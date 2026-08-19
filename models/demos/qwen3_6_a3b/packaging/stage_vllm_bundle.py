# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage a SELF-CONTAINED tt-kernel vLLM bundle: the adapter plus a vendored copy of the
model code it imports.

Why this exists
---------------
``tt-kernel push --backend vllm`` ships only the ``--bundle-dir`` subtree. The checked-in
``vllm_bundle/`` holds just the adapter, which does ``from models.demos.qwen3_6_a3b...`` --
resolved on the serve host from *its* tt-metal checkout. So a pushed bundle inherits whatever
model code that host happens to have, and a fix in this repo does not travel with the artifact.

Vendoring under the name ``models`` does NOT work: the plugin deliberately *appends* the bundle
folder to ``sys.path`` ("never insert(0), so an installed package of the same name always wins"),
while ``tt-kernel serve`` *prepends* the tt-metal checkout to ``PYTHONPATH``. The host's
``models`` would therefore always shadow ours -- silently.

So we vendor under a unique root (``qwen36_vendored``) and rewrite ``from models.`` ->
``from qwen36_vendored.models.`` in every staged file, including the adapter. The rewrite happens
only in the staging output, never in the repo, so in-repo development keeps using the plain
``models.`` imports.

Usage
-----
    python models/demos/qwen3_6_a3b/packaging/stage_vllm_bundle.py --out /tmp/qwen36_bundle
    tt-kernel push <ns>/qwen3.6-a3b-blackhole --backend vllm \
        --manifest models/demos/qwen3_6_a3b/packaging/qwen36_v4_manifest.json \
        --bundle-dir /tmp/qwen36_bundle
"""
from __future__ import annotations

import argparse
import ast
import collections
import re
import shutil
import sys
from pathlib import Path

VENDOR_ROOT = "qwen36_vendored"
# Entry modules the adapter imports; the closure is walked from here.
ENTRY_MODULES = [
    "models.demos.qwen3_6_a3b.tt.model",
    "models.demos.qwen3_6_a3b.tt.model_config",
    "models.demos.qwen3_6_a3b.tt.load_checkpoints",
    "models.tt_transformers.tt.generator",
]
BUNDLE_SUBDIR = Path("models/demos/qwen3_6_a3b/packaging/vllm_bundle")
# `from models.X import ...` / `import models.X` -> vendored root. Anchored so a bare
# "models" elsewhere in the line (a string, a comment) is untouched.
_IMPORT_RE = re.compile(r"^(\s*)(from|import)(\s+)models\b", re.MULTILINE)


def _module_to_path(repo: Path, mod: str) -> str | None:
    rel = mod.replace(".", "/")
    for cand in (rel + ".py", rel + "/__init__.py"):
        if (repo / cand).is_file():
            return cand
    return None


def compute_closure(repo: Path, entries: list[str]) -> list[str]:
    """Transitive closure of ``models.*`` modules reachable from ``entries`` (static AST walk)."""
    seen: set[str] = set()
    queue = collections.deque(entries)
    while queue:
        mod = queue.popleft()
        if mod in seen:
            continue
        seen.add(mod)
        path = _module_to_path(repo, mod)
        if path is None:
            continue
        try:
            tree = ast.parse((repo / path).read_text())
        except (SyntaxError, UnicodeDecodeError) as exc:
            print(f"  warn: could not parse {path}: {exc}", file=sys.stderr)
            continue
        # Package of `mod`, for resolving relative imports.
        parts = mod.split(".")
        pkg = parts if path.endswith("__init__.py") else parts[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                queue.extend(a.name for a in node.names if a.name.startswith("models."))
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # `from . import x` / `from .mod import x`
                    base = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                    target = ".".join(base + ([node.module] if node.module else []))
                elif node.module and node.module.startswith("models."):
                    target = node.module
                else:
                    continue
                if not target.startswith("models."):
                    continue
                queue.append(target)
                # `from pkg import mod` may name a submodule rather than an attribute.
                queue.extend(f"{target}.{a.name}" for a in node.names)
    return sorted({p for m in seen if (p := _module_to_path(repo, m))})


def rewrite_imports(text: str) -> str:
    return _IMPORT_RE.sub(rf"\1\2\3{VENDOR_ROOT}.models", text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="staging directory to write (recreated)")
    ap.add_argument("--repo", default=None, help="tt-metal checkout root (default: infer from this file)")
    args = ap.parse_args()

    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parents[4]
    bundle_src = repo / BUNDLE_SUBDIR
    if not bundle_src.is_dir():
        print(f"error: bundle dir not found: {bundle_src}", file=sys.stderr)
        return 1
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # 1. adapter files (skip caches), with imports rewritten to the vendored root.
    #    vllm_metadata.json is deliberately NOT staged: with a v4 --manifest, tt-kernel is the
    #    source of truth and RENDERS that file on pull (cli.py: write_vllm_metadata(render_...)),
    #    overwriting anything shipped. Shipping the checked-in copy only publishes a stale env
    #    that contradicts the manifest.
    n_adapter = 0
    for src in sorted(bundle_src.iterdir()):
        if src.is_dir() or src.name.endswith(".pyc") or src.name == "vllm_metadata.json":
            continue
        text = src.read_text()
        (out / src.name).write_text(rewrite_imports(text) if src.suffix == ".py" else text)
        n_adapter += 1

    # 2. vendored model code
    closure = compute_closure(repo, ENTRY_MODULES)
    vroot = out / VENDOR_ROOT
    for rel in closure:
        dst = vroot / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(rewrite_imports((repo / rel).read_text()))

    # 3. non-.py data files that vendored modules load by path at import time
    data_dirs = {Path(rel).parent for rel in closure}
    n_data = 0
    for d in sorted(data_dirs):
        for src in (repo / d).rglob("*"):
            if not src.is_file() or src.suffix in (".py", ".pyc") or "__pycache__" in src.parts:
                continue
            dst = vroot / src.relative_to(repo)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_data += 1

    # 4. __init__.py for every package level (tt-metal leans on PEP 420 namespace packages;
    #    make the vendored tree explicit so it cannot half-resolve against a host package)
    (vroot / "__init__.py").write_text(f'"""Vendored model code for the {VENDOR_ROOT} bundle."""\n')
    n_init = 1
    for rel in closure:
        d = (vroot / rel).parent
        while d != vroot:
            init = d / "__init__.py"
            if not init.exists():
                init.write_text("")
                n_init += 1
            d = d.parent

    total_kb = sum(f.stat().st_size for f in out.rglob("*.py")) / 1024
    print(f"staged {out}")
    print(f"  adapter files      : {n_adapter}")
    print(f"  vendored modules   : {len(closure)}")
    print(f"  data files         : {n_data}")
    print(f"  __init__.py created: {n_init}")
    print(f"  total python       : {total_kb:.0f} KB")
    leaked = [
        f"{f.relative_to(out)}:{i}"
        for f in out.rglob("*.py")
        for i, line in enumerate(f.read_text().splitlines(), 1)
        if re.match(r"^\s*(from|import)\s+models\b", line)
    ]
    if leaked:
        print(f"  ERROR: {len(leaked)} un-rewritten 'models' imports remain:", file=sys.stderr)
        for x in leaked[:10]:
            print(f"    {x}", file=sys.stderr)
        return 1
    print("  no un-rewritten 'models' imports remain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
