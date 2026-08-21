# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage a self-contained tt-kernel vLLM bundle for Qwen3.6-35B-A3B.

What ships, and why
-------------------
``tt-kernel push --backend vllm`` ships only the ``--bundle-dir`` subtree. The checked-in
``vllm_bundle/`` holds just the adapter, whose ``from models.demos.qwen3_6_a3b...`` imports
resolve on the serve host from *its* tt-metal checkout -- so a fix in this repo does not travel
with the published artifact.

We therefore vendor **our model package** into the bundle and host-resolve **the platform**:

  * vendored: everything under ``models/demos/qwen3_6_a3b/`` that the adapter's import closure
    reaches, laid down as ``models/autoports/qwen3_6_a3b/`` with the adapter moved inside it.
  * host-resolved: ``models.common.lightweightmodule`` and ``models.tt_transformers.*`` (the
    ``Generator`` base class). These are upstream tt-metal code we do not modify, so depending on
    the host for them is exactly as safe as depending on it for ``ttnn`` -- and vendoring the
    ``Generator`` import alone would drag in 32 extra modules / ~890 KB, three quarters of the
    bundle, to inherit one base class.

Host-resolution is now an ALLOWLIST (``HOST_RESOLVED``) and staging FAILS on anything else, because
the previous "all of ``models.common`` is upstream" rule shipped a broken bundle: ``moe_gather.py``
was OUR file living in ``models/common/``, so it existed on the build machine and nowhere else, and
``tt/moe.py``'s unconditional ``from models.common import moe_gather`` made every pulled bundle fail
at import. Two things hid it: the module was in a directory that reads as upstream, and
:func:`external_imports` recorded ``from models.common import moe_gather`` as just ``models.common``
-- a package, not a file -- so the audit called it an attribute import and passed. It now expands
imported NAMES to modules, and the file itself has moved into this package where the closure picks it
up automatically and no host copy can shadow it.

Why ``models/qwen3_6_a3b`` and not ``models/demos/qwen3_6_a3b``
--------------------------------------------------------------
``models/`` has no ``__init__.py`` anywhere, in this repo or in the bundle, so it is a PEP 420
namespace package: Python MERGES it across every ``sys.path`` entry. That is what lets the
host's ``models.common`` and our ``models.qwen3_6_a3b`` coexist. It also means the vendored path
must not collide with one the host already provides -- a vendored ``models/demos/qwen3_6_a3b``
would lose to the host's copy, because ``tt-kernel serve`` *prepends* the tt-metal checkout to
``PYTHONPATH`` while the plugin *appends* the bundle folder ("never insert(0), so an installed
package of the same name always wins"). tt-metal has no ``models/qwen3_6_a3b``, so there is
nothing to shadow it.

Do not write ``models/__init__.py`` into the bundle -- that would turn it into a regular package
and break the namespace merge.

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

SRC_PKG = "models.demos.qwen3_6_a3b"  # in-repo package name
DST_PKG = "models.qwen3_6_a3b"  # name inside the bundle
SRC_DIR = Path(SRC_PKG.replace(".", "/"))
DST_DIR = Path(DST_PKG.replace(".", "/"))
ADAPTER = "generator_vllm.py"
BUNDLE_SUBDIR = SRC_DIR / "packaging" / "vllm_bundle"
# Entry modules of the adapter's closure. `models.tt_transformers.tt.generator` is deliberately
# absent: it is host-resolved (see the module docstring).
ENTRY_MODULES = [f"{SRC_PKG}.tt.model", f"{SRC_PKG}.tt.model_config", f"{SRC_PKG}.tt.load_checkpoints"]
# Also vendored: the standalone OpenAI-compatible server and the CLI demo. Nothing in the vLLM path
# imports these -- they ship so the artifact is self-sufficient for the one thing vLLM cannot do on
# this backend, speculative decode (MTP is unreachable through the plugin: it asserts no
# speculative_config, its runner is one-token-per-step, and vLLM has no hook to advance the
# gated-delta recurrent state by an accepted count). Their extra third-party deps
# (fastapi / uvicorn / transformers) are needed ONLY if you run them, so they cost the vLLM path
# nothing but disk.
SERVER_MODULES = [
    f"{SRC_PKG}.demo.server",
    f"{SRC_PKG}.demo.demo",
    f"{SRC_PKG}.demo.runner",
    f"{SRC_PKG}.demo.generate_weight_cache",
]
# Namespace-package levels that must NOT get an __init__.py.
NAMESPACE_DIRS = {Path("models")}
# The ONLY `models.*` modules the bundle may resolve from the serve host. Anything else must be
# vendored (i.e. must live under SRC_PKG so the closure picks it up). Keep this list tiny and only
# for code that genuinely ships with tt-metal.
# MODULES only -- deliberately no bare-package entries. `from models.common import moe_gather`
# reports the PACKAGE (`models.common`), and if the name cannot be resolved to a file in this repo
# there is no way to tell a submodule from an attribute -- which is the exact shape of the bug that
# shipped. Listing only leaf modules means any `from <pkg> import name` surfaces `<pkg>`, fails the
# check, and forces an explicit decision. `from models.common.lightweightmodule import X` reports the
# module itself and passes.
HOST_RESOLVED = {
    "models.common.lightweightmodule",
    "models.tt_transformers.tt.generator",
}
# `vllm_metadata.json` is not staged: with a v4 --manifest, tt-kernel renders it on pull from the
# authoritative manifest and overwrites anything shipped, so shipping the checked-in copy only
# publishes a stale env that contradicts the manifest.
SKIP_BUNDLE_FILES = {"vllm_metadata.json"}
_SRC_RE = re.compile(rf"\b{re.escape(SRC_PKG)}\b")


def _module_to_path(repo: Path, mod: str) -> str | None:
    rel = mod.replace(".", "/")
    for cand in (rel + ".py", rel + "/__init__.py"):
        if (repo / cand).is_file():
            return cand
    return None


def compute_closure(repo: Path, entries: list[str], within: str = SRC_PKG) -> list[str]:
    """Modules under ``within`` reachable from ``entries`` (static AST walk, relative imports
    included). Imports that leave ``within`` are recorded by :func:`external_imports`, not
    followed -- they are host-resolved."""
    seen: set[str] = set()
    queue = collections.deque(entries)
    while queue:
        mod = queue.popleft()
        if mod in seen or not mod.startswith(within):
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
                queue.append(target)
                # `from pkg import mod` may name a submodule rather than an attribute.
                queue.extend(f"{target}.{a.name}" for a in node.names)
    return sorted({p for m in seen if (p := _module_to_path(repo, m))})


def external_imports(repo: Path, files: list[str]) -> set[str]:
    """`models.*` imports in the staged set that fall OUTSIDE SRC_PKG (i.e. host-resolved).

    Expands ``from pkg import name`` to ``pkg.name`` when that resolves to a module FILE. Without
    that, ``from models.common import moe_gather`` was recorded as ``models.common`` -- a package
    that obviously exists on the host -- and the audit passed while the bundle was missing the
    module it actually needed."""
    out: set[str] = set()
    for rel in files:
        for node in ast.walk(ast.parse((repo / rel).read_text())):
            if isinstance(node, ast.Import):
                out.update(
                    a.name for a in node.names if a.name.startswith("models.") and not a.name.startswith(SRC_PKG)
                )
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("models."):
                if node.module.startswith(SRC_PKG):
                    continue
                out.add(node.module)
                for a in node.names:  # `from pkg import mod` may name a SUBMODULE, not an attribute
                    if _module_to_path(repo, f"{node.module}.{a.name}"):
                        out.add(f"{node.module}.{a.name}")
    return out


def check_host_resolved(external: set[str]) -> list[str]:
    """External imports that are NOT on the HOST_RESOLVED allowlist -- i.e. things that will not
    exist on a serve host and must be vendored instead. Returns the offenders."""
    return sorted(m for m in external if m not in HOST_RESOLVED)


def rewrite(text: str) -> str:
    return _SRC_RE.sub(DST_PKG, text)


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

    # 1. bundle-level files. The adapter moves INSIDE the vendored package so it is addressed by
    #    a dotted path (models.autoports.qwen3_6_a3b.generator_vllm:Class), matching the
    #    convention; everything else (README) stays at the bundle root.
    pkg_root = out / DST_DIR
    pkg_root.mkdir(parents=True)
    staged_docs = 0
    for src in sorted(bundle_src.iterdir()):
        if src.is_dir() or src.suffix == ".pyc" or src.name in SKIP_BUNDLE_FILES:
            continue
        dst = (pkg_root / src.name) if src.name == ADAPTER else (out / src.name)
        dst.write_text(rewrite(src.read_text()) if src.suffix == ".py" else src.read_text())
        staged_docs += src.name != ADAPTER

    # 2. the model package, remapped SRC_DIR -> DST_DIR
    closure = compute_closure(repo, ENTRY_MODULES + SERVER_MODULES)
    for rel in closure:
        dst = out / DST_DIR / Path(rel).relative_to(SRC_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(rewrite((repo / rel).read_text()))

    # 3. non-.py data files those modules load by path
    n_data = 0
    for d in sorted({Path(rel).parent for rel in closure}):
        for src in (repo / d).rglob("*"):
            if not src.is_file() or src.suffix in (".py", ".pyc") or "__pycache__" in src.parts:
                continue
            dst = out / DST_DIR / src.relative_to(repo / SRC_DIR)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_data += 1

    # 4. __init__.py for real package levels only -- never for the namespace levels
    n_init = 0
    for d in {p.parent for p in out.rglob("*.py")}:
        rel = d.relative_to(out)
        if rel in NAMESPACE_DIRS or rel == Path("."):
            continue
        init = d / "__init__.py"
        if not init.exists():
            init.write_text("")
            n_init += 1
    for ns in NAMESPACE_DIRS:
        assert not (out / ns / "__init__.py").exists(), f"{ns} must stay a namespace package"

    py = sorted(out.rglob("*.py"))
    print(f"staged {out}")
    print(f"  bundle docs        : {staged_docs}")
    n_server = sum(1 for rel in closure if "/demo/" in rel)
    print(f"  vendored modules   : {len(closure)} (+ adapter)  [{n_server} standalone-server]")
    print(f"  data files         : {n_data}")
    print(f"  __init__.py created: {n_init}")
    print(f"  total python       : {sum(f.stat().st_size for f in py) / 1024:.0f} KB in {len(py)} files")
    print(f"  main_class         : {DST_PKG}.{ADAPTER[:-3]}:Qwen36ForCausalLM")

    leaked = [
        f"{f.relative_to(out)}:{i}"
        for f in py
        for i, line in enumerate(f.read_text().splitlines(), 1)
        if SRC_PKG in line
    ]
    if leaked:
        print(f"  ERROR: {len(leaked)} reference(s) to {SRC_PKG} remain:", file=sys.stderr)
        for x in leaked[:10]:
            print(f"    {x}", file=sys.stderr)
        return 1
    # Audit the ADAPTER too, not just the closure: it is staged separately (step 1) and its imports
    # were never checked. Then fail on anything the serve host will not provide -- shipping a bundle
    # whose import graph reaches a build-machine-only module is the one failure mode that looks like
    # "plug and play" right up until someone pulls it.
    ext = external_imports(repo, closure + [str(BUNDLE_SUBDIR / ADAPTER)])
    print(f"  host-resolved      : {', '.join(sorted(ext)) or '(none)'}")
    bad = check_host_resolved(ext)
    if bad:
        print(
            f"  ERROR: {len(bad)} import(s) resolve to neither this bundle nor tt-metal:",
            file=sys.stderr,
        )
        for m in bad:
            where = _module_to_path(repo, m)
            print(
                f"    {m}" + (f"  (exists here as {where} -- move it under {SRC_PKG})" if where else ""),
                file=sys.stderr,
            )
        print(
            "  Either move the module under "
            f"{SRC_PKG} so the closure vendors it, or add it to HOST_RESOLVED if it genuinely "
            "ships with tt-metal.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
