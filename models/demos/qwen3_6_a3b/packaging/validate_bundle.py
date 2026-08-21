# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prove a staged bundle is self-contained by IMPORTING it the way a serve host would.

Why this exists
---------------
The v4 bundle shipped with a hidden dependency: ``tt/moe.py`` did an unconditional
``from models.common import moe_gather``, and ``moe_gather.py`` was our own file that happened to live
in ``models/common/`` on the build machine. It is not in tt-metal, not in any fork, and was not staged
-- so the artifact imported fine for us and failed for everyone else. Static checks in
``stage_vllm_bundle.py`` now reject that class of import, but the only thing that actually proves
self-containment is loading the package with the model directory OUT of reach and seeing where each
module resolves from.

What it checks
--------------
1. Every ``models.*`` module the import graph touches resolves either from the BUNDLE or from the
   short allowlist of things tt-metal genuinely provides -- nothing from ``models/demos/...``.
2. The entry modules import at all (catches missing files, not just mislocated ones).

Usage
-----
    python models/demos/qwen3_6_a3b/packaging/stage_vllm_bundle.py --out /tmp/b
    python models/demos/qwen3_6_a3b/packaging/validate_bundle.py --bundle /tmp/b

Needs ttnn importable (no device). Does not import the vLLM adapter -- that needs vllm installed;
its `models.*` imports are covered by the stager's static audit.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

DST_PKG = "models.qwen3_6_a3b"
ENTRIES = [f"{DST_PKG}.tt.model", f"{DST_PKG}.tt.model_config", f"{DST_PKG}.tt.load_checkpoints"]
# The standalone server/demo ship in the bundle but need fastapi/uvicorn/transformers, which the vLLM
# path does not. Checked BEST-EFFORT: a missing third-party dep is reported and skipped, while a broken
# `models.*` import still fails -- otherwise the one code path that is only reachable by hand would be
# the one nothing verifies.
OPTIONAL_ENTRIES = [f"{DST_PKG}.demo.server", f"{DST_PKG}.demo.demo"]
# What a serve host legitimately supplies (mirrors stage_vllm_bundle.HOST_RESOLVED).
HOST_OK = {"models.common.lightweightmodule", "models.tt_transformers.tt.generator"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="staged bundle directory")
    ap.add_argument("--repo", default=None, help="tt-metal checkout that plays the serve host")
    a = ap.parse_args()
    bundle = Path(a.bundle).expanduser().resolve()
    repo = Path(a.repo).resolve() if a.repo else Path(__file__).resolve().parents[4]
    if not (bundle / DST_PKG.replace(".", "/")).is_dir():
        print(f"error: {bundle} does not contain {DST_PKG.replace('.', '/')}", file=sys.stderr)
        return 1

    # A serve host prepends its tt-metal checkout and APPENDS the bundle (see stage_vllm_bundle);
    # reproduce that order so we are not flattered by an import that only works bundle-first.
    sys.path[:0] = [str(repo)]
    sys.path.append(str(bundle))

    for mod in ENTRIES:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001 -- report, do not traceback-spam
            print(f"FAIL: import {mod}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    skipped = []
    for mod in OPTIONAL_ENTRIES:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as exc:
            missing = (exc.name or "").split(".")[0]
            if missing and missing != "models":  # a third-party dep, not a bundle problem
                skipped.append(f"{mod} (needs {missing})")
                continue
            print(f"FAIL: import {mod}: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: import {mod}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    bad, n_bundle, n_host = [], 0, 0
    for name, m in sorted(sys.modules.items()):
        if not name.startswith("models.") or m is None:
            continue
        f = getattr(m, "__file__", None)
        if f is None:  # namespace package (models, models.common) -- no single file, fine
            continue
        p = Path(f).resolve()
        if bundle in p.parents:
            n_bundle += 1
        elif name in HOST_OK:
            n_host += 1
        else:
            bad.append((name, str(p)))

    print(f"imported {len(ENTRIES)} entry + {len(OPTIONAL_ENTRIES) - len(skipped)} optional modules")
    for s_ in skipped:
        print(f"  skipped            : {s_}")
    print(f"  from bundle        : {n_bundle}")
    print(f"  from host (allowed): {n_host}  {sorted(HOST_OK & set(sys.modules))}")
    if bad:
        print(f"  ERROR: {len(bad)} module(s) resolved from OUTSIDE the bundle:", file=sys.stderr)
        for name, path in bad:
            print(f"    {name}  <-  {path}", file=sys.stderr)
        print("  A serve host will not have these. Vendor them (move under the model package).", file=sys.stderr)
        return 1
    print("  OK: self-contained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
