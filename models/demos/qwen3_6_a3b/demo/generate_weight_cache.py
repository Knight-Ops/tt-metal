# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One-time weight-cache generator for Qwen3.6-35B-A3B.

This is an OPTIONAL speedup, not a required step. Building ``TtModel`` writes one ``.tensorbin`` per
big weight (experts, attention/linear-attn projections, embed, lm_head) under the weight-cache dir
the first time it runs; every later load (demo, server, benches) then reads those back and skips the
block-float quantization + tilize, which is the bulk of the ~800s/40-layer cold load.

Running this script just builds the model once so the cache is populated ahead of time. If caching
is unavailable for any reason (cache disabled, unwritable dir, ttnn.as_tensor error) the build falls
back to a normal from-scratch load with a single warning and still succeeds -- you simply don't get
the cached files. Subsequent real runs behave identically.

Usage (tt-metal python_env):
    QWEN36_LAYERS=40 python models/demos/qwen3_6_a3b/demo/generate_weight_cache.py
    # cache somewhere other than <ckpt>/tt_weight_cache:
    TT_CACHE_PATH=/mnt/nas/qwen36_cache python models/demos/qwen3_6_a3b/demo/generate_weight_cache.py
"""
import argparse
import os
import time

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--layers",
        type=int,
        default=int(os.environ.get("QWEN36_LAYERS", "40")),
        help="number of decoder layers to build/cache (default: all)",
    )
    cli = ap.parse_args()

    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=int(os.environ.get("QWEN36_MAX_SEQ", "512")))
        if args.weight_cache_path is None:
            print("[gen-cache] QWEN36_WEIGHT_CACHE=0 -> caching disabled; nothing to generate.")
            return
        print(f"[gen-cache] populating weight cache at {args.weight_cache_path} ({cli.layers} layers)...")
        loader = CheckpointLoader(ckpt)
        t0 = time.time()
        TtModel(mesh, args, loader, num_layers=cli.layers)
        print(f"[gen-cache] done in {time.time() - t0:.1f}s. Later loads will reuse this cache.")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
