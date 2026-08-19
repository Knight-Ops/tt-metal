# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Focused Tracy profiling of ONE Gated-DeltaNet decode step (the now-dominant decode cost, ~53%).

Isolates the real gated-delta mixer (real weights + the layer's decode cache) and runs its decode
forward in a signpost-bounded EAGER region, so every internal op (conv, recurrence, the in_proj
matmuls, norms, reshapes) is attributed by name + shape in the device report. Use this to find the
largest sub-cost before picking a lever (program-config tune / DRAM-shard / precision / fused kernel).

Run (profiler build):
    QWEN36_LAYERS=4 ./python_env/bin/python -m tracy -r -v --op-support-count 6000 \
        models/demos/qwen3_6_a3b/tests/prof_gated_delta.py
Then parse generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv between gdn_start/gdn_stop.
"""
import os

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs

try:
    from tracy import signpost
except Exception:

    def signpost(*a, **k):
        pass


@torch.no_grad()
def main():
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=128)
        loader = CheckpointLoader(ckpt)
        model = TtModel(mesh, args, loader, num_layers=n_layers)

        torch.manual_seed(0)
        lin_idx = next(i for i, l in enumerate(model.layers) if l.is_linear)
        gdn = model.layers[lin_idx].mixer
        cache_g = model.caches[lin_idx]
        x_g = ttnn.from_torch(
            torch.randn(1, 1, 1, args.dim) * 0.5,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        # warmup: compile all gated-delta decode kernels (NOT measured)
        gdn.forward(x_g, cache_g)
        gdn.forward(x_g, cache_g)
        ttnn.synchronize_device(mesh)

        # --- EAGER measured region: a few steps so per-op kernel times average cleanly ---
        signpost("gdn_start")
        for _ in range(5):
            gdn.forward(x_g, cache_g)
        ttnn.synchronize_device(mesh)
        signpost("gdn_stop")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
