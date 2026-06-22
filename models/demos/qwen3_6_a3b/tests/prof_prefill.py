# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prefill profiling companion to prof_decode.py — profiles TWO signpost-bounded regions in one run:
an EAGER prefill forward and a TRACED prefill replay, both warm. Eager exposes the per-op host-dispatch
gaps; traced replays the captured graph (those gaps removed). They are separable in the device report
by the METAL TRACE ID column (traced ops have it set), so you can read off how much of prefill wall
time is real compute vs on-device dispatch — i.e. exactly what the trace recovers.

Run (profiler build; note the larger trace_region_size below and the high op-support-count):
    QWEN36_LAYERS=4 QWEN36_SEQ=256 ./python_env/bin/python -m tracy -r -v --op-support-count 6000 \
        models/demos/qwen3_6_a3b/tests/prof_prefill.py
Report: generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv (load the <ts> folder in
ttnn-visualizer; filter to prefill_eager_* / prefill_traced_*). 4 layers is an exact 1/10 scale of the
40-layer model with identical op shapes, so proportions match the full model.
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
    seq = int(os.environ.get("QWEN36_SEQ", "256"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=512)
        model = TtModel(mesh, args, CheckpointLoader(ckpt), num_layers=n_layers)
        torch.manual_seed(0)
        ids = torch.randint(0, args.vocab_size, (1, seq))

        # --- EAGER prefill (host-dispatched; per-op kernel + host gaps) ---
        model.forward(ids)  # warmup: compile (NOT measured)
        ttnn.synchronize_device(mesh)
        signpost("prefill_eager_start")
        model.forward(ids)  # measured eager prefill
        ttnn.synchronize_device(mesh)
        signpost("prefill_eager_stop")

        # --- TRACED prefill (captured graph replayed; per-op dispatch removed) ---
        model.capture_prefill_trace(seq)  # capture (NOT measured)
        model.forward_prefill_traced(ids)  # warm replay (NOT measured)
        ttnn.synchronize_device(mesh)
        signpost("prefill_traced_start")
        model.forward_prefill_traced(ids)  # measured traced replay (METAL TRACE ID set in the report)
        ttnn.synchronize_device(mesh)
        signpost("prefill_traced_stop")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
