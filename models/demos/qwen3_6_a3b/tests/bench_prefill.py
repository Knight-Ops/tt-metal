# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standing prefill benchmark: warm eager vs traced prefill (tok/s + TTFT) at one or more sequence
lengths, plus a roofline reference. One model load; eager timed first (forward() reallocates caches),
then all traces captured, then traced replays timed.

  QWEN36_LAYERS=40 QWEN36_MAX_SEQ=512 ./python_env/bin/python \
      models/demos/qwen3_6_a3b/tests/bench_prefill.py --seqs 128,256
"""
import argparse
import os
import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs


def _warm_time(fn, mesh, iters=3):
    for _ in range(2):
        fn()
        ttnn.synchronize_device(mesh)
    best = 1e9
    for _ in range(iters):
        t0 = time.time()
        fn()
        ttnn.synchronize_device(mesh)
        best = min(best, time.time() - t0)
    return best * 1000.0  # ms (best-of to reduce noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=str, default="256", help="comma-separated seq lengths (multiples of 64)")
    a = ap.parse_args()
    seqs = [int(s) for s in a.seqs.split(",")]
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=600_000_000)
    try:
        args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=max(512, max(seqs)))
        t0 = time.time()
        model = TtModel(mesh, args, CheckpointLoader(ckpt), num_layers=n_layers)
        print(f"[bench] {n_layers}-layer model built in {time.time()-t0:.0f}s; seqs={seqs}")
        torch.manual_seed(0)
        ids = {L: torch.randint(0, args.vocab_size, (1, L)) for L in seqs}

        eager = {L: _warm_time(lambda L=L: model.forward(ids[L]), mesh) for L in seqs}
        model.setup_prefill_traces(seqs)
        traced = {L: _warm_time(lambda L=L: model.forward_prefill_traced(ids[L]), mesh) for L in seqs}

        print(
            f"\n{'seq':>5} | {'eager ms':>9} {'eager tok/s':>12} | {'traced ms':>10} {'traced tok/s':>13} | {'speedup':>8}"
        )
        for L in seqs:
            print(
                f"{L:>5} | {eager[L]:>9.0f} {L/eager[L]*1000:>12.0f} | {traced[L]:>10.0f} "
                f"{L/traced[L]*1000:>13.0f} | {eager[L]/traced[L]:>7.2f}x"
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
