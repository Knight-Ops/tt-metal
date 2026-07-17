# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Instrumented prefill+decode profiling — ground-truth overhead via the tt-metal device profiler
(Tracy). See DEBUGGING.md for the full workflow (this is the script it documents).

Profiles THREE signpost-bounded regions in one run: a PREFILL forward, one EAGER decode step, and one
TRACED decode step. Eager (not traced) attributes each op by name and exposes the host-dispatch gaps;
the per-op DEVICE KERNEL DURATION is identical to traced replay (same kernels), so summing it gives the
pure-silicon compute floor. Comparing eager vs traced (separable by the METAL TRACE ID column) shows
how much of the wall-clock is real compute vs on-device dispatch/gap overhead — i.e. whether the
remaining win is "fuse the ops" (overhead-bound) or "faster kernels / fewer bytes" (compute-bound).

Captures BOTH an EAGER decode step and a TRACED decode step in one run. They are separable in the
device report by the METAL TRACE ID column (traced ops have it set). This lets us compare:
  - eager:  per-op kernel time + HOST dispatch gaps (host-bound; what trace removes)
  - traced: per-op kernel time + ON-DEVICE inter-op gaps (the irreducible-without-fusion overhead)

Run (profiler-enabled build, ENABLE_TRACY=ON):
    QWEN36_LAYERS=4 ./python_env/bin/python -m tracy -r -v --op-support-count 6000 \
        models/demos/qwen3_6_a3b/tests/prof_decode.py
Output: generated/profiler/.logs/cpp_device_perf_report.csv (per-op device timings; parse directly).
"""
import os

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs

try:
    from tracy import signpost
except Exception:  # tracy not importable outside the wrapper

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
        seq = int(os.environ.get("QWEN36_SEQ", "32"))
        prompt = torch.randint(0, args.vocab_size, (1, seq))
        model.forward(prompt)  # warm prefill: compile prefill kernels (NOT measured)

        # --- PREFILL measured (eager, fused-chunked delta + dense MoE) ---
        ttnn.synchronize_device(mesh)
        signpost("prefill_start")
        logits = model.forward(prompt)
        ttnn.synchronize_device(mesh)
        signpost("prefill_stop")

        nxt = int(logits[0, -1].argmax())
        model.start_decode(nxt)
        model.decode_step_eager()  # warmup: compile all decode kernels (NOT measured)
        model.decode_step_eager()  # second warmup so caches/state are steady-state

        # --- EAGER measured step (host-dispatched; per-op kernel + host gaps) ---
        ttnn.synchronize_device(mesh)
        signpost("eager_start")
        model.decode_step_eager()
        signpost("eager_stop")
        ttnn.synchronize_device(mesh)

        # --- TRACED measured step (same kernels; on-device inter-op gaps) ---
        model.capture_decode_trace()
        model.decode_step_traced()  # warm replay (not measured)
        ttnn.synchronize_device(mesh)
        signpost("trace_start")
        model.decode_step_traced()  # measured traced step (METAL TRACE ID set in the report)
        signpost("trace_stop")
        ttnn.synchronize_device(mesh)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
