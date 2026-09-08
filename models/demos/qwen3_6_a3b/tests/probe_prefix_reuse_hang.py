# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SOLVED. Bisection record for the multi-turn device hang found while enabling prefix reuse.

ROOT CAUSE: the device PROGRAM CACHE accumulates, and past some point the device hangs inside an
ordinary prefill op (a `ttnn.concat` in the conv, a MoE `swiglu` -- it moves). Proven causally, not by
correlation: dropping the cache between turns makes a conversation that otherwise hangs on turn 3 run
indefinitely.

NOT memory, all measured at 40 layers at the moment of the hang: DRAM 10.6 GB free with 1330 MiB/bank
still CONTIGUOUS (flat across turns), L1 ~1% used, trace region 19.1 of 200 MiB with one trace
resident. Not a fixed entry count either -- it hung at 553 having previously survived 832.

THE FIX, both halves needed (see IMPLEMENTATION_GUIDE.md §2 and §5):
  1. Bucket the prefill shape space so the cache saturates instead of growing ~90 entries/turn --
     ragged blocks are padded and the pad rows MASKED out of the recurrence rather than the block
     being sliced to `valid_len` (QWEN36_GDN_RAGGED_MASK, PCC-gated in tests/test_ragged_mask.py:
     masked vs sliced is exactly equal, 1.000000). Short prompts are bucketed the same way. This
     alone moved the hang two turns later but did NOT stop it: a few ops key on the ABSOLUTE position
     (chunked-SDPA chunk_start_idx, fill_cache update_idx), which advances forever.
  2. Recycle the cache past QWEN36_PROGRAM_CACHE_LIMIT (TtModel.maybe_recycle_programs), releasing
     traces first because they reference the binaries being freed. 400 only delayed the hang (turn 5
     instead of 3); 280 ran 16 consecutive turns at 40 layers -- 6 streaming plus 10 long-context out
     to 2586 prompt tokens -- with prefill holding 127-508 tok/s.

TWO OTHER REAL BUGS found on the way, both fixed and both independent of prefix reuse:
  * warmup captured the GREEDY decode trace before the larger SAMPLING one. Capturing the larger graph
    into the hole left by releasing the smaller one wedged the device on a REPEATED PROMPT. Measured:
    none/greedy-only/sampling-only/greedy-then-greedy all pass, greedy-then-sampling hangs,
    sampling-then-greedy passes. Guarded device-free by test_server_stream.py.
  * the gated-delta checkpoint ring was allocated after the first prefill had churned the allocator
    (MEMORY.md §1-3). At 4 layers it silently DEGRADED the device (prefill 149 vs 743 tok/s, decode
    5.5 vs 24.5); at 40 layers, where it is 180 MiB, it wedged request two. Now pre-allocated from
    `args` before any prefill -- TtModel.alloc_state_checkpoints.

RULED OUT, each by a passing on-device run: block width, offset alignment and q_chunk (the rule the
kernel enforces is loud -- "TT_FATAL: chunk_start_idx must be a multiple of q_chunk_size"); the state
restore itself; a live decode trace during a resumed prefill; the sampled decode tail; worker-thread
dispatch; the fast-vs-safe dense-MoE config; tile alignment of the snapshot buffers; DRAM
fragmentation (largest contiguous free per bank is flat).

THE PROCESS LESSON, which cost the most: everything above was first bisected, fixed and "validated"
at QWEN36_LAYERS=4, with green PCC gates and a passing end-to-end run -- and then failed immediately
at 40. A 4-layer model has so much headroom that every one of these faults disappears. Small-layer
runs are for correctness only.

USING THIS SCRIPT: reproduces the original failing turn pair by driving Qwen36Engine directly (real
warmup, no uvicorn). With the fixes in place it passes; set QWEN36_PROGRAM_CACHE_LIMIT=0 and
QWEN36_GDN_RAGGED_MASK=0 to watch it hang again.

    QWEN36_LAYERS=40 REPRO_SAMPLE=1 REPRO_THREAD=1 python .../probe_prefix_reuse_hang.py

Instruments worth knowing: QWEN36_PREFILL_LAYER_SYNC=1 syncs and logs after every layer and MoE stage
so a hang localizes to one op; `mesh.num_program_cache_entries()` and
`memstat.region(mesh, ttnn.BufferType.DRAM|L1|TRACE)` (note `largest_free_per_bank`, not just `free`)
are what turned guessing into measurement."""
import os
import sys
import threading
import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs

# Geometry of the failing turn pair, from the server log (QWEN36_LOG_REQUESTS=1).
T1 = int(os.environ.get("REPRO_T1", "636"))
T2 = int(os.environ.get("REPRO_T2", "766"))
DIV = int(os.environ.get("REPRO_DIV", "634"))  # T1 - 2: where a re-rendered turn first differs
WITH_TRACE = os.environ.get("REPRO_TRACE", "1") == "1"
SAMPLE = os.environ.get("REPRO_SAMPLE", "0") == "1"
THREAD = os.environ.get("REPRO_THREAD", "0") == "1"


def log(*a):
    print(*a, flush=True)
    sys.stdout.flush()


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200 * 1024 * 1024)
    try:
        ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
        args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=8192)
        model = TtModel(mesh, args, CheckpointLoader(ckpt), num_layers=int(os.environ.get("QWEN36_LAYERS", "4")))
        torch.manual_seed(0)
        samp = lambda: model.enable_sampling(0.6, top_k=20, top_p=0.95, seed=0, presence_penalty=1.5)
        if SAMPLE:
            samp()  # the server's default tail (temperature 0.6 / top_k 20 / top_p 0.95 / pp 1.5)

        p1 = torch.randint(0, args.vocab_size, (1, T1))
        log(f"[1] cold prefill T={T1}")
        l1, reused = model.prefill_reuse(p1, None)
        log(f"[1] reused={reused} pos={model.pos} checkpoints={model.checkpoint_positions()}")

        if WITH_TRACE:
            log("[2] decode 24 tokens + capture the trace the server has live at turn 2")
            model.start_decode(int(l1[0, -1].argmax()))
            model.decode_step_eager()
            model.capture_decode_trace()
            for _ in range(23):
                model.decode_step_traced()
            log(f"[2] pos={model.pos} trace_id={model.trace_id}")

        p2 = torch.cat([p1[:, :DIV], torch.randint(0, args.vocab_size, (1, T2 - DIV))], dim=1)
        if SAMPLE:
            samp()  # server.py calls enable_sampling per request, BEFORE the prefill
        log(f"[3] resumed prefill T={T2} (divergence {DIV}, worker thread={THREAD})")
        t0, out = time.time(), {}

        def work():
            out["r"] = model.prefill_reuse(p2, p1[0].tolist())

        if THREAD:  # FastAPI runs sync endpoints in anyio's threadpool, not on the capturing thread
            th = threading.Thread(target=work, daemon=True)
            th.start()
            th.join(timeout=120)
            if th.is_alive():
                log("[3] HUNG -- reproduced. Take a py-spy dump of this pid before killing it.")
                os._exit(3)
        else:
            work()
        log(f"[3] reused={out['r'][1]} pos={model.pos} in {time.time() - t0:.2f}s")
        log("OK -- did not reproduce")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
