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
from models.demos.qwen3_6_a3b.tt import attention as attn_mod
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
    ap.add_argument(
        "--no-trace",
        action="store_true",
        help="skip prefill trace capture and report eager only. Prefer raising QWEN36_TRACE_REGION (MiB) "
        "instead -- long-T capture needs ~1400 vs the 600 default. Use this when you do not want to "
        "spend the memory: PREFILL.md's headline table is eager anyway, and tracing prefill is worth "
        "only ~1.02x since the GDN fused op landed",
    )
    ap.add_argument(
        "--kchunks",
        type=str,
        default="",
        help="sweep QWEN36_PREFILL_KCHUNK in-process, e.g. 64,128,256,512. One model build for all of "
        "them, so the comparison is paired. Only affects T > QWEN36_LONG_PREFILL_THRESHOLD, where "
        "forward() streams chunks through forward_prefill_incremental",
    )
    a = ap.parse_args()
    seqs = [int(s) for s in a.seqs.split(",")]
    kchunks = [int(k) for k in a.kchunks.split(",")] if a.kchunks else None
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))
    # Every other entry point in this tree makes the trace region configurable (demo/server.py reads
    # QWEN36_TRACE_REGION in MiB, evaluation/ reads QWEN36_TRACE_REGION_MB); this script hardcoded it,
    # which made long-T tracing look impossible when it is only unbudgeted. A prefill trace's command
    # stream scales with CHUNK COUNT, so it grows fast: capturing 4096+8192+32768 together asks for
    # 1401 MB against the 600 MB default. Weights (~20.3 GB at 40L) plus KV leave ~10 GB free, so a
    # couple of GB here is affordable -- raise it rather than dropping to --no-trace.
    # NB this is a CAPACITY knob only. The multi-bucket prefill CORRUPTION (trace address aliasing,
    # fixed in 365180d31c1) was explicitly not a footprint problem and raising this never helped it.
    trace_region_mb = int(os.environ.get("QWEN36_TRACE_REGION", "600"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=trace_region_mb * 1024 * 1024)
    try:
        args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=max(512, max(seqs)))
        t0 = time.time()
        model = TtModel(mesh, args, CheckpointLoader(ckpt), num_layers=n_layers)
        print(f"[bench] {n_layers}-layer model built in {time.time()-t0:.0f}s; seqs={seqs}")
        torch.manual_seed(0)
        ids = {L: torch.randint(0, args.vocab_size, (1, L)) for L in seqs}

        if kchunks:
            # Paired k_chunk sweep: same weights, same process, same device state. The setting is a
            # module global rather than a per-call argument so the model does not need rebuilding.
            saved = attn_mod._PREFILL_KCHUNK_MAX
            rows = {}
            try:
                for kc in kchunks:
                    attn_mod._PREFILL_KCHUNK_MAX = kc
                    rows[kc] = {}
                    for L in seqs:
                        # A larger k tile can overflow the SDPA kernel's L1 at this head_dim. That is a
                        # RESULT, not a crash: record it and keep the rest of the sweep.
                        try:
                            rows[kc][L] = _warm_time(lambda L=L: model.forward(ids[L]), mesh)
                        except Exception as e:  # noqa: BLE001
                            rows[kc][L] = None
                            print(f"[bench] kchunk={kc} T={L} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    done = "  ".join(f"T={L} " + (f"{rows[kc][L]:.0f}ms" if rows[kc][L] else "ERR") for L in seqs)
                    print(f"[bench] kchunk={kc}: {done}", flush=True)
            finally:
                attn_mod._PREFILL_KCHUNK_MAX = saved
            base = kchunks[0]
            print(f"\n{'seq':>7} | " + " | ".join(f"kc={kc:<4} tok/s   vs kc={base}" for kc in kchunks))
            for L in seqs:
                cells = []
                for kc in kchunks:
                    ms, ref = rows[kc][L], rows[base][L]
                    if ms is None:
                        cells.append(f"{'ERR':>7}    {'':>6}  {'':>7}")
                    else:
                        rel = f"{ref/ms:>6.3f}x" if ref else f"{'':>7}"
                        cells.append(f"{ms:>7.0f}ms {L/ms*1000:>6.0f}  {rel}")
                print(f"{L:>7} | " + " | ".join(cells))
            print("\n  Only T > QWEN36_LONG_PREFILL_THRESHOLD (2048) uses the chunked path; shorter")
            print("  lengths go single-shot through forward_prefill and MUST read as 1.000x.")
            return

        eager = {L: _warm_time(lambda L=L: model.forward(ids[L]), mesh) for L in seqs}
        if a.no_trace:
            print(f"\n{'seq':>7} | {'eager ms':>9} {'eager tok/s':>12}")
            for L in seqs:
                print(f"{L:>7} | {eager[L]:>9.0f} {L/eager[L]*1000:>12.0f}")
            return
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
