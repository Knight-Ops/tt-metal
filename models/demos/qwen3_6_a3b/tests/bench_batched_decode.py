# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Throughput of the scan-based batched multi-user decode vs batch size B.

For each B: prefill one prompt, replicate to B slots (setup_decode_batch), compile + capture the decode
trace (greedy on-device select is already B-compatible), and time pure execute_trace replays. Each step
produces B tokens, so aggregate tok/s = B * 1000 / step_ms; per-user tok/s = 1000 / step_ms. Confirms the
measured amortization (MoE dense flat, projections ~32x, GDN recurrence batching-resistant) -> ~3x aggregate.

Run: QWEN36_LAYERS=40 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/bench_batched_decode.py
"""
from __future__ import annotations

import os
import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tests.bench_decode import build_model

# Default sweep includes B=16/32: the scan-based batched decode OOMs there at 40L (it pins B× GDN
# intermediates across all layers in the trace), but the fused GDN kernel (QWEN36_GDN_BATCH_FUSED=1,
# on-core per-user state) removes that ceiling. Override with QWEN36_BENCH_BS=1,2,4,8,16,32.
BS = [int(x) for x in os.environ.get("QWEN36_BENCH_BS", "1,2,4,8,16,32").split(",")]


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=300_000_000)
    try:
        ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
        n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))
        model, args = build_model(mesh, n_layers, ckpt, int(os.environ.get("QWEN36_MAX_SEQ", "128")), seq=8)
        torch.manual_seed(0)
        prompt = torch.randint(0, args.vocab_size, (1, 16))
        first = int(model.forward(prompt)[0, -1].argmax())

        print(f"\n=== batched decode throughput ({n_layers} layers, traced, greedy) ===")
        print(f"{'B':>4}{'step ms':>10}{'per-user tok/s':>16}{'aggregate tok/s':>18}{'vs B=1':>9}")
        base = None
        for B in BS:
            try:
                model.caches = None  # reallocate [1,...] caches, then re-prefill and re-batch
                model._pf_caches_ready = False
                model.trace_id = None  # prior trace already released at end of last iter
                model.forward(prompt)
                model.setup_decode_batch([first] * B)
                model.decode_step_eager()  # compile kernels for capture
                model.capture_decode_trace()
                for _ in range(3):
                    ttnn.execute_trace(mesh, model.trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                t0 = time.time()
                N = 50
                for _ in range(N):
                    ttnn.execute_trace(mesh, model.trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                step_ms = (time.time() - t0) / N * 1e3
                agg = B * 1000.0 / step_ms
                base = base or agg
                print(f"{B:>4}{step_ms:>10.2f}{1000.0/step_ms:>16.2f}{agg:>18.1f}{agg/base:>8.2f}x")
                ttnn.release_trace(mesh, model.trace_id)
            except Exception as e:
                print(f"{B:>4}  [failed: {type(e).__name__}: {str(e)[:80]}]")
                break
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
