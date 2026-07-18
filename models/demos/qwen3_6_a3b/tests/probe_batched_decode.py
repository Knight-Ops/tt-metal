# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-0 probe for BATCHED multi-user decode — validate the ~5-7x aggregate estimate before committing
to the B-axis plumbing. Measures the pieces the prior aggregate ESTIMATED but never directly measured:

  1. REAL dense-MoE at T=B (moe.forward, the T>1 dense path = all 256 experts over B rows) — the
     aggregate FLOOR/ceiling. B=1 runs the sparse decode path (0.32 ms/layer baseline); B>1 runs dense.
     Resolves whether dense per-token really falls toward ~0.09 ms at B=32.
  2. Projection amortization at M=B (synth): a decode projection matmul should go ~flat in ms as B grows
     (per-token ~1/B) — confirms the router/proj/lm_head 15-32x scaling the aggregate assumed.
  3. GDN recurrence-core bmm at B (synth): the fused T=1 kernel is M=1/head-hardcoded, so batched GDN must
     add a B axis to the recurrence matmuls ([B*nv,1,Dk] @ [B*nv,Dk,Dv]). Measures how that core scales.

Assembles a per-token cost at B=32 vs the 28.9 ms single-user baseline to sanity-check the aggregate.
NOTE: GDN/attn FULL mixer at B>1 needs the plumbing (M=1 hardcoding) — this bounds it, not exact.
Run: QWEN36_LAYERS=4 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_batched_decode.py
"""
from __future__ import annotations

import os
import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tests.bench_decode import build_model

H = 2048
BS = [1, 4, 8, 16, 32]
BASE_MS = 28.9  # single-user decode step (40L)
N_MOE, N_GDN, N_ATTN = 40, 30, 10


def time_traced(mesh, fn, iters=200, warmup=5):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.time() - t0) / iters * 1e6  # us
    ttnn.release_trace(mesh, tid)
    return dt


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=400_000_000)
    try:
        ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
        n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
        model, args = build_model(mesh, n_layers, ckpt, int(os.environ.get("QWEN36_MAX_SEQ", "128")), seq=8)
        lin_idx = next(i for i, l in enumerate(model.layers) if l.is_linear)
        moe = model.layers[lin_idx].moe

        # ---- 1. REAL MoE at batch B (B=1 sparse; B>1 dense) ----
        print("\n=== 1. REAL MoE at batch B (B=1 sparse-gather, B>1 dense-256) ===")
        print(f"{'B':>4}{'path':>8}{'ms/layer':>10}{'per-tok ms/layer':>18}{'x40 step ms':>13}{'per-tok step':>13}")
        moe_ms = {}
        for B in BS:
            x = ttnn.from_torch(
                torch.randn(1, 1, B, H) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
            )
            try:
                us = time_traced(mesh, lambda: moe.forward(x))
            except Exception as e:
                print(f"{B:>4}  [FAIL {type(e).__name__}: {str(e)[:60]}]")
                continue
            ms = us / 1000.0
            moe_ms[B] = ms
            path = "sparse" if B == 1 else "dense"
            print(f"{B:>4}{path:>8}{ms:>10.3f}{ms/B:>18.4f}{ms*N_MOE:>13.1f}{ms*N_MOE/B:>13.3f}")

        # ---- 2. projection amortization at M=B (synth interleaved decode matmul) ----
        print("\n=== 2. projection matmul amortization at M=B (per-token should ~1/B) ===")
        for name, K, N in [("attn.wq K2048 N4096", 2048, 4096), ("gdn.w_out K4096 N2048", 4096, 2048)]:
            w = ttnn.from_torch(
                torch.randn(K, N) * 0.02,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            row = []
            for B in BS:
                x = ttnn.from_torch(torch.randn(B, K) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                us = time_traced(mesh, lambda: ttnn.linear(x, w))
                row.append((B, us / 1000.0))
            print(f"  {name:24s}: " + "  ".join(f"B{B}={ms:.3f}({ms/B:.4f}/tok)" for B, ms in row))

        # ---- 3. GDN recurrence-core bmm at B: [B*nv,1,Dk] @ [B*nv,Dk,Dv] (q@S) ----
        print("\n=== 3. GDN recurrence-core bmm q@S at B ([B*32,1,128]@[B*32,128,128]) ===")
        nv, Dk, Dv = 32, 128, 128
        gdn_core = {}
        for B in BS:
            q = ttnn.from_torch(
                torch.randn(B * nv, 1, Dk) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
            )
            S = ttnn.from_torch(
                torch.randn(B * nv, Dk, Dv) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
            )
            us = time_traced(mesh, lambda: ttnn.matmul(q, S))
            gdn_core[B] = us / 1000.0
            print(f"  B={B:>3}: {us/1000.0:.4f} ms  ({us/1000.0/B:.5f}/tok)")

        # ---- aggregate sanity-check at B=32 (MoE real; GDN/attn bounded) ----
        print("\n=== aggregate sanity-check ===")
        print(f"  single-user baseline: {BASE_MS:.1f} ms/token ({1000/BASE_MS:.1f} tok/s)")
        if 32 in moe_ms:
            moe32_step = moe_ms[32] * N_MOE  # dense MoE, whole step, batch of 32
            moe32_tok = moe32_step / 32
            moe1_tok = moe_ms.get(1, float("nan")) * N_MOE  # single-user sparse MoE per token
            print(
                f"  MoE per-token: single-user {moe1_tok:.2f} ms  ->  B=32 dense {moe32_tok:.3f} ms "
                f"({moe1_tok/moe32_tok:.1f}x cheaper/token)"
            )
            print(f"  MoE-dense floor at B=32 = {moe32_step:.1f} ms/step for 32 users (dominates the aggregate).")
            print(f"  If GDN+attn+proj+head amortize well (see §2), aggregate per-token approaches the MoE floor")
            print(
                f"  {moe32_tok:.3f} ms -> ~{1000/max(moe32_tok, 0.01):.0f} tok/s aggregate ceiling (vs {1000/BASE_MS:.1f} single)."
            )
            print(f"  GO if B=32 dense MoE per-token << single-user MoE per-token AND projections amortize (§2).")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
