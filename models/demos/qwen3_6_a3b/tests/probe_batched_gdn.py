# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-0 probe: can the fused GDN decode kernel (decode_step_tt) take a B (multi-user) axis, at what cost?

FEASIBILITY (from reading tt/ttl_delta.py `_decode_step`): the kernel grids 32 value-heads onto 32 cores,
one head/core, ct=1 (one token). B users × 32 heads ≫ available cores, and each user has an INDEPENDENT
recurrent state S, so the recurrence `k@S`/`q@S` cannot collapse into one shared-weight matmul — each core
must LOOP its head over B users (B sequential rank-1 updates). So a B axis is feasible but the recurrence is
inherently linear in B (batching-resistant); the fused kernel's advantage over the ttnn scan is dispatch
avoidance (one launch, on-core state), not parallelism.

This probe measures the FULL batched recurrence in the SCAN form (ttnn ops, the fallback batched GDN would
use without a kernel rewrite) vs B, to get the concrete cost and bracket the looped-fused-kernel ceiling.
Recurrence per (user,head), batched over B*nv: Sg=S*g; kv=k@Sg; delta=(v-kv)*beta; outer=kᵀ@delta;
Snew=Sg+outer; core=q@Snew. nv=32, Dk=Dv=128 (real dims). Synthetic — no model load.
"""
from __future__ import annotations

import time

import torch

import ttnn

NV, DK, DV = 32, 128, 128
BS = [1, 4, 8, 16, 32]
GDN_LAYERS = 30
FUSED_B1_MS = 0.323  # measured single-user fused GDN /layer (the kernel we'd batch)
SCAN_B1_MS = 0.51  # measured single-user pre-fusion scan /layer (fused gives ~1.6x over this)


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
    dt = (time.time() - t0) / iters * 1e6
    ttnn.release_trace(mesh, tid)
    return dt


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=400_000_000)
    try:

        def mk(*shape):
            return ttnn.from_torch(torch.randn(*shape) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)

        print("\n=== batched GDN recurrence (scan form) vs B — the batching-resistant mixer ===")
        print(f"{'B':>4}{'ms/layer':>10}{'per-tok /layer':>16}{'x30 step ms':>13}{'per-tok step':>14}")
        rec = {}
        for B in BS:
            n = B * NV
            S = mk(n, DK, DV)
            q = mk(n, 1, DK)
            k = mk(n, 1, DK)
            v = mk(n, 1, DV)
            g = mk(n, 1, DV)
            beta = mk(n, 1, DV)

            def step():
                Sg = ttnn.multiply(S, g)  # decay (g broadcast over Dk rows)
                kv = ttnn.matmul(k, Sg)  # [n,1,Dv]
                delta = ttnn.multiply(ttnn.subtract(v, kv), beta)
                kcol = ttnn.transpose(k, 1, 2)  # [n,Dk,1]
                outer = ttnn.matmul(kcol, delta)  # [n,Dk,Dv]
                Snew = ttnn.add(Sg, outer)
                core = ttnn.matmul(q, Snew)  # [n,1,Dv]
                return core

            us = time_traced(mesh, step)
            ms = us / 1000.0
            rec[B] = ms
            print(f"{B:>4}{ms:>10.3f}{ms/B:>16.4f}{ms*GDN_LAYERS:>13.1f}{ms*GDN_LAYERS/B:>14.3f}")

        print("\n=== interpretation ===")
        r1, r32 = rec.get(1, float("nan")), rec.get(32, float("nan"))
        print(
            f"  recurrence scales {r32/r1:.1f}x from B=1->32 (linear would be 32x); per-token "
            f"{'~FLAT (batching-resistant)' if r32/r1 > 20 else 'sub-linear (some amortization)'}"
        )
        print(
            f"  scan batched GDN @B=32 ~= {r32*GDN_LAYERS:.0f} ms/step for 32 users -> {r32*GDN_LAYERS/32:.2f} ms/token"
        )
        # looped-fused ceiling: fused gives ~1.6x over scan at B=1; assume it holds -> divide
        fused_ratio = SCAN_B1_MS / FUSED_B1_MS
        fused32_tok = r32 * GDN_LAYERS / 32 / fused_ratio
        print(
            f"  looped-FUSED kernel ceiling (~{fused_ratio:.1f}x over scan, dispatch avoidance): "
            f"~{fused32_tok:.2f} ms/token GDN"
        )
        print(f"\n  Combined w/ MoE-dense floor (~3.71 ms/token @B=32):")
        print(
            f"    scan-GDN aggregate  ~= {3.71 + r32*GDN_LAYERS/32:.1f} ms/token -> ~{1000/(3.71+r32*GDN_LAYERS/32):.0f} tok/s (+attn)"
        )
        print(
            f"    fused-GDN aggregate ~= {3.71 + fused32_tok:.1f} ms/token -> ~{1000/(3.71+fused32_tok):.0f} tok/s (+attn)"
        )
        print(f"  vs single-user 34.6 tok/s. GDN is the aggregate-limiting floor; the batched fused kernel")
        print(f"  (on-core B-loop) is the lever that moves ~3x -> toward ~5x. Feasible but real ttl work.")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
