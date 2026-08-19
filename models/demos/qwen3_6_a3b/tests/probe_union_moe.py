# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-0 probe: the SMALL-M SPARSE-UNION MoE cost surface — the shared linchpin for BOTH batched
multi-user decode AND an MTP/speculative-decode revival.

The union-MoE (moe.py `_experts_for_tile`) computes, for a 32-row tile, ONLY the experts in the tile's
union mask (nonzero routing across the ≤32 tokens) via `ttnn.sparse_matmul(nnz=None)` — it iterates the
256 expert slots but skips the DRAM reads of zero-union experts, so its cost is a function of the UNION
SIZE U (unique active experts), not of the 256 total. Crucially the sparse program config uses
`per_core_M = max(32, m)//32 = 1`, so 1..32 token rows cost the SAME — this probe measures whether the
cost is U-bound (expert weight-reads / per-block handshake) or M-bound (token rows), which decides:
  * BATCHED decode: pack up to 32 users into one tile; per-token cost = cost(U(B)) / min(B,32). Union
    beats dense-256 (~3.0 ms/step) when the batch union is well below 256.
  * MTP verify: one sequence's (γ+1) tokens in one tile; verify-MoE = cost(U≈8(γ+1)) × 40 layers vs a
    decode step's MoE (~12.8 ms = 0.32×40). Prior MTP Stage-0 found ~1.0 ms/layer @ U~24 → ~40 ms =
    ~3× a decode step = the NO-GO. This re-measures freshly and extends the U range.

Synthetic real-SHAPE weights (E=256, hidden=2048, inter=512, gate_up fused 2I=1024, bf4) — MoE cost is
value-independent, so no model load. Mirrors `_experts_for_tile` exactly (same ops, same _sparse_pc).
"""
from __future__ import annotations

import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.moe import TtMoE

E, H, I = 256, 2048, 512  # experts, hidden, moe_intermediate
DECODE_MOE_MS = 12.8  # a full decode step's MoE (0.32 ms/op × 40 layers), the MTP comparison base


def time_traced(mesh, fn, iters=300, warmup=5):
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


def make_sparsity(mesh, U, M):
    """[1,1,1,E] bf16 ROW_MAJOR union mask with the first U experts active, and [E,M,1] weights."""
    s = torch.zeros(E)
    s[:U] = 1.0 / max(U, 1)
    sparsity = ttnn.from_torch(s.reshape(1, 1, 1, E), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
    wt_t = torch.zeros(E, M, 1)
    wt_t[:U] = 1.0 / max(U, 1)
    wt = ttnn.from_torch(wt_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    return sparsity, wt


def _split_last(t, half):
    """Rank-generic last-dim split (mirror of moe.py TtMoE._split_last)."""
    sh = list(t.shape)
    r = len(sh)
    lo = ttnn.slice(t, [0] * r, sh[:-1] + [half])
    hi = ttnn.slice(t, [0] * (r - 1) + [half], list(sh))
    return lo, hi


def experts_tile(x4, gate_up_sp, down_sp, sparsity, wt, M):
    """Mirror of moe.py `_experts_for_tile` (union path), M rows (padded internally to one 32-tile)."""
    gu = ttnn.sparse_matmul(
        x4,
        gate_up_sp,
        sparsity=sparsity,
        nnz=None,
        program_config=TtMoE._sparse_pc(M, 2 * I),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    gate, up = _split_last(gu, I)
    h = ttnn.multiply(ttnn.silu(gate), up)
    h = ttnn.reshape(h, [1, E, M, I])
    down = ttnn.sparse_matmul(
        h,
        down_sp,
        sparsity=sparsity,
        nnz=None,
        is_input_a_sparse=True,
        program_config=TtMoE._sparse_pc(M, H),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    down = ttnn.reshape(down, [E, M, H])
    return ttnn.sum(ttnn.multiply(down, wt), dim=0)


def expected_union(B):
    """Expected unique experts across B users w/ random top-8 of 256."""
    return E * (1 - (1 - 8 / E) ** B)


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=300_000_000)
    try:
        # synthetic real-shape expert weights (bf4), built once
        gate_up_sp = ttnn.from_torch(
            torch.randn(1, E, H, 2 * I) * 0.02,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        down_sp = ttnn.from_torch(
            torch.randn(1, E, I, H) * 0.02,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        print("\n=== union-MoE cost vs U (one 32-row tile) — the shared linchpin ===")
        print(f"{'U':>5}{'ms/tile':>10}{'x40 (ms/step)':>16}{'vs decode-MoE 12.8ms':>22}")
        surface = {}
        for U in [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]:
            M = 32
            x4 = ttnn.from_torch(
                torch.randn(1, 1, M, H) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
            )
            sparsity, wt = make_sparsity(mesh, U, M)
            try:
                us = time_traced(mesh, lambda: experts_tile(x4, gate_up_sp, down_sp, sparsity, wt, M))
            except Exception as e:
                print(f"{U:>5}  [FAIL {type(e).__name__}: {str(e)[:70]}]")
                continue
            surface[U] = us / 1000.0  # ms
            print(f"{U:>5}{us/1000.0:>10.3f}{us/1000.0*40:>16.1f}{us/1000.0*40/DECODE_MOE_MS:>21.2f}x")

        # tile-linearity: fixed U, M=32 (1 tile) vs 64 (2 tiles) — is cost per-tile-linear or per-row?
        print("\n=== M-scaling at fixed U=24 (1 tile=32 rows vs 2 tiles=64 rows) ===")
        for M in [32, 64]:
            ntile = M // 32
            x4 = ttnn.from_torch(
                torch.randn(1, 1, 32, H) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
            )
            sparsity, wt = make_sparsity(mesh, 24, 32)
            us = time_traced(
                mesh, lambda: [experts_tile(x4, gate_up_sp, down_sp, sparsity, wt, 32) for _ in range(ntile)][-1]
            )
            print(f"  M={M:>3} ({ntile} tile): {us/1000.0:.3f} ms  -> {us/1000.0/ntile:.3f} ms/tile")

        # ---- economics ----
        def cost(U):  # nearest measured
            ks = sorted(surface)
            k = min(ks, key=lambda z: abs(z - U))
            return surface.get(k, float("nan"))

        print("\n=== BATCHED decode economics (union packs up to 32 users/tile) ===")
        print(f"  single-user sparse decode MoE ~= 0.32 ms/step; dense-256 ~= 3.0 ms/step (flat)")
        print(f"{'B':>4}{'E[U]':>7}{'union ms/step':>15}{'per-token ms':>14}{'vs dense/B':>12}{'vs sparse .32':>15}")
        for B in [4, 8, 16, 32]:
            U = expected_union(B)
            step_ms = cost(U) * 40  # one tile handles min(B,32) users; ×40 layers
            per_tok = step_ms / min(B, 32)
            dense_pt = 3.0 / B
            print(f"{B:>4}{U:>7.0f}{step_ms:>15.1f}{per_tok:>14.3f}{dense_pt:>12.3f}{per_tok/0.32:>14.2f}x")

        print("\n=== MTP verify economics (one sequence, γ+1 tokens/tile, U≈8(γ+1)) ===")
        print(f"{'γ':>4}{'γ+1 tok':>9}{'E[U]':>7}{'verify-MoE ms':>15}{'x decode-MoE':>14}")
        for g in [1, 2, 3, 4]:
            m = g + 1
            U = E * (1 - (1 - 8 / E) ** m)  # union of m tokens
            vms = cost(U) * 40
            print(f"{g:>4}{m:>9}{U:>7.0f}{vms:>15.1f}{vms/DECODE_MOE_MS:>13.2f}x")
        print("\n  (MTP verify-MoE alone must be << (γ+1)×accept-benefit; prior NO-GO was ~3× a decode step.)")
        print("  GO for the union kernel build if: batched per-token << 0.32 at useful B, OR the cost is")
        print("  clearly M-bound (ragged tile<32 would then cut MTP verify). NO-GO if U-bound & already flat.")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
