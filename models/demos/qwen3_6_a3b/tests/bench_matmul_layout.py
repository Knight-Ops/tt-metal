# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Directly test the memory-layout hypothesis for DECODE (T=1) projection matmuls.

For each real Qwen3.6 decode matmul (weight interleaved in DRAM, activation 1 row), time it
traced under several layouts and compare to the analytical DRAM-bandwidth floor:
  A) interleaved-DRAM weight, interleaved-DRAM out  (CURRENT production path)
  B) interleaved-DRAM weight, L1 out
  C) DRAM-SHARDED weight (width-sharded across banks) + L1 activation + 1D multicast config
If A is already near the BW floor, memory layout is NOT a lever. If C is much faster than A,
it is. Weight bytes are the real .tensorbin sizes; DRAM peak = 432 GB/s (tt-npe P150 model).
"""
from __future__ import annotations

import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.common import build_dram_shard

DRAM_PEAK_GBs = 432.0  # 8 controllers x 54 GB/s (tt-npe blackhole P150 model)

# (name, K=in, N=out, weight_dtype, bytes) — real decode projections
BF8 = ttnn.bfloat8_b
BF16 = ttnn.bfloat16
CASES = [
    ("gdn.w_qkv", 2048, 8192, BF8),
    ("gdn.w_z", 2048, 4096, BF8),
    ("gdn.w_out", 4096, 2048, BF8),
    ("attn.wq", 2048, 4096, BF8),
    ("attn.wgate", 2048, 4096, BF8),
    ("attn.wo", 4096, 2048, BF8),
    (
        "moe.se_down",
        512,
        2048,
        BF16,
    ),  # NOTE: floor_us/util% are bf8-based (undercounts bf16 bytes); use A vs C directly
]


def bytes_bf8(K, N):
    # bf8_b tile: 16x16 datums? tt stores 32x32 tiles, 1 byte/datum + 4 bytes exponent per 16 datums.
    # empirically ~1.0625 B/elem. Use measured ratio from tensorbin instead where possible.
    return int(K * N * 1.0625)


def time_traced(mesh, fn, iters=500, warmup=5):
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
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        g = mesh.compute_with_storage_grid_size()
        grid = (g.x, g.y)
        print(f"grid {grid}, DRAM peak {DRAM_PEAK_GBs} GB/s\n")
        print(f"{'case':14s}{'floor_us':>9}{'A_intl':>9}{'B_L1out':>9}{'C_dramsh':>10}{'A_util%':>9}{'C_util%':>9}")
        for name, K, N, wdt in CASES:
            wt = torch.randn(K, N) * 0.02
            x = ttnn.from_torch(torch.randn(1, K) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            w_dram = ttnn.from_torch(
                wt, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            floor = bytes_bf8(K, N) / (DRAM_PEAK_GBs * 1e9) * 1e6

            # A: interleaved weight, interleaved out (current)
            a = time_traced(mesh, lambda: ttnn.linear(x, w_dram))
            # B: interleaved weight, L1 out
            b = time_traced(mesh, lambda: ttnn.linear(x, w_dram, memory_config=ttnn.L1_MEMORY_CONFIG))

            # C: DRAM width-sharded weight + L1 width-sharded activation + DRAM-sharded matmul config
            # via the PRODUCTION helper (tt.common.build_dram_shard) so the microbench matches exactly
            # what the model runs (tt_transformers decode pattern: in0 sharded across banks==cores). The
            # output MUST use the sharded out_mc the helper returns (an interleaved out fails validation).
            c = float("nan")
            try:
                w_dsh, act_mc, out_mc, pc = build_dram_shard(w_dram, K, N, 8)
                x_sh = ttnn.to_memory_config(x, act_mc)
                c = time_traced(
                    mesh,
                    lambda: ttnn.linear(x_sh, w_dsh, program_config=pc, memory_config=out_mc),
                )
            except Exception as e:
                print(f"  [C failed for {name}: {type(e).__name__}: {str(e)[:110]}]")

            print(
                f"{name:14s}{floor:9.1f}{a:9.1f}{b:9.1f}{c:10.1f}"
                f"{floor/a*100:9.1f}{(floor/c*100 if c==c else 0):9.1f}"
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
