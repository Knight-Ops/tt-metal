# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Directly test the memory-layout hypothesis for DECODE (T=1) projection matmuls.

For each real Qwen3.6 decode matmul (weight interleaved in DRAM, activation 1 row), time it
traced under several layouts and compare to the analytical DRAM-bandwidth floor:
  A) interleaved-DRAM weight, interleaved-DRAM out  (CURRENT production path)
  B) interleaved-DRAM weight, L1 out
  C) DRAM-SHARDED weight (width-sharded across banks) + L1 activation + 1D multicast config
  D) interleaved-DRAM weight + TUNED WIDE 1D mcast_in0 config, swept over (num_cores, in0_block_w)
If A is already near the BW floor, memory layout is NOT a lever. If C is much faster than A,
it is. Weight bytes are the real .tensorbin sizes; DRAM peak = 432 GB/s (tt-npe P150 model).

VARIANT D is the point of this revision. The original sweep behind the shipped DRAM-shard decision
(FUTURE_OPTIMIZATIONS.md Lever 1) varied the DRAM-sharded core count (8/16/32/64) and correctly found
8 best -- but it never tried a TUNED 1D mcast config on the plain INTERLEAVED weight, which is what
two independent single-P150 implementations measure as the winner on this exact class of skinny (M=1)
matmul (see tt/common.py:wide_1d_decode_pc for the numbers and the mechanism). D also needs no
reshard in/out and no duplicate weight copy, so if it merely TIES C it is still the better choice.

Two knobs, swept jointly because they interact through the in0 circular buffer:
  * num_cores -- shaped WIDE-first (up to 11 columns on a P150), so a wide-short grid shortens the
    in0 multicast column.
  * in0_block_w -- only has to DIVIDE k_tiles under mcast_in0, so it can be far larger than
    k_tiles // grid_x. MuseGlimmer reports 4 K-tiles "nearly halves this projection's device time"
    by fixing DRAM-reader under-utilization on Blackhole.

CAVEAT, stated because this harness cannot see it: in0_block_w sizes a circular buffer that competes
with any resident L1 output for the same 1536 KB/core. A value that wins here can still overflow in
the full model. Always re-A/B the winner at QWEN36_LAYERS=40 before shipping it.
"""
from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.common import build_dram_shard, wide_1d_decode_pc

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
    ("moe.se_down", 512, 2048, BF16),
    # Below: decode matmuls that today run on the ttnn AUTO program config with no tuning at all.
    # se_gate_up / gate_w are bf16 because ModelArgs.mlp_weight_dtype is never read (see D5).
    ("moe.se_gate_up", 2048, 1024, BF16),
    ("moe.gate_w", 2048, 256, BF16),
    ("gdn.w_in_proj", 2048, 12352, BF8),  # the fused qkv|z|b|a projection, 30x/step
    ("attn.wk", 2048, 512, BF8),
    ("lm_head.bf4", 2048, 248320, ttnn.bfloat4_b),  # 1.23 ms/token at ~54% of peak
]

# Variant-D sweep grid. num_cores is shaped wide-first (cols = min(11, num_cores)); in0_block_w is
# skipped when it does not divide k_tiles. Keep this small enough that the whole sweep is one run.
D_CORES = [8, 22, 33, 44, 55, 66, 88, 110]
D_BLOCK_W = [1, 2, 4, 8, 16]


# Bytes per element by dtype. BFP8_B and BFP4_B are 1 resp. 0.5 mantissa bytes plus a shared 8-bit
# exponent per 16 datums -> 1.0625 and 0.5625.
_BPE = {ttnn.bfloat16: 2.0, ttnn.bfloat8_b: 1.0625, ttnn.bfloat4_b: 0.5625}


def weight_bytes(K, N, dtype):
    """DRAM bytes for a [K, N] weight at `dtype`.

    FIXED 2026-08-22: this used to be `bytes_bf8()`, which applied 1.0625 B/elem to EVERY case. That
    silently understated the floor by 1.88x for the bf16 rows (se_down / se_gate_up / gate_w) and
    OVERSTATED it by 1.89x for bf4 (lm_head) -- which made lm_head read as ">100% of DRAM peak", an
    impossible number that looked like evidence the 432 GB/s figure was wrong. It was the harness.
    Any utilization number printed by an older run of this script is wrong for non-bf8 weights."""
    return int(K * N * _BPE[dtype])


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="also print the top variant-D configs per case")
    verbose = ap.parse_args().verbose
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        g = mesh.compute_with_storage_grid_size()
        grid = (g.x, g.y)
        print(f"grid {grid}, DRAM peak {DRAM_PEAK_GBs} GB/s\n")
        print(
            f"{'case':14s}{'floor_us':>9}{'A_intl':>9}{'B_L1out':>9}{'C_dramsh':>10}{'D_wide1d':>9}"
            f"{'C_net':>9}{'A_util%':>9}{'C_util%':>9}{'D_util%':>9}   D_best(cores,bw)"
        )
        for name, K, N, wdt in CASES:
            wt = torch.randn(K, N) * 0.02
            x = ttnn.from_torch(torch.randn(1, K) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            w_dram = ttnn.from_torch(
                wt, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            floor = weight_bytes(K, N, wdt) / (DRAM_PEAK_GBs * 1e9) * 1e6

            # A: interleaved weight, interleaved out (current)
            a = time_traced(mesh, lambda: ttnn.linear(x, w_dram))
            # B: interleaved weight, L1 out
            b = time_traced(mesh, lambda: ttnn.linear(x, w_dram, memory_config=ttnn.L1_MEMORY_CONFIG))

            # C: DRAM width-sharded weight + L1 width-sharded activation + DRAM-sharded matmul config
            # via the PRODUCTION helper (tt.common.build_dram_shard) so the microbench matches exactly
            # what the model runs (tt_transformers decode pattern: in0 sharded across banks==cores). The
            # output MUST use the sharded out_mc the helper returns (an interleaved out fails validation).
            c = cn = float("nan")
            try:
                w_dsh, act_mc, out_mc, pc = build_dram_shard(w_dram, K, N, 8)
                x_sh = ttnn.to_memory_config(x, act_mc)
                c = time_traced(
                    mesh,
                    lambda: ttnn.linear(x_sh, w_dsh, program_config=pc, memory_config=out_mc),
                )

                # C_net: the SAME config timed the way the model actually calls it -- resharding the
                # activation in and the output back out. Bare C is unfair to D, which needs neither:
                # every DRAM-sharded matmul in the decode step pays this pair (~110 of them per step),
                # and C also costs a SECOND copy of the weight (the interleaved original stays alive
                # for prefill). Compare D against C_net, not C, when deciding what the model ships.
                def _c_net():
                    xs = ttnn.to_memory_config(x, act_mc)
                    o = ttnn.linear(xs, w_dsh, program_config=pc, memory_config=out_mc)
                    return ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)

                cn = time_traced(mesh, _c_net)
            except Exception as e:
                print(f"  [C failed for {name}: {type(e).__name__}: {str(e)[:110]}]")

            # D: interleaved weight + tuned WIDE 1D mcast config, swept over (num_cores, in0_block_w).
            # No reshard, no second weight copy -- so a mere tie with C already wins on footprint.
            k_tiles = (K + 31) // 32
            # mcast_in0 streams the full K from in0 to every core, so give it an L1-resident
            # activation (what tp_common.matmul_1d_decode does). One copy, hoisted out of the sweep.
            x_l1 = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            best_d, best_cfg, d_rows = float("nan"), None, []
            for nc in D_CORES:
                for bw in D_BLOCK_W:
                    if k_tiles % bw or bw > k_tiles:
                        continue  # mcast_in0 streams the full K, so in0_block_w must divide k_tiles
                    try:
                        pc = wide_1d_decode_pc(32, K, N, nc, grid_w=grid[0], in0_block_w=bw)
                        t = time_traced(
                            mesh,
                            lambda pc=pc: ttnn.linear(
                                x_l1, w_dram, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG
                            ),
                            iters=200,
                        )
                    except Exception:  # illegal shape/grid combos are expected; just skip them
                        continue
                    d_rows.append((t, nc, bw))
                    if not (best_d == best_d) or t < best_d:
                        best_d, best_cfg = t, (nc, bw)

            print(
                f"{name:14s}{floor:9.1f}{a:9.1f}{b:9.1f}{c:10.1f}{best_d:9.1f}{cn:9.1f}"
                f"{floor/a*100:9.1f}{(floor/c*100 if c==c else 0):9.1f}"
                f"{(floor/best_d*100 if best_d==best_d else 0):9.1f}"
                f"   {best_cfg if best_cfg else '-'}"
            )
            if verbose and d_rows:
                for t, nc, bw in sorted(d_rows)[:6]:
                    print(f"{'':16s}D  cores={nc:3d} in0_block_w={bw:2d}  {t:8.1f} us  ({floor/t*100:5.1f}% peak)")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
