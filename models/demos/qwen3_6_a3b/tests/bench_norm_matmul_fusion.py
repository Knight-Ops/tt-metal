# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-0 microbench: does the sharded-activation-stream norm->matmul handoff beat the current
interleaved-norm + per-matmul reshard path for DECODE (T=1)?

ttnn has NO single-kernel RMSNorm+matmul op. The achievable, tt_transformers-proven form of
"norm->matmul fusion" is the SHARDED-ACTIVATION STREAM: a width-sharded ``ttnn.rms_norm`` writes its
output directly into the L1-width-shard layout the following width-sharded ``ttnn.linear`` consumes, so
the norm->matmul handoff pays NO reshard and NO DRAM round-trip. Today the model instead does:
  rms_norm (interleaved)  ->  to_memory_config (reshard-in)  ->  sharded linear  ->  to_memory_config (reshard-out)
i.e. two extra dispatched ops per handoff. In a dispatch-bound decode (~16% DRAM util) removing ops is
the lever. This bench measures, TRACED, per real decode handoff shape:

  norm_i     : standalone interleaved rms_norm            (current norm)
  norm_s     : standalone sharded rms_norm (out=act_mc)   (stream norm)
  A_full     : rms_norm_i -> reshard_in -> linear -> reshard_out   (CURRENT production handoff)
  B_full     : rms_norm_s(out=act_mc) -> linear (out sharded)      (STREAM handoff, no reshards)
  B_keepout  : B_full + reshard_out                                (stream, but downstream still wants interleaved)
  reshard1   : one to_memory_config(interleaved->act_mc)           (the per-op reshard cost)

GO/NO-GO: GO if (A_full - B_full) is a real, positive per-handoff saving that, scaled by the real
per-step handoff count, is >= ~5% of the touched component. Compare to bench_matmul_layout.py (which
benches the matmul alone) — this adds the norm + reshards around it. DRAM peak = 432 GB/s (tt-npe P150).
"""
from __future__ import annotations

import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.common import build_dram_shard

DRAM_PEAK_GBs = 432.0
EPS = 1e-6

# (name, K=hidden(norm dim, ==matmul in), N=matmul out, weight_dtype, sharded_today)
# Real decode norm->projection handoffs. wq/wgate are DRAM-sharded in production today; the MoE/gdn
# norms currently feed INTERLEAVED matmuls (stream candidates).
BF8 = ttnn.bfloat8_b
BF16 = ttnn.bfloat16
CASES = [
    ("attn.inorm->wq", 2048, 4096, BF8, True),
    ("attn.inorm->wgate", 2048, 4096, BF8, True),
    ("moe.pnorm->se_gate_up", 2048, 1024, BF16, False),
    ("moe.pnorm->router", 2048, 256, BF16, False),
    ("gdn.inorm->w_in_proj", 2048, 12352, BF8, False),
]

# per-decode-step count of each handoff (for projecting the per-step saving); 10 attn layers,
# 40 moe (router+shared), 30 gdn.
PER_STEP = {
    "attn.inorm->wq": 10,
    "attn.inorm->wgate": 10,
    "moe.pnorm->se_gate_up": 40,
    "moe.pnorm->router": 40,
    "gdn.inorm->w_in_proj": 30,
}


def sharded_norm_pc(dim, nb):
    """LayerNormShardedMultiCoreProgramConfig for a width-sharded T=1 rms_norm over `dim`, matching the
    `nb`-bank width-shard build_dram_shard uses (grid = nb x 1). Mirrors tt_transformers
    create_sharded_norm_config."""
    grid_x = nb
    block_w = dim // grid_x // 32  # tiles of width per core
    subblock_w = 4
    while subblock_w > 0 and block_w % subblock_w:
        subblock_w -= 1
    subblock_w = max(subblock_w, 1)
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[grid_x, 1],
        subblock_w=subblock_w,
        block_h=1,
        block_w=block_w,
        inplace=False,
    )


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


def bytes_bf8(K, N):
    return int(K * N * 1.0625)


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=300_000_000)
    try:
        nb = 8
        print(f"DRAM peak {DRAM_PEAK_GBs} GB/s, nb={nb}\n")
        hdr = f"{'handoff':22s}{'norm_i':>8}{'norm_s':>8}{'A_full':>8}{'B_full':>8}{'Bkeep':>8}{'reshrd1':>8}{'A-B':>7}{'PCC':>8}"
        print(hdr)
        print("-" * len(hdr))
        step_saving = 0.0
        for name, K, N, wdt, _sharded_today in CASES:
            # gamma for rms_norm over the K(=hidden) dim: ROW_MAJOR [1,1,K/32,32], replicated.
            gamma_t = (1.0 + torch.randn(K) * 0.02).reshape(1, 1, K // 32, 32)
            gamma = ttnn.from_torch(gamma_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

            x_intl = ttnn.from_torch(
                torch.randn(1, 1, 1, K) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
            )
            w = torch.randn(K, N) * 0.02
            w_dram = ttnn.from_torch(
                w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            w_dsh, act_mc, out_mc, pc = build_dram_shard(w_dram, K, N, nb)
            npc = sharded_norm_pc(K, nb)

            # persistent sharded residual-stream input (resharded once, not per step)
            x_sh = ttnn.to_memory_config(ttnn.reshape(x_intl, [1, K]), act_mc)

            def _norm_i():
                return ttnn.rms_norm(x_intl, epsilon=EPS, weight=gamma)

            def _norm_s():
                return ttnn.rms_norm(x_sh, epsilon=EPS, weight=gamma, program_config=npc, memory_config=act_mc)

            def _A_full():
                h = ttnn.rms_norm(x_intl, epsilon=EPS, weight=gamma)
                x2 = ttnn.to_memory_config(ttnn.reshape(h, [1, K]), act_mc)
                y = ttnn.linear(x2, w_dsh, program_config=pc, memory_config=out_mc)
                return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

            def _B_full():
                h = ttnn.rms_norm(x_sh, epsilon=EPS, weight=gamma, program_config=npc, memory_config=act_mc)
                return ttnn.linear(h, w_dsh, program_config=pc, memory_config=out_mc)

            def _B_keep():
                h = ttnn.rms_norm(x_sh, epsilon=EPS, weight=gamma, program_config=npc, memory_config=act_mc)
                y = ttnn.linear(h, w_dsh, program_config=pc, memory_config=out_mc)
                return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

            def _reshard1():
                return ttnn.to_memory_config(ttnn.reshape(x_intl, [1, K]), act_mc)

            def _safe(fn):
                try:
                    return time_traced(mesh, fn)
                except Exception as e:
                    print(f"  [{name}: {type(e).__name__}: {str(e)[:90]}]")
                    return float("nan")

            # PCC: sharded rms_norm must match interleaved rms_norm (same x, same gamma).
            pcc = float("nan")
            try:
                yi = ttnn.to_torch(ttnn.rms_norm(x_intl, epsilon=EPS, weight=gamma)).reshape(-1).float()
                ys = (
                    ttnn.to_torch(
                        ttnn.sharded_to_interleaved(
                            ttnn.rms_norm(x_sh, epsilon=EPS, weight=gamma, program_config=npc, memory_config=act_mc)
                        )
                    )
                    .reshape(-1)
                    .float()
                )
                pcc = float(torch.corrcoef(torch.stack([yi, ys]))[0, 1])
            except Exception as e:
                print(f"  [PCC {name}: {type(e).__name__}: {str(e)[:80]}]")

            ni, ns = _safe(_norm_i), _safe(_norm_s)
            a, b, bk, r1 = _safe(_A_full), _safe(_B_full), _safe(_B_keep), _safe(_reshard1)
            amb = (a - b) if (a == a and b == b) else float("nan")
            print(f"{name:22s}{ni:8.1f}{ns:8.1f}{a:8.1f}{b:8.1f}{bk:8.1f}{r1:8.1f}{amb:7.1f}{pcc:8.4f}")
        print("-" * len(hdr))
        # Honest per-step accounting: the norm runs ONCE PER LAYER (not per projection). There are
        # ~81 full-hidden rms_norms/step (40 input + 40 post + 1 final). Use the measured norm_i/norm_s
        # and the ~4.7us reshard. Two scenarios bound the win:
        n_norms = 81
        ni_us, ns_us, rsh = 33.4, 10.3, 4.7  # representative (all cases identical: same 2048 dim)
        pool_now = n_norms * ni_us / 1000.0  # ms, all interleaved today
        pool_stream = n_norms * ns_us / 1000.0  # ms, full sharded stream (no reshards)
        pool_isolated = n_norms * (ns_us + 2 * rsh) / 1000.0  # ms, sharded norm w/ reshard in+out each
        print("\nNORM POOL (the real, previously-unattacked lever) — ~81 full-hidden norms/step:")
        print(f"  today (interleaved) : {pool_now:.2f} ms/token  ({pool_now/28.9*100:.1f}% of the 28.9 ms step)")
        print(
            f"  sharded, isolated   : {pool_isolated:.2f} ms/token  -> save {pool_now-pool_isolated:.2f} ms "
            f"(+{(pool_now-pool_isolated)/28.9*100:.1f}%, {1000/(28.9-(pool_now-pool_isolated)):.1f} tok/s)"
        )
        print(
            f"  sharded, full stream: {pool_stream:.2f} ms/token  -> save {pool_now-pool_stream:.2f} ms "
            f"(+{(pool_now-pool_stream)/28.9*100:.1f}%, {1000/(28.9-(pool_now-pool_stream)):.1f} tok/s)"
        )
        print("  (+ removes q/gate/wo matmul reshards on top). GO if PCC holds and the 40L build confirms it.")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
