# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sweep `in0_block_w` (and the core grid) for the two DECODE expert `sparse_matmul`s.

WHY. These are the largest remaining decode pool: 8-of-256 experts is 14.16 MB of BFP4 weight per
layer, 566 MB/token over 40 layers, and `ROOFLINE.md` §2.1 measures them at ~30% of DRAM peak (4.36
ms/token). Their `in0_block_w` was tuned in 2026-07 (gate_up 32, down 16) on the theory that a larger
K-block means fewer per-block multicast handshakes. The 2026-08-22 dense sweep found a second, stronger
mechanism for the same knob -- DRAM READER under-utilisation on Blackhole, where the win came from
FEWER, LARGER contiguous reads per reader rather than fewer handshakes -- and that mechanism was never
applied here. Same knob, different reason to move it, so re-sweep.

Also swept: the core grid. `TtMoE._sparse_pc` derives it from Nt with `per_core_N=1`, which pins
gate_up to 32 cores and down to 64 of 110 -- the "grid loss" that killed the union-indexed prefill
variant (PREFILL.md §2.1). This probe checks whether any legal config reaches more cores.

Synthetic weights: timing for these ops is value-independent (a fixed top_k is always active), so no
checkpoint load. Run:
    ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_sparse_in0bw.py
"""
from __future__ import annotations

import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt import moe_gather
from models.demos.qwen3_6_a3b.tt.moe import TtMoE

E, H, I, TOPK = 256, 2048, 512, 8
BPE_BF4 = 0.5625
DRAM_PEAK_GBs = 432.0


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
    dt = (time.time() - t0) / iters * 1e6
    ttnn.release_trace(mesh, tid)
    return dt


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        g = mesh.compute_with_storage_grid_size()
        print(f"grid {g.x}x{g.y}   DRAM peak {DRAM_PEAK_GBs} GB/s   top_k={TOPK}\n")
        torch.manual_seed(0)
        gate_up = ttnn.from_torch(
            (torch.randn(1, E, H, 2 * I) * 0.02), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        down = ttnn.from_torch(
            (torch.randn(1, E, I, H) * 0.02), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        x4 = ttnn.from_torch(torch.randn(1, 1, 1, H) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        # sparsity is a required-but-unread operand in indexed mode (see TtMoE._const_sparsity)
        sp = ttnn.from_torch(torch.ones(1, 1, 1, E) / E, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        topi = ttnn.from_torch(
            torch.arange(TOPK, dtype=torch.int32).reshape(1, TOPK),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        idx = moe_gather.topk_to_indices(topi, TOPK)

        h_in = ttnn.from_torch(
            torch.randn(1, TOPK, 1, I) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )

        cases = [
            ("gate_up", gate_up, x4, 2 * I, H, False, [1, 2, 4, 8, 16, 32, 64]),
            ("down", down, h_in, H, I, True, [1, 2, 4, 8, 16]),
        ]
        for name, w, in0, n, k, a_sparse, bws in cases:
            floor = TOPK * k * n * BPE_BF4 / (DRAM_PEAK_GBs * 1e9) * 1e6
            Nt = (n + 31) // 32
            cx, cy = TtMoE._grid_for(Nt)
            print(
                f"--- {name}: K={k} N={n}  Kt={(k+31)//32} Nt={Nt} -> _sparse_pc grid {cx}x{cy}"
                f" = {cx*cy} cores of {g.x*g.y}   byte floor {floor:.1f} us"
            )
            best = (1e9, None)
            for bw in bws:
                if ((k + 31) // 32) % bw:
                    continue
                pc = TtMoE._sparse_pc(1, n, in0bw=bw)
                try:
                    t = time_traced(
                        mesh,
                        lambda pc=pc: ttnn.sparse_matmul(
                            in0,
                            w,
                            sparsity=sp,
                            indices=idx,
                            is_input_a_sparse=a_sparse,
                            program_config=pc,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                        ),
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"      in0_block_w={bw:3d}   FAILED {type(e).__name__}: {str(e)[:70]}")
                    continue
                tag = "  <- current default" if bw == (32 if name == "gate_up" else 16) else ""
                print(f"      in0_block_w={bw:3d}   {t:8.1f} us   {floor/t*100:5.1f}% of peak{tag}")
                if t < best[0]:
                    best = (t, bw)
            print(f"      BEST: in0_block_w={best[1]} at {best[0]:.1f} us\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
