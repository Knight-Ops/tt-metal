# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What does PAGING the KV cache cost at decode time? Flat vs paged SDPA-decode + cache write.

Context. `TtModel._alloc_cache` allocates a FLAT `[1, n_kv, max_seq, head_dim]` cache per attention
layer and calls `paged_update_cache(..., page_table=None)` / `scaled_dot_product_attention_decode`.
Because it is flat, every slot pays `max_seq` whether it uses it or not -- measured, a 512-token prefill
at max_seq=262144 allocates **5.23 GB** of cache. Paging would make that proportional to real length,
which is the largest memory lever in the model and is on the critical path for traced incremental
prefill (`PREFILL.md` §4a) and for vLLM continuous batching (`README.md`).

The open question is whether paging costs decode LATENCY. Same bytes are read either way (decode reads
are bounded by `cur_pos`, not by `max_seq`), so any cost is the page-table indirection and a
block-strided rather than contiguous access pattern -- which should amortise once a block is large
(64 tokens x 2 kv heads x 256 x 2 B = 64 KB). Should. Measured here instead:

  * `scaled_dot_product_attention_decode`        on a flat  [1, n_kv, max_seq, hd] cache
  * `paged_scaled_dot_product_attention_decode`  on a paged [n_blocks, n_kv, block, hd] cache
  * `paged_update_cache` with page_table=None vs a real page table

across realistic positions, at this model's attention dims. Also prints the KV bytes read per step,
because at long context that -- not the paging -- is what dominates decode.

MEASURED 2026-08-23 (this model's attention dims, block=64, us per attention layer per step):

     pos   dtype   KV MB   flat us  paged us  pg/flat  wr flat  wr paged
    1024    bf16     2.0      36.2      36.1     1.00      6.9       7.4
    1024     bf8     1.1      33.8      33.3     0.98      7.0       7.4
    8192    bf16    16.0      83.2      83.1     1.00      7.0       7.4
    8192     bf8     8.5      62.4      62.7     1.01      6.9       7.4
   32768    bf16    64.0     227.7     226.1     0.99      7.0       7.5
   32768     bf8    34.0     137.4     138.9     1.01      7.0       7.4
  131072    bf16   256.0     796.7     792.0     0.99      7.0       7.7
  131072     bf8   136.0     438.8     444.7     1.01      7.0       7.6

TWO CONCLUSIONS, and they are about different things.

1. PAGING IS FREE. `pg/flat` is 0.98-1.01 at every position, and the write costs +0.4-0.7 us (over 10
   attention layers, +4-7 us on a ~26 ms step). A 64-token block is 64 KB, big enough that the
   page-table indirection amortises. So paging is a MEMORY/FLEXIBILITY change, not a speedup -- read
   `ROOFLINE.md` before treating it as one.

2. KV DTYPE IS THE REAL LEVER, and only at long context. bf16 -> BFP8 is 1.07x at 1K but **1.33x at 8K,
   1.66x at 32K and 1.82x at 128K** -- i.e. -0.02 / -0.21 / -0.90 / -3.58 ms per token across the 10
   attention layers. The byte ratio is 2 / 1.0625 = 1.88x, so at 128K the kernel is realising 96.5% of
   the ideal, which says it is doing nothing but moving the cache: there is no compute to tune and
   halving the bytes is the only available lever. See `QWEN36_KV_DTYPE`, FUTURE_OPTIMIZATIONS "Lever 6"
   for the end-to-end number (and why bench_decode.py UNDER-states it), and EXPERIMENTS.md for accuracy.

Run:  ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_paged_kv_perf.py
"""
from __future__ import annotations

import argparse
import time

import torch

import ttnn

NH, NKV, HD = 16, 2, 256  # Qwen3.6: 16 q heads, 2 kv heads, head_dim 256
BLOCK = 64  # tokens per page; what blackhole/qwen36 and tt_transformers use
DRAM_PEAK_GBs = 432.0


def time_traced(mesh, fn, iters=100, warmup=5):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="1024,8192,32768,131072")
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    positions = [int(p) for p in a.positions.split(",")]

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        g = mesh.compute_with_storage_grid_size()
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=128,
        )
        print(f"grid {g.x}x{g.y}  n_heads={NH} n_kv={NKV} head_dim={HD}  block={BLOCK}\n")
        print(
            f"{'pos':>8}{'dtype':>8}{'KV MB':>8}{'flat us':>10}{'paged us':>10}{'pg/flat':>9}"
            f"{'wr flat':>9}{'wr paged':>10}"
        )
        torch.manual_seed(0)
        q = ttnn.from_torch(torch.randn(1, 1, NH, HD) * 0.3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        for pos, KVDT in [(p, d) for p in positions for d in (ttnn.bfloat16, ttnn.bfloat8_b)]:
            max_seq = pos * 2  # a realistic over-allocation for the flat form
            nblk = max_seq // BLOCK
            bpe = 2.0 if KVDT == ttnn.bfloat16 else 1.0625
            kv_mb = 2 * pos * NKV * HD * bpe / 2**20  # K+V at the cache dtype, rows 0..pos
            cur = ttnn.from_torch(
                torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
            )
            # paged_update_cache TT_FATALs unless the update tensor is SHARDED -- mirror the model's
            # `_kv_update_mc` (HEIGHT_SHARDED L1, one core, shard [TILE_SIZE, head_dim]).
            one_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
            upd_mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(one_core, [ttnn.TILE_SIZE, HD], ttnn.ShardOrientation.ROW_MAJOR),
            )
            upd = ttnn.from_torch(
                torch.zeros(1, 1, ttnn.TILE_SIZE, HD),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=upd_mc,
            )
            row = {}
            flat = [
                ttnn.zeros([1, NKV, max_seq, HD], dtype=KVDT, layout=ttnn.TILE_LAYOUT, device=mesh) for _ in range(2)
            ]
            try:
                row["flat"] = time_traced(
                    mesh,
                    lambda: ttnn.transformer.scaled_dot_product_attention_decode(
                        q, flat[0], flat[1], cur_pos_tensor=cur, scale=HD**-0.5, program_config=pc
                    ),
                    a.iters,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [flat sdpa pos={pos} FAILED] {type(e).__name__}: {str(e)[:260]}")
                row["flat"] = "ERR"
            try:
                row["wr_flat"] = time_traced(
                    mesh,
                    lambda: ttnn.experimental.paged_update_cache(flat[0], upd, update_idxs_tensor=cur, page_table=None),
                    a.iters,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [flat write pos={pos} FAILED] {type(e).__name__}: {str(e)[:260]}")
                row["wr_flat"] = "ERR"
            for t in flat:
                ttnn.deallocate(t)
            paged = [
                ttnn.zeros([nblk, NKV, BLOCK, HD], dtype=KVDT, layout=ttnn.TILE_LAYOUT, device=mesh) for _ in range(2)
            ]
            # identity page table: block i of the sequence lives in physical block i
            pt = ttnn.from_torch(
                torch.arange(nblk, dtype=torch.int32).reshape(1, nblk),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
            )
            try:
                row["paged"] = time_traced(
                    mesh,
                    lambda: ttnn.transformer.paged_scaled_dot_product_attention_decode(
                        q,
                        paged[0],
                        paged[1],
                        cur_pos_tensor=cur,
                        page_table_tensor=pt,
                        scale=HD**-0.5,
                        program_config=pc,
                    ),
                    a.iters,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [paged sdpa pos={pos} FAILED] {type(e).__name__}: {str(e)[:260]}")
                row["paged"] = "ERR"
            try:
                row["wr_paged"] = time_traced(
                    mesh,
                    lambda: ttnn.experimental.paged_update_cache(paged[0], upd, update_idxs_tensor=cur, page_table=pt),
                    a.iters,
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [paged write pos={pos} FAILED] {type(e).__name__}: {str(e)[:260]}")
                row["wr_paged"] = "ERR"
            for t in paged:
                ttnn.deallocate(t)

            def c(v):
                return f"{v:10.1f}" if isinstance(v, float) else f"{str(v)[:10]:>10}"

            ratio = (
                (row["paged"] / row["flat"])
                if all(isinstance(row[k], float) for k in ("paged", "flat"))
                else float("nan")
            )
            dn = "bf16" if KVDT == ttnn.bfloat16 else "bf8"
            print(
                f"{pos:>8}{dn:>8}{kv_mb:>8.1f}{c(row['flat'])}{c(row['paged'])}{ratio:>9.2f}"
                f"{c(row['wr_flat'])[-9:]}{c(row['wr_paged'])}"
            )
        print("\nNote: 'KV MB read' is what a decode step must read at that position. Compare it to the")
        print("2.6 GB of WEIGHTS per token -- past ~64K the cache, not the weights, sets decode latency,")
        print("and that is true of the flat cache too. Paging changes allocation, not that.")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
