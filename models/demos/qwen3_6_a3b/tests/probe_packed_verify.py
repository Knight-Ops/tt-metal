# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Packed verify: can K speculative rows be ONE sdpa_decode instead of K?

`Attention.forward_verify` (tt/attention.py:349) loops the K speculative rows, paying per row
2x `paged_update_cache` + 1x `sdpa_decode` -- 3K ops per attention layer per verify. Its docstring
records WHY: `sdpa_decode` takes its output batch from the KV cache's batch dim, so a batch-1 cache
returns one row, and `share_cache=True` cannot express K rows over one sequence.

That is true of the FLAT op. It is not true of the PAGED one. `sdpa_decode_device_operation.cpp:196`
takes the batch from the PAGE TABLE instead:

    const auto B = page_table_tensor.is_sharded() ? q_shape[1] : page_table_tensor.padded_shape()[0];

so a page table with K IDENTICAL rows presents K "users" that all read the same physical pages, each
masked by its own `cur_pos`. That is the trick Muse Glimmer's packed verifier uses
(`tt/attention/decode.py`: "Treat packed positions as decode users which all map to the same physical
pages"), and it needs no extra KV memory.

The same reading applies to the WRITE: `paged_update_cache` dispatches one user per core, so K users
with identical page-table rows write K rows of one sequence in one op -- IF row-granular writes into a
shared 32-row tile are safe. Glimmer did not risk it; they hand-wrote a NOC kernel whose comment says
"each core serializes its row writes, so several packed rows can safely update the same physical cache
tile", implying cross-core row writes were the thing to avoid. This probe measures it AND PCC-checks
it, so the answer is measured rather than inferred.

Variants, per K, at this model's attention dims:
  loop_flat     today's path: K x (2 write + 1 sdpa) on a flat [1,nkv,max_seq,hd] cache
  loop_paged    same, on a paged [nblk,nkv,64,hd] cache
  batch_sdpa    K x 2 writes, then ONE paged sdpa with a K-row page table
  batch_all     ONE K-user write + ONE paged sdpa  (the write-race question)
  flat1blk      the FLAT cache read as a single-block paged cache (page_table = zeros[K,1]);
                no paged cache needed, so it drops into the shipping model unchanged

Run:  ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_packed_verify.py
      ... --positions 1024,8192 --ks 1,2,3,4,8,16 --dtypes bf16,bf8
"""
from __future__ import annotations

import argparse
import time

import torch

import ttnn

NH, NKV, HD = 16, 2, 256  # Qwen3.6: 16 q heads, 2 kv heads, head_dim 256
BLOCK = 64
PCC_GATE = 0.995


def time_traced(mesh, fn, iters=50, warmup=3):
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


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    a, b = a - a.mean(), b - b.mean()
    d = (a.norm() * b.norm()).item()
    return float("nan") if d == 0 else (a @ b).item() / d


def ref_attention(q, k, v, positions):
    """q [K,NH,HD]; k,v [S,NKV,HD] logical rows; row j attends 0..positions[j] inclusive. GQA.
    Vectorised over heads -- the row loop stays because each row has its own key range."""
    out = torch.zeros(len(positions), NH, HD, dtype=torch.float32)
    rep = NH // NKV
    kv_of = torch.arange(NH) // rep  # q head -> kv head
    for j, p in enumerate(positions):
        kk = k[: p + 1].float()[:, kv_of]  # [p+1, NH, HD]
        vv = v[: p + 1].float()[:, kv_of]
        sc = torch.einsum("snd,nd->ns", kk, q[j].float()) * (HD**-0.5)  # [NH, p+1]
        out[j] = torch.einsum("ns,snd->nd", torch.softmax(sc, dim=-1), vv)
    return out


def pc_sweep(mesh, g, positions, ks, dts, iters):
    """The batched verify presents nh*K head-batches instead of nh, so the config tuned for a 1-row
    decode is not obviously right for it. Sweep the knobs that decide the core split.

    Only the SDPA is timed here -- the looped K/V writes are identical across configs, and mixing them
    in would dilute a difference we are trying to see."""
    print(f"{'pos':>7}{'dt':>5}{'K':>4}{'kchunk':>8}{'grid':>8}{'maxcore':>9}{'sdpa us':>10}{'vs base':>9}")
    torch.manual_seed(0)
    cands = [
        (128, (8, 8), None),  # the shipping decode config == today's baseline
        (128, (8, 8), 16),
        (128, (8, 8), 64),
        (64, (8, 8), None),
        (256, (8, 8), None),
        (128, (11, 10), None),
        (256, (11, 10), None),
    ]
    for pos in positions:
        for KVDT in dts:
            max_seq = ((pos + max(ks)) // 512 + 2) * 512  # multiple of every candidate k_chunk
            cache = [
                ttnn.from_torch(
                    (torch.randn(1, NKV, max_seq, HD) * 0.25), dtype=KVDT, layout=ttnn.TILE_LAYOUT, device=mesh
                )
                for _ in range(2)
            ]
            for K in ks:
                q = ttnn.from_torch(
                    torch.randn(1, K, NH, HD) * 0.3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
                )
                cur = ttnn.from_torch(
                    torch.tensor([pos + j for j in range(K)], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                )
                pt = ttnn.from_torch(
                    torch.zeros(K, 1, dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                )
                base = None
                for kc, grid, mc in cands:
                    kw = {} if mc is None else {"max_cores_per_head_batch": mc}
                    cfg = ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                        exp_approx_mode=False,
                        q_chunk_size=0,
                        k_chunk_size=kc,
                        **kw,
                    )
                    dn = "bf16" if KVDT == ttnn.bfloat16 else "bf8"
                    try:
                        us = time_traced(
                            mesh,
                            lambda: ttnn.transformer.paged_scaled_dot_product_attention_decode(
                                q,
                                cache[0],
                                cache[1],
                                cur_pos_tensor=cur,
                                page_table_tensor=pt,
                                scale=HD**-0.5,
                                program_config=cfg,
                            ),
                            iters,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(
                            f"{pos:>7}{dn:>5}{K:>4}{kc:>8}{grid[0]}x{grid[1]:<6}{str(mc):>9}"
                            f"   ERR {type(e).__name__}: {str(e)[:70]}"
                        )
                        continue
                    base = us if base is None else base
                    print(
                        f"{pos:>7}{dn:>5}{K:>4}{kc:>8}{f'{grid[0]}x{grid[1]}':>8}{str(mc):>9}"
                        f"{us:>10.1f}{us/base:>8.2f}x"
                    )
                for t in (q, cur, pt):
                    ttnn.deallocate(t)
            for t in cache:
                ttnn.deallocate(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="1024,8192")
    ap.add_argument("--ks", default="1,2,3,4,8,16")
    ap.add_argument("--dtypes", default="bf16,bf8")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument(
        "--pcsweep",
        action="store_true",
        help="sweep the SDPA program config for the BATCHED verify shape only "
        "(k_chunk_size / grid / max_cores_per_head_batch)",
    )
    ap.add_argument(
        "--timing-only",
        action="store_true",
        help="skip the torch reference + PCC (the O(pos) host attention dominates at 32K)",
    )
    a = ap.parse_args()
    positions = [int(p) for p in a.positions.split(",")]
    ks = [int(k) for k in a.ks.split(",")]
    dts = [{"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}[d] for d in a.dtypes.split(",")]

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=128,
        )
        g = mesh.compute_with_storage_grid_size()
        print(f"grid {g.x}x{g.y}  nh={NH} nkv={NKV} hd={HD} block={BLOCK}  iters={a.iters}\n")
        if a.pcsweep:
            pc_sweep(mesh, g, positions, ks, dts, a.iters)
            return
        hdr = (
            f"{'pos':>7}{'dt':>5}{'K':>4}"
            f"{'loop_flat':>11}{'loop_pgd':>10}{'batch_sdpa':>12}{'batch_all':>11}{'flat1blk':>10}"
            f"{'best/loop':>11}{'saved us':>10}"
        )
        print(hdr)
        print("-" * len(hdr))
        torch.manual_seed(0)
        results = {}
        correctness = []

        for pos in positions:
            for KVDT in dts:
                # flat sdpa requires K-seq % k_chunk_size == 0; keep it a multiple of 128 AND of BLOCK
                max_seq = ((pos + max(ks) + 2 * max(ks) * 32 + 127) // 128 + 1) * 128
                nblk = max_seq // BLOCK
                # ---- host ground truth ------------------------------------------------------
                k_host = torch.randn(max_seq, NKV, HD) * 0.25
                v_host = torch.randn(max_seq, NKV, HD) * 0.25
                for K in ks:
                    q_host = torch.randn(K, NH, HD) * 0.3
                    poss = [pos + j for j in range(K)]
                    # the K new rows land at poss; everything below pos is pre-existing context
                    kv_rows_k = k_host[pos : pos + K]  # [K, NKV, HD]
                    kv_rows_v = v_host[pos : pos + K]

                    q_dev = ttnn.from_torch(
                        q_host.reshape(1, K, NH, HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
                    )
                    cur = ttnn.from_torch(
                        torch.tensor(poss, dtype=torch.int32),
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh,
                    )
                    curs = [
                        ttnn.from_torch(
                            torch.tensor([p], dtype=torch.int32),
                            dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=mesh,
                        )
                        for p in poss
                    ]
                    # page tables
                    pt_1 = ttnn.from_torch(
                        torch.arange(nblk, dtype=torch.int32).reshape(1, nblk),
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh,
                    )
                    pt_K = ttnn.from_torch(
                        torch.arange(nblk, dtype=torch.int32).reshape(1, nblk).repeat(K, 1).contiguous(),
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh,
                    )
                    pt_flatK = ttnn.from_torch(
                        torch.zeros(K, 1, dtype=torch.int32),
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh,
                    )

                    # write inputs: per-row single-user [1,1,TILE,HD], and one K-user [1,K,TILE,HD]
                    def shard_mc(B):
                        cores = ttnn.num_cores_to_corerangeset(B, g, row_wise=True)
                        return ttnn.MemoryConfig(
                            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                            ttnn.BufferType.L1,
                            ttnn.ShardSpec(cores, [ttnn.TILE_SIZE, HD], ttnn.ShardOrientation.ROW_MAJOR),
                        )

                    def upd_tensor(rows):  # rows [B, NKV, HD] -> [1,B,TILE,HD] sharded
                        B = rows.shape[0]
                        t = torch.zeros(1, B, ttnn.TILE_SIZE, HD)
                        t[0, :, :NKV] = rows
                        return ttnn.from_torch(
                            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=shard_mc(B)
                        )

                    upd_k_rows = [upd_tensor(kv_rows_k[j : j + 1]) for j in range(K)]
                    upd_v_rows = [upd_tensor(kv_rows_v[j : j + 1]) for j in range(K)]
                    upd_k_all = upd_tensor(kv_rows_k)
                    upd_v_all = upd_tensor(kv_rows_v)

                    def fresh(paged):
                        """Cache holding context rows 0..pos-1 and ZEROS at the K new rows."""
                        kt, vt = k_host.clone(), v_host.clone()
                        kt[pos:] = 0
                        vt[pos:] = 0
                        if paged:
                            shp = (nblk, BLOCK, NKV, HD)
                            kk = kt.reshape(shp).permute(0, 2, 1, 3).contiguous()
                            vv = vt.reshape(shp).permute(0, 2, 1, 3).contiguous()
                        else:
                            kk = kt.permute(1, 0, 2).unsqueeze(0).contiguous()
                            vv = vt.permute(1, 0, 2).unsqueeze(0).contiguous()
                        return [ttnn.from_torch(x, dtype=KVDT, layout=ttnn.TILE_LAYOUT, device=mesh) for x in (kk, vv)]

                    def read_cache(c, paged):
                        t = ttnn.to_torch(c)
                        if paged:  # [nblk,NKV,BLOCK,HD] -> [S,NKV,HD]
                            return t.permute(0, 2, 1, 3).reshape(-1, NKV, HD)
                        return t[0].permute(1, 0, 2)

                    def row_err(cache, paged, want_rows, at):
                        """max abs error on the K rows the verify wrote, vs the cache dtype's own
                        round-trip of them -- so a nonzero result is CORRUPTION, not quantisation."""
                        got = read_cache(cache, paged)
                        e = 0.0
                        for j, p_ in enumerate(at):
                            e = max(e, (got[p_] - want_rows[j]).abs().max().item())
                        return e

                    ref = None if a.timing_only else ref_attention(q_host, k_host, v_host, poss)
                    row = {}

                    def bench(name, paged, body, at=None, want=None, ref_out=None, timeit=True):
                        cache = fresh(paged)
                        at = poss if at is None else at
                        want = kv_rows_k if want is None else want
                        try:
                            if a.timing_only:
                                row[name] = time_traced(mesh, lambda: body(cache), a.iters) if timeit else None
                                return
                            out = body(cache)
                            got = ttnn.to_torch(out).reshape(len(at), NH, HD)
                            p_out = pcc(ref if ref_out is None else ref_out, got)
                            # quantise the intended rows through the cache dtype so the residual is
                            # corruption only
                            q = ttnn.to_torch(
                                ttnn.from_torch(
                                    want.unsqueeze(0).permute(0, 2, 1, 3).contiguous(),
                                    dtype=KVDT,
                                    layout=ttnn.TILE_LAYOUT,
                                    device=mesh,
                                )
                            )[0].permute(1, 0, 2)
                            err = row_err(cache[0], paged, q, at)
                            correctness.append((pos, KVDT, len(at), name, p_out, err))
                            row[name] = time_traced(mesh, lambda: body(cache), a.iters) if timeit else None
                        except Exception as e:  # noqa: BLE001
                            row[name] = None
                            msg = f"ERR {type(e).__name__}: {str(e)[:120]}"
                            correctness.append((pos, KVDT, len(at), name, msg, ""))
                        finally:
                            for t in cache:
                                ttnn.deallocate(t)

                    # ---- loop_flat: today's shipping path -----------------------------------
                    def body_loop(cache, page_table):
                        rows = []
                        for j in range(K):
                            ttnn.experimental.paged_update_cache(
                                cache[0], upd_k_rows[j], update_idxs_tensor=curs[j], page_table=page_table
                            )
                            ttnn.experimental.paged_update_cache(
                                cache[1], upd_v_rows[j], update_idxs_tensor=curs[j], page_table=page_table
                            )
                            qj = ttnn.slice(q_dev, [0, j, 0, 0], [1, j + 1, NH, HD])
                            if page_table is None:
                                rows.append(
                                    ttnn.transformer.scaled_dot_product_attention_decode(
                                        qj,
                                        cache[0],
                                        cache[1],
                                        cur_pos_tensor=curs[j],
                                        scale=HD**-0.5,
                                        program_config=pc,
                                    )
                                )
                            else:
                                rows.append(
                                    ttnn.transformer.paged_scaled_dot_product_attention_decode(
                                        qj,
                                        cache[0],
                                        cache[1],
                                        cur_pos_tensor=curs[j],
                                        page_table_tensor=pt_1,
                                        scale=HD**-0.5,
                                        program_config=pc,
                                    )
                                )
                        return rows[0] if K == 1 else ttnn.concat(rows, dim=1)

                    bench("loop_flat", False, lambda c: body_loop(c, None))
                    bench("loop_pgd", True, lambda c: body_loop(c, pt_1))

                    # ---- batch_sdpa: looped writes, ONE paged sdpa over K page-table rows ---
                    def body_batch_sdpa(cache):
                        for j in range(K):
                            ttnn.experimental.paged_update_cache(
                                cache[0], upd_k_rows[j], update_idxs_tensor=curs[j], page_table=pt_1
                            )
                            ttnn.experimental.paged_update_cache(
                                cache[1], upd_v_rows[j], update_idxs_tensor=curs[j], page_table=pt_1
                            )
                        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                            q_dev,
                            cache[0],
                            cache[1],
                            cur_pos_tensor=cur,
                            page_table_tensor=pt_K,
                            scale=HD**-0.5,
                            program_config=pc,
                        )

                    bench("batch_sdpa", True, body_batch_sdpa)

                    # ---- batch_all: ONE K-user write + ONE paged sdpa ----------------------
                    def body_batch_all(cache):
                        ttnn.experimental.paged_update_cache(
                            cache[0], upd_k_all, update_idxs_tensor=cur, page_table=pt_K
                        )
                        ttnn.experimental.paged_update_cache(
                            cache[1], upd_v_all, update_idxs_tensor=cur, page_table=pt_K
                        )
                        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                            q_dev,
                            cache[0],
                            cache[1],
                            cur_pos_tensor=cur,
                            page_table_tensor=pt_K,
                            scale=HD**-0.5,
                            program_config=pc,
                        )

                    bench("batch_all", True, body_batch_all)

                    # ---- diagnostic: same batched write, rows in DISTINCT 32-row tiles ------
                    # If batch_all corrupts because K cores read-modify-write ONE tile, then
                    # spreading the rows 32 apart must be clean. That distinguishes "the op cannot
                    # batch" from "the op cannot batch INTO A SHARED TILE".
                    if K > 1:
                        poss_sp = [pos + 32 * j for j in range(K)]
                        cur_sp = ttnn.from_torch(
                            torch.tensor(poss_sp, dtype=torch.int32),
                            dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=mesh,
                        )
                        rows_k_sp = k_host[poss_sp]
                        rows_v_sp = v_host[poss_sp]
                        upd_k_sp, upd_v_sp = upd_tensor(rows_k_sp), upd_tensor(rows_v_sp)
                        k_sp, v_sp = k_host.clone(), v_host.clone()
                        k_sp[pos:] = 0
                        v_sp[pos:] = 0
                        for j, p_ in enumerate(poss_sp):
                            k_sp[p_], v_sp[p_] = rows_k_sp[j], rows_v_sp[j]
                        ref_sp = ref_attention(q_host, k_sp, v_sp, poss_sp)

                        def body_spread(cache):
                            ttnn.experimental.paged_update_cache(
                                cache[0], upd_k_sp, update_idxs_tensor=cur_sp, page_table=pt_K
                            )
                            ttnn.experimental.paged_update_cache(
                                cache[1], upd_v_sp, update_idxs_tensor=cur_sp, page_table=pt_K
                            )
                            return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                                q_dev,
                                cache[0],
                                cache[1],
                                cur_pos_tensor=cur_sp,
                                page_table_tensor=pt_K,
                                scale=HD**-0.5,
                                program_config=pc,
                            )

                        bench(
                            "wr_spread32", True, body_spread, at=poss_sp, want=rows_k_sp, ref_out=ref_sp, timeit=False
                        )
                        for t in (cur_sp, upd_k_sp, upd_v_sp):
                            ttnn.deallocate(t)

                    # ---- flat1blk: the FLAT cache as a 1-block paged cache -----------------
                    def body_flat1blk(cache):
                        for j in range(K):
                            ttnn.experimental.paged_update_cache(
                                cache[0], upd_k_rows[j], update_idxs_tensor=curs[j], page_table=None
                            )
                            ttnn.experimental.paged_update_cache(
                                cache[1], upd_v_rows[j], update_idxs_tensor=curs[j], page_table=None
                            )
                        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                            q_dev,
                            cache[0],
                            cache[1],
                            cur_pos_tensor=cur,
                            page_table_tensor=pt_flatK,
                            scale=HD**-0.5,
                            program_config=pc,
                        )

                    bench("flat1blk", False, body_flat1blk)

                    base = row.get("loop_flat")
                    cands = {k: v for k, v in row.items() if v is not None and k != "loop_flat"}
                    best = min(cands.values()) if cands else None

                    def c(v):
                        return f"{v:.1f}" if isinstance(v, float) else "ERR"

                    dn = "bf16" if KVDT == ttnn.bfloat16 else "bf8"
                    ratio = f"{best/base:.2f}x" if (best and base) else "-"
                    saved = f"{base-best:.1f}" if (best and base) else "-"
                    print(
                        f"{pos:>7}{dn:>5}{K:>4}"
                        f"{c(row.get('loop_flat')):>11}{c(row.get('loop_pgd')):>10}"
                        f"{c(row.get('batch_sdpa')):>12}{c(row.get('batch_all')):>11}"
                        f"{c(row.get('flat1blk')):>10}{ratio:>11}{saved:>10}"
                    )
                    results[(pos, dn, K)] = row
                    for t in (
                        q_dev,
                        cur,
                        pt_1,
                        pt_K,
                        pt_flatK,
                        upd_k_all,
                        upd_v_all,
                        *curs,
                        *upd_k_rows,
                        *upd_v_rows,
                    ):
                        ttnn.deallocate(t)

        print("\nCORRECTNESS (out = sdpa vs torch; kv row err = max|written - intended| on the K new rows;")
        print("             a nonzero kv err is CORRUPTION -- the intended rows are pre-quantised to the cache dtype)")
        print(f"{'pos':>7}{'dt':>5}{'K':>4}{'variant':>14}{'pcc out':>12}{'kv row err':>12}")
        bad = []
        # the loop path is ground truth for the write; bf8 has a nonzero baseline err because a block
        # float tile's shared exponent depends on rows OUTSIDE the K written ones
        loop_err = {}
        for pos, dt, K, name, p_out, p_k in correctness:
            if name == "loop_flat" and not isinstance(p_out, str):
                loop_err[(pos, "bf16" if dt == ttnn.bfloat16 else "bf8", K)] = p_k
        for pos, dt, K, name, p_out, p_k in correctness:
            dn = "bf16" if dt == ttnn.bfloat16 else "bf8"
            if isinstance(p_out, str):
                print(f"{pos:>7}{dn:>5}{K:>4}{name:>14}  {p_out}")
                continue
            base_err = loop_err.get((pos, dn, K), 0.0)
            flag = "" if (p_out > PCC_GATE and p_k <= base_err + 1e-6) else "   <-- FAIL"
            if flag:
                bad.append((pos, dn, K, name, p_out, p_k))
            print(f"{pos:>7}{dn:>5}{K:>4}{name:>14}{p_out:>12.6f}{p_k:>12.3e}{flag}")
        print("\nFAILURES:" if bad else "\nAll variants PCC-clean.")
        for b in bad:
            print("  ", b)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
