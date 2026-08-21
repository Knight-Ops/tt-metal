# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Phase-A measurement gate for MTP / speculative decode — the cost of a K-token VERIFY step.

Answers the three questions that decide GO/NO-GO, in one device session (synthetic real-SHAPE
tensors; all three costs are value-independent, so no checkpoint load):

  A1  verify-MoE: does the expert cost scale with the number of active expert-SLOTS (=K*top_k) in
      INDICES-GATHER mode, and is it cheaper than K separate single-token decode MoE calls?
      This re-measures what the July NO-GO got wrong: `probe_union_moe.py` used `nnz=None`, which
      still spins all 256 sparsity slots (moe.py:234-241 calls that scan "the dominant decode cost
      ~1.4 ms/layer"). Gather mode iterates only `num_active`.

  A2  verify-GDN: the per-extra-token increment delta of an on-core K-step recurrence. Compares
      (a) K independent single-user kernel launches  [what both prior MTP attempts did => break-even]
      (b) the existing BATCHED kernel at B=K          [K on-core iterations, K state reads]
      (c) a new CHAINED kernel: K on-core iterations, ONE state read, state carried in a DFB
      Gate: delta must be well under the 0.323 ms/layer that a whole decode-step GDN costs.

  A3  verify-attention: sdpa_decode(share_cache=True) + paged_update_cache(share_cache=True) with
      cur_pos = pos + arange(K) gives exact causal masking for K speculative rows against a batch-1
      KV cache. Expect ~flat vs a 1-row decode.

Run: ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_mtp_verify.py
"""
from __future__ import annotations

import json
import os
import time

import torch
import ttl

import ttnn
from models.demos.qwen3_6_a3b.tt.moe import TtMoE
from models.demos.qwen3_6_a3b.tt.ttl_delta import decode_step_batch_tt, decode_step_tt

# ---- real Qwen3.6-35B-A3B dims (config.json) ----
E, H, I = 256, 2048, 512  # experts, hidden, moe_intermediate
TOP_K = 8
NV, NK, DK, DV = 32, 16, 128, 128  # gated-delta value/key heads + head dims
N_HEADS, N_KV, HEAD_DIM = 16, 2, 256  # full-attention heads
TILE = 32
REP = NV // NK
KT_T, VT, CT = DK // TILE, DV // TILE, 1
KEY_DIM, VALUE_DIM = NK * DK, NV * DV
MAX_SEQ = 512

# measured per-layer decode costs (CURRENT_BENCHMARK.md:43-51), the comparison base
# A0 measured 2026-08-20, v0.77, 40L real weights: full step 29.01 ms (34.5 tok/s).
# The 26.8 ms / 37.3 tok/s "breakdown" line is the SUM OF ISOLATED per-component traces and
# under-counts the real step by 7.6% (inter-op dispatch gap) -- 29.01 is the baseline to beat.
BASE_STEP_MS = 29.01
MOE_MS, GDN_MS, ATTN_MS, HEAD_MS = 0.326, 0.319, 0.295, 1.226
# A0: real step 29.01 ms vs component sum 26.80 ms -> 2.21 ms of on-device inter-op dispatch gap.
# A verify step pays this ONCE for K tokens instead of K times; charge it once, not zero times.
GAP_MS = 2.21
MOE_EXPERTS_MS = MOE_MS * 0.387 / (0.387 + 0.181)  # experts:router split measured 0.387 : 0.181
N_GDN, N_ATTN, N_MOE = 30, 10, 40

KS = [int(k) for k in os.environ.get("QWEN36_MTP_KS", "1,2,3,4,6,8").split(",")]
RESULTS = os.environ.get("QWEN36_MTP_PROBE_JSON", "/tmp/qwen36_mtp_probe.json")


def _save(section, payload):
    """Append one section's numbers to the results JSON (survives a lost stdout)."""
    try:
        cur = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    except Exception:
        cur = {}
    cur[section] = payload
    with open(RESULTS, "w") as f:
        json.dump(cur, f, indent=2)
    print(f"  [saved {section} -> {RESULTS}]", flush=True)


def time_traced(mesh, fn, iters=200, warmup=5):
    """ms per replay of one traced call. Mirrors tests/bench_decode.py:44."""
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
    dt = (time.time() - t0) / iters * 1e3
    ttnn.release_trace(mesh, tid)
    return dt


# ==================================================================================================
# A1 — verify-MoE in indices-gather mode
# ==================================================================================================
def _split_last(t, half):
    sh = list(t.shape)
    r = len(sh)
    return ttnn.slice(t, [0] * r, sh[:-1] + [half]), ttnn.slice(t, [0] * (r - 1) + [half], list(sh))


def _pc(m, n, in0bw=None):
    """`TtMoE._sparse_pc` with an overridable in0_block_w (K-tiles per multicast block)."""
    if in0bw is None:
        return TtMoE._sparse_pc(m, n)
    Nt = (n + 31) // 32
    cx, cy = TtMoE._grid_for(Nt)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=in0bw,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=max(32, m) // 32,
        per_core_N=1,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def experts_gather(x4, gate_up_sp, down_sp, sparsity, indices, wt, M, num_active, gu_bw=None, dn_bw=None):
    """`_experts_for_tile` generalised to INDICES-GATHER: iterate only `num_active` expert slots.
    x4 [1,1,M,H]; out [M,H]. Mirrors the shipping decode gather path (moe.py:245-266) at M rows."""
    gu = ttnn.sparse_matmul(
        x4,
        gate_up_sp,
        sparsity=sparsity,
        indices=indices,
        program_config=_pc(M, 2 * I, gu_bw),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )  # [1, num_active, M, 2I]
    gate, up = _split_last(gu, I)
    h = ttnn.multiply(ttnn.silu(gate), up)
    h = ttnn.reshape(h, [1, num_active, M, I])
    down = ttnn.sparse_matmul(
        h,
        down_sp,
        sparsity=sparsity,
        indices=indices,
        is_input_a_sparse=True,
        program_config=_pc(M, H, dn_bw),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )  # [1, num_active, M, H]
    down = ttnn.reshape(down, [num_active, M, H])
    return ttnn.sum(ttnn.multiply(down, wt), dim=0)  # [M, H]


def run_a1(mesh):
    print("\n" + "=" * 96)
    print("A1  verify-MoE — indices-gather cost vs active expert slots (num_active = K * top_k)")
    print("=" * 96)
    # bf16 host buffers: the values are irrelevant (sparse_matmul cost is value-independent) and
    # generating 800M elements in fp32 costs ~4 GB and several minutes on this box.
    gate_up_sp = ttnn.from_torch(
        (torch.randn(1, E, H, 2 * I, dtype=torch.bfloat16) * 0.02),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
    )
    down_sp = ttnn.from_torch(
        (torch.randn(1, E, I, H, dtype=torch.bfloat16) * 0.02),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
    )
    sparsity = ttnn.from_torch(
        torch.ones(1, 1, 1, E) / E, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
    )  # required operand, NOT read in indices mode

    def idx(n):
        """[1,1,1,n] UINT16 ROW_MAJOR expert ids -- the dtype/layout sparse_matmul validates. Built
        via typecast like tt/moe_gather.py:topk_to_indices (from_torch cannot emit uint16
        from an int32 host tensor). Ids are non-monotonic and duplicates are legal: gather slots are
        independent."""
        ids = (torch.arange(n) * 7 + 3) % E
        t = ttnn.from_torch(
            ids.reshape(1, 1, 1, n).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
        )
        return ttnn.to_layout(ttnn.typecast(t, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT)

    # reference: the SHIPPING single-token decode expert cost (M=1, num_active=top_k)
    x1 = ttnn.from_torch(torch.randn(1, 1, 1, H) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    wt1 = ttnn.from_torch(torch.randn(TOP_K, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    # every device tensor must exist BEFORE trace capture: a from_torch inside the timed lambda is a
    # host write during capture and fatals ("Writes are not supported during trace capture").
    i1 = idx(TOP_K)
    base = time_traced(mesh, lambda: experts_gather(x1, gate_up_sp, down_sp, sparsity, i1, wt1, 1, TOP_K))
    print(f"\n  decode reference (M=1, num_active=8): {base:.4f} ms/layer   [shipping path]")
    print(f"  measured decode MoE experts share:    {MOE_EXPERTS_MS:.4f} ms/layer  (CURRENT_BENCHMARK)")

    print(f"\n  {'K':>3}{'num_active':>12}{'ms/layer':>11}{'x40 (ms)':>11}{'vs K x base':>13}{'verdict':>12}")
    out = {}
    for K in KS:
        na = K * TOP_K
        M = TILE  # K real rows padded into one tile; per_core_M=1 either way, so rows are free
        x4 = ttnn.from_torch(torch.randn(1, 1, M, H) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        wt = ttnn.from_torch(torch.randn(na, M, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        ii = idx(na)
        try:
            ms = time_traced(mesh, lambda: experts_gather(x4, gate_up_sp, down_sp, sparsity, ii, wt, M, na))
        except Exception as e:
            print(f"  {K:>3}{na:>12}   [FAIL {type(e).__name__}: {str(e)[:50]}]")
            continue
        out[K] = ms
        ratio = ms / (K * base)
        verdict = "flat!" if ratio < 0.75 else ("wash" if ratio < 1.1 else "WORSE")
        print(f"  {K:>3}{na:>12}{ms:>11.4f}{ms*N_MOE:>11.1f}{ratio:>12.2f}x{verdict:>12}")
    _save("a1", {"per_k_ms": out, "decode_ref_ms": base})

    # ---------------------------------------------------------------- A1b
    # `_sparse_pc` uses ONE in0_block_w for BOTH expert matmuls, and moe.py:204-208 caps it at 16
    # because it "must divide Kt for both (gate_up Kt=64, down Kt=16)". But the two matmuls take
    # SEPARATE program configs -- gate_up can use 32 or 64. in0_block_w is K-tiles per multicast
    # block, and the per-block multicast-semaphore handshake is the measured decode MoE bottleneck,
    # so gate_up at 64 would do 1 handshake per slot instead of 4. This is orthogonal to MTP: it
    # would speed up plain single-user decode too (MoE is 45% of the step).
    print("\n  --- A1b: per-matmul in0_block_w (gate_up Kt=64 is NOT capped by down's Kt=16) ---")
    print(f"  {'gu_bw':>6}{'dn_bw':>6}{'M=1 ms':>10}{'M=32,na=32 ms':>15}{'vs bw=16':>10}")
    x1b = ttnn.from_torch(torch.randn(1, 1, 1, H) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    w1b = ttnn.from_torch(torch.randn(TOP_K, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    x32 = ttnn.from_torch(torch.randn(1, 1, 32, H) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    w32 = ttnn.from_torch(torch.randn(32, 32, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    bw = {}
    ib = {TOP_K: idx(TOP_K), 32: idx(32)}  # pre-built: no allocation inside a traced lambda
    for gu_bw, dn_bw in [(16, 16), (32, 16), (64, 16), (64, 8), (32, 8)]:
        row = {}
        for tag, xx, ww, mm, na in (("m1", x1b, w1b, 1, TOP_K), ("m32", x32, w32, 32, 32)):
            try:
                row[tag] = time_traced(
                    mesh,
                    lambda xx=xx, ww=ww, mm=mm, na=na, g=gu_bw, d=dn_bw: experts_gather(
                        xx, gate_up_sp, down_sp, sparsity, ib[na], ww, mm, na, g, d
                    ),
                )
            except Exception as e:
                row[tag] = float("nan")
                row[tag + "_err"] = f"{type(e).__name__}: {str(e)[:60]}"
        bw[f"{gu_bw}/{dn_bw}"] = row
        ref = bw.get("16/16", {}).get("m1") or float("nan")
        err = row.get("m1_err") or row.get("m32_err") or ""
        print(
            f"  {gu_bw:>6}{dn_bw:>6}{row['m1']:>10.4f}{row['m32']:>15.4f}"
            f"{row['m1']/ref if ref == ref else float('nan'):>9.2f}x  {err}"
        )
    _save("a1b", bw)
    return out, base


# ==================================================================================================
# A2 — verify-GDN: chained on-core K-step recurrence
# ==================================================================================================
_chain_ops = {}


def _grid_dims(n_heads):
    for gm in range(min(n_heads, 10), 0, -1):
        if n_heads % gm == 0 and n_heads // gm <= 11:
            return n_heads // gm, gm
    raise ValueError(f"cannot fit {n_heads} heads in an 11x10 grid")


def _get_chain_op(K):
    """Recurrence-only CHAINED kernel: K sequential rank-1 updates to ONE state, on-core.

    Derived from tests/probe_batched_gdn_kernel.py `_rec_step` (B independent states). Two changes:
      * read(): the S fetch is hoisted OUT of the loop -- one [nv*Dk,Dv] read, not K.
      * compute(): the state for step i>0 is the previous step's `st`, carried in the `carry` DFB
        (reserved at the end of step i, waited at the start of step i+1 -- different scopes, so this
        never trips the "cannot wait and reserve the same buffer in one scope" rule).
    `Snew` is still written PER STEP so verify can roll back to the accepted prefix by slot index.
    """
    if K in _chain_ops:
        return _chain_ops[K]
    gn, gm = _grid_dims(NV)

    @ttl.operation(grid=(gn, gm), fp32_dest_acc_en=False)
    def _chain_step(
        q: ttnn.Tensor,  # [K*TILE, key_dim]    pre-normed q, step t on tile-row t
        kr: ttnn.Tensor,  # [K*TILE, key_dim]
        v: ttnn.Tensor,  # [K*TILE, value_dim]
        gtile: ttnn.Tensor,  # [K*nv*TILE, TILE]  g_exp per (t,h)
        btile: ttnn.Tensor,  # [K*nv*TILE, TILE]  beta per (t,h)
        S: ttnn.Tensor,  # [nv*Dk, Dv]        ONE state (not K)
        out: ttnn.Tensor,  # [K*TILE, value_dim]
        Snew: ttnn.Tensor,  # [K*nv*Dk, Dv]      per-step states, for rollback
    ) -> None:
        def mk(t, s):
            return ttl.make_dataflow_buffer_like(t, shape=s, block_count=2)

        qd, krd, vd = mk(q, (CT, KT_T)), mk(kr, (CT, KT_T)), mk(v, (CT, VT))
        gd, bd = mk(gtile, (CT, 1)), mk(btile, (CT, 1))
        Sd = mk(S, (KT_T, VT))
        gbd, bbd = mk(S, (KT_T, VT)), mk(v, (CT, VT))
        kn2, kcol = mk(kr, (CT, KT_T)), mk(S, (KT_T, CT))
        Sg, Sg2, kv, dl, outer, st = (
            mk(S, (KT_T, VT)),
            mk(S, (KT_T, VT)),
            mk(v, (CT, VT)),
            mk(v, (CT, VT)),
            mk(S, (KT_T, VT)),
            mk(S, (KT_T, VT)),
        )
        carry = mk(S, (KT_T, VT))  # the chained state handoff, step i -> step i+1
        core_d = mk(v, (CT, VT))
        od, Snd = mk(out, (CT, VT)), mk(Snew, (KT_T, VT))

        @ttl.datamovement()
        def read():
            node_n, node_m = ttl.node(dims=2)
            h = node_n * gm + node_m
            hk = h // REP
            cq, cv = hk * KT_T, h * VT
            with Sd.reserve() as a5:  # ONE state read, hoisted out of the loop
                ttl.copy(S[h * KT_T : h * KT_T + KT_T, 0:VT], a5).wait()
            for t in range(K):
                rMv, rb = (t * NV + h) * CT, t * CT
                with qd.reserve() as a0, krd.reserve() as a1, vd.reserve() as a2, gd.reserve() as a3, bd.reserve() as a4:
                    t0 = ttl.copy(q[rb : rb + CT, cq : cq + KT_T], a0)
                    t1 = ttl.copy(kr[rb : rb + CT, cq : cq + KT_T], a1)
                    t2 = ttl.copy(v[rb : rb + CT, cv : cv + VT], a2)
                    t3 = ttl.copy(gtile[rMv : rMv + CT, 0:1], a3)
                    t4 = ttl.copy(btile[rMv : rMv + CT, 0:1], a4)
                    t0.wait()
                    t1.wait()
                    t2.wait()
                    t3.wait()
                    t4.wait()

        @ttl.compute()
        def compute():
            # ttl rejects conditionals in kernel source ("conditional expressions are not supported
            # in TT-Lang kernels"), so the state chain cannot be `Sd if t == 0 else carry`. Instead
            # seed `carry` from the single S read, then run K-1 steps that re-produce `carry` and peel
            # the last step (which must NOT produce one, or a block dangles unconsumed). The two loop
            # bodies are duplicated source on purpose -- Python-level `if` is not available here.
            with Sd.wait() as Sb:
                with carry.reserve() as o:
                    o.store(Sb)
            for _t in range(K - 1):
                with gd.wait() as gg:
                    with gbd.reserve() as o:
                        o.store(ttl.block.broadcast(gg, dims=[-2, -1], shape=(KT_T, VT)))
                with bd.wait() as bb:
                    with bbd.reserve() as o:
                        o.store(ttl.block.broadcast(bb, dims=[-1], shape=(CT, VT)))
                with krd.wait() as kb:
                    with kcol.reserve() as o:
                        o.store(ttl.block.transpose(kb))
                    with kn2.reserve() as o:
                        o.store(kb)
                with carry.wait() as Sb, gbd.wait() as gb:
                    with Sg.reserve() as o:
                        o.store(Sb * gb)
                with kn2.wait() as knb, Sg.wait() as Sgb:
                    with kv.reserve() as o:
                        o.store(knb @ Sgb)
                    with Sg2.reserve() as o:
                        o.store(Sgb)
                with vd.wait() as vb, kv.wait() as kvb, bbd.wait() as betab:
                    with dl.reserve() as o:
                        o.store((vb - kvb) * betab)
                with kcol.wait() as kcolb, dl.wait() as dlb:
                    with outer.reserve() as o:
                        o.store(kcolb @ dlb)
                with Sg2.wait() as Sgb, outer.wait() as ob:
                    with st.reserve() as o:
                        o.store(Sgb + ob)
                with qd.wait() as qnb, st.wait() as stb:
                    with core_d.reserve() as o:
                        o.store(qnb @ stb)
                    with Snd.reserve() as o:
                        o.store(stb)  # per-step state -> rollback slot t
                    with carry.reserve() as o:
                        o.store(stb)  # chain into step t+1
                with core_d.wait() as cb:
                    with od.reserve() as o:
                        o.store(cb)
            for _t in range(1):  # final step: consumes the chain, produces no new carry
                with gd.wait() as gg:
                    with gbd.reserve() as o:
                        o.store(ttl.block.broadcast(gg, dims=[-2, -1], shape=(KT_T, VT)))
                with bd.wait() as bb:
                    with bbd.reserve() as o:
                        o.store(ttl.block.broadcast(bb, dims=[-1], shape=(CT, VT)))
                with krd.wait() as kb:
                    with kcol.reserve() as o:
                        o.store(ttl.block.transpose(kb))
                    with kn2.reserve() as o:
                        o.store(kb)
                with carry.wait() as Sb, gbd.wait() as gb:
                    with Sg.reserve() as o:
                        o.store(Sb * gb)
                with kn2.wait() as knb, Sg.wait() as Sgb:
                    with kv.reserve() as o:
                        o.store(knb @ Sgb)
                    with Sg2.reserve() as o:
                        o.store(Sgb)
                with vd.wait() as vb, kv.wait() as kvb, bbd.wait() as betab:
                    with dl.reserve() as o:
                        o.store((vb - kvb) * betab)
                with kcol.wait() as kcolb, dl.wait() as dlb:
                    with outer.reserve() as o:
                        o.store(kcolb @ dlb)
                with Sg2.wait() as Sgb, outer.wait() as ob:
                    with st.reserve() as o:
                        o.store(Sgb + ob)
                with qd.wait() as qnb, st.wait() as stb:
                    with core_d.reserve() as o:
                        o.store(qnb @ stb)
                    with Snd.reserve() as o:
                        o.store(stb)
                with core_d.wait() as cb:
                    with od.reserve() as o:
                        o.store(cb)

        @ttl.datamovement()
        def write():
            node_n, node_m = ttl.node(dims=2)
            h = node_n * gm + node_m
            cv = h * VT
            for t in range(K):
                rKv, rb = (t * NV + h) * KT_T, t * CT
                with od.wait() as ob:
                    ttl.copy(ob, out[rb : rb + CT, cv : cv + VT]).wait()
                with Snd.wait() as snb:
                    ttl.copy(snb, Snew[rKv : rKv + KT_T, 0:VT]).wait()

    _chain_ops[K] = _chain_step
    return _chain_step


def _chain_ref(q, kr, v, g, beta, S):
    """K sequential rank-1 updates to ONE state. q/kr [K,NK,DK], v [K,NV,DV], g/beta [K,NV],
    S [NV,DK,DV]. Returns core [K,NV,DV], states [K,NV,DK,DV]."""
    K = q.shape[0]
    core = torch.zeros(K, NV, DV, dtype=torch.float32)
    states = torch.zeros(K, NV, DK, DV, dtype=torch.float32)
    cur = S.float().clone()
    for t in range(K):
        for h in range(NV):
            hk = h // REP
            qh, kh, vh = q[t, hk].float(), kr[t, hk].float(), v[t, h].float()
            Sg = cur[h] * g[t, h].item()
            delta = (vh - kh @ Sg) * beta[t, h].item()
            stt = Sg + torch.outer(kh, delta)
            core[t, h] = qh @ stt
            states[t, h] = stt
            cur[h] = stt
    return core, states


def _rows(mesh, t2, feat, K):
    """[K, feat] -> [K*TILE, feat], step t on its OWN tile-row, padding rows ZERO (the k^T@delta
    outer product contracts over all 32 tile rows)."""
    z = torch.zeros(K * TILE, feat)
    z[::TILE] = t2
    return ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)


def _uni(mesh, t2, K):
    """[K, NV] -> [K*NV*TILE, TILE], per-(t,h) uniform tile."""
    z = torch.zeros(K * NV * TILE, TILE)
    for t in range(K):
        for h in range(NV):
            z[(t * NV + h) * TILE : (t * NV + h) * TILE + TILE, :] = t2[t, h]
    return ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)


def run_a2(mesh):
    print("\n" + "=" * 96)
    print("A2  verify-GDN — on-core K-step chained recurrence (the decisive number)")
    print("=" * 96)
    torch.manual_seed(0)
    print(f"\n  {'K':>3}{'chain ms':>11}{'batch ms':>11}{'K launch':>11}{'delta/tok':>11}{'x30 chain':>11}{'PCC':>10}")
    res = {}
    for K in KS:
        q = torch.nn.functional.normalize(torch.randn(K, NK, DK), dim=-1) * (DK**-0.5)
        kr = torch.nn.functional.normalize(torch.randn(K, NK, DK), dim=-1)
        v = torch.randn(K, NV, DV) * 0.5
        g = torch.rand(K, NV) * 0.2 + 0.8
        beta = torch.rand(K, NV) * 0.5 + 0.25
        S0 = torch.randn(NV, DK, DV) * 0.1

        qt = _rows(mesh, q.reshape(K, KEY_DIM), KEY_DIM, K)
        krt = _rows(mesh, kr.reshape(K, KEY_DIM), KEY_DIM, K)
        vt = _rows(mesh, v.reshape(K, VALUE_DIM), VALUE_DIM, K)
        gt, bt = _uni(mesh, g, K), _uni(mesh, beta, K)
        St = ttnn.from_torch(S0.reshape(NV * DK, DV), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        outt = ttnn.from_torch(
            torch.zeros(K * TILE, VALUE_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        Snt = ttnn.from_torch(torch.zeros(K * NV * DK, DV), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)

        op = _get_chain_op(K)
        try:
            op(qt, krt, vt, gt, bt, St, outt, Snt)
        except Exception as e:
            print(f"  {K:>3}   [COMPILE FAIL {type(e).__name__}: {str(e)[:55]}]")
            continue
        ttnn.synchronize_device(mesh)

        # PCC of the chained recurrence vs the sequential torch reference
        core_ref, st_ref = _chain_ref(q, kr, v, g, beta, S0)
        got = ttnn.to_torch(outt).reshape(K, TILE, NV, DV)[:, 0, :, :].float()
        gs = ttnn.to_torch(Snt).reshape(K, NV, DK, DV).float()
        pcc = min(
            torch.corrcoef(torch.stack([got.flatten(), core_ref.flatten()]))[0, 1].item(),
            torch.corrcoef(torch.stack([gs.flatten(), st_ref.flatten()]))[0, 1].item(),
        )

        chain_ms = time_traced(mesh, lambda: op(qt, krt, vt, gt, bt, St, outt, Snt))

        # (b) the existing BATCHED kernel at B=K: same K on-core iterations, K state reads
        try:
            Sb = ttnn.from_torch(
                S0.repeat(K, 1, 1).reshape(K * NV * DK, DV), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
            )
            nw = ttnn.from_torch(torch.ones(TILE, DV), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            zt = _rows(mesh, torch.randn(K, VALUE_DIM) * 0.3, VALUE_DIM, K)
            nA = _uni(mesh, -torch.rand(K, NV) - 0.5, K)
            dtb = _uni(mesh, torch.zeros(K, NV), K)
            ob2 = ttnn.from_torch(
                torch.zeros(K * TILE, VALUE_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
            )
            sn2 = ttnn.from_torch(
                torch.zeros(K * NV * DK, DV), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
            )
            batch_ms = time_traced(
                mesh,
                lambda: decode_step_batch_tt(
                    qt, krt, vt, zt, gt, bt, nA, dtb, Sb, nw, ob2, sn2, NV, NK, DK**-0.5, 1e-6, K
                ),
            )
        except Exception as e:
            batch_ms = float("nan")
            print(f"       [batched B={K} n/a: {type(e).__name__}: {str(e)[:45]}]")

        # (c) K independent single-user launches — what both prior MTP attempts effectively did
        q1 = _rows(mesh, q[:1].reshape(1, KEY_DIM), KEY_DIM, 1)
        k1 = _rows(mesh, kr[:1].reshape(1, KEY_DIM), KEY_DIM, 1)
        v1 = _rows(mesh, v[:1].reshape(1, VALUE_DIM), VALUE_DIM, 1)
        z1 = _rows(mesh, torch.randn(1, VALUE_DIM) * 0.3, VALUE_DIM, 1)
        g1, b1 = _uni(mesh, g[:1], 1), _uni(mesh, beta[:1], 1)
        nA1 = _uni(mesh, -torch.rand(1, NV) - 0.5, 1)
        dt1 = _uni(mesh, torch.zeros(1, NV), 1)
        S1 = ttnn.from_torch(S0.reshape(NV * DK, DV), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        nw1 = ttnn.from_torch(torch.ones(TILE, DV), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        o1 = ttnn.from_torch(torch.zeros(TILE, VALUE_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        s1 = ttnn.from_torch(torch.zeros(NV * DK, DV), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)

        def klaunch():
            for _ in range(K):
                decode_step_tt(q1, k1, v1, z1, g1, b1, nA1, dt1, S1, nw1, o1, s1, NV, NK, DK**-0.5, 1e-6)

        seq_ms = time_traced(mesh, klaunch)
        res[K] = (chain_ms, batch_ms, seq_ms, pcc)
        delta = (chain_ms - res[1][0]) / (K - 1) if K > 1 and 1 in res else float("nan")
        print(
            f"  {K:>3}{chain_ms:>11.4f}{batch_ms:>11.4f}{seq_ms:>11.4f}"
            f"{delta:>11.4f}{chain_ms*N_GDN:>11.1f}{pcc:>10.5f}"
        )
    print(f"\n  GATE: delta/tok must be << {GDN_MS:.3f} ms (a whole decode-step GDN layer).")
    print("  'K launch' is the prior-attempt cost -- if chain ~= K launch, MTP is break-even again.")
    _save("a2", {str(k): {"chain_ms": v[0], "batch_ms": v[1], "seq_ms": v[2], "pcc": v[3]} for k, v in res.items()})
    return res


# ==================================================================================================
# A3 — verify-attention via share_cache
# ==================================================================================================
def run_a3(mesh):
    print("\n" + "=" * 96)
    print("A3  verify-attention — sdpa_decode(share_cache=True), K rows vs a batch-1 KV cache")
    print("=" * 96)
    pos = 128
    kc = ttnn.from_torch(
        torch.randn(1, N_KV, MAX_SEQ, HEAD_DIM) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
    )
    vc = ttnn.from_torch(
        torch.randn(1, N_KV, MAX_SEQ, HEAD_DIM) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
    )
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=128
    )
    print(f"\n  {'K':>3}{'sdpa ms':>11}{'vs K=1':>10}{'x10 (ms)':>11}{'status':>10}")
    res = {}
    for K in [1] + [k for k in KS if k > 1]:
        q = ttnn.from_torch(
            torch.randn(1, K, N_HEADS, HEAD_DIM) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        cp = ttnn.from_torch(
            (torch.arange(K) + pos).to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
        )
        try:
            ms = time_traced(
                mesh,
                lambda: ttnn.transformer.scaled_dot_product_attention_decode(
                    q, kc, vc, cur_pos_tensor=cp, scale=HEAD_DIM**-0.5, program_config=pc, share_cache=True
                ),
            )
        except Exception as e:
            print(f"  {K:>3}   [FAIL {type(e).__name__}: {str(e)[:60]}]")
            continue
        res[K] = ms
        print(f"  {K:>3}{ms:>11.4f}{ms/res[1]:>9.2f}x{ms*N_ATTN:>11.2f}{'ok':>10}")
    _save("a3", res)
    return res


# ==================================================================================================
def economics(a1, a1_base, a2, a3):
    print("\n" + "=" * 96)
    print(f"VERIFY-STEP ECONOMICS  (A0 baseline {BASE_STEP_MS:.2f} ms/token = {1000/BASE_STEP_MS:.1f} tok/s, 40L)")
    print("=" * 96)
    base_step = BASE_STEP_MS
    router_shared = (MOE_MS - MOE_EXPERTS_MS) * N_MOE  # flat in K
    print(f"\n  flat terms: MoE router+shared {router_shared:.2f} ms, lm_head {HEAD_MS:.2f} ms")
    print(f"  (verify also charges one {GAP_MS:.2f} ms dispatch gap, vs K of them for K decode steps)")
    print(f"\n  {'K':>3}{'MoE':>8}{'GDN':>8}{'attn':>7}{'head':>7}{'verify':>9}{'vs K steps':>12}")
    verify = {}
    for K in KS:
        if K not in a1 or K not in a2:
            continue
        moe = a1[K] * N_MOE + router_shared
        gdn = a2[K][0] * N_GDN + (GDN_MS - a2[1][0]) * N_GDN  # kernel + flat projections/conv
        attn = a3.get(K, ATTN_MS) * N_ATTN
        head = HEAD_MS
        v = moe + gdn + attn + head + GAP_MS
        verify[K] = v
        print(f"  {K:>3}{moe:>8.1f}{gdn:>8.1f}{attn:>7.1f}{head:>7.2f}{v:>9.1f}{v/(K*base_step):>11.2f}x")

    print(f"\n  {'gamma':>6}{'K':>4}{'round':>8}" + "".join(f"{'a='+str(a):>9}" for a in (0.6, 0.7, 0.8, 0.9)))
    draft_ms = MOE_MS + ATTN_MS + HEAD_MS + 0.15  # MTP head = 1 MoE decoder layer + shared lm_head
    best = (0, 0, 0.0)
    for g in [1, 2, 3, 4, 5, 7]:
        K = g + 1
        if K not in verify:
            continue
        rnd = verify[K] + g * draft_ms + 0.5
        row = f"  {g:>6}{K:>4}{rnd:>8.1f}"
        for a in (0.6, 0.7, 0.8, 0.9):
            toks = sum(a**i for i in range(K))
            sp = base_step / (rnd / toks)
            row += f"{sp:>8.2f}x"
            if a == 0.8 and sp > best[2]:
                best = (g, K, sp)
        print(row)
    print(f"\n  draft step assumed {draft_ms:.2f} ms (MTP head = full MoE decoder layer + lm_head; A4 refines)")
    print(f"  BEST at alpha=0.8: gamma={best[0]} -> {best[2]:.2f}x  ({1000/BASE_STEP_MS*best[2]:.0f} tok/s)")
    print("  GATE: proceed only if >= 1.25x at the A5-measured alpha; KILL below 1.15x.")


def main():
    import sys

    want = set(sys.argv[1:]) or {"a1", "a2", "a3"}
    import traceback

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=300_000_000)
    try:
        # Print any traceback HERE, not via propagation: close_mesh_device below routinely takes
        # minutes (or hangs) with big bf4 DRAM tensors resident, and a propagated traceback would be
        # stuck behind it. Section results are also persisted by _save() as each one finishes.
        a1, a1_base, a2, a3 = {}, 0.0, {}, {}
        for name, fn in (("a1", lambda: run_a1(mesh)), ("a2", lambda: run_a2(mesh)), ("a3", lambda: run_a3(mesh))):
            if name not in want:
                continue
            try:
                r = fn()
            except Exception:
                print(f"\n!! section {name} FAILED:", flush=True)
                traceback.print_exc()
                continue
            if name == "a1":
                a1, a1_base = r
            elif name == "a2":
                a2 = r
            else:
                a3 = r
        if a1 and a2:
            try:
                economics(a1, a1_base, a2, a3)
            except Exception:
                traceback.print_exc()
        print("\n[closing device -- this can take minutes with large bf4 tensors resident]", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
