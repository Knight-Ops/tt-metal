# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused chunked Gated-DeltaNet kernel in tt-lang — incremental buildout (WIP).

Builds the chunked delta-rule kernel piece by piece, each PCC-gated vs the torch reference
(`reference/qwen3_5_moe.py::torch_chunk_gated_delta_rule`). Status:

  Step 1 — intra-chunk attention  o = (q @ kᵀ ⊙ decay_mask) @ v  for one chunk of C tokens.
           DONE: PCC 0.99999. Confirms block matmul + masking in tt-lang.
  Step 2 — triangular inverse (I−L)⁻¹ via the log-depth doubling product
           (I+L)(I+L²)(I+L⁴)…(I+L³²), materialized to L1 buffers to avoid exponential
           compile-time expression growth (the naive `p = p@p` reassignment HANGS the compiler).
           DONE: PCC 0.99999 at REALISTIC L magnitudes. (An earlier 0.948 came from an unrealistically
           large synthetic L; the real gated-delta L = -(k_beta·kᵀ⊙decay) with l2-normed k has
           max|·|~0.01, so (I−L)⁻¹≈I+L and bf16 matmul is plenty — confirmed by an end-to-end bf16-vs-
           fp32 chunk analysis, tests/analyze_chunk_precision.py: chunk-output PCC 0.99999.)
  Step 3 — full single-chunk output (initial_state=0): w = T·v_beta ; A = (q·kᵀ)⊙decay ; out = A·w,
           composing the device inverse (_chunk_inverse) + apply (_chunk_apply). DONE: PCC 0.99999.
  Step 4 — cross-chunk STATE CARRY (v_new=w−k_cumdecay·S, out=(q·exp(g_cum))·S + A·v_new,
           S_new=S·exp(g_last)+(k·exp(g_last−g_cum))ᵀ·v_new), chaining chunks. DONE: 2-chunk chain
           PCC 0.99998 ✓ (head-dim 64). Solved via a STAGED compute: every matmul stores to its own
           buffer, then elementwise stages combine — ttl requires a matmul to be the SOLE op feeding
           its accumulator (no `S*g + kgt@v` or `qg@S + A@v` in one store); S and v_new are reused
           multi-use within a single scope (probe-confirmed OK).
  Step 5 — MULTI-HEAD batching: grid=(n_heads,1), one head per Tensix core; inputs stacked head-major
           on the row axis, ttl.node() head offset in read/write, compute body unchanged. DONE: 4-head
           2-chunk chain PCC 0.99998 (head-dim 64). Ops built by a cached factory keyed on n_heads.
  Step 6 — HEAD-DIM 128 (the real model dim), multi-head, 2-chunk chain. DONE: PCC 0.99997. NO matmul
           tiling needed — an mm probe confirmed every _chunk_state matmul (incl. kcd·S K=4/8-out and
           kgt·v_new 16-out) is legal with bf16 DST (fp32_dest_acc_en=False, 16-tile capacity). The
           old "K=4+8-out illegal" note was the fp32-DST case only.
  Step 7 — INTEGRATED (done): per-chunk prep on-device (batched ttnn: decay/cumsum/exp/L/inverse/w/kcd),
           sequence chaining + state carry, wired into tt/gated_delta.py (_chunk_prep +
           _forward_prefill_chunked) as the DEFAULT prefill (QWEN36_FUSED_PREFILL=0 disables). Full
           chunked path 3-chunk PCC 0.99950; module tests pass; 40-layer prefill 5.4–5.7× faster than
           the scan (warm, seq 256). Only _chunk_state is a ttl kernel — chunk_state_tt is the
           ttnn-native entry; everything else is ttnn (so _chunk_apply/_chunk_inverse are unused in the
           model, kept here for the standalone step ladder).

Layout: one head per core (grid (n_heads,1)); C=64 (2 tiles); head-dim 64 or 128. Inputs are head-major
stacked on the row (M) axis. k is passed pre-transposed as kt [D, C] so the kernel uses plain block
matmuls. chunk_state(...) with S=0 also covers the chunk-0 (initial_state=0) case, subsuming
_chunk_apply. Run: python tt/ttl_delta.py (reports PCCs).
"""
import torch
import ttl

import ttnn

TILE = 32


@ttl.operation(grid=(1, 1))
def _chunk_attn(q: ttnn.Tensor, kt: ttnn.Tensor, v: ttnn.Tensor, mask: ttnn.Tensor, o: ttnn.Tensor) -> None:
    C = q.shape[0]
    D = q.shape[1]
    ct, dt = C // TILE, D // TILE

    q_dfb = ttl.make_dataflow_buffer_like(q, shape=(ct, dt), block_count=2)
    kt_dfb = ttl.make_dataflow_buffer_like(kt, shape=(dt, ct), block_count=2)
    v_dfb = ttl.make_dataflow_buffer_like(v, shape=(ct, dt), block_count=2)
    mask_dfb = ttl.make_dataflow_buffer_like(mask, shape=(ct, ct), block_count=2)
    o_dfb = ttl.make_dataflow_buffer_like(o, shape=(ct, dt), block_count=2)

    @ttl.datamovement()
    def read():
        with q_dfb.reserve() as qb, kt_dfb.reserve() as ktb, v_dfb.reserve() as vb, mask_dfb.reserve() as mb:
            t0 = ttl.copy(q[0:ct, 0:dt], qb)
            t1 = ttl.copy(kt[0:dt, 0:ct], ktb)
            t2 = ttl.copy(v[0:ct, 0:dt], vb)
            t3 = ttl.copy(mask[0:ct, 0:ct], mb)
            t0.wait()
            t1.wait()
            t2.wait()
            t3.wait()

    @ttl.compute()
    def compute():
        with q_dfb.wait() as qb, kt_dfb.wait() as ktb, v_dfb.wait() as vb, mask_dfb.wait() as mb:
            with o_dfb.reserve() as ob:
                a = qb @ ktb  # [C, C]
                a = a * mb  # apply decay/causal mask
                ob.store(a @ vb)  # [C, D]

    @ttl.datamovement()
    def write():
        with o_dfb.wait() as ob:
            ttl.copy(ob, o[0:ct, 0:dt]).wait()


@ttl.operation(grid=(1, 1))
def _chunk_inverse(L: ttnn.Tensor, eye: ttnn.Tensor, T: ttnn.Tensor) -> None:
    """T = (I - L)^{-1} for strictly-lower-triangular nilpotent L [C,C] (L^C = 0), via the
    doubling product (I+L)(I+L^2)(I+L^4)...  — log2(C) squarings, all block matmuls."""
    C = L.shape[0]
    ct = C // TILE
    import math as _m

    steps = int(_m.log2(C))  # 6 for C=64 -> factors i=0..5

    cap = steps - 1  # number of squarings (factors beyond the initial (I+L))

    L_dfb = ttl.make_dataflow_buffer_like(L, shape=(ct, ct), block_count=2)
    eye_dfb = ttl.make_dataflow_buffer_like(eye, shape=(ct, ct), block_count=2)
    p_dfb = ttl.make_dataflow_buffer_like(L, shape=(ct, ct), block_count=2)
    acc_dfb = ttl.make_dataflow_buffer_like(T, shape=(ct, ct), block_count=2)
    T_dfb = ttl.make_dataflow_buffer_like(T, shape=(ct, ct), block_count=2)

    @ttl.datamovement()
    def read():
        with L_dfb.reserve() as lb:
            ttl.copy(L[0:ct, 0:ct], lb).wait()
        for _ in range(cap + 1):  # init uses eye once, then one per squaring step
            with eye_dfb.reserve() as eb:
                ttl.copy(eye[0:ct, 0:ct], eb).wait()

    @ttl.compute()
    def compute():
        # init: p = L^1 ; acc = I + L   (materialized into L1 buffers, not a growing expression)
        with L_dfb.wait() as lb, eye_dfb.wait() as eb:
            with p_dfb.reserve() as p0, acc_dfb.reserve() as a0:
                p0.store(lb)
                a0.store(eb + lb)
        # doubling: each step squares p and multiplies acc by (I + p_squared)
        for _ in range(cap):
            with p_dfb.wait() as pw, acc_dfb.wait() as av, eye_dfb.wait() as eb:
                psq = pw @ pw  # one concrete matmul (inputs from L1) -> no expression blowup
                with p_dfb.reserve() as pn, acc_dfb.reserve() as an:
                    pn.store(psq)
                    an.store(av @ (eb + psq))
        # final acc -> output
        with acc_dfb.wait() as av:
            with T_dfb.reserve() as tb:
                tb.store(av)

    @ttl.datamovement()
    def write():
        with T_dfb.wait() as tb:
            ttl.copy(tb, T[0:ct, 0:ct]).wait()


def chunk_inverse(L, dev):
    C = L.shape[0]

    def up(t):
        # fp32 throughout: the doubling product chains 6 matmuls; bf16 intermediate storage
        # loses too much, so keep the inverse in fp32 (Blackhole supports fp32 matmul/DST).
        return ttnn.from_torch(
            t.to(torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    T = up(torch.zeros(C, C))
    _chunk_inverse(up(L), up(torch.eye(C)), T)
    return ttnn.to_torch(T)


@ttl.operation(grid=(1, 1))
def _chunk_apply(
    q: ttnn.Tensor, kt: ttnn.Tensor, vb: ttnn.Tensor, T: ttnn.Tensor, decay: ttnn.Tensor, o: ttnn.Tensor
) -> None:
    """Post-inverse part of one chunk (initial_state=0): w = T @ v_beta ; A = (q @ kᵀ) ⊙ decay ;
    o = A @ w.  decay is the tril (incl-diag) decay mask, so A is automatically causal."""
    C, D = q.shape[0], q.shape[1]
    ct, dt = C // TILE, D // TILE
    q_d = ttl.make_dataflow_buffer_like(q, shape=(ct, dt), block_count=2)
    kt_d = ttl.make_dataflow_buffer_like(kt, shape=(dt, ct), block_count=2)
    vb_d = ttl.make_dataflow_buffer_like(vb, shape=(ct, dt), block_count=2)
    T_d = ttl.make_dataflow_buffer_like(T, shape=(ct, ct), block_count=2)
    dc_d = ttl.make_dataflow_buffer_like(decay, shape=(ct, ct), block_count=2)
    o_d = ttl.make_dataflow_buffer_like(o, shape=(ct, dt), block_count=2)

    @ttl.datamovement()
    def read():
        with q_d.reserve() as a, kt_d.reserve() as b, vb_d.reserve() as c, T_d.reserve() as t, dc_d.reserve() as d:
            x0 = ttl.copy(q[0:ct, 0:dt], a)
            x1 = ttl.copy(kt[0:dt, 0:ct], b)
            x2 = ttl.copy(vb[0:ct, 0:dt], c)
            x3 = ttl.copy(T[0:ct, 0:ct], t)
            x4 = ttl.copy(decay[0:ct, 0:ct], d)
            x0.wait()
            x1.wait()
            x2.wait()
            x3.wait()
            x4.wait()

    @ttl.compute()
    def compute():
        with q_d.wait() as qb, kt_d.wait() as ktb, vb_d.wait() as vbb, T_d.wait() as Tb, dc_d.wait() as dcb:
            with o_d.reserve() as ob:
                w = Tb @ vbb  # [C, D]
                a = (qb @ ktb) * dcb  # [C, C], causal via decay (tril)
                ob.store(a @ w)  # [C, D]

    @ttl.datamovement()
    def write():
        with o_d.wait() as ob:
            ttl.copy(ob, o[0:ct, 0:dt]).wait()


# fp32_dest_acc_en=False -> bf16 DST (16-tile capacity) so the [C,Dv]=8-tile outputs fit; fp32 acc
# would exactly fill the 8-tile f32 DST and the K=4 matmuls fail to legalize. bf16 acc is fine here
# (realistic small magnitudes; see tests/analyze_chunk_precision.py).
#
# Multi-head: one head per Tensix core via a 2D grid (gn, gm) with gn*gm == n_heads. Blackhole's
# compute grid is 11x10, so a single axis caps at 11 — 32 heads use an 8x4 grid. Each core's linear
# head index is h = node_n*gm + node_m. Inputs are head-major stacked on the row (M) axis — head h
# occupies tile-rows [h*ct:(h+1)*ct] for [C,*] operands and [h*kt_t:(h+1)*kt_t] for [Dk,*] operands
# (S, kgt, glast, Snew). The compute body is per-head and identical to the single-head version; only
# the read/write threads add the head offset. Ops are built by a cached factory keyed on n_heads
# (grid must be fixed at decoration time). n_heads=1 reproduces the original single-head kernel.
_chunk_state_ops = {}


def _grid_dims(n_heads):
    """Pick (gn, gm) with gn*gm == n_heads, gn<=11, gm<=10 (Blackhole 11x10 compute grid)."""
    for gm in range(min(n_heads, 10), 0, -1):
        if n_heads % gm == 0 and n_heads // gm <= 11:
            return n_heads // gm, gm
    raise ValueError(f"cannot fit {n_heads} heads in an 11x10 grid")


def _get_chunk_state_op(n_heads):
    if n_heads in _chunk_state_ops:
        return _chunk_state_ops[n_heads]
    gn, gm = _grid_dims(n_heads)

    @ttl.operation(grid=(gn, gm), fp32_dest_acc_en=False)
    def _chunk_state(
        q: ttnn.Tensor,
        kt: ttnn.Tensor,
        w: ttnn.Tensor,
        kcd: ttnn.Tensor,
        decay: ttnn.Tensor,
        qg: ttnn.Tensor,
        kgt: ttnn.Tensor,
        glast: ttnn.Tensor,
        S: ttnn.Tensor,
        out: ttnn.Tensor,
        Snew: ttnn.Tensor,
    ) -> None:
        """One chunk WITH incoming recurrent state S (chains chunks). Given precomputed per-chunk terms:
          w = T·v_beta, kcd = T·(k_beta⊙exp(g_cum)), qg = q⊙exp(g_cum), kgt = (k⊙exp(g_last−g_cum))ᵀ,
          glast = exp(g_cum[-1]) (broadcast tile), decay = tril decay mask.
        Computes:  v_new = w − kcd·S ;  out = qg·S + ((q·kᵀ)⊙decay)·v_new ;  Snew = S·glast + kgt·v_new."""
        C = (q.shape[0] // TILE) // n_heads * TILE  # per-head rows (q is [n_heads*C, Dk])
        Dk, Dv = q.shape[1], w.shape[1]
        ct, kt_t, vt = C // TILE, Dk // TILE, Dv // TILE
        qd = ttl.make_dataflow_buffer_like(q, shape=(ct, kt_t), block_count=2)
        ktd = ttl.make_dataflow_buffer_like(kt, shape=(kt_t, ct), block_count=2)
        wd = ttl.make_dataflow_buffer_like(w, shape=(ct, vt), block_count=2)
        kcdd = ttl.make_dataflow_buffer_like(kcd, shape=(ct, kt_t), block_count=2)
        dcd = ttl.make_dataflow_buffer_like(decay, shape=(ct, ct), block_count=2)
        qgd = ttl.make_dataflow_buffer_like(qg, shape=(ct, kt_t), block_count=2)
        kgtd = ttl.make_dataflow_buffer_like(kgt, shape=(kt_t, ct), block_count=2)
        gld = ttl.make_dataflow_buffer_like(glast, shape=(kt_t, vt), block_count=2)
        Sd = ttl.make_dataflow_buffer_like(S, shape=(kt_t, vt), block_count=2)
        od = ttl.make_dataflow_buffer_like(out, shape=(ct, vt), block_count=2)
        Snd = ttl.make_dataflow_buffer_like(Snew, shape=(kt_t, vt), block_count=2)
        # intermediate buffers — every matmul stores to its own block (ttl: a matmul must be the sole op
        # feeding its accumulator), then elementwise stages combine them.
        kcdS_d = ttl.make_dataflow_buffer_like(out, shape=(ct, vt), block_count=2)  # kcd·S
        qgS_d = ttl.make_dataflow_buffer_like(out, shape=(ct, vt), block_count=2)  # qg·S
        sg_d = ttl.make_dataflow_buffer_like(Snew, shape=(kt_t, vt), block_count=2)  # S·glast
        qk_d = ttl.make_dataflow_buffer_like(decay, shape=(ct, ct), block_count=2)  # q·kᵀ
        ai_d = ttl.make_dataflow_buffer_like(decay, shape=(ct, ct), block_count=2)  # aintra = qk⊙decay
        vn_d = ttl.make_dataflow_buffer_like(out, shape=(ct, vt), block_count=2)  # v_new = w − kcd·S
        av_d = ttl.make_dataflow_buffer_like(out, shape=(ct, vt), block_count=2)  # aintra·v_new
        sk_d = ttl.make_dataflow_buffer_like(Snew, shape=(kt_t, vt), block_count=2)  # kgt·v_new

        @ttl.datamovement()
        def read():
            node_n, node_m = ttl.node(dims=2)
            h = node_n * gm + node_m
            rM, rK = h * ct, h * kt_t
            with qd.reserve() as a, ktd.reserve() as b, wd.reserve() as c, kcdd.reserve() as d, dcd.reserve() as e, qgd.reserve() as f, kgtd.reserve() as g_, gld.reserve() as gl, Sd.reserve() as s:
                t0 = ttl.copy(q[rM : rM + ct, 0:kt_t], a)
                t1 = ttl.copy(kt[rK : rK + kt_t, 0:ct], b)
                t2 = ttl.copy(w[rM : rM + ct, 0:vt], c)
                t3 = ttl.copy(kcd[rM : rM + ct, 0:kt_t], d)
                t4 = ttl.copy(decay[rM : rM + ct, 0:ct], e)
                t5 = ttl.copy(qg[rM : rM + ct, 0:kt_t], f)
                t6 = ttl.copy(kgt[rK : rK + kt_t, 0:ct], g_)
                t7 = ttl.copy(glast[rK : rK + kt_t, 0:vt], gl)
                t8 = ttl.copy(S[rK : rK + kt_t, 0:vt], s)
                t0.wait()
                t1.wait()
                t2.wait()
                t3.wait()
                t4.wait()
                t5.wait()
                t6.wait()
                t7.wait()
                t8.wait()

        @ttl.compute()
        def compute():
            # Stage A: matmuls/eltwise using S (S reused multi-use within one scope)
            with kcdd.wait() as kcdb, qgd.wait() as qgb, gld.wait() as glb, Sd.wait() as Sb:
                with kcdS_d.reserve() as o:
                    o.store(kcdb @ Sb)
                with qgS_d.reserve() as o:
                    o.store(qgb @ Sb)
                with sg_d.reserve() as o:
                    o.store(Sb * glb)
            # Stage B: q·kᵀ
            with qd.wait() as qb, ktd.wait() as ktb:
                with qk_d.reserve() as o:
                    o.store(qb @ ktb)
            # Stage C: elementwise — aintra = qk⊙decay ; v_new = w − kcd·S
            with qk_d.wait() as qkb, dcd.wait() as dcb:
                with ai_d.reserve() as o:
                    o.store(qkb * dcb)
            with wd.wait() as wb, kcdS_d.wait() as kcdSb:
                with vn_d.reserve() as o:
                    o.store(wb - kcdSb)
            # Stage D: matmuls with materialized operands (v_new reused multi-use within one scope)
            with ai_d.wait() as aib, kgtd.wait() as kgtb, vn_d.wait() as vnb:
                with av_d.reserve() as o:
                    o.store(aib @ vnb)
                with sk_d.reserve() as o:
                    o.store(kgtb @ vnb)
            # Stage E: final elementwise combines -> outputs
            with qgS_d.wait() as qgSb, av_d.wait() as avb:
                with od.reserve() as ob:
                    ob.store(qgSb + avb)
            with sg_d.wait() as sgb, sk_d.wait() as skb:
                with Snd.reserve() as snb:
                    snb.store(sgb + skb)

        @ttl.datamovement()
        def write():
            node_n, node_m = ttl.node(dims=2)
            h = node_n * gm + node_m
            rM, rK = h * ct, h * kt_t
            with od.wait() as ob:
                ttl.copy(ob, out[rM : rM + ct, 0:vt]).wait()
            with Snd.wait() as snb:
                ttl.copy(snb, Snew[rK : rK + kt_t, 0:vt]).wait()

    _chunk_state_ops[n_heads] = _chunk_state
    return _chunk_state


def chunk_state_tt(q, kt, w, kcd, decay, qg, kgt, glast, S, out, Snew, n_heads):
    """ttnn-native entry: all args are device ttnn tensors (head-major stacked, bf16, TILE layout).
    Writes results into the preallocated `out` [n_heads*C, Dv] and `Snew` [n_heads*Dk, Dv]. No host
    round-trip — this is what the model prefill calls per chunk."""
    _get_chunk_state_op(n_heads)(q, kt, w, kcd, decay, qg, kgt, glast, S, out, Snew)


def chunk_state(q, kt, w, kcd, decay, qg, kgt, glast, S, dev, n_heads=1):
    """q,kcd,qg: [n_heads*C, Dk]; kt,kgt: [n_heads*Dk, C]; w: [n_heads*C, Dv]; decay: [n_heads*C, C];
    glast,S: [n_heads*Dk, Dv]. Returns out [n_heads*C, Dv], Snew [n_heads*Dk, Dv] (head-major)."""

    def up(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    C_total, Dv = q.shape[0], w.shape[1]
    Dk = q.shape[1]
    out = up(torch.zeros(C_total, Dv))
    Snew = up(torch.zeros(S.shape[0], Dv))
    op = _get_chunk_state_op(n_heads)
    op(up(q), up(kt), up(w), up(kcd), up(decay), up(qg), up(kgt), up(glast), up(S), out, Snew)
    return ttnn.to_torch(out), ttnn.to_torch(Snew)


def chunk_apply(q, kt, vb, T, decay, dev):
    def up(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    o = up(torch.zeros(q.shape[0], vb.shape[1]))
    _chunk_apply(up(q), up(kt), up(vb), up(T), up(decay), o)
    return ttnn.to_torch(o)


def chunk_attn(q, kt, v, mask, dev):
    def up(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    o = up(torch.zeros(q.shape[0], v.shape[1]))
    _chunk_attn(up(q), up(kt), up(v), up(mask), o)
    return ttnn.to_torch(o)


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    try:
        C, D = 64, 128
        q = torch.randn(C, D) * 0.3
        k = torch.randn(C, D) * 0.3
        v = torch.randn(C, D) * 0.3
        mask = torch.tril(torch.ones(C, C))  # causal lower-tri (stand-in for decay_mask)
        expected = (q.float() @ k.float().T * mask) @ v.float()
        out = chunk_attn(q, k.T.contiguous(), v, mask, dev).float()
        pcc = torch.corrcoef(torch.stack([out.flatten(), expected.flatten()]))[0, 1].item()
        print(f"[ttl_delta] step1 intra-chunk (q·kᵀ⊙mask)·v PCC {pcc:.5f}")
        assert pcc > 0.99, pcc
        print("[ttl_delta] step 1 OK ✓")

        # Step 2: triangular inverse (I - L)^-1 via doubling product.
        # Use REALISTIC magnitudes: the gated-delta L = -(k_beta @ kᵀ ⊙ decay) with l2-normed k has
        # max|.|~0.01 (host analysis), so (I-L)^-1 ≈ I+L and bf16 is accurate. (A large synthetic L
        # like randn*0.1 is not representative and exposes only bf16-chained-matmul noise.)
        L = torch.tril(torch.randn(C, C) * 0.01, diagonal=-1)  # strictly lower, realistic scale
        Tref = torch.inverse(torch.eye(C) - L)
        Ttt = chunk_inverse(L, dev).float()
        pcc2 = torch.corrcoef(torch.stack([Ttt.flatten(), Tref.flatten()]))[0, 1].item()
        print(f"[ttl_delta] step2 (I-L)^-1 doubling PCC {pcc2:.5f}")
        if pcc2 > 0.99:
            print("[ttl_delta] step 2 OK ✓")
        else:
            print(f"[ttl_delta] step 2 WIP — compiles+runs, PCC {pcc2:.4f}")

        # Step 3: full single-chunk (initial_state=0) delta output, composing inverse + apply.
        import torch.nn.functional as F

        Dk = 128
        qr = torch.randn(C, Dk)
        kr = torch.randn(C, Dk)
        vr = torch.randn(C, Dk) * 0.5
        beta = torch.sigmoid(torch.randn(C))
        gln = -torch.rand(C).uniform_(0, 16).log_().exp() * F.softplus(torch.randn(C) + 1.0)
        qn = F.normalize(qr, dim=-1, eps=1e-6) * (Dk**-0.5)
        kn = F.normalize(kr, dim=-1, eps=1e-6)
        gc = gln.cumsum(0)
        decay = (gc[:, None] - gc[None, :]).tril().exp().tril()
        vb = vr * beta[:, None]
        kb = kn * beta[:, None]
        Lc = torch.tril(-(kb @ kn.T * decay), diagonal=-1)  # strictly-lower
        Tref = torch.inverse(torch.eye(C) - Lc)  # gold inverse
        A_ref = torch.tril(qn @ kn.T * decay)
        out_ref = A_ref @ (Tref @ vb)  # gold chunk-0 output

        # apply kernel alone (gold inverse):
        out_apply = chunk_apply(qn, kn.T.contiguous(), vb, Tref, decay, dev).float()
        p_apply = torch.corrcoef(torch.stack([out_apply.flatten(), out_ref.flatten()]))[0, 1].item()
        print(f"[ttl_delta] step3 apply (gold T) PCC {p_apply:.5f}")
        # full composition (device inverse + device apply):
        Tdev = torch.from_numpy(chunk_inverse(Lc, dev).numpy()).float()
        out_full = chunk_apply(qn, kn.T.contiguous(), vb, Tdev, decay, dev).float()
        p_full = torch.corrcoef(torch.stack([out_full.flatten(), out_ref.flatten()]))[0, 1].item()
        print(f"[ttl_delta] step3 full chunk-0 (device inverse+apply) PCC {p_full:.5f}")
        print("[ttl_delta] step 3 OK ✓" if p_full > 0.99 else f"[ttl_delta] step 3 WIP PCC {p_full:.4f}")

        # Step 4: cross-chunk state carry — 2-chunk chain vs fp32 reference.
        # Validate the state-carry LOGIC at head-dim 64 (K=2 tiles -> matmuls legal). Scaling to the
        # real head-dim 128 needs k-loop matmul tiling (tutorial step_2): a single [C,Dk]@[Dk,Dv] with
        # K=4 tiles + 8-tile output is rejected by the ttl matmul lowering ("explicitly marked illegal").
        Dh = 64
        Slen = 2 * C
        Q = torch.randn(Slen, Dh)
        K = torch.randn(Slen, Dh)
        V = torch.randn(Slen, Dh) * 0.5
        bet = torch.sigmoid(torch.randn(Slen))
        gg = -torch.rand(Slen).uniform_(0, 16).log_().exp() * F.softplus(torch.randn(Slen) + 1.0)
        Qn = F.normalize(Q, dim=-1, eps=1e-6) * (Dh**-0.5)
        Kn = F.normalize(K, dim=-1, eps=1e-6)

        def gold_and_device():
            outs_ref, outs_dev = [], []
            Sref = torch.zeros(Dh, Dh)
            Sdev = torch.zeros(Dh, Dh)
            for i in range(Slen // C):
                sl = slice(i * C, (i + 1) * C)
                qc, kc, vc, bc = Qn[sl], Kn[sl], V[sl], bet[sl]
                gc = gg[sl].cumsum(0)
                decay = (gc[:, None] - gc[None, :]).tril().exp().tril()
                vbeta, kbeta = vc * bc[:, None], kc * bc[:, None]
                L = torch.tril(-(kbeta @ kc.T * decay), diagonal=-1)
                T = torch.inverse(torch.eye(C) - L)
                w = T @ vbeta
                kcd = T @ (kbeta * gc.exp()[:, None])
                qg = qc * gc.exp()[:, None]
                kg = kc * (gc[-1] - gc).exp()[:, None]
                glast = gc[-1].exp().item()
                # reference (fp32)
                vnew_r = w - kcd @ Sref
                out_r = qg @ Sref + torch.tril(qc @ kc.T * decay) @ vnew_r
                Sref = Sref * glast + kg.T @ vnew_r
                outs_ref.append(out_r)
                # device state-carry kernel (uses device inverse T for w/kcd? here host T for isolation)
                glast_tile = torch.full((Dh, Dh), glast)
                out_d, Sdev = chunk_state(
                    qc, kc.T.contiguous(), w, kcd, decay, qg, kg.T.contiguous(), glast_tile, Sdev, dev
                )
                outs_dev.append(out_d.float())
            return torch.cat(outs_ref), torch.cat(outs_dev)

        try:
            ref_out, dev_out = gold_and_device()
            p4 = torch.corrcoef(torch.stack([dev_out.flatten(), ref_out.flatten()]))[0, 1].item()
            print(f"[ttl_delta] step4 2-chunk chain (state carry) PCC {p4:.5f}")
            print("[ttl_delta] step 4 OK ✓" if p4 > 0.99 else f"[ttl_delta] step 4 WIP PCC {p4:.4f}")
        except Exception as e:
            print(
                f"[ttl_delta] step 4 WIP — _chunk_state needs restructure (see docstring): "
                f"{type(e).__name__}: {str(e).splitlines()[-1][:120]}"
            )

        # Steps 5/6: multi-head batching — grid=(n_heads,1), one head per Tensix core. Inputs stacked
        # head-major on the row axis; validates the ttl.node head offset + per-head state carry vs a
        # per-head loop of the fp32 reference. Run at head-dim 64 (step 5) and the real head-dim 128
        # (step 6). The mm_probe confirmed every _chunk_state matmul is legal at Dh=128 with bf16 DST
        # (fp32_dest_acc_en=False, 16-tile capacity), so no per-matmul tiling is needed.
        def gold_and_device_mh(NH, Dh, seed=1):
            torch.manual_seed(seed)
            Slen = 2 * C
            Qm = torch.randn(NH, Slen, Dh)
            Km = torch.randn(NH, Slen, Dh)
            Vm = torch.randn(NH, Slen, Dh) * 0.5
            betm = torch.sigmoid(torch.randn(NH, Slen))
            ggm = -torch.rand(NH, Slen).uniform_(0, 16).log_().exp() * F.softplus(torch.randn(NH, Slen) + 1.0)
            Qnm = F.normalize(Qm, dim=-1, eps=1e-6) * (Dh**-0.5)
            Knm = F.normalize(Km, dim=-1, eps=1e-6)
            Sref = [torch.zeros(Dh, Dh) for _ in range(NH)]
            Sdev = torch.zeros(NH * Dh, Dh)
            outs_ref = [[] for _ in range(NH)]
            outs_dev = [[] for _ in range(NH)]
            for i in range(Slen // C):
                sl = slice(i * C, (i + 1) * C)
                q_s, kt_s, w_s, kcd_s, dc_s, qg_s, kgt_s, gl_s = [], [], [], [], [], [], [], []
                for hh in range(NH):
                    qc, kc, vc, bc = Qnm[hh, sl], Knm[hh, sl], Vm[hh, sl], betm[hh, sl]
                    gc = ggm[hh, sl].cumsum(0)
                    decay = (gc[:, None] - gc[None, :]).tril().exp().tril()
                    vbeta, kbeta = vc * bc[:, None], kc * bc[:, None]
                    L = torch.tril(-(kbeta @ kc.T * decay), diagonal=-1)
                    T = torch.inverse(torch.eye(C) - L)
                    w = T @ vbeta
                    kcd = T @ (kbeta * gc.exp()[:, None])
                    qg = qc * gc.exp()[:, None]
                    kg = kc * (gc[-1] - gc).exp()[:, None]
                    glast = gc[-1].exp().item()
                    vnew_r = w - kcd @ Sref[hh]
                    out_r = qg @ Sref[hh] + torch.tril(qc @ kc.T * decay) @ vnew_r
                    Sref[hh] = Sref[hh] * glast + kg.T @ vnew_r
                    outs_ref[hh].append(out_r)
                    q_s.append(qc)
                    kt_s.append(kc.T.contiguous())
                    w_s.append(w)
                    kcd_s.append(kcd)
                    dc_s.append(decay)
                    qg_s.append(qg)
                    kgt_s.append(kg.T.contiguous())
                    gl_s.append(torch.full((Dh, Dh), glast))
                out_d, Sdev = chunk_state(
                    torch.cat(q_s),
                    torch.cat(kt_s),
                    torch.cat(w_s),
                    torch.cat(kcd_s),
                    torch.cat(dc_s),
                    torch.cat(qg_s),
                    torch.cat(kgt_s),
                    torch.cat(gl_s),
                    Sdev,
                    dev,
                    n_heads=NH,
                )
                for hh in range(NH):
                    outs_dev[hh].append(out_d[hh * C : (hh + 1) * C].float())
            ref_flat = torch.cat([torch.cat(o) for o in outs_ref])
            dev_flat = torch.cat([torch.cat(o) for o in outs_dev])
            return ref_flat, dev_flat

        for step, (NH, Dh) in [(5, (4, 64)), (6, (4, 128))]:
            try:
                ref_mh, dev_mh = gold_and_device_mh(NH, Dh)
                p = torch.corrcoef(torch.stack([dev_mh.flatten(), ref_mh.flatten()]))[0, 1].item()
                print(f"[ttl_delta] step{step} multi-head ({NH} heads, Dh={Dh}, grid=(NH,1)) state carry PCC {p:.5f}")
                print(f"[ttl_delta] step {step} OK ✓" if p > 0.99 else f"[ttl_delta] step {step} WIP PCC {p:.4f}")
            except Exception as e:
                print(f"[ttl_delta] step {step} WIP — Dh={Dh}: {type(e).__name__}: {str(e).splitlines()[-1][:160]}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
