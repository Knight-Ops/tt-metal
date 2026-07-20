# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-0 probe for the BATCHED fused Gated-DeltaNet decode kernel.

Validates the core design risk of Fused_gdn_handoff.md: can the single-user fused decode kernel
(`tt/ttl_delta.py:_decode_step`, 32 value-heads -> 32 cores, one head/core, ct=1) take a batch
axis B by wrapping read/compute/write in `for b in range(B)` — reusing the SAME dataflow buffers
per iteration (the `_chunk_inverse` loop pattern) — with each user on its OWN tile-row?

This probe builds a RECURRENCE-ONLY batched kernel (skips l2norm/gating/gated-norm; feeds pre-normed
q/k and precomputed g/beta) and PCC-gates its `out`(=core) and `Snew` against a torch B-user
recurrence for B in {1,2,4,8}. It confirms (a) the B-unrolled op compiles, (b) the per-(b,h) outer
product legalizes under bf16 DST, (c) the DFB budget holds, (d) the tile-row-per-user layout + offsets
are correct. Stages 1+ fold the gating/l2norm/norm back in and port into tt/ttl_delta.py.

Batched tensor contract (each user b on its own tile-row; state stacks (b,h) on rows):
  q, kr : [B*TILE, key_dim]     user b tile-row, key-head hk=h//rep column slab hk*kt_t
  v     : [B*TILE, value_dim]   user b tile-row, value-head h column slab h*vt
  gtile, btile : [B*nv*TILE, TILE]  per-(b,h) uniform tile at tile-row (b*nv+h), col 0
  S, Snew : [B*nv*Dk, Dv]       head-major, (b,h) at tile-rows (b*nv+h)*kt_t
  out   : [B*TILE, value_dim]   user b tile-row, value-head h column slab h*vt

Recurrence per (user b, head h), k-head hk=h//rep:
  Sg = S_bh * g_bh ; kv = k_bh @ Sg ; delta = (v_bh - kv) * beta_bh ;
  outer = k_bh^T @ delta ; st = Sg + outer (=Snew_bh) ; core = q_bh @ st (=out_bh)

Synthetic, no model load. Run: ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_batched_gdn_kernel.py
"""
from __future__ import annotations

import torch
import ttl

import ttnn
from models.demos.qwen3_6_a3b.tt.ttl_delta import decode_step_batch_tt

NV, NK, DK, DV = 32, 16, 128, 128
TILE = 32
REP = NV // NK
KT_T, VT, CT = DK // TILE, DV // TILE, 1
KEY_DIM, VALUE_DIM = NK * DK, NV * DV
BS = [1, 2, 4, 8]
PCC_GATE = 0.999
SCALE = DK**-0.5
NORM_EPS = 1e-6
GDN_EPS = 1e-6  # matches _GDN_EPS in ttl_delta

_ops = {}


def _grid_dims(n_heads):
    """(gn, gm) with gn*gm == n_heads, gn<=11, gm<=10 (Blackhole 11x10 grid). 32 -> (8, 4)."""
    for gm in range(min(n_heads, 10), 0, -1):
        if n_heads % gm == 0 and n_heads // gm <= 11:
            return n_heads // gm, gm
    raise ValueError(f"cannot fit {n_heads} heads in an 11x10 grid")


def _get_op(B):
    """Recurrence-only batched decode op, B baked as a compile-time constant (loop unrolls)."""
    if B in _ops:
        return _ops[B]
    gn, gm = _grid_dims(NV)

    @ttl.operation(grid=(gn, gm), fp32_dest_acc_en=False)
    def _rec_step(
        q: ttnn.Tensor,  # [B*TILE, key_dim]   pre-normed q (heads along columns)
        kr: ttnn.Tensor,  # [B*TILE, key_dim]   pre-normed k
        v: ttnn.Tensor,  # [B*TILE, value_dim]
        gtile: ttnn.Tensor,  # [B*nv*TILE, TILE]  g_exp per (b,h) uniform
        btile: ttnn.Tensor,  # [B*nv*TILE, TILE]  beta per (b,h) uniform
        S: ttnn.Tensor,  # [B*nv*Dk, Dv]
        out: ttnn.Tensor,  # [B*TILE, value_dim]  = core (recurrence-only, no norm)
        Snew: ttnn.Tensor,  # [B*nv*Dk, Dv]
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
        core_d = mk(v, (CT, VT))
        od, Snd = mk(out, (CT, VT)), mk(Snew, (KT_T, VT))

        @ttl.datamovement()
        def read():
            node_n, node_m = ttl.node(dims=2)
            h = node_n * gm + node_m
            hk = h // REP
            cq, cv = hk * KT_T, h * VT
            for b in range(B):
                rMv, rKv, rb = (b * NV + h) * CT, (b * NV + h) * KT_T, b * CT
                with (
                    qd.reserve() as a0,
                    krd.reserve() as a1,
                    vd.reserve() as a2,
                    gd.reserve() as a3,
                    bd.reserve() as a4,
                    Sd.reserve() as a5,
                ):
                    t0 = ttl.copy(q[rb : rb + CT, cq : cq + KT_T], a0)
                    t1 = ttl.copy(kr[rb : rb + CT, cq : cq + KT_T], a1)
                    t2 = ttl.copy(v[rb : rb + CT, cv : cv + VT], a2)
                    t3 = ttl.copy(gtile[rMv : rMv + CT, 0:1], a3)
                    t4 = ttl.copy(btile[rMv : rMv + CT, 0:1], a4)
                    t5 = ttl.copy(S[rKv : rKv + KT_T, 0:VT], a5)
                    t0.wait()
                    t1.wait()
                    t2.wait()
                    t3.wait()
                    t4.wait()
                    t5.wait()

        @ttl.compute()
        def compute():
            for _ in range(B):
                # broadcast g -> state shape, beta -> value shape
                with gd.wait() as gg:
                    with gbd.reserve() as o:
                        o.store(ttl.block.broadcast(gg, dims=[-2, -1], shape=(KT_T, VT)))
                with bd.wait() as bb:
                    with bbd.reserve() as o:
                        o.store(ttl.block.broadcast(bb, dims=[-1], shape=(CT, VT)))
                # transpose k, keep a copy for the k@Sg matmul
                with krd.wait() as kb:
                    with kcol.reserve() as o:
                        o.store(ttl.block.transpose(kb))
                    with kn2.reserve() as o:
                        o.store(kb)
                # rank-1 recurrence
                with Sd.wait() as Sb, gbd.wait() as gb:
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
            for b in range(B):
                rKv, rb = (b * NV + h) * KT_T, b * CT
                with od.wait() as ob:
                    ttl.copy(ob, out[rb : rb + CT, cv : cv + VT]).wait()
                with Snd.wait() as snb:
                    ttl.copy(snb, Snew[rKv : rKv + KT_T, 0:VT]).wait()

    _ops[B] = _rec_step
    return _rec_step


def _torch_ref(q, kr, v, g, beta, S):
    """B-user recurrence reference. q/kr [B,NK,DK], v [B,NV,DV], g/beta [B,NV], S [B,NV,DK,DV]."""
    B = q.shape[0]
    core = torch.zeros(B, NV, DV, dtype=torch.float32)
    Snew = torch.zeros(B, NV, DK, DV, dtype=torch.float32)
    for b in range(B):
        for h in range(NV):
            hk = h // REP
            qh, kh = q[b, hk].float(), kr[b, hk].float()
            vh = v[b, h].float()
            Sg = S[b, h].float() * g[b, h].item()
            kv = kh @ Sg  # [DV]
            delta = (vh - kv) * beta[b, h].item()
            st = Sg + torch.outer(kh, delta)
            core[b, h] = qh @ st
            Snew[b, h] = st
    return core, Snew


def _torch_ref_full(q, kr, v, z, araw, braw, negA, dtb, nweight, S):
    """Full-kernel reference: l2norm(q)*scale + l2norm(k) + gate/beta + recurrence + gated RMSNorm.
    q/kr [B,NK,DK], v/z [B,NV,DV], araw/braw [B,NV], negA/dtb [NV], nweight [DV], S [B,NV,DK,DV]."""
    B = q.shape[0]
    out = torch.zeros(B, NV, DV, dtype=torch.float32)
    Snew = torch.zeros(B, NV, DK, DV, dtype=torch.float32)
    w = nweight.float()
    for b in range(B):
        for h in range(NV):
            hk = h // REP
            qh, kh = q[b, hk].float(), kr[b, hk].float()
            vh, zh = v[b, h].float(), z[b, h].float()
            beta = torch.sigmoid(braw[b, h].float()).item()
            softplus = torch.log(torch.exp(araw[b, h].float() + dtb[h].float()) + 1.0)
            g = torch.exp(negA[h].float() * softplus).item()
            qhat = qh * SCALE * torch.rsqrt((qh * qh).sum() + GDN_EPS)
            khat = kh * torch.rsqrt((kh * kh).sum() + GDN_EPS)
            Sg = S[b, h].float() * g
            kv = khat @ Sg
            delta = (vh - kv) * beta
            st = Sg + torch.outer(khat, delta)
            core = qhat @ st
            rms = torch.rsqrt((core * core).mean() + NORM_EPS)
            out[b, h] = (core * rms * w) * torch.nn.functional.silu(zh)
            Snew[b, h] = st
    return out, Snew


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    if a.numel() != b.numel():
        return float("nan")
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)

    def up(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _rows_to_bh(t, B):  # [B*TILE, F] -> per-user row b*TILE -> [B, NH, D]
        return torch.stack([t[b * TILE] for b in range(B)])

    def _uniform_bh(vals_bh, B):  # [B,NV] -> [B*nv*TILE, TILE], tile (b*nv+h) filled with vals_bh[b,h]
        out = torch.zeros(B * NV * TILE, TILE)
        for b in range(B):
            for h in range(NV):
                r = (b * NV + h) * TILE
                out[r : r + TILE, :] = vals_bh[b, h].item()
        return out

    def _rows_pack(vecs, B, F):  # list/tensor [B, F] -> [B*TILE, F], user b at tile-row b
        out = torch.zeros(B * TILE, F)
        for b in range(B):
            out[b * TILE, :] = vecs[b].reshape(-1)
        return out

    try:
        all_ok = True

        # ---- Stage 0: recurrence-only (isolates the B-loop + tile-row layout + outer-product legality) ----
        print("=== Stage 0: recurrence-only batched kernel vs torch ===")
        for B in BS:
            q = torch.randn(B, NK, DK) * 0.3
            kr = torch.randn(B, NK, DK) * 0.3
            v = torch.randn(B, NV, DV) * 0.3
            g = torch.rand(B, NV) * 0.5 + 0.5  # decay in (0.5,1.0)
            beta = torch.rand(B, NV)  # in (0,1)
            S = torch.randn(B, NV, DK, DV) * 0.1

            qt, kt, vt_ = up(_rows_pack(q, B, KEY_DIM)), up(_rows_pack(kr, B, KEY_DIM)), up(_rows_pack(v, B, VALUE_DIM))
            gt, bt, St = up(_uniform_bh(g, B)), up(_uniform_bh(beta, B)), up(S.reshape(B * NV * DK, DV))
            out_t, Snew_t = up(torch.zeros(B * TILE, VALUE_DIM)), up(torch.zeros(B * NV * DK, DV))

            _get_op(B)(qt, kt, vt_, gt, bt, St, out_t, Snew_t)

            core_dev = _rows_to_bh(ttnn.to_torch(out_t), B).reshape(B, NV, DV)
            Snew_dev = ttnn.to_torch(Snew_t).reshape(B, NV, DK, DV)
            core_ref, Snew_ref = _torch_ref(q, kr, v, g, beta, S)
            p_core, p_state = _pcc(core_ref, core_dev), _pcc(Snew_ref, Snew_dev)
            ok = p_core > PCC_GATE and p_state > PCC_GATE
            all_ok = all_ok and ok
            print(f"  B={B:>2}  core PCC {p_core:.6f}  state PCC {p_state:.6f}  {'OK' if ok else 'FAIL'}")

        # ---- Stage 1: FULL batched kernel (decode_step_batch_tt) vs full torch reference ----
        print("\n=== Stage 1: full batched kernel (l2norm+gate+recurrence+gated-norm) vs torch ===")
        for B in BS:
            q = torch.randn(B, NK, DK) * 0.5
            kr = torch.randn(B, NK, DK) * 0.5
            v = torch.randn(B, NV, DV) * 0.5
            z = torch.randn(B, NV, DV) * 0.5
            araw = torch.randn(B, NV)
            braw = torch.randn(B, NV)
            negA = -torch.rand(NV).uniform_(0.0, 16.0).exp()  # -exp(A_log), per head
            dtb = torch.randn(NV) * 0.5
            nweight = torch.randn(DV) * 0.3 + 1.0
            S = torch.randn(B, NV, DK, DV) * 0.1

            qt = up(_rows_pack(q, B, KEY_DIM))
            kt = up(_rows_pack(kr, B, KEY_DIM))
            vt_ = up(_rows_pack(v, B, VALUE_DIM))
            zt = up(_rows_pack(z, B, VALUE_DIM))
            at = up(_uniform_bh(araw, B))
            bt = up(_uniform_bh(braw, B))
            nAt = up(_uniform_bh(negA.unsqueeze(0).expand(B, NV), B))
            dtt = up(_uniform_bh(dtb.unsqueeze(0).expand(B, NV), B))
            St = up(S.reshape(B * NV * DK, DV))
            nwt = up(nweight.unsqueeze(0).expand(TILE, DV).contiguous())
            out_t, Snew_t = up(torch.zeros(B * TILE, VALUE_DIM)), up(torch.zeros(B * NV * DK, DV))

            decode_step_batch_tt(qt, kt, vt_, zt, at, bt, nAt, dtt, St, nwt, out_t, Snew_t, NV, NK, SCALE, NORM_EPS, B)

            out_dev = _rows_to_bh(ttnn.to_torch(out_t), B).reshape(B, NV, DV)
            Snew_dev = ttnn.to_torch(Snew_t).reshape(B, NV, DK, DV)
            out_ref, Snew_ref = _torch_ref_full(q, kr, v, z, araw, braw, negA, dtb, nweight, S)
            p_out, p_state = _pcc(out_ref, out_dev), _pcc(Snew_ref, Snew_dev)
            ok = p_out > PCC_GATE and p_state > PCC_GATE
            all_ok = all_ok and ok
            print(f"  B={B:>2}  out PCC {p_out:.6f}  state PCC {p_state:.6f}  {'OK' if ok else 'FAIL'}")

        print(
            f"\nRESULT: {'PASS' if all_ok else 'FAIL'} — batched fused GDN decode kernel "
            f"{'compiles + matches torch at all B (Stage 0 + Stage 1)' if all_ok else 'FAILED'}"
        )
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
