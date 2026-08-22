# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""STAGE-0 GO/NO-GO: can the fused decode kernel run an fp32 recurrent state, and what does it cost?

`probe_gate_vs_state.py` settled that the decode drift needs BOTH an fp32 gate and an fp32 carry
(bf16 or TF32 for either is not enough), and that bf16 matmul read-outs are free (PCC 0.999998). The
cheap implementation of that would be a mixed kernel: fp32 carry, bf16 everything else.

**ttl 1.1.6 cannot express it.** It rejects an f32 tile argument alongside any bf16 one -- "mixed f32
and non-f32 tile arguments" (TTLOpsUtils.h:679-691) -- and an f32->f32 self-cast does NOT satisfy the
checker (probed). So the kernel is all-fp32 or nothing, which drags in:
  * every tensor argument fp32, so the caller must produce fp32 q/k/v/z/out;
  * fp32 DST, halving capacity (4 tiles double-buffered / 8 with dst_full_sync_en) -- and `outer` is a
    16-tile matmul output, exactly the shape ttl_delta.py:65-67 records as rejected under fp32 DST;
  * the state DFBs double in L1, so they drop to block_count=1.

This probe measures whether that is affordable, BEFORE any cross-cutting allocator work.

Budget: 5% of 28.16 ms/token over 30 GDN layers = 47 us/layer. The kernel is ~61 us today, so the
gate is <= ~108 us/launch INCLUDING whatever the caller must spend making its inputs fp32.

    ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_fp32_state_kernel.py
"""

from __future__ import annotations

import argparse
import os
import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig, l2norm
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.ttl_delta import decode_step_tt

TILE = 32


def ref_step(S, q, k, v, z, a, b, negA, dtb, nw, scale, eps):
    """The recurrence the kernel implements, in torch fp64 -- the ground truth for both builds."""
    S = S.double()
    qn = l2norm(q.double(), dim=-1, eps=1e-6) * scale
    kn = l2norm(k.double(), dim=-1, eps=1e-6)
    beta = torch.sigmoid(b.double()).unsqueeze(-1).unsqueeze(-1)
    g = (negA.double() * torch.nn.functional.softplus(a.double() + dtb.double())).exp()
    g = g.unsqueeze(-1).unsqueeze(-1)
    Sg = S * g
    kv = (Sg * kn.unsqueeze(-1)).sum(dim=-2)
    delta = (v.double() - kv) * beta.squeeze(-1)
    Snew = Sg + kn.unsqueeze(-1) * delta.unsqueeze(-2)
    core = (Snew * qn.unsqueeze(-1)).sum(dim=-2)
    rms = torch.rsqrt(core.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = (core * rms) * nw.double() * torch.sigmoid(z.double()) * z.double()
    return Snew, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument(
        "--no-gate", action="store_true", help="force negA=0 so g=exp(0)=1: isolates the ACCUMULATE path from the GATE"
    )
    ap.add_argument(
        "--no-alias",
        action="store_true",
        help="pass a SEPARATE Snew buffer instead of aliasing S (isolates the in-place write)",
    )
    ap.add_argument(
        "--identity",
        action="store_true",
        help="negA=0 AND beta~0 (braw=-20) so S' == S exactly. Any deviation is a pure "
        "plumbing/precision bug in the state read-store path, not SFPU function bias.",
    )
    ap.add_argument("--ckpt", default=os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36")))
    args = ap.parse_args()

    cfg = Qwen35MoeConfig.from_hf_config(args.ckpt)
    V, Kh = cfg.linear_num_value_heads, cfg.linear_num_key_heads
    Dk, Dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    key_dim, value_dim = Kh * Dk, V * Dv
    w = {k: (v() if callable(v) else v) for k, v in CheckpointLoader(args.ckpt).gated_delta_weights(args.layer).items()}
    A_log, dt_bias, norm_w = w["A_log"].float(), w["dt_bias"].float(), w["norm"].float()
    negA_v = -A_log.exp()
    if args.no_gate or args.identity:
        negA_v = torch.zeros_like(negA_v)  # g = exp(0 * softplus) = 1 -> pure accumulation
    scale, eps = Dk**-0.5, cfg.rms_norm_eps
    print(f"real dims: V={V} Kh={Kh} Dk={Dk} Dv={Dv}  key_dim={key_dim} value_dim={value_dim}")
    print(f"state = {V * Dk * Dv * 2 / 2**20:.1f} MiB bf16 / {V * Dk * Dv * 4 / 2**20:.1f} MiB fp32\n")

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        torch.manual_seed(0)
        h_q = torch.randn(1, key_dim) * 0.1
        h_k = torch.randn(1, key_dim) * 0.1
        h_v = torch.randn(1, value_dim) * 0.1
        h_z = torch.randn(1, value_dim) * 0.1
        h_a, h_b = torch.randn(V) * 0.5, torch.randn(V) * 0.5
        if args.identity:
            h_b = torch.full((V,), -20.0)  # sigmoid(-20) ~ 2e-9 -> delta ~ 0 -> S' == S*g == S
        h_S = torch.randn(V * Dk, Dv) * 0.05

        def build(dt):
            t = lambda x: ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            rep = lambda x: t(x.reshape(V, 1, 1).expand(V, TILE, TILE).reshape(V * TILE, TILE).contiguous())
            d = dict(
                q=t(h_q),
                kr=t(h_k),
                v=t(h_v),
                z=t(h_z),
                araw=rep(h_a),
                braw=rep(h_b),
                negA=rep(negA_v),
                dtb=rep(dt_bias),
                S=t(h_S),
                nweight=t(norm_w.reshape(1, Dv).expand(TILE, Dv).contiguous()),
                out=t(torch.zeros(1, value_dim)),
            )
            d["snew"] = t(torch.zeros(V * Dk, Dv)) if args.no_alias else d["S"]
            return d

        # fp64 ground truth
        Sr, outr = ref_step(
            h_S.reshape(V, Dk, Dv),
            h_q.reshape(Kh, Dk).repeat_interleave(V // Kh, 0),
            h_k.reshape(Kh, Dk).repeat_interleave(V // Kh, 0),
            h_v.reshape(V, Dv),
            h_z.reshape(V, Dv),
            h_a,
            h_b,
            negA_v,
            dt_bias,
            norm_w,
            scale,
            eps,
        )

        def pcc(a, b):
            a, b = a.flatten().double(), b.flatten().double()
            return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

        results = {}
        for name, dt, fp32 in (("bf16 (today)", ttnn.bfloat16, False), ("fp32 (all-f32)", ttnn.float32, True)):
            bufs = build(dt)
            try:
                decode_step_tt(
                    bufs["q"],
                    bufs["kr"],
                    bufs["v"],
                    bufs["z"],
                    bufs["araw"],
                    bufs["braw"],
                    bufs["negA"],
                    bufs["dtb"],
                    bufs["S"],
                    bufs["nweight"],
                    bufs["out"],
                    bufs["snew"],
                    V,
                    Kh,
                    scale,
                    eps,
                    fp32_state=fp32,
                )
                ttnn.synchronize_device(dev)
            except Exception as e:
                print(f"{name:<16} COMPILE/RUN FAILED:\n  {str(e)[:600]}\n")
                results[name] = None
                continue
            gotS = ttnn.to_torch(bufs["snew"]).float().reshape(V, Dk, Dv)
            p = pcc(Sr, gotS)
            nr = (gotS.double().norm() / Sr.norm()).item()
            print(f"{name:<16} ran OK   state PCC vs fp64 = {p:.8f}   |S|ratio = {nr:.6f}")
            if args.identity:
                # g == 1 and beta ~ 2e-9, so S' must equal the INPUT state to ~4e-9. Comparing against
                # the input (not the fp64 reference) isolates the kernel's own read->store path.
                S_in = h_S.reshape(V, Dk, Dv).double()
                d = (gotS.double() - S_in).abs()
                print(
                    f"{'':<16} vs INPUT state: max|d| {d.max():.3e}  rel {d.max() / S_in.abs().max():.3e}  "
                    f"|S'|/|S_in| {(gotS.double().norm() / S_in.norm()).item():.9f}"
                )

            for _ in range(20):
                decode_step_tt(
                    bufs["q"],
                    bufs["kr"],
                    bufs["v"],
                    bufs["z"],
                    bufs["araw"],
                    bufs["braw"],
                    bufs["negA"],
                    bufs["dtb"],
                    bufs["S"],
                    bufs["nweight"],
                    bufs["out"],
                    bufs["snew"],
                    V,
                    Kh,
                    scale,
                    eps,
                    fp32_state=fp32,
                )
            ttnn.synchronize_device(dev)
            t0 = time.time()
            for _ in range(args.iters):
                decode_step_tt(
                    bufs["q"],
                    bufs["kr"],
                    bufs["v"],
                    bufs["z"],
                    bufs["araw"],
                    bufs["braw"],
                    bufs["negA"],
                    bufs["dtb"],
                    bufs["S"],
                    bufs["nweight"],
                    bufs["out"],
                    bufs["snew"],
                    V,
                    Kh,
                    scale,
                    eps,
                    fp32_state=fp32,
                )
            ttnn.synchronize_device(dev)
            us = (time.time() - t0) / args.iters * 1e6
            print(f"{'':<16} {us:8.1f} us/launch (eager; includes ~190 us host dispatch)")
            results[name] = (p, nr, us)

        # what the CALLER must additionally spend to hand the fp32 kernel fp32 inputs
        print("\ncaller-side cost of producing fp32 q/k/v/z (4 ttnn.typecast on the decode path):")
        src = [
            ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for x in (h_q, h_k, h_v, h_z)
        ]
        for _ in range(20):
            for x in src:
                ttnn.typecast(x, ttnn.float32)
        ttnn.synchronize_device(dev)
        t0 = time.time()
        for _ in range(args.iters):
            for x in src:
                ttnn.typecast(x, ttnn.float32)
        ttnn.synchronize_device(dev)
        us_tc = (time.time() - t0) / args.iters * 1e6
        print(f"  4x typecast: {us_tc:8.1f} us  ({us_tc / 4:.1f} us each)")

        b, f = results.get("bf16 (today)"), results.get("fp32 (all-f32)")
        print("\n--- VERDICT ---")
        if f is None:
            print("NO-GO: the all-fp32 kernel does not build/run. With mixed dtypes also rejected by")
            print("       ttl 1.1.6, there is no fp32-state kernel available at this ttl version.")
        else:
            print(f"accuracy: bf16 PCC {b[0]:.8f} -> fp32 PCC {f[0]:.8f}")
            print(f"|S|ratio: bf16 {b[1]:.6f} -> fp32 {f[1]:.6f}   (1.0 is exact)")
            print(f"kernel:   bf16 {b[2]:.1f} us -> fp32 {f[2]:.1f} us  ({f[2] - b[2]:+.1f} us)")
            print(f"plus caller typecasts: {us_tc:.1f} us -> total delta {f[2] - b[2] + us_tc:+.1f} us/layer")
            print("NOTE eager launches carry ~190 us of HOST dispatch that a traced replay does not, so")
            print("     the eager delta OVERSTATES nothing but the fixed part -- confirm on bench_decode.py.")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
