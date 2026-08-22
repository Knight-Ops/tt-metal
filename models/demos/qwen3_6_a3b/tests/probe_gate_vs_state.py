# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""STAGE 0: is the measured GDN drift caused by the bf16 GATE or the bf16 STATE?

`probe_state_drift.py` measured, on device with real layer-0 weights, that the bf16 recurrent state
drifts: state PCC 0.99986 -> 0.93323 and |S|dev/|S|ref -> 1.187 over 8192 steps, monotone. That says
*something* bf16 in the recurrence compounds. It does NOT say which bf16 value is losing the
information, and the two candidates have fixes that differ by orders of magnitude in cost:

  GATE  -- `g = exp(-exp(A_log)*softplus(a+dt_bias))` is materialised into a bf16 CB in the ttl
           kernel (`gbd = mk(S, ...)` inherits S's dtype; ttl_delta.py:520-524). bf16 has NO
           representable value between 0.99609375 and 1.0, so ANY g > 0.998046875 rounds to exactly
           1.0 and the decay is not degraded, it is DELETED. A long-memory head then applies no decay
           at all while the reference applies g^N -- which predicts exactly the observed signature
           (state fails to shrink, |S| inflates, monotone, no plateau).
           Fix: store d = 1-g instead of g (bf16 gives a small d full RELATIVE precision). ~1 us,
           no dtype change anywhere.

  STATE -- storing S itself with an 8-bit mantissa, so `S*g` rounds back to S for g close to 1.
           This is round-to-nearest and S's position inside its ulp bin is re-randomised every step
           by the rank-1 update, so it should be a zero-mean sqrt(N) walk, not a ramp.
           Fix: fp32 state -> the full fp32 kernel, with L1, DST-capacity and DFB consequences.

The device measurement cannot separate them (both mechanisms need g near 1, and the "random weights
are FLAT" datapoint suppresses BOTH -- random A_log gives exp(A_log) in [1, 8.9e6], i.e. g ~ 0 and a
memoryless state). So this probe separates them on the HOST: no device, no kernel edits, minutes.

It simulates the ttl kernel's exact arithmetic -- including the intermediate bf16 `Sg` buffer and the
bf16 `st` writeback -- on the REAL layer-0 A_log/dt_bias and the real projections, with the gate
and/or the state selectively quantised.

    (i)   fp32 g / fp32 S    reference
    (ii)  bf16 g / fp32 S    GATE mechanism alone
    (iii) fp32 g / bf16 S    STATE mechanism alone
    (iv)  bf16 g / bf16 S    what ships today  (must reproduce the device number)
    (v)   bf16 d / bf16 S    the proposed cheap fix

HOW TO READ IT. If (ii) reproduces the device signature and (iii) is materially flatter, the gate
dominates and the cheap fix is the fix. If (iii) dominates, fp32 state is required. If (iv) does not
reproduce the device number at all, the drift is not a dtype effect and the whole fp32 plan is aimed
at the wrong target -- which is worth finding out for the price of this script.

Run (loads ONE layer, so seconds not the ~800 s full build):
    ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_gate_vs_state.py --steps 8192
    ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_gate_vs_state.py --steps 2048 --layer 12
Env: QWEN36_CKPT. --random-weights runs without a checkpoint (and is EXPECTED to show no drift --
that is the false-negative this probe exists to explain, not a pass).
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig, Qwen35MoeGatedDeltaNet, l2norm
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader

BF16_NEXTBELOW_ONE = 0.99609375  # largest bf16 < 1.0; any g above (1+this)/2 rounds to exactly 1.0
GATE_CLIFF = (1.0 + BF16_NEXTBELOW_ONE) / 2  # 0.998046875


def bf16(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through bfloat16 -- what storing into a bf16 CB does."""
    return x.to(torch.bfloat16).to(torch.float32)


def bf16_trunc(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 by TRUNCATION toward zero (drop the low 16 bits), not round-to-nearest.

    THE HYPOTHESIS THIS TESTS. ttl_delta.py:67 records that matmul INPUTS are bf16 on Tensix
    regardless of the CB dtype. So in the all-fp32 kernel every matmul operand is *converted* f32->bf16,
    while in the all-bf16 kernel the values are already bf16 and nothing is converted. If that
    conversion truncates rather than rounds, it is a one-sided magnitude deficit on every step -- which
    would explain why the measured fp32 build LOSES magnitude (|S| x0.735 at 8192 steps) where the bf16
    build GAINS it (x1.187), and why this host model -- which assumed round-to-nearest -- predicted
    x1.000 instead.
    """
    i = x.float().view(torch.int32)
    return (i & ~0xFFFF).view(torch.float32)


def tf32(x: torch.Tensor) -> torch.Tensor:
    """Round-trip through TF32 (1+8+10 bits) -- what an f32 CB costs WITHOUT UnpackToDestFp32.

    tt-metal's accuracy guide (tech_reports/op_kernel_dev/accuracy_tips/accuracy_tips.md:58-60): an
    intermediate CB not configured UnpackToDestFp32 has "the data converted to TF32 before being
    copied to dest". ttl sets that mode only for SFPU consumers; FPU-eligible add/sub/mul route
    through SRCA/SRCB and must stay Default. So a naively-promoted f32 carry consumed by an
    FPU-eligible multiply gets 10 mantissa bits, not 23 -- and whether 10 is enough decides whether
    the kernel has to force the carry onto the SFPU path (`--no-ttl-fpu-binary-ops`) or not.
    Emulated by truncating fp32's 23-bit mantissa to 10 (round-to-nearest-even).
    """
    i = x.float().view(torch.int32)
    # round-to-nearest-even on the 13 bits being discarded
    lsb = 1 << 13
    half = lsb >> 1
    rem = i & (lsb - 1)
    up = (rem > half) | ((rem == half) & ((i & lsb) != 0))
    i = (i & ~(lsb - 1)) + torch.where(up, torch.full_like(i, lsb), torch.zeros_like(i))
    return i.view(torch.float32)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _load_layer(ckpt: str, layer: int, cfg, random_weights: bool):
    ref = Qwen35MoeGatedDeltaNet(cfg, layer_idx=0).eval()
    if random_weights:
        torch.manual_seed(0)
        for p in ref.parameters():
            p.normal_(0, 0.05)
        ref.A_log.data.uniform_(0, 16).log_()
        ref.dt_bias.data.uniform_(0.5, 1.5)
        ref.norm.weight.data.normal_(1.0, 0.05)
        return ref
    w = {k: (v() if callable(v) else v) for k, v in CheckpointLoader(ckpt).gated_delta_weights(layer).items()}
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "conv1d", "norm"):
        getattr(ref, name).weight.data = w[name].float()
    ref.A_log.data = w["A_log"].float()
    ref.dt_bias.data = w["dt_bias"].float()
    return ref


@torch.no_grad()
def build_inputs(ref, cfg, xs):
    """The per-step (q, k, v, beta, g) stream, computed ONCE so every config sees identical inputs.

    The conv1d is causal, so running the projections over the whole sequence at once yields exactly
    the q/k/v a step-by-step decode would see. q/k are l2-normalised and q is scaled here, matching
    `use_qk_l2norm_in_kernel=True` and the kernel's fused `l2norm(q)*scale`.
    """
    B, T, _ = xs.shape
    Vh, Kh = cfg.linear_num_value_heads, cfg.linear_num_key_heads
    Dk, Dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim

    mixed = ref.in_proj_qkv(xs).transpose(1, 2)  # [B, conv_dim, T]
    mixed = F.silu(ref.conv1d(mixed)[:, :, :T]).transpose(1, 2)  # causal conv -> [B, T, conv_dim]
    q, k, v = torch.split(mixed, [ref.key_dim, ref.key_dim, ref.value_dim], dim=-1)
    q = q.reshape(B, T, Kh, Dk)
    k = k.reshape(B, T, Kh, Dk)
    v = v.reshape(B, T, Vh, Dv)

    beta = ref.in_proj_b(xs).sigmoid()  # [B, T, Vh]
    a = ref.in_proj_a(xs)
    g_log = -ref.A_log.float().exp() * F.softplus(a.float() + ref.dt_bias)  # [B, T, Vh], negative

    q = l2norm(q, dim=-1, eps=1e-6) * (Dk**-0.5)
    k = l2norm(k, dim=-1, eps=1e-6)
    rep = Vh // Kh
    q = q.repeat_interleave(rep, dim=2)
    k = k.repeat_interleave(rep, dim=2)
    return q[0], k[0], v[0], beta[0], g_log[0].exp()  # drop batch; g is the multiplicative decay


@torch.no_grad()
def run(
    q,
    k,
    v,
    beta,
    g,
    *,
    quant_gate: bool,
    quant_state: bool,
    quant_readout: bool,
    d_form: bool,
    steps: int,
    every: int,
    carry: str = "auto",
    gate: str = "auto",
    readout: str = "rne",
):
    """Simulate the ttl kernel's recurrence, selectively quantising gate / carry / read-out.

    Mirrors _get_decode_op's arithmetic (ttl_delta.py:564-586) including every intermediate the
    kernel really stores into a bf16 DFB today:
        Sg  = S*g      -> `Sg`    (mk(S,...)  -> inherits S's dtype)
        kv  = k @ Sg   -> `kv`    (mk(out,...) -> bf16)
        dl  = (v-kv)*b -> `dl`    (mk(out,...) -> bf16)
        out = k^T @ dl -> `outer` (mk(S,...)  -> inherits S's dtype)
        st  = Sg + out -> `st`/`Snew`
    `quant_state` covers the two on the CARRY path (Sg, st); `quant_readout` covers the read-out
    path (kv, dl, outer), which stays bf16 in the Stage-2 design because matmul inputs are bf16 on
    Tensix regardless (ttl_delta.py:67). Setting both reproduces what ships today.
    """
    # `carry`/`gate` override the bool flags with an explicit precision: "bf16" | "tf32" | "fp32".
    qc = {"bf16": bf16, "tf32": tf32, "fp32": lambda t: t}[
        carry if carry != "auto" else ("bf16" if quant_state else "fp32")
    ]
    qg = {"bf16": bf16, "tf32": tf32, "fp32": lambda t: t}[
        gate if gate != "auto" else ("bf16" if quant_gate else "fp32")
    ]
    # `readout` = "rne" (round-to-nearest bf16) or "trunc" (truncate toward zero, the suspected
    # Tensix f32->bf16 matmul-operand conversion).
    qr = {"rne": bf16, "trunc": bf16_trunc}[readout]
    Vh, Dk, Dv = v.shape[1], q.shape[2], v.shape[2]
    S = torch.zeros(Vh, Dk, Dv, dtype=torch.float32)
    snaps = {}
    for t in range(steps):
        gt = g[t].view(Vh, 1, 1)
        if d_form:
            Sg = S - S * qg(1.0 - gt)
        else:
            Sg = S * qg(gt)
        Sg = qc(Sg)
        kt, vt, bt = k[t].unsqueeze(-1), v[t], beta[t].unsqueeze(-1)
        # the matmul operands are what get converted, so quantise Sg/kt on the way IN
        kv = (qr(Sg) * qr(kt)).sum(dim=-2) if quant_readout else (Sg * kt).sum(dim=-2)
        if quant_readout:
            kv = qr(kv)
        delta = (vt - kv) * bt
        if quant_readout:
            delta = qr(delta)
        outer = qr(kt) * qr(delta).unsqueeze(-2) if quant_readout else kt * delta.unsqueeze(-2)
        if quant_readout:
            outer = qr(outer)
        S = Sg + outer
        S = qc(S)
        step = t + 1
        if step == 1 or step % every == 0 or step == steps:
            snaps[step] = S.clone()
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8192)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--every", type=int, default=1024)
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--ckpt", default=os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36")))
    args = ap.parse_args()

    cfg = Qwen35MoeConfig() if args.random_weights else Qwen35MoeConfig.from_hf_config(args.ckpt)
    ref = _load_layer(args.ckpt, args.layer, cfg, args.random_weights)

    torch.manual_seed(0)
    xs = torch.randn(1, args.steps, cfg.hidden_size) * 0.5
    q, k, v, beta, g = build_inputs(ref, cfg, xs)

    # --- the gate distribution: how much of this layer is even ABOVE the bf16 cliff? ---
    gmean = g.mean(dim=0)  # [Vh] mean decay per head over the run
    n_del = int((gmean > GATE_CLIFF).sum())
    print(
        f"\nlayer {args.layer}  {'RANDOM WEIGHTS' if args.random_weights else 'real weights'}  "
        f"heads {g.shape[1]}  steps {args.steps}"
    )
    print(f"bf16 cliff: any g > {GATE_CLIFF} rounds to EXACTLY 1.0 (decay deleted)")
    print(f"  mean-g per head: min {gmean.min():.6f}  median {gmean.median():.6f}  max {gmean.max():.8f}")
    print(
        f"  heads with mean g above the cliff: {n_del}/{g.shape[1]}"
        f"   ({100.0 * n_del / g.shape[1]:.0f}% would lose their decay entirely)"
    )
    frac_above = (g > GATE_CLIFF).float().mean().item()
    print(f"  fraction of ALL (step, head) gate values above the cliff: {frac_above:.4f}")
    print(
        f"  N*mean(1-g) per head: min {(args.steps * (1 - gmean)).min():.3f}  "
        f"max {(args.steps * (1 - gmean)).max():.1f}   (how much decay the run should apply)"
    )

    # `quant_readout` is bf16 in EVERY shippable config (matmul inputs are bf16 on Tensix), so it is
    # on for all of them; only the gate and the carry are the design variables.
    RO = dict(quant_readout=True)
    configs = [
        ("i   fp32 g / fp32 carry (reference)", dict(quant_gate=False, quant_state=False, d_form=False, **RO)),
        ("ii  bf16 g / fp32 carry (GATE only)", dict(quant_gate=True, quant_state=False, d_form=False, **RO)),
        ("iii fp32 g / bf16 carry (CARRY only)", dict(quant_gate=False, quant_state=True, d_form=False, **RO)),
        ("iv  bf16 g / bf16 carry (SHIPS TODAY)", dict(quant_gate=True, quant_state=True, d_form=False, **RO)),
        ("v   bf16 d / bf16 carry (Stage 1 fix)", dict(quant_gate=True, quant_state=True, d_form=True, **RO)),
        ("vi  fp32 d / fp32 carry (Stage 2 fix)", dict(quant_gate=False, quant_state=False, d_form=True, **RO)),
        # Does the carry actually need TRUE fp32, or is TF32 (what an f32 CB gives an FPU-eligible
        # consumer, with no UnpackToDestFp32) enough? This decides whether the kernel must force the
        # carry onto the SFPU path via --no-ttl-fpu-binary-ops.
        ("vii fp32 g / TF32 carry", dict(quant_gate=False, quant_state=False, d_form=False, carry="tf32", **RO)),
        (
            "viii TF32 g / TF32 carry",
            dict(quant_gate=False, quant_state=False, d_form=False, carry="tf32", gate="tf32", **RO),
        ),
        # THE DEVICE-DISAGREEMENT TEST: fp32 gate+carry, but the matmul operands converted by
        # TRUNCATION rather than round-to-nearest. If this reproduces the measured |S| x0.735, the
        # f32->bf16 matmul-operand conversion is the reason the fp32 build loses magnitude.
        (
            "ix  fp32/fp32, TRUNCATING read-out",
            dict(quant_gate=False, quant_state=False, d_form=False, quant_readout=True, readout="trunc"),
        ),
        (
            "x   bf16/bf16, TRUNCATING read-out",
            dict(quant_gate=True, quant_state=True, d_form=False, quant_readout=True, readout="trunc"),
        ),
    ]
    results = {}
    for name, kw in configs:
        results[name] = run(q, k, v, beta, g, steps=args.steps, every=args.every, **kw)
        print(f"  ...ran {name.split()[0]}")

    # Ground truth is the all-fp32 recurrence (no bf16 anywhere), which is what the torch
    # reference in probe_state_drift.py computes -- NOT config i (which has bf16 read-outs).
    ref_snaps = run(
        q,
        k,
        v,
        beta,
        g,
        quant_gate=False,
        quant_state=False,
        quant_readout=False,
        d_form=False,
        steps=args.steps,
        every=args.every,
    )
    steps_sorted = sorted(ref_snaps)

    print(f"\n{'config':<36} " + "  ".join(f"{s:>9}" for s in steps_sorted))
    print(f"{'-' * 36} " + "  ".join("-" * 9 for _ in steps_sorted))
    for name, _ in configs:
        pccs = [f"{_pcc(ref_snaps[s], results[name][s]):>9.6f}" for s in steps_sorted]
        print(f"{'PCC  ' + name:<36} " + "  ".join(pccs))
    print()
    for name, _ in configs:
        rats = [f"{(results[name][s].norm() / ref_snaps[s].norm()).item():>9.4f}" for s in steps_sorted]
        print(f"{'|S|ratio ' + name:<36} " + "  ".join(rats))

    # --- verdict ---
    last = steps_sorted[-1]
    p_base = _pcc(ref_snaps[last], results[configs[0][0]][last])
    p_gate = _pcc(ref_snaps[last], results[configs[1][0]][last])
    p_state = _pcc(ref_snaps[last], results[configs[2][0]][last])
    p_today = _pcc(ref_snaps[last], results[configs[3][0]][last])
    p_fix = _pcc(ref_snaps[last], results[configs[4][0]][last])
    r_today = (results[configs[3][0]][last].norm() / ref_snaps[last].norm()).item()
    r_fix = (results[configs[4][0]][last].norm() / ref_snaps[last].norm()).item()
    p_s2 = _pcc(ref_snaps[last], results[configs[5][0]][last])
    r_s2 = (results[configs[5][0]][last].norm() / ref_snaps[last].norm()).item()
    p_tf = _pcc(ref_snaps[last], results[configs[6][0]][last])
    r_tf = (results[configs[6][0]][last].norm() / ref_snaps[last].norm()).item()
    p_tf2 = _pcc(ref_snaps[last], results[configs[7][0]][last])
    r_tf2 = (results[configs[7][0]][last].norm() / ref_snaps[last].norm()).item()
    p_tr = _pcc(ref_snaps[last], results[configs[8][0]][last])
    r_tr = (results[configs[8][0]][last].norm() / ref_snaps[last].norm()).item()

    print(f"\n--- at {last} steps (vs the all-fp32 recurrence)")
    print(f"    read-out-bf16 floor {p_base:.6f} | gate-only {p_gate:.6f} | carry-only {p_state:.6f}")
    print(f"    ships today {p_today:.6f} (|S| x{r_today:.3f})")
    print(f"    Stage 1 d-form {p_fix:.6f} (|S| x{r_fix:.3f})   Stage 2 fp32 {p_s2:.6f} (|S| x{r_s2:.3f})")
    print(f"    TF32 carry {p_tf:.6f} (|S| x{r_tf:.3f})   TF32 gate+carry {p_tf2:.6f} (|S| x{r_tf2:.3f})")
    print(f"    fp32 + TRUNCATING read-out {p_tr:.6f} (|S| x{r_tr:.3f})   <- device measured x0.735")
    if r_tr < 0.95:
        print("    -> TRUNCATION REPRODUCES the device's magnitude loss. The f32->bf16 matmul-operand")
        print("       conversion is the bug, not the fp32 carry. Fix: keep the carry off the matmul")
        print("       operand path, or pre-round before the matmul consumes it.")
    else:
        print("    -> truncation does NOT explain the device result; look elsewhere.")
    if p_tf > 0.999 and abs(r_tf - 1.0) < 0.01:
        print("    -> TF32 carry SUFFICES: an f32 CB on the FPU path is enough; no need to force SFPU")
        print("       routing (--no-ttl-fpu-binary-ops). Removes the plan's top precision risk.")
    else:
        print("    -> TF32 carry is NOT enough: the carry must reach the SFPU/UnpackToDestFp32 path")
        print("       (--no-ttl-fpu-binary-ops or a DST-resident multiplicand). Plan for it.")
    print("Device reference for 'today' (probe_state_drift.py, 8192 steps): PCC 0.93323, |S| x1.187\n")

    if p_today > 0.999 and abs(r_today - 1.0) < 0.01:
        print("VERDICT: NO DRIFT REPRODUCED. The host simulation of today's arithmetic does not show")
        print("         the device's drift. Either the input stream is unrepresentative or the drift")
        print("         is NOT a dtype effect -- in which case the fp32 plan is aimed at the wrong")
        print("         target. Do NOT proceed to the kernel work.")
    elif p_fix <= p_today:
        print("VERDICT: THE d-FORM IS REFUTED -- it is WORSE than what ships today")
        print(f"         (PCC {p_fix:.6f} vs {p_today:.6f}; |S| x{r_fix:.3f} vs x{r_today:.3f}).")
        print("         Reason: with a bf16 CARRY, `S - S*d` rounds straight back to S -- the decay")
        print("         correction is smaller than bf16's ulp at S -- so an exact gate buys nothing.")
        print("         The two bf16 errors were also partially CANCELLING, so removing one exposes")
        print("         the other. Each mechanism ALONE is worse than both together:")
        print(f"           gate-only {p_gate:.6f} | carry-only {p_state:.6f} | both {p_today:.6f}")
        if p_s2 > 0.999 and abs(r_s2 - 1.0) < 0.01:
            print(f"         fp32 gate + fp32 carry reaches PCC {p_s2:.6f} / |S| x{r_s2:.3f} -- that is the fix,")
            print("         and it must be BOTH. Note the bf16 read-out floor is")
            print(f"         {p_base:.6f}, so leaving the matmul operands (Sgm/stm/outer) bf16 costs")
            print("         nothing -- which is what keeps L1 and the DST budget affordable.")
        else:
            print(f"         WARNING: even fp32 gate + fp32 carry only reaches {p_s2:.6f} / |S| x{r_s2:.3f}.")
            print("         Something outside the gate/carry pair is contributing; investigate before")
            print("         committing to the kernel work.")
    elif (1.0 - p_gate) > 3 * (1.0 - p_state):
        print("VERDICT: GATE-DOMINATED and the d-form helps. Ship the d-form first (~1 us, no dtype")
        print(f"         change): PCC {p_fix:.6f} / |S| x{r_fix:.3f} at {last} steps.")
    else:
        print("VERDICT: BOTH MECHANISMS MATERIAL, and the d-form helps but does not close the gap.")
        print(f"         d-form {p_fix:.6f} vs fp32 {p_s2:.6f}. Plan on the fp32 carry.")
    print("\nCaveat: ONE layer, and a random-normal hidden stream rather than a real prompt's. This")
    print("separates the mechanisms; it does not replace the on-device number.")


if __name__ == "__main__":
    main()
