# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced DECODE logit drift: what a decode-path change actually costs, per step.

WHY THIS EXISTS. This model's decode sits at the edge of bf16 determinism -- `EXPERIMENTS.md` records
that a change which *improved* PCC 0.9997 -> 0.99998 still flipped a near-tie and derailed a token
stream. So token-identity is not a usable gate, and MMLU-Redux does not cover the gap either: it is
scored on PREFILL last-token logits, while most decode-path optimisations here are gated on `T == 1`
and are therefore never executed by an MMLU run. `FUTURE_OPTIMIZATIONS.md` names the right instrument
-- "a teacher-forced decode logit-PCC vs baseline" -- and it has only ever been run ad hoc. This is it,
as a standing harness.

WHAT IT MEASURES. One model build, both arms, perfectly paired:
  1. arm A (baseline): prefill, then N greedy decode steps, recording (token, full logits) per step.
  2. arm B (candidate): prefill the same prompt, then replay arm A's tokens -- TEACHER FORCING -- so
     both arms see identical inputs at every step and the only difference is the code under test.
  3. per step: logit PCC, argmax agreement, top-5 overlap, and the argmax MARGIN (how close the step
     was to flipping anyway).

Reading it: a *stable* PCC across steps means the change is a fixed rounding offset -- benign. A PCC
that DRIFTS downward means error is accumulating through the recurrent state, which is the failure mode
that matters for a linear-attention model, and is what killed the reverted conv-fold (0.885). Margins
say how much of any token divergence is attributable to the change rather than to ties that were always
going to be coin tosses.

Arms are selected by nulling the model's own config attributes and module flags at runtime, so no
re-import and no second model load is needed.

Run (40 layers, warm cache ~50-85 s):
    QWEN36_LAYERS=40 ./python_env/bin/python \
        models/demos/qwen3_6_a3b/tests/probe_teacher_forced_drift.py --steps 48
"""
from __future__ import annotations

import argparse
import os
import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt import moe as moe_mod
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs

PROMPT = "The Tenstorrent Blackhole architecture is interesting because"


def snapshot(model):
    """Capture every runtime-selectable perf knob this probe can toggle."""
    return {
        "router_fast": moe_mod._ROUTER_FAST,
        "const_sparsity": moe_mod._CONST_SPARSITY,
        "lm": model._lm_head_pc,
        "layers": [
            (
                layer.moe._pc_gate_w,
                layer.moe._pc_se_gu,
                layer.moe._pc_se_down,
                getattr(layer.mixer, "_pc_kv", None),
                getattr(layer.mixer, "_pc_in_proj", None),
            )
            for layer in model.layers
        ],
    }


def apply_arm(model, snap, on):
    """on=True -> the shipping config; on=False -> every knob in `snap` reverted."""
    moe_mod._ROUTER_FAST = snap["router_fast"] if on else False
    moe_mod._CONST_SPARSITY = snap["const_sparsity"] if on else False
    model._lm_head_pc = snap["lm"] if on else None
    for layer, (g, u, d, kv, ip) in zip(model.layers, snap["layers"]):
        layer.moe._pc_gate_w = g if on else None
        layer.moe._pc_se_gu = u if on else None
        layer.moe._pc_se_down = d if on else None
        if hasattr(layer.mixer, "_pc_kv"):
            layer.mixer._pc_kv = kv if on else None
        if hasattr(layer.mixer, "_pc_in_proj"):
            layer.mixer._pc_in_proj = ip if on else None


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--prompt", type=str, default=PROMPT)
    a = ap.parse_args()

    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=1024)
        t0 = time.time()
        model = TtModel(mesh, args, CheckpointLoader(ckpt), num_layers=n_layers)
        print(f"[drift] {n_layers}-layer model built in {time.time()-t0:.0f}s", flush=True)

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(ckpt)
        ids = torch.tensor([tok.encode(a.prompt)], dtype=torch.long)
        snap = snapshot(model)

        # ---- arm A: baseline, free-running greedy; record tokens AND logits ----
        apply_arm(model, snap, on=False)
        base_logits, base_tokens = [], []
        first = int(model.forward(ids)[0, -1].argmax())
        model.start_decode(first)
        for _ in range(a.steps):
            lg = model.decode_forward_logits()  # [1, vocab] host
            t = int(lg[0].argmax())
            base_logits.append(lg[0].clone())
            base_tokens.append(t)
            model.set_decode_tokens(t)
        print(f"[drift] arm A (baseline) done, {len(base_tokens)} steps", flush=True)

        # ---- arm B: candidate, TEACHER-FORCED on arm A's tokens ----
        apply_arm(model, snap, on=True)
        first_b = int(model.forward(ids)[0, -1].argmax())
        model.start_decode(first)  # force the SAME start, so step 0 is comparable
        rows = []
        for i in range(a.steps):
            lg = model.decode_forward_logits()[0]
            b = base_logits[i]
            top5_b = set(torch.topk(b, 5).indices.tolist())
            top5_n = set(torch.topk(lg, 5).indices.tolist())
            srt = torch.topk(b, 2).values
            rows.append(
                {
                    "pcc": pcc(b, lg),
                    "argmax_same": int(lg.argmax()) == int(b.argmax()),
                    "top5": len(top5_b & top5_n),
                    "margin": (srt[0] - srt[1]).item(),  # baseline's own top1-top2 gap
                }
            )
            model.set_decode_tokens(base_tokens[i])  # teacher forcing

        print(
            f"\n[drift] prefill first token: baseline {first} vs candidate {first_b}"
            f" ({'same' if first == first_b else 'DIFFER'})"
        )
        p = torch.tensor([r["pcc"] for r in rows])
        same = sum(r["argmax_same"] for r in rows)
        t5 = torch.tensor([float(r["top5"]) for r in rows])
        # Trend, judged on the MEDIAN and on a least-squares slope -- not on two half-means. A single
        # bad step (one near-tie whose logits transiently disagree) drags a half-mean by ~1/n of its
        # depth and can read as "drift" when the series is flat, which is exactly what a 96-step run
        # of this model does. Accumulating error in a recurrent state shows up as a NEGATIVE SLOPE
        # over the whole series; a fixed rounding offset shows up as slope ~0 plus outliers.
        idx = torch.arange(len(p), dtype=torch.float64)
        pd = p.double()

        def ols(x, y):
            return (((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()).item()

        slope = ols(idx, pd)
        med = pd.median().item()
        q10 = pd.sort().values[max(0, int(0.1 * len(pd)) - 1)].item()
        h1, h2 = pd[: len(pd) // 2], pd[len(pd) // 2 :]
        # The verdict is taken on MEDIANS, and the OLS slope is reported both with and without the
        # single worst step. One deep outlier at position i contributes (i - mean_i)*(y_i - mean_y) /
        # sum((i - mean_i)^2) to the slope, which for a late outlier in a ~100-step series is enough
        # to fake a drift signal on its own -- measured: at 96 steps one 0.947 step at #94 supplied
        # ~90% of a -3.5e-5 slope whose series was otherwise flat. Judge on the median trend.
        keep = [i for i in range(len(pd)) if i != int(pd.argmin())]
        slope_rb = ols(idx[keep], pd[keep])
        med_trend = h2.median().item() - h1.median().item()
        drift = med_trend < -5e-4 and slope_rb < -2e-5
        print(
            f"[drift] steps={len(rows)}  logit PCC  mean {p.mean():.6f}  median {med:.6f}"
            f"  10th pct {q10:.6f}  min {p.min():.6f}"
        )
        print(
            f"[drift]   median 1st half {h1.median():.6f} -> 2nd half {h2.median():.6f}" f"  (delta {med_trend:+.2e})"
        )
        print(f"[drift]   OLS slope {slope:+.2e}/step   excluding the single worst step {slope_rb:+.2e}/step")
        print(
            f"[drift]   VERDICT: {'DRIFTING (error accumulating in the recurrent state)' if drift else 'STABLE (fixed rounding offset + outliers; no accumulation)'}"
        )
        print(f"[drift]   at the robust slope, PCC after 1000 steps would be {med + slope_rb * 1000:.4f}")
        worst = sorted(range(len(rows)), key=lambda i: rows[i]["pcc"])[:5]
        print("[drift]   worst 5 steps: " + "  ".join(f"#{i}={rows[i]['pcc']:.4f}" for i in worst))
        print(f"[drift] argmax agreement {same}/{len(rows)}   mean top-5 overlap {t5.mean():.2f}/5")
        flips = [(i, r) for i, r in enumerate(rows) if not r["argmax_same"]]
        if flips:
            print("[drift] argmax flips (with the BASELINE's own top1-top2 margin at that step):")
            for i, r in flips:
                print(f"          step {i:3d}  pcc {r['pcc']:.6f}  baseline margin {r['margin']:.4f}")
            m = torch.tensor([r["margin"] for _, r in flips])
            allm = torch.tensor([r["margin"] for r in rows])
            print(
                f"[drift]   flip margins mean {m.mean():.4f} vs all-step mean {allm.mean():.4f}"
                f"  -> flips are {'ON NEAR-TIES (expected)' if m.mean() < allm.mean() else 'NOT tie-driven (investigate)'}"
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
