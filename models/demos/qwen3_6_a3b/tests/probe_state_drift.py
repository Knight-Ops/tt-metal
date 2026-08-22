# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does the bf16 Gated-DeltaNet recurrent state DRIFT over a long decode?

THE QUESTION. The checkpoint asks for `mamba_ssm_dtype: "float32"`. Prefill honours that (the
fp32/HiFi4 chunk-state path, QWEN36_TRACED_FP32_RECURRENCE), but the PERSISTENT state and the whole
decode path are bfloat16: the buffer in `TtModel._alloc_cache`, the fused kernel's scratch, and the
in-place write-back in `_forward_decode_fused`.

WHY IT IS NOT THE SAME AS BFP4 WEIGHTS OR A BF16 KV CACHE. Those are read-only: quantisation error
enters once per use and does not accumulate. The GDN state is a FEEDBACK accumulator — every step is
`S <- S*decay + rank-1 update` through an 8-bit mantissa, so rounding error can compound with the
step count. Nothing in the existing validation can see that: module PCC runs at short T (8 prefill +
6 decode in test_gated_delta_decode) and the MMLU-Redux 79.5% number is short-generation.

WHAT THIS MEASURES. One REAL gated-delta layer (real checkpoint weights) stepped N times on device
against the torch reference's fp32 recurrence on identical inputs, reporting how state error and
output error evolve with step count. A per-layer accumulator is the right unit: the whole 40-layer
torch reference will not fit host RAM (72 GB checkpoint, 15 GB box), and drift is a property of the
recurrence, not of the stack.

HOW TO READ THE RESULT. Error that is flat in N (a fixed bf16 noise floor, state PCC steady around
its step-1 value) means the state is fine as bf16 and the question closes. Error that GROWS with N —
state PCC falling, relative error trending up, or the state norm walking away from the reference's —
means the accumulator is compounding and the persistent buffer + `_fused_scratch` should be promoted
to fp32. Do not pre-emptively promote: the conv-fold and sharded-norm results in EXPERIMENTS.md are
a standing reminder that this hardware does not reward changes that only look right on paper.

Run (one layer's weights only, so the load is seconds, not the ~800 s full-model build):
    ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_state_drift.py --steps 4096
    ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_state_drift.py --steps 512 --layer 8
Env: QWEN36_CKPT. Add --random-weights to run without a checkpoint (shape-faithful, values are not).
"""

from __future__ import annotations

import argparse
import os

import torch

import ttnn
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig, Qwen35MoeGatedDeltaNet
from models.demos.qwen3_6_a3b.tt.common import from_tt, to_tt
from models.demos.qwen3_6_a3b.tt.gated_delta import TtGatedDeltaNet
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    if a.numel() == 1:
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _rel_err(ref, got):
    """Max |got-ref| relative to the reference's RMS — the scale-free error the accumulator would grow."""
    denom = ref.float().pow(2).mean().sqrt().clamp_min(1e-12)
    return ((got.float() - ref.float()).abs().max() / denom).item()


def _load_layer(ckpt, layer, cfg, random_weights):
    """Real weights for ONE linear_attention layer, as torch tensors (the loader hands back thunks)."""
    if random_weights:
        ref = Qwen35MoeGatedDeltaNet(cfg, layer_idx=0).eval()
        torch.manual_seed(0)
        for p in ref.parameters():
            p.normal_(0, 0.05)
        ref.A_log.data.uniform_(0, 16).log_()
        ref.dt_bias.data.uniform_(0.5, 1.5)
        ref.norm.weight.data.normal_(1.0, 0.05)
        return ref, {
            "in_proj_qkv": ref.in_proj_qkv.weight.data,
            "in_proj_z": ref.in_proj_z.weight.data,
            "in_proj_b": ref.in_proj_b.weight.data,
            "in_proj_a": ref.in_proj_a.weight.data,
            "out_proj": ref.out_proj.weight.data,
            "conv1d": ref.conv1d.weight.data,
            "norm": ref.norm.weight.data,
            "A_log": ref.A_log.data,
            "dt_bias": ref.dt_bias.data,
        }

    loader = CheckpointLoader(ckpt)
    w = {k: (v() if callable(v) else v) for k, v in loader.gated_delta_weights(layer).items()}
    ref = Qwen35MoeGatedDeltaNet(cfg, layer_idx=0).eval()
    ref.in_proj_qkv.weight.data = w["in_proj_qkv"].float()
    ref.in_proj_z.weight.data = w["in_proj_z"].float()
    ref.in_proj_b.weight.data = w["in_proj_b"].float()
    ref.in_proj_a.weight.data = w["in_proj_a"].float()
    ref.out_proj.weight.data = w["out_proj"].float()
    ref.conv1d.weight.data = w["conv1d"].float()
    ref.norm.weight.data = w["norm"].float()
    ref.A_log.data = w["A_log"].float()
    ref.dt_bias.data = w["dt_bias"].float()
    return ref, w


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1024, help="decode steps to run")
    ap.add_argument("--layer", type=int, default=0, help="linear_attention layer index to probe")
    ap.add_argument("--every", type=int, default=64, help="report interval")
    ap.add_argument("--prefill", type=int, default=64, help="prefill length before decoding")
    ap.add_argument("--random-weights", action="store_true", help="skip the checkpoint")
    ap.add_argument("--ckpt", default=os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36")))
    args = ap.parse_args()

    cfg = Qwen35MoeConfig.from_hf_config(args.ckpt) if not args.random_weights else Qwen35MoeConfig()
    H = cfg.hidden_size
    ref, weights = _load_layer(args.ckpt, args.layer, cfg, args.random_weights)

    torch.manual_seed(0)
    # A decode stream with realistic per-token scale. The residual stream this layer sees is normed,
    # so unit-ish variance is the honest input; the state's own dynamics do the rest.
    xs = torch.randn(1, args.prefill + args.steps, H) * 0.5

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        gdn = TtGatedDeltaNet(mesh_device, weights, cfg, dtype=ttnn.bfloat16)

        # --- prefill both sides to the same starting state ---
        ref_cache = {
            "conv_state": torch.zeros(1, gdn.conv_dim, cfg.linear_conv_kernel_dim - 1),
            "recurrent_state": None,
        }
        pre = xs[:, : args.prefill]
        ref(pre, cache=ref_cache)

        dev_cache = {}
        x_pre = to_tt(pre.reshape(1, 1, args.prefill, H), mesh_device)
        gdn.forward(x_pre, dev_cache)
        gdn.sync_conv_rows(dev_cache)  # decode reads conv_rows; prefill filled conv_state

        V, Dk, Dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        print(
            f"\nlayer {args.layer}  hidden {H}  state [{V},{Dk},{Dv}] = "
            f"{V * Dk * Dv * 2 / 2**20:.2f} MiB bf16   prefill {args.prefill}   steps {args.steps}"
        )
        print(f"{'step':>6}  {'state PCC':>11}  {'state relerr':>12}  {'out PCC':>9}  {'|S|dev/|S|ref':>13}")

        rows = []
        for i in range(args.steps):
            xt = xs[:, args.prefill + i : args.prefill + i + 1]
            y_ref = ref(xt, cache=ref_cache)
            y_dev = gdn.forward(to_tt(xt.reshape(1, 1, 1, H), mesh_device), dev_cache, decode=True)

            step = i + 1
            if step % args.every == 0 or step == 1:
                s_ref = ref_cache["recurrent_state"].reshape(V, Dk, Dv).float()
                s_dev = from_tt(dev_cache["recurrent_state"], mesh_device).reshape(V, Dk, Dv).float()
                nr = s_ref.norm().item()
                row = (
                    step,
                    _pcc(s_ref, s_dev),
                    _rel_err(s_ref, s_dev),
                    _pcc(y_ref, from_tt(y_dev, mesh_device).reshape(y_ref.shape)),
                    (s_dev.norm().item() / nr) if nr > 0 else float("nan"),
                )
                rows.append(row)
                print(f"{row[0]:>6}  {row[1]:>11.6f}  {row[2]:>12.3e}  {row[3]:>9.6f}  {row[4]:>13.6f}")

        # --- verdict: is the error a floor, or is it growing? ---
        if len(rows) >= 3:
            first, last = rows[1], rows[-1]  # rows[0] is step 1 (pure round-off, not yet accumulated)
            d_pcc = last[1] - first[1]
            growth = last[2] / max(first[2], 1e-12)
            print(
                f"\nstate PCC {first[1]:.6f} (step {first[0]}) -> {last[1]:.6f} (step {last[0]}), "
                f"delta {d_pcc:+.2e}; relerr x{growth:.2f} over the run"
            )
            if d_pcc < -1e-3 or growth > 3.0:
                print("VERDICT: DRIFTING — bf16 state error compounds with step count. Promote the")
                print("         persistent recurrent_state + _fused_scratch to fp32 and re-measure.")
            else:
                print("VERDICT: FLAT — error is a bf16 noise floor, not an accumulator. bf16 state is")
                print("         adequate for this length; record the number and close the question.")
            print("\nCaveat: ONE layer. A 40-layer stack composes 40 of these per token, and the")
            print("residual stream can amplify a per-layer floor; treat this as necessary, not sufficient.")
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
