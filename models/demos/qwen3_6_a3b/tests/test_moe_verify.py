# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the K-token speculative VERIFY path of the MoE block (`TtMoE.forward_verify`).

Gates that one K-row indexed/gather call equals (a) the torch reference and (b) K independent
single-token decode calls — i.e. the K rows do not contaminate each other through the shared
expert-slot axis, which is the one real hazard in the ROWSEL combine.
"""
import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig, Qwen35MoeSparseMoeBlock
from models.demos.qwen3_6_a3b.tt.common import from_tt, to_tt
from models.demos.qwen3_6_a3b.tt.moe import TtMoE

PCC_GATE = 0.99


def _build(num_experts=32, seed=0):
    torch.manual_seed(seed)
    # Real hidden/intermediate dims: _sparse_pc's in0_block_w=16 must divide Kt=inter/32.
    cfg = Qwen35MoeConfig(
        hidden_size=2048,
        num_experts=num_experts,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        rms_norm_eps=1e-6,
    )
    ref = Qwen35MoeSparseMoeBlock(cfg).eval()
    ref.experts.gate_up_proj.data.normal_(0, 0.04)
    ref.experts.down_proj.data.normal_(0, 0.04)
    ref.gate.weight.data.normal_(0, 0.04)
    ref.shared_expert_gate.weight.data.normal_(0, 0.04)
    for p in ref.shared_expert.parameters():
        p.data.normal_(0, 0.04)
    weights = {
        "gate": ref.gate.weight.data,
        "gate_up_proj": ref.experts.gate_up_proj.data,
        "down_proj": ref.experts.down_proj.data,
        "se_gate_proj": ref.shared_expert.gate_proj.weight.data,
        "se_up_proj": ref.shared_expert.up_proj.weight.data,
        "se_down_proj": ref.shared_expert.down_proj.weight.data,
        "se_router": ref.shared_expert_gate.weight.data,
    }
    return cfg, ref, weights


@torch.no_grad()
@pytest.mark.parametrize("K", [1, 2, 3, 4, 8])
def test_moe_verify(mesh_device, K):
    # sparse_matmul requires num_active (= K*top_k) <= num_experts. On the real model E=256 bounds
    # K <= 32 (gamma <= 31), never binding; the reduced test needs the expert count sized to K.
    cfg, ref, weights = _build(num_experts=max(32, K * 8))
    H = cfg.hidden_size
    x = torch.randn(1, K, H) * 0.5
    y_ref = ref(x)  # [1, K, hidden]

    moe = TtMoE(
        mesh_device,
        weights,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        expert_dtype=ttnn.bfloat16,
        dtype=ttnn.bfloat16,
    )
    x_tt = to_tt(x.reshape(1, 1, K, H), mesh_device)

    got = from_tt(moe.forward_verify(x_tt), mesh_device).reshape(1, K, H)
    ok, msg = comp_pcc(y_ref, got, PCC_GATE)
    print(f"[moe-verify] K={K} vs reference   PCC {msg}")

    # ...and each row must equal the SAME token run as a lone decode step (no cross-row leakage)
    per_row = torch.cat(
        [
            from_tt(moe.forward(to_tt(x[:, j : j + 1].reshape(1, 1, 1, H), mesh_device)), mesh_device).reshape(1, 1, H)
            for j in range(K)
        ],
        dim=1,
    )
    ok_rows, msg_rows = comp_pcc(per_row, got, PCC_GATE)
    print(f"[moe-verify] K={K} vs {K}x decode  PCC {msg_rows}")
    assert ok, f"vs reference: {msg}"
    assert ok_rows, f"vs per-token decode: {msg_rows}"


@torch.no_grad()
def test_moe_verify_no_row_leakage(mesh_device):
    """Perturbing ONE row must leave the other rows bit-comparable: the expert-slot axis is shared
    across rows, so a wrong ROWSEL/combine would smear one token's experts onto its neighbours."""
    cfg, ref, weights = _build(seed=3)
    H, K = cfg.hidden_size, 4
    moe = TtMoE(
        mesh_device,
        weights,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        expert_dtype=ttnn.bfloat16,
        dtype=ttnn.bfloat16,
    )
    x = torch.randn(1, K, H) * 0.5
    base = from_tt(moe.forward_verify(to_tt(x.reshape(1, 1, K, H), mesh_device)), mesh_device).reshape(1, K, H)
    x2 = x.clone()
    x2[:, 1] += 5.0  # perturb row 1 only (large enough to change its expert routing)
    pert = from_tt(moe.forward_verify(to_tt(x2.reshape(1, 1, K, H), mesh_device)), mesh_device).reshape(1, K, H)

    keep = torch.cat([base[:, 0:1], base[:, 2:]], dim=1)
    keep_p = torch.cat([pert[:, 0:1], pert[:, 2:]], dim=1)
    ok, msg = comp_pcc(keep, keep_p, 0.9999)
    moved = float((base[:, 1] - pert[:, 1]).abs().max())
    print(f"[moe-verify] untouched rows unchanged  PCC {msg}")
    print(f"[moe-verify] perturbed row moved       max abs delta {moved:.4f}")
    assert ok, f"row leakage: {msg}"
    assert moved > 1e-2, "perturbation had no effect; test not exercising the path"
