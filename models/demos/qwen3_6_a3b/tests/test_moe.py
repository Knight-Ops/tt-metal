# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""On-device PCC test for the tt-nn MoE block vs the validated reference."""
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig, Qwen35MoeSparseMoeBlock
from models.demos.qwen3_6_a3b.tt.common import from_tt, to_tt
from models.demos.qwen3_6_a3b.tt.moe import TtMoE


def _build_ref_and_weights(num_experts, seed=0):
    torch.manual_seed(seed)
    # Reduced expert count for a fast test; the tt-nn code is parameterized to the real 256/8.
    cfg = Qwen35MoeConfig(
        hidden_size=2048,
        num_experts=num_experts,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        rms_norm_eps=1e-6,
    )
    ref = Qwen35MoeSparseMoeBlock(cfg).eval()
    ref.experts.gate_up_proj.normal_(0, 0.04)
    ref.experts.down_proj.normal_(0, 0.04)
    ref.gate.weight.normal_(0, 0.04)
    ref.shared_expert_gate.weight.normal_(0, 0.04)
    for p in ref.shared_expert.parameters():
        p.normal_(0, 0.04)
    weights = {
        "gate": ref.gate.weight.data,  # [E, hidden]
        "gate_up_proj": ref.experts.gate_up_proj.data,  # [E, 2*inter, hidden]
        "down_proj": ref.experts.down_proj.data,  # [E, hidden, inter]
        "se_gate_proj": ref.shared_expert.gate_proj.weight.data,
        "se_up_proj": ref.shared_expert.up_proj.weight.data,
        "se_down_proj": ref.shared_expert.down_proj.weight.data,
        "se_router": ref.shared_expert_gate.weight.data,  # [1, hidden]
    }
    return cfg, ref, weights


@torch.no_grad()
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("T", [64, 96])  # multi-tile (T%32==0) exercises the per-tile sparse path
def test_moe(mesh_device, T, sparse):
    """Prefill PCC vs the reference, for both the default dense path and the opt-in sparse path."""
    cfg, ref, weights = _build_ref_and_weights(num_experts=64)
    x = torch.randn(1, T, cfg.hidden_size) * 0.5
    y_ref = ref(x)  # [1, T, hidden]

    # bf16 experts for a clean correctness check (real run uses bfloat4_b).
    moe = TtMoE(
        mesh_device, weights, cfg.num_experts, cfg.num_experts_per_tok, expert_dtype=ttnn.bfloat16, dtype=ttnn.bfloat16
    )
    x_tt = to_tt(x.reshape(1, 1, T, cfg.hidden_size), mesh_device)

    if sparse:
        os.environ["QWEN36_SPARSE_PREFILL"] = "1"
    try:
        y_tt = from_tt(moe.forward(x_tt), mesh_device).reshape(y_ref.shape)
    finally:
        os.environ.pop("QWEN36_SPARSE_PREFILL", None)

    ok, pcc = comp_pcc(y_ref, y_tt, 0.99)
    assert ok, f"MoE {'sparse' if sparse else 'dense'}-prefill PCC {pcc}"


@torch.no_grad()
def test_moe_prefill_sparse_matches_dense(mesh_device):
    """The opt-in sparse per-tile prefill must equal the default dense prefill to bf16 noise."""
    cfg, ref, weights = _build_ref_and_weights(num_experts=64)
    T = 96
    x = torch.randn(1, T, cfg.hidden_size) * 0.5
    moe = TtMoE(
        mesh_device, weights, cfg.num_experts, cfg.num_experts_per_tok, expert_dtype=ttnn.bfloat16, dtype=ttnn.bfloat16
    )
    x_tt = to_tt(x.reshape(1, 1, T, cfg.hidden_size), mesh_device)

    y_dense = from_tt(moe.forward(x_tt), mesh_device)  # default
    os.environ["QWEN36_SPARSE_PREFILL"] = "1"
    try:
        y_sparse = from_tt(moe.forward(x_tt), mesh_device)
    finally:
        os.environ.pop("QWEN36_SPARSE_PREFILL", None)

    ok, pcc = comp_pcc(y_dense, y_sparse, 0.999)
    assert ok, f"sparse vs dense prefill PCC {pcc}"


@torch.no_grad()
def test_moe_decode_sparse(mesh_device):
    """T=1 exercises the gather-top-k sparse decode path."""
    torch.manual_seed(1)
    cfg = Qwen35MoeConfig(
        hidden_size=2048,
        num_experts=64,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        rms_norm_eps=1e-6,
    )
    ref = Qwen35MoeSparseMoeBlock(cfg).eval()
    ref.experts.gate_up_proj.normal_(0, 0.04)
    ref.experts.down_proj.normal_(0, 0.04)
    ref.gate.weight.normal_(0, 0.04)
    ref.shared_expert_gate.weight.normal_(0, 0.04)
    for p in ref.shared_expert.parameters():
        p.normal_(0, 0.04)
    x = torch.randn(1, 1, cfg.hidden_size) * 0.5
    y_ref = ref(x)

    weights = {
        "gate": ref.gate.weight.data,
        "gate_up_proj": ref.experts.gate_up_proj.data,
        "down_proj": ref.experts.down_proj.data,
        "se_gate_proj": ref.shared_expert.gate_proj.weight.data,
        "se_up_proj": ref.shared_expert.up_proj.weight.data,
        "se_down_proj": ref.shared_expert.down_proj.weight.data,
        "se_router": ref.shared_expert_gate.weight.data,
    }
    moe = TtMoE(
        mesh_device, weights, cfg.num_experts, cfg.num_experts_per_tok, expert_dtype=ttnn.bfloat16, dtype=ttnn.bfloat16
    )
    x_tt = to_tt(x.reshape(1, 1, 1, cfg.hidden_size), mesh_device)
    y_tt = from_tt(moe.forward(x_tt), mesh_device).reshape(y_ref.shape)
    ok, pcc = comp_pcc(y_ref, y_tt, 0.99)
    assert ok, f"MoE decode-sparse PCC {pcc}"
