# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the ttnn.sparse_matmul MoE decode path (compute only selected experts)."""
import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig, Qwen35MoeSparseMoeBlock
from models.demos.qwen3_6_a3b.tt import moe_gather
from models.demos.qwen3_6_a3b.tt.common import from_tt, to_tt
from models.demos.qwen3_6_a3b.tt.moe import TtMoE


@torch.no_grad()
def test_moe_sparse_matmul_decode(mesh_device):
    torch.manual_seed(1)
    cfg = Qwen35MoeConfig(
        hidden_size=2048,
        num_experts=32,
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
    x2 = to_tt(x.reshape(1, cfg.hidden_size), mesh_device)
    shared, topv, topi, probs = moe._shared_and_router(x2)
    routing = ttnn.scatter(ttnn.zeros_like(probs), 1, topi, topv)  # [1, E]
    sparsity = ttnn.to_layout(ttnn.reshape(routing, [1, 1, 1, cfg.num_experts]), ttnn.ROW_MAJOR_LAYOUT)
    routed = moe.forward_sparse_decode(x2, sparsity)  # [hidden]
    out = ttnn.add(ttnn.reshape(routed, [1, cfg.hidden_size]), shared)
    y_tt = from_tt(out, mesh_device).reshape(y_ref.shape)

    ok, pcc = comp_pcc(y_ref, y_tt, 0.99)
    assert ok, f"sparse_matmul MoE PCC {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("num_experts", [256])
def test_moe_gather_decode(mesh_device, num_experts):
    """Three-way: the indexed/gather decode path vs the legacy 256-slot scan path vs the torch
    reference, at the production expert count. The gather path must (a) match the reference and (b)
    agree with the scan path to near bit-level (both compute the same selected experts)."""
    torch.manual_seed(2)
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
    x2 = to_tt(x.reshape(1, cfg.hidden_size), mesh_device)
    shared, topv, topi, probs = moe._shared_and_router(x2)
    routing = ttnn.scatter(ttnn.zeros_like(probs), 1, topi, topv)
    sparsity = ttnn.to_layout(ttnn.reshape(routing, [1, 1, 1, cfg.num_experts]), ttnn.ROW_MAJOR_LAYOUT)

    # legacy 256-slot scan path
    routed_scan = moe.forward_sparse_decode(x2, sparsity)
    y_scan = from_tt(ttnn.add(ttnn.reshape(routed_scan, [1, cfg.hidden_size]), shared), mesh_device).reshape(
        y_ref.shape
    )

    # indexed/gather path
    indices = moe_gather.topk_to_indices(topi, cfg.num_experts_per_tok)
    routed_gather = moe.forward_sparse_decode(x2, sparsity, topv, indices)
    y_gather = from_tt(ttnn.add(ttnn.reshape(routed_gather, [1, cfg.hidden_size]), shared), mesh_device).reshape(
        y_ref.shape
    )

    ok_ref, pcc_ref = comp_pcc(y_ref, y_gather, 0.99)
    ok_scan, pcc_scan = comp_pcc(y_scan, y_gather, 0.999)
    assert ok_ref, f"gather vs reference PCC {pcc_ref}"
    assert ok_scan, f"gather vs scan PCC {pcc_scan} (index/weight-pairing bug if low)"
