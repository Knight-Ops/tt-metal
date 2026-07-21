# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""On-device PCC test for tt-nn full attention vs the validated reference (prefill)."""
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import (
    Qwen35MoeAttention,
    Qwen35MoeConfig,
    Qwen35TextRotaryEmbedding,
)
from models.demos.qwen3_6_a3b.tt.attention import TtAttention
from models.demos.qwen3_6_a3b.tt.common import from_tt, to_tt


@torch.no_grad()
def test_attention_prefill(mesh_device):
    torch.manual_seed(0)
    cfg = Qwen35MoeConfig(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        partial_rotary_factor=0.25,
        rope_theta=1e7,
        rms_norm_eps=1e-6,
    )
    S = 128
    ref = Qwen35MoeAttention(cfg, layer_idx=3).eval()
    for p in ref.parameters():
        p.normal_(0, 0.04)
    rope = Qwen35TextRotaryEmbedding(cfg)
    x = torch.randn(1, S, cfg.hidden_size) * 0.5
    pos = torch.arange(S)[None]
    cos, sin = rope(x, pos)  # [1, S, 64]
    mask = torch.full((S, S), float("-inf")).triu(1)[None, None]
    y_ref = ref(x, (cos, sin), mask)  # [1, S, hidden]

    weights = {
        "q_proj": ref.q_proj.weight.data,
        "k_proj": ref.k_proj.weight.data,
        "v_proj": ref.v_proj.weight.data,
        "o_proj": ref.o_proj.weight.data,
        "q_norm": ref.q_norm.weight.data,
        "k_norm": ref.k_norm.weight.data,
    }
    rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    attn = TtAttention(
        mesh_device,
        weights,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        rotary_dim,
        cfg.rms_norm_eps,
        dtype=ttnn.bfloat16,
    )
    x_tt = to_tt(x.reshape(1, 1, S, cfg.hidden_size), mesh_device)
    cos_tt = to_tt(cos.reshape(1, 1, S, rotary_dim), mesh_device)
    sin_tt = to_tt(sin.reshape(1, 1, S, rotary_dim), mesh_device)
    y_tt = from_tt(attn.forward(x_tt, cos_tt, sin_tt), mesh_device).reshape(y_ref.shape)

    ok, pcc = comp_pcc(y_ref, y_tt, 0.99)
    assert ok, f"Attention prefill PCC {pcc}"
