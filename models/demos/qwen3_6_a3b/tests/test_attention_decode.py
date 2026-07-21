# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test for fixed-KV-cache prefill + sdpa_decode single-token decode vs reference."""
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
def test_attention_decode_cache(mesh_device):
    torch.manual_seed(0)
    cfg = Qwen35MoeConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        partial_rotary_factor=0.25,
        rope_theta=1e7,
        rms_norm_eps=1e-6,
    )
    Tp, N = 8, 4
    S = Tp + N
    max_seq = 512
    ref = Qwen35MoeAttention(cfg, layer_idx=3).eval()
    for p in ref.parameters():
        p.normal_(0, 0.04)
    rope = Qwen35TextRotaryEmbedding(cfg)
    x = torch.randn(1, S, cfg.hidden_size) * 0.5
    cos, sin = rope(x, torch.arange(S)[None])
    mask = torch.full((S, S), float("-inf")).triu(1)[None, None]
    y_ref = ref(x, (cos, sin), mask)  # [1, S, hidden]

    weights = {k: getattr(ref, k).weight.data for k in ["q_proj", "k_proj", "v_proj", "o_proj"]}
    weights["q_norm"] = ref.q_norm.weight.data
    weights["k_norm"] = ref.k_norm.weight.data
    rdim = int(cfg.head_dim * cfg.partial_rotary_factor)
    attn = TtAttention(
        mesh_device,
        weights,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        rdim,
        cfg.rms_norm_eps,
        dtype=ttnn.bfloat16,
    )

    kv = [
        ttnn.zeros(
            [1, cfg.num_key_value_heads, max_seq, cfg.head_dim],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )
        for _ in range(2)
    ]

    # prefill Tp
    cos_p = to_tt(cos[:, :Tp].reshape(1, 1, Tp, rdim), mesh_device)
    sin_p = to_tt(sin[:, :Tp].reshape(1, 1, Tp, rdim), mesh_device)
    x_p = to_tt(x[:, :Tp].reshape(1, 1, Tp, cfg.hidden_size), mesh_device)
    outs = [from_tt(attn.forward_prefill(x_p, cos_p, sin_p, kv), mesh_device).reshape(1, Tp, cfg.hidden_size)]

    # decode N (O(1) KV write via paged_update_cache at the current_pos tensor)
    for t in range(Tp, S):
        cos_t = to_tt(cos[:, t : t + 1].reshape(1, 1, 1, rdim), mesh_device)
        sin_t = to_tt(sin[:, t : t + 1].reshape(1, 1, 1, rdim), mesh_device)
        x_t = to_tt(x[:, t : t + 1].reshape(1, 1, 1, cfg.hidden_size), mesh_device)
        cur = to_tt(torch.tensor([t], dtype=torch.int32), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        outs.append(
            from_tt(attn.forward_decode(x_t, cos_t, sin_t, kv, cur), mesh_device).reshape(1, 1, cfg.hidden_size)
        )
    y_tt = torch.cat(outs, dim=1)

    ok, pcc = comp_pcc(y_ref, y_tt, 0.99)
    assert ok, f"Attention decode-cache PCC {pcc}"
