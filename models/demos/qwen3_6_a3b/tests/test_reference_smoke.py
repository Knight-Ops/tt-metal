# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Self-consistency smoke tests for the standalone PyTorch reference (no HW, no transformers)."""
import torch

from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import (
    Qwen35MoeAttention,
    Qwen35MoeConfig,
    Qwen35MoeForCausalLM,
    Qwen35MoeGatedDeltaNet,
    Qwen35MoeSparseMoeBlock,
    Qwen35TextRotaryEmbedding,
)


def tiny_config():
    # Small but architecturally faithful: keeps head dims/ratios from the real model.
    return Qwen35MoeConfig(
        vocab_size=512,
        hidden_size=256,
        num_hidden_layers=4,  # exercises 3 linear + 1 full (interval 4)
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        partial_rotary_factor=0.25,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        full_attention_interval=4,
    )


def test_attention_shapes():
    torch.manual_seed(0)
    cfg = tiny_config()
    attn = Qwen35MoeAttention(cfg, layer_idx=3).eval()
    rope = Qwen35TextRotaryEmbedding(cfg)
    x = torch.randn(1, 16, cfg.hidden_size)
    pos = torch.arange(16)[None]
    pe = rope(x, pos)
    mask = torch.full((16, 16), float("-inf")).triu(1)[None, None]
    out = attn(x, pe, mask)
    assert out.shape == x.shape


def test_moe_shapes_and_routing():
    torch.manual_seed(0)
    cfg = tiny_config()
    moe = Qwen35MoeSparseMoeBlock(cfg).eval()
    # give experts non-trivial weights
    with torch.no_grad():
        moe.experts.gate_up_proj.normal_(0, 0.02)
        moe.experts.down_proj.normal_(0, 0.02)
        moe.gate.weight.normal_(0, 0.02)
    x = torch.randn(1, 16, cfg.hidden_size)
    out = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_gated_delta_prefill_matches_recurrent_decode():
    """Chunk (prefill) and recurrent (per-step decode) delta-rule are mathematically equivalent.
    Running the same sequence both ways must give matching outputs -> validates both paths + cache."""
    torch.manual_seed(0)
    cfg = tiny_config()
    gdn = Qwen35MoeGatedDeltaNet(cfg, layer_idx=0).double().eval()
    seq = 12
    x = torch.randn(1, seq, cfg.hidden_size, dtype=torch.float64)

    # Prefill: process whole sequence at once (chunk path), keep state.
    cache_pf = {"conv_state": None, "recurrent_state": None}
    out_prefill = gdn(x, cache=cache_pf)

    # Decode: process token-by-token through the recurrent path with state carried forward.
    conv_dim = gdn.conv_dim
    cache = {
        "conv_state": torch.zeros(1, conv_dim, cfg.linear_conv_kernel_dim - 1, dtype=torch.float64),
        "recurrent_state": torch.zeros(
            1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim, dtype=torch.float64
        ),
    }
    outs = []
    for t in range(seq):
        outs.append(gdn(x[:, t : t + 1], cache=cache))
    out_decode = torch.cat(outs, dim=1)

    pcc = torch.corrcoef(torch.stack([out_prefill.flatten(), out_decode.flatten()]))[0, 1].item()
    assert pcc > 0.999, f"chunk vs recurrent PCC too low: {pcc}"


def test_full_model_runs():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = Qwen35MoeForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.02)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    logits = model(ids)
    assert logits.shape == (1, 16, cfg.vocab_size)
    assert torch.isfinite(logits).all()
