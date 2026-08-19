# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Cross-validate the standalone PyTorch reference against the canonical HuggingFace
``transformers`` implementation of Qwen3.5-MoE.

Run with the ISOLATED reference venv (transformers>=5.2), NOT the tt-metal python_env:
    /home/carl/qwen_ref_venv/bin/python models/demos/qwen3_6_a3b/tests/cross_validate_reference.py

Builds tiny HF modules, copies their weights into our reference modules (names mirror HF),
runs both on identical inputs, and reports PCC. PCC ~1.0 confirms the port is numerically faithful.
"""
import importlib.util
import os
import sys

import torch

REF_PATH = os.path.join(os.path.dirname(__file__), "..", "reference", "qwen3_5_moe.py")
spec = importlib.util.spec_from_file_location("qref", os.path.abspath(REF_PATH))
qref = importlib.util.module_from_spec(spec)
sys.modules["qref"] = qref  # needed for @dataclass introspection
spec.loader.exec_module(qref)

from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf  # noqa: E402


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def tiny_hf_config():
    return hf.Qwen3_5MoeTextConfig(
        vocab_size=512,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=8,
        num_experts_per_tok=2,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        full_attention_interval=4,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
        max_position_embeddings=4096,
    )


def my_config_from_hf(hfc):
    return qref.Qwen35MoeConfig(
        vocab_size=hfc.vocab_size,
        hidden_size=hfc.hidden_size,
        num_hidden_layers=hfc.num_hidden_layers,
        num_attention_heads=hfc.num_attention_heads,
        num_key_value_heads=hfc.num_key_value_heads,
        head_dim=hfc.head_dim,
        rms_norm_eps=hfc.rms_norm_eps,
        rope_theta=1e7,
        partial_rotary_factor=0.25,
        linear_conv_kernel_dim=hfc.linear_conv_kernel_dim,
        linear_key_head_dim=hfc.linear_key_head_dim,
        linear_value_head_dim=hfc.linear_value_head_dim,
        linear_num_key_heads=hfc.linear_num_key_heads,
        linear_num_value_heads=hfc.linear_num_value_heads,
        num_experts=hfc.num_experts,
        num_experts_per_tok=hfc.num_experts_per_tok,
        moe_intermediate_size=hfc.moe_intermediate_size,
        shared_expert_intermediate_size=hfc.shared_expert_intermediate_size,
        full_attention_interval=4,
    )


def copy_state(dst, src):
    """Load src(HF) state_dict into dst(reference); names mirror HF so this is mostly direct."""
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    return missing, unexpected


def check_attention(hfc, mycfg):
    torch.manual_seed(0)
    hf_mod = hf.Qwen3_5MoeAttention(hfc, layer_idx=3).eval()
    my_mod = qref.Qwen35MoeAttention(mycfg, layer_idx=3).eval()
    m, u = copy_state(my_mod, hf_mod)
    x = torch.randn(1, 16, hfc.hidden_size)
    pos = torch.arange(16)[None]
    rope = hf.Qwen3_5MoeTextRotaryEmbedding(hfc)
    cos, sin = rope(x, pos.expand(3, -1).reshape(3, 1, 16) if False else pos)
    pe = (cos, sin)
    mask = torch.full((16, 16), float("-inf")).triu(1)[None, None]
    with torch.no_grad():
        out_hf, _ = hf_mod(x, position_embeddings=pe, attention_mask=mask)
        out_my = my_mod(x, pe, mask)
    return pcc(out_hf, out_my), (m, u)


def check_moe(hfc, mycfg):
    torch.manual_seed(0)
    hf_mod = hf.Qwen3_5MoeSparseMoeBlock(hfc).eval()
    my_mod = qref.Qwen35MoeSparseMoeBlock(mycfg).eval()
    with torch.no_grad():
        hf_mod.experts.gate_up_proj.normal_(0, 0.05)
        hf_mod.experts.down_proj.normal_(0, 0.05)
        hf_mod.gate.weight.normal_(0, 0.05)
        hf_mod.shared_expert_gate.weight.normal_(0, 0.05)
    copy_state(my_mod, hf_mod)
    x = torch.randn(1, 16, hfc.hidden_size)
    with torch.no_grad():
        out_hf = hf_mod(x)
        out_hf = out_hf[0] if isinstance(out_hf, tuple) else out_hf
        out_my = my_mod(x)
    return pcc(out_hf, out_my)


def check_gated_delta(hfc, mycfg):
    torch.manual_seed(0)
    hf_mod = hf.Qwen3_5MoeGatedDeltaNet(hfc, layer_idx=0).eval()
    my_mod = qref.Qwen35MoeGatedDeltaNet(mycfg, layer_idx=0).eval()
    copy_state(my_mod, hf_mod)
    x = torch.randn(1, 16, hfc.hidden_size)
    with torch.no_grad():
        out_hf = hf_mod(x, cache_params=None)
        out_hf = out_hf[0] if isinstance(out_hf, tuple) else out_hf
        out_my = my_mod(x, cache=None)
    return pcc(out_hf, out_my)


if __name__ == "__main__":
    hfc = tiny_hf_config()
    mycfg = my_config_from_hf(hfc)
    results = {}
    try:
        p, (m, u) = check_attention(hfc, mycfg)
        results["attention"] = p
        if m or u:
            print(f"[attention] missing={list(m)[:6]} unexpected={list(u)[:6]}")
    except Exception as e:
        results["attention"] = f"ERR {type(e).__name__}: {e}"
    for name, fn in [("moe", check_moe), ("gated_delta", check_gated_delta)]:
        try:
            results[name] = fn(hfc, mycfg)
        except Exception as e:
            results[name] = f"ERR {type(e).__name__}: {e}"
    print("\n=== PCC vs HuggingFace ===")
    for k, v in results.items():
        print(f"  {k:14s}: {v if isinstance(v, str) else f'{v:.6f}'}")
    ok = all(isinstance(v, float) and v > 0.99 for v in results.values())
    sys.exit(0 if ok else 1)
