# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""On-device PCC test for the tt-nn MTP (multi-token-prediction) draft head vs the torch reference.

Checkpoint-free: builds a reduced-size `Qwen35MoeMTP` with random weights and feeds the same tensors
to `TtMtpHead` through a stub loader. Covers both entry points:
  * forward_prefill  — teacher-forced over a sequence, which is how the head's own KV cache is primed
  * forward_decode   — one draft step against that primed cache

Also pins the two contract details that probe A5 resolved empirically (see MTP.md): the `fc` concat
order is token-embedding-first, and `forward_decode` returns (post-`mtp.norm`, pre-`mtp.norm`) so a
gamma>1 draft chain can feed the pre-norm state back.
"""
from types import SimpleNamespace

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig, Qwen35MoeMTP, Qwen35TextRotaryEmbedding
from models.demos.qwen3_6_a3b.tt.common import from_tt, to_tt
from models.demos.qwen3_6_a3b.tt.mtp import TtMtpHead

PCC_GATE = 0.99


def _stub_loader(ref):
    """The four `mtp_*` dicts the real CheckpointLoader returns, sourced from the reference module."""
    lay = ref.layers[0]
    return SimpleNamespace(
        mtp_head_weights=lambda: {
            "fc": ref.fc.weight.data,
            "pre_fc_norm_embedding": ref.pre_fc_norm_embedding.weight.data,
            "pre_fc_norm_hidden": ref.pre_fc_norm_hidden.weight.data,
            "norm": ref.norm.weight.data,
        },
        mtp_norm_weights=lambda: {
            "input_layernorm": lay.input_layernorm.weight.data,
            "post_attention_layernorm": lay.post_attention_layernorm.weight.data,
        },
        mtp_attention_weights=lambda: {
            "q_proj": lay.self_attn.q_proj.weight.data,
            "k_proj": lay.self_attn.k_proj.weight.data,
            "v_proj": lay.self_attn.v_proj.weight.data,
            "o_proj": lay.self_attn.o_proj.weight.data,
            "q_norm": lay.self_attn.q_norm.weight.data,
            "k_norm": lay.self_attn.k_norm.weight.data,
        },
        mtp_moe_weights=lambda: {
            "gate": lay.mlp.gate.weight.data,
            "gate_up_proj": lay.mlp.experts.gate_up_proj.data,
            "down_proj": lay.mlp.experts.down_proj.data,
            "se_gate_proj": lay.mlp.shared_expert.gate_proj.weight.data,
            "se_up_proj": lay.mlp.shared_expert.up_proj.weight.data,
            "se_down_proj": lay.mlp.shared_expert.down_proj.weight.data,
            "se_router": lay.mlp.shared_expert_gate.weight.data,
        },
    )


def _stub_args(cfg):
    return SimpleNamespace(
        dim=cfg.hidden_size,
        n_heads=cfg.num_attention_heads,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rotary_dim=int(cfg.head_dim * cfg.partial_rotary_factor),
        norm_eps=cfg.rms_norm_eps,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        moe_intermediate_size=cfg.moe_intermediate_size,
        shared_expert_intermediate_size=cfg.shared_expert_intermediate_size,
        attn_weight_dtype=ttnn.bfloat16,  # bf16 for a clean correctness check
        expert_weight_dtype=ttnn.bfloat16,
        expert_down_weight_dtype=ttnn.bfloat16,
        activation_dtype=ttnn.bfloat16,
        sparse_moe_decode=False,
        compute_kernel_lofi=None,
        weight_cache_path=None,
    )


def _build(seed=0):
    torch.manual_seed(seed)
    # MoE dims stay REAL (hidden 2048 / inter 512): the sparse decode path needs Kt=inter/32=16 to be
    # divisible by _sparse_pc's in0_block_w=16, so a shrunken intermediate_size fatals. Only the
    # expert count and the attention head geometry are reduced, exactly as test_moe.py does.
    cfg = Qwen35MoeConfig(
        vocab_size=256,
        hidden_size=2048,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        partial_rotary_factor=0.25,
        rope_theta=1e7,
        rms_norm_eps=1e-6,
        num_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        mtp_num_hidden_layers=1,
    )
    ref = Qwen35MoeMTP(cfg, concat_order="token_first").eval()
    for p in ref.parameters():
        p.data.normal_(0, 0.04)
    return cfg, ref


@torch.no_grad()
def test_mtp_head(mesh_device):
    """Prefill (KV priming) + one decode step, both vs the reference."""
    cfg, ref = _build()
    Tp, max_seq = 8, 128
    S = Tp + 1
    rdim = int(cfg.head_dim * cfg.partial_rotary_factor)
    H = cfg.hidden_size

    # backbone hidden + the next-token ids this head consumes
    hidden = torch.randn(1, S, H) * 0.5
    tok = torch.randint(0, cfg.vocab_size, (S,))
    emb_w = torch.randn(cfg.vocab_size, H) * 0.04
    y_ref, pre_ref = ref(hidden, emb_w[tok].unsqueeze(0), return_prenorm=True)

    args = _stub_args(cfg)
    head = TtMtpHead(
        mesh_device,
        args,
        _stub_loader(ref),
        embed_weight=to_tt(emb_w, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT),
        lm_head_w=None,
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

    rope = Qwen35TextRotaryEmbedding(cfg)
    cos, sin = rope(hidden, torch.arange(S)[None])

    # --- prefill the first Tp positions (primes the head's own KV cache) ---
    out_p, _ = head.forward_prefill(
        to_tt(hidden[:, :Tp].reshape(1, 1, Tp, H), mesh_device),
        to_tt(emb_w[tok[:Tp]].reshape(1, 1, Tp, H), mesh_device),
        to_tt(cos[:, :Tp].reshape(1, 1, Tp, rdim), mesh_device),
        to_tt(sin[:, :Tp].reshape(1, 1, Tp, rdim), mesh_device),
        kv,
    )
    got_p = from_tt(out_p, mesh_device).reshape(1, Tp, H)
    ok_p, msg_p = comp_pcc(y_ref[:, :Tp], got_p, PCC_GATE)
    print(f"[mtp] prefill  PCC {msg_p}")

    # --- one decode step at position Tp, against the primed cache ---
    out_d, pre_d = head.forward_decode(
        to_tt(hidden[:, Tp : Tp + 1].reshape(1, 1, 1, H), mesh_device),
        to_tt(tok[Tp : Tp + 1].to(torch.int32), mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        to_tt(cos[:, Tp : Tp + 1].reshape(1, 1, 1, rdim), mesh_device),
        to_tt(sin[:, Tp : Tp + 1].reshape(1, 1, 1, rdim), mesh_device),
        kv,
        to_tt(torch.tensor([Tp], dtype=torch.int32), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
    )
    got_d = from_tt(out_d, mesh_device).reshape(1, 1, H)
    got_pre = from_tt(pre_d, mesh_device).reshape(1, 1, H)
    ok_d, msg_d = comp_pcc(y_ref[:, Tp : Tp + 1], got_d, PCC_GATE)
    ok_pre, msg_pre = comp_pcc(pre_ref[:, Tp : Tp + 1], got_pre, PCC_GATE)
    print(f"[mtp] decode   PCC {msg_d}")
    print(f"[mtp] pre-norm PCC {msg_pre}   (the state a gamma>1 draft chain feeds back)")

    assert ok_p, f"prefill {msg_p}"
    assert ok_d, f"decode {msg_d}"
    assert ok_pre, f"pre-norm {msg_pre}"


@torch.no_grad()
def test_mtp_concat_order_is_load_bearing(mesh_device):
    """`fc` halves multiply DISTINCT column blocks, so the concat order is not a free choice: probe A5
    measured alpha = 0.9115 for token-first vs EXACTLY 0.0000 for hidden-first. Pin that the tt head
    matches token-first and is clearly WRONG against a hidden-first reference."""
    cfg, ref = _build()
    H = cfg.hidden_size
    hidden = torch.randn(1, 4, H) * 0.5
    emb = torch.randn(1, 4, H) * 0.04

    y_token_first = ref.combine(hidden, emb)
    ref.concat_order = "hidden_first"
    y_hidden_first = ref.combine(hidden, emb)

    args = _stub_args(cfg)
    head = TtMtpHead(mesh_device, args, _stub_loader(ref), embed_weight=None, lm_head_w=None)
    got = from_tt(
        head.combine(to_tt(hidden.reshape(1, 1, 4, H), mesh_device), to_tt(emb.reshape(1, 1, 4, H), mesh_device)),
        mesh_device,
    ).reshape(1, 4, H)

    ok_t, msg_t = comp_pcc(y_token_first, got, PCC_GATE)
    _, msg_h = comp_pcc(y_hidden_first, got, PCC_GATE)
    print(f"[mtp] combine vs token_first  {msg_t}")
    print(f"[mtp] combine vs hidden_first {msg_h}  (must NOT match)")
    assert ok_t, f"tt combine must match token_first: {msg_t}"
    assert not comp_pcc(y_hidden_first, got, PCC_GATE)[0], "hidden_first must not also match"
