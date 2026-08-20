# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the K-row speculative VERIFY path of full attention (share_cache=True).

`forward_verify(x, cos, sin, kv_cache, positions)` runs the projections/RoPE/norms ONCE over the K
rows (M=K is free) and loops only the tiny (kv-write, sdpa) pair, so each row attends the batch-1
cache up to its own position — exact causal verification of K speculative tokens with no extra memory
and no chunk-alignment constraint. (`sdpa_decode(share_cache=True)` cannot do this: its output batch
comes from the KV cache's batch dim, so a batch-1 cache always returns ONE row.)

`positions` is a DEVICE tensor so the whole path is trace-capturable — a `to_tt` per step would be a
host write during capture.

Two gates:
  1. vs the torch reference at those positions (ground truth),
  2. vs K SEQUENTIAL single-row decodes (the thing the verify pass replaces).
"""
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

PCC_GATE = 0.99
K = 4  # speculative window (gamma + 1)


def _cfg():
    return Qwen35MoeConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        partial_rotary_factor=0.25,
        rope_theta=1e7,
        rms_norm_eps=1e-6,
    )


@torch.no_grad()
def test_attention_verify_share_cache(mesh_device):
    torch.manual_seed(0)
    cfg = _cfg()
    Tp, max_seq = 8, 128
    S = Tp + K
    H, rdim = cfg.hidden_size, int(cfg.head_dim * cfg.partial_rotary_factor)

    ref = Qwen35MoeAttention(cfg, layer_idx=3).eval()
    for p in ref.parameters():
        p.data.normal_(0, 0.04)
    rope = Qwen35TextRotaryEmbedding(cfg)
    x = torch.randn(1, S, H) * 0.5
    cos, sin = rope(x, torch.arange(S)[None])
    y_ref = ref(x, (cos, sin), torch.full((S, S), float("-inf")).triu(1)[None, None])

    weights = {k: getattr(ref, k).weight.data for k in ["q_proj", "k_proj", "v_proj", "o_proj"]}
    weights["q_norm"] = ref.q_norm.weight.data
    weights["k_norm"] = ref.k_norm.weight.data
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

    def fresh_cache():
        kv = [
            ttnn.zeros(
                [1, cfg.num_key_value_heads, max_seq, cfg.head_dim],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )
            for _ in range(2)
        ]
        attn.forward_prefill(
            to_tt(x[:, :Tp].reshape(1, 1, Tp, H), mesh_device),
            to_tt(cos[:, :Tp].reshape(1, 1, Tp, rdim), mesh_device),
            to_tt(sin[:, :Tp].reshape(1, 1, Tp, rdim), mesh_device),
            kv,
        )
        return kv

    def pos_tensor(vals):
        return to_tt(torch.tensor(vals, dtype=torch.int32), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    # --- (2) K sequential single-row decodes: what verify replaces ---
    kv_seq = fresh_cache()
    seq_rows = []
    for j in range(K):
        p = Tp + j
        o = attn.forward_decode(
            to_tt(x[:, p : p + 1].reshape(1, 1, 1, H), mesh_device),
            to_tt(cos[:, p : p + 1].reshape(1, 1, 1, rdim), mesh_device),
            to_tt(sin[:, p : p + 1].reshape(1, 1, 1, rdim), mesh_device),
            kv_seq,
            pos_tensor([p]),
        )
        seq_rows.append(from_tt(o, mesh_device).reshape(1, 1, H))
    y_seq = torch.cat(seq_rows, dim=1)  # [1, K, H]

    # --- (1) ONE share_cache verify call over all K rows ---
    kv_ver = fresh_cache()
    o = attn.forward_verify(
        to_tt(x[:, Tp:S].reshape(1, 1, K, H), mesh_device),
        to_tt(cos[:, Tp:S].reshape(1, 1, K, rdim), mesh_device),
        to_tt(sin[:, Tp:S].reshape(1, 1, K, rdim), mesh_device),
        kv_ver,
        pos_tensor([Tp + j for j in range(K)]),
    )
    y_ver = from_tt(o, mesh_device).reshape(1, K, H)

    ok_ref, msg_ref = comp_pcc(y_ref[:, Tp:S], y_ver, PCC_GATE)
    ok_seq, msg_seq = comp_pcc(y_seq, y_ver, PCC_GATE)
    print(f"[verify] forward_verify(K={K}) vs torch reference  PCC {msg_ref}")
    print(f"[verify] forward_verify(K={K}) vs {K} sequential    PCC {msg_seq}")
    for j in range(K):
        _, m = comp_pcc(y_ref[:, Tp + j : Tp + j + 1], y_ver[:, j : j + 1], PCC_GATE)
        print(f"           row {j} (pos {Tp+j}) vs reference   PCC {m}")

    assert ok_ref, f"vs reference: {msg_ref}"
    assert ok_seq, f"vs sequential decode: {msg_seq}"


@torch.no_grad()
def test_attention_verify_is_causal(mesh_device):
    """The rows must NOT see each other's future: perturbing the LAST speculative token may not change
    any earlier row's output. This is what proves the per-row cur_pos masking is real."""
    torch.manual_seed(1)
    cfg = _cfg()
    Tp, max_seq = 8, 128
    H, rdim = cfg.hidden_size, int(cfg.head_dim * cfg.partial_rotary_factor)

    ref = Qwen35MoeAttention(cfg, layer_idx=3).eval()
    for p in ref.parameters():
        p.data.normal_(0, 0.04)
    rope = Qwen35TextRotaryEmbedding(cfg)
    x = torch.randn(1, Tp + K, H) * 0.5
    cos, sin = rope(x, torch.arange(Tp + K)[None])

    weights = {k: getattr(ref, k).weight.data for k in ["q_proj", "k_proj", "v_proj", "o_proj"]}
    weights["q_norm"] = ref.q_norm.weight.data
    weights["k_norm"] = ref.k_norm.weight.data
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

    def run(xin):
        kv = [
            ttnn.zeros(
                [1, cfg.num_key_value_heads, max_seq, cfg.head_dim],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )
            for _ in range(2)
        ]
        attn.forward_prefill(
            to_tt(xin[:, :Tp].reshape(1, 1, Tp, H), mesh_device),
            to_tt(cos[:, :Tp].reshape(1, 1, Tp, rdim), mesh_device),
            to_tt(sin[:, :Tp].reshape(1, 1, Tp, rdim), mesh_device),
            kv,
        )
        o = attn.forward_verify(
            to_tt(xin[:, Tp:].reshape(1, 1, K, H), mesh_device),
            to_tt(cos[:, Tp:].reshape(1, 1, K, rdim), mesh_device),
            to_tt(sin[:, Tp:].reshape(1, 1, K, rdim), mesh_device),
            kv,
            to_tt(
                torch.tensor([Tp + j for j in range(K)], dtype=torch.int32),
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
        )
        return from_tt(o, mesh_device).reshape(1, K, H)

    base = run(x)
    x2 = x.clone()
    x2[:, -1] += 3.0  # perturb only the LAST speculative token
    pert = run(x2)

    ok_prefix, msg_prefix = comp_pcc(base[:, : K - 1], pert[:, : K - 1], 0.9999)
    changed = float((base[:, -1] - pert[:, -1]).abs().max())
    print(f"[verify] earlier rows unchanged by a last-token perturbation  PCC {msg_prefix}")
    print(f"[verify] last row DID change (max abs delta {changed:.4f})")
    assert ok_prefix, f"causality violated — earlier rows saw the future: {msg_prefix}"
    assert changed > 1e-2, "perturbation had no effect at all; the test is not exercising the path"
