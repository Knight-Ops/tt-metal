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


@torch.no_grad()
def test_attention_verify_batched_matches_loop(mesh_device):
    """PAIRED gate on the batched-SDPA verify: the row loop is the reference implementation.

    `_verify_batched` (QWEN36_VERIFY_BATCH_SDPA, default on) emits all K rows from ONE paged
    sdpa_decode over a [K,1] page table instead of looping one flat sdpa_decode per row. The two must
    agree, and the KV cache each leaves behind must be IDENTICAL -- the batched form writes all K rows
    before the single SDPA instead of interleaving write and read, so this also pins that the reordering
    is causally harmless.

    Batching `paged_update_cache` the same way is NOT safe (K cores read-modify-write one 32-row tile);
    tests/probe_packed_verify.py has the measurement. Hence: batched read, looped write.
    """
    import models.demos.qwen3_6_a3b.tt.attention as attn_mod

    torch.manual_seed(2)
    cfg = _cfg()
    Tp, max_seq = 64, 256
    Kb = 8  # wider than the shipping gamma=2 so the K-row page table is exercised properly
    S = Tp + Kb
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

    def run(batched):
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
        saved = attn_mod._VERIFY_BATCH_SDPA
        attn_mod._VERIFY_BATCH_SDPA = batched
        try:
            o = attn.forward_verify(
                to_tt(x[:, Tp:S].reshape(1, 1, Kb, H), mesh_device),
                to_tt(cos[:, Tp:S].reshape(1, 1, Kb, rdim), mesh_device),
                to_tt(sin[:, Tp:S].reshape(1, 1, Kb, rdim), mesh_device),
                kv,
                to_tt(
                    torch.tensor([Tp + j for j in range(Kb)], dtype=torch.int32),
                    mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
            )
        finally:
            attn_mod._VERIFY_BATCH_SDPA = saved
        return from_tt(o, mesh_device).reshape(1, Kb, H), [from_tt(c, mesh_device) for c in kv]

    y_loop, kv_loop = run(False)
    y_batch, kv_batch = run(True)

    ok_ab, msg_ab = comp_pcc(y_loop, y_batch, 0.9999)
    ok_ref, msg_ref = comp_pcc(y_ref[:, Tp:S], y_batch, PCC_GATE)
    print(f"[verify] batched(K={Kb}) vs row loop      PCC {msg_ab}")
    print(f"[verify] batched(K={Kb}) vs torch         PCC {msg_ref}")
    for name, a, b in (("K", kv_loop[0], kv_batch[0]), ("V", kv_loop[1], kv_batch[1])):
        err = (a - b).abs().max().item()
        print(f"[verify] {name} cache max|loop - batched| = {err:.3e}")
        assert err == 0.0, f"{name} cache differs between the two write orders (max {err})"
    assert ok_ab, f"batched vs loop: {msg_ab}"
    assert ok_ref, f"batched vs reference: {msg_ref}"


@torch.no_grad()
def test_attention_verify_wide_grid_matches_loop(mesh_device):
    """The WIDE-grid verify SDPA must still agree with the row loop.

    `_verify_sdpa_pc` widens the SDPA core grid once `K * pos_hint` clears
    QWEN36_VERIFY_SDPA_WIDE_WORK, which is worth 0.89-0.95x of the verify SDPA at long context. A
    different grid means a different flash-attention reduction order, so unlike the narrow-grid batched
    path this is NOT expected to be bit-identical to the loop -- it is expected to be numerically
    equivalent. Gate it as such, and pin that the KV writes (which the grid does not touch) stay exact.

    Note the default `pos_hint=None` keeps the shipped config, which is why
    test_attention_verify_batched_matches_loop can still demand PCC 1.0: only a caller with a host-side
    position opts in.
    """
    import models.demos.qwen3_6_a3b.tt.attention as attn_mod

    torch.manual_seed(3)
    cfg = _cfg()
    Tp, max_seq, Kb = 128, 512, 8
    S = Tp + Kb
    H, rdim = cfg.hidden_size, int(cfg.head_dim * cfg.partial_rotary_factor)

    ref = Qwen35MoeAttention(cfg, layer_idx=3).eval()
    for p in ref.parameters():
        p.data.normal_(0, 0.04)
    rope = Qwen35TextRotaryEmbedding(cfg)
    x = torch.randn(1, S, H) * 0.5
    cos, sin = rope(x, torch.arange(S)[None])

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
    # force the wide branch regardless of this tiny config's K*pos
    assert attn._verify_sdpa_pc(Kb, None) is attn.sdpa_decode_pc, "pos_hint=None must keep the shipped config"
    saved_work = attn_mod._VERIFY_SDPA_WIDE_WORK
    saved_wide = attn_mod._VERIFY_SDPA_WIDE
    attn_mod._VERIFY_SDPA_WIDE_WORK = 1
    attn_mod._VERIFY_SDPA_WIDE = True  # default is OFF (see the flag comment); this test is its gate
    wide = attn._verify_sdpa_pc(Kb, Tp)
    assert wide is not attn.sdpa_decode_pc, "the wide branch did not engage"

    def run(batched, hint):
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
        saved = attn_mod._VERIFY_BATCH_SDPA
        attn_mod._VERIFY_BATCH_SDPA = batched
        try:
            o = attn.forward_verify(
                to_tt(x[:, Tp:S].reshape(1, 1, Kb, H), mesh_device),
                to_tt(cos[:, Tp:S].reshape(1, 1, Kb, rdim), mesh_device),
                to_tt(sin[:, Tp:S].reshape(1, 1, Kb, rdim), mesh_device),
                kv,
                to_tt(
                    torch.tensor([Tp + j for j in range(Kb)], dtype=torch.int32),
                    mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                pos_hint=hint,
            )
        finally:
            attn_mod._VERIFY_BATCH_SDPA = saved
        return from_tt(o, mesh_device).reshape(1, Kb, H), [from_tt(c, mesh_device) for c in kv]

    try:
        y_loop, kv_loop = run(False, None)
        y_wide, kv_wide = run(True, Tp)
    finally:
        attn_mod._VERIFY_SDPA_WIDE_WORK = saved_work
        attn_mod._VERIFY_SDPA_WIDE = saved_wide

    ok, msg = comp_pcc(y_loop, y_wide, 0.999)
    print(f"[verify] wide-grid batched(K={Kb}) vs row loop  PCC {msg}")
    for name, a, b in (("K", kv_loop[0], kv_wide[0]), ("V", kv_loop[1], kv_wide[1])):
        err = (a - b).abs().max().item()
        print(f"[verify] {name} cache max|loop - wide| = {err:.3e}")
        assert err == 0.0, f"{name} cache differs; the grid must not touch the KV writes (max {err})"
    assert ok, f"wide grid vs loop: {msg}"
