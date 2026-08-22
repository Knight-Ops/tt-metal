# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the K-token speculative VERIFY path of Gated-DeltaNet + its ROLLBACK.

`forward(..., verify=True)` runs the projections/conv ONCE over the K rows and chains K single-step
ttl launches, leaving the persistent recurrent_state/conv_state untouched so the caller can accept any
prefix and then `commit_verify(cache, j)`.

Three gates:
  1. verify's K outputs == K sequential single-token decode steps (the thing it replaces),
  2. commit_verify(j) leaves the state exactly where j sequential decodes would have,
  3. commit_verify(0) is a no-op (full rejection must not corrupt anything).
"""
import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig
from models.demos.qwen3_6_a3b.tt.common import from_tt, to_tt
from models.demos.qwen3_6_a3b.tt.gated_delta import _STATE_DT, TtGatedDeltaNet

PCC_GATE = 0.99


def _cfg():
    # head_k_dim/head_v_dim = 128 and 32 v-heads: the fused ttl kernel's grid is head-count-driven,
    # so these must stay real (the kernel maps 32 v-heads -> 32 cores).
    return Qwen35MoeConfig(
        hidden_size=512,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )


def _weights(cfg, seed=0):
    torch.manual_seed(seed)
    H = cfg.hidden_size
    kd = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    vd = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    V = cfg.linear_num_value_heads
    conv_dim = 2 * kd + vd
    g = lambda *s: (torch.randn(*s) * 0.04)
    return {
        "in_proj_qkv": g(conv_dim, H),
        "in_proj_z": g(vd, H),
        "in_proj_b": g(V, H),
        "in_proj_a": g(V, H),
        "out_proj": g(H, vd),
        "conv1d": g(conv_dim, 1, cfg.linear_conv_kernel_dim),
        "norm": g(cfg.linear_value_head_dim),
        "A_log": torch.rand(V) * 0.5,
        "dt_bias": torch.rand(V) * 0.1,
    }


def _fresh(mesh_device, cfg, gdn, Tp, x_pre):
    """Prefill Tp tokens to get a non-trivial recurrent_state + conv_state, then return the cache."""
    V = cfg.linear_num_value_heads
    kd, vd = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    cache = {
        "conv_state": ttnn.zeros(
            [cfg.linear_conv_kernel_dim - 1, gdn.conv_dim],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        ),
        # dtype must follow QWEN36_GDN_FP32_STATE: the fused kernel is all-bf16 or all-fp32 (ttl
        # rejects mixed tile arguments), so a hardcoded bf16 state fails to compile against an
        # fp32 build rather than testing it.
        "recurrent_state": ttnn.zeros([1, V, kd, vd], dtype=_STATE_DT, layout=ttnn.TILE_LAYOUT, device=mesh_device),
    }
    gdn.forward(to_tt(x_pre.reshape(1, 1, Tp, cfg.hidden_size), mesh_device), cache=cache)
    return cache


def _state(cache, mesh_device):
    return (from_tt(cache["recurrent_state"], mesh_device).float(), from_tt(cache["conv_state"], mesh_device).float())


@torch.no_grad()
@pytest.mark.parametrize("K", [2, 3, 4])
def test_gated_delta_verify(mesh_device, K):
    cfg = _cfg()
    H, Tp = cfg.hidden_size, 32
    gdn = TtGatedDeltaNet(mesh_device, _weights(cfg), cfg, dtype=ttnn.bfloat16)
    torch.manual_seed(7)
    x_pre = torch.randn(1, Tp, H) * 0.5
    x_new = torch.randn(1, K, H) * 0.5

    # --- reference: K sequential single-token decode steps ---
    c_seq = _fresh(mesh_device, cfg, gdn, Tp, x_pre)
    seq_rows, seq_states = [], []
    for j in range(K):
        y = gdn.forward(to_tt(x_new[:, j : j + 1].reshape(1, 1, 1, H), mesh_device), cache=c_seq, decode=True)
        seq_rows.append(from_tt(y, mesh_device).reshape(1, 1, H))
        seq_states.append(_state(c_seq, mesh_device))
    y_seq = torch.cat(seq_rows, dim=1)

    # --- verify: ONE call over all K rows ---
    c_ver = _fresh(mesh_device, cfg, gdn, Tp, x_pre)
    before = _state(c_ver, mesh_device)
    y_ver = from_tt(
        gdn.forward(to_tt(x_new.reshape(1, 1, K, H), mesh_device), cache=c_ver, verify=True), mesh_device
    ).reshape(1, K, H)

    ok, msg = comp_pcc(y_seq, y_ver, PCC_GATE)
    print(f"[gdn-verify] K={K} outputs vs {K} sequential decodes  PCC {msg}")

    # verify must NOT have advanced the persistent state
    after = _state(c_ver, mesh_device)
    _, m_rs = comp_pcc(before[0], after[0], 0.99999)
    _, m_cs = comp_pcc(before[1], after[1], 0.99999)
    print(f"[gdn-verify] state untouched by verify: rec {m_rs}  conv {m_cs}")
    assert comp_pcc(before[0], after[0], 0.99999)[0], "verify mutated recurrent_state"
    assert comp_pcc(before[1], after[1], 0.99999)[0], "verify mutated conv_state"
    assert ok, msg


@torch.no_grad()
@pytest.mark.parametrize("j", [0, 1, 2, 3])
def test_gated_delta_commit_verify(mesh_device, j):
    """commit_verify(j) must land the state exactly where j sequential decodes would."""
    cfg = _cfg()
    H, Tp, K = cfg.hidden_size, 32, 3
    gdn = TtGatedDeltaNet(mesh_device, _weights(cfg), cfg, dtype=ttnn.bfloat16)
    torch.manual_seed(11)
    x_pre = torch.randn(1, Tp, H) * 0.5
    x_new = torch.randn(1, K, H) * 0.5

    # golden: j sequential decode steps
    c_seq = _fresh(mesh_device, cfg, gdn, Tp, x_pre)
    for t in range(j):
        gdn.forward(to_tt(x_new[:, t : t + 1].reshape(1, 1, 1, H), mesh_device), cache=c_seq, decode=True)
    want_rs, want_cs = _state(c_seq, mesh_device)

    # verify(K) then accept only j
    c_ver = _fresh(mesh_device, cfg, gdn, Tp, x_pre)
    gdn.forward(to_tt(x_new.reshape(1, 1, K, H), mesh_device), cache=c_ver, verify=True)
    gdn.commit_verify(c_ver, j)
    got_rs, got_cs = _state(c_ver, mesh_device)

    ok_rs, m_rs = comp_pcc(want_rs, got_rs, PCC_GATE)
    ok_cs, m_cs = comp_pcc(want_cs, got_cs, PCC_GATE)
    print(f"[gdn-verify] commit(j={j}) recurrent_state  PCC {m_rs}")
    print(f"[gdn-verify] commit(j={j}) conv_state       PCC {m_cs}")
    assert ok_rs, f"recurrent_state after commit({j}): {m_rs}"
    assert ok_cs, f"conv_state after commit({j}): {m_cs}"
