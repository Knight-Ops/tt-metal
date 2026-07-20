# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test for cached single-step decode of the on-device Gated DeltaNet.

prefill(T) then decode the rest one token at a time from the (conv_state, recurrent_state) cache;
compare to the reference run over the FULL sequence. Validates cache continuation == full recompute.
"""
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig, Qwen35MoeGatedDeltaNet
from models.demos.qwen3_6_a3b.tt.common import from_tt, to_tt
from models.demos.qwen3_6_a3b.tt.gated_delta import TtGatedDeltaNet


@torch.no_grad()
def test_gated_delta_decode_cache(mesh_device):
    torch.manual_seed(0)
    cfg = Qwen35MoeConfig(
        hidden_size=256,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    Tp, N = 8, 6  # prefill 8, decode 6 more
    S = Tp + N
    ref = Qwen35MoeGatedDeltaNet(cfg, layer_idx=0).eval()
    for p in ref.parameters():
        p.normal_(0, 0.05)
    ref.A_log.data.uniform_(0, 2).log_()
    ref.dt_bias.data.uniform_(0.5, 1.5)
    x = torch.randn(1, S, cfg.hidden_size) * 0.5
    y_ref = ref(x, cache=None)  # [1, S, hidden]

    weights = {
        "in_proj_qkv": ref.in_proj_qkv.weight.data,
        "in_proj_z": ref.in_proj_z.weight.data,
        "in_proj_b": ref.in_proj_b.weight.data,
        "in_proj_a": ref.in_proj_a.weight.data,
        "out_proj": ref.out_proj.weight.data,
        "conv1d": ref.conv1d.weight.data,
        "norm": ref.norm.weight.data,
        "A_log": ref.A_log.data,
        "dt_bias": ref.dt_bias.data,
    }
    gdn = TtGatedDeltaNet(mesh_device, weights, cfg, dtype=ttnn.bfloat16)

    x_tt = to_tt(x.reshape(1, 1, S, cfg.hidden_size), mesh_device)
    cache = {}
    outs = []
    y0 = gdn.forward(ttnn.slice(x_tt, [0, 0, 0, 0], [1, 1, Tp, cfg.hidden_size]), cache)
    outs.append(from_tt(y0, mesh_device).reshape(1, Tp, cfg.hidden_size))
    for t in range(Tp, S):
        xt = ttnn.slice(x_tt, [0, 0, t, 0], [1, 1, t + 1, cfg.hidden_size])
        yt = gdn.forward(xt, cache)
        outs.append(from_tt(yt, mesh_device).reshape(1, 1, cfg.hidden_size))
    y_tt = torch.cat(outs, dim=1)

    ok, pcc = comp_pcc(y_ref, y_tt, 0.99)
    assert ok, f"GatedDeltaNet decode-cache PCC {pcc}"
