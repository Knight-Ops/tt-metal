# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""On-device PCC test for the tt-nn Gated DeltaNet (Phase A) vs the validated reference."""
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig, Qwen35MoeGatedDeltaNet
from models.demos.qwen3_6_a3b.tt.common import from_tt, to_tt
from models.demos.qwen3_6_a3b.tt.gated_delta import TtGatedDeltaNet


@torch.no_grad()
def test_gated_delta_prefill(mesh_device):
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
    T = 16
    ref = Qwen35MoeGatedDeltaNet(cfg, layer_idx=0).eval()
    for p in ref.parameters():
        p.normal_(0, 0.05)
    # keep A_log/dt_bias in their natural init ranges
    ref.A_log.data.uniform_(0, 2).log_()
    ref.dt_bias.data.uniform_(0.5, 1.5)
    x = torch.randn(1, T, cfg.hidden_size) * 0.5
    y_ref = ref(x, cache=None)  # [1, T, hidden]

    weights = {
        "in_proj_qkv": ref.in_proj_qkv.weight.data,
        "in_proj_z": ref.in_proj_z.weight.data,
        "in_proj_b": ref.in_proj_b.weight.data,
        "in_proj_a": ref.in_proj_a.weight.data,
        "out_proj": ref.out_proj.weight.data,
        "conv1d": ref.conv1d.weight.data,  # [conv_dim, 1, K]
        "norm": ref.norm.weight.data,
        "A_log": ref.A_log.data,
        "dt_bias": ref.dt_bias.data,
    }
    gdn = TtGatedDeltaNet(mesh_device, weights, cfg, dtype=ttnn.bfloat16)
    x_tt = to_tt(x.reshape(1, 1, T, cfg.hidden_size), mesh_device)
    y_tt = from_tt(gdn.forward(x_tt), mesh_device).reshape(y_ref.shape)

    ok, pcc = comp_pcc(y_ref, y_tt, 0.99)
    assert ok, f"GatedDeltaNet prefill PCC {pcc}"
