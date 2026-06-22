# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""On-device PCC tests for RMSNorm variants vs the validated reference."""
import torch

from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen3NextRMSNorm, Qwen3NextRMSNormGated
from models.demos.qwen3_6_a3b.tt.common import from_tt, to_tt
from models.demos.qwen3_6_a3b.tt.rms_norm import TtRMSNorm, TtRMSNormGated


@torch.no_grad()
def test_rmsnorm(mesh_device):
    torch.manual_seed(0)
    dim = 2048
    ref = Qwen3NextRMSNorm(dim, eps=1e-6).eval()
    ref.weight.normal_(0, 0.2)
    x = torch.randn(1, 1, 32, dim)
    y_ref = ref(x)

    tt = TtRMSNorm(mesh_device, ref.weight.data, eps=1e-6, add_unit_offset=True)
    x_tt = to_tt(x, mesh_device)
    y_tt = from_tt(tt.forward(x_tt), mesh_device).reshape(y_ref.shape)

    ok, pcc = comp_pcc(y_ref, y_tt, 0.999)
    assert ok, f"RMSNorm PCC {pcc}"


@torch.no_grad()
def test_rmsnorm_gated(mesh_device):
    torch.manual_seed(0)
    dim = 128  # head_v_dim
    ref = Qwen3NextRMSNormGated(dim, eps=1e-6).eval()
    ref.weight.normal_(1.0, 0.1)
    x = torch.randn(1, 1, 64, dim)
    gate = torch.randn(1, 1, 64, dim)
    y_ref = ref(x, gate)

    tt = TtRMSNormGated(mesh_device, ref.weight.data, eps=1e-6)
    x_tt = to_tt(x, mesh_device)
    g_tt = to_tt(gate, mesh_device)
    y_tt = from_tt(tt.forward(x_tt, g_tt), mesh_device).reshape(y_ref.shape)

    ok, pcc = comp_pcc(y_ref, y_tt, 0.999)
    assert ok, f"RMSNormGated PCC {pcc}"
