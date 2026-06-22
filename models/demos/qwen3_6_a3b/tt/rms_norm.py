# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm variants for Qwen3.6-35B-A3B.

- ``TtRMSNorm``: standard norm with the Qwen3-Next "(1 + weight)" convention. The +1 is folded
  into the stored weight at load time.
- ``TtRMSNormGated``: norm (weight ones-init) followed by gating with ``silu(gate)`` — used inside
  the Gated DeltaNet on the value head dim.
"""
from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_a3b.tt.common import to_tt


class TtRMSNorm(LightweightModule):
    def __init__(self, mesh_device, weight, eps, add_unit_offset=True, dtype=ttnn.bfloat16):
        super().__init__()
        self.eps = eps
        w = weight.float()
        if add_unit_offset:
            w = 1.0 + w  # fold the (1 + weight) convention into the stored weight
        # gamma layout required by ttnn.rms_norm: [1, 1, dim//32, 32], row-major
        dim = w.numel()
        self.weight = to_tt(w.reshape(1, 1, dim // 32, 32), mesh_device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    def forward(self, x):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)


class TtRMSNormGated(LightweightModule):
    def __init__(self, mesh_device, weight, eps, dtype=ttnn.bfloat16):
        super().__init__()
        self.eps = eps
        dim = weight.numel()
        self.weight = to_tt(
            weight.float().reshape(1, 1, dim // 32, 32), mesh_device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    def forward(self, x, gate):
        y = ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)
        return ttnn.multiply(y, ttnn.silu(gate))
