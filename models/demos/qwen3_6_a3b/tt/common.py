# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared tt-nn helpers for the Qwen3.6-35B-A3B demo (single-device oriented)."""
from __future__ import annotations

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def to_tt(torch_tensor, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=DRAM):
    """Replicate a torch tensor onto the (single- or multi-) device mesh."""
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device is not None else None
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=mapper,
    )


def from_tt(tt_tensor, mesh_device=None):
    """Bring a (replicated) tt tensor back to torch, taking the first device shard."""
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0) if mesh_device is not None else None
    if composer is not None:
        t = ttnn.to_torch(tt_tensor, mesh_composer=composer)
        # replicated -> all shards identical; if concatenated along dim0 take the first copy
        n = mesh_device.get_num_devices()
        if n > 1 and t.shape[0] % n == 0:
            t = t[: t.shape[0] // n]
        return t
    return ttnn.to_torch(tt_tensor)


def as_weight(torch_weight, mesh_device, dtype=ttnn.bfloat16):
    """Convert an nn.Linear weight (out, in) to a tt tensor laid out for ttnn.linear (in, out)."""
    return to_tt(torch_weight.t().contiguous(), mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
