# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Unit test for ttnn.experimental.deepseek.moe.fused_experts (first version).

The first version of the op performs the expert *selection* that the model currently
does on host:

    rw_host = ttnn.to_torch(routing_weights).reshape(T, E)
    hit = (rw_host.abs().sum(dim=0) > 0).nonzero().flatten().tolist()

A single core reads the routing weights, computes the hit expert ids on-device, and
writes them to a [1, 1, 1, E] UINT32 output tensor: the sorted hit ids compacted at the
front, with the remaining slots padded with the sentinel value E ("no expert").

Decode-only: sequence length T == 1.
"""

import pytest
import torch
import ttnn
import random


def _hit_ids(routing: torch.Tensor, num_experts: int) -> list[int]:
    """Host reference matching the model's `hit` computation, padded to length E."""
    hit = (routing.abs().sum(dim=0) > 0).nonzero().flatten().tolist()
    return hit + [num_experts] * (num_experts - len(hit))


@pytest.mark.parametrize(
    "hidden, intermediate, num_experts, num_nonzero",
    [
        (128, 64, 8, 5),
        (256, 128, 16, 11),
        (128, 64, 8, 8),  # all experts selected
        # DeepSeek-V4-Flash config sizes (hidden_size=4096, moe_intermediate_size=2048).
        # The model has n_routed_experts=256; we use fewer here to keep DRAM/host memory
        # tractable for a unit test (each [4096, 4096] gate_up weight is ~32 MB).
        (4096, 2048, 64, 6),
    ],
)
def test_fused_experts_expert_ids(device, hidden, intermediate, num_experts, num_nonzero):
    torch.manual_seed(0)
    limit = 7.0
    tokens = 1  # decode: sequence length T == 1

    x = (torch.rand((tokens, hidden), dtype=torch.bfloat16) - 0.5).float()
    x_flat = x.reshape(1, 1, tokens, hidden)

    # Routing weights [T, E]; nonzero columns are the "selected" experts. Use values
    # well above bf16 rounding noise so abs() stays strictly positive. `num_nonzero`
    # evenly spaced columns stay nonzero; the rest are zeroed.
    routing = torch.rand((tokens, num_experts), dtype=torch.bfloat16).float() + 0.5
    nonzero_cols = random.sample(range(num_experts), num_nonzero)
    for c in range(num_experts):
        if c not in nonzero_cols:
            routing[:, c] = 0.0
    routing_4d = routing.reshape(1, 1, tokens, num_experts)

    expected_ids = _hit_ids(routing, num_experts)

    # Weights are required by the op signature (E experts) but unused by this version.
    gate_up_weights = [
        (torch.rand((hidden, 2 * intermediate), dtype=torch.bfloat16) - 0.5).float() for _ in range(num_experts)
    ]
    down_weights = [
        (torch.rand((intermediate, hidden), dtype=torch.bfloat16) - 0.5).float() for _ in range(num_experts)
    ]

    def to_tt(t, layout, dtype=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dtype, device=device, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    tt_out = ttnn.experimental.deepseek.moe.fused_experts(
        to_tt(x_flat, ttnn.TILE_LAYOUT),
        routing_weights=ttnn.from_torch(
            routing_4d,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        gate_up_weights=[to_tt(w, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b) for w in gate_up_weights],
        down_weights=[to_tt(w, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b) for w in down_weights],
        intermediate_size=intermediate,
        swiglu_limit=limit,
    )

    got_ids = ttnn.to_torch(tt_out).flatten().to(torch.int64).tolist()

    assert got_ids == expected_ids, f"expert ids mismatch: got {got_ids}, expected {expected_ids}"
