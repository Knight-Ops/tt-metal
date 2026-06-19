// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek::moe {

// Fused routed-expert FFN for DeepSeek V4-Flash decode (sequence length T == 1).
//
// Replaces the per-expert host loop
//   for e in experts:
//       gate_up = matmul(x, gate_up_w[e]); act = swiglu(gate_up, intermediate, limit)
//       down    = matmul(act, down_w[e]);  acc += down * routing_weights[:, e]
// with a single device operation. Expert selection/scaling is read on-device from
// `routing_weights` (no host-side expert-id / "hit" list): the i-th weight pair is scaled by
// routing_weights column i, so experts with zero routing weight contribute nothing.
//
// Args:
//   input_tensor:     activations, [1, 1, 1, H] (decode).
//   routing_weights:  per-token routing weights, [1, 1, 1, E], with E == gate_up_weights.size().
//   gate_up_weights:  one [H, 2I] weight tensor per expert.
//   down_weights:     one [I, H] weight tensor per expert.
//   intermediate_size: SwiGLU intermediate size I.
//   swiglu_limit:     clamp limit used by the SwiGLU activation.
//   memory_config:    optional output memory config (defaults to the input's).
//
// FIRST VERSION: instead of the FFN result, this op currently returns the on-device-computed
// selected expert ids as a [1, 1, 1, E] UINT32 tensor (slots [0, num_hits) hold the sorted hit
// expert ids; remaining slots are padded with E). routing_weights must be ROW_MAJOR bfloat16.
// The fused matmul/SwiGLU output ([1, 1, 1, H]) is a later milestone.
Tensor fused_experts(
    const Tensor& input_tensor,
    const Tensor& routing_weights,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    uint32_t intermediate_size,
    float swiglu_limit,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental::deepseek::moe
