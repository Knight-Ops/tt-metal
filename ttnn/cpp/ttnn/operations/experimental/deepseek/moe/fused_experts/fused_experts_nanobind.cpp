// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_nanobind.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/deepseek/moe/fused_experts/fused_experts.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts::detail {

void bind_fused_experts(nb::module_& mod) {
    ttnn::bind_function<"fused_experts", "ttnn.experimental.deepseek.moe.">(
        mod,
        R"doc(
        Experimental fused routed-expert FFN for DeepSeek V4-Flash decode (sequence length 1).

        Fuses the per-expert matmul -> SwiGLU -> matmul -> weighted-accumulate loop
        into a single device operation. Expert selection/scaling is read on-device from
        ``routing_weights``: the i-th weight pair is scaled by ``routing_weights`` column i,
        so experts with zero routing weight contribute nothing (no host-side expert-id list).

        FIRST VERSION: a single core reads ``routing_weights`` and computes the selected
        ("hit") expert ids on-device, returning them as a [1, 1, 1, E] UINT32 tensor (sorted
        hit ids compacted at the front, remaining slots padded with E). ``routing_weights``
        must be ROW_MAJOR bfloat16. The full fused FFN output is a later milestone.

        Args:
            input_tensor: Activations, [1, 1, 1, H].
            routing_weights: Per-token routing weights, [1, 1, 1, E] (ROW_MAJOR bfloat16),
                with E == len(gate_up_weights).
            gate_up_weights: List of [H, 2I] weight tensors, one per expert.
            down_weights: List of [I, H] weight tensors, one per expert.
            intermediate_size: SwiGLU intermediate size I.
            swiglu_limit: Clamp limit used by the SwiGLU activation.
            memory_config: Optional output memory config.
        )doc",
        &ttnn::experimental::deepseek::moe::fused_experts,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("routing_weights"),
        nb::arg("gate_up_weights"),
        nb::arg("down_weights"),
        nb::arg("intermediate_size"),
        nb::arg("swiglu_limit"),
        nb::arg("memory_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts::detail

namespace ttnn::operations::experimental::deepseek::moe::detail {

void bind_fused_experts(::nanobind::module_& mod) { fused_experts::detail::bind_fused_experts(mod); }

}  // namespace ttnn::operations::experimental::deepseek::moe::detail
