// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

FusedExpertsDeviceOperation::program_factory_t FusedExpertsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return MultiCore{};
}

void FusedExpertsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.input_tensor;
    const auto& rw = tensor_args.routing_weights;

    (void)attributes;
    TT_FATAL(x.storage_type() == StorageType::DEVICE, "fused_experts: input_tensor must be on device");
    TT_FATAL(rw.storage_type() == StorageType::DEVICE, "fused_experts: routing_weights must be on device");
    // input_tensor is broadcast verbatim page-by-page, so any layout (ROW_MAJOR / TILE) is fine.

    // First version reads routing_weights element-by-element on a single core, so it must be a
    // contiguous ROW_MAJOR bfloat16 row.
    TT_FATAL(rw.layout() == tt::tt_metal::Layout::ROW_MAJOR, "fused_experts: routing_weights must be ROW_MAJOR layout");
    TT_FATAL(
        rw.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "fused_experts: routing_weights must be BFLOAT16, got {}",
        rw.dtype());

    TT_FATAL(
        tensor_args.gate_up_weights.size() == tensor_args.down_weights.size(),
        "fused_experts: gate_up_weights ({}) and down_weights ({}) must have the same length",
        tensor_args.gate_up_weights.size(),
        tensor_args.down_weights.size());

    // Expert selection/scaling is on-device: weight i is scaled by routing_weights column i, so the
    // routing-weight width must match the number of weight pairs.
    const uint32_t num_experts = static_cast<uint32_t>(tensor_args.gate_up_weights.size());
    TT_FATAL(num_experts > 0, "fused_experts: need at least one expert");
    TT_FATAL(
        static_cast<uint32_t>(rw.logical_shape()[-1]) == num_experts,
        "fused_experts: routing_weights last dim ({}) must equal the number of experts ({})",
        rw.logical_shape()[-1],
        num_experts);

    // gate_up weights must be DRAM ND-sharded so that each shard is exactly one core's
    // [K, 64] column slice (read in a single NoC read by the dataflow kernels). The 8x8
    // compute grid owns 64 column slices of 2 tiles (64 cols) each.
    constexpr uint32_t kColsPerCore = 64;
    for (uint32_t e = 0; e < num_experts; ++e) {
        const auto& w = tensor_args.gate_up_weights[e];
        TT_FATAL(
            w.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "fused_experts: gate_up_weights[{}] must be in DRAM",
            e);
        const auto& nd = w.memory_config().nd_shard_spec();
        TT_FATAL(nd.has_value(), "fused_experts: gate_up_weights[{}] must be ND-sharded (one shard per core)", e);
        const auto& shard_shape = nd->shard_shape;
        TT_FATAL(
            static_cast<uint32_t>(shard_shape[-1]) == kColsPerCore,
            "fused_experts: gate_up_weights[{}] shard last dim ({}) must be {} (one core's 2-tile column slice)",
            e,
            shard_shape[-1],
            kColsPerCore);
        TT_FATAL(
            static_cast<uint32_t>(shard_shape[-2]) == static_cast<uint32_t>(w.logical_shape()[-2]),
            "fused_experts: gate_up_weights[{}] shard must span the full K dim ({} rows), got {}",
            e,
            w.logical_shape()[-2],
            shard_shape[-2]);
    }

    // Decode-only: sequence length T == 1.
    TT_FATAL(
        static_cast<uint32_t>(x.logical_shape()[-2]) == 1,
        "fused_experts: decode op expects sequence length 1, got {}",
        x.logical_shape()[-2]);

    // TODO: validate per-expert weight shapes against H / 2I / I, dtypes, and tile alignment.
}

void FusedExpertsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

FusedExpertsDeviceOperation::spec_return_value_t FusedExpertsDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // First version: the output holds the selected expert ids, [1, 1, 1, E] UINT32 ROW_MAJOR.
    // Slots [0, num_hits) hold the sorted hit expert ids; the rest are padded with `E` ("no expert").
    const uint32_t num_experts = static_cast<uint32_t>(tensor_args.gate_up_weights.size());
    const ttnn::Shape output_shape({1, 1, 1, num_experts});
    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            attributes.output_memory_config));
}

FusedExpertsDeviceOperation::tensor_return_value_t FusedExpertsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

std::tuple<FusedExpertsDeviceOperation::operation_attributes_t, FusedExpertsDeviceOperation::tensor_args_t>
FusedExpertsDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& routing_weights,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    uint32_t intermediate_size,
    float swiglu_limit,
    const std::optional<MemoryConfig>& memory_config) {
    operation_attributes_t attributes{
        .intermediate_size = intermediate_size,
        .swiglu_limit = swiglu_limit,
        .output_memory_config = memory_config.value_or(input_tensor.memory_config()),
    };
    tensor_args_t tensor_args{
        .input_tensor = input_tensor,
        .routing_weights = routing_weights,
        .gate_up_weights = gate_up_weights,
        .down_weights = down_weights,
    };
    return {std::move(attributes), std::move(tensor_args)};
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

namespace ttnn::prim {
ttnn::operations::experimental::deepseek::moe::fused_experts::FusedExpertsDeviceOperation::tensor_return_value_t
fused_experts(
    const Tensor& input_tensor,
    const Tensor& routing_weights,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    uint32_t intermediate_size,
    float swiglu_limit,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::experimental::deepseek::moe::fused_experts::FusedExpertsDeviceOperation;
    auto [operation_attributes, tensor_args] = OperationType::invoke(
        input_tensor, routing_weights, gate_up_weights, down_weights, intermediate_size, swiglu_limit, memory_config);
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
