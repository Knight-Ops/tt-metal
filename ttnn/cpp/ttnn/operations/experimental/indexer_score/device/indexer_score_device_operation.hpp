// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "indexer_score_device_operation_types.hpp"
#include "indexer_score_program_factory.hpp"

namespace ttnn::operations::experimental::indexer_score {

struct IndexerScoreDeviceOperation {
    using operation_attributes_t = indexer_score::operation_attributes_t;
    using tensor_args_t = indexer_score::tensor_args_t;
    using spec_return_value_t = indexer_score::spec_return_value_t;
    using tensor_return_value_t = indexer_score::tensor_return_value_t;
    using program_factory_t = std::variant<program::IndexerScoreProgramFactory>;

    // No custom compute_program_hash: operation_attributes_t (chunk_start_idx + every config field)
    // is a reflectable aggregate, so the default reflection hash already keys distinct programs on all
    // fields. Do not add a hand-rolled hash that drops fields.
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Matmul-FLOP performance model (mirrors SDPAOperation::create_op_performance_model): reports ideal
    // matmul cycles so tracy's ideal/actual utilization equals the math-util test's mm_flops/(cores x
    // cycles x peak). FLOPs count causal-valid tiles only; cores/fidelity match the factory.
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& q,
        const Tensor& k,
        const Tensor& weights,
        uint32_t chunk_start_idx,
        bool apply_relu,
        uint32_t num_groups,
        uint32_t block_size,
        const IndexerScoreProgramConfig& program_config,
        const DeviceComputeKernelConfig& compute_kernel_config);
};

}  // namespace ttnn::operations::experimental::indexer_score

namespace ttnn::experimental {

// DeepSeek-V3.2 DSA / MiniMax-M3 MSA lightning-indexer scorer (the public callable,
// ttnn.experimental.indexer_score):
//   score[b, s, t] = sum_h act(q[b,h,s,:] . k[b,t,:]) * weights[b,h,s]
//     act = relu (apply_relu=true, DeepSeek/GLM) or identity (apply_relu=false, MiniMax M3).
// q [B, Hi, Sq, D], k [B, 1, T, D] -> score [B, num_groups, Sq, T] (row-major bf16).
// weights [B, Hi, Sq, 1] are DeepSeek/GLM's learned per-head gates (scale pre-folded). MiniMax M3 has
// NO gates, only a 1/sqrt(d) scale: pass weights=nullopt and scale=1/sqrt(d), and the op runs with a
// constant gate (= scale) so no dummy gate tensor is needed. `scale` is used only when weights==nullopt.
// num_groups: 1 sums all Hi heads into one plane (DeepSeek/GLM). G>1 partitions the heads into G groups
// of Hi/G and sums within each group -> G output planes (MiniMax M3 per-GQA-group selection, multiple
// groups on one chip); G>1 needs all heads resident and k_chunk_size>=64.
// block_size: 0 = no pooling -> score [B,G,Sq,T]. >0 = block-max-pool over block_size keys ->
// score [B,G,Sq,T/block_size] (MiniMax M3 block selection; downstream topk picks per-group top-k blocks).
// Causality from chunk_start_idx: key t visible to query s iff t <= chunk_start_idx + s.
ttnn::Tensor indexer_score(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const std::optional<ttnn::Tensor>& weights = std::nullopt,
    uint32_t chunk_start_idx = 0,
    bool apply_relu = true,
    float scale = 1.0f,
    uint32_t num_groups = 1,
    uint32_t block_size = 0,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config = {},
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental
