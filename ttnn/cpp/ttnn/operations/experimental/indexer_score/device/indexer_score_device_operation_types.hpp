// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::indexer_score {

// Work-unit knobs (elements, tile-aligned; SDPAProgramConfig analogue).
// One unit = q_chunk rows x k_chunk keys; heads stream in head_group blocks.
struct IndexerScoreProgramConfig {
    std::size_t q_chunk_size = 32;
    std::size_t k_chunk_size = 32;
    std::size_t head_group_size = 1;  // heads resident at once; 1 always fits L1, raise for perf (0 = all)
};

// Resolve head_group_size to a concrete head count (0 = all Hi). Single-sourced so validate and the
// factory can't drift on the "0 = all" contract.
inline uint32_t resolve_head_group(const IndexerScoreProgramConfig& cfg, uint32_t Hi) {
    return cfg.head_group_size == 0 ? Hi : static_cast<uint32_t>(cfg.head_group_size);
}

struct operation_attributes_t {
    uint32_t chunk_start_idx{0};
    // ReLU on each per-head q.kT before the gate-multiply. true = DeepSeek/GLM lightning indexer
    // (relu(q.k)*w); false = raw dot product q.k*w (e.g. MiniMax M3 MSA, which has no ReLU). A
    // compile-time arg in the compute kernel, so the apply_relu==true path is byte-identical to before.
    bool apply_relu{true};
    // Number of output groups. 1 = sum ALL Hi heads into one plane -> score [B,1,Sq,T] (DeepSeek/GLM).
    // G>1 = partition the Hi heads into G contiguous groups of Hi/G, sum WITHIN each group only ->
    // score [B,G,Sq,T] (MiniMax M3 MSA per-GQA-group selection, multiple groups resident on one chip).
    // Compile-time, with G==1 byte-identical to before. G>1 requires all heads resident (head_group_size
    // 0 or Hi) and the full-strip path (k_chunk_size>=64).
    uint32_t num_groups{1};
    IndexerScoreProgramConfig program_config{};
    // Resolved (not optional) so it is part of the reflected program-cache key; the public callable
    // fills it from the user's optional config, defaulting math_fidelity to the dtype-derived choice.
    DeviceComputeKernelConfig compute_kernel_config{};
};

struct tensor_args_t {
    const Tensor& q;
    const Tensor& k;
    const Tensor& weights;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::indexer_score
