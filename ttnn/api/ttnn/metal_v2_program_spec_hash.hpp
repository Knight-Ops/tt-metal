// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Content hash of a Metal 2.0 ProgramSpec, for use as a program-cache key.
//
// The ProgramSpec fully defines a compiled Program, so hashing it keys the cache
// at program identity: two specs collide iff they describe the same Program. This
// is the correct-by-construction replacement for hand-maintained per-op
// compute_program_hash, which has to be kept in sync with the program by hand.
//
// ProgramSpec is reflection-hashable via ttsl::hash out of the box, so almost all
// of it is hashed generically. The single semantic generic reflection can't express
// is shape *relaxation*: a TensorParameter may declare dynamic_tensor_shape /
// match_padded_shape_only, meaning the Program is invariant to (part of) the tensor
// shape. We honor that here so volume-equivalent shapes (e.g. [2,3] and [3,2], both
// one tile) share a cache entry instead of fragmenting it. Every other field is
// folded in via reflection, so adding a field to KernelSpec/DataflowBufferSpec/etc.
// is picked up automatically; a static_assert below guards the one place that isn't
// (the top-level ProgramSpec field set), so a new ProgramSpec member can't silently
// slip out of the key.

#include <cstddef>
#include <cstdint>

#include <reflect>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt_stl/reflection.hpp>

namespace ttnn::device_operation {

namespace detail {

// TensorParameter, with shape folded in per its declared relaxations. (Everything
// else about the parameter is identity-relevant and always hashed.)
inline std::size_t hash_tensor_parameter(const tt::tt_metal::experimental::TensorParameter& p) {
    const auto& spec = p.spec;
    // Data type, layout and memory config (which carries the sharding configuration) are always
    // identity-relevant, so they always go into the hash. Only the *shape* is folded in conditionally,
    // per the parameter's declared relaxations below.
    std::size_t hash = ttsl::hash::hash_objects_with_default_seed(
        p.unique_id.get(), spec.data_type(), spec.layout(), spec.memory_config());

    if (p.advanced_options.dynamic_tensor_shape) {
        // Program depends only on element/page count, not on tensor shape.
        hash = ttsl::hash::hash_objects(hash, static_cast<uint64_t>(spec.padded_shape().volume()));
    } else if (p.advanced_options.match_padded_shape_only) {
        // Logical shape may vary; the padded shape (and thus access pattern) is fixed.
        hash = ttsl::hash::hash_objects(hash, spec.padded_shape());
    } else {
        hash = ttsl::hash::hash_objects(hash, spec.logical_shape(), spec.padded_shape());
    }
    return hash;
}

}  // namespace detail

// Content hash of `spec`, suitable as a program-cache key. Combine with the op's
// type hash at the call site so distinct ops with coincidentally-identical specs
// don't collide.
inline std::size_t program_spec_cache_key(const tt::tt_metal::experimental::ProgramSpec& spec) {
    // ProgramSpec currently has 7 fields; tensor_parameters is the only one needing
    // relaxation-aware handling, so it is hashed separately below and the other six
    // are hashed generically. If a field is added/removed, update this function (and
    // this count) so the new field participates in the key.
    static_assert(reflect::size<tt::tt_metal::experimental::ProgramSpec>() == 7);

    std::size_t hash = ttsl::hash::hash_objects_with_default_seed(
        spec.name,
        spec.kernels,
        spec.dataflow_buffers,
        spec.cross_node_dataflow_buffers,
        spec.semaphores,
        spec.work_units);

    for (const auto& p : spec.tensor_parameters) {
        hash = ttsl::hash::hash_objects(hash, detail::hash_tensor_parameter(p));
    }
    return hash;
}

}  // namespace ttnn::device_operation
