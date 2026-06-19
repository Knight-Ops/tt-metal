// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"

// Per-core gate_up weight fetch (shared by all kernels in this op).
//
// Each core owns a 2-tile-wide (64 column) slice of the first matmul's N == 2I
// dimension: the tiles [col_start_tile, col_start_tile + 1]. For a gate_up weight
// of shape [K, N] (K == H rows, N == 2I cols) laid out in TILE layout, this core
// needs the [K, 64] column slice, i.e. k_tiles * 2 tiles.
//
// The gate_up weights are stored DRAM ND-sharded so that each shard is exactly one
// core's [K, 64] column slice (shard shape [K, 64] in elements -> [k_tiles, 2] in
// tiles). The shards are round-robin distributed across the DRAM banks, and the
// pages of a shard are contiguous within its bank, so this core's entire weight
// slice for one expert can be pulled in a *single* NoC read of the whole shard.
// Shard id == this core's compute index == col_start_tile / 2.
//
// The expert ids (compacted at the front of cb_bcast, sentinel-padded) tell the
// core which experts are active; the slice is fetched once per active expert.
//
// Arguments:
//   noc               NoC instance to use for the reads.
//   cb_bcast_id       CB holding the UINT32 expert ids.
//   cb_weights_id     CB receiving this core's weight slice.
//   num_experts       E (also the sentinel value for "no expert").
//   k_tiles           K / 32 (number of tile rows of the weight).
//   n_tiles           N / 32 (number of tile cols of the weight).
//   tile_bytes        Size of one tile in bytes.
//   col_start_tile    This core's first output N-tile (= compute_index * 2).
//   gate_up_args      Shared TensorAccessorArgs (all experts share one layout).
//   rt_w_addr_base    Runtime-arg index of the first gate_up base address.
template <typename GateUpArgs>
void fetch_gate_up_slices(
    const Noc& noc,
    uint32_t cb_bcast_id,
    uint32_t cb_weights_id,
    uint32_t num_experts,
    uint32_t k_tiles,
    uint32_t n_tiles,
    uint32_t tile_bytes,
    uint32_t col_start_tile,
    const GateUpArgs& gate_up_args,
    uint32_t rt_w_addr_base) {
    // Cores whose column slice falls outside the weight do nothing.
    if (col_start_tile >= n_tiles) {
        return;
    }

    constexpr uint32_t kTilesPerCore = 2;
    const uint32_t slice_tiles = k_tiles * kTilesPerCore;
    // One DRAM shard == this core's whole [K, 64] slice; read it in one shot.
    const uint32_t slice_bytes = slice_tiles * tile_bytes;
    const uint32_t shard_id = col_start_tile / kTilesPerCore;

    CircularBuffer cb_bcast(cb_bcast_id);
    CircularBuffer cb_weights(cb_weights_id);
    CoreLocalMem<volatile uint32_t> ids(cb_bcast.get_write_ptr());

    // Reserve once and reuse: there is no consumer yet, so each expert overwrites
    // the slice. A later compute milestone will push/pop per expert instead.
    cb_weights.reserve_back(slice_tiles);

    for (uint32_t i = 0; i < num_experts; ++i) {
        const uint32_t e = ids[i];
        if (e >= num_experts) {
            break;  // sentinel: active ids are compacted at the front.
        }

        const uint32_t w_addr = get_arg_val<uint32_t>(rt_w_addr_base + e);
        const auto w = TensorAccessor(gate_up_args, w_addr);

        // Single NoC read of this expert's entire shard for this core.
        ShardView w_shard(w);
        noc.async_read(w_shard, cb_weights, slice_bytes, {.shard_id = shard_id}, {.offset_bytes = 0});
        noc.async_read_barrier();
    }
}
