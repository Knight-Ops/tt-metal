// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

#include "fetch_gate_up.h"

// Receiver kernel (runs on every compute core except the two senders {0,0} and {1,0}).
//
// Waits for both broadcasts to land in this core's L1:
//   - {0,0} multicasts the expert ids into cb_bcast and bumps sem_id.
//   - {1,0} multicasts the activation row into cb_input and bumps sem_input_id.
// Once both are observed, it fetches this core's gate_up weight slice for each
// active expert.
//
// Compile-time args:
//   0: sem_id        (expert-ids-ready semaphore)
//   1: sem_input_id  (input-ready semaphore)
//   2: num_experts (E)
//   3: cb_bcast      (L1 buffer holding the expert ids)
//   4: cb_weights    (L1 buffer for this core's gate_up slice)
//   5: k_tiles       (H / 32)
//   6: n_tiles       (2I / 32)
//   7: tile_bytes
//   8+: TensorAccessorArgs(gate_up)
//
// Runtime args:
//   0: col_start_tile  (this core's first output N-tile)
//   1+: gate_up base addresses (one per expert)
void kernel_main() {
    constexpr uint32_t sem_id = get_compile_time_arg_val(0);
    constexpr uint32_t sem_input_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_experts = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bcast_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(4);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);

    constexpr auto gate_up_args = TensorAccessorArgs<8>();

    const uint32_t col_start_tile = get_arg_val<uint32_t>(0);
    constexpr uint32_t kWeightAddrBase = 1;

    Semaphore<>(sem_id).wait(1);
    Semaphore<>(sem_input_id).wait(1);
    // Both semaphores observed: expert ids (cb_bcast) and activations (cb_input) are in L1.

    Noc noc;
    fetch_gate_up_slices(
        noc,
        cb_bcast_id,
        cb_weights_id,
        num_experts,
        k_tiles,
        n_tiles,
        tile_bytes,
        col_start_tile,
        gate_up_args,
        kWeightAddrBase);
}
