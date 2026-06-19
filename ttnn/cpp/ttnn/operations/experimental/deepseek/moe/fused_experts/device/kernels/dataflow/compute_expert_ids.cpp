// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"

#include "fetch_gate_up.h"

// Sender kernel (runs on core {0,0}).
//
// 1. Reads the routing-weight row and computes the selected ("hit") expert ids on
//    device (matching the host `hit = (rw.abs().sum(0) > 0).nonzero()`).
// 2. Writes the ids to the UINT32 output tensor [1, 1, 1, E].
// 3. Multicasts the ids buffer to all other compute cores' L1 (same CB address).
// 4. Sets + multicasts a semaphore to signal the other cores that the data is ready.
// 5. Fetches this core's gate_up weight slice for each active expert (it already
//    holds the expert ids locally).
//
// Compile-time args:
//   0: num_experts (E)
//   1: sentinel value for unused output slots (= E)
//   2: cb_routing  (L1 scratch for the routing-weight row)
//   3: cb_bcast    (L1 buffer holding the expert ids; broadcast to all cores)
//   4: routing_page_bytes (E * sizeof(bfloat16))
//   5: bcast_page_bytes    (E * sizeof(uint32))
//   6: sem_id      (broadcast-ready semaphore)
//   7: cb_weights  (L1 buffer for this core's gate_up slice)
//   8: k_tiles     (H / 32)
//   9: n_tiles     (2I / 32)
//   10: tile_bytes
//   11+: TensorAccessorArgs(routing_weights), TensorAccessorArgs(output), TensorAccessorArgs(gate_up)
//
// Runtime args:
//   0: routing_weights base address
//   1: output base address
//   2: mcast_start_x   3: mcast_start_y
//   4: mcast_end_x     5: mcast_end_y
//   6: num_dests       (number of receiver cores = total cores - 1)
//   7: col_start_tile  (this core's first output N-tile)
//   8+: gate_up base addresses (one per expert)
void kernel_main() {
    constexpr uint32_t num_experts = get_compile_time_arg_val(0);
    constexpr uint32_t sentinel = get_compile_time_arg_val(1);
    constexpr uint32_t cb_routing_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_bcast_id = get_compile_time_arg_val(3);
    constexpr uint32_t routing_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t bcast_page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(7);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(10);

    constexpr auto routing_args = TensorAccessorArgs<11>();
    constexpr auto out_args = TensorAccessorArgs<routing_args.next_compile_time_args_offset()>();
    constexpr auto gate_up_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t routing_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t mcast_start_x = get_arg_val<uint32_t>(2);
    const uint32_t mcast_start_y = get_arg_val<uint32_t>(3);
    const uint32_t mcast_end_x = get_arg_val<uint32_t>(4);
    const uint32_t mcast_end_y = get_arg_val<uint32_t>(5);
    const uint32_t num_dests = get_arg_val<uint32_t>(6);
    const uint32_t col_start_tile = get_arg_val<uint32_t>(7);
    constexpr uint32_t kWeightAddrBase = 8;

    // Pin the expert-id sender to NoC 0; the input broadcaster on the second core uses NoC 1.
    Noc noc(0);
    const auto routing = TensorAccessor(routing_args, routing_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    CircularBuffer cb_routing(cb_routing_id);
    CircularBuffer cb_bcast(cb_bcast_id);

    // ---- 1. Read routing row + compute expert ids into cb_bcast. ----
    cb_routing.reserve_back(1);
    noc.async_read(routing, cb_routing, routing_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    cb_bcast.reserve_back(1);
    const uint32_t bcast_l1 = cb_bcast.get_write_ptr();

    CoreLocalMem<volatile uint16_t> rw(cb_routing.get_write_ptr());
    CoreLocalMem<volatile uint32_t> ids(bcast_l1);

    uint32_t n = 0;
    for (uint32_t e = 0; e < num_experts; ++e) {
        if ((rw[e] & 0x7FFF) != 0) {
            ids[n++] = e;
        }
    }
    for (uint32_t i = n; i < num_experts; ++i) {
        ids[i] = sentinel;
    }

    // ---- 2. Write ids to the output tensor. ----
    noc.async_write(cb_bcast, out, bcast_page_bytes, {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();

    // ---- 3. Broadcast the ids to all other cores' L1 (same cb_bcast address). ----
    noc.async_write_multicast(
        CoreLocalMem<uint32_t>(bcast_l1),
        MulticastEndpoint{},
        bcast_page_bytes,
        num_dests,
        {.offset_bytes = 0},
        {.noc_x_start = mcast_start_x,
         .noc_y_start = mcast_start_y,
         .noc_x_end = mcast_end_x,
         .noc_y_end = mcast_end_y,
         .addr = bcast_l1},
        /*linked=*/false);
    noc.async_write_barrier();

    // ---- 4. Signal the other cores via the broadcast-ready semaphore. ----
    Semaphore<> sem(sem_id);
    sem.set(1);
    sem.set_multicast(noc, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, num_dests, /*linked=*/false);

    // ---- 5. Fetch this core's gate_up weight slice for each active expert. ----
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
