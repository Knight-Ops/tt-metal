// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"

// Input-broadcaster kernel (runs on the second core, {1,0}).
//
// Uses the *other* NoC (NoC 1) so it runs concurrently with the expert-id sender
// on {0,0} (NoC 0).
//
// The activations are broadcast verbatim, page-by-page, so the kernel is layout
// agnostic (works for ROW_MAJOR and TILE inputs alike): it copies every page of
// input_tensor into a contiguous L1 buffer and multicasts the whole buffer.
//
//   1. Reads all input_num_pages pages of input_tensor into cb_input.
//   2. Multicasts the buffer to every other core's L1 (same cb_input address).
//   3. Sets + multicasts the input-ready semaphore to signal the other cores.
//
// NoC 1 multicasts traverse from high to low coordinates, so the host passes the
// multicast rectangle with start = bottom-right corner, end = top-left corner.
//
// Compile-time args:
//   0: cb_input         (L1 buffer holding the activations; broadcast to all cores)
//   1: input_page_size  (bytes per page of input_tensor)
//   2: input_num_pages  (number of pages of input_tensor)
//   3: sem_input_id     (input-ready semaphore)
//   4+: TensorAccessorArgs(input_tensor)
//
// Runtime args:
//   0: input_tensor base address
//   1: mcast_start_x   2: mcast_start_y
//   3: mcast_end_x     4: mcast_end_y
//   5: num_dests       (number of receiver cores = total cores - 1)
void kernel_main() {
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t input_num_pages = get_compile_time_arg_val(2);
    constexpr uint32_t sem_input_id = get_compile_time_arg_val(3);

    constexpr auto input_args = TensorAccessorArgs<4>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t mcast_start_x = get_arg_val<uint32_t>(1);
    const uint32_t mcast_start_y = get_arg_val<uint32_t>(2);
    const uint32_t mcast_end_x = get_arg_val<uint32_t>(3);
    const uint32_t mcast_end_y = get_arg_val<uint32_t>(4);
    const uint32_t num_dests = get_arg_val<uint32_t>(5);

    // Use NoC 1 ("the other NoC") so this runs in parallel with the {0,0} sender on NoC 0.
    Noc noc(1);
    const auto input = TensorAccessor(input_args, input_addr);

    CircularBuffer cb_input(cb_input_id);

    // ---- 1. Read all activation pages into cb_input. ----
    cb_input.reserve_back(1);
    const uint32_t input_l1 = cb_input.get_write_ptr();
    for (uint32_t p = 0; p < input_num_pages; ++p) {
        noc.async_read(input, cb_input, input_page_size, {.page_id = p}, {.offset_bytes = p * input_page_size});
    }
    noc.async_read_barrier();

    // ---- 2. Broadcast the activations to all other cores' L1 (same cb_input address). ----
    const uint32_t total_bytes = input_page_size * input_num_pages;
    noc.async_write_multicast(
        CoreLocalMem<uint32_t>(input_l1),
        MulticastEndpoint{},
        total_bytes,
        num_dests,
        {.offset_bytes = 0},
        {.noc_x_start = mcast_start_x,
         .noc_y_start = mcast_start_y,
         .noc_x_end = mcast_end_x,
         .noc_y_end = mcast_end_y,
         .addr = input_l1},
        /*linked=*/false);
    noc.async_write_barrier();

    // ---- 3. Signal the other cores via the input-ready semaphore. ----
    Semaphore<> sem(sem_input_id);
    sem.set(1);
    sem.set_multicast(noc, mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, num_dests, /*linked=*/false);
}
