// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

// Receiver kernel (runs on every compute core except the two senders {0,0} and {1,0}).
//
// Waits for both broadcasts to land in this core's L1:
//   - {0,0} multicasts the expert ids into cb_bcast and bumps sem_id.
//   - {1,0} multicasts the activation row into cb_input and bumps sem_input_id.
// Once both semaphores are observed the data is resident in L1 for the future
// fused-FFN compute milestone, and the kernel exits.
//
// Compile-time args:
//   0: sem_id        (expert-ids-ready semaphore)
//   1: sem_input_id  (input-ready semaphore)
void kernel_main() {
    constexpr uint32_t sem_id = get_compile_time_arg_val(0);
    constexpr uint32_t sem_input_id = get_compile_time_arg_val(1);

    Semaphore<>(sem_id).wait(1);
    Semaphore<>(sem_input_id).wait(1);
    // Both semaphores observed: expert ids (cb_bcast) and activations (cb_input) are in L1. Exit.
}
