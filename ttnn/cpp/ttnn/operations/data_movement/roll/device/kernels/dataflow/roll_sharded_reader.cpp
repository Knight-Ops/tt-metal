// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

// Native sharded roll copy kernel.
//
// Roll is realised as a gather: the host computes, per destination core, a list of
// segment-copy descriptors. Each descriptor copies a contiguous run of `copy_size` bytes
// from a source shard (possibly on another core) into the local output shard.
//
// Segment offsets/sizes are arbitrary (a within-row rotation produces unaligned runs), so
// every segment is staged through a scratch CB: we NOC-read an alignment-padded superset of
// the run into the scratch buffer, then copy the exact bytes out locally. This sidesteps the
// NOC L1 alignment requirement for both same-core and cross-core transfers.
//
// Source addresses are given as (src_cb_id, offset) so program-cache reuse stays correct
// after the backing buffer moves.

void kernel_main() {
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t scratch_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t alignment = get_compile_time_arg_val(2);

    CircularBuffer output_cb(output_cb_id);
    CircularBuffer scratch_cb(scratch_cb_id);
    const uint32_t dst_base = output_cb.get_write_ptr();
    const uint32_t scratch_base = scratch_cb.get_write_ptr();

    uint32_t arg_idx = 0;
    const uint32_t num_transfers = get_arg_val<uint32_t>(arg_idx++);

    for (uint32_t t = 0; t < num_transfers; t++) {
        const uint32_t src_noc_x = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t src_noc_y = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t src_cb_id = get_arg_val<uint32_t>(arg_idx++);
        uint32_t src_l1_offset = get_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_offset = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t copy_size = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t src_stride = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t dst_stride = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t num_rows = get_arg_val<uint32_t>(arg_idx++);

        CircularBuffer src_cb(src_cb_id);
        const uint32_t src_cb_base = src_cb.get_read_ptr();

        // One copy per row; identical column run, advanced by src/dst strides.
        for (uint32_t row = 0; row < num_rows; row++) {
            // Align the source offset down; read a padded superset into scratch.
            const uint32_t pre_pad = src_l1_offset & (alignment - 1);
            const uint32_t aligned_src_offset = src_l1_offset - pre_pad;
            uint32_t read_size = pre_pad + copy_size;
            read_size = (read_size + alignment - 1) & ~(alignment - 1);

            const uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_cb_base + aligned_src_offset);
            noc_async_read(src_noc_addr, scratch_base, read_size);
            noc_async_read_barrier();

            // Local copy of the exact run into the output shard. Prefer 32-bit word copies
            // (4x fewer L1 accesses); fall back to bytes when either address is not word
            // aligned (e.g. an odd bf16 column offset), keeping the copy alignment-agnostic.
            const uint32_t src_addr = scratch_base + pre_pad;
            const uint32_t dst_addr = dst_base + dst_offset;
            if (((src_addr | dst_addr) & 3u) == 0) {
                const uint32_t words = copy_size >> 2;
                volatile tt_l1_ptr uint32_t* s32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
                volatile tt_l1_ptr uint32_t* d32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
                for (uint32_t w = 0; w < words; w++) {
                    d32[w] = s32[w];
                }
                volatile tt_l1_ptr uint8_t* s8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src_addr);
                volatile tt_l1_ptr uint8_t* d8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst_addr);
                for (uint32_t b = words << 2; b < copy_size; b++) {
                    d8[b] = s8[b];
                }
            } else {
                volatile tt_l1_ptr uint8_t* s8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src_addr);
                volatile tt_l1_ptr uint8_t* d8 = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst_addr);
                for (uint32_t b = 0; b < copy_size; b++) {
                    d8[b] = s8[b];
                }
            }
            src_l1_offset += src_stride;
            dst_offset += dst_stride;
        }
    }
}
