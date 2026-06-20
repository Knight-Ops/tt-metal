// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for unified_routed_expert_ffn.
//
// Primary responsibility: pop `cb_out` (the down matmul's per-core final
// block, packed one subblock at a time) and write tiles to the DRAM-
// interleaved output tensor at this core's (mt, nt_d) tile region, looped
// over `effective_chunks` chunks.
//
// Two-RISC weight read (writer_handles_up == 1): the gate/up phase is bound by
// the gate+up weight DRAM read, which the reader (BRISC, NoC 0) would otherwise
// issue serially. To split it, the writer (NCRISC, NoC 1) owns the `up` weight
// entirely — it reads up from DRAM and, at GRID_Y > 1, multicasts it down its
// N-column — running concurrently with the reader's `gate` read on the other
// NoC. The reader skips all `up` handling in this mode (see reader_handles_up).
// Both feed the fused gate+up compute phase. Enabled for all layouts: per chunk
// the writer produces all `up` K-blocks, then drains that chunk's `cb_out`
// (the drain is small vs the read saved, so even multi-chunk long sequences
// win despite the writer serialising up-read and cb_out within a chunk).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_mt = get_arg_val<uint32_t>(1);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(2);
    // Up-weight (two-RISC) runtime args. Only meaningful when writer_handles_up.
    const uint32_t up_addr = get_arg_val<uint32_t>(3);
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(4);
    const bool is_up_sender = get_arg_val<uint32_t>(5) != 0;
    const uint32_t up_ready_sem_id = get_arg_val<uint32_t>(6);
    const uint32_t up_valid_sem_id = get_arg_val<uint32_t>(7);
    const uint32_t up_num_receivers = get_arg_val<uint32_t>(8);
    const uint32_t up_mcast_nx_start = get_arg_val<uint32_t>(9);
    const uint32_t up_mcast_ny_start = get_arg_val<uint32_t>(10);
    const uint32_t up_mcast_nx_end = get_arg_val<uint32_t>(11);
    const uint32_t up_mcast_ny_end = get_arg_val<uint32_t>(12);
    const uint32_t up_sender_nx = get_arg_val<uint32_t>(13);
    const uint32_t up_sender_ny = get_arg_val<uint32_t>(14);

    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_N_gu = get_compile_time_arg_val(3);
    constexpr uint32_t per_core_N_d = get_compile_time_arg_val(4);
    constexpr uint32_t d_out_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t d_out_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t N_gate_tiles_full = get_compile_time_arg_val(9);
    constexpr uint32_t N_down_tiles_full = get_compile_time_arg_val(10);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t chunk_M_tiles = get_compile_time_arg_val(12);
    // device-side count read.
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(13);
    constexpr uint32_t cb_idx_scratch = get_compile_time_arg_val(14);
    constexpr uint32_t local_expert_id = get_compile_time_arg_val(15);
    // M_tiles_full: total tile-row count of the output tensor. When the
    // kernel runs more chunks than strictly needed (because
    // M_tiles_full % chunk_M_tiles != 0), the last chunk has writer
    // destinations past M_tiles_full — we skip those writes here so we
    // don't OOB-write the output buffer.
    constexpr uint32_t M_tiles_full = get_compile_time_arg_val(16);
    // Two-RISC up-weight read (short-seq only).
    constexpr uint32_t writer_handles_up = get_compile_time_arg_val(17);
    constexpr uint32_t cb_in1_up = get_compile_time_arg_val(18);
    constexpr uint32_t in0_block_w_gu = get_compile_time_arg_val(19);
    constexpr uint32_t K_gate_tiles = get_compile_time_arg_val(20);

    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_in1_num_subblocks_M = per_core_M / d_out_subblock_h;
    constexpr uint32_t d_in1_num_subblocks_N = per_core_N_d / d_out_subblock_w;
    constexpr uint32_t num_blocks_gu = K_gate_tiles / in0_block_w_gu;
    constexpr uint32_t g_in1_block_num_tiles = per_core_N_gu * in0_block_w_gu;

    constexpr uint32_t out_accessor_offset = 21;
    constexpr auto out_args = TensorAccessorArgs<out_accessor_offset>();
    const auto out_acc = TensorAccessor(out_args, output_addr, get_tile_size(cb_out));

    // up accessor follows the out accessor in the compile-arg stream (always
    // appended host-side; constructed here unconditionally, used only when
    // writer_handles_up).
    constexpr uint32_t up_accessor_offset = out_args.next_compile_time_args_offset();
    constexpr auto up_args = TensorAccessorArgs<up_accessor_offset>();
    const auto up_acc = TensorAccessor(up_args, up_addr, get_tile_size(cb_in1_up));

    const uint32_t out_tile_bytes = get_tile_size(cb_out);

    // Wait for the reader's counts/idx push and compute effective_chunks =
    // ceil(count / chunk_M_tiles). The writer drains cb_out per chunk;
    // bounding the loop here is required because the reader and compute
    // bound theirs too — without this, the writer would wait forever on
    // cb_out for chunks the compute never pushes.
    cb_wait_front(cb_counts_scratch, 1);
    cb_wait_front(cb_idx_scratch, 1);
    const volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_counts_scratch));
    const uint32_t idx_l1 = get_read_ptr(cb_idx_scratch);
    const volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(idx_l1);
    const uint32_t global_expert_id = idx_ptr[local_expert_id];
    const uint32_t count_value = counts_ptr[global_expert_id];
    const uint32_t count_tiles = (count_value + 31) / 32;
    const uint32_t effective_chunks_runtime = (count_tiles + chunk_M_tiles - 1) / chunk_M_tiles;
    const uint32_t effective_chunks = effective_chunks_runtime < num_chunks ? effective_chunks_runtime : num_chunks;

    // ---- Two-RISC up-weight read setup (short-seq only) ----
    // The writer reads `up` on NoC 1 (kUpNoc, its dedicated NoC) concurrent
    // with the reader's NoC-0 `gate` read. At GRID_Y > 1 it also multicasts the
    // block down its N-column (sender gy=0 -> receivers gy=1..GRID_Y-1) using
    // its own ready/valid sem pair, so each block is read once and shared.
    //
    // NoC-1 multicast rectangle corners must be SWAPPED relative to NoC 0: NoC 0
    // expects (min_x,min_y .. max_x,max_y), NoC 1 traverses the opposite way and
    // expects (max .. min). The host passes NoC-0-ordered corners
    // (up_mcast_*_start = first/min receiver, *_end = last/max), so here we pass
    // (end, start) into get_noc_multicast_addr with noc=1 (which also applies
    // the per-NoC coordinate flip). get_noc_addr for the unicast ready-inc is a
    // single point — no swap, just the noc=1 flip.
    constexpr uint8_t kUpNoc = 1;
    const uint32_t up_tile_bytes = get_tile_size(cb_in1_up);
    const uint32_t up_ready_sem_addr = get_semaphore(up_ready_sem_id);
    const uint32_t up_valid_sem_addr = get_semaphore(up_valid_sem_id);
    volatile tt_l1_ptr uint32_t* up_ready_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(up_ready_sem_addr);
    volatile tt_l1_ptr uint32_t* up_valid_local = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(up_valid_sem_addr);
    const uint64_t up_sender_ready_noc = get_noc_addr(up_sender_nx, up_sender_ny, up_ready_sem_addr, kUpNoc);
    const uint64_t up_mcast_valid_noc = get_noc_multicast_addr(
        up_mcast_nx_end, up_mcast_ny_end, up_mcast_nx_start, up_mcast_ny_start, up_valid_sem_addr, kUpNoc);
    constexpr uint32_t UP_VALID = 1;

    for (uint32_t chunk = 0; chunk < effective_chunks; ++chunk) {
        // ---- Phase 1/2 weight feed: produce cb_in1_up (writer owns `up`) ----
        // Mirrors the reader's `gate` in1 multicast but on the writer's RISC /
        // NoC, so gate and up stream from DRAM concurrently. Runs before the
        // cb_out drain; short-seq is single-chunk so there is no overlap with a
        // later chunk's down output.
        if constexpr (writer_handles_up) {
            // Sender (gy=0) reads its `up` N-column slice on NoC 1 and, when
            // there are column receivers, multicasts it down the column;
            // receivers (gy>0) ack the sender then wait for the block. Mirrors
            // the reader's `gate` in1 multicast but on the writer's RISC / NoC.
            for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
                cb_reserve_back(cb_in1_up, g_in1_block_num_tiles);

                if (!is_up_sender) {
                    *up_valid_local = 0;
                    noc_semaphore_inc(up_sender_ready_noc, 1, kUpNoc);
                }

                if (is_up_sender) {
                    if (up_num_receivers > 0) {
                        noc_semaphore_wait(up_ready_local, up_num_receivers);
                        *up_ready_local = 0;
                    }
                    uint32_t l1_w_up = get_write_ptr(cb_in1_up);
                    const uint32_t up_block_start = l1_w_up;
                    for (uint32_t k = 0; k < in0_block_w_gu; ++k) {
                        for (uint32_t n = 0; n < per_core_N_gu; ++n) {
                            const uint32_t row = kb * in0_block_w_gu + k;
                            const uint32_t col = my_nt_gu * per_core_N_gu + n;
                            if (col < N_gate_tiles_full) {
                                const uint32_t tile_idx = row * N_gate_tiles_full + col;
                                noc_async_read_page(tile_idx, up_acc, l1_w_up, /*offset=*/0, kUpNoc);
                            } else {
                                volatile tt_l1_ptr uint64_t* p =
                                    reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_w_up);
                                for (uint32_t i = 0; i < up_tile_bytes / 8; ++i) {
                                    p[i] = 0;
                                }
                            }
                            l1_w_up += up_tile_bytes;
                        }
                    }
                    noc_async_read_barrier(kUpNoc);

                    if (up_num_receivers > 0) {
                        // Swapped corners for NoC 1 (see setup comment above).
                        const uint64_t up_mcast_noc = get_noc_multicast_addr(
                            up_mcast_nx_end,
                            up_mcast_ny_end,
                            up_mcast_nx_start,
                            up_mcast_ny_start,
                            up_block_start,
                            kUpNoc);
                        const uint32_t up_block_bytes = g_in1_block_num_tiles * up_tile_bytes;
                        noc_async_write_multicast(
                            up_block_start, up_mcast_noc, up_block_bytes, up_num_receivers, /*linked=*/false, kUpNoc);
                    }

                    cb_push_back(cb_in1_up, g_in1_block_num_tiles);

                    if (up_num_receivers > 0) {
                        noc_async_writes_flushed(kUpNoc);
                        *up_valid_local = UP_VALID;
                        noc_semaphore_set_multicast(
                            up_valid_sem_addr, up_mcast_valid_noc, up_num_receivers, /*linked=*/false, kUpNoc);
                    }
                }

                if (!is_up_sender) {
                    noc_semaphore_wait(up_valid_local, UP_VALID);
                    cb_push_back(cb_in1_up, g_in1_block_num_tiles);
                }
            }
        }

        // ---- Drain cb_out (down matmul output) to DRAM ----
        const uint32_t row0 = chunk * chunk_M_tiles + my_mt * per_core_M;
        const uint32_t col0 = my_nt_d * per_core_N_d;
        for (uint32_t sb_m = 0; sb_m < d_in1_num_subblocks_M; ++sb_m) {
            for (uint32_t sb_n = 0; sb_n < d_in1_num_subblocks_N; ++sb_n) {
                cb_wait_front(cb_out, d_out_subblock_num_tiles);
                uint32_t l1_read = get_read_ptr(cb_out);
                for (uint32_t i = 0; i < d_out_subblock_h; ++i) {
                    for (uint32_t j = 0; j < d_out_subblock_w; ++j) {
                        const uint32_t row = row0 + sb_m * d_out_subblock_h + i;
                        const uint32_t col = col0 + sb_n * d_out_subblock_w + j;
                        // Skip OOB writes:
                        //   * col >= N_down_tiles_full: GRID_X ceil_div
                        //     produces phantom output cols past actual N.
                        //   * row >= M_tiles_full: ceil_div of M produces a
                        //     last-chunk tail past actual M.
                        //   * row >= count_tiles: the last chunk's per_core_M
                        //     rows extend past count_tiles when count_tiles
                        //     is not chunk-aligned.
                        if (col < N_down_tiles_full && row < M_tiles_full && row < count_tiles) {
                            const uint32_t tile_idx = row * N_down_tiles_full + col;
                            noc_async_write_page(tile_idx, out_acc, l1_read);
                        }
                        l1_read += out_tile_bytes;
                    }
                }
                // Wait for the writes to LEAVE this core (departed sender);
                // doesn't wait for the DRAM round-trip. Safe to reuse the L1
                // slot now — the NoC has captured the data. ~10x faster than
                // noc_async_write_barrier per subblock at small per_core_M.
                noc_async_writes_flushed();
                cb_pop_front(cb_out, d_out_subblock_num_tiles);
            }
        }
    }
    // Ensure all outstanding writes complete at the destination before the
    // kernel returns (the next dispatched op may read this output).
    noc_async_write_barrier();
    // The two-RISC `up` weight multicast issues its valid-semaphore broadcast
    // as a POSTED atomic on NoC 1 (kUpNoc). Like the reader's valid-sem mcasts
    // (which it flushes with noc_async_atomic_barrier), the last one can still
    // be in flight at kernel exit; without this barrier it leaks into the next
    // dispatched program and corrupts/hangs it (timing-dependent, surfaces deep
    // in long multi-expert/multi-layer runs). Flush the NoC-1 atomics here.
    if constexpr (writer_handles_up) {
        noc_async_atomic_barrier(kUpNoc);
    }
}
