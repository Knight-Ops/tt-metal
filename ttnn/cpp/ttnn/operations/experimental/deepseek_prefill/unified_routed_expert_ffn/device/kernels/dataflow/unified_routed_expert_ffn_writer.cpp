// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for unified_routed_expert_ffn.
//
// Primary responsibility: pop `cb_out` (the down matmul's per-core final block)
// and write tiles to the DRAM output tensor at this core's (mt, nt_d) region,
// looped over `effective_chunks` chunks.
//
// Two-RISC weight read: the writer (NCRISC) reads `up` from DRAM on NoC 1
// concurrent with the reader's NoC-0 `gate` read. The program factory selects
// UP_SPLIT (writer_split_up) for ALL layouts: the writer reads `up` into the
// gy=0 sender's cb_in1_up slot and the reader multicasts it on NoC 0, ordered by
// a local up_go/up_done handshake. Only a NoC-1 DRAM read happens here — no
// worker multicast and no NoC-1 atomics — so it is safe beside the fabric CCL
// ops. The legacy writer-side NoC-1 multicast mode (UP_WRITER_MCAST /
// writer_mcasts_up) is retired and never selected.
// Per chunk the writer produces all `up` K-blocks, then drains `cb_out`.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t my_mt = get_arg_val<uint32_t>(1);
    const uint32_t my_nt_d = get_arg_val<uint32_t>(2);
    // UP_SPLIT up-weight read args: up tensor base, this core's N-column, and
    // whether this core is the gy=0 sender (only senders read `up`).
    const uint32_t up_addr = get_arg_val<uint32_t>(3);
    const uint32_t my_nt_gu = get_arg_val<uint32_t>(4);
    const bool is_up_sender = get_arg_val<uint32_t>(5) != 0;
    // UP_SPLIT local handshake sems (see reader): up_go = slot reserved,
    // up_done = up landed.
    const uint32_t up_go_sem_id = get_arg_val<uint32_t>(6);
    const uint32_t up_done_sem_id = get_arg_val<uint32_t>(7);

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
    // UP_SPLIT up-weight read (see header): the writer reads `up` on NoC 1 into
    // the gy=0 sender's cb_in1_up slot; the reader mcasts it on NoC 0.
    // writer_split_up gates it (1 = UP_SPLIT, 0 = LEGACY: reader owns `up`).
    constexpr uint32_t cb_in1_up = get_compile_time_arg_val(17);
    constexpr uint32_t in0_block_w_gu = get_compile_time_arg_val(18);
    constexpr uint32_t K_gate_tiles = get_compile_time_arg_val(19);
    constexpr uint32_t writer_split_up = get_compile_time_arg_val(20);

    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_in1_num_subblocks_M = per_core_M / d_out_subblock_h;
    constexpr uint32_t d_in1_num_subblocks_N = per_core_N_d / d_out_subblock_w;
    constexpr uint32_t num_blocks_gu = K_gate_tiles / in0_block_w_gu;
    constexpr uint32_t g_in1_block_num_tiles = per_core_N_gu * in0_block_w_gu;

    constexpr uint32_t out_accessor_offset = 21;
    constexpr auto out_args = TensorAccessorArgs<out_accessor_offset>();
    const auto out_acc = TensorAccessor(out_args, output_addr, get_tile_size(cb_out));

    // up accessor follows out in the compile-arg stream; used only when the
    // writer handles `up`.
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

    // ---- UP_SPLIT up-weight read setup ----
    // The writer reads `up` from DRAM on NoC 1 (kUpNoc) concurrent with the
    // reader's NoC-0 `gate` read, into the gy=0 sender's cb_in1_up slot; the
    // reader multicasts it on NoC 0. A local same-core (BRISC reader <-> NCRISC
    // writer) handshake orders the two: up_go (reader: slot reserved) and
    // up_done (writer: up landed in L1), monotonic counters.
    constexpr uint8_t kUpNoc = 1;
    const uint32_t up_tile_bytes = get_tile_size(cb_in1_up);
    volatile tt_l1_ptr uint32_t* up_go_local =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(up_go_sem_id));
    volatile tt_l1_ptr uint32_t* up_done_local =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(up_done_sem_id));
    uint32_t up_seq = 0;

    for (uint32_t chunk = 0; chunk < effective_chunks; ++chunk) {
        // ---- Phase 1/2 weight feed: writer reads `up` on NoC 1 (UP_SPLIT) ----
        // Streams `up` from DRAM concurrent with the reader's NoC-0 `gate` read.
        // Runs before the cb_out drain.
        if constexpr (writer_split_up) {
            // UP_SPLIT: only gy=0 in1-sender cores read `up`. Per K-block: wait
            // for the reader to reserve the slot (up_go), read this column's
            // `up` slice on NoC 1 into it, then signal up_done so the reader
            // mcasts on NoC 0. Only a NoC-1 DRAM read here (fabric-safe); the
            // reader owns cb_in1_up reserve/push.
            if (is_up_sender) {
                // The CB write pointer is PER-RISC and the reader owns push, so
                // the writer's get_write_ptr never advances. Replicate the
                // reader's cadence: cb_in1_up is double-buffered, one push per
                // K-block, so the live slot is base + (up_seq-1)%2 * slot.
                constexpr uint32_t kUpNumSlots = 2;
                const uint32_t up_cb_base = get_write_ptr(cb_in1_up);
                const uint32_t up_slot_bytes = g_in1_block_num_tiles * up_tile_bytes;
                for (uint32_t kb = 0; kb < num_blocks_gu; ++kb) {
                    ++up_seq;
                    noc_semaphore_wait_min(up_go_local, up_seq);
                    uint32_t l1_w_up = up_cb_base + ((up_seq - 1) % kUpNumSlots) * up_slot_bytes;
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
                    *up_done_local = up_seq;
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
    // UP_SPLIT issues only per-K-block-barriered NoC-1 `up` reads (no NoC-1
    // worker multicast and no NoC-1 atomics), so no extra NoC-1 drain is needed
    // here — which is exactly why it is safe beside the fabric CCL ops.
}
