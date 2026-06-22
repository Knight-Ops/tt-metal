// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "roll_program_factory.hpp"

#include <algorithm>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

// Why ROW_MAJOR sharded roll needs a dedicated kernel instead of the slice + concat composite
// used for interleaved roll:
//
//   * slice cuts a tensor into two unequal halves (e.g. widths W-shift and shift). Re-sharding
//     those odd-shaped halves with the original shard spec produces invalid shards — the shard
//     dims no longer divide the sliced shape ("page size must equal buffer size").
//   * concat's sharded program factories only accept specific (axis, layout) pairs (width concat
//     on height/block-sharded, height concat on width/block-sharded, ...). The slice→concat
//     decomposition of an arbitrary roll does not fit those combinations across HEIGHT/WIDTH/
//     BLOCK sharding and every dim.
//
// So a roll on a sharded tensor cannot be expressed as composite slice + concat for the general
// case. This factory instead performs the roll as a single per-core gather directly over the
// sharded buffers: it stays in one shard spec (no intermediate slices) and is not bound by
// slice's or concat's sharded constraints.

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {

struct TransferDesc {
    CoreCoord src_physical_core;
    uint32_t src_l1_offset;
    uint32_t dst_offset;
    uint32_t copy_size;
    uint32_t src_stride;
    uint32_t dst_stride;
    uint32_t num_rows;
};

// A contiguous cell-column run: dst cells [dst_col, dst_col+len) <- src cells [src_col, ...).
// The same column structure repeats for every cell-row, so it is computed once.
struct ColPiece {
    uint32_t dst_col;
    uint32_t src_col;
    uint32_t len;
};

}  // namespace

ProgramDescriptor RollShardedProgramFactory::create_descriptor(
    const RollParams& operation_attributes, const RollInputs& tensor_args, Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    Tensor& output = tensor_return_value;

    TT_FATAL(input.is_sharded() && output.is_sharded(), "Native sharded roll requires sharded input and output");

    const auto& shape = input.padded_shape();
    const uint32_t rank = shape.rank();
    const uint32_t shift = operation_attributes.shift;
    const int32_t dim = operation_attributes.dim;
    const bool is_last_dim = (static_cast<uint32_t>(dim) == rank - 1);

    // The gather works in "cells": a cell is one element for ROW_MAJOR, one tile for TILE.
    // Tile cells are contiguous in L1 and naturally aligned, so a tile-aligned roll is just a
    // permutation/rotation of whole tiles — identical to the row-major element gather.
    const bool is_tile = input.layout() == Layout::TILE;
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t cell_h = is_tile ? tt::constants::TILE_HEIGHT : 1;
    const uint32_t cell_w = is_tile ? tt::constants::TILE_WIDTH : 1;
    const uint32_t cell_size = is_tile ? tt::tile_size(cb_data_format) : input.element_size();

    // Tile-aligned shift is required along the last two dims when tilized.
    if (is_tile) {
        if (is_last_dim) {
            TT_FATAL(shift % cell_w == 0, "Native tilized roll on the width dim requires a tile-aligned shift");
        } else if (static_cast<uint32_t>(dim) == rank - 2) {
            TT_FATAL(shift % cell_h == 0, "Native tilized roll on the height dim requires a tile-aligned shift");
        }
    }

    const uint32_t W_cells = shape[rank - 1] / cell_w;

    const auto& out_ss = output.shard_spec().value();
    const auto& in_ss = input.shard_spec().value();
    const uint32_t shard_cells_h = out_ss.shape[0] / cell_h;
    const uint32_t shard_cells_w = out_ss.shape[1] / cell_w;
    TT_FATAL(
        in_ss.shape[0] == out_ss.shape[0] && in_ss.shape[1] == out_ss.shape[1],
        "Native sharded roll expects identical input/output shard shapes");
    TT_FATAL(W_cells % shard_cells_w == 0, "Shard width must evenly divide the tensor");
    const uint32_t num_shard_cols = W_cells / shard_cells_w;

    // Cell-row dim sizes (dims 0..rank-2 collapsed): the height dim is measured in tile-rows.
    std::vector<uint32_t> rd(rank, 1);
    uint32_t H_cells = 1;
    for (uint32_t i = 0; i + 1 < rank; i++) {
        rd[i] = (i == rank - 2) ? shape[i] / cell_h : shape[i];
        H_cells *= rd[i];
    }
    TT_FATAL(H_cells % shard_cells_h == 0, "Shard height must evenly divide the tensor");

    auto* device = input.device();

    // Row-major enumeration of the shard grid. shard_linear = sr*num_shard_cols + sc indexes
    // both the logical core (for runtime args) and the physical core (for NOC source reads).
    std::vector<CoreCoord> logical_cores;
    std::vector<CoreCoord> ordered_cores;
    for (const auto& range : out_ss.grid.ranges()) {
        for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; y++) {
            for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; x++) {
                logical_cores.push_back(CoreCoord(x, y));
                ordered_cores.push_back(device->worker_core_from_logical_core(CoreCoord(x, y)));
            }
        }
    }
    const uint32_t num_cores = ordered_cores.size();

    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    // Sharded buffers store one shard cell-row per page, padded up to the L1 alignment.
    const uint32_t row_pitch_bytes = ((shard_cells_w * cell_size + l1_alignment - 1) / l1_alignment) * l1_alignment;

    auto shard_linear = [&](uint32_t row, uint32_t col) {
        return (row / shard_cells_h) * num_shard_cols + (col / shard_cells_w);
    };
    auto local_offset = [&](uint32_t row, uint32_t col) {
        const uint32_t local_r = row % shard_cells_h;
        const uint32_t local_c = col % shard_cells_w;
        return local_r * row_pitch_bytes + local_c * cell_size;
    };

    // Strides over the cell-row dims (all dims except the last), row-major, for higher-dim rolls.
    std::vector<uint32_t> row_stride(rank, 0);
    if (!is_last_dim) {
        uint32_t s = 1;
        for (int32_t k = static_cast<int32_t>(rank) - 2; k >= 0; k--) {
            row_stride[k] = s;
            s *= rd[k];
        }
    }
    // Coordinate shift for the rolled dim, in cell-row units (tile-rows for the height dim).
    const uint32_t dim_size_cells = (static_cast<uint32_t>(dim) == rank - 2) ? rd[dim] : shape[dim];
    const uint32_t shift_cells = (static_cast<uint32_t>(dim) == rank - 2) ? shift / cell_h : shift;

    auto rolled_src_row = [&](uint32_t r) -> uint32_t {
        // Decrement the dim-th coordinate by shift (mod dim_size); other coords unchanged.
        const uint32_t coord_d = (r / row_stride[dim]) % dim_size_cells;
        const uint32_t src_coord_d = (coord_d + dim_size_cells - (shift_cells % dim_size_cells)) % dim_size_cells;
        return r + (src_coord_d - coord_d) * row_stride[dim];
    };

    // --- Column-piece structure (identical for every cell-row) ---
    // Last-dim roll rotates columns within a row; higher-dim rolls keep columns identity.
    auto split_into_pieces = [&](uint32_t dst_col0, uint32_t src_col0, uint32_t len, std::vector<ColPiece>& out) {
        uint32_t done = 0;
        while (done < len) {
            const uint32_t dcol = dst_col0 + done;
            const uint32_t scol = src_col0 + done;
            const uint32_t dst_boundary = (dcol / shard_cells_w + 1) * shard_cells_w;
            const uint32_t src_boundary = (scol / shard_cells_w + 1) * shard_cells_w;
            const uint32_t piece = std::min({len - done, dst_boundary - dcol, src_boundary - scol});
            out.push_back(ColPiece{.dst_col = dcol, .src_col = scol, .len = piece});
            done += piece;
        }
    };

    std::vector<ColPiece> col_pieces;
    if (is_last_dim) {
        const uint32_t s = (shift / cell_w) % W_cells;
        if (s > 0) {
            split_into_pieces(0, W_cells - s, s, col_pieces);  // out[:, 0:s)   <- in[:, W-s:W)
        }
        split_into_pieces(s, 0, W_cells - s, col_pieces);  // out[:, s:W)   <- in[:, 0:W-s)
    } else {
        split_into_pieces(0, 0, W_cells, col_pieces);  // identity columns, permuted rows
    }

    // --- Build transfers, coalescing consecutive cell-rows into strided runs ---
    std::vector<std::vector<TransferDesc>> all_transfers(num_cores);

    for (const auto& p : col_pieces) {
        const uint32_t copy_bytes = p.len * cell_size;
        bool have_run = false;
        uint32_t run_dst_core = 0, run_src_core = 0;
        uint32_t run_src_off = 0, run_dst_off = 0, run_num = 0;
        auto flush = [&]() {
            if (!have_run) {
                return;
            }
            all_transfers[run_dst_core].push_back(TransferDesc{
                .src_physical_core = ordered_cores[run_src_core],
                .src_l1_offset = run_src_off,
                .dst_offset = run_dst_off,
                .copy_size = copy_bytes,
                .src_stride = row_pitch_bytes,
                .dst_stride = row_pitch_bytes,
                .num_rows = run_num,
            });
            have_run = false;
        };
        for (uint32_t r = 0; r < H_cells; r++) {
            const uint32_t src_row = is_last_dim ? r : rolled_src_row(r);
            const uint32_t dst_core = shard_linear(r, p.dst_col);
            const uint32_t src_core = shard_linear(src_row, p.src_col);
            const uint32_t src_off = local_offset(src_row, p.src_col);
            const uint32_t dst_off = local_offset(r, p.dst_col);
            // Extend the run only if cores match and both offsets advance by exactly one pitch.
            if (have_run && dst_core == run_dst_core && src_core == run_src_core &&
                dst_off == run_dst_off + run_num * row_pitch_bytes &&
                src_off == run_src_off + run_num * row_pitch_bytes) {
                run_num++;
            } else {
                flush();
                have_run = true;
                run_dst_core = dst_core;
                run_src_core = src_core;
                run_src_off = src_off;
                run_dst_off = dst_off;
                run_num = 1;
            }
        }
        flush();
    }

    // --- Circular buffers: cb0 over input, cb16 over output, cb1 scratch (own L1) ---
    constexpr uint32_t input_cb_id = 0;
    constexpr uint32_t output_cb_id = 16;
    constexpr uint32_t scratch_cb_id = 1;
    const uint32_t cb_page_size = is_tile ? cell_size : row_pitch_bytes;

    ProgramDescriptor desc;
    const uint32_t shard_l1_size = shard_cells_h * row_pitch_bytes;
    desc.cbs.push_back(CBDescriptor{
        .total_size = shard_l1_size,
        .core_ranges = out_ss.grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_cb_id),
            .data_format = cb_data_format,
            .page_size = cb_page_size,
        }}},
        .buffer = input.buffer(),
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = shard_l1_size,
        .core_ranges = out_ss.grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_id),
            .data_format = cb_data_format,
            .page_size = cb_page_size,
        }}},
        .buffer = output.buffer(),
    });
    // Scratch CB: holds one alignment-padded segment (max copy is one shard row width).
    const uint32_t scratch_size = row_pitch_bytes + 2 * l1_alignment;
    desc.cbs.push_back(CBDescriptor{
        .total_size = scratch_size,
        .core_ranges = out_ss.grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(scratch_cb_id),
            .data_format = cb_data_format,
            .page_size = scratch_size,
        }}},
    });

    // --- Kernel ---
    uint32_t max_num_transfers = 0;
    for (const auto& t : all_transfers) {
        max_num_transfers = std::max(max_num_transfers, static_cast<uint32_t>(t.size()));
    }
    constexpr uint32_t runtime_args_limit = 256;
    constexpr uint32_t args_per_transfer = 9;
    TT_FATAL(
        1 + max_num_transfers * args_per_transfer <= runtime_args_limit,
        "Native sharded roll: too many copy segments per core ({}). Reduce grid/shape.",
        max_num_transfers);

    const std::vector<uint32_t> compile_time_args = {output_cb_id, scratch_cb_id, l1_alignment};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/roll/device/kernels/dataflow/roll_sharded_reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = out_ss.grid;
    reader_desc.compile_time_args = compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    auto build_runtime_args = [&](const std::vector<TransferDesc>& descs) {
        KernelDescriptor::CoreRuntimeArgs args;
        args.reserve(1 + descs.size() * args_per_transfer);
        args.push_back(static_cast<uint32_t>(descs.size()));
        for (const auto& td : descs) {
            args.push_back(td.src_physical_core.x);
            args.push_back(td.src_physical_core.y);
            args.push_back(input_cb_id);
            args.push_back(td.src_l1_offset);
            args.push_back(td.dst_offset);
            args.push_back(td.copy_size);
            args.push_back(td.src_stride);
            args.push_back(td.dst_stride);
            args.push_back(td.num_rows);
        }
        return args;
    };

    for (uint32_t c = 0; c < num_cores; c++) {
        reader_desc.runtime_args.emplace_back(logical_cores[c], build_runtime_args(all_transfers[c]));
    }

    desc.kernels.push_back(std::move(reader_desc));
    return desc;
}

}  // namespace ttnn::prim
