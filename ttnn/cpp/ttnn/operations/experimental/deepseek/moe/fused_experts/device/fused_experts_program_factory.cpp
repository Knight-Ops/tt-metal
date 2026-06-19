// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_device_operation.hpp"

#include <algorithm>
#include <string>

#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

using namespace tt;
using namespace tt::tt_metal;

namespace {
constexpr std::string_view kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/device/kernels";

// Compute grid for the broadcast: 8x8 = 64 cores.
constexpr uint32_t GRID_X = 8;
constexpr uint32_t GRID_Y = 8;

uint32_t align_up_32(uint32_t x) { return (x + 31u) & ~31u; }
}  // namespace

// First version: two sender cores drive the grid in parallel on separate NoCs.
//   - {0,0} (NoC 0) reads the routing weights, computes the selected ("hit") expert
//     ids on-device, writes them to the UINT32 output tensor, and broadcasts the ids
//     to all other cores' L1 (cb_bcast), bumping sem_id.
//   - {1,0} (NoC 1) reads the decode activation row (input_tensor) and broadcasts it
//     to all other cores' L1 (cb_input), bumping sem_input_id.
// The remaining 62 cores wait on both semaphores and exit. The full fused
// matmul/SwiGLU pipeline (using all 64 cores for compute) is a later milestone.
ProgramDescriptor FusedExpertsDeviceOperation::MultiCore::create_descriptor(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& routing_weights = tensor_args.routing_weights;
    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* routing_buffer = routing_weights.buffer();
    auto* input_buffer = input_tensor.buffer();
    auto* out_buffer = output_tensor.buffer();
    auto* device = routing_weights.device();

    const auto grid = device->compute_with_storage_grid_size();
    TT_FATAL(
        grid.x >= GRID_X && grid.y >= GRID_Y,
        "fused_experts: expected at least {}x{} compute grid, got {}x{}",
        GRID_X,
        GRID_Y,
        grid.x,
        grid.y);

    const uint32_t num_experts = static_cast<uint32_t>(tensor_args.gate_up_weights.size());
    const uint32_t sentinel = num_experts;  // "no expert" marker for unused output slots

    // gate_up weights are [K=H, N=2I] per expert (TILE layout); each core fetches a
    // [K, 64] (k_tiles x 2-tile) column slice for every active expert. All experts
    // share the same layout, so one TensorAccessorArgs (from weight 0) is reused.
    const auto& gate_up0 = tensor_args.gate_up_weights.front();
    auto* gate_up0_buffer = gate_up0.buffer();
    constexpr uint32_t TILE_DIM = 32;
    const uint32_t k_tiles = static_cast<uint32_t>(gate_up0.logical_shape()[-2]) / TILE_DIM;
    const uint32_t n_tiles = static_cast<uint32_t>(gate_up0.logical_shape()[-1]) / TILE_DIM;
    const uint32_t weight_tile_bytes = static_cast<uint32_t>(gate_up0_buffer->page_size());
    constexpr uint32_t kTilesPerCore = 2;
    const uint32_t weights_cb_bytes = k_tiles * kTilesPerCore * weight_tile_bytes;

    const tt::DataFormat gate_up_df = datatype_to_dataformat_converter(gate_up0.dtype());

    const tt::DataFormat routing_df = datatype_to_dataformat_converter(routing_weights.dtype());
    const tt::DataFormat out_df = datatype_to_dataformat_converter(output_tensor.dtype());
    const tt::DataFormat input_df = datatype_to_dataformat_converter(input_tensor.dtype());

    constexpr uint32_t routing_elem_bytes = 2;  // bfloat16
    constexpr uint32_t out_elem_bytes = 4;      // uint32
    const uint32_t routing_page_bytes = num_experts * routing_elem_bytes;
    const uint32_t bcast_page_bytes = num_experts * out_elem_bytes;

    // Activations are broadcast verbatim, page-by-page, so the kernel is layout agnostic
    // (handles ROW_MAJOR and TILE inputs alike).
    const uint32_t input_page_size = static_cast<uint32_t>(input_buffer->page_size());
    const uint32_t input_num_pages = static_cast<uint32_t>(input_buffer->num_pages());
    const uint32_t input_total_bytes = input_page_size * input_num_pages;

    const uint32_t routing_cb_bytes = std::max<uint32_t>(align_up_32(routing_page_bytes), 32u);
    const uint32_t bcast_cb_bytes = std::max<uint32_t>(align_up_32(bcast_page_bytes), 32u);
    const uint32_t input_cb_bytes = std::max<uint32_t>(align_up_32(input_total_bytes), 32u);

    // Core sets: the full 8x8 grid, the two senders {0,0} (expert ids) and {1,0}
    // (input activations), and the 62 receivers.
    const CoreCoord sender{0, 0};
    const CoreCoord input_sender{1, 0};
    const CoreRange all_range({0, 0}, {GRID_X - 1, GRID_Y - 1});
    const CoreRangeSet all_cores{all_range};
    const CoreRangeSet sender_set{CoreRange{sender, sender}};
    const CoreRangeSet input_sender_set{CoreRange{input_sender, input_sender}};
    // Receivers = full grid minus {0,0} and {1,0}: row 0 (x=2..7) plus rows 1..7 (all x).
    const CoreRangeSet receiver_cores{std::vector<CoreRange>{
        CoreRange{{2, 0}, {GRID_X - 1, 0}},
        CoreRange{{0, 1}, {GRID_X - 1, GRID_Y - 1}},
    }};

    ProgramDescriptor desc;

    // Two broadcast-ready semaphores on ALL cores: one for the expert ids ({0,0})
    // and one for the activation row ({1,0}).
    constexpr uint32_t sem_id = 0;
    constexpr uint32_t sem_input_id = 1;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sem_id,
        .core_type = CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sem_input_id,
        .core_type = CoreType::WORKER,
        .core_ranges = all_cores,
        .initial_value = 0,
    });

    // CBs are allocated identically on all cores so cb_bcast lands at the same L1
    // address everywhere (required for the multicast write to be valid).
    constexpr uint32_t cb_routing = CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = routing_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_routing,
            .data_format = routing_df,
            .page_size = routing_cb_bytes,
        }}},
    });

    constexpr uint32_t cb_bcast = CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = bcast_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_bcast,
            .data_format = out_df,
            .page_size = bcast_cb_bytes,
        }}},
    });

    constexpr uint32_t cb_input = CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_input,
            .data_format = input_df,
            .page_size = input_cb_bytes,
        }}},
    });

    // Per-core gate_up weight slice ([K, 64] = k_tiles x 2 tiles).
    constexpr uint32_t cb_weights = CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = weights_cb_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_weights,
            .data_format = gate_up_df,
            .page_size = weight_tile_bytes,
        }}},
    });

    // Multicast rectangle (NoC coords) covering the whole grid. Non-loopback
    // multicast excludes the sender, so num_dests = total cores - 1.
    const auto corner_a = device->worker_core_from_logical_core(CoreCoord{0, 0});
    const auto corner_b = device->worker_core_from_logical_core(CoreCoord{GRID_X - 1, GRID_Y - 1});
    const uint32_t mcast_start_x = std::min<uint32_t>(corner_a.x, corner_b.x);
    const uint32_t mcast_start_y = std::min<uint32_t>(corner_a.y, corner_b.y);
    const uint32_t mcast_end_x = std::max<uint32_t>(corner_a.x, corner_b.x);
    const uint32_t mcast_end_y = std::max<uint32_t>(corner_a.y, corner_b.y);
    const uint32_t num_dests = GRID_X * GRID_Y - 1;

    // Each core owns the output N-tile pair [2*idx, 2*idx+1], idx = y*GRID_X + x.
    auto col_start_tile_for = [](const CoreCoord& c) -> uint32_t { return (c.y * GRID_X + c.x) * kTilesPerCore; };
    // Base address of every expert's gate_up weight, in expert-id order.
    std::vector<uint32_t> gate_up_addrs;
    gate_up_addrs.reserve(num_experts);
    for (const auto& w : tensor_args.gate_up_weights) {
        gate_up_addrs.push_back(static_cast<uint32_t>(w.buffer()->address()));
    }
    auto append_addrs = [&](KernelDescriptor::CoreRuntimeArgs& args) {
        for (uint32_t a : gate_up_addrs) {
            args.push_back(a);
        }
    };

    // ---- Sender kernel on {0,0}. ----
    std::vector<uint32_t> sender_ct_args = {
        num_experts,
        sentinel,
        cb_routing,
        cb_bcast,
        routing_page_bytes,
        bcast_page_bytes,
        sem_id,
        cb_weights,
        k_tiles,
        n_tiles,
        weight_tile_bytes,
    };
    TensorAccessorArgs(*routing_buffer).append_to(sender_ct_args);
    TensorAccessorArgs(*out_buffer).append_to(sender_ct_args);
    TensorAccessorArgs(*gate_up0_buffer).append_to(sender_ct_args);

    KernelDescriptor sender_desc;
    sender_desc.kernel_source = std::string(kKernelDir) + "/dataflow/compute_expert_ids.cpp";
    sender_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_desc.core_ranges = sender_set;
    sender_desc.compile_time_args = sender_ct_args;
    // BRISC (RISCV_0) owns NoC 0 (matches `Noc noc(0)` in the kernel).
    sender_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_0,
    };
    {
        KernelDescriptor::CoreRuntimeArgs args{
            routing_buffer->address(),
            out_buffer->address(),
            mcast_start_x,
            mcast_start_y,
            mcast_end_x,
            mcast_end_y,
            num_dests,
            col_start_tile_for(sender),
        };
        append_addrs(args);
        sender_desc.runtime_args.emplace_back(sender, std::move(args));
    }
    desc.kernels.push_back(std::move(sender_desc));

    // ---- Input-broadcaster kernel on {1,0} (uses NoC 1). ----
    std::vector<uint32_t> input_ct_args = {
        cb_input,
        input_page_size,
        input_num_pages,
        sem_input_id,
        sem_id,
        num_experts,
        cb_bcast,
        cb_weights,
        k_tiles,
        n_tiles,
        weight_tile_bytes,
    };
    TensorAccessorArgs(*input_buffer).append_to(input_ct_args);
    TensorAccessorArgs(*gate_up0_buffer).append_to(input_ct_args);

    KernelDescriptor input_sender_desc;
    input_sender_desc.kernel_source = std::string(kKernelDir) + "/dataflow/broadcast_input.cpp";
    input_sender_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    input_sender_desc.core_ranges = input_sender_set;
    input_sender_desc.compile_time_args = input_ct_args;
    // NCRISC (RISCV_1) owns NoC 1 (matches `Noc noc(1)` in the kernel) so it runs in
    // parallel with the {0,0} expert-id sender on NoC 0.
    input_sender_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_1,
    };
    // NoC 1 multicasts traverse from high to low coordinates, so the rectangle's
    // start/end must be swapped relative to the NoC 0 sender.
    {
        KernelDescriptor::CoreRuntimeArgs args{
            input_buffer->address(),
            mcast_end_x,
            mcast_end_y,
            mcast_start_x,
            mcast_start_y,
            num_dests,
            col_start_tile_for(input_sender),
        };
        append_addrs(args);
        input_sender_desc.runtime_args.emplace_back(input_sender, std::move(args));
    }
    desc.kernels.push_back(std::move(input_sender_desc));

    // ---- Receiver kernel on the other 62 cores. ----
    std::vector<uint32_t> receiver_ct_args = {
        sem_id,
        sem_input_id,
        num_experts,
        cb_bcast,
        cb_weights,
        k_tiles,
        n_tiles,
        weight_tile_bytes,
    };
    TensorAccessorArgs(*gate_up0_buffer).append_to(receiver_ct_args);

    KernelDescriptor receiver_desc;
    receiver_desc.kernel_source = std::string(kKernelDir) + "/dataflow/wait_expert_ids.cpp";
    receiver_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    receiver_desc.core_ranges = receiver_cores;
    receiver_desc.compile_time_args = receiver_ct_args;
    receiver_desc.config = ReaderConfigDescriptor{};
    // Per-core: this core's column slice start, then every expert's gate_up address.
    for (const auto& cr : receiver_cores.ranges()) {
        for (const auto& core : cr) {
            KernelDescriptor::CoreRuntimeArgs args{col_start_tile_for(core)};
            append_addrs(args);
            receiver_desc.runtime_args.emplace_back(core, std::move(args));
        }
    }
    desc.kernels.push_back(std::move(receiver_desc));

    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts
