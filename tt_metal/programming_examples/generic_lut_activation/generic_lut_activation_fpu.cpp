// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host code for RISC-V / TF32 piecewise polynomial activation kernels.
// Identical to generic_lut_activation.cpp except:
//   1. Always passes POLY_DEGREE + NUM_SEGMENTS as compute compile-time args.
//   2. Creates scratch CBs 26, 27, 28 (unused by piecewise_riscv; still needed by piecewise_fpu_tf32).
//   3. For the TF32 variant, CB 26/27 use Float32 (TF32 stored in 4-byte words).

#include <fmt/ostream.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/work_split.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "lut_loader.hpp"

using namespace tt::tt_metal;

#define START_TIMER(name) auto timer_##name##_start = std::chrono::high_resolution_clock::now()
#define END_TIMER(name) \
    do { \
        auto timer_##name##_end = std::chrono::high_resolution_clock::now(); \
        auto timer_##name##_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(timer_##name##_end - timer_##name##_start); \
        fmt::print("TIMING_{}: {:.6f}\n", #name, timer_##name##_duration.count() / 1e6); \
    } while(0)

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// KERNEL_VARIANT must be defined (piecewise_riscv or piecewise_fpu_tf32)
#ifndef KERNEL_VARIANT
#error "KERNEL_VARIANT must be defined for this host (piecewise_riscv or piecewise_fpu_tf32)"
#endif
#define COMPUTE_KERNEL_PATH OVERRIDE_KERNEL_PREFIX "generic_lut_activation/kernels/compute/" KERNEL_VARIANT ".cpp"

#ifndef LUT_SIZE
#error "LUT_SIZE must be defined"
#endif

#ifndef POLY_DEGREE
#error "POLY_DEGREE must be defined for FPU variants"
#endif

#ifndef NUM_SEGMENTS
#error "NUM_SEGMENTS must be defined for FPU variants"
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        fmt::print(stderr,
            "Usage: {} <csv_file> --activation <name> --precision <bf16|fp32> "
            "--range-min <min> --range-max <max> [--tiles N] [--compute-loops N]\n", argv[0]);
        return 1;
    }

    std::string csv_filename = argv[1];
    std::string activation_name;
    std::string precision_arg;
    float range_min = 0.0f;
    float range_max = 0.0f;
    uint32_t n_tiles = 32;
    uint32_t compute_loop_factor = 1;

    bool has_activation = false;
    bool has_precision = false;
    bool has_range_min = false;
    bool has_range_max = false;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--activation" || arg == "-a") {
            activation_name = argv[++i];
            has_activation = true;
        } else if (arg == "--range-min" || arg == "-rmin") {
            range_min = std::stof(argv[++i]);
            has_range_min = true;
        } else if (arg == "--range-max" || arg == "-rmax") {
            range_max = std::stof(argv[++i]);
            has_range_max = true;
        } else if (arg == "--tiles" || arg == "-t") {
            n_tiles = std::stoi(argv[++i]);
        } else if (arg == "--precision" || arg == "-p") {
            precision_arg = argv[++i];
            has_precision = true;
        } else if (arg == "--compute-loops" || arg == "-cl") {
            compute_loop_factor = std::stoi(argv[++i]);
        } else {
            fmt::print(stderr, "Error: unknown argument '{}'\n", arg);
            return 1;
        }
    }

    if (!has_activation || !has_precision || !has_range_min || !has_range_max) {
        fmt::print(stderr, "Error: --activation, --precision, --range-min, --range-max are required\n");
        return 1;
    }
    if (range_min >= range_max) {
        fmt::print(stderr, "Error: range-min must be less than range-max\n");
        return 1;
    }

    bool pass = true;
    try {
        fmt::print("{}\n", std::string(60, '='));
        fmt::print("FPU Piecewise Polynomial Activation (kernel: " KERNEL_VARIANT ")\n");
        fmt::print("POLY_DEGREE={}, NUM_SEGMENTS={}, LUT_SIZE={}\n",
                   (uint32_t)POLY_DEGREE, (uint32_t)NUM_SEGMENTS, (uint32_t)LUT_SIZE);
        fmt::print("{}\n", std::string(60, '='));

        bool use_bf16_mode = (precision_arg == "bf16");
        const bool is_tf32_variant = (std::string(KERNEL_VARIANT) == "piecewise_fpu_tf32");

        const auto input_data_format = use_bf16_mode ? tt::DataFormat::Float16_b : tt::DataFormat::Float32;

        fmt::print("Loading coefficients from: {}\n", csv_filename);
        fmt::print("Precision: {}\n", use_bf16_mode ? "BF16" : "FP32");
        fmt::print("TF32 variant: {}\n", is_tf32_variant ? "yes" : "no");

        std::string range_reduction_method = LUTLoader::extract_range_reduction_method(csv_filename);

        std::vector<float> lut_data = LUTLoader::load(csv_filename);
        fmt::print("Loaded {} entries from CSV\n", lut_data.size());

        if (lut_data.size() < LUT_SIZE) {
            lut_data.resize(LUT_SIZE, 0.0f);
        } else if (lut_data.size() > LUT_SIZE) {
            fmt::print("ERROR: LUT file has {} entries but LUT_SIZE={}\n", lut_data.size(), (uint32_t)LUT_SIZE);
            return 1;
        }

        fmt::print("Activation: {} | Test range: [{}, {}]\n", activation_name, range_min, range_max);

        // Device + program
        START_TIMER(DEVICE_INIT);
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        END_TIMER(DEVICE_INIT);

        START_TIMER(PROGRAM_CREATION);
        Program program = CreateProgram();

        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        const uint32_t bytes_per_element = use_bf16_mode ? sizeof(bfloat16) : sizeof(float);
        const uint32_t tile_size_bytes = bytes_per_element * elements_per_tile;
        const uint32_t dram_buffer_size = tile_size_bytes * n_tiles;

        fmt::print("Processing {} tiles ({} elements total)\n", n_tiles, n_tiles * elements_per_tile);

        CoreCoord grid_size = mesh_device->compute_with_storage_grid_size();
        fmt::print("Grid size: {}x{}\n", grid_size.x, grid_size.y);

        auto [num_cores, all_cores, core_group_1, core_group_2,
              tiles_per_core_1, tiles_per_core_2] =
            split_work_to_cores(grid_size, n_tiles);

        fmt::print("Work split: {} cores total\n", num_cores);

        // Per-core LUT buffers (CB 25)
        constexpr auto cb_lut = tt::CBIndex::c_25;
        const uint32_t lut_size_bytes = LUT_SIZE * sizeof(float);
        uint32_t num_cores_y = grid_size.y;

        for (uint32_t i = 0; i < num_cores; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            ShardSpecBuffer shard_spec(
                CoreRangeSet(CoreRange(core, core)),
                {1, LUT_SIZE},
                ShardOrientation::ROW_MAJOR,
                {1, LUT_SIZE},
                {1, LUT_SIZE}
            );
            BufferShardingArgs sharding_args(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
            distributed::DeviceLocalBufferConfig lut_local_config{
                .page_size = lut_size_bytes,
                .buffer_type = BufferType::L1,
                .sharding_args = sharding_args
            };
            distributed::ReplicatedBufferConfig lut_buffer_config{.size = lut_size_bytes};
            auto core_lut_buffer = distributed::MeshBuffer::create(lut_buffer_config, lut_local_config, mesh_device.get());

            distributed::EnqueueWriteMeshBuffer(cq, core_lut_buffer, lut_data, false);

            CreateCircularBuffer(
                program, core,
                CircularBufferConfig(lut_size_bytes, {{cb_lut, tt::DataFormat::Float32}})
                    .set_page_size(cb_lut, lut_size_bytes)
                    .set_globally_allocated_address(*core_lut_buffer->get_backing_buffer())
            );
        }

        fmt::print("LUT loaded into L1 on {} cores\n", num_cores);

        // Input / output CBs (CB 0 and CB 16)
        constexpr uint32_t tiles_per_cb = 2;
        constexpr auto cb_in  = tt::CBIndex::c_0;
        constexpr auto cb_out = tt::CBIndex::c_16;

        CreateCircularBuffer(program, all_cores,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{cb_in, input_data_format}})
                .set_page_size(cb_in, tile_size_bytes));

        CreateCircularBuffer(program, all_cores,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{cb_out, input_data_format}})
                .set_page_size(cb_out, tile_size_bytes));

        // Scratch CBs for FPU matmul (CB 26, 27, 28)
        // C and P matrices: TF32 variant uses Float32 (4 bytes/elem), BF16 variant uses Float16_b
        // Y result: TF32 variant FP32 dest acc → Float32; BF16 variant → Float16_b
        {
            const auto scratch_cp_fmt = is_tf32_variant ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
            const auto scratch_y_fmt  = is_tf32_variant ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;

            // BF16 tile = 32×32 × 2 bytes = 2048; FP32/TF32 tile = 32×32 × 4 bytes = 4096
            const uint32_t scratch_cp_tile_size = (scratch_cp_fmt == tt::DataFormat::Float32) ? 4096u : 2048u;
            const uint32_t scratch_y_tile_size  = (scratch_y_fmt  == tt::DataFormat::Float32) ? 4096u : 2048u;

            constexpr auto cb_scratch_c = tt::CBIndex::c_26;
            constexpr auto cb_scratch_p = tt::CBIndex::c_27;
            constexpr auto cb_scratch_y = tt::CBIndex::c_28;

            // Double-buffer scratch CBs to allow overlap between load and compute
            CreateCircularBuffer(program, all_cores,
                CircularBufferConfig(2 * scratch_cp_tile_size, {{cb_scratch_c, scratch_cp_fmt}})
                    .set_page_size(cb_scratch_c, scratch_cp_tile_size));

            CreateCircularBuffer(program, all_cores,
                CircularBufferConfig(2 * scratch_cp_tile_size, {{cb_scratch_p, scratch_cp_fmt}})
                    .set_page_size(cb_scratch_p, scratch_cp_tile_size));

            CreateCircularBuffer(program, all_cores,
                CircularBufferConfig(2 * scratch_y_tile_size, {{cb_scratch_y, scratch_y_fmt}})
                    .set_page_size(cb_scratch_y, scratch_y_tile_size));

            fmt::print("Created scratch CBs 26/27/28 ({}, tile_cp={}B, tile_y={}B)\n",
                       is_tf32_variant ? "TF32/FP32" : "BF16",
                       scratch_cp_tile_size, scratch_y_tile_size);
        }

        END_TIMER(PROGRAM_CREATION);

        // DRAM buffers
        START_TIMER(BUFFER_ALLOCATION);
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = tile_size_bytes,
            .buffer_type = BufferType::DRAM
        };
        distributed::ReplicatedBufferConfig buffer_config{.size = dram_buffer_size};

        auto input_buffer  = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto output_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        END_TIMER(BUFFER_ALLOCATION);

        // Test input data
        START_TIMER(DATA_PREPARATION);
        const size_t num_elements = elements_per_tile * n_tiles;
        const float test_range = range_max - range_min;

        std::vector<bfloat16> input_data_bf16;
        std::vector<float>    input_data_fp32;

        if (use_bf16_mode) {
            input_data_bf16.resize(num_elements);
            for (size_t i = 0; i < num_elements; i++) {
                input_data_bf16[i] = bfloat16(range_min + test_range * (i / float(num_elements)));
            }
            END_TIMER(DATA_PREPARATION);
            START_TIMER(HOST_TO_DEVICE);
            distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_data_bf16, false);
            END_TIMER(HOST_TO_DEVICE);
        } else {
            input_data_fp32.resize(num_elements);
            for (size_t i = 0; i < num_elements; i++) {
                input_data_fp32[i] = range_min + test_range * (i / float(num_elements));
            }
            END_TIMER(DATA_PREPARATION);
            START_TIMER(HOST_TO_DEVICE);
            distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_data_fp32, false);
            END_TIMER(HOST_TO_DEVICE);
        }

        auto get_input_value = [&](size_t i) -> float {
            return use_bf16_mode ? static_cast<float>(input_data_bf16[i]) : input_data_fp32[i];
        };

        // Kernels
        START_TIMER(KERNEL_CREATION);

        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*input_buffer->get_backing_buffer()).append_to(reader_compile_time_args);
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "generic_lut_activation/kernels/dataflow/reader.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_compile_time_args
            });

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*output_buffer->get_backing_buffer()).append_to(writer_compile_time_args);
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "generic_lut_activation/kernels/dataflow/writer.cpp",
            all_cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_compile_time_args
            });

        // Compute compile args: LUT_SIZE, POLY_DEGREE, NUM_SEGMENTS
        std::vector<uint32_t> compute_compile_args = {LUT_SIZE, POLY_DEGREE, NUM_SEGMENTS};

        std::map<std::string, std::string> compute_defines;
        std::vector<UnpackToDestMode> unpack_to_dest_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

        if (!use_bf16_mode || is_tf32_variant) {
            // FP32 mode: hardware handles FP32 via fp32_dest_acc_en + UnpackToDestFp32
            unpack_to_dest_modes[static_cast<uint32_t>(cb_in)]  = UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_modes[static_cast<uint32_t>(cb_out)] = UnpackToDestMode::UnpackToDestFp32;
        }

        if (!range_reduction_method.empty()) {
            if (range_reduction_method == "exp") {
                compute_defines["RANGE_REDUCTION_EXP"] = "1";
            } else if (range_reduction_method == "trig") {
                compute_defines["RANGE_REDUCTION_TRIG"] = "1";
            } else if (range_reduction_method == "tan") {
                compute_defines["RANGE_REDUCTION_TAN"] = "1";
            } else if (range_reduction_method == "cbrt") {
                compute_defines["RANGE_REDUCTION_CBRT"] = "1";
            }
        }

        fmt::print("Compute compile args: LUT_SIZE={}, POLY_DEGREE={}, NUM_SEGMENTS={}\n",
                   LUT_SIZE, POLY_DEGREE, NUM_SEGMENTS);

        auto make_compute_config = [&]() {
            return ComputeConfig{
                .fp32_dest_acc_en = (!use_bf16_mode || is_tf32_variant),
                .unpack_to_dest_mode = unpack_to_dest_modes,
                .math_approx_mode = (use_bf16_mode && !is_tf32_variant),
                .compile_args = compute_compile_args,
                .defines = compute_defines
            };
        };

        auto compute_kernel_1 = CreateKernel(program, COMPUTE_KERNEL_PATH, core_group_1, make_compute_config());

        KernelHandle compute_kernel_2 = 0;
        if (!core_group_2.ranges().empty()) {
            compute_kernel_2 = CreateKernel(program, COMPUTE_KERNEL_PATH, core_group_2, make_compute_config());
        }

        // Runtime args
        uint32_t tiles_written = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            uint32_t tiles_this_core = 0;
            KernelHandle compute_kernel_id = 0;

            if (core_group_1.contains(core)) {
                tiles_this_core = tiles_per_core_1;
                compute_kernel_id = compute_kernel_1;
            } else {
                tiles_this_core = tiles_per_core_2;
                compute_kernel_id = compute_kernel_2;
            }

            SetRuntimeArgs(program, reader, core,
                {input_buffer->address(), tiles_this_core, tiles_written});
            SetRuntimeArgs(program, writer, core,
                {output_buffer->address(), tiles_this_core, tiles_written});
            SetRuntimeArgs(program, compute_kernel_id, core,
                {tiles_this_core, compute_loop_factor});

            tiles_written += tiles_this_core;
        }

        TT_FATAL(tiles_written == n_tiles, "Tile distribution mismatch! {} != {}", tiles_written, n_tiles);
        END_TIMER(KERNEL_CREATION);
        fmt::print("Kernels created and configured\n");

        // Execute
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        workload.add_program(device_range, std::move(program));

        START_TIMER(KERNEL_EXECUTION);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        END_TIMER(KERNEL_EXECUTION);

        if (std::getenv("TT_METAL_DEVICE_PROFILER")) {
            ReadMeshDeviceProfilerResults(*mesh_device);
        }

        fmt::print("Execution complete\n\n");

        // Read results
        START_TIMER(DEVICE_TO_HOST);
        std::vector<bfloat16> output_data_bf16;
        std::vector<float>    output_data_fp32;

        if (use_bf16_mode && !is_tf32_variant) {
            distributed::EnqueueReadMeshBuffer(cq, output_data_bf16, output_buffer, true);
        } else {
            distributed::EnqueueReadMeshBuffer(cq, output_data_fp32, output_buffer, true);
        }
        END_TIMER(DEVICE_TO_HOST);

        auto get_output_value = [&](size_t i) -> float {
            return (use_bf16_mode && !is_tf32_variant)
                ? static_cast<float>(output_data_bf16[i])
                : output_data_fp32[i];
        };

        const size_t output_size = (use_bf16_mode && !is_tf32_variant)
            ? output_data_bf16.size() : output_data_fp32.size();

        fmt::print("Results (samples):\n");
        fmt::print("{}\n", std::string(60, '-'));
        fmt::print("{:>5} {:>12} {:>12}\n", "Index", "Input", "Output");
        fmt::print("{}\n", std::string(60, '-'));

        const int samples[] = {0, 3276, 6553, 9830, 13107, 16384, 19660, 22937, 26214, 29491, 32767};
        for (auto idx : samples) {
            if (static_cast<size_t>(idx) < output_size) {
                fmt::print("{:5d} {:12.6f} {:12.6f}\n", idx, get_input_value(idx), get_output_value(idx));
            }
        }
        fmt::print("{}\n", std::string(60, '-'));

        // CSV dump
        const char* dump_csv = std::getenv("DUMP_OUTPUT_CSV");
        if (dump_csv) {
            std::ofstream csv_file(dump_csv);
            if (csv_file.is_open()) {
                csv_file << std::fixed << std::setprecision(10);
                csv_file << "input,output\n";
                for (size_t i = 0; i < output_size; ++i) {
                    csv_file << get_input_value(i) << "," << get_output_value(i) << "\n";
                }
                fmt::print("Output dumped to: {}\n", dump_csv);
            }
        }

        if (!mesh_device->close()) {
            pass = false;
        }

        fmt::print("\n{}\n", std::string(60, '='));
        fmt::print("{}\n", pass ? "Test PASSED" : "Test FAILED");
        fmt::print("{}\n", std::string(60, '='));

    } catch (const std::exception& e) {
        fmt::print(stderr, "\nTest failed: {}\n", e.what());
        return 1;
    }

    return pass ? 0 : 1;
}
