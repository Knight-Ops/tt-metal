// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/ostream.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iomanip>

using namespace tt::tt_metal;

// Timing helper macro
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

int main(int argc, char** argv) {
    if (argc < 2) {
        fmt::print(stderr, "Usage: {} <activation_type> --range-min <min> --range-max <max> [--tiles N]\n", argv[0]);
        fmt::print(stderr, "\nRequired arguments:\n");
        fmt::print(stderr, "  <activation_type>        Activation function name\n");
        fmt::print(stderr, "                           Available: gelu, relu, tanh, softplus, exp, leaky_relu,\n");
        fmt::print(stderr, "                                      elu, selu, hardsigmoid, sin, cos, erf, cosh,\n");
        fmt::print(stderr, "                                      sinh, atanh, celu, prelu, softsign, softshrink,\n");
        fmt::print(stderr, "                                      hardtanh, threshold\n");
        fmt::print(stderr, "  --range-min, -rmin <val> Minimum test range value\n");
        fmt::print(stderr, "  --range-max, -rmax <val> Maximum test range value\n");
        fmt::print(stderr, "\nOptional arguments:\n");
        fmt::print(stderr, "  --tiles, -t N            Number of tiles to process (default: 32, range: 1-1024)\n");
        fmt::print(stderr, "  --precision, -p <type>   Precision mode: bf16 (default) or fp32\n");
        fmt::print(stderr, "  --fast-approx            Use fast approximate SFPU mode (default: true)\n");
        fmt::print(stderr, "  --no-fast-approx         Use precise SFPU mode (slower but more accurate)\n");
        fmt::print(stderr, "\nExamples:\n");
        fmt::print(stderr, "  {} gelu --range-min -10 --range-max 10\n", argv[0]);
        fmt::print(stderr, "  {} sigmoid -rmin -10 -rmax 10 --tiles 256\n", argv[0]);
        return 1;
    }

    std::string activation = argv[1];
    uint32_t activation_type;
    float range_min = 0.0f;
    float range_max = 0.0f;
    uint32_t n_tiles = 32;  // Default value
    std::string precision_mode = "bf16";  // Default to BF16
    bool fast_approx = true;  // Default to fast approximate mode

    bool has_range_min = false;
    bool has_range_max = false;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--range-min" || arg == "-rmin") {
            if (i + 1 >= argc) {
                fmt::print(stderr, "Error: --range-min requires an argument\n");
                return 1;
            }
            try {
                range_min = std::stof(argv[++i]);
                has_range_min = true;
            } catch (const std::exception& e) {
                fmt::print(stderr, "Error: invalid range-min value '{}'\n", argv[i]);
                return 1;
            }
        } else if (arg == "--range-max" || arg == "-rmax") {
            if (i + 1 >= argc) {
                fmt::print(stderr, "Error: --range-max requires an argument\n");
                return 1;
            }
            try {
                range_max = std::stof(argv[++i]);
                has_range_max = true;
            } catch (const std::exception& e) {
                fmt::print(stderr, "Error: invalid range-max value '{}'\n", argv[i]);
                return 1;
            }
        } else if (arg == "--tiles" || arg == "-t") {
            if (i + 1 >= argc) {
                fmt::print(stderr, "Error: --tiles requires an argument\n");
                return 1;
            }
            try {
                n_tiles = std::stoi(argv[++i]);
                if (n_tiles == 0 || n_tiles > 1024) {
                    fmt::print(stderr, "Error: tiles must be between 1 and 1024 (got {})\n", n_tiles);
                    return 1;
                }
            } catch (const std::exception& e) {
                fmt::print(stderr, "Error: invalid tiles value '{}'\n", argv[i]);
                return 1;
            }
        } else if (arg == "--precision" || arg == "-p") {
            if (i + 1 >= argc) {
                fmt::print(stderr, "Error: --precision requires an argument\n");
                return 1;
            }
            precision_mode = argv[++i];
            if (precision_mode != "bf16" && precision_mode != "fp32") {
                fmt::print(stderr, "Error: --precision must be 'bf16' or 'fp32'\n");
                return 1;
            }
        } else if (arg == "--fast-approx") {
            fast_approx = true;
        } else if (arg == "--no-fast-approx") {
            fast_approx = false;
        } else {
            fmt::print(stderr, "Error: unknown argument '{}'\n", arg);
            return 1;
        }
    }

    // Check required arguments
    if (!has_range_min) {
        fmt::print(stderr, "Error: --range-min is required\n");
        return 1;
    }
    if (!has_range_max) {
        fmt::print(stderr, "Error: --range-max is required\n");
        return 1;
    }
    if (range_min >= range_max) {
        fmt::print(stderr, "Error: range-min ({}) must be less than range-max ({})\n", range_min, range_max);
        return 1;
    }

    // Map activation name to type ID (matching native_sfpu.cpp kernel)
    if (activation == "gelu") {
        activation_type = 0;
    } else if (activation == "relu") {
        activation_type = 1;
    } else if (activation == "tanh") {
        activation_type = 2;
    } else if (activation == "softplus") {
        activation_type = 3;
    } else if (activation == "exp") {
        activation_type = 4;
    } else if (activation == "leaky_relu") {
        activation_type = 5;
    } else if (activation == "elu") {
        activation_type = 6;
    } else if (activation == "selu") {
        activation_type = 7;
    } else if (activation == "hardsigmoid") {
        activation_type = 8;
    } else if (activation == "sin") {
        activation_type = 9;
    } else if (activation == "cos") {
        activation_type = 10;
    } else if (activation == "erf") {
        activation_type = 11;
    } else if (activation == "cosh") {
        activation_type = 12;
    } else if (activation == "sinh") {
        activation_type = 13;
    } else if (activation == "atanh") {
        activation_type = 14;
    } else if (activation == "celu") {
        activation_type = 15;
    } else if (activation == "prelu") {
        activation_type = 16;
    } else if (activation == "softsign") {
        activation_type = 17;
    } else if (activation == "softshrink") {
        activation_type = 18;
    } else if (activation == "hardtanh") {
        activation_type = 19;
    } else if (activation == "threshold") {
        activation_type = 20;
    } else {
        fmt::print(stderr, "Unknown activation: {}\n", activation);
        fmt::print(stderr, "Available: gelu, relu, tanh, softplus, exp, leaky_relu, elu, selu, hardsigmoid, sin, cos, erf, cosh, sinh, atanh, celu, prelu, softsign, softshrink, hardtanh, threshold\n");
        return 1;
    }

    // Use test range from command-line arguments
    float test_min = range_min;
    float test_max = range_max;

    // Detect precision mode
    const bool use_bf16_mode = (precision_mode == "bf16");
    const auto input_data_format = use_bf16_mode ? tt::DataFormat::Float16_b : tt::DataFormat::Float32;

    fmt::print("Activation: {} | Test range: [{}, {}]\n", activation, test_min, test_max);
    fmt::print("Precision mode: {}\n", use_bf16_mode ? "BF16" : "FP32");
    fmt::print("SFPU mode: {}\n", fast_approx ? "Fast approximate" : "Precise (slower)");

    bool pass = true;

    try {
        fmt::print("{}\n", std::string(60, '='));
        fmt::print("Native SFPU Activation Function Example\n");
        fmt::print("{}\n", std::string(60, '='));
        fmt::print("Activation: {}\n\n", activation);

        // STEP 1: Create device and program
        START_TIMER(DEVICE_INIT);
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        END_TIMER(DEVICE_INIT);

        START_TIMER(PROGRAM_CREATION);
        Program program = CreateProgram();

        constexpr CoreCoord core = {0, 0};

        // Tile configuration (n_tiles parsed from command-line arguments, default: 32)
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        const uint32_t bytes_per_element = use_bf16_mode ? sizeof(bfloat16) : sizeof(float);
        const uint32_t tile_size_bytes = bytes_per_element * elements_per_tile;
        const uint32_t dram_buffer_size = tile_size_bytes * n_tiles;

        fmt::print("Processing {} tiles ({} elements total)\n", n_tiles, n_tiles * elements_per_tile);

        // STEP 2: Create input/output circular buffers
        constexpr uint32_t tiles_per_cb = 2;
        constexpr auto cb_in = tt::CBIndex::c_0;
        constexpr auto cb_out = tt::CBIndex::c_16;

        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(
                tiles_per_cb * tile_size_bytes,
                {{cb_in, input_data_format}})
                .set_page_size(cb_in, tile_size_bytes));

        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(
                tiles_per_cb * tile_size_bytes,
                {{cb_out, input_data_format}})
                .set_page_size(cb_out, tile_size_bytes));
        END_TIMER(PROGRAM_CREATION);

        // STEP 3: Allocate DRAM buffers
        START_TIMER(BUFFER_ALLOCATION);
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = tile_size_bytes,
            .buffer_type = BufferType::DRAM
        };
        distributed::ReplicatedBufferConfig buffer_config{.size = dram_buffer_size};

        auto input_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto output_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        END_TIMER(BUFFER_ALLOCATION);

        // STEP 4: Create test input data using activation's test range
        START_TIMER(DATA_PREPARATION);
        fmt::print("Creating test input data (range [{}, {}])...\n", test_min, test_max);
        const size_t num_elements = elements_per_tile * n_tiles;
        const float test_range = test_max - test_min;

        std::vector<bfloat16> input_data_bf16;
        std::vector<float> input_data_fp32;

        if (use_bf16_mode) {
            input_data_bf16.resize(num_elements);
            for (size_t i = 0; i < num_elements; i++) {
                float val = test_min + test_range * (i / float(num_elements));
                input_data_bf16[i] = bfloat16(val);
            }
            END_TIMER(DATA_PREPARATION);
            START_TIMER(HOST_TO_DEVICE);
            distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_data_bf16, false);
            END_TIMER(HOST_TO_DEVICE);
        } else {
            input_data_fp32.resize(num_elements);
            for (size_t i = 0; i < num_elements; i++) {
                input_data_fp32[i] = test_min + test_range * (i / float(num_elements));
            }
            END_TIMER(DATA_PREPARATION);
            START_TIMER(HOST_TO_DEVICE);
            distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_data_fp32, false);
            END_TIMER(HOST_TO_DEVICE);
        }

        // STEP 5: Create kernels
        START_TIMER(KERNEL_CREATION);
        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*input_buffer->get_backing_buffer()).append_to(reader_compile_time_args);
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "generic_lut_activation/kernels/dataflow/reader.cpp",
            core,
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
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_compile_time_args
            });

        // Compute kernel - native SFPU
        // Add compiler defines for FP32 mode
        std::map<std::string, std::string> compute_defines;
        std::vector<UnpackToDestMode> unpack_to_dest_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

        if (!use_bf16_mode) {
            // FP32 mode: hardware handles FP32 via fp32_dest_acc_en + UnpackToDestFp32
            unpack_to_dest_modes[static_cast<uint32_t>(cb_in)] = UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_modes[static_cast<uint32_t>(cb_out)] = UnpackToDestMode::UnpackToDestFp32;
        }

        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "generic_lut_activation/kernels/compute/native_sfpu.cpp",
            core,
            ComputeConfig{
                .fp32_dest_acc_en = !use_bf16_mode,
                .unpack_to_dest_mode = unpack_to_dest_modes,
                .math_approx_mode = use_bf16_mode,
                .compile_args = {},
                .defines = compute_defines
            });

        // Set runtime arguments
        SetRuntimeArgs(program, reader, core, {input_buffer->address(), n_tiles});
        SetRuntimeArgs(program, writer, core, {output_buffer->address(), n_tiles});
        SetRuntimeArgs(program, compute, core, {n_tiles, activation_type, static_cast<uint32_t>(fast_approx)});
        END_TIMER(KERNEL_CREATION);

        fmt::print("✓ Kernels created and configured\n\n");

        // STEP 6: Execute program
        fmt::print("Executing program with native SFPU: {}\n", activation);
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        workload.add_program(device_range, std::move(program));

        START_TIMER(KERNEL_EXECUTION);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        END_TIMER(KERNEL_EXECUTION);

        fmt::print("✓ Execution complete\n\n");

        // STEP 7: Read and verify results
        START_TIMER(DEVICE_TO_HOST);
        std::vector<bfloat16> output_data_bf16;
        std::vector<float> output_data_fp32;

        if (use_bf16_mode) {
            distributed::EnqueueReadMeshBuffer(cq, output_data_bf16, output_buffer, true);
        } else {
            distributed::EnqueueReadMeshBuffer(cq, output_data_fp32, output_buffer, true);
        }
        END_TIMER(DEVICE_TO_HOST);

        // Read mesh device profiler results if enabled
        if (std::getenv("TT_METAL_DEVICE_PROFILER")) {
            ReadMeshDeviceProfilerResults(*mesh_device);
            fmt::print("✓ Device profiler results written to generated/profiler/\n");
        }

        // Helper lambdas for value reading
        auto get_input_value = [&](size_t i) -> float {
            return use_bf16_mode ? static_cast<float>(input_data_bf16[i]) : input_data_fp32[i];
        };

        auto get_output_value = [&](size_t i) -> float {
            return use_bf16_mode ? static_cast<float>(output_data_bf16[i]) : output_data_fp32[i];
        };

        const size_t output_size = use_bf16_mode ? output_data_bf16.size() : output_data_fp32.size();

        fmt::print("Results (samples across input range -10 to +10):\n");
        fmt::print("{}\n", std::string(60, '-'));
        fmt::print("{:>5} {:>12} {:>12}\n", "Index", "Input", "Output");
        fmt::print("{}\n", std::string(60, '-'));

        // Show samples from start, middle, and end to demonstrate full range
        const int samples[] = {0, 3276, 6553, 9830, 13107, 16384, 19660, 22937, 26214, 29491, 32767};
        for (auto i : samples) {
            if (static_cast<size_t>(i) < output_size) {
                fmt::print("{:5d} {:12.6f} {:12.6f}\n",
                          i, get_input_value(i), get_output_value(i));
            }
        }

        fmt::print("{}\n", std::string(60, '-'));
        fmt::print("\nProcessed {} tiles ({} elements total)\n", n_tiles, output_size);

        // OPTIONAL: Dump full input/output to CSV for plotting
        const char* dump_csv = std::getenv("DUMP_OUTPUT_CSV");
        if (dump_csv) {
            std::string csv_filename = dump_csv;
            std::ofstream csv_file(csv_filename);
            if (csv_file.is_open()) {
                csv_file << std::fixed << std::setprecision(10);
                csv_file << "input,output\n";
                for (size_t i = 0; i < output_size; ++i) {
                    csv_file << get_input_value(i) << "," << get_output_value(i) << "\n";
                }
                csv_file.close();
                fmt::print("\n✓ Output data dumped to: {}\n", csv_filename);
            }
        }

        // Close device
        if (!mesh_device->close()) {
            pass = false;
        }

        fmt::print("\n{}\n", std::string(60, '='));
        if (pass) {
            fmt::print("✓ Test PASSED\n");
        } else {
            fmt::print("✗ Test FAILED\n");
        }
        fmt::print("{}\n", std::string(60, '='));

    } catch (const std::exception& e) {
        fmt::print(stderr, "\n✗ Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());
        return 1;
    }

    return pass ? 0 : 1;
}
