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
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "lut_loader_rational.hpp"
#include "exhaustive_bf16_generator.hpp"

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

// Rational kernel path
#define COMPUTE_KERNEL_PATH OVERRIDE_KERNEL_PREFIX "generic_lut_activation/kernels/compute/piecewise_rational.cpp"

// =============================================================================
// ADHOC MODE: Read config from generated header (sweep scripts update this)
// =============================================================================
#ifdef ADHOC_MODE
#include "adhoc_rational_config.h"
[[maybe_unused]] constexpr uint32_t LUT_SIZE = ADHOC_LUT_SIZE;
constexpr uint32_t NUM_DEGREE = ADHOC_NUM_DEGREE;
constexpr uint32_t DEN_DEGREE = ADHOC_DEN_DEGREE;
constexpr uint32_t NUM_SEGMENTS = ADHOC_NUM_SEGMENTS;
#else
// LUT size must be compile-time constant for template instantiation
#ifndef LUT_SIZE
constexpr uint32_t LUT_SIZE = 1024;
#endif

// Rational degrees and number of segments
#ifndef NUM_DEGREE
[[maybe_unused]] constexpr uint32_t NUM_DEGREE = 4;  // Default numerator degree
#endif

#ifndef DEN_DEGREE
[[maybe_unused]] constexpr uint32_t DEN_DEGREE = 4;  // Default denominator degree
#endif

#ifndef NUM_SEGMENTS
[[maybe_unused]] constexpr uint32_t NUM_SEGMENTS = 2;  // Default segments
#endif
#endif  // ADHOC_MODE

int main(int argc, char** argv) {
    if (argc < 2) {
        fmt::print(stderr, "Usage: {} <rational_csv_file> --activation <name> --precision <bf16|fp32> --range-min <min> --range-max <max> [--tiles N]\n", argv[0]);
        fmt::print(stderr, "\nRequired arguments:\n");
        fmt::print(stderr, "  <rational_csv_file>      Path to rational coefficient CSV file\n");
        fmt::print(stderr, "  --activation, -a <name>  Activation function name (e.g., sigmoid, erf, atanh)\n");
        fmt::print(stderr, "  --precision <type>       Precision mode: bf16 or fp32\n");
        fmt::print(stderr, "  --range-min, -rmin <val> Minimum test range value\n");
        fmt::print(stderr, "  --range-max, -rmax <val> Maximum test range value\n");
        fmt::print(stderr, "\nOptional arguments:\n");
        fmt::print(stderr, "  --tiles, -t N            Number of tiles to process (default: 32, range: 1-1024)\n");
        fmt::print(stderr, "  --compute-loops, -cl N   Compute amplification factor (default: 1, range: 1-10000)\n");
        fmt::print(stderr, "\nExamples:\n");
        fmt::print(stderr, "  {} sigmoid_fp32_2_4_4_uniform_rational.csv --activation sigmoid --precision fp32 --range-min -10 --range-max 10\n", argv[0]);
        fmt::print(stderr, "  {} erf_fp32_1_9_8_uniform_rational.csv -a erf --precision fp32 -rmin -5 -rmax 5 --tiles 256\n", argv[0]);
        return 1;
    }

    std::string csv_filename = argv[1];
    std::string activation_name;
    std::string precision_arg;
    float range_min = 0.0f;
    float range_max = 0.0f;
    uint32_t n_tiles = 32;  // Default value
    uint32_t compute_loop_factor = 1;  // Default: no amplification

    bool has_activation = false;
    bool has_precision = false;
    bool has_range_min = false;
    bool has_range_max = false;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--activation" || arg == "-a") {
            if (i + 1 >= argc) {
                fmt::print(stderr, "Error: --activation requires an argument\n");
                return 1;
            }
            activation_name = argv[++i];
            has_activation = true;
        } else if (arg == "--range-min" || arg == "-rmin") {
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
            precision_arg = argv[++i];
            if (precision_arg != "bf16" && precision_arg != "fp32") {
                fmt::print(stderr, "Error: --precision must be 'bf16' or 'fp32' (got '{}')\n", precision_arg);
                return 1;
            }
            has_precision = true;
        } else if (arg == "--compute-loops" || arg == "-cl") {
            if (i + 1 >= argc) {
                fmt::print(stderr, "Error: --compute-loops requires an argument\n");
                return 1;
            }
            try {
                compute_loop_factor = std::stoi(argv[++i]);
                if (compute_loop_factor == 0 || compute_loop_factor > 10000) {
                    fmt::print(stderr, "Error: compute-loops must be between 1 and 10000 (got {})\n", compute_loop_factor);
                    return 1;
                }
            } catch (const std::exception& e) {
                fmt::print(stderr, "Error: invalid compute-loops value '{}'\n", argv[i]);
                return 1;
            }
        } else {
            fmt::print(stderr, "Error: unknown argument '{}'\n", arg);
            return 1;
        }
    }

    // Check required arguments
    if (!has_activation) {
        fmt::print(stderr, "Error: --activation is required\n");
        return 1;
    }
    if (!has_precision) {
        fmt::print(stderr, "Error: --precision is required\n");
        return 1;
    }
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

    bool pass = true;

    try {
        fmt::print("{}\n", std::string(60, '='));
        fmt::print("Generic Rational Approximation LUT Example\n");
        fmt::print("{}\n", std::string(60, '='));

        // STEP 1: Use precision from command-line argument
        bool use_bf16_mode = (precision_arg == "bf16");
        const auto input_data_format = use_bf16_mode ? tt::DataFormat::Float16_b : tt::DataFormat::Float32;

        fmt::print("Loading rational coefficients from: {}\n", csv_filename);
        fmt::print("Precision: {}\n", use_bf16_mode ? "BF16" : "FP32");

        // Detect range reduction method from CSV metadata
        std::string range_reduction_method = RationalLUTLoader::extract_range_reduction_method(csv_filename);
        if (!range_reduction_method.empty()) {
            fmt::print("Detected range reduction: {}\n", range_reduction_method);
        }

        // STEP 2: Load rational LUT from CSV file
        RationalLUTLoader::RationalLUTInfo lut_info = RationalLUTLoader::load(csv_filename);
        fmt::print("✓ Loaded rational LUT:\n");
        fmt::print("  Segments: {}\n", lut_info.num_segments);
        fmt::print("  Numerator degree: {}\n", lut_info.num_degree);
        fmt::print("  Denominator degree: {}\n", lut_info.den_degree);
        fmt::print("  Total LUT size: {} floats\n", lut_info.lut_size);

        // Verify LUT size matches compile-time constant
        uint32_t expected_lut_size = (lut_info.num_segments + 1) +
                                     lut_info.num_segments * (lut_info.num_degree + lut_info.den_degree + 2);
        if (lut_info.lut_size != expected_lut_size) {
            fmt::print("ERROR: LUT size mismatch (got {}, expected {})\n",
                      lut_info.lut_size, expected_lut_size);
            return 1;
        }

        // Verify compile-time parameters match CSV
        if (NUM_DEGREE != 0 && lut_info.num_degree != NUM_DEGREE) {
            fmt::print("ERROR: Numerator degree mismatch (CSV: {}, compile-time: {})\n",
                      lut_info.num_degree, NUM_DEGREE);
            return 1;
        }
        if (DEN_DEGREE != 0 && lut_info.den_degree != DEN_DEGREE) {
            fmt::print("ERROR: Denominator degree mismatch (CSV: {}, compile-time: {})\n",
                      lut_info.den_degree, DEN_DEGREE);
            return 1;
        }
        if (NUM_SEGMENTS != 0 && lut_info.num_segments != NUM_SEGMENTS) {
            fmt::print("ERROR: Segments mismatch (CSV: {}, compile-time: {})\n",
                      lut_info.num_segments, NUM_SEGMENTS);
            return 1;
        }

        // DEBUG: Print LUT data
        fmt::print("\nDEBUG Rational LUT Data:\n");
        fmt::print("Boundaries ({}):\n", lut_info.num_segments + 1);
        for (uint32_t i = 0; i <= lut_info.num_segments; i++) {
            fmt::print("  b{}: {:.6f}\n", i, lut_info.lut_data[i]);
        }

        fmt::print("\nCoefficients ({} segments):\n", lut_info.num_segments);
        uint32_t coeff_offset = lut_info.num_segments + 1;
        uint32_t coeffs_per_seg = lut_info.num_degree + lut_info.den_degree + 2;
        for (uint32_t seg = 0; seg < lut_info.num_segments; seg++) {
            fmt::print("  Segment {}:\n", seg);
            uint32_t base = coeff_offset + seg * coeffs_per_seg;
            fmt::print("    Numerator (degree {}): ", lut_info.num_degree);
            for (uint32_t i = 0; i <= lut_info.num_degree; i++) {
                fmt::print("{:.6e}{}", lut_info.lut_data[base + i],
                          i < lut_info.num_degree ? ", " : "\n");
            }
            fmt::print("    Denominator (degree {}): ", lut_info.den_degree);
            for (uint32_t i = 0; i <= lut_info.den_degree; i++) {
                fmt::print("{:.6e}{}", lut_info.lut_data[base + lut_info.num_degree + 1 + i],
                          i < lut_info.den_degree ? ", " : "\n");
            }
        }

        // Use combined LUT format directly from loader:
        // [b0, b1, ..., bN][n0_seg0, n1_seg0, ..., d0_seg0, d1_seg0, ...][n0_seg1, ...]
        // The kernel evaluates: result = P(x) / Q(x) where P and Q are polynomials
        const std::vector<float>& lut_combined = lut_info.lut_data;

        fmt::print("\n✓ Using combined rational LUT: size={}\n", lut_combined.size());

        // DEBUG: Print combined LUT contents for verification
        fmt::print("\n=== COMBINED RATIONAL LUT ===\n");
        for (size_t i = 0; i < lut_combined.size(); i++) {
            fmt::print("  lut[{}] = {:.6e}\n", i, lut_combined[i]);
        }

        RationalLUTLoader::print_stats(lut_info);
        fmt::print("\n");

        // Use test range from command-line arguments
        float test_min = range_min;
        float test_max = range_max;
        fmt::print("Activation: {} | Test range: [{}, {}]\n", activation_name, test_min, test_max);

        if (compute_loop_factor > 1) {
            fmt::print("⚡ Compute amplification enabled: {}× (recompute each tile {} times)\n",
                       compute_loop_factor, compute_loop_factor);
        }

        // STEP 3: Create device and program
        START_TIMER(DEVICE_INIT);
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        END_TIMER(DEVICE_INIT);

        START_TIMER(PROGRAM_CREATION);
        Program program = CreateProgram();

        constexpr CoreCoord core = {0, 0};

        // Tile configuration
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        const uint32_t bytes_per_element = use_bf16_mode ? sizeof(bfloat16) : sizeof(float);
        const uint32_t tile_size_bytes = bytes_per_element * elements_per_tile;
        const uint32_t dram_buffer_size = tile_size_bytes * n_tiles;

        fmt::print("Processing {} tiles ({} elements total, {} bytes per element)\n",
                   n_tiles, n_tiles * elements_per_tile, bytes_per_element);

        // STEP 4: Create combined rational LUT buffer
        // LUT format: [boundaries][num_coeffs, den_coeffs per segment]
        constexpr auto cb_lut = tt::CBIndex::c_25;

        const uint32_t lut_size = lut_combined.size();
        const uint32_t lut_size_bytes = lut_size * sizeof(float);
        const auto lut_cb_data_format = tt::DataFormat::Float32;

        fmt::print("\n=== RATIONAL LUT ({} entries, {} bytes) ===\n", lut_size, lut_size_bytes);

        // Create L1 buffer for combined LUT
        ShardSpecBuffer shard_spec(
            CoreRangeSet(CoreRange(core, core)),
            {1, lut_size},
            ShardOrientation::ROW_MAJOR,
            {1, lut_size},
            {1, lut_size}
        );
        BufferShardingArgs sharding_args(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED);
        distributed::DeviceLocalBufferConfig lut_local_config{
            .page_size = lut_size_bytes,
            .buffer_type = BufferType::L1,
            .sharding_args = sharding_args
        };
        distributed::ReplicatedBufferConfig lut_buffer_config{.size = lut_size_bytes};
        auto lut_buffer = distributed::MeshBuffer::create(lut_buffer_config, lut_local_config, mesh_device.get());
        distributed::EnqueueWriteMeshBuffer(cq, lut_buffer, lut_combined, false);

        // STEP 5: Create circular buffer for rational LUT
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(lut_size_bytes, {{cb_lut, lut_cb_data_format}})
                .set_page_size(cb_lut, lut_size_bytes)
                .set_globally_allocated_address(*lut_buffer->get_backing_buffer())
        );

        fmt::print("✓ Rational LUT loaded into L1 memory (CB {}, {} bytes)\n",
                   static_cast<int>(cb_lut), lut_size_bytes);

        // STEP 6: Create input/output circular buffers
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

        // STEP 7: Allocate DRAM buffers
        START_TIMER(BUFFER_ALLOCATION);
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = tile_size_bytes,
            .buffer_type = BufferType::DRAM
        };
        distributed::ReplicatedBufferConfig buffer_config{.size = dram_buffer_size};

        auto input_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        auto output_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
        END_TIMER(BUFFER_ALLOCATION);

        // STEP 8: Create test input data
        START_TIMER(DATA_PREPARATION);
        fmt::print("Creating test input data (range [{}, {}])...\n", test_min, test_max);

        const size_t num_elements = elements_per_tile * n_tiles;

        std::vector<bfloat16> input_data_bf16;
        std::vector<float> input_data_fp32;

        if (use_bf16_mode) {
            // Generate EXHAUSTIVE BF16 inputs (all possible BF16 values in range, excluding subnormals)
            // This ensures we test worst-case bit patterns, not just linearly-spaced values
            fmt::print("Generating exhaustive BF16 inputs (excluding subnormals)...\n");

            input_data_bf16.resize(num_elements);
            size_t unique_count = fill_buffer_with_exhaustive_bf16(
                input_data_bf16.data(), num_elements, test_min, test_max);

            fmt::print("  Found {} unique BF16 values in range [{}, {}]\n",
                      unique_count, test_min, test_max);
            fmt::print("  Generated {} total elements ({} tiles, repeating exhaustive set)\n",
                      num_elements, n_tiles);

            END_TIMER(DATA_PREPARATION);

            START_TIMER(HOST_TO_DEVICE);
            distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_data_bf16, false);
            END_TIMER(HOST_TO_DEVICE);
        } else {
            // FP32: Use linear spacing (exhaustive FP32 not feasible - 2^32 values)
            fmt::print("Generating linearly-spaced FP32 inputs...\n");
            const float test_range = test_max - test_min;
            input_data_fp32.resize(num_elements);
            for (size_t i = 0; i < num_elements; i++) {
                input_data_fp32[i] = test_min + test_range * (i / float(num_elements));
            }
            END_TIMER(DATA_PREPARATION);

            START_TIMER(HOST_TO_DEVICE);
            distributed::EnqueueWriteMeshBuffer(cq, input_buffer, input_data_fp32, false);
            END_TIMER(HOST_TO_DEVICE);
        }

        // STEP 9: Create kernels
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

        // Compute kernel with compile-time args for rational approximation
        // Pass: LUT_SIZE, NUM_DEGREE (poly_degree), DEN_DEGREE, NUM_SEGMENTS
        // Matches polynomial kernel: {lut_size, poly_degree, num_segments} but with extra den_degree
        std::vector<uint32_t> compute_compile_args = {
            lut_size,  // Same as polynomial: actual LUT size (4 for linear/1-segment)
            lut_info.num_degree,
            lut_info.den_degree,
            lut_info.num_segments
        };

        std::map<std::string, std::string> compute_defines;
        std::vector<UnpackToDestMode> unpack_to_dest_modes(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

        if (!use_bf16_mode) {
            // FP32 mode: hardware handles FP32 via fp32_dest_acc_en + UnpackToDestFp32
            constexpr auto cb_in = tt::CBIndex::c_0;
            constexpr auto cb_out = tt::CBIndex::c_16;
            unpack_to_dest_modes[static_cast<uint32_t>(cb_in)] = UnpackToDestMode::UnpackToDestFp32;
            unpack_to_dest_modes[static_cast<uint32_t>(cb_out)] = UnpackToDestMode::UnpackToDestFp32;
        }

        // Set range reduction compile defines based on CSV metadata
        if (range_reduction_method == "exp") {
            compute_defines["RANGE_REDUCTION_EXP"] = "1";
        } else if (range_reduction_method == "trig") {
            compute_defines["RANGE_REDUCTION_TRIG"] = "1";
        } else if (range_reduction_method == "tan") {
            compute_defines["RANGE_REDUCTION_TAN"] = "1";
        } else if (range_reduction_method == "cbrt") {
            compute_defines["RANGE_REDUCTION_CBRT"] = "1";
        }

        fmt::print("✓ Creating rational kernel: NUM_DEGREE={}, DEN_DEGREE={}, NUM_SEGMENTS={}\n",
                   lut_info.num_degree, lut_info.den_degree, lut_info.num_segments);
        fmt::print("  Kernel path: {}\n", COMPUTE_KERNEL_PATH);
        fmt::print("  Compile args: lut_size={}, num_degree={}, den_degree={}, num_segments={}\n",
                   lut_size, lut_info.num_degree, lut_info.den_degree, lut_info.num_segments);

        auto compute = CreateKernel(
            program,
            COMPUTE_KERNEL_PATH,
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = !use_bf16_mode,
                .unpack_to_dest_mode = unpack_to_dest_modes,
                .math_approx_mode = false,
                .compile_args = compute_compile_args,
                .defines = compute_defines
            });

        END_TIMER(KERNEL_CREATION);

        // STEP 10: Set runtime arguments
        SetRuntimeArgs(program, reader, core, {input_buffer->address(), n_tiles});
        SetRuntimeArgs(program, writer, core, {output_buffer->address(), n_tiles});
        SetRuntimeArgs(program, compute, core, {n_tiles, compute_loop_factor});

        // STEP 11: Execute program
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        workload.add_program(device_range, std::move(program));

        START_TIMER(KERNEL_EXECUTION);
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        END_TIMER(KERNEL_EXECUTION);

        fmt::print("✓ Kernel execution completed\n\n");

        // STEP 12: Read results from device
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

        // Helper lambda to get output value at index
        auto get_output_value = [&](size_t i) -> float {
            return use_bf16_mode ? static_cast<float>(output_data_bf16[i]) : output_data_fp32[i];
        };

        // STEP 13: Print sample results
        fmt::print("Sample Results (first 5 and last 5 elements):\n");
        fmt::print("  {:>15} {:>15}\n", "Input", "Output");
        for (size_t i = 0; i < std::min(size_t(5), num_elements); i++) {
            float input_val = use_bf16_mode ? static_cast<float>(input_data_bf16[i]) : input_data_fp32[i];
            float output_val = get_output_value(i);
            fmt::print("  {:15.6f} {:15.6f}\n", input_val, output_val);
        }
        if (num_elements > 10) {
            fmt::print("  ...\n");
            for (size_t i = num_elements - 5; i < num_elements; i++) {
                float input_val = use_bf16_mode ? static_cast<float>(input_data_bf16[i]) : input_data_fp32[i];
                float output_val = get_output_value(i);
                fmt::print("  {:15.6f} {:15.6f}\n", input_val, output_val);
            }
        }

        // STEP 14: Dump output CSV if requested
        const char* dump_csv = std::getenv("DUMP_OUTPUT_CSV");
        if (dump_csv) {
            std::string csv_filename = dump_csv;
            std::ofstream csv_file(csv_filename);
            if (csv_file.is_open()) {
                csv_file << std::fixed << std::setprecision(10);
                csv_file << "input,output\n";
                for (size_t i = 0; i < num_elements; ++i) {
                    float input_val = use_bf16_mode ? static_cast<float>(input_data_bf16[i]) : input_data_fp32[i];
                    float output_val = get_output_value(i);
                    csv_file << input_val << "," << output_val << "\n";
                }
                csv_file.close();
                fmt::print("\n✓ Output data dumped to: {}\n", csv_filename);
            }
        }

        fmt::print("\n{}\n", std::string(60, '='));
        fmt::print("Test PASSED\n");
        fmt::print("{}\n", std::string(60, '='));

    } catch (const std::exception& e) {
        pass = false;
        fmt::print(stderr, "\nTest FAILED with exception:\n{}\n", e.what());
    }

    return pass ? 0 : 1;
}
