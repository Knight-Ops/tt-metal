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
#include <tt-metalium/work_split.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include "lut_loader.hpp"
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

#ifdef KERNEL_VARIANT
#define COMPUTE_KERNEL_PATH OVERRIDE_KERNEL_PREFIX "generic_lut_activation/kernels/compute/" KERNEL_VARIANT ".cpp"
#else
#define COMPUTE_KERNEL_PATH OVERRIDE_KERNEL_PREFIX "generic_lut_activation/kernels/compute/generic_lut_activation.cpp"
#endif

// =============================================================================
// ADHOC MODE: Read config from generated header (sweep scripts update this)
// =============================================================================
#ifdef ADHOC_MODE
#include "adhoc_config.h"
constexpr uint32_t LUT_SIZE = ADHOC_LUT_SIZE;
constexpr uint32_t POLY_DEGREE = ADHOC_POLY_DEGREE;
constexpr uint32_t NUM_SEGMENTS = ADHOC_NUM_SEGMENTS;
#else
// LUT size must be compile-time constant for template instantiation
#ifndef LUT_SIZE
constexpr uint32_t LUT_SIZE = 1024;
#endif

// Polynomial degree and number of segments for piecewise_generic.cpp
// These are optional - only used when KERNEL_VARIANT="piecewise_generic"
#ifndef POLY_DEGREE
[[maybe_unused]] constexpr uint32_t POLY_DEGREE = 0;  // 0 means not specified
#endif

#ifndef NUM_SEGMENTS
[[maybe_unused]] constexpr uint32_t NUM_SEGMENTS = 0;  // 0 means not specified
#endif
#endif  // ADHOC_MODE

int main(int argc, char** argv) {
    if (argc < 2) {
        fmt::print(stderr, "Usage: {} <csv_file> --activation <name> --precision <bf16|fp32|both> --range-min <min> --range-max <max> [--tiles N] [--compute-loops N]\n", argv[0]);
        fmt::print(stderr, "\nRequired arguments:\n");
        fmt::print(stderr, "  <csv_file>               Path to coefficient CSV file\n");
        fmt::print(stderr, "  --activation, -a <name>  Activation function name (e.g., sigmoid, gelu, tanh)\n");
        fmt::print(stderr, "  --precision <type>       Precision mode: bf16, fp32, or both\n");
        fmt::print(stderr, "  --range-min, -rmin <val> Minimum test range value\n");
        fmt::print(stderr, "  --range-max, -rmax <val> Maximum test range value\n");
        fmt::print(stderr, "\nOptional arguments:\n");
        fmt::print(stderr, "  --tiles, -t N            Number of tiles to process (default: 32, range: 1-100000)\n");
        fmt::print(stderr, "  --batch-tiles 1,8,256    Comma-separated tile counts (runs all in one device session)\n");
        fmt::print(stderr, "  --compute-loops, -cl N   Compute amplification factor (default: 1, recompute N times per tile)\n");
        fmt::print(stderr, "  --staggered              Force use of staggered Horner evaluation (interleaved coeff-select + Horner step)\n");
        fmt::print(stderr, "\nExamples:\n");
        fmt::print(stderr, "  {} sigmoid_bf16_32_3_errordriven_any.csv --activation sigmoid --precision bf16 --range-min -10 --range-max 10\n", argv[0]);
        fmt::print(stderr, "  {} sigmoid_bf16_32_3_errordriven_any.csv -a sigmoid --precision fp32 -rmin -10 -rmax 10 --tiles 256 --compute-loops 100\n", argv[0]);
        return 1;
    }

    std::string csv_filename = argv[1];
    std::string activation_name;
    std::string precision_arg;
    float range_min = 0.0f;
    float range_max = 0.0f;
    uint32_t n_tiles = 32;  // Default value
    std::vector<uint32_t> batch_tiles;
    uint32_t compute_loop_factor = 1;  // Default: no amplification
    bool use_staggered_horner = false;  // Default: use standard dispatch logic

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
                if (n_tiles == 0 || n_tiles > 100000) {
                    fmt::print(stderr, "Error: tiles must be between 1 and 100000 (got {})\n", n_tiles);
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
            if (precision_arg != "bf16" && precision_arg != "fp32" && precision_arg != "both") {
                fmt::print(stderr, "Error: --precision must be 'bf16', 'fp32', or 'both' (got '{}')\n", precision_arg);
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
        } else if (arg == "--batch-tiles") {
            if (i + 1 >= argc) {
                fmt::print(stderr, "Error: --batch-tiles requires a comma-separated list of tile counts\n");
                return 1;
            }
            std::string tiles_str = argv[++i];
            std::istringstream ss(tiles_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    uint32_t t = std::stoi(token);
                    if (t == 0 || t > 100000) {
                        fmt::print(stderr, "Error: tile count must be between 1 and 100000 (got {})\n", t);
                        return 1;
                    }
                    batch_tiles.push_back(t);
                } catch (const std::exception& e) {
                    fmt::print(stderr, "Error: invalid tile count '{}' in --batch-tiles\n", token);
                    return 1;
                }
            }
        } else if (arg == "--staggered") {
            use_staggered_horner = true;
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

    // If no --batch-tiles, use single n_tiles value (backward compat)
    if (batch_tiles.empty()) {
        batch_tiles.push_back(n_tiles);
    }
    bool is_batch_mode = (batch_tiles.size() > 1);

    // Build list of precisions to iterate over
    std::vector<std::string> precisions;
    if (precision_arg == "both") {
        precisions = {"bf16", "fp32"};
    } else {
        precisions = {precision_arg};
    }
    bool is_multi_precision = (precisions.size() > 1);

    // Capture DUMP_OUTPUT_CSV base path before the loop
    std::string dump_csv_base;
    const char* dump_csv_env = std::getenv("DUMP_OUTPUT_CSV");
    if (dump_csv_env) {
        dump_csv_base = dump_csv_env;
    }

    bool pass = true;

    try {
        fmt::print("{}\n", std::string(60, '='));
        fmt::print("Generic LUT Activation Function Example\n");
        fmt::print("{}\n", std::string(60, '='));

        fmt::print("Loading coefficients from: {}\n", csv_filename);
        fmt::print("Precision: {}\n", precision_arg);

        // Detect range reduction method from CSV metadata
        std::string range_reduction_method = LUTLoader::extract_range_reduction_method(csv_filename);
        if (!range_reduction_method.empty()) {
            fmt::print("Detected range reduction: {}\n", range_reduction_method);
        }

        // STEP 2: Load LUT from coefficient CSV file
        std::vector<float> lut_data;
#ifdef USE_DOUBLE_FLOAT
        // Double-float mode: load DD coefficient pairs (hi, lo)
        lut_data = LUTLoader::load_dd(csv_filename);
        fmt::print("✓ Loaded {} entries from CSV (double-float mode: 2× coefficients)\n", lut_data.size());
#else
        // Standard mode: single-precision coefficients
        lut_data = LUTLoader::load(csv_filename);
        fmt::print("✓ Loaded {} entries from CSV\n", lut_data.size());
#endif

        // Pad LUT to LUT_SIZE if needed (for kernels that expect fixed size)
        if (lut_data.size() < LUT_SIZE) {
            size_t original_size = lut_data.size();
            lut_data.resize(LUT_SIZE, 0.0f);  // Pad with zeros
            fmt::print("✓ Padded LUT from {} to {} entries\n", original_size, LUT_SIZE);
        } else if (lut_data.size() > LUT_SIZE) {
            fmt::print("ERROR: LUT file has {} entries but LUT_SIZE={}\n", lut_data.size(), LUT_SIZE);
            return 1;
        }

        fmt::print("✓ Successfully loaded LUT with {} entries\n", lut_data.size());
        LUTLoader::print_stats(lut_data);
        fmt::print("\n");

        // Use test range from command-line arguments
        float test_min = range_min;
        float test_max = range_max;
        fmt::print("Activation: {} | Test range: [{}, {}]\n", activation_name, test_min, test_max);

        // STEP 3: Create device and program
        START_TIMER(DEVICE_INIT);
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        END_TIMER(DEVICE_INIT);

        // ===== PRECISION LOOP × BATCH LOOP =====
        for (const auto& current_precision : precisions) {
            bool use_bf16_mode = (current_precision == "bf16");
            const auto input_data_format = use_bf16_mode ? tt::DataFormat::Float16_b : tt::DataFormat::Float32;

            if (is_multi_precision) {
                fmt::print("\n===== Precision: {} ({}) =====\n", current_precision, use_bf16_mode ? "bfloat16" : "float32");
            }

        for (uint32_t current_tiles : batch_tiles) {
            // Build batch prefix for timing extraction
            // IMPORTANT: single-precision batch must keep "BATCH[tiles=N]:" format
            // for backward compat with sweep_helpers.sh extract_shape_timing()
            std::string batch_prefix;
            if (is_multi_precision) {
                batch_prefix = fmt::format("BATCH[{},tiles={}]:", current_precision, current_tiles);
            } else if (is_batch_mode) {
                batch_prefix = fmt::format("BATCH[tiles={}]:", current_tiles);
            }

            if (is_batch_mode || is_multi_precision) {
                fmt::print("\n{}BEGIN\n", batch_prefix);
            }

        START_TIMER(PROGRAM_CREATION);
        Program program = CreateProgram();

        // Tile configuration
        constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
        // Compute tile size based on detected precision: BF16 = 2 bytes, FP32 = 4 bytes
        const uint32_t bytes_per_element = use_bf16_mode ? sizeof(bfloat16) : sizeof(float);
        const uint32_t tile_size_bytes = bytes_per_element * elements_per_tile;
        const uint32_t dram_buffer_size = tile_size_bytes * current_tiles;

        fmt::print("Processing {} tiles ({} elements total, {} bytes per element)\n",
                   current_tiles, current_tiles * elements_per_tile, bytes_per_element);

        // MULTI-CORE: Get grid size and split work across cores
        CoreCoord grid_size = mesh_device->compute_with_storage_grid_size();
        fmt::print("Grid size: {}x{}\n", grid_size.x, grid_size.y);

        auto [num_cores, all_cores, core_group_1, core_group_2,
              tiles_per_core_1, tiles_per_core_2] =
            split_work_to_cores(grid_size, current_tiles);

        fmt::print("Work split: {} cores total\n", num_cores);
        fmt::print("Group 1: {} cores, {} tiles/core\n",
                   core_group_1.num_cores(), tiles_per_core_1);
        if (!core_group_2.ranges().empty()) {
            fmt::print("Group 2: {} cores, {} tiles/core\n",
                       core_group_2.num_cores(), tiles_per_core_2);
        }

        // Verify work distribution
        uint32_t total_tiles = core_group_1.num_cores() * tiles_per_core_1;
        if (!core_group_2.ranges().empty()) {
            total_tiles += core_group_2.num_cores() * tiles_per_core_2;
        }
        TT_FATAL(total_tiles == current_tiles,
                 "Work split mismatch! {} cores × tiles != {} tiles total",
                 total_tiles, current_tiles);

        // STEP 4 & 5: Create LUT buffer and CB for EACH core individually
        constexpr auto cb_lut = tt::CBIndex::c_25;
        const uint32_t lut_size_bytes = LUT_SIZE * sizeof(float);
        const auto lut_cb_data_format = tt::DataFormat::Float32;

        // Create per-core LUT buffers and CBs
        uint32_t num_cores_y = grid_size.y;
        for (uint32_t i = 0; i < num_cores; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            // Create L1 buffer for this core's LUT
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

            // Write LUT data to this core's L1
            distributed::EnqueueWriteMeshBuffer(cq, core_lut_buffer, lut_data, false);

            // Create CB for this core, pointing to its L1 buffer
            CreateCircularBuffer(
                program,
                core,
                CircularBufferConfig(lut_size_bytes, {{cb_lut, lut_cb_data_format}})
                    .set_page_size(cb_lut, lut_size_bytes)
                    .set_globally_allocated_address(*core_lut_buffer->get_backing_buffer())
            );
        }

        fmt::print("✓ LUT loaded into L1 memory on {} cores (CB index {})\n", num_cores, static_cast<int>(cb_lut));

        // STEP 6: Create input/output circular buffers on ALL cores
        constexpr uint32_t tiles_per_cb = 2;
        constexpr auto cb_in = tt::CBIndex::c_0;
        constexpr auto cb_out = tt::CBIndex::c_16;

        CreateCircularBuffer(
            program,
            all_cores,  // Create on ALL cores
            CircularBufferConfig(
                tiles_per_cb * tile_size_bytes,
                {{cb_in, input_data_format}})
                .set_page_size(cb_in, tile_size_bytes));

        CreateCircularBuffer(
            program,
            all_cores,  // Create on ALL cores
            CircularBufferConfig(
                tiles_per_cb * tile_size_bytes,
                {{cb_out, input_data_format}})
                .set_page_size(cb_out, tile_size_bytes));

        // FPU variant: create scratch CBs for coefficient matrix, power matrix, result matrix.
        // These are always BF16 regardless of the host precision mode.
#ifdef KERNEL_VARIANT
        if (std::string(KERNEL_VARIANT) == "piecewise_riscv") {
            constexpr uint32_t bf16_tile_size = elements_per_tile * sizeof(uint16_t);
            // CB_SCRATCH_C (c_26): coefficient matrix [num_segs × K], 1 tile, never popped
            CreateCircularBuffer(program, all_cores,
                CircularBufferConfig(1 * bf16_tile_size, {{tt::CBIndex::c_26, tt::DataFormat::Float16_b}})
                    .set_page_size(tt::CBIndex::c_26, bf16_tile_size));
            // CB_SCRATCH_P (c_27): power matrix, capacity=2 for double-buffering
            CreateCircularBuffer(program, all_cores,
                CircularBufferConfig(2 * bf16_tile_size, {{tt::CBIndex::c_27, tt::DataFormat::Float16_b}})
                    .set_page_size(tt::CBIndex::c_27, bf16_tile_size));
            // CB_SCRATCH_Y (c_28): matmul result per row, capacity=1
            CreateCircularBuffer(program, all_cores,
                CircularBufferConfig(1 * bf16_tile_size, {{tt::CBIndex::c_28, tt::DataFormat::Float16_b}})
                    .set_page_size(tt::CBIndex::c_28, bf16_tile_size));
            fmt::print("✓ Created FPU scratch CBs (c_26 x1, c_27 x2, c_28 x1, {} bytes/tile)\n", bf16_tile_size);
        }
#endif
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

        // STEP 8: Create test input data using activation's test range
        START_TIMER(DATA_PREPARATION);
        fmt::print("Creating test input data (range [{}, {}])...\n", test_min, test_max);

        const size_t num_elements = elements_per_tile * current_tiles;

        // Use appropriate data type based on precision mode
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
                      num_elements, current_tiles);

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

        // Helper lambda to get input value at index (works for both BF16 and FP32)
        auto get_input_value = [&](size_t i) -> float {
            return use_bf16_mode ? static_cast<float>(input_data_bf16[i]) : input_data_fp32[i];
        };

        // STEP 9: Create kernels following TTNN's pattern
        START_TIMER(KERNEL_CREATION);

        // Create reader kernel on ALL cores
        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*input_buffer->get_backing_buffer()).append_to(reader_compile_time_args);
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "generic_lut_activation/kernels/dataflow/reader.cpp",
            all_cores,  // Create on ALL cores
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_compile_time_args
            });

        // Create writer kernel on ALL cores
        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*output_buffer->get_backing_buffer()).append_to(writer_compile_time_args);
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "generic_lut_activation/kernels/dataflow/writer.cpp",
            all_cores,  // Create on ALL cores
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_compile_time_args
            });

        fmt::print("✓ Created reader/writer kernels on {} cores\n", num_cores);

        // Compute kernel with compile-time args
        // Always pass LUT_SIZE - kernels will derive NUM_SEGMENTS based on their format:
        // - piecewise_linear: NUM_SEGMENTS = LUT_SIZE / 2 (2 values per segment)
        // - piecewise_quadratic: NUM_SEGMENTS = LUT_SIZE / 3 (3 values per segment)
        // - piecewise_cubic: NUM_SEGMENTS = LUT_SIZE / 4 (4 values per segment)
        // - piecewise_generic: uses explicit POLY_DEGREE and NUM_SEGMENTS
        std::vector<uint32_t> compute_compile_args = {LUT_SIZE};

        // Define PACKER_L1_ACC for FP32 mode to enable FP32 data format
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

        // Set staggered Horner flag if requested
        if (use_staggered_horner) {
            compute_defines["USE_STAGGERED_HORNER"] = "1";
            fmt::print("✓ Staggered Horner evaluation enabled\n");
        }

        #ifdef KERNEL_VARIANT
        std::string kernel_variant = KERNEL_VARIANT;

        // piecewise_generic, opt, and fpu variants require POLY_DEGREE and NUM_SEGMENTS as compile-time args
        if (kernel_variant == "piecewise_generic" ||
            kernel_variant == "piecewise_generic_opt_v2" ||
            kernel_variant == "piecewise_riscv") {
            if (POLY_DEGREE == 0 || NUM_SEGMENTS == 0) {
                fmt::print(stderr, "Error: {} requires POLY_DEGREE and NUM_SEGMENTS to be defined\n", kernel_variant);
                return 1;
            }
            compute_compile_args.push_back(POLY_DEGREE);
            compute_compile_args.push_back(NUM_SEGMENTS);
            fmt::print("✓ Using kernel variant '{}' with LUT_SIZE={}, POLY_DEGREE={}, NUM_SEGMENTS={}\n",
                       kernel_variant, LUT_SIZE, POLY_DEGREE, NUM_SEGMENTS);
        }
        // Estrin kernel requires precision mode as second compile-time arg
        else if (kernel_variant == "piecewise_quadratic_estrin") {
            // PRECISION_MODE: 0=BF16, 2=FP32
            uint32_t precision_mode = use_bf16_mode ? 0 : 2;
            compute_compile_args.push_back(precision_mode);
            fmt::print("✓ Using kernel variant '{}' with LUT_SIZE={}\n", kernel_variant, LUT_SIZE);
        }
        else {
            fmt::print("✓ Using kernel variant '{}' with LUT_SIZE={}\n", kernel_variant, LUT_SIZE);
        }
        #else
        fmt::print("✓ Using generic kernel with LUT_SIZE={}\n", LUT_SIZE);
        #endif

        // Create compute kernel for core_group_1
        auto compute_kernel_1 = CreateKernel(
            program,
            COMPUTE_KERNEL_PATH,
            core_group_1,
            ComputeConfig{
                .fp32_dest_acc_en = !use_bf16_mode,
                .unpack_to_dest_mode = unpack_to_dest_modes,
                .math_approx_mode = use_bf16_mode,
                .compile_args = compute_compile_args,
                .defines = compute_defines
            });

        // Create compute kernel for core_group_2 (if it exists)
        KernelHandle compute_kernel_2 = 0;
        if (!core_group_2.ranges().empty()) {
            compute_kernel_2 = CreateKernel(
                program,
                COMPUTE_KERNEL_PATH,
                core_group_2,
                ComputeConfig{
                    .fp32_dest_acc_en = !use_bf16_mode,
                    .unpack_to_dest_mode = unpack_to_dest_modes,
                    .math_approx_mode = use_bf16_mode,
                    .compile_args = compute_compile_args,
                    .defines = compute_defines
                });
            fmt::print("✓ Created compute kernels for group 1 and group 2\n");
        } else {
            fmt::print("✓ Created compute kernel for group 1 only\n");
        }

        // Set runtime arguments with EXTENSIVE logging
        // num_cores_y already defined above
        uint32_t tiles_written = 0;

        fmt::print("\nSetting runtime arguments:\n");
        fmt::print("{}\n", std::string(80, '-'));

        for (uint32_t i = 0; i < num_cores; i++) {
            // CRITICAL: Use TTNN's formula exactly
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            // Check which group this core belongs to
            uint32_t tiles_this_core = 0;
            KernelHandle compute_kernel_id = 0;

            if (core_group_1.contains(core)) {
                tiles_this_core = tiles_per_core_1;
                compute_kernel_id = compute_kernel_1;
                fmt::print("Core ({},{}) → Group 1: {} tiles, offset {}\n",
                          core.x, core.y, tiles_this_core, tiles_written);
            } else if (core_group_2.contains(core)) {
                tiles_this_core = tiles_per_core_2;
                compute_kernel_id = compute_kernel_2;
                fmt::print("Core ({},{}) → Group 2: {} tiles, offset {}\n",
                          core.x, core.y, tiles_this_core, tiles_written);
            } else {
                fmt::print("ERROR: Core ({},{}) not in any group!\n", core.x, core.y);
                TT_FATAL(false, "Core not assigned to any group");
            }

            // Set reader args: (input_addr, num_tiles, start_tile_id)
            SetRuntimeArgs(
                program, reader, core,
                {input_buffer->address(), tiles_this_core, tiles_written}
            );

            // Set writer args: (buffer_addr, num_tiles, start_tile_id)
            SetRuntimeArgs(
                program, writer, core,
                {output_buffer->address(), tiles_this_core, tiles_written}
            );

            // Set compute args: (num_tiles, compute_loop_factor)
            SetRuntimeArgs(program, compute_kernel_id, core, {tiles_this_core, compute_loop_factor});

            tiles_written += tiles_this_core;
        }

        fmt::print("{}\n", std::string(80, '-'));
        fmt::print("Total tiles distributed: {}\n", tiles_written);
        TT_FATAL(tiles_written == current_tiles, "Tile distribution mismatch! {} != {}", tiles_written, current_tiles);

        END_TIMER(KERNEL_CREATION);

        fmt::print("✓ Kernels created and configured\n");
        if (compute_loop_factor > 1) {
            fmt::print("⚡ Compute amplification enabled: {}× (recompute each tile {} times)\n", compute_loop_factor, compute_loop_factor);
        }
        fmt::print("\n");

        // STEP 10: Execute program with device profiler
        fmt::print("Executing program with CSV: {}\n", csv_filename);

        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        workload.add_program(device_range, std::move(program));

        auto kernel_exec_start = std::chrono::high_resolution_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);
        auto kernel_exec_end = std::chrono::high_resolution_clock::now();
        auto kernel_exec_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_exec_end - kernel_exec_start);
        fmt::print("{}TIMING_KERNEL_EXECUTION: {:.6f}\n", batch_prefix, kernel_exec_duration.count() / 1e6);

        // Read mesh device profiler results if enabled
        if (std::getenv("TT_METAL_DEVICE_PROFILER")) {
            ReadMeshDeviceProfilerResults(*mesh_device);
            fmt::print("✓ Device profiler results written to generated/profiler/\n");
        }

        fmt::print("✓ Execution complete\n\n");

        // STEP 11: Read and verify results
        START_TIMER(DEVICE_TO_HOST);
        std::vector<bfloat16> output_data_bf16;
        std::vector<float> output_data_fp32;

        if (use_bf16_mode) {
            distributed::EnqueueReadMeshBuffer(cq, output_data_bf16, output_buffer, true);
        } else {
            distributed::EnqueueReadMeshBuffer(cq, output_data_fp32, output_buffer, true);
        }
        END_TIMER(DEVICE_TO_HOST);

        // Helper lambda to get output value at index (works for both BF16 and FP32)
        auto get_output_value = [&](size_t i) -> float {
            return use_bf16_mode ? static_cast<float>(output_data_bf16[i]) : output_data_fp32[i];
        };

        const size_t output_size = use_bf16_mode ? output_data_bf16.size() : output_data_fp32.size();

        fmt::print("Results (samples across input range [{}, {}]):\n", test_min, test_max);
        fmt::print("{}\n", std::string(60, '-'));
        fmt::print("{:>5} {:>12} {:>12}\n", "Index", "Input", "Output");
        fmt::print("{}\n", std::string(60, '-'));

        // Show samples from start, middle, and end to demonstrate full range
        const int samples[] = {0, 3276, 6553, 9830, 13107, 16384, 19660, 22937, 26214, 29491, 32767};
        for (auto i : samples) {
            if (static_cast<size_t>(i) < output_size) {
                fmt::print("{:5d} {:12.6f} {:12.6f}\n",
                          i,
                          get_input_value(i),
                          get_output_value(i));
            }
        }

        fmt::print("{}\n", std::string(60, '-'));
        fmt::print("\nProcessed {} tiles ({} elements total)\n", current_tiles, output_size);

        // OPTIONAL: Dump full input/output to CSV for plotting
        if (!dump_csv_base.empty()) {
            // Insert precision and/or _tilesN before .csv extension
            std::string dump_csv_path;
            size_t ext_pos = dump_csv_base.rfind(".csv");
            std::string base_no_ext = (ext_pos != std::string::npos) ? dump_csv_base.substr(0, ext_pos) : dump_csv_base;

            std::string suffix;
            if (is_multi_precision) {
                suffix += "_" + current_precision;
            }
            if (is_batch_mode) {
                suffix += "_tiles" + std::to_string(current_tiles);
            }

            if (!suffix.empty()) {
                dump_csv_path = base_no_ext + suffix + ".csv";
            } else {
                dump_csv_path = dump_csv_base;
            }

            std::ofstream csv_file(dump_csv_path);
            if (csv_file.is_open()) {
                // Set high precision (17 significant figures, scientific notation) to preserve exhaustive BF16 values
                csv_file << std::scientific << std::setprecision(17);
                csv_file << "input,output\n";
                for (size_t i = 0; i < output_size; ++i) {
                    csv_file << get_input_value(i) << "," << get_output_value(i) << "\n";
                }
                csv_file.close();
                fmt::print("\n✓ Output data dumped to: {}\n", dump_csv_path);
            }
        }

            if (is_batch_mode || is_multi_precision) {
                fmt::print("{}END\n", batch_prefix);
            }
        } // end batch loop
        } // end precision loop

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
