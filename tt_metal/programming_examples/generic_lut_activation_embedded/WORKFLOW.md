# Embedded LUT Activation Function Workflow

Complete workflow for generating embedded LUT kernels, running benchmarks, and creating visualizations.

## Overview

This directory implements **embedded LUT kernels** where LUT coefficients are compiled directly into kernel binaries as `constexpr` arrays. This provides:
- **Zero L1 memory overhead** (no runtime LUT loading)
- **Runtime precision selection** (bf16/fp32 via JIT compilation)
- **One binary per activation/method/depth/segmentation** (fast kernel dispatch)

## Directory Structure

```
generic_lut_activation_embedded/
├── {arch}_*.csv                    # Result CSVs (in root directory)
├── {arch}_hardware_outputs/        # Raw hardware output CSVs (in root directory)
├── {arch}_plots_*/                 # Generated plots (in root directory)
├── kernels/compute/
│   ├── generated_luts/             # Generated C++ headers with embedded LUTs
│   │   └── {activation}/           # One directory per activation
│   │       ├── constant_depth_4_uniform.hpp
│   │       ├── linear_depth_8_adaptive.hpp
│   │       └── ...
│   ├── piecewise_constant.cpp      # Base kernel implementations
│   ├── piecewise_linear.cpp
│   └── ...
│   └── piecewise_*_embedded/       # Generated embedded kernels
│       └── {activation}_{method}_{depth}_{seg}.cpp
├── cmake/
│   └── EmbeddedTargets.cmake       # Generated CMake build targets
├── tools/                          # Code generation scripts
│   ├── generate_lut_headers.py     # Generate C++ headers from LUT files
│   ├── generate_embedded_kernels.py # Generate kernel variants
│   └── generate_cmake_targets.py   # Generate CMake build targets
└── *.sh                            # Workflow scripts
```

## Workflow Steps

The workflow is organized into 4 phases:
1. **Code Generation** (steps 1-3) - Run when LUTs change or adding new activations
2. **Build** (step 4) - Run after code generation or C++ changes
3. **Hardware Benchmarking** (step 5) - Main iteration loop
4. **Post-processing** (steps 6-7) - After benchmarks complete

---

## Phase 1: Code Generation (When LUTs change)

### 1. Generate LUT Headers
```bash
python3 tools/generate_lut_headers.py
```
Generates C++ header files from LUT files in `../generic_lut_activation/luts/`.

**Options:**
- `--activation <name>` - Generate for specific activation only
- `--segmentation <type>` - uniform | adaptive | both (default: both)
- `--degree <type>` - constant | linear | quadratic | cubic | hexic | octic
- `--depth <depths>` - Comma-separated depths (e.g., '4,8' or '8')

**Examples:**
```bash
python3 tools/generate_lut_headers.py                           # All headers
python3 tools/generate_lut_headers.py --activation sigmoid      # Sigmoid only
python3 tools/generate_lut_headers.py --degree cubic --depth 8  # Cubic depth 8
```

**Output:** `kernels/compute/generated_luts/{activation}/*.hpp` files with both bf16 and fp32 LUTs

**Run when:**
- Parent directory regenerates LUTs
- Adding new activation functions
- Changing LUT generation algorithms

### 2. Generate Embedded Kernels
```bash
python3 tools/generate_embedded_kernels.py
```
Generates kernel variants with inlined LUT data from the headers.

**Options:** Same as `generate_lut_headers.py`

**Examples:**
```bash
python3 tools/generate_embedded_kernels.py                      # All kernels
python3 tools/generate_embedded_kernels.py --degree linear      # Linear only
```

**Output:** `kernels/compute/*_embedded/{activation}_{method}_{depth}_{seg}.cpp`

**Run when:**
- LUT headers are regenerated
- Changing kernel implementations

### 3. Generate CMake Targets
```bash
python3 tools/generate_cmake_targets.py > cmake/EmbeddedTargets.cmake
```
Generates CMake build targets for all embedded binaries.

**Options:** Same as `generate_lut_headers.py`

**Examples:**
```bash
# Generate all targets
python3 tools/generate_cmake_targets.py > cmake/EmbeddedTargets.cmake

# Generate specific subset
python3 tools/generate_cmake_targets.py --degree cubic > cmake/EmbeddedTargets.cmake
```

**Output:** `cmake/EmbeddedTargets.cmake` (included by main CMakeLists.txt)

**Run when:**
- Adding new activations or configurations
- LUT headers are regenerated

---

## Phase 2: Build (After code generation)

### 4. Build Embedded Binaries
```bash
cd $TT_METAL_HOME
./build_metal.sh --build-programming-examples
```

Builds all embedded LUT activation binaries.

**Output:**
- Binaries in `build/programming_examples/`
- Format: `programming_examples_generic_lut_activation_embedded_{activation}_{method}_{depth}_{seg}`

**Run when:**
- After code generation
- After modifying C++ host code or kernel implementations

---

## Phase 3: Hardware Benchmarking (Main iteration loop)

### 5. Run Hardware Benchmarks (On Server)
```bash
./sweep_embedded.sh
```
Runs embedded LUT benchmarks on the connected server.

**Options:**
- `--degree <type>` - constant | linear | quadratic | cubic | hexic | octic (default: all)
- `--depth <depths>` - Comma-separated depths (e.g., '4,8')
- `--activation <name>` - Single activation only
- `--segmentation <type>` - uniform | adaptive | both (default: both)
- `--precision <type>` - bf16 | fp32 (default: bf16)

**Examples:**
```bash
./sweep_embedded.sh                                    # All tests, BF16
./sweep_embedded.sh --precision fp32                   # All tests, FP32
./sweep_embedded.sh --degree cubic                     # Cubic only, BF16
./sweep_embedded.sh --degree cubic --precision fp32    # Cubic only, FP32
./sweep_embedded.sh --activation sigmoid               # Sigmoid all degrees
./sweep_embedded.sh --degree linear --segmentation uniform  # Linear uniform only
./sweep_embedded.sh --degree hexic --depth 4,8 --activation tanh  # Specific config
```

**Output:**
- CSVs: `{arch}_piecewise_{method}_results.csv` (in root directory)
- Hardware outputs: `{arch}_hardware_outputs/piecewise_{method}_{activation}_{depth}_{seg}.csv` (in root directory)

**Run after:**
- Rebuilding binaries
- Fixing kernel implementations
- Parent directory regenerates LUTs (after full code generation)

---

## Phase 4: Post-processing

### 6. Pull Results from Server
```bash
./pull_results.sh
```
Downloads result CSVs and hardware outputs from remote server to local root directory.

**Options:**
- `--arch <arch>` - Filter by architecture (wormhole | blackhole)
- `--precision <prec>` - Filter by precision (bf16 | fp32)
- `--degree <degree>` - Filter by polynomial degree (constant | linear | quadratic | cubic | hexic | octic)
- `--status` - Show status without downloading hardware outputs

**Examples:**
```bash
./pull_results.sh                                # Pull everything
./pull_results.sh --status                       # Quick check (CSV only)
./pull_results.sh --arch wormhole --precision bf16  # Wormhole BF16 results
./pull_results.sh --degree cubic --precision fp32   # Cubic FP32 results
```

**Output:** CSV files and `{arch}_hardware_outputs/` directories in root

### 7. Generate All Plots
```bash
./plots.sh
```
Generates all visualization plots from the collected data using parent directory's plotting scripts.

**Options:**
- `--arch <arch>` - wormhole | blackhole | both (default: both)

**Examples:**
```bash
./plots.sh                    # All plots, both architectures
./plots.sh --arch wormhole    # Wormhole plots only
```

**Output:**
- `{arch}_plots_runtime/` - Runtime comparison plots
- `{arch}_plots_error/` - Error vs runtime plots
- `{arch}_plots_pareto/` - Pareto frontier analysis
- `plots_values/` - Activation function plots

---

## Quick Reference

### Full Workflow (Local + Remote)
```bash
# ═══════════════════════════════════════════════════════════════
# PHASE 1: CODE GENERATION (run when LUTs change)
# ═══════════════════════════════════════════════════════════════
python3 tools/generate_lut_headers.py
python3 tools/generate_embedded_kernels.py
python3 tools/generate_cmake_targets.py > cmake/EmbeddedTargets.cmake

# ═══════════════════════════════════════════════════════════════
# PHASE 2: BUILD (run after code generation)
# ═══════════════════════════════════════════════════════════════
cd $TT_METAL_HOME
./build_metal.sh --build-programming-examples

# ═══════════════════════════════════════════════════════════════
# PHASE 3: HARDWARE BENCHMARKING (main iteration loop)
# ═══════════════════════════════════════════════════════════════
cd tt_metal/programming_examples/generic_lut_activation_embedded
./sweep_embedded.sh --precision bf16    # Run benchmarks on server

# ═══════════════════════════════════════════════════════════════
# PHASE 4: POST-PROCESSING (after benchmarks)
# ═══════════════════════════════════════════════════════════════
./pull_results.sh      # Download results to root directory
./plots.sh             # Generate all visualizations
```

### Typical Iteration Cycle
```bash
# After fixing kernel code:
cd $TT_METAL_HOME
./build_metal.sh --build-programming-examples  # Rebuild
cd tt_metal/programming_examples/generic_lut_activation_embedded
./sweep_embedded.sh --precision bf16           # Run benchmarks
./pull_results.sh                              # Pull results
./plots.sh                                     # Update plots

# After parent regenerates LUTs:
python3 tools/generate_lut_headers.py
python3 tools/generate_embedded_kernels.py
python3 tools/generate_cmake_targets.py > cmake/EmbeddedTargets.cmake
cd $TT_METAL_HOME
./build_metal.sh --build-programming-examples
cd tt_metal/programming_examples/generic_lut_activation_embedded
./sweep_embedded.sh --precision bf16
./pull_results.sh
./plots.sh
```

### Quick Test - Single Config
Test one specific configuration (fastest iteration):

```bash
# 1. Generate code for one config
python3 tools/generate_lut_headers.py --activation sigmoid --degree cubic --depth 8
python3 tools/generate_embedded_kernels.py --activation sigmoid --degree cubic --depth 8
python3 tools/generate_cmake_targets.py > cmake/EmbeddedTargets.cmake

# 2. Build
cd $TT_METAL_HOME
./build_metal.sh --build-programming-examples

# 3. Run benchmark on server
cd tt_metal/programming_examples/generic_lut_activation_embedded
./sweep_embedded.sh --degree cubic --activation sigmoid --depth 8 --precision bf16

# 4. Pull and plot
./pull_results.sh --degree cubic --precision bf16
./plots.sh --arch wormhole
```

### Compare BF16 vs FP32
```bash
# Run both precisions
./sweep_embedded.sh --degree cubic --precision bf16
./sweep_embedded.sh --degree cubic --precision fp32

# Pull both
./pull_results.sh --degree cubic --precision bf16
./pull_results.sh --degree cubic --precision fp32

# Plot (parent scripts auto-detect both precisions)
./plots.sh --arch wormhole
```

### Quick Test - Single Activation (All Degrees/Depths)
```bash
# 1. Generate code for one activation
python3 tools/generate_lut_headers.py --activation sigmoid
python3 tools/generate_embedded_kernels.py --activation sigmoid
python3 tools/generate_cmake_targets.py --activation sigmoid > cmake/EmbeddedTargets.cmake

# 2. Build
cd $TT_METAL_HOME
./build_metal.sh --build-programming-examples

# 3. Run benchmarks
cd tt_metal/programming_examples/generic_lut_activation_embedded
./sweep_embedded.sh --activation sigmoid --precision bf16

# 4. Pull and plot
./pull_results.sh --activation sigmoid
./plots.sh --arch wormhole
```

---

## Output Files

### Result Files (Root Directory)
Result CSVs:
- `{arch}_piecewise_constant_results.csv`
- `{arch}_piecewise_linear_results.csv`
- `{arch}_piecewise_quadratic_results.csv`
- `{arch}_piecewise_cubic_results.csv`
- `{arch}_piecewise_hexic_results.csv`
- `{arch}_piecewise_octic_results.csv`
- `{arch}_hardware_outputs/*.csv` (raw kernel outputs)

### Plot Directories
- `{arch}_plots_runtime/` - Runtime comparison plots
- `{arch}_plots_error/` - Error vs runtime tradeoff plots
- `{arch}_plots_pareto/` - Pareto frontier analysis
- `plots_values/` - Activation function value plots

---

## Key Differences from Parent Directory

### Embedded Kernels
- **LUT storage**: Compiled into kernels as `constexpr` (not loaded at runtime)
- **Memory overhead**: Zero L1 usage for LUTs
- **Binary count**: One binary per (activation × method × depth × segmentation)
- **JIT compilation**: Precision selected at kernel creation time

### Runtime Precision Selection
```cpp
// Example: Binary supports both bf16 and fp32
auto defines = use_bf16 ? std::map<string, string>{{"USE_BF16", "1"}} : std::map<string, string>{};
auto kernel = CreateKernel(program, kernel_path, core, ComputeConfig{.defines = defines});
```

### Depth Limits (Match Parent)
- **constant, linear**: 4, 8, 16, 32
- **quadratic, cubic**: 4, 8, 16
- **hexic, octic**: 4, 8

### Not Included
- **Native SFPU**: Not applicable (uses parent's for comparison)
- **Piecewise constant**: Included (constant method with embedded LUTs)
- **Setup scripts**: Uses parent's server setup

---

## Troubleshooting

### Headers not found during kernel generation
```bash
python3 tools/generate_lut_headers.py  # Regenerate headers first
```

### Build fails with missing targets
```bash
python3 tools/generate_cmake_targets.py > cmake/EmbeddedTargets.cmake
cd $TT_METAL_HOME
./build_metal.sh --clean
./build_metal.sh --build-programming-examples
```

### Binary not found during sweep
```bash
ls build/programming_examples/ | grep generic_lut_activation_embedded  # Check binaries
./build_metal.sh --build-programming-examples  # Rebuild if missing
```

### Pull results fails
```bash
./pull_results.sh --status  # Check what's available on server
```

### Plots show missing data
```bash
./pull_results.sh           # Ensure all CSVs are downloaded
ls -la *_piecewise_*.csv    # Verify CSV files exist
ls -la *_hardware_outputs/  # Verify hardware outputs exist
```

### LUT files missing (parent directory)
```bash
cd ../generic_lut_activation
./generate_luts.sh          # Generate LUTs in parent
cd ../generic_lut_activation_embedded
python3 tools/generate_lut_headers.py  # Then regenerate headers
```

---

## Performance Notes

### Compile-Time Overhead
- **Headers**: Each header contains both bf16 and fp32 LUTs
- **Kernel compilation**: JIT compilation happens once per unique (kernel × precision) combination
- **Binary count**: ~1000 binaries total for all configurations

### Runtime Benefits
- **Zero L1 overhead**: No LUT storage in local memory
- **Fast dispatch**: Direct kernel execution (no LUT loading)
- **Precision flexibility**: One binary supports both bf16 and fp32

### Build Time
- **Full build**: ~10-15 minutes (1000+ binaries)
- **Incremental**: Seconds (only changed configs)
- **Recommended**: Use filters during development to reduce build scope
