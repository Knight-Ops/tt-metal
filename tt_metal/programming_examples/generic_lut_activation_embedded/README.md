# Generic LUT-Based Activation Functions (Embedded)

> **Note:** This embedded implementation focuses exclusively on LUT-based methods with coefficients embedded in compute kernels. For Native SFPU (hardware-accelerated) functionality, see the parent `generic_lut_activation` folder.

Comprehensive implementation and benchmarking of activation functions on Tenstorrent hardware using piecewise polynomial approximation methods: Piecewise Constant (PC), Piecewise Linear (PL), Piecewise Quadratic (PQ), Piecewise Quadratic Remez (PQR), and Piecewise Cubic Remez (PCR).

> 📘 **New here / learning SFPU optimization?** See [`tutorials/sfpu_optimization/SFPU_OPTIMIZATION_STORY.md`](tutorials/sfpu_optimization/SFPU_OPTIMIZATION_STORY.md) — a runnable, step-by-step walkthrough of optimizing an SFPU vector kernel from a naive baseline to ~5x faster, measuring each win on silicon.

## Setup & Build (from a clean checkout)

These are the exact steps required to build and run this example against current `tt-metal` HEAD. A clean checkout does **not** build out of the box — the steps below were verified on Blackhole (also applies to Wormhole).

### 1. Prerequisites
- A Tenstorrent device (Wormhole B0 or Blackhole). The build auto-detects the arch from silicon — `ARCH_NAME` is informational only.
- The **`tt-polynomial-fitter`** repo checked out (provides coefficient CSVs and `extract_accuracy.py` for ground-truth/ULP). Default location: `/localdev/<user>/tt-polynomial-fitter`. Override with `TT_POLY_FIT_DIR`.
- **System Python with `numpy`** (for `extract_accuracy.py` and `best_all.py`). The accuracy scripts use `/usr/bin/python3` on purpose — `python_env`'s torch has broken BF16 ULP spacing.

### 2. Wire the example into the build (REQUIRED — not done by default)
Neither `generic_lut_activation` nor `generic_lut_activation_embedded` is registered in the programming-examples CMake tree, so the targets don't exist until you add them. In `tt_metal/programming_examples/CMakeLists.txt`, alongside the other `add_subdirectory(...)` lines:

```cmake
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/generic_lut_activation)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/generic_lut_activation_embedded)
```

### 3. Configure & build the adhoc target
Programming examples are off by default; the device profiler (Tracy) is required for `run_csv.sh` timing (it's on by default in `build_metal.sh`).

```bash
cd $TT_METAL_HOME
# Enable programming examples (Tracy/ENABLE_TRACY is already ON in a default build)
cmake -DBUILD_PROGRAMMING_EXAMPLES=ON build_Release
# Build only the adhoc target used by run_csv.sh
ninja -C build_Release programming_examples_generic_lut_activation_embedded_adhoc
```
The binary lands at `build_Release/programming_examples/programming_examples_generic_lut_activation_embedded_adhoc`.

### 4. Kernel compatibility fixes for current tt-metal HEAD
The compute/dataflow kernels were written against an older `tt-metal` + SFPI toolchain. Building against current HEAD requires these (already applied in this branch — listed so the drift is documented):

| File | Old (broken) | New (current API) |
|------|--------------|-------------------|
| `kernels/dataflow/reader.cpp`, `writer.cpp` | `DPRINT << x << ENDL()` (now a hard `static_assert`) | removed the debug-print lines |
| `kernels/compute/piecewise_generic.cpp`, `piecewise_rational.cpp` | bare `#pragma unroll` (`-Werror=unknown-pragmas`) | `#pragma GCC unroll 16` |
| `kernels/compute/piecewise_generic.cpp` | `_sfpu_reciprocal_<3>(x)` | `sfpu_reciprocal_iter<3>(x)` |
| `piecewise_generic.cpp`, `piecewise_rational.cpp`, `piecewise_generic_specialized.cpp` | `int32_to_float(x, 0)` | `int32_to_float(x, RoundMode::Nearest)` |
| `kernels/compute/piecewise_generic_specialized.cpp` (dispatcher) | parity x²-Horner + dual-eval at high degree → GCC `-O3 -flto` register-reload **ICE** | fall back to single-eval for `POLY_PARITY_* && POLY_DEGREE > 4` |

> JIT does not track ckernel-header changes — after editing shared headers, `rm -rf ~/.cache/tt-metal-cache` to force recompilation.

### 5. Run a coefficient CSV on silicon
Run from the **repo root** (the binary resolves soc descriptors relative to cwd):

```bash
export TT_POLY_FIT_DIR=/localdev/<user>/tt-polynomial-fitter
cd tt_metal/programming_examples/generic_lut_activation_embedded
./run_csv.sh $TT_POLY_FIT_DIR/data/coefficients/<activation>_<cfg>.csv \
    --activation <name> --precision fp32 --tiles 256 --runs 1
```
`run_csv.sh` auto-detects degree/segments/range/range-reduction from the CSV, JIT-compiles the kernel, runs 5 standard shapes (or `--tiles N` for one), and reports MAE / MaxErr / MaxULP / MeanULP + Tracy timing.

### 6. (Optional) Compare against TTNN native
Both comparison tools need a TTNN Python env:
```bash
cd $TT_METAL_HOME
./create_venv.sh           # builds python_env with ttnn (required for native ttnn.<op>)
```

**Recommended — safe two-way (native vs ours), no header surgery:**
`tools/compare_native_vs_embedded.sh` runs `ttnn.<activation>` (native) and our embedded LUT kernel (`run_csv.sh`) over the **same** input range and reports MAE + MaxULP for both, plus the embedded Tracy kernel time. It never touches `tt_metal/hw/ckernels`, so it is safe on stock activations.
```bash
export TT_POLY_FIT_DIR=/localdev/<user>/tt-polynomial-fitter
cd tt_metal/programming_examples/generic_lut_activation_embedded
./tools/compare_native_vs_embedded.sh --activation tanh \
    --csv $TT_POLY_FIT_DIR/data/coefficients/tanh_n8d8_s1_uniform_rational_ulp.csv \
    --precision both
# Batch: loop over an "act,prec,csvname" worklist, calling the script per line.
```

**Batch sweep over a `best*.csv` (replicable native-vs-ours table):**
`tools/sweep_best_native_vs_embedded.sh` resolves every `(activation, precision)` row of a `best*.csv` to its best-ULP coefficient file (by header name + `source_metric` — robust to schema changes) and runs the two-way comparison for each. Use it to compare selection policies — e.g. `best.csv` (lowest ULP) vs `best95.csv` (cheapest within 95% of peak accuracy) vs `best99.csv`:
```bash
export TT_POLY_FIT_DIR=/localdev/<user>/tt-polynomial-fitter
cd tt_metal/programming_examples/generic_lut_activation_embedded
./tools/sweep_best_native_vs_embedded.sh --best-csv $TT_POLY_FIT_DIR/best95.csv --precision bf16
./tools/sweep_best_native_vs_embedded.sh --best-csv $TT_POLY_FIT_DIR/best99.csv --precision bf16
# subset: --activations tanh,asin,gelu ; single shape: --tiles 256
```
> bf16 note: `best.csv` minimizes ULP with no cost awareness, so it can pick 32-segment polynomials that win a meaningless sub-1.0 ULP over a single-segment rational. Since the segment selector is a *predicated* `v_if` cascade (every segment's polynomial runs on every lane), that costs ~segments×degree per element. For bf16 deployment prefer **`best95.csv`** — cheapest fit at ULP≈native-parity — which collapses that cost ~10-40× at no meaningful accuracy loss.

**Full three-way (native vs drop-in vs embedded):** `tools/compare_three_way.sh`.
> ⚠️ `compare_three_way.sh` assumes a drop-in header has **already been applied** for the activation (via `apply_dropin.sh`). To measure the "original" baseline it *deletes* `ckernel_sfpu_<act>.h` expecting a composite-op fallback — on a stock activation whose header is the real git-tracked one, this **deletes the real header and breaks the build**. Apply the drop-in first, or use `compare_native_vs_embedded.sh` instead.

### 7. (Optional) Regenerate best.csv coefficient index
`best.csv` in `tt-polynomial-fitter` can be stale/inconsistent with `data/coefficients/`. Regenerate over the current coefficient set (needs `numpy`):

```bash
cd $TT_POLY_FIT_DIR
./best_all.sh --input-dir data/coefficients --output-dir .
```
> Caveat: coefficient files carry embedded accuracy metadata that has been observed to be fabricated/cloned across degrees for some families (e.g. `log_*_s128`). Trust on-silicon measurement over both the file metadata and the `best.csv` claim.

## Overview

This example provides five LUT-based methods for computing activation functions on Tenstorrent accelerators:

1. **Piecewise Constant (PC)** - Flat segments using 1024-entry LUT
2. **Piecewise Linear (PL)** - Linear segments with slope and offset coefficients (least squares fitting)
3. **Piecewise Quadratic (PQ)** - Quadratic segments with a, b, c coefficients (least squares fitting)
4. **Piecewise Quadratic Remez (PQR)** - Quadratic minimax approximation (equioscillation)
5. **Piecewise Cubic Remez (PCR)** - Cubic minimax approximation (a, b, c, d coefficients)

### Supported Activations

**All LUT-based methods support 29 activations:**

These activations can be approximated with 1D piecewise polynomial LUTs. GLU variants are excluded as they require 2D operations.

- **atanh**, **celu**, **cos**, **cosh**, **elu**, **erf**, **exp**, **gelu**
- **hardshrink**, **hardsigmoid**, **hardswish**, **hardtanh**, **leaky_relu**, **logsigmoid**
- **mish**, **prelu**, **relu**, **relu6**, **selu**, **sigmoid**
- **sin**, **sinh**, **softplus**, **softshrink**, **softsign**, **swish**, **tanh**
- **tanhshrink**, **threshold**

**Configuration:** All activation parameters (test ranges, plot ranges) are centralized in `activations_config.dat` (from parent folder) for consistent behavior across Python and C++ code.

### Key Findings (from hardware benchmarks)

**Best overall accuracy**: PL-128 (MAE 0.00149) - excellent balance of accuracy and memory usage

**Accuracy vs Memory tradeoffs**:
- **PL-128**: MAE 0.00149, 1.0 KB memory
- **PQ-64**: MAE 0.00184, 768 B memory
- **PQ-32**: MAE 0.00261, 384 B memory
- **PQ-16**: MAE 0.00318, 196 B memory
- **PQ-8**: MAE 0.00639, 100 B memory

**Memory-constrained systems**: Use PQ-8 (100B, MAE 0.006) or PQ-16 (196B, MAE 0.003)

**Note:** For comparisons with Native SFPU hardware-accelerated operations, see the parent `generic_lut_activation` folder.

## Recent Improvements (January 2026)

### Input Range Fix
**Problem:** All tests previously used hardcoded [-10, 10] input range, causing numerical instability for functions like logsigmoid (which has domain issues outside [-1, 1]).

**Solution:** Test programs now read per-activation `test_range` from `activations_config.dat` and generate input data within safe ranges. LUTs remain generated over full `plot_range` [-10, 10] for visualization, but tests use appropriate subranges.

**Impact:** Dramatically improved accuracy metrics for numerically sensitive activations (e.g., logsigmoid relative error reduced from 7000+ to reasonable values).

### Hardware Capture Integration
**Before:** Sweep scripts ran tests for timing, then `capture_hardware_outputs.sh` re-ran same tests to dump CSVs.

**After:** Hardware capture integrated into sweep scripts using `DUMP_OUTPUT_CSV` environment variable on the last run of each configuration. No duplicate test execution.

**Impact:** ~2× faster workflow, eliminates redundant computation.

### Centralized Configuration
All activation lists, ranges, and parameters consolidated from scattered hardcoded arrays into:
- `../generic_lut_activation/activations_config.dat` (single source of truth in parent folder)
- `activations_config.sh` (Bash)
- `activations_config.py` (Python)
- `activation_config.hpp` (C++)

**Impact:** Consistent behavior across all scripts, easier maintenance, no drift between Python and C++.

## Configuration Management

All activation function parameters are centralized in configuration files for consistency across the codebase:

**`../generic_lut_activation/activations_config.dat`** - Single source of truth for:
- Activation names and descriptions
- Native SFPU hardware support flags
- Test ranges (safe input ranges for testing: e.g., logsigmoid uses [-1, 1])
- Plot ranges (visualization ranges: typically [-10, 10])
- Full mathematical ranges (domain of definition)

**`activations_config.sh`** - Bash version sourced by sweep scripts
**`activations_config.py`** - Python version imported by plotting scripts
**`activation_config.hpp`** - C++ header for test programs to read CSV at runtime

This centralized approach ensures:
- Test programs use correct input ranges per activation (preventing numerical instability)
- Sweep scripts iterate over consistent activation lists
- Plotting scripts use proper labels and ranges
- No hardcoded activation arrays scattered across codebase

**Example: logsigmoid**
- `test_range`: [-1, 1] (numerically safe range for testing)
- `plot_range`: [-10, 10] (full visualization range)
- `full_range`: [-∞, ∞] (mathematical domain)

LUTs are generated over `plot_range`, but test data uses `test_range` to avoid overflow/underflow errors.

## Quick Start

**Complete workflow for remote hardware benchmarks**:
```
generate_luts.sh → auto_reserve.sh --setup → sweep_all.sh → pull_results.sh → plots.sh
```

See [Section 7: Complete Experiment Workflow](#7-complete-experiment-workflow) for detailed step-by-step instructions.

### 1. Generate LUT Files

```bash
cd tt_metal/programming_examples/generic_lut_activation

# Parallel generation (RECOMMENDED - 3.75× faster)
# Generates all 464 LUTs in ~277s using 16 parallel processes
bash generate_luts.sh

# Or generate individually (slower, sequential)
python3 generate_piecewise_constant_luts.py       # PC (1024-entry LUTs)
python3 generate_piecewise_linear_luts.py         # PL (4-256 segments)
python3 generate_piecewise_quadratic_luts.py      # PQ (4-256 segments)
python3 generate_piecewise_quadratic_remez_luts.py # PQ Remez (4-256 segments)
python3 generate_piecewise_cubic_remez_luts.py    # Cubic Remez (4-256 segments)
```

**Parallel generation features:**
- 16 concurrent processes (one per activation)
- Each process generates all approximation types for its activation
- Progress monitoring with completion timestamps
- Automatic verification of LUT counts

### 2. Build

```bash
./build_metal.sh --build-programming-examples
```

### 3. Run

```bash
export ARCH_NAME=wormhole_b0  # or blackhole

# Native SFPU (hardware-accelerated)
./build_Release/programming_examples/programming_examples_generic_lut_activation_native_sfpu tanh

# Piecewise Constant (8-256 segments, 1024-entry LUT)
./build_Release/programming_examples/programming_examples_generic_lut_activation_pc_8 \
  tt_metal/programming_examples/generic_lut_activation/luts/sigmoid_1024.lut

# Piecewise Linear (8-256 segments)
./build_Release/programming_examples/programming_examples_generic_lut_activation_pwl_128 \
  tt_metal/programming_examples/generic_lut_activation/luts/piecewise_linear_sigmoid_128.lut

# Piecewise Quadratic (8-256 segments)
./build_Release/programming_examples/programming_examples_generic_lut_activation_pq_16 \
  tt_metal/programming_examples/generic_lut_activation/luts/piecewise_quadratic_sigmoid_16.lut
```

#### Configurable Tile Counts

All binaries support the `--tiles` flag to control workload size (default: 32 tiles):

```bash
# Test with 256 tiles for more stable runtime measurements
./build_Release/programming_examples/programming_examples_generic_lut_activation_pc_128 \
  tt_metal/programming_examples/generic_lut_activation/luts/sigmoid_1024.lut --tiles 256

# Test with default 32 tiles (32,768 elements)
./build_Release/programming_examples/programming_examples_generic_lut_activation_pwl_64 \
  tt_metal/programming_examples/generic_lut_activation/luts/piecewise_linear_gelu_64.lut
```

**Why adjust tile count?**
- Larger tile counts (128, 256) reduce runtime variance by amortizing kernel launch overhead
- Smaller tile counts (32, 64) run faster but may show higher variance
- Full sweep uses tile_count=256 for stable measurements; quick-run uses tile_count=32

### 4. Run Comprehensive Benchmarks

```bash
# Change to working directory
cd tt_metal/programming_examples/generic_lut_activation_embedded

# Run all sweeps (unified script - RECOMMENDED)
export ARCH_NAME=wormhole_b0  # or blackhole
bash sweep_all.sh              # Full sweep: all LUT methods, depths [4,8,16,32], tile_count=256

# Targeted sweeps (specific models, activations, or depths)
bash sweep_all.sh --degree linear                     # Only piecewise linear
bash sweep_all.sh --degree cubic --activation sigmoid # Only cubic sigmoid
bash sweep_all.sh --activation atanh                  # All methods, only atanh
bash sweep_all.sh --depth 32                          # All methods, only depth 32
bash sweep_all.sh --depth 8,16 --activation gelu      # All methods, gelu at depths 8,16

# Or run individual method sweeps
bash sweep_embedded.sh constant    # Piecewise Constant (29 activations × depths)
bash sweep_embedded.sh linear      # Piecewise Linear (29 activations × depths)
bash sweep_embedded.sh quadratic   # Piecewise Quadratic (29 activations × depths)
bash sweep_embedded.sh cubic       # Piecewise Cubic (29 activations × depths)
```

**sweep_all.sh options:**
- `--depth <depths>`: Comma-separated depths (e.g., '32' or '4,8,16,32')
- `--activation <name>`: Run specific activation only (default: all activations)
- `--model <type>`: Run specific model type: sfpu|constant|linear|quadratic|cubic (default: all models)

**sweep_all.sh features:**
- Runs selected methods sequentially (all 5 by default, or specific model with --model)
- Automatically detects architecture from `ARCH_NAME` or hostname
- Resets devices before sweeps
- **Integrated hardware capture**: Dumps output CSVs during last run of each config (no separate capture step needed)
- Creates timestamped log file: `sweep_{arch}.log`

Results saved to:
- `{arch}_native_sfpu_results.csv`
- `{arch}_piecewise_constant_results.csv`
- `{arch}_piecewise_linear_results.csv`
- `{arch}_piecewise_quadratic_remez_results.csv`
- `{arch}_piecewise_cubic_remez_results.csv`
- `{arch}_hardware_outputs/*.csv` (captured inline during sweeps)

**CSV Format:**

Each sweep script generates results with detailed timing breakdown across multiple runs (2 executions per configuration):

```csv
# Native SFPU format
activation,tile_count,device_init_ms,program_creation_ms,buffer_alloc_ms,data_prep_ms,
host_to_device_ms,kernel_creation_ms,kernel_exec_ms,device_to_host_ms,
runtime_mean_ms,runtime_min_ms,runtime_stddev_ms,mae,rmse,max_error,mean_rel_error,status

# Piecewise methods format
depth,activation,tile_count,device_init_ms,program_creation_ms,buffer_alloc_ms,
data_prep_ms,host_to_device_ms,kernel_creation_ms,kernel_exec_ms,device_to_host_ms,
runtime_mean_ms,runtime_min_ms,runtime_stddev_ms,mae,rmse,max_error,mean_rel_error,status
```

**Key columns:**
- `tile_count`: Number of tiles processed (256 for full sweep, 32 for quick-run)
- `device_init_ms`: Device initialization time (varies between first/subsequent runs)
- `kernel_exec_ms`: Actual kernel execution time (most relevant for performance)
- `runtime_mean_ms`: Mean total runtime excluding device init (milliseconds)
- `runtime_min_ms`: Minimum runtime (most reliable for comparison)
- `runtime_stddev_ms`: Standard deviation (kernel_exec only, indicates stability)
- `mae/rmse/max_error`: Accuracy metrics vs ground truth (tested over safe `test_range`)

**Hardware capture:** On the last run (run 2 of 2), the sweep sets `DUMP_OUTPUT_CSV` to save hardware outputs to `{arch}_hardware_outputs/{method}_{activation}_{depth}.csv` for plotting.

### 5. Remote Server Initial Setup

When launching Docker containers with access to Tenstorrent silicon/cards, perform these one-time setup steps:

#### Prerequisites on Remote Servers

1. **Install numpy for system Python** (required by sweep scripts for accuracy calculations):
   ```bash
   pip3 install --user numpy
   ```

2. **Verify Python virtual environment** (required by LUT generation scripts):
   ```bash
   cd /localdev/nkapre/tt-metal  # Or your tt-metal path
   ls python_env/  # Should exist
   source python_env/bin/activate
   python3 -c "import numpy; print('numpy OK')"
   deactivate
   ```

3. **Build tt-metal with programming examples**:
   ```bash
   cd /localdev/nkapre/tt-metal
   export ARCH_NAME=wormhole_b0  # or blackhole
   ./build_metal.sh --build-programming-examples
   ```

4. **Verify LD_LIBRARY_PATH** (needed for shared library loading):
   ```bash
   export LD_LIBRARY_PATH=/localdev/nkapre/tt-metal/build_Release/lib:$LD_LIBRARY_PATH
   export TT_METAL_RUNTIME_ROOT=/localdev/nkapre/tt-metal
   ./build_Release/programming_examples/programming_examples_generic_lut_activation_native_sfpu gelu
   # Should run successfully and show "✓ Test PASSED"
   ```

#### Server Hostnames

Configure these hostnames in `wormhole.sh` and `blackhole.sh`:

- **Wormhole B0**: `yyzc-wh-03` (port 49819)
- **Blackhole P100**: `bh-34` (port 49819)

```bash
# SSH access test
ssh -A yyzc-wh-03 -p 49819 "hostname"
ssh -A bh-34 -p 49819 "hostname"
```

#### Common Issues After Container Launch

**"ModuleNotFoundError: No module named 'numpy'"**
- **Cause**: Sweep scripts use system `python3` for accuracy calculations
- **Fix**: `pip3 install --user numpy` on the remote server

**"libtt_metal.so: cannot open shared object file"**
- **Cause**: Shared library not in search path
- **Fix**: Ensure `LD_LIBRARY_PATH` includes `$REMOTE_DIR/build_Release/lib`
- **Auto-handled**: Remote scripts (`wormhole.sh`, `blackhole.sh`) set this automatically

**"Binary not found" errors**
- **Cause**: Programming examples not built
- **Fix**: Run `./build_metal.sh --build-programming-examples` on remote server

**LUT generation fails in remote scripts**
- **Cause**: Python virtual environment not activated
- **Fix**: Remote scripts now automatically activate `python_env` before LUT generation
- **Verify**: Check that `source python_env/bin/activate` appears in script output

**Device access errors**
- **Cause**: Device in use or needs reset
- **Fix**: `tt-smi -r 0` (Blackhole) or `tt-smi -r 0,1,2,3,4,5,6,7` (Wormhole)
- **Note**: Remote scripts attempt device reset but continue if `tt-smi` unavailable

### 6. Shell Scripts Overview

The benchmark workflow is orchestrated by 12 shell scripts organized into functional groups:

**Configuration (2 scripts)**:
- `activations_config.sh` - Centralized activation lists sourced by sweep scripts
- `hosts.sh` - Auto-generated server configuration (hostnames, ports, passwords)

**Setup & Infrastructure (3 scripts)**:
- `auto_reserve.sh` - Reserve Wormhole and Blackhole servers via IRD (with optional `--setup` flag)
- `setup_server.sh` - Automated server setup (dual-mode: local orchestration or remote execution)
- `generate_luts.sh` - Parallel LUT generation for all methods

**Sweep Scripts (5 scripts)**:
- `sweep_native_sfpu.sh` - Native SFPU benchmarks
- `sweep_piecewise_constant.sh` - Piecewise constant benchmarks
- `sweep_piecewise_linear.sh` - Piecewise linear benchmarks
- `sweep_piecewise_quadratic.sh` - Piecewise quadratic + Remez benchmarks
- `sweep_piecewise_cubic.sh` - Piecewise cubic Remez benchmarks
- `sweep_all.sh` - Unified sweep orchestration (dual-mode: local orchestration or remote execution)

**Results (2 scripts)**:
- `pull_results.sh` - Pull CSV results and hardware outputs from servers
- `plots.sh` - Generate all visualization plots from results

**Complete Workflow**:
```
generate_luts.sh → auto_reserve.sh --setup → sweep_all.sh → pull_results.sh → plots.sh
```

**Key Features**:
- All scripts use `#!/bin/bash` and `set -e` for consistency
- Configuration centralized in `activations_config.sh` and `hosts.sh`
- Architecture detection via `ARCH_NAME` environment variable or hostname
- Dual-mode scripts (`setup_server.sh`, `sweep_all.sh`) auto-detect local vs remote execution
- No hardcoded hostnames or ports (all use `hosts.sh`)

### 7. Complete Experiment Workflow

This section documents the end-to-end workflow for running benchmarks on remote Tenstorrent hardware.

#### Step 0: Generate LUTs Locally (One-Time Setup)

**IMPORTANT**: Generate all LUT files once on your local machine before provisioning servers. This step takes ~4-5 minutes and only needs to be done once.

```bash
cd tt_metal/programming_examples/generic_lut_activation

# Ensure Python virtual environment is activated
source ../../../python_env/bin/activate

# Generate all LUTs in parallel (464 LUTs in ~277 seconds)
bash generate_luts.sh
```

**What this generates:**
- **16 Constant LUTs** (1024-entry): One per activation
- **112 Linear LUTs**: 16 activations × 7 depths (4, 8, 16, 32, 64, 128, 256)
- **112 Quadratic LUTs**: 16 activations × 7 depths
- **112 Quadratic Remez LUTs**: 16 activations × 7 depths
- **112 Cubic Remez LUTs**: 16 activations × 7 depths
- **Total: 464 LUTs** saved to `luts/` directory

**Note**: Full sweeps only test depths 4-128 (not 256), but all LUTs are generated for completeness.

**Performance:**
- Parallelized by activation (16 concurrent processes)
- ~277s on modern CPU (3.75× faster than sequential generation)
- Each process generates all approximation types for one activation

**Verify LUT generation:**
```bash
# Check total count (should be 464)
ls luts/*.lut | wc -l

# Check per-activation breakdown (each should have 29)
for act in sigmoid tanh gelu swish relu leaky_relu softplus exp elu selu mish hardsigmoid softsign sin cos erf; do
    echo "$act: $(ls luts/${act}_1024.lut luts/piecewise_*_${act}_*.lut 2>/dev/null | wc -l) LUTs"
done
```

**Once generated locally, these LUTs will be automatically copied to remote servers during setup.**

#### Step 1: Reserve Remote Servers

Use `auto_reserve.sh` to automatically reserve both Wormhole and Blackhole servers via IRD:

```bash
cd tt_metal/programming_examples/generic_lut_activation

# Reserve servers with automatic setup (RECOMMENDED - one-step workflow)
bash auto_reserve.sh --setup

# Or reserve without setup and run setup manually later
bash auto_reserve.sh

# Specify custom duration (e.g., 2 hours) with automatic setup
bash auto_reserve.sh --setup 2:00:00

# Choose specific architectures and machine models
bash auto_reserve.sh --arch wormhole_b0 --wormhole-model x2
bash auto_reserve.sh --arch blackhole --blackhole-model p150
bash auto_reserve.sh --arch both --wormhole-model x1 --blackhole-model p100
```

**auto_reserve.sh options:**
- `--setup`: Automatically run setup_server.sh on reserved servers (recommended for streamlined workflow)
- `--arch <wormhole_b0|blackhole|both>`: Which architectures to reserve (default: both)
- `--wormhole-model <x1|x2>`: Wormhole machine type (default: x1, single-card)
- `--blackhole-model <p100|p150>`: Blackhole machine type (default: p100, single-card)
- `--skip-build`, `--skip-luts`, `--skip-reset`: Pass-through flags for setup (when using --setup)

This script:
- Reserves Wormhole B0 and/or Blackhole P100 servers (single-card by default)
- Automatically generates `hosts.sh` with server details
- Flushes old SSH keys
- Optionally runs automated setup via `--setup` flag (skips Step 2)

#### Step 2: Set Up Remote Servers

**NOTE**: If you used `auto_reserve.sh --setup` in Step 1, this step is already complete and you can skip to Step 3.

Otherwise, use the automated setup orchestrator to configure both servers from your local machine:

```bash
cd tt_metal/programming_examples/generic_lut_activation

# Automated setup from local machine (RECOMMENDED - dual-mode orchestration)
bash setup_server.sh --arch both

# Or setup specific architectures only
bash setup_server.sh --arch wormhole_b0
bash setup_server.sh --arch blackhole

# Setup with options (skip build/LUTs/reset if needed)
bash setup_server.sh --arch both --skip-build
bash setup_server.sh --arch both --skip-luts
bash setup_server.sh --arch both --skip-reset
```

The `setup_server.sh` script operates in **dual-mode**:
- **Local mode** (with hosts.sh): Orchestrates remote setup via SSH, runs setup on both servers
- **Remote mode** (without hosts.sh): Performs actual setup tasks on the server

**What setup does:**
- Installs system dependencies (tmux, zsh, oh-my-zsh)
- Cleans cache directory (`~/.cache` - prevents build issues)
- Installs Python packages (numpy, mpmath, OptimalPoly)
- Pulls latest code from current git branch
- Builds tt-metal with programming examples
- Resets devices via tt-smi
- Generates all LUT files (PC, PL, PQ, Cubic, Hexic, Octic)

**Note**: If you've already generated LUTs locally (Step 0), the setup script will regenerate them on the server. This is safe but redundant. LUTs are deterministic and identical regardless of where they're generated.

#### Step 3: Run Benchmarks

Use the unified `sweep_all.sh` script to launch benchmarks on remote servers:

```bash
cd tt_metal/programming_examples/generic_lut_activation

# Full sweep mode from local machine (~2-3 hours per arch, all models, all activations)
# - All 7 model types (Native SFPU, PC, PL, PQ, Cubic, Hexic, Octic)
# - All 29 activations
# - Depths: 4, 8, 16, 32 (default)
# - Tile count: 256 (262,144 elements)
# - 2 runs per configuration
bash sweep_all.sh --arch both

# Run on specific architecture only
bash sweep_all.sh --arch wormhole_b0  # or --arch blackhole

# Targeted sweeps (faster, run synchronously)
bash sweep_all.sh --model linear                      # Only piecewise linear
bash sweep_all.sh --model cubic --activation sigmoid  # Only cubic sigmoid
bash sweep_all.sh --model quadratic --depth 32        # Only quadratic at depth 32
bash sweep_all.sh --depth 32 --activation atanh       # All models, only atanh at depth 32
bash sweep_all.sh --depth 8,16,32                     # All models, selected depths
```

**sweep_all.sh options:**
- `--arch <wormhole_b0|blackhole|both>`: Target architecture (local mode only, default: both)
- `--depth <depths>`: Comma-separated depths (e.g., '32' or '4,8,16,32'). Default: 4,8,16,32
- `--activation <name>`: Run specific activation only (default: all activations)
- `--model <type>`: Run specific model type: sfpu|constant|linear|quadratic|cubic|hexic|octic (default: all models)

The `sweep_all.sh` script operates in **dual-mode**:
- **Local mode** (with hosts.sh): Orchestrates remote execution via SSH
  - Pulls latest code from current git branch to remote servers
  - Launches sweeps in detached tmux sessions (full sweeps) or runs synchronously (targeted sweeps)
  - Runs on both architectures in parallel by default
- **Remote mode** (without hosts.sh): Performs actual sweep execution on the server

**Execution modes:**
- **Full sweep** (no filters): Runs in detached tmux sessions, can disconnect and sweeps continue
- **Targeted sweep** (with --depth, --activation, or --model): Runs synchronously, waits for completion

#### Step 4: Monitor Progress

Check on sweep progress using tmux sessions:

```bash
# Source hosts.sh to get server details
source hosts.sh

# Check tmux sessions
ssh -p $BLACKHOLE_PORT $BLACKHOLE_HOST 'tmux ls'
ssh -p $WORMHOLE_PORT $WORMHOLE_HOST 'tmux ls'

# Attach to live session (Ctrl+B, D to detach)
ssh -A -p $BLACKHOLE_PORT $BLACKHOLE_HOST -t 'tmux attach -t blackhole_sweep'
ssh -A -p $WORMHOLE_PORT $WORMHOLE_HOST -t 'tmux attach -t wormhole_sweep'

# Tail log files
ssh -p $BLACKHOLE_PORT $BLACKHOLE_HOST 'tail -f /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/sweep_blackhole.log'
ssh -p $WORMHOLE_PORT $WORMHOLE_HOST 'tail -f /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/sweep_wormhole_b0.log'
```

#### Step 5: Collect Results

The `run_sweeps.sh` script automatically syncs results back to your local machine after sweeps complete.

Results are saved with architecture-specific prefixes:
- **CSV files**: `{arch}_native_sfpu_results.csv`, `{arch}_piecewise_*.csv`
- **Hardware outputs**: `{arch}_hardware_outputs/`

You can also manually download results:
```bash
source hosts.sh

# Download CSV files
scp -P $BLACKHOLE_PORT $BLACKHOLE_HOST:/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/blackhole_*.csv .
scp -P $WORMHOLE_PORT $WORMHOLE_HOST:/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/wormhole_*.csv .

# Download hardware outputs
scp -P $BLACKHOLE_PORT -r $BLACKHOLE_HOST:/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/blackhole_hardware_outputs .
scp -P $WORMHOLE_PORT -r $WORMHOLE_HOST:/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/wormhole_hardware_outputs .
```

#### Step 6: Generate Plots

Generate visualizations from the collected CSV results and hardware outputs using the unified plotting script:

```bash
cd tt_metal/programming_examples/generic_lut_activation

# Generate plots for both architectures (RECOMMENDED)
bash plots.sh

# Or generate for specific architecture only
bash plots.sh --arch wormhole   # Wormhole B0 only
bash plots.sh --arch blackhole  # Blackhole P100 only
```

The `plots.sh` script automatically:
1. Generates runtime comparison plots for each architecture
2. Generates error vs runtime tradeoff plots
3. Generates activation function comparison plots (from hardware outputs)
4. Generates Pareto frontier analysis (polynomial degree × depth tradeoffs)
5. Generates theoretical error analysis plots (from LUTs)

**Individual plotting scripts** (for advanced use):
```bash
# Runtime and error plots (per architecture)
python3 plot_runtime_comparison.py --arch-prefix wormhole
python3 plot_error_vs_runtime.py --arch-prefix wormhole

# Activation comparison (uses hardware_outputs directories)
python3 plot_activation_comparison.py

# Pareto frontier analysis (polynomial degree × depth tradeoffs)
python3 plots/plot_pareto_frontier.py --arch-prefix wormhole

# Theoretical error analysis (uses LUT files)
python3 plot_theoretical_error.py
```

**Output directories:**

- **`{arch}_plots_runtime/`**: Runtime comparison plots
  - Individual activation runtime plots showing all methods
  - Runtime vs tile count scaling plots
  - Grouped comparison plots

- **`{arch}_plots_error/`**: Error vs runtime tradeoff plots
  - Individual activation error vs runtime plots (multi-panel by tile count)
  - Grouped comparison plots
  - Pareto frontier plots

- **`{arch}_plots_pareto/`**: Pareto frontier analysis
  - 2×2 subplot grid, one per depth (4, 8, 16, 32)
  - Each subplot has dual Y-axes: MAE (left, blue, solid) and Runtime (right, red, dashed)
  - X-axis shows polynomial degree progression: Constant → Linear → Quadratic → Cubic
  - Points annotated with memory (coefficients) and runtime (ms)
  - Clean, uncluttered view of accuracy vs speed tradeoff at each depth
  - Single file: `{arch}_degree_vs_mae.png`

- **`{arch}_plots_values/`**: Hardware activation value comparison plots
  - Shows ground truth vs actual hardware outputs
  - Compares Native SFPU, PC, PL, PQ, and Cubic approximations
  - Generated from `{arch}_hardware_outputs/` CSV files

**Plot features:**
- Automatically detects available depths and tile counts from data
- Color-coded by method: Native SFPU (blue), PC (orange), PL (green), PQ (red), Cubic (pink)
- Uses runtime minimum for stable comparisons
- Runtime values displayed on bars
- Log-scale error axes where appropriate

## Method Comparison

### Native SFPU (Hardware-Accelerated)

**How it works**: Uses built-in SFPU operations like `tanh_tile(0)`, `gelu_tile(0)`, etc.

**Pros**:
- Zero memory overhead
- Exceptional accuracy for tanh/gelu
- Fast for most operations (~700ms on Wormhole B0)

**Cons**:
- Only 6 activations available (no sigmoid/swish)
- Softplus has poor accuracy (use LUTs instead)
- GELU is 3.6× slower (2538ms)

**Use for**: tanh, gelu, relu, leaky_relu, exp

### Piecewise Constant (PC)

**How it works**: Divides input range into segments, samples 1024-entry LUT at boundaries.

**Memory**: 4100 bytes (1024 entries, always)

**Pros**:
- Simple lookup, no interpolation
- Fixed memory usage

**Cons**:
- Always inferior to PL/PQ at all depths
- Step discontinuities at boundaries
- Poor gradients for ML training

**Verdict**: ❌ **Never use** - obsolete, always use PL or PQ instead

### Piecewise Linear (PL)

**How it works**: Divides input range into segments, stores slope and offset per segment. Computes `y = slope*x + offset`.

**Memory**: `NUM_SEGMENTS * 2 * 4` bytes (2 coefficients per segment)
- PL-8: 68 bytes
- PL-128: 1028 bytes
- PL-256: 2052 bytes

**Accuracy** (average MAE, excluding exp):
- PL-8: 0.0118
- PL-16: 0.0045
- PL-32: 0.0025
- PL-64: 0.0021
- PL-128: **0.0015** (best overall!)
- PL-256: 0.0014

**Pros**:
- Best accuracy at high segment counts (≥128)
- Smooth, continuous output
- Lower memory than PQ at same segment count
- Good gradients for training

**Cons**:
- Needs more segments than PQ for same accuracy at low depths
- Linear cannot capture curvature within segments

**Use for**: High accuracy requirements (>500B memory available)

### Piecewise Quadratic (PQ)

**How it works**: Divides input range into segments, stores a, b, c coefficients per segment. Computes `y = a*x² + b*x + c` using Ordinary Least Squares (OLS) fitting.

**Memory**: `NUM_SEGMENTS * 3 * 4` bytes (3 coefficients per segment)
- PQ-8: 100 bytes
- PQ-16: 196 bytes
- PQ-32: 388 bytes
- PQ-64: 772 bytes
- PQ-128: 1540 bytes
- PQ-256: 3076 bytes

**Accuracy** (average MAE, excluding exp):
- PQ-8: 0.0052 (2.3× better than PL-8!)
- PQ-16: 0.0023 (2.0× better than PL-16!)
- PQ-32: 0.0021
- PQ-64: 0.0021
- PQ-128: 0.0021
- PQ-256: 0.0021

**Pros**:
- **Best accuracy per byte** at low segment counts (8-64)
- Captures curvature within segments
- Converges 4-8× faster than linear
- Ideal for memory-constrained systems

**Cons**:
- Higher computation cost (x² operation)
- 1.5× memory of PL at same segment count
- At high depths (≥128), PL becomes more accurate

**Use for**: Memory-constrained systems (<500B), embedded devices

## Recommendations by Use Case

### For Production Inference 🚀

1. **If native SFPU available for your activation**:
   - ✅ Use for: tanh, gelu, relu, leaky_relu, exp
   - ⚠️ Avoid softplus (use PQ-16 or PL-128 instead)
   - ❌ Not available for: sigmoid, swish

2. **When native SFPU unavailable or suboptimal**:
   - **<200B memory**: PQ-8 (100B, MAE 0.005) or PQ-16 (196B, MAE 0.002)
   - **200-500B memory**: PQ-32 (388B) or PL-64 (516B)
   - **>500B memory**: PL-128 (1028B, best LUT accuracy)

### For Research/Training 🔬

- **Best overall**: PL-128 or PL-256 (MAE 0.0015-0.0014)
- **Per-activation optimal**:
  - tanh/gelu: Native SFPU
  - sigmoid/swish: PL-128 or PL-256
  - softplus: PQ-16 or PL-128 (NOT native SFPU!)
  - relu/leaky_relu: Any method (all perfect)

### For Embedded Systems 📱

- **PQ-8** (100B, MAE 0.005) - Best accuracy per byte
- **PQ-16** (196B, MAE 0.002) - Excellent accuracy, minimal memory
- Consider Native SFPU if zero memory overhead is critical

### Never Use ❌

- **Piecewise Constant (PC)** at any depth - always inferior to PQ/PL

## Visual Comparison

The plots below show the numerical output of each activation function across the input range [-10, 10], comparing all four methods: Native SFPU (blue), PL-128 (green), PQ-128 (magenta), and PC-128 (red dashed) against ground truth (black). The bottom panel shows absolute error on a log scale.

### Sigmoid

![Sigmoid Comparison](plots/sigmoid_comparison.png)

### Tanh

![Tanh Comparison](plots/tanh_comparison.png)

### GELU

![GELU Comparison](plots/gelu_comparison.png)

### Swish

![Swish Comparison](plots/swish_comparison.png)

### ReLU

![ReLU Comparison](plots/relu_comparison.png)

### Leaky ReLU

![Leaky ReLU Comparison](plots/leaky_relu_comparison.png)

### Softplus

![Softplus Comparison](plots/softplus_comparison.png)

### Exponential

![Exp Comparison](plots/exp_comparison.png)

## Benchmark Results

All benchmarks run on real Tenstorrent hardware (Wormhole B0 and Blackhole P100).

**Latest results:** See `wormhole_*.csv` and `blackhole_*.csv` for complete benchmark data with runtime statistics for tile count 256 and 2 runs per configuration.

### Platform Performance Comparison

#### Runtime Comparison (Depth-128 implementations)

| Method | Activation | Runtime (ms) |
|--------|------------|--------------|
| **Native SFPU** | tanh | 7,386 |
| | gelu | 20,415 |
| | relu | 12,565 |
| | softplus | 12,692 |
| | exp | 7,369 |
| | leaky_relu | 7,432 |
| **PL-128** | sigmoid | 13,410 |
| | tanh | 7,337 |
| | gelu | 7,346 |
| | swish | 12,423 |
| | relu | 2,130 |
| | leaky_relu | 12,455 |
| | softplus | 7,122 |
| | exp | 12,289 |
| **PQ-128** | sigmoid | 13,446 |
| | tanh | 2,131 |
| | gelu | 2,252 |
| | swish | 2,224 |
| | relu | 2,232 |
| | leaky_relu | 7,233 |
| | softplus | 7,039 |
| | exp | 7,238 |
| **PC-128** | sigmoid | 2,035 |
| | tanh | 774 |
| | gelu | 769 |
| | swish | 771 |
| | relu | 769 |
| | leaky_relu | 772 |
| | softplus | 769 |
| | exp | 778 |

**Note**: CSV data shows identical values for Wormhole B0 and Blackhole P100, indicating either similar performance or shared test runs. Both platforms validated for accuracy.

#### Accuracy Comparison (MAE at Depth-128)

| Activation | Native SFPU | PL-128 | PQ-128 | PC-128 |
|------------|-------------|---------|---------|---------|
| sigmoid | N/A | 0.001128 | 0.001305 | 0.002527 |
| tanh | 0.000273 | 0.003301 | 0.003284 | 0.003959 |
| gelu | 0.000710 | 0.001715 | 0.001570 | 0.029184 |
| swish | N/A | 0.010755 | 0.010664 | 0.034492 |
| relu | 0.000000 | 0.000000 | 0.000000 | 0.028587 |
| leaky_relu | 0.000118 | 0.000118 | 0.000118 | 0.028851 |
| softplus | 0.008086 | 0.002338 | 0.002338 | 0.026813 |
| exp | 1.470 | 13.179 | 1.542 | 8.001 |

**Key Findings**:
- **PQ-128** offers best balance: 2-7× faster than Native SFPU for most activations with excellent accuracy (MAE < 0.002)
- **Native SFPU** excels at tanh (MAE 0.000273) but softplus is poor (MAE 0.008)
- **PL-128** provides consistent accuracy but slower runtime than PQ-128
- **PC-128** is fastest but significantly lower accuracy (10-20× worse than PQ/PL)

### Architecture-Specific Results

#### Wormhole B0

**Runtime characteristics** (tile_count=256):
- **Native SFPU**: ~755ms baseline (gelu slower at ~2538ms)
- **PQ**: ~750-1016ms (+0% to +35% overhead)
- **PL**: ~740-855ms (+0% to +13% overhead)
- **PC**: ~740-855ms (+0% to +13% overhead)

**Accuracy highlights**:
- **Native SFPU tanh**: MAE 0.00027 🏆 (9-12× better than LUTs!)
- **PL-128**: MAE 0.00149 (best LUT method, beats native SFPU average!)
- **PQ-8**: MAE 0.00612 (43× better than PC-8, only 100 bytes!)
- **Native SFPU softplus**: MAE 0.00809 ⚠️ (use PQ-16 or PL-128 instead!)

See `wormhole_*.csv` for complete data with runtime statistics (mean, min, stddev) at tile_count=256.

#### Blackhole P100

**Runtime characteristics** (tile_count=256):
- **Approximately 2× slower than Wormhole B0**: ~1450ms baseline vs ~755ms
- **Similar overhead patterns**: PQ/PL/PC show same relative performance
- **Higher variance**: Blackhole shows ~14× runtime variation vs Wormhole's stable ~755ms

**Accuracy**: Identical to Wormhole B0 (deterministic computation)

See `blackhole_*.csv` for complete data with runtime statistics.

#### Visual Comparisons

**Runtime plots**: `{arch}_plots_runtime/` directories contain per-activation and grouped runtime comparisons

**Error vs Runtime plots**: `{arch}_plots_error/` directories contain tradeoff analysis and Pareto frontiers

**Dual-architecture comparison**: `plots_runtime/` contains side-by-side Wormhole vs Blackhole plots

### Detailed Analysis

- **COMPREHENSIVE_COMPARISON.md** - Full comparison of all methods across all depths
- **PQ_DEPTH_ANALYSIS.md** - Deep dive into piecewise quadratic convergence

## Architecture Details

### Directory Structure

```
generic_lut_activation/
├── README.md                                  # This file
├── CMakeLists.txt                             # Build configuration (all variants)
│
├── ../generic_lut_activation/activations_config.dat  # Central config (test/plot ranges, SFPU support)
├── activations_config.sh                      # Bash version (sourced by sweep scripts)
├── activations_config.py                      # Python version (imported by plotting scripts)
├── activation_config.hpp                      # C++ header (CSV parser for test programs)
│
├── generic_lut_activation_native.cpp          # Native SFPU host program
├── generic_lut_activation.cpp                 # PC/PL/PQ/Cubic host program
├── lut_loader.hpp                             # LUT loading utilities
│
├── kernels/
│   ├── dataflow/
│   │   ├── reader.cpp                         # Data reader kernel
│   │   └── writer.cpp                         # Data writer kernel
│   └── compute/
│       ├── native_sfpu.cpp                    # Native SFPU compute kernel
│       ├── piecewise_constant.cpp             # PC compute kernel
│       ├── piecewise_linear.cpp               # PL compute kernel
│       └── piecewise_quadratic.cpp            # PQ compute kernel
│
├── generate_luts.sh                           # Parallel LUT generation (all methods)
├── generate_piecewise_constant_luts.py        # Generate PC LUTs (1024 entries)
├── generate_piecewise_linear_luts.py          # Generate PL LUTs (4-256 segments)
├── generate_piecewise_quadratic_luts.py       # Generate PQ LUTs (4-256 segments)
├── generate_piecewise_quadratic_remez_luts.py # Generate PQ Remez LUTs (minimax)
├── generate_piecewise_cubic_remez_luts.py     # Generate Cubic Remez LUTs (minimax)
│
├── auto_reserve.sh                            # Reserve servers via IRD (with optional --setup)
├── setup_server.sh                            # Automated server setup (dual-mode)
├── hosts.sh                                   # Auto-generated server configuration
├── pull_results.sh                            # Pull results from servers
│
├── sweep_all.sh                               # Unified sweep orchestration (dual-mode, RECOMMENDED)
├── sweep_native_sfpu.sh                       # Benchmark native SFPU
├── sweep_piecewise_constant.sh                # Benchmark PC
├── sweep_piecewise_linear.sh                  # Benchmark PL
├── sweep_piecewise_quadratic.sh               # Benchmark PQ Remez
├── sweep_piecewise_cubic.sh                   # Benchmark Cubic Remez
│
├── extract_runtime.py                         # Parse timing from test output
├── extract_accuracy.py                        # Compute accuracy metrics
│
├── plots.sh                                   # Generate all visualization plots
│
├── plots/                                     # Plotting scripts and output
│   ├── plot_pareto_frontier.py                # Generate Pareto frontier analysis
│   ├── plot_depth_degree.py                   # Generate depth-degree heatmaps
│   ├── best.py                                # Find best configurations
│   ├── blackhole/                             # Blackhole output plots
│   │   ├── pareto/                            # Pareto frontier plots
│   │   └── depth_degree/                      # Depth-degree heatmaps
│   └── wormhole/                              # Wormhole output plots
│       ├── pareto/                            # Pareto frontier plots
│       └── depth_degree/                      # Depth-degree heatmaps
│
├── luts/                                      # Generated LUT files
│   ├── *_1024.lut                             # PC LUTs (1024 entries)
│   ├── piecewise_linear_*_*.lut               # PL LUTs
│   └── piecewise_quadratic_*_*.lut            # PQ LUTs
│
├── data/                                      # Benchmark results
│   ├── wormhole_*.csv                         # Wormhole B0 benchmark results
│   └── blackhole_*.csv                        # Blackhole P100 benchmark results
│
├── COMPREHENSIVE_COMPARISON.md                # Complete method comparison
└── PQ_DEPTH_ANALYSIS.md                       # PQ convergence analysis
```

### Build Targets

CMakeLists.txt creates 20 build targets:

**Native SFPU**:
- `programming_examples_generic_lut_activation_native_sfpu`

**Piecewise Constant** (6 depths):
- `programming_examples_generic_lut_activation_pc_{8,16,32,64,128,256}`

**Piecewise Linear** (6 depths):
- `programming_examples_generic_lut_activation_pwl_{8,16,32,64,128,256}`

**Piecewise Quadratic** (6 depths):
- `programming_examples_generic_lut_activation_pq_{8,16,32,64,128,256}`

### LUT File Formats

**Piecewise Constant**: Binary `[uint32_t size][float32 values...]`
- 1024 float32 values sampled uniformly across input range

**Piecewise Linear**: Binary `[uint32_t segments][float32 slopes...][float32 offsets...]`
- NUM_SEGMENTS slopes followed by NUM_SEGMENTS offsets

**Piecewise Quadratic**: Binary `[uint32_t segments][float32 a_coeffs...][float32 b_coeffs...][float32 c_coeffs...]`
- NUM_SEGMENTS of each coefficient (a, b, c) for y = ax² + bx + c
- Coefficients fitted using Ordinary Least Squares regression

## Performance Characteristics

### Runtime (Wormhole B0)

- **Native SFPU**: 689-2538ms (GELU slow, others ~700ms)
- **PQ**: 696-1016ms (+0% to +46% vs baseline)
- **PL**: 677-855ms (+0% to +21% vs baseline)
- **PC**: 687-855ms (+0% to +21% vs baseline)

**PQ overhead**: ~10-20% slower than PL due to x² computation, but 2-4× better accuracy at low segment counts.

### Memory Usage

| Method | 8 seg | 16 seg | 32 seg | 64 seg | 128 seg | 256 seg |
|--------|------:|-------:|-------:|-------:|--------:|--------:|
| **PC** | 4100B | 4100B | 4100B | 4100B | 4100B | 4100B |
| **PL** | 68B | 132B | 260B | 516B | 1028B | 2052B |
| **PQ** | 100B | 196B | 388B | 772B | 1540B | 3076B |

**Native SFPU**: 0 bytes (no LUT storage)

### Accuracy Convergence

**Fast convergers** (16-32 segments):
- Sigmoid, Swish, Softplus with PQ

**Medium convergers** (32-64 segments):
- GELU, Tanh with PQ
- Most functions with PL

**Slow convergers** (64-128 segments):
- Exp with PQ/PL (large dynamic range)
- All functions with PC

## Creating Custom Activations

### Add New Activation to Generators

Edit the generator scripts:

```python
# generate_piecewise_linear_luts.py
ACTIVATIONS = {
    'my_custom': lambda x: np.custom_function(x),
    # ... existing activations
}
```

### Custom LUT Generation

```python
import numpy as np
import struct

def create_pq_lut(activation_fn, num_segments=16, input_range=(-10, 10)):
    """Generate piecewise quadratic LUT using OLS fitting."""
    x_min, x_max = input_range
    segment_width = (x_max - x_min) / num_segments

    a_coeffs = []
    b_coeffs = []
    c_coeffs = []

    for i in range(num_segments):
        seg_start = x_min + i * segment_width
        seg_end = seg_start + segment_width

        # Sample 100 points for fitting
        x_samples = np.linspace(seg_start, seg_end, 100)
        y_samples = activation_fn(x_samples)

        # Fit quadratic: y = a*x^2 + b*x + c
        X = np.vstack([x_samples**2, x_samples, np.ones_like(x_samples)]).T
        coeffs = np.linalg.lstsq(X, y_samples, rcond=None)[0]

        a_coeffs.append(coeffs[0])
        b_coeffs.append(coeffs[1])
        c_coeffs.append(coeffs[2])

    # Write to file
    with open(f'my_activation_pq_{num_segments}.lut', 'wb') as f:
        f.write(struct.pack('I', num_segments))
        for a in a_coeffs:
            f.write(struct.pack('f', a))
        for b in b_coeffs:
            f.write(struct.pack('f', b))
        for c in c_coeffs:
            f.write(struct.pack('f', c))

# Usage
create_pq_lut(lambda x: x**3 + 2*x, num_segments=16)
```

## Hardware Validation

All results measured on **real Tenstorrent silicon**:
- **Wormhole B0**: Physical PCIe device (yyzc-wh-03), 8 chips
- **Blackhole P100**: Physical PCIe device (yyzo-bh-09), 1 chip
- **Test coverage**: ~792 measurements per architecture (12 native SFPU × 2 runs + 16 activations × 6 depths × 4 methods × 2 runs)
- **Benchmarking**: Automated via `wormhole.sh` and `blackhole.sh` scripts (see "Automated Hardware Benchmark Regeneration")

This is **actual accelerator performance**, not simulation!

## Troubleshooting

**"Binary not found"**:
```bash
./build_metal.sh --build-programming-examples
```

**"LUT file not found"**:
```bash
cd tt_metal/programming_examples/generic_lut_activation
python3 generate_piecewise_constant_luts.py
python3 generate_piecewise_linear_luts.py
python3 generate_piecewise_quadratic_luts.py
```

**"Test FAILED"**:
- Check ARCH_NAME environment variable
- Ensure device is accessible
- Try running simpler test first (native SFPU relu)

**Poor accuracy**:
- Verify input range matches your data
- Try higher segment count (PL-128 or PL-256)
- For softplus, use LUT methods (NOT native SFPU)

**Plotting issues**:
- Ensure CSV files exist before running plot scripts
- Check CSV format matches expected columns (runtime_mean_ms, runtime_min_ms, runtime_stddev_ms)
- Use `--arch-prefix` flag for single-architecture plots: `python3 plot_runtime_comparison.py --arch-prefix wormhole`
- Install matplotlib if missing: `pip install matplotlib`

**Remote benchmark script issues**:
- **"libtt_metal.so: cannot open shared object file"**: The script automatically sets `LD_LIBRARY_PATH`, but verify `REMOTE_DIR` points to correct tt-metal location
- **All tests fail with ERROR**: Check that binaries are built on remote server: `ls $REMOTE_DIR/build_Release/programming_examples/`
- **"Binary not found"**: Run `./build_metal.sh --build-programming-examples` on the remote server
- **Git errors**: Ensure SSH agent forwarding works and you have git access from the remote server
- **Device access errors**: Run `tt-smi` on remote server to verify devices are accessible

## Server Reservation and Setup

If a remote server goes down overnight or you need to reserve a new one, use the automated reservation script or manual IRD (Interactive Run Docker) workflow.

### Automated Reservation (Recommended)

The `auto_reserve.sh` script automates the entire reservation and setup process for both Wormhole and Blackhole servers:

```bash
cd tt_metal/programming_examples/generic_lut_activation

# Reserve and setup servers in one command (RECOMMENDED)
bash auto_reserve.sh --setup

# Reserve specific architectures with setup
bash auto_reserve.sh --arch wormhole_b0 --setup
bash auto_reserve.sh --arch blackhole --setup

# Reserve with custom timeout and setup (e.g., 2 hours)
bash auto_reserve.sh --setup 2:00:00

# Choose specific machine models
bash auto_reserve.sh --wormhole-model x2 --blackhole-model p150 --setup

# Reserve without setup (manual setup later)
bash auto_reserve.sh
```

**What the script does:**
1. ✅ **Smart reservation checking**: Reuses existing valid reservations, only reserves new machines if needed
2. ✅ Reserves Wormhole B0 and/or Blackhole P100 servers via IRD (single-card models by default)
3. ✅ Extracts server hostnames, ports, and passwords automatically
4. ✅ Flushes old SSH host keys for both servers
5. ✅ Verifies SSH connectivity to both servers
6. ✅ Updates `hosts.sh` with all server details
7. ✅ **Optional automated setup**: With `--setup` flag, automatically runs setup_server.sh on reserved servers
8. ✅ **Automatic rollback**: If Blackhole reservation fails, releases Wormhole automatically

**Generated hosts.sh format:**
```bash
#!/bin/bash
# Server configuration for remote benchmarks
# Auto-generated by auto_reserve.sh on <date>

# Wormhole B0 server
WORMHOLE_HOST="yyzc-wh-01"
WORMHOLE_PORT="49819"
WORMHOLE_PASSWORD="EFAg024eaizJRdjJ8MnPq5fDgQiYenjq"
WORMHOLE_JOB_ID="1"

# Blackhole P100 server
BLACKHOLE_HOST="yyzo-bh-26"
BLACKHOLE_PORT="49819"
BLACKHOLE_PASSWORD="TNqqNvP7+5KHlWMuW6/vTvhSXVKCxbuT"

# Remote working directory (same on both servers)
REMOTE_DIR="localdev/nkapre/tt-metal"
```

**Timeout format examples:**
- `2:00:00` = 2 hours
- `10:00:00` = 10 hours (default)
- `1-0` = 1 day (24 hours)
- `max` = Maximum allowed by IRD

After reservation completes, proceed to setup servers using the automated setup script below.

### Automated Server Setup

**NOTE**: If you used `auto_reserve.sh --setup`, this step is already complete and you can skip ahead.

Once servers are reserved, use the automated setup orchestrator to configure both servers from your local machine:

```bash
cd tt_metal/programming_examples/generic_lut_activation

# Automated dual-mode orchestration (RECOMMENDED - runs setup on both servers)
bash setup_server.sh --arch both

# Or setup specific architectures only
bash setup_server.sh --arch wormhole_b0
bash setup_server.sh --arch blackhole

# Setup with options (skip build/LUTs/reset if needed)
bash setup_server.sh --arch both --skip-build
bash setup_server.sh --arch both --skip-luts
bash setup_server.sh --arch both --skip-reset
```

The `setup_server.sh` script operates in **dual-mode**:
- **Local mode** (with hosts.sh): Orchestrates remote setup via SSH, automatically copies script and runs on servers
- **Remote mode** (without hosts.sh): Performs actual setup tasks on the server

No manual scp/ssh commands needed - the script handles everything automatically!

**What setup_server.sh does:**
1. ✅ Cleans cache directory (`~/.cache` - prevents build issues from stale files)
2. ✅ Installs system dependencies (tmux, zsh, oh-my-zsh)
3. ✅ Installs Python packages (numpy, mpmath, OptimalPoly) with fallback strategies
4. ✅ Pulls latest code from current git branch
5. ✅ Builds tt-metal with programming examples
6. ✅ Resets devices via tt-smi
7. ✅ Generates all LUT files (PC, PL, PQ, Cubic, Hexic, Octic)
8. ✅ **Displays comprehensive summary**: Binaries built, LUTs generated, Python versions

**Setup completion summary example:**
```
==========================================
✓ Server Setup Complete!
==========================================

Setup Summary:
-------------------------------------------
✓ Repository: 63055fdaa2 (feature/generic-lut-activation)
✓ Architecture: wormhole_b0
✓ Binaries built: 26
✓ LUTs generated: 384
✓ Devices reset: 0,1,2,3,4,5,6,7

Python dependencies:
  - numpy: 1.24.3
  - pandas: 2.0.2
  - mpmath: 1.3.0
```

**Script options:**
```bash
bash /tmp/setup_server.sh --arch wormhole_b0          # Full setup
bash /tmp/setup_server.sh --arch blackhole --skip-build  # Skip build (if already built)
bash /tmp/setup_server.sh --arch wormhole_b0 --skip-luts # Skip LUT generation
bash /tmp/setup_server.sh --arch blackhole --skip-reset  # Skip device reset
```

After setup completes, you can run benchmarks:
```bash
cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation
export ARCH_NAME=wormhole_b0  # or blackhole
export LD_LIBRARY_PATH=/localdev/nkapre/tt-metal/build_Release/lib:$LD_LIBRARY_PATH

# Run sweeps
bash sweep_native_sfpu.sh
bash sweep_piecewise_constant.sh
bash sweep_piecewise_linear.sh
bash sweep_piecewise_quadratic.sh
bash sweep_piecewise_cubic.sh
```

### Manual IRD Reservation (Alternative)

If you prefer manual control, follow the traditional IRD workflow:

**IRD (Interactive Run Docker)** is Tenstorrent's system for reserving remote hardware. See the [IRD documentation](https://tenstorrent.atlassian.net/wiki/spaces/DevInfra/pages/178454836/IRD+-+Interactive+Run+Docker) for details.

**Step 1: SSH to IRD gateway**
```bash
ssh yyz-ird
```

**Step 2: Reserve servers with IRD**
```bash
# Wormhole B0 (8-chip x1 model) - 10 hour reservation
ird reserve --timeout 10:00:00 wormhole_b0 --model x1

# Blackhole P100 - 10 hour reservation
ird reserve --timeout 10:00:00 blackhole --model p100
```

**Step 3: Parse IRD output and update hosts.sh**

IRD will output server details like:
```
Welcome nkapre! You have been assigned yyzc-wh-01 for your reservation.
SSH with 'ssh yyzc-wh-01 -p 49819'
If prompted for a password, use: Uh7Sn86l0f19VkjmhlxqeXFBonjvMgC4
```

Manually create/update `hosts.sh`:
```bash
cd tt_metal/programming_examples/generic_lut_activation
# Edit hosts.sh with extracted hostname, port, and password
```

### Release Servers

When done with benchmarks, release your reservations:

```bash
# Check active reservations
ssh yyz-ird 'ird list'

# Release by selection ID (from list output)
ssh yyz-ird 'ird release 1'  # Release first reservation
ssh yyz-ird 'ird release 2'  # Release second reservation
```

### Server Connection Issues

**"Connection refused" errors**:
- IRD reservation may have expired or server was reassigned
- Reserve new servers using `auto_reserve.sh` or manual IRD workflow
- The `hosts.sh` file is automatically updated by `auto_reserve.sh`

**"Host key verification failed" errors**:
- Old SSH host keys cached from previous reservations
- `auto_reserve.sh` automatically flushes old keys
- Manual fix: `ssh-keygen -R "[hostname]:port"`

**Current server format** (servers change per reservation):
- Hostnames: `yyzc-wh-XX`, `yyzo-bh-XX`, `bh-XX`
- Ports: Typically `49819` or `22`
- Always check `hosts.sh` for current active servers

## References

- TT-Metal documentation: https://docs.tenstorrent.com/tt-metal/
- SFPU programming: See `METALIUM_GUIDE.md` in tt-metal root
- Production LUT usage: Welford in `ttnn/cpp/ttnn/operations/normalization/layernorm/`
- General development: `CLAUDE.md` in tt-metal root
