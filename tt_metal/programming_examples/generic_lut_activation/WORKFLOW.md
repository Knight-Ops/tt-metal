# LUT Activation Function Workflow

Complete workflow for generating LUTs, running benchmarks, and creating visualizations.

## Directory Structure

```
generic_lut_activation/
├── data/                           # All experimental result CSVs and hardware outputs
│   ├── wormhole_*.csv              # Wormhole result files
│   ├── blackhole_*.csv             # Blackhole result files
│   ├── wormhole_hardware_outputs/  # Raw hardware output CSVs
│   └── blackhole_hardware_outputs/ # Raw hardware output CSVs
├── luts/                           # LUT files and generators
│   ├── *.lut                       # Generated LUT files
│   └── *.py                        # LUT generation scripts
├── plots/                          # All plotting scripts and outputs
│   ├── *.py                        # Plotting scripts
│   ├── wormhole/                   # Wormhole architecture plots
│   ├── blackhole/                  # Blackhole architecture plots
│   └── output/                     # Architecture-independent plots
└── *.sh                            # Main workflow scripts
```

## Workflow Steps

The workflow is organized into 4 phases:
1. **One-time Setup** (steps 1-2) - Run once at the start
2. **LUT Generation** (steps 3-4) - Run once, or when fixing LUT generation
3. **Hardware Benchmarking** (step 5) - Main iteration loop after fixing LUTs/C++/kernels
4. **Post-processing** (steps 6-7) - After benchmarks complete

---

## Phase 1: One-Time Setup

### 1. Reserve/Connect to Server (Optional - if using remote hardware)
```bash
./auto_reserve.sh
```

### 2. Setup Server Environment
```bash
./setup_server.sh
```
Sets up the build, environment variables, and copies files to remote server.

**Run once at project start.**

---

## Phase 2: LUT Generation (One-time unless fixing generation)

### 3. Generate LUT Files (Local)
```bash
./generate_luts.sh
```
**Options:**
- `--activation <name>` - Generate for specific activation only
- `--segmentation <type>` - uniform | adaptive (default: both)
- `--degree <type>` - constant | linear | quadratic | cubic | hexic | octic
- `--depth <depths>` - Comma-separated depths (e.g., '4,8' or '8')

**Examples:**
```bash
./generate_luts.sh                                    # All LUTs
./generate_luts.sh --activation sigmoid               # Sigmoid only
./generate_luts.sh --degree cubic --depth 8           # Cubic depth 8 only
./generate_luts.sh --activation digamma --depth 4,8   # Digamma depths 4,8
```

**Output:** `luts/*.lut` files

**Run once, or when fixing LUT generation algorithms.**

### 4. Theoretical Error Analysis (Optional - Local)
```bash
./theoretical_error_analysis.sh
```
Generates theoretical error metrics and plots by evaluating LUTs against ground truth.

**Options:**
- `--activation <name>` - Single activation (e.g., sigmoid, gelu, tanh)
- `--depth <depths>` - Comma-separated depths (e.g., '4,8,16' or '16')
- `--degree <type>` - linear | quadratic | cubic | hexic | octic
- `--simulate-bf16` - Simulate BF16 quantization (hardware-accurate)
- `--samples <N>` - Number of test samples (default: 100000)
- `--compare-adaptive` - Compare uniform vs adaptive segmentation
- `--arch <name>` - Architecture for plot output (wormhole|blackhole, default: output)
- `--skip-plots` - Skip plot generation (CSV only)

**Examples:**
```bash
./theoretical_error_analysis.sh                                    # All configs, FP32
./theoretical_error_analysis.sh --simulate-bf16                    # All configs, BF16
./theoretical_error_analysis.sh --activation sigmoid --simulate-bf16   # Sigmoid, BF16
./theoretical_error_analysis.sh --degree cubic --depth 8,16 --simulate-bf16
./theoretical_error_analysis.sh --arch wormhole                    # Output to plots/wormhole/
```

**Output:**
- CSV: `plots/{arch}/theoretical_error_summary_{bf16|fp32}.csv`
- Plots: `plots/{arch}/*.png`

**Run once after generating LUTs, or when fixing LUT generation.**

---

## Phase 3: Hardware Benchmarking (Main iteration loop)

### 5. Run Hardware Benchmarks (On Server)
```bash
./sweep_all.sh
```
Runs all hardware benchmarks on the connected server:
- Native SFPU
- Piecewise constant
- Piecewise polynomial (linear, quadratic, cubic, hexic, octic)

**Options:**
- `--depth <depths>` - Comma-separated depths (e.g., '4,8')
- `--activation <name>` - Single activation only
- `--quick-run` - Quick test mode

**Output:** Result CSVs and hardware_outputs/ directory on the server

**Run this after:**
- Fixing C++ host code
- Fixing kernel implementations
- Regenerating LUTs (if testing new approximations)

---

## Phase 4: Post-processing

### 6. Pull Results from Server
```bash
./pull_results.sh
```
Downloads result CSVs and hardware outputs from remote server to local `data/` directory.

**Options:**
- `--arch <arch>` - Filter by architecture (wormhole|blackhole)
- `--precision <prec>` - Filter by precision (bf16|fp32)
- `--degree <degree>` - Filter by polynomial degree (linear|quadratic|cubic|hexic|octic)
- `--activation <name>` - Filter by activation function (sigmoid|gelu|tanh|exp|etc.)
- `--depth <depth>` - Filter by LUT depth (4|8|16|32)
- `--segmentation <seg>` - Filter by segmentation (uniform|adaptive)
- `--status` - Show status without downloading hardware outputs

**Notes:**
- `--degree`, `--precision`: Apply to both CSV results and hardware outputs
- `--activation`, `--depth`, `--segmentation`: Apply only to hardware outputs

**Examples:**
```bash
./pull_results.sh                                      # Pull everything
./pull_results.sh --status                             # Quick check (CSV only)
./pull_results.sh --arch wormhole --precision fp32     # Wormhole FP32 results
./pull_results.sh --degree cubic --status              # Quick cubic status
./pull_results.sh --activation sigmoid --depth 8       # Sigmoid depth-8 hardware outputs
./pull_results.sh --degree cubic --segmentation adaptive  # Cubic adaptive results
```

**Output:** `data/` directory populated with CSV files and hardware_outputs/

### 7. Generate All Plots
```bash
./plots.sh
```
Generates all visualization plots from the collected data.

**Options:**
- `--arch <arch>` - wormhole | blackhole | both (default: both)
- `--activation <name>` - Filter plots by activation function
- `--degree <degree>` - Filter plots by polynomial degree (linear|quadratic|cubic|hexic|octic)
- `--depth <depth>` - Filter plots by LUT depth (4|8|16|32)
- `--segmentation <seg>` - Filter plots by segmentation (uniform|adaptive)
- `--precision <prec>` - Filter plots by precision (bf16|fp32)

**Smart Skip Logic:**
- **No filters**: Run all plots
- **Degree filter**: Skip degree comparison grid (no point comparing with one degree)
- **Very specific filters** (activation+degree+depth): Only run theoretical error analysis

The script shows an execution plan upfront listing what will/won't be run based on filters.

**Examples:**
```bash
./plots.sh                                          # All plots, all data
./plots.sh --arch wormhole                          # Wormhole plots only
./plots.sh --degree cubic                           # Skip degree comparison grid
./plots.sh --activation sigmoid                     # All plots for sigmoid
./plots.sh --activation sigmoid --degree cubic --depth 8  # Only theoretical error for this config
```

**Output:**
- `plots/wormhole/` - Wormhole plots (runtime, error, pareto)
- `plots/blackhole/` - Blackhole plots (runtime, error, pareto)
- `plots/output/` - Architecture-independent plots (theoretical, comparison)

## Quick Reference

### Full Workflow (Local + Remote)
```bash
# ═══════════════════════════════════════════════════════════════
# PHASE 1: ONE-TIME SETUP (run once at start)
# ═══════════════════════════════════════════════════════════════
./auto_reserve.sh      # (Optional) Reserve server
./setup_server.sh      # Setup server environment

# ═══════════════════════════════════════════════════════════════
# PHASE 2: LUT GENERATION (run once, or when fixing LUT generation)
# ═══════════════════════════════════════════════════════════════
./generate_luts.sh                                # Generate all LUTs
./theoretical_error_analysis.sh --simulate-bf16   # (Optional) Theoretical analysis

# ═══════════════════════════════════════════════════════════════
# PHASE 3: HARDWARE BENCHMARKING (main iteration loop)
# ═══════════════════════════════════════════════════════════════
# Run after fixing C++/kernels or regenerating LUTs
./sweep_all.sh         # Run benchmarks on server

# ═══════════════════════════════════════════════════════════════
# PHASE 4: POST-PROCESSING (after benchmarks)
# ═══════════════════════════════════════════════════════════════
./pull_results.sh      # Download results to data/
./plots.sh             # Generate all visualizations
```

### Typical Iteration Cycle
```bash
# After fixing C++ or kernel code:
./sweep_all.sh         # Run benchmarks
./pull_results.sh      # Pull results
./plots.sh             # Update plots

# After fixing LUT generation:
./generate_luts.sh     # Regenerate LUTs
./sweep_all.sh         # Run benchmarks
./pull_results.sh      # Pull results
./plots.sh             # Update plots
```

### Minimal Quick Test (Single Config)
Test one specific configuration (fastest iteration):

```bash
# 1. Generate LUT for one config
./generate_luts.sh --activation sigmoid --degree cubic --depth 8

# 2. Run benchmark on server for that config
./sweep_piecewise.sh --degree cubic --activation sigmoid --depth 8

# 3. Pull just that result and plot
./pull_results.sh --activation sigmoid --degree cubic --depth 8
./plots.sh --arch wormhole
```

**With theoretical error analysis:**
```bash
# Add theoretical analysis before hardware benchmark
./generate_luts.sh --activation sigmoid --degree cubic --depth 8
./theoretical_error_analysis.sh --activation sigmoid --degree cubic --depth 8 --simulate-bf16
./sweep_piecewise.sh --degree cubic --activation sigmoid --depth 8
./pull_results.sh --activation sigmoid --degree cubic --depth 8
./plots.sh --arch wormhole
```

### Quick Test - Single Activation (All Degrees/Depths)
Test one activation with multiple configurations:

```bash
# 1. Generate all LUTs for one activation
./generate_luts.sh --activation sigmoid

# 2. (Optional) Theoretical analysis
./theoretical_error_analysis.sh --activation sigmoid --simulate-bf16

# 3. Run all benchmarks for that activation
./sweep_piecewise.sh --activation sigmoid

# 4. Pull results for that activation
./pull_results.sh --activation sigmoid
./plots.sh --arch wormhole
```

### Local-Only Workflow (Theoretical Analysis)
```bash
# 1. Generate LUTs
./generate_luts.sh

# 2. Analyze theoretical error
./theoretical_error_analysis.sh --simulate-bf16
```

### Individual Sweeps (Advanced)
```bash
# Run individual method sweeps
./sweep_native_sfpu.sh
./sweep_piecewise_constant.sh
./sweep_piecewise.sh --degree cubic
./sweep_piecewise.sh --degree cubic --precision fp32
./sweep_piecewise.sh --degree hexic --segmentation adaptive
```

## Output Files

### Data Directory (`data/`)
- `{arch}_native_sfpu_results.csv`
- `{arch}_piecewise_constant_results.csv`
- `{arch}_piecewise_linear_results.csv`
- `{arch}_piecewise_quadratic_remez_results.csv`
- `{arch}_piecewise_cubic_remez_results.csv`
- `{arch}_piecewise_hexic_remez_results.csv`
- `{arch}_piecewise_octic_remez_results.csv`
- `{arch}_hardware_outputs/*.csv` (raw kernel outputs)

### Plot Directories
- `plots/wormhole/` - Wormhole hardware plots (runtime, error, pareto) + theoretical (if `--arch wormhole`)
- `plots/blackhole/` - Blackhole hardware plots (runtime, error, pareto) + theoretical (if `--arch blackhole`)
- `plots/output/` - Architecture-independent plots (theoretical error, comparison)

**Note:** Theoretical plots from `theoretical_error_analysis.sh` go to:
- `plots/output/` by default
- `plots/{arch}/` when using `--arch` flag

## Troubleshooting

### LUT files not found
```bash
./generate_luts.sh  # Generate missing LUTs
```

### Pull results fails
```bash
./pull_results.sh --status  # Check what's available on server
```

### Plots show missing data
```bash
./pull_results.sh           # Ensure all CSVs are downloaded
ls data/                    # Verify CSV files exist
```

### BF16 simulation error
```bash
python3 -m pip install ml_dtypes  # Required for BF16 analysis
```
