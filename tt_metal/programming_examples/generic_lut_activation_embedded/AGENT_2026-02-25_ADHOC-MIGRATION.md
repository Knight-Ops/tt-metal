# AGENT: Test Adhoc Workflow Migration on Remote Server

**Date:** 2026-02-25
**Branch:** `feature/generic-lut-activation`
**Related:** `tt-polynomial-fitter/AGENT_2026-02-25_CSV-FORMAT-CHANGE-NEW-RECURSIVE-ULP-SEG.md` (incoming)

## Overview

This agent document describes how to test the adhoc workflow migration that eliminates the need for pre-compiled kernel binaries. The migration affects two directories:

| Directory | Before | After | Reduction |
|-----------|--------|-------|-----------|
| `generic_lut_activation_embedded` | 522+ kernels, 6851-line CMake | 4 adhoc targets | ~99% |
| `generic_lut_activation` | 205 targets (96 poly + 108 rational + 1 native) | 4 targets | ~98% |

## Adhoc Workflow Pattern

Instead of pre-compiling hundreds of binaries for every degree/segment combination:

```
OLD: compile 200+ binaries upfront (hours) → run binary for specific config
NEW: generate config header → ninja rebuild (~5-10s) → run binary
```

### Key Files

**Embedded version (`generic_lut_activation_embedded/`):**
- `tools/generate_adhoc_kernel.py` - Unified kernel generator (polynomial + rational)
- `kernels/compute/adhoc/adhoc.cpp` - Generated kernel (overwritten per config)
- `sweep_best.sh`, `sweep_polynomial.sh`, `sweep_rational.sh` - Use adhoc workflow

**Non-embedded version (`generic_lut_activation/`):**
- `adhoc_config.h` - Generated config for polynomial (POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE)
- `adhoc_rational_config.h` - Generated config for rational (NUM_DEGREE, DEN_DEGREE, etc.)
- `generic_lut_activation.cpp` - Uses `#ifdef ADHOC_MODE` to include config headers
- CMake targets: `_adhoc`, `_rational_adhoc`, `_native_sfpu`, default

## Remote Server Testing

### Prerequisites

1. Source server configuration:
   ```bash
   source hosts.sh
   ```

2. Connect to server (Wormhole or Blackhole):
   ```bash
   # Wormhole
   sshpass -p "$WORMHOLE_PASSWORD" ssh -p $WORMHOLE_PORT $WORMHOLE_HOST

   # Blackhole
   sshpass -p "$BLACKHOLE_PASSWORD" ssh -p $BLACKHOLE_PORT $BLACKHOLE_HOST
   ```

3. On remote server:
   ```bash
   cd /localdev/nkapre/tt-metal
   git fetch origin
   git checkout feature/generic-lut-activation
   git pull origin feature/generic-lut-activation
   ```

### Environment Setup

```bash
# On remote server
cd /localdev/nkapre/tt-metal
source python_env/bin/activate
export ARCH_NAME=wormhole_b0  # or blackhole

# Set polynomial fitter path
export TT_POLY_FIT_DIR=/localdev/nkapre/tt-polynomial-fitter

# Build (first time or after major changes)
./build_metal.sh --build-programming-examples
```

### Test 1: Embedded Adhoc Workflow

```bash
cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded

# Test kernel generation
python3 tools/generate_adhoc_kernel.py \
    --activation gelu \
    --degree 4 \
    --segments 16 \
    --segmentation chebyshev \
    --metric ulp \
    --output kernels/compute/adhoc/adhoc.cpp

# Verify generated kernel
head -50 kernels/compute/adhoc/adhoc.cpp

# Build adhoc target
ninja -C $BUILD_DIR programming_examples_generic_lut_activation_embedded_adhoc

# Run sweep (single activation)
./sweep_best.sh --activation gelu --precision bf16 --metric ulp

# Full sweep (all activations)
./sweep_best.sh --activation gelu --precision fp32 --metric max
./sweep_best.sh --activation sigmoid --precision bf16 --metric ulp
```

**Expected output:**
```
=== gelu (bf16, range: -4.0 to 4.0) ===
Config                    Coeffs          MAE       MaxErr       MaxULP      MeanULP   Prof(us)  Host(ms)
-------------------------  --------  ------------  ------------  ----------  ----------  ----------  ----------
single_tile_p4_s16_che          5     1.23e-04     4.56e-04       2.50        1.20     12.34us    0.89ms
8_tiles_p4_s16_che              5     1.24e-04     4.57e-04       2.51        1.21     45.67us    1.23ms
```

### Test 2: Non-Embedded Adhoc Workflow

The non-embedded version loads LUT data from CSV at runtime but still needs compile-time constants for POLY_DEGREE/NUM_SEGMENTS.

```bash
cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation

# Manually update config header (sweep scripts will automate this)
cat > adhoc_config.h << 'EOF'
#pragma once
#define ADHOC_POLY_DEGREE 4
#define ADHOC_NUM_SEGMENTS 16
#define ADHOC_LUT_SIZE 97  // (16+1) boundaries + 16*5 coefficients
EOF

# Build adhoc target
ninja -C $BUILD_DIR programming_examples_generic_lut_activation_adhoc

# Run with CSV input
$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_adhoc \
    $TT_POLY_FIT_DIR/data/coefficients/gelu_16_4_chebyshev_any_ulp.csv \
    --activation gelu \
    --precision bf16 \
    --range-min -4.0 \
    --range-max 4.0 \
    --tiles 256
```

### Test 3: Rational Approximations

```bash
cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded

# Generate rational kernel
python3 tools/generate_adhoc_kernel.py \
    --activation gelu \
    --num-degree 3 \
    --den-degree 2 \
    --segments 8 \
    --segmentation chebyshev \
    --metric ulp \
    --output kernels/compute/adhoc/adhoc.cpp

# Build and test
ninja -C $BUILD_DIR programming_examples_generic_lut_activation_embedded_adhoc

./sweep_rational.sh --activation gelu --precision fp32 --metric max
```

### Test 4: Verify No Regression

Compare adhoc results with pre-existing results (if available):

```bash
# Check existing results
ls -la data/hardware_outputs/*/gelu/

# Run sweep and compare timing/accuracy
./sweep_best.sh --activation gelu --precision bf16 --metric ulp

# Results should match within noise margin (~5%)
```

## Validation Checklist

- [ ] `generate_adhoc_kernel.py` generates valid kernel code
- [ ] `ninja` incremental rebuild completes in <15 seconds
- [ ] Polynomial sweep produces accurate results (compare to theoretical)
- [ ] Rational sweep produces accurate results
- [ ] Hardware profiler timing (`Prof(us)`) is reasonable
- [ ] Host timing (`Host(ms)`) matches previous runs
- [ ] ULP error metrics are within expected bounds
- [ ] No device hangs or timeouts

## CSV Format Changes (COMPLETE)

The CSV naming convention has changed. See:
`tt-polynomial-fitter/AGENT_2026-02-25_CSV-FORMAT-CHANGE-NEW-RECURSIVE-ULP-SEG.md`

### New Canonical Format

```
Polynomial: {activation}_p{degree}_s{segments}_{segmentation}_{fitting}_{metric}.csv
Rational:   {activation}_n{num}d{den}_s{segments}_{segmentation}_{fitting}_{metric}.csv
```

**Examples:**
```
gelu_p4_s32_uniform_any_ulp.csv              # OLD: gelu_32_4_uniform_any.csv
gelu_n5d4_s37_recursive_ulp_rminimax_ulp.csv # Rational with recursive_ulp segmentation
leaky_relu_p3_s16_curvature_any_mae.csv      # Activation with underscore
```

### Key Changes

| Component | Old Format | New Format |
|-----------|------------|------------|
| Degree prefix | None (`gelu_32_4_...`) | `p` for poly, `n{N}d{D}` for rational |
| Segments prefix | None | `s` prefix (`s32`) |
| Metric | Not in filename | Required suffix (`_ulp`, `_max`, `_mae`) |

### New Segmentation: `recursive_ulp`

Dynamic segment count based on ULP target:
```bash
./generate_coefficients.sh --activation gelu --segmentation-method recursive_ulp \
    --target-ulp 1 --max-depth 10 --subdivision-degree 4
```

### Impact on tt-metal Sweep Scripts

The sweep scripts in `generic_lut_activation/` reference old CSV paths like:
```bash
# OLD (broken)
lut_file="${TT_POLY_FIT_DIR}/data/coefficients/${activation}_${segments}_${degree}_${segmentation}_${fitting}.csv"

# NEW (correct)
lut_file="${TT_POLY_FIT_DIR}/data/coefficients/${activation}_p${degree}_s${segments}_${segmentation}_${fitting}_${metric}.csv"
```

**Action:** Update sweep scripts to use new filename format or use centralized parsers:
- Python: `from csv_filename import parse_csv_filename, build_csv_filename`
- Bash: `source csv_filename.sh`

## Troubleshooting

### Build failures

```bash
# Clean rebuild
ninja -C $BUILD_DIR -t clean programming_examples_generic_lut_activation_embedded_adhoc
ninja -C $BUILD_DIR programming_examples_generic_lut_activation_embedded_adhoc
```

### Device hangs

```bash
# Reset device
/opt/venv/bin/tt-smi -r 0

# Check device status
/opt/venv/bin/tt-smi
```

### Missing coefficients

```bash
# Verify polynomial fitter path
ls $TT_POLY_FIT_DIR/data/coefficients/gelu*.csv

# Check best.csv exists
head $TT_POLY_FIT_DIR/best.csv
```

### Kernel generation errors

```bash
# Debug mode
python3 tools/generate_adhoc_kernel.py \
    --activation gelu --degree 4 --segments 16 \
    --segmentation chebyshev --metric ulp \
    --output /dev/stdout 2>&1 | head -100
```

## Success Criteria

1. **Build time:** Adhoc rebuild < 15 seconds (vs hours for full compilation)
2. **Accuracy:** Hardware ULP error within 10% of theoretical
3. **Performance:** Profiler timing within 5% of pre-compiled baseline
4. **No regressions:** All existing sweep functionality works

## Files Changed in This Migration

### Embedded (`generic_lut_activation_embedded/`)
- `CMakeLists.txt` - Removed `include(EmbeddedTargets.cmake)`, added adhoc targets
- `tools/generate_adhoc_kernel.py` - NEW: unified generator
- `sweep_best.sh`, `sweep_polynomial.sh`, `sweep_rational.sh` - Use adhoc workflow
- DELETED: `tools/generate_embedded_kernels.py`, `tools/generate_cmake_embedded.py`, etc.
- DELETED: `cmake/EmbeddedTargets.cmake` (6851 lines)
- DELETED: 522+ kernel wrapper files

### Non-embedded (`generic_lut_activation/`)
- `CMakeLists.txt` - 327 lines → 67 lines (205 targets → 4 targets)
- `adhoc_config.h` - NEW: polynomial config header
- `adhoc_rational_config.h` - NEW: rational config header
- `generic_lut_activation.cpp` - Added `#ifdef ADHOC_MODE`
- `generic_lut_activation_rational.cpp` - Added `#ifdef ADHOC_MODE`
- DELETED: `generate_cmake_all.py`, `generate_cmake_lists.py`
