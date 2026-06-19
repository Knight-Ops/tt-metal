#!/bin/bash
# Compare all three kernel variants: SFPU, BF16 FPU, TF32 FPU
# For GELU with degree 8, 16 segments

set -e

if [ -z "$TT_POLY_FIT_DIR" ]; then
    echo "ERROR: TT_POLY_FIT_DIR not set"
    exit 1
fi

DEGREE=8
SEGMENTS=16
LUT_SIZE=$((($SEGMENTS + 1) + $SEGMENTS * ($DEGREE + 1)))  # 161

echo "========================================"
echo "Comparing All Three Kernel Variants"
echo "========================================"
echo "Configuration: GELU Degree=$DEGREE Segments=$SEGMENTS LUT_SIZE=$LUT_SIZE"
echo ""

# Find GELU coefficient file
COEFF_FILE=$(find "$TT_POLY_FIT_DIR" -name "*gelu*${SEGMENTS}*${DEGREE}*.csv" -o -name "*gelu*.csv" 2>/dev/null | head -1)
if [ -z "$COEFF_FILE" ]; then
    echo "ERROR: Could not find GELU coefficient file in $TT_POLY_FIT_DIR"
    exit 1
fi
echo "Using coefficient file: $COEFF_FILE"
echo ""

# Get to repo root
cd "$(dirname "$0")"
REPO_ROOT="../../.."
BUILD_DIR="$REPO_ROOT/build"

# Test parameters
TILES=256
COMPUTE_LOOPS=100

# Enable Tracy profiler
export TT_METAL_DEVICE_PROFILER=1

echo "========================================"
echo "1. SFPU Horner (piecewise_generic)"
echo "========================================"

cmake -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DKERNEL_VARIANT=piecewise_generic \
  -DPOLY_DEGREE=$DEGREE \
  -DNUM_SEGMENTS=$SEGMENTS \
  -DLUT_SIZE=$LUT_SIZE

cmake --build "$BUILD_DIR" --target generic_lut_activation

echo ""
echo "Running SFPU Horner..."
DUMP_OUTPUT_CSV=output_sfpu.csv \
"$BUILD_DIR/bin/generic_lut_activation" \
  "$COEFF_FILE" \
  --activation gelu \
  --precision fp32 \
  --range-min -10 \
  --range-max 10 \
  --tiles $TILES \
  --compute-loops $COMPUTE_LOOPS

echo ""
echo "========================================"
echo "2. BF16 FPU (piecewise_fpu)"
echo "========================================"

cmake -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DKERNEL_VARIANT=piecewise_fpu \
  -DPOLY_DEGREE=$DEGREE \
  -DNUM_SEGMENTS=$SEGMENTS \
  -DLUT_SIZE=$LUT_SIZE

cmake --build "$BUILD_DIR" --target generic_lut_activation

echo ""
echo "Running BF16 FPU..."
DUMP_OUTPUT_CSV=output_bf16_fpu.csv \
"$BUILD_DIR/bin/generic_lut_activation" \
  "$COEFF_FILE" \
  --activation gelu \
  --precision fp32 \
  --range-min -10 \
  --range-max 10 \
  --tiles $TILES \
  --compute-loops $COMPUTE_LOOPS

echo ""
echo "========================================"
echo "3. TF32 FPU (piecewise_fpu_tf32)"
echo "========================================"

cmake -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DKERNEL_VARIANT=piecewise_fpu_tf32 \
  -DPOLY_DEGREE=$DEGREE \
  -DNUM_SEGMENTS=$SEGMENTS \
  -DLUT_SIZE=$LUT_SIZE

cmake --build "$BUILD_DIR" --target generic_lut_activation

echo ""
echo "Running TF32 FPU..."
DUMP_OUTPUT_CSV=output_tf32_fpu.csv \
"$BUILD_DIR/bin/generic_lut_activation" \
  "$COEFF_FILE" \
  --activation gelu \
  --precision fp32 \
  --range-min -10 \
  --range-max 10 \
  --tiles $TILES \
  --compute-loops $COMPUTE_LOOPS

echo ""
echo "========================================"
echo "Comparison Summary"
echo "========================================"

# Analyze errors using csv2error.py from TT_POLY_FIT_DIR
echo ""
echo "1. SFPU Horner Errors:"
python3 "$TT_POLY_FIT_DIR/csv2error.py" output_sfpu.csv gelu

echo ""
echo "2. BF16 FPU Errors:"
python3 "$TT_POLY_FIT_DIR/csv2error.py" output_bf16_fpu.csv gelu

echo ""
echo "3. TF32 FPU Errors:"
python3 "$TT_POLY_FIT_DIR/csv2error.py" output_tf32_fpu.csv gelu

echo ""
echo "Tracy profiler results saved to: generated/profiler/"
echo ""
echo "To view timing breakdown:"
echo "  cat generated/profiler/blackhole_1/device_profiler_blackhole_1.csv"
echo ""
echo "Key timing metrics (from program output above):"
echo "  - TIMING_KERNEL_EXECUTION: Total kernel runtime"
echo "  - Look for 'Compute amplification: 100×' speedup calculation"
echo ""
