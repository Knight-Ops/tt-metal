#!/bin/bash
# Quick comparison of all 3 kernels for GELU P8 S16
# Uses csv2error.py and GELU coefficients from TT_POLY_FIT_DIR

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/argparse_common.sh"
source "$SCRIPT_DIR/sweep_helpers.sh"

# Initialize common environment (sets REPO_ROOT, BUILD_DIR, TT_POLY_FIT_DIR, ARCH_NAME)
init_common_env

D=8
S=16
LUT=$(( (S + 1) + S * (D + 1) ))
TILES=256
ACTIVATION="gelu"
SEGMENTATION="uniform"
FITTING="any"
METRIC="ulp"

# Get test range for GELU
RANGE_MIN=-10
RANGE_MAX=10

# Coefficient file using canonical CSV format: {activation}_p{degree}_s{segments}_{segmentation}_{fitting}_{metric}.csv
COEFF_FILE=$(get_poly_csv_path "$ACTIVATION" "$D" "$S" "$SEGMENTATION" "$FITTING" "$METRIC")
if [ ! -f "$COEFF_FILE" ]; then
    echo "ERROR: Could not find GELU coefficient file: $COEFF_FILE"
    exit 1
fi

echo "========================================"
echo "Quick Test: GELU P8 S16 (3 kernels)"
echo "========================================"
echo "Using coefficient file: $COEFF_FILE"
echo "Build dir: $BUILD_DIR"
echo "Tiles: $TILES"
echo ""

# Run binaries from repo root so device descriptors resolve correctly
cd "$REPO_ROOT"
WORK_DIR="tt_metal/programming_examples/generic_lut_activation"

# Separate local build dir for FPU variants (avoids overwriting build_Release)
LOCAL_BUILD_DIR="${REPO_ROOT}/build_fpu_compare"

# Test each kernel variant
for VARIANT in piecewise_generic piecewise_fpu piecewise_fpu_tf32; do
    echo "========================================"
    echo "Testing: $VARIANT"
    echo "========================================"

    # piecewise_generic: use the pre-built named binary from BUILD_DIR if available.
    # FPU variants have no dedicated named binary, so always build them locally.
    if [ "$VARIANT" = "piecewise_generic" ]; then
        BINARY="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_p${D}_s${S}"
        if [ -x "$BINARY" ]; then
            echo "Using pre-built binary: $BINARY"
        else
            echo "Pre-built binary not found, building $VARIANT locally..."
            cmake -S "$REPO_ROOT/$WORK_DIR" \
                  -B "$LOCAL_BUILD_DIR" \
                  -G Ninja \
                  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                  -DCMAKE_PREFIX_PATH="$BUILD_DIR" \
                  -DKERNEL_VARIANT="$VARIANT" \
                  -DPOLY_DEGREE=$D \
                  -DNUM_SEGMENTS=$S \
                  -DLUT_SIZE=$LUT \
                  > /dev/null 2>&1
            cmake --build "$LOCAL_BUILD_DIR" \
                  --target programming_examples_generic_lut_activation \
                  > /dev/null 2>&1
            BINARY="$LOCAL_BUILD_DIR/programming_examples_generic_lut_activation"
        fi
    else
        # FPU variants: reconfigure with new KERNEL_VARIANT and build
        echo "Building $VARIANT..."
        cmake -S "$REPO_ROOT/$WORK_DIR" \
              -B "$LOCAL_BUILD_DIR" \
              -G Ninja \
              -DCMAKE_BUILD_TYPE=RelWithDebInfo \
              -DCMAKE_PREFIX_PATH="$BUILD_DIR" \
              -DKERNEL_VARIANT="$VARIANT" \
              -DPOLY_DEGREE=$D \
              -DNUM_SEGMENTS=$S \
              -DLUT_SIZE=$LUT \
              > /dev/null 2>&1
        cmake --build "$LOCAL_BUILD_DIR" \
              --target programming_examples_generic_lut_activation \
              > /dev/null 2>&1
        BINARY="$LOCAL_BUILD_DIR/programming_examples_generic_lut_activation"
    fi

    if [ ! -x "$BINARY" ]; then
        echo "ERROR: Binary not found at $BINARY"
        exit 1
    fi

    OUTPUT_CSV="/tmp/out_${VARIANT}_p${D}_s${S}.csv"

    # Run and capture timing
    echo "Running..."
    DUMP_OUTPUT_CSV="$OUTPUT_CSV" \
    "$BINARY" \
      "$COEFF_FILE" \
      --activation "$ACTIVATION" \
      --precision fp32 \
      --range-min $RANGE_MIN \
      --range-max $RANGE_MAX \
      --tiles $TILES \
      2>&1 | grep "TIMING_KERNEL_EXECUTION"

    # Analyze error
    if [ -f "$OUTPUT_CSV" ]; then
        python3 "$TT_POLY_FIT_DIR/csv2error.py" "$OUTPUT_CSV" "$ACTIVATION" \
          | grep -E "(MAE|Max Relative)"
    else
        echo "WARNING: Output CSV not found"
    fi

    echo ""
done

echo "========================================"
echo "Done!"
echo "========================================"
