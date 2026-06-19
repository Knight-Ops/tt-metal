#!/bin/bash
# Build and Test Script for FPU Piecewise Polynomial Kernels
# ===========================================================
#
# Tests both the BF16 and TF32 FPU variants and compares accuracy
# against the SFPU baseline using ground truth from tt-polynomial-fitter.
#
# Usage:
#   ./build_and_test_tf32.sh [activation] [degree] [segments]
#
# Examples:
#   ./build_and_test_tf32.sh sigmoid 6 16
#   ./build_and_test_tf32.sh gelu 4 8
#   ./build_and_test_tf32.sh tanh 8 32
#
# Prerequisites:
#   - tt-polynomial-fitter repo at $TT_POLY_FIT_DIR (or ../tt-polynomial-fitter)
#   - Build already done: ./build_metal.sh --build-programming-examples
#     (from /localdev/nkapre/tt-metal repo root)

set -e

ACTIVATION="${1:-sigmoid}"
POLY_DEGREE="${2:-6}"
NUM_SEGMENTS="${3:-16}"

# Locate repo root (script lives in tt_metal/programming_examples/generic_lut_activation/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Source helpers for get_test_range (reads from sweep_config.dat)
source "$SCRIPT_DIR/sweep_helpers.sh"
WORK_DIR="tt_metal/programming_examples/generic_lut_activation"
ensure_sweep_config "$WORK_DIR"
CONFIG_FILE="$WORK_DIR/sweep_config.dat"
if [[ -f "$CONFIG_FILE" ]]; then
    eval "$(grep -v '^\s*#' "$CONFIG_FILE" | grep -v '^\s*$' | sed 's/,/ /g' | sed 's/\([A-Z_]*\)=/\1=(/' | sed 's/$/)/')"
fi

# Build dir
BUILD_DIR="$REPO_ROOT/build"

# tt-polynomial-fitter location
TT_POLY_FIT_DIR="${TT_POLY_FIT_DIR:-$REPO_ROOT/../tt-polynomial-fitter}"

# Binary names follow: programming_examples_generic_lut_activation_p{deg}_s{segs}{suffix}
SFPU_BINARY="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_p${POLY_DEGREE}_s${NUM_SEGMENTS}"
FPU_BF16_BINARY="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_p${POLY_DEGREE}_s${NUM_SEGMENTS}_fpu"
FPU_TF32_BINARY="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_p${POLY_DEGREE}_s${NUM_SEGMENTS}_fpu_tf32"

# Coefficient file — look for polynomial CSV in tt-polynomial-fitter data dir
COEFF_DIR="$TT_POLY_FIT_DIR/data/coefficients"
CSV_FILE=$(ls "${COEFF_DIR}/${ACTIVATION}_*_${NUM_SEGMENTS}_${POLY_DEGREE}_*.csv" 2>/dev/null | head -n 1)

echo "========================================"
echo "FPU Piecewise Polynomial Kernel Test"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Activation:    $ACTIVATION"
echo "  Polynomial:    Degree $POLY_DEGREE, $NUM_SEGMENTS segments"
echo "  Repo root:     $REPO_ROOT"
echo "  Build dir:     $BUILD_DIR"
echo "  Poly fitter:   $TT_POLY_FIT_DIR"
echo ""

# ---- Step 1: Check binaries ----
echo "========================================"
echo "Step 1: Checking binaries"
echo "========================================"
echo ""

check_binary() {
    local bin="$1"
    local name="$2"
    if [ -f "$bin" ]; then
        echo "  ✓ $name: $bin"
    else
        echo "  ✗ $name NOT FOUND: $bin"
        echo ""
        echo "  Build with:"
        echo "    cd $REPO_ROOT && ./build_metal.sh --build-programming-examples"
        return 1
    fi
}

check_binary "$SFPU_BINARY" "SFPU baseline"
check_binary "$FPU_BF16_BINARY" "FPU BF16"
check_binary "$FPU_TF32_BINARY" "FPU TF32"
echo ""

# ---- Step 2: Find coefficient file ----
echo "========================================"
echo "Step 2: Finding coefficient file"
echo "========================================"
echo ""

if [ -z "$CSV_FILE" ]; then
    echo "ERROR: No coefficient file found."
    echo "Searched: ${COEFF_DIR}/${ACTIVATION}_*_${NUM_SEGMENTS}_${POLY_DEGREE}_*.csv"
    echo ""
    echo "Available files matching '${ACTIVATION}':"
    ls "${COEFF_DIR}/${ACTIVATION}_"*.csv 2>/dev/null | head -10 || echo "  (none)"
    exit 1
fi

echo "  Coefficient file: $CSV_FILE"
echo ""

# ---- Step 3: Test range (from sweep_config.dat via get_test_range) ----
read -r TEST_MIN TEST_MAX < <(get_test_range "$ACTIVATION")

# Precision: use fp32 for TF32 variant, bf16 for BF16 variants
# (Check CSV filename to determine precision used in fitting)
if echo "$CSV_FILE" | grep -q "_fp32_"; then
    PRECISION="fp32"
else
    PRECISION="bf16"
fi

echo "  Test range:    [$TEST_MIN, $TEST_MAX]"
echo "  Precision:     $PRECISION"
echo ""

# ---- Step 4: Device reset ----
echo "========================================"
echo "Step 3: Device reset"
echo "========================================"
echo ""
echo "  Resetting device to clear stale state..."
tt-smi -r 0 2>/dev/null || /opt/venv/bin/tt-smi -r 0 2>/dev/null || echo "  (tt-smi not found, skipping reset)"
sleep 2
echo ""

# ---- Step 5: Run kernels ----
echo "========================================"
echo "Step 4: Running kernels"
echo "========================================"
echo ""

TILES=32
OUT_SFPU="out_sfpu_${ACTIVATION}_p${POLY_DEGREE}_s${NUM_SEGMENTS}.csv"
OUT_FPU_BF16="out_fpu_bf16_${ACTIVATION}_p${POLY_DEGREE}_s${NUM_SEGMENTS}.csv"
OUT_FPU_TF32="out_fpu_tf32_${ACTIVATION}_p${POLY_DEGREE}_s${NUM_SEGMENTS}.csv"

run_kernel() {
    local binary="$1"
    local output="$2"
    local label="$3"
    local precision="$4"

    echo "  Running $label..."
    (cd "$REPO_ROOT" && \
        DUMP_OUTPUT_CSV="$SCRIPT_DIR/$output" \
        "$binary" \
        "$CSV_FILE" \
        --activation "$ACTIVATION" \
        --precision "$precision" \
        --range-min "$TEST_MIN" \
        --range-max "$TEST_MAX" \
        --tiles "$TILES" \
        > /dev/null 2>&1) \
    && echo "    ✓ $label done → $output" \
    || echo "    ✗ $label FAILED"
    echo ""
}

run_kernel "$SFPU_BINARY"    "$OUT_SFPU"     "SFPU baseline" "$PRECISION"
run_kernel "$FPU_BF16_BINARY" "$OUT_FPU_BF16" "FPU BF16"      "$PRECISION"
run_kernel "$FPU_TF32_BINARY" "$OUT_FPU_TF32" "FPU TF32"      "fp32"

# ---- Step 6: Accuracy analysis ----
echo "========================================"
echo "Step 5: Accuracy analysis"
echo "========================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "  ⚠ Python3 not found — skipping analysis"
    exit 0
fi

# Check for ground_truth module
if [ ! -d "$TT_POLY_FIT_DIR" ]; then
    echo "  ⚠ tt-polynomial-fitter not found at $TT_POLY_FIT_DIR"
    echo "    Set TT_POLY_FIT_DIR to use ground_truth module"
    echo "    Skipping accuracy analysis"
    exit 0
fi

python3 - <<PYTHON_EOF
import sys
sys.path.insert(0, "$TT_POLY_FIT_DIR")
import numpy as np
import pandas as pd

try:
    from ground_truth import compute_ground_truth
except ImportError:
    print("  ⚠ Could not import ground_truth from $TT_POLY_FIT_DIR")
    print("    Skipping accuracy analysis")
    sys.exit(0)

activation = "$ACTIVATION"
files = {
    "SFPU baseline": "$SCRIPT_DIR/$OUT_SFPU",
    "FPU BF16":      "$SCRIPT_DIR/$OUT_FPU_BF16",
    "FPU TF32":      "$SCRIPT_DIR/$OUT_FPU_TF32",
}

print(f"  Activation: {activation}")
print(f"  {'Variant':<16} {'MAE':>12} {'Max AE':>12} {'ratio':>8}  {'Passthrough?':>14}")
print(f"  {'-'*68}")

for label, fname in files.items():
    try:
        df = pd.read_csv(fname)
    except Exception as e:
        print(f"  {label:<16}  (missing: {e})")
        continue

    inputs  = df["input"].values.astype(np.float64)
    outputs = df["output"].values.astype(np.float64)

    # Passthrough check
    passthrough = np.allclose(inputs, outputs, rtol=1e-4, atol=1e-6)

    try:
        gt = compute_ground_truth(activation, inputs)
    except Exception as e:
        print(f"  {label:<16}  (ground_truth error: {e})")
        continue

    ae  = np.abs(outputs - gt)
    mae = float(np.mean(ae))
    max_ae = float(np.max(ae))

    # Ratio: MAE compared to a tiny perturbation of the reference (measures relative scale)
    perturb = compute_ground_truth(activation, inputs + 1e-4) - gt
    scale = np.mean(np.abs(perturb)) / 1e-4  # ~|f'(x)|
    ratio = mae / (scale * 1e-3) if scale > 0 else float('nan')  # MAE in units of 1e-3 * |f'|

    pt_str = "PASSTHROUGH!" if passthrough else "ok"
    print(f"  {label:<16} {mae:>12.3e} {max_ae:>12.3e} {ratio:>8.3f}  {pt_str:>14}")

print()
print("  ratio < 2.0: hardware accuracy within 2x of expected")
print("  FPU TF32 ratio should be <= FPU BF16 ratio (same or better)")
PYTHON_EOF

echo ""
echo "========================================"
echo "✓ FPU Kernel Test Complete"
echo "========================================"
echo ""
echo "Output files:"
echo "  $OUT_SFPU"
echo "  $OUT_FPU_BF16"
echo "  $OUT_FPU_TF32"
echo ""
