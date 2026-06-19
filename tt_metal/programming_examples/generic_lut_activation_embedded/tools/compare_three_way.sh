#!/bin/bash
# =============================================================================
# compare_three_way.sh — Three-way comparison: Original vs Drop-in vs Embedded
#
# Measures ULP (extract_accuracy.py) and yolov4 runtime for:
#   1. Original SFPU (git baseline)
#   2. Drop-in replacement (current working tree)
#   3. Embedded kernel (run_csv.sh)
#
# Usage:
#   ./compare_three_way.sh --activation softplus --csv <rational_csv> [--precision bf16]
#
# Requires:
#   - Modified ckernel_sfpu_<activation>.h already in working tree
#   - Coefficient CSV for run_csv.sh
#   - TT_POLY_FIT_DIR set (for extract_accuracy.py)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
WORK_DIR="tt_metal/programming_examples/generic_lut_activation_embedded"

source "$SCRIPT_DIR/../profiler_helpers.sh"
source "$SCRIPT_DIR/../sweep_helpers.sh"
init_arch_detection

TT_POLY_FIT_DIR="${TT_POLY_FIT_DIR:-/localdev/nkapre/tt-polynomial-fitter}"
ACCURACY_SCRIPT="$TT_POLY_FIT_DIR/extract_accuracy.py"
# Use system python for accuracy (no TTNN dependency; python_env torch has broken ULP spacing)
ACCURACY_PYTHON="/usr/bin/python3"

# Parse args
ACTIVATION=""
CSV_FILE=""
PRECISION="bf16"

while [[ $# -gt 0 ]]; do
    case $1 in
        --activation|-a) ACTIVATION="$2"; shift 2 ;;
        --csv|-c)        CSV_FILE="$2"; shift 2 ;;
        --precision|-p)  PRECISION="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --activation <name> --csv <coeff_csv> [--precision bf16|fp32]"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[[ -z "$ACTIVATION" ]] && { echo "Error: --activation required"; exit 1; }
[[ -z "$CSV_FILE" ]] && { echo "Error: --csv required"; exit 1; }
[[ ! -f "$CSV_FILE" ]] && { echo "Error: CSV not found: $CSV_FILE"; exit 1; }
[[ ! -f "$ACCURACY_SCRIPT" ]] && { echo "Error: extract_accuracy.py not found"; exit 1; }

CSV_FILE="$(cd "$(dirname "$CSV_FILE")" && pwd)/$(basename "$CSV_FILE")"

TTNN_TEST_SCRIPT=$(mktemp /tmp/three_way_ttnn_XXXX.py)
cat > "$TTNN_TEST_SCRIPT" << 'PYEOF'
import sys, os, torch, ttnn, numpy as np

activation = sys.argv[1]
precision = sys.argv[2]
output_csv = sys.argv[3]
label = sys.argv[4]
profiler_dir = sys.argv[5] if len(sys.argv) > 5 else ""

device = ttnn.open_device(device_id=0)
is_bf16 = precision == "bf16"
dtype_tt = ttnn.bfloat16 if is_bf16 else ttnn.float32
dtype_torch = torch.bfloat16 if is_bf16 else torch.float32

# Exhaustive BF16 or dense FP32 in [-10, 10]
if is_bf16:
    all_bits = np.arange(65536, dtype=np.uint16)
    f32_bits = all_bits.astype(np.uint32) << 16
    all_vals = np.frombuffer(f32_bits.tobytes(), dtype=np.float32)
    mask = np.isfinite(all_vals) & (all_vals >= -10) & (all_vals <= 10)
    x_cpu = torch.from_numpy(np.sort(all_vals[mask])).bfloat16()
else:
    x_cpu = torch.linspace(-10, 10, 262144, dtype=torch.float32)
n = len(x_cpu)

# Pad + evaluate
pad = ((n + 1023) // 1024) * 1024
xp = torch.zeros(pad, dtype=dtype_torch)
xp[:n] = x_cpu
xt = ttnn.from_torch(xp.reshape(1, 1, 1, -1), device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype_tt)
fn = getattr(ttnn, activation)
yt = fn(xt)
hw = ttnn.to_torch(yt).squeeze().float().numpy()[:n]

# Dump CSV
with open(output_csv, 'w') as f:
    f.write('input,output\n')
    x_np = x_cpu.float().numpy()
    for i in range(n):
        f.write(f'{x_np[i]},{hw[i]}\n')

# yolov4 Tracy-profiled run (3 runs, take min)
xt_t = ttnn.from_torch(torch.randn(1, 1, 5120, 320, dtype=dtype_torch), device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype_tt)
# Warmup (unprofiled)
for _ in range(3):
    fn(xt_t)
ttnn.synchronize_device(device)

# Profiled run
fn(xt_t)
ttnn.synchronize_device(device)

print(f'{label},{n}')
ttnn.close_device(device)
PYEOF

cd "$REPO_ROOT"
source python_env/bin/activate
export PYTHONPATH="$REPO_ROOT"

BH_SFPU="tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"
WH_SFPU="tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu"
HEADER="ckernel_sfpu_${ACTIVATION}.h"
SHARED="ckernel_sfpu_piecewise_rational.h"

echo "================================================================"
echo "  Three-Way Comparison: $ACTIVATION ($PRECISION, Blackhole)"
echo "================================================================"
echo ""

# --- 1. ORIGINAL ---
echo ">>> Step 1/3: Original SFPU baseline"
# Save current files (may not exist for new SFPU kernels / composite ops)
IS_COMPOSITE=false
if [[ -f "$BH_SFPU/$HEADER" ]]; then
    cp "$BH_SFPU/$HEADER" /tmp/dropin_header_backup.h
    cp "$WH_SFPU/$HEADER" /tmp/dropin_header_wh_backup.h
    [[ -f "$BH_SFPU/$SHARED" ]] && cp "$BH_SFPU/$SHARED" /tmp/dropin_shared_backup.h
    [[ -f "$WH_SFPU/$SHARED" ]] && cp "$WH_SFPU/$SHARED" /tmp/dropin_shared_wh_backup.h

    # Revert to original from upstream/main
    git fetch upstream main --quiet 2>/dev/null || true
    if git show upstream/main:"$BH_SFPU/$HEADER" > /dev/null 2>&1; then
        git show upstream/main:"$BH_SFPU/$HEADER" > "$BH_SFPU/$HEADER" 2>/dev/null
        git show upstream/main:"$WH_SFPU/$HEADER" > "$WH_SFPU/$HEADER" 2>/dev/null
    else
        echo "  No original SFPU header in upstream/main — measuring composite op baseline"
        IS_COMPOSITE=true
        # Remove our drop-in so the composite op path runs instead
        rm -f "$BH_SFPU/$HEADER" "$WH_SFPU/$HEADER"
    fi
else
    echo "  No SFPU header found — measuring composite op baseline"
    IS_COMPOSITE=true
fi
./build_metal.sh 2>&1 | tail -1
# Use unique cache dir per step to prevent cross-contamination
export TT_METAL_CACHE="/tmp/tt_cache_original_${PRECISION}_$$"
rm -rf "$TT_METAL_CACHE"
find "$REPO_ROOT" -maxdepth 1 -name "ckernel_sfpu_*.h" -delete 2>/dev/null

# Run 3 Tracy-profiled iterations, take min
ORIG_BEST_US="999999"
for run in 1 2 3; do
    PROF_DIR="/tmp/three_way_orig_${PRECISION}_run${run}"
    mkdir -p "$PROF_DIR"
    [[ "$run" -eq 1 ]] && export DUMP_CSV_ARG="/tmp/three_way_original_${PRECISION}.csv" || DUMP_CSV_ARG="/dev/null"
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$PROF_DIR" \
        python3 "$TTNN_TEST_SCRIPT" "$ACTIVATION" "$PRECISION" "$DUMP_CSV_ARG" "ORIGINAL" 2>/dev/null
    PROF_CSV="$PROF_DIR/.logs/profile_log_device.csv"
    if [[ -f "$PROF_CSV" ]]; then
        t=$(extract_profiler_compute_time "$PROF_CSV" "$WORK_DIR")
        if [[ -n "$t" && "$t" != "0" ]] && (( $(echo "$t < $ORIG_BEST_US" | bc -l 2>/dev/null || echo 0) )); then
            ORIG_BEST_US="$t"
        fi
    fi
done
ORIG_US="$ORIG_BEST_US"
ORIG_ACC=$($ACCURACY_PYTHON "$ACCURACY_SCRIPT" "$ACTIVATION" /tmp/three_way_original_${PRECISION}.csv 2>/dev/null)
ORIG_MAX_ULP=$(echo "$ORIG_ACC" | cut -d',' -f5)
ORIG_MEAN_ULP=$(echo "$ORIG_ACC" | cut -d',' -f6)
echo "  MaxULP=$ORIG_MAX_ULP MeanULP=$ORIG_MEAN_ULP yolov4=${ORIG_US}us (Tracy)"

# --- 2. DROP-IN ---
echo ">>> Step 2/3: Drop-in replacement"
# Restore drop-in files
if [[ -f /tmp/dropin_header_backup.h ]]; then
    cp /tmp/dropin_header_backup.h "$BH_SFPU/$HEADER"
    cp /tmp/dropin_header_wh_backup.h "$WH_SFPU/$HEADER"
fi
[[ -f /tmp/dropin_shared_backup.h ]] && cp /tmp/dropin_shared_backup.h "$BH_SFPU/$SHARED"
[[ -f /tmp/dropin_shared_wh_backup.h ]] && cp /tmp/dropin_shared_wh_backup.h "$WH_SFPU/$SHARED"
# Also restore polynomial header if it exists
[[ -f "$BH_SFPU/ckernel_sfpu_piecewise_polynomial.h" ]] || \
    cp "$WH_SFPU/ckernel_sfpu_piecewise_polynomial.h" "$BH_SFPU/ckernel_sfpu_piecewise_polynomial.h" 2>/dev/null || true
./build_metal.sh 2>&1 | tail -1
# Use unique cache dir per step to prevent cross-contamination
export TT_METAL_CACHE="/tmp/tt_cache_dropin_${PRECISION}_$$"
rm -rf "$TT_METAL_CACHE"
find "$REPO_ROOT" -maxdepth 1 -name "ckernel_sfpu_*.h" -delete 2>/dev/null

# Run 3 Tracy-profiled iterations, take min
DROPIN_BEST_US="999999"
for run in 1 2 3; do
    PROF_DIR="/tmp/three_way_dropin_${PRECISION}_run${run}"
    mkdir -p "$PROF_DIR"
    [[ "$run" -eq 1 ]] && DUMP_CSV_ARG="/tmp/three_way_dropin_${PRECISION}.csv" || DUMP_CSV_ARG="/dev/null"
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$PROF_DIR" \
        python3 "$TTNN_TEST_SCRIPT" "$ACTIVATION" "$PRECISION" "$DUMP_CSV_ARG" "DROPIN" 2>/dev/null
    PROF_CSV="$PROF_DIR/.logs/profile_log_device.csv"
    if [[ -f "$PROF_CSV" ]]; then
        t=$(extract_profiler_compute_time "$PROF_CSV" "$WORK_DIR")
        if [[ -n "$t" && "$t" != "0" ]] && (( $(echo "$t < $DROPIN_BEST_US" | bc -l 2>/dev/null || echo 0) )); then
            DROPIN_BEST_US="$t"
        fi
    fi
done
DROPIN_US="$DROPIN_BEST_US"
DROPIN_ACC=$($ACCURACY_PYTHON "$ACCURACY_SCRIPT" "$ACTIVATION" /tmp/three_way_dropin_${PRECISION}.csv 2>/dev/null)
DROPIN_MAX_ULP=$(echo "$DROPIN_ACC" | cut -d',' -f5)
DROPIN_MEAN_ULP=$(echo "$DROPIN_ACC" | cut -d',' -f6)
echo "  MaxULP=$DROPIN_MAX_ULP MeanULP=$DROPIN_MEAN_ULP yolov4=${DROPIN_US}us (Tracy)"

# --- 3. EMBEDDED ---
echo ">>> Step 3/3: Embedded kernel (run_csv.sh)"
EMBEDDED_RESULT=$("$SCRIPT_DIR/../run_csv.sh" "$CSV_FILE" --activation "$ACTIVATION" --precision "$PRECISION" --runs 3 2>&1)
# Extract yolov4 line
EMBEDDED_YOLOV4=$(echo "$EMBEDDED_RESULT" | grep "yolov4" | tail -1)
EMBEDDED_TRACY=$(echo "$EMBEDDED_YOLOV4" | awk '{print $(NF-1)}' | sed 's/µs//')
EMBEDDED_MAX_ULP=$(echo "$EMBEDDED_YOLOV4" | awk '{print $5}')
EMBEDDED_MEAN_ULP=$(echo "$EMBEDDED_YOLOV4" | awk '{print $6}')
echo "  MaxULP=$EMBEDDED_MAX_ULP MeanULP=$EMBEDDED_MEAN_ULP yolov4=${EMBEDDED_TRACY}us (Tracy)"

# --- Summary ---
echo ""
echo "================================================================"
echo "  RESULTS: $ACTIVATION $PRECISION (Blackhole)"
echo "================================================================"
echo ""
printf "%-12s %10s %10s %12s\n" "Impl" "MaxULP" "MeanULP" "yolov4(us)"
printf "%-12s %10s %10s %12s\n" "------------" "----------" "----------" "------------"
ORIG_LABEL=$( [[ "$IS_COMPOSITE" == true ]] && echo "Composite" || echo "Original" )
printf "%-12s %10s %10s %12s\n" "$ORIG_LABEL" "$ORIG_MAX_ULP" "$ORIG_MEAN_ULP" "$ORIG_US"
printf "%-12s %10s %10s %12s\n" "Drop-in" "$DROPIN_MAX_ULP" "$DROPIN_MEAN_ULP" "$DROPIN_US"
printf "%-12s %10s %10s %12s\n" "Embedded" "$EMBEDDED_MAX_ULP" "$EMBEDDED_MEAN_ULP" "$EMBEDDED_TRACY"
echo "(All timings: Tracy device profiler, yolov4 5120x320)"
echo ""
echo "================================================================"

# Cleanup
rm -f "$TTNN_TEST_SCRIPT" /tmp/dropin_header_backup.h /tmp/dropin_header_wh_backup.h
rm -f /tmp/dropin_shared_backup.h /tmp/dropin_shared_wh_backup.h
rm -rf "/tmp/tt_cache_original_${PRECISION}_$$" "/tmp/tt_cache_dropin_${PRECISION}_$$"
unset TT_METAL_CACHE
