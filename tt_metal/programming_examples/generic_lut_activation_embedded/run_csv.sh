#!/bin/bash
# =============================================================================
# run_csv.sh — Run an arbitrary coefficient CSV through the embedded flow
#
# Usage:
#   ./run_csv.sh <csv_file> --activation <name> [--precision fp32|bf16|both] [--tiles N] [--range-min X] [--range-max X] [--skip-build] [--dump-csv <path>]
#
# Example:
#   ./run_csv.sh /path/to/sigmoid_16_6_uniform_any_ulp.csv --activation sigmoid
#   ./run_csv.sh my_coeffs.csv --activation gelu --precision bf16 --tiles 64
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
WORK_DIR="tt_metal/programming_examples/generic_lut_activation_embedded"
BUILD_DIR="${TT_METAL_RUNTIME_ROOT:-$REPO_ROOT}/build_Release"
BINARY="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_embedded_adhoc"
KERNEL_DIR="$SCRIPT_DIR/kernels/compute/adhoc"

source "$SCRIPT_DIR/profiler_helpers.sh"
source "$SCRIPT_DIR/sweep_helpers.sh"
init_arch_detection

# TT_POLY_FIT_DIR needed for extract_accuracy.py (ground truth + ULP computation)
TT_POLY_FIT_DIR="${TT_POLY_FIT_DIR:-/localdev/nkapre/tt-polynomial-fitter}"
ACCURACY_SCRIPT="$TT_POLY_FIT_DIR/extract_accuracy.py"
# Use system python for accuracy (python_env's torch has broken BF16 ULP spacing)
ACCURACY_PYTHON="/usr/bin/python3"
HAS_ACCURACY=false
if [[ -f "$ACCURACY_SCRIPT" ]]; then
    HAS_ACCURACY=true
fi

# Standard test shapes (same as sweep scripts)
declare -a TEST_SHAPES=(
    "single_tile:32:32"
    "8_tiles:64:128"
    "256_tiles:512:512"
    "height_sharded:25600:128"
    "yolov4:5120:320"
)

NUM_RUNS=3

# --- Parse args ---
CSV_FILE=""
ACTIVATION=""
PRECISION="fp32"
TILES_OVERRIDE=""
RANGE_MIN=""
RANGE_MAX=""
SKIP_BUILD=false
DUMP_CSV=""
EXTRA_BINARY_ARGS=()

show_help() {
    echo "Usage: $0 <csv_file> --activation <name> [OPTIONS]"
    echo ""
    echo "Run an arbitrary coefficient CSV through the embedded kernel flow."
    echo "Auto-detects polynomial degree and segment count from the CSV."
    echo "Runs all 5 standard shapes with Tracy profiler timing (3 runs, takes min)."
    echo ""
    echo "Required:"
    echo "  <csv_file>                Path to coefficient CSV (segment_id,lo,hi,c0,c1,...)"
    echo "  --activation <name>       Activation function name (for ground truth comparison)"
    echo ""
    echo "Optional:"
    echo "  --precision <fp32|bf16|both>  Precision mode (default: fp32). 'both' runs bf16 then fp32."
    echo "  --tiles <N>               Override: run only this tile count (skip standard shapes)"
    echo "  --range-min <X>           Override input range min (default: from CSV)"
    echo "  --range-max <X>           Override input range max (default: from CSV)"
    echo "  --runs <N>                Number of timing runs per shape (default: 3)"
    echo "  --skip-build              Skip build (reuse last binary)"
    echo "  --dump-csv <path>         Dump per-element hardware output CSV"
    echo "  --no-dual-eval            Disable dual x-vector evaluation"
    echo "  --no-adaptive-degree      Disable per-segment degree optimization"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Standard shapes (run by default):"
    echo "  single_tile    32x32      (1 tile)"
    echo "  8_tiles        64x128     (8 tiles)"
    echo "  256_tiles      512x512    (256 tiles)"
    echo "  height_sharded 25600x128  (3200 tiles)"
    echo "  yolov4         5120x320   (51200 tiles)"
    echo ""
    echo "Examples:"
    echo "  $0 coeffs/sigmoid_16_6_uniform_any_ulp.csv --activation sigmoid"
    echo "  $0 my_gelu.csv --activation gelu --precision bf16"
    echo "  $0 my_gelu.csv --activation gelu --tiles 256          # single shape only"
    echo "  $0 exp_coeffs.csv --activation exp --dump-csv /tmp/hw_out.csv"
    exit 0
}

# First positional arg is CSV file
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
fi

if [[ "$1" != --* ]]; then
    CSV_FILE="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --activation|-a) ACTIVATION="$2"; shift 2 ;;
        --precision|-p)  PRECISION="$2"; shift 2 ;;
        --tiles)         TILES_OVERRIDE="$2"; shift 2 ;;
        --range-min)     RANGE_MIN="$2"; shift 2 ;;
        --range-max)     RANGE_MAX="$2"; shift 2 ;;
        --runs)          NUM_RUNS="$2"; shift 2 ;;
        --skip-build)    SKIP_BUILD=true; shift ;;
        --dump-csv)      DUMP_CSV="$2"; shift 2 ;;
        --no-dual-eval)  EXTRA_BINARY_ARGS+=("--no-dual-eval"); shift ;;
        --no-adaptive-degree) EXTRA_BINARY_ARGS+=("--no-adaptive-degree"); shift ;;
        -h|--help)       show_help ;;
        *)               echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CSV_FILE" ]]; then
    echo "Error: CSV file path required as first argument"
    exit 1
fi
if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi
if [[ -z "$ACTIVATION" ]]; then
    echo "Error: --activation is required"
    exit 1
fi

# Make CSV path absolute
CSV_FILE="$(cd "$(dirname "$CSV_FILE")" && pwd)/$(basename "$CSV_FILE")"

# If --tiles is given, replace standard shapes with a single custom shape
if [[ -n "$TILES_OVERRIDE" ]]; then
    TEST_SHAPES=("custom:1:$((TILES_OVERRIDE * 1024))")
    # Actually, tiles = rows/32 * cols/32. Simplest: use rows=32, cols=TILES*32
    cols=$((TILES_OVERRIDE * 32))
    TEST_SHAPES=("custom_${TILES_OVERRIDE}t:32:${cols}")
fi

# --- Step 1: Generate embedded kernel from CSV ---
echo "================================================================================"
echo "  run_csv.sh — Embedded flow for arbitrary coefficient CSV"
echo "================================================================================"
echo "CSV:        $CSV_FILE"
echo "Activation: $ACTIVATION"
echo "Precision:  $PRECISION"
echo "Runs/shape: $NUM_RUNS"
echo ""

mkdir -p "$KERNEL_DIR"

# Inline Python: parse CSV, auto-detect degree/segments, write kernel .cpp
python3 -c "
import csv, sys, os

csv_path = '$CSV_FILE'
kernel_path = '$KERNEL_DIR/adhoc.cpp'
range_min_override = '$RANGE_MIN' or None
range_max_override = '$RANGE_MAX' or None

# Parse CSV
import math
boundaries = []
coefficients = []
segment_degrees = []
asymptotic_flags = []  # per-segment: True if asymptotic
dominant_factors = []  # per-segment: dominant factor string
metadata = {}
degree = 0

with open(csv_path) as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames

    # Auto-detect: polynomial (c0,c1,...) vs rational (n0,n1,...,d0,d1,...)
    num_cols = sorted([h for h in headers if h.startswith('n') and h[1:].isdigit()], key=lambda h: int(h[1:]))
    den_cols = sorted([h for h in headers if h.startswith('d') and h[1:].isdigit()], key=lambda h: int(h[1:]))
    coeff_cols = sorted([h for h in headers if h.startswith('c') and h[1:].isdigit()], key=lambda h: int(h[1:]))

    is_rational = len(num_cols) > 0 and len(den_cols) > 0
    num_degree = 0
    den_degree = 0

    if is_rational:
        num_degree = len(num_cols) - 1
        den_degree = len(den_cols) - 1
        degree = num_degree  # for compatibility with degree_macros logic below
        print(f'Auto-detected RATIONAL approximation: n{num_degree}/d{den_degree} (num: {num_cols[0]}..{num_cols[-1]}, den: {den_cols[0]}..{den_cols[-1]})')
    else:
        degree = len(coeff_cols) - 1
        print(f'Auto-detected polynomial degree: {degree} (columns: {coeff_cols[0]}..{coeff_cols[-1]})')

    num_coefficients = []  # rational numerator
    den_coefficients = []  # rational denominator

    for row in reader:
        if row.get('segment_id', '').upper() == 'METADATA':
            key = row.get('lo', '')
            val = row.get('hi', '')
            if key:
                metadata[key] = val
            continue

        if len(boundaries) == 0:
            boundaries.append(float(row['lo']))
        boundaries.append(float(row['hi']))

        if is_rational:
            seg_num = [float(row[col]) for col in num_cols]
            seg_den = [float(row[col]) for col in den_cols]
            num_coefficients.extend(seg_num)
            den_coefficients.extend(seg_den)
            coefficients.extend(seg_num + seg_den)
            # Effective degree = max of num and den effective degrees
            seg_deg = 0
            for d in range(num_degree, -1, -1):
                if seg_num[d] != 0.0:
                    seg_deg = d
                    break
            segment_degrees.append(seg_deg)
        else:
            seg_coeffs = [float(row[col]) for col in coeff_cols]
            coefficients.extend(seg_coeffs)
            # Detect effective degree (highest non-zero coefficient)
            seg_deg = 0
            for d in range(degree, -1, -1):
                if seg_coeffs[d] != 0.0:
                    seg_deg = d
                    break
            segment_degrees.append(seg_deg)

        # Asymptotic factoring metadata
        is_asym = row.get('is_asymptotic', '').strip().lower() == 'true'
        dom_factor = row.get('dominant_factor', '').strip() if is_asym else ''
        asymptotic_flags.append(is_asym)
        dominant_factors.append(dom_factor)

num_segments = len(boundaries) - 1
input_min = float(range_min_override) if range_min_override else boundaries[0]
input_max = float(range_max_override) if range_max_override else boundaries[-1]

print(f'Segments: {num_segments}')
print(f'Range: [{input_min}, {input_max}]')
print(f'LUT entries: {len(boundaries)} boundaries + {len(coefficients)} coefficients = {len(boundaries) + len(coefficients)} total')

# Clamp float32
def clamp(v):
    if abs(v) > 3.4028234663852886e38:
        return 3.4028234663852886e38 if v > 0 else -3.4028234663852886e38
    if abs(v) < 1.4e-45:
        return 0.0
    return v

if is_rational:
    # Rational LUT layout: boundaries + [num_coeffs_seg0 + den_coeffs_seg0 + num_coeffs_seg1 + ...]
    lut_values = boundaries + coefficients
else:
    lut_values = boundaries + coefficients
lut_size = len(lut_values)
lut_str = ',\n    '.join(f'{clamp(v):.10e}f' for v in lut_values)

# Per-segment adaptive degree (skip wasted FMA on zero high-order coefficients)
# Two mechanisms:
#   1. SEGi_DEGREE macros — used by hand-written 4/8/16/32 specializations
#   2. SEGMENT_DEGREES[] constexpr array — used by recursive _N unroller
degree_macros = ''
if any(d < degree for d in segment_degrees):
    # SEGi_DEGREE macros for hand-written specializations (4/8/16/32)
    seg_macro_lines = [f'#define SEG{i}_DEGREE {d}' for i, d in enumerate(segment_degrees)]
    # constexpr array for recursive unroller (any segment count)
    deg_array = ', '.join(str(d) for d in segment_degrees)
    degree_macros = (
        '\n#ifndef DISABLE_ADAPTIVE_DEGREE\n'
        + '\n'.join(seg_macro_lines) + '\n'
        + f'#define HAS_SEGMENT_DEGREES\n'
        + f'constexpr uint32_t SEGMENT_DEGREES[] = {{{deg_array}}};\n'
        + '#endif\n'
    )
    reduced = sum(1 for d in segment_degrees if d < degree)
    avg_deg = sum(segment_degrees) / len(segment_degrees)
    print(f'Adaptive degree: {reduced}/{num_segments} segments reduced (avg effective degree: {avg_deg:.1f} vs max {degree})')

# Detect parity from coefficient values
poly_parity_macro = ''
threshold = 1e-30
if is_rational and num_degree >= 2:
    # Rational parity: check num (odd-index) and den (even-index) separately
    num_even_zero = True  # even-index num coeffs zero → odd numerator
    den_odd_zero = True   # odd-index den coeffs zero → even denominator
    ncps = num_degree + 1
    dcps = den_degree + 1
    for s in range(num_segments):
        seg_num = num_coefficients[s * ncps : (s + 1) * ncps]
        seg_den = den_coefficients[s * dcps : (s + 1) * dcps]
        for i in range(ncps):
            if i % 2 == 0 and abs(seg_num[i]) > threshold:
                num_even_zero = False
        for i in range(dcps):
            if i % 2 == 1 and abs(seg_den[i]) > threshold:
                den_odd_zero = False
    if num_even_zero and den_odd_zero:
        poly_parity_macro = '\n// Rational parity: odd num / even den -> x^2-Horner\n#define RATIONAL_NUM_PARITY_ODD\n#define RATIONAL_DEN_PARITY_EVEN\n'
        print(f'Rational parity: odd num / even den -> x^2-Horner enabled')
elif not is_rational and degree >= 2:
    # Polynomial parity
    cps = degree + 1
    seg_coeffs_list = []
    for s in range(num_segments):
        seg_coeffs_list.append(coefficients[s * cps : (s + 1) * cps])

    even_all_zero = True
    odd_all_zero = True
    for seg in seg_coeffs_list:
        for i in range(degree + 1):
            if i % 2 == 0 and abs(seg[i]) > threshold:
                even_all_zero = False
            if i % 2 == 1 and abs(seg[i]) > threshold:
                odd_all_zero = False

    if even_all_zero:
        poly_parity_macro = '\n// Polynomial parity: odd function (c0=c2=c4=...=0) -> x^2-Horner\n#define POLY_PARITY_ODD\n'
        print(f'Polynomial parity: ODD (c0=c2=c4=...=0) -> x^2-Horner enabled')
    elif odd_all_zero:
        poly_parity_macro = '\n// Polynomial parity: even function (c1=c3=c5=...=0) -> x^2-Horner\n#define POLY_PARITY_EVEN\n'
        print(f'Polynomial parity: EVEN (c1=c3=c5=...=0) -> x^2-Horner enabled')

# Check for range reduction
rr_macro = ''
rr_method = metadata.get('range_reduction_method', '')
if rr_method == 'exp':
    rr_macro = '\n#define RANGE_REDUCTION_EXP\n'
    print(f'Range reduction: exp')
elif rr_method == 'trig':
    rr_macro = '\n#define RANGE_REDUCTION_TRIG\n'
    print(f'Range reduction: trig')
elif rr_method == 'log':
    expand_const = metadata.get('log_ln2_constant', '0.6931471805599453')
    rr_macro = f'\n#define RANGE_REDUCTION_LOG\n#define LOG_EXPAND_CONSTANT {expand_const}f\n'
    print(f'Range reduction: log (expand_const={expand_const})')
elif rr_method == 'tan':
    rr_macro = '\n#define RANGE_REDUCTION_TAN\n'
    print(f'Range reduction: tan')
elif rr_method == 'cbrt':
    rr_macro = '\n#define RANGE_REDUCTION_CBRT\n'
    print(f'Range reduction: cbrt')

# Detect asymptotic factoring from CSV columns
DOMINANT_FACTOR_MAP = {
    '-exp(-x^2/2) / sqrt(2*pi)': ('EXP_QUADRATIC', -0.5, -1.0 / math.sqrt(2 * math.pi)),
    'exp(-x^2/2) / sqrt(2*pi)': ('EXP_QUADRATIC', -0.5, 1.0 / math.sqrt(2 * math.pi)),
    'exp(x)': ('EXP_LINEAR', 1.0, 1.0),
    'exp(-x)': ('EXP_LINEAR', -1.0, 1.0),
    '-exp(-x)': ('EXP_LINEAR', -1.0, -1.0),
    'x * exp(x)': ('X_EXP_LINEAR', 1.0, 1.0),
    'x': ('X', 0.0, 1.0),
}

asymptotic_macro = ''
if any(asymptotic_flags):
    active_factors = [f for f, a in zip(dominant_factors, asymptotic_flags) if a and f]
    unique_factors = set(active_factors)
    if len(unique_factors) == 1 and active_factors[0] in DOMINANT_FACTOR_MAP:
        dom_str = active_factors[0]
        factor_class, arg_scale, output_scale = DOMINANT_FACTOR_MAP[dom_str]
        asymptotic_macro = f'\n// Asymptotic factoring: {dom_str}\n#define ASYMPTOTIC_FACTOR_{factor_class}\n'
        if factor_class != 'X':
            asymptotic_macro += f'constexpr float ASYMPTOTIC_EXP_ARG_SCALE = {arg_scale:.16e}f;\n'
        asymptotic_macro += f'constexpr float ASYMPTOTIC_SCALE = {output_scale:.16e}f;\n'
        # Determine bound: left tail (x < bound) or right tail (x > bound)
        first_asym = next(i for i, a in enumerate(asymptotic_flags) if a)
        last_asym = next(i for i in range(num_segments - 1, -1, -1) if asymptotic_flags[i])
        if first_asym == 0 and last_asym < num_segments - 1:
            bound = boundaries[last_asym + 1]
            asymptotic_macro += f'constexpr float ASYMPTOTIC_UPPER_BOUND = {bound:.16e}f;\n'
            print(f'Asymptotic factoring: {dom_str} (left tail, x < {bound})')
        elif last_asym == num_segments - 1 and first_asym > 0:
            bound = boundaries[first_asym]
            asymptotic_macro += f'constexpr float ASYMPTOTIC_LOWER_BOUND = {bound:.16e}f;\n'
            print(f'Asymptotic factoring: {dom_str} (right tail, x > {bound})')
        elif first_asym == 0 and last_asym == num_segments - 1:
            asymptotic_macro += 'constexpr float ASYMPTOTIC_UPPER_BOUND = 1.0e38f;\n'
            print(f'Asymptotic factoring: {dom_str} (all segments)')
        else:
            asymptotic_macro = ''
            print(f'WARNING: non-contiguous asymptotic segments, skipping')
    elif len(unique_factors) > 1:
        print(f'WARNING: mixed dominant factors not supported: {unique_factors}')

# Write kernel .cpp
if is_rational:
    kernel = f'''// Auto-generated by run_csv.sh from: {os.path.basename(csv_path)}
// Rational n{num_degree}/d{den_degree}, {num_segments} segments, range [{input_min}, {input_max}]
#include <array>
#include <cstdint>

#define EMBEDDED_LUT
constexpr uint32_t NUM_DEGREE = {num_degree};
constexpr uint32_t DEN_DEGREE = {den_degree};
constexpr uint32_t NUM_SEGMENTS = {num_segments};

constexpr float INPUT_MIN = {input_min:.10e}f;
constexpr float INPUT_MAX = {input_max:.10e}f;

constexpr uint32_t LUT_SIZE = {lut_size};
constexpr std::array<float, LUT_SIZE> LUT_DATA = {{{{
    {lut_str}
}}}};
{poly_parity_macro}{rr_macro}{asymptotic_macro}
#include \"../piecewise_rational.cpp\"
'''
    # Override degree-related output variables for rational
    degree = num_degree
else:
    kernel = f'''// Auto-generated by run_csv.sh from: {os.path.basename(csv_path)}
// Degree {degree}, {num_segments} segments, range [{input_min}, {input_max}]
#include <array>
#include <cstdint>

#define EMBEDDED_LUT
constexpr uint32_t POLY_DEGREE = {degree};
constexpr uint32_t NUM_SEGMENTS = {num_segments};

constexpr float INPUT_MIN = {input_min:.10e}f;
constexpr float INPUT_MAX = {input_max:.10e}f;

constexpr uint32_t LUT_SIZE_BF16 = {lut_size};
constexpr std::array<float, LUT_SIZE_BF16> LUT_DATA_BF16 = {{{{
    {lut_str}
}}}};

constexpr uint32_t LUT_SIZE_FP32 = {lut_size};
constexpr std::array<float, LUT_SIZE_FP32> LUT_DATA_FP32 = {{{{
    {lut_str}
}}}};

#ifdef USE_BF16
    constexpr auto& LUT_DATA = LUT_DATA_BF16;
    constexpr uint32_t LUT_SIZE = LUT_SIZE_BF16;
#else
    constexpr auto& LUT_DATA = LUT_DATA_FP32;
    constexpr uint32_t LUT_SIZE = LUT_SIZE_FP32;
#endif
{degree_macros}{poly_parity_macro}{rr_macro}{asymptotic_macro}
#include \"../piecewise_generic.cpp\"
'''

with open(kernel_path, 'w') as f:
    f.write(kernel)

print(f'Generated: {kernel_path}')

# Write detected values to stdout for bash to capture
print(f'DETECTED_RANGE_MIN={input_min}')
print(f'DETECTED_RANGE_MAX={input_max}')
print(f'DETECTED_DEGREE={degree}')
print(f'DETECTED_SEGMENTS={num_segments}')
print(f'DETECTED_DEGREE_SUM={sum(segment_degrees)}')
print(f'DETECTED_IS_RATIONAL={1 if is_rational else 0}')
if is_rational:
    print(f'DETECTED_NUM_DEGREE={num_degree}')
    print(f'DETECTED_DEN_DEGREE={den_degree}')
" 2>&1 | tee /tmp/run_csv_gen.log

# Extract detected values from Python output
if [[ -z "$RANGE_MIN" ]]; then
    RANGE_MIN=$(grep '^DETECTED_RANGE_MIN=' /tmp/run_csv_gen.log | tail -1 | cut -d= -f2)
fi
if [[ -z "$RANGE_MAX" ]]; then
    RANGE_MAX=$(grep '^DETECTED_RANGE_MAX=' /tmp/run_csv_gen.log | tail -1 | cut -d= -f2)
fi
POLY_DEGREE=$(grep '^DETECTED_DEGREE=' /tmp/run_csv_gen.log | tail -1 | cut -d= -f2)
NUM_SEGMENTS=$(grep '^DETECTED_SEGMENTS=' /tmp/run_csv_gen.log | tail -1 | cut -d= -f2)
COEFFS=$(grep '^DETECTED_DEGREE_SUM=' /tmp/run_csv_gen.log | tail -1 | cut -d= -f2)
IS_RATIONAL=$(grep '^DETECTED_IS_RATIONAL=' /tmp/run_csv_gen.log | tail -1 | cut -d= -f2)
[[ -z "$COEFFS" || "$COEFFS" == "0" ]] && COEFFS=$((POLY_DEGREE * NUM_SEGMENTS))

if [[ "$IS_RATIONAL" == "1" ]]; then
    NUM_DEG=$(grep '^DETECTED_NUM_DEGREE=' /tmp/run_csv_gen.log | tail -1 | cut -d= -f2)
    DEN_DEG=$(grep '^DETECTED_DEN_DEGREE=' /tmp/run_csv_gen.log | tail -1 | cut -d= -f2)
    CONFIG_NAME="n${NUM_DEG}d${DEN_DEG}_s${NUM_SEGMENTS}"
    COEFFS=$(( (NUM_DEG + 1 + DEN_DEG + 1) * NUM_SEGMENTS ))
else
    CONFIG_NAME="p${POLY_DEGREE}_s${NUM_SEGMENTS}"
fi

echo "Config:     $CONFIG_NAME ($COEFFS coeffs/segment)"
echo ""

# Build list of precisions to iterate for accuracy/reporting
if [[ "$PRECISION" == "both" ]]; then
    PRECISION_LIST=(bf16 fp32)
else
    PRECISION_LIST=("$PRECISION")
fi

# --- Step 2: Build ---
if [[ "$SKIP_BUILD" == true ]]; then
    echo "Skipping build (--skip-build)"
else
    echo "Building adhoc target..."
    cd "$REPO_ROOT"
    ninja -C "$BUILD_DIR" programming_examples_generic_lut_activation_embedded_adhoc
    echo "Build complete."
fi

echo ""

# --- Step 3: Run all shapes with profiler timing and accuracy ---
cd "$REPO_ROOT"

# Hardware output directory for accuracy CSVs
output_dir=$(get_hardware_output_dir "$ACTIVATION" "$WORK_DIR")

# Build batch tile list from TEST_SHAPES
batch_tiles=""
declare -A shape_name_by_tiles=()
for test_shape in "${TEST_SHAPES[@]}"; do
    parse_shape "$test_shape"
    batch_tiles="${batch_tiles:+$batch_tiles,}$tile_count"
    shape_name_by_tiles[$tile_count]="$shape_name"
done

# Determine if we should use batch mode (multiple shapes → single binary invocation)
use_batch=false
if [[ ${#TEST_SHAPES[@]} -gt 1 ]]; then
    use_batch=true
fi

# Print table header (matching sweep_best.sh format)
echo "=== $ACTIVATION ($PRECISION, range: $RANGE_MIN to $RANGE_MAX) ==="
printf "%-25s %8s %12s %12s %12s %12s %10s %10s\n" "Config" "DegSum" "MAE" "MaxErr" "MaxULP" "MeanULP" "Prof(µs)" "Host(ms)"
printf "%-25s %8s %12s %12s %12s %12s %10s %10s\n" "-------------------------" "--------" "------------" "------------" "------------" "------------" "----------" "----------"

# Per-shape result accumulators (associative arrays keyed by shape_name)
declare -A profiler_times_csv=()   # shape_name → "t1,t2,t3" (comma-separated across runs)
declare -A host_times_csv=()       # shape_name → "t1,t2,t3"
declare -A csv_output_paths=()     # shape_name → path to hardware output CSV

# Precompute per-shape, per-precision CSV output paths
# Must match what the C++ binary produces from DUMP_OUTPUT_CSV base path:
#   --precision both:  base (no precision) + _${prec}_tiles${N}.csv
#   single precision:  base (with precision) + _tiles${N}.csv  (backward compat)
for prec in "${PRECISION_LIST[@]}"; do
    for test_shape in "${TEST_SHAPES[@]}"; do
        parse_shape "$test_shape"
        if [[ "$PRECISION" == "both" && "$use_batch" == true ]]; then
            # Binary base: ${ACTIVATION}_${NUM_SEGMENTS}_${POLY_DEGREE}
            # Binary suffix: _${prec}_tiles${N} (is_multi_precision + is_batch_mode)
            csv_output_paths["${prec}_${shape_name}"]="${output_dir}/${ACTIVATION}_${NUM_SEGMENTS}_${POLY_DEGREE}_${prec}_tiles${tile_count}.csv"
        elif [[ "$PRECISION" == "both" ]]; then
            # Binary base: ${ACTIVATION}_${NUM_SEGMENTS}_${POLY_DEGREE}
            # Binary suffix: _${prec} (is_multi_precision only, no batch)
            csv_output_paths["${prec}_${shape_name}"]="${output_dir}/${ACTIVATION}_${NUM_SEGMENTS}_${POLY_DEGREE}_${prec}.csv"
        elif [[ "$use_batch" == true ]]; then
            # Single precision batch: base has precision, binary adds _tiles${N}
            csv_output_paths["${prec}_${shape_name}"]="${output_dir}/${ACTIVATION}_${PRECISION}_${NUM_SEGMENTS}_${POLY_DEGREE}_tiles${tile_count}.csv"
        else
            # Single precision, single shape: exact path (no suffix added by binary)
            csv_output_paths["${prec}_${shape_name}"]="${output_dir}/${ACTIVATION}_${PRECISION}_${NUM_SEGMENTS}_${POLY_DEGREE}_tiles${tile_count}.csv"
        fi
    done
done

# ===== STEP 1: Profiler + host timing + accuracy (single pass) =====
# First run also dumps hardware output for accuracy computation
profiler_success=true
host_success=true

for run in $(seq 1 $NUM_RUNS); do
    PROFILER_BASE="$WORK_DIR/profiler_results/reports/adhoc_${ACTIVATION}_run${run}"
    mkdir -p "$PROFILER_BASE"

    # Dump hardware output on first run (for accuracy computation)
    if [[ "$run" -eq 1 ]]; then
        if [[ "$PRECISION" == "both" ]]; then
            # Binary inserts _bf16/_fp32 and _tilesN before .csv (is_multi_precision + is_batch_mode)
            export DUMP_OUTPUT_CSV="${output_dir}/${ACTIVATION}_${NUM_SEGMENTS}_${POLY_DEGREE}.csv"
        elif [[ "$use_batch" == true ]]; then
            # Single precision, binary inserts _tilesN before .csv
            export DUMP_OUTPUT_CSV="${output_dir}/${ACTIVATION}_${PRECISION}_${NUM_SEGMENTS}_${POLY_DEGREE}.csv"
        else
            parse_shape "${TEST_SHAPES[0]}"
            export DUMP_OUTPUT_CSV="${csv_output_paths[${PRECISION_LIST[0]}_${shape_name}]}"
        fi
    else
        unset DUMP_OUTPUT_CSV
    fi

    set +e
    if [[ "$use_batch" == true ]]; then
        run_output=$(TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$PROFILER_BASE" \
            "$BINARY" --activation "$ACTIVATION" --precision "$PRECISION" \
            --range-min "$RANGE_MIN" --range-max "$RANGE_MAX" \
            --batch-tiles "$batch_tiles" "${EXTRA_BINARY_ARGS[@]}" 2>&1)
    else
        parse_shape "${TEST_SHAPES[0]}"
        run_output=$(TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$PROFILER_BASE" \
            "$BINARY" --activation "$ACTIVATION" --precision "$PRECISION" \
            --range-min "$RANGE_MIN" --range-max "$RANGE_MAX" --tiles "$tile_count" \
            "${EXTRA_BINARY_ARGS[@]}" 2>&1)
    fi
    run_exit=$?
    set -e

    if [[ $run_exit -ne 0 ]]; then
        profiler_success=false
        host_success=false
        break
    fi

    # Extract per-shape profiler times from single CSV (batch clustering)
    PROFILER_CSV_PATH="${PROFILER_BASE}/.logs/profile_log_device.csv"
    if [[ -f "$PROFILER_CSV_PATH" ]]; then
        total_shapes=$(( ${#PRECISION_LIST[@]} * ${#TEST_SHAPES[@]} ))
        batch_times=$(extract_batch_profiler_times "$PROFILER_CSV_PATH" "$total_shapes" "$WORK_DIR")
        i=0
        IFS=',' read -ra _ptimes <<< "$batch_times"
        for prec in "${PRECISION_LIST[@]}"; do
            for test_shape in "${TEST_SHAPES[@]}"; do
                parse_shape "$test_shape"
                pkey="${prec}_${shape_name}"
                ptime="${_ptimes[$i]:-0}"
                if [[ -n "$ptime" && "$ptime" != "0" && "$ptime" != "0.00" ]]; then
                    profiler_times_csv[$pkey]="${profiler_times_csv[$pkey]:+${profiler_times_csv[$pkey]},}$ptime"
                fi
                i=$((i + 1))
            done
        done
    fi

    # Extract per-shape, per-precision host timing from same output
    for prec in "${PRECISION_LIST[@]}"; do
        for test_shape in "${TEST_SHAPES[@]}"; do
            parse_shape "$test_shape"
            pkey="${prec}_${shape_name}"
            if [[ "$PRECISION" == "both" ]]; then
                # Multi-precision format: BATCH[bf16,tiles=N]:TIMING_KERNEL_EXECUTION:
                kernel_exec=$(echo "$run_output" | grep "BATCH\[${prec},tiles=${tile_count}\]:TIMING_KERNEL_EXECUTION:" | awk -F': ' '{print $2}')
            elif [[ "$use_batch" == true ]]; then
                # Single-precision batch: BATCH[tiles=N]:TIMING_KERNEL_EXECUTION:
                kernel_exec=$(echo "$run_output" | grep "BATCH\[tiles=${tile_count}\]:TIMING_KERNEL_EXECUTION:" | awk -F': ' '{print $2}')
            else
                # Single shape, single precision: TIMING_KERNEL_EXECUTION:
                kernel_exec=$(echo "$run_output" | grep "TIMING_KERNEL_EXECUTION:" | head -1 | awk -F': ' '{print $2}')
            fi
            if [[ -n "$kernel_exec" ]]; then
                host_times_csv[$pkey]="${host_times_csv[$pkey]:+${host_times_csv[$pkey]},}$kernel_exec"
            fi
        done
    done
done
unset DUMP_OUTPUT_CSV

# ===== STEP 3: Compute results per shape and print table =====
# Arrays to collect results for summary table
declare -a result_configs=()
declare -a result_coeffs=()
declare -a result_mae=()
declare -a result_max_err=()
declare -a result_max_ulp=()
declare -a result_mean_ulp=()
declare -a result_profiler_us=()
declare -a result_host_ms=()

all_runs_passed=$( [[ "$profiler_success" == true && "$host_success" == true ]] && echo true || echo false )

for prec in "${PRECISION_LIST[@]}"; do
    for test_shape in "${TEST_SHAPES[@]}"; do
        parse_shape "$test_shape"
        pkey="${prec}_${shape_name}"
        csv_output="${csv_output_paths[$pkey]}"

        # If --dump-csv was specified, copy first shape's output to user-requested path
        if [[ -n "$DUMP_CSV" && "${test_shape}" == "${TEST_SHAPES[0]}" && "$prec" == "${PRECISION_LIST[0]}" && -f "$csv_output" ]]; then
            cp "$csv_output" "$DUMP_CSV"
        fi

        if [[ "$all_runs_passed" == true ]]; then
            # Compute profiler timing (in us)
            profiler_min_us="0"
            if [[ -n "${profiler_times_csv[$pkey]:-}" ]]; then
                profiler_min_us=$(compute_kernel_exec_min "${profiler_times_csv[$pkey]}")
            fi

            # Compute host timing (in ms)
            host_min_ms="0"
            if [[ -n "${host_times_csv[$pkey]:-}" ]]; then
                host_min_ms=$(compute_kernel_exec_min "${host_times_csv[$pkey]}")
            fi

            # Compute accuracy metrics
            mae_hw="0" max_hw="0" max_ulp_hw="0" mean_ulp_hw="0"
            if [[ "$HAS_ACCURACY" == true && -f "$csv_output" ]]; then
                accuracy_stats=$($ACCURACY_PYTHON "$ACCURACY_SCRIPT" "$ACTIVATION" "$csv_output" 2>/dev/null) || true
                if [[ -n "$accuracy_stats" ]]; then
                    mae_hw=$(echo "$accuracy_stats" | cut -d',' -f1)
                    max_hw=$(echo "$accuracy_stats" | cut -d',' -f3)
                    max_ulp_hw=$(echo "$accuracy_stats" | cut -d',' -f5)
                    mean_ulp_hw=$(echo "$accuracy_stats" | cut -d',' -f6)
                fi
            fi

            # Compress hardware output CSV to save disk space (original deleted by gzip)
            if [[ -f "$csv_output" ]]; then
                gzip -f "$csv_output"
            fi

            # Print row — include precision prefix when running both
            if [[ ${#PRECISION_LIST[@]} -gt 1 ]]; then
                config_display="${prec}_${shape_name}_${CONFIG_NAME}"
            else
                config_display="${shape_name}_${CONFIG_NAME}"
            fi
            [[ "$shape_name" == "yolov4" ]] && printf "${GREEN}"
            printf "%-25s %8d %12.2e %12.2e %12.2f %12.2f %9.2fµs %9.2fms\n" \
                "$config_display" "$COEFFS" "$mae_hw" "$max_hw" "$max_ulp_hw" "$mean_ulp_hw" "$profiler_min_us" "$host_min_ms"
            [[ "$shape_name" == "yolov4" ]] && printf "${NC}"

            # Store for summary
            result_configs+=("$config_display")
            result_coeffs+=("$COEFFS")
            result_mae+=("$mae_hw")
            result_max_err+=("$max_hw")
            result_max_ulp+=("$max_ulp_hw")
            result_mean_ulp+=("$mean_ulp_hw")
            result_profiler_us+=("$profiler_min_us")
            result_host_ms+=("$host_min_ms")
        else
            if [[ ${#PRECISION_LIST[@]} -gt 1 ]]; then
                config_display="${prec}_${shape_name}_${CONFIG_NAME}"
            else
                config_display="${shape_name}_${CONFIG_NAME}"
            fi
            printf "%-25s %8d %12s %12s %12s %12s %10s\n" "$config_display" "$COEFFS" "-" "-" "-" "-" "FAILED"

            result_configs+=("$config_display")
            result_coeffs+=("$COEFFS")
            result_mae+=("-")
            result_max_err+=("-")
            result_max_ulp+=("-")
            result_mean_ulp+=("-")
            result_profiler_us+=("-")
            result_host_ms+=("-")
        fi
    done
done

# --- Summary table ---
echo ""
echo "================================================================================"
echo "  SUMMARY: $ACTIVATION  $PRECISION  $(basename "$CSV_FILE")"
echo "================================================================================"
printf "%-25s %8s %12s %12s %12s %12s %10s %10s\n" "Config" "DegSum" "MAE" "MaxErr" "MaxULP" "MeanULP" "Prof(µs)" "Host(ms)"
printf "%-25s %8s %12s %12s %12s %12s %10s %10s\n" "-------------------------" "--------" "------------" "------------" "------------" "------------" "----------" "----------"

for i in "${!result_configs[@]}"; do
    if [[ "${result_mae[$i]}" == "-" ]]; then
        printf "%-25s %8s %12s %12s %12s %12s %10s\n" \
            "${result_configs[$i]}" "${result_coeffs[$i]}" "-" "-" "-" "-" "FAILED"
    else
        [[ "${result_configs[$i]}" == yolov4_* ]] && printf "${GREEN}"
        printf "%-25s %8d %12.2e %12.2e %12.2f %12.2f %9.2fµs %9.2fms\n" \
            "${result_configs[$i]}" "${result_coeffs[$i]}" \
            "${result_mae[$i]}" "${result_max_err[$i]}" \
            "${result_max_ulp[$i]}" "${result_mean_ulp[$i]}" \
            "${result_profiler_us[$i]}" "${result_host_ms[$i]}"
        [[ "${result_configs[$i]}" == yolov4_* ]] && printf "${NC}"
    fi
done

echo "================================================================================"
