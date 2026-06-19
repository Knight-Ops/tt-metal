#!/bin/bash
set -e

# =============================================================================
# Source common argument parsing library
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/argparse_common.sh"
source "$SCRIPT_DIR/sweep_helpers.sh"
source "$SCRIPT_DIR/profiler_helpers.sh"

# =============================================================================
# Initialize Common Environment
# =============================================================================

init_common_env
cd "$REPO_ROOT"
WORK_DIR="tt_metal/programming_examples/generic_lut_activation"

# Ensure sweep_config.dat is up-to-date, then load it
ensure_sweep_config "$WORK_DIR"
CONFIG_FILE="$WORK_DIR/sweep_config.dat"
if [[ -f "$CONFIG_FILE" ]]; then
    eval "$(grep -v '^\s*#' "$CONFIG_FILE" | grep -v '^\s*$' | sed 's/,/ /g' | sed 's/\([A-Z_]*\)=/\1=(/' | sed 's/$/)/')"
fi

# =============================================================================
# Helper Functions (now in sweep_helpers.sh)
# =============================================================================
# get_test_range() - moved to sweep_helpers.sh

# =============================================================================
# Script-Specific Variables
# =============================================================================

# Note: NUM_RUNS is set to 2 (override common default of 3)
NUM_RUNS=2

# Tracy profiling - always run both modes to get accurate Prof(µs) and Host(ms)
TRACY_ITERATIONS=10

# =============================================================================
# Script-Specific Help
# =============================================================================

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Tests native hardware SFPU implementations (no LUT)"
    echo ""
    show_common_options
    echo ""
    echo "Script-Specific Options:"
    echo "  --sfpu-mode <mode>        SFPU mode: fast, precise, both (default: precise)"
    echo "  --depth                   Accepted but ignored (for consistency with other scripts)"
    echo "  --tracy-iterations N      Number of Tracy benchmark iterations (default: 10)"
    echo ""
    show_common_env_vars
    echo ""
    echo "Examples:"
    echo "  $0                        # Run all native SFPU activations, both precisions"
    echo "  $0 --activation gelu      # Run gelu only"
    echo "  $0 --precision bf16       # Run all activations in bf16 only"
    echo "  $0 --sfpu-mode precise    # Run only precise mode (skip fast-approx)"
    echo "  $0 --dry-run              # Show what would be tested"
    echo "  $0 --skip-run             # Load and display existing results only"
    echo ""
    exit 0
}

# =============================================================================
# Parse Arguments
# =============================================================================

# Parse common arguments first
parse_common_args "$@"
set -- "${PARSED_ARGS[@]}"

# Show help if requested
if [[ "$HELP_REQUESTED" == "true" ]]; then
    show_help
fi

# Parse script-specific arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --depth)
            # Native SFPU doesn't use depths, but accept for consistency
            shift 2
            ;;
        --sfpu-mode)
            FILTER_SFPU_MODE="$2"
            if [[ "$FILTER_SFPU_MODE" != "fast" && "$FILTER_SFPU_MODE" != "precise" && "$FILTER_SFPU_MODE" != "both" ]]; then
                echo "Error: --sfpu-mode must be 'fast', 'precise', or 'both'" >&2
                exit 1
            fi
            shift 2
            ;;
        --tracy-iterations)
            TRACY_ITERATIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Configuration
# =============================================================================

# Determine which precisions to run
if [[ -z "$FILTER_PRECISION" ]]; then
    PRECISIONS=("bf16" "fp32")  # Run both by default
else
    PRECISIONS=("$FILTER_PRECISION")  # Run only specified precision
fi

# SFPU modes to test (can be filtered by --sfpu-mode flag)
if [[ "$FILTER_SFPU_MODE" == "both" ]]; then
    SFPU_MODES=("fast" "precise")
elif [[ -n "$FILTER_SFPU_MODE" ]]; then
    SFPU_MODES=("$FILTER_SFPU_MODE")
else
    SFPU_MODES=("precise")  # Default: precise only
fi

# Configure test parameters
if [[ -n "$FILTER_ACTIVATION" ]]; then
    # Comma-separated list or single activation
    IFS=',' read -ra ACTIVATIONS <<< "$FILTER_ACTIVATION"
else
    # Use only activations with native SFPU support (auto-generated from activations/*.json)
    ACTIVATIONS=("${NATIVE_SFPU_ACTIVATIONS[@]}")
fi

# Test configurations: name:rows:cols format (each tile is 32×32)
# Matches standard op-runner test cases for comparable results
# Use standard test shapes from sweep_helpers.sh (filtered by --shape if specified)
TEST_SHAPES=("${STANDARD_TEST_SHAPES[@]}")
apply_shape_filter
# Native SFPU hardware outputs go to per-activation directories via get_hardware_output_dir()
# No need to create OUTPUT_DIR here - it's created per-activation in the execution loop

# =============================================================================
# Get Results File Path
# =============================================================================

RESULTS_FILE=$(get_results_file "native_sfpu" "$WORK_DIR")
EXPECTED_HEADER="precision,activation,sfpu_mode,shape,time_profiler_us,time_host_ms,mae,rmse,max_error,mean_rel_error,max_ulp_error,mean_ulp_error,median_ulp_error,p99_ulp_error,status"


echo "================================================================================"
echo "        Activation Function Sweep (via TTNN)"
echo "================================================================================"
echo "Architecture: $ARCH_NAME"
echo "Activations: ${ACTIVATIONS[@]}"
echo "Precisions: ${PRECISIONS[@]}"
echo "SFPU modes: ${SFPU_MODES[@]}"
echo "Test shapes:"
for shape in "${TEST_SHAPES[@]}"; do
    echo "  $shape"
done
echo "Runs per config: $NUM_RUNS"
echo "Timeout per test: ${TIMEOUT_SECONDS}s"
echo "Skip build: $SKIP_BUILD (no build required)"
echo "Skip run: $SKIP_RUN"
echo "Dry run: $DRY_RUN"
echo "Tracy iterations: $TRACY_ITERATIONS (for profiler timing)"
echo "Host timing runs: $NUM_RUNS"
echo "Results file: $RESULTS_FILE"
echo ""
echo "Note: All activations tested via TTNN API (native SFPU + composite ops)"
echo ""

# =============================================================================
# Dry-Run Mode
# =============================================================================

if [[ "$DRY_RUN" == true ]]; then
    echo "================================================================================"
    echo "DRY-RUN MODE: Showing what would be tested"
    echo "================================================================================"
    echo ""

    PYTHON_RUNNER="$WORK_DIR/ttnn_runner.py"
    TRACY_RUNNER="$WORK_DIR/ttnn_tracy_runner.py"

    if [[ -f "$TRACY_RUNNER" ]]; then
        echo "✓ Tracy runner found: $TRACY_RUNNER"
    else
        echo "❌ Tracy runner not found: $TRACY_RUNNER"
    fi
    if [[ -f "$PYTHON_RUNNER" ]]; then
        echo "✓ Python runner found: $PYTHON_RUNNER"
    else
        echo "❌ Python runner not found: $PYTHON_RUNNER"
    fi
    echo ""

    config_num=0
    for precision in "${PRECISIONS[@]}"; do
        for sfpu_mode in "${SFPU_MODES[@]}"; do
            for activation in "${ACTIVATIONS[@]}"; do
                config_num=$((config_num + 1))
                if [[ "$precision" == "bf16" && "$USE_FITTING_RANGE" != "true" ]]; then
                    range_min="-3.3895e+38"; range_max="3.3895e+38"
                else
                    read -r range_min range_max < <(get_test_range "$activation")
                fi
                printf "%3d. %-12s %-6s %-8s (range: %4s to %4s)\n" \
                    "$config_num" "$activation" "$precision" "$sfpu_mode" "$range_min" "$range_max"
            done
        done
    done

    echo ""
    echo "Total configurations: $config_num"
    echo ""
    echo "To run the sweep, remove --dry-run flag"
    echo ""
    exit 0
fi

# =============================================================================
# Skip-Run Mode: Load and Display Existing Results
# =============================================================================

if [[ "$SKIP_RUN" == true ]]; then
    echo "================================================================================"
    echo "SKIP-RUN MODE: Loading results from $RESULTS_FILE"
    echo "================================================================================"
    echo ""

    if [[ ! -f "$RESULTS_FILE" ]]; then
        echo "❌ ERROR: Results file not found: $RESULTS_FILE"
        echo ""
        echo "Run without --skip-run to generate results first."
        echo ""
        exit 1
    fi

    # Display summary for each precision
    for CURRENT_PRECISION in "${PRECISIONS[@]}"; do
        echo "================================================================================"
        echo "                    RESULTS SUMMARY (${CURRENT_PRECISION^^})"
        echo "================================================================================"
        echo ""

        export CURRENT_PRECISION  # Export for Python script

        python3 << 'EOFPYTHON'
import csv
import os
import sys
from collections import defaultdict

precision = os.environ.get('CURRENT_PRECISION', 'bf16')
results_file = os.environ.get('RESULTS_FILE', 'data/native_sfpu_results.csv')

try:
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r['status'] == 'pass' and r['precision'] == precision]
except FileNotFoundError:
    print(f"ERROR: Results file not found: {results_file}")
    sys.exit(1)

if not rows:
    print(f"No successful results found for {precision.upper()}")
    sys.exit(0)

print("=" * 100)
print(f"Native SFPU Performance ({precision.upper()})")
print("=" * 100)

# Group by (activation, sfpu_mode) — each shape is a separate row
grouped = defaultdict(dict)
for r in rows:
    key = (r['activation'], r['sfpu_mode'])
    grouped[key][r['shape']] = r

shape_order = ['single_tile', '8_tiles', '256_tiles', 'height_sharded', 'yolov4']

print(f"\n{'Activation':<15} {'Mode':<8} {'Shape':<16} {'Prof(µs)':<10} {'Host(ms)':<10} {'MAE':<12} {'MaxULP':<12} {'MeanULP':<12}")
print("-" * 100)

for (activation, sfpu_mode), shapes in sorted(grouped.items()):
    for shape in shape_order:
        if shape not in shapes:
            continue
        r = shapes[shape]
        print(f"{activation:<15} {sfpu_mode:<8} {shape:<16} {float(r['time_profiler_us']):>8.2f}  {float(r['time_host_ms']):>8.2f}  {float(r['mae']):>10.2e}  {float(r['max_ulp_error']):>10.2g}  {float(r['mean_ulp_error']):>10.2g}")

print("\n" + "=" * 100)
EOFPYTHON

        echo ""
    done

    echo "================================================================================"
    echo "SKIP-RUN COMPLETE"
    echo "================================================================================"
    echo ""
    exit 0
fi

# =============================================================================
# Execution Phase
# =============================================================================

# Reset device to clear any stale profiler state from previous interrupted runs
clear_shm
reset_device 0

# Path to Python in virtual environment
PYTHON_BIN="$REPO_ROOT/python_env/bin/python3"
if [[ ! -f "$PYTHON_BIN" ]]; then
    echo "❌ ERROR: Python venv not found: $PYTHON_BIN"
    echo "   Please run: ./create_venv.sh"
    exit 1
fi

# Path to Python runners
PYTHON_RUNNER="$WORK_DIR/ttnn_runner.py"
TRACY_RUNNER="$WORK_DIR/ttnn_tracy_runner.py"

# Check if both runners exist (we need both now)
if [[ ! -f "$TRACY_RUNNER" ]]; then
    echo "❌ ERROR: Tracy runner not found: $TRACY_RUNNER"
    echo "   This file should exist in the repository"
    echo ""
    exit 1
fi

if [[ ! -f "$PYTHON_RUNNER" ]]; then
    echo "❌ ERROR: Python runner not found: $PYTHON_RUNNER"
    echo "   This file should exist in the repository"
    echo ""
    exit 1
fi

# Always recreate results file fresh (avoids stale rows from previous runs)
mkdir -p "$(dirname "$RESULTS_FILE")"
echo "$EXPECTED_HEADER" > "$RESULTS_FILE"

# Initialize per-activation files for all-activations mode
if [[ -z "$FILTER_ACTIVATION" ]]; then
    # All-activations mode: recreate per-activation files fresh
    per_activation_dir="$WORK_DIR/data/${ARCH_NAME/_b0/}/native_sfpu"
    mkdir -p "$per_activation_dir"
    for activation in "${ACTIVATIONS[@]}"; do
        echo "$EXPECTED_HEADER" > "$per_activation_dir/${activation}.csv"
    done
fi

# Loop through each precision
for CURRENT_PRECISION in "${PRECISIONS[@]}"; do
for CURRENT_SFPU_MODE in "${SFPU_MODES[@]}"; do

echo "================================================================================"
echo "        Native SFPU - ${CURRENT_PRECISION^^} - ${CURRENT_SFPU_MODE^^}"
echo "================================================================================"
echo "Activations: ${ACTIVATIONS[@]}"
echo "Test shapes: ${#TEST_SHAPES[@]} configurations"
echo "Runs per config: $NUM_RUNS"
echo ""

# Determine flag for SFPU mode
if [[ "$CURRENT_SFPU_MODE" == "fast" ]]; then
    SFPU_FLAG="--fast-approx"
else
    SFPU_FLAG="--no-fast-approx"
fi

# Track current activation for header display
current_activation=""

for activation in "${ACTIVATIONS[@]}"; do
    # Get test range: default full bf16 range, --use-fitting-range restricts to JSON domain
    if [[ "$CURRENT_PRECISION" == "bf16" && "$USE_FITTING_RANGE" != "true" ]]; then
        range_min="-3.3895e+38"
        range_max="3.3895e+38"
    else
        read -r range_min range_max < <(get_test_range "$activation")
    fi

    # Print header for new activation
    if [[ "$activation" != "$current_activation" ]]; then
        current_activation="$activation"
        echo ""
        echo "=== $activation ($CURRENT_PRECISION, $CURRENT_SFPU_MODE, range: $range_min to $range_max) ==="
        # Always use µs for direct comparison between Tracy and standard mode
        printf "%-25s %8s %12s %12s %12s %12s %10s %10s\n" "Config" "Coeffs" "MAE" "MaxErr" "MaxULP" "MeanULP" "Prof(µs)" "Host(ms)"
        printf "%-25s %8s %12s %12s %12s %12s %10s %10s\n" "-------------------------" "--------" "------------" "------------" "------------" "------------" "----------" "----------"
    fi

    # Collect timing for all tile sizes, then output single row
    declare -A profiler_timing_us   # Store profiler timing (µs) for each shape
    declare -A host_timing_ms       # Store host timing (ms) for each shape
    declare -A mae_metrics          # Store MAE for each shape
    declare -A rmse_metrics         # Store RMSE for each shape
    declare -A max_error_metrics    # Store max error for each shape
    declare -A mean_rel_metrics     # Store mean relative error for each shape
    declare -A max_ulp_metrics      # Store max ULP error for each shape
    declare -A mean_ulp_metrics     # Store mean ULP error for each shape
    declare -A shape_status         # Store pass/fail/timeout status for each shape
    declare -A median_ulp_metrics   # Store median ULP error for each shape
    declare -A p99_ulp_metrics      # Store P99 ULP error for each shape
    config_passed=false
    config_failed=false
    config_timeout=false
    # Reset error metrics for this activation/SFPU mode combination
    config_mae=""
    config_rmse=""
    config_max=""
    config_mean_rel=""
    config_max_ulp=""
    config_mean_ulp=""

    for test_shape in "${TEST_SHAPES[@]}"; do
        # Parse shape: name:rows:cols
        shape_name="${test_shape%%:*}"
        shape_rest="${test_shape#*:}"
        shape_rows="${shape_rest%%:*}"
        shape_cols="${shape_rest#*:}"
        n_tiles=$(( (shape_rows / 32) * (shape_cols / 32) ))

        # ===== STEP 1: Run with Tracy profiling to get Prof(µs) =====

        # Preventive cleanup: clear stale shm segments before each device cycle
        # to avoid accumulation that causes open_device() to hang
        clear_shm

        # Set up output CSV path (dumped during host timing run, NOT during Tracy)
        output_dir=$(get_hardware_output_dir "$activation" "$WORK_DIR")
        output_csv="${output_dir}/native_sfpu_${activation}_${CURRENT_PRECISION}_${CURRENT_SFPU_MODE}_tiles${n_tiles}.csv"

        # Run Tracy profiling (no CSV dump — Tracy overhead on writerow() causes timeouts)
        set +e
        tracy_name="run_${shape_name}"
        tracy_output_dir="${WORK_DIR}/tracy_results/.temp_${activation}_${CURRENT_PRECISION}_${CURRENT_SFPU_MODE}"
        rm -rf "$tracy_output_dir"
        mkdir -p "$tracy_output_dir"

        # Build Tracy command with fast-approx flag if needed
        tracy_fast_flag=""
        if [[ "$CURRENT_SFPU_MODE" == "fast" ]]; then
            tracy_fast_flag="--fast-approx"
        fi

        # Increase device profiler buffer from default 1000 to 4000 ops.
        # Composite activations like multigammaln decompose into ~130 device ops
        # per iteration, so 10 iterations can generate ~1700 ops and overflow the
        # default buffer, causing "Device data mismatch" assertions in Tracy.
        # 4000 is generous enough for any activation; DRAM cost is negligible.

        # Run Tracy (without DUMP_OUTPUT_CSV to avoid profiling CSV I/O)
        # TT_METAL_HOME must be set for tracy module to find profiling tools
        tracy_output=$(source "$REPO_ROOT/python_env/bin/activate" && \
            TT_METAL_HOME="$REPO_ROOT" PYTHONNOUSERSITE=1 timeout --foreground "$TIMEOUT_SECONDS" python3 -m tracy \
            -r -p --no-runtime-analysis \
            --op-support-count 4000 \
            -o "$tracy_output_dir" \
            -n "$tracy_name" \
            "$TRACY_RUNNER" \
            "$TRACY_ITERATIONS" "$CURRENT_PRECISION" "$activation" "$shape_rows" "$shape_cols" \
            "$range_min" "$range_max" $tracy_fast_flag 2>&1)
        tracy_exit=$?
        set -e
        clear_shm  # prevent /tmp/ompi.* accumulation between iterations

        # Check Tracy exit status
        if [[ $tracy_exit -eq 124 ]]; then
            profiler_timing_us["$shape_name"]="0"
            host_timing_ms["$shape_name"]="0"
            mae_metrics["$shape_name"]="0"
            rmse_metrics["$shape_name"]="0"
            max_error_metrics["$shape_name"]="0"
            mean_rel_metrics["$shape_name"]="0"
            max_ulp_metrics["$shape_name"]="0"
            mean_ulp_metrics["$shape_name"]="0"
            shape_status["$shape_name"]="timeout"
            config_timeout=true
            printf "%-25s %8s %12s %12s %12s %12s %10s %10s\n" "${shape_name}_$CURRENT_SFPU_MODE" "-" "-" "-" "-" "-" "TIMEOUT" "-"
            # Show last 10 lines of Tracy output for debugging
            echo "  Timeout after ${TIMEOUT_SECONDS}s, last output:"
            echo "$tracy_output" | tail -10 | sed 's/^/    /'
            reset_device 0
            continue
        fi

        if [[ $tracy_exit -ne 0 ]] || ! echo "$tracy_output" | grep -q "Tracy profiling completed"; then
            # Tracy profiling failed — set profiler time to 0 but DON'T skip
            # accuracy measurement. The kernel ran; we just can't profile it.
            profiler_timing_us["$shape_name"]="0"
            echo "  Tracy profiling unavailable (exit=$tracy_exit) — continuing with host timing + accuracy"
            reset_device 0
            # Fall through to STEP 2 for host timing and accuracy
        fi

        # Extract profiler timing
        # Extract profiler timing (best-effort — may be 0 if Tracy didn't capture data)
        profiler_csv="${tracy_output_dir}/.logs/profile_log_device.csv"
        if wait_for_profiler_csv "$profiler_csv" 10; then
            profiler_compute_us=$(extract_profiler_compute_time "$profiler_csv" "$WORK_DIR")
            if [[ -n "$profiler_compute_us" && "$profiler_compute_us" != "0" ]]; then
                profiler_timing_us["$shape_name"]=$profiler_compute_us
            else
                profiler_timing_us["$shape_name"]="0"
            fi
        else
            profiler_timing_us["$shape_name"]="0"
        fi

        # ===== STEP 2: Run WITHOUT profiling to get Host(ms) =====

        # Preventive cleanup before host timing runs
        clear_shm

        host_exec_times=()
        for run in $(seq 1 $NUM_RUNS); do
            set +e
            # Dump CSV on first run only (for accuracy measurement)
            if [[ $run -eq 1 ]]; then
                host_output=$(DUMP_OUTPUT_CSV="$output_csv" timeout --foreground "$TIMEOUT_SECONDS" "$PYTHON_BIN" "$PYTHON_RUNNER" "$activation" --precision "$CURRENT_PRECISION" --range-min="$range_min" --range-max="$range_max" --rows "$shape_rows" --cols "$shape_cols" "$SFPU_FLAG" 2>&1)
            else
                host_output=$(timeout --foreground "$TIMEOUT_SECONDS" "$PYTHON_BIN" "$PYTHON_RUNNER" "$activation" --precision "$CURRENT_PRECISION" --range-min="$range_min" --range-max="$range_max" --rows "$shape_rows" --cols "$shape_cols" "$SFPU_FLAG" 2>&1)
            fi
            host_exit=$?
            set -e
            clear_shm  # prevent /tmp/ompi.* accumulation between iterations

            if [[ $host_exit -eq 0 ]] && echo "$host_output" | grep -q "Test PASSED"; then
                host_time=$(echo "$host_output" | grep "TIMING_KERNEL_EXECUTION:" | awk '{print $2}')
                host_exec_times+=("$host_time")
            fi
        done

        # Compute min host timing
        if [[ ${#host_exec_times[@]} -gt 0 ]]; then
            host_values=$(IFS=,; echo "${host_exec_times[*]}")
            host_min=$(python3 -c "import sys; values = [float(x) for x in '$host_values'.split(',') if x]; print(min(values) if values else 0)")
            host_timing_ms["$shape_name"]=$host_min
        else
            host_timing_ms["$shape_name"]="0"
        fi

        # Compute accuracy using hardware output CSV for this tile configuration
        output_dir=$(get_hardware_output_dir "$activation" "$WORK_DIR")
        output_csv="${output_dir}/native_sfpu_${activation}_${CURRENT_PRECISION}_${CURRENT_SFPU_MODE}_tiles${n_tiles}.csv"
        if [[ ! -f "$output_csv" ]]; then
            profiler_timing_us["$shape_name"]="0"
            host_timing_ms["$shape_name"]="0"
            mae_metrics["$shape_name"]="0"
            rmse_metrics["$shape_name"]="0"
            max_error_metrics["$shape_name"]="0"
            mean_rel_metrics["$shape_name"]="0"
            max_ulp_metrics["$shape_name"]="0"
            mean_ulp_metrics["$shape_name"]="0"
            median_ulp_metrics["$shape_name"]="0"
            p99_ulp_metrics["$shape_name"]="0"
            shape_status["$shape_name"]="no_csv"
            printf "%-25s %8s %12s %12s %12s %12s %10s %10s\n" "${shape_name}_${CURRENT_SFPU_MODE}" "-" "-" "-" "-" "-" "NO_CSV" "-"
            continue
        fi

        # Use centralized extract_accuracy.py
        GROUND_TRUTH_DIR="${GROUND_TRUTH:-$TT_POLY_FIT_DIR}"
        accuracy=$(python3 "$GROUND_TRUTH_DIR/extract_accuracy.py" "$activation" "$output_csv" 2>/dev/null | grep -v "Warning:")

        mae=$(echo "$accuracy" | cut -d, -f1)
        rmse=$(echo "$accuracy" | cut -d, -f2)
        max_err=$(echo "$accuracy" | cut -d, -f3)
        mre=$(echo "$accuracy" | cut -d, -f4)
        max_ulp=$(echo "$accuracy" | cut -d, -f5)
        mean_ulp=$(echo "$accuracy" | cut -d, -f6)
        median_ulp=$(echo "$accuracy" | cut -d, -f7)
        p99_ulp=$(echo "$accuracy" | cut -d, -f8)

        # Compress hardware output CSV to save disk space (original deleted by gzip)
        if [[ -f "$output_csv" ]]; then
            gzip -f "$output_csv"
        fi

        # Store error metrics for this shape
        mae_metrics["$shape_name"]=$mae
        rmse_metrics["$shape_name"]=$rmse
        max_error_metrics["$shape_name"]=$max_err
        mean_rel_metrics["$shape_name"]=$mre
        max_ulp_metrics["$shape_name"]=$max_ulp
        mean_ulp_metrics["$shape_name"]=$mean_ulp
        median_ulp_metrics["$shape_name"]=$median_ulp
        p99_ulp_metrics["$shape_name"]=$p99_ulp

        # Mark as passed
        shape_status["$shape_name"]="pass"
        config_passed=true

        # Print progress (highlight yolov4 in green)
        config_name="${shape_name}_${CURRENT_SFPU_MODE}"
        [[ "$shape_name" == "yolov4" ]] && printf "${GREEN}"
        printf "%-25s %8s %12.2e %12.2e %12.2g %12.2g %9.2fµs %9.2fms\n" "$config_name" "-" "$mae" "$max_err" "$max_ulp" "$mean_ulp" "${profiler_timing_us["$shape_name"]}" "${host_timing_ms["$shape_name"]}"
        [[ "$shape_name" == "yolov4" ]] && printf "${NC}"

    done

    # Compute per-activation CSV path (used when running all activations)
    if [[ -z "$FILTER_ACTIVATION" ]]; then
        per_activation_file="$WORK_DIR/data/${ARCH_NAME/_b0/}/native_sfpu/${activation}.csv"
    fi

    # Write one CSV row per shape (row-per-shape format)
    for shape in single_tile 8_tiles 256_tiles height_sharded yolov4; do
        local_profiler=${profiler_timing_us[$shape]:-0}
        local_host=${host_timing_ms[$shape]:-0}
        local_mae=${mae_metrics[$shape]:-0}
        local_rmse=${rmse_metrics[$shape]:-0}
        local_max=${max_error_metrics[$shape]:-0}
        local_mean_rel=${mean_rel_metrics[$shape]:-0}
        local_max_ulp=${max_ulp_metrics[$shape]:-0}
        local_mean_ulp=${mean_ulp_metrics[$shape]:-0}
        local_median_ulp=${median_ulp_metrics[$shape]:-0}
        local_p99_ulp=${p99_ulp_metrics[$shape]:-0}
        local_status=${shape_status[$shape]:-skip}

        # Only write row if we have data for this shape
        if [[ "$local_profiler" != "0" ]] || [[ "$local_host" != "0" ]]; then
            csv_row="$CURRENT_PRECISION,$activation,$CURRENT_SFPU_MODE,$shape,$local_profiler,$local_host,$local_mae,$local_rmse,$local_max,$local_mean_rel,$local_max_ulp,$local_mean_ulp,$local_median_ulp,$local_p99_ulp,$local_status"

            echo "$csv_row" >> "$RESULTS_FILE"

            # All-activations mode: also write per-activation file
            if [[ -z "$FILTER_ACTIVATION" ]]; then
                echo "$csv_row" >> "$per_activation_file"
            fi
        fi
    done

done

echo ""
echo "================================================================================"
echo "                    RESULTS SUMMARY (${CURRENT_PRECISION^^})"
echo "================================================================================"
echo ""

export CURRENT_PRECISION RESULTS_FILE

python3 << 'EOFPYTHON'
import csv
import os
import sys
from collections import defaultdict

precision = os.environ.get('CURRENT_PRECISION', 'bf16')
results_file = os.environ.get('RESULTS_FILE', 'data/native_sfpu_results.csv')

try:
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r['status'] == 'pass' and r['precision'] == precision]
except FileNotFoundError:
    print(f"ERROR: Results file not found: {results_file}")
    sys.exit(1)

if not rows:
    print("No successful results to analyze.")
    sys.exit(0)

print("=" * 100)
print(f"Native SFPU Performance ({precision.upper()})")
print("=" * 100)

# Group by (activation, sfpu_mode) — each shape is a separate row
grouped = defaultdict(dict)
for r in rows:
    key = (r['activation'], r['sfpu_mode'])
    grouped[key][r['shape']] = r

shape_order = ['single_tile', '8_tiles', '256_tiles', 'height_sharded', 'yolov4']

print(f"\n{'Activation':<15} {'Mode':<8} {'Shape':<16} {'Prof(µs)':<10} {'Host(ms)':<10} {'MAE':<12} {'MaxULP':<12} {'MeanULP':<12}")
print("-" * 100)

for (activation, sfpu_mode), shapes in sorted(grouped.items()):
    for shape in shape_order:
        if shape not in shapes:
            continue
        r = shapes[shape]
        print(f"{activation:<15} {sfpu_mode:<8} {shape:<16} {float(r['time_profiler_us']):>8.2f}  {float(r['time_host_ms']):>8.2f}  {float(r['mae']):>10.2e}  {float(r['max_ulp_error']):>10.2g}  {float(r['mean_ulp_error']):>10.2g}")

print("\n" + "=" * 100)

EOFPYTHON

echo ""

done  # End of SFPU mode loop
done  # End of precision loop

echo ""
echo "================================================================================"
echo "                    ALL PRECISIONS COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""

# =============================================================================
# Symlink SFPU Hardware Outputs to Embedded Directory
# =============================================================================

EMBEDDED_DIR="$SCRIPT_DIR/../generic_lut_activation_embedded"
if [[ -d "$EMBEDDED_DIR" ]]; then
    echo "================================================================================"
    echo "Symlinking SFPU Hardware Outputs to Embedded Directory"
    echo "================================================================================"
    echo ""

    EMBEDDED_OUTPUT_DIR="$EMBEDDED_DIR/data/hardware_outputs"
    mkdir -p "$EMBEDDED_OUTPUT_DIR"

    # Strip _b0 suffix from ARCH_NAME to get directory name (e.g., blackhole_b0 -> blackhole)
    ARCH_DIR="${ARCH_NAME/_b0/}"

    # For each activation that was tested
    for activation in "${ACTIVATIONS[@]}"; do
        OUTPUT_DIR=$(get_hardware_output_dir "$activation" "$WORK_DIR")
        ACTIVATION_NAME=$(basename "$(dirname "$OUTPUT_DIR")")
        EMBEDDED_ACTIVATION_DIR="$EMBEDDED_OUTPUT_DIR/$ARCH_DIR/$ACTIVATION_NAME"

            mkdir -p "$EMBEDDED_ACTIVATION_DIR"

            # Remove old SFPU CSVs and create symlinks
            (cd "$EMBEDDED_ACTIVATION_DIR" && rm -f native_sfpu_*.csv 2>/dev/null || true)

            # Create relative symlinks to parent directory
            for sfpu_file in "$OUTPUT_DIR"/native_sfpu_*.csv; do
                if [[ -f "$sfpu_file" ]]; then
                    filename=$(basename "$sfpu_file")
                    (cd "$EMBEDDED_ACTIVATION_DIR" && ln -sf "../../../../../generic_lut_activation/data/hardware_outputs/$ARCH_DIR/$ACTIVATION_NAME/$filename" "$filename")
                fi
            done

            # Count symlinks created
            num_links=$(find "$EMBEDDED_ACTIVATION_DIR" -name "native_sfpu_*.csv" -type l 2>/dev/null | wc -l)
            if [[ $num_links -gt 0 ]]; then
                echo "  ✓ $ACTIVATION_NAME: $num_links SFPU files symlinked"
            fi
        done

    echo ""
    echo "✓ Symlinking complete"
    echo ""
fi

echo ""
