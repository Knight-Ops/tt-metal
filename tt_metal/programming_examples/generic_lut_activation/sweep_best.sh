#!/bin/bash
set -e

# =============================================================================
# Unified Best Configuration Sweep Script
#
# Supports two modes:
#   - embedded: Uses ADHOC workflow (generate kernel + compile per config)
#   - standard: Uses pre-built binaries with runtime LUT loading
#
# Mode is determined by:
#   1. SWEEP_MODE environment variable (embedded|standard)
#   2. Auto-detect based on directory name (generic_lut_activation_embedded)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/argparse_common.sh"
source "$SCRIPT_DIR/sweep_helpers.sh"
source "$SCRIPT_DIR/profiler_helpers.sh"

# =============================================================================
# Auto-detect mode if not specified
# =============================================================================

if [[ -z "$SWEEP_MODE" ]]; then
    if [[ "$SCRIPT_DIR" == *"embedded"* ]]; then
        SWEEP_MODE="embedded"
    else
        SWEEP_MODE="standard"
    fi
fi

# =============================================================================
# Initialize Environment
# =============================================================================

init_common_env
cd "$REPO_ROOT"

if [[ "$SWEEP_MODE" == "embedded" ]]; then
    WORK_DIR="tt_metal/programming_examples/generic_lut_activation_embedded"
    ADHOC_KERNEL="$SCRIPT_DIR/kernels/compute/adhoc/adhoc.cpp"
    ADHOC_BINARY="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_embedded_adhoc"
    ADHOC_TARGET="programming_examples_generic_lut_activation_embedded_adhoc"
    GENERATE_SCRIPT="$SCRIPT_DIR/tools/generate_adhoc_kernel.py"
else
    WORK_DIR="tt_metal/programming_examples/generic_lut_activation"
    ADHOC_CONFIG_H="$REPO_ROOT/$WORK_DIR/adhoc_config.h"
    ADHOC_BINARY="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_adhoc"
    ADHOC_TARGET="programming_examples_generic_lut_activation_adhoc"
    ADHOC_RATIONAL_CONFIG_H="$REPO_ROOT/$WORK_DIR/adhoc_rational_config.h"
    ADHOC_RATIONAL_BINARY="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_rational_adhoc"
    ADHOC_RATIONAL_TARGET="programming_examples_generic_lut_activation_rational_adhoc"
fi

# Ensure sweep_config.dat is up-to-date, then load it
ensure_sweep_config "$WORK_DIR"
CONFIG_FILE="$WORK_DIR/sweep_config.dat"
if [[ -f "$CONFIG_FILE" ]]; then
    eval "$(grep -v '^\s*#' "$CONFIG_FILE" | grep -v '^\s*$' | sed 's/,/ /g' | sed 's/\([A-Z_]*\)=/\1=(/' | sed 's/$/)/')"
fi

# =============================================================================
# Script-Specific Variables
# =============================================================================

ERROR_METRIC="ulp"
TEST_SHAPES=("${STANDARD_TEST_SHAPES[@]}")
apply_shape_filter
NUM_RUNS=${NUM_RUNS:-3}

# =============================================================================
# Help
# =============================================================================

show_help() {
    echo "Usage: $0 --activation <name> --precision <type> --metric <type> [OPTIONS]"
    echo ""
    if [[ "$SWEEP_MODE" == "embedded" ]]; then
        echo "Runs best configurations using ADHOC WORKFLOW (embedded coefficients)"
    else
        echo "Runs best configurations using ADHOC WORKFLOW (runtime LUT loading)"
    fi
    echo ""
    echo "REQUIRED Arguments:"
    echo "  --activation <name>[,<name>,...]  Activation function(s) to test (comma-separated)"
    echo "  --precision <type>        Precision: fp32 or bf16"
    echo "  --metric <type>           Error metric: max, mae, or ulp (default: ulp)"
    echo ""
    show_common_options
    echo ""
    echo "Mode: $SWEEP_MODE"
    echo ""
    echo "Examples:"
    echo "  $0 --activation gelu --precision fp32 --metric max"
    echo "  $0 --activation gelu,sigmoid,silu --precision bf16 --metric ulp"
    echo "  $0 --activation sigmoid --precision bf16 --metric ulp --skip-build"
    echo ""
    exit 0
}

# =============================================================================
# Parse Arguments
# =============================================================================

parse_common_args "$@"
set -- "${PARSED_ARGS[@]}"

if [[ "$HELP_REQUESTED" == "true" ]]; then
    show_help
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --metric|-m)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Error: --metric requires an argument (mae|max|ulp)"
                exit 1
            fi
            ERROR_METRIC="$2"
            if [[ "$ERROR_METRIC" != "mae" && "$ERROR_METRIC" != "max" && "$ERROR_METRIC" != "ulp" ]]; then
                echo "Error: --metric must be one of: mae, max, ulp (got: $ERROR_METRIC)"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$FILTER_ACTIVATION" ]]; then
    echo "Error: --activation is required"
    exit 1
fi
if [[ -z "$FILTER_PRECISION" ]]; then
    FILTER_PRECISION="both"
fi

# =============================================================================
# Check Prerequisites
# =============================================================================

BEST_CSV="$TT_POLY_FIT_DIR/best.csv"
if [[ ! -f "$BEST_CSV" ]]; then
    echo "ERROR: Best configuration file not found: $BEST_CSV"
    exit 1
fi

if [[ "$SWEEP_MODE" == "embedded" && ! -f "$GENERATE_SCRIPT" ]]; then
    echo "ERROR: Adhoc kernel generator not found: $GENERATE_SCRIPT"
    exit 1
fi

echo "================================================================================"
if [[ "$SWEEP_MODE" == "embedded" ]]; then
    echo "        Best Configuration Sweep (ADHOC - embedded coefficients)"
else
    echo "        Best Configuration Sweep (ADHOC - runtime LUT loading)"
fi
echo "================================================================================"
echo "Build directory: $BUILD_DIR"
echo "Poly fitter dir: $TT_POLY_FIT_DIR"
echo "Architecture: $ARCH_NAME"
echo "Error metric: $ERROR_METRIC"
echo "Activations: ${FILTER_ACTIVATION_LIST[*]}"
echo "Precision: $FILTER_PRECISION"
echo "Test shapes: ${#TEST_SHAPES[@]}"
echo "Runs per config: $NUM_RUNS"
if [[ "$SWEEP_MODE" == "embedded" ]]; then
    echo "Adhoc kernel: $ADHOC_KERNEL"
    echo "Adhoc binary: $ADHOC_BINARY"
fi
# Pre-read best.csv to display LUT config in header
if [[ -f "$BEST_CSV" ]]; then
    while IFS=',' read -r _act _prec \
        _max_seg _max_deg _max_mae _max_max _max_ulp _max_rmse _max_nseg _max_segm _max_fit _max_src \
        _mae_seg _mae_deg _mae_mae _mae_max _mae_ulp _mae_rmse _mae_nseg _mae_segm _mae_fit _mae_src \
        _ulp_seg _ulp_deg _ulp_mae _ulp_max _ulp_ulp _ulp_rmse _ulp_nseg _ulp_segm _ulp_fit _ulp_src; do
        _act=$(echo "$_act" | tr -d ' ')
        _prec=$(echo "$_prec" | tr -d ' ')
        activation_matches "$_act" || continue
        [[ "$FILTER_PRECISION" != "both" && "$_prec" != "$FILTER_PRECISION" ]] && continue
        if [[ "$ERROR_METRIC" == "mae" ]]; then
            _deg=$(echo "$_mae_deg" | tr -d ' '); _seg=$(echo "$_mae_seg" | tr -d ' ')
            _segm=$(echo "$_mae_segm" | tr -d ' '); _fit=$(echo "$_mae_fit" | tr -d ' ')
        elif [[ "$ERROR_METRIC" == "ulp" ]]; then
            _deg=$(echo "$_ulp_deg" | tr -d ' '); _seg=$(echo "$_ulp_seg" | tr -d ' ')
            _segm=$(echo "$_ulp_segm" | tr -d ' '); _fit=$(echo "$_ulp_fit" | tr -d ' ')
        else
            _deg=$(echo "$_max_deg" | tr -d ' '); _seg=$(echo "$_max_seg" | tr -d ' ')
            _segm=$(echo "$_max_segm" | tr -d ' '); _fit=$(echo "$_max_fit" | tr -d ' ')
        fi
        if [[ "$_deg" == *"/"* ]]; then
            echo "LUT config ($_prec): rational n${_deg%%/*}/d${_deg##*/}, ${_seg} segments, ${_segm}, fitting=${_fit}"
        else
            echo "LUT config ($_prec): polynomial p${_deg}, ${_seg} segments, ${_segm}, fitting=${_fit}"
        fi
    done < <(tail -n +2 "$BEST_CSV")
fi
echo ""

# =============================================================================
# Mode-Specific Functions
# =============================================================================

prepare_best_config() {
    local activation="$1"
    local degree="$2"
    local segments="$3"
    local segmentation="$4"
    local metric="$5"

    if [[ "$SWEEP_MODE" == "embedded" ]]; then
        # Embedded: Generate adhoc kernel with embedded coefficients and compile
        local gen_args="--activation $activation --segments $segments --segmentation $segmentation --metric $metric --output $ADHOC_KERNEL"

        if [[ "$degree" == *"/"* ]]; then
            local num_deg="${degree%%/*}"
            local den_deg="${degree##*/}"
            gen_args="$gen_args --num-degree $num_deg --den-degree $den_deg"
        else
            gen_args="$gen_args --degree $degree"
        fi

        python3 "$GENERATE_SCRIPT" $gen_args > /dev/null 2>&1 || return 1
        if [[ "$VERBOSE_BUILD" == true ]]; then
            ninja -v -C "$BUILD_DIR" "$ADHOC_TARGET" || return 1
        else
            ninja -C "$BUILD_DIR" "$ADHOC_TARGET" > /dev/null 2>&1 || return 1
        fi
    else
        # Non-embedded: Check for LUT file first
        local lut_file
        if [[ "$degree" == *"/"* ]]; then
            local num_deg="${degree%%/*}"
            local den_deg="${degree##*/}"
            lut_file=$(get_rational_csv_path "$activation" "$num_deg" "$den_deg" "$segments" "$segmentation" "rational" "$metric")
        else
            lut_file=$(get_poly_csv_path "$activation" "$degree" "$segments" "$segmentation" "any" "$metric")
        fi

        if [[ ! -f "$lut_file" ]]; then
            return 1
        fi
        CURRENT_LUT_FILE="$lut_file"

        # Update adhoc config header and build
        if [[ "$degree" == *"/"* ]]; then
            # Rational: update rational config header
            local num_deg="${degree%%/*}"
            local den_deg="${degree##*/}"
            local lut_size=$(( (segments + 1) + segments * (num_deg + 1 + den_deg + 1) ))
            cat > "$ADHOC_RATIONAL_CONFIG_H" << EOF
// Auto-generated adhoc rational configuration header
// Generated by sweep_best.sh - DO NOT EDIT MANUALLY.
#pragma once
#define ADHOC_NUM_DEGREE $num_deg
#define ADHOC_DEN_DEGREE $den_deg
#define ADHOC_NUM_SEGMENTS $segments
#define ADHOC_LUT_SIZE $lut_size
EOF
            if [[ "$VERBOSE_BUILD" == true ]]; then
                ninja -v -C "$BUILD_DIR" "$ADHOC_RATIONAL_TARGET" || return 1
            else
                ninja -C "$BUILD_DIR" "$ADHOC_RATIONAL_TARGET" > /dev/null 2>&1 || return 1
            fi
            IS_RATIONAL=true
        else
            # Polynomial: update polynomial config header
            local lut_size=$(( (segments + 1) + segments * (degree + 1) ))
            cat > "$ADHOC_CONFIG_H" << EOF
// Auto-generated adhoc configuration header
// Generated by sweep_best.sh - DO NOT EDIT MANUALLY.
#pragma once
#define ADHOC_POLY_DEGREE $degree
#define ADHOC_NUM_SEGMENTS $segments
#define ADHOC_LUT_SIZE $lut_size
EOF
            if [[ "$VERBOSE_BUILD" == true ]]; then
                ninja -v -C "$BUILD_DIR" "$ADHOC_TARGET" || return 1
            else
                ninja -C "$BUILD_DIR" "$ADHOC_TARGET" > /dev/null 2>&1 || return 1
            fi
            IS_RATIONAL=false
        fi
    fi
    return 0
}

get_best_binary_path() {
    local degree="$1"
    local segments="$2"

    if [[ "$SWEEP_MODE" == "embedded" ]]; then
        echo "$ADHOC_BINARY"
    else
        # Non-embedded: use appropriate adhoc binary based on type
        if [[ "$degree" == *"/"* ]]; then
            echo "$ADHOC_RATIONAL_BINARY"
        else
            echo "$ADHOC_BINARY"
        fi
    fi
}

get_best_binary_args() {
    local activation="$1"
    local precision="$2"
    local range_min="$3"
    local range_max="$4"
    local tile_count="$5"

    if [[ "$SWEEP_MODE" == "embedded" ]]; then
        echo "--activation $activation --precision $precision --range-min $range_min --range-max $range_max --tiles $tile_count"
    else
        echo "$CURRENT_LUT_FILE --activation $activation --precision $precision --range-min $range_min --range-max $range_max --tiles $tile_count"
    fi
}

# =============================================================================
# Dry-Run Mode (standard mode only)
# =============================================================================

if [[ "$DRY_RUN" == true && "$SWEEP_MODE" == "standard" ]]; then
    echo "================================================================================"
    echo "DRY-RUN MODE: Showing configurations that would be tested"
    echo "================================================================================"
    echo ""

    config_num=0
    found_binaries=0
    missing_binaries=0
    found_csvs=0
    missing_csvs=0

    while IFS=',' read -r activation precision \
        best_max_segments best_max_degree best_max_mae best_max_max best_max_ulp best_max_rmse \
        best_max_num_segments best_max_segmentation best_max_fitting best_max_source \
        best_mae_segments best_mae_degree best_mae_mae best_mae_max best_mae_ulp best_mae_rmse \
        best_mae_num_segments best_mae_segmentation best_mae_fitting best_mae_source \
        best_ulp_segments best_ulp_degree best_ulp_mae best_ulp_max best_ulp_ulp best_ulp_rmse \
        best_ulp_num_segments best_ulp_segmentation best_ulp_fitting best_ulp_source; do

        activation=$(echo "$activation" | tr -d ' ')
        precision=$(echo "$precision" | tr -d ' ')

        if ! activation_matches "$activation"; then
            continue
        fi
        if [[ "$FILTER_PRECISION" != "both" && "$precision" != "$FILTER_PRECISION" ]]; then
            continue
        fi

        # Choose config based on metric
        if [[ "$ERROR_METRIC" == "mae" ]]; then
            segments=$(echo "$best_mae_segments" | tr -d ' ')
            degree=$(echo "$best_mae_degree" | tr -d ' ')
            segmentation=$(echo "$best_mae_segmentation" | tr -d ' ')
        elif [[ "$ERROR_METRIC" == "ulp" ]]; then
            segments=$(echo "$best_ulp_segments" | tr -d ' ')
            degree=$(echo "$best_ulp_degree" | tr -d ' ')
            segmentation=$(echo "$best_ulp_segmentation" | tr -d ' ')
        else
            segments=$(echo "$best_max_segments" | tr -d ' ')
            degree=$(echo "$best_max_degree" | tr -d ' ')
            segmentation=$(echo "$best_max_segmentation" | tr -d ' ')
        fi

        config_num=$((config_num + 1))

        if [[ "$degree" == *"/"* ]]; then
            num_deg="${degree%%/*}"
            den_deg="${degree##*/}"
            binary="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_n${num_deg}_d${den_deg}_s${segments}"
            lut_file=$(get_rational_csv_path "$activation" "$num_deg" "$den_deg" "$segments" "$segmentation" "rational" "$ERROR_METRIC")
            coeffs=$(( (num_deg + den_deg) * segments ))  # rough degree_sum for rational
            config_name="${activation}_n${num_deg}d${den_deg}_s${segments}"
        else
            binary="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_p${degree}_s${segments}"
            lut_file=$(get_poly_csv_path "$activation" "$degree" "$segments" "$segmentation" "any" "$ERROR_METRIC")
            deg_result=$(get_poly_actual_degree "$activation" "$degree" "$segments" "$segmentation")
            coeffs="${deg_result##*,}"  # degree_sum
            [[ "$coeffs" == "0" ]] && coeffs=$((degree * segments))
            config_name="${activation}_p${degree}_s${segments}"
        fi

        binary_exists="NO"
        csv_exists="NO"

        if [[ -f "$binary" ]]; then
            binary_exists="YES"
            found_binaries=$((found_binaries + 1))
        else
            missing_binaries=$((missing_binaries + 1))
        fi

        if [[ -f "$lut_file" ]]; then
            csv_exists="YES"
            found_csvs=$((found_csvs + 1))
        else
            missing_csvs=$((missing_csvs + 1))
        fi

        printf "%3d. %-25s  Binary:%s  CSV:%s  Coeffs:%d\n" \
            "$config_num" "$config_name" "$binary_exists" "$csv_exists" "$coeffs"
    done < <(tail -n +2 "$BEST_CSV")

    echo ""
    echo "Summary: Found $found_binaries/$config_num binaries, $found_csvs/$config_num CSVs"
    exit 0
fi

# =============================================================================
# Execution
# =============================================================================

cd "$REPO_ROOT"

RESULTS_FILE=$(get_results_file "best" "$WORK_DIR")
mkdir -p "$(dirname "$RESULTS_FILE")"

# Append mode: write header if file doesn't exist, then strip any existing rows
# for this (precision, metric) so re-runs don't duplicate and all 6 combos
# (fp32/bf16 × max/mae/ulp) accumulate in one file per activation.
HEADER="activation,precision,metric,depth,degree,segmentation,shape,time_profiler_us,time_host_ms,mae,max_error,max_ulp,status"
if [[ ! -f "$RESULTS_FILE" ]]; then
    echo "$HEADER" > "$RESULTS_FILE"
else
    # Remove stale rows matching this (precision, metric) combination
    tmp="${RESULTS_FILE}.tmp"
    head -1 "$RESULTS_FILE" > "$tmp"
    if [[ "$FILTER_PRECISION" == "both" ]]; then
        # Running both precisions: strip all rows for this metric
        tail -n +2 "$RESULTS_FILE" | grep -v "^[^,]*,[^,]*,${ERROR_METRIC}," >> "$tmp" 2>/dev/null || true
    else
        # Strip rows matching exact (precision, metric) pair
        tail -n +2 "$RESULTS_FILE" | grep -v "^[^,]*,${FILTER_PRECISION},${ERROR_METRIC}," >> "$tmp" 2>/dev/null || true
    fi
    mv "$tmp" "$RESULTS_FILE"
fi

echo "================================================================================"
echo "RUNNING BEST CONFIGURATIONS"
echo "================================================================================"
echo ""

reset_device 0

processed=0
failed=0
skipped=0
current_activation=""

# Read best.csv
while IFS=',' read -r activation precision \
    best_max_segments best_max_degree best_max_mae best_max_max best_max_ulp best_max_rmse \
    best_max_num_segments best_max_segmentation best_max_fitting best_max_source \
    best_mae_segments best_mae_degree best_mae_mae best_mae_max best_mae_ulp best_mae_rmse \
    best_mae_num_segments best_mae_segmentation best_mae_fitting best_mae_source \
    best_ulp_segments best_ulp_degree best_ulp_mae best_ulp_max best_ulp_ulp best_ulp_rmse \
    best_ulp_num_segments best_ulp_segmentation best_ulp_fitting best_ulp_source; do

    activation=$(echo "$activation" | tr -d ' ')
    precision=$(echo "$precision" | tr -d ' ')

    # Filter
    if ! activation_matches "$activation"; then
        continue
    fi
    if [[ "$FILTER_PRECISION" != "both" && "$precision" != "$FILTER_PRECISION" ]]; then
        continue
    fi

    # Choose config based on metric
    if [[ "$ERROR_METRIC" == "mae" ]]; then
        segments=$(echo "$best_mae_segments" | tr -d ' ')
        degree=$(echo "$best_mae_degree" | tr -d ' ')
        segmentation=$(echo "$best_mae_segmentation" | tr -d ' ')
        fitting=$(echo "$best_mae_fitting" | tr -d ' ')
    elif [[ "$ERROR_METRIC" == "ulp" ]]; then
        segments=$(echo "$best_ulp_segments" | tr -d ' ')
        degree=$(echo "$best_ulp_degree" | tr -d ' ')
        segmentation=$(echo "$best_ulp_segmentation" | tr -d ' ')
        fitting=$(echo "$best_ulp_fitting" | tr -d ' ')
    else
        segments=$(echo "$best_max_segments" | tr -d ' ')
        degree=$(echo "$best_max_degree" | tr -d ' ')
        segmentation=$(echo "$best_max_segmentation" | tr -d ' ')
        fitting=$(echo "$best_max_fitting" | tr -d ' ')
    fi

    # Config name for display
    if [[ "$degree" == *"/"* ]]; then
        num_deg="${degree%%/*}"
        den_deg="${degree##*/}"
        config_name="${precision}_n${num_deg}d${den_deg}_s${segments}_${segmentation:0:3}"
        coeffs=$(( (num_deg + den_deg) * segments ))  # rough degree_sum for rational
    else
        config_name="${precision}_p${degree}_s${segments}_${segmentation:0:3}"
        deg_result=$(get_poly_actual_degree "$activation" "$degree" "$segments" "$segmentation")
        coeffs="${deg_result##*,}"  # degree_sum
        [[ "$coeffs" == "0" ]] && coeffs=$((degree * segments))
    fi

    # Get test range (always needed for args)
    # Default for bf16: test ALL finite bf16 values. --use-fitting-range restricts to JSON domain.
    # fp32 always uses fitting domain (exhaustive fp32 not feasible).
    if [[ "$activation:$precision" != "$current_activation" ]]; then
        if [[ "$precision" == "bf16" && "$USE_FITTING_RANGE" != "true" ]]; then
            range_min="-3.3895e+38"
            range_max="3.3895e+38"
        else
            range_output=$(get_test_range "$activation")
            range_min=$(echo "$range_output" | awk '{print $1}')
            range_max=$(echo "$range_output" | awk '{print $2}')
        fi
    fi

    # Prepare config (mode-specific) — must run before header so we can extract LUT range
    if ! prepare_best_config "$activation" "$degree" "$segments" "$segmentation" "$ERROR_METRIC"; then
        printf "%-25s %8d %12s %12s %12s %12s %10s\n" "$config_name" "$coeffs" "-" "-" "-" "-" "SKIP"
        skipped=$((skipped + 1))
        continue
    fi

    # Print header when activation changes (after prepare so we can extract LUT range)
    if [[ "$activation:$precision" != "$current_activation" ]]; then
        current_activation="$activation:$precision"
        # Extract LUT fitting range from generated adhoc kernel
        lut_min="" ; lut_max=""
        if [[ "$SWEEP_MODE" == "embedded" && -f "$ADHOC_KERNEL" ]]; then
            lut_min=$(grep -oP 'INPUT_MIN = \K[-0-9.e+]+' "$ADHOC_KERNEL" | head -1)
            lut_max=$(grep -oP 'INPUT_MAX = \K[-0-9.e+]+' "$ADHOC_KERNEL" | head -1)
        fi
        echo ""
        # Compare numerically (string compare fails: "-10.0" vs "-1.0000000000e+01")
        ranges_differ=false
        if [[ -n "$lut_min" && -n "$lut_max" ]]; then
            ranges_differ=$(awk "BEGIN { print ($lut_min != $range_min || $lut_max != $range_max) ? \"true\" : \"false\" }")
        fi
        if [[ "$ranges_differ" == "true" ]]; then
            echo "=== $activation ($precision, test: $range_min to $range_max, LUT: $lut_min to $lut_max) ==="
        else
            echo "=== $activation ($precision, range: $range_min to $range_max) ==="
        fi
        if [[ "$degree" == *"/"* ]]; then
            echo "  LUT: rational n${num_deg}/d${den_deg}, ${segments} segments, ${segmentation}, fitting=${fitting:-unknown}, metric=${ERROR_METRIC}"
        else
            echo "  LUT: polynomial p${degree}, ${segments} segments, ${segmentation}, fitting=${fitting:-unknown}, metric=${ERROR_METRIC}"
        fi
        printf "%-25s %8s %12s %12s %12s %12s %10s %10s\n" "Config" "DegSum" "MAE" "MaxErr" "MaxULP" "MeanULP" "Prof(us)" "Host(ms)"
        printf "%s\n" "--------------------------------------------------------------------------------------------------------"
    fi

    binary=$(get_best_binary_path "$degree" "$segments")

    # Build batch tile list and base args (without --tiles)
    batch_tiles=$(build_batch_tiles)
    tile_arg=$(build_tile_arg "$batch_tiles")
    # Use per-row precision from best.csv (not FILTER_PRECISION which may be "both")
    if [[ "$SWEEP_MODE" == "embedded" ]]; then
        base_args="--activation $activation --precision $precision --range-min $range_min --range-max $range_max"
    else
        base_args="$CURRENT_LUT_FILE --activation $activation --precision $precision --range-min $range_min --range-max $range_max"
    fi

    # Precompute per-shape CSV output paths
    declare -A shape_csv_outputs=()
    for test_shape in "${TEST_SHAPES[@]}"; do
        parse_shape "$test_shape"
        if [[ "$degree" == *"/"* ]]; then
            shape_csv_outputs[$shape_name]=$(get_hw_rational_csv_path "$activation" "$precision" "$num_deg" "$den_deg" "$segments" "$segmentation" "adhoc" "$tile_count" "$WORK_DIR")
        else
            shape_csv_outputs[$shape_name]=$(get_hw_poly_csv_path "$activation" "$precision" "$degree" "$segments" "$segmentation" "adhoc" "$tile_count" "$WORK_DIR")
        fi
    done

    # Profiler + host timing pass: batch all shapes per run
    declare -A shape_profiler_times=()
    declare -A shape_host_times=()
    run_success=true
    PROFILER_DIR="$WORK_DIR/profiler_results/${activation}_${config_name}"
    mkdir -p "$PROFILER_DIR"

    for run in $(seq 1 $NUM_RUNS); do
        # Dump hardware output on first run (for accuracy computation)
        if [[ "$run" -eq 1 ]]; then
            if use_batch_mode; then
                local_output_dir=$(get_hardware_output_dir "$activation" "$WORK_DIR")
                if [[ "$degree" == *"/"* ]]; then
                    export DUMP_OUTPUT_CSV="${local_output_dir}/${activation}_${precision}_n${num_deg}d${den_deg}_s${segments}_${segmentation}_adhoc.csv"
                else
                    export DUMP_OUTPUT_CSV="${local_output_dir}/${activation}_${precision}_p${degree}_s${segments}_${segmentation}_adhoc.csv"
                fi
            else
                parse_shape "${TEST_SHAPES[0]}"
                export DUMP_OUTPUT_CSV="${shape_csv_outputs[$shape_name]}"
            fi
        else
            unset DUMP_OUTPUT_CSV
        fi

        set +e
        if [[ "$VERBOSE_BUILD" == true ]]; then
            export TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1
        fi
        output=$(TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$PROFILER_DIR" \
            $binary $base_args $tile_arg 2>&1)
        exit_code=$?
        set -e
        unset DUMP_OUTPUT_CSV
        clear_shm  # prevent /tmp/ompi.* accumulation between iterations

        if [[ $exit_code -ne 0 ]]; then
            run_success=false
            if [[ "$VERBOSE_BUILD" == true ]]; then
                echo "--- JIT compile output (exit code $exit_code) ---"
                echo "$output"
                # Report JIT build log locations
                JIT_CACHE_DIR="$HOME/.cache/tt-metal-cache"
                echo "--- JIT build logs ---"
                find "$JIT_CACHE_DIR" -name "*.log" -newer "$binary" -type f 2>/dev/null | head -10
                echo "---"
            fi
            break
        fi

        extract_and_accum_batch_profiler shape_profiler_times "$(get_profiler_csv "$PROFILER_DIR")" "$WORK_DIR"

        for test_shape in "${TEST_SHAPES[@]}"; do
            parse_shape "$test_shape"
            htime=$(extract_shape_timing "$output" "$tile_count")
            [[ -n "$htime" ]] && accum_timing shape_host_times "$shape_name" "$htime"
        done
    done

    # Output results per shape
    for test_shape in "${TEST_SHAPES[@]}"; do
        parse_shape "$test_shape"
        csv_output="${shape_csv_outputs[$shape_name]}"

        if [[ "$run_success" == true ]] && \
           [[ -n "${shape_profiler_times[$shape_name]:-}" ]] && \
           [[ -n "${shape_host_times[$shape_name]:-}" ]]; then
            profiler_min=$(compute_kernel_exec_min "${shape_profiler_times[$shape_name]}")
            host_min=$(compute_kernel_exec_min "${shape_host_times[$shape_name]}")

            # Accuracy metrics
            mae="0" max_err="0" max_ulp="0" mean_ulp="0"
            if [[ -f "$TT_POLY_FIT_DIR/extract_accuracy.py" && -f "$csv_output" ]]; then
                accuracy=$(/usr/bin/python3 "$TT_POLY_FIT_DIR/extract_accuracy.py" "$activation" "$csv_output" 2>/dev/null) || true
                if [[ -n "$accuracy" ]]; then
                    mae=$(echo "$accuracy" | cut -d',' -f1)
                    max_err=$(echo "$accuracy" | cut -d',' -f3)
                    max_ulp=$(echo "$accuracy" | cut -d',' -f5)
                    mean_ulp=$(echo "$accuracy" | cut -d',' -f6)
                fi
            fi

            # Compress hardware output CSV to save disk space (original deleted by gzip)
            if [[ -f "$csv_output" ]]; then
                gzip -f "$csv_output"
            fi

            display_name="${shape_name}_${config_name}"
            [[ "$shape_name" == "yolov4" ]] && printf "${GREEN}"
            printf "%-25s %8d %12.2e %12.2e %12.2g %12.2g %9.2fus %9.2fms\n" \
                "$display_name" "$coeffs" "$mae" "$max_err" "$max_ulp" "$mean_ulp" "$profiler_min" "$host_min"
            [[ "$shape_name" == "yolov4" ]] && printf "${NC}"

            echo "$activation,$precision,$ERROR_METRIC,$segments,$degree,$segmentation,$shape_name,$profiler_min,$host_min,$mae,$max_err,$max_ulp,pass" >> "$RESULTS_FILE"
            processed=$((processed + 1))
        else
            printf "%-25s %8d %12s %12s %12s %12s %10s\n" "${shape_name}_${config_name}" "$coeffs" "-" "-" "-" "-" "FAILED"
            echo "$activation,$precision,$ERROR_METRIC,$segments,$degree,$segmentation,$shape_name,0,0,0,0,0,fail" >> "$RESULTS_FILE"
            failed=$((failed + 1))
        fi
    done


done < <(tail -n +2 "$BEST_CSV")

echo ""
echo "================================================================================"
echo "SWEEP COMPLETE: Processed=$processed, Skipped=$skipped, Failed=$failed"
echo "Results: $RESULTS_FILE"
echo "================================================================================"
