#!/bin/bash
set -e

# =============================================================================
# Unified Rational Approximation Sweep Script
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
    ADHOC_RATIONAL_CONFIG_H="$REPO_ROOT/$WORK_DIR/adhoc_rational_config.h"
    ADHOC_BINARY="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_rational_adhoc"
    ADHOC_TARGET="programming_examples_generic_lut_activation_rational_adhoc"
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

FILTER_DEGREE=""
FILTER_DEPTHS=""
RATIONAL_SEGMENTATION="uniform"
NUM_RUNS=${NUM_RUNS:-2}

# =============================================================================
# Help
# =============================================================================

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    if [[ "$SWEEP_MODE" == "embedded" ]]; then
        echo "Explores rational approximations using ADHOC WORKFLOW (embedded coefficients)"
    else
        echo "Explores rational approximations using ADHOC WORKFLOW (runtime LUT loading)"
    fi
    echo ""
    show_common_options
    echo ""
    echo "Script-Specific Options:"
    echo "  --degree <type>           Rational degree: rational_1d1, rational_2d2, ... (default: all)"
    echo "  --depth <depths>          Comma-separated segment counts (default: from config)"
    echo "  --segmentation <type>     Segmentation type: uniform, chebyshev (default: uniform)"
    echo ""
    echo "Mode: $SWEEP_MODE"
    echo ""
    echo "Examples:"
    echo "  $0 --activation exp --degree rational_2d2"
    echo "  $0 --activation sigmoid --depth 2,4"
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
        --degree) FILTER_DEGREE="$2"; shift 2 ;;
        --depth) FILTER_DEPTHS="$2"; shift 2 ;;
        --segmentation) RATIONAL_SEGMENTATION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Apply shape filter
TEST_SHAPES=("${STANDARD_TEST_SHAPES[@]}")
apply_shape_filter

# Default precision for rational (needs higher precision)
if [[ -z "$FILTER_PRECISION" ]]; then
    FILTER_PRECISION="fp32"
fi

# =============================================================================
# Configuration
# =============================================================================

# Generate rational degree patterns
declare -a RATIONAL_CONFIGS=()

if [[ -n "$FILTER_DEGREE" ]]; then
    # Parse specific degree like "rational_2d2"
    if [[ "$FILTER_DEGREE" =~ ^rational_([0-9]+)d([0-9]+)$ ]]; then
        RATIONAL_CONFIGS+=("${BASH_REMATCH[1]}/${BASH_REMATCH[2]}")
    else
        echo "Error: Invalid degree format. Use rational_NdM (e.g., rational_2d2)"
        exit 1
    fi
else
    # Generate all patterns from config
    for deg in ${RATIONAL_DEGREES[@]:-1 2 3 4 5}; do
        RATIONAL_CONFIGS+=("$deg/$deg")          # Symmetric: n/n
        RATIONAL_CONFIGS+=("$((deg+1))/$deg")    # Asymmetric: (n+1)/n
    done
fi

# Depths (segment counts)
if [[ -n "$FILTER_DEPTHS" ]]; then
    IFS=',' read -ra DEPTHS <<< "$FILTER_DEPTHS"
else
    DEPTHS=(${RATIONAL_SEGMENTS[@]:-1 2 4})
fi

# Segmentations
SEGMENTATIONS=("$RATIONAL_SEGMENTATION")

# Activations
if [[ -n "$FILTER_ACTIVATION" ]]; then
    ACTIVATIONS_TO_TEST=("$FILTER_ACTIVATION")
else
    ACTIVATIONS_TO_TEST=(${ACTIVATIONS[@]:-exp sigmoid tanh})
fi

PRECISIONS=("$FILTER_PRECISION")

# Calculate total
total=$((${#ACTIVATIONS_TO_TEST[@]} * ${#RATIONAL_CONFIGS[@]} * ${#DEPTHS[@]}))

echo "================================================================================"
if [[ "$SWEEP_MODE" == "embedded" ]]; then
    echo "        Rational Sweep (ADHOC - embedded coefficients)"
else
    echo "        Rational Sweep (ADHOC - runtime LUT loading)"
fi
echo "================================================================================"
echo "Activations: ${ACTIVATIONS_TO_TEST[*]}"
echo "Precision: $FILTER_PRECISION"
echo "Rational configs: ${RATIONAL_CONFIGS[*]}"
echo "Depths: ${DEPTHS[*]}"
echo "Total configs: $total"
if [[ "$SWEEP_MODE" == "embedded" ]]; then
    echo "Adhoc kernel: $ADHOC_KERNEL"
    echo "Adhoc binary: $ADHOC_BINARY"
fi
echo ""

# =============================================================================
# Mode-Specific Functions
# =============================================================================

prepare_rational_config() {
    local activation="$1"
    local num_degree="$2"
    local den_degree="$3"
    local segments="$4"
    local segmentation="$5"

    if [[ "$SWEEP_MODE" == "embedded" ]]; then
        # Embedded: Generate adhoc kernel with embedded coefficients and compile
        python3 "$GENERATE_SCRIPT" \
            --activation "$activation" \
            --num-degree "$num_degree" \
            --den-degree "$den_degree" \
            --segments "$segments" \
            --segmentation "$segmentation" \
            --output "$ADHOC_KERNEL" > /dev/null 2>&1 || return 1

        ninja -C "$BUILD_DIR" "$ADHOC_TARGET" > /dev/null 2>&1 || return 1
    else
        # Non-embedded: Check for LUT file first
        local lut_file
        lut_file=$(get_rational_csv_path "$activation" "$num_degree" "$den_degree" "$segments" "$segmentation" "rational" "ulp")
        if [[ ! -f "$lut_file" ]]; then
            return 1
        fi
        CURRENT_LUT_FILE="$lut_file"

        # Update adhoc rational config header and build
        local lut_size=$(( (segments + 1) + segments * (num_degree + 1 + den_degree + 1) ))
        cat > "$ADHOC_RATIONAL_CONFIG_H" << EOF
// Auto-generated adhoc rational configuration header
// Generated by sweep_rational.sh - DO NOT EDIT MANUALLY.
#pragma once
#define ADHOC_NUM_DEGREE $num_degree
#define ADHOC_DEN_DEGREE $den_degree
#define ADHOC_NUM_SEGMENTS $segments
#define ADHOC_LUT_SIZE $lut_size
EOF

        ninja -C "$BUILD_DIR" "$ADHOC_TARGET" > /dev/null 2>&1 || return 1
    fi
    return 0
}

get_rational_binary_path() {
    local num_degree="$1"
    local den_degree="$2"
    local segments="$3"

    # Both modes now use adhoc binary (compiled on-the-fly)
    echo "$ADHOC_BINARY"
}

get_rational_binary_args() {
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

    for activation in "${ACTIVATIONS_TO_TEST[@]}"; do
        for rational_config in "${RATIONAL_CONFIGS[@]}"; do
            num_degree="${rational_config%%/*}"
            den_degree="${rational_config##*/}"

            for depth in "${DEPTHS[@]}"; do
                config_num=$((config_num + 1))

                binary="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_n${num_degree}_d${den_degree}_s${depth}"
                lut_file=$(get_rational_csv_path "$activation" "$num_degree" "$den_degree" "$depth" "$RATIONAL_SEGMENTATION" "rational" "ulp")
                coeffs=$(( (num_degree + den_degree) * depth ))  # degree_sum for rational
                config_name="${activation}_n${num_degree}d${den_degree}_s${depth}"

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

                printf "%3d. %-30s  Binary:%s  CSV:%s  Coeffs:%d\n" \
                    "$config_num" "$config_name" "$binary_exists" "$csv_exists" "$coeffs"
            done
        done
    done

    echo ""
    echo "Summary: Found $found_binaries/$config_num binaries, $found_csvs/$config_num CSVs"
    exit 0
fi

# =============================================================================
# Execution
# =============================================================================

cd "$REPO_ROOT"

RESULTS_FILE=$(get_results_file "rational" "$WORK_DIR")
mkdir -p "$(dirname "$RESULTS_FILE")"

echo "activation,precision,depth,num_degree,den_degree,segmentation,shape,time_profiler_us,time_host_ms,mae,rmse,max_error,mean_rel_error,max_ulp,mean_ulp,median_ulp,p99_ulp,status" > "$RESULTS_FILE"

reset_device 0

processed=0
failed=0
skipped=0

# Determine binary precision arg: use "both" when multiple precisions requested
if [[ ${#PRECISIONS[@]} -gt 1 ]]; then
    binary_precision="both"
else
    binary_precision="${PRECISIONS[0]}"
fi

for activation in "${ACTIVATIONS_TO_TEST[@]}"; do
    range_output=$(get_test_range "$activation")
    range_min=$(echo "$range_output" | awk '{print $1}')
    range_max=$(echo "$range_output" | awk '{print $2}')

    echo ""
    echo "=== $activation (${binary_precision}, range: $range_min to $range_max) ==="
    printf "%-25s %8s %12s %12s %12s %10s %10s\n" "Config" "DegSum" "MAE" "MaxErr" "MaxULP" "Prof(us)" "Host(ms)"
    printf "%s\n" "--------------------------------------------------------------------------------------------"

    for rational_config in "${RATIONAL_CONFIGS[@]}"; do
        num_degree="${rational_config%%/*}"
        den_degree="${rational_config##*/}"

        for depth in "${DEPTHS[@]}"; do
            for segmentation in "${SEGMENTATIONS[@]}"; do
                config_name="n${num_degree}d${den_degree}_s${depth}_${segmentation:0:3}"
                coeffs=$(( (num_degree + den_degree) * depth ))  # degree_sum for rational

                # Prepare config (mode-specific)
                if ! prepare_rational_config "$activation" "$num_degree" "$den_degree" "$depth" "$segmentation"; then
                    for precision in "${PRECISIONS[@]}"; do
                        for test_shape in "${TEST_SHAPES[@]}"; do
                            skip_shape_name="${test_shape%%:*}"
                            printf "%-25s %8d %12s %12s %12s %10s\n" "${skip_shape_name}_${config_name}" "$coeffs" "-" "-" "-" "SKIP"
                        done
                    done
                    skipped=$((skipped + 1))
                    continue
                fi

                binary=$(get_rational_binary_path "$num_degree" "$den_degree" "$depth")

                # Build batch tile list and base args
                batch_tiles=$(build_batch_tiles)
                tile_arg=$(build_tile_arg "$batch_tiles")
                if [[ "$SWEEP_MODE" == "embedded" ]]; then
                    base_args="--activation $activation --precision $binary_precision --range-min $range_min --range-max $range_max"
                else
                    base_args="$CURRENT_LUT_FILE --activation $activation --precision $binary_precision --range-min $range_min --range-max $range_max"
                fi

                # Precompute per-shape, per-precision CSV output paths
                declare -A shape_csv_outputs=()
                local_output_dir=$(get_hardware_output_dir "$activation" "$WORK_DIR")
                for precision in "${PRECISIONS[@]}"; do
                    for test_shape in "${TEST_SHAPES[@]}"; do
                        parse_shape "$test_shape"
                        if [[ "$binary_precision" == "both" ]]; then
                            # Binary appends _{precision}_tiles{N} to base path
                            shape_csv_outputs["${precision}_${shape_name}"]="${local_output_dir}/${activation}_n${num_degree}d${den_degree}_s${depth}_${segmentation}_adhoc_${precision}_tiles${tile_count}.csv"
                        else
                            shape_csv_outputs["${precision}_${shape_name}"]=$(get_hw_rational_csv_path "$activation" "$precision" "$num_degree" "$den_degree" "$depth" "$segmentation" "adhoc" "$tile_count" "$WORK_DIR")
                        fi
                    done
                done

                # Profiler + host timing: keyed by precision_shape
                declare -A shape_profiler_times=()
                declare -A shape_host_times=()
                run_success=true
                PROFILER_DIR="$WORK_DIR/profiler_results/rational_${activation}_${config_name}"
                mkdir -p "$PROFILER_DIR"

                for run in $(seq 1 $NUM_RUNS); do
                    if [[ "$run" -eq 1 ]]; then
                        local_output_dir=$(get_hardware_output_dir "$activation" "$WORK_DIR")
                        if [[ "$binary_precision" == "both" ]]; then
                            export DUMP_OUTPUT_CSV="${local_output_dir}/${activation}_n${num_degree}d${den_degree}_s${depth}_${segmentation}_adhoc.csv"
                        elif use_batch_mode; then
                            export DUMP_OUTPUT_CSV="${local_output_dir}/${activation}_${binary_precision}_n${num_degree}d${den_degree}_s${depth}_${segmentation}_adhoc.csv"
                        else
                            parse_shape "${TEST_SHAPES[0]}"
                            export DUMP_OUTPUT_CSV="${shape_csv_outputs[${PRECISIONS[0]}_${shape_name}]}"
                        fi
                    else
                        unset DUMP_OUTPUT_CSV
                    fi

                    set +e
                    output=$(TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$PROFILER_DIR" \
                        $binary $base_args $tile_arg 2>&1)
                    exit_code=$?
                    set -e
                    unset DUMP_OUTPUT_CSV
                    clear_shm  # prevent /tmp/ompi.* accumulation between iterations

                    if [[ $exit_code -ne 0 ]]; then
                        run_success=false
                        break
                    fi

                    # Profiler: total clusters = precisions × shapes
                    total_profiler_shapes=$(( ${#PRECISIONS[@]} * ${#TEST_SHAPES[@]} ))
                    PROFILER_CSV_PATH="$(get_profiler_csv "$PROFILER_DIR")"
                    if [[ -f "$PROFILER_CSV_PATH" ]]; then
                        batch_times=$(extract_batch_profiler_times "$PROFILER_CSV_PATH" "$total_profiler_shapes" "$WORK_DIR")
                        pi=0
                        IFS=',' read -ra _ptimes <<< "$batch_times"
                        for precision in "${PRECISIONS[@]}"; do
                            for test_shape in "${TEST_SHAPES[@]}"; do
                                parse_shape "$test_shape"
                                pkey="${precision}_${shape_name}"
                                ptime="${_ptimes[$pi]:-0}"
                                if [[ -n "$ptime" && "$ptime" != "0" && "$ptime" != "0.00" ]]; then
                                    accum_timing shape_profiler_times "$pkey" "$ptime"
                                fi
                                pi=$((pi + 1))
                            done
                        done
                    fi

                    for precision in "${PRECISIONS[@]}"; do
                        for test_shape in "${TEST_SHAPES[@]}"; do
                            parse_shape "$test_shape"
                            pkey="${precision}_${shape_name}"
                            htime=$(extract_shape_timing "$output" "$tile_count" "$precision")
                            [[ -n "$htime" ]] && accum_timing shape_host_times "$pkey" "$htime"
                        done
                    done
                done

                # Output results per precision per shape
                for precision in "${PRECISIONS[@]}"; do
                    for test_shape in "${TEST_SHAPES[@]}"; do
                        parse_shape "$test_shape"
                        pkey="${precision}_${shape_name}"
                        csv_output="${shape_csv_outputs[$pkey]}"

                        if [[ "$run_success" == true ]] && \
                           [[ -n "${shape_profiler_times[$pkey]:-}" ]] && \
                           [[ -n "${shape_host_times[$pkey]:-}" ]]; then
                            profiler_min=$(compute_kernel_exec_min "${shape_profiler_times[$pkey]}")
                            host_min=$(compute_kernel_exec_min "${shape_host_times[$pkey]}")

                            mae="0" rmse="0" max_err="0" mean_rel="0" max_ulp="0" mean_ulp="0" median_ulp="0" p99_ulp="0"
                            if [[ -f "$TT_POLY_FIT_DIR/extract_accuracy.py" && -f "$csv_output" ]]; then
                                accuracy=$(python3 "$TT_POLY_FIT_DIR/extract_accuracy.py" "$activation" "$csv_output" 2>/dev/null) || true
                                if [[ -n "$accuracy" ]]; then
                                    mae=$(echo "$accuracy" | cut -d',' -f1)
                                    rmse=$(echo "$accuracy" | cut -d',' -f2)
                                    max_err=$(echo "$accuracy" | cut -d',' -f3)
                                    mean_rel=$(echo "$accuracy" | cut -d',' -f4)
                                    max_ulp=$(echo "$accuracy" | cut -d',' -f5)
                                    mean_ulp=$(echo "$accuracy" | cut -d',' -f6)
                                    median_ulp=$(echo "$accuracy" | cut -d',' -f7)
                                    p99_ulp=$(echo "$accuracy" | cut -d',' -f8)
                                fi
                            fi

                            if [[ -f "$csv_output" ]]; then
                                gzip -f "$csv_output"
                            fi

                            if [[ ${#PRECISIONS[@]} -gt 1 ]]; then
                                display_name="${precision}_${shape_name}_${config_name}"
                            else
                                display_name="${shape_name}_${config_name}"
                            fi
                            [[ "$shape_name" == "yolov4" ]] && printf "${GREEN}"
                            printf "%-25s %8d %12.2e %12.2e %12.2g %9.2fus %9.2fms\n" \
                                "$display_name" "$coeffs" "$mae" "$max_err" "$max_ulp" "$profiler_min" "$host_min"
                            [[ "$shape_name" == "yolov4" ]] && printf "${NC}"

                            echo "$activation,$precision,$depth,$num_degree,$den_degree,$segmentation,$shape_name,$profiler_min,$host_min,$mae,$rmse,$max_err,$mean_rel,$max_ulp,$mean_ulp,$median_ulp,$p99_ulp,pass" >> "$RESULTS_FILE"
                            processed=$((processed + 1))
                        else
                            if [[ ${#PRECISIONS[@]} -gt 1 ]]; then
                                display_name="${precision}_${shape_name}_${config_name}"
                            else
                                display_name="${shape_name}_${config_name}"
                            fi
                            printf "%-25s %8d %12s %12s %12s %10s\n" "$display_name" "$coeffs" "-" "-" "-" "FAILED"
                            echo "$activation,$precision,$depth,$num_degree,$den_degree,$segmentation,$shape_name,0,0,0,0,0,0,0,0,0,0,fail" >> "$RESULTS_FILE"
                            failed=$((failed + 1))
                        fi
                    done
                done
            done
        done
    done
done

echo ""
echo "================================================================================"
echo "SWEEP COMPLETE: Processed=$processed, Skipped=$skipped, Failed=$failed"
echo "Results: $RESULTS_FILE"
echo "================================================================================"
