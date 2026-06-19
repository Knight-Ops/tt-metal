#!/bin/bash
set -e

# =============================================================================
# Unified Polynomial Sweep Script
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
    ADHOC_KERNEL="$REPO_ROOT/$WORK_DIR/kernels/compute/adhoc/adhoc.cpp"
    ADHOC_BINARY="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_embedded_adhoc"
    ADHOC_TARGET="programming_examples_generic_lut_activation_embedded_adhoc"
    GENERATE_SCRIPT="$REPO_ROOT/$WORK_DIR/tools/generate_adhoc_kernel.py"
else
    WORK_DIR="tt_metal/programming_examples/generic_lut_activation"
    ADHOC_CONFIG_H="$REPO_ROOT/$WORK_DIR/adhoc_config.h"
    ADHOC_BINARY="$BUILD_DIR/programming_examples/programming_examples_generic_lut_activation_adhoc"
    ADHOC_TARGET="programming_examples_generic_lut_activation_adhoc"
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
FILTER_SEGMENTATION=""
NUM_RUNS=${NUM_RUNS:-2}

# =============================================================================
# Help
# =============================================================================

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    if [[ "$SWEEP_MODE" == "embedded" ]]; then
        echo "Polynomial sweep using ADHOC WORKFLOW (embedded coefficients)"
    else
        echo "Polynomial sweep using ADHOC WORKFLOW (runtime LUT loading)"
    fi
    echo ""
    show_common_options
    echo ""
    echo "Script-Specific Options:"
    echo "  --degree <degrees>        Comma-separated degrees (e.g., '3,4,8') (default: all)"
    echo "  --depth <depths>          Comma-separated depths (e.g., '4,8,16') (default: all)"
    echo "  --segmentation <type>     uniform | chebyshev | curvature (default: all)"
    echo ""
    echo "Mode: $SWEEP_MODE"
    echo ""
    echo "Examples:"
    echo "  $0 --activation sigmoid --degree 4 --depth 16"
    echo "  $0 --activation gelu --precision fp32 --degree 3,4,5"
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
        --segmentation) FILTER_SEGMENTATION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Apply shape filter
TEST_SHAPES=("${STANDARD_TEST_SHAPES[@]}")
apply_shape_filter

# =============================================================================
# Configuration
# =============================================================================

# Degrees to sweep
if [[ -n "$FILTER_DEGREE" ]]; then
    IFS=',' read -ra DEGREES <<< "$FILTER_DEGREE"
else
    DEGREES=(${POLY_DEGREES[@]:-1 2 3 4 5 6 7 8})
fi

# Depths (segment counts) to sweep
if [[ -n "$FILTER_DEPTHS" ]]; then
    IFS=',' read -ra DEPTHS <<< "$FILTER_DEPTHS"
else
    DEPTHS=(${POLY_SEGMENTS[@]:-4 8 16 32})
fi

# Segmentation methods
if [[ -n "$FILTER_SEGMENTATION" ]]; then
    SEGMENTATIONS=("$FILTER_SEGMENTATION")
else
    SEGMENTATIONS=(${POLY_SEGMENTATIONS[@]:-uniform chebyshev curvature})
fi

# Activations
if [[ -n "$FILTER_ACTIVATION" ]]; then
    ACTIVATIONS_TO_TEST=("$FILTER_ACTIVATION")
else
    ACTIVATIONS_TO_TEST=(${ACTIVATIONS[@]:-sigmoid gelu tanh})
fi

# Precisions
if [[ -n "$FILTER_PRECISION" ]]; then
    PRECISIONS=("$FILTER_PRECISION")
else
    PRECISIONS=(${POLY_PRECISIONS[@]:-fp32 bf16})
fi

# Calculate total
total=$((${#ACTIVATIONS_TO_TEST[@]} * ${#PRECISIONS[@]} * ${#DEGREES[@]} * ${#DEPTHS[@]} * ${#SEGMENTATIONS[@]}))

echo "================================================================================"
if [[ "$SWEEP_MODE" == "embedded" ]]; then
    echo "        Polynomial Sweep (ADHOC - embedded coefficients)"
else
    echo "        Polynomial Sweep (ADHOC - runtime LUT loading)"
fi
echo "================================================================================"
echo "Activations: ${ACTIVATIONS_TO_TEST[*]}"
echo "Precisions: ${PRECISIONS[*]}"
echo "Degrees: ${DEGREES[*]}"
echo "Depths: ${DEPTHS[*]}"
echo "Segmentations: ${SEGMENTATIONS[*]}"
echo "Total configs: $total"
echo ""

# =============================================================================
# Mode-Specific Functions
# =============================================================================

prepare_config() {
    local activation="$1"
    local degree="$2"
    local depth="$3"
    local segmentation="$4"

    if [[ "$SWEEP_MODE" == "embedded" ]]; then
        # Embedded: Generate adhoc kernel with embedded coefficients and compile
        python3 "$GENERATE_SCRIPT" \
            --activation "$activation" \
            --degree "$degree" \
            --segments "$depth" \
            --segmentation "$segmentation" \
            --output "$ADHOC_KERNEL" > /dev/null 2>&1 || return 1

        ninja -C "$BUILD_DIR" "$ADHOC_TARGET" > /dev/null 2>&1 || return 1
    else
        # Non-embedded: Check for LUT file first
        local lut_file
        for metric in ulp mae max; do
            lut_file=$(get_poly_csv_path "$activation" "$degree" "$depth" "$segmentation" "any" "$metric")
            if [[ -f "$lut_file" ]]; then
                CURRENT_LUT_FILE="$lut_file"
                break
            fi
        done
        if [[ -z "$CURRENT_LUT_FILE" || ! -f "$CURRENT_LUT_FILE" ]]; then
            return 1
        fi

        # Update adhoc_config.h with degree/segments
        local lut_size=$(( (depth + 1) + depth * (degree + 1) ))
        cat > "$ADHOC_CONFIG_H" << EOF
// Auto-generated adhoc configuration header
// Generated by sweep_polynomial.sh - DO NOT EDIT MANUALLY.
#pragma once
#define ADHOC_POLY_DEGREE $degree
#define ADHOC_NUM_SEGMENTS $depth
#define ADHOC_LUT_SIZE $lut_size
EOF

        # Build the adhoc target
        ninja -C "$BUILD_DIR" "$ADHOC_TARGET" > /dev/null 2>&1 || return 1
    fi
    return 0
}

get_binary_path() {
    local degree="$1"
    local depth="$2"

    # Both modes now use adhoc binary (compiled on-the-fly)
    echo "$ADHOC_BINARY"
}

get_binary_args() {
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

# Build phase removed - both modes now compile on-the-fly per config

# =============================================================================
# Execution
# =============================================================================

cd "$REPO_ROOT"

RESULTS_FILE=$(get_results_file "polynomial" "$WORK_DIR")
mkdir -p "$(dirname "$RESULTS_FILE")"

echo "activation,precision,depth,degree,segmentation,shape,time_profiler_us,time_host_ms,mae,rmse,max_error,mean_rel_error,max_ulp,mean_ulp,median_ulp,p99_ulp,status" > "$RESULTS_FILE"

reset_device 0

processed=0
failed=0
skipped=0

# Determine binary precision arg: use "both" when multiple precisions requested
# so the binary runs bf16+fp32 in a single device session (one device init)
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

    for depth in "${DEPTHS[@]}"; do
        for degree in "${DEGREES[@]}"; do
            for segmentation in "${SEGMENTATIONS[@]}"; do
                config_name="p${degree}_s${depth}_${segmentation:0:3}"
                deg_result=$(get_poly_actual_degree "$activation" "$degree" "$depth" "$segmentation")
                coeffs="${deg_result##*,}"  # degree_sum
                [[ "$coeffs" == "0" ]] && coeffs=$((degree * depth))  # fallback

                # Prepare config (mode-specific)
                if ! prepare_config "$activation" "$degree" "$depth" "$segmentation"; then
                    # Show skip message with shape prefix if shape filter is active
                    for precision in "${PRECISIONS[@]}"; do
                        for test_shape in "${TEST_SHAPES[@]}"; do
                            skip_shape_name="${test_shape%%:*}"
                            printf "%-25s %8d %12s %12s %12s %10s\n" "${skip_shape_name}_${config_name}" "$coeffs" "-" "-" "-" "SKIP"
                        done
                    done
                    skipped=$((skipped + 1))
                    continue
                fi

                binary=$(get_binary_path "$degree" "$depth")

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
                            shape_csv_outputs["${precision}_${shape_name}"]="${local_output_dir}/${activation}_p${degree}_s${depth}_${segmentation}_adhoc_${precision}_tiles${tile_count}.csv"
                        else
                            shape_csv_outputs["${precision}_${shape_name}"]=$(get_hw_poly_csv_path "$activation" "$precision" "$degree" "$depth" "$segmentation" "adhoc" "$tile_count" "$WORK_DIR")
                        fi
                    done
                done

                # Profiler + host timing: keyed by precision_shape
                declare -A shape_profiler_times=()
                declare -A shape_host_times=()
                run_success=true
                PROFILER_DIR="$WORK_DIR/profiler_results/${activation}_${config_name}"
                mkdir -p "$PROFILER_DIR"

                for run in $(seq 1 $NUM_RUNS); do
                    # Dump hardware output on first run (for accuracy computation)
                    if [[ "$run" -eq 1 ]]; then
                        local_output_dir=$(get_hardware_output_dir "$activation" "$WORK_DIR")
                        if [[ "$binary_precision" == "both" ]]; then
                            # Binary inserts _bf16/_fp32 and _tilesN
                            export DUMP_OUTPUT_CSV="${local_output_dir}/${activation}_p${degree}_s${depth}_${segmentation}_adhoc.csv"
                        elif use_batch_mode; then
                            export DUMP_OUTPUT_CSV="${local_output_dir}/${activation}_${binary_precision}_p${degree}_s${depth}_${segmentation}_adhoc.csv"
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

                    # Host timing: extract per precision per shape
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

                            # Accuracy metrics (all 8 fields from extract_accuracy.py)
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

                            # Compress hardware output CSV to save disk space
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

                            echo "$activation,$precision,$depth,$degree,$segmentation,$shape_name,$profiler_min,$host_min,$mae,$rmse,$max_err,$mean_rel,$max_ulp,$mean_ulp,$median_ulp,$p99_ulp,pass" >> "$RESULTS_FILE"
                            processed=$((processed + 1))
                        else
                            if [[ ${#PRECISIONS[@]} -gt 1 ]]; then
                                display_name="${precision}_${shape_name}_${config_name}"
                            else
                                display_name="${shape_name}_${config_name}"
                            fi
                            printf "%-25s %8d %12s %12s %12s %10s\n" "$display_name" "$coeffs" "-" "-" "-" "FAIL"
                            echo "$activation,$precision,$depth,$degree,$segmentation,$shape_name,0,0,0,0,0,0,0,0,0,0,fail" >> "$RESULTS_FILE"
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
