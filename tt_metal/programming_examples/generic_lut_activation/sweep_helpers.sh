#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Common helper functions for sweep scripts
# Provides reusable code for testing multiple tile sizes and collecting results

# Standard test shapes matching op-runner benchmarks
declare -a STANDARD_TEST_SHAPES=(
    "single_tile:32:32"           # 1 tile
    "8_tiles:64:128"              # 8 tiles
    "256_tiles:512:512"           # 256 tiles (16×16) - square
    "height_sharded:25600:128"    # 3200 tiles (800×4) - tall
    "yolov4:5120:320"             # 1600 tiles (160×10) - 16 batch × 320×320
)

# Apply shape filter to TEST_SHAPES array
# Usage: apply_shape_filter  (reads FILTER_SHAPE, modifies TEST_SHAPES in place)
apply_shape_filter() {
    if [[ -z "$FILTER_SHAPE" ]]; then
        return
    fi
    # Parse comma-separated filter into array
    local -a allowed
    IFS=',' read -ra allowed <<< "$FILTER_SHAPE"
    local -a filtered=()
    for shape in "${TEST_SHAPES[@]}"; do
        local name="${shape%%:*}"
        for a in "${allowed[@]}"; do
            if [[ "$name" == "$a" ]]; then
                filtered+=("$shape")
                break
            fi
        done
    done
    if [[ ${#filtered[@]} -eq 0 ]]; then
        echo "WARNING: --shape '$FILTER_SHAPE' matched no shapes. Available: single_tile, 8_tiles, 256_tiles, height_sharded, yolov4" >&2
        exit 1
    fi
    TEST_SHAPES=("${filtered[@]}")
}

# Initialize result storage arrays
# Usage: init_shape_results
init_shape_results() {
    declare -gA profiler_timing_us   # Global associative array for profiler timing (µs)
    declare -gA host_timing_ms       # Global associative array for host timing (ms)
    declare -gA shape_status         # Global associative array for status
    declare -gA shape_mae            # Global associative array for MAE per shape
    declare -gA shape_rmse           # Global associative array for RMSE per shape
    declare -gA shape_max_error      # Global associative array for max error per shape
    declare -gA shape_mean_rel       # Global associative array for mean relative error per shape
    declare -gA shape_max_ulp        # Global associative array for max ULP error per shape
    declare -gA shape_mean_ulp       # Global associative array for mean ULP error per shape
    declare -gA shape_median_ulp     # Global associative array for median ULP error per shape
    declare -gA shape_p99_ulp        # Global associative array for P99 ULP error per shape
    config_passed=false
    config_failed=false
    config_timeout=false
    config_mae=""
    config_rmse=""
    config_max=""
    config_mean_rel=""
    config_max_ulp=""
    config_mean_ulp=""
    config_median_ulp=""
    config_p99_ulp=""
}

# Compute minimum from comma-separated timing values
# Usage: result=$(compute_kernel_exec_min "1.5,2.3,1.8")
compute_kernel_exec_min() {
    local values="$1"
    python3 -c "
import sys
values = [float(x) for x in '$values'.split(',') if x]
print(min(values) if values else 0)
"
}

# Parse shape string into components
# Usage: parse_shape "single_tile:32:32"
# Sets: shape_name, shape_rows, shape_cols, tile_count
parse_shape() {
    local test_shape="$1"
    shape_name="${test_shape%%:*}"
    shape_rest="${test_shape#*:}"
    shape_rows="${shape_rest%%:*}"
    shape_cols="${shape_rest#*:}"
    tile_count=$(( (shape_rows / 32) * (shape_cols / 32) ))
}

# =============================================================================
# Batch Mode Helpers (--batch-tiles support)
# =============================================================================

# Build comma-separated batch tile list from TEST_SHAPES
# Usage: batch_tiles=$(build_batch_tiles)
build_batch_tiles() {
    local result=""
    for test_shape in "${TEST_SHAPES[@]}"; do
        parse_shape "$test_shape"
        result="${result:+$result,}$tile_count"
    done
    echo "$result"
}

# Check if batch mode should be used (multiple shapes)
# Usage: if use_batch_mode; then ...
use_batch_mode() {
    [[ ${#TEST_SHAPES[@]} -gt 1 ]]
}

# Extract per-shape kernel execution timing from batch or non-batch output
# Handles three formats:
#   Single shape:       TIMING_KERNEL_EXECUTION: <val>
#   Batch (tiles only): BATCH[tiles=N]:TIMING_KERNEL_EXECUTION: <val>
#   Batch (precision):  BATCH[bf16,tiles=N]:TIMING_KERNEL_EXECUTION: <val>
# Usage: timing=$(extract_shape_timing "$output" "$tile_count" ["$precision"])
extract_shape_timing() {
    local output="$1"
    local tiles="$2"
    local precision="${3:-}"
    if use_batch_mode || [[ -n "$precision" ]]; then
        local result=""
        if [[ -n "$precision" ]]; then
            # Try multi-precision format first: BATCH[bf16,tiles=N]:
            result=$(echo "$output" | grep "BATCH\[${precision},tiles=${tiles}\]:TIMING_KERNEL_EXECUTION:" | awk -F': ' '{print $2}')
        fi
        if [[ -z "$result" ]]; then
            # Fall back to single-precision batch: BATCH[tiles=N]:
            result=$(echo "$output" | grep "BATCH\[tiles=${tiles}\]:TIMING_KERNEL_EXECUTION:" | awk -F': ' '{print $2}')
        fi
        echo "$result"
    else
        echo "$output" | grep "TIMING_KERNEL_EXECUTION:" | awk -F': ' '{print $2}'
    fi
}

# Get profiler CSV path (always the top-level .logs/ dir)
# In batch mode, all shapes' data accumulates in a single CSV.
# Usage: csv_path=$(get_profiler_csv "$profiler_dir")
get_profiler_csv() {
    echo "${1}/.logs/profile_log_device.csv"
}

# Extract per-shape profiler times from a batch run and populate an associative array.
# Uses time-gap clustering to separate per-shape data from the single CSV.
# Args:
#   $1 - name of associative array to populate (e.g. shape_profiler_times)
#   $2 - path to profiler CSV
#   $3 - work_dir for Python module path
# Populates: array[shape_name] with comma-appended profiler times
extract_and_accum_batch_profiler() {
    local -n _arr=$1
    local profiler_csv="$2"
    local work_dir="$3"

    if [[ ! -f "$profiler_csv" ]]; then
        return 1
    fi

    local num_shapes=${#TEST_SHAPES[@]}
    local batch_times
    batch_times=$(extract_batch_profiler_times "$profiler_csv" "$num_shapes" "$work_dir")

    # Split comma-separated times and assign to shapes in order
    local i=0
    IFS=',' read -ra times_arr <<< "$batch_times"
    for test_shape in "${TEST_SHAPES[@]}"; do
        parse_shape "$test_shape"
        local ptime="${times_arr[$i]:-0}"
        if [[ -n "$ptime" && "$ptime" != "0" && "$ptime" != "0.00" ]]; then
            _arr[$shape_name]="${_arr[$shape_name]:+${_arr[$shape_name]},}$ptime"
        fi
        i=$((i + 1))
    done
}

# Build binary invocation args with tile specification (batch or single)
# Usage: tile_arg=$(build_tile_arg "$batch_tiles")
# Returns: "--batch-tiles 1,8,256,..." in batch mode, "--tiles N" in single mode
build_tile_arg() {
    local batch_tiles="$1"
    if use_batch_mode; then
        echo "--batch-tiles $batch_tiles"
    else
        # Single shape: extract the one tile count
        parse_shape "${TEST_SHAPES[0]}"
        echo "--tiles $tile_count"
    fi
}

# Accumulate a timing value into a comma-separated associative array entry
# Usage: accum_timing shape_profiler_times "$shape_name" "$ptime"
accum_timing() {
    local -n _arr=$1
    local key="$2"
    local val="$3"
    _arr[$key]="${_arr[$key]:+${_arr[$key]},}$val"
}

# Write CSV rows in row-per-shape format (one row per shape)
# This is the new standard format used by embedded sweeps
# Usage: write_csv_rows_per_shape "$activation" "$precision" "$depth" "$degree" "$segmentation" "$RESULTS_FILE"
write_csv_rows_per_shape() {
    local activation="$1"
    local precision="$2"
    local depth="$3"
    local degree="$4"
    local segmentation="$5"
    local results_file="$6"

    # Determine overall status
    local status
    if [[ "$config_passed" == true ]]; then
        status="pass"
    elif [[ "$config_timeout" == true ]]; then
        status="timeout"
    else
        status="fail"
    fi

    # Write one row per shape
    for shape in single_tile 8_tiles 256_tiles height_sharded yolov4; do
        local profiler_us=${profiler_timing_us[$shape]:-0}
        local host_ms=${host_timing_ms[$shape]:-0}
        local mae=${shape_mae[$shape]:-0}
        local rmse=${shape_rmse[$shape]:-0}
        local max_err=${shape_max_error[$shape]:-0}
        local mean_rel=${shape_mean_rel[$shape]:-0}
        local max_ulp=${shape_max_ulp[$shape]:-0}
        local mean_ulp=${shape_mean_ulp[$shape]:-0}
        local median_ulp=${shape_median_ulp[$shape]:-0}
        local p99_ulp=${shape_p99_ulp[$shape]:-0}

        # Only write row if we have timing data for this shape
        if [[ "$profiler_us" != "0" ]] || [[ "$host_ms" != "0" ]]; then
            echo "$activation,$precision,$depth,$degree,$segmentation,$shape,$profiler_us,$host_ms,$mae,$rmse,$max_err,$mean_rel,$max_ulp,$mean_ulp,$median_ulp,$p99_ulp,$status" >> "$results_file"
        fi
    done
}

# Legacy: Write CSV row with results for all tile sizes in multi-column format
# Kept for backward compatibility
# Usage: write_csv_row_with_shapes "$activation" "$precision" ... "$RESULTS_FILE"
write_csv_row_polynomial() {
    local activation="$1"
    local precision="$2"
    local depth="$3"
    local degree="$4"
    local segmentation="$5"
    local fitting="$6"
    local refined="$7"
    local theoretical_mae="$8"
    local theoretical_max="$9"
    local results_file="${10}"

    # Look up actual degree from coefficient CSV
    local actual_deg_result
    actual_deg_result=$(get_poly_actual_degree "$activation" "$degree" "$depth" "$segmentation")
    local actual_degree_max="${actual_deg_result%%,*}"
    local actual_degree_sum="${actual_deg_result##*,}"

    # Extract per-shape timing and error metrics
    local profiler_single=${profiler_timing_us[single_tile]:-0}
    local host_single=${host_timing_ms[single_tile]:-0}
    local mae_single=${shape_mae[single_tile]:-0}
    local rmse_single=${shape_rmse[single_tile]:-0}
    local max_single=${shape_max_error[single_tile]:-0}
    local mean_rel_single=${shape_mean_rel[single_tile]:-0}
    local max_ulp_single=${shape_max_ulp[single_tile]:-0}
    local mean_ulp_single=${shape_mean_ulp[single_tile]:-0}
    local median_ulp_single=${shape_median_ulp[single_tile]:-0}
    local p99_ulp_single=${shape_p99_ulp[single_tile]:-0}

    local profiler_8tiles=${profiler_timing_us[8_tiles]:-0}
    local host_8tiles=${host_timing_ms[8_tiles]:-0}
    local mae_8tiles=${shape_mae[8_tiles]:-0}
    local rmse_8tiles=${shape_rmse[8_tiles]:-0}
    local max_8tiles=${shape_max_error[8_tiles]:-0}
    local mean_rel_8tiles=${shape_mean_rel[8_tiles]:-0}
    local max_ulp_8tiles=${shape_max_ulp[8_tiles]:-0}
    local mean_ulp_8tiles=${shape_mean_ulp[8_tiles]:-0}
    local median_ulp_8tiles=${shape_median_ulp[8_tiles]:-0}
    local p99_ulp_8tiles=${shape_p99_ulp[8_tiles]:-0}

    local profiler_256tiles=${profiler_timing_us[256_tiles]:-0}
    local host_256tiles=${host_timing_ms[256_tiles]:-0}
    local mae_256tiles=${shape_mae[256_tiles]:-0}
    local rmse_256tiles=${shape_rmse[256_tiles]:-0}
    local max_256tiles=${shape_max_error[256_tiles]:-0}
    local mean_rel_256tiles=${shape_mean_rel[256_tiles]:-0}
    local max_ulp_256tiles=${shape_max_ulp[256_tiles]:-0}
    local mean_ulp_256tiles=${shape_mean_ulp[256_tiles]:-0}
    local median_ulp_256tiles=${shape_median_ulp[256_tiles]:-0}
    local p99_ulp_256tiles=${shape_p99_ulp[256_tiles]:-0}

    local profiler_height=${profiler_timing_us[height_sharded]:-0}
    local host_height=${host_timing_ms[height_sharded]:-0}
    local mae_height=${shape_mae[height_sharded]:-0}
    local rmse_height=${shape_rmse[height_sharded]:-0}
    local max_height=${shape_max_error[height_sharded]:-0}
    local mean_rel_height=${shape_mean_rel[height_sharded]:-0}
    local max_ulp_height=${shape_max_ulp[height_sharded]:-0}
    local mean_ulp_height=${shape_mean_ulp[height_sharded]:-0}
    local median_ulp_height=${shape_median_ulp[height_sharded]:-0}
    local p99_ulp_height=${shape_p99_ulp[height_sharded]:-0}

    local profiler_yolov4=${profiler_timing_us[yolov4]:-0}
    local host_yolov4=${host_timing_ms[yolov4]:-0}
    local mae_yolov4=${shape_mae[yolov4]:-0}
    local rmse_yolov4=${shape_rmse[yolov4]:-0}
    local max_yolov4=${shape_max_error[yolov4]:-0}
    local mean_rel_yolov4=${shape_mean_rel[yolov4]:-0}
    local max_ulp_yolov4=${shape_max_ulp[yolov4]:-0}
    local mean_ulp_yolov4=${shape_mean_ulp[yolov4]:-0}
    local median_ulp_yolov4=${shape_median_ulp[yolov4]:-0}
    local p99_ulp_yolov4=${shape_p99_ulp[yolov4]:-0}

    # Determine overall status
    local status
    if [[ "$config_passed" == true ]]; then
        status="pass"
    elif [[ "$config_timeout" == true ]]; then
        status="timeout"
    else
        status="fail"
    fi

    # Write CSV row with per-shape timing and error metrics
    # Note: actual_degree_max and actual_degree_sum are inserted after degree column
    echo "$activation,$precision,$depth,$degree,$actual_degree_max,$actual_degree_sum,$segmentation,$fitting,$refined,$profiler_single,$host_single,$mae_single,$rmse_single,$max_single,$mean_rel_single,$max_ulp_single,$mean_ulp_single,$median_ulp_single,$p99_ulp_single,$profiler_8tiles,$host_8tiles,$mae_8tiles,$rmse_8tiles,$max_8tiles,$mean_rel_8tiles,$max_ulp_8tiles,$mean_ulp_8tiles,$median_ulp_8tiles,$p99_ulp_8tiles,$profiler_256tiles,$host_256tiles,$mae_256tiles,$rmse_256tiles,$max_256tiles,$mean_rel_256tiles,$max_ulp_256tiles,$mean_ulp_256tiles,$median_ulp_256tiles,$p99_ulp_256tiles,$profiler_height,$host_height,$mae_height,$rmse_height,$max_height,$mean_rel_height,$max_ulp_height,$mean_ulp_height,$median_ulp_height,$p99_ulp_height,$profiler_yolov4,$host_yolov4,$mae_yolov4,$rmse_yolov4,$max_yolov4,$mean_rel_yolov4,$max_ulp_yolov4,$mean_ulp_yolov4,$median_ulp_yolov4,$p99_ulp_yolov4,$theoretical_mae,$theoretical_max,$status" >> "$results_file"
}

# Write CSV row for best/native configurations
write_csv_row_simple() {
    local prefix="$1"       # e.g., "activation,precision,config_specific_fields"
    local results_file="$2"

    # Extract timing for each shape
    local single_tile_time=${kernel_exec_results[single_tile]:-0}
    local eight_tiles_time=${kernel_exec_results[8_tiles]:-0}
    local tiles_256_time=${kernel_exec_results[256_tiles]:-0}
    local height_sharded_time=${kernel_exec_results[height_sharded]:-0}
    local yolov4_time=${kernel_exec_results[yolov4]:-0}

    # Determine status
    local status
    if [[ "$config_passed" == true ]]; then
        status="pass"
    elif [[ "$config_timeout" == true ]]; then
        status="timeout"
    else
        status="fail"
    fi

    # Write CSV row: prefix, 5 timing columns, accuracy metrics, status
    echo "$prefix,$single_tile_time,$eight_tiles_time,$tiles_256_time,$height_sharded_time,$yolov4_time,${config_mae:-nan},${config_rmse:-nan},${config_max:-nan},${config_mean_rel:-inf},${config_max_ulp:-inf},${config_mean_ulp:-inf},$status" >> "$results_file"
}

# Standard CSV header columns: per-shape timing interleaved with per-shape error metrics
CSV_SHAPE_COLUMNS="time_profiler_single_tile_us,time_host_single_tile_ms,mae_single_tile,rmse_single_tile,max_error_single_tile,mean_rel_error_single_tile,max_ulp_error_single_tile,mean_ulp_error_single_tile,median_ulp_error_single_tile,p99_ulp_error_single_tile,time_profiler_8_tiles_us,time_host_8_tiles_ms,mae_8_tiles,rmse_8_tiles,max_error_8_tiles,mean_rel_error_8_tiles,max_ulp_error_8_tiles,mean_ulp_error_8_tiles,median_ulp_error_8_tiles,p99_ulp_error_8_tiles,time_profiler_256_tiles_us,time_host_256_tiles_ms,mae_256_tiles,rmse_256_tiles,max_error_256_tiles,mean_rel_error_256_tiles,max_ulp_error_256_tiles,mean_ulp_error_256_tiles,median_ulp_error_256_tiles,p99_ulp_error_256_tiles,time_profiler_height_sharded_us,time_host_height_sharded_ms,mae_height_sharded,rmse_height_sharded,max_error_height_sharded,mean_rel_error_height_sharded,max_ulp_error_height_sharded,mean_ulp_error_height_sharded,median_ulp_error_height_sharded,p99_ulp_error_height_sharded,time_profiler_yolov4_us,time_host_yolov4_ms,mae_yolov4,rmse_yolov4,max_error_yolov4,mean_rel_error_yolov4,max_ulp_error_yolov4,mean_ulp_error_yolov4,median_ulp_error_yolov4,p99_ulp_error_yolov4"

# Clear stale shared memory segments left by killed TT-Metal processes
# Stale /dev/shm files block device initialization (ARC core timeout)
# Also cleans up OpenMPI session directories that accumulate per-process
# and cause segfaults when they grow too large (~24k+ subdirs)
# Usage: clear_shm [username]
clear_shm() {
    local user="${1:-$(whoami)}"
    local count
    count=$(ls /dev/shm/sm_segment.*"${user}"* 2>/dev/null | wc -l)
    if [[ "$count" -gt 0 ]]; then
        # Silently clean up (no need to spam sweep output)
        for f in /dev/shm/sm_segment.*"${user}"*; do
            rm -f "$f"
        done
    fi

    # Clean up stale OpenMPI session directories
    # Each TTNN/MPI invocation creates a subdirectory; they accumulate and
    # eventually cause MPI runtime init to segfault
    local uid
    uid=$(id -u)
    for d in /tmp/ompi.*."${uid}"; do
        if [[ -d "$d" ]]; then
            rm -rf "$d"
        fi
    done
}

# Reset device using tt-smi (clears shm first to avoid ARC timeout)
# Usage: reset_device [device_id]
reset_device() {
    local device_id="${1:-0}"
    clear_shm
    if command -v tt-smi &> /dev/null; then
        tt-smi -r "$device_id" &> /dev/null || true
        sleep 2  # Give device time to reset
    elif [[ -f /opt/venv/bin/tt-smi ]]; then
        /opt/venv/bin/tt-smi -r "$device_id" &> /dev/null || true
        sleep 2
    fi
}

# Convert degree method name to numeric degree
# Usage: degree=$(degree_to_number "cubic")  # Returns 3
degree_to_number() {
    case "$1" in
        constant) echo 0 ;;
        linear) echo 1 ;;
        quadratic) echo 2 ;;
        cubic) echo 3 ;;
        quartic) echo 4 ;;
        quintic) echo 5 ;;
        hexic) echo 6 ;;
        septic) echo 7 ;;
        octic) echo 8 ;;
        nonic) echo 9 ;;
        decic) echo 10 ;;
        undecic) echo 11 ;;
        dodecic) echo 12 ;;
        tredecic) echo 13 ;;
        tetradecic) echo 14 ;;
        pentadecic) echo 15 ;;
        hexadecic) echo 16 ;;
        *) echo "" ;;
    esac
}

# Get test range for an activation from ACTIVATION_RANGES
# Usage: get_test_range "sigmoid" -> "-10 10"
get_test_range() {
    local activation="$1"
    for entry in "${ACTIVATION_RANGES[@]}"; do
        local name="${entry%%:*}"
        local rest="${entry#*:}"
        local min="${rest%%:*}"
        local max="${rest#*:}"
        if [[ "$name" == "$activation" ]]; then
            echo "$min $max"
            return
        fi
    done
    # Default range if not found
    echo "-10 10"
}

# Get hardware output directory for an activation
# Creates directory structure: data/hardware_outputs/<arch>/<activation>/
# Usage: output_dir=$(get_hardware_output_dir "gelu" "$WORK_DIR")
# Note: work_dir can be absolute path or relative to REPO_ROOT
get_hardware_output_dir() {
    local activation="$1"
    local work_dir="$2"

    # Strip _b0 suffix from architecture name for cleaner paths
    local arch_short="${ARCH_NAME/_b0/}"

    # Build directory path - handle both absolute and relative paths
    local base_dir
    if [[ "$work_dir" == /* ]]; then
        # Absolute path - use directly
        base_dir="$work_dir"
    else
        # Relative path - prefix with REPO_ROOT
        base_dir="$REPO_ROOT/${work_dir}"
    fi
    local output_dir="${base_dir}/data/hardware_outputs/${arch_short}/${activation}"

    # Create directory if it doesn't exist
    mkdir -p "$output_dir"

    echo "$output_dir"
}

# Build hardware output CSV filename (polynomial)
# Canonical format: {activation}_{precision}_p{degree}_s{segments}_{segmentation}_{fitting}_tiles{count}.csv
# Usage: filename=$(build_hw_poly_csv_filename activation precision degree segments segmentation fitting tile_count)
build_hw_poly_csv_filename() {
    local activation="$1"
    local precision="$2"
    local degree="$3"
    local segments="$4"
    local segmentation="$5"
    local fitting="$6"
    local tile_count="$7"
    echo "${activation}_${precision}_p${degree}_s${segments}_${segmentation}_${fitting}_tiles${tile_count}.csv"
}

# Build hardware output CSV filename (rational)
# Canonical format: {activation}_{precision}_n{num}d{den}_s{segments}_{segmentation}_{fitting}_tiles{count}.csv
# Usage: filename=$(build_hw_rational_csv_filename activation precision num_deg den_deg segments segmentation fitting tile_count)
build_hw_rational_csv_filename() {
    local activation="$1"
    local precision="$2"
    local num_deg="$3"
    local den_deg="$4"
    local segments="$5"
    local segmentation="$6"
    local fitting="$7"
    local tile_count="$8"
    echo "${activation}_${precision}_n${num_deg}d${den_deg}_s${segments}_${segmentation}_${fitting}_tiles${tile_count}.csv"
}

# Get full hardware output CSV path (polynomial)
# Usage: csv_path=$(get_hw_poly_csv_path activation precision degree segments segmentation fitting tile_count work_dir)
get_hw_poly_csv_path() {
    local activation="$1"
    local precision="$2"
    local degree="$3"
    local segments="$4"
    local segmentation="$5"
    local fitting="$6"
    local tile_count="$7"
    local work_dir="$8"

    local output_dir
    output_dir=$(get_hardware_output_dir "$activation" "$work_dir")
    local filename
    filename=$(build_hw_poly_csv_filename "$activation" "$precision" "$degree" "$segments" "$segmentation" "$fitting" "$tile_count")
    echo "${output_dir}/${filename}"
}

# Get full hardware output CSV path (rational)
# Usage: csv_path=$(get_hw_rational_csv_path activation precision num_deg den_deg segments segmentation fitting tile_count work_dir)
get_hw_rational_csv_path() {
    local activation="$1"
    local precision="$2"
    local num_deg="$3"
    local den_deg="$4"
    local segments="$5"
    local segmentation="$6"
    local fitting="$7"
    local tile_count="$8"
    local work_dir="$9"

    local output_dir
    output_dir=$(get_hardware_output_dir "$activation" "$work_dir")
    local filename
    filename=$(build_hw_rational_csv_filename "$activation" "$precision" "$num_deg" "$den_deg" "$segments" "$segmentation" "$fitting" "$tile_count")
    echo "${output_dir}/${filename}"
}

# Initialize architecture detection
# Detects architecture from hostname and sets ARCH_NAME and ARCH_PREFIX
# Hostname is authoritative — overrides stale env vars to prevent cross-arch writes
# Usage: init_arch_detection
init_arch_detection() {
    # Detect architecture from hostname (authoritative)
    if [[ $(hostname) == *"bh"* ]]; then
        local detected_arch=blackhole
    else
        local detected_arch=wormhole_b0
    fi

    if [[ -n "$ARCH_NAME" && "$ARCH_NAME" != "$detected_arch" ]]; then
        echo "WARNING: ARCH_NAME='$ARCH_NAME' doesn't match hostname '$(hostname)' (expected '$detected_arch'). Overriding." >&2
    fi
    export ARCH_NAME="$detected_arch"

    # Strip _b0 suffix for cleaner directory names (wormhole_b0 -> wormhole)
    ARCH_PREFIX=$(echo "$ARCH_NAME" | sed 's/_b0$//')
    export ARCH_PREFIX
}

# ANSI color codes for terminal output
# Usage: echo -e "${BLUE}[INFO]${NC} Message"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# =============================================================================
# CSV Filename Helpers (canonical format)
# =============================================================================
# Source from tt-polynomial-fitter if available, otherwise define inline
#
# Canonical CSV format:
#   Polynomial: {activation}_p{degree}_s{segments}_{segmentation}_{fitting}_{metric}.csv
#   Rational:   {activation}_n{num}d{den}_s{segments}_{segmentation}_{fitting}_{metric}.csv

# Coefficient filename builders
build_poly_csv_filename() {
    # Usage: build_poly_csv_filename activation degree segments segmentation fitting metric
    echo "${1}_p${2}_s${3}_${4}_${5}_${6}.csv"
}

build_rational_csv_filename() {
    # Usage: build_rational_csv_filename activation num_deg den_deg segments segmentation fitting metric
    echo "${1}_n${2}d${3}_s${4}_${5}_${6}_${7}.csv"
}

    parse_csv_filename() {
        local basename="${1%.csv}"
        CSV_ACTIVATION="" CSV_DEGREE="" CSV_NUM_DEG="" CSV_DEN_DEG=""
        CSV_SEGMENTS="" CSV_SEGMENTATION="" CSV_FITTING="" CSV_METRIC="" CSV_APPROX_TYPE=""

        # Try rational: {activation}_n{num}d{den}_s{segments}_{segmentation}_{fitting}_{metric}
        if [[ $basename =~ ^(.+)_n([0-9]+)d([0-9]+)_s([0-9]+)_(.+)_([a-z]+)_(ulp|mae|max)$ ]]; then
            CSV_ACTIVATION="${BASH_REMATCH[1]}"
            CSV_NUM_DEG="${BASH_REMATCH[2]}"
            CSV_DEN_DEG="${BASH_REMATCH[3]}"
            CSV_DEGREE="${CSV_NUM_DEG}/${CSV_DEN_DEG}"
            CSV_SEGMENTS="${BASH_REMATCH[4]}"
            CSV_SEGMENTATION="${BASH_REMATCH[5]}"
            CSV_FITTING="${BASH_REMATCH[6]}"
            CSV_METRIC="${BASH_REMATCH[7]}"
            CSV_APPROX_TYPE="rational"
            return 0
        fi

        # Try polynomial: {activation}_p{degree}_s{segments}_{segmentation}_{fitting}_{metric}
        if [[ $basename =~ ^(.+)_p([0-9]+)_s([0-9]+)_(.+)_([a-z]+)_(ulp|mae|max)$ ]]; then
            CSV_ACTIVATION="${BASH_REMATCH[1]}"
            CSV_DEGREE="${BASH_REMATCH[2]}"
            CSV_SEGMENTS="${BASH_REMATCH[3]}"
            CSV_SEGMENTATION="${BASH_REMATCH[4]}"
            CSV_FITTING="${BASH_REMATCH[5]}"
            CSV_METRIC="${BASH_REMATCH[6]}"
            CSV_APPROX_TYPE="polynomial"
            return 0
        fi
        return 1
    }

# Build full path to coefficient CSV
# Usage: csv_path=$(get_poly_csv_path activation degree segments segmentation fitting metric)
get_poly_csv_path() {
    local filename
    filename=$(build_poly_csv_filename "$1" "$2" "$3" "$4" "$5" "$6")
    echo "${TT_POLY_FIT_DIR}/data/coefficients/${filename}"
}

# Usage: csv_path=$(get_rational_csv_path activation num_deg den_deg segments segmentation fitting metric)
get_rational_csv_path() {
    local filename
    filename=$(build_rational_csv_filename "$1" "$2" "$3" "$4" "$5" "$6" "$7")
    echo "${TT_POLY_FIT_DIR}/data/coefficients/${filename}"
}

# Compute actual (used) polynomial degree from coefficient CSV
# Returns: max_actual_degree,degree_sum
# The actual degree is the highest non-zero coefficient across all segments
# Usage: result=$(get_actual_degree "/path/to/coeffs.csv")
#        max_deg=$(echo "$result" | cut -d',' -f1)
#        deg_sum=$(echo "$result" | cut -d',' -f2)
get_actual_degree() {
    local csv_path="$1"
    python3 -c "
import csv
import sys

csv_path = '$csv_path'
if not csv_path:
    print('0,0')
    sys.exit(0)

try:
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

    # Find coefficient columns (c0, c1, c2, ...)
    coeff_cols = sorted([h for h in headers if h.startswith('c') and h[1:].isdigit()],
                        key=lambda h: int(h[1:]))
    if not coeff_cols:
        print('0,0')
        sys.exit(0)

    max_degree = int(coeff_cols[-1][1:])
    threshold = 1e-10

    segment_degrees = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('segment_id', '').upper() == 'METADATA':
                continue
            # Find highest non-zero coefficient
            seg_deg = 0
            for deg in range(max_degree, -1, -1):
                coeff_key = f'c{deg}'
                if coeff_key in row and row[coeff_key]:
                    try:
                        if abs(float(row[coeff_key])) > threshold:
                            seg_deg = deg
                            break
                    except ValueError:
                        pass
            segment_degrees.append(seg_deg)

    if segment_degrees:
        print(f'{max(segment_degrees)},{sum(segment_degrees)}')
    else:
        print('0,0')
except Exception as e:
    print('0,0', file=sys.stderr)
    print('0,0')
"
}

# Find polynomial coefficient CSV and compute actual degree
# Usage: result=$(get_poly_actual_degree activation degree segments segmentation)
get_poly_actual_degree() {
    local activation="$1"
    local degree="$2"
    local segments="$3"
    local segmentation="$4"

    # Find coefficient CSV (try different metrics/fittings)
    local coeff_dir="${TT_POLY_FIT_DIR:-$HOME/workspace/tt-polynomial-fitter}/data/coefficients"
    local csv_path=""

    for metric in ulp max mae; do
        for fitting in any remez fpminimax lstsq; do
            local candidate="${coeff_dir}/${activation}_p${degree}_s${segments}_${segmentation}_${fitting}_${metric}.csv"
            if [[ -f "$candidate" ]]; then
                csv_path="$candidate"
                break 2
            fi
        done
    done

    if [[ -z "$csv_path" ]]; then
        echo "0,0"
        return
    fi

    get_actual_degree "$csv_path"
}

# =============================================================================
# Sweep Config Generation
# =============================================================================

# Regenerate sweep_config.dat from tt-polynomial-fitter activations/*.json
# Usage: regenerate_sweep_config [output_path]
# Environment:
#   TT_POLY_FIT_DIR: Path to tt-polynomial-fitter (auto-detected if not set)
#   SWEEP_CONFIG_AUTO_REGEN: Set to "1" to auto-regenerate before each sweep
regenerate_sweep_config() {
    local output_path="${1:-}"
    local script_dir="${BASH_SOURCE[0]%/*}"

    # Find generate_sweep_config.py
    local gen_script=""
    local search_paths=(
        "$script_dir/../generic_lut_activation_embedded/tools/generate_sweep_config.py"
        "$script_dir/tools/generate_sweep_config.py"
    )
    for p in "${search_paths[@]}"; do
        if [[ -f "$p" ]]; then
            gen_script="$p"
            break
        fi
    done

    if [[ -z "$gen_script" ]]; then
        echo "Warning: generate_sweep_config.py not found" >&2
        return 1
    fi

    # Determine output path
    if [[ -z "$output_path" ]]; then
        output_path="$script_dir/sweep_config.dat"
    fi

    # Find tt-polynomial-fitter
    local activations_dir=""
    if [[ -n "$TT_POLY_FIT_DIR" && -d "$TT_POLY_FIT_DIR/activations" ]]; then
        activations_dir="$TT_POLY_FIT_DIR/activations"
    elif [[ -d "/localdev/nkapre/tt-polynomial-fitter/activations" ]]; then
        activations_dir="/localdev/nkapre/tt-polynomial-fitter/activations"
    elif [[ -d "$HOME/workspace/tt-polynomial-fitter/activations" ]]; then
        activations_dir="$HOME/workspace/tt-polynomial-fitter/activations"
    fi

    if [[ -z "$activations_dir" ]]; then
        echo "Warning: tt-polynomial-fitter activations directory not found" >&2
        return 1
    fi

    python3 "$gen_script" --activations-dir "$activations_dir" --output "$output_path"
}

# Ensure sweep_config.dat is up-to-date
# Call this before loading sweep_config.dat in sweep scripts
# Usage: ensure_sweep_config [work_dir]
# Always attempts to regenerate from activations/*.json
ensure_sweep_config() {
    local work_dir="${1:-$WORK_DIR}"
    local config_file="${work_dir}/sweep_config.dat"

    # Always try to regenerate from JSON source of truth
    if regenerate_sweep_config "$config_file" 2>/dev/null; then
        echo "Regenerated sweep_config.dat from activations/*.json"
    fi

    # Check if config exists
    if [[ ! -f "$config_file" ]]; then
        echo "Warning: $config_file not found. Run: python3 tools/generate_sweep_config.py" >&2
        return 1
    fi

    return 0
}

