#!/usr/bin/env bash
# Depth/degree host timing sweep (no profiler overhead)
# Measures TIMING_KERNEL_EXECUTION across tile shapes
# Works for both embedded (adhoc kernel generation) and non-embedded (runtime LUT loading)

set -e

# =============================================================================
# Auto-detect mode based on directory
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sweep_helpers.sh"
source "$SCRIPT_DIR/profiler_helpers.sh"

if [[ "$SCRIPT_DIR" == *"embedded"* ]]; then
    SWEEP_MODE="embedded"
else
    SWEEP_MODE="standard"
fi

# Initialize architecture detection and color codes
init_arch_detection

# =============================================================================
# Configuration
# =============================================================================

NUM_RUNS=3

# Test shapes
TEST_SHAPES=(
    "single_tile:32:32"
    "8_tiles:64:128"
    "256_tiles:512:512"
    "height_sharded:25600:128"
    "yolov4:5120:320"
)

# Sweep parameters (configure these to control test matrix)
ACTIVATIONS=("gelu")
DEGREES=(1 2 4 6 8)
SEGMENTS=(4 8 16 32)
PRECISIONS=("fp32" "bf16")
SEGMENTATIONS=("curvature")

# =============================================================================
# Environment Setup
# =============================================================================

REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo /localdev/nkapre/tt-metal)"
export TT_METAL_RUNTIME_ROOT="$REPO_ROOT"

if [[ "$SWEEP_MODE" == "embedded" ]]; then
    WORK_DIR="$REPO_ROOT/tt_metal/programming_examples/generic_lut_activation_embedded"
    ADHOC_KERNEL="$WORK_DIR/kernels/compute/adhoc/adhoc.cpp"
    ADHOC_BINARY="$REPO_ROOT/build_Release/programming_examples/programming_examples_generic_lut_activation_embedded_adhoc"
    ADHOC_TARGET="programming_examples_generic_lut_activation_embedded_adhoc"
    GENERATE_SCRIPT="$WORK_DIR/tools/generate_adhoc_kernel.py"
else
    WORK_DIR="$REPO_ROOT/tt_metal/programming_examples/generic_lut_activation"
    ADHOC_CONFIG_H="$WORK_DIR/adhoc_config.h"
    ADHOC_BINARY="$REPO_ROOT/build_Release/programming_examples/programming_examples_generic_lut_activation_adhoc"
    ADHOC_TARGET="programming_examples_generic_lut_activation_adhoc"
fi

BUILD_DIR="$REPO_ROOT/build_Release"

# Results directory
RESULTS_DIR="$WORK_DIR/logs/$ARCH_PREFIX/depth_degree_host_logs"
mkdir -p "$RESULTS_DIR"
mkdir -p "$WORK_DIR/data/$ARCH_PREFIX"

# CSV output
csv_file="$WORK_DIR/data/$ARCH_PREFIX/depth_degree_host.csv"

# =============================================================================
# Display Info
# =============================================================================

echo -e "${BLUE}[INFO]${NC} Depth/Degree Host Timing Sweep (no profiler)"
if [[ "$SWEEP_MODE" == "embedded" ]]; then
    echo -e "${BLUE}[INFO]${NC} Mode: EMBEDDED (adhoc kernel generation)"
else
    echo -e "${BLUE}[INFO]${NC} Mode: STANDARD (runtime LUT loading)"
fi
echo -e "${BLUE}[INFO]${NC} ======================================"
echo -e "${BLUE}[INFO]${NC} Architecture: $ARCH_NAME"
echo -e "${BLUE}[INFO]${NC} Activations: ${ACTIVATIONS[*]}"
echo -e "${BLUE}[INFO]${NC} Degrees: ${DEGREES[*]}"
echo -e "${BLUE}[INFO]${NC} Segments: ${SEGMENTS[*]}"
echo -e "${BLUE}[INFO]${NC} Precisions: ${PRECISIONS[*]}"
echo -e "${BLUE}[INFO]${NC} Test shapes:"
for shape in "${TEST_SHAPES[@]}"; do
    shape_name="${shape%%:*}"
    shape_rest="${shape#*:}"
    shape_rows="${shape_rest%%:*}"
    shape_cols="${shape_rest#*:}"
    tile_count=$(( (shape_rows / 32) * (shape_cols / 32) ))
    echo -e "${BLUE}[INFO]${NC}   - $shape_name: ${shape_rows}x${shape_cols} (${tile_count} tiles)"
done
echo -e "${BLUE}[INFO]${NC} Runs per config: $NUM_RUNS"
echo -e "${BLUE}[INFO]${NC}"

# =============================================================================
# Prepare Config Function
# =============================================================================

prepare_config() {
    local activation="$1"
    local degree="$2"
    local segments="$3"
    local segmentation="$4"

    if [[ "$SWEEP_MODE" == "embedded" ]]; then
        # Generate adhoc kernel and compile
        python3 "$GENERATE_SCRIPT" \
            --activation "$activation" \
            --degree "$degree" \
            --segments "$segments" \
            --segmentation "$segmentation" \
            --output "$ADHOC_KERNEL" > /dev/null 2>&1 || return 1

        ninja -C "$BUILD_DIR" "$ADHOC_TARGET" > /dev/null 2>&1 || return 1
    else
        # Check for LUT file first
        local lut_file
        for metric in ulp mae max; do
            lut_file=$(get_poly_csv_path "$activation" "$degree" "$segments" "$segmentation" "any" "$metric")
            if [[ -f "$lut_file" ]]; then
                CURRENT_LUT_FILE="$lut_file"
                break
            fi
        done
        if [[ -z "$CURRENT_LUT_FILE" || ! -f "$CURRENT_LUT_FILE" ]]; then
            return 1
        fi

        # Update adhoc_config.h and build
        local lut_size=$(( (segments + 1) + segments * (degree + 1) ))
        cat > "$ADHOC_CONFIG_H" << EOF
// Auto-generated adhoc configuration header
#pragma once
#define ADHOC_POLY_DEGREE $degree
#define ADHOC_NUM_SEGMENTS $segments
#define ADHOC_LUT_SIZE $lut_size
EOF

        ninja -C "$BUILD_DIR" "$ADHOC_TARGET" > /dev/null 2>&1 || return 1
    fi
    return 0
}

# =============================================================================
# Main Sweep
# =============================================================================

echo "activation,degree,segments,precision,segmentation,shape,tiles,host_kernel_exec_ms,runs" > "$csv_file"

success_count=0
fail_count=0
total_configs=0

for activation in "${ACTIVATIONS[@]}"; do
    range_output=$(get_test_range "$activation")
    range_min=$(echo "$range_output" | awk '{print $1}')
    range_max=$(echo "$range_output" | awk '{print $2}')

    for degree in "${DEGREES[@]}"; do
        for segments in "${SEGMENTS[@]}"; do
            for segmentation in "${SEGMENTATIONS[@]}"; do
                total_configs=$((total_configs + 1))
                config_name="${activation}_p${degree}_s${segments}_${segmentation}"

                echo -e "${BLUE}[INFO]${NC} Preparing $config_name..."

                # Prepare config (mode-specific)
                if ! prepare_config "$activation" "$degree" "$segments" "$segmentation"; then
                    echo -e "${YELLOW}[SKIP]${NC} Could not prepare $config_name"
                    continue
                fi

                config_success=true

                # Use --precision both when multiple precisions requested
                if [[ ${#PRECISIONS[@]} -gt 1 ]]; then
                    binary_precision="both"
                else
                    binary_precision="${PRECISIONS[0]}"
                fi

                echo -e "${BLUE}[INFO]${NC}   Testing precision: ${binary_precision}"

                # Build batch tile list and base args
                batch_tiles=$(build_batch_tiles)
                tile_arg=$(build_tile_arg "$batch_tiles")
                if [[ "$SWEEP_MODE" == "embedded" ]]; then
                    base_args="--activation $activation --precision $binary_precision --range-min $range_min --range-max $range_max"
                else
                    base_args="$CURRENT_LUT_FILE --activation $activation --precision $binary_precision --range-min $range_min --range-max $range_max"
                fi

                # Host timing pass: keyed by precision_shape
                declare -A shape_host_times=()
                shape_success=true

                for run in $(seq 1 $NUM_RUNS); do
                    set +e
                    output=$($ADHOC_BINARY $base_args $tile_arg 2>&1)
                    exit_code=$?
                    set -e

                    if [[ $exit_code -ne 0 ]]; then
                        shape_success=false
                        break
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

                        if [[ "$shape_success" == true && -n "${shape_host_times[$pkey]:-}" ]]; then
                            host_min=$(compute_kernel_exec_min "${shape_host_times[$pkey]}" || echo "0")
                            IFS=',' read -ra _times <<< "${shape_host_times[$pkey]}"
                            num_shape_runs=${#_times[@]}

                            echo -e "    ${precision} ${shape_name} (${tile_count} tiles): Host=${host_min}ms"

                            echo "$activation,$degree,$segments,$precision,$segmentation,$shape_name,$tile_count,$host_min,$num_shape_runs" >> "$csv_file"
                        else
                            echo -e "${RED}[ERROR]${NC}     ${precision} ${shape_name} test failed"
                            config_success=false
                            fail_count=$((fail_count + 1))
                        fi
                    done
                done

                if [[ "$config_success" == true ]]; then
                    success_count=$((success_count + 1))
                    echo -e "${GREEN}[OK]${NC} Completed $config_name"
                else
                    echo -e "${YELLOW}[WARN]${NC} Partial failure in $config_name"
                fi
                echo ""
            done
        done
    done
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo -e "${BLUE}[INFO]${NC} Testing complete"
echo -e "${BLUE}[INFO]${NC} ======================================"
if [[ $fail_count -eq 0 ]]; then
    echo -e "${GREEN}[SUCCESS]${NC} All ${success_count} configurations passed!"
else
    echo -e "${BLUE}[INFO]${NC} Configurations passed: ${success_count}/${total_configs}"
    echo -e "${RED}[INFO]${NC} Failed tests: ${fail_count}"
fi
echo -e "${GREEN}[SUCCESS]${NC} CSV written to: $csv_file"
echo ""
