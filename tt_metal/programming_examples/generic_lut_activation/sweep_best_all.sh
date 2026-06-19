#!/bin/bash
# =============================================================================
# Wrapper script to run sweep_best.sh with all metric/precision combinations
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_BEST="$SCRIPT_DIR/sweep_best.sh"

# Source helpers for arch detection and TT_POLY_FIT_DIR
source "$SCRIPT_DIR/argparse_common.sh"
source "$SCRIPT_DIR/sweep_helpers.sh"
init_common_env
init_arch_detection
ARCH_SHORT="$ARCH_PREFIX"

# =============================================================================
# Verify SFPI version matches sfpi-version spec
# =============================================================================
SFPI_VERSION_FILE="$REPO_ROOT/tt_metal/sfpi-version"
if [[ -f "$SFPI_VERSION_FILE" ]]; then
    source "$SFPI_VERSION_FILE"
    expected_version="$sfpi_version"
    expected_build="$sfpi_build"

    # Find the installed SFPI compiler
    SFPI_GPP=$(find "$BUILD_DIR" -name "riscv-tt-elf-g++" -type f 2>/dev/null | head -1)
    if [[ -n "$SFPI_GPP" ]]; then
        installed_ver=$("$SFPI_GPP" --version 2>&1 | head -1)
        # Extract version from format: riscv-tt-elf-g++ (tenstorrent/sfpi:7.29.0-kapre-ice[301]) 15.1.0
        if [[ "$installed_ver" =~ :([^\[]+)\[([0-9]+)\] ]]; then
            actual_version="${BASH_REMATCH[1]}"
            actual_build="${BASH_REMATCH[2]}"
            if [[ "$actual_version" != "$expected_version" || "$actual_build" != "$expected_build" ]]; then
                echo "ERROR: SFPI version mismatch!"
                echo "  Expected: $expected_version [build $expected_build]"
                echo "  Installed: $actual_version [build $actual_build]"
                echo "  Compiler: $SFPI_GPP"
                echo "  Run ./build_metal.sh --clean to rebuild with correct SFPI version."
                exit 1
            fi
            SFPI_INFO="$actual_version [build $actual_build]"
        else
            echo "WARNING: Could not parse SFPI version from: $installed_ver"
            SFPI_INFO="unknown"
        fi
    else
        echo "WARNING: SFPI compiler not found in $BUILD_DIR"
        SFPI_INFO="not found"
    fi
else
    echo "WARNING: sfpi-version file not found at $SFPI_VERSION_FILE"
    SFPI_INFO="no version file"
fi

# Parse arguments to determine which combinations to run
ACTIVATION=""
PRECISION=""
METRIC=""
RESUME_FROM=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --activation|-a)
            ACTIVATION="$2"
            shift 2
            ;;
        --precision|-p)
            PRECISION="$2"
            shift 2
            ;;
        --metric|-m)
            METRIC="$2"
            if [[ "$METRIC" != "mae" && "$METRIC" != "max" && "$METRIC" != "ulp" ]]; then
                echo "Error: --metric must be one of: mae, max, ulp (got: $METRIC)"
                exit 1
            fi
            shift 2
            ;;
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--activation <name>[,<name>,...]] [OPTIONS]"
            echo ""
            echo "Automatically runs sweep_best.sh with multiple metric/precision combinations."
            echo "If --activation is omitted, runs all activations found in best.csv."
            echo ""
            echo "  (no --activation)                              Test all activations × all precisions × all metrics"
            echo "  --activation <name>                            Test all precisions × all metrics"
            echo "  --activation <name1>,<name2>,...               Test listed activations × all precisions × all metrics"
            echo "  --activation <name> --precision <type>         Test all metrics for that precision"
            echo "  --activation <name> --metric <type>            Test both precisions for that metric"
            echo ""
            echo "Options:"
            echo "  --resume-from <activation>  Skip all activations before <activation> and start from it"
            echo ""
            echo "Examples:"
            echo "  $0                                        # All activations, all combinations"
            echo "  $0 --activation gelu                      # 6 tests (fp32+bf16) × (max+mae+ulp)"
            echo "  $0 --activation gelu,silu,mish            # 18 tests: 3 activations × 2 prec × 3 metrics"
            echo "  $0 --activation gelu --precision fp32     # 3 tests: max, mae, ulp for fp32"
            echo "  $0 --activation gelu --metric mae         # 2 tests: fp32-mae, bf16-mae"
            echo "  $0 --resume-from mish                     # Resume full sweep starting at mish"
            echo ""
            echo "All other options are passed through to sweep_best.sh"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Determine which activations to run
ACTIVATIONS_TO_TEST=()
if [[ -n "$ACTIVATION" ]]; then
    IFS=',' read -ra ACTIVATIONS_TO_TEST <<< "$ACTIVATION"
else
    # Discover all unique activations from best.csv
    BEST_CSV="$TT_POLY_FIT_DIR/best.csv"
    if [[ ! -f "$BEST_CSV" ]]; then
        echo "Error: best.csv not found at $BEST_CSV"
        echo "Set TT_POLY_FIT_DIR or provide --activation <name>"
        exit 1
    fi
    while IFS=',' read -r act _rest; do
        ACTIVATIONS_TO_TEST+=("$act")
    done < <(tail -n +2 "$BEST_CSV" | awk -F',' '{print $1}' | sort -u)
    echo "No --activation specified: running all ${#ACTIVATIONS_TO_TEST[@]} activations from best.csv"
fi

# Determine what combinations to run
PRECISIONS_TO_TEST=()
METRICS_TO_TEST=()

if [[ -n "$PRECISION" ]]; then
    PRECISIONS_TO_TEST=("$PRECISION")
else
    PRECISIONS_TO_TEST=("bf16" "fp32")
fi

if [[ -n "$METRIC" ]]; then
    METRICS_TO_TEST=("$METRIC")
else
    METRICS_TO_TEST=("max" "mae" "ulp")
fi

# Calculate total tests
TOTAL_TESTS=$((${#ACTIVATIONS_TO_TEST[@]} * ${#PRECISIONS_TO_TEST[@]} * ${#METRICS_TO_TEST[@]}))

echo "================================================================================"
echo "        Sweep Best - Multi-Combination Runner"
echo "================================================================================"
echo "Activations to test: ${ACTIVATIONS_TO_TEST[*]}"
echo "Precisions to test: ${PRECISIONS_TO_TEST[*]}"
echo "Metrics to test: ${METRICS_TO_TEST[*]}"
echo "SFPI version: ${SFPI_INFO:-unknown}"
echo "Total test combinations: $TOTAL_TESTS"
echo "================================================================================"
echo ""

# Validate --resume-from value if given
if [[ -n "$RESUME_FROM" ]]; then
    found=false
    for act in "${ACTIVATIONS_TO_TEST[@]}"; do
        if [[ "$act" == "$RESUME_FROM" ]]; then found=true; break; fi
    done
    if [[ "$found" == false ]]; then
        echo "Error: --resume-from '$RESUME_FROM' not found in activation list: ${ACTIVATIONS_TO_TEST[*]}"
        exit 1
    fi
    echo "Resuming from: $RESUME_FROM (skipping earlier activations)"
    echo ""
fi

# Run all combinations
test_num=0
SKIPPING="${RESUME_FROM:+true}"  # true if RESUME_FROM is set, else empty
for act in "${ACTIVATIONS_TO_TEST[@]}"; do
    # If resuming, skip until we reach the target activation
    if [[ "$SKIPPING" == true ]]; then
        if [[ "$act" == "$RESUME_FROM" ]]; then
            SKIPPING=false
        else
            echo "Skipping (resume): $act"
            test_num=$((test_num + ${#PRECISIONS_TO_TEST[@]} * ${#METRICS_TO_TEST[@]}))
            continue
        fi
    fi

    # Remove stale results CSV so this run starts fresh (fp32+bf16 will re-accumulate below)
    rm -f "$SCRIPT_DIR/data/$ARCH_SHORT/best/${act}.csv"

    for prec in "${PRECISIONS_TO_TEST[@]}"; do
        for metric in "${METRICS_TO_TEST[@]}"; do
            test_num=$((test_num + 1))

            echo ""
            echo "[$test_num/$TOTAL_TESTS] Running: $act $prec $metric"
            echo "================================================================================"

            # Build command
            cmd=("$SWEEP_BEST" "--activation" "$act" "--precision" "$prec" "--metric" "$metric")
            cmd+=("${EXTRA_ARGS[@]}")

            # Run sweep_best.sh with this combination
            "${cmd[@]}"

            echo "✓ Completed: $act $prec $metric"
        done
    done
done

echo ""
echo "================================================================================"
echo "ALL COMBINATIONS COMPLETE"
echo "================================================================================"
echo "Tested $TOTAL_TESTS combinations for: ${ACTIVATIONS_TO_TEST[*]}"
echo ""
echo "Results saved to: $SCRIPT_DIR/data/${ARCH_SHORT}/best/"
echo ""
