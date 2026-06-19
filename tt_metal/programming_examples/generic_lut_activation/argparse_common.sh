#!/bin/bash
# Common argument parsing for sweep scripts
# Source this file and call parse_common_args "$@"
#
# Usage in sweep script:
#   source "$(dirname "${BASH_SOURCE[0]}")/argparse_common.sh"
#   parse_common_args "$@"
#   # Script-specific parsing goes here
#   handle_remaining_args "$@"

# =============================================================================
# Common Variables (set defaults before parsing)
# =============================================================================

# Common arguments shared across all sweep scripts
FILTER_ACTIVATION=""          # Activation filter: single or comma-separated list (empty = all)
FILTER_ACTIVATION_LIST=()    # Parsed array form of FILTER_ACTIVATION
FILTER_PRECISION=""           # Precision filter: fp32, bf16, both (empty = both)
FILTER_SHAPE=""               # Shape filter: comma-separated shape names (empty = all)
TIMEOUT_SECONDS=120           # Timeout per test in seconds
SKIP_BUILD=false              # Skip build phase
SKIP_RUN=false                # Skip run phase (load existing results)
DRY_RUN=false                 # Show what would run without executing
SHOW_COMPARISON=false         # Show detailed comparison table after sweep
HELP_REQUESTED=false          # Help message requested
RESUME_FROM=""                # Resume sweep from this activation (skip earlier ones)
VERBOSE_BUILD=false           # Show verbose SFPI compiler output
USE_FITTING_RANGE=false       # bf16: test full range (default) vs fitting domain only

# Common benchmarking configuration
TILE_COUNTS=(256)             # Tile counts for testing
NUM_RUNS=3                    # Number of runs per config (can be overridden)

# =============================================================================
# Architecture Detection and Setup
# =============================================================================

setup_common_env() {
    # Detect architecture from hostname (authoritative — ignores stale env vars)
    if [[ $(hostname) == *"bh"* ]]; then
        local detected_arch=blackhole
    else
        local detected_arch=wormhole_b0
    fi

    if [[ -n "$ARCH_NAME" && "$ARCH_NAME" != "$detected_arch" ]]; then
        echo "WARNING: ARCH_NAME='$ARCH_NAME' doesn't match hostname '$(hostname)' (expected '$detected_arch'). Overriding." >&2
    fi
    export ARCH_NAME="$detected_arch"

    # Set platform prefix for result files
    if [[ "$ARCH_NAME" == "wormhole_b0" ]]; then
        PLATFORM_PREFIX="wormhole_"
    elif [[ "$ARCH_NAME" == "blackhole" ]]; then
        PLATFORM_PREFIX="blackhole_"
    else
        PLATFORM_PREFIX=""
    fi

    # Repository and build paths
    REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
    export TT_METAL_RUNTIME_ROOT=${TT_METAL_RUNTIME_ROOT:-$REPO_ROOT}
    BUILD_DIR="$TT_METAL_RUNTIME_ROOT/build_Release"

    # Polynomial fitter directory - search in order and pick first match
    if [[ -z "$TT_POLY_FIT_DIR" ]]; then
        if [[ -d "/localdev/nkapre/tt-polynomial-fitter" ]]; then
            TT_POLY_FIT_DIR="/localdev/nkapre/tt-polynomial-fitter"
        elif [[ -d "/proj_sw/user_dev/nkapre/tt-polynomial-fitter" ]]; then
            TT_POLY_FIT_DIR="/proj_sw/user_dev/nkapre/tt-polynomial-fitter"
        else
            TT_POLY_FIT_DIR="$HOME/workspace/tt-polynomial-fitter"
        fi
    fi

    # Export common variables for use in calling script
    export ARCH_NAME PLATFORM_PREFIX REPO_ROOT BUILD_DIR TT_POLY_FIT_DIR
}

# =============================================================================
# Common Help Text Fragments
# =============================================================================

show_common_options() {
    echo "Common Options:"
    echo "  --activation <name>[,...]  Activation(s) to test, comma-separated (default: all)"
    echo "  --precision <type>        Filter by precision: fp32, bf16, both (default: both)"
    echo "  --shape <names>           Comma-separated shape names to test (default: all)"
    echo "                            Available: single_tile, 8_tiles, 256_tiles, height_sharded, yolov4"
    echo "  --timeout <seconds>       Timeout per test in seconds (default: 120)"
    echo "  --tiles <count>           Number of tiles to test (default: 256)"
    echo "  --runs <n>                Number of benchmark runs per config (default: 3)"
    echo "  --skip-build              Skip build phase (use existing binaries)"
    echo "  --skip-run                Skip run phase (build only)"
    echo "  --dry-run                 Show what would be tested without actually running"
    echo "  --compare                 Show detailed comparison table after sweep"
    echo "  --resume-from <name>      Skip activations before <name> and append to existing CSV"
    echo "  --verbose-build           Show verbose SFPI compiler output (ninja -v)"
    echo "  --use-fitting-range       bf16: restrict test to fitting domain (default: all bf16 values)"
    echo "  -h, --help                Show this help message"
}

show_common_env_vars() {
    echo "Common Environment Variables:"
    echo "  ARCH_NAME                 Architecture (wormhole_b0, blackhole) - default: auto-detect"
    echo "  TT_POLY_FIT_DIR           Polynomial fitter directory (contains best.csv and data/)"
    echo "                            (default: search /localdev/nkapre, then /proj_sw/user_dev/nkapre)"
    echo "  TT_METAL_RUNTIME_ROOT     TT-Metal runtime root (default: repo root)"
}

# =============================================================================
# Argument Parsing
# =============================================================================

# Parse common arguments and remove them from $@
# Returns remaining unparsed arguments
parse_common_args() {
    PARSED_ARGS=()

    while [[ $# -gt 0 ]]; do
        case $1 in
            --activation)
                FILTER_ACTIVATION="$2"
                IFS=',' read -ra FILTER_ACTIVATION_LIST <<< "$2"
                shift 2
                ;;
            --precision)
                FILTER_PRECISION="$2"
                validate_precision "$FILTER_PRECISION" || exit 1
                # Convert "both" to empty string (means test both)
                if [[ "$FILTER_PRECISION" == "both" ]]; then
                    FILTER_PRECISION=""
                fi
                shift 2
                ;;
            --shape)
                FILTER_SHAPE="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT_SECONDS="$2"
                # Validate timeout is a positive integer
                if ! [[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
                    echo "Error: --timeout must be a positive integer (seconds)" >&2
                    exit 1
                fi
                shift 2
                ;;
            --tiles)
                TILE_COUNTS=($2)
                if ! [[ "$2" =~ ^[0-9]+$ ]]; then
                    echo "Error: --tiles must be a positive integer" >&2
                    exit 1
                fi
                shift 2
                ;;
            --runs)
                NUM_RUNS="$2"
                if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]]; then
                    echo "Error: --runs must be a positive integer" >&2
                    exit 1
                fi
                shift 2
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-run)
                SKIP_RUN=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --compare)
                SHOW_COMPARISON=true
                shift
                ;;
            --resume-from)
                RESUME_FROM="$2"
                shift 2
                ;;
            --verbose-build)
                VERBOSE_BUILD=true
                shift
                ;;
            --use-fitting-range)
                USE_FITTING_RANGE=true
                shift
                ;;
            -h|--help)
                HELP_REQUESTED=true
                shift
                ;;
            *)
                # Not a common arg, save for script-specific parsing
                PARSED_ARGS+=("$1")
                shift
                ;;
        esac
    done

    # Return remaining args to caller
    set -- "${PARSED_ARGS[@]}"
    export FILTER_ACTIVATION FILTER_ACTIVATION_LIST FILTER_PRECISION FILTER_SHAPE TIMEOUT_SECONDS TILE_COUNTS NUM_RUNS SKIP_BUILD SKIP_RUN DRY_RUN SHOW_COMPARISON HELP_REQUESTED RESUME_FROM VERBOSE_BUILD USE_FITTING_RANGE
}

# =============================================================================
# Activation Filter Helpers
# =============================================================================

# Check if an activation matches the current filter.
# Usage: activation_matches "gelu"
# Returns 0 (true) if activation matches or no filter is set, 1 (false) otherwise.
activation_matches() {
    local name="$1"
    if [[ -z "$FILTER_ACTIVATION" ]]; then
        return 0  # No filter = match all
    fi
    for _a in "${FILTER_ACTIVATION_LIST[@]}"; do
        [[ "$_a" == "$name" ]] && return 0
    done
    return 1
}

# =============================================================================
# Result File Path Management
# =============================================================================

# Get result file path for current script
# Args: $1 = result type (best, polynomial, rational, native_sfpu)
#       $2 = work directory (optional, default: current)
get_results_file() {
    local result_type="$1"
    local work_dir="${2:-.}"

    # Strip _b0 suffix for cleaner directory names (wormhole_b0 -> wormhole)
    local arch_short="${ARCH_NAME/_b0/}"

    if [[ ${#FILTER_ACTIVATION_LIST[@]} -eq 1 ]]; then
        # Single activation: per-activation file: data/{arch}/{type}/{activation}.csv
        local results_dir="$work_dir/data/$arch_short/$result_type"
        mkdir -p "$results_dir"
        echo "$results_dir/${FILTER_ACTIVATION_LIST[0]}.csv"
    elif [[ -n "$FILTER_ACTIVATION" ]]; then
        # Multiple activations: consolidated file
        echo "$work_dir/data/$arch_short/${result_type}_results.csv"
    else
        # Consolidated file: data/{arch}/{type}_results.csv
        echo "$work_dir/data/$arch_short/${result_type}_results.csv"
    fi
}

# Get hardware output directory for per-element CSV dumps
# Args: $1 = activation name
#       $2 = work directory (optional, default: current)
# Returns: Path to activation-specific hardware output directory
get_hardware_output_dir() {
    local activation="$1"
    local work_dir="${2:-.}"

    # Strip _b0 suffix for cleaner directory names (wormhole_b0 -> wormhole)
    local arch_short="${ARCH_NAME/_b0/}"

    # Per-activation directory: data/hardware_outputs/{arch}/{activation}/
    local output_dir="$work_dir/data/hardware_outputs/$arch_short/$activation"
    mkdir -p "$output_dir"
    echo "$output_dir"
}

# =============================================================================
# Common Validation Functions
# =============================================================================

# Validate precision value
validate_precision() {
    local precision="$1"
    case "$precision" in
        fp32|bf16|both)
            return 0
            ;;
        *)
            echo "Error: Invalid precision '$precision'. Must be: fp32, bf16, or both" >&2
            return 1
            ;;
    esac
}

# Validate segmentation type
validate_segmentation() {
    local seg="$1"
    case "$seg" in
        uniform|chebyshev|curvature)
            return 0
            ;;
        *)
            echo "Error: Invalid segmentation '$seg'. Must be: uniform, chebyshev, or curvature" >&2
            return 1
            ;;
    esac
}

# =============================================================================
# Initialization
# =============================================================================

# Call this at the start of your script to set up common environment
init_common_env() {
    setup_common_env
}

# =============================================================================
# Usage Example (for reference - not executed when sourced)
# =============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    cat <<'EOF'
Usage example in sweep script:

#!/bin/bash
set -e

# Source common argument parsing
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/argparse_common.sh"

# Initialize common environment
init_common_env

# Define script-specific defaults
PRECISION="both"
COMPARE=false

# Define script-specific help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    show_common_options
    echo ""
    echo "Script-Specific Options:"
    echo "  --precision <type>        Filter by precision (fp32, bf16, both) (default: both)"
    echo "  --compare                 Show comparison table"
    echo ""
    show_common_env_vars
    echo ""
    exit 0
}

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
        --precision)
            PRECISION="$2"
            validate_precision "$PRECISION" || exit 1
            shift 2
            ;;
        --compare)
            COMPARE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Get results file path
WORK_DIR="tt_metal/programming_examples/generic_lut_activation"
RESULTS_FILE=$(get_results_file "polynomial" "$WORK_DIR")

echo "Results file: $RESULTS_FILE"
echo "Architecture: $ARCH_NAME"
echo "Filter activation: ${FILTER_ACTIVATION:-all}"
EOF
fi
