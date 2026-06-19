#!/bin/bash
# Generate all visualization plots from benchmark results
# Works for both embedded and non-embedded directories

set -e

# =============================================================================
# Auto-detect mode based on directory
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

if [[ "$SCRIPT_DIR" == *"embedded"* ]]; then
    WORK_DIR="$SCRIPT_DIR"
else
    WORK_DIR="$SCRIPT_DIR"
fi

# Parse command-line arguments
ARCH="both"
ACTIVATION_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"
            if [[ "$ARCH" != "wormhole" && "$ARCH" != "blackhole" && "$ARCH" != "both" ]]; then
                echo "Error: --arch must be 'wormhole', 'blackhole', or 'both'"
                exit 1
            fi
            shift 2
            ;;
        --activation)
            ACTIVATION_FILTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Generates all visualization plots from benchmark CSV results."
            echo ""
            echo "Options:"
            echo "  --arch ARCH           Generate plots for specific architecture (default: both)"
            echo "  --activation NAME     PER-ACTIVATION MODE: Generate only plots for this activation"
            echo "                        (generates {activation}_*.png files only)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Plotting Modes:"
            echo "  WITHOUT --activation: AGGREGATE MODE (default)"
            echo "    - Generates ONLY aggregate plots (all_activations_*.png)"
            echo "    - Skips individual per-activation plots"
            echo "    - Use for final consolidated plots"
            echo ""
            echo "  WITH --activation: PER-ACTIVATION MODE"
            echo "    - Generates ONLY plots for specified activation ({activation}_*.png)"
            echo "    - Skips aggregate plots"
            echo "    - Use for parallel CI execution"
            echo ""
            echo "Plots generated:"
            echo "  Per-Activation Plots (with --activation):"
            echo "    • {activation}_mae_pareto.png          - Error vs runtime scatter"
            echo "    • {activation}_mae_vs_runtime.png      - Error vs runtime line plot"
            echo "    • {activation}_heatmap.png             - Depth-degree heatmap"
            echo "    • {activation}_degree_vs_mae.png       - Pareto frontier analysis"
            echo ""
            echo "  Aggregate Plots (without --activation):"
            echo "    • all_activations_mae_grid.png         - Grid comparison across activations"
            echo "    • all_activations_degree_vs_mae*.png   - Aggregate Pareto frontiers"
            echo "    • all_activations_heatmap.png          - Aggregate depth-degree heatmap"
            echo "    • per_activation_mae_lut_vs_sfpu.png   - LUT vs SFPU comparison"
            echo ""
            echo "Output directories:"
            echo "  plots/{arch}/error_vs_runtime/   - Error vs runtime plots"
            echo "  plots/{arch}/pareto/             - Pareto frontier plots"
            echo "  plots/{arch}/depth_degree/       - Depth-degree heatmaps"
            echo "  plots/{arch}/depth_degree_tracy/ - Tracy profiler SFPU heatmaps"
            echo ""
            echo "Examples:"
            echo "  # Aggregate mode (generate all_activations_* plots):"
            echo "  $0                                # All aggregate plots"
            echo "  $0 --arch blackhole               # Blackhole aggregate plots only"
            echo ""
            echo "  # Per-activation mode (for parallel CI):"
            echo "  $0 --activation gelu              # Only gelu_*.png plots"
            echo "  $0 --activation sigmoid           # Only sigmoid_*.png plots"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Generating Visualization Plots"
echo "Working directory: $WORK_DIR"
if [[ "$ARCH" != "both" ]]; then
    echo "Architecture: $ARCH"
fi
echo "=========================================="
echo ""

# Function to check if data exists for an architecture
check_arch_data() {
    local arch=$1
    local data_dir="$WORK_DIR/data/$arch"

    # Check for per-activation CSVs or consolidated CSVs
    if [[ -d "$data_dir/polynomial" ]] || [[ -d "$data_dir/best" ]] || \
       [[ -f "$data_dir/polynomial_results.csv" ]] || [[ -f "$data_dir/best_results.csv" ]]; then
        return 0
    fi
    return 1
}

# Function to generate plots for a specific architecture
generate_arch_plots() {
    local arch=$1

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Generating plots for: $arch"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Check if data exists
    if ! check_arch_data "$arch"; then
        echo "⚠️  Warning: No CSV results found in $WORK_DIR/data/$arch/, skipping"
        echo "   Run sweep_polynomial.sh, sweep_rational.sh, or sweep_best.sh first"
        return
    fi

    # 1. Error vs runtime tradeoff plots
    echo "1. Generating error vs runtime plots..."
    ERROR_ARGS="--arch-prefix $arch --data-dir $WORK_DIR/data"
    [[ -n "$ACTIVATION_FILTER" ]] && ERROR_ARGS="$ERROR_ARGS --activation $ACTIVATION_FILTER"
    python3 "$WORK_DIR/plots/plot_error_vs_runtime.py" $ERROR_ARGS
    echo "   ✓ Saved to $WORK_DIR/plots/${arch}/error_vs_runtime/"
    echo ""

    # 1b. Tracy profiler depth-degree plots (if available)
    local tracy_csv="$WORK_DIR/logs/${arch}/depth_degree_tracy.csv"
    if [[ -f "$tracy_csv" ]]; then
        echo "1b. Generating tracy profiler plots..."
        python3 "$WORK_DIR/plots/plot_depth_degree_tracy.py" --arch "$arch" --data-dir "$WORK_DIR"
        echo "   ✓ Saved to $WORK_DIR/plots/${arch}/depth_degree_tracy/"
        echo ""
    fi
}

# Determine which architectures have data
HAVE_WORMHOLE=false
HAVE_BLACKHOLE=false

if check_arch_data "wormhole"; then
    HAVE_WORMHOLE=true
fi
if check_arch_data "blackhole"; then
    HAVE_BLACKHOLE=true
fi

# Generate architecture-specific plots
if [[ "$ARCH" == "both" ]]; then
    if [[ "$HAVE_WORMHOLE" == "false" ]] && [[ "$HAVE_BLACKHOLE" == "false" ]]; then
        echo "❌ No CSV result files found in $WORK_DIR/data/ directory"
        echo "   Run sweep_polynomial.sh, sweep_rational.sh, or sweep_best.sh first"
        exit 1
    fi

    [[ "$HAVE_WORMHOLE" == "true" ]] && generate_arch_plots "wormhole"
    [[ "$HAVE_BLACKHOLE" == "true" ]] && generate_arch_plots "blackhole"
else
    generate_arch_plots "$ARCH"
fi

# 2. Pareto frontier and depth-degree analysis
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Generating Pareto frontier and depth-degree plots"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

generate_analysis_plots() {
    local arch=$1

    if ! check_arch_data "$arch"; then
        return
    fi

    echo "Generating analysis plots for $arch..."
    PARETO_ARGS="--arch-prefix $arch --data-dir $WORK_DIR/data"
    [[ -n "$ACTIVATION_FILTER" ]] && PARETO_ARGS="$PARETO_ARGS --activation $ACTIVATION_FILTER"

    echo "   - Pareto frontier (MAE + max_error)..."
    python3 "$WORK_DIR/plots/plot_pareto_frontier.py" $PARETO_ARGS

    echo "   - Depth-degree heatmaps..."
    python3 "$WORK_DIR/plots/plot_depth_degree.py" $PARETO_ARGS

    echo "   ✓ Saved to $WORK_DIR/plots/$arch/"
    echo ""
}

if [[ "$ARCH" == "both" ]]; then
    [[ "$HAVE_WORMHOLE" == "true" ]] && generate_analysis_plots "wormhole"
    [[ "$HAVE_BLACKHOLE" == "true" ]] && generate_analysis_plots "blackhole"
else
    generate_analysis_plots "$ARCH"
fi

echo "=========================================="
echo "✓ Plot generation complete!"
echo "=========================================="
echo ""
echo "Generated plot directories:"
if [[ "$ARCH" == "both" ]] || [[ "$ARCH" == "wormhole" ]]; then
    [[ -d "$WORK_DIR/plots/wormhole/pareto" ]] && echo "  • plots/wormhole/pareto/             - Pareto frontier analysis"
    [[ -d "$WORK_DIR/plots/wormhole/depth_degree" ]] && echo "  • plots/wormhole/depth_degree/       - Depth-degree heatmaps"
    [[ -d "$WORK_DIR/plots/wormhole/depth_degree_tracy" ]] && echo "  • plots/wormhole/depth_degree_tracy/ - Tracy profiler heatmaps"
    [[ -d "$WORK_DIR/plots/wormhole/error_vs_runtime" ]] && echo "  • plots/wormhole/error_vs_runtime/   - Error vs runtime plots"
fi
if [[ "$ARCH" == "both" ]] || [[ "$ARCH" == "blackhole" ]]; then
    [[ -d "$WORK_DIR/plots/blackhole/pareto" ]] && echo "  • plots/blackhole/pareto/            - Pareto frontier analysis"
    [[ -d "$WORK_DIR/plots/blackhole/depth_degree" ]] && echo "  • plots/blackhole/depth_degree/      - Depth-degree heatmaps"
    [[ -d "$WORK_DIR/plots/blackhole/depth_degree_tracy" ]] && echo "  • plots/blackhole/depth_degree_tracy/ - Tracy profiler heatmaps"
    [[ -d "$WORK_DIR/plots/blackhole/error_vs_runtime" ]] && echo "  • plots/blackhole/error_vs_runtime/  - Error vs runtime plots"
fi
echo ""
