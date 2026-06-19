#!/bin/bash
# ASCII plotting for LUT sweep results
# Shows impact of depth, degree, and precision on accuracy metrics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Default parameters
FILTER_ACTIVATION=""
FILTER_PRECISION=""
METRIC="mae"  # mae, rmse, or max_error
SHOW_THEORETICAL=""  # Show theoretical MAE alongside hardware MAE
INCLUDE_EMBEDDED=""  # Include embedded data for comparison with CB-based

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --activation <name>       Filter by activation (default: all)"
    echo "  --precision <bf16|fp32>   Filter by precision (default: both)"
    echo "  --metric <name>           Metric to plot: mae|rmse|max_error (default: mae)"
    echo "  --show-theoretical        Show theoretical MAE from LUT generation alongside hardware MAE"
    echo "  --include-embedded        Include embedded data alongside CB-based data for comparison"
    echo "  -h, --help                Show this help"
    echo ""
    echo "Generates ASCII plots showing:"
    echo "  1. Impact of depth on error"
    echo "  2. Impact of degree on error"
    echo "  3. Impact of precision on error"
    echo "  4. Impact of segmentation on error"
    echo ""
    echo "Examples:"
    echo "  $0 --activation digamma"
    echo "  $0 --activation digamma --precision bf16"
    echo "  $0 --metric rmse"
    echo "  $0 --activation sigmoid --show-theoretical"
    echo "  $0 --activation sigmoid --include-embedded"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --activation)
            FILTER_ACTIVATION="$2"
            shift 2
            ;;
        --precision)
            FILTER_PRECISION="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --show-theoretical)
            SHOW_THEORETICAL="true"
            shift
            ;;
        --include-embedded)
            INCLUDE_EMBEDDED="true"
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage"
            exit 1
            ;;
    esac
done

# Activate venv if available
if [[ -f "$REPO_ROOT/python_env/bin/activate" ]]; then
    source "$REPO_ROOT/python_env/bin/activate"
fi

# Export variables so Python can access them via os.environ
export FILTER_ACTIVATION
export FILTER_PRECISION
export FILTER_SEGMENTATION
export METRIC
export SHOW_THEORETICAL
export INCLUDE_EMBEDDED

# Run Python plotting script
python3 << 'PYTHON_SCRIPT'
import sys
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# Get parameters from environment
# Use current working directory (where user runs the script from), not where script file lives
script_dir = os.getcwd()
filter_activation = os.environ.get('FILTER_ACTIVATION', '')
filter_precision = os.environ.get('FILTER_PRECISION', '')
filter_segmentation = os.environ.get('FILTER_SEGMENTATION', '')
metric = os.environ.get('METRIC', 'mae')
show_theoretical = os.environ.get('SHOW_THEORETICAL', '') == 'true'
include_embedded = os.environ.get('INCLUDE_EMBEDDED', '') == 'true'

# Validate metric
valid_metrics = ['mae', 'rmse', 'max_error']
if metric not in valid_metrics:
    print(f"Error: Invalid metric '{metric}'. Must be one of: {', '.join(valid_metrics)}")
    sys.exit(1)

# Validate precision if specified
if filter_precision and filter_precision not in ['bf16', 'fp32']:
    print(f"Error: Invalid precision '{filter_precision}'. Must be: bf16 or fp32")
    sys.exit(1)

metric_display = metric.upper().replace('_', ' ')

# Find all CSV result files
def load_csv_files_from_dir(directory, source_label):
    """Load CSV files from a directory and tag them with source"""
    csv_pattern = f"{directory}/wormhole_piecewise_*.csv"
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        csv_pattern = f"{directory}/piecewise_*.csv"
        csv_files = glob.glob(csv_pattern)

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                # Extract degree and precision from filename
                filename = Path(csv_file).stem
                parts = filename.replace('wormhole_', '').replace('piecewise_', '').split('_')

                # Parse filename: piecewise_<degree>_<precision>_results.csv
                # Check in order of specificity (hexic/octic before linear to avoid substring match)
                if 'constant' in filename:
                    degree = 'constant'
                elif 'hexic' in filename:
                    degree = 'hexic'
                elif 'octic' in filename:
                    degree = 'octic'
                elif 'quadratic' in filename:
                    degree = 'quadratic'
                elif 'cubic' in filename:
                    degree = 'cubic'
                elif 'linear' in filename:
                    degree = 'linear'
                else:
                    degree = 'unknown'

                precision = 'bf16' if 'bf16' in filename else 'fp32'

                df['degree'] = degree
                df['precision'] = precision
                df['source'] = source_label
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")

    return dfs, csv_files

# Load embedded data (current directory)
dfs, csv_files = load_csv_files_from_dir(script_dir, 'embedded')

if not csv_files:
    print("Error: No CSV result files found in current directory")
    sys.exit(1)

print(f"✓ Loaded {len(dfs)} embedded CSV files from current directory")

# Load CB-based data if requested
if include_embedded:
    cb_dir = Path(script_dir).parent / "generic_lut_activation"
    if cb_dir.exists():
        cb_dfs, cb_csv_files = load_csv_files_from_dir(str(cb_dir), 'cb-based')
        if cb_dfs:
            print(f"✓ Loaded {len(cb_dfs)} CB-based CSV files from {cb_dir}")
            dfs.extend(cb_dfs)
        else:
            print(f"Warning: No CB-based CSV files found in {cb_dir}")
    else:
        print(f"Warning: CB-based directory not found: {cb_dir}")

if not dfs:
    print("Error: No valid data found in CSV files")
    sys.exit(1)

# Combine all dataframes
data = pd.concat(dfs, ignore_index=True)

# Filter by activation if specified
if filter_activation:
    data = data[data['activation'] == filter_activation]
    if data.empty:
        print(f"Error: No data found for activation '{filter_activation}'")
        sys.exit(1)

# Filter by precision if specified
if filter_precision:
    data = data[data['precision'] == filter_precision]
    if data.empty:
        print(f"Error: No data found for precision '{filter_precision}'")
        sys.exit(1)

# Filter by segmentation if specified
if filter_segmentation:
    data = data[data['segmentation'] == filter_segmentation]
    if data.empty:
        print(f"Error: No data found for segmentation '{filter_segmentation}'")
        sys.exit(1)

# Filter out failed runs
data = data[data['status'] == 'pass']

if data.empty:
    print("Error: No successful runs found in data")
    sys.exit(1)

# Load theoretical error data if requested
theoretical_data = None
if show_theoretical:
    # Look for theoretical error summary CSV (fp32 by default, or bf16)
    precision_suffix = filter_precision if filter_precision else 'fp32'
    theoretical_csv = f"{script_dir}/plots/output/theoretical_error_summary_{precision_suffix}.csv"

    if os.path.exists(theoretical_csv):
        try:
            theoretical_df = pd.read_csv(theoretical_csv)
            # Rename MAE column for clarity
            if 'mae' in theoretical_df.columns:
                theoretical_df = theoretical_df.rename(columns={'mae': 'theoretical_mae'})
            theoretical_data = theoretical_df
            print(f"✓ Loaded theoretical errors from: {theoretical_csv}")
        except Exception as e:
            print(f"Warning: Could not load theoretical errors: {e}")
    else:
        print(f"Warning: Theoretical error file not found: {theoretical_csv}")
        print("         Run ./theoretical_error_analysis.sh first to generate it")

# Simple ASCII bar chart function
def ascii_bar_chart(labels, values, title, max_width=60, global_max=None, theoretical_values=None, embedded_values=None):
    """Create a simple ASCII bar chart with optional theoretical and embedded comparison"""
    print("\n" + "="*80)
    print(title)
    print("="*80)

    if len(values) == 0:
        print("No data to plot")
        return

    # Use global max if provided for consistent scaling
    max_val = global_max if global_max is not None else max(values)
    if embedded_values is not None and len(embedded_values) > 0:
        max_val = max(max_val, max(embedded_values))
    if max_val == 0:
        max_val = 1

    # Sort by value for better readability
    if theoretical_values is not None:
        sorted_data = sorted(zip(labels, values, theoretical_values), key=lambda x: x[1])
    elif embedded_values is not None:
        sorted_data = sorted(zip(labels, values, embedded_values), key=lambda x: x[1])
    else:
        sorted_data = sorted(zip(labels, values), key=lambda x: x[1])

    for item in sorted_data:
        if theoretical_values is not None:
            label, hw_val, theo_val = item
            bar_len = int((hw_val / max_val) * max_width)
            bar = "█" * bar_len

            # Show comparison: Hardware vs Theoretical
            if theo_val > 0:
                ratio = hw_val / theo_val
                if ratio > 10:
                    status = f"❌ {ratio:.0f}×"
                elif ratio > 2:
                    status = f"⚠️  {ratio:.1f}×"
                else:
                    status = f"✓ {ratio:.2f}×"
                print(f"{label:20s} {bar} HW={hw_val:.6f}  Theo={theo_val:.6f}  {status}")
            else:
                print(f"{label:20s} {bar} HW={hw_val:.6f}  Theo=N/A")
        elif embedded_values is not None:
            label, cb_val, emb_val = item
            # Show both CB-based and Embedded bars
            cb_bar_len = int((cb_val / max_val) * max_width)
            emb_bar_len = int((emb_val / max_val) * max_width)
            cb_bar = "█" * cb_bar_len
            emb_bar = "▓" * emb_bar_len

            # Calculate improvement ratio
            if emb_val > 0:
                ratio = cb_val / emb_val
                if ratio < 0.5:
                    status = f"✓ CB {ratio:.2f}× (CB better)"
                elif ratio > 2.0:
                    status = f"✓ Emb {1/ratio:.2f}× (Emb better)"
                else:
                    status = f"≈ {ratio:.2f}×"
            else:
                status = "N/A"

            print(f"{label:20s}")
            print(f"  CB:  {cb_bar} {cb_val:.6f}")
            print(f"  Emb: {emb_bar} {emb_val:.6f}  {status}")
        else:
            label, val = item
            bar_len = int((val / max_val) * max_width)
            bar = "█" * bar_len
            print(f"{label:20s} {bar} {val:.6f}")

    print()

# Simple ASCII line plot function
def ascii_line_plot(x_vals, y_vals, x_label, y_label, title, height=20, width=60):
    """Create a simple ASCII line plot"""
    print("\n" + "="*80)
    print(title)
    print("="*80)

    if len(x_vals) == 0 or len(y_vals) == 0:
        print("No data to plot")
        return

    # Create plot grid
    y_min, y_max = min(y_vals), max(y_vals)
    if y_min == y_max:
        y_max = y_min + 1

    x_min, x_max = min(x_vals), max(x_vals)

    # Normalize values to grid
    def normalize_y(val):
        return int((val - y_min) / (y_max - y_min) * (height - 1))

    def normalize_x(val):
        if x_max == x_min:
            return width // 2
        return int((val - x_min) / (x_max - x_min) * (width - 1))

    # Create empty grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot points
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        x_pos = normalize_x(x)
        y_pos = height - 1 - normalize_y(y)
        if 0 <= y_pos < height and 0 <= x_pos < width:
            grid[y_pos][x_pos] = '●'

    # Print grid with axes
    print(f"{y_label}:")
    print(f"{y_max:>10.6f} ┤", end='')
    print(''.join(grid[0]))

    for row in grid[1:-1]:
        print(" " * 11 + "│", end='')
        print(''.join(row))

    print(f"{y_min:>10.6f} ┤", end='')
    print(''.join(grid[-1]))
    print(" " * 11 + "└" + "─" * width)
    print(" " * 12 + f"{x_min:<10} {' ' * (width - 25)} {x_max:>10}")
    print(" " * 12 + f"{x_label}")
    print()

# Calculate global max for consistent scaling
# Use local max when filtering by activation or precision for better visibility
if filter_activation or filter_precision:
    # For single activation or precision, use None to auto-scale each chart
    global_max_value = None
else:
    # For all activations/precisions, use global max for fair comparison
    global_max_value = data[metric].max()

# Sample-by-sample comparison: Hardware vs Theoretical outputs
# Find matching hardware/theoretical CSV pairs and show divergence summary
import subprocess
from pathlib import Path as PathLib

hw_output_dir = PathLib(script_dir) / "data" / "hardware_outputs"
theo_output_dir = PathLib(script_dir) / "plots" / "output"

if hw_output_dir.exists() and theo_output_dir.exists():
    hw_csvs = list(hw_output_dir.glob("piecewise_*_*_*_*_*.csv"))
    theo_csvs = list(theo_output_dir.glob("theoretical_output_piecewise_*_*_*_*_*.csv"))

    # Build mapping: (degree, activation, depth, seg, precision) -> file
    hw_map = {}
    for csv in hw_csvs:
        parts = csv.stem.split('_')
        if len(parts) >= 5:
            # piecewise_{degree}_{activation}_{depth}_{seg}_{precision}.csv
            degree, activation, depth, seg, precision = parts[1], parts[2], parts[3], parts[4], parts[5].split('.')[0]
            hw_map[(degree, activation, depth, seg, precision)] = csv

    theo_map = {}
    for csv in theo_csvs:
        # theoretical_output_piecewise_{degree}_{activation}_{depth}_{seg}_{precision}.csv
        parts = csv.stem.replace('theoretical_output_', '').split('_')
        if len(parts) >= 5:
            degree, activation, depth, seg, precision = parts[1], parts[2], parts[3], parts[4], parts[5].split('.')[0]
            theo_map[(degree, activation, depth, seg, precision)] = csv

    # Find matching pairs
    matching_keys = set(hw_map.keys()) & set(theo_map.keys())

    if matching_keys and filter_activation:
        # Only show comparisons for filtered activation
        matching_keys = [k for k in matching_keys if k[1] == filter_activation]

    if matching_keys:
        print("\n" + "="*80)
        print("SAMPLE-BY-SAMPLE COMPARISON: Hardware vs Theoretical Outputs")
        print("="*80)
        print(f"Found {len(matching_keys)} matching hardware/theoretical CSV pairs\n")

        comparison_results = []

        for key in sorted(matching_keys):
            degree, activation, depth, seg, precision = key
            hw_csv = hw_map[key]
            theo_csv = theo_map[key]

            # Run compare_outputs.py
            try:
                result = subprocess.run(
                    ['python3', f'{script_dir}/compare_outputs.py',
                     str(hw_csv), str(theo_csv), '--top', '0'],
                    capture_output=True, text=True, timeout=30
                )

                if result.returncode == 0:
                    # Parse output for summary stats
                    lines = result.stdout.split('\n')
                    mean_diff = None
                    max_diff = None

                    for line in lines:
                        if 'Mean absolute difference:' in line:
                            mean_diff = float(line.split(':')[1].strip())
                        elif 'Max absolute difference:' in line:
                            max_diff = float(line.split(':')[1].strip())

                    if mean_diff is not None and max_diff is not None:
                        comparison_results.append({
                            'config': f"{degree}/{activation}/{depth}/{seg}",
                            'precision': precision,
                            'mean_diff': mean_diff,
                            'max_diff': max_diff
                        })
            except Exception as e:
                print(f"  Warning: Could not compare {key}: {e}")

        if comparison_results:
            # Show summary table
            print(f"{'Config':<30} {'Precision':<10} {'Mean Diff':<15} {'Max Diff':<15} {'Status':<10}")
            print("-" * 80)

            for result in sorted(comparison_results, key=lambda x: x['mean_diff'], reverse=True):
                # Classify divergence severity
                if result['mean_diff'] > 1.0:
                    status = "❌ CATASTROPHIC"
                elif result['mean_diff'] > 0.1:
                    status = "⚠️  HIGH"
                elif result['mean_diff'] > 0.01:
                    status = "⚠️  MODERATE"
                else:
                    status = "✓ GOOD"

                print(f"{result['config']:<30} {result['precision']:<10} "
                      f"{result['mean_diff']:<15.6e} {result['max_diff']:<15.6e} {status:<10}")

            print("\nNote: Run 'python3 compare_outputs.py <hw_csv> <theo_csv>' for detailed sample-by-sample analysis")

        print()

# Plot 1: Impact of depth on error (grouped by degree)
print("\n" + "█"*80)
print("  LUT SWEEP RESULTS: ASCII VISUALIZATION")
print("█"*80)
print(f"\nMetric: {metric_display}")
if filter_activation:
    print(f"Activation: {filter_activation}")
if filter_precision:
    print(f"Precision: {filter_precision}")
if filter_segmentation:
    print(f"Segmentation: {filter_segmentation}")
if include_embedded and 'source' in data.columns:
    sources = data['source'].unique()
    print(f"Comparison: {', '.join(sorted(sources))}")
print(f"{metric_display} range: [{data[metric].min():.6f}, {data[metric].max():.6f}]")

# Group by degree and compute average error per depth (show only key degrees)
degrees = ['constant', 'linear', 'quadratic', 'cubic', 'hexic', 'octic']
key_degrees = []
for degree in degrees:
    if not data[data['degree'] == degree].empty:
        key_degrees.append(degree)

# Helper function to extract theoretical MAEs
def get_theoretical_maes(degree, depths, activation, segmentation, theoretical_df):
    """Extract theoretical MAEs for given degree/depths/activation"""
    if theoretical_df is None:
        return None

    theo_values = []
    for depth in depths:
        # Map degree names to theoretical CSV method names
        degree_map = {
            'linear': 'piecewise_linear',
            'quadratic': 'piecewise_quadratic_remez',
            'cubic': 'piecewise_cubic_remez',
            'hexic': 'piecewise_hexic_remez',
            'octic': 'piecewise_octic_remez'
        }
        method_name = degree_map.get(degree, degree)

        # Filter theoretical data
        mask = (theoretical_df['activation'] == activation) & \
               (theoretical_df['method'] == method_name) & \
               (theoretical_df['depth'] == depth)

        if segmentation:
            mask = mask & (theoretical_df['segmentation'] == segmentation)

        matches = theoretical_df[mask]
        if not matches.empty:
            theo_values.append(matches['theoretical_mae'].iloc[0])
        else:
            theo_values.append(0)  # Missing data

    return theo_values if any(v > 0 for v in theo_values) else None

# Show all available degrees
for degree in key_degrees:
    degree_data = data[data['degree'] == degree]

    if include_embedded and 'source' in degree_data.columns:
        # Separate CB-based and embedded data
        cb_data = degree_data[degree_data['source'] == 'cb-based']
        emb_data = degree_data[degree_data['source'] == 'embedded']

        if not cb_data.empty and not emb_data.empty:
            # Get stats for both sources
            cb_depth_stats = cb_data.groupby('depth')[metric].mean().sort_index()
            emb_depth_stats = emb_data.groupby('depth')[metric].mean().sort_index()

            # Find common depths
            common_depths = sorted(set(cb_depth_stats.index) & set(emb_depth_stats.index))

            if common_depths:
                labels = [f"Depth {d:3d}" for d in common_depths]
                cb_values = [cb_depth_stats[d] for d in common_depths]
                emb_values = [emb_depth_stats[d] for d in common_depths]

                ascii_bar_chart(labels, cb_values,
                               f"Impact of Depth on {metric_display} ({degree.title()}) - CB vs Embedded",
                               global_max=global_max_value,
                               embedded_values=emb_values)
        else:
            # Only one source has data, plot normally
            depth_stats = degree_data.groupby('depth')[metric].mean().sort_index()
            if not depth_stats.empty:
                labels = [f"Depth {d:3d}" for d in depth_stats.index]
                values = depth_stats.values
                ascii_bar_chart(labels, values,
                               f"Impact of Depth on {metric_display} ({degree.title()})",
                               global_max=global_max_value)
    else:
        # Normal single-source plotting
        depth_stats = degree_data.groupby('depth')[metric].mean().sort_index()

        if not depth_stats.empty:
            labels = [f"Depth {d:3d}" for d in depth_stats.index]
            values = depth_stats.values

            # Get theoretical values if available
            theo_values = None
            if theoretical_data is not None and filter_activation and metric == 'mae':
                seg = filter_segmentation if filter_segmentation else 'uniform'
                theo_values = get_theoretical_maes(degree, depth_stats.index, filter_activation, seg, theoretical_data)

            ascii_bar_chart(labels, values,
                           f"Impact of Depth on {metric_display} ({degree.title()})",
                           global_max=global_max_value,
                           theoretical_values=theo_values)

# Plot 2: Impact of degree on error (show only one representative depth)
depths = sorted(data['depth'].unique())
# Pick middle depth as most representative
representative_depth = depths[len(depths)//2] if depths else depths[0]

depth_data = data[data['depth'] == representative_depth]
degree_stats = []
degree_labels = []

for degree in degrees:
    degree_subset = depth_data[depth_data['degree'] == degree]
    if not degree_subset.empty:
        degree_stats.append(degree_subset[metric].mean())
        degree_labels.append(degree.title())

if degree_stats:
    ascii_bar_chart(degree_labels, degree_stats,
                   f"Impact of Degree on {metric_display} (Depth={representative_depth})",
                   global_max=global_max_value)

# Plot 3: Impact of precision on error (compact)
precision_stats = data.groupby(['degree', 'precision'])[metric].mean().unstack(fill_value=0)

if not precision_stats.empty and len(precision_stats.columns) > 1:
    print("\n" + "="*80)
    print(f"Precision: BF16 vs FP32 {metric_display}")
    print("="*80)
    print(f"{'Degree':<12} {'BF16':>12} {'FP32':>12} {'FP32 Δ%':>12}")
    print("-" * 50)

    for degree in precision_stats.index:
        bf16_val = precision_stats.loc[degree, 'bf16'] if 'bf16' in precision_stats.columns else 0
        fp32_val = precision_stats.loc[degree, 'fp32'] if 'fp32' in precision_stats.columns else 0

        # Calculate percentage improvement (negative = FP32 better, positive = FP32 worse)
        if bf16_val > 0:
            pct_improvement = ((fp32_val - bf16_val) / bf16_val) * 100
            pct_str = f"{pct_improvement:+.1f}%"
        else:
            pct_str = "N/A"

        print(f"{degree:<12} {bf16_val:>12.6f} {fp32_val:>12.6f} {pct_str:>12}")
    print()

    # Add runtime comparison if runtime_min_ms column exists
    if 'runtime_min_ms' in data.columns:
        runtime_stats = data.groupby(['degree', 'precision'])['runtime_min_ms'].mean().unstack(fill_value=0)

        if not runtime_stats.empty and len(runtime_stats.columns) > 1:
            print("="*80)
            print(f"Precision: BF16 vs FP32 Runtime (Min)")
            print("="*80)
            print(f"{'Degree':<12} {'BF16 (ms)':>12} {'FP32 (ms)':>12} {'FP32 Δ%':>12}")
            print("-" * 50)

            for degree in runtime_stats.index:
                bf16_time = runtime_stats.loc[degree, 'bf16'] if 'bf16' in runtime_stats.columns else 0
                fp32_time = runtime_stats.loc[degree, 'fp32'] if 'fp32' in runtime_stats.columns else 0

                # Calculate percentage difference (negative = FP32 faster, positive = FP32 slower)
                if bf16_time > 0:
                    pct_diff = ((fp32_time - bf16_time) / bf16_time) * 100
                    pct_str = f"{pct_diff:+.1f}%"
                else:
                    pct_str = "N/A"

                print(f"{degree:<12} {bf16_time:>12.2f} {fp32_time:>12.2f} {pct_str:>12}")
            print()

# Plot 4: Impact of segmentation on error (compact)
if 'segmentation' in data.columns and len(data['segmentation'].unique()) > 1:
    segmentation_stats = data.groupby(['degree', 'segmentation'])[metric].mean().unstack(fill_value=0)

    if not segmentation_stats.empty and len(segmentation_stats.columns) > 1:
        print("="*80)
        print(f"Segmentation: Uniform vs Adaptive {metric_display}")
        print("="*80)
        print(f"{'Degree':<12} {'Uniform':>12} {'Adaptive':>12} {'Δ%':>12}")
        print("-" * 50)

        for degree in segmentation_stats.index:
            uniform_val = segmentation_stats.loc[degree, 'uniform'] if 'uniform' in segmentation_stats.columns else 0
            adaptive_val = segmentation_stats.loc[degree, 'adaptive'] if 'adaptive' in segmentation_stats.columns else 0

            if uniform_val > 0:
                improvement = ((uniform_val - adaptive_val) / uniform_val * 100)
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"

            print(f"{degree:<12} {uniform_val:>12.6f} {adaptive_val:>12.6f} {improvement_str:>12}")
        print()

# Summary statistics (compact)
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Find best configuration
best_idx = data[metric].idxmin()
best_row = data.loc[best_idx]

# Compact two-column layout
print(f"\n{'Runs:':<20} {len(data):>15}    {'Best ' + metric_display + ':':<20} {best_row[metric]:>15.6f}")
print(f"{'Activations:':<20} {len(data['activation'].unique()):>15}    {'Config:':<20} {best_row['degree']}/{best_row['depth']}/{best_row['segmentation'][:3]}/{best_row['precision']}")
print(f"{'Degrees:':<20} {len(data['degree'].unique()):>15}    {metric_display + ' Range:':<20} [{data[metric].min():.6f}, {data[metric].max():.6f}]")

# If comparing CB-based vs Embedded, show comparison summary
if include_embedded and 'source' in data.columns:
    cb_data = data[data['source'] == 'cb-based']
    emb_data = data[data['source'] == 'embedded']

    if not cb_data.empty and not emb_data.empty:
        print("\n" + "="*80)
        print("CB-BASED vs EMBEDDED COMPARISON")
        print("="*80)

        print(f"\n{'Source':<15} {'Runs':<10} {'Mean ' + metric_display:<15} {'Min ' + metric_display:<15} {'Max ' + metric_display:<15}")
        print("-" * 80)
        print(f"{'CB-based':<15} {len(cb_data):<10} {cb_data[metric].mean():<15.6f} {cb_data[metric].min():<15.6f} {cb_data[metric].max():<15.6f}")
        print(f"{'Embedded':<15} {len(emb_data):<10} {emb_data[metric].mean():<15.6f} {emb_data[metric].min():<15.6f} {emb_data[metric].max():<15.6f}")

        # Calculate improvement
        if cb_data[metric].mean() > 0:
            improvement = ((cb_data[metric].mean() - emb_data[metric].mean()) / cb_data[metric].mean()) * 100
            if improvement > 0:
                print(f"\n✓ Embedded is {improvement:.1f}% better (lower {metric_display})")
            else:
                print(f"\n⚠️  CB-based is {-improvement:.1f}% better (lower {metric_display})")

        # Show per-degree comparison
        print("\n" + "-" * 80)
        print(f"{'Degree':<12} {'CB-based':>15} {'Embedded':>15} {'Δ%':>12}")
        print("-" * 80)

        for degree in ['constant', 'linear', 'quadratic', 'cubic', 'hexic', 'octic']:
            cb_degree = cb_data[cb_data['degree'] == degree]
            emb_degree = emb_data[emb_data['degree'] == degree]

            if not cb_degree.empty and not emb_degree.empty:
                cb_mean = cb_degree[metric].mean()
                emb_mean = emb_degree[metric].mean()

                if cb_mean > 0:
                    pct_diff = ((cb_mean - emb_mean) / cb_mean) * 100
                    pct_str = f"{pct_diff:+.1f}%"
                else:
                    pct_str = "N/A"

                print(f"{degree:<12} {cb_mean:>15.6f} {emb_mean:>15.6f} {pct_str:>12}")

print("\n" + "="*80)
print("✓ Visualization complete!")
print("="*80 + "\n")

PYTHON_SCRIPT

echo ""
echo "To generate plots with different parameters:"
echo "  $0 --activation <name> --segmentation <type> --metric <mae|rmse|max_error>"
echo "  $0 --activation <name> --include-embedded  # Compare CB-based vs Embedded"
echo "  $0 --activation <name> --precision <bf16|fp32> --show-theoretical"
