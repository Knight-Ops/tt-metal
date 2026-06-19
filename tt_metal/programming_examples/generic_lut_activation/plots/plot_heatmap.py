#!/usr/bin/env python3
"""
Generate Pareto frontier plot showing polynomial degree vs depth tradeoffs.

2x2 subplot grid (one subplot per depth: 4, 8, 16, 32):
Each subplot has dual Y-axes:
- X-axis: Polynomial Degree (Const, Linear, Quad, Cubic)
- Left Y-axis: MAE (log scale, blue) - solid line with circle markers
  - Points annotated with memory usage (coefficients)
- Right Y-axis: Runtime (ms, red) - dashed line with square markers
  - Points annotated with runtime values

Cleaner than single plot with all depths overlaid.
Shows clear progression: Constant → Linear → Quadratic → Cubic for each depth.
Helps answer: "What polynomial degree should I use for my accuracy/speed needs at depth X?"
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import os
import argparse
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# Timing metrics to plot (5 profiler metrics configurations)
TIMING_METRICS = [
    'single_tile',
    '8_tiles',
    '256_tiles',
    'height_sharded',
    'yolov4'
]

METRIC_LABELS = [
    'Single Tile',
    '8 Tiles',
    '256 Tiles',
    'Height Sharded',
    'YOLOv4'
]


def pivot_row_per_shape_to_columns(df):
    """Transform row-per-shape format to multi-column format.

    Input format (one row per shape):
        activation, precision, depth, degree, segmentation, shape, time_profiler_us, mae, max_error, max_ulp

    Output format (one row per config, multiple shape columns):
        activation, precision, depth, degree, segmentation, mae_single_tile, mae_yolov4, ...
    """
    if 'shape' not in df.columns:
        return df  # Already in multi-column format

    # Group by config columns and pivot shape data
    config_cols = ['activation', 'precision', 'depth', 'degree', 'segmentation']
    config_cols = [c for c in config_cols if c in df.columns]

    # Metrics to pivot
    metrics = ['time_profiler_us', 'mae', 'rmse', 'max_error', 'mean_rel_error',
               'max_ulp', 'mean_ulp', 'median_ulp', 'p99_ulp',
               'max_ulp_error', 'mean_ulp_error', 'median_ulp_error', 'p99_ulp_error']
    metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        return df

    # Create pivoted dataframe
    result_dfs = []
    for shape in df['shape'].unique():
        shape_df = df[df['shape'] == shape].copy()
        # Rename metric columns with shape suffix
        for metric in metrics:
            if metric in shape_df.columns:
                new_name = f'{metric}_{shape}' if metric != 'time_profiler_us' else f'time_profiler_{shape}_us'
                shape_df = shape_df.rename(columns={metric: new_name})
        shape_df = shape_df.drop(columns=['shape'], errors='ignore')
        result_dfs.append(shape_df)

    if not result_dfs:
        return df

    # Merge all shape dataframes on config columns
    result = result_dfs[0]
    for shape_df in result_dfs[1:]:
        result = result.merge(shape_df, on=config_cols, how='outer', suffixes=('', '_dup'))
        # Drop duplicate columns
        result = result[[c for c in result.columns if not c.endswith('_dup')]]

    return result


# Tufte-inspired minimalist style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['grid.linewidth'] = 0.5

def compute_memory_usage(method, depth):
    """Compute total memory usage (number of float32 coefficients) for a method+depth combo."""
    if 'constant' in method:
        return 1024  # Always 1024-entry LUT
    elif 'linear' in method:
        return depth * 2  # 2 coefficients per segment (slope, offset)
    elif 'quadratic' in method:
        return depth * 3  # 3 coefficients per segment (a, b, c)
    elif 'cubic' in method:
        return depth * 4  # 4 coefficients per segment (a, b, c, d)
    elif 'hexic' in method:
        return depth * 7  # 7 coefficients per segment (a6...a0)
    elif 'octic' in method:
        return depth * 9  # 9 coefficients per segment (a8...a0)
    return 0

def get_polynomial_degree(method):
    """Get polynomial degree for a method."""
    if 'constant' in method:
        return 0
    elif 'linear' in method:
        return 1
    elif 'quadratic' in method:
        return 2
    elif 'cubic' in method:
        return 3
    elif 'hexic' in method:
        return 6
    elif 'octic' in method:
        return 8
    return -1

def load_results(arch_prefix=None):
    """Load all result CSVs for an architecture from new data structure.

    New format: data/{arch}/polynomial/{activation}.csv
    """
    if not arch_prefix:
        return None

    # Load from new directory structure
    polynomial_dir = f"data/{arch_prefix}/polynomial"

    if not os.path.exists(polynomial_dir):
        print(f"ERROR: Directory not found: {polynomial_dir}")
        return None

    # Find all activation CSV files
    activation_files = [f for f in os.listdir(polynomial_dir) if f.endswith('.csv')]

    if not activation_files:
        print(f"ERROR: No CSV files found in {polynomial_dir}")
        return None

    results = []

    print(f"Loading polynomial results from {len(activation_files)} activation files...")

    for csv_file in activation_files:
        csv_path = os.path.join(polynomial_dir, csv_file)
        activation_name = csv_file[:-4]  # Remove .csv extension

        df = pd.read_csv(csv_path)
        df = df[df['status'] == 'pass']  # Only successful runs

        # Add activation column if not present
        if 'activation' not in df.columns:
            df['activation'] = activation_name

        # Transform row-per-shape format to multi-column format if needed
        df = pivot_row_per_shape_to_columns(df)

        # Rename 'degree' to 'poly_degree' for consistency
        if 'degree' in df.columns:
            df['poly_degree'] = df['degree']

        # Compute memory coefficients based on degree and depth
        # Generic LUT format: degree 0=constant (1024), 1=linear (depth*2), 2=quad (depth*3), 3=cubic (depth*4), etc
        df['memory_coeffs'] = df.apply(
            lambda row: 1024 if row['poly_degree'] == 0 else row['depth'] * (row['poly_degree'] + 1),
            axis=1
        )

        # Add runtime column - use single_tile timing as baseline runtime
        # Convert from microseconds to milliseconds
        if 'time_profiler_single_tile_us' in df.columns:
            df['runtime_us'] = df['time_profiler_single_tile_us']

        # Add unified error metrics using yolov4 as baseline (single_tile often has degenerate near-zero values)
        # Prefer yolov4 > 256_tiles > height_sharded > single_tile
        for shape_preference in ['yolov4', '256_tiles', 'height_sharded', 'single_tile']:
            mae_col = f'mae_{shape_preference}'
            if mae_col in df.columns and 'mae' not in df.columns:
                df['mae'] = df[mae_col]
            max_error_col = f'max_error_{shape_preference}'
            if max_error_col in df.columns and 'max_error' not in df.columns:
                df['max_error'] = df[max_error_col]
            # Handle both column naming conventions for ULP
            for ulp_prefix in ['max_ulp_error', 'max_ulp']:
                ulp_col = f'{ulp_prefix}_{shape_preference}'
                if ulp_col in df.columns and 'max_ulp_error' not in df.columns:
                    df['max_ulp_error'] = df[ulp_col]
            for ulp_prefix in ['mean_ulp_error', 'mean_ulp']:
                ulp_col = f'{ulp_prefix}_{shape_preference}'
                if ulp_col in df.columns and 'mean_ulp_error' not in df.columns:
                    df['mean_ulp_error'] = df[ulp_col]

        results.append(df)

    if not results:
        return None

    combined_df = pd.concat(results, ignore_index=True)

    # Exclude hardshrink and softshrink from pareto analysis
    combined_df = combined_df[~combined_df['activation'].isin(['hardshrink', 'softshrink'])]

    # Cap invalid ULP values (overflow/NaN artifacts from ULP calculation bugs)
    ULP_CAP = 1e10
    ulp_cols = [col for col in combined_df.columns if 'ulp' in col.lower()]
    for col in ulp_cols:
        invalid_count = (combined_df[col] > ULP_CAP).sum()
        if invalid_count > 0:
            print(f"  Capping {invalid_count} invalid ULP values in '{col}' (>{ULP_CAP:.0e})")
            combined_df[col] = combined_df[col].clip(upper=ULP_CAP)

    print(f"Loaded {len(combined_df)} measurements across {len(combined_df['activation'].unique())} activations")

    return combined_df

def find_cheapest_best(heatmap_pivot, heatmap_values):
    """Find the cheapest (smallest depth) cell that achieves the best (minimum) error.

    Returns:
        Tuple of (row_idx, col_idx) for the cell to highlight, or None if not found.
    """
    if heatmap_values.size == 0:
        return None

    # Find minimum error value
    min_error = np.nanmin(heatmap_values)

    # Find all cells that achieve this minimum
    min_indices = np.argwhere(heatmap_values == min_error)

    if len(min_indices) == 0:
        return None

    # Among cells with minimum error, find the one with smallest depth (leftmost column)
    # heatmap_pivot columns are depths, so we want the cell with smallest column index
    cheapest = min(min_indices, key=lambda idx: heatmap_pivot.columns[idx[1]])

    return tuple(cheapest)

def plot_depth_degree_heatmap_per_activation(df, arch_prefix=None, output_dir='plots_heatmap', metric='mae'):
    """Generate separate heatmap for each activation showing error values across depth and polynomial degree."""
    # Get metric configuration
    config = get_metric_config(metric)
    metric_name = config['metric_name']
    metric_filename = config['metric_filename']
    use_log = config['use_log']

    # Generate heatmaps for all activations
    activations = sorted(df['activation'].unique())

    print(f"Generating depth-degree heatmaps ({metric}) for all activations: {len(activations)}")

    for activation in activations:
        # Create per-activation subdirectory
        activation_output_dir = f"{output_dir}/per_activation/{activation}"
        os.makedirs(activation_output_dir, exist_ok=True)

        activation_df = df[df['activation'] == activation]

        # Skip if metric column doesn't exist
        if metric not in activation_df.columns:
            print(f"  Warning: {metric} column not found for {activation}, skipping heatmap")
            continue

        # Compute minimum error for each (depth, degree) combination for this activation
        heatmap_data = activation_df.groupby(['depth', 'poly_degree']).agg({
            metric: 'min'
        }).reset_index()

        # Skip if no data
        if len(heatmap_data) == 0:
            continue

        # Pivot to create matrix for heatmap
        heatmap_pivot = heatmap_data.pivot(index='poly_degree', columns='depth', values=metric)

        # Prepare values for visualization (log or linear scale)
        heatmap_values = heatmap_pivot.values
        if use_log:
            heatmap_values_for_display = np.where(heatmap_values == 0, 1e-10, heatmap_values)
            display_values = np.log10(heatmap_values_for_display)
        else:
            display_values = heatmap_values

        # Create figure with GridSpec for proper sizing
        num_cols = heatmap_pivot.shape[1]
        num_rows = heatmap_pivot.shape[0]

        # Adjust figure height based on number of rows
        fig_height = min(12.0, max(4.0, num_rows * 0.35))
        fig = plt.figure(figsize=(8.0, fig_height))
        gs = gridspec.GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0, 0])

        # Aspect ratio adjustment for readability
        if num_rows > 15:
            aspect_ratio = num_cols / (num_rows * 0.8)
        elif num_rows > 10:
            aspect_ratio = num_cols / (num_rows * 1.0)
        elif num_rows > 6:
            aspect_ratio = num_cols / (num_rows * 1.2)
        else:
            aspect_ratio = 'auto'

        # Create heatmap - scale colors based on actual data range (green=low/good, red=high/bad)
        vmin = np.nanmin(display_values)
        vmax = np.nanmax(display_values)
        im = ax.imshow(display_values, cmap='RdYlGn_r', aspect=aspect_ratio, interpolation='nearest',
                       origin='lower', vmin=vmin, vmax=vmax)

        # Find cheapest best configuration
        cheapest_best_idx = find_cheapest_best(heatmap_pivot, heatmap_values)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(heatmap_pivot.columns)))
        ax.set_yticks(np.arange(len(heatmap_pivot.index)))
        ax.set_xticklabels(heatmap_pivot.columns, fontsize=11)

        # Format Y-axis labels with dynamic font sizing
        degree_to_label = {i: str(i) for i in range(17)}
        degree_labels = [degree_to_label.get(deg, str(deg)) for deg in heatmap_pivot.index]
        num_labels = len(degree_labels)
        if num_labels > 20:
            label_fontsize = 9
        elif num_labels > 15:
            label_fontsize = 10
        elif num_labels > 10:
            label_fontsize = 11
        elif num_labels > 6:
            label_fontsize = 12
        else:
            label_fontsize = 13
        ax.set_yticklabels(degree_labels, fontsize=label_fontsize)

        # Labels
        ax.set_xlabel('Depth', fontsize=13, fontweight='bold')
        ax.set_ylabel('Polynomial Degree', fontsize=13, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar_label = f'{metric_name} (log10)' if use_log else metric_name
        cbar.set_label(cbar_label, fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        # Annotate each cell with scientific notation - polynomial-fitter style
        if num_rows > 20:
            annotation_fontsize = 8
        elif num_rows > 15:
            annotation_fontsize = 9
        elif num_rows > 10:
            annotation_fontsize = 10
        elif num_rows > 6:
            annotation_fontsize = 11
        else:
            annotation_fontsize = 12

        is_ulp_metric = 'ulp' in metric.lower()

        for i in range(len(heatmap_pivot.index)):
            for j in range(len(heatmap_pivot.columns)):
                error_value = heatmap_values[i, j]
                if error_value == 0:
                    text = '0'
                elif is_ulp_metric and abs(error_value) < 10000:
                    # ULP metrics: show as integer for small values
                    if abs(error_value) >= 1:
                        text = f'{error_value:.0f}'
                    else:
                        text = f'{error_value:.1f}'
                else:
                    # MAE/max_error or large ULP: use scientific notation with superscript exponent
                    exp_str = f'{error_value:.1e}'
                    if 'e' in exp_str:
                        mantissa, exp = exp_str.split('e')
                        text = f'{mantissa}e$^{{{exp}}}$'
                    else:
                        text = exp_str

                # Use black text directly on cell
                ax.text(j, i, text, ha='center', va='center',
                       color='black', fontsize=annotation_fontsize)

        # Draw prominent black box around cheapest best configuration
        if cheapest_best_idx is not None:
            i, j = cheapest_best_idx
            # Rectangle centered on cell (j, i) with width/height 1
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     linewidth=3, edgecolor='black', facecolor='none',
                                     zorder=10)
            ax.add_patch(rect)

        # No title (Tufte minimalist style)

        plt.tight_layout()

        # Save plot in per-activation subdirectory
        filename = f'{activation_output_dir}/{activation}_heatmap_{metric_filename}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()

def plot_depth_degree_heatmap(df, arch_prefix=None, output_dir='plots_heatmap', metric='mae'):
    """Generate combined heatmap showing hit count of how many activations selected each (depth, degree) as their best configuration."""
    # Get metric configuration
    config = get_metric_config(metric)
    metric_dir = config['metric_dir']

    all_activations_dir = f"{output_dir}/all_activations/{metric_dir}"
    os.makedirs(all_activations_dir, exist_ok=True)

    # Count how many activations selected each (depth, degree) as their "best" (cheapest best) configuration
    hit_counts = {}

    # For each activation, find its cheapest best configuration
    for activation in df['activation'].unique():
        activation_df = df[df['activation'] == activation]

        # Group by (depth, poly_degree) and find best MAE for each
        grouped = activation_df.groupby(['depth', 'poly_degree']).agg({'mae': 'min'}).reset_index()
        grouped_pivot = grouped.pivot(index='poly_degree', columns='depth', values='mae')

        # Find cheapest best configuration
        best_idx = find_cheapest_best(grouped_pivot, grouped_pivot.values)

        if best_idx is not None:
            i, j = best_idx
            degree = grouped_pivot.index[i]
            depth = grouped_pivot.columns[j]
            key = (depth, degree)
            hit_counts[key] = hit_counts.get(key, 0) + 1

    # Get all possible (depth, degree) combinations from the data
    all_combinations = df.groupby(['depth', 'poly_degree']).size().reset_index()[['depth', 'poly_degree']]

    # Create matrix with hit counts
    heatmap_data = []
    for _, row in all_combinations.iterrows():
        depth = row['depth']
        degree = row['poly_degree']
        count = hit_counts.get((depth, degree), 0)
        heatmap_data.append({'depth': depth, 'poly_degree': degree, 'hit_count': count})

    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_pivot = heatmap_df.pivot(index='poly_degree', columns='depth', values='hit_count').fillna(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Get hit count values
    heatmap_values = heatmap_pivot.values

    # Create heatmap - use Blues colormap (more hits = darker blue)
    max_hits = int(heatmap_values.max())
    im = ax.imshow(heatmap_values, cmap='Blues', aspect='auto', interpolation='nearest',
                   origin='lower', vmin=0, vmax=max(max_hits, 1))

    # Find most popular configuration (highest hit count)
    most_popular_idx = None
    if max_hits > 0:
        max_positions = np.argwhere(heatmap_values == max_hits)
        if len(max_positions) > 0:
            # Pick the one with lowest depth (leftmost)
            max_positions = sorted(max_positions, key=lambda x: (x[1], x[0]))
            most_popular_idx = tuple(max_positions[0])

    # Set ticks and labels
    ax.set_xticks(np.arange(len(heatmap_pivot.columns)))
    ax.set_yticks(np.arange(len(heatmap_pivot.index)))
    ax.set_xticklabels(heatmap_pivot.columns, fontsize=11)

    # Format Y-axis labels
    degree_to_label = {i: str(i) for i in range(17)}
    degree_labels = [degree_to_label.get(int(deg), str(int(deg))) for deg in heatmap_pivot.index]
    ax.set_yticklabels(degree_labels, fontsize=11)

    # Labels
    ax.set_xlabel('Depth', fontsize=13, fontweight='bold')
    ax.set_ylabel('Polynomial Degree', fontsize=13, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Hit Count', fontsize=11, fontweight='bold')

    # Annotate each cell with the hit count
    for i in range(len(heatmap_pivot.index)):
        for j in range(len(heatmap_pivot.columns)):
            count = int(heatmap_values[i, j])
            text = str(count) if count > 0 else ''

            # Use white text on dark cells, black on light cells
            text_color = 'white' if count > max_hits / 2 else 'black'

            ax.text(j, i, text, ha='center', va='center',
                   color=text_color, fontsize=10, fontweight='bold')

    # Draw prominent black box around most popular configuration
    if most_popular_idx is not None:
        i, j = most_popular_idx
        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 linewidth=3, edgecolor='black', facecolor='none',
                                 zorder=10)
        ax.add_patch(rect)

    # No title (Tufte minimalist style)

    plt.tight_layout()

    # Save plot in all_activations subdirectory
    filename = f'{all_activations_dir}/all_activations_heatmap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def load_native_sfpu_results(arch_prefix):
    """Load native SFPU results from new directory structure."""
    sfpu_dir = f'../generic_lut_activation/data/{arch_prefix}/native_sfpu'
    # Try alternative path if running from parent directory
    if not os.path.exists(sfpu_dir):
        sfpu_dir = f'data/{arch_prefix}/native_sfpu'

    if not os.path.exists(sfpu_dir):
        return None

    # Find all activation CSV files
    activation_files = [f for f in os.listdir(sfpu_dir) if f.endswith('.csv')]

    if not activation_files:
        return None

    results = []

    for csv_file in activation_files:
        csv_path = os.path.join(sfpu_dir, csv_file)
        activation_name = csv_file[:-4]  # Remove .csv extension

        df = pd.read_csv(csv_path)
        df = df[df['status'] == 'pass']  # Only successful runs

        # Filter to fp32 precision only to match LUT results
        if 'precision' in df.columns:
            df = df[df['precision'] == 'fp32']

        # Add activation column if not present
        if 'activation' not in df.columns:
            df['activation'] = activation_name

        # Row-per-shape format: pivot to multi-column for pareto analysis
        if 'shape' in df.columns:
            df = pivot_row_per_shape_to_columns(df)

        # Add runtime column - use single_tile timing as baseline
        if 'time_profiler_single_tile_us' in df.columns:
            df['runtime_us'] = df['time_profiler_single_tile_us']

        # Add unified error metrics using yolov4 as baseline (single_tile often has degenerate near-zero values)
        for shape_preference in ['yolov4', '256_tiles', 'height_sharded', 'single_tile']:
            mae_col = f'mae_{shape_preference}'
            if mae_col in df.columns and 'mae' not in df.columns:
                df['mae'] = df[mae_col]
            max_error_col = f'max_error_{shape_preference}'
            if max_error_col in df.columns and 'max_error' not in df.columns:
                df['max_error'] = df[max_error_col]
            for ulp_prefix in ['max_ulp_error', 'max_ulp']:
                ulp_col = f'{ulp_prefix}_{shape_preference}'
                if ulp_col in df.columns and 'max_ulp_error' not in df.columns:
                    df['max_ulp_error'] = df[ulp_col]
            for ulp_prefix in ['mean_ulp_error', 'mean_ulp']:
                ulp_col = f'{ulp_prefix}_{shape_preference}'
                if ulp_col in df.columns and 'mean_ulp_error' not in df.columns:
                    df['mean_ulp_error'] = df[ulp_col]

        results.append(df)

    if not results:
        return None

    combined_df = pd.concat(results, ignore_index=True)

    # Exclude hardshrink and softshrink
    combined_df = combined_df[~combined_df['activation'].isin(['hardshrink', 'softshrink'])]

    # Cap invalid ULP values
    ULP_CAP = 1e10
    ulp_cols = [col for col in combined_df.columns if 'ulp' in col.lower()]
    for col in ulp_cols:
        combined_df[col] = combined_df[col].clip(upper=ULP_CAP)

    return combined_df

def load_rational_results(arch_prefix):
    """Load rational approximation results from new directory structure."""
    rational_dir = f'data/{arch_prefix}/rational'

    if not os.path.exists(rational_dir):
        return None

    # Find all activation CSV files
    activation_files = [f for f in os.listdir(rational_dir) if f.endswith('.csv')]

    if not activation_files:
        return None

    results = []

    print(f"Loading rational results from {len(activation_files)} activation files...")

    for csv_file in activation_files:
        csv_path = os.path.join(rational_dir, csv_file)
        activation_name = csv_file[:-4]  # Remove .csv extension

        df = pd.read_csv(csv_path)
        df = df[df['status'] == 'pass']  # Only successful runs

        # Add activation column if not present
        if 'activation' not in df.columns:
            df['activation'] = activation_name

        # Transform row-per-shape format to multi-column format if needed
        df = pivot_row_per_shape_to_columns(df)

        # Compute rational degree as sum of numerator and denominator degrees
        if 'num_degree' in df.columns and 'den_degree' in df.columns:
            df['rational_degree'] = df['num_degree'] + df['den_degree']

        # Add runtime column - use single_tile timing as baseline
        if 'time_profiler_single_tile_us' in df.columns:
            df['runtime_us'] = df['time_profiler_single_tile_us']

        # Add unified error metrics using yolov4 as baseline (single_tile often has degenerate near-zero values)
        for shape_preference in ['yolov4', '256_tiles', 'height_sharded', 'single_tile']:
            mae_col = f'mae_{shape_preference}'
            if mae_col in df.columns and 'mae' not in df.columns:
                df['mae'] = df[mae_col]
            max_error_col = f'max_error_{shape_preference}'
            if max_error_col in df.columns and 'max_error' not in df.columns:
                df['max_error'] = df[max_error_col]
            for ulp_prefix in ['max_ulp_error', 'max_ulp']:
                ulp_col = f'{ulp_prefix}_{shape_preference}'
                if ulp_col in df.columns and 'max_ulp_error' not in df.columns:
                    df['max_ulp_error'] = df[ulp_col]
            for ulp_prefix in ['mean_ulp_error', 'mean_ulp']:
                ulp_col = f'{ulp_prefix}_{shape_preference}'
                if ulp_col in df.columns and 'mean_ulp_error' not in df.columns:
                    df['mean_ulp_error'] = df[ulp_col]

        results.append(df)

    if not results:
        return None

    combined_df = pd.concat(results, ignore_index=True)

    # Exclude hardshrink and softshrink
    combined_df = combined_df[~combined_df['activation'].isin(['hardshrink', 'softshrink'])]

    # Cap invalid ULP values
    ULP_CAP = 1e10
    ulp_cols = [col for col in combined_df.columns if 'ulp' in col.lower()]
    for col in ulp_cols:
        combined_df[col] = combined_df[col].clip(upper=ULP_CAP)

    print(f"Loaded {len(combined_df)} rational measurements")

    return combined_df

def get_metric_config(metric):
    """Get metric-specific configuration for plotting.

    Returns a dict with:
        - metric_name: Display name (e.g., 'MAE', 'Max ULP')
        - metric_base: Column name base (e.g., 'mae', 'max_ulp_error')
        - metric_dir: Directory name (e.g., 'mae', 'ulp_max')
        - metric_filename: Filename component (e.g., 'mae', 'ulp_max')
        - default_min: Default Y-axis minimum
        - default_max: Default Y-axis maximum
        - min_clip: Minimum value for clipping
        - show_break_indicator: Whether to show axis break zigzag
        - break_threshold: Y-value threshold for showing clipping label
        - use_log: Whether to use log scale for Y-axis
    """
    is_ulp = metric in ['max_ulp_error', 'mean_ulp_error']

    if metric == 'mae':
        return {
            'metric_name': 'MAE',
            'metric_base': 'mae',
            'metric_dir': 'mae',
            'metric_filename': 'mae',
            'default_min': 1e-7,
            'default_max': 1e-3,
            'min_clip': 1e-7,
            'show_break_indicator': True,
            'break_threshold': 1e-7 * 1.5,
            'use_log': True
        }
    elif metric == 'max_error':
        return {
            'metric_name': 'Max Error',
            'metric_base': 'max_error',
            'metric_dir': 'max',
            'metric_filename': 'max',
            'default_min': 1e-7,
            'default_max': 1e-3,
            'min_clip': 1e-7,
            'show_break_indicator': True,
            'break_threshold': 1e-7 * 1.5,
            'use_log': True
        }
    elif metric == 'max_ulp_error':
        return {
            'metric_name': 'Max ULP',
            'metric_base': 'max_ulp_error',
            'metric_dir': 'ulp_max',
            'metric_filename': 'ulp_max',
            'default_min': 1,
            'default_max': 1e6,
            'min_clip': 1,
            'show_break_indicator': False,
            'break_threshold': None,
            'use_log': True  # Log scale for ULP
        }
    elif metric == 'mean_ulp_error':
        return {
            'metric_name': 'Mean ULP',
            'metric_base': 'mean_ulp_error',
            'metric_dir': 'ulp_mean',
            'metric_filename': 'ulp_mean',
            'default_min': 1,
            'default_max': 1e6,
            'min_clip': 1,
            'show_break_indicator': False,
            'break_threshold': None,
            'use_log': True  # Log scale for ULP
        }
    else:
        # Fallback for unknown metrics
        return {
            'metric_name': metric.upper(),
            'metric_base': metric,
            'metric_dir': metric,
            'metric_filename': metric,
            'default_min': 1e-7,
            'default_max': 1e-3,
            'min_clip': 1e-7,
            'show_break_indicator': False,
            'break_threshold': None,
            'use_log': True
        }

def plot_pareto_frontier(df, arch_prefix=None, output_dir='plots_heatmap', activations_filter=None, show_native_sfpu=True, filename_suffix='', metric='mae', timing_config='single_tile'):
    """Generate Pareto frontier plot for a specific timing configuration with subplots by depth, each with dual Y-axes.

    Args:
        df: DataFrame with piecewise method results
        arch_prefix: Architecture prefix (e.g., 'blackhole', 'wormhole')
        output_dir: Output directory for plots
        activations_filter: List of activations to include (None = all)
        show_native_sfpu: Whether to show Native SFPU baseline
        filename_suffix: Suffix to add to output filename
        metric: Error metric to plot ('mae', 'max_error', 'max_ulp_error', 'mean_ulp_error')
        timing_config: Timing configuration to plot ('single_tile', '8_tiles', '256_tiles', 'height_sharded', 'yolov4')
    """
    # Get metric configuration (unified for all metrics)
    config = get_metric_config(metric)
    metric_name = config['metric_name']
    metric_base = config['metric_base']
    metric_dir = config['metric_dir']
    metric_filename = config['metric_filename']
    default_min = config['default_min']
    default_max = config['default_max']
    min_clip = config['min_clip']
    show_break_indicator = config['show_break_indicator']
    break_threshold = config['break_threshold']
    use_log = config['use_log']

    all_activations_dir = f"{output_dir}/all_activations/{metric_dir}"
    os.makedirs(all_activations_dir, exist_ok=True)

    # Get metric and runtime column names for this timing configuration
    metric_col = f'{metric_base}_{timing_config}'
    runtime_col = f'time_profiler_{timing_config}_us'

    # Check if columns exist in dataframe
    if metric_col not in df.columns:
        print(f"Warning: Column '{metric_col}' not found in dataframe. Skipping {timing_config}.")
        return
    if runtime_col not in df.columns:
        print(f"Warning: Column '{runtime_col}' not found in dataframe. Skipping {timing_config}.")
        return

    # Create temporary columns with standard names for processing
    df['metric_value'] = df[metric_col]
    df['runtime_us'] = df[runtime_col] / 1000.0  # Convert microseconds to milliseconds

    # Load rational approximation data
    rational_df = None
    if arch_prefix:
        rational_df = load_rational_results(arch_prefix)
        if rational_df is not None:
            print(f"Loaded {len(rational_df)} rational approximation measurements")
            # Add metric and runtime columns for rational data
            if metric_col in rational_df.columns and runtime_col in rational_df.columns:
                rational_df['metric_value'] = rational_df[metric_col]
                rational_df['runtime_us'] = rational_df[runtime_col] 
    # Filter activations if requested
    if activations_filter is not None:
        df = df[df['activation'].isin(activations_filter)]
        if df.empty:
            print(f"Warning: No data after filtering for activations: {activations_filter}")
            return
        if rational_df is not None:
            rational_df = rational_df[rational_df['activation'].isin(activations_filter)]

    # Process data for each timing configuration
    data_by_timing = {}

    # Load native SFPU data for baseline
    native_sfpu_error = None
    native_sfpu_error_min = None
    native_sfpu_error_max = None
    native_sfpu_error_p25 = None
    native_sfpu_error_p75 = None
    native_sfpu_runtime = None
    native_sfpu_runtime_sem = None
    if show_native_sfpu:
        native_df = load_native_sfpu_results(arch_prefix)
        if native_df is not None and not native_df.empty:
            # Filter to same activations if activations_filter is provided
            if activations_filter is not None:
                native_df = native_df[native_df['activation'].isin(activations_filter)]
            if not native_df.empty:
                # Check if timing-specific columns exist for SFPU data
                if metric_col in native_df.columns and runtime_col in native_df.columns:
                    native_sfpu_error = native_df[metric_col].median()  # Use median for consistency with LUT data
                    native_sfpu_error_min = native_df[metric_col].min()  # Best TTNN performance
                    native_sfpu_error_max = native_df[metric_col].max()  # Worst TTNN performance
                    native_sfpu_error_p25 = native_df[metric_col].quantile(0.25)
                    native_sfpu_error_p75 = native_df[metric_col].quantile(0.75)
                    native_sfpu_runtime = (native_df[runtime_col] / 1000.0).median()  # Convert to ms
                    native_sfpu_runtime_sem = (native_df[runtime_col] / 1000.0).std() / np.sqrt(len(native_df))
                    print(f"TTNN baseline ({timing_config}): {metric_name}={native_sfpu_error:.6f} (min={native_sfpu_error_min:.6f}, max={native_sfpu_error_max:.6f}), Runtime={native_sfpu_runtime:.2f}±{native_sfpu_runtime_sem:.2f}ms")

    # Compute BEST metric for each (activation, degree, depth) by taking min across segmentation types
    # This represents optimal choice of uniform vs adaptive for each activation
    best_per_activation = df.groupby(['activation', 'poly_degree', 'depth']).agg({
        'metric_value': 'min',  # Best metric for this activation at this (degree, depth)
        'runtime_us': 'min',  # Best runtime for this activation
        'memory_coeffs': 'first'
    }).reset_index()

    # Now compute statistics across activations for each (degree, depth) combination
    # This shows: "If I optimally pick uniform/adaptive for each activation, what's the median metric?"
    # Use MEDIAN instead of MEAN because outliers (exp, cosh, sinh) skew the mean
    # Also include MIN and MAX to show best/worst achievable performance
    summary = best_per_activation.groupby(['poly_degree', 'depth']).agg({
        'metric_value': ['median', 'std', 'count', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), 'min', 'max'],
        'runtime_us': ['median', 'std'],
        'memory_coeffs': 'first'
    }).reset_index()

    # Flatten multi-level column names
    summary.columns = ['poly_degree', 'depth', 'metric_median', 'metric_std', 'metric_count', 'metric_p25', 'metric_p75', 'metric_min', 'metric_max',
                       'runtime_median', 'runtime_std', 'memory_coeffs']

    # Rename for backwards compatibility (now using median, not mean)
    summary['metric_value'] = summary['metric_median']
    summary['runtime_us'] = summary['runtime_median']

    # Use available depths (exclude 32 - too slow)
    depths = sorted([d for d in summary['depth'].unique() if d < 32])

    # Filter summary to only include depths we're plotting
    summary_filtered = summary[summary['depth'].isin(depths)]

    # Get all unique polynomial degrees across all depths for consistent x-axis
    all_poly_degrees = sorted(summary_filtered['poly_degree'].unique())

    # Process rational data if available
    rational_summary = None
    if rational_df is not None and not rational_df.empty:
        # Group rational data by segments (similar to depth for piecewise)
        # Include num_degree and den_degree for sorting
        best_rational_per_activation = rational_df.groupby(['activation', 'num_degree', 'den_degree', 'rational_degree', 'segments']).agg({
            'metric_value': 'min',
            'runtime_us': 'min'
        }).reset_index()

        rational_summary = best_rational_per_activation.groupby(['num_degree', 'den_degree', 'rational_degree', 'segments']).agg({
            'metric_value': ['median', 'std', 'count', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), 'min', 'max'],
            'runtime_us': ['median', 'std']
        }).reset_index()

        rational_summary.columns = ['num_degree', 'den_degree', 'rational_degree', 'segments', 'metric_median', 'metric_std', 'metric_count', 'metric_p25', 'metric_p75', 'metric_min', 'metric_max',
                                   'runtime_median', 'runtime_std']
        rational_summary['metric_value'] = rational_summary['metric_median']
        rational_summary['runtime_us'] = rational_summary['runtime_median']

        # Sort by num_degree first, then den_degree
        rational_summary = rational_summary.sort_values(['num_degree', 'den_degree', 'segments'])

        # Get unique segments for rational (similar to depths)
        rational_segments = sorted(rational_summary['segments'].unique())
        print(f"Rational segments: {rational_segments}")

    # Create subplot grid dynamically based on number of depths + rational subplots
    num_depths = len(depths)
    # Add 2 subplots for rational: one for num=den, one for num=den+1
    num_rational_subplots = 2 if rational_summary is not None else 0
    num_subplots = num_depths + num_rational_subplots
    # IEEE two-column format: each subplot ~2.8" wide, 2.8" tall with unified legend at top
    fig, axes = plt.subplots(1, num_subplots, figsize=(2.8 * num_subplots, 3.0))
    if num_subplots == 1:
        axes = [axes]  # Make it iterable
    else:
        axes = axes.flatten()

    # Compute global min/max for consistent axis ranges (include minimum and maximum metric values)
    # Use metric_min for lower bound and metric_max for upper bound to show full error scope
    # Filter out inf and NaN values to avoid invalid axis ranges
    metric_min_values = summary_filtered['metric_min'].replace([np.inf, -np.inf], np.nan).dropna()
    metric_max_values = summary_filtered['metric_max'].replace([np.inf, -np.inf], np.nan).dropna()

    # Use metric-specific defaults from config
    global_metric_min = metric_min_values.min() if len(metric_min_values) > 0 else default_min
    global_metric_max = metric_max_values.max() if len(metric_max_values) > 0 else default_max
    global_runtime_min = summary_filtered['runtime_us'].min()
    global_runtime_max = summary_filtered['runtime_us'].max()

    # Include rational data in global ranges
    if rational_summary is not None:
        rational_metric_min = rational_summary['metric_min'].replace([np.inf, -np.inf], np.nan).dropna()
        rational_metric_max = rational_summary['metric_max'].replace([np.inf, -np.inf], np.nan).dropna()

        if len(rational_metric_min) > 0:
            global_metric_min = min(global_metric_min, rational_metric_min.min())
        if len(rational_metric_max) > 0:
            global_metric_max = max(global_metric_max, rational_metric_max.max())
        global_runtime_min = min(global_runtime_min, rational_summary['runtime_us'].min())
        global_runtime_max = max(global_runtime_max, rational_summary['runtime_us'].max())

    # Include Native SFPU baseline in global ranges
    # This ensures SFPU baselines are visible in the plot
    if native_sfpu_error_min is not None and native_sfpu_error_max is not None:
        global_metric_min = min(global_metric_min, native_sfpu_error_min)
        global_metric_max = max(global_metric_max, native_sfpu_error_max)
    if native_sfpu_runtime is not None:
        global_runtime_min = min(global_runtime_min, native_sfpu_runtime)
        global_runtime_max = max(global_runtime_max, native_sfpu_runtime)

    # Add padding to ranges (20% on each side in log space for MAE, linear space for runtime)
    # Ensure lower bound is above zero for log scale (use metric-specific min_clip from config)
    metric_log_min = np.log10(max(global_metric_min, min_clip))  # Avoid log(0)
    metric_log_max = np.log10(max(global_metric_max, min_clip))  # Avoid log(0)

    # Ensure valid log values (not NaN or Inf)
    if not np.isfinite(metric_log_min):
        metric_log_min = -7
    if not np.isfinite(metric_log_max):
        metric_log_max = -7

    metric_log_range = metric_log_max - metric_log_min
    # Ensure valid ylim range (handle case where min == max or invalid range)
    if not np.isfinite(metric_log_range) or metric_log_range == 0:
        metric_ylim = (10**(metric_log_min - 1), 10**(metric_log_max + 1))
    else:
        metric_ylim = (10**(metric_log_min - 0.2 * metric_log_range), 10**(metric_log_max + 0.2 * metric_log_range))

    # Runtime uses linear scale
    runtime_range = global_runtime_max - global_runtime_min
    runtime_ylim = (global_runtime_min - 0.1 * runtime_range, global_runtime_max + 0.1 * runtime_range)

    # Plot each depth in its own subplot (organized by depth, x-axis = degree)
    for subplot_idx, depth in enumerate(depths):
        ax1 = axes[subplot_idx]
        ax2 = ax1.twinx()  # Create second Y-axis

        depth_df = summary[summary['depth'] == depth]

        # Sort by polynomial degree for proper plotting
        depth_df = depth_df.sort_values('poly_degree')

        # ========== LEFT Y-AXIS: Degree vs MAE (median across activations) ==========
        # Tufte-inspired muted blue
        ax1.plot(depth_df['poly_degree'], depth_df['metric_value'],
                color='#4A90E2', linewidth=2.5, alpha=1.0,
                marker='o', markersize=8, linestyle='-',
                markeredgewidth=1.5, markeredgecolor='white',
                label=f'LUT {metric_name} (median)' if subplot_idx == 0 else None)

        # Add translucent region showing full spread from best to worst (min to max)
        # This covers ALL activation metric values, showing the complete achievable range
        if 'metric_min' in depth_df.columns and 'metric_max' in depth_df.columns and not depth_df['metric_min'].isna().all():
            # Clip metric values for log scale plotting (use metric-specific min_clip from config)
            metric_min_clipped = depth_df['metric_min'].clip(lower=min_clip)
            ax1.fill_between(depth_df['poly_degree'],
                           metric_min_clipped,
                           depth_df['metric_max'],
                           color='#4A90E2', alpha=0.2, linewidth=0,
                           zorder=2, interpolate=True)

        # Scatter points removed for cleaner visualization
        # The blue shaded region already shows the spread across activations

        # ========== RIGHT Y-AXIS: Degree vs Runtime (dashed line) ==========
        # Tufte-inspired muted red
        ax2.plot(depth_df['poly_degree'], depth_df['runtime_us'],
                color='#E27A7A', linewidth=2.5, alpha=1.0,
                marker='s', markersize=7, linestyle='--',
                markeredgewidth=1.5, markeredgecolor='white',
                label='LUT Runtime' if subplot_idx == 0 else None)

        # SFPU scatter points removed for cleaner visualization
        # The green shaded region already shows the SFPU spread across activations

        # Add TTNN baseline lines if available - make them more visible with green color
        if native_sfpu_error is not None:
            # Only add labels on first subplot
            label_median = f'TTNN {metric_name}' if subplot_idx == 0 else None

            ax1.axhline(y=native_sfpu_error, color='#6AAA71', linestyle='--', linewidth=2.0,
                       alpha=0.85, label=label_median, zorder=3)
            # Add shaded region for TTNN showing full spread (min to max)
            # This covers ALL TTNN activations to match the LUT blue region
            if native_sfpu_error_min is not None and native_sfpu_error_max is not None:
                # Use 1e-7 instead of 0 to avoid log(0)
                lower_bound = 1e-7 if native_sfpu_error_min == 0 else max(native_sfpu_error_min, 1e-7)
                # Get the actual x-axis range from the data
                x_min = depth_df['poly_degree'].min() - 0.5
                x_max = depth_df['poly_degree'].max() + 0.5
                ax1.fill_between([x_min, x_max],
                               lower_bound,
                               native_sfpu_error_max,
                               color='#6AAA71', alpha=0.2, linewidth=0, zorder=2)
        if native_sfpu_runtime is not None:
            # Use dotted line for TTNN to distinguish from LUT dashed line (color-blind friendly)
            label_runtime = 'TTNN Runtime' if subplot_idx == 0 else None
            ax2.axhline(y=native_sfpu_runtime, color='#E27A7A', linestyle=':', linewidth=2.5,
                       alpha=0.85, label=label_runtime, zorder=3)

        # Configure axes - Tufte style with larger sizing for readability
        ax1.set_xlabel(f'Polynomial Degree (Seg={depth})', fontsize=9, color='#333333')
        # Use short label to prevent overlap between subplots
        ylabel_suffix = ' (log)' if use_log else ''
        ax1.set_ylabel(f'{metric_name}{ylabel_suffix}', fontsize=9, color='#4A90E2')

        # Use all polynomial degrees for consistent x-axis (missing data will show as gaps in line)
        degree_to_label = {i: str(i) for i in range(17)}
        ax1.set_xticks(all_poly_degrees)
        ax1.set_xticklabels([degree_to_label.get(deg, str(deg)) for deg in all_poly_degrees],
                           fontsize=8, rotation=90, ha='center')
        if use_log:
            ax1.set_yscale('log')
        ax1.set_ylim(metric_ylim)  # Set consistent range

        # Custom y-axis formatter to show clipping indicator for MAE/max_error
        # ULP uses standard scientific notation without clipping labels
        def metric_formatter(y, pos):
            if not use_log:
                # Linear scale: show as-is, scientific notation for large values
                if abs(y) >= 1000:
                    return f'{y:.1e}'
                return f'{y:.0f}'
            if y <= 0:
                return '$10^{0}$'
            exp = int(np.floor(np.log10(abs(y))))
            # Show clipping indicator only if break_threshold is set and y is below it
            if break_threshold is not None and y <= break_threshold:
                return '≤$10^{-22}$'
            else:
                return f'$10^{{{exp}}}$'
        ax1.yaxis.set_major_formatter(FuncFormatter(metric_formatter))

        ax1.tick_params(axis='y', labelcolor='#4A90E2', labelsize=8)
        ax1.tick_params(axis='x', labelsize=8)

        # Add axis break indicators (zigzag lines) only for MAE/max_error plots
        # This shows that values below 1e-7 are clipped (actual values may be ~1e-22 or zero)
        # ULP plots don't need this since they don't clip at small values
        if show_break_indicator and break_threshold is not None:
            ylim = ax1.get_ylim()
            xlim = ax1.get_xlim()
            # Draw zigzag at break_threshold level (slightly above the clipped threshold)
            break_y_pos = break_threshold * 1.2  # Position in data coordinates
            x_left = xlim[0] - (xlim[1] - xlim[0]) * 0.015  # Slightly left of left edge
            x_width = (xlim[1] - xlim[0]) * 0.025  # Width of zigzag
            # Create zigzag pattern (two diagonal lines forming a "Z")
            ax1.plot([x_left, x_left + x_width * 0.5, x_left + x_width],
                    [break_y_pos * 0.7, break_y_pos * 1.3, break_y_pos * 0.7],
                    color='#4A90E2', linewidth=1.8, clip_on=False, zorder=10,
                    solid_capstyle='round')

        # Minimal gridlines - only major horizontal
        ax1.grid(True, alpha=0.15, which='major', axis='y', linestyle='-', linewidth=0.5)
        ax1.grid(False, axis='x')

        ax2.set_ylabel('Runtime (μs)', fontsize=9, color='#E27A7A')
        ax2.set_ylim(runtime_ylim)  # Set consistent runtime range
        ax2.tick_params(axis='y', labelcolor='#E27A7A', labelsize=8)
        # Remove right spine (Tufte principle - already set globally but reinforce)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        # No subplot titles (Tufte minimalist style - info is in x-axis label)

        # Store legend handles from first subplot for unified legend
        if subplot_idx == 0:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            legend_lines = lines1 + lines2
            legend_labels = labels1 + labels2

    # Plot rational approximation subplots if data is available
    # Split into two subplots: num=den and num=den+1
    if rational_summary is not None:
        segments_list = sorted(rational_summary['segments'].unique())

        # Define two subplot configurations: num=den and num=den+1
        rational_configs = [
            ('num=den', lambda row: row['num_degree'] == row['den_degree']),
            ('num=den+1', lambda row: row['num_degree'] == row['den_degree'] + 1)
        ]

        for config_idx, (config_name, filter_func) in enumerate(rational_configs):
            rational_subplot_idx = num_depths + config_idx
            ax1 = axes[rational_subplot_idx]
            ax2 = ax1.twinx()

            # Filter rational_summary for this configuration
            filtered_summary = rational_summary[rational_summary.apply(filter_func, axis=1)]

            if filtered_summary.empty:
                # No data for this configuration, hide subplot
                ax1.axis('off')
                ax2.axis('off')
                continue

            # Get unique (num_degree, den_degree) pairs sorted by num_degree
            rational_degrees = filtered_summary[['num_degree', 'den_degree']].drop_duplicates().sort_values(['num_degree', 'den_degree'])

            # Create x-axis positions (one per degree pair, shared across all segments)
            x_positions = []
            x_labels = []
            for idx, row in rational_degrees.iterrows():
                num_deg = int(row['num_degree'])
                den_deg = int(row['den_degree'])
                x_positions.append((num_deg, den_deg))
                x_labels.append(f'{num_deg}/{den_deg}')

            x_tick_positions = list(range(len(x_positions)))

            # Plot data by grouping by segments (each segment gets its own line)
            for seg_idx, segments in enumerate(segments_list):
                seg_data = filtered_summary[filtered_summary['segments'] == segments]
                if seg_data.empty:
                    continue

                # Create data arrays aligned to x_positions
                x_vals = []
                mae_vals = []
                metric_min_vals = []
                metric_max_vals = []
                runtime_vals = []

                for x_idx, (num_deg, den_deg) in enumerate(x_positions):
                    deg_data = seg_data[(seg_data['num_degree'] == num_deg) & (seg_data['den_degree'] == den_deg)]
                    if not deg_data.empty:
                        x_vals.append(x_idx)
                        mae_vals.append(deg_data['metric_value'].values[0])
                        metric_min_vals.append(deg_data['metric_min'].values[0])
                        metric_max_vals.append(deg_data['metric_max'].values[0])
                        runtime_vals.append(deg_data['runtime_us'].values[0])

                if x_vals:
                    # Plot error line for this segment group
                    # Only show legend label for first segment in first rational subplot
                    label_error = f'Rational {metric_name}' if (rational_subplot_idx == num_depths and seg_idx == 0) else None
                    ax1.plot(x_vals, mae_vals,
                            color='#003366', linewidth=2.5, alpha=1.0,
                            marker='p', markersize=8, linestyle='-',
                            markeredgewidth=1.5, markeredgecolor='white',
                            label=label_error)

                    # Add shaded region (use metric-specific min_clip from config)
                    metric_min_clipped = [max(m, min_clip) for m in metric_min_vals]
                    ax1.fill_between(x_vals, metric_min_clipped, metric_max_vals,
                                   color='#003366', alpha=0.2, linewidth=0,
                                   zorder=2, interpolate=True)

                    # Plot runtime line for this segment group
                    # Only show legend label for first segment in first rational subplot
                    label_runtime = f'Rational Runtime' if (rational_subplot_idx == num_depths and seg_idx == 0) else None
                    ax2.plot(x_vals, runtime_vals,
                            color='#E27A7A', linewidth=2.5, alpha=1.0,
                            marker='s', markersize=7, linestyle='--',
                            markeredgewidth=1.5, markeredgecolor='white',
                            label=label_runtime)

            # Configure axes
            # Convert segments_list to clean integer list for display
            segments_str = ','.join([str(int(s)) for s in segments_list])
            ax1.set_xlabel(f'Rational {config_name} (Segs=[{segments_str}])', fontsize=9, color='#333333')
            scale_label = ', log scale' if use_log else ''
            ax1.set_ylabel(f'Error ({metric_name}{scale_label})', fontsize=9, color='#4A90E2')
            ax1.set_xticks(x_tick_positions)
            ax1.set_xticklabels(x_labels, fontsize=8, rotation=90, ha='center')
            if use_log:
                ax1.set_yscale('log')
            ax1.set_ylim(metric_ylim)
            ax1.yaxis.set_major_formatter(FuncFormatter(metric_formatter))
            ax1.tick_params(axis='y', labelcolor='#4A90E2', labelsize=8)
            ax1.tick_params(axis='x', labelsize=8)
            ax1.grid(True, alpha=0.15, which='major', axis='y', linestyle='-', linewidth=0.5)
            ax1.grid(False, axis='x')

            ax2.set_ylabel('Runtime (μs)', fontsize=9, color='#E27A7A')
            ax2.set_ylim(runtime_ylim)
            ax2.tick_params(axis='y', labelcolor='#E27A7A', labelsize=8)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)

            # Store legend handles for unified legend (only from first rational subplot)
            if rational_subplot_idx == num_depths:
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                # Append rational legend entries to existing legend
                legend_lines.extend(lines1 + lines2)
                legend_labels.extend(labels1 + labels2)

    # Use tight_layout FIRST to compute proper subplot positions
    # Increase w_pad significantly to prevent Y-axis label overlap between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.94], pad=1.5, w_pad=5.0, h_pad=1.0)

    # Create unified legend at the top of the figure AFTER tight_layout
    # We have ~6 legend items total, so ncol=6 ensures single row
    # Place legend in the reserved space at top (above 0.94)
    # Set clip_on=True to prevent legend elements from escaping
    legend = fig.legend(legend_lines, legend_labels, loc='upper center',
              bbox_to_anchor=(0.5, 0.985), ncol=6, fontsize=8.5,
              framealpha=0.95, edgecolor='none', fancybox=False,
              columnspacing=1.2, handletextpad=0.5)
    legend.set_clip_on(True)

    # No overall title (Tufte minimalist style)

    # Save plot - publication quality 300 DPI
    # Use metric_filename from config (already set at top of function)
    filename = f'{all_activations_dir}/all_activations_degree_vs_{metric_filename}_{timing_config}{filename_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_pareto_frontier_per_activation(df, arch_prefix=None, output_dir='plots_heatmap', metric='mae'):
    """Generate individual Pareto frontier plots for each activation.

    Creates one plot per activation showing degree vs error/runtime tradeoffs.
    Similar to aggregate plot but focused on single activation.

    Args:
        df: DataFrame with piecewise method results
        arch_prefix: Architecture prefix (e.g., 'blackhole', 'wormhole')
        output_dir: Output directory for plots
        metric: Error metric to plot ('mae', 'max_error', 'max_ulp_error', 'mean_ulp_error')
    """

    # Get metric configuration
    config = get_metric_config(metric)
    metric_name = config['metric_name']
    metric_filename = config['metric_filename']
    use_log = config['use_log']

    # Load rational approximation data
    rational_df = None
    if arch_prefix:
        rational_df = load_rational_results(arch_prefix)

    # Load TTNN data for baseline
    native_df = load_native_sfpu_results(arch_prefix)

    # Generate plot for each activation
    activations = sorted(df['activation'].unique())
    print(f"Generating per-activation Pareto frontier plots for {len(activations)} activations...")

    for activation in activations:
        # Create per-activation subdirectory
        activation_output_dir = f"{output_dir}/per_activation/{activation}"
        os.makedirs(activation_output_dir, exist_ok=True)

        activation_df = df[df['activation'] == activation]

        # Check if this activation has TTNN support
        has_ttnn = False
        ttnn_error = None
        ttnn_runtime = None
        if native_df is not None:
            ttnn_data = native_df[native_df['activation'] == activation]
            if not ttnn_data.empty:
                has_ttnn = True
                ttnn_error = ttnn_data[metric].median()
                ttnn_runtime = ttnn_data['runtime_us'].median()

        # Get best error for each (degree, depth) combination
        summary = activation_df.groupby(['poly_degree', 'depth']).agg({
            metric: 'min',
            'runtime_us': 'min',
            'memory_coeffs': 'first'
        }).reset_index()

        # Use available depths (exclude 32)
        depths = sorted([d for d in summary['depth'].unique() if d < 32])
        summary_filtered = summary[summary['depth'].isin(depths)]

        if summary_filtered.empty:
            continue

        # Get rational data for this activation
        rational_summary = None
        if rational_df is not None:
            rational_activation = rational_df[rational_df['activation'] == activation]
            if not rational_activation.empty:
                rational_summary = rational_activation.groupby(['num_degree', 'den_degree', 'rational_degree', 'segments']).agg({
                    metric: 'min',
                    'runtime_us': 'min'
                }).reset_index()

        # Compute Y-axis limits based on worst-case metric value
        metric_values = summary_filtered[metric].dropna()
        if has_ttnn and ttnn_error is not None:
            metric_values = pd.concat([metric_values, pd.Series([ttnn_error])])
        if rational_summary is not None and not rational_summary.empty:
            rational_values = rational_summary[metric].dropna()
            metric_values = pd.concat([metric_values, rational_values])

        metric_max = metric_values.max() if len(metric_values) > 0 else 1

        # Set Y-axis limits with padding
        if use_log:
            # Log scale: use sensible floor (values below 1e-8 are numerical noise)
            MIN_LOG_FLOOR = 1e-8
            metric_min = max(MIN_LOG_FLOOR, metric_values[metric_values > 0].min()) if len(metric_values[metric_values > 0]) > 0 else MIN_LOG_FLOOR
            log_min = np.log10(metric_min)
            log_max = np.log10(metric_max)
            log_range = log_max - log_min
            padding = max(0.5, log_range * 0.1)
            metric_ylim = (10**(log_min - padding), 10**(log_max + padding))
        else:
            # Linear scale: add 10% padding
            metric_min = metric_values.min() if len(metric_values) > 0 else 0
            padding = (metric_max - metric_min) * 0.1
            metric_ylim = (max(0, metric_min - padding), metric_max + padding)

        # Create subplot grid
        num_depths = len(depths)
        num_rational_subplots = 2 if rational_summary is not None and not rational_summary.empty else 0
        num_subplots = num_depths + num_rational_subplots
        fig, axes = plt.subplots(1, num_subplots, figsize=(2.8 * num_subplots, 3.0))
        if num_subplots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Plot polynomial data
        for i, depth in enumerate(depths):
            ax1 = axes[i]
            ax2 = ax1.twinx()

            depth_data = summary_filtered[summary_filtered['depth'] == depth].sort_values('poly_degree')

            if len(depth_data) > 0:
                # Plot error (left Y-axis)
                ax1.plot(depth_data['poly_degree'], depth_data[metric],
                        'o-', color='#2E5FA3', linewidth=2, markersize=8, label=f'{metric_name}')

                # Plot runtime (right Y-axis)
                ax2.plot(depth_data['poly_degree'], depth_data['runtime_us'],
                        's--', color='#C44E52', linewidth=2, markersize=6, label='Runtime')

                # Add TTNN baseline if available
                if has_ttnn:
                    ax1.axhline(y=ttnn_error, color='#7FBC41', linestyle=':', linewidth=2, label='TTNN')

                # Configure axes
                ax1.set_xlabel(f'Polynomial Degree (Depth={depth})', fontsize=10, fontweight='bold')
                ax1.set_ylabel(f'{metric_name}', color='#2E5FA3', fontsize=10, fontweight='bold')
                ax2.set_ylabel('Runtime (μs)', color='#C44E52', fontsize=10, fontweight='bold')

                if use_log:
                    ax1.set_yscale('log')
                ax1.set_ylim(metric_ylim)
                ax1.tick_params(axis='y', labelcolor='#2E5FA3')
                ax2.tick_params(axis='y', labelcolor='#C44E52')

                ax1.grid(True, alpha=0.2, which='both')

        # Plot rational data if available
        if rational_summary is not None and not rational_summary.empty:
            # Plot num=den (subplot after polynomial plots)
            ax1 = axes[num_depths]
            ax2 = ax1.twinx()

            rational_equal = rational_summary[rational_summary['num_degree'] == rational_summary['den_degree']].sort_values('segments')
            if len(rational_equal) > 0:
                ax1.plot(rational_equal['segments'], rational_equal[metric],
                        'o-', color='#8C6BB1', linewidth=2, markersize=8, label=f'{metric_name}')
                ax2.plot(rational_equal['segments'], rational_equal['runtime_us'],
                        's--', color='#F28E2B', linewidth=2, markersize=6, label='Runtime')

                if has_ttnn:
                    ax1.axhline(y=ttnn_error, color='#7FBC41', linestyle=':', linewidth=2, label='TTNN')

                ax1.set_xlabel('Segments (num=den)', fontsize=10, fontweight='bold')
                ax1.set_ylabel(f'{metric_name}', color='#8C6BB1', fontsize=10, fontweight='bold')
                ax2.set_ylabel('Runtime (μs)', color='#F28E2B', fontsize=10, fontweight='bold')
                if use_log:
                    ax1.set_yscale('log')
                ax1.set_ylim(metric_ylim)
                ax1.grid(True, alpha=0.2, which='both')

        # Add title
        fig.suptitle(f'{activation.upper()} - Degree vs {metric_name} Tradeoff',
                    fontsize=12, fontweight='bold', y=1.02)

        # Add shared legend at top
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#2E5FA3', marker='o', linestyle='-', linewidth=2, markersize=6, label=metric_name),
            Line2D([0], [0], color='#C44E52', marker='s', linestyle='--', linewidth=2, markersize=5, label='Runtime (μs)'),
        ]
        if has_ttnn:
            legend_elements.append(
                Line2D([0], [0], color='#7FBC41', linestyle=':', linewidth=2, label='TTNN Baseline')
            )
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99),
                   ncol=len(legend_elements), fontsize=9, frameon=False)

        plt.tight_layout(rect=[0, 0, 1, 0.91])

        # Save plot in per-activation subdirectory
        filename = f'{activation_output_dir}/{activation}_degree_vs_{metric_filename}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    print(f"✓ Saved {len(activations)} per-activation Pareto frontier plots to {output_dir}/per_activation/")


def plot_per_activation_comparison(df, arch_prefix=None, output_dir='plots_heatmap', metric='mae', output_filename=None, rational_df=None):
    """
    Plot per-activation comparison of LUT vs TTNN performance.

    X-axis: Activation functions
    Y-axis: Error metric normalized to TTNN baseline (1.0 = TTNN performance)
    Shows Pareto-optimal LUT configurations as a cluster of points for each activation.

    Args:
        metric: Error metric to compare ('mae' or 'max_error')
        output_filename: Output filename (default: 'per_activation_{metric}_lut_vs_ttnn.png')
        rational_df: DataFrame with rational approximation results
    """
    # Get metric configuration
    config = get_metric_config(metric)
    metric_dir = config['metric_dir']
    metric_name = config['metric_name']
    metric_filename = config['metric_filename']

    all_activations_dir = f"{output_dir}/all_activations/{metric_dir}"
    os.makedirs(all_activations_dir, exist_ok=True)

    # Load TTNN baseline data
    native_df = load_native_sfpu_results(arch_prefix)

    # Only include activations with TTNN support for comparison
    if native_df is None or native_df.empty:
        print("Skipping per-activation comparison plot (no TTNN baseline)")
        return

    # Only compare activations that have TTNN support
    ttnn_activations = set(native_df['activation'].unique())
    comparison_df = df[df['activation'].isin(ttnn_activations)].copy()

    if len(comparison_df) == 0:
        print("No activations with TTNN support found")
        return

    # Add rational data if available
    if rational_df is not None and not rational_df.empty:
        rational_comparison = rational_df[rational_df['activation'].isin(ttnn_activations)].copy()
        # Mark rational data with a special degree marker (-1 for sorting purposes)
        rational_comparison['poly_degree'] = -1  # Will be used for coloring
        # Combine with polynomial data
        comparison_df = pd.concat([comparison_df, rational_comparison], ignore_index=True)

    # Create TTNN baseline lookup
    ttnn_baseline = {}
    for _, row in native_df.iterrows():
        # Use small epsilon for activations with metric=0 (perfect TTNN implementations like relu)
        # This allows comparison while avoiding division by zero
        baseline_value = row[metric] if row[metric] > 0 else 1e-22
        ttnn_baseline[row['activation']] = baseline_value

    # Compute normalized error (LUT_metric / TTNN_metric)
    # For activations with perfect TTNN (metric=0), we use 1e-22 as denominator
    # So LUT with metric=0 would give normalized=1.0, LUT with metric=1e-10 would give normalized>1
    comparison_df['metric_normalized'] = comparison_df.apply(
        lambda row: row[metric] / ttnn_baseline.get(row['activation'], 1e-22) if ttnn_baseline.get(row['activation'], 1e-22) > 0 else 1.0,
        axis=1
    )

    # Find Pareto frontier for each activation on metric vs Runtime tradeoff
    # True frontier: no point should have both worse metric AND worse runtime than another
    # This shows configurations where you can't improve one metric without degrading the other
    pareto_points = []

    for activation in sorted(comparison_df['activation'].unique()):
        act_df = comparison_df[comparison_df['activation'] == activation].copy()

        # Find Pareto frontier: for each point, check if any other point dominates it
        # Point A dominates point B if A has (metric <= B.metric) AND (runtime <= B.runtime) with at least one strict inequality
        frontier = []

        for i, row in act_df.iterrows():
            is_dominated = False
            for j, other in act_df.iterrows():
                if i != j:
                    # Check if other point dominates this one (better or equal on both, strictly better on at least one)
                    if (other[metric] <= row[metric] and other['runtime_us'] <= row['runtime_us'] and
                        (other[metric] < row[metric] or other['runtime_us'] < row['runtime_us'])):
                        is_dominated = True
                        break

            if not is_dominated:
                frontier.append(row)

        if frontier:
            pareto_act_df = pd.DataFrame(frontier)
            pareto_points.append(pareto_act_df)

    pareto_df = pd.concat(pareto_points, ignore_index=True)

    # Create figure - very compact Tufte style
    # Use ALL activations with TTNN support for consistent x-axis across plots
    # This ensures both MAE and max_error plots have the same x-axis for animation
    activations = sorted(ttnn_activations)
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))

    # Tufte-inspired minimalist style: remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # Create color map for polynomial degrees
    degree_colors = {
        -1: '#003366', # Rational - dark blue
        0: '#B8B8B8',  # Deg 0 - gray
        1: '#8B4789',  # Deg 1 - purple
        2: '#4A90E2',  # Deg 2 - blue
        3: '#E27A7A',  # Deg 3 - red
        6: '#6AAA71',  # Deg 6 - green
    }

    # Create marker map for polynomial degrees
    degree_markers = {
        -1: 'p',  # Rational - pentagon
        0: 'o',   # Deg 0 - circle
        1: 'o',   # Deg 1 - circle
        2: 'o',   # Deg 2 - circle
        3: 'o',   # Deg 3 - circle
        6: 'o',   # Deg 6 - circle
    }

    # Plot TTNN baseline at y=1.0
    ax.axhline(y=1.0, color='#333333', linestyle='--', linewidth=1.5,
               alpha=0.7, zorder=1)

    # Add TTNN label directly on the plot next to the line
    ax.text(len(activations) - 0.5, 1.0, 'TTNN',
            fontsize=8, color='#333333', ha='right', va='bottom')

    # Plot Pareto-optimal LUT points
    x_positions = {act: i for i, act in enumerate(activations)}

    # Add translucent red background for activations where TTNN has significantly lower error
    # Use threshold of 1.1x (10% tolerance) to avoid marking near-identical performance as "worse"
    # Examples: hardsigmoid (1.000x), erf (1.003x) are essentially identical to TTNN
    ttnn_threshold = 1.1
    for activation in activations:
        act_data = pareto_df[pareto_df['activation'] == activation]
        if not act_data.empty:
            min_normalized = act_data['metric_normalized'].min()
            if min_normalized > ttnn_threshold:
                # TTNN wins significantly - LUT configs are >10% worse
                x_pos = x_positions[activation]
                ax.axvspan(x_pos - 0.4, x_pos + 0.4, color='#E27A7A', alpha=0.15, zorder=0)

    # Log scale for y-axis to show both better and worse than SFPU
    ax.set_yscale('log')
    y_min, y_max = 0.05, 50
    ax.set_ylim(y_min, y_max)

    # Plot all Pareto points (no clipping tracking needed)
    for degree in sorted(pareto_df['poly_degree'].unique()):
        degree_df = pareto_df[pareto_df['poly_degree'] == degree]
        degree_label = 'Rational' if degree == -1 else str(degree)

        x_coords = [x_positions[act] for act in degree_df['activation']]
        y_coords = degree_df['metric_normalized'].values

        # Filter to only plot points within y-axis bounds
        x_coords_in_range = []
        y_coords_in_range = []

        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            if y_min <= y <= y_max:
                x_coords_in_range.append(x)
                y_coords_in_range.append(y)

        # Only plot points that are within the y-axis bounds
        if len(x_coords_in_range) > 0:
            ax.scatter(x_coords_in_range, y_coords_in_range,
                      color=degree_colors.get(degree, '#000000'),
                      marker=degree_markers.get(degree, 'o'),
                      s=60, alpha=0.7, label=f'{degree_label}',
                      edgecolors='white', linewidths=0.8, zorder=3)

    # Configure axes - Tufte style
    ax.set_xlabel('Activation', fontsize=9, color='#333333')
    ax.set_ylabel(f'Normalized {metric_name} (to TTNN)', fontsize=9, color='#333333')
    ax.set_xticks(range(len(activations)))
    ax.set_xticklabels(activations, rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='both', labelsize=8, colors='#333333')

    # Set explicit x-axis limits with padding to prevent margin shifts from red bars
    # This ensures both MAE and max plots have identical x-axis layout for animation
    ax.set_xlim(-0.7, len(activations) - 0.3)

    # Add horizontal reference lines (subtle)
    ax.axhline(y=0.5, color='#6AAA71', linestyle=':', linewidth=0.8, alpha=0.3)
    ax.axhline(y=2.0, color='#E27A7A', linestyle=':', linewidth=0.8, alpha=0.3)

    # Minimal grid - Tufte style
    ax.grid(True, alpha=0.1, which='major', axis='y', linestyle='-', linewidth=0.5)
    ax.grid(False, axis='x')

    # Get legend handles and labels for top placement
    handles, labels = ax.get_legend_handles_labels()

    # Legend at top - Tufte style (no box)
    # Use number of labels as ncol to ensure single row
    fig.legend(handles, labels, loc='upper center',
              bbox_to_anchor=(0.5, 0.98), ncol=len(labels), fontsize=8,
              framealpha=0, edgecolor='none', columnspacing=1.0, handletextpad=0.5)

    # No title (Tufte minimalist principle)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save plot in all_activations subdirectory
    if output_filename is None:
        # Use metric_filename from config
        output_filename = f'per_activation_{metric_filename}_lut_vs_ttnn.png'
    filename = f'{all_activations_dir}/{output_filename}'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filename}")
    plt.close()


def print_pareto_summary(df, arch_prefix=None):
    """Print summary of Pareto-optimal configurations."""
    print()
    print("=" * 80)
    if arch_prefix:
        print(f"Pareto Frontier Summary ({arch_prefix.capitalize()})")
    else:
        print("Pareto Frontier Summary")
    print("=" * 80)
    print()

    # Compute average MAE for each (poly_degree, depth) combination
    summary = df.groupby(['poly_degree', 'depth']).agg({
        'mae': 'mean',
        'memory_coeffs': 'first'
    }).reset_index()

    summary = summary.sort_values('memory_coeffs')

    # Identify Pareto-optimal points
    pareto_optimal = []
    min_mae_so_far = float('inf')

    for _, row in summary.iterrows():
        if row['mae'] < min_mae_so_far:
            pareto_optimal.append(row)
            min_mae_so_far = row['mae']

    print("Pareto-Optimal Configurations (Best Accuracy for Each Memory Budget):")
    print("-" * 80)
    print(f"{'Degree':<10} {'Depth':>6} {'Memory':>10} {'Avg MAE':>12}")
    print("-" * 80)

    for config in pareto_optimal:
        print(f"{'p' + str(int(config['poly_degree'])):<10} {int(config['depth']):>6} "
              f"{int(config['memory_coeffs']):>10} {config['mae']:>12.6f}")

    print()
    print("Key Insights:")
    print("-" * 80)

    # Find best overall
    best_overall = summary.loc[summary['mae'].idxmin()]
    print(f"• Best overall accuracy: p{int(best_overall['poly_degree'])}_s{int(best_overall['depth'])} "
          f"(MAE={best_overall['mae']:.6f}, {int(best_overall['memory_coeffs'])} coeffs)")

    # Find best for low memory (<100 coeffs)
    low_mem = summary[summary['memory_coeffs'] < 100]
    if not low_mem.empty:
        best_low_mem = low_mem.loc[low_mem['mae'].idxmin()]
        print(f"• Best for <100 coeffs: p{int(best_low_mem['poly_degree'])}_s{int(best_low_mem['depth'])} "
              f"(MAE={best_low_mem['mae']:.6f}, {int(best_low_mem['memory_coeffs'])} coeffs)")

    # Find best for medium memory (100-500 coeffs)
    med_mem = summary[(summary['memory_coeffs'] >= 100) & (summary['memory_coeffs'] < 500)]
    if not med_mem.empty:
        best_med_mem = med_mem.loc[med_mem['mae'].idxmin()]
        print(f"• Best for 100-500 coeffs: p{int(best_med_mem['poly_degree'])}_s{int(best_med_mem['depth'])} "
              f"(MAE={best_med_mem['mae']:.6f}, {int(best_med_mem['memory_coeffs'])} coeffs)")

    print()

def main(arch_prefix=None, activation=None, degree=None, depth=None, segmentation=None, precision=None):
    """Generate Pareto frontier analysis."""
    if not arch_prefix:
        print("ERROR: --arch-prefix is required (e.g., 'blackhole' or 'wormhole')")
        print("This prevents accidentally creating generic plot directories.")
        print()
        print("Usage: python3 plot_heatmap.py --arch-prefix blackhole")
        return

    df = load_results(arch_prefix)

    if df is None or df.empty:
        print(f"❌ No result CSV files found")
        print(f"   Expected directory: data/{arch_prefix}/polynomial/")
        return

    # Apply filters
    initial_count = len(df)
    if activation:
        df = df[df['activation'] == activation]
    if degree is not None:
        df = df[df['poly_degree'] == degree]
    if depth is not None:
        df = df[df['depth'] == depth]
    if segmentation:
        df = df[df['segmentation'] == segmentation]
    if precision:
        df = df[df['precision'] == precision]

    if df.empty:
        print(f"❌ No data after applying filters (started with {initial_count} rows)")
        return

    if activation or degree or depth or segmentation or precision:
        print(f"Applied filters: {initial_count} → {len(df)} measurements")

    print(f"Loaded {len(df)} measurements across {len(df['activation'].unique())} activations")
    available_degrees = sorted(df['poly_degree'].unique())
    print(f"Degrees: {', '.join(map(str, available_degrees))}")
    available_depths = sorted([d for d in df['depth'].unique() if d < 32])
    print(f"Depths (excluding 32): {', '.join(map(str, available_depths))}")

    # Get list of activations with TTNN support
    activations_with_ttnn = []
    native_df = load_native_sfpu_results(arch_prefix)
    if native_df is not None and not native_df.empty:
        ttnn_activations = set(native_df['activation'].unique())
        # Intersect with current dataframe activations (respects filters)
        activations_with_ttnn = sorted(set(df['activation'].unique()) & ttnn_activations)
        if activations_with_ttnn:
            print(f"\nActivations with TTNN support ({len(activations_with_ttnn)}): {', '.join(activations_with_ttnn)}")

    # Get activations without TTNN support
    all_activations = set(df['activation'].unique())
    activations_without_ttnn = sorted(all_activations - set(activations_with_ttnn))
    if activations_without_ttnn:
        print(f"Activations without TTNN support ({len(activations_without_ttnn)}): {', '.join(activations_without_ttnn)}")

    # Load rational approximation data
    rational_df = load_rational_results(arch_prefix)

    # Generate plots with organized subdirectories
    # All plots go under plots/{arch}/heatmap with subdirectories:
    # - all_activations/ for aggregate plots
    # - per_activation/{act}/ for per-activation plots
    heatmap_dir = f"plots/{arch_prefix}/heatmap"

    # CLEAN SEPARATION:
    # - If --activation is specified: generate ONLY per-activation plots for that activation
    # - If --activation is NOT specified: generate ONLY aggregate plots

    if activation:
        # PER-ACTIVATION MODE: Generate only plots for the specified activation
        print("=" * 80)
        print(f"PER-ACTIVATION MODE: Generating plots for {activation} only")
        print("=" * 80)
        print()

        # Generate plots for all error metrics
        for metric in ['mae', 'max_error', 'max_ulp_error', 'mean_ulp_error']:
            print(f"\nGenerating {activation} Pareto frontier plot ({metric})...")
            plot_pareto_frontier_per_activation(df, arch_prefix, heatmap_dir, metric=metric)

        # Generate heatmaps for all metrics
        for metric in ['mae', 'max_error', 'max_ulp_error', 'mean_ulp_error']:
            print(f"\nGenerating {activation} depth-degree heatmap ({metric})...")
            plot_depth_degree_heatmap_per_activation(df, arch_prefix, heatmap_dir, metric=metric)

        print()
        print("=" * 80)
        print(f"✓ Per-activation plots generated for {activation} in:")
        print(f"  - {heatmap_dir}/per_activation/{activation}/{activation}_degree_vs_*.png")
        print(f"  - {heatmap_dir}/per_activation/{activation}/{activation}_heatmap_*.png")
        print("=" * 80)

    else:
        # AGGREGATE MODE: Generate only aggregate plots (all_activations_*)
        print("=" * 80)
        print("AGGREGATE MODE: Generating aggregate plots for all activations")
        print("=" * 80)
        print()

        # Generate plots for all metrics and all timing configurations
        metrics_to_plot = ['mae', 'max_error', 'max_ulp_error', 'mean_ulp_error']

        for metric in metrics_to_plot:
            # Get metric display name
            if metric == 'mae':
                metric_name = 'MAE'
            elif metric == 'max_error':
                metric_name = 'Max Error'
            elif metric == 'max_ulp_error':
                metric_name = 'Max ULP Error'
            elif metric == 'mean_ulp_error':
                metric_name = 'Mean ULP Error'
            else:
                metric_name = metric.upper()

            # Generate plots for each timing configuration
            for timing_config, timing_label in zip(TIMING_METRICS, METRIC_LABELS):
                print(f"\nGenerating degree_vs_{metric}_{timing_config} plots...")

                # Plot 1: Activations with TTNN support (with TTNN baseline)
                if activations_with_ttnn:
                    print(f"  - {timing_label} WITH TTNN support...")
                    plot_pareto_frontier(df, arch_prefix, heatmap_dir,
                                       activations_filter=activations_with_ttnn,
                                       show_native_sfpu=True,
                                       filename_suffix='_with_sfpu',
                                       metric=metric,
                                       timing_config=timing_config)

                # Plot 2: Activations without TTNN support (no TTNN baseline)
                if activations_without_ttnn:
                    print(f"  - {timing_label} WITHOUT TTNN support...")
                    plot_pareto_frontier(df, arch_prefix, heatmap_dir,
                                       activations_filter=activations_without_ttnn,
                                       show_native_sfpu=False,
                                       filename_suffix='_without_sfpu',
                                       metric=metric,
                                       timing_config=timing_config)

        # Generate aggregate heatmap for all activations (for each metric)
        for metric in metrics_to_plot:
            print(f"\nGenerating aggregate depth-degree heatmap for {metric}...")
            plot_depth_degree_heatmap(df, arch_prefix, heatmap_dir, metric=metric)

        # Generate per-activation comparison plots (LUT vs TTNN) for both MAE and max_error
        if activations_with_ttnn:
            print(f"\nGenerating per-activation LUT vs TTNN comparison plot (MAE)...")
            plot_per_activation_comparison(df, arch_prefix, heatmap_dir, metric='mae', rational_df=rational_df)

            print(f"\nGenerating per-activation LUT vs TTNN comparison plot (max_error)...")
            plot_per_activation_comparison(df, arch_prefix, heatmap_dir, metric='max_error', rational_df=rational_df)

        print()
        print("=" * 80)
        print(f"✓ Aggregate plots generated in:")
        print(f"  - {heatmap_dir}/all_activations/{{mae,max,ulp_max,ulp_mean}}/*.png")
        print("=" * 80)

    # Print summary
    print_pareto_summary(df, arch_prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Pareto frontier analysis plots')
    parser.add_argument('--arch-prefix', type=str, default=None,
                        help='Architecture prefix for CSVs (e.g., "wormhole" or "blackhole")')
    parser.add_argument('--activation', type=str, default=None,
                        help='Filter by specific activation function')
    parser.add_argument('--degree', type=int, default=None,
                        help='Filter by polynomial degree (0=constant, 1=linear, 2=quadratic, 3=cubic, 6=hexic, 8=octic)')
    parser.add_argument('--depth', type=int, default=None,
                        help='Filter by LUT depth (4, 8, 16, 32)')
    parser.add_argument('--segmentation', type=str, default=None,
                        help='Filter by segmentation type (uniform or adaptive)')
    parser.add_argument('--precision', type=str, default=None,
                        help='Filter by precision (bf16 or fp32)')
    args = parser.parse_args()

    main(arch_prefix=args.arch_prefix, activation=args.activation, degree=args.degree,
         depth=args.depth, segmentation=args.segmentation, precision=args.precision)
