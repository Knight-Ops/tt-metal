#!/usr/bin/env python3
"""
Generate depth-degree heatmaps showing error values across depth and polynomial degree.

For each activation function, creates heatmaps showing:
- Polynomial MAE and Max Error
- Rational MAE and Max Error (when rational data is available)

Uses polynomial-fitter styling with RdYlGn_r colormap and log scale.
Highlights the cheapest (lowest memory) best configuration with a prominent black box.

Data is loaded from per-activation CSV hierarchy:
- data/<arch>/polynomial/<activation>.csv
- data/<arch>/rational/<activation>.csv

Outputs are organized into:
- plots/<arch>/depth_degree/all_activations/heatmap.png (combined hit count)
- plots/<arch>/depth_degree/per_activation/<activation>/heatmap.png (individual)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import os
import argparse

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


def detect_activations(arch_prefix):
    """Detect available activations from data/<arch>/polynomial/ and rational/ directories."""
    activations = set()

    # Check polynomial and rational directories
    for subdir in ['polynomial', 'rational']:
        dir_path = f'data/{arch_prefix}/{subdir}'
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith('.csv'):
                    activations.add(filename[:-4])  # Remove .csv

    # Exclude hardshrink and softshrink
    activations.discard('hardshrink')
    activations.discard('softshrink')

    return sorted(list(activations))


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
    metrics = ['time_profiler_us', 'mae', 'max_error', 'max_ulp']
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


def load_results(arch_prefix, activation=None):
    """Load polynomial benchmark results from per-activation CSV hierarchy.

    Supports both row-per-shape format (new) and multi-column format (legacy).

    Args:
        arch_prefix: Architecture name (e.g., 'blackhole', 'wormhole')
        activation: Specific activation to load, or None to load all

    Returns:
        DataFrame with all polynomial data (with all config-specific columns preserved)
    """
    all_dfs = []

    # Determine which activations to load
    if activation:
        activations = [activation]
    else:
        activations = detect_activations(arch_prefix)

    if not activations:
        print(f"❌ No activation CSV files found in data/{arch_prefix}/polynomial/")
        return None

    # Load data from polynomial directory
    for act in activations:
        csv_path = f'data/{arch_prefix}/polynomial/{act}.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df[df['status'] == 'pass']  # Only successful runs
            df['activation'] = act
            # Transform row-per-shape format to multi-column format if needed
            df = pivot_row_per_shape_to_columns(df)
            all_dfs.append(df)

    if not all_dfs:
        return None

    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure poly_degree column exists
    if 'degree' in combined_df.columns:
        combined_df['poly_degree'] = combined_df['degree']
    else:
        raise ValueError("CSV files missing 'degree' column")

    return combined_df


def load_rational_data(arch_prefix, activation=None):
    """Load rational approximation data from per-activation CSV hierarchy.

    Args:
        arch_prefix: Architecture name (e.g., 'blackhole', 'wormhole')
        activation: Specific activation to load, or None to load all

    Returns:
        DataFrame with all rational data or None (with all config-specific columns preserved)
    """
    all_dfs = []

    # Determine which activations to load
    if activation:
        activations = [activation]
    else:
        activations = detect_activations(arch_prefix)

    # Load data from rational directory
    for act in activations:
        csv_path = f'data/{arch_prefix}/rational/{act}.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df[df['status'] == 'pass']
            df['activation'] = act
            # Transform row-per-shape format to multi-column format if needed
            df = pivot_row_per_shape_to_columns(df)
            all_dfs.append(df)

    if not all_dfs:
        return None

    # Combine all dataframes
    rational_df = pd.concat(all_dfs, ignore_index=True)

    # Create degree string "num/den" for display
    if 'num_degree' in rational_df.columns and 'den_degree' in rational_df.columns:
        rational_df['degree_str'] = rational_df['num_degree'].astype(str) + '/' + rational_df['den_degree'].astype(str)

    # Use segments as depth
    if 'segments' in rational_df.columns:
        rational_df['depth'] = rational_df['segments']

    return rational_df


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


def plot_heatmap_subplot(ax, data, metric_name, approx_type, degree_col, activation, colorbar_shrink=1.0):
    """Plot a single heatmap subplot with highlighting for cheapest best configuration."""
    if data.empty:
        return

    # Pivot to create matrix for heatmap
    heatmap_pivot = data.pivot(index=degree_col, columns='depth', values=metric_name)

    if heatmap_pivot.empty:
        return

    # Prepare values for log scale visualization
    heatmap_values = heatmap_pivot.values
    heatmap_values_for_display = np.where(heatmap_values == 0, 1e-10, heatmap_values)
    log_values = np.log10(heatmap_values_for_display)

    # Create heatmap - polynomial-fitter styling
    # With proper subplot height allocation via GridSpec, we can use simpler aspect ratio
    num_cols = heatmap_pivot.shape[1]
    num_rows = heatmap_pivot.shape[0]

    # Slight adjustment for readability, but not as aggressive since GridSpec handles height
    # For very tall heatmaps (>15 rows), use less aggressive aspect ratio to allow taller cells
    if num_rows > 15:
        aspect_ratio = num_cols / (num_rows * 0.8)
    elif num_rows > 10:
        aspect_ratio = num_cols / (num_rows * 1.0)
    elif num_rows > 6:
        aspect_ratio = num_cols / (num_rows * 1.2)
    else:
        aspect_ratio = 'auto'

    # Use dynamic color range based on data
    vmin_val = np.min(log_values)
    vmax_val = np.max(log_values)
    im = ax.imshow(log_values, cmap='RdYlGn_r', aspect=aspect_ratio, interpolation='nearest',
                   origin='lower', vmin=vmin_val, vmax=vmax_val)

    # Find cheapest best configuration
    cheapest_best_idx = find_cheapest_best(heatmap_pivot, heatmap_values)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(heatmap_pivot.columns)))
    ax.set_yticks(np.arange(len(heatmap_pivot.index)))
    ax.set_xticklabels(heatmap_pivot.columns, fontsize=14)

    # Format Y-axis labels - adjust font based on number of labels
    # With GridSpec allocation, use reasonable readable font sizes
    degree_labels = [f'p{deg}' for deg in heatmap_pivot.index]
    num_labels = len(degree_labels)
    if num_labels > 20:
        label_fontsize = 11
    elif num_labels > 15:
        label_fontsize = 12
    elif num_labels > 10:
        label_fontsize = 13
    elif num_labels > 6:
        label_fontsize = 14
    else:
        label_fontsize = 15
    ax.set_yticklabels(degree_labels, fontsize=label_fontsize)

    # Labels - polynomial-fitter style
    metric_label = 'MAE' if metric_name == 'mae' else 'Max Error'
    ax.set_xlabel('Depth (segments)', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'{approx_type} Degree', fontsize=16, fontweight='bold')
    ax.set_title(f'{activation.upper()} {approx_type} - {metric_label}', fontsize=18, fontweight='bold')

    # Add colorbar with optional shrink to keep reasonable size
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=colorbar_shrink)
    cbar.set_label(f'{metric_label} (log₁₀)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    # Annotate each cell with the actual error value - polynomial-fitter style
    # With GridSpec giving proper vertical space, use consistent readable font sizes
    num_rows = len(heatmap_pivot.index)
    if num_rows > 20:
        annotation_fontsize = 10
    elif num_rows > 15:
        annotation_fontsize = 11
    elif num_rows > 10:
        annotation_fontsize = 12
    elif num_rows > 6:
        annotation_fontsize = 13
    else:
        annotation_fontsize = 14
    for i in range(len(heatmap_pivot.index)):
        for j in range(len(heatmap_pivot.columns)):
            error_value = heatmap_values[i, j]
            if error_value == 0:
                text = '0'
            else:
                # Format: scientific notation with superscript exponent
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


# Timing configurations to plot (5 profiler metrics)
TIMING_CONFIGS = [
    'single_tile',
    '8_tiles',
    '256_tiles',
    'height_sharded',
    'yolov4'
]

CONFIG_LABELS = [
    'Single Tile',
    '8 Tiles',
    '256 Tiles',
    'Height Sharded',
    'YOLOv4'
]

# Error metrics to plot
ERROR_METRICS = [
    ('mae', 'MAE'),
    ('max_error', 'Max Error'),
    ('max_ulp_error', 'Max ULP Error'),
    ('mean_ulp_error', 'Mean ULP Error')
]

def plot_depth_degree_heatmap_per_activation(df, arch_prefix, base_output_dir='plots'):
    """Generate separate heatmap for each activation showing error values across depth and polynomial degree.

    Creates separate 1x5 grid plots for each error metric (MAE, Max Error, Max ULP, Mean ULP) and precision (bf16, fp32).
    Each plot has one subplot for each configuration (Single Tile, 8 Tiles, 256 Tiles, Height Sharded, YOLOv4).
    Highlights cheapest best configuration with prominent black box in each subplot.
    """
    # Generate for ALL activations
    all_activations = sorted(df['activation'].unique())

    print(f"Generating depth-degree heatmaps for {len(all_activations)} activations...")

    for activation in all_activations:
        activation_df = df[df['activation'] == activation]

        # Skip if no data
        if len(activation_df) == 0:
            continue

        # Get available precisions
        precisions = ['bf16', 'fp32'] if 'precision' in activation_df.columns else [None]

        # Generate plots for each precision
        for precision in precisions:
            # Filter by precision if applicable
            if precision:
                precision_df = activation_df[activation_df['precision'].str.lower() == precision.lower()]
                if len(precision_df) == 0:
                    continue
            else:
                precision_df = activation_df

            # Generate a plot for each error metric
            for error_metric, metric_label in ERROR_METRICS:
                # Create 1x5 grid for the 5 configurations
                fig, axes = plt.subplots(1, 5, figsize=(25.0, 5.0))

                has_any_data = False

                # First pass: compute global min/max across all configs for consistent color scale
                global_log_min = float('inf')
                global_log_max = float('-inf')

                for config in TIMING_CONFIGS:
                    error_col = f'{error_metric}_{config}'
                    if error_col in precision_df.columns:
                        poly_data = precision_df.groupby(['depth', 'poly_degree']).agg({error_col: 'min'}).reset_index()
                        if len(poly_data) > 0:
                            error_values = poly_data[error_col].values
                            error_values_safe = np.where(error_values == 0, 1e-10, error_values)
                            log_vals = np.log10(error_values_safe)
                            global_log_min = min(global_log_min, np.min(log_vals))
                            global_log_max = max(global_log_max, np.max(log_vals))

                # Second pass: plot with consistent color range
                for config_idx, (config, config_label) in enumerate(zip(TIMING_CONFIGS, CONFIG_LABELS)):
                    ax = axes[config_idx]

                    # Get config-specific error column
                    error_col = f'{error_metric}_{config}'

                    if error_col not in precision_df.columns:
                        ax.set_visible(False)
                        continue

                    # Compute minimum error for each (depth, degree) combination
                    poly_data = precision_df.groupby(['depth', 'poly_degree']).agg({error_col: 'min'}).reset_index()
                poly_data = poly_data.rename(columns={error_col: 'error'})

                if len(poly_data) == 0:
                    ax.set_visible(False)
                    continue

                # Pivot to create matrix for heatmap
                heatmap_pivot = poly_data.pivot(index='poly_degree', columns='depth', values='error')

                if heatmap_pivot.empty:
                    ax.set_visible(False)
                    continue

                # Prepare values for log scale visualization
                heatmap_values = heatmap_pivot.values
                heatmap_values_for_display = np.where(heatmap_values == 0, 1e-10, heatmap_values)
                log_values = np.log10(heatmap_values_for_display)

                # Create heatmap with consistent global color range across all subplots
                im = ax.imshow(log_values, cmap='RdYlGn_r', aspect='auto', interpolation='nearest',
                               origin='lower', vmin=global_log_min, vmax=global_log_max)

                # Find cheapest best configuration
                cheapest_best_idx = find_cheapest_best(heatmap_pivot, heatmap_values)

                # Set ticks and labels
                ax.set_xticks(np.arange(len(heatmap_pivot.columns)))
                ax.set_yticks(np.arange(len(heatmap_pivot.index)))
                ax.set_xticklabels(heatmap_pivot.columns, fontsize=14)
                ax.set_yticklabels([str(deg) for deg in heatmap_pivot.index], fontsize=14)

                # Labels
                ax.set_xlabel('Depth', fontsize=16, fontweight='bold')
                if config_idx == 0:
                    ax.set_ylabel('Degree', fontsize=16, fontweight='bold')
                ax.set_title(f'{config_label}', fontsize=18, fontweight='bold')

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
                cbar.set_label(f'{metric_label} (log₁₀)', fontsize=14, fontweight='bold')
                cbar.ax.tick_params(labelsize=12)

                # Annotate each cell with the actual error value
                for i in range(len(heatmap_pivot.index)):
                    for j in range(len(heatmap_pivot.columns)):
                        error_value = heatmap_values[i, j]
                        if error_value == 0:
                            text = '0'
                        else:
                            # Format: scientific notation
                            exp_str = f'{error_value:.1e}'
                            if 'e' in exp_str:
                                mantissa, exp = exp_str.split('e')
                                text = f'{mantissa}e$^{{{exp}}}$'
                            else:
                                text = exp_str

                        # Use black text
                        ax.text(j, i, text, ha='center', va='center',
                               color='black', fontsize=11)

                # Draw prominent black box around cheapest best configuration
                if cheapest_best_idx is not None:
                    i, j = cheapest_best_idx
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                             linewidth=3, edgecolor='black', facecolor='none',
                                             zorder=10)
                    ax.add_patch(rect)

                has_any_data = True

                if not has_any_data:
                    plt.close()
                    continue

                # Add overall title with precision
                precision_label = f' ({precision.upper()})' if precision else ''
                fig.suptitle(f'{activation.upper()} - Depth vs Degree Heatmaps ({metric_label}){precision_label}',
                            fontsize=20, fontweight='bold', y=1.02)

                plt.tight_layout()

                # Save plot to plots/<arch>/depth_degree/per_activation/<activation>/
                output_dir = f'{base_output_dir}/{arch_prefix}/depth_degree/per_activation/{activation}'
                os.makedirs(output_dir, exist_ok=True)

                # Include precision in filename
                precision_suffix = f'_{precision.lower()}' if precision else ''
                filename = f'{output_dir}/{error_metric}{precision_suffix}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  ✓ {filename}")
                plt.close()


def plot_depth_degree_heatmap(df, arch_prefix, base_output_dir='plots', config='256_tiles', error_metric='mae', precision=None):
    """Generate combined heatmap showing hit count of how many activations selected each (depth, degree) as their best configuration.

    Highlights most popular configuration with prominent black box.
    Uses a specific configuration and error metric for determining best (default: 256_tiles, mae).

    Args:
        df: DataFrame with activation data
        arch_prefix: Architecture name
        base_output_dir: Base directory for output
        config: Configuration to use (default: '256_tiles')
        error_metric: Error metric to use (default: 'mae')
        precision: Precision filter ('bf16' or 'fp32', None for all)
    """
    # Filter by precision if specified
    if precision and 'precision' in df.columns:
        df = df[df['precision'].str.lower() == precision.lower()]
        if len(df) == 0:
            print(f"⚠ Warning: No data for precision '{precision}', skipping combined heatmap")
            return

    error_col = f'{error_metric}_{config}'

    if error_col not in df.columns:
        print(f"⚠ Warning: Column '{error_col}' not found, skipping combined heatmap")
        return

    # Count how many activations selected each (depth, degree) as their "best" (cheapest best) configuration
    hit_counts = {}

    # For each activation, find its cheapest best configuration
    for activation in df['activation'].unique():
        activation_df = df[df['activation'] == activation]

        # Group by (depth, poly_degree) and find best error for each
        grouped = activation_df.groupby(['depth', 'poly_degree']).agg({error_col: 'min'}).reset_index()
        grouped_pivot = grouped.pivot(index='poly_degree', columns='depth', values=error_col)

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
    ax.set_xticklabels(heatmap_pivot.columns, fontsize=14)

    # Format Y-axis labels
    degree_labels = [f'p{int(deg)}' for deg in heatmap_pivot.index]
    ax.set_yticklabels(degree_labels, fontsize=14)

    # Labels
    config_label = config.replace('_', ' ').title()
    metric_label = error_metric.replace('_', ' ').upper()
    precision_label = f' ({precision.upper()})' if precision else ''
    ax.set_xlabel('Depth (segments)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Polynomial Degree', fontsize=16, fontweight='bold')
    ax.set_title(f'All Activations - Best Config Hit Count ({config_label}, {metric_label}){precision_label}', fontsize=18, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Hit Count', fontsize=14, fontweight='bold')

    # Annotate each cell with the hit count
    for i in range(len(heatmap_pivot.index)):
        for j in range(len(heatmap_pivot.columns)):
            count = int(heatmap_values[i, j])
            text = str(count) if count > 0 else ''

            # Use white text on dark cells, black on light cells
            text_color = 'white' if count > max_hits / 2 else 'black'

            ax.text(j, i, text, ha='center', va='center',
                   color=text_color, fontsize=13, fontweight='bold')

    # Draw prominent black box around most popular configuration
    if most_popular_idx is not None:
        i, j = most_popular_idx
        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 linewidth=3, edgecolor='black', facecolor='none',
                                 zorder=10)
        ax.add_patch(rect)

    plt.tight_layout()

    # Save plot to plots/<arch>/depth_degree/all_activations/
    output_dir = f'{base_output_dir}/{arch_prefix}/depth_degree/all_activations'
    os.makedirs(output_dir, exist_ok=True)
    precision_suffix = f'_{precision.lower()}' if precision else ''
    filename = f'{output_dir}/{error_metric}_{config}{precision_suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def main(arch_prefix):
    """Generate depth-degree heatmaps."""
    if not arch_prefix:
        print("ERROR: --arch-prefix is required (e.g., 'blackhole' or 'wormhole')")
        print("Usage: python3 plot_depth_degree.py --arch-prefix blackhole")
        return

    df = load_results(arch_prefix)

    if df is None or df.empty:
        print(f"❌ No result CSV files found")
        print(f"   Looking for: data/{arch_prefix}/polynomial/<activation>.csv")
        return

    print(f"Loaded {len(df)} measurements across {len(df['activation'].unique())} activations")

    # Get available precisions
    precisions = ['bf16', 'fp32'] if 'precision' in df.columns else [None]

    # Generate combined heatmaps for each metric and precision
    print(f"\nGenerating combined heatmaps...")
    for precision in precisions:
        for error_metric, metric_label in ERROR_METRICS:
            print(f"  {metric_label} ({precision if precision else 'all'})...")
            plot_depth_degree_heatmap(df, arch_prefix, base_output_dir='plots',
                                     config='256_tiles', error_metric=error_metric, precision=precision)

    # Generate per-activation heatmaps (handles all metrics and precisions internally)
    print(f"\nGenerating per-activation heatmaps...")
    plot_depth_degree_heatmap_per_activation(df, arch_prefix, base_output_dir='plots')

    print(f"\n✓ All heatmaps saved to:")
    print(f"  - plots/{arch_prefix}/depth_degree/all_activations/")
    print(f"  - plots/{arch_prefix}/depth_degree/per_activation/<activation>/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate depth-degree heatmaps for LUT approximations')
    parser.add_argument('--arch-prefix', type=str, required=True,
                        help='Architecture prefix (e.g., "blackhole" or "wormhole")')
    args = parser.parse_args()

    main(args.arch_prefix)
