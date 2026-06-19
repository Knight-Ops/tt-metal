#!/usr/bin/env python3
"""
Generate error vs runtime scatter plots for activation-specific analysis using new data structure.
Generates plots to plots/<arch>/<activation>/*.png with 5 subplots for different timing metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import csv
import os
import sys

from brokenaxes import brokenaxes
from adjustText import adjust_text
from pathlib import Path


# Coefficient directory for looking up actual degree
COEFF_DIR = Path(os.environ.get("TT_POLY_FIT_DIR", os.path.expanduser("~/workspace/tt-polynomial-fitter"))) / "data" / "coefficients"


def get_actual_degree(activation: str, degree: int, segments: int, segmentation: str) -> tuple:
    """
    Look up actual (used) polynomial degree from coefficient CSV.

    Returns (max_actual_degree, degree_sum) or (degree, degree * segments) if not found.
    The actual degree is the highest non-zero coefficient across all segments.
    """
    # Find coefficient CSV
    csv_path = None
    for metric in ["ulp", "max", "mae"]:
        for fitting in ["any", "remez", "fpminimax", "lstsq"]:
            candidate = COEFF_DIR / f"{activation}_p{degree}_s{segments}_{segmentation}_{fitting}_{metric}.csv"
            if candidate.exists():
                csv_path = candidate
                break
        if csv_path:
            break

    if not csv_path:
        # Fallback: assume all segments use full advertised degree
        return degree, degree * segments

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        # Find coefficient columns (c0, c1, c2, ...)
        coeff_cols = sorted([h for h in headers if h.startswith('c') and h[1:].isdigit()],
                            key=lambda h: int(h[1:]))
        if not coeff_cols:
            return degree, degree * segments

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
            return max(segment_degrees), sum(segment_degrees)
        return degree, degree * segments
    except Exception:
        return degree, degree * segments


def compute_pareto_frontier(points):
    """
    Compute the Pareto frontier from a list of points.

    For error vs runtime plots, Pareto-optimal points are those where:
    - No other point has BOTH lower error AND lower runtime

    Args:
        points: List of dicts with 'runtime', 'error', and optionally other keys

    Returns:
        List of Pareto-optimal points sorted by runtime
    """
    if not points:
        return []

    # Filter out invalid points
    valid_points = [p for p in points if p['runtime'] > 0 and p['error'] > 0]
    if not valid_points:
        return []

    # Sort by runtime
    sorted_points = sorted(valid_points, key=lambda p: p['runtime'])

    pareto_points = []
    min_error_so_far = float('inf')

    for point in sorted_points:
        # A point is Pareto-optimal if it has lower error than all points with lower runtime
        if point['error'] < min_error_so_far:
            pareto_points.append(point)
            min_error_so_far = point['error']

    return pareto_points


def detect_broken_axis_ranges(errors, gap_threshold_orders=10):
    """
    Detect if data has a large gap that would benefit from a broken axis.

    Uses adaptive thresholding:
    - For outliers (few extreme points vs many normal points): lower threshold (5 orders)
    - For balanced splits: higher threshold (gap_threshold_orders)

    Args:
        errors: List of error values
        gap_threshold_orders: Minimum orders of magnitude gap to trigger broken axis
                              for balanced data splits (default: 10)

    Returns:
        tuple: (needs_break, lower_range, upper_range) where ranges are (min, max) tuples
               Returns (False, None, None) if no break needed
    """
    if not errors or len(errors) < 2:
        return False, None, None

    # Filter out zeros and invalid values
    valid_errors = [e for e in errors if e > 0 and np.isfinite(e)]
    if len(valid_errors) < 2:
        return False, None, None

    # Sort and convert to log scale
    sorted_errors = sorted(valid_errors)
    log_errors = np.log10(sorted_errors)

    # Find the largest gap between consecutive points
    gaps = np.diff(log_errors)
    if len(gaps) == 0:
        return False, None, None

    max_gap_idx = np.argmax(gaps)
    max_gap = gaps[max_gap_idx]

    # Split data into lower and upper groups
    split_value = sorted_errors[max_gap_idx]
    lower_errors = [e for e in valid_errors if e <= split_value]
    upper_errors = [e for e in valid_errors if e > split_value]

    if not lower_errors or not upper_errors:
        return False, None, None

    # Adaptive threshold based on split imbalance
    # If one side has very few points (outliers), use a lower threshold
    n_lower = len(lower_errors)
    n_upper = len(upper_errors)
    n_total = len(valid_errors)

    # Calculate imbalance ratio (0 = balanced, 1 = maximally imbalanced)
    imbalance = abs(n_lower - n_upper) / n_total

    # For extreme outliers (1-3 points on one side vs many on other), use lower threshold
    min_side = min(n_lower, n_upper)
    if min_side <= 3 and imbalance > 0.7:
        # Outlier case: use 5 orders of magnitude as threshold
        effective_threshold = 5
    elif min_side <= 5 and imbalance > 0.5:
        # Moderate outlier case: use 7 orders
        effective_threshold = 7
    else:
        # Balanced split: use the provided threshold
        effective_threshold = gap_threshold_orders

    # Check if gap is large enough to warrant a break
    if max_gap < effective_threshold:
        return False, None, None

    # For outlier case, we allow even single points on one side
    # For balanced splits, require at least 2 points on each side
    if min_side < 1:
        return False, None, None
    if imbalance <= 0.5 and (n_lower < 2 or n_upper < 2):
        return False, None, None

    # Calculate ranges with some padding (in log space)
    lower_min = min(lower_errors) / 10  # One order of magnitude below
    lower_max = max(lower_errors) * 10  # One order of magnitude above
    upper_min = min(upper_errors) / 10
    upper_max = max(upper_errors) * 10

    # Final sanity check: ensure the ranges don't overlap after padding
    # If they do, it means the "gap" isn't really significant visually
    if lower_max >= upper_min:
        return False, None, None

    return True, (lower_min, lower_max), (upper_min, upper_max)

# Timing metrics to plot (5 profiler metrics in microseconds)
# Shape names used in row-per-shape CSV format
SHAPE_NAMES = ['single_tile', '8_tiles', '256_tiles', 'height_sharded', 'yolov4']

# Legacy column names for multi-column CSV format (for backward compatibility)
TIMING_METRICS = [
    'time_profiler_single_tile_us',
    'time_profiler_8_tiles_us',
    'time_profiler_256_tiles_us',
    'time_profiler_height_sharded_us',
    'time_profiler_yolov4_us'
]

METRIC_LABELS = [
    'Single Tile',
    '8 Tiles',
    '256 Tiles',
    'Height Sharded',
    'YOLOv4'
]

# Tufte-inspired minimalist style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['grid.alpha'] = 0.15
plt.rcParams['grid.linewidth'] = 0.5

def load_activation_data(arch_prefix, activation, error_metric='mae', segmentation=None, depth_filter=None, precision_filter=None):
    """
    Load data for a specific activation from data/<arch>/polynomial/<activation>.csv
    and data/<arch>/rational/<activation>.csv

    Supports two CSV formats:
    1. Row-per-shape format (new/embedded): shape column with time_profiler_us, mae, max_error, max_ulp
    2. Multi-column format (legacy): time_profiler_<shape>_us, mae_<shape>, etc.

    Returns:
        dict: {timing_metric: {degree: {depth: {segmentation: {'runtime_us': float, 'error': float}}}}}
    """
    # Try both polynomial and rational directories
    csv_paths = [
        f'data/{arch_prefix}/polynomial/{activation}.csv',
        f'data/{arch_prefix}/rational/{activation}.csv'
    ]

    # Find first existing path
    csv_path = None
    for path in csv_paths:
        if os.path.exists(path):
            csv_path = path
            break

    if csv_path is None:
        return None

    # Map error metric to CSV column name (row-per-shape format)
    error_column_map = {
        'mae': 'mae',
        'rmse': 'rmse',
        'max': 'max_error',
        'mean_rel': 'mean_rel_error',
        'ulp_max': 'max_ulp',
        'ulp_mean': 'mean_ulp',
        'ulp_median': 'median_ulp',
        'ulp_p99': 'p99_ulp'
    }
    error_column = error_column_map.get(error_metric, 'mae')

    data_by_metric = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Skip failed tests
            if row.get('status') != 'pass':
                continue

            # Apply precision filter
            if precision_filter and row.get('precision', '').lower() != precision_filter.lower():
                continue
            if segmentation and row.get('segmentation') != segmentation:
                continue

            # Handle both polynomial and rational CSV formats
            if 'degree' in row and row.get('degree'):
                degree = int(row.get('degree', 0))
                depth = int(row.get('depth', 0))
            elif 'num_degree' in row and 'den_degree' in row:
                num_deg = row.get('num_degree', '0')
                den_deg = row.get('den_degree', '0')
                degree = f"{num_deg}/{den_deg}"
                depth = int(row.get('segments', 0))
            else:
                continue

            # Apply depth filter
            if depth_filter is not None and depth != depth_filter:
                continue

            seg = row.get('segmentation', 'unkn')

            # Row-per-shape format: one row per (config, shape)
            shape = row.get('shape', '')
            if shape not in SHAPE_NAMES:
                continue

            runtime_us = float(row.get('time_profiler_us', 0))
            if runtime_us <= 0:
                continue

            error = float(row.get(error_column, 0))
            if error == 0.0:
                error = 1e-30

            # Map shape to timing metric name for consistency
            timing_metric = f'time_profiler_{shape}_us'

            if timing_metric not in data_by_metric:
                data_by_metric[timing_metric] = {}
            if degree not in data_by_metric[timing_metric]:
                data_by_metric[timing_metric][degree] = {}
            if depth not in data_by_metric[timing_metric][degree]:
                data_by_metric[timing_metric][degree][depth] = {}

            data_by_metric[timing_metric][degree][depth][seg] = {
                'runtime_us': runtime_us,
                'error': error
            }

    return data_by_metric

def plot_activation_error_vs_runtime(arch_prefix, activation, error_metric='mae', segmentation=None, precision=None, skip_best=False, best_only=False, use_broken_axis=False, gap_threshold=10, shape_filter=None, standalone_shapes=None):
    """
    Generate a plot for a single activation with subplots (one per timing metric/shape).

    Args:
        shape_filter: If specified, only plot this shape (single_tile, 8_tiles, 256_tiles, height_sharded, yolov4)
        standalone_shapes: List of shapes to generate standalone plots for (default: all shapes)
    """
    data = None if best_only else load_activation_data(arch_prefix, activation, error_metric=error_metric, segmentation=segmentation, precision_filter=precision)
    best_data = None if skip_best else load_best_data(arch_prefix, activation, error_metric=error_metric, precision_filter=precision)
    sfpu_fast_data = load_sfpu_data(arch_prefix, activation, error_metric=error_metric, precision_filter=precision, sfpu_mode_filter='fast')
    sfpu_precise_data = load_sfpu_data(arch_prefix, activation, error_metric=error_metric, precision_filter=precision, sfpu_mode_filter='precise')

    if not data and not best_data and not sfpu_fast_data and not sfpu_precise_data:
        return f"⊘ Skipped {activation} (no data)"

    # Determine y-axis label based on error metric
    ylabel_map = {
        'mae': 'Mean Absolute Error (MAE)',
        'rmse': 'Root Mean Squared Error (RMSE)',
        'max': 'Max Error',
        'mean_rel': 'Mean Relative Error',
        'ulp_max': 'Max ULP Error',
        'ulp_mean': 'Mean ULP Error',
        'ulp_median': 'Median ULP Error',
        'ulp_p99': 'P99 ULP Error'
    }
    ylabel = ylabel_map.get(error_metric, 'Error')

    # Filter shapes if shape_filter is specified
    if shape_filter:
        # Map shape name to index
        shape_to_idx = {name: idx for idx, name in enumerate(SHAPE_NAMES)}
        if shape_filter not in shape_to_idx:
            return f"⊘ Skipped {activation} (invalid shape: {shape_filter})"
        shape_idx = shape_to_idx[shape_filter]
        timing_metrics_filtered = [TIMING_METRICS[shape_idx]]
        metric_labels_filtered = [METRIC_LABELS[shape_idx]]
        num_subplots = 1
        fig, axes = plt.subplots(1, 1, figsize=(5.0, 4.5))
        axes = [axes]  # Make iterable
    else:
        timing_metrics_filtered = TIMING_METRICS
        metric_labels_filtered = METRIC_LABELS
        num_subplots = 5
        fig, axes = plt.subplots(1, 5, figsize=(18.0, 4.5))

    has_any_data = False

    # Color palette for degrees (auto-assign colors to any degree)
    from matplotlib import cm
    import numpy as np

    # Define explicit colors for common degrees, auto-generate for others
    explicit_degree_colors = {
        1: '#90C695',   # Linear - green
        2: '#E27A7A',   # Quadratic - red
        3: '#B4A7D6',   # Cubic - purple
        4: '#F4A460',   # Quartic - sandy brown
        5: '#6FA8DC',   # Quintic - light blue
        6: '#87B5C4',   # Hexic - blue
        7: '#E6B8AF',   # Septic - light pink
        8: '#D4A5B8',   # Octic - pink
        10: '#98D8C8',  # Decic - turquoise
        12: '#F7DC6F',  # Dodecic - yellow
        14: '#BB8FCE',  # Tetradecic - lavender
        16: '#A4C2F4',  # Hexadecic - sky blue
    }

    explicit_degree_markers = {
        1: 's',  # Square
        2: 'D',  # Diamond
        3: '^',  # Triangle up
        4: 'o',  # Circle
        5: 'v',  # Triangle down
        6: '<',  # Triangle left
        7: '>',  # Triangle right
        8: 'p',  # Pentagon
        10: 'h',  # Hexagon
        12: '*',  # Star
        14: 'X',  # X
        16: 'P',  # Plus
    }

    # Function to get color for any degree
    def get_degree_color(degree):
        if degree in explicit_degree_colors:
            return explicit_degree_colors[degree]
        # Auto-generate color from colormap
        # Handle both integer and string (rational) degrees
        if isinstance(degree, str):
            # For rational degrees like "1/1", hash to get a color index
            degree_hash = hash(degree) % 20
            return cm.tab20(degree_hash)
        return cm.tab20(degree % 20)

    def get_degree_marker(degree):
        if degree in explicit_degree_markers:
            return explicit_degree_markers[degree]
        # For rational degrees (strings), use default marker
        return 'o'  # Default circle

    depth_sizes = {4: 40, 8: 60, 16: 90, 32: 130}  # Increased sizes for better visibility

    for metric_idx, (timing_metric, metric_label) in enumerate(zip(timing_metrics_filtered, metric_labels_filtered)):
        ax = axes[metric_idx]

        has_data = False

        # Track actual data min/max for annotation positioning
        all_errors = []

        # Per-subplot legend tracking
        subplot_legend_handles = []
        subplot_legend_labels = []

        # Plot TTNN-fast data if available (yellow star)
        if sfpu_fast_data and timing_metric in sfpu_fast_data:
            sfpu_point = sfpu_fast_data[timing_metric]

            # Yellow filled circle (larger, underneath)
            circle = ax.scatter(sfpu_point['runtime_us'], sfpu_point['error'],
                            marker='o', s=300, color='#FFEB3B',
                            edgecolors='none', alpha=1.0, zorder=9)

            # Star marker on top
            star = ax.scatter(sfpu_point['runtime_us'], sfpu_point['error'],
                            marker='*', s=250, color='#888888',
                            edgecolors='none', linewidth=0, alpha=0.95, zorder=10,
                            label='TTNN-fast')

            # Add to subplot legend
            subplot_legend_handles.append(star)
            subplot_legend_labels.append('TTNN-fast')

            all_errors.append(sfpu_point['error'])
            has_data = True
            has_any_data = True

        # Plot TTNN-precise data if available (blue star)
        if sfpu_precise_data and timing_metric in sfpu_precise_data:
            sfpu_point = sfpu_precise_data[timing_metric]

            # Blue filled circle (larger, underneath)
            circle = ax.scatter(sfpu_point['runtime_us'], sfpu_point['error'],
                            marker='o', s=300, color='#64B5F6',
                            edgecolors='none', alpha=1.0, zorder=9)

            # Star marker on top
            star = ax.scatter(sfpu_point['runtime_us'], sfpu_point['error'],
                            marker='*', s=250, color='#1976D2',
                            edgecolors='none', linewidth=0, alpha=0.95, zorder=10,
                            label='TTNN-precise')

            # Add to subplot legend
            subplot_legend_handles.append(star)
            subplot_legend_labels.append('TTNN-precise')

            all_errors.append(sfpu_point['error'])
            has_data = True
            has_any_data = True

        # Plot polynomial data if available
        min_error_point = None
        min_error_value = float('inf')

        if data and timing_metric in data:
            metric_data = data[timing_metric]

            # First pass: plot all points and track minimum error
            for degree, depth_data in metric_data.items():
                # No filtering - plot ALL degrees!
                color = get_degree_color(degree)
                marker = get_degree_marker(degree)

                for depth, seg_data in depth_data.items():
                    for seg, point in seg_data.items():
                        size = depth_sizes.get(depth, 40)
                        if isinstance(degree, str) and '/' in degree:
                            label = f"n{degree.replace('/', 'd')}_s{depth}_{seg[:4]}"
                        else:
                            label = f'p{degree}_s{depth}_{seg[:4]}'

                        scatter = ax.scatter(point['runtime_us'], point['error'],
                                  marker=marker, s=size, color=color,
                                  label=label,
                                  edgecolors='black', linewidth=0.8, alpha=0.85, zorder=5)

                        # Track minimum error point
                        if point['error'] < min_error_value:
                            min_error_value = point['error']
                            min_error_point = (point['runtime_us'], point['error'], degree, depth)

                        # Track all errors for annotation positioning
                        all_errors.append(point['error'])

                    # Skip adding polynomial sweep data to legend (too cluttered)
                    # Only show SFPU and best configs in per-subplot legend

                    has_data = True
                    has_any_data = True


        # Plot best configuration data if available (with colored pentagon markers)
        if best_data and timing_metric in best_data:
            metric_data = best_data[timing_metric]

            # Define colors for each best type - clearly distinguishable
            best_colors = {
                'MAE': '#FF8C00',      # Dark orange for Best MAE
                'Max': '#DC143C',      # Crimson red for Best Max
                'ULP': '#00CED1',      # Dark turquoise for Best ULP
                'MAE/Max': '#FF1493',  # Deep pink for MAE+Max
                'MAE/ULP': '#32CD32',  # Lime green for MAE+ULP
                'Max/ULP': '#9370DB',  # Medium purple for Max+ULP
                'MAE/Max/ULP': '#FFD700'  # Gold for all three
            }

            # Plot points for each degree and depth
            for degree, depth_data in metric_data.items():
                for depth, point in depth_data.items():
                    # Include what this config was best for in the label
                    best_for = point.get('best_for', '')

                    # Format label - all configs shown here are best for THIS timing
                    segmentation = point.get('segmentation', '')
                    seg_abbrev = segmentation[:4] if segmentation else 'unkn'  # First 4 chars
                    if isinstance(degree, str) and '/' in degree:
                        # Rational approximation (e.g., "3/3")
                        if best_for:
                            label = f"Best {best_for}: n{degree.replace('/', 'd')}_s{depth}_{seg_abbrev}"
                        else:
                            label = f"Best: n{degree.replace('/', 'd')}_s{depth}_{seg_abbrev}"
                    else:
                        # Polynomial (e.g., "5")
                        if best_for:
                            label = f'Best {best_for}: p{degree}_s{depth}_{seg_abbrev}'
                        else:
                            label = f'Best: p{degree}_s{depth}_{seg_abbrev}'

                    # Get color based on best_for type
                    color = best_colors.get(best_for, '#FF8C00')

                    # Pentagon marker with black edge and type-specific color
                    scatter = ax.scatter(point['runtime_us'], point['error'],
                              marker='p', s=120, color=color,
                              label=label,
                              edgecolors='black', linewidth=1.2, alpha=0.95, zorder=8)

                    # Track all errors for annotation positioning
                    all_errors.append(point['error'])

                    # Add to subplot legend (show best configs)
                    if label not in subplot_legend_labels:
                        subplot_legend_handles.append(scatter)
                        subplot_legend_labels.append(label)

                    has_data = True
                    has_any_data = True

        if not has_data:
            ax.set_visible(False)
            continue

        # Find SFPU runtime and minimum error configuration runtime for speedup annotation
        sfpu_runtime = None
        min_error_runtime = float('inf')
        min_error = float('inf')

        # Use TTNN-precise runtime for speedup comparison (always compare to precise baseline)
        if sfpu_precise_data and timing_metric in sfpu_precise_data:
            sfpu_runtime = sfpu_precise_data[timing_metric]['runtime_us']

        # Find the configuration with the absolute minimum error (from ALL polynomial data + best data)
        # When errors tie, prefer the faster runtime
        if data and timing_metric in data:
            for degree, depth_data in data[timing_metric].items():
                for depth, seg_data in depth_data.items():
                    for seg, point in seg_data.items():
                        if point['error'] < min_error or (point['error'] == min_error and point['runtime_us'] < min_error_runtime):
                            min_error = point['error']
                            min_error_runtime = point['runtime_us']

        if best_data and timing_metric in best_data:
            for degree, depth_data in best_data[timing_metric].items():
                for depth, point in depth_data.items():
                    if point['error'] < min_error or (point['error'] == min_error and point['runtime_us'] < min_error_runtime):
                        min_error = point['error']
                        min_error_runtime = point['runtime_us']

        # Draw speedup annotation comparing SFPU to minimum error configuration
        if sfpu_runtime is not None and min_error_runtime < float('inf') and sfpu_runtime > 0 and min_error_runtime > 0 and all_errors:
            # Calculate speedup
            speedup = min_error_runtime / sfpu_runtime

            # Draw vertical lines at SFPU and min error config runtimes (full height)
            ax.axvline(x=sfpu_runtime, color='black', linestyle='--', linewidth=1.0, alpha=0.5, zorder=1)
            ax.axvline(x=min_error_runtime, color='black', linestyle='--', linewidth=1.0, alpha=0.5, zorder=1)

            # Position horizontal speedup arrow - will be positioned later after we know sfpu_error
            import numpy as np
            x_mid = (sfpu_runtime + min_error_runtime) / 2

            # Format the speedup text
            if speedup < 1.0:
                # LUT is slower than SFPU - show as negative percentage
                percentage = (1.0 - speedup) * 100
                speedup_text = f'-{percentage:.0f}%'
            elif speedup < 2.0:
                # LUT is slightly faster - show as percentage
                percentage = (speedup - 1.0) * 100
                speedup_text = f'+{percentage:.0f}%'
            else:
                # LUT is significantly faster - show as multiplier
                speedup_text = f'{speedup:.1f}x'

            # Store for later positioning (after we determine sfpu_error)
            speedup_annotation_data = {
                'x_mid': x_mid,
                'sfpu_runtime': sfpu_runtime,
                'min_error_runtime': min_error_runtime,
                'speedup_text': speedup_text
            }

        # Draw error improvement annotation (vertical arrow showing error gap)
        # Always compare to TTNN-precise baseline
        sfpu_error = None
        if sfpu_precise_data and timing_metric in sfpu_precise_data:
            sfpu_error = sfpu_precise_data[timing_metric]['error']

        if sfpu_error is not None:
            # Find minimum error from LUT sweep
            if min_error < float('inf') and min_error > 0 and sfpu_error > 0:
                error_improvement = sfpu_error / min_error

                # Position arrow at SFPU runtime location (offset to right to avoid hiding y-axis)
                if sfpu_runtime is not None and np.isfinite(sfpu_runtime):
                    # Draw horizontal dotted red lines at SFPU error and min error levels
                    ax.axhline(y=sfpu_error, color='red', linestyle='--', linewidth=1.0, alpha=0.5, zorder=1)
                    ax.axhline(y=min_error, color='red', linestyle='--', linewidth=1.0, alpha=0.5, zorder=1)

                    # Position arrow exactly at SFPU runtime (on top of left vertical dotted line)
                    arrow_x_pos = sfpu_runtime

                    # Draw vertical arrow from min_error to sfpu_error
                    ax.annotate('', xy=(arrow_x_pos, sfpu_error), xytext=(arrow_x_pos, min_error),
                               arrowprops=dict(arrowstyle='<->', color='red', lw=2, shrinkA=0, shrinkB=0),
                               zorder=100)

                    # Position text label centered on the arrow
                    y_mid = np.sqrt(sfpu_error * min_error)  # Geometric mean in log space
                    # Format the error improvement text
                    if error_improvement < 1.0:
                        # LUT is worse than SFPU - show as negative percentage
                        percentage = (1.0 - error_improvement) * 100
                        error_text = f'-{percentage:.0f}%'
                    elif error_improvement < 2.0:
                        # LUT is slightly better - show as percentage
                        percentage = (error_improvement - 1.0) * 100
                        error_text = f'+{percentage:.0f}%'
                    else:
                        # LUT is significantly better - show as multiplier
                        error_text = f'{error_improvement:.1f}x'
                    ax.text(arrow_x_pos, y_mid, error_text,
                           ha='center', va='center', fontsize=9, color='red', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red',
                                   alpha=0.95, linewidth=1.2), zorder=101, clip_on=True)

                    # Draw black horizontal speedup arrow at the top red line (sfpu_error level)
                    if 'speedup_annotation_data' in locals():
                        anno_data = speedup_annotation_data
                        # Position at sfpu_error level (top red line)
                        ax.annotate('', xy=(anno_data['min_error_runtime'], sfpu_error),
                                   xytext=(anno_data['sfpu_runtime'], sfpu_error),
                                   arrowprops=dict(arrowstyle='<->', color='black', lw=2, shrinkA=0, shrinkB=0),
                                   zorder=100)
                        ax.text(anno_data['x_mid'], sfpu_error, anno_data['speedup_text'],
                               ha='center', va='center', fontsize=9, color='black', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='black',
                                       alpha=0.95, linewidth=1.2), zorder=101, clip_on=True)

        # Style the subplot
        ax.set_xlabel(f'{metric_label} (μs)', fontsize=10, color='#333333')
        if metric_idx == 0:
            ax.set_ylabel(ylabel, fontsize=10, color='#333333')
        ax.set_title(metric_label, fontsize=11, color='#333333', fontweight='bold', pad=12)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.tick_params(axis='both', labelsize=9)
        ax.grid(True, alpha=0.12, linestyle='-', linewidth=0.5, which='major')

        # Add padding to x-axis to prevent clipping datapoints at edges
        if has_data:
            xlim = ax.get_xlim()
            x_range = xlim[1] - xlim[0]
            ax.set_xlim(xlim[0] - 0.05 * x_range, xlim[1] + 0.15 * x_range)  # 5% left, 15% right padding

        # Add per-subplot legend (above subplot title, horizontal)
        if subplot_legend_handles:
            ax.legend(subplot_legend_handles, subplot_legend_labels,
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.20),  # Above the subplot title
                     fontsize=6,
                     framealpha=0.9,
                     edgecolor='none',
                     fancybox=False,
                     markerscale=0.6,
                     handlelength=0.8,
                     handletextpad=0.2,
                     columnspacing=0.5,
                     ncol=len(subplot_legend_labels))  # Single row (all items horizontal)

    if not has_any_data:
        plt.close()
        return f"⊘ Skipped {activation} (no valid data points)"

    # Adjust layout with space at top for per-subplot legends
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave 5% at top for legends

    # Save to plots/<arch>/error_vs_runtime/<activation>/
    output_dir = f'plots/{arch_prefix}/error_vs_runtime/{activation}'
    os.makedirs(output_dir, exist_ok=True)

    # Build filename with optional precision and shape filters
    filename_parts = [error_metric, 'vs', 'runtime']
    if shape_filter:
        filename_parts.append(shape_filter)
    if precision:
        filename_parts.append(precision.lower())
    output_path = f'{output_dir}/{"_".join(filename_parts)}.png'

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # Also generate standalone plots for each timing configuration (skip if shape_filter is specified)
    standalone_results = None
    if not shape_filter:
        standalone_results = generate_standalone_plots(arch_prefix, activation, error_metric, precision,
                                                        data, best_data, sfpu_fast_data, sfpu_precise_data,
                                                        ylabel, best_only=best_only,
                                                        use_broken_axis=use_broken_axis, gap_threshold=gap_threshold,
                                                        standalone_shapes=standalone_shapes)

    return f"✓ Generated {output_path}" + (f"\n    {standalone_results}" if standalone_results else "")

def generate_standalone_plots(arch_prefix, activation, error_metric, precision, data, best_data,
                               sfpu_fast_data, sfpu_precise_data, ylabel, best_only=False,
                               use_broken_axis=False, gap_threshold=10, standalone_shapes=None):
    """
    Generate individual standalone plots for each timing configuration.
    Uses Tufte-inspired minimal styling with color-coding by metric type.

    Args:
        standalone_shapes: List of shapes to generate plots for (default: all shapes)

    Returns a status string.
    """
    from matplotlib import cm
    import numpy as np

    # All LUT sweep data is green
    data_color = '#2ECC71'  # Green for all LUT datapoints

    # Metric labels
    all_metric_labels = {
        'time_profiler_single_tile_us': 'Single Tile',
        'time_profiler_8_tiles_us': '8 Tiles',
        'time_profiler_256_tiles_us': '256 Tiles',
        'time_profiler_height_sharded_us': 'Height Sharded',
        'time_profiler_yolov4_us': 'YOLOv4'
    }

    # Filter metric_labels based on standalone_shapes
    if standalone_shapes:
        metric_labels = {}
        for shape in standalone_shapes:
            timing_metric = f'time_profiler_{shape}_us'
            if timing_metric in all_metric_labels:
                metric_labels[timing_metric] = all_metric_labels[timing_metric]
    else:
        metric_labels = all_metric_labels

    # Create hierarchical directory structure: standalone/<metric>/
    base_output_dir = f'plots/{arch_prefix}/error_vs_runtime/{activation}/standalone/{error_metric}'
    os.makedirs(base_output_dir, exist_ok=True)

    # Apply Tufte styling (matching plot_hardware_error_analysis.py)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['grid.color'] = '#CCCCCC'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'

    count = 0

    # Define colors for best configurations
    best_colors = {
        'MAE': '#FF8C00',      # Dark orange for Best MAE
        'Max': '#DC143C',      # Crimson red for Best Max
        'ULP': '#00CED1',      # Dark turquoise for Best ULP
        'MAE/Max': '#FF1493',  # Deep pink for MAE+Max
        'MAE/ULP': '#32CD32',  # Lime green for MAE+ULP
        'Max/ULP': '#9370DB',  # Medium purple for Max+ULP
        'MAE/Max/ULP': '#FFD700'  # Gold for all three
    }

    # Map error_metric to display label for "Best X" annotations
    error_metric_to_label = {
        'mae': 'MAE',
        'max': 'Max',
        'ulp_max': 'ULP',
        'ulp_mean': 'ULP'
    }
    current_metric_label = error_metric_to_label.get(error_metric, error_metric.upper())

    # Marker shapes by depth (segments)
    depth_markers = {
        1: '.',    # Point
        2: 'o',    # Circle
        3: 'o',    # Circle (same as 2)
        4: 's',    # Square
        8: '^',    # Triangle up
        16: 'D',   # Diamond
        32: 'v',   # Triangle down
        64: 'p',   # Pentagon
        128: 'h',  # Hexagon
    }

    # Marker sizes by degree (larger degree = larger marker)
    def get_degree_size(degree):
        if isinstance(degree, str) and '/' in degree:
            # Rational: sum of numerator and denominator
            parts = degree.split('/')
            deg = int(parts[0]) + int(parts[1])
        else:
            deg = int(degree) if degree else 1
        # Scale: degree 1 = 40, degree 16 = 200
        return 40 + deg * 10

    for timing_metric, timing_label in metric_labels.items():
        # First pass: collect all data to determine if broken axis is needed
        all_errors = []
        plot_data = {
            'sweep': [],  # List of dicts with runtime, error, degree, depth, segmentation
            'sfpu_fast': None,
            'sfpu_precise': None,
            'best': []
        }

        # Collect polynomial/rational sweep data
        if not best_only and data and timing_metric in data:
            metric_data = data[timing_metric]
            for degree, depth_data in sorted(metric_data.items()):
                for depth, seg_data in depth_data.items():
                    for seg, point in seg_data.items():
                        if point and isinstance(point, dict):
                            plot_data['sweep'].append({
                                'runtime': point['runtime_us'],
                                'error': point['error'],
                                'degree': degree,
                                'depth': depth,
                                'segmentation': seg
                            })
                            all_errors.append(point['error'])

        # Collect SFPU fast data
        if sfpu_fast_data and timing_metric in sfpu_fast_data:
            point = sfpu_fast_data[timing_metric]
            plot_data['sfpu_fast'] = point
            all_errors.append(point['error'])

        # Collect SFPU precise data
        if sfpu_precise_data and timing_metric in sfpu_precise_data:
            point = sfpu_precise_data[timing_metric]
            plot_data['sfpu_precise'] = point
            all_errors.append(point['error'])

        # Collect best configurations
        if best_data and timing_metric in best_data:
            metric_data = best_data[timing_metric]
            for degree, depth_data in metric_data.items():
                for depth, point in depth_data.items():
                    best_for = point.get('best_for', '')
                    if not best_for:
                        continue
                    segmentation = point.get('segmentation', '')
                    seg_abbrev = segmentation[:4] if segmentation else 'unkn'
                    if isinstance(degree, str) and '/' in degree:
                        config_str = f"n{degree.replace('/', 'd')}_s{depth}_{seg_abbrev}"
                    else:
                        config_str = f'p{degree}_s{depth}_{seg_abbrev}'
                    # Use the current error metric for the label, not the pre-computed best_for
                    label = f'LUT: {config_str}'
                    color = best_colors.get(current_metric_label, '#FF8C00')
                    plot_data['best'].append({
                        'runtime': point['runtime_us'],
                        'error': point['error'],
                        'label': label,
                        'color': color
                    })
                    all_errors.append(point['error'])

        if not all_errors:
            continue

        # Determine if broken axis is needed
        needs_break = False
        lower_range = None
        upper_range = None

        # No broken axis - use simple clipped axis at 1e3
        # TTNN baselines will be shown as arrows pointing off-chart
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        is_broken = False

        has_data = False
        legend_handles = []
        legend_labels = []
        text_annotations = []  # Collect for adjustText

        # Helper to extract scatter/plot handle (brokenaxes returns list, regular axes returns single)
        def get_handle(result):
            if isinstance(result, list):
                return result[0] if result else None
            return result

        # Plot polynomial/rational sweep data with shape/size encoding
        # Shape = depth (segments), Size = degree
        if plot_data['sweep']:
            # Group points by (depth, degree) for efficient plotting
            from collections import defaultdict
            grouped_points = defaultdict(list)
            for p in plot_data['sweep']:
                key = (p.get('depth', 8), p.get('degree', 1))
                grouped_points[key].append(p)

            legend_added = False
            for (depth, degree), points in grouped_points.items():
                runtimes = [p['runtime'] for p in points]
                errors = [p['error'] for p in points]
                marker = depth_markers.get(depth, 'o')
                size = get_degree_size(degree)

                scatter = ax.scatter(
                    runtimes, errors,
                    marker=marker, s=size, c=data_color, alpha=0.6,
                    edgecolors='#666666', linewidth=0.5, zorder=5
                )
                if not legend_added and 'LUT Sweep' not in legend_labels:
                    legend_handles.append(get_handle(scatter))
                    legend_labels.append('LUT Sweep')
                    legend_added = True
            has_data = True

        # Compute and plot Pareto frontier as thick red curve
        pareto_points = compute_pareto_frontier(plot_data['sweep'])
        best_in_slowdown_region = None  # Will track best error within 30% slowdown
        slowdown_threshold_runtime = None  # Will be set if SFPU precise is available

        if pareto_points and len(pareto_points) >= 2:
            # Sort Pareto points by runtime for line plotting
            pareto_sorted = sorted(pareto_points, key=lambda p: p['runtime'])

            # Draw smooth line connecting Pareto points
            pareto_runtimes = [p['runtime'] for p in pareto_sorted]
            pareto_errors = [p['error'] for p in pareto_sorted]

            # Plot straight line connecting Pareto points
            pareto_line = ax.plot(pareto_runtimes, pareto_errors, color='#CC0000', linewidth=3.0,
                                  alpha=0.9, zorder=8, label='Pareto Frontier')
            if 'Pareto Frontier' not in legend_labels:
                # Create a proxy artist for legend (works with both regular axes and brokenaxes)
                from matplotlib.lines import Line2D
                pareto_proxy = Line2D([0], [0], color='#CC0000', linewidth=4.0, solid_capstyle='round')
                legend_handles.append(pareto_proxy)
                legend_labels.append('Pareto Frontier')

            # Plot individual Pareto points with markers based on depth and sized by degree
            for pt in pareto_sorted:
                depth = pt.get('depth', 8)
                degree = pt.get('degree', 1)
                marker = depth_markers.get(depth, 'o')
                size = get_degree_size(degree)

                ax.scatter(pt['runtime'], pt['error'],
                          marker=marker, s=size, color='#CC0000',
                          edgecolors='black', linewidth=2.0, alpha=0.95, zorder=9)


        # Y-axis clip threshold
        y_clip = 1e3

        # Helper to format error as 10^N
        def format_exp(err):
            import math
            if err <= 0:
                return '0'
            exp = int(math.log10(err))
            return f'$10^{{{exp}}}$'

        # Helper to format error value
        def format_err(err):
            if err < 0.01:
                return f'{err:.4f}'
            elif err < 1:
                return f'{err:.2f}'
            elif err < 1000:
                return f'{err:.1f}'
            else:
                return format_exp(err)

        # Collect all in-chart annotations to position later
        pending_annotations = []  # (x, y, label, color)

        # Plot SFPU fast data - clip to y_clip if above threshold
        # Skip if TTNN-fast is identical to TTNN-precise (same runtime and error)
        show_sfpu_fast = False
        if plot_data['sfpu_fast']:
            fast_pt = plot_data['sfpu_fast']
            prec_pt = plot_data['sfpu_precise']
            # Check if they're different enough to show both
            if prec_pt is None:
                show_sfpu_fast = True
            else:
                runtime_diff = abs(fast_pt['runtime_us'] - prec_pt['runtime_us'])
                error_ratio = fast_pt['error'] / max(prec_pt['error'], 1e-30) if prec_pt['error'] > 0 else 1.0
                # Show fast if runtime differs by >1μs OR error differs by >10%
                if runtime_diff > 1.0 or abs(error_ratio - 1.0) > 0.1:
                    show_sfpu_fast = True

        # Minimum error threshold for plotting (below this is considered essentially 0)
        y_min_threshold = 0.1

        if show_sfpu_fast and plot_data['sfpu_fast']:
            point = plot_data['sfpu_fast']
            actual_err = point['error']
            # Clip to visible range: [y_min_threshold, y_clip * 0.9]
            plot_y = max(y_min_threshold, min(actual_err, y_clip * 0.9))
            is_clipped_high = actual_err > y_clip
            is_clipped_low = actual_err < y_min_threshold

            ax.scatter(point['runtime_us'], plot_y,
                      marker='*', s=120, color='#FFD700',
                      edgecolors='none', linewidth=0, alpha=0.95, zorder=10)

            # Label shows actual error (with arrow pointing up/down if clipped)
            err_str = format_err(actual_err)
            arrow_indicator = ' ↑' if is_clipped_high else (' ↓' if is_clipped_low else '')
            label = f'TTNN-fast\n{err_str}' + arrow_indicator
            pending_annotations.append((point['runtime_us'], plot_y, label, '#FF6600'))
            has_data = True

        # Plot SFPU precise data - clip to visible range
        if plot_data['sfpu_precise']:
            point = plot_data['sfpu_precise']
            actual_err = point['error']
            # Clip to visible range: [y_min_threshold, y_clip * 0.9]
            plot_y = max(y_min_threshold, min(actual_err, y_clip * 0.9))
            is_clipped_high = actual_err > y_clip
            is_clipped_low = actual_err < y_min_threshold

            ax.scatter(point['runtime_us'], plot_y,
                      marker='*', s=150, color='#1F77B4',
                      edgecolors='none', linewidth=0, alpha=0.95, zorder=10)

            # Label shows actual error (with arrow pointing up/down if clipped)
            err_str = format_err(actual_err)
            arrow_indicator = ' ↑' if is_clipped_high else (' ↓' if is_clipped_low else '')
            label = f'TTNN\n{err_str}' + arrow_indicator
            pending_annotations.append((point['runtime_us'], plot_y, label, '#1F77B4'))
            has_data = True

        # Highlight best configurations with green color
        best_highlight_color = '#228B22'

        for best_point in plot_data['best']:
            matching_sweep = None
            for p in plot_data['sweep']:
                if abs(p['runtime'] - best_point['runtime']) < 0.1 and abs(p['error'] - best_point['error']) / max(p['error'], 1e-10) < 0.01:
                    matching_sweep = p
                    break

            if matching_sweep:
                depth = matching_sweep.get('depth', 8)
                degree = matching_sweep.get('degree', 1)
                size = get_degree_size(degree) * 2.5
            else:
                size = 150
                depth = 8
                degree = best_point.get('degree', '?')

            # Plot the marker (needed for axis limits)
            ax.scatter(
                best_point['runtime'], best_point['error'],
                marker='o', s=size, color=best_highlight_color,
                edgecolors='none', linewidth=0, alpha=1.0, zorder=14
            )

            label_text = best_point['label'].replace('Best ', '')
            err = best_point['error']
            if err < 0.01:
                err_str = f'{err:.4f}'
            elif err < 1:
                err_str = f'{err:.2f}'
            elif err < 1000:
                err_str = f'{err:.1f}'
            else:
                err_str = f'{err:.1e}'
            label_text = f'{label_text}\n{err_str}'
            pending_annotations.append((best_point['runtime'], best_point['error'], label_text, best_highlight_color))
            has_data = True

        # Add legend entry for best configuration if any were plotted
        if plot_data['best'] and 'Best Config' not in legend_labels:
            from matplotlib.lines import Line2D
            best_proxy = Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6600',
                               markeredgecolor='black', markersize=10, linewidth=0)
            legend_handles.append(best_proxy)
            legend_labels.append('Best Config')

        if not has_data:
            plt.close()
            continue

        # Add right-angle connection from TTNN-precise to best LUT point
        sfpu_runtime = None
        sfpu_error = None
        min_error_runtime = float('inf')
        min_error = float('inf')
        right_angle_data = None

        # Get SFPU runtime/error (always use precise as baseline)
        if plot_data['sfpu_precise']:
            sfpu_runtime = plot_data['sfpu_precise']['runtime_us']
            sfpu_error = plot_data['sfpu_precise']['error']

        # Find minimum error point from sweep or best data
        # When errors tie, prefer the faster runtime
        for pt in plot_data['sweep']:
            if pt['error'] < min_error or (pt['error'] == min_error and pt['runtime'] < min_error_runtime):
                min_error = pt['error']
                min_error_runtime = pt['runtime']
        for pt in plot_data['best']:
            if pt['error'] < min_error or (pt['error'] == min_error and pt['runtime'] < min_error_runtime):
                min_error = pt['error']
                min_error_runtime = pt['runtime']

        # Prepare right-angle connection data if we have both SFPU precise and best LUT data
        if sfpu_runtime is not None and sfpu_error is not None and min_error_runtime < float('inf') and min_error < float('inf'):
            if sfpu_runtime > 0 and min_error_runtime > 0 and sfpu_error > 0 and min_error > 0:
                # Calculate runtime ratio (LUT runtime / SFPU runtime)
                runtime_ratio = min_error_runtime / sfpu_runtime
                # Calculate error ratio (SFPU error / LUT error) - higher means LUT is better
                error_ratio = sfpu_error / min_error

                # Calculate absolute runtime difference
                runtime_diff = min_error_runtime - sfpu_runtime  # positive = LUT slower

                # Format runtime text: percentage + absolute μs for clarity
                if runtime_ratio < 2.0:
                    # Show as percentage change with absolute μs
                    if runtime_ratio < 1.0:
                        # LUT is faster
                        pct = (1.0 - runtime_ratio) * 100
                        runtime_text = f'-{pct:.0f}%\n({abs(runtime_diff):.1f}μs)'
                    else:
                        # LUT is slower
                        pct = (runtime_ratio - 1.0) * 100
                        runtime_text = f'+{pct:.0f}%\n(+{runtime_diff:.1f}μs)'
                elif runtime_ratio < 10.0:
                    # Show as multiplier with absolute μs
                    runtime_text = f'{runtime_ratio:.1f}x\n({min_error_runtime:.1f}μs)'
                else:
                    # Large multiplier - use scientific notation with absolute μs
                    runtime_text = f'{runtime_ratio:.0e}x\n({min_error_runtime:.0f}μs)'

                # Format error text: percentage if <200%, multiplier if <10x, scientific if >10x
                if error_ratio < 2.0:
                    # Show as percentage change
                    if error_ratio < 1.0:
                        # LUT is worse (higher error)
                        pct = (1.0 - error_ratio) * 100
                        error_text = f'+{pct:.0f}%'
                    else:
                        # LUT is better (lower error)
                        pct = (error_ratio - 1.0) * 100
                        error_text = f'-{pct:.0f}%'
                elif error_ratio < 10.0:
                    # Show as multiplier (error reduction)
                    error_text = f'{error_ratio:.1f}x'
                else:
                    # Astronomical - use scientific notation
                    error_text = f'{error_ratio:.0e}x'

                # Clip SFPU error for plotting but keep actual value for annotation
                # Clip both high (above y_clip) AND low (below y_min threshold) values
                # Use 0.1 as minimum since that's our typical y_min for log scale
                y_min_threshold = 0.1
                sfpu_error_clipped = max(y_min_threshold, min(sfpu_error, y_clip * 0.9))

                right_angle_data = {
                    'sfpu_runtime': sfpu_runtime,
                    'sfpu_error': sfpu_error,  # Actual error for ratio text
                    'sfpu_error_plot': sfpu_error_clipped,  # Clipped for drawing
                    'lut_runtime': min_error_runtime,
                    'lut_error': min_error,
                    'runtime_text': runtime_text,
                    'error_text': error_text
                }

        # Configure axes (Tufte style)
        ax.set_xlabel('Runtime (μs)', fontsize=11, color='#333333')
        ax.set_ylabel(ylabel, fontsize=11, color='#333333')

        ax.set_yscale('log')
        # X-axis is linear (runtime values don't need log scale)
        # Set y-axis limits based on actual data (excluding astronomical values)
        # Filter to only include reasonable errors (below y_clip) when computing limits
        reasonable_errors = [e for e in all_errors if 0 < e <= y_clip]
        if reasonable_errors:
            min_error_val = min(reasonable_errors)
            max_error_val = max(reasonable_errors)
            # Set lower limit to one order of magnitude below minimum, but at least 0.1
            y_min = max(0.1, min_error_val / 10)
            # Set upper limit based on actual max data + padding (one order of magnitude above)
            # Only use y_clip if data actually exceeds it
            y_max = min(y_clip, max_error_val * 10)
        else:
            # All errors are astronomical - use a default reasonable range
            y_min = 0.1
            y_max = y_clip
        ax.set_ylim(y_min, y_max)

        # Collect ALL plotted data points for x-axis bounds
        all_x_values = []

        # Add SFPU fast point if exists
        if plot_data['sfpu_fast']:
            all_x_values.append(plot_data['sfpu_fast']['runtime_us'])

        # Add SFPU precise point if exists
        if plot_data['sfpu_precise']:
            all_x_values.append(plot_data['sfpu_precise']['runtime_us'])

        # Add best/LUT points
        for pt in plot_data['best']:
            all_x_values.append(pt['runtime'])

        # Add sweep points
        for pt in plot_data['sweep']:
            all_x_values.append(pt['runtime'])

        # Calculate bounds from actual data
        if all_x_values:
            x_min_data = min(all_x_values)
            x_max_data = max(all_x_values)
        else:
            xlim = ax.get_xlim()
            x_min_data, x_max_data = xlim[0], xlim[1]

        x_range = x_max_data - x_min_data
        x_center = (x_min_data + x_max_data) / 2

        # Make the data span ~50% of the CENTER of the plot
        # Add 25% padding on each side for labels
        if x_range > 0:
            # Data should span 50% of center, with 25% label space on each side
            # So total width = data_range / 0.5 = data_range * 2
            center_range = max(x_range * 2, 6.0)  # At least 6μs for data area
        else:
            center_range = 6.0  # Default for single point

        # Add 25% padding on each side for label boxes
        total_range = center_range * 1.5  # 50% extra total (25% each side)
        half_range = total_range / 2
        new_xlim = (max(0, x_center - half_range), x_center + half_range)
        ax.set_xlim(new_xlim[0], new_xlim[1])

        # Minimal grid (only horizontal lines, subtle)
        ax.grid(True, alpha=0.2, linewidth=0.5, axis='y', linestyle='-')
        ax.grid(False, axis='x')

        # ==========================================================================
        # CLEAN TRIANGLE + LABELS: All labels OUTSIDE triangle, expand axes to fit
        # ==========================================================================
        # Check if triangle is meaningful (both errors should be > 0 and different)
        draw_triangle = False
        if right_angle_data:
            from matplotlib.patches import Polygon
            import numpy as np
            import math

            sfpu_x = right_angle_data['sfpu_runtime']
            sfpu_y_plot = right_angle_data['sfpu_error_plot']
            lut_x = right_angle_data['lut_runtime']
            lut_y = right_angle_data['lut_error']

            # Only draw triangle if there's a meaningful error difference
            # Skip if both are essentially zero or nearly identical
            if sfpu_y_plot > 0.1 and lut_y > 0.1 and abs(sfpu_y_plot - lut_y) > 0.05:
                draw_triangle = True
            elif sfpu_y_plot > 1.0 or lut_y > 1.0:  # At least one has significant error
                draw_triangle = True

        if right_angle_data and draw_triangle:
            corner_x, corner_y = lut_x, sfpu_y_plot

            # Determine triangle orientation
            lut_is_right = lut_x > sfpu_x

            # Get current limits
            xlim = list(ax.get_xlim())
            ylim = list(ax.get_ylim())
            log_ymin, log_ymax = math.log10(ylim[0]), math.log10(ylim[1])
            log_y_range = log_ymax - log_ymin

            # ===== CALCULATE ALL LABEL POSITIONS FIRST =====
            # Label width estimate in data units (rough: 8% of x-range per label)
            x_range = xlim[1] - xlim[0]
            label_width = x_range * 0.12
            label_pad = x_range * 0.03  # Small gap between label and point/edge

            # 1. TTNN label: OUTSIDE triangle, on SFPU side
            if lut_is_right:
                ttnn_label_x = sfpu_x - label_pad
                ttnn_label_ha = 'right'
            else:
                ttnn_label_x = sfpu_x + label_pad
                ttnn_label_ha = 'left'
            ttnn_label_y = 10 ** (log_ymax - log_y_range * 0.15)  # Near top

            # 2. LUT label: OUTSIDE triangle, on LUT side
            if lut_is_right:
                lut_label_x = lut_x + label_pad
                lut_label_ha = 'left'
            else:
                lut_label_x = lut_x - label_pad
                lut_label_ha = 'right'
            lut_label_y = 10 ** (log_ymin + log_y_range * 0.20)  # Near bottom

            # 3. Runtime label: ABOVE horizontal leg (outside triangle)
            runtime_x = (sfpu_x + corner_x) / 2
            runtime_y = 10 ** (np.log10(sfpu_y_plot) + log_y_range * 0.10)
            runtime_va = 'bottom'

            # 4. Error label: OUTSIDE vertical edge (opposite side from triangle body)
            error_y = np.sqrt(corner_y * lut_y)  # Geometric mean
            if lut_is_right:
                # Triangle body is LEFT of vertical edge, so error label goes RIGHT
                error_x = corner_x + label_pad
                error_ha = 'left'
            else:
                # Triangle body is RIGHT of vertical edge, so error label goes LEFT
                error_x = corner_x - label_pad
                error_ha = 'right'

            # ===== EXPAND AXIS LIMITS TO FIT ALL LABELS =====
            # Expand X limits
            needed_x_min = min(sfpu_x, lut_x, ttnn_label_x - label_width, lut_label_x - label_width, error_x - label_width)
            needed_x_max = max(sfpu_x, lut_x, ttnn_label_x + label_width, lut_label_x + label_width, error_x + label_width)

            # Add 15% padding
            x_pad = (needed_x_max - needed_x_min) * 0.15
            new_xmin = max(0, needed_x_min - x_pad)
            new_xmax = needed_x_max + x_pad

            # Expand Y limits (in log space) - minimal padding since labels already offset
            needed_log_ymin = min(np.log10(lut_y), np.log10(lut_label_y)) - log_y_range * 0.05
            needed_log_ymax = max(np.log10(sfpu_y_plot), np.log10(runtime_y), np.log10(ttnn_label_y)) + log_y_range * 0.05

            ax.set_xlim(new_xmin, new_xmax)
            ax.set_ylim(10 ** needed_log_ymin, 10 ** needed_log_ymax)

            # ===== NOW DRAW EVERYTHING =====
            # 1. Draw filled grey triangle
            triangle = Polygon(
                [(sfpu_x, sfpu_y_plot), (corner_x, corner_y), (lut_x, lut_y)],
                closed=True, facecolor='#E0E0E0', edgecolor='none', alpha=0.5, zorder=2
            )
            ax.add_patch(triangle)

            # 2. Draw triangle edges
            ax.plot([sfpu_x, corner_x], [sfpu_y_plot, sfpu_y_plot], color='#555555', linewidth=2.0, zorder=50)
            ax.plot([corner_x, corner_x], [corner_y, lut_y], color='#555555', linewidth=2.0, zorder=50)

            # 3. Draw endpoint markers
            ax.scatter([sfpu_x], [sfpu_y_plot], marker='o', s=30, color='#555555', zorder=51)
            ax.scatter([lut_x], [lut_y], marker='o', s=30, color='#555555', zorder=51)

            # 4. Runtime annotation with connecting line
            ax.plot([runtime_x, runtime_x], [sfpu_y_plot, runtime_y], color='black', linewidth=0.8, zorder=51, alpha=0.7)
            ax.text(runtime_x, runtime_y, right_angle_data['runtime_text'],
                   ha='center', va=runtime_va, fontsize=9, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', edgecolor='black', alpha=0.95),
                   zorder=52, clip_on=False)

            # 5. Error annotation with connecting line
            ax.plot([corner_x, error_x], [error_y, error_y], color='black', linewidth=0.8, zorder=51, alpha=0.7)
            ax.text(error_x, error_y, right_angle_data['error_text'],
                   ha=error_ha, va='center', fontsize=9, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', edgecolor='black', alpha=0.95),
                   zorder=52, clip_on=False)

        # Title with activation name and precision
        # Use larger pad to ensure title stays above any labels that extend beyond axes
        ax.set_title(f'{activation.upper()} ({precision.upper()})', fontsize=14, fontweight='bold', color='#333333', pad=30)

        # Minimize whitespace, but reserve space at top for title + labels
        if not is_broken:
            plt.tight_layout(pad=0.5)
            plt.subplots_adjust(top=0.88)  # Reserve 12% at top for title above labels

        # ==========================================================================
        # DRAW TTNN/LUT LABELS - positioned outside triangle (or simple if no triangle)
        # ==========================================================================
        if pending_annotations and not is_broken:
            import math

            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            x_range = xlim[1] - xlim[0]
            log_ymin, log_ymax = math.log10(ylim[0]), math.log10(ylim[1])
            log_y_range = log_ymax - log_ymin

            # Filter out duplicate TTNN labels (if fast == precise, only show precise)
            filtered_annotations = []
            ttnn_fast_val = None
            ttnn_precise_val = None
            for data_x, data_y, label, color in pending_annotations:
                if 'TTNN-fast' in label or 'TTNN fast' in label:
                    ttnn_fast_val = data_y
                elif 'TTNN' in label and 'fast' not in label.lower() and 'LUT' not in label:
                    ttnn_precise_val = data_y

            for data_x, data_y, label, color in pending_annotations:
                is_fast = 'fast' in label.lower() and 'TTNN' in label
                # Skip TTNN-fast if it's identical to TTNN-precise
                if is_fast and ttnn_fast_val is not None and ttnn_precise_val is not None:
                    if abs(ttnn_fast_val - ttnn_precise_val) < 0.01 * max(ttnn_fast_val, ttnn_precise_val, 0.001):
                        continue  # Skip this one, only show precise
                filtered_annotations.append((data_x, data_y, label, color))

            # Compute triangle centroid for radial positioning (in log space for y)
            if right_angle_data and draw_triangle:
                tri_x1 = right_angle_data['sfpu_runtime']  # TTNN corner
                tri_y1 = right_angle_data['sfpu_error_plot']  # TTNN corner (clipped)
                tri_x2 = right_angle_data['lut_runtime']  # LUT corner
                tri_y2 = right_angle_data['lut_error']  # LUT corner
                tri_x3 = tri_x1  # Right angle corner (same x as TTNN)
                tri_y3 = tri_y2  # Right angle corner (same y as LUT)

                # Centroid in linear x, log y
                centroid_x = (tri_x1 + tri_x2 + tri_x3) / 3
                log_tri_y1 = np.log10(max(tri_y1, 1e-30))
                log_tri_y2 = np.log10(max(tri_y2, 1e-30))
                log_tri_y3 = np.log10(max(tri_y3, 1e-30))
                centroid_log_y = (log_tri_y1 + log_tri_y2 + log_tri_y3) / 3

            for idx, (data_x, data_y, label, color) in enumerate(filtered_annotations):
                is_lut = 'LUT' in label
                is_fast = 'fast' in label.lower()

                if right_angle_data and draw_triangle:
                    # Position label radially outward from triangle centroid
                    log_data_y = np.log10(max(data_y, 1e-30))

                    # Direction from centroid to data point
                    dx = data_x - centroid_x
                    dy = log_data_y - centroid_log_y

                    # Normalize and scale for offset
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > 0:
                        # Offset distance: scale based on plot dimensions
                        offset_x = x_range * 0.08 * (dx / dist)
                        offset_log_y = log_y_range * 0.15 * (dy / dist)
                    else:
                        offset_x = x_range * 0.08
                        offset_log_y = log_y_range * 0.15

                    label_x = data_x + offset_x
                    label_log_y = log_data_y + offset_log_y
                    label_y = 10 ** label_log_y

                    # Determine horizontal alignment based on direction
                    if offset_x > 0:
                        ha = 'left'
                    elif offset_x < 0:
                        ha = 'right'
                    else:
                        ha = 'center'
                else:
                    # No triangle - simple positioning near data point
                    if data_x > (xlim[0] + xlim[1]) / 2:
                        label_x = data_x + x_range * 0.05
                        ha = 'left'
                    else:
                        label_x = data_x - x_range * 0.05
                        ha = 'right'
                    label_y = data_y

                ax.text(label_x, label_y, label,
                       fontsize=9, color='white', fontweight='bold', zorder=16,
                       ha=ha, va='center', clip_on=False,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                                edgecolor=color, alpha=0.95, linewidth=1.5))

                # Draw arrow from label to data point
                ax.annotate('', xy=(data_x, data_y), xytext=(label_x, label_y),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.8),
                           zorder=15, clip_on=True)

        # Save with simplified filename (metric is in directory path)
        timing_key = timing_metric.replace('time_profiler_', '').replace('_us', '')
        if precision:
            output_path = f'{base_output_dir}/{timing_key}_{precision.lower()}.png'
        else:
            output_path = f'{base_output_dir}/{timing_key}.png'

        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        count += 1

    # Reset matplotlib to defaults (don't affect other plots)
    plt.rcdefaults()

    return f"✓ Generated {count} standalone plots"

def detect_activations(arch_prefix):
    """Detect available activations from data/<arch>/polynomial/, rational/, and best/ directories."""
    activations = set()

    # Check polynomial, rational, and best directories
    for subdir in ['polynomial', 'rational', 'best']:
        dir_path = f'data/{arch_prefix}/{subdir}'
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith('.csv'):
                    activations.add(filename[:-4])  # Remove .csv

    return sorted(list(activations))

def load_best_data(arch_prefix, activation, error_metric='mae', precision_filter=None):
    """
    Load best configuration data from data/<arch>/best/<activation>.csv

    Supports row-per-shape format with shape column.

    Returns:
        dict: {timing_metric: {degree: {depth: {'runtime_us': float, 'error': float, 'best_for': str}}}}
    """
    best_csv = f'data/{arch_prefix}/best/{activation}.csv'

    if not os.path.exists(best_csv):
        return None

    # Map error metric to column
    error_column_map = {
        'mae': 'mae',
        'rmse': 'rmse',
        'max': 'max_error',
        'mean_rel': 'mean_rel_error',
        'ulp_max': 'max_ulp',
        'ulp_mean': 'mean_ulp',
        'ulp_median': 'median_ulp',
        'ulp_p99': 'p99_ulp'
    }
    error_column = error_column_map.get(error_metric, 'mae')

    # Load all rows
    with open(best_csv, 'r') as f:
        reader = csv.DictReader(f)
        all_rows = [row for row in reader if row.get('status') == 'pass' and
                    (not precision_filter or row.get('precision', '').lower() == precision_filter.lower())]

    if not all_rows:
        return None

    # Check if this CSV has a shape column (row-per-shape format)
    has_shape_column = 'shape' in all_rows[0] if all_rows else False

    data_by_metric = {}

    if has_shape_column:
        # Row-per-shape format: each row is for a specific shape
        # Group by shape first, then find best config per shape
        for shape in SHAPE_NAMES:
            timing_metric = f'time_profiler_{shape}_us'
            shape_rows = [r for r in all_rows if r.get('shape') == shape]

            if not shape_rows:
                continue

            # Find the best row for this shape based on error metric
            best_row = min(shape_rows, key=lambda r: float(r.get(error_column, float('inf'))))

            degree_str = best_row.get('degree', '0')
            degree = degree_str if '/' in degree_str else int(degree_str)
            depth = int(best_row.get('depth', 0))
            segmentation = best_row.get('segmentation', 'unknown')

            runtime_us = float(best_row.get('time_profiler_us', 0))
            if runtime_us <= 0:
                continue

            error = float(best_row.get(error_column, 0))
            if error == 0.0:
                error = 1e-30

            # Determine what this config is best for
            labels = []
            if best_row == min(shape_rows, key=lambda r: float(r.get('mae', float('inf')))):
                labels.append('MAE')
            if best_row == min(shape_rows, key=lambda r: float(r.get('max_error', float('inf')))):
                labels.append('Max')
            if best_row == min(shape_rows, key=lambda r: float(r.get('max_ulp', float('inf')))):
                labels.append('ULP')
            best_for_label = '/'.join(labels) if labels else ''

            if timing_metric not in data_by_metric:
                data_by_metric[timing_metric] = {}
            if degree not in data_by_metric[timing_metric]:
                data_by_metric[timing_metric][degree] = {}

            data_by_metric[timing_metric][degree][depth] = {
                'runtime_us': runtime_us,
                'error': error,
                'best_for': best_for_label,
                'segmentation': segmentation
            }
    else:
        # Legacy format without shape column - apply to all timing metrics
        # Find best row for the current error metric
        best_row = min(all_rows, key=lambda r: float(r.get(error_column, float('inf'))))
        best_degree_str = best_row.get('degree', '0')
        best_degree = best_degree_str if '/' in best_degree_str else int(best_degree_str)
        best_depth = int(best_row.get('depth', 0))

        for row in all_rows:
            degree_str = row.get('degree', '0')
            degree = degree_str if '/' in degree_str else int(degree_str)
            depth = int(row.get('depth', 0))
            segmentation = row.get('segmentation', 'unknown')

            runtime_us = float(row.get('time_profiler_us', 0))
            if runtime_us <= 0:
                continue

            error = float(row.get(error_column, 0))
            if error == 0.0:
                error = 1e-30

            # Only include the best row
            if degree != best_degree or depth != best_depth:
                continue

            # Determine best_for label
            labels = []
            if row == min(all_rows, key=lambda r: float(r.get('mae', float('inf')))):
                labels.append('MAE')
            if row == min(all_rows, key=lambda r: float(r.get('max_error', float('inf')))):
                labels.append('Max')
            if row == min(all_rows, key=lambda r: float(r.get('max_ulp', float('inf')))):
                labels.append('ULP')
            best_for_label = '/'.join(labels) if labels else ''

            # Add to all timing metrics
            for timing_metric in TIMING_METRICS:
                if timing_metric not in data_by_metric:
                    data_by_metric[timing_metric] = {}
                if degree not in data_by_metric[timing_metric]:
                    data_by_metric[timing_metric][degree] = {}

                data_by_metric[timing_metric][degree][depth] = {
                    'runtime_us': runtime_us,
                    'error': error,
                    'best_for': best_for_label,
                    'segmentation': segmentation
                }

    return data_by_metric if data_by_metric else None

def load_sfpu_data(arch_prefix, activation, error_metric='mae', precision_filter=None, sfpu_mode_filter=None):
    """
    Load native SFPU data from ../generic_lut_activation/data/<arch>/native_sfpu/<activation>.csv
    or from ../generic_lut_activation/data/<arch>/native_sfpu_results.csv (consolidated file)

    Args:
        sfpu_mode_filter: Filter by sfpu_mode ('fast' or 'precise')

    Returns:
        dict: {timing_metric: {'runtime_us': float, 'error': float}} or None
    """
    # Try per-activation file first
    sfpu_csv = f'../generic_lut_activation/data/{arch_prefix}/native_sfpu/{activation}.csv'

    # Fallback to consolidated file
    if not os.path.exists(sfpu_csv):
        sfpu_csv = f'../generic_lut_activation/data/{arch_prefix}/native_sfpu_results.csv'

    if not os.path.exists(sfpu_csv):
        return None

    # Map error metric to column
    error_column_map = {
        'mae': 'mae',
        'max': 'max_error',
        'ulp_max': 'max_ulp_error',
        'ulp_mean': 'mean_ulp_error',
        'ulp_median': 'median_ulp_error',
        'ulp_p99': 'p99_ulp_error'
    }
    error_column = error_column_map.get(error_metric, 'mae')

    sfpu_data = {}

    with open(sfpu_csv, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Skip if wrong activation (for combined CSV)
            if row.get('activation') != activation:
                continue

            # Skip failed tests
            if row.get('status') != 'pass':
                continue

            # Filter by precision
            if precision_filter and row.get('precision', '').lower() != precision_filter.lower():
                continue

            # Filter by sfpu_mode
            if sfpu_mode_filter and row.get('sfpu_mode', '').lower() != sfpu_mode_filter.lower():
                continue

            # Row-per-shape format: one row per (config, shape)
            shape = row.get('shape', '')
            if shape not in SHAPE_NAMES:
                continue

            runtime_us = float(row.get('time_profiler_us', 0))
            if runtime_us <= 0:
                continue

            error = float(row.get(error_column, 0))
            if error == 0.0:
                error = 1e-30

            # Map shape to timing metric name for consistency with LUT data
            timing_metric = f'time_profiler_{shape}_us'

            sfpu_data[timing_metric] = {
                'runtime_us': runtime_us,
                'error': error
            }

    return sfpu_data if sfpu_data else None

def process_single_plot(args_tuple):
    """
    Process a single (activation, precision, error_metric) combination.
    Designed to be called in parallel via ProcessPoolExecutor.

    Returns:
        tuple: (activation, precision, error_metric, result_message)
    """
    (arch_prefix, act, error_metric, segmentation, prec, skip_best,
     best_only, use_broken_axis, gap_threshold, shape_filter,
     standalone_shapes_list) = args_tuple

    result = plot_activation_error_vs_runtime(
        arch_prefix, act, error_metric=error_metric,
        segmentation=segmentation, precision=prec, skip_best=skip_best,
        best_only=best_only, use_broken_axis=use_broken_axis,
        gap_threshold=gap_threshold, shape_filter=shape_filter,
        standalone_shapes=standalone_shapes_list
    )
    return (act, prec, error_metric, result)


def main(arch_prefix, activation=None, segmentation=None, precision=None, skip_best=False, best_only=False,
         use_broken_axis=False, gap_threshold=10, shape_filter=None, metric_filter=None, workers=None,
         standalone_shapes=None):
    """
    Generate error vs runtime plots for activations.

    Args:
        arch_prefix: Architecture (e.g., 'blackhole', 'wormhole')
        activation: Specific activation to plot (if None, plots all)
        segmentation: Segmentation type filter (uniform, curvature, chebyshev)
        precision: Precision filter (bf16, fp32, or None for both)
        skip_best: Skip loading best configuration data
        best_only: Only plot best configurations (hide polynomial sweep data)
        use_broken_axis: Use broken/discontinuous y-axis when large gaps exist
        gap_threshold: Orders of magnitude gap required to trigger broken axis (default: 10)
        shape_filter: Shape filter (single_tile, 8_tiles, 256_tiles, height_sharded, yolov4)
        metric_filter: Error metric filter (mae, rmse, max, mean_rel, ulp_max, ulp_mean, ulp_median, ulp_p99)
        workers: Number of parallel workers (None = CPU count)
        standalone_shapes: Shapes for standalone plots (comma-separated or "all")
    """
    import concurrent.futures
    import multiprocessing

    if not arch_prefix:
        print("ERROR: --arch-prefix is required")
        return

    # Determine worker count
    max_workers = workers if workers else multiprocessing.cpu_count()

    print("=" * 80)
    print(f"Generating {arch_prefix.upper()} Error vs Runtime Plots")
    print("=" * 80)
    print()

    # Determine which activations to plot
    if activation:
        activations = [activation]
        print(f"Plotting specific activation: {activation}")
    else:
        activations = detect_activations(arch_prefix)
        print(f"Found {len(activations)} activations: {', '.join(activations)}")

    # Determine which precisions to plot
    if precision:
        precisions = [precision]
        print(f"Precision filter: {precision}")
    else:
        precisions = ['bf16', 'fp32']
        print(f"Generating plots for both bf16 and fp32")

    # Determine which shapes to plot
    if shape_filter:
        print(f"Shape filter: {shape_filter}")

    # Determine which error metrics to plot
    all_metrics = ['mae', 'rmse', 'max', 'mean_rel', 'ulp_max', 'ulp_mean', 'ulp_median', 'ulp_p99']
    if metric_filter:
        error_metrics = [metric_filter]
        print(f"Metric filter: {metric_filter}")
    else:
        error_metrics = all_metrics
        print(f"Generating plots for all {len(all_metrics)} error metrics")

    if not use_broken_axis:
        print("Broken axis disabled")

    print(f"Workers: {max_workers}")
    print()

    # Parse standalone_shapes (default to yolov4 only)
    if standalone_shapes:
        if standalone_shapes.lower() == 'all':
            standalone_shapes_list = SHAPE_NAMES
        else:
            standalone_shapes_list = [s.strip() for s in standalone_shapes.split(',')]
        print(f"Standalone shapes: {', '.join(standalone_shapes_list)}")
    else:
        standalone_shapes_list = ['yolov4']  # Default to yolov4 only
        print(f"Standalone shapes: yolov4 (use --standalone-shapes all for all shapes)")

    # Build list of all tasks
    tasks = []
    for prec in precisions:
        for error_metric in error_metrics:
            for act in activations:
                tasks.append((
                    arch_prefix, act, error_metric, segmentation, prec,
                    skip_best, best_only, use_broken_axis, gap_threshold, shape_filter,
                    standalone_shapes_list
                ))

    total_tasks = len(tasks)
    print(f"Total plots to generate: {total_tasks}")
    print()

    # Process in parallel
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_plot, task): task for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            try:
                act, prec, error_metric, result = future.result()
                completed += 1

                # Progress indicator - show first line of result
                result_first_line = result.split('\n')[0] if result else "done"
                print(f"  [{completed}/{total_tasks}] {prec}/{error_metric}/{act}: {result_first_line}")

            except Exception as e:
                task = futures[future]
                act = task[1]
                prec = task[4]
                error_metric = task[2]
                completed += 1
                print(f"  [{completed}/{total_tasks}] {prec}/{error_metric}/{act}: ERROR - {e}")

    print()
    print("=" * 80)
    print(f"✓ All {total_tasks} plots generated in plots/{arch_prefix}/error_vs_runtime/<activation>/ directories")
    print("=" * 80)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate error vs runtime plots (new data structure)')
    parser.add_argument('--arch-prefix', type=str, required=True,
                        help='Architecture prefix (e.g., "blackhole" or "wormhole")')
    parser.add_argument('--activation', type=str, default=None,
                        help='Specific activation to plot (plots all if not specified)')
    parser.add_argument('--segmentation', type=str, default=None,
                        help='Segmentation type filter (uniform, curvature, chebyshev)')
    parser.add_argument('--precision', type=str, default=None,
                        help='Precision filter (bf16 or fp32, generates both if not specified)')
    parser.add_argument('--shape', type=str, default=None,
                        help='Shape filter for main plot (single_tile, 8_tiles, 256_tiles, height_sharded, yolov4)')
    parser.add_argument('--standalone-shapes', type=str, default=None,
                        help='Shapes for standalone plots: comma-separated list (e.g., "yolov4" or "256_tiles,yolov4") or "all" for all shapes')
    parser.add_argument('--metric', type=str, default=None,
                        help='Error metric filter (mae, rmse, max, mean_rel, ulp_max, ulp_mean, ulp_median, ulp_p99)')
    parser.add_argument('--skip-best', action='store_true',
                        help='Skip loading best configuration data')
    parser.add_argument('--best-only', action='store_true',
                        help='Only plot best configurations (hide polynomial sweep data)')
    parser.add_argument('--no-broken-axis', action='store_true',
                        help='Disable broken/discontinuous y-axis (enabled by default when large gaps exist)')
    parser.add_argument('--gap-threshold', type=int, default=10,
                        help='Orders of magnitude gap required to trigger broken axis (default: 10)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')

    args = parser.parse_args()
    main(args.arch_prefix, args.activation, args.segmentation, args.precision,
         skip_best=args.skip_best, best_only=args.best_only,
         use_broken_axis=not args.no_broken_axis, gap_threshold=args.gap_threshold,
         shape_filter=args.shape, metric_filter=args.metric, workers=args.workers,
         standalone_shapes=args.standalone_shapes)
