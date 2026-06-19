#!/usr/bin/env python3
"""
Hardware Error Analysis Script

Compares TTNN native and best approximation implementations against PyTorch baseline
using actual hardware outputs. Styled to match plot_theoretical_error.py.

Usage:
    python3 plot_hardware_error_analysis.py --arch blackhole --activation gelu --precision bf16
    python3 plot_hardware_error_analysis.py --arch blackhole --activation all --metric ulp
"""

import argparse
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import os
import sys
from pathlib import Path

# Import shared CSV naming helpers
from csv_naming import degree_to_safe, build_hw_poly_csv_pattern, build_hw_rational_csv_pattern, get_hardware_output_dir

# Import shared error computation utilities (with caching support)
from hwoutcsv2errcsv import (
    load_or_compute_errors,
    load_hardware_outputs,
    compute_pytorch_baseline,
    downcast_to_precision,
    is_normal_number,
    ACTIVATION_FUNCTIONS,
)

# Activations with native SFPU support (legacy list, kept for reference)
NATIVE_SFPU_ACTIVATIONS = [
    "cos",
    "elu",
    "erf",
    "exp",
    "gelu",
    "hardsigmoid",
    "leaky_relu",
    "relu",
    "selu",
    "sin",
    "softplus",
    "tanh",
    "cosh",
    "sinh",
    "atanh",
    "celu",
]

# Default max points for scatter plots (performance optimization)
DEFAULT_MAX_SCATTER_POINTS = 50000


def downsample_for_plotting(x, y, max_points=DEFAULT_MAX_SCATTER_POINTS, preserve_extrema=True):
    """Downsample data for faster plotting while preserving visual fidelity.

    Uses ULP-aware sampling that:
    1. ALWAYS preserves worst errors (highest y values)
    2. Samples more densely near x=0 where ULP errors tend to spike
    3. Preserves local maxima in each bin

    Args:
        x: Input x coordinates (1D array)
        y: Input y coordinates (1D array)
        max_points: Maximum number of points to return
        preserve_extrema: If True, always include points with highest y values (worst errors)

    Returns:
        (x_downsampled, y_downsampled) tuple
    """
    n = len(x)
    if n <= max_points:
        return x, y

    # Sort by x for consistent results
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Reserve indices for worst errors (extrema)
    if preserve_extrema:
        # Keep top 10% of budget for worst errors, but at least 100 and at most 5000
        n_extrema = min(max(100, max_points // 10), 5000, n)
        # Find indices of highest y values (worst errors) - this is the key for error plots!
        extrema_idx = np.argsort(y_sorted)[-n_extrema:]
        remaining_budget = max_points - n_extrema
    else:
        extrema_idx = np.array([], dtype=int)
        remaining_budget = max_points

    # Also preserve local maxima in bins (catches spikes that might be missed)
    if preserve_extrema and remaining_budget > 1000:
        n_bins = min(500, remaining_budget // 4)
        bin_size = n // n_bins
        local_max_idx = []
        for i in range(n_bins):
            start = i * bin_size
            end = min((i + 1) * bin_size, n)
            if end > start:
                local_max_idx.append(start + np.argmax(y_sorted[start:end]))
        local_max_idx = np.array(local_max_idx)
        extrema_idx = np.unique(np.concatenate([extrema_idx, local_max_idx]))
        remaining_budget = max_points - len(extrema_idx)

    # ULP-aware sampling: denser near x=0 where ULP errors spike
    if remaining_budget > 0:
        # Compute sampling weights: higher weight = more likely to be sampled
        # Use inverse distance to zero with a floor to avoid division issues
        x_range = np.max(np.abs(x_sorted)) + 1e-10
        dist_from_zero = np.abs(x_sorted) / x_range  # Normalized to [0, 1]

        # Weight function: higher near zero, decays towards edges
        # w(d) = 1 / (0.1 + d) gives ~10x more density at x=0 vs edges
        weights = 1.0 / (0.1 + dist_from_zero)

        # Normalize to probability distribution
        weights = weights / np.sum(weights)

        # Sample without replacement according to weights
        # Use deterministic approach: sort by weight and take top remaining_budget
        # This is faster than np.random.choice and reproducible
        weight_order = np.argsort(weights)[::-1]  # Highest weight first

        # Take indices with highest weights, excluding already-selected extrema
        extrema_set = set(extrema_idx)
        weighted_idx = []
        for idx in weight_order:
            if idx not in extrema_set:
                weighted_idx.append(idx)
                if len(weighted_idx) >= remaining_budget:
                    break
        weighted_idx = np.array(weighted_idx) if weighted_idx else np.array([], dtype=int)
    else:
        weighted_idx = np.array([], dtype=int)

    # Combine and deduplicate
    all_idx = np.unique(np.concatenate([weighted_idx, extrema_idx]))
    all_idx = np.sort(all_idx)

    return x_sorted[all_idx], y_sorted[all_idx]


def discover_activations(arch):
    """Auto-detect available activations from the data directory."""
    data_dir = Path(f"data/hardware_outputs/{arch}")
    if not data_dir.exists():
        return NATIVE_SFPU_ACTIVATIONS  # Fallback to legacy list

    activations = []
    for entry in sorted(data_dir.iterdir()):
        # Only include directories that look like activation names
        # Skip non-activation directories (e.g., 'blackhole')
        if entry.is_dir() and entry.name != arch and "," not in entry.name:
            activations.append(entry.name)

    return activations if activations else NATIVE_SFPU_ACTIVATIONS


def load_activation_ranges():
    """Load activation domain ranges from sweep_config.dat (ACTIVATION_RANGES).
    Returns dict: {activation: (range_min, range_max)}"""
    ranges = {}
    for search_dir in [
        Path(__file__).parent.parent,
        Path(__file__).parent.parent.parent / "generic_lut_activation_embedded",
    ]:
        config_path = search_dir / "sweep_config.dat"
        if config_path.exists():
            with open(config_path) as f:
                for line in f:
                    if line.startswith("ACTIVATION_RANGES="):
                        for pair in line.strip().split("=", 1)[1].split(","):
                            parts = pair.split(":")
                            if len(parts) == 3:
                                ranges[parts[0]] = (float(parts[1]), float(parts[2]))
                        break
            if ranges:
                break
    return ranges


def load_best_configs(precision="fp32", metric="ulp"):
    """Load best configurations from $TT_POLY_FIT_DIR/best.csv"""
    poly_fit_dir = os.environ.get("TT_POLY_FIT_DIR")
    if not poly_fit_dir:
        search_paths = [
            "/localdev/nkapre/tt-polynomial-fitter",
            "/proj_sw/user_dev/nkapre/tt-polynomial-fitter",
            os.path.expanduser("~/workspace/tt-polynomial-fitter"),
        ]
        for path in search_paths:
            if os.path.isdir(path):
                poly_fit_dir = path
                break
        else:
            poly_fit_dir = search_paths[-1]

    best_csv = os.path.join(poly_fit_dir, "best.csv")
    if not os.path.exists(best_csv):
        return {}

    configs = {}
    prefix = f"best_{metric}"

    with open(best_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["precision"] == precision:
                activation = row["activation"].strip()
                degree = row[f"{prefix}_degree"].strip()
                if f"{prefix}_segments" in row:
                    segments = row[f"{prefix}_segments"].strip()
                elif f"{prefix}_num_segments" in row:
                    segments = row[f"{prefix}_num_segments"].strip()
                else:
                    segments = row["max_num_segments"].strip()
                segmentation = row[f"{prefix}_segmentation"].strip()
                fitting = row[f"{prefix}_fitting"].strip()

                ulp_field = f"{prefix}_ulp"
                theoretical_ulp = None
                if ulp_field in row and row[ulp_field].strip() not in ("", "inf", "nan"):
                    try:
                        theoretical_ulp = float(row[ulp_field].strip())
                    except ValueError:
                        pass

                configs[activation] = (degree, segments, segmentation, fitting, theoretical_ulp)

    return configs


def plot_error_analysis(
    inputs_dict,
    errors_dict,
    outputs_dict,
    activation,
    output_path,
    approx_label=None,
    exclude_near_zero=False,
    zero_threshold=0.1,
    precision="bf16",
    theoretical_ulp=None,
    breakpoints=None,
    metric="ulp",
    max_scatter_points=DEFAULT_MAX_SCATTER_POINTS,
):
    """Create hardware error analysis plot matching tt-polynomial-fitter style.

    Args:
        max_scatter_points: Maximum points per scatter plot (default 50k for fast rendering).
                           Set to None or 0 to disable downsampling.
    """

    # Save original input ranges BEFORE filtering (for x-axis limits)
    original_x_min = None
    original_x_max = None
    for method in ["sfpu_fast", "sfpu_precise", "approx", "override"]:
        if method in inputs_dict:
            if original_x_min is None:
                original_x_min = np.min(inputs_dict[method])
                original_x_max = np.max(inputs_dict[method])
            else:
                original_x_min = min(original_x_min, np.min(inputs_dict[method]))
                original_x_max = max(original_x_max, np.max(inputs_dict[method]))

    # Note: Subnormal filtering is already done in load_or_compute_errors() with filter_subnormals=True
    # The cached .err.npz files have subnormals pre-filtered, so we don't filter again.
    # outputs_dict may have different sizes than inputs_dict due to filtering - handle this in plotting.

    # Tufte-inspired style
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["font.size"] = 9
    plt.rcParams["agg.path.chunksize"] = 10000

    # Create standard 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    axes = axes.flatten()

    ax1 = axes[0]  # Absolute Error
    ax2 = axes[1]  # Relative Error
    ax3 = axes[2]  # ULP Error
    ax4 = axes[3]  # Approximation Quality

    # Color scheme
    sfpu_fast_color = "#FFA500"  # Orange
    sfpu_precise_color = "#1E88E5"  # Blue
    approx_color = "#43A047"  # Green
    override_color = "#FFD700"  # Yellow (gold) for override CSV

    tt_metal_colors = {"fast": sfpu_fast_color, "precise": sfpu_precise_color}
    tt_metal_alpha = 0.6
    tt_metal_point_size = 4
    tt_metal_zorder = 50

    approx_alpha = 1.0
    approx_linewidth = 2.5
    approx_zorder = 200

    error_types = [("abs", "Absolute Error", ax1, 0), ("rel", "Relative Error", ax2, 1), ("ulp", "ULP Error", ax3, 2)]

    legend_handles = []
    legend_labels = []

    # Plot each error type
    for error_key, error_name, ax, row in error_types:
        has_data = False

        # Plot SFPU Fast
        if "sfpu_fast" in errors_dict and "sfpu_fast" in inputs_dict:
            sfpu_errors = errors_dict["sfpu_fast"][row]
            sfpu_inputs = inputs_dict["sfpu_fast"]

            valid_mask = np.isfinite(sfpu_errors) & (sfpu_errors > 0)
            valid_inputs = sfpu_inputs[valid_mask]
            valid_errors = sfpu_errors[valid_mask]

            if len(valid_errors) > 0:
                # Find max BEFORE downsampling (for annotation)
                max_idx = np.argmax(valid_errors)
                max_val = valid_errors[max_idx]
                max_x = valid_inputs[max_idx]

                # Downsample for faster rendering
                if max_scatter_points and len(valid_inputs) > max_scatter_points:
                    plot_inputs, plot_errors = downsample_for_plotting(valid_inputs, valid_errors, max_scatter_points)
                else:
                    plot_inputs, plot_errors = valid_inputs, valid_errors

                scatter = ax.scatter(
                    plot_inputs,
                    plot_errors,
                    c=sfpu_fast_color,
                    alpha=tt_metal_alpha,
                    s=tt_metal_point_size,
                    edgecolors="none",
                    label="tt-metal (fast)",
                    zorder=tt_metal_zorder,
                    rasterized=True,
                )
                has_data = True
                if "tt-metal (fast)" not in legend_labels:
                    legend_handles.append(scatter)
                    legend_labels.append("tt-metal (fast)")

                # Highlight max error point (use original max_x, max_val)
                ax.plot(
                    max_x,
                    max_val,
                    marker="o",
                    markersize=40,
                    color=sfpu_fast_color,
                    alpha=0.15,
                    markeredgecolor="none",
                    zorder=tt_metal_zorder + 5,
                )

                ax.plot(
                    max_x,
                    max_val,
                    marker="o",
                    markersize=8,
                    color=sfpu_fast_color,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    zorder=tt_metal_zorder + 10,
                )

                # Add text annotation with light colored background
                import matplotlib.colors as mcolors

                rgb = mcolors.to_rgb(sfpu_fast_color)
                light_color = tuple(min(1.0, c + 0.3) for c in rgb)

                if error_key == "ulp":
                    text = f"{max_val:.1e}" if max_val >= 1000 else f"{max_val:.1f}"
                else:
                    text = f"{max_val:.1e}"

                ax.annotate(
                    text,
                    xy=(max_x, max_val),
                    xytext=(8, -15),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor=light_color,
                        edgecolor=sfpu_fast_color,
                        linewidth=0.5,
                        alpha=0.85,
                    ),
                    zorder=tt_metal_zorder + 15,
                )

        # Plot SFPU Precise
        if "sfpu_precise" in errors_dict and "sfpu_precise" in inputs_dict:
            sfpu_errors = errors_dict["sfpu_precise"][row]
            sfpu_inputs = inputs_dict["sfpu_precise"]

            valid_mask = np.isfinite(sfpu_errors) & (sfpu_errors > 0)
            valid_inputs = sfpu_inputs[valid_mask]
            valid_errors = sfpu_errors[valid_mask]

            if len(valid_errors) > 0:
                # Find max BEFORE downsampling (for annotation)
                max_idx = np.argmax(valid_errors)
                max_val = valid_errors[max_idx]
                max_x = valid_inputs[max_idx]

                # Downsample for faster rendering
                if max_scatter_points and len(valid_inputs) > max_scatter_points:
                    plot_inputs, plot_errors = downsample_for_plotting(valid_inputs, valid_errors, max_scatter_points)
                else:
                    plot_inputs, plot_errors = valid_inputs, valid_errors

                scatter = ax.scatter(
                    plot_inputs,
                    plot_errors,
                    c=sfpu_precise_color,
                    alpha=tt_metal_alpha,
                    s=tt_metal_point_size,
                    edgecolors="none",
                    label="tt-metal (precise)",
                    zorder=tt_metal_zorder + 15,
                    rasterized=True,
                )
                has_data = True
                if "tt-metal (precise)" not in legend_labels:
                    legend_handles.append(scatter)
                    legend_labels.append("tt-metal (precise)")

                # Highlight max error point (use original max_x, max_val)
                ax.plot(
                    max_x,
                    max_val,
                    marker="o",
                    markersize=40,
                    color=sfpu_precise_color,
                    alpha=0.15,
                    markeredgecolor="none",
                    zorder=tt_metal_zorder + 20,
                )

                ax.plot(
                    max_x,
                    max_val,
                    marker="o",
                    markersize=8,
                    color=sfpu_precise_color,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    zorder=tt_metal_zorder + 25,
                )

                # Add text annotation with light colored background
                import matplotlib.colors as mcolors

                rgb = mcolors.to_rgb(sfpu_precise_color)
                light_color = tuple(min(1.0, c + 0.3) for c in rgb)

                if error_key == "ulp":
                    text = f"{max_val:.1e}" if max_val >= 1000 else f"{max_val:.1f}"
                else:
                    text = f"{max_val:.1e}"

                ax.annotate(
                    text,
                    xy=(max_x, max_val),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor=light_color,
                        edgecolor=sfpu_precise_color,
                        linewidth=0.5,
                        alpha=0.85,
                    ),
                    zorder=tt_metal_zorder + 30,
                )

        # Plot approximation (line on top)
        if "approx" in errors_dict and "approx" in inputs_dict:
            approx_errors = errors_dict["approx"][row]
            approx_inputs = inputs_dict["approx"]

            valid_mask = np.isfinite(approx_errors) & (approx_errors > 0)
            valid_inputs = approx_inputs[valid_mask]
            valid_errors = approx_errors[valid_mask]

            if len(valid_errors) > 0:
                # Sort by x-coordinate
                sort_idx = np.argsort(valid_inputs)
                valid_inputs = valid_inputs[sort_idx]
                valid_errors = valid_errors[sort_idx]

                # Add star to indicate best config (matching theoretical plot style)
                display_label = approx_label if approx_label else "LUT"
                display_label = f"★ {display_label}"

                (line,) = ax.plot(
                    valid_inputs,
                    valid_errors,
                    color=approx_color,
                    alpha=approx_alpha,
                    linewidth=approx_linewidth,
                    label=display_label,
                    zorder=approx_zorder,
                )
                has_data = True
                if display_label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(display_label)

                # Highlight max error point
                max_idx = np.argmax(valid_errors)
                max_val = valid_errors[max_idx]

                ax.plot(
                    valid_inputs[max_idx],
                    max_val,
                    marker="o",
                    markersize=40,
                    color=approx_color,
                    alpha=0.15,
                    markeredgecolor="none",
                    zorder=approx_zorder + 5,
                )

                ax.plot(
                    valid_inputs[max_idx],
                    max_val,
                    marker="o",
                    markersize=8,
                    color=approx_color,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    zorder=approx_zorder + 10,
                )

                if error_key == "ulp":
                    text = f"{max_val:.1f}" if max_val < 1000 else f"{max_val:.1e}"
                else:
                    text = f"{max_val:.1e}"

                ax.annotate(
                    text,
                    xy=(valid_inputs[max_idx], max_val),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="#C8E6C9", edgecolor=approx_color, linewidth=0.5, alpha=0.85
                    ),
                    zorder=approx_zorder + 15,
                )

        # Plot override CSV (yellow line on top)
        override_zorder = approx_zorder + 50  # Higher than approx to be on top
        if "override" in errors_dict and "override" in inputs_dict:
            override_errors = errors_dict["override"][row]
            override_inputs = inputs_dict["override"]

            valid_mask = np.isfinite(override_errors) & (override_errors > 0)
            valid_inputs = override_inputs[valid_mask]
            valid_errors = override_errors[valid_mask]

            if len(valid_errors) > 0:
                # Sort by x-coordinate
                sort_idx = np.argsort(valid_inputs)
                valid_inputs = valid_inputs[sort_idx]
                valid_errors = valid_errors[sort_idx]

                display_label = "Override CSV"

                (line,) = ax.plot(
                    valid_inputs,
                    valid_errors,
                    color=override_color,
                    alpha=approx_alpha,
                    linewidth=approx_linewidth,
                    label=display_label,
                    zorder=override_zorder,
                )
                has_data = True
                if display_label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(display_label)

                # Highlight max error point
                max_idx = np.argmax(valid_errors)
                max_val = valid_errors[max_idx]

                ax.plot(
                    valid_inputs[max_idx],
                    max_val,
                    marker="o",
                    markersize=40,
                    color=override_color,
                    alpha=0.15,
                    markeredgecolor="none",
                    zorder=override_zorder + 5,
                )

                ax.plot(
                    valid_inputs[max_idx],
                    max_val,
                    marker="o",
                    markersize=8,
                    color=override_color,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    zorder=override_zorder + 10,
                )

                if error_key == "ulp":
                    text = f"{max_val:.1f}" if max_val < 1000 else f"{max_val:.1e}"
                else:
                    text = f"{max_val:.1e}"

                # Light yellow background for annotation
                import matplotlib.colors as mcolors

                rgb = mcolors.to_rgb(override_color)
                light_override_color = tuple(min(1.0, c + 0.2) for c in rgb)

                ax.annotate(
                    text,
                    xy=(valid_inputs[max_idx], max_val),
                    xytext=(8, -20),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor=light_override_color,
                        edgecolor=override_color,
                        linewidth=0.5,
                        alpha=0.85,
                    ),
                    zorder=override_zorder + 15,
                )

        if has_data:
            ax.set_yscale("log")

            # ULP threshold lines
            if error_key == "ulp":
                x_range = ax.get_xlim()
                ax.axhline(y=1.0, color="#2E7D32", linestyle="--", linewidth=1.5, alpha=0.7, zorder=200)
                ax.axhline(y=3.0, color="#F57C00", linestyle="--", linewidth=1.5, alpha=0.7, zorder=200)
                ax.axhline(y=10.0, color="#C62828", linestyle="--", linewidth=1.5, alpha=0.7, zorder=200)

                # Smart positioning to avoid text overlap on log scale
                # Use geometric spacing to maintain visual separation on log scale
                y_min, y_max = ax.get_ylim()

                # Calculate positions based on log scale spacing
                def get_text_ypos(target_y, y_min, y_max):
                    """Adjust text position if too close to y_min on log scale"""
                    # Guard against invalid values for log
                    if y_min <= 0 or y_max <= 0 or target_y <= 0:
                        return target_y
                    log_ymin = np.log10(y_min)
                    log_ymax = np.log10(y_max)
                    log_target = np.log10(target_y)

                    # If target is too close to bottom (within 10% of log range)
                    if (log_target - log_ymin) / (log_ymax - log_ymin) < 0.1:
                        # Position at 15% of log range from bottom
                        return 10 ** (log_ymin + 0.15 * (log_ymax - log_ymin))
                    return target_y

                y1_text = get_text_ypos(1.0, y_min, y_max)
                y3_text = get_text_ypos(3.0, y_min, y_max)
                y10_text = get_text_ypos(10.0, y_min, y_max)

                ax.text(
                    x_range[1],
                    y1_text,
                    ' 1 ULP - "great"',
                    verticalalignment="center",
                    fontsize=9,
                    color="#2E7D32",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=0.9),
                    zorder=200,
                )
                ax.text(
                    x_range[1],
                    y3_text,
                    ' 3 ULP - "accurate"',
                    verticalalignment="center",
                    fontsize=9,
                    color="#F57C00",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=0.9),
                    zorder=200,
                )
                ax.text(
                    x_range[1],
                    y10_text,
                    ' 10 ULP - "approx"',
                    verticalalignment="center",
                    fontsize=9,
                    color="#C62828",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=0.9),
                    zorder=200,
                )

            # Plot breakpoints
            if breakpoints is not None:
                for bp in breakpoints:
                    ax.axvline(x=bp, color="black", linestyle="-", linewidth=0.5, alpha=1.0, zorder=0)

        ax.set_xlabel("Input Value", fontsize=10)
        ax.set_ylabel(error_name, fontsize=10)
        ax.set_title(error_name, fontsize=11, fontweight="normal", pad=10)

    # Compute y-axis limits PER SUBPLOT - clip extreme values and add arrows
    # Clip thresholds for each error type
    CLIP_THRESHOLDS = {0: 1e2, 1: 1e6, 2: 1e3}  # abs, rel, ulp (ULP threshold lower to show labels for large errors)
    method_colors = {
        "sfpu_fast": sfpu_fast_color,
        "sfpu_precise": sfpu_precise_color,
        "approx": approx_color,
        "override": override_color,
    }

    for error_idx, (ax, error_name_short) in enumerate([(ax1, "abs"), (ax2, "rel"), (ax3, "ulp")]):
        subplot_errors = []
        clip_threshold = CLIP_THRESHOLDS[error_idx]

        # Track max values per method for arrow annotations
        method_max_values = {}

        for method in ["sfpu_fast", "sfpu_precise", "approx", "override"]:
            if method in errors_dict:
                abs_err, rel_err, ulp_err = errors_dict[method]

                if error_idx == 0:
                    err_data = abs_err
                elif error_idx == 1:
                    err_data = rel_err
                else:
                    err_data = ulp_err

                valid_mask = np.isfinite(err_data) & (err_data > 0)
                valid_errors = err_data[valid_mask]
                subplot_errors.extend(valid_errors)

                if len(valid_errors) > 0:
                    max_val = np.max(valid_errors)
                    max_idx = np.argmax(err_data[valid_mask])
                    max_x = inputs_dict[method][valid_mask][max_idx]
                    method_max_values[method] = (max_val, max_x)

        if len(subplot_errors) > 0:
            subplot_errors = np.array(subplot_errors)
            err_min = np.percentile(subplot_errors, 0.5)
            err_max = np.max(subplot_errors)

            if error_idx == 0:
                ymin = max(err_min * 0.5, 1e-10)
            elif error_idx == 1:
                ymin = max(err_min * 0.5, 1e-6)
            else:
                ymin = max(err_min * 0.5, 1e-4)

            # Clip ymax if exceeds threshold
            if err_max >= clip_threshold:
                ymax = clip_threshold * 3  # A bit above threshold for visual
                ax.set_ylim(ymin, ymax)

                # Add arrows for methods that exceed threshold
                arrow_y = clip_threshold * 1.5
                arrow_offset = 0
                for method, (max_val, max_x) in method_max_values.items():
                    if max_val >= clip_threshold:
                        color = method_colors[method]
                        # Arrow pointing up
                        ax.annotate(
                            "",
                            xy=(max_x, ymax * 0.9),
                            xytext=(max_x, arrow_y),
                            arrowprops=dict(arrowstyle="->", color=color, lw=2),
                            zorder=300,
                        )
                        # Text label with actual max
                        label = f"{max_val:.1e}"
                        ax.text(
                            max_x,
                            ymax * 0.95,
                            label,
                            fontsize=8,
                            color=color,
                            ha="center",
                            va="bottom",
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.9),
                            zorder=301,
                        )
                        arrow_offset += 0.5
            else:
                ymax = err_max * 2.5
                if error_idx == 2:  # ULP
                    ymax = max(ymax, 10)
                ax.set_ylim(ymin, ymax)

    # Set x-axis limits to full input domain for all error subplots
    if original_x_min is not None and original_x_max is not None:
        x_padding = (original_x_max - original_x_min) * 0.02

        ax1.set_xlim(original_x_min - x_padding, original_x_max + x_padding)
        ax2.set_xlim(original_x_min - x_padding, original_x_max + x_padding)
        ax3.set_xlim(original_x_min - x_padding, original_x_max + x_padding)

    # Plot 4: Approximation Quality
    ax4.set_xlabel("Input Value", fontsize=10)
    ax4.set_ylabel("Output Value", fontsize=10)
    ax4.set_title("Approximation Quality", fontsize=11, fontweight="normal", pad=10)

    if "approx" in inputs_dict:
        x_for_truth = inputs_dict["approx"]
    elif "sfpu_fast" in inputs_dict:
        x_for_truth = inputs_dict["sfpu_fast"]
    elif "sfpu_precise" in inputs_dict:
        x_for_truth = inputs_dict["sfpu_precise"]
    else:
        x_for_truth = None

    if x_for_truth is not None:
        sort_idx_truth = np.argsort(x_for_truth)
        x_truth_sorted = x_for_truth[sort_idx_truth]
        y_truth = compute_pytorch_baseline(x_truth_sorted, activation)

        ax4.plot(x_truth_sorted, y_truth, "k--", linewidth=2.5, label="Ground Truth", alpha=0.8, zorder=15)

        # Plot SFPU fast outputs (use outputs_dict directly - same size as raw CSV)
        if "sfpu_fast" in outputs_dict:
            # Load inputs/outputs from the same source to ensure matching sizes
            x_data = outputs_dict.get("sfpu_fast_inputs", None)
            y_data = outputs_dict["sfpu_fast"]
            if x_data is not None and len(x_data) == len(y_data):
                # Downsample for faster rendering
                if max_scatter_points and len(x_data) > max_scatter_points:
                    plot_x, plot_y = downsample_for_plotting(x_data, y_data, max_scatter_points, preserve_extrema=False)
                else:
                    plot_x, plot_y = x_data, y_data
                ax4.scatter(
                    plot_x,
                    plot_y,
                    c=sfpu_fast_color,
                    alpha=tt_metal_alpha,
                    s=tt_metal_point_size,
                    edgecolors="none",
                    label="tt-metal (fast)",
                    zorder=tt_metal_zorder,
                    rasterized=True,
                )

        # Plot SFPU precise outputs
        if "sfpu_precise" in outputs_dict:
            x_data = outputs_dict.get("sfpu_precise_inputs", None)
            y_data = outputs_dict["sfpu_precise"]
            if x_data is not None and len(x_data) == len(y_data):
                # Downsample for faster rendering
                if max_scatter_points and len(x_data) > max_scatter_points:
                    plot_x, plot_y = downsample_for_plotting(x_data, y_data, max_scatter_points, preserve_extrema=False)
                else:
                    plot_x, plot_y = x_data, y_data
                ax4.scatter(
                    plot_x,
                    plot_y,
                    c=sfpu_precise_color,
                    alpha=tt_metal_alpha,
                    s=tt_metal_point_size,
                    edgecolors="none",
                    label="tt-metal (precise)",
                    zorder=tt_metal_zorder + 5,
                    rasterized=True,
                )

        # Plot LUT approximation outputs
        if "approx" in outputs_dict:
            x_data = outputs_dict.get("approx_inputs", None)
            y_data = outputs_dict["approx"]
            if x_data is not None and len(x_data) == len(y_data):
                sort_idx = np.argsort(x_data)
                x_approx = x_data[sort_idx]
                y_approx = y_data[sort_idx]

                ax4.plot(
                    x_approx,
                    y_approx,
                    color=approx_color,
                    alpha=approx_alpha,
                    linewidth=approx_linewidth,
                    label=approx_label if approx_label else "LUT",
                    zorder=approx_zorder,
                )

        # Plot override CSV (yellow) on the approximation quality plot
        if "override" in outputs_dict:
            x_data = outputs_dict.get("override_inputs", None)
            y_data = outputs_dict["override"]
            if x_data is not None and len(x_data) == len(y_data):
                sort_idx = np.argsort(x_data)
                x_override = x_data[sort_idx]
                y_override = y_data[sort_idx]

                ax4.plot(
                    x_override,
                    y_override,
                    color=override_color,
                    alpha=approx_alpha,
                    linewidth=approx_linewidth,
                    label="Override CSV",
                    zorder=approx_zorder + 50,
                )

        y_min = np.min(y_truth)
        y_max = np.max(y_truth)
        y_range = y_max - y_min
        buffer = 0.1 * y_range if y_range > 0 else 0.1
        ax4.set_ylim(y_min - buffer, y_max + buffer)

        # Set x-axis limits to full input domain (use original unfiltered range)
        if original_x_min is not None and original_x_max is not None:
            x_padding_truth = (original_x_max - original_x_min) * 0.02
            ax4.set_xlim(original_x_min - x_padding_truth, original_x_max + x_padding_truth)

        # Statistics boxes
        y_position = 0.98

        # Override CSV stats box (yellow) - shown first if present
        if "override" in errors_dict:
            abs_err, rel_err, ulp_err = errors_dict["override"]
            valid_abs = abs_err[np.isfinite(abs_err)]
            valid_ulp = ulp_err[np.isfinite(ulp_err)]

            if len(valid_abs) > 0:
                override_text = f"Override CSV\n"
                max_ulp = np.max(valid_ulp) if len(valid_ulp) > 0 else 0
                if max_ulp < 0.01 or max_ulp >= 100:
                    override_text += f"Max ULP Err={max_ulp:.2e}\n"
                else:
                    override_text += f"Max ULP Err={max_ulp:.2f}\n"
                override_text += f"MAE={np.mean(valid_abs):.2e}\n"
                override_text += f"Max Err={np.max(valid_abs):.2e}"

                ax4.text(
                    0.02,
                    y_position,
                    override_text,
                    transform=ax4.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(
                        boxstyle="round,pad=0.6", facecolor="#FFFACD", alpha=0.95, edgecolor="#DAA520", linewidth=1.0
                    ),
                )
                y_position -= 0.19

        if "approx" in errors_dict:
            abs_err, rel_err, ulp_err = errors_dict["approx"]
            valid_abs = abs_err[np.isfinite(abs_err)]
            valid_ulp = ulp_err[np.isfinite(ulp_err)]

            if len(valid_abs) > 0:
                approx_text = f"BEST (by ULP): {approx_label}\n"
                max_ulp = np.max(valid_ulp) if len(valid_ulp) > 0 else 0
                if max_ulp < 0.01 or max_ulp >= 100:
                    approx_text += f"Max ULP Err={max_ulp:.2e}\n"
                else:
                    approx_text += f"Max ULP Err={max_ulp:.2f}\n"
                approx_text += f"MAE={np.mean(valid_abs):.2e}\n"
                approx_text += f"Max Err={np.max(valid_abs):.2e}"

                ax4.text(
                    0.02,
                    y_position,
                    approx_text,
                    transform=ax4.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(
                        boxstyle="round,pad=0.6", facecolor="#C8E6C9", alpha=0.85, edgecolor=approx_color, linewidth=1.5
                    ),
                )
                y_position -= 0.19

        for mode in ["precise", "fast"]:
            key = f"sfpu_{mode}"
            if key in errors_dict:
                abs_err, rel_err, ulp_err = errors_dict[key]
                valid_abs = abs_err[np.isfinite(abs_err)]
                valid_ulp = ulp_err[np.isfinite(ulp_err)]

                if len(valid_abs) > 0:
                    tt_text = f"tt-metal ({mode})\n"
                    max_ulp = np.max(valid_ulp) if len(valid_ulp) > 0 else 0
                    if max_ulp < 0.01 or max_ulp >= 100:
                        tt_text += f"Max ULP Err={max_ulp:.2e}\n"
                    else:
                        tt_text += f"Max ULP Err={max_ulp:.2f}\n"
                    tt_text += f"MAE={np.mean(valid_abs):.2e}\n"
                    tt_text += f"Max Err={np.max(valid_abs):.2e}"

                    box_color = "#ADD8E6" if mode == "precise" else "#FFCC80"
                    ax4.text(
                        0.02,
                        y_position,
                        tt_text,
                        transform=ax4.transAxes,
                        fontsize=10,
                        fontweight="bold",
                        verticalalignment="top",
                        horizontalalignment="left",
                        bbox=dict(
                            boxstyle="round,pad=0.6",
                            facecolor=box_color,
                            alpha=0.95,
                            edgecolor="#666666",
                            linewidth=1.0,
                        ),
                    )
                    y_position -= 0.19

    # Apply Tufte minimalist style
    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.12, linewidth=0.4, linestyle="-", color="#cccccc")
        ax.grid(False, axis="x")
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        ax.tick_params(labelsize=9, length=3, width=0.5, color="#888888")

    # Legend - all items in one row at top
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(legend_handles),  # All items in one row
            fontsize=9,
            frameon=False,
            handlelength=1.5,
            handletextpad=0.4,
            columnspacing=1.5,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Collect messages to return (don't print from worker processes)
    messages = [f"  Saved: {output_path}"]

    # Generate standalone ULP plot only when metric is 'ulp'
    if metric == "ulp":
        ulp_msg = plot_standalone_ulp(
            inputs_dict,
            errors_dict,
            activation,
            output_path,
            approx_label=approx_label,
            precision=precision,
            theoretical_ulp=theoretical_ulp,
            original_x_min=original_x_min,
            original_x_max=original_x_max,
            max_scatter_points=max_scatter_points,
        )
        if ulp_msg:
            messages.append(ulp_msg)

    return messages


def plot_standalone_ulp(
    inputs_dict,
    errors_dict,
    activation,
    main_output_path,
    approx_label=None,
    precision="bf16",
    theoretical_ulp=None,
    original_x_min=None,
    original_x_max=None,
    max_scatter_points=DEFAULT_MAX_SCATTER_POINTS,
):
    """Generate standalone ULP error plot matching tt-polynomial-fitter style."""

    # Create ulp_only subfolder
    main_path = Path(main_output_path)
    ulp_subfolder = main_path.parent / "ulp_only"
    ulp_subfolder.mkdir(parents=True, exist_ok=True)

    # Generate filename
    activation_name = activation.__name__ if hasattr(activation, "__name__") else str(activation)
    if activation_name.startswith("<lambda>"):
        activation_name = main_path.stem.split("_error_analysis")[0]
    ulp_output_file = ulp_subfolder / f"{activation_name}_{precision}_ulp.png"

    # Tufte-inspired style
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["font.size"] = 11
    plt.rcParams["agg.path.chunksize"] = 10000
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["ytick.major.size"] = 3
    plt.rcParams["xtick.color"] = "#333333"
    plt.rcParams["ytick.color"] = "#333333"

    # Create figure - moderate size with larger fonts for PDF clarity
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Color scheme (matching main plot)
    sfpu_fast_color = "#FFA500"  # Orange
    sfpu_precise_color = "#1E88E5"  # Blue
    approx_color = "#43A047"  # Green
    override_color = "#FFD700"  # Yellow (gold) for override CSV
    method_colors = {
        "sfpu_fast": sfpu_fast_color,
        "sfpu_precise": sfpu_precise_color,
        "approx": approx_color,
        "override": override_color,
    }

    # Clip threshold for ULP
    ULP_CLIP_THRESHOLD = 1e6

    tt_metal_alpha = 0.6
    tt_metal_point_size = 4
    tt_metal_zorder = 50

    approx_alpha = 1.0
    approx_linewidth = 2.5
    approx_zorder = 200

    legend_handles = []
    legend_labels = []

    # Collect all ULP errors for y-axis scaling
    all_ulp_errors = []

    # Plot in order: TTNN fast (back) -> TTNN precise (middle) -> LUT (front)

    # 1. Plot TTNN fast (orange)
    if "sfpu_fast" in errors_dict and "sfpu_fast" in inputs_dict:
        inputs = inputs_dict["sfpu_fast"]
        _, _, ulp_errors = errors_dict["sfpu_fast"]

        # Sort by x-coordinate
        sort_idx = np.argsort(inputs)
        inputs_sorted = inputs[sort_idx]
        ulp_sorted = ulp_errors[sort_idx]

        # Find max BEFORE downsampling (for annotation)
        max_idx = np.argmax(ulp_sorted)
        max_val = ulp_sorted[max_idx]
        max_x = inputs_sorted[max_idx]

        # Downsample for faster rendering
        if max_scatter_points and len(inputs_sorted) > max_scatter_points:
            plot_inputs, plot_ulp = downsample_for_plotting(inputs_sorted, ulp_sorted, max_scatter_points)
        else:
            plot_inputs, plot_ulp = inputs_sorted, ulp_sorted

        # Plot scatter (downsampled)
        ax.scatter(
            plot_inputs,
            plot_ulp,
            s=tt_metal_point_size,
            color=sfpu_fast_color,
            alpha=tt_metal_alpha,
            rasterized=True,
            zorder=tt_metal_zorder,
        )

        # Add bubble annotation for max error
        ax.plot(
            max_x,
            max_val,
            marker="o",
            markersize=30,
            color=sfpu_fast_color,
            alpha=0.2,
            markeredgecolor="none",
            zorder=tt_metal_zorder + 5,
        )
        ax.plot(
            max_x,
            max_val,
            marker="o",
            markersize=6,
            color=sfpu_fast_color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=tt_metal_zorder + 10,
        )

        # Add text annotation
        text = f"{max_val:.1f}" if max_val < 1000 else f"{max_val:.0e}"
        ax.annotate(
            text,
            xy=(max_x, max_val),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=sfpu_fast_color, linewidth=0.5, alpha=0.9),
            zorder=tt_metal_zorder + 15,
        )

        legend_handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=sfpu_fast_color, markersize=6, alpha=tt_metal_alpha
            )
        )
        legend_labels.append("TTNN fast")

        all_ulp_errors.extend(ulp_sorted)

    # 2. Plot TTNN precise (blue)
    if "sfpu_precise" in errors_dict and "sfpu_precise" in inputs_dict:
        inputs = inputs_dict["sfpu_precise"]
        _, _, ulp_errors = errors_dict["sfpu_precise"]

        # Sort by x-coordinate
        sort_idx = np.argsort(inputs)
        inputs_sorted = inputs[sort_idx]
        ulp_sorted = ulp_errors[sort_idx]

        # Find max BEFORE downsampling (for annotation)
        max_idx = np.argmax(ulp_sorted)
        max_val = ulp_sorted[max_idx]
        max_x = inputs_sorted[max_idx]

        # Downsample for faster rendering
        if max_scatter_points and len(inputs_sorted) > max_scatter_points:
            plot_inputs, plot_ulp = downsample_for_plotting(inputs_sorted, ulp_sorted, max_scatter_points)
        else:
            plot_inputs, plot_ulp = inputs_sorted, ulp_sorted

        # Plot scatter (downsampled)
        ax.scatter(
            plot_inputs,
            plot_ulp,
            s=tt_metal_point_size,
            color=sfpu_precise_color,
            alpha=tt_metal_alpha,
            rasterized=True,
            zorder=tt_metal_zorder,
        )

        # Add bubble annotation for max error
        ax.plot(
            max_x,
            max_val,
            marker="o",
            markersize=30,
            color=sfpu_precise_color,
            alpha=0.2,
            markeredgecolor="none",
            zorder=tt_metal_zorder + 5,
        )
        ax.plot(
            max_x,
            max_val,
            marker="o",
            markersize=6,
            color=sfpu_precise_color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=tt_metal_zorder + 10,
        )

        # Add text annotation
        text = f"{max_val:.1f}" if max_val < 1000 else f"{max_val:.0e}"
        ax.annotate(
            text,
            xy=(max_x, max_val),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=9,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", edgecolor=sfpu_precise_color, linewidth=0.5, alpha=0.9
            ),
            zorder=tt_metal_zorder + 15,
        )

        legend_handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=sfpu_precise_color, markersize=6, alpha=tt_metal_alpha
            )
        )
        legend_labels.append("TTNN precise")

        all_ulp_errors.extend(ulp_sorted)

    # 3. Plot LUT/approximation (green, on top)
    if "approx" in errors_dict and "approx" in inputs_dict:
        inputs = inputs_dict["approx"]
        _, _, ulp_errors = errors_dict["approx"]

        # Sort by x-coordinate
        sort_idx = np.argsort(inputs)
        inputs_sorted = inputs[sort_idx]
        ulp_sorted = ulp_errors[sort_idx]

        # Plot as line
        ax.plot(
            inputs_sorted,
            ulp_sorted,
            color=approx_color,
            linewidth=approx_linewidth,
            alpha=approx_alpha,
            zorder=approx_zorder,
        )

        # Find max error for annotation
        max_idx = np.argmax(ulp_sorted)
        max_val = ulp_sorted[max_idx]
        max_x = inputs_sorted[max_idx]

        # Add bubble annotation
        ax.plot(
            max_x,
            max_val,
            marker="o",
            markersize=40,
            color=approx_color,
            alpha=0.15,
            markeredgecolor="none",
            zorder=approx_zorder + 5,
        )
        ax.plot(
            max_x,
            max_val,
            marker="o",
            markersize=8,
            color=approx_color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=approx_zorder + 10,
        )

        # Add text annotation
        import matplotlib.colors as mcolors

        rgb = mcolors.to_rgb(approx_color)
        light_color = tuple(min(1.0, c + 0.3) for c in rgb)

        text = f"{max_val:.1e}" if max_val >= 1000 else f"{max_val:.1f}"
        ax.annotate(
            text,
            xy=(max_x, max_val),
            xytext=(8, -15),
            textcoords="offset points",
            fontsize=9,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=light_color, edgecolor=approx_color, linewidth=0.5, alpha=0.85
            ),
            zorder=approx_zorder + 15,
        )

        # Add star to indicate best config (matching theoretical plot style)
        label = approx_label if approx_label else "Approximation"
        label = f"★ {label}"  # Add star prefix
        legend_handles.append(plt.Line2D([0], [0], color=approx_color, linewidth=approx_linewidth, alpha=approx_alpha))
        legend_labels.append(label)

        all_ulp_errors.extend(ulp_sorted)

    # 4. Plot override CSV (yellow, on top of everything)
    override_zorder = approx_zorder + 50
    if "override" in errors_dict and "override" in inputs_dict:
        inputs = inputs_dict["override"]
        _, _, ulp_errors = errors_dict["override"]

        # Sort by x-coordinate
        sort_idx = np.argsort(inputs)
        inputs_sorted = inputs[sort_idx]
        ulp_sorted = ulp_errors[sort_idx]

        # Plot as line
        ax.plot(
            inputs_sorted,
            ulp_sorted,
            color=override_color,
            linewidth=approx_linewidth,
            alpha=approx_alpha,
            zorder=override_zorder,
        )

        # Find max error for annotation
        max_idx = np.argmax(ulp_sorted)
        max_val = ulp_sorted[max_idx]
        max_x = inputs_sorted[max_idx]

        # Add bubble annotation
        ax.plot(
            max_x,
            max_val,
            marker="o",
            markersize=40,
            color=override_color,
            alpha=0.15,
            markeredgecolor="none",
            zorder=override_zorder + 5,
        )
        ax.plot(
            max_x,
            max_val,
            marker="o",
            markersize=8,
            color=override_color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=override_zorder + 10,
        )

        # Add text annotation
        import matplotlib.colors as mcolors

        rgb = mcolors.to_rgb(override_color)
        light_color = tuple(min(1.0, c + 0.2) for c in rgb)

        text = f"{max_val:.1e}" if max_val >= 1000 else f"{max_val:.1f}"
        ax.annotate(
            text,
            xy=(max_x, max_val),
            xytext=(8, -20),
            textcoords="offset points",
            fontsize=9,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=light_color, edgecolor=override_color, linewidth=0.5, alpha=0.85
            ),
            zorder=override_zorder + 15,
        )

        legend_handles.append(
            plt.Line2D([0], [0], color=override_color, linewidth=approx_linewidth, alpha=approx_alpha)
        )
        legend_labels.append("Override CSV")

        all_ulp_errors.extend(ulp_sorted)

    # Collect max values per method for arrow annotations if clipped
    method_max_info = {}
    for method, color in [
        ("sfpu_fast", sfpu_fast_color),
        ("sfpu_precise", sfpu_precise_color),
        ("approx", approx_color),
        ("override", override_color),
    ]:
        if method in errors_dict and method in inputs_dict:
            _, _, ulp_err = errors_dict[method]
            valid_mask = np.isfinite(ulp_err) & (ulp_err > 0)
            if np.any(valid_mask):
                max_val = np.max(ulp_err[valid_mask])
                max_idx = np.argmax(ulp_err[valid_mask])
                max_x = inputs_dict[method][valid_mask][max_idx]
                method_max_info[method] = (max_val, max_x, color)

    # Set y-axis limits with clipping
    if all_ulp_errors:
        ulp_finite = [e for e in all_ulp_errors if np.isfinite(e) and e > 0]
        if ulp_finite:
            ulp_min = np.percentile(ulp_finite, 0.5)
            ulp_max = np.max(ulp_finite)
            ulp_ymin = max(ulp_min * 0.5, 1e-4)

            # Clip at threshold and add arrows for extreme values
            if ulp_max > ULP_CLIP_THRESHOLD:
                ulp_ymax = ULP_CLIP_THRESHOLD * 3
                ax.set_ylim(ulp_ymin, ulp_ymax)

                # Add arrows for methods exceeding threshold
                for method, (max_val, max_x, color) in method_max_info.items():
                    if max_val > ULP_CLIP_THRESHOLD:
                        # Arrow pointing up
                        ax.annotate(
                            "",
                            xy=(max_x, ulp_ymax * 0.9),
                            xytext=(max_x, ULP_CLIP_THRESHOLD * 1.5),
                            arrowprops=dict(arrowstyle="->", color=color, lw=2.5),
                            zorder=300,
                        )
                        # Text label with actual max (placed lower to avoid legend overlap)
                        ax.text(
                            max_x,
                            ulp_ymax * 0.45,
                            f"max={max_val:.1e}",
                            fontsize=9,
                            color=color,
                            ha="center",
                            va="bottom",
                            fontweight="bold",
                            bbox=dict(
                                boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, linewidth=1, alpha=0.95
                            ),
                            zorder=301,
                        )
            else:
                ulp_ymax = max(ulp_max * 2.5, 10)
                ax.set_ylim(ulp_ymin, ulp_ymax)

    # Add ULP reference lines
    if all_ulp_errors:
        x_min = min(inputs_dict[k].min() for k in inputs_dict if k in inputs_dict)
        x_max = max(inputs_dict[k].max() for k in inputs_dict if k in inputs_dict)

        ax.axhline(y=1.0, color="#2E7D32", linestyle="--", linewidth=1.5, alpha=0.7, zorder=200)
        ax.axhline(y=3.0, color="#F57C00", linestyle="--", linewidth=1.5, alpha=0.7, zorder=200)
        ax.axhline(y=10.0, color="#C62828", linestyle="--", linewidth=1.5, alpha=0.7, zorder=200)

        y_min, y_max = ax.get_ylim()

        def get_text_ypos(target_y, y_min, y_max):
            if y_min <= 0 or y_max <= 0 or target_y <= 0:
                return target_y
            log_ymin = np.log10(y_min)
            log_ymax = np.log10(y_max)
            log_target = np.log10(target_y)
            if (log_target - log_ymin) / (log_ymax - log_ymin) < 0.1:
                return 10 ** (log_ymin + 0.15 * (log_ymax - log_ymin))
            return target_y

        y1_text = get_text_ypos(1.0, y_min, y_max)
        y3_text = get_text_ypos(3.0, y_min, y_max)
        y10_text = get_text_ypos(10.0, y_min, y_max)

        # Position labels inside plot using axes coordinates (right side)
        ax.text(
            0.98,
            y1_text,
            "1 ULP",
            verticalalignment="center",
            horizontalalignment="right",
            transform=ax.get_yaxis_transform(),
            fontsize=9,
            color="#2E7D32",
            fontweight="bold",
            alpha=1.0,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#2E7D32", alpha=0.9, linewidth=0.5),
            zorder=200,
        )
        ax.text(
            0.98,
            y3_text,
            "3 ULP",
            verticalalignment="center",
            horizontalalignment="right",
            transform=ax.get_yaxis_transform(),
            fontsize=9,
            color="#F57C00",
            fontweight="bold",
            alpha=1.0,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#F57C00", alpha=0.9, linewidth=0.5),
            zorder=200,
        )
        ax.text(
            0.98,
            y10_text,
            "10 ULP",
            verticalalignment="center",
            horizontalalignment="right",
            transform=ax.get_yaxis_transform(),
            fontsize=9,
            color="#C62828",
            fontweight="bold",
            alpha=1.0,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#C62828", alpha=0.9, linewidth=0.5),
            zorder=200,
        )

    # Set x-axis limits
    if original_x_min is not None and original_x_max is not None:
        x_padding_ulp = (original_x_max - original_x_min) * 0.02
        ax.set_xlim(original_x_min - x_padding_ulp, original_x_max + x_padding_ulp)
    else:
        all_x_inputs = []
        for method in ["sfpu_fast", "sfpu_precise", "approx"]:
            if method in inputs_dict:
                all_x_inputs.extend(inputs_dict[method])

        if len(all_x_inputs) > 0:
            x_min_ulp = np.min(all_x_inputs)
            x_max_ulp = np.max(all_x_inputs)
            x_padding_ulp = (x_max_ulp - x_min_ulp) * 0.02
            ax.set_xlim(x_min_ulp - x_padding_ulp, x_max_ulp + x_padding_ulp)

    # Styling
    ax.set_yscale("log")
    ax.set_xlabel("Input Value", fontsize=11, color="#333333")
    ax.set_ylabel("Max ULP Error", fontsize=11, color="#333333")

    # Grid and spines (Tufte style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.3, color="#CCCCCC", alpha=0.7, zorder=0)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.2, color="#DDDDDD", alpha=0.5, zorder=0)

    ax.tick_params(labelsize=10, length=4, width=0.5, color="#888888")

    # Title at top
    ax.set_title(
        f"{activation_name.upper()} ({precision.upper()})", fontsize=14, fontweight="bold", color="#333333", pad=35
    )

    # Legend just below title
    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=len(legend_labels),
            fontsize=10,
            frameon=False,
            markerscale=2.0,
        )

    plt.tight_layout(pad=0.5)
    plt.savefig(ulp_output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # Return message instead of printing (avoid worker process output interleaving)
    return f"  ✓ Generated standalone ULP plot: {ulp_output_file}"


def combine_precision_plots(arch, activations, base_output_dir):
    """
    Combine bf16 and fp32 ULP plots into side-by-side images.

    Creates combined plots in ulp_only/combined/ subfolder.
    """
    from PIL import Image

    combined_count = 0

    for activation in activations:
        act_dir = base_output_dir / activation / "ulp_only"
        bf16_path = act_dir / f"{activation}_bf16_ulp.png"
        fp32_path = act_dir / f"{activation}_fp32_ulp.png"

        if not bf16_path.exists() or not fp32_path.exists():
            continue

        # Load both images
        bf16_img = Image.open(bf16_path)
        fp32_img = Image.open(fp32_path)

        # Create combined image (side by side)
        combined_width = bf16_img.width + fp32_img.width
        combined_height = max(bf16_img.height, fp32_img.height)

        combined = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
        combined.paste(bf16_img, (0, 0))
        combined.paste(fp32_img, (bf16_img.width, 0))

        # Save combined image
        combined_dir = act_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_path = combined_dir / f"{activation}_ulp_combined.png"
        combined.save(combined_path, dpi=(300, 300))

        combined_count += 1

    return combined_count


def process_activation(
    activation,
    arch,
    precision,
    metric,
    best_configs,
    base_output_dir,
    override_csv_path=None,
    verbose=False,
    max_scatter_points=DEFAULT_MAX_SCATTER_POINTS,
    standalone_only=False,
    activation_ranges=None,
):
    """Process a single activation - designed to be called in parallel."""
    import glob

    inputs_dict = {}
    errors_dict = {}
    outputs_dict = {}
    theoretical_ulp = None
    loaded_csvs = []  # Track loaded CSV paths for verbose output
    searched_paths = []  # Track all paths searched (for error reporting)

    # Load SFPU data (always from ../generic_lut_activation/data/)
    for variant in ["fast", "precise"]:
        pattern = f"../generic_lut_activation/data/hardware_outputs/{arch}/{activation}/native_sfpu_{activation}_{precision}_{variant}_tiles*.csv"
        searched_paths.append(("sfpu_" + variant, pattern))
        matching_files = glob.glob(pattern) + glob.glob(pattern + ".gz")

        if matching_files:
            matching_files.sort(
                key=lambda f: int(f.split("tiles")[-1].replace(".csv.gz", "").replace(".csv", "")), reverse=True
            )
            sfpu_csv = Path(matching_files[0])
            loaded_csvs.append(("sfpu_" + variant, str(sfpu_csv)))

            error_data = load_or_compute_errors(sfpu_csv, activation, precision, verbose=False)
            if error_data:
                inputs_dict[f"sfpu_{variant}"] = error_data["inputs"]
                errors_dict[f"sfpu_{variant}"] = (
                    error_data["abs_errors"],
                    error_data["rel_errors"],
                    error_data["ulp_errors"],
                )
                inputs_sfpu, outputs_sfpu = load_hardware_outputs(sfpu_csv)
                outputs_dict[f"sfpu_{variant}"] = outputs_sfpu
                outputs_dict[f"sfpu_{variant}_inputs"] = inputs_sfpu

    # Load override CSV if specified (plotted in yellow)
    if override_csv_path:
        override_csv = Path(override_csv_path)
        if override_csv.exists():
            loaded_csvs.append(("override", str(override_csv)))
            error_data = load_or_compute_errors(override_csv, activation, precision, verbose=False)
            if error_data:
                inputs_dict["override"] = error_data["inputs"]
                errors_dict["override"] = (error_data["abs_errors"], error_data["rel_errors"], error_data["ulp_errors"])
                inputs_override, outputs_override = load_hardware_outputs(override_csv)
                outputs_dict["override"] = outputs_override
                outputs_dict["override_inputs"] = inputs_override

    # Load approximation data (from ./data/)
    approx_label = None
    if activation in best_configs:
        degree, segments, seg_type, fitting, theoretical_ulp = best_configs[activation]

        activation_dir = Path(f"data/hardware_outputs/{arch}/{activation}")
        if activation_dir.exists():
            # Use canonical CSV naming format
            if "/" in degree:
                num_deg, den_deg = degree.split("/")
                csv_pattern = build_hw_rational_csv_pattern(activation, precision, num_deg, den_deg, segments, seg_type)
            else:
                csv_pattern = build_hw_poly_csv_pattern(activation, precision, degree, segments, seg_type, fitting)
            pattern = f"{activation_dir}/{csv_pattern}"
            searched_paths.append(("approx", str(pattern)))

            matching_files = glob.glob(str(pattern)) + glob.glob(str(pattern) + ".gz")
            if matching_files:
                matching_files.sort(
                    key=lambda f: int(f.split("tiles")[-1].replace(".csv.gz", "").replace(".csv", "")), reverse=True
                )
                approx_csv = Path(matching_files[0])
                loaded_csvs.append(("approx", str(approx_csv)))

                error_data = load_or_compute_errors(approx_csv, activation, precision, verbose=False)
                if error_data:
                    inputs_dict["approx"] = error_data["inputs"]
                    errors_dict["approx"] = (
                        error_data["abs_errors"],
                        error_data["rel_errors"],
                        error_data["ulp_errors"],
                    )
                    inputs_approx, outputs_approx = load_hardware_outputs(approx_csv)
                    outputs_dict["approx"] = outputs_approx
                    outputs_dict["approx_inputs"] = inputs_approx

                if "/" in degree:
                    approx_label = f"n{degree.replace('/', 'd')}_s{segments}_{seg_type[:4]}"
                else:
                    approx_label = f"p{degree}_s{segments}_{seg_type[:4]}"

    # Filter data to activation's fitting domain (removes out-of-range BF16 points entirely)
    if activation_ranges and activation in activation_ranges:
        domain_min, domain_max = activation_ranges[activation]
        for method in list(inputs_dict.keys()):
            if method.endswith("_inputs"):
                continue  # Skip the paired _inputs keys in outputs_dict
            inp = inputs_dict[method]
            mask = (inp >= domain_min) & (inp <= domain_max)
            inputs_dict[method] = inp[mask]
            if method in errors_dict:
                abs_e, rel_e, ulp_e = errors_dict[method]
                errors_dict[method] = (abs_e[mask], rel_e[mask], ulp_e[mask])
        # Filter outputs_dict too
        for method in list(outputs_dict.keys()):
            if method.endswith("_inputs"):
                inp_key = method
                out_key = method.replace("_inputs", "")
                if inp_key in outputs_dict and out_key in outputs_dict:
                    inp = outputs_dict[inp_key]
                    mask = (inp >= domain_min) & (inp <= domain_max)
                    outputs_dict[inp_key] = inp[mask]
                    outputs_dict[out_key] = outputs_dict[out_key][mask]

    # Generate plot
    if errors_dict:
        output_dir = base_output_dir / activation
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{activation}_error_analysis_{precision}_{metric}.png"

        if standalone_only:
            # Compute x range from (now filtered) data
            original_x_min = None
            original_x_max = None
            for method in ["sfpu_fast", "sfpu_precise", "approx", "override"]:
                if method in inputs_dict:
                    if original_x_min is None:
                        original_x_min = np.min(inputs_dict[method])
                        original_x_max = np.max(inputs_dict[method])
                    else:
                        original_x_min = min(original_x_min, np.min(inputs_dict[method]))
                        original_x_max = max(original_x_max, np.max(inputs_dict[method]))

            ulp_msg = plot_standalone_ulp(
                inputs_dict,
                errors_dict,
                activation,
                output_path,
                approx_label=approx_label,
                precision=precision,
                theoretical_ulp=theoretical_ulp,
                original_x_min=original_x_min,
                original_x_max=original_x_max,
                max_scatter_points=max_scatter_points,
            )
            messages = [ulp_msg] if ulp_msg else []
        else:
            messages = plot_error_analysis(
                inputs_dict,
                errors_dict,
                outputs_dict,
                activation,
                output_path,
                approx_label=approx_label,
                precision=precision,
                theoretical_ulp=theoretical_ulp,
                breakpoints=None,
                metric=metric,
                max_scatter_points=max_scatter_points,
            )
        return activation, True, str(output_path), loaded_csvs, messages, []

    return activation, False, None, loaded_csvs, [], searched_paths


def main():
    import concurrent.futures
    import multiprocessing

    parser = argparse.ArgumentParser(description="Generate hardware error analysis plots")
    parser.add_argument("--arch", type=str, required=True, choices=["wormhole", "blackhole"], help="Architecture")
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="Activation function(s), comma-separated (e.g., gelu,tanh,relu). Default: all activations",
    )
    parser.add_argument(
        "--precision", type=str, default="both", choices=["bf16", "fp32", "both"], help="Precision (default: both)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=["mae", "max", "ulp"],
        help="Optimization metric (default: all metrics)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--csv", type=str, default=None, help="Override CSV file path (plotted in yellow)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print paths to hardware output CSVs being used")
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_SCATTER_POINTS,
        help=f"Max points per scatter plot for speed (default: {DEFAULT_MAX_SCATTER_POINTS}). "
        "Use 0 to disable downsampling.",
    )
    parser.add_argument(
        "--full", action="store_true", help="Disable downsampling (plot all points). Equivalent to --max-points 0"
    )
    parser.add_argument(
        "--standalone", action="store_true", help="Only generate standalone ULP plots (skip the 2x2 grid plot)"
    )

    args = parser.parse_args()

    precisions = ["bf16", "fp32"] if args.precision == "both" else [args.precision]
    metrics = ["mae", "max", "ulp"] if args.metric is None else [args.metric]
    max_workers = args.workers if args.workers else multiprocessing.cpu_count()
    max_scatter_points = 0 if args.full else args.max_points

    activation_ranges = load_activation_ranges()

    for precision in precisions:
        for metric in metrics:
            best_configs = load_best_configs(precision=precision, metric=metric)

            if args.output_dir:
                base_output_dir = Path(args.output_dir)
            else:
                base_output_dir = Path(f"plots/{args.arch}/hardware_error_analysis")

            if args.activation is None or args.activation == "all":
                activations = discover_activations(args.arch)
            else:
                # Support comma-separated list of activations
                activations = [a.strip() for a in args.activation.split(",")]

            print("=" * 80)
            print(f"Hardware Error Analysis - {args.arch.upper()} - {precision.upper()} - metric={metric}")
            print("=" * 80)
            print(f"Activations: {', '.join(activations)}")
            print(f"Metric: {metric}")
            print(f"Workers: {max_workers}")
            if max_scatter_points:
                print(f"Max scatter points: {max_scatter_points:,} (use --full for all points)")
            else:
                print("Downsampling: DISABLED (plotting all points)")
            if args.standalone:
                print("Standalone mode: Only generating ULP plots (skipping 2x2 grid)")
            print()

            # Process activations in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_activation,
                        activation,
                        args.arch,
                        precision,
                        metric,
                        best_configs,
                        base_output_dir,
                        args.csv,
                        args.verbose,
                        max_scatter_points,
                        args.standalone,
                        activation_ranges,
                    ): activation
                    for activation in activations
                }

                for future in concurrent.futures.as_completed(futures):
                    activation = futures[future]
                    try:
                        act_name, success, output_path, loaded_csvs, messages, searched_paths = future.result()
                        if success:
                            if args.verbose and loaded_csvs:
                                print(f"\n  {act_name} - Hardware output CSVs:")
                                for csv_type, csv_path in loaded_csvs:
                                    print(f"    [{csv_type}] {csv_path}")
                            # Print messages from worker (collected, not interleaved)
                            for msg in messages:
                                print(msg)
                        else:
                            # Show what paths were searched when no data is found
                            if searched_paths:
                                print(f"  ✗ {act_name}: No data found. Searched patterns:")
                                for csv_type, pattern in searched_paths:
                                    print(f"      [{csv_type}] {pattern}")
                            else:
                                print(f"  ✗ {act_name}: No data found (no search patterns available)")
                    except Exception as e:
                        print(f"  ✗ {activation}: Error - {e}")

            print()
            print("=" * 80)
            print(f"✓ {precision.upper()} / {metric} analysis complete")
            print("=" * 80)
            print()

    # Combine bf16 and fp32 plots if both precisions were generated
    if args.precision == "both" and args.standalone:
        print("=" * 80)
        print("Combining bf16 and fp32 plots into side-by-side images...")
        print("=" * 80)

        if args.output_dir:
            base_output_dir = Path(args.output_dir)
        else:
            base_output_dir = Path(f"plots/{args.arch}/hardware_error_analysis")

        if args.activation is None or args.activation == "all":
            activations = discover_activations(args.arch)
        else:
            activations = [a.strip() for a in args.activation.split(",")]

        combined_count = combine_precision_plots(args.arch, activations, base_output_dir)
        print(f"  Created {combined_count} combined plots in ulp_only/combined/ subfolders")
        print()


if __name__ == "__main__":
    main()
