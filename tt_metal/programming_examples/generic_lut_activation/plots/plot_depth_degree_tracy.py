#!/usr/bin/env python3
"""
Generate depth-degree profiler heatmaps showing SFPU compute time.

Uses device profiler data (tracy) to visualize pure SFPU compute performance
across polynomial degrees and segment counts, independent of data movement.
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


def load_tracy_results(arch):
    """Load profiler tracy results from arch-specific directory."""
    tracy_file = f"data/{arch}/depth_degree_tracy.csv"
    if not os.path.exists(tracy_file):
        print(f"❌ No tracy CSV file found: {tracy_file}")
        return None

    print(f"Loading tracy profiler results from {tracy_file}")
    df = pd.read_csv(tracy_file)

    # Filter out invalid/zero profiler times
    df = df[df['profiler_compute_us'] > 0]

    return df


def plot_sfpu_heatmap_per_activation(df, arch, output_dir, precision='bf16'):
    """Generate SFPU compute time heatmaps for gelu activation with all tile shapes in subplots."""
    os.makedirs(output_dir, exist_ok=True)

    activation = 'gelu'
    # Preserve order from CSV (order shapes were run)
    all_shapes = df['shape'].unique().tolist()

    print(f"Generating SFPU compute heatmap for {activation} ({precision.upper()}) across {len(all_shapes)} tile shapes...")

    # Filter to activation and precision
    activation_df = df[(df['activation'] == activation) & (df['precision'] == precision)]

    if len(activation_df) == 0:
        print(f"❌ No data found for {activation} with {precision}")
        return

    # Create subplots (1 row x N columns for N tile shapes)
    fig, axes = plt.subplots(1, len(all_shapes), figsize=(5.0 * len(all_shapes), 5.0))

    # Handle single shape case
    if len(all_shapes) == 1:
        axes = [axes]

    for ax, shape in zip(axes, all_shapes):
        shape_df = activation_df[activation_df['shape'] == shape]

        if len(shape_df) == 0:
            ax.axis('off')
            continue

        # Aggregate: minimum profiler_compute_us for each (degree, segments)
        heatmap_data = shape_df.groupby(['degree', 'segments']).agg({'profiler_compute_us': 'min'}).reset_index()

        # Pivot to create matrix
        heatmap_pivot = heatmap_data.pivot(index='degree', columns='segments', values='profiler_compute_us')

        if heatmap_pivot.empty:
            ax.axis('off')
            continue

        # Create heatmap with RdYlGn_r colormap (green = fast, red = slow)
        heatmap_values = heatmap_pivot.values
        im = ax.imshow(heatmap_values, cmap='RdYlGn_r', aspect='auto', interpolation='nearest', origin='lower')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(heatmap_pivot.columns)))
        ax.set_yticks(np.arange(len(heatmap_pivot.index)))
        ax.set_xticklabels(heatmap_pivot.columns, fontsize=9)
        ax.set_yticklabels([f'p{int(d)}' for d in heatmap_pivot.index], fontsize=9)

        # Labels
        ax.set_xlabel('Depth (segments)', fontsize=10, fontweight='bold')
        if ax == axes[0]:
            ax.set_ylabel('Degree', fontsize=10, fontweight='bold')

        # Title showing shape
        ax.set_title(f'{shape.replace("_", " ").title()} - {precision.upper()}', fontsize=11, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Time (µs)', fontsize=9, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

        # Annotate cells with timing values
        # Normalize values to [0, 1] for colormap
        vmin, vmax = np.nanmin(heatmap_values), np.nanmax(heatmap_values)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.RdYlGn_r

        for i in range(len(heatmap_pivot.index)):
            for j in range(len(heatmap_pivot.columns)):
                time_us = heatmap_values[i, j]
                if not np.isnan(time_us):
                    text = f'{time_us:.1f}µs'
                    # Get the color from colormap and compute luminance
                    normalized_val = norm(time_us)
                    rgba = cmap(normalized_val)
                    # Compute luminance (perceived brightness)
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    # Use black text on light backgrounds (luminance > 0.5)
                    text_color = 'black' if luminance > 0.5 else 'white'
                    ax.text(j, i, text, ha='center', va='center',
                           color=text_color, fontsize=12, fontweight='bold')

    # Add overall title
    fig.suptitle(f'GELU Tracy Profiler Timing - {precision.upper()} (Depth vs Degree)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot with arch in filename
    filename = f'{output_dir}/gelu_sfpu_heatmap_{arch}_{precision}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ {filename}")
    plt.close()


def plot_scaling_across_shapes(df, arch, output_dir, target_activation='gelu', target_degree=6, target_segments=16):
    """Generate plot showing SFPU compute scaling across tile shapes."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter to target activation/degree/segments
    filtered_df = df[
        (df['activation'] == target_activation) &
        (df['degree'] == target_degree) &
        (df['segments'] == target_segments)
    ]

    if len(filtered_df) == 0:
        print(f"⚠️  No data for {target_activation} p{target_degree}_s{target_segments}, skipping scaling plot")
        return

    # Aggregate: minimum profiler_compute_us for each (shape, precision)
    scaling_data = filtered_df.groupby(['shape', 'tiles', 'precision']).agg({'profiler_compute_us': 'min'}).reset_index()

    # Sort by tile count for proper line plotting
    scaling_data = scaling_data.sort_values('tiles')

    # Create plot
    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    for precision in ['fp32', 'bf16']:
        prec_data = scaling_data[scaling_data['precision'] == precision]
        if len(prec_data) > 0:
            ax.plot(prec_data['tiles'], prec_data['profiler_compute_us'],
                   marker='o', markersize=8, linewidth=2, label=precision.upper())

    ax.set_xlabel('Number of Tiles', fontsize=13, fontweight='bold')
    ax.set_ylabel('SFPU Compute Time (µs)', fontsize=13, fontweight='bold')
    ax.set_title(f'{target_activation.upper()} p{target_degree}_s{target_segments} - SFPU Scaling Across Tile Counts',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    filename = f'{output_dir}/{target_activation}_p{target_degree}_s{target_segments}_scaling.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()


def plot_fp32_vs_bf16_comparison(df, arch, output_dir, target_shape='256_tiles'):
    """Generate heatmap comparing FP32 vs BF16 SFPU time ratio."""
    os.makedirs(output_dir, exist_ok=True)

    all_activations = sorted(df['activation'].unique())
    print(f"Generating FP32 vs BF16 comparison for {len(all_activations)} activations (shape: {target_shape})...")

    for activation in all_activations:
        # Filter to target shape
        activation_df = df[(df['activation'] == activation) & (df['shape'] == target_shape)]

        if len(activation_df) == 0:
            continue

        # Aggregate: minimum profiler_compute_us for each (degree, segments, precision)
        heatmap_data = activation_df.groupby(['degree', 'segments', 'precision']).agg({'profiler_compute_us': 'min'}).reset_index()

        # Pivot to separate FP32 and BF16
        fp32_pivot = heatmap_data[heatmap_data['precision'] == 'fp32'].pivot(index='degree', columns='segments', values='profiler_compute_us')
        bf16_pivot = heatmap_data[heatmap_data['precision'] == 'bf16'].pivot(index='degree', columns='segments', values='profiler_compute_us')

        if fp32_pivot.empty or bf16_pivot.empty:
            continue

        # Compute ratio: bf16 / fp32 (< 1.0 means BF16 is faster)
        ratio_pivot = bf16_pivot / fp32_pivot

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8.0, 5.0))

        # Use RdYlGn colormap: green = BF16 faster, red = FP32 faster
        heatmap_values = ratio_pivot.values
        im = ax.imshow(heatmap_values, cmap='RdYlGn_r', aspect='auto', interpolation='nearest',
                      origin='lower', vmin=0.8, vmax=1.2)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(ratio_pivot.columns)))
        ax.set_yticks(np.arange(len(ratio_pivot.index)))
        ax.set_xticklabels(ratio_pivot.columns, fontsize=11)
        ax.set_yticklabels([f'p{int(d)}' for d in ratio_pivot.index], fontsize=11)

        # Labels
        ax.set_xlabel('Segments', fontsize=13, fontweight='bold')
        ax.set_ylabel('Polynomial Degree', fontsize=13, fontweight='bold')
        ax.set_title(f'{activation.upper()} - BF16/FP32 SFPU Time Ratio', fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('BF16/FP32 Ratio', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        # Annotate cells with ratio values
        for i in range(len(ratio_pivot.index)):
            for j in range(len(ratio_pivot.columns)):
                ratio = heatmap_values[i, j]
                if not np.isnan(ratio):
                    text = f'{ratio:.2f}×'
                    # Use white text on extreme cells, black on middle
                    text_color = 'white' if (ratio < 0.9 or ratio > 1.1) else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                           color=text_color, fontsize=10, fontweight='bold')

        plt.tight_layout()

        # Save plot
        filename = f'{output_dir}/{activation}_fp32_vs_bf16.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ {filename}")
        plt.close()


def main(arch, precision_filter=None):
    """Generate depth-degree tracy profiler plots."""
    if not arch:
        print("ERROR: --arch is required (e.g., 'blackhole' or 'wormhole_b0')")
        print("Usage: python3 plot_depth_degree_tracy.py --arch blackhole")
        return

    # Normalize arch name for file paths
    arch_prefix = 'wormhole' if arch == 'wormhole_b0' else arch

    df = load_tracy_results(arch_prefix)

    if df is None or df.empty:
        print(f"❌ No tracy result CSV found")
        print(f"   Looking for: data/{arch_prefix}/depth_degree_tracy.csv")
        print(f"   Run sweep_depth_degree_tracy.sh first")
        return

    print(f"Loaded {len(df)} measurements across {len(df['activation'].unique())} activations")

    # Show available precisions in data
    available_precisions = sorted(df['precision'].unique())
    print(f"Available precisions: {', '.join(available_precisions)}")

    output_dir = f"plots/{arch_prefix}/depth_degree_tracy"

    # Determine which precisions to plot
    if precision_filter:
        precisions_to_plot = [precision_filter] if precision_filter in available_precisions else []
        if not precisions_to_plot:
            print(f"⚠️  Precision '{precision_filter}' not found in data. Available: {', '.join(available_precisions)}")
            return
    else:
        # Plot all available precisions
        precisions_to_plot = available_precisions

    # Generate GELU SFPU heatmaps (all tile shapes in subplots)
    print(f"\nGenerating GELU SFPU compute heatmaps...")
    for precision in precisions_to_plot:
        plot_sfpu_heatmap_per_activation(df, arch_prefix, output_dir, precision=precision)

    print(f"\n✓ Plots saved to: {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate depth-degree tracy profiler heatmaps')
    parser.add_argument('--arch', type=str, required=True,
                        help='Architecture (e.g., "blackhole" or "wormhole_b0")')
    parser.add_argument('--precision', type=str, choices=['fp32', 'bf16'], default=None,
                        help='Plot only specific precision (default: plot all available)')
    args = parser.parse_args()

    main(args.arch, args.precision)
