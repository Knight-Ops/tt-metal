#!/usr/bin/env python3
"""
Generate comparison plots showing MAE and Max Error separately, comparing:
- Theoretical best configs (from tt-polynomial-fitter)
- Best configs (from sweep_best)
- Empirically best configs (from sweep_all)
- SFPU baseline (if available)
"""

import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict

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
plt.rcParams['grid.alpha'] = 0.15
plt.rcParams['grid.linewidth'] = 0.5


def load_theoretical_best(theoretical_best_file):
    """Load theoretical predictions from tt-polynomial-fitter's best.csv."""
    theoretical = {}
    if not os.path.exists(theoretical_best_file):
        print(f"Warning: Theoretical best file not found: {theoretical_best_file}")
        return theoretical

    with open(theoretical_best_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            activation = row['activation'].strip()
            precision = row['precision'].strip()
            key = (activation, precision)

            # Skip if errors are inf (not approximable)
            if row['mae'].strip() == 'inf' or row['max_error'].strip() == 'inf':
                continue

            # Handle rational degrees in theoretical data
            degree_str = row['best_mae_degree'].strip()
            if '/' in degree_str:
                degree = degree_str  # Keep as string for rational
            else:
                degree = int(degree_str)  # Convert to int for polynomial

            theoretical[key] = {
                'depth': int(row['best_mae_segments'].strip()),
                'degree': degree,
                'mae': float(row['mae'].strip()),
                'max_error': float(row['max_error'].strip()),
                'segmentation': row['segmentation'].strip()
            }

    return theoretical


def load_best_results(best_results_file):
    """Load best results (one config per activation from sweep_best)."""
    best_results = {}
    if not os.path.exists(best_results_file):
        print(f"Warning: Best results file not found: {best_results_file}")
        return best_results

    with open(best_results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] != 'pass':
                continue

            activation = row['activation']
            precision = row['precision']
            key = (activation, precision)

            # Handle rational degrees (e.g., "10/9") - store as string
            degree_str = row['degree']
            if '/' in degree_str:
                degree = degree_str  # Keep as string for rational
            else:
                degree = int(degree_str)  # Convert to int for polynomial

            best_results[key] = {
                'depth': int(row['depth']),
                'degree': degree,
                'mae': float(row['mae']),
                'max_error': float(row['max_error']),
                'runtime': float(row['kernel_exec_ms'])
            }

    return best_results


def load_all_results(all_results_file):
    """Load all results and group by activation/precision."""
    all_configs = defaultdict(list)
    if not os.path.exists(all_results_file):
        print(f"Warning: All results file not found: {all_results_file}")
        return all_configs

    with open(all_results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] != 'pass':
                continue

            activation = row['activation']
            precision = row['precision']
            key = (activation, precision)

            # Handle rational degrees in all results
            degree_str = row['degree']
            if '/' in degree_str:
                degree = degree_str  # Keep as string for rational
            else:
                degree = int(degree_str)  # Convert to int for polynomial

            all_configs[key].append({
                'depth': int(row['depth']),
                'degree': degree,
                'segmentation': row['segmentation'],
                'mae': float(row['mae']),
                'max_error': float(row['max_error']),
                'runtime': float(row['kernel_exec_ms'])
            })

    return all_configs


def load_sfpu_results(sfpu_results_file):
    """Load SFPU baseline results."""
    sfpu_results = {}
    if not os.path.exists(sfpu_results_file):
        print(f"Warning: SFPU results file not found: {sfpu_results_file}")
        return sfpu_results

    with open(sfpu_results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] != 'pass':
                continue

            activation = row['activation']
            precision = row['precision']
            key = (activation, precision)

            sfpu_results[key] = {
                'mae': float(row['mae']),
                'max_error': float(row['max_error']),
                'runtime': float(row['kernel_exec_ms'])
            }

    return sfpu_results


def plot_comparison(theoretical, best_results, all_configs, sfpu_results, output_dir, platform_prefix=''):
    """Generate separate plots for MAE and Max Error comparisons."""

    os.makedirs(output_dir, exist_ok=True)

    # Filter to only keys that exist in both best_results and all_configs
    common_keys = sorted([k for k in best_results.keys() if k in all_configs])

    if not common_keys:
        print("No common activations found between best and all results!")
        return

    # Prepare data for plotting
    activations = []

    # Data arrays for MAE
    theory_mae = []
    best_mae = []
    empirical_mae = []
    sfpu_mae = []

    # Data arrays for Max Error
    theory_max = []
    best_max = []
    empirical_max = []
    sfpu_max = []

    for key in common_keys:
        activation, precision = key
        label = f"{activation}\n({precision})"
        activations.append(label)

        # Best config
        best = best_results[key]
        best_mae.append(best['mae'])
        best_max.append(best['max_error'])

        # Empirically best config (min MAE from all configs)
        all_cfgs = all_configs[key]
        empirical_best = min(all_cfgs, key=lambda c: c['mae'])
        empirical_mae.append(empirical_best['mae'])
        empirical_max.append(empirical_best['max_error'])

        # Theoretical
        if key in theoretical:
            theory = theoretical[key]
            theory_mae.append(theory['mae'])
            theory_max.append(theory['max_error'])
        else:
            theory_mae.append(None)
            theory_max.append(None)

        # SFPU baseline
        if key in sfpu_results:
            sfpu = sfpu_results[key]
            sfpu_mae.append(sfpu['mae'])
            sfpu_max.append(sfpu['max_error'])
        else:
            sfpu_mae.append(None)
            sfpu_max.append(None)

    # Generate MAE plot
    _plot_metric(activations, theory_mae, best_mae, empirical_mae, sfpu_mae,
                 'Mean Absolute Error (MAE)', 'mae', output_dir, platform_prefix)

    # Generate Max Error plot
    _plot_metric(activations, theory_max, best_max, empirical_max, sfpu_max,
                 'Maximum Error', 'max_error', output_dir, platform_prefix)


def _plot_metric(activations, theory_vals, best_vals, empirical_vals, sfpu_vals,
                 ylabel, metric_name, output_dir, platform_prefix):
    """Generate a single plot for a specific metric."""

    n_activations = len(activations)
    x = np.arange(n_activations)
    width = 0.2

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, n_activations * 0.4), 6))

    # Colors (Tufte-inspired muted colors)
    theory_color = '#90C695'  # Muted green
    best_color = '#5791A4'    # Muted blue
    empirical_color = '#E27A7A'  # Muted red
    sfpu_color = '#7A8B99'    # Muted grey-blue

    # Plot bars
    bars1 = ax.bar(x - 1.5*width, [v if v is not None else 0 for v in theory_vals],
                   width, label='Theory', color=theory_color, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x - 0.5*width, best_vals, width, label='Best Config',
                   color=best_color, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + 0.5*width, empirical_vals, width, label='Empirical Best',
                   color=empirical_color, edgecolor='black', linewidth=0.5)
    bars4 = ax.bar(x + 1.5*width, [v if v is not None else 0 for v in sfpu_vals],
                   width, label='SFPU Baseline', color=sfpu_color, edgecolor='black', linewidth=0.5)

    # Mark bars with None values (missing data) with hatching
    for i, v in enumerate(theory_vals):
        if v is None:
            bars1[i].set_facecolor('white')
            bars1[i].set_edgecolor('#CCCCCC')
            bars1[i].set_hatch('///')

    for i, v in enumerate(sfpu_vals):
        if v is None:
            bars4[i].set_facecolor('white')
            bars4[i].set_edgecolor('#CCCCCC')
            bars4[i].set_hatch('///')

    # Labels and styling
    ax.set_xlabel('Activation Function (Precision)', fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.set_title(f'{ylabel} Comparison: Theory vs Best Config vs Empirical Best vs SFPU',
                 fontsize=12, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(activations, rotation=45, ha='right', fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5, which='major', axis='y')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='none')

    # Add horizontal line for visual separation
    ax.axhline(y=1e-6, color='#CCCCCC', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{platform_prefix}comparison_{metric_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Generated {output_path}")


def main():
    """Main entry point."""
    if len(sys.argv) < 4:
        print("Usage: python3 plot_sweep_comparison.py <all_results.csv> <best_results.csv> <theoretical_best.csv> [sfpu_results.csv] [output_dir] [platform_prefix]")
        print()
        print("Arguments:")
        print("  all_results.csv      : CSV with all sweep combinations")
        print("  best_results.csv     : CSV with best configurations")
        print("  theoretical_best.csv : CSV with theoretical predictions")
        print("  sfpu_results.csv     : (Optional) CSV with SFPU baseline results")
        print("  output_dir           : (Optional) Output directory for plots (default: plots/comparison)")
        print("  platform_prefix      : (Optional) Platform prefix for output files (e.g., 'blackhole_')")
        sys.exit(1)

    all_results_file = sys.argv[1]
    best_results_file = sys.argv[2]
    theoretical_best_file = sys.argv[3]
    sfpu_results_file = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
    output_dir = sys.argv[5] if len(sys.argv) > 5 else 'plots/comparison'
    platform_prefix = sys.argv[6] if len(sys.argv) > 6 else ''

    print("=" * 80)
    print("Generating Sweep Comparison Plots")
    print("=" * 80)
    print()
    print(f"All results     : {all_results_file}")
    print(f"Best results    : {best_results_file}")
    print(f"Theoretical best: {theoretical_best_file}")
    print(f"SFPU results    : {sfpu_results_file if sfpu_results_file else 'N/A'}")
    print(f"Output directory: {output_dir}")
    print(f"Platform prefix : {platform_prefix if platform_prefix else 'N/A'}")
    print()

    # Load data
    print("Loading data...")
    theoretical = load_theoretical_best(theoretical_best_file)
    best_results = load_best_results(best_results_file)
    all_configs = load_all_results(all_results_file)
    sfpu_results = load_sfpu_results(sfpu_results_file) if sfpu_results_file else {}

    print(f"  Theoretical configs: {len(theoretical)}")
    print(f"  Best configs       : {len(best_results)}")
    print(f"  All configs        : {sum(len(v) for v in all_configs.values())} total ({len(all_configs)} activations)")
    print(f"  SFPU baselines     : {len(sfpu_results)}")
    print()

    # Generate plots
    print("Generating comparison plots...")
    plot_comparison(theoretical, best_results, all_configs, sfpu_results, output_dir, platform_prefix)
    print()

    print("=" * 80)
    print(f"✓ All comparison plots generated in {output_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
