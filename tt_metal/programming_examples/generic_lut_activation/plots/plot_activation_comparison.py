#!/usr/bin/env python3
"""
Generate comparison plots for activation functions across different methods.
Shows Native SFPU, PC, PL, PQ outputs for visual comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os
import sys
import csv
from multiprocessing import Pool, cpu_count

# Add tt-polynomial-fitter to path for ground_truth module
_poly_fit_dir = os.environ.get('TT_POLY_FIT_DIR', '/localdev/nkapre/tt-polynomial-fitter')
sys.path.insert(0, _poly_fit_dir)
from ground_truth import ACTIVATION_FUNCTIONS, get_all_activations

# Tufte-inspired minimalist style for publication-quality plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 9
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

# Default input range (fallback)
INPUT_RANGE = (-10, 10)
NUM_POINTS = 1000

def load_activation_configs(csv_path='activations_config.dat'):
    """Load per-activation test ranges from CSV."""
    configs = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                activation = row['activation']
                range_min = float(row['range_min'])
                range_max = float(row['range_max'])
                configs[activation] = (range_min, range_max)
    except Exception as e:
        print(f"Warning: Could not load {csv_path}: {e}")
    return configs

# Load activation configs at module level
ACTIVATION_CONFIGS = load_activation_configs()

def load_native_sfpu_results(csv_path='wormhole_native_sfpu_results.csv'):
    """Load native SFPU results from CSV file (fast mode by default)."""
    try:
        results = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            has_sfpu_mode = 'sfpu_mode' in reader.fieldnames if reader.fieldnames else False
            for row in reader:
                # Skip failed tests
                if row.get('status') != 'pass':
                    continue
                # Filter to fast mode (default baseline) if sfpu_mode column exists
                if has_sfpu_mode and row.get('sfpu_mode') != 'fast':
                    continue
                activation = row['activation']
                mae = float(row['mae'])
                results[activation] = mae
        return results
    except Exception as e:
        print(f"Warning: Could not load {csv_path}: {e}")
        return {}

def simulate_piecewise_linear(x, func, num_segments=128):
    """Simulate piecewise linear approximation."""
    x_min, x_max = INPUT_RANGE
    segment_width = (x_max - x_min) / num_segments
    
    # Fit linear segments
    result = np.zeros_like(x)
    for i in range(num_segments):
        seg_start = x_min + i * segment_width
        seg_end = seg_start + segment_width
        seg_mid = (seg_start + seg_end) / 2
        
        # Get slope and offset from segment endpoints
        y_start = func(np.array([seg_start]))[0]
        y_end = func(np.array([seg_end]))[0]
        slope = (y_end - y_start) / segment_width
        offset = y_start - slope * seg_start
        
        # Apply to points in this segment
        mask = (x >= seg_start) & (x < seg_end)
        result[mask] = slope * x[mask] + offset
    
    # Handle last point
    result[x >= x_max] = func(np.array([x_max]))[0]
    return result

def simulate_piecewise_quadratic_remez(x, func, num_segments=128):
    """Simulate piecewise quadratic approximation using OLS fitting."""
    x_min, x_max = INPUT_RANGE
    segment_width = (x_max - x_min) / num_segments
    
    result = np.zeros_like(x)
    for i in range(num_segments):
        seg_start = x_min + i * segment_width
        seg_end = seg_start + segment_width
        
        # Sample points for OLS fitting
        x_samples = np.linspace(seg_start, seg_end, 100)
        y_samples = func(x_samples)
        
        # Fit quadratic: y = a*x^2 + b*x + c
        X = np.vstack([x_samples**2, x_samples, np.ones_like(x_samples)]).T
        coeffs = np.linalg.lstsq(X, y_samples, rcond=None)[0]
        a, b, c = coeffs
        
        # Apply to points in this segment
        mask = (x >= seg_start) & (x < seg_end)
        result[mask] = a * x[mask]**2 + b * x[mask] + c
    
    # Handle last point
    result[x >= x_max] = func(np.array([x_max]))[0]
    return result

def load_hardware_output(filepath):
    """Load hardware output CSV file with input,output columns."""
    if not os.path.exists(filepath):
        return None, None

    inputs = []
    outputs = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                inputs.append(float(row['input']))
                outputs.append(float(row['output']))
        return np.array(inputs), np.array(outputs)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None, None

def get_available_depths(hardware_dir, activation_name, method_prefix):
    """
    Detect available depths for a given activation and method from hardware output files.

    Args:
        hardware_dir: Directory containing hardware output files
        activation_name: Name of the activation function
        method_prefix: Method prefix (e.g., 'piecewise_constant', 'piecewise_linear')

    Returns:
        Sorted list of available depth values
    """
    if not os.path.exists(hardware_dir):
        return []

    depths = []
    import glob
    pattern = os.path.join(hardware_dir, f'{method_prefix}_{activation_name}_*.csv')
    files = glob.glob(pattern)

    for filepath in files:
        filename = os.path.basename(filepath)
        # Extract depth from filename: method_activation_DEPTH.csv
        parts = filename.replace('.csv', '').split('_')
        try:
            depth = int(parts[-1])
            depths.append(depth)
        except (ValueError, IndexError):
            continue

    return sorted(set(depths))

def plot_hardware_activation_comparison(activation_name, hardware_dir='data/hardware_outputs',
                                       output_dir='wormhole_plots_values', arch_name='Wormhole',
                                       platform_prefix='wormhole_'):
    """
    Generate comparison plot using actual hardware outputs.
    Shows ground truth, native SFPU, and piecewise constant/linear/quadratic/cubic.
    """
    os.makedirs(output_dir, exist_ok=True)

    if activation_name not in ACTIVATION_FUNCTIONS:
        print(f"Unknown activation: {activation_name}")
        return

    func = ACTIVATION_FUNCTIONS[activation_name]

    # Get per-activation test range
    input_range = ACTIVATION_CONFIGS.get(activation_name, INPUT_RANGE)

    # Generate ground truth over activation-specific range
    x_gt = np.linspace(*input_range, NUM_POINTS)
    y_gt = func(x_gt)

    # Create figure - IEEE single-column format
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Plot ground truth - Tufte style (thin, precise lines)
    ax.plot(x_gt, y_gt, color='#333333', linewidth=1.0, label='Ground Truth', alpha=0.8, zorder=1)

    # Load and plot Native SFPU - muted blue-grey
    native_file = os.path.join(hardware_dir, f'native_sfpu_{activation_name}.csv')
    x_native, y_native = load_hardware_output(native_file)
    if x_native is not None:
        # Sort by input for proper line plotting
        sorted_idx = np.argsort(x_native)
        ax.plot(x_native[sorted_idx], y_native[sorted_idx], color='#7A8B99', linewidth=1.0,
                label='Native SFPU', alpha=0.9, zorder=5)

    # Dynamically detect available depths for each method
    depths_pl = get_available_depths(hardware_dir, activation_name, 'piecewise_linear')
    depths_pq = get_available_depths(hardware_dir, activation_name, 'piecewise_quadratic_remez')
    depths_pqr = get_available_depths(hardware_dir, activation_name, 'piecewise_quadratic_remez')
    depths_cubic = get_available_depths(hardware_dir, activation_name, 'piecewise_cubic_remez')
    depths_hexic = get_available_depths(hardware_dir, activation_name, 'piecewise_hexic_remez')
    depths_octic = get_available_depths(hardware_dir, activation_name, 'piecewise_octic_remez')

    # Filter out octic depth 8 for Blackhole
    if arch_name == 'Blackhole' and 8 in depths_octic:
        depths_octic = [d for d in depths_octic if d != 8]

    # Use the union of all available depths for consistent display
    all_depths = sorted(set(depths_pl + depths_pq + depths_pqr + depths_cubic + depths_hexic + depths_octic))

    if not all_depths:
        print(f"Warning: No depth data found for {activation_name}")
        return

    # Generate colors and line styles dynamically based on number of depths
    def get_colors_and_styles(base_colors, num_depths):
        """Generate colors and line styles for given number of depths."""
        if num_depths == 0:
            return [], []
        elif num_depths == 1:
            linestyles = ['-']
            colors = [base_colors[1]]  # Use middle color
        elif num_depths == 2:
            linestyles = ['--', '-']
            colors = [base_colors[0], base_colors[2]]
        elif num_depths == 3:
            linestyles = [':', '--', '-']
            colors = base_colors
        else:  # 4 or more depths
            linestyles = [':', '-.', '--', '-'] + ['-'] * (num_depths - 4)
            # Interpolate colors
            import matplotlib.colors as mcolors
            cmap = mcolors.LinearSegmentedColormap.from_list("", [base_colors[0], base_colors[-1]])
            colors = [mcolors.to_hex(cmap(i / (num_depths - 1))) for i in range(num_depths)]
        return colors, linestyles

    # Plot Piecewise Linear at different depths - Tufte-inspired muted colors
    base_colors_pl = ['#90C695', '#7DB883', '#6AAA71']  # Muted green shades
    colors_pl, linestyles_pl = get_colors_and_styles(base_colors_pl, len(depths_pl))
    for i, depth in enumerate(depths_pl):
        pl_file = os.path.join(hardware_dir, f'piecewise_linear_{activation_name}_{depth}.csv')
        x_pl, y_pl = load_hardware_output(pl_file)
        if x_pl is not None:
            sorted_idx = np.argsort(x_pl)
            alpha = 0.5 + (i / max(1, len(depths_pl) - 1)) * 0.4
            ax.plot(x_pl[sorted_idx], y_pl[sorted_idx], linestyles_pl[i],
                   linewidth=1.0, color=colors_pl[i], label=f'Linear-{depth}',
                   alpha=alpha, zorder=3)

    # Plot Piecewise Quadratic (Least Squares) at different depths - Tufte-inspired muted colors
    base_colors_pq = ['#E27A7A', '#D66868', '#CA5656']  # Muted red shades
    colors_pq, linestyles_pq = get_colors_and_styles(base_colors_pq, len(depths_pq))
    for i, depth in enumerate(depths_pq):
        pq_file = os.path.join(hardware_dir, f'piecewise_quadratic_remez_{activation_name}_{depth}.csv')
        x_pq, y_pq = load_hardware_output(pq_file)
        if x_pq is not None:
            sorted_idx = np.argsort(x_pq)
            alpha = 0.5 + (i / max(1, len(depths_pq) - 1)) * 0.4
            ax.plot(x_pq[sorted_idx], y_pq[sorted_idx], linestyles_pq[i],
                   linewidth=1.0, color=colors_pq[i], label=f'Quad-LS-{depth}',
                   alpha=alpha, zorder=4)

    # Plot Piecewise Quadratic (optimized polynomial fit) at different depths - Tufte-inspired muted colors
    base_colors_pqr = ['#B4A7D6', '#A294C8', '#9081BA']  # Muted purple shades
    colors_pqr, linestyles_pqr = get_colors_and_styles(base_colors_pqr, len(depths_pqr))
    for i, depth in enumerate(depths_pqr):
        pqr_file = os.path.join(hardware_dir, f'piecewise_quadratic_remez_{activation_name}_{depth}.csv')
        x_pqr, y_pqr = load_hardware_output(pqr_file)
        if x_pqr is not None:
            sorted_idx = np.argsort(x_pqr)
            alpha = 0.5 + (i / max(1, len(depths_pqr) - 1)) * 0.4
            ax.plot(x_pqr[sorted_idx], y_pqr[sorted_idx], linestyles_pqr[i],
                   linewidth=1.0, color=colors_pqr[i], label=f'Quad-{depth}',
                   alpha=alpha, zorder=4)

    # Piecewise Cubic - Tufte-inspired muted teals
    base_colors_cubic = ['#87B5C4', '#6FA3B4', '#5791A4']  # Muted teal shades
    colors_cubic, linestyles_cubic = get_colors_and_styles(base_colors_cubic, len(depths_cubic))
    for i, depth in enumerate(depths_cubic):
        cubic_file = os.path.join(hardware_dir, f'piecewise_cubic_remez_{activation_name}_{depth}.csv')
        x_cubic, y_cubic = load_hardware_output(cubic_file)
        if x_cubic is not None:
            sorted_idx = np.argsort(x_cubic)
            alpha = 0.5 + (i / max(1, len(depths_cubic) - 1)) * 0.4
            ax.plot(x_cubic[sorted_idx], y_cubic[sorted_idx], linestyles_cubic[i],
                   linewidth=1.0, color=colors_cubic[i], label=f'Cubic-{depth}',
                   alpha=alpha, zorder=4)

    # Piecewise Hexic - Tufte-inspired muted mauves
    base_colors_hexic = ['#D4A5B8', '#C793A8', '#BA8198']  # Muted mauve shades
    colors_hexic, linestyles_hexic = get_colors_and_styles(base_colors_hexic, len(depths_hexic))
    for i, depth in enumerate(depths_hexic):
        hexic_file = os.path.join(hardware_dir, f'piecewise_hexic_remez_{activation_name}_{depth}.csv')
        x_hexic, y_hexic = load_hardware_output(hexic_file)
        if x_hexic is not None:
            sorted_idx = np.argsort(x_hexic)
            alpha = 0.5 + (i / max(1, len(depths_hexic) - 1)) * 0.4
            ax.plot(x_hexic[sorted_idx], y_hexic[sorted_idx], linestyles_hexic[i],
                   linewidth=1.0, color=colors_hexic[i], label=f'Hexic-{depth}',
                   alpha=alpha, zorder=4)

    # Piecewise Octic - Tufte-inspired muted browns/tans
    base_colors_octic = ['#C4A57B', '#B39469', '#A28357']  # Muted brown/tan shades
    colors_octic, linestyles_octic = get_colors_and_styles(base_colors_octic, len(depths_octic))
    for i, depth in enumerate(depths_octic):
        octic_file = os.path.join(hardware_dir, f'piecewise_octic_remez_{activation_name}_{depth}.csv')
        x_octic, y_octic = load_hardware_output(octic_file)
        if x_octic is not None:
            sorted_idx = np.argsort(x_octic)
            alpha = 0.5 + (i / max(1, len(depths_octic) - 1)) * 0.4
            ax.plot(x_octic[sorted_idx], y_octic[sorted_idx], linestyles_octic[i],
                   linewidth=1.0, color=colors_octic[i], label=f'Octic-{depth}',
                   alpha=alpha, zorder=4)

    ax.set_xlabel('Input', fontsize=8, color='#333333')
    ax.set_ylabel('Output', fontsize=8, color='#333333')
    # No title (Tufte minimalist style)
    # Very compact legend at top - spread across more columns for fewer rows
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=6, fontsize=4,
             framealpha=0.95, edgecolor='none', fancybox=False,
             markerscale=0.5, handlelength=1.2, columnspacing=0.5, handletextpad=0.3)
    ax.tick_params(axis='both', labelsize=7)
    # Minimal gridlines (Tufte style)
    ax.grid(True, alpha=0.12, linestyle='-', linewidth=0.5, which='major')

    # Set x-axis limits to match the activation's test range
    ax.set_xlim(input_range)

    # Set y-axis limits to fit ground truth range (with small padding)
    # This prevents showing numerical artifacts/ringing beyond expected output range
    y_min, y_max = y_gt.min(), y_gt.max()
    y_range = y_max - y_min
    padding = 0.05 * y_range  # 5% padding

    # Use log scale for functions with large dynamic range
    if activation_name in ['exp', 'sinh', 'cosh']:
        ax.set_yscale('log')
        ax.set_ylim(bottom=max(1e-4, y_min * 0.95), top=y_max * 1.05)
    else:
        ax.set_ylim(y_min - padding, y_max + padding)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{activation_name}_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Generated {output_path}")

def plot_activation_comparison(activation_name, output_dir='plots', native_sfpu_results=None):
    """Generate comparison plot for a single activation function."""
    os.makedirs(output_dir, exist_ok=True)

    if activation_name not in ACTIVATION_FUNCTIONS:
        print(f"Unknown activation: {activation_name}")
        return

    func = ACTIVATION_FUNCTIONS[activation_name]
    x = np.linspace(*INPUT_RANGE, NUM_POINTS)

    # Compute outputs for all methods
    y_ground_truth = func(x)

    # For native SFPU, add noise based on actual MAE if available
    if native_sfpu_results and activation_name in native_sfpu_results:
        mae = native_sfpu_results[activation_name]
        # Add realistic noise based on measured MAE
        noise = np.random.normal(0, mae, size=x.shape)
        y_native = y_ground_truth + noise
    else:
        y_native = y_ground_truth  # Fallback to ground truth if no data

    # Compute multiple depths for each method to show convergence
    depths = [8, 32, 128]

    # Piecewise Linear at different depths
    y_pl = {}
    for d in depths:
        y_pl[d] = simulate_piecewise_linear(x, func, num_segments=d)

    # Piecewise Quadratic at different depths
    y_pq = {}
    for d in depths:
        y_pq[d] = simulate_piecewise_quadratic_remez(x, func, num_segments=d)

    # Create 3x2 subplot layout (3 comparisons, each with output+error pair)
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle(f'{activation_name.upper()} Activation Function Comparison', fontsize=18, fontweight='bold', y=0.995)

    # Comparison 1: Native SFPU + PL
    # Output plot
    ax = plt.subplot(3, 2, 1)
    ax.plot(x, y_ground_truth, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax.plot(x, y_native, 'b-', linewidth=1.8, label='Native SFPU', alpha=0.9)
    ax.plot(x, y_pl[8], 'g:', linewidth=1.3, label='PL-8', alpha=0.5)
    ax.plot(x, y_pl[32], 'g--', linewidth=1.5, label='PL-32', alpha=0.7)
    ax.plot(x, y_pl[128], 'g-', linewidth=1.8, label='PL-128', alpha=0.9)
    ax.set_ylabel('Output', fontsize=11)
    ax.set_title('Native SFPU vs Piecewise Linear (PL)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    # Error plot
    ax = plt.subplot(3, 2, 2)
    ax.plot(x, np.abs(y_native - y_ground_truth), 'b-', linewidth=1.8, label='Native SFPU', alpha=0.9)
    ax.plot(x, np.abs(y_pl[8] - y_ground_truth), 'g:', linewidth=1.3, label='PL-8', alpha=0.5)
    ax.plot(x, np.abs(y_pl[32] - y_ground_truth), 'g--', linewidth=1.5, label='PL-32', alpha=0.7)
    ax.plot(x, np.abs(y_pl[128] - y_ground_truth), 'g-', linewidth=1.8, label='PL-128', alpha=0.9)
    ax.set_ylabel('Absolute Error', fontsize=11)
    ax.set_title('Error: SFPU vs PL', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Comparison 2: Native SFPU + PQ
    # Output plot
    ax = plt.subplot(3, 2, 3)
    ax.plot(x, y_ground_truth, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax.plot(x, y_native, 'b-', linewidth=1.8, label='Native SFPU', alpha=0.9)
    ax.plot(x, y_pq[8], 'm:', linewidth=1.3, label='PQ-8', alpha=0.5)
    ax.plot(x, y_pq[32], 'm--', linewidth=1.5, label='PQ-32', alpha=0.7)
    ax.plot(x, y_pq[128], 'm-', linewidth=1.8, label='PQ-128', alpha=0.9)
    ax.set_ylabel('Output', fontsize=11)
    ax.set_title('Native SFPU vs Piecewise Quadratic (PQ)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    # Error plot
    ax = plt.subplot(3, 2, 4)
    ax.plot(x, np.abs(y_native - y_ground_truth), 'b-', linewidth=1.8, label='Native SFPU', alpha=0.9)
    ax.plot(x, np.abs(y_pq[8] - y_ground_truth), 'm:', linewidth=1.3, label='PQ-8', alpha=0.5)
    ax.plot(x, np.abs(y_pq[32] - y_ground_truth), 'm--', linewidth=1.5, label='PQ-32', alpha=0.7)
    ax.plot(x, np.abs(y_pq[128] - y_ground_truth), 'm-', linewidth=1.8, label='PQ-128', alpha=0.9)
    ax.set_ylabel('Absolute Error', fontsize=11)
    ax.set_title('Error: SFPU vs PQ', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Comparison 3: Best of each method
    # Output plot
    ax = plt.subplot(3, 2, 5)
    ax.plot(x, y_ground_truth, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax.plot(x, y_native, 'b-', linewidth=1.8, label='Native SFPU', alpha=0.9)
    ax.plot(x, y_pl[128], 'g-', linewidth=1.8, label='PL-128', alpha=0.8)
    ax.plot(x, y_pq[128], 'm-', linewidth=1.8, label='PQ-128', alpha=0.8)
    ax.set_xlabel('Input', fontsize=11)
    ax.set_ylabel('Output', fontsize=11)
    ax.set_title('Best of Each Method (128 segments)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    # Error plot
    ax = plt.subplot(3, 2, 6)
    ax.plot(x, np.abs(y_native - y_ground_truth), 'b-', linewidth=1.8, label='Native SFPU', alpha=0.9)
    ax.plot(x, np.abs(y_pl[128] - y_ground_truth), 'g-', linewidth=1.8, label='PL-128', alpha=0.8)
    ax.plot(x, np.abs(y_pq[128] - y_ground_truth), 'm-', linewidth=1.8, label='PQ-128', alpha=0.8)
    ax.set_xlabel('Input', fontsize=11)
    ax.set_ylabel('Absolute Error', fontsize=11)
    ax.set_title('Error: Best of Each', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{activation_name}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated {output_path}")

def _plot_activation_wrapper(args):
    """Wrapper function for parallel processing of activation comparisons."""
    activation, hardware_dir, output_dir, arch_name, platform_prefix = args
    plot_hardware_activation_comparison(activation, hardware_dir, output_dir, arch_name, platform_prefix)
    return f"✓ Generated {activation} plot for {arch_name}"


def plot_activation_comparison_grid(hardware_dir, output_dir, arch_name, platform_prefix='wormhole_'):
    """
    Generate a single grid plot showing all activation function comparisons.
    Each subplot shows ground truth + Native SFPU + best piecewise methods.

    Args:
        hardware_dir: Directory containing hardware output CSVs
        output_dir: Directory to save the grid plot
        arch_name: Architecture name (for title and output filename)
    """
    ALL_ACTIVATIONS = get_all_activations()

    os.makedirs(output_dir, exist_ok=True)

    # Filter to activations that exist in ACTIVATION_FUNCTIONS
    activations = sorted([act for act in ALL_ACTIVATIONS if act in ACTIVATION_FUNCTIONS])

    # Determine grid size
    n_activations = len(activations)
    if n_activations <= 12:
        nrows, ncols = 3, 4
    elif n_activations <= 15:
        nrows, ncols = 3, 5
    elif n_activations <= 20:
        nrows, ncols = 4, 5
    else:
        nrows, ncols = 5, 6

    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10), sharex=False, sharey=False)
    axes = axes.flatten() if n_activations > 1 else [axes]

    # Piecewise methods to plot
    piecewise_methods = [
        'piecewise_constant',
        'piecewise_linear',
        'piecewise_quadratic_remez',
        'piecewise_cubic_remez',
        'piecewise_hexic_remez',
    ]

    # Color scheme for different methods
    colors = {
        'ground_truth': '#000000',      # Black
        'native_sfpu': '#FF0000',       # Red
        'piecewise_constant': '#4A90E2',      # Blue
        'piecewise_linear': '#50C878',        # Green
        'piecewise_quadratic_remez': '#9B59B6',  # Purple
        'piecewise_cubic_remez': '#E67E22',      # Orange
        'piecewise_hexic_remez': '#E91E63',      # Pink
    }

    # Line styles for different depths
    depth_styles = {
        4: ':',      # Dotted - lightest/most approximate
        8: '--',     # Dashed
        16: '-.',    # Dash-dot
        32: '-'      # Solid - best quality
    }

    # Plot each activation
    for idx, activation in enumerate(activations):
        ax = axes[idx]

        if activation not in ACTIVATION_FUNCTIONS:
            ax.axis('off')
            continue

        func = ACTIVATION_FUNCTIONS[activation]

        # Get input range for this activation
        if activation in ACTIVATION_CONFIGS:
            input_range = ACTIVATION_CONFIGS[activation]
        else:
            input_range = INPUT_RANGE

        x = np.linspace(*input_range, NUM_POINTS)
        y_ground_truth = func(x)

        # Plot ground truth
        ax.plot(x, y_ground_truth, color=colors['ground_truth'], linewidth=1.5,
               label='Ground Truth', alpha=0.8, zorder=5)

        # Set y-axis limits based on ground truth range
        y_min_gt = np.min(y_ground_truth)
        y_max_gt = np.max(y_ground_truth)
        y_range = y_max_gt - y_min_gt
        y_padding = y_range * 0.1  # 10% padding
        ax.set_ylim(y_min_gt - y_padding, y_max_gt + y_padding)

        # Plot Native SFPU if available
        native_file = os.path.join(hardware_dir, f'native_sfpu_{activation}.csv')
        if os.path.exists(native_file):
            x_native, y_native = load_hardware_output(native_file)
            if x_native is not None and y_native is not None:
                ax.plot(x_native, y_native, color=colors['native_sfpu'], linewidth=1.2,
                       label='Native SFPU', alpha=0.9, zorder=4)

        # Plot all piecewise methods from hardware data
        for method in piecewise_methods:
            if method not in colors:
                continue

            # Try different depths for this method
            for depth in [4, 8, 16, 32]:
                filepath = os.path.join(hardware_dir, f'{method}_{activation}_{depth}.csv')
                if os.path.exists(filepath):
                    x_data, y_data = load_hardware_output(filepath)
                    if x_data is not None and y_data is not None:
                        linestyle = depth_styles.get(depth, '-')
                        # Vary line width by depth - deeper = thicker
                        linewidth = 0.5 + (depth / 32) * 0.5  # 0.5 to 1.0
                        alpha = 0.5 + (depth / 32) * 0.3  # 0.5 to 0.8
                        ax.plot(x_data, y_data, color=colors[method], linewidth=linewidth,
                               linestyle=linestyle, alpha=alpha, zorder=3)

        # Styling
        ax.set_title(activation, fontsize=7, color='#333333')
        ax.set_xlabel('x', fontsize=6, color='#333333')
        ax.set_ylabel('y', fontsize=6, color='#333333')
        ax.tick_params(axis='both', labelsize=5)
        ax.grid(True, alpha=0.12, linestyle='-', linewidth=0.5)

    # Hide unused subplots
    for idx in range(len(activations), len(axes)):
        axes[idx].axis('off')

    # Create unified legend with all methods and depths
    legend_handles = []
    legend_labels = []

    # Ground truth
    line = plt.Line2D([0], [0], color=colors['ground_truth'], linewidth=1.5, alpha=0.8)
    legend_handles.append(line)
    legend_labels.append('Ground Truth')

    # Native SFPU
    line = plt.Line2D([0], [0], color=colors['native_sfpu'], linewidth=1.2, alpha=0.9)
    legend_handles.append(line)
    legend_labels.append('Native SFPU')

    # Method labels (short names)
    method_labels = {
        'piecewise_constant': 'PC',
        'piecewise_linear': 'PL',
        'piecewise_quadratic_remez': 'PQ',
        'piecewise_cubic_remez': 'PCub',
        'piecewise_hexic_remez': 'PHex',
    }

    # Add all methods with all depths
    for method in piecewise_methods:
        if method not in colors:
            continue
        method_label = method_labels.get(method, method)
        for depth in [4, 8, 16, 32]:
            linestyle = depth_styles.get(depth, '-')
            linewidth = 0.5 + (depth / 32) * 0.5
            line = plt.Line2D([0], [0], color=colors[method], linewidth=linewidth,
                            linestyle=linestyle, alpha=0.7)
            legend_handles.append(line)
            legend_labels.append(f'{method_label}-{depth}')

    # Multi-row legend at top spanning entire figure
    fig.legend(legend_handles, legend_labels, loc='upper center',
              bbox_to_anchor=(0.5, 1.00), ncol=11, fontsize=9,
              framealpha=0.95, edgecolor='none', fancybox=False,
              markerscale=1.5, handlelength=2.5, columnspacing=1.2, handletextpad=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Minimize gap between legend and plots

    # Save plot
    output_path = os.path.join(output_dir, f'all_activations_comparison_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Generated {output_path}")


def main():
    """Generate all activation function comparison plots using hardware data."""
    import argparse
    parser = argparse.ArgumentParser(description='Generate activation function comparison plots')
    parser.add_argument('--activation', type=str, default=None,
                       help='Generate plot for specific activation only (default: all)')
    args = parser.parse_args()

    # Use all activation functions from ground_truth module, excluding GLU variants (vector ops)
    all_activations = sorted([k for k in ACTIVATION_FUNCTIONS.keys() if not k.endswith('glu')])

    # Filter to specific activation if requested
    if args.activation:
        if args.activation not in all_activations:
            print(f"Error: Unknown activation '{args.activation}'")
            print(f"Available: {', '.join(all_activations)}")
            return
        activations = [args.activation]
    else:
        activations = all_activations

    print("=" * 80)
    print("Generating Hardware Activation Function Comparison Plots")
    print("=" * 80)
    print(f"Activations: {len(activations)} total")
    if args.activation:
        print(f"Filtering to: {args.activation}")
    print()

    num_workers = max(1, cpu_count() - 1)  # Leave one core free

    # Generate plots for all activations (or just the one specified via --activation)
    sample_activations = activations

    # Check unified hardware outputs directory
    hardware_dir = 'data/hardware_outputs'

    if not os.path.exists(hardware_dir):
        print(f"WARNING: No hardware outputs found at {hardware_dir}")
        return

    # Check for Wormhole files
    wormhole_files = [f for f in os.listdir(hardware_dir) if f.startswith('wormhole_')]
    if wormhole_files:
        if sample_activations:
            print(f"Generating Wormhole individual hardware comparison plots (samples: {', '.join(sample_activations)})...")
            args_list = [
                (activation, hardware_dir, 'wormhole_plots_values', 'Wormhole', 'wormhole_')
                for activation in sample_activations
            ]
            with Pool(num_workers) as pool:
                results = pool.map(_plot_activation_wrapper, args_list)
            for result in results:
                print(result)
            print()

        # Generate grid plot for all activations
        if not args.activation:  # Only generate grid for full run
            print("Generating Wormhole activation comparison grid plot (all activations)...")
            plot_activation_comparison_grid(hardware_dir, 'wormhole_plots_values', 'Wormhole', 'wormhole_')
            print()

    # Check for Blackhole files
    blackhole_files = [f for f in os.listdir(hardware_dir) if f.startswith('blackhole_')]
    if blackhole_files:
        if sample_activations:
            print(f"Generating Blackhole individual hardware comparison plots (samples: {', '.join(sample_activations)})...")
            args_list = [
                (activation, hardware_dir, 'blackhole_plots_values', 'Blackhole', 'blackhole_')
                for activation in sample_activations
            ]
            with Pool(num_workers) as pool:
                results = pool.map(_plot_activation_wrapper, args_list)
            for result in results:
                print(result)
            print()

        # Generate grid plot for all activations
        if not args.activation:  # Only generate grid for full run
            print("Generating Blackhole activation comparison grid plot (all activations)...")
            plot_activation_comparison_grid(hardware_dir, 'blackhole_plots_values', 'Blackhole', 'blackhole_')
            print()

    print("=" * 80)
    print("✓ All hardware comparison plots generated")
    print("=" * 80)

if __name__ == '__main__':
    main()
