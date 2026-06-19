# Adaptive Segmentation Implementation Plan

**Status**: Waiting for boundary-value LUT testing to complete
**Goal**: Implement adaptive segmentation with critical boundary preservation

---

## Part 1: Generate and Store Adaptive Boundaries

### Decision: Separate CSV Files Per Depth ✅ RECOMMENDED

**Rationale:**
- ✅ Cleaner than massive CSV with multiple columns
- ✅ Easy to inspect specific depth configuration
- ✅ Matches existing pattern (we already vary depth in experiments)
- ✅ Avoids parsing complex array formats in CSV
- ✅ Each file is self-contained and simple

**File structure:**
```
generic_lut_activation/
├── activations_config.csv                    # Keep as-is (domain ranges only)
├── adaptive_segments/
│   ├── depth_4_segments.csv                  # 4 segments
│   ├── depth_8_segments.csv                  # 8 segments
│   ├── depth_16_segments.csv                 # 16 segments
│   ├── depth_32_segments.csv                 # 32 segments
│   └── README.md                             # Documentation
```

**CSV format (simple, one row per activation):**
```csv
activation,num_segments,breakpoints
sigmoid,4,"-5.0000,-0.5000,0.0000,0.5000,5.0000"
sigmoid,8,"-5.0000,-2.5000,-1.5000,-0.5000,0.0000,0.5000,1.5000,2.5000,5.0000"
relu,4,"-2.0000,-0.0010,0.0000,2.5000,5.0000"
...
```

**Critical boundaries guarantee:**
The generation algorithm must ensure:
1. Discontinuities always get a breakpoint (e.g., x=0 for relu)
2. When increasing depth, critical points from lower depths are preserved
3. Example: If depth=4 has breakpoint at x=0, depth=8 MUST also have x=0

---

## Script 1: Generate Adaptive Segment CSVs

**File:** `generate_adaptive_segments.py`

```python
#!/usr/bin/env python3
"""
generate_adaptive_segments.py - Generate adaptive segment boundaries for all activations

Creates CSV files for each depth with adaptive breakpoints that:
1. Concentrate segments in high-curvature regions
2. Always include discontinuities and inflection points
3. Preserve critical boundaries across different depths
4. Output separate CSV per depth for clarity

Usage:
    python3 generate_adaptive_segments.py                    # All depths
    python3 generate_adaptive_segments.py --depth 16         # Single depth
    python3 generate_adaptive_segments.py --verify           # Check consistency
"""

import numpy as np
import csv
import os
from pathlib import Path
from ground_truth import ACTIVATION_FUNCTIONS, get_activation_domain
from analyze_breakpoints import (
    suggest_adaptive_breakpoints,
    detect_derivative_discontinuities,
    find_inflection_points
)


def identify_critical_points(func_name, func, input_range):
    """
    Identify critical points that MUST appear in all depth configurations.

    Returns:
        List of x-coordinates that are mandatory breakpoints
    """
    critical = []

    # 1. Domain boundaries (always critical)
    x_min, x_max = input_range
    critical.extend([x_min, x_max])

    # 2. Derivative discontinuities (highest priority)
    discontinuities = detect_derivative_discontinuities(func_name, func, input_range)
    for disc in discontinuities:
        critical.append(disc['x'])

    # 3. Inflection points (high priority)
    inflections = find_inflection_points(func, input_range)
    for infl in inflections:
        critical.append(infl['x'])

    # 4. Function-specific critical points
    if func_name == 'relu' or func_name == 'leaky_relu' or func_name == 'prelu':
        # Ensure x=0 is always included
        if 0.0 not in critical and x_min < 0 < x_max:
            critical.append(0.0)

    elif func_name in ['hardsigmoid', 'hardswish']:
        # Piecewise boundaries at x = ±3
        if x_min < -3 < x_max:
            critical.append(-3.0)
        if x_min < 3 < x_max:
            critical.append(3.0)

    elif func_name == 'hardtanh':
        # Clip boundaries at x = ±1
        if x_min < -1 < x_max:
            critical.append(-1.0)
        if x_min < 1 < x_max:
            critical.append(1.0)

    elif func_name == 'relu6':
        # ReLU at 0, clip at 6
        if x_min < 0 < x_max:
            critical.append(0.0)
        if x_min < 6 < x_max:
            critical.append(6.0)

    elif func_name == 'threshold':
        # Threshold at x=0
        if x_min < 0 < x_max:
            critical.append(0.0)

    # Remove duplicates and sort
    critical = sorted(set(critical))

    return critical


def generate_adaptive_breakpoints_with_critical(func_name, func, input_range,
                                                target_segments, critical_points):
    """
    Generate adaptive breakpoints while ensuring critical points are included.

    Args:
        func_name: Activation function name
        func: Activation function
        input_range: (min, max) domain
        target_segments: Desired number of segments
        critical_points: List of x-coordinates that must be included

    Returns:
        Array of breakpoints (length = target_segments + 1)
    """
    # Start with base adaptive breakpoints
    base_breakpoints = suggest_adaptive_breakpoints(
        func_name, func, input_range, target_segments
    )

    # Merge critical points into breakpoints
    # Strategy: For each critical point, find nearest breakpoint and replace it
    breakpoints = base_breakpoints.copy()

    for critical_x in critical_points:
        # Find closest breakpoint (excluding boundaries)
        interior_breakpoints = breakpoints[1:-1]
        if len(interior_breakpoints) > 0:
            distances = np.abs(interior_breakpoints - critical_x)
            closest_idx = np.argmin(distances) + 1  # +1 to account for skipped first element

            # If critical point is close enough, snap to it
            if distances[closest_idx - 1] < (input_range[1] - input_range[0]) / target_segments:
                breakpoints[closest_idx] = critical_x
            else:
                # Otherwise, insert it and remove a nearby non-critical point
                # Insert critical point
                breakpoints = np.sort(np.append(breakpoints, critical_x))
                # Remove a point in a low-curvature region to maintain segment count
                # (Simple heuristic: remove point with largest adjacent segment)
                if len(breakpoints) > target_segments + 1:
                    widths = np.diff(breakpoints)
                    # Don't remove boundaries or critical points
                    removable_indices = [i+1 for i in range(len(widths)-1)
                                       if breakpoints[i+1] not in critical_points]
                    if removable_indices:
                        # Remove point with largest segment on either side
                        max_width_idx = max(removable_indices,
                                          key=lambda i: max(widths[i-1], widths[i]))
                        breakpoints = np.delete(breakpoints, max_width_idx)

    # Ensure exactly target_segments + 1 breakpoints
    while len(breakpoints) < target_segments + 1:
        # Add breakpoint in largest segment
        widths = np.diff(breakpoints)
        max_idx = np.argmax(widths)
        new_point = (breakpoints[max_idx] + breakpoints[max_idx + 1]) / 2
        breakpoints = np.sort(np.append(breakpoints, new_point))

    while len(breakpoints) > target_segments + 1:
        # Remove non-critical point in smallest segment
        widths = np.diff(breakpoints)
        removable = [i+1 for i in range(len(widths)-1)
                    if breakpoints[i+1] not in critical_points]
        if removable:
            min_idx = min(removable, key=lambda i: min(widths[i-1], widths[i]))
            breakpoints = np.delete(breakpoints, min_idx)
        else:
            break  # All remaining points are critical

    return breakpoints


def verify_critical_preservation(all_breakpoints, activation):
    """
    Verify that critical points from lower depths appear in higher depths.

    Args:
        all_breakpoints: Dict mapping depth -> breakpoints array
        activation: Function name for reporting

    Returns:
        (is_valid, warnings)
    """
    warnings = []
    depths = sorted(all_breakpoints.keys())

    for i in range(len(depths) - 1):
        lower_depth = depths[i]
        higher_depth = depths[i + 1]

        lower_points = set(np.round(all_breakpoints[lower_depth], 6))
        higher_points = set(np.round(all_breakpoints[higher_depth], 6))

        # Check if all lower depth points exist in higher depth (within tolerance)
        missing = []
        for point in lower_points:
            # Check if point exists in higher_points (within 1e-4 tolerance)
            if not any(abs(point - hp) < 1e-4 for hp in higher_points):
                missing.append(point)

        if missing:
            warnings.append(
                f"{activation}: depth {lower_depth} -> {higher_depth}: "
                f"Missing critical points: {missing}"
            )

    return len(warnings) == 0, warnings


def generate_all_adaptive_segments(depths=[4, 8, 16, 32], output_dir='adaptive_segments'):
    """
    Generate adaptive segment CSV files for all depths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all activations (exclude GLU variants)
    activations = [name for name in sorted(ACTIVATION_FUNCTIONS.keys())
                  if name not in ['glu', 'geglu', 'swiglu', 'reglu']]

    print("="*80)
    print("ADAPTIVE SEGMENT GENERATION")
    print("="*80)
    print(f"Activations: {len(activations)}")
    print(f"Depths: {depths}")
    print(f"Output: {output_dir}/")
    print()

    # Generate for each depth
    all_results = {depth: {} for depth in depths}

    for depth in depths:
        print(f"\nGenerating depth={depth} segments...")
        print("-"*80)

        csv_file = Path(output_dir) / f'depth_{depth}_segments.csv'

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['activation', 'num_segments', 'breakpoints'])

            for func_name in activations:
                try:
                    func = ACTIVATION_FUNCTIONS[func_name]
                    input_range = get_activation_domain(func_name)

                    # Identify critical points
                    critical = identify_critical_points(func_name, func, input_range)

                    # Generate adaptive breakpoints
                    breakpoints = generate_adaptive_breakpoints_with_critical(
                        func_name, func, input_range, depth, critical
                    )

                    # Store for verification
                    all_results[depth][func_name] = breakpoints

                    # Format breakpoints as comma-separated string
                    breakpoints_str = ','.join(f'{x:.6f}' for x in breakpoints)

                    writer.writerow([func_name, depth, breakpoints_str])

                    print(f"  {func_name:<15} {len(breakpoints)} points, "
                          f"{len(critical)} critical")

                except Exception as e:
                    print(f"  ERROR {func_name}: {e}")

        print(f"\n✓ Saved: {csv_file}")

    # Verification phase
    print("\n" + "="*80)
    print("VERIFICATION: Critical Point Preservation")
    print("="*80)

    all_valid = True
    for func_name in activations:
        func_breakpoints = {depth: all_results[depth][func_name] for depth in depths}
        is_valid, warnings = verify_critical_preservation(func_breakpoints, func_name)

        if not is_valid:
            all_valid = False
            for warning in warnings:
                print(f"⚠️  {warning}")

    if all_valid:
        print("✓ All critical points preserved across depths")

    print("\n" + "="*80)
    print("✓ Generation complete!")
    print(f"  Files: {output_dir}/depth_{{4,8,16,32}}_segments.csv")
    print("="*80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate adaptive segment boundaries')
    parser.add_argument('--depth', type=int, help='Generate for single depth only')
    parser.add_argument('--verify', action='store_true', help='Verify existing files')
    parser.add_argument('--output', type=str, default='adaptive_segments',
                       help='Output directory')
    args = parser.parse_args()

    if args.depth:
        generate_all_adaptive_segments(depths=[args.depth], output_dir=args.output)
    else:
        generate_all_adaptive_segments(output_dir=args.output)


if __name__ == "__main__":
    main()
```

---

## Part 2: Theoretical Error Analysis (Uniform vs Adaptive)

### Modify Existing Script

**File:** `plot_theoretical_error_comparison.py` (new, extends `plot_theoretical_error.py`)

```python
#!/usr/bin/env python3
"""
plot_theoretical_error_comparison.py - Compare uniform vs adaptive segmentation error

Generates side-by-side comparison plots showing:
1. Error distribution across domain (uniform vs adaptive)
2. MAE/Max Error bar charts
3. Error reduction percentage

Uses adaptive segment CSVs generated by generate_adaptive_segments.py

Usage:
    python3 plot_theoretical_error_comparison.py                      # All activations
    python3 plot_theoretical_error_comparison.py --activation sigmoid # Single activation
    python3 plot_theoretical_error_comparison.py --depth 16           # Single depth
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
from pathlib import Path
from ground_truth import ACTIVATION_FUNCTIONS, get_activation_domain


def load_adaptive_breakpoints(activation, depth, segments_dir='adaptive_segments'):
    """
    Load adaptive breakpoints from CSV.

    Returns:
        Array of breakpoints, or None if not found
    """
    csv_file = Path(segments_dir) / f'depth_{depth}_segments.csv'

    if not csv_file.exists():
        print(f"Warning: {csv_file} not found, using uniform segmentation")
        return None

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['activation'] == activation:
                # Parse comma-separated breakpoints
                breakpoints_str = row['breakpoints']
                breakpoints = np.array([float(x) for x in breakpoints_str.split(',')])
                return breakpoints

    return None


def evaluate_piecewise_linear_error(func, breakpoints):
    """
    Compute theoretical error for piecewise linear approximation.

    Args:
        func: Ground truth function
        breakpoints: Array of segment boundaries (length = num_segments + 1)

    Returns:
        (x_test, y_true, y_approx, errors)
    """
    num_segments = len(breakpoints) - 1
    x_min, x_max = breakpoints[0], breakpoints[-1]

    # Dense evaluation points
    x_test = np.linspace(x_min, x_max, 10000, dtype=np.float32)
    y_true = func(x_test)
    y_approx = np.zeros_like(x_test)

    # Fit and evaluate each segment
    for i in range(num_segments):
        seg_start = breakpoints[i]
        seg_end = breakpoints[i + 1]

        # Fit line using least squares on segment
        x_fit = np.linspace(seg_start, seg_end, 100)
        y_fit = func(x_fit)

        # Least squares: y = slope * x + offset
        X = np.vstack([x_fit, np.ones_like(x_fit)]).T
        coeffs = np.linalg.lstsq(X, y_fit, rcond=None)[0]
        slope, offset = coeffs[0], coeffs[1]

        # Evaluate on test points in this segment
        mask = (x_test >= seg_start) & (x_test < seg_end)
        y_approx[mask] = slope * x_test[mask] + offset

    # Handle last point
    y_approx[-1] = slope * x_test[-1] + offset

    # Compute errors
    errors = np.abs(y_true - y_approx)

    return x_test, y_true, y_approx, errors


def compare_segmentation_error(func_name, depth=16, method='piecewise_linear'):
    """
    Compare uniform vs adaptive segmentation error.

    Returns:
        Dictionary with comparison metrics
    """
    func = ACTIVATION_FUNCTIONS[func_name]
    input_range = get_activation_domain(func_name)
    x_min, x_max = input_range

    # 1. Uniform segmentation
    uniform_breakpoints = np.linspace(x_min, x_max, depth + 1)
    x_uni, y_true, y_uni, err_uni = evaluate_piecewise_linear_error(func, uniform_breakpoints)

    # 2. Adaptive segmentation
    adaptive_breakpoints = load_adaptive_breakpoints(func_name, depth)
    if adaptive_breakpoints is None:
        print(f"⚠️  No adaptive breakpoints for {func_name}, depth={depth}")
        return None

    x_adp, _, y_adp, err_adp = evaluate_piecewise_linear_error(func, adaptive_breakpoints)

    # Compute metrics
    results = {
        'activation': func_name,
        'depth': depth,
        'uniform': {
            'breakpoints': uniform_breakpoints,
            'mae': np.mean(err_uni),
            'rmse': np.sqrt(np.mean(err_uni**2)),
            'max_error': np.max(err_uni),
            'errors': err_uni,
            'x': x_uni,
            'y': y_uni
        },
        'adaptive': {
            'breakpoints': adaptive_breakpoints,
            'mae': np.mean(err_adp),
            'rmse': np.sqrt(np.mean(err_adp**2)),
            'max_error': np.max(err_adp),
            'errors': err_adp,
            'x': x_adp,
            'y': y_adp
        },
        'y_true': y_true
    }

    # Improvement ratios
    results['mae_improvement'] = results['uniform']['mae'] / results['adaptive']['mae']
    results['max_error_improvement'] = results['uniform']['max_error'] / results['adaptive']['max_error']

    return results


def plot_comparison(results, output_dir='plots_error_comparison'):
    """
    Generate comparison visualization.
    """
    os.makedirs(output_dir, exist_ok=True)

    func_name = results['activation']
    depth = results['depth']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{func_name} - Uniform vs Adaptive Segmentation (depth={depth})',
                 fontsize=16, fontweight='bold')

    # Plot 1: Error distribution (Uniform)
    ax = axes[0, 0]
    x_uni = results['uniform']['x']
    err_uni = results['uniform']['errors']
    ax.fill_between(x_uni, 0, err_uni, alpha=0.3, color='red', label='Error region')
    ax.plot(x_uni, err_uni, 'r-', linewidth=1, label='Absolute error')

    # Mark segment boundaries
    for bp in results['uniform']['breakpoints']:
        ax.axvline(bp, color='gray', alpha=0.3, linestyle='--', linewidth=1)

    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title(f'UNIFORM - MAE: {results["uniform"]["mae"]:.6f}, '
                f'Max: {results["uniform"]["max_error"]:.6f}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Error distribution (Adaptive)
    ax = axes[0, 1]
    x_adp = results['adaptive']['x']
    err_adp = results['adaptive']['errors']
    ax.fill_between(x_adp, 0, err_adp, alpha=0.3, color='green', label='Error region')
    ax.plot(x_adp, err_adp, 'g-', linewidth=1, label='Absolute error')

    # Mark segment boundaries
    for bp in results['adaptive']['breakpoints']:
        ax.axvline(bp, color='gray', alpha=0.3, linestyle='--', linewidth=1)

    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title(f'ADAPTIVE - MAE: {results["adaptive"]["mae"]:.6f}, '
                f'Max: {results["adaptive"]["max_error"]:.6f}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: Direct comparison (overlay)
    ax = axes[1, 0]
    ax.plot(x_uni, err_uni, 'r-', linewidth=2, label='Uniform', alpha=0.7)
    ax.plot(x_adp, err_adp, 'g-', linewidth=2, label='Adaptive', alpha=0.7)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Error Comparison (log scale)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 4: Metrics bar chart
    ax = axes[1, 1]
    metrics = ['MAE', 'Max Error']
    uniform_vals = [results['uniform']['mae'], results['uniform']['max_error']]
    adaptive_vals = [results['adaptive']['mae'], results['adaptive']['max_error']]

    x_pos = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, uniform_vals, width, label='Uniform', color='red', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, adaptive_vals, width, label='Adaptive', color='green', alpha=0.7)

    ax.set_ylabel('Error Value', fontsize=12)
    ax.set_title(f'Error Metrics Comparison\n'
                f'MAE: {results["mae_improvement"]:.2f}× better, '
                f'Max: {results["max_error_improvement"]:.2f}× better',
                fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Add improvement text
    for i, (u, a) in enumerate(zip(uniform_vals, adaptive_vals)):
        improvement = u / a
        ax.text(i, max(u, a) * 1.5, f'{improvement:.1f}×',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    output_file = Path(output_dir) / f'{func_name}_depth{depth}_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def generate_summary_table(all_results, output_file='error_comparison_summary.csv'):
    """
    Generate CSV summary of all comparisons.
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Activation', 'Depth',
            'Uniform MAE', 'Adaptive MAE', 'MAE Improvement',
            'Uniform Max', 'Adaptive Max', 'Max Improvement'
        ])

        for result in all_results:
            writer.writerow([
                result['activation'],
                result['depth'],
                f"{result['uniform']['mae']:.8f}",
                f"{result['adaptive']['mae']:.8f}",
                f"{result['mae_improvement']:.2f}",
                f"{result['uniform']['max_error']:.8f}",
                f"{result['adaptive']['max_error']:.8f}",
                f"{result['max_error_improvement']:.2f}"
            ])

    print(f"✓ Summary table: {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare uniform vs adaptive segmentation error')
    parser.add_argument('--activation', type=str, help='Single activation to analyze')
    parser.add_argument('--depth', type=int, default=16, help='Segment depth')
    parser.add_argument('--segments-dir', type=str, default='adaptive_segments',
                       help='Directory containing adaptive segment CSVs')
    args = parser.parse_args()

    print("="*80)
    print("THEORETICAL ERROR COMPARISON: Uniform vs Adaptive")
    print("="*80)

    # Determine activations
    if args.activation:
        activations = [args.activation]
    else:
        activations = [name for name in sorted(ACTIVATION_FUNCTIONS.keys())
                      if name not in ['glu', 'geglu', 'swiglu', 'reglu']]

    print(f"Activations: {len(activations)}")
    print(f"Depth: {args.depth}")
    print()

    all_results = []

    for func_name in activations:
        try:
            print(f"Analyzing {func_name}...")
            results = compare_segmentation_error(func_name, depth=args.depth)

            if results:
                all_results.append(results)

                # Print summary
                print(f"  Uniform:  MAE={results['uniform']['mae']:.6f}, "
                     f"Max={results['uniform']['max_error']:.6f}")
                print(f"  Adaptive: MAE={results['adaptive']['mae']:.6f}, "
                     f"Max={results['adaptive']['max_error']:.6f}")
                print(f"  Improvement: {results['mae_improvement']:.2f}× MAE, "
                     f"{results['max_error_improvement']:.2f}× Max")

                # Generate plot
                plot_comparison(results)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary table
    if all_results:
        generate_summary_table(all_results)

        # Print overall statistics
        mae_improvements = [r['mae_improvement'] for r in all_results]
        max_improvements = [r['max_error_improvement'] for r in all_results]

        print("\n" + "="*80)
        print("OVERALL STATISTICS")
        print("="*80)
        print(f"Functions analyzed: {len(all_results)}")
        print(f"Average MAE improvement: {np.mean(mae_improvements):.2f}×")
        print(f"Average Max Error improvement: {np.mean(max_improvements):.2f}×")
        print(f"Best MAE improvement: {np.max(mae_improvements):.2f}× "
             f"({all_results[np.argmax(mae_improvements)]['activation']})")
        print(f"Best Max improvement: {np.max(max_improvements):.2f}× "
             f"({all_results[np.argmax(max_improvements)]['activation']})")
        print("="*80)


if __name__ == "__main__":
    main()
```

---

## Implementation Timeline

### Phase 1: Generate Adaptive Segments (Day 1)
```bash
# 1. Create the generation script
vim generate_adaptive_segments.py  # Copy from above

# 2. Generate all adaptive segment CSVs
python3 generate_adaptive_segments.py

# 3. Verify output
ls adaptive_segments/
# Should see: depth_4_segments.csv, depth_8_segments.csv, depth_16_segments.csv, depth_32_segments.csv

# 4. Inspect one file
head -20 adaptive_segments/depth_16_segments.csv
```

**Expected output structure:**
```
adaptive_segments/
├── depth_4_segments.csv    (29 rows + header)
├── depth_8_segments.csv    (29 rows + header)
├── depth_16_segments.csv   (29 rows + header)
├── depth_32_segments.csv   (29 rows + header)
└── README.md
```

### Phase 2: Theoretical Error Analysis (Day 2-3)
```bash
# 1. Create comparison script
vim plot_theoretical_error_comparison.py  # Copy from above

# 2. Run for single activation (quick test)
python3 plot_theoretical_error_comparison.py --activation sigmoid --depth 16

# 3. Run for all activations
python3 plot_theoretical_error_comparison.py --depth 16

# 4. Check output
ls plots_error_comparison/
# Should see: sigmoid_depth16_comparison.png, relu_depth16_comparison.png, ...
# Also: error_comparison_summary.csv
```

### Phase 3: Multi-Depth Analysis (Day 4)
```bash
# Compare across all depths
for depth in 4 8 16 32; do
    python3 plot_theoretical_error_comparison.py --depth $depth
done

# Generate aggregate plots
python3 plot_depth_vs_improvement.py  # Create this script
```

---

## Deliverables

### 1. CSV Files (Ready for LUT generation)
```
adaptive_segments/
├── depth_4_segments.csv
├── depth_8_segments.csv
├── depth_16_segments.csv
└── depth_32_segments.csv
```

### 2. Error Analysis Plots
```
plots_error_comparison/
├── sigmoid_depth16_comparison.png
├── relu_depth16_comparison.png
├── exp_depth16_comparison.png
└── ... (29 activations)
```

### 3. Summary Tables
```
error_comparison_summary.csv
error_improvement_by_depth.csv
```

---

## Usage in LUT Generation (Future)

Once boundary-value testing is complete:

```python
# In generate_piecewise_*_luts.py
def load_adaptive_breakpoints(activation, depth):
    csv_file = f'adaptive_segments/depth_{depth}_segments.csv'
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['activation'] == activation:
                return [float(x) for x in row['breakpoints'].split(',')]
    return None

# Modified main loop
for depth in [4, 8, 16, 32]:
    for func_name in activations:
        # Load adaptive breakpoints
        breakpoints = load_adaptive_breakpoints(func_name, depth)

        # Generate LUT using these breakpoints
        coeffs = fit_polynomial_segments(func, breakpoints)
        write_lut(f'piecewise_linear_adaptive_{func_name}_{depth}.lut', coeffs)
```

---

## Next Steps

1. ✅ Wait for boundary-value LUT testing to complete
2. ⬜ Implement `generate_adaptive_segments.py`
3. ⬜ Run and verify CSV generation
4. ⬜ Implement `plot_theoretical_error_comparison.py`
5. ⬜ Generate comparison plots for all activations
6. ⬜ Review results and tune algorithm if needed
7. ⬜ Integrate with LUT generation pipeline

Let me know when boundary testing is done and we'll execute this plan!
