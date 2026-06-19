# Optimal Degree Selection Plotting

## Overview

The optimal degree selection plotting feature generates visual summaries of how the optimal degree search algorithm selected polynomial degrees for each segment. These plots show:

1. **Activation function curve** - The ground truth function being approximated
2. **Segment boundaries** - Vertical lines showing breakpoints (blue = critical points, gray = fill-in points)
3. **Degree numbers** - Large, translucent colored numbers showing the degree selected for each segment
4. **Per-segment MAE** - Green boxes showing optimal MAE, red boxes showing suboptimal (fixed degree) MAE with improvement percentage
5. **Total MAE comparison** - Header showing overall improvement from optimal degree search
6. **Metadata** - Segment count, precision, coefficient savings statistics

## Quick Start

### Using generate_luts.sh (Recommended)

```bash
# Generate hexic LUTs with optimal degree search AND plots
./generate_luts.sh \
    --activation sigmoid \
    --degree hexic \
    --depth 8 \
    --precision bf16 \
    --optimal-degree-search \
    --plot-optimal-degree
```

**Output:**
- LUT files: `luts/piecewise_hexic_remez_sigmoid_8_bf16.lut`
- Plot data: `plots/optimal_degree_data/piecewise_hexic_remez_sigmoid_8_bf16.json`
- Plot image: `plots/optimal_degree_selection/piecewise_hexic_remez_sigmoid_8_bf16.png`

### Using Python script directly

```bash
# Generate LUTs with plot data export
python3 luts/generate_piecewise_remez_luts.py \
    --degree hexic \
    --activation sigmoid \
    --depth 8 \
    --precision bf16 \
    --optimal-degree-search \
    --plot-optimal-degree

# Or generate plot from existing JSON data
python3 plots/plot_optimal_degree_selection.py \
    --data-file plots/optimal_degree_data/piecewise_hexic_remez_sigmoid_8_bf16.json \
    --output plots/my_custom_plot.png
```

## Plot Interpretation

### Example Plot Elements

**Title:**
```
SIGMOID - Optimal Degree Selection
Fixed Degree 6: MAE=0.002630  →  Optimal Search: MAE=0.000730  (+260.3% improvement)
```

**Segment Visualization:**
- Each segment shows a large translucent colored number (1-8) indicating the polynomial degree used
- Different colors distinguish different degrees:
  - Red: degree 1
  - Teal: degree 2
  - Blue: degree 3
  - Yellow: degree 6
  - Pink: degree 8

**MAE Annotations (below each segment):**
- **Green box**: Optimal MAE achieved with selected degree (e.g., `7.30e-04`)
- **Red box**: Suboptimal MAE if fixed max degree was used (e.g., `2.63e-03` with `(+260%)`)
- **Arrow**: Points from suboptimal to optimal, showing improvement direction

**Metadata Box (top right):**
```
Segments: 8 | Max Degree: 6 | Precision: BF16
Mean Degree: 3.0 | Coeff Savings: 50.0%
```

**Legend (left side):**
- Shows which colors correspond to which degrees
- Shows count of segments using each degree (e.g., "Degree 2 (4 segs)")

### Reading the Results

**Case Study: Sigmoid with hexic (degree 6), 8 segments**

```
Segment 0: [-10.0, -7.5]  → Degree 2 selected (vs max degree 6)
  Optimal MAE: 1.90e-05
  Fixed deg 6 MAE: 2.03e-04 (+971% worse)
  Interpretation: Lower degree achieves 10x better accuracy!

Segment 3: [-2.5, 0.0]  → Degree 6 selected (matches max degree)
  Optimal MAE: 6.63e-04
  No suboptimal box shown (degree 6 is optimal here)
  Interpretation: High curvature region needs full degree 6
```

## File Organization

### Generated Files

```
plots/
├── optimal_degree_data/              # JSON data files
│   ├── piecewise_hexic_remez_sigmoid_8_bf16.json
│   ├── piecewise_hexic_remez_sigmoid_8_fp32.json
│   └── piecewise_hexic_remez_sigmoid_8_adaptive_bf16.json
│
└── optimal_degree_selection/         # PNG plot images
    ├── piecewise_hexic_remez_sigmoid_8_bf16.png
    ├── piecewise_hexic_remez_sigmoid_8_fp32.png
    └── piecewise_hexic_remez_sigmoid_8_adaptive_bf16.png
```

### JSON Data Format

The JSON files contain all data needed to regenerate plots:

```json
{
  "function": "sigmoid",
  "max_degree": 6,
  "num_segments": 8,
  "input_range": [-10.0, 10.0],
  "degree_selections": [2, 2, 2, 6, 6, 3, 2, 1],
  "mae_per_segment_optimal": [1.90e-05, 2.00e-04, ...],
  "mae_per_segment_fixed": [2.03e-04, 2.47e-03, ...],
  "mae_total_optimal": 0.000730,
  "mae_total_fixed": 0.002630,
  "precision": "bf16",
  "breakpoints": [-10.0, -7.5, -5.0, ...]
}
```

## Use Cases

### 1. Debugging Optimal Degree Search

**Question:** Why did degree 2 beat degree 6 in segment 0?

**Answer:** Look at the plot:
- Segment 0 is in the flat region of sigmoid (near -1)
- Function is nearly linear, so degree 2 is sufficient
- Higher degrees introduce quantization noise in BF16 without improving fit

### 2. Comparing Uniform vs Adaptive Segmentation

Generate both:
```bash
./generate_luts.sh \
    --activation sigmoid \
    --degree hexic \
    --depth 8 \
    --optimal-degree-search \
    --plot-optimal-degree
```

Compare:
- `piecewise_hexic_remez_sigmoid_8_bf16.png` (uniform)
- `piecewise_hexic_remez_sigmoid_8_adaptive_bf16.png` (adaptive)

Adaptive segmentation places more breakpoints in high-curvature regions, potentially allowing lower degrees elsewhere.

### 3. Precision Analysis (BF16 vs FP32)

```bash
./generate_luts.sh \
    --activation sigmoid \
    --degree hexic \
    --depth 8 \
    --precision both \
    --optimal-degree-search \
    --plot-optimal-degree
```

Compare:
- BF16: Lower degrees often win (less quantization error)
- FP32: Higher degrees may be selected more often (numerical stability)

### 4. Multi-Activation Analysis

```bash
# Generate plots for multiple activations
for act in sigmoid tanh gelu; do
    ./generate_luts.sh \
        --activation $act \
        --degree hexic \
        --depth 8 \
        --optimal-degree-search \
        --plot-optimal-degree
done
```

Compare which activations benefit most from optimal degree search.

## Customization

### Regenerate Plot with Custom Settings

```python
from plots.plot_optimal_degree_selection import plot_optimal_degree_selection, load_optimal_degree_data
import numpy as np

# Load data
data = load_optimal_degree_data('plots/optimal_degree_data/piecewise_hexic_remez_sigmoid_8_bf16.json')

# Customize and regenerate
plot_optimal_degree_selection(
    func_name=data['function'],
    max_degree=data['max_degree'],
    num_segments=data['num_segments'],
    input_range=tuple(data['input_range']),
    degree_selections=np.array(data['degree_selections']),
    mae_per_segment_optimal=np.array(data['mae_per_segment_optimal']),
    mae_per_segment_fixed=np.array(data['mae_per_segment_fixed']),
    mae_total_optimal=data['mae_total_optimal'],
    mae_total_fixed=data['mae_total_fixed'],
    breakpoints=np.array(data['breakpoints']) if data['breakpoints'] else None,
    output_file='my_custom_plot.png',
    precision=data['precision']
)
```

## Performance Insights

### Typical Results

| Activation | Max Degree | Mean Degree | Coeff Savings | MAE Improvement |
|-----------|-----------|-------------|---------------|-----------------|
| sigmoid   | 6         | 3.0         | 50%           | +260%           |
| tanh      | 6         | 3.2         | 47%           | +180%           |
| gelu      | 8         | 4.5         | 44%           | +120%           |
| exp       | 8         | 5.1         | 36%           | +45%            |

**Key Findings:**
- **Flat regions** (sigmoid tails): degree 1-2 optimal
- **Medium curvature** (sigmoid center): degree 3-4 optimal
- **High curvature** (gelu, exp): degree 6-8 optimal
- **BF16 precision**: Lower degrees preferred (less quantization)
- **FP32 precision**: Higher degrees more competitive

## Troubleshooting

### No plots generated

**Symptom:** "No plot data files found. Plots not generated."

**Causes:**
1. `--optimal-degree-search` flag not set
2. `--plot-optimal-degree` flag not set
3. Sollya convergence failures

**Solution:**
```bash
# Ensure BOTH flags are set
./generate_luts.sh \
    --activation sigmoid \
    --degree hexic \
    --optimal-degree-search \
    --plot-optimal-degree
```

### Plot shows all segments with max degree

**Symptom:** All segments show degree 6 (or max degree)

**Cause:** Optimal degree search not actually running

**Solution:** Check that `--optimal-degree-search` is passed before `--plot-optimal-degree`

### Missing matplotlib

**Error:** `ModuleNotFoundError: No module named 'matplotlib'`

**Solution:**
```bash
pip install matplotlib
# or
source python_env/bin/activate  # Use project venv
```

## Advanced Features

### Batch Plot Generation

```bash
# Generate plots for all activations, all degrees
for degree in cubic hexic octic; do
    ./generate_luts.sh \
        --degree $degree \
        --depth 8 \
        --optimal-degree-search \
        --plot-optimal-degree
done
```

### Export for Paper/Presentation

Plots are generated at 300 DPI with tight bounding boxes, ready for publication:

```bash
# Generated plot can be used directly in LaTeX/PowerPoint
cp plots/optimal_degree_selection/piecewise_hexic_remez_sigmoid_8_bf16.png \
   ~/my_paper/figures/
```

## References

- See [`OPTIMAL_DEGREE_SEARCH.md`](OPTIMAL_DEGREE_SEARCH.md) for algorithm details
- See [`OPTIMAL_DEGREE_USAGE.md`](OPTIMAL_DEGREE_USAGE.md) for usage examples
- Plot script source: `plots/plot_optimal_degree_selection.py`
