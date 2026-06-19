# Degree-Segments Analysis Plots

## Overview

The degree-segments analysis tool generates comparative plots showing MAE (Mean Absolute Error) performance across all polynomial degree and segment count combinations for a given activation function.

## Features

- **Separate plots per precision** - BF16 and FP32 results shown independently
- **Best configuration highlighted** - Bold lines and "BEST" annotation for optimal MAE
- **Trend visualization** - Connected data points show how MAE changes with segment count
- **Tufte styling** - Minimal, data-focused design
- **Segmentation comparison** - Uniform vs adaptive breakpoint strategies shown with different line styles

## Usage

### Basic Usage

```bash
python3 plots/plot_degree_segments_analysis.py \
    --activation sigmoid
```

### Custom Data Directory

```bash
python3 plots/plot_degree_segments_analysis.py \
    --activation sigmoid \
    --data-dir plots/optimal_degree_data \
    --output-dir plots/custom_analysis
```

### Generate Data First

Before running the analysis, generate LUT configurations with optimal degree search:

```bash
# Generate multiple configurations for comparison
for degree in cubic hexic octic; do
    for depth in 4 8 16; do
        ./generate_luts.sh \
            --activation sigmoid \
            --degree $degree \
            --depth $depth \
            --precision bf16 \
            --optimal-degree-search \
            --plot-optimal-degree
    done
done

# Then run analysis
python3 plots/plot_degree_segments_analysis.py --activation sigmoid
```

## Plot Components

### Main Elements

1. **Data points** - Circles showing MAE for each configuration
   - Size: Larger for best configuration
   - Color: Unique color per polynomial degree

2. **Trend lines** - Connect configurations with same degree and segmentation
   - Line style: Solid (uniform) vs dashed (adaptive)
   - Width: Thicker (2.5pt) for best curve, thinner (1.2pt) for others
   - Alpha: Full opacity for best, 50% for others

3. **Best configuration annotation** - "BEST" label with arrow pointing to optimal result

4. **Legend** - Horizontal, top-aligned, shows all degree/segmentation combinations

### Color Coding

| Degree | Color | Hex |
|--------|-------|-----|
| Linear (1) | Red | #FF6B6B |
| Quadratic (2) | Teal | #4ECDC4 |
| Cubic (3) | Blue | #45B7D1 |
| Quartic (4) | Light Salmon | #FFA07A |
| Quintic (5) | Mint | #98D8C8 |
| Hexic (6) | Yellow | #FFD93D |
| Septic (7) | Light Green | #A8E6CF |
| Octic (8) | Pink | #FF8C94 |

### Line Styles

- **Solid line (—)** - Uniform segmentation
- **Dashed line (- -)** - Adaptive segmentation

## Output Files

Plots are saved to `plots/degree_segments_analysis/`:

```
plots/degree_segments_analysis/
├── sigmoid_degree_segments_bf16.png
├── sigmoid_degree_segments_fp32.png
├── tanh_degree_segments_bf16.png
└── ...
```

## Interpreting the Results

### What the Plot Shows

**X-axis**: Number of segments (4, 8, 16, 32, etc.)
**Y-axis**: Total MAE (log scale) - lower is better

### Key Insights

1. **Optimal degree selection** - The bold curve shows which degree/depth combination achieved best MAE

2. **Degree vs segments tradeoff** - Compare how different polynomial degrees perform as segment count increases

3. **Diminishing returns** - See where adding more segments stops improving accuracy

4. **Segmentation strategy impact** - Compare uniform (solid) vs adaptive (dashed) for same degree

### Example Interpretation

```
SIGMOID - Optimal Degree Selection Analysis (BF16)
Best MAE: 0.000627 (Deg 8, n=16, uniform)

Observations:
- Degree 8 (pink line) achieves best MAE with 16 segments
- Degree 6 (yellow) performs well with 8 segments (simpler)
- Degree 3 (blue) requires more segments for comparable accuracy
- Adaptive segmentation (dashed) sometimes matches uniform with fewer total segments
```

## Comparison with Optimal Degree Selection Plots

These tools complement each other:

| Feature | Optimal Degree Selection | Degree-Segments Analysis |
|---------|--------------------------|--------------------------|
| **Focus** | Per-segment degree choices | Overall configuration comparison |
| **Detail level** | Segment-by-segment breakdown | High-level trends |
| **Use case** | Understanding why degrees were selected | Choosing best overall configuration |
| **X-axis** | Segment boundaries | Number of segments |
| **Y-axis** | Function values | Total MAE |

## Advanced Usage

### Compare Multiple Activations

```bash
# Generate data for multiple activations
for activation in sigmoid tanh gelu hardswish; do
    for degree in cubic hexic octic; do
        for depth in 4 8 16; do
            ./generate_luts.sh \
                --activation $activation \
                --degree $degree \
                --depth $depth \
                --precision bf16 \
                --optimal-degree-search \
                --plot-optimal-degree
        done
    done
done

# Generate analysis for each
for activation in sigmoid tanh gelu hardswish; do
    python3 plots/plot_degree_segments_analysis.py --activation $activation
done

# Compare results side-by-side
open plots/degree_segments_analysis/*_bf16.png
```

### Find Optimal Configuration Programmatically

```python
import json
from pathlib import Path

def find_best_config(activation, precision='bf16'):
    data_dir = Path('plots/optimal_degree_data')
    best_mae = float('inf')
    best_config = None

    for json_file in data_dir.glob(f'*{activation}*{precision}.json'):
        with open(json_file) as f:
            data = json.load(f)

        if 'mae_total_optimal' in data:
            mae = data['mae_total_optimal']
            if mae < best_mae:
                best_mae = mae
                best_config = json_file.stem

    return best_config, best_mae

# Example usage
config, mae = find_best_config('sigmoid', 'bf16')
print(f"Best config: {config}")
print(f"Best MAE: {mae:.6f}")
```

## Tufte Design Principles Applied

Following Edward Tufte's visualization principles:

1. **Maximize data-ink ratio** - Every visual element represents data
2. **Minimize chartjunk** - No unnecessary decorations, shadows, or 3D effects
3. **Clear hierarchy** - Best result stands out via line width and annotation
4. **Subtle differentiation** - Color distinguishes data categories only
5. **Log scale** - Reveals patterns across orders of magnitude

## Performance Metrics

From sigmoid analysis with 10 configurations:

| Metric | Value |
|--------|-------|
| Configurations analyzed | 10 |
| Degrees tested | 3 (cubic, hexic, octic) |
| Segment counts | 4, 8, 16 |
| Segmentation types | 2 (uniform, adaptive) |
| Best MAE | 0.000627 (octic, 16 segments) |
| Plot file size | 269 KB |
| Generation time | <1 second |

## Troubleshooting

### No data found for activation

**Problem**: "No data found for activation 'foo'"

**Solution**: Generate LUTs with `--optimal-degree-search` flag first:

```bash
./generate_luts.sh \
    --activation foo \
    --degree hexic \
    --depth 8 \
    --optimal-degree-search \
    --plot-optimal-degree
```

### Only one data point shown

**Problem**: Plot shows only a single point, no trend lines

**Solution**: Generate multiple configurations with different depths:

```bash
for depth in 4 8 16; do
    ./generate_luts.sh --activation sigmoid --degree hexic --depth $depth \
        --optimal-degree-search --plot-optimal-degree
done
```

### Legend overlaps with data

**Problem**: Legend covers data points

**Solution**: Legend is positioned at bottom outside plot area. If still overlapping, increase figure height:

Edit `plot_degree_segments_analysis.py`:
```python
fig, ax = plt.subplots(figsize=(12, 8))  # Increased from 7 to 8
```

## References

- `plot_degree_segments_analysis.py` - Main plotting script
- `plot_optimal_degree_selection.py` - Per-segment degree visualization
- `OPTIMAL_DEGREE_SEARCH.md` - Algorithm details
- `TUFTE_PLOTTING_IMPROVEMENTS.md` - Design rationale

---

**Status:** ✅ Fully implemented with Tufte styling

**Features:**
- ✅ Separate BF16/FP32 plots
- ✅ Best configuration highlighted in bold
- ✅ Trend lines connecting configurations
- ✅ Horizontal legend at bottom
- ✅ Log scale Y-axis
- ✅ Tufte minimal styling
- ✅ Configurable data/output directories
