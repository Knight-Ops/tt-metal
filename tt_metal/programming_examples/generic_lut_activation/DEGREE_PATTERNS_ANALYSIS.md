# Degree Selection Patterns Analysis

## Overview

The degree selection patterns plot shows which polynomial degree was chosen for each segment across different {max_degree|segmentation} configurations.

## Key Features

- **X-axis**: Segment index (0, 1, 2, ..., n-1)
- **Y-axis**: Optimal degree selected (1, 2, 3, ..., max_degree)
- **Lines**: One per {max_degree|segmentation} configuration
- **Tufte styling**: Only the best configuration is highlighted, all others faded
- **Star markers**: Best configuration uses ★ markers on all data points

## Tufte Design Principles

### Highlight Only What Matters
- **Best configuration**: Dark green, bold line (2.5pt), star markers, full opacity
- **All other configurations**: Gray, thin lines (0.8pt), small markers, faded (35% opacity)

### Minimal Non-Data Ink
- Ultra-light grid (5% opacity)
- No top/right spines
- Thin bottom/left spines (0.5pt, gray)
- Legend at top outside plot area
- No shadows, rounded corners, or decorations

### Clear Information Hierarchy
1. Best curve stands out immediately (green + bold + stars)
2. Losing curves recede into background (gray + faded)
3. Data dominates, decoration minimized

## Usage

### Generate Data First

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
```

### Create Patterns Plot

```bash
# For a specific depth
python3 plots/plot_degree_selection_patterns.py \
    --activation sigmoid \
    --depth 8

# For different precision
python3 plots/plot_degree_selection_patterns.py \
    --activation sigmoid \
    --depth 8 \
    --precision fp32

# Custom data/output directories
python3 plots/plot_degree_selection_patterns.py \
    --activation sigmoid \
    --depth 8 \
    --data-dir ../data \
    --output-dir my_plots
```

## Output

Plots saved to: `plots/degree_segments_analysis/`

**Naming**: `{activation}_degree_patterns_d{depth}_{precision}.png`

Example: `sigmoid_degree_patterns_d8_bf16.png`

## Interpreting the Plot

### What It Shows

For each segment (x-axis), you can see which degree (y-axis) was selected by different configurations.

**Example interpretation**:
```
Sigmoid, depth=8, BF16

Segment 0: All configs choose degree 2
Segment 1: All configs choose degree 2
Segment 2: Max deg 3 → 2, Max deg 6 → 2, Max deg 8 → 2
Segment 3: Max deg 3 → 3, Max deg 6 → 6, Max deg 8 → 6 ★ (best)
...

Best config: Max deg 8 (adaptive) - MAE=0.000627
```

### Key Insights

1. **Degree consistency**: If all lines overlap, different max degrees choose the same degree pattern
2. **Complexity hotspots**: Segments where higher degrees are chosen indicate function complexity
3. **Diminishing returns**: If max deg 6 and max deg 8 choose same degrees, deg 8 doesn't help
4. **Segmentation impact**: Compare uniform (solid) vs adaptive (dashed) patterns

### Visual Cues

| Element | Meaning |
|---------|---------|
| **Green line + ★** | Best overall configuration (lowest MAE) |
| **Gray faded lines** | Suboptimal configurations |
| **Solid line** | Uniform segmentation |
| **Dashed line** | Adaptive segmentation |
| **Flat horizontal** | All segments use same degree |
| **Jagged pattern** | Varying complexity across segments |

## Example Output Analysis

### Sigmoid, depth=8, BF16

```
Configurations tested:
- Max deg 6 (uniform):  MAE=0.000733 - gray, faded
- Max deg 6 (adaptive): MAE=0.000694 - green, bold ★ BEST

Pattern insights:
- Segments 0-2: Both choose degree 2 (simple region)
- Segment 3: Both choose degree 6 (complex region)
- Segments 4-7: Both choose degree 1-3 (simple region)

Conclusion: Adaptive segmentation wins by 5.3%
```

### Digamma, depth=8, BF16

```
Configurations tested:
- Max deg 6 (uniform):  MAE=0.012345 - gray, faded
- Max deg 6 (adaptive): MAE=0.009876 - gray, faded
- Max deg 8 (uniform):  MAE=0.008765 - green, bold ★ BEST
- Max deg 8 (adaptive): MAE=0.009123 - gray, faded

Pattern insights:
- All segments: Higher degrees needed (4-8)
- Best uses uniform segmentation with max deg 8
- Digamma is challenging (poles/singularities)

Conclusion: Max deg 8 uniform wins, adaptive doesn't help for digamma
```

## Statistics Box

Bottom-left corner shows:
```
Best: Max deg 6 (adaptive)
MAE: 0.000694 | Avg deg: 3.2
Configs: 4
```

- **Best**: Winning configuration
- **MAE**: Total mean absolute error
- **Avg deg**: Average degree across all segments (coefficient efficiency)
- **Configs**: Number of configurations compared

## File Sizes

Typical file sizes with Tufte styling:

| Depth | File Size |
|-------|-----------|
| 4 | 220-240 KB |
| 8 | 220-230 KB |
| 16 | 300-310 KB |

Smaller than previous designs due to:
- Minimal grid (5% opacity vs 8%)
- Faded curves (35% alpha)
- No shadows or decorations
- Thin lines (0.5-0.8pt)

## Comparison with Optimal Degree Selection Plot

| Feature | Degree Patterns | Optimal Degree Selection |
|---------|-----------------|--------------------------|
| **Purpose** | Compare configurations | Understand single config |
| **X-axis** | Segment index | Function domain |
| **Y-axis** | Degree selected | Function values |
| **Shows** | Which degrees chosen | Why degrees chosen |
| **Highlights** | Best overall config | Per-segment MAE breakdown |
| **Use case** | Pick best setup | Debug degree choices |

## Best Practices

### 1. Generate Multiple Configs

Don't compare just one config - you need at least 2-3 to see patterns:

```bash
# Good - multiple max degrees
for deg in cubic hexic octic; do
    ./generate_luts.sh --activation sigmoid --degree $deg --depth 8 \
        --optimal-degree-search --plot-optimal-degree
done

# Bad - only one config
./generate_luts.sh --activation sigmoid --degree hexic --depth 8 \
    --optimal-degree-search --plot-optimal-degree
```

### 2. Compare Same Depth

Patterns are only comparable at the same depth:

```bash
# Good - same depth, different max degrees
python3 plots/plot_degree_selection_patterns.py --activation sigmoid --depth 8

# Bad - mixing depths (requires separate plots)
# Can't compare depth 4 and depth 8 in same plot
```

### 3. Look for Overlap

If lines overlap completely, higher max degree doesn't help:
- All choose same degrees → Lower max degree is sufficient
- Lines diverge → Higher max degree beneficial in some segments

### 4. Check Segmentation Impact

Compare solid (uniform) vs dashed (adaptive):
- Adaptive wins → Function has non-uniform complexity
- Uniform wins → Function complexity is evenly distributed

## Advanced Usage

### Batch Generate All Depths

```bash
for depth in 4 8 16 32; do
    python3 plots/plot_degree_selection_patterns.py \
        --activation sigmoid \
        --depth $depth
done

open plots/degree_segments_analysis/sigmoid_degree_patterns_*.png
```

### Compare Activations

```bash
for act in sigmoid tanh gelu hardswish; do
    python3 plots/plot_degree_selection_patterns.py \
        --activation $act \
        --depth 8
done

# Compare results
ls -lh plots/degree_segments_analysis/*_d8_bf16.png
```

### Programmatic Analysis

```python
import json
from pathlib import Path

def analyze_degree_patterns(activation, depth, precision='bf16'):
    """Find which segments need high degrees."""
    data_dir = Path('../data')

    for json_file in data_dir.glob(f'*{activation}*{depth}*{precision}.json'):
        with open(json_file) as f:
            data = json.load(f)

        degree_selections = data['degree_selections']
        high_degree_segments = [
            i for i, deg in enumerate(degree_selections) if deg >= 5
        ]

        print(f"{json_file.stem}:")
        print(f"  High-degree segments: {high_degree_segments}")
        print(f"  Average degree: {sum(degree_selections) / len(degree_selections):.1f}")

# Example
analyze_degree_patterns('sigmoid', 8)
```

## Troubleshooting

### No data found

**Problem**: "No data found for activation 'foo' with depth 8"

**Solution**: Generate LUTs first with `--optimal-degree-search`:

```bash
./generate_luts.sh --activation foo --degree hexic --depth 8 \
    --optimal-degree-search --plot-optimal-degree
```

### Only one curve shown

**Problem**: Plot shows only one line

**Solution**: Generate multiple configurations:

```bash
# Need at least 2 configs
./generate_luts.sh --activation sigmoid --degree cubic --depth 8 \
    --optimal-degree-search
./generate_luts.sh --activation sigmoid --degree hexic --depth 8 \
    --optimal-degree-search
```

### All curves look the same

**Problem**: All lines overlap perfectly

**Interpretation**: This is actually good data! It means:
- Higher max degrees don't help (choose same degrees anyway)
- Lower max degree is sufficient
- Save memory/computation with lower max degree

## References

- **Tufte, E. R.** (1983). *The Visual Display of Quantitative Information*
- **Tufte, E. R.** (2006). *Beautiful Evidence*

## Related Documentation

- `TUFTE_PLOTTING_IMPROVEMENTS.md` - Design rationale
- `README_OPTIMAL_DEGREE_PLOTS.md` - Optimal degree selection plots
- `PLOTTING_UPDATES_SUMMARY.md` - All plotting improvements

---

**Design Philosophy**: Show the winner clearly, fade the losers gracefully. Let the data speak for itself without visual clutter.
