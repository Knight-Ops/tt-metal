# Optimal Degree Selection Plots - Complete Feature Guide

## Overview

The optimal degree selection plotting system now includes **ALL** requested features:

### Plot Components

1. ✅ **Activation function curve** - The ground truth function
2. ✅ **Segment boundaries (breakpoints)** - Blue for critical points, gray for fill-in points
3. ✅ **Degree numbers** - Large, colored, translucent numbers showing selected degree in each segment
4. ✅ **Per-segment MAE** - Green boxes for optimal MAE, red boxes for suboptimal with improvement %
5. ✅ **Total MAE comparison** - Header showing overall improvement
6. ✅ **Method labels** - Small colored boxes showing which algorithm won (Sollya/Remez/LstSq)
7. ✅ **Metadata** - Statistics box with coefficient savings, mean degree, etc.
8. ✅ **Overlap handling** - Smart positioning, font sizing, and alternating heights

## Complete Usage Example

```bash
# Generate LUTs with complete visualization
./generate_luts.sh \
    --activation sigmoid \
    --degree hexic \
    --depth 8 \
    --precision bf16 \
    --optimal-degree-search \
    --plot-optimal-degree
```

**Generated Plot Shows:**

```
┌─────────────────────────────────────────────────────────────┐
│ SIGMOID - Optimal Degree Selection                          │
│ Fixed Degree 6: MAE=0.002630 → Optimal: MAE=0.000730 (+260%)│
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Legend          Sigmoid Curve with:                        │
│  □ Deg 1 (1)     - 8 vertical lines (breakpoints)          │
│  □ Deg 2 (4)     - Large colored numbers (degrees)          │
│  □ Deg 3 (1)     - Green boxes (optimal MAE)                │
│  □ Deg 6 (2)     - Red boxes (suboptimal MAE)              │
│                  - Method labels (Sollya/Remez/LstSq)       │
│                                                              │
│  Per Segment:                                               │
│  Seg 0: Degree "2" (teal) + "Sollya" label                 │
│         Green box: 1.90e-05                                 │
│         Red box: 2.03e-04 (+971%)                          │
│         Arrow: suboptimal → optimal                         │
│                                                              │
│  Seg 3: Degree "6" (yellow) + "Sollya" label              │
│         Green box: 6.63e-04                                 │
│         (No red box - degree 6 is optimal here)            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Method Labels Explained

Each segment shows which algorithm produced the best coefficients:

### Method Colors

- **Sollya** (dark green): Sollya's fpminimax succeeded and won
- **Remez** (dark blue): OptimalPoly Remez succeeded and won
- **LstSq** (dark orange): Least-squares fallback used and won
- **Failed** (dark red): All methods failed (rare)

### When Each Method Wins

**Sollya wins most often because:**
- BF16-aware fitting (considers quantization in optimization)
- Minimax criterion (optimizes worst-case error)
- Built-in function library (exp, erf, tanh, etc.)

**OptimalPoly Remez wins when:**
- Sollya fails to converge (e.g., digamma, wide ranges)
- Classic Remez gives better FP64 coefficients that survive BF16 quantization
- Function not in Sollya's library

**Least-Squares wins when:**
- Both Sollya and Remez fail or produce poor results
- Very simple segments (near-linear)
- Fallback provides stability

## Overlap Handling Features

The plotting script now handles text overlaps intelligently:

### 1. Adaptive Font Sizes

```python
# Degree numbers scaled based on narrowest segment
degree_fontsize = max(40, min(80, 80 * min_segment_width / 0.1))

# MAE boxes scaled similarly
mae_fontsize = max(7, min(9, 9 * min_segment_width / 0.1))
```

### 2. Alternating Heights

```python
# Narrow segments use alternating vertical positions
base_offset = 0.35 if i % 2 == 0 else 0.40
```

### 3. Text Rotation

```python
# Rotate text 45° for very narrow segments
if seg_width / x_range < 0.08:
    text_rotation = 45
```

### 4. Compact Formatting

```python
# Scientific notation for huge improvements
if improvement_pct > 10000:
    improvement_str = f'+{improvement_pct:.0e}'  # e.g., "+8.2e4"
elif improvement_pct > 1000:
    improvement_str = f'+{improvement_pct/1000:.1f}k'  # e.g., "+82.3k"
```

### 5. Smart Legend Positioning

```python
# Place legend in corner with most whitespace
if function_low_on_left:
    legend_loc = 'upper left'
elif function_low_on_right:
    legend_loc = 'upper right'
```

### 6. Dynamic Figure Height

```python
# Taller figure for many segments
fig_height = 8 if num_segments <= 8 else 10
```

## Example Output Annotations

### Segment with Large Improvement

```
┌────────────────────────┐
│ Segment 2: [-5.0,-2.5] │
│                        │
│    Degree: 2 (teal)    │ ← Large, translucent
│    Method: Sollya      │ ← Green box, top of segment
│                        │
│  ┌──────────────┐      │
│  │ 1.05e-03     │      │ ← Green: Optimal MAE
│  └──────────────┘      │
│         ↑              │
│         │              │ ← Red arrow
│  ┌──────────────┐      │
│  │ 8.64e-01     │      │ ← Red: Suboptimal MAE
│  │ (+82.3k%)    │      │ ← Improvement percentage
│  └──────────────┘      │
└────────────────────────┘
```

### Segment Where Max Degree is Optimal

```
┌────────────────────────┐
│ Segment 3: [-2.5, 0.0] │
│                        │
│    Degree: 6 (yellow)  │ ← Large, translucent
│    Method: Sollya      │ ← Green box, top of segment
│                        │
│  ┌──────────────┐      │
│  │ 6.63e-04     │      │ ← Green: Optimal MAE
│  └──────────────┘      │
│                        │ ← No red box (6 is optimal)
└────────────────────────┘
```

## File Outputs

Each run produces:

1. **LUT file**: `luts/piecewise_hexic_remez_sigmoid_8_bf16.lut`
2. **JSON data**: `plots/optimal_degree_data/piecewise_hexic_remez_sigmoid_8_bf16.json`
3. **PNG plot**: `plots/optimal_degree_selection/piecewise_hexic_remez_sigmoid_8_bf16.png`

### JSON Data Format

```json
{
  "function": "sigmoid",
  "max_degree": 6,
  "num_segments": 8,
  "degree_selections": [2, 2, 2, 6, 6, 3, 2, 1],
  "method_selections": ["Sollya", "Sollya", "Sollya", ...],
  "mae_per_segment_optimal": [1.90e-05, 2.00e-04, ...],
  "mae_per_segment_fixed": [2.03e-04, 2.47e-03, ...],
  "mae_total_optimal": 0.000730,
  "mae_total_fixed": 0.002630,
  "precision": "bf16",
  "breakpoints": [-10.0, -7.5, -5.0, ...]
}
```

## Interpreting the Plot

### Questions the Plot Answers

**Q: Which segments need high-degree polynomials?**
A: Look for large yellow/pink degree numbers (6, 8)

**Q: Where do lower degrees win?**
A: Look for red improvement boxes with large percentages

**Q: Which method is most reliable?**
A: Count method labels - typically Sollya dominates

**Q: Where is the function complex?**
A: High-degree segments with small MAE improvements

**Q: Where is the function simple?**
A: Low-degree segments (1-2) with huge MAE improvements

### Color Coding Guide

| Element | Color | Meaning |
|---------|-------|---------|
| Degree 1 | Red | Linear approximation |
| Degree 2 | Teal | Quadratic |
| Degree 3 | Blue | Cubic |
| Degree 6 | Yellow | Hexic |
| Degree 8 | Pink | Octic |
| Sollya | Dark Green | Sollya fpminimax |
| Remez | Dark Blue | OptimalPoly Remez |
| LstSq | Dark Orange | Least-squares |
| Optimal MAE | Light Green | Best achievable |
| Suboptimal MAE | Light Red | Fixed max degree |

## Advanced Usage

### Generate for Multiple Activations

```bash
for act in sigmoid tanh gelu exp; do
    ./generate_luts.sh \
        --activation $act \
        --degree hexic \
        --depth 8 \
        --optimal-degree-search \
        --plot-optimal-degree
done

# Compare plots side-by-side
open plots/optimal_degree_selection/piecewise_hexic_remez_{sigmoid,tanh,gelu,exp}_8_bf16.png
```

### Compare BF16 vs FP32

```bash
./generate_luts.sh \
    --activation sigmoid \
    --degree hexic \
    --depth 8 \
    --precision both \
    --optimal-degree-search \
    --plot-optimal-degree

# Compare
open plots/optimal_degree_selection/piecewise_hexic_remez_sigmoid_8_{bf16,fp32}.png
```

**Expected differences:**
- BF16: More low-degree selections (quantization favors simplicity)
- FP32: Higher degrees more competitive (better numerical stability)

### Regenerate Plot from Existing Data

```python
# Customize plot without re-running LUT generation
python3 plots/plot_optimal_degree_selection.py \
    --data-file plots/optimal_degree_data/piecewise_hexic_remez_sigmoid_8_bf16.json \
    --output my_custom_plot.png
```

## Performance Summary

From the sigmoid hexic example:

| Metric | Value |
|--------|-------|
| Segments | 8 |
| Max degree | 6 |
| Mean degree selected | 3.0 |
| Coefficient savings | 50% |
| MAE improvement | +260% |
| Methods used | 100% Sollya |

**Key Insight:** Even with max degree 6 available, optimal search selected mean degree 3.0, achieving **2.6× better accuracy** with **half the coefficients**!

## Troubleshooting

### Plot text overlaps

✅ **FIXED** - Now includes:
- Adaptive font sizing
- Alternating heights
- Text rotation
- Smart positioning

### Method labels missing

**Check:** JSON file has `method_selections` field

```bash
cat plots/optimal_degree_data/*.json | grep method_selections
```

If missing, regenerate with latest code.

### All methods show "Sollya"

**This is normal!** Sollya usually wins because:
1. BF16-aware optimization
2. Minimax criterion
3. Robust convergence

Remez/LstSq only win when Sollya fails or produces suboptimal results.

## References

- [`OPTIMAL_DEGREE_SEARCH.md`](OPTIMAL_DEGREE_SEARCH.md) - Algorithm details
- [`OPTIMAL_DEGREE_USAGE.md`](OPTIMAL_DEGREE_USAGE.md) - Usage examples
- [`OPTIMAL_DEGREE_PLOTTING.md`](OPTIMAL_DEGREE_PLOTTING.md) - Plot documentation
- [`PLOTTING_SUMMARY.md`](PLOTTING_SUMMARY.md) - Implementation summary

---

**Status:** ✅ **All requested features implemented and tested!**

- Breakpoints visualization
- Degree numbers (large, colored, translucent)
- Per-segment MAE (optimal + suboptimal with % improvement)
- Total MAE comparison
- Method labels (Sollya/Remez/LstSq)
- Overlap handling (smart positioning, sizing, rotation)
