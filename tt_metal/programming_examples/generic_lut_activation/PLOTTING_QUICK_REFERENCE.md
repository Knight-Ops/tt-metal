# Plotting Tools - Quick Reference

## TL;DR

Two plotting tools for analyzing optimal degree selection:

1. **plot_optimal_degree_selection.py** - Per-segment degree choices, MAE breakdown
2. **plot_degree_segments_analysis.py** - Cross-configuration comparison, find best setup

## Quick Commands

### Generate LUT with Optimal Degree Search

```bash
./generate_luts.sh \
    --activation sigmoid \
    --degree hexic \
    --depth 8 \
    --precision bf16 \
    --optimal-degree-search \
    --plot-optimal-degree
```

**Output**:
- LUT file: `luts/piecewise_hexic_remez_sigmoid_8_bf16.lut`
- JSON data: `plots/optimal_degree_data/piecewise_hexic_remez_sigmoid_8_bf16.json`
- Plot: `plots/optimal_degree_selection/piecewise_hexic_remez_sigmoid_8_bf16.png`

### Compare Configurations

```bash
# Generate multiple configs
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

# Generate comparison plot
python3 plots/plot_degree_segments_analysis.py --activation sigmoid
```

**Output**: `plots/degree_segments_analysis/sigmoid_degree_segments_bf16.png`

## Plot Types Comparison

| Feature | Optimal Degree Selection | Degree-Segments Analysis |
|---------|--------------------------|--------------------------|
| **Shows** | Per-segment degree choices | Overall MAE trends |
| **X-axis** | Function domain | Number of segments |
| **Y-axis** | Function values | Total MAE (log scale) |
| **Use when** | Understanding why degrees chosen | Finding best configuration |
| **Highlights** | Optimal vs suboptimal MAE per segment | Best overall MAE curve |
| **Colors** | Degree numbers + MAE boxes | Degree lines |

## What's New

### ✅ Tufte-Style Design

- **Legend**: Horizontal single row at top (was: inside plot area)
- **MAE boxes**: Aligned rows (was: alternating heights)
- **Notation**: Compact `7.3e-4` (was: `7.30e-04`)
- **Grid**: Ultra-light 8% opacity (was: denser)
- **File size**: 218-308 KB (was: 334-460 KB)

### ✅ Three-Way Method Competition

Each degree now tries **all three methods** and picks the winner:

1. **Sollya fpminimax** - BF16-aware minimax
2. **OptimalPoly Remez** - Classic Remez exchange
3. **Least-squares** - Iterative precision-aware fit

**Method labels** at top of plot show which won per segment.

### ✅ Degree-Segments Analysis

New tool for cross-configuration comparison:

- Plots MAE vs segments for all degree/depth combinations
- Highlights best configuration in bold
- Separate plots for BF16 and FP32

## Reading the Plots

### Optimal Degree Selection Plot

```
┌─────────────────────────────────────────────────────┐
│ Legend (horizontal, top): Deg 2 | Deg 3 | Deg 6 ... │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Method Labels (top of plot):                       │
│    [Sollya] [Sollya] [LstSq] [Sollya] ...          │
│                                                      │
│  Function Curve + Degree Numbers (large, colored):  │
│    2  2  3  6  6  3  2  1                           │
│                                                      │
│  MAE Boxes (aligned rows):                          │
│    ┌─────┐ ┌─────┐ ┌─────┐  ← Green (optimal)      │
│    │7.3e-4│ │2.1e-4│ │8.2e-4│                       │
│    └─────┘ └─────┘ └─────┘                          │
│       ↑        ↑        ↑                            │
│    ┌─────┐ ┌─────┐ ┌─────┐  ← Red (suboptimal)     │
│    │2.1e-3│ │4.5e-3│ │1.2e-2│                       │
│    └─────┘ └─────┘ └─────┘                          │
│                                                      │
│  Metadata (top-left): n=8 | deg≤6 | x̄=3.0 | Δ=50%  │
└─────────────────────────────────────────────────────┘
```

**Key**:
- **Green box**: MAE with optimal degree
- **Red box**: MAE with fixed max degree
- **Arrow**: Improvement direction
- **Method label**: Which algorithm won (Sollya/Remez/LstSq)
- **Degree number**: Selected degree (large, colored, translucent)

### Degree-Segments Analysis Plot

```
┌─────────────────────────────────────────────────────┐
│ Title: SIGMOID - Optimal Degree Selection (BF16)   │
│        Best MAE: 0.000627 (Deg 8, n=16, uniform)   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  MAE                                                 │
│  (log)                                               │
│   ^                                                  │
│   │     ●──────●──────● Deg 8 (bold, pink)          │
│   │    /                                             │
│   │   ●───●───● Deg 6 (yellow)                      │
│   │  /                                               │
│   │ ●──● Deg 3 (blue)                               │
│   │                                                  │
│   └─────────────────────> Segments (4, 8, 16)       │
│                                                      │
│  Legend (bottom): Deg 3 | Deg 6 | Deg 8 | ...       │
└─────────────────────────────────────────────────────┘
```

**Key**:
- **Solid line**: Uniform segmentation
- **Dashed line**: Adaptive segmentation
- **Bold line**: Best configuration
- **"BEST" annotation**: Points to optimal result

## Method Competition Examples

### Sigmoid (smooth function)
```
Segment 0-6: Sollya wins (BF16-aware fitting optimal)
Segment 7:   LstSq wins  (edge case, numerical stability matters)
```

### Digamma (challenging function)
```
All segments: LstSq wins (Sollya struggles with poles/singularities)
```

## Interpreting Method Labels

| Method | When It Wins | Why |
|--------|--------------|-----|
| **Sollya** | Smooth functions, moderate domains | BF16-aware, minimax criterion |
| **Remez** | Sollya fails to converge | Classic algorithm, robust |
| **LstSq** | Complex/singular functions | Fallback, always stable |

**Gray color**: All three methods are valid, color only highlights failures (red)

## Common Questions

### Q: Why does Sollya win most segments?

**A**: Sollya's BF16-aware fitting accounts for quantization during optimization, producing better results after rounding to BF16.

### Q: Why aligned MAE rows?

**A**: Makes visual comparison easier - green boxes at same height, red boxes at same height.

### Q: Why scientific notation?

**A**: Compact format (7.3e-4 vs 7.30e-04) reduces clutter, easier to read.

### Q: Why legend at top?

**A**: Doesn't overlap with function curve, more space for data.

### Q: How to find best configuration?

**A**: Use degree-segments analysis - shows MAE trends across all configs, highlights best in bold.

## File Locations

```
plots/
├── optimal_degree_selection/          # Per-segment analysis
│   ├── piecewise_hexic_remez_sigmoid_8_bf16.png
│   ├── piecewise_octic_remez_sigmoid_16_bf16.png
│   └── ...
├── degree_segments_analysis/          # Cross-config comparison
│   ├── sigmoid_degree_segments_bf16.png
│   ├── sigmoid_degree_segments_fp32.png
│   └── ...
└── optimal_degree_data/               # JSON data
    ├── piecewise_hexic_remez_sigmoid_8_bf16.json
    └── ...
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Plot generation time | <1 second |
| File size range | 218-308 KB |
| Data-ink ratio | ~75% |
| Tufte compliance | ✅ Full |

## Customization

### Regenerate Plot from JSON

```bash
python3 plots/plot_optimal_degree_selection.py \
    --data-file plots/optimal_degree_data/my_config.json \
    --output custom_plot.png
```

### Change Output Directory

```bash
python3 plots/plot_degree_segments_analysis.py \
    --activation sigmoid \
    --output-dir my_custom_dir
```

## Documentation

- **TUFTE_PLOTTING_IMPROVEMENTS.md** - Design rationale
- **PLOTTING_UPDATES_SUMMARY.md** - Complete changelog
- **DEGREE_SEGMENTS_ANALYSIS.md** - New tool guide
- **README_OPTIMAL_DEGREE_PLOTS.md** - Feature reference

## Tips

1. **Generate multiple configs** before running degree-segments analysis
2. **Use BF16** for production (matches hardware precision)
3. **Check method labels** to understand why certain degrees won
4. **Compare uniform vs adaptive** segmentation strategies
5. **Look for plateau** in degree-segments plot (diminishing returns)

---

**Quick Start**: Run the "Generate LUT with Optimal Degree Search" command above, then open the generated plot!
