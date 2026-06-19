# Plotting Guide for Embedded LUT Results

This guide shows how to visualize your benchmark results from the `data/*.csv` files.

## Quick Start

Generate Pareto frontier plots (both MAE and max_error):
```bash
python3 plots/plot_pareto_frontier.py --arch-prefix blackhole --metric both
python3 plots/plot_pareto_frontier.py --arch-prefix wormhole --metric both
```

Or generate plots for a specific metric only:
```bash
python3 plots/plot_pareto_frontier.py --arch-prefix blackhole --metric mae
python3 plots/plot_pareto_frontier.py --arch-prefix blackhole --metric max_error
```

Generate depth-degree heatmaps:
```bash
python3 plots/plot_depth_degree.py --arch-prefix blackhole
python3 plots/plot_depth_degree.py --arch-prefix wormhole
```

Find best configurations for each activation (based on min max_error):
```bash
python3 plots/best.py --arch blackhole
python3 plots/best.py --arch blackhole --save  # Also saves CSV + Markdown
```

## Available Plotting Tools

### 1. Pareto Frontier Analysis (`plots/plot_pareto_frontier.py`)

**What it does:**
- Shows polynomial degree vs depth tradeoffs
- Supports both MAE (average error) and max_error (worst-case error) metrics
- Compares different segmentation strategies
- Shows all degrees: Deg 1, Deg 2, Deg 3, Deg 4, Deg 6, Deg 8
- For heatmaps, use `plot_depth_degree.py` instead

**Usage:**
```bash
# Generate both MAE and max_error plots
python3 plots/plot_pareto_frontier.py --arch-prefix blackhole --metric both

# Generate only MAE plots
python3 plots/plot_pareto_frontier.py --arch-prefix blackhole --metric mae

# Generate only max_error plots
python3 plots/plot_pareto_frontier.py --arch-prefix blackhole --metric max_error
```

**Output files (in `plots/{arch}/pareto/`):**
- `all_activations_degree_vs_mae_with_sfpu.png` - MAE Pareto plot WITH SFPU support (includes SFPU baseline)
- `all_activations_degree_vs_mae_without_sfpu.png` - MAE Pareto plot WITHOUT SFPU support
- `per_activation_mae_lut_vs_sfpu.png` - Per-activation MAE comparison (if SFPU data available)
- `all_activations_degree_vs_max_with_sfpu.png` - Max error Pareto plot WITH SFPU support
- `all_activations_degree_vs_max_without_sfpu.png` - Max error Pareto plot WITHOUT SFPU support
- `per_activation_max_lut_vs_sfpu.png` - Per-activation max error comparison (if SFPU data available)

**Note:** SFPU data is loaded from `../generic_lut_activation/data/{arch}_native_sfpu_results.csv`

**Key insights provided:**
- Best overall accuracy configuration
- Best configuration for memory budgets (<100 coeffs, 100-500 coeffs)
- Pareto-optimal points (no better accuracy without using more memory)

### 2. Depth-Degree Heatmaps (`plots/plot_depth_degree.py`)

**What it does:**
- Generates heatmaps showing error values across depth (segments) and polynomial degree
- Creates both combined (all activations) and per-activation heatmaps
- Includes rational function data when available (2x2 layout)
- Highlights cheapest best configuration with prominent black box
- Uses polynomial-fitter styling (RdYlGn_r colormap, log scale)

**Usage:**
```bash
python3 plots/plot_depth_degree.py --arch-prefix blackhole
```

**Output files (in `plots/{arch}/depth_degree/`):**
- `all_activations_heatmap.png` - Combined heatmap (all activations)
- `{activation}_heatmap.png` - Per-activation heatmaps (all 27 activations)

**Layout:**
- 1x2 grid when only polynomial data: [Polynomial MAE, Polynomial Max]
- 2x2 grid when rational data available: [Polynomial MAE, Polynomial Max, Rational MAE, Rational Max]

**Black box highlighting:**
- Shows the cheapest (lowest memory) configuration that achieves the best (minimum) error
- Helps identify optimal configurations when memory is constrained

**Styling:**
- RdYlGn_r colormap (green=good, red=bad)
- Log scale with vmin=-10 for zero values
- Scientific notation with superscript exponents
- Black text annotations on cells

### 3. Best Configuration Finder (`plots/best.py`)

**What it does:**
- Identifies optimal LUT configuration for each activation
- Selects minimum max_error (best worst-case bound), then fastest runtime among ties
- Provides summary statistics

**Usage:**
```bash
# Show all activations
python3 plots/best.py --arch blackhole

# Show specific activation
python3 plots/best.py --arch blackhole --activation gelu

# Save to CSV and Markdown
python3 plots/best.py --arch blackhole --save
```

**Output:**
- Terminal table with best configs
- `{arch}_best_lut_configs.csv` (if --save)
- `{arch}_best_lut_configs.md` (if --save)

### 4. ASCII Terminal Plots (`ascii_plot.sh`)

**What it does:**
- Quick terminal-based visualization
- No matplotlib/GUI required
- Great for quick checks on remote servers

**Usage:**
```bash
./ascii_plot.sh --activation gelu --metric mae
./ascii_plot.sh --precision fp32 --metric rmse
```

### 5. Parent Directory Tools (Advanced)

Located in `../generic_lut_activation/plots/`:

**Error vs Runtime Tradeoff:**
```bash
python3 ../generic_lut_activation/plots/plot_error_vs_runtime.py --arch-prefix blackhole
```
- Scatter plots showing accuracy/speed tradeoffs
- Output: `{arch}_plots_error/`

**Runtime Comparison:**
```bash
python3 ../generic_lut_activation/plots/plot_runtime_comparison.py --arch-prefix blackhole
```
- Bar charts comparing execution times
- Output: `{arch}_plots_runtime/`

**Activation Function Visualization:**
```bash
python3 ../generic_lut_activation/plots/plot_activation_comparison.py
```
- Plots actual activation function curves
- Shows hardware output vs reference
- Requires: `data/hardware_outputs/*.csv`

## Data File Format

Your CSV files should have these columns:
- `activation` - Activation function name
- `depth` - Number of piecewise segments
- `degree` - Polynomial degree (1=linear, 2=quadratic, 3=cubic, 6=hexic, 8=octic)
- `segmentation` - Segmentation strategy (uniform, chebyshev, curvature)
- `mae` - Mean Absolute Error
- `rmse` - Root Mean Square Error
- `max_error` - Maximum error
- `runtime_min_ms` - Minimum runtime in milliseconds
- `runtime_mean_ms` - Mean runtime
- `status` - Pass/fail status

## Understanding the Plots

### Pareto Frontier Plot
- **X-axis:** Polynomial degree (Linear → Quadratic → Cubic → Hexic)
- **Left Y-axis (blue):** MAE (log scale) - Lower is better
- **Right Y-axis (red):** Runtime (ms) - Lower is faster
- **Blue shaded region:** Shows min-max error range across all activations
- **Blue scatter points:** Individual activation results

**How to read:**
- Moving right increases accuracy but uses more coefficients
- Look for "elbow points" where accuracy gains diminish
- Trade off accuracy vs memory based on your needs

### Heatmap
- **Rows:** Polynomial degree
- **Columns:** Depth (number of segments)
- **Color:** MAE (green=good, red=bad)
- **Numbers:** Actual MAE values

**How to read:**
- Find the best (greenest) cell for your memory budget
- Memory usage = depth × coeffs_per_segment
  - Linear: depth × 2
  - Quadratic: depth × 3
  - Cubic: depth × 4
  - Hexic: depth × 7

## Tips

1. **Start with Pareto plots** to understand the accuracy/memory tradeoff landscape
2. **Use best.py** to get concrete recommendations for each activation
3. **Check heatmaps** (`plots/plot_depth_degree.py`) if you have a specific memory budget constraint
   - Look for the black box to find the cheapest best configuration
4. **Validate with ASCII plots** for quick sanity checks

## Troubleshooting

**"No result CSV files found"**
- Ensure your CSV is in `data/{arch}_embedded_all_results.csv`
- Or use split files: `{arch}_piecewise_{method}_results.csv`

**Missing plots**
- Check that you have matplotlib installed: `pip install matplotlib pandas numpy`
- Verify CSV has required columns

**Empty plots**
- Filter out failed runs: Only rows with `status='pass'` are plotted
- Check for NaN values in mae/runtime columns
