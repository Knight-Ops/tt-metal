# Hardware Error Analysis Tool

## Overview

This tool analyzes the accuracy of hardware implementations (SFPU native and LUT approximations) against PyTorch baseline, computing three types of errors:
- **Absolute Error**: `|hardware_output - pytorch_baseline|`
- **Relative Error**: `|hardware_output - pytorch_baseline| / |pytorch_baseline|`
- **ULP Error**: Error normalized by the spacing between representable floating-point values

## ULP Error Calculation

**ULP (Units in the Last Place)** measures error relative to floating-point precision:

```python
ulp_error = abs_error / spacing(reference)
```

Where `spacing(reference)` is the distance to the next representable floating-point number at that magnitude.

### Why This Matters

- **Accounts for varying precision**: FP numbers have more precision near zero, less at large magnitudes
- **Standardized metric**: 1 ULP means error equals one representable value spacing
- **Matches ttnn tests**: Same calculation as [tt-metal utility_functions.py](https://github.com/tenstorrent/tt-metal/blob/c7bf98755635a0fab9867cce6341c2d8bd154155/models/common/utility_functions.py#L576)

### Interpretation

- **0-1 ULP**: Perfect (within rounding error)
- **1-3 ULP**: Excellent - typical for well-implemented math functions
- **3-10 ULP**: Good - acceptable for most applications
- **10-50 ULP**: Fair - reasonable for ML approximations
- **>50 ULP**: Poor approximation quality

### BF16 vs FP32 ULP

BF16 has 7 mantissa bits (vs 23 for FP32), so:
- BF16 spacing = 2^16 × FP32 spacing (65,536x larger)
- Same absolute error → 65,536x lower ULP in BF16 space
- **Always measure ULP in the precision space of your data!**

## Usage

### Basic Usage
```bash
# Single activation with best approximation
python3 tools/plot_hardware_error_analysis.py --arch wormhole --activation gelu --precision bf16

# All activations with SFPU support
python3 tools/plot_hardware_error_analysis.py --arch wormhole --activation all --precision bf16

# Compare different approximations
python3 tools/plot_hardware_error_analysis.py --arch wormhole --activation gelu --precision bf16 \
    --approx-method quadratic --approx-depth 8 --approx-seg-type uniform
```

### Options

**Required:**
- `--arch {wormhole,blackhole}` - Hardware architecture
- `--activation {name}` - Activation function (or "all" for all with SFPU)

**Optional:**
- `--precision {bf16,fp32,both}` - Precision to analyze (default: bf16)
- `--approx-method {best,cubic,quadratic,etc}` - Approximation method (default: best)
- `--approx-depth N` - Approximation depth/segments (default: 16)
- `--approx-seg-type {adaptive,uniform,chebyshev}` - Segmentation (default: adaptive)
- `--output-dir PATH` - Custom output directory

### Examples

```bash
# Use best configuration from sweep results
python3 tools/plot_hardware_error_analysis.py --arch wormhole --activation gelu

# Compare cubic depth-8 vs depth-16
python3 tools/plot_hardware_error_analysis.py --arch wormhole --activation gelu \
    --approx-method cubic --approx-depth 8

# Analyze FP32 precision
python3 tools/plot_hardware_error_analysis.py --arch wormhole --activation gelu --precision fp32

# Generate both BF16 and FP32 plots
python3 tools/plot_hardware_error_analysis.py --arch wormhole --activation all --precision both
```

## Output

### Plot Layout
3×2 grid of subplots:
- **Rows**: Absolute Error, Relative Error, ULP Error
- **Columns**: SFPU Native (left), Best Approximation (right)

Each subplot shows:
- Scatter plot of errors vs input values
- Statistics box (max, mean, median)
- Log scale for absolute and ULP errors
- Clipped y-axis for relative error (99th percentile) to handle outliers near zero

### Directory Structure
```
plots/{arch}/hardware_error_analysis/
  ├── gelu_error_analysis.png
  ├── tanh_error_analysis.png
  └── ...
```

With `--precision both`:
```
plots/{arch}/hardware_error_analysis/
  ├── bf16/
  │   ├── gelu_error_analysis.png
  │   └── ...
  └── fp32/
      ├── gelu_error_analysis.png
      └── ...
```

## Data Requirements

The tool requires hardware output CSV files with precision tags:

```
data/hardware_outputs/
  ├── {arch}_native_sfpu_{activation}_{precision}.csv
  ├── {arch}_piecewise_{method}_{activation}_{depth}_{segmentation}_{precision}.csv
  └── ...
```

CSV format:
```csv
input,output
-10.0,0.00205994
-9.99,0.00208745
...
```

**Generate data with:**
```bash
# SFPU native
./sweep_native_sfpu.sh --arch wormhole --activation gelu --precision bf16

# Polynomial approximations
./sweep_polynomial.sh --arch wormhole --activation gelu --precision bf16
```

## Technical Details

### ULP Calculation for BF16

BF16 has:
- 1 sign bit
- 8 exponent bits (same as FP32)
- 7 mantissa bits (vs 23 in FP32)

The spacing at value `x` is:
```python
mantissa, exponent = np.frexp(abs(x))
spacing = 2^(exponent - 8)  # 8 = 7 mantissa bits + 1 for [0.5, 1.0) range
```

### Relative Error Clipping

Relative error can explode when the reference value is near zero. The tool:
1. Clips y-axis at 99th percentile for visualization
2. Shows full max/mean/median in statistics box
3. Adds "(Y-axis clipped)" note when applicable

### Consistent Y-Axis

Both SFPU and approximation plots share the same y-axis limits for direct comparison, computed as:
- **Absolute/ULP**: Full range with 2x padding on log scale
- **Relative**: 99th percentile with 20% padding on linear scale

## Comparison with Reference

This tool matches the accuracy measurement methodology from:
- [ttnn-eltwise-op-tester](https://github.com/nmauriceTT/ttnn-eltwise-op-tester/blob/main/measure_accuracy.py)
- [tt-metal utility_functions.py](https://github.com/tenstorrent/tt-metal/blob/c7bf98755635a0fab9867cce6341c2d8bd154155/models/common/utility_functions.py#L576)

Key differences:
- **Reference**: Exhaustively tests all possible BF16 input values (~65K values)
- **This tool**: Uses your actual test range (typically -10 to 10, ~262K samples)
- **Reference**: Tests ttnn ops end-to-end
- **This tool**: Compares raw hardware outputs vs PyTorch baseline

## Troubleshooting

**"File not found" errors:**
- Make sure you've run sweeps with the new format (includes precision tags)
- Delete old CSV files: `rm -f data/hardware_outputs/wormhole_*.csv`
- Regenerate with: `./sweep_native_sfpu.sh` and `./sweep_polynomial.sh`

**High ULP errors:**
- Check if you're measuring in the right precision space (bf16 vs fp32)
- BF16 data measured in FP32 ULP will show ~65K errors per 1 BF16 ULP
- Review input range - errors may be concentrated in specific regions

**Empty plots:**
- Check for extreme outliers in relative error
- Tool automatically clips at 99th percentile - check statistics box for actual values

## See Also
- `plot_error_vs_runtime.py` - Runtime vs accuracy tradeoff analysis
- `SWEEP_USAGE.md` - How to run hardware sweeps
- `README.md` - Project overview
