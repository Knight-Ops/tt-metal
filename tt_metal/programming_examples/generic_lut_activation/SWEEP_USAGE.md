# Sweep Scripts Usage Guide

This guide explains how to run sweeps across various activations, depths, polynomial degrees, and segmentation types for both LUT generation, hardware testing, and theoretical error analysis.

## Quick Reference

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `sweep_piecewise.sh` | Hardware testing sweep | `--degree`, `--depth`, `--activation`, `--segmentation` |
| `sweep_polynomial.sh` | Master orchestrator for all hardware sweeps | `--arch`, `--degree`, `--depth`, `--activation` |
| `generate_piecewise_remez_luts.py` | Generate LUTs | `--degree`, `--depth`, `--activation`, `--adaptive` |

## Common Options Across Scripts

### Universal Options
- `--activation <name>` - Target specific activation (e.g., `sigmoid`, `gelu`, `tanh`, `exp`, `digamma`)
- `--depth <depths>` - Comma-separated segment depths (e.g., `"4,8,16"` or `"16"`)
- `--degree <type>` - Polynomial degree: `linear`, `quadratic`, `cubic`, `hexic`, `octic`
- `--segmentation <type>` - Segmentation strategy: `uniform` or `adaptive`

### Precision Options (Error Analysis Only)
- `--simulate-bf16` - Simulate BF16 quantization to match hardware precision

## 1. LUT Generation

### Generate LUTs for specific configuration
```bash
# Generate cubic LUTs for sigmoid at depth 8
python3 generate_piecewise_remez_luts.py --degree cubic --activation sigmoid --depth 8

# Generate quadratic LUTs for all activations at depths 4 and 8
python3 generate_piecewise_remez_luts.py --degree quadratic --depth "4,8"

# Generate cubic LUTs with adaptive segmentation
python3 generate_piecewise_remez_luts.py --degree cubic --activation tanh --depth 16 --adaptive
```

### Generate all LUTs
```bash
# Generate all standard LUTs (uses generate_luts.sh)
./generate_luts.sh

# Generate specific degree for all activations
./generate_luts.sh --degree cubic
```

## 2. Hardware Testing Sweeps

### Run hardware tests for specific configuration
```bash
# Test cubic polynomials at depth 8 (both uniform and adaptive)
./sweep_piecewise.sh --degree cubic --depth 8

# Test only uniform segmentation for sigmoid
./sweep_piecewise.sh --degree cubic --activation sigmoid --segmentation uniform

# Test adaptive segmentation only for tanh at depths 4 and 8
./sweep_piecewise.sh --degree cubic --activation tanh --depth "4,8" --segmentation adaptive

# Test hexic polynomials for digamma (single activation, all depths)
./sweep_piecewise.sh --degree hexic --activation digamma
```

### Run complete hardware sweeps
```bash
# Run all hardware tests on current architecture
./sweep_polynomial.sh

# Run only cubic tests
./sweep_polynomial.sh --degree cubic

# Run only uniform segmentation tests
./sweep_polynomial.sh --segmentation uniform

# Run specific activation across all degrees
./sweep_polynomial.sh --activation sigmoid
```

## 3. Theoretical Error Analysis


## 4. Specialized Analysis Tools

### Analyze BF16 quantization overhead
```bash
# Show how much error BF16 quantization adds
python3 analyze_bf16_overhead.py sigmoid 8
python3 analyze_bf16_overhead.py exp 16 piecewise_cubic_remez
python3 analyze_bf16_overhead.py tanh 16
```

### Compare FP32 vs BF16 side-by-side
```bash
python3 compare_bf16_fp32_error.py sigmoid 8
python3 compare_bf16_fp32_error.py gelu 16 piecewise_cubic_remez
```

## Complete Workflow Examples

### Example 1: Full pipeline for sigmoid with BF16 analysis
```bash
# 1. Generate LUTs
python3 generate_piecewise_remez_luts.py --degree cubic --activation sigmoid --depth "4,8,16"

# 2. Run hardware tests
./sweep_piecewise.sh --degree cubic --activation sigmoid

# 3. Analyze quantization overhead
python3 analyze_bf16_overhead.py sigmoid 8
```

### Example 2: Compare uniform vs adaptive segmentation
```bash
# 1. Generate adaptive breakpoints
python3 generate_adaptive_segments.py --depth 16

# 2. Generate both uniform and adaptive LUTs
python3 generate_piecewise_remez_luts.py --degree cubic --activation tanh --depth 16
python3 generate_piecewise_remez_luts.py --degree cubic --activation tanh --depth 16 --adaptive

# 3. Run hardware tests for both
./sweep_piecewise.sh --degree cubic --activation tanh --depth 16

# 4. Compare results
```

### Example 3: Quick validation of new activation
```bash
# Test single configuration
python3 generate_piecewise_remez_luts.py --degree cubic --activation digamma --depth 8
./sweep_piecewise.sh --degree cubic --activation digamma --depth 8
python3 analyze_bf16_overhead.py digamma 8
```

## Output Files

### Hardware Testing Outputs
- `wormhole_piecewise_cubic_remez_results.csv` - Timing and accuracy results
- `wormhole_hardware_outputs/piecewise_cubic_sigmoid_8.csv` - Raw hardware outputs

### Theoretical Error Outputs
- `plots_error_theoretical/` - Error analysis plots
- `sweep_summary_bf16.csv` - BF16-aware error metrics across all configs
- `sweep_summary_fp32.csv` - FP32 ideal error metrics

### Comparison Outputs
- `adaptive_comparison_summary.csv` - Uniform vs adaptive comparison metrics

## Important Notes

### BF16 Precision Considerations

**Always use `--simulate-bf16` for theoretical error when comparing to hardware!**

The hardware path is:
```
FP32 input → BF16 quantize → FP32 LUT eval → BF16 quantize → FP32 output
```

Without `--simulate-bf16`, theoretical error only measures polynomial approximation error in FP32, which will be **lower** than actual hardware error.

With `--simulate-bf16`, theoretical error includes:
- Input quantization error (FP32 → BF16)
- Polynomial approximation error
- Output quantization error (FP32 → BF16)

This typically adds **1-3% to MAE** and **36-60% to max error**.

### Depth Limitations
- **Hexic/Octic**: Avoid depths 16/32 (27+ minute hangs per activation during generation)
- **Cubic/Quadratic**: All depths work fine (4, 8, 16, 32)
- **Linear**: Fast at all depths

### Segmentation Types
- **Uniform**: Equal-width segments across input domain (default)
- **Adaptive**: Error-guided segment placement (requires generating adaptive segments first)
- Adaptive typically improves MAE by 1.5-3× for same segment count

## Filter Combinations

You can combine multiple filters:

```bash
# Very specific test
./sweep_piecewise.sh --degree cubic --activation sigmoid --depth 8 --segmentation uniform

# Test multiple depths for one activation
./sweep_piecewise.sh --activation tanh --depth "4,8,16" --degree cubic
```

## Performance Tuning

### For faster sweeps:
- Use `--samples 10000` instead of default 100000 for theoretical error
- Test single activation first: `--activation sigmoid`
- Test single depth first: `--depth 8`
- Use only one segmentation mode: `--segmentation uniform`

### For comprehensive analysis:
- Use `--samples 100000` or higher
- Include both uniform and adaptive segmentation
- Test all depths: 4, 8, 16, 32
- Always use `--simulate-bf16` for hardware comparison

## Troubleshooting

### "LUTs not found"
Run: `./generate_luts.sh --degree <type>` first

### "Binary not found"
Run: `./build_metal.sh --build-programming-examples` from repo root

### "ml_dtypes not installed" (for BF16 simulation)
Run: `python3 -m pip install ml_dtypes`

### Adaptive LUTs not found
Generate adaptive breakpoints first: `python3 generate_adaptive_segments.py --depth <N>`
Then regenerate LUTs with: `python3 generate_piecewise_remez_luts.py --degree <type> --adaptive --depth <N>`
