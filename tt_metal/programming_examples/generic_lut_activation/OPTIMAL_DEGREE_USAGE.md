# Optimal Degree Search - Quick Usage Guide

## Using generate_luts.sh (Recommended)

The `--optimal-degree-search` flag is now available in `generate_luts.sh`:

```bash
# Generate hexic LUTs for all activations with optimal degree search
./generate_luts.sh --degree hexic --optimal-degree-search

# Generate cubic LUTs for sigmoid only with optimal degree search
./generate_luts.sh --activation sigmoid --degree cubic --optimal-degree-search

# Generate hexic LUTs for specific depth with optimal degree search
./generate_luts.sh --activation sigmoid --degree hexic --depth 8 --optimal-degree-search

# BF16 precision only with optimal degree search
./generate_luts.sh --activation sigmoid --degree hexic --precision bf16 --optimal-degree-search

# Adaptive segmentation with optimal degree search
./generate_luts.sh --activation sigmoid --degree hexic --segmentation adaptive --optimal-degree-search
```

## What It Does

When `--optimal-degree-search` is enabled:

1. **For each segment**: Tries all degrees from 1 to max_degree (e.g., 1-6 for hexic)
2. **Evaluates MAE**: Tests each degree in target precision (BF16 or FP32)
3. **Picks best**: Selects the degree with **lowest MAE** (not highest!)
4. **Sets zeros**: Higher-degree coefficients are set to zero if lower degree is optimal

## Example Output

```bash
./generate_luts.sh --activation sigmoid --degree hexic --depth 8 --precision bf16 --optimal-degree-search
```

**Output:**
```
Optimization: Optimal degree search ENABLED (tries degrees 1-N, picks best MAE)
...
sigmoid (uniform) bf16 → MAE: 0.00073  RMSE: 0.00108  MaxErr: 0.00446
                         Degrees: {1: 1, 2: 4, 3: 1, 6: 2} (mean=3.0)
```

**Interpretation:**
- **1 segment** uses degree 1
- **4 segments** use degree 2
- **1 segment** uses degree 3
- **2 segments** use degree 6
- **Mean degree**: 3.0 (instead of always 6)
- **Coefficient savings**: 50% fewer coefficients on average

## Direct Python Usage

You can also call the Python script directly:

```bash
# Using optimal degree search
python3 luts/generate_piecewise_remez_luts.py \
    --degree hexic \
    --activation sigmoid \
    --depth 8 \
    --precision bf16 \
    --optimal-degree-search

# Without optimal degree search (always uses max degree)
python3 luts/generate_piecewise_remez_luts.py \
    --degree hexic \
    --activation sigmoid \
    --depth 8 \
    --precision bf16
```

## Comparison Tool

Use the comparison script to see the difference:

```bash
python3 compare_fixed_vs_optimal_degree.py \
    --activation sigmoid \
    --max-degree 6 \
    --segments 8 \
    --precision bf16
```

**Sample Output:**
```
MAE Improvement: +260.3% (BETTER)
Coefficient Savings: 50.0% fewer coefficients on average

Per-Segment Breakdown:
Seg                Range  Fixed Deg  Optimal Deg     Improvement
  0     [-10.00,  -7.50]          6            2             6→2
  1     [ -7.50,  -5.00]          6            2             6→2
  2     [ -5.00,  -2.50]          6            2             6→2
  3     [ -2.50,   0.00]          6            6         optimal
  4     [  0.00,   2.50]          6            6         optimal
  5     [  2.50,   5.00]          6            3             6→3
  6     [  5.00,   7.50]          6            2             6→2
  7     [  7.50,  10.00]          6            1             6→1
```

## When to Use

**Use Optimal Degree Search When:**
- Working with BF16 precision (quantization makes lower degrees more robust)
- Memory/compute is constrained (save up to 50-60% coefficients)
- Exploring design space (understand which segments need high accuracy)
- Maximizing accuracy (lower degrees often beat higher degrees!)

**Use Fixed Degree When:**
- Need predictable compute cost across all segments
- Already know optimal degree from prior analysis
- Simplicity is more important than efficiency

## Technical Details

See [`OPTIMAL_DEGREE_SEARCH.md`](OPTIMAL_DEGREE_SEARCH.md) for:
- Why higher degrees can be worse (260-80,000% worse in some cases!)
- Algorithm details
- Per-segment MAE evaluation methodology
- BF16 vs FP32 precision handling
