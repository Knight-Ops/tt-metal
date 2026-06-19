# Quick Start: Unified Remez LUT Generator

## One-Liners

```bash
# Most common use case (cubic, all activations, standard depths)
python3 generate_piecewise_remez_luts.py --degree cubic

# Specific activation only
python3 generate_piecewise_remez_luts.py --degree cubic --activation sigmoid

# Specific depth only
python3 generate_piecewise_remez_luts.py --degree cubic --depth 8

# High accuracy (hexic)
python3 generate_piecewise_remez_luts.py --degree hexic --depth 8

# Adaptive segmentation
python3 generate_piecewise_remez_luts.py --degree cubic --adaptive
```

## Quick Comparison

### Command Equivalence

| Old (4 scripts) | New (1 unified) |
|-----------------|-----------------|
| `generate_piecewise_quadratic_remez_luts.py` | `generate_piecewise_remez_luts.py --degree quadratic` |
| `generate_piecewise_cubic_remez_luts.py` | `generate_piecewise_remez_luts.py --degree cubic` |
| `generate_piecewise_hexic_remez_luts.py` | `generate_piecewise_remez_luts.py --degree hexic` |
| `generate_piecewise_octic_remez_luts.py` | `generate_piecewise_remez_luts.py --degree octic` |

### Accuracy vs Compute Trade-off

```
Degree      Accuracy   Compute    Memory     Recommendation
-----------------------------------------------------------------
quadratic   Baseline   Fastest    Smallest   Simple functions only
cubic       3-5× ↑     Fast       Small      ★ RECOMMENDED DEFAULT
hexic       100× ↑     Moderate   Medium     Complex functions (gelu, erf)
octic       500× ↑     Slow       Large      Research/comparison only
```

## Cheat Sheet

### Arguments

```bash
--degree <quadratic|cubic|hexic|octic>   # REQUIRED: polynomial degree
--activation <name>                      # Optional: filter by activation
--depth <4,8,16,32>                      # Optional: filter by depth
--adaptive                               # Optional: use adaptive segmentation
```

### Examples by Use Case

**Production (balanced accuracy/compute):**
```bash
python3 generate_piecewise_remez_luts.py --degree cubic
```

**High accuracy (ML training):**
```bash
python3 generate_piecewise_remez_luts.py --degree hexic
```

**Fast evaluation (embedded):**
```bash
python3 generate_piecewise_remez_luts.py --degree quadratic
```

**Research (maximum accuracy):**
```bash
python3 generate_piecewise_remez_luts.py --degree octic --depth 4,8
```

**Adaptive segmentation (complex functions):**
```bash
python3 generate_piecewise_remez_luts.py --degree cubic --adaptive
```

**Single activation (iterative development):**
```bash
python3 generate_piecewise_remez_luts.py --degree cubic --activation gelu --depth 8
```

## Output

### File Location
```
luts/piecewise_<degree>_remez_<activation>_<depth>.lut
```

### Example Outputs
```
luts/piecewise_cubic_remez_sigmoid_8.lut
luts/piecewise_hexic_remez_gelu_16.lut
luts/piecewise_quadratic_remez_tanh_4_adaptive.lut
```

## Expected Accuracy (sigmoid, depth=8)

```
Degree      MAE        RMSE       MaxErr
---------------------------------------------
quadratic   0.00518    0.00791    0.01614
cubic       0.00126    0.00191    0.00387  ← 4× better
hexic       0.00003    0.00004    0.00008  ← 200× better
octic       0.00001    0.00001    0.00002  ← 800× better
```

## Troubleshooting

**Script not found:**
```bash
cd /path/to/tt-metal/tt_metal/programming_examples/generic_lut_activation
```

**Import error (mpmath/remez_poly):**
```bash
pip3 install --user mpmath
cd /tmp && git clone https://github.com/DKenefake/OptimalPoly.git
```

**Remez convergence failure:**
- Script automatically falls back to least-squares
- Warning messages are informational only
- Output files are still valid

**Adaptive breakpoints not found:**
- Generate adaptive segments first: `python3 generate_adaptive_segments.py`
- Or use uniform segmentation (omit `--adaptive`)

## Migration from Original Scripts

**Replace this:**
```bash
python3 generate_piecewise_cubic_remez_luts.py --activation sigmoid
```

**With this:**
```bash
python3 generate_piecewise_remez_luts.py --degree cubic --activation sigmoid
```

**Benefits:**
- ✓ Same output files (binary identical)
- ✓ Same accuracy metrics
- ✓ Single script to maintain
- ✓ Consistent interface

## Help

```bash
python3 generate_piecewise_remez_luts.py --help
```

## See Also

- Full documentation: `UNIFIED_REMEZ_GENERATOR.md`
- Binary format spec: `LUT_FORMAT_CONSISTENCY_ANALYSIS.md`
- Validation test: `test_unified_remez.sh`
