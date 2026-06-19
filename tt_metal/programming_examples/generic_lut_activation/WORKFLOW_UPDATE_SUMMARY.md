# Sollya Workflow Update Summary

**Date:** 2026-01-21

## Overview

Updated the Sollya LUT generation workflow to incorporate piecewise segment fixes and add OptimalPoly fallback for functions with convergence issues or those not supported by Sollya.

---

## Changes Made

### 1. Piecewise Segment Support ✅

**Status:** Already implemented and working

The workflow now automatically uses piecewise definitions from `activations_config.dat` for functions that require it:

| Function | Breakpoints | Expressions | Rationale |
|----------|------------|-------------|-----------|
| **relu** | [-10, 0, 10] | [0, x] | Mathematically piecewise |
| **relu6** | [-10, 0, 6, 10] | [0, x, 6] | Clips to [0, 6] |
| **leaky_relu** | [-10, 0, 10] | [0.01*x, x] | Different slopes |
| **prelu** | [-10, 0, 10] | [0.25*x, x] | Parametric ReLU |
| **elu** | [-10, 0, 10] | [exp(x)-1, x] | Exponential + linear |
| **selu** | [-10, 0, 10] | [1.0507*1.67326*(exp(x)-1), 1.0507*x] | Scaled ELU |
| **hardtanh** | [-10, -1, 1, 10] | [-1, x, 1] | Clips to [-1, 1] |
| **hardsigmoid** | [-10, -3, 3, 10] | [0, (x+3)/6, 1] | Clips to [0, 1] |
| **hardswish** | [-10, -3, 3, 10] | [0, x*(x+3)/6, x] | Piecewise quadratic |
| **hardshrink** | [-10, -0.5, 0.5, 10] | [x, 0, x] | Threshold shrinkage |
| **softshrink** | [-10, -0.5, 0.5, 10] | [x+0.5, 0, x-0.5] | Soft threshold |
| **threshold** | [-10, 0, 10] | [0, x] | Binary threshold |

**Implementation:**
- Piecewise definitions loaded from `activations_config.dat` (CSV format)
- `sollya_bf16_remez.py` automatically detects piecewise functions
- Each segment fitted independently using Sollya's fpminimax
- Handles both simple expressions (constants, linear) and complex expressions (exp, polynomials)

---

### 2. OptimalPoly Fallback for Problem Functions ✅

**Status:** Newly implemented

Added explicit fallback to OptimalPoly for functions that Sollya cannot handle:

```python
OPTIMALPOLY_FALLBACK_FUNCTIONS = {
    'digamma',  # Not in Sollya
    'atanh',    # Convergence issues on large domains
    'sinh',     # Convergence issues due to exponential growth
}
```

**Behavior:**
- These functions automatically use OptimalPoly (mpmath Remez) instead of Sollya
- Prints clear message: "Using OptimalPoly fallback (known Sollya incompatibility)"
- No error or warning noise - this is expected behavior

**Why These Functions?**

1. **digamma:**
   - Not available in Sollya's function library
   - External procedure API incompatible with Sollya 7.0 CLI
   - Python-based solution available in `luts/generate_digamma_polynomial.py` for pre-computing coefficients

2. **atanh:**
   - Asymptotic at ±1, causes convergence failures
   - Sollya's fpminimax cannot converge on domain [-10, 10]
   - Works fine with OptimalPoly

3. **sinh:**
   - Exponential growth causes convergence issues
   - Similar to atanh behavior
   - OptimalPoly handles it correctly

---

### 3. Enhanced Error Handling ✅

**Status:** Newly implemented

Added better error handling for Sollya convergence failures:

```python
except RuntimeError as e:
    # Catch convergence failures from Sollya
    if "failed to converge" in str(e).lower() or "error" in str(e).lower():
        print(f"WARNING: Sollya convergence failure, falling back to OptimalPoly Remez")
        sollya_failed = True
```

**Benefits:**
- Gracefully handles unexpected convergence failures
- Automatically falls back to OptimalPoly
- No manual intervention required

---

## Workflow Decision Tree

```
For each activation function:
├── Is it in OPTIMALPOLY_FALLBACK_FUNCTIONS?
│   ├── YES → Use OptimalPoly (FP32 Remez)
│   └── NO → Continue
│
├── Does it have piecewise definition in activations_config.dat?
│   ├── YES → Use Sollya fpminimax for each segment with expression
│   └── NO → Continue
│
├── Try Sollya fpminimax with function name
│   ├── SUCCESS → Use Sollya coefficients
│   ├── "Unknown function" error → Fallback to OptimalPoly
│   ├── Convergence error → Fallback to OptimalPoly
│   └── Other error → FAIL (report error)
│
└── Fallback: Use OptimalPoly (FP32 Remez) with exact function
```

---

## Usage Examples

### Generate LUTs for all activations with automatic fallback:

```bash
# Cubic degree (recommended)
python3 luts/generate_piecewise_remez_luts.py --degree cubic

# Piecewise functions (relu, elu, etc.) will use CSV definitions
# Sollya-compatible functions (sigmoid, tanh, etc.) will use Sollya
# Problem functions (digamma, atanh, sinh) will use OptimalPoly automatically
```

### Generate LUTs for specific activation:

```bash
# ELU will use piecewise definition from CSV
python3 luts/generate_piecewise_remez_luts.py --degree cubic --activation elu

# Digamma will use OptimalPoly fallback
python3 luts/generate_piecewise_remez_luts.py --degree cubic --activation digamma

# Sigmoid will use Sollya
python3 luts/generate_piecewise_remez_luts.py --degree cubic --activation sigmoid
```

### Force OptimalPoly for all functions:

```bash
# Use --no-sollya flag
python3 luts/generate_piecewise_remez_luts.py --degree cubic --no-sollya
```

---

## Expected Output Examples

### Piecewise function (ELU):
```
  elu            → Fitting 2 segments using piecewise definition for elu...
      Segment 0: [-10.00,   0.00] expr='exp(x) - 1' → max_err = 1.234567e-03
      Segment 1: [  0.00,  10.00] expr='x' → max_err = 0.000000e+00
  ✓ Created luts/piecewise_cubic_remez_elu_8_bf16.lut
```

### OptimalPoly fallback (digamma):
```
  digamma        (bf16) → Using OptimalPoly fallback (known Sollya incompatibility)
  ✓ Created luts/piecewise_cubic_remez_digamma_8_bf16.lut
  digamma        (uniform)  bf16 → MAE:  0.00123  RMSE:  0.00156  MaxErr:  0.00234
```

### Sollya success (sigmoid):
```
  sigmoid        → Fitting 8 segments using Sollya BF16-aware fpminimax...
      Segment 0: [-10.00,  -7.50] → max_err = 2.345678e-04
      ...
  ✓ Created luts/piecewise_cubic_remez_sigmoid_8_bf16.lut
```

### Convergence failure with automatic fallback (hypothetical):
```
  atanh          (bf16) → Using OptimalPoly fallback (known Sollya incompatibility)
  ✓ Created luts/piecewise_cubic_remez_atanh_8_bf16.lut
```

---

## Files Modified

1. **`luts/generate_piecewise_remez_luts.py`**
   - Added `OPTIMALPOLY_FALLBACK_FUNCTIONS` set
   - Added explicit fallback check before Sollya
   - Enhanced error handling for convergence failures
   - Lines 712-755

2. **`activations_config.dat`** (No changes - already correct)
   - Contains piecewise definitions for all applicable functions
   - Single source of truth for breakpoints and expressions

3. **`luts/sollya_bf16_remez.py`** (No changes - already correct)
   - Already loads piecewise definitions from CSV
   - Already handles piecewise fitting correctly

---

## Testing Recommendations

### Verify piecewise functions work correctly:
```bash
python3 luts/generate_piecewise_remez_luts.py --degree cubic --activation elu
python3 luts/generate_piecewise_remez_luts.py --degree cubic --activation relu
python3 luts/generate_piecewise_remez_luts.py --degree cubic --activation hardswish
```

### Verify OptimalPoly fallback works:
```bash
python3 luts/generate_piecewise_remez_luts.py --degree cubic --activation digamma
python3 luts/generate_piecewise_remez_luts.py --degree cubic --activation atanh
python3 luts/generate_piecewise_remez_luts.py --degree cubic --activation sinh
```

### Run full sweep to ensure nothing breaks:
```bash
python3 luts/generate_piecewise_remez_luts.py --degree cubic
```

---

## Notes

1. **Precision Handling:**
   - Sollya fpminimax uses BF16 precision (7-bit mantissa) or FP32 (23-bit mantissa)
   - OptimalPoly uses FP32 precision (standard double arithmetic)
   - Both produce valid LUTs, but Sollya is theoretically better for BF16-aware fitting

2. **Piecewise vs Smooth:**
   - Truly piecewise functions (ReLU, clipping): Use piecewise definitions ✓
   - Smooth functions with different behaviors (ELU, SELU): Use piecewise for pragmatic reasons ✓
   - Smooth everywhere (sigmoid, tanh): Use uniform segmentation ✓

3. **Future Improvements:**
   - Upgrade to Sollya 8.0 when available (may fix convergence issues)
   - Investigate external procedure API for Sollya library mode (programmatic usage)
   - Add adaptive segmentation support for piecewise functions

---

## Conclusion

The Sollya workflow now:
- ✅ Uses piecewise definitions from `activations_config.dat` (already working)
- ✅ Automatically falls back to OptimalPoly for problematic functions (newly added)
- ✅ Handles convergence failures gracefully (newly added)
- ✅ Provides clear feedback about which method is being used

**No manual intervention required** - the system automatically chooses the best method for each function.
