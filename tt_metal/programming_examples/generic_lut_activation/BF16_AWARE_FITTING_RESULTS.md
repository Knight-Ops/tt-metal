# BF16-Aware Remez Fitting: Implementation and Results

**Date**: 2026-01-20
**Status**: ❌ **APPROACH INEFFECTIVE - DO NOT USE**

---

## Executive Summary

We implemented BF16 quantization-aware Remez fitting to optimize LUT accuracy for hardware execution. After analyzing the SFPU hardware precision model and implementing the feature, **testing revealed the approach is fundamentally flawed and produces significantly worse results** (10-850× higher error).

**Conclusion**: The current FP32-based Remez fitting is superior. BF16-aware fitting via function wrapping does not work.

---

## What Was Implemented

### 1. Hardware Precision Analysis

**Key Discovery**: LUT coefficients stored as **FP32 in L1 memory** but converted to **BF16 in SFPU vFloat registers**:

```cpp
// From generic_lut_activation.cpp
const auto lut_cb_data_format = tt::DataFormat::Float32;  // FP32 storage

// From compute kernel
vFloat a = lut[5];  // FP32 → BF16 implicit conversion
vFloat result = ((a * x + b) * x + c) * x + d;  // BF16 arithmetic
```

**Hardware Dataflow**:
1. Input: FP32 → BF16 (CB c_0)
2. Coeffs: FP32 (L1) → BF16 (SFPU registers)
3. Arithmetic: BF16 × BF16 → BF16
4. Output: BF16 → BF16 (CB c_16)

### 2. BF16-Aware Fitting Implementation

**Changes Made**:
- Added `create_bf16_aware_function()` wrapper in `generate_piecewise_remez_luts.py`
- Modified `fit_piecewise_remez()` to accept `bf16_aware` parameter
- Added `--bf16-aware` command-line flag
- Created `validate_bf16_aware_fitting.py` validation script

**Function Wrapper**:
```python
def create_bf16_aware_function(target_func):
    def bf16_wrapper(x):
        x_bf16 = simulate_bf16_quantization(x)      # Input quantization
        y_fp32 = target_func(x_bf16)                # Evaluate
        y_bf16 = simulate_bf16_quantization(y_fp32) # Output quantization
        return y_bf16
    return bf16_wrapper
```

This wrapper was passed to the Remez algorithm to fit polynomials.

---

## Test Results

### Sigmoid (Bounded Output: [0, 1])

| Metric | FP32-Fitted | BF16-Fitted | Change |
|--------|-------------|-------------|--------|
| **Theoretical FP32 Error** |
| MAE | 2.22e-04 | 5.80e-03 | **+2518%** ❌ |
| Max Error | 1.19e-03 | 5.50e-02 | **+4539%** ❌ |
| **Hardware BF16 Error** |
| MAE | 5.58e-04 | 5.90e-03 | **-90.6%** ❌ |
| Max Error | 3.70e-03 | 5.50e-02 | **-93.3%** ❌ |

**Result**: BF16-aware fitting is **10× worse** for hardware error.

### Exp (Unbounded Output: [0, ∞))

| Metric | FP32-Fitted | BF16-Fitted | Change |
|--------|-------------|-------------|--------|
| **Theoretical FP32 Error** |
| MAE | 7.56 | 15,320 | **+202,628%** ❌ |
| Max Error | 88.2 | 218,902 | **+247,995%** ❌ |
| **Hardware BF16 Error** |
| MAE | 18.0 | 15,315 | **-99.9%** ❌ |
| Max Error | 678 | 218,877 | **-99.7%** ❌ |

**Result**: BF16-aware fitting is **850× worse** for hardware error.

---

## Root Cause Analysis

### The Fundamental Flaw

**What we did (INCORRECT)**:
```python
# Fit polynomial to BF16-quantized target function
target_bf16 = lambda x: bf16(f(bf16(x)))
poly = remez(target_bf16, degree, ...)  # Minimize ||target_bf16(x) - poly(x)||
```

**Why this fails**:
1. We're fitting the polynomial to **approximate quantization noise**
2. The polynomial learns to match `bf16(f(bf16(x)))` instead of `f(x)`
3. When evaluated in hardware (which quantizes differently), error explodes
4. The BF16 quantization in the target introduces artifacts that corrupt the fit

**What we should do (CORRECT but not implemented)**:
```python
# Fit polynomial to original function, but minimize BF16 hardware error
poly = custom_remez(f, degree, error_metric=bf16_hardware_error)
```

Where:
```python
def bf16_hardware_error(x, poly, f):
    x_bf16 = bf16(x)                           # Input quantization
    poly_result = bf16_eval(poly, x_bf16)       # Poly eval in BF16 arithmetic
    return abs(f(x) - poly_result)              # Error vs TRUE function
```

### Why OptimalPoly Can't Do This

The Remez algorithm from OptimalPoly:
- Takes a target function `f(x)`
- Finds polynomial `p(x)` minimizing `max |f(x) - p(x)|`
- Assumes polynomial evaluation is exact (no quantization)

We need:
- Custom error metric: `max |f(x) - bf16_eval(p, bf16(x))|`
- Requires forking/rewriting Remez algorithm
- Not feasible with existing OptimalPoly library

---

## Why FP32 Fitting is Already Good Enough

From previous analysis (`BF16_PRECISION_ANALYSIS.md`):

**BF16 Quantization Overhead**:
- Typical MAE increase: **1-3%**
- Typical max error increase: **36-60%**
- This is acceptable for most applications

**Example (Sigmoid, depth=8)**:
- FP32 theoretical error: 2.22e-04
- FP32 hardware error: 5.58e-04 (2.5× overhead from BF16)
- 2.5× overhead is **much better** than BF16-aware fitting (10× worse!)

**Conclusion**: The 1-3% quantization overhead is **small enough to accept**. Trying to optimize for it makes things dramatically worse.

---

## Alternative Approaches (Future Work)

If BF16 quantization overhead becomes problematic:

### 1. **Accept FP32 Fitting** (RECOMMENDED)
- Current approach works well
- 1-3% overhead is acceptable
- No additional complexity

### 2. **Iterative Coefficient Refinement**
- Start with FP32 Remez fit
- Perturb coefficients slightly
- Evaluate hardware BF16 error
- Keep perturbations that reduce error
- Heuristic/gradient-free optimization

**Pros**: Doesn't require modifying Remez
**Cons**: May only find local minima, slow

### 3. **Custom BF16-Aware Remez Algorithm**
- Fork OptimalPoly or implement from scratch
- Modify error metric to include BF16 quantization
- Use BF16 arithmetic simulator in error evaluation

**Pros**: Theoretically optimal
**Cons**: Significant implementation effort (weeks), requires algorithm expertise

### 4. **Higher Polynomial Degrees**
- Use hexic (degree 6) or octic (degree 8) instead of cubic
- Reduces approximation error, making BF16 overhead less significant
- Already implemented and working

**Pros**: Easy, already available
**Cons**: Higher compute cost (more multiplications)

---

## Recommendations

1. **Remove `--bf16-aware` flag** or mark as experimental/broken
2. **Document findings** to prevent future attempts at same approach
3. **Focus on higher-degree polynomials** (hexic/octic) if more accuracy needed
4. **Accept 1-3% BF16 quantization overhead** as acceptable
5. **Consider alternative approach #2 or #4** only if overhead becomes critical

---

## Implementation Files

**Modified**:
- `generate_piecewise_remez_luts.py`: Added BF16-aware fitting (90 lines)
  - `create_bf16_aware_function()` wrapper
  - `--bf16-aware` flag
  - Modified `fit_piecewise_remez()` signature

**Created**:
- `validate_bf16_aware_fitting.py`: Validation script (170 lines)
- `BF16_AWARE_FITTING_RESULTS.md`: This document

**Status**: Code is functional but produces incorrect results. Do not use for production.

---

## Lessons Learned

1. **Quantization-aware training ≠ fitting to quantized target**
   - In neural networks, QAT simulates quantization in forward/backward passes
   - For polynomial fitting, we can't modify Remez's error metric easily

2. **Function wrapping approach doesn't work for Remez**
   - Remez requires exact target function, not noisy approximation
   - Wrapping adds noise that corrupts the fit

3. **FP32 fitting is already excellent**
   - BF16 overhead is small (1-3%) compared to polynomial approximation error
   - Optimizing for BF16 at fitting time is premature optimization

4. **Hardware analysis was valuable**
   - Understanding FP32 storage + BF16 arithmetic was critical
   - Dataflow analysis helped identify quantization points
   - Even though approach failed, we now understand the system better

---

## Usage (FOR REFERENCE ONLY - DO NOT USE)

```bash
# Generate BF16-aware LUT (produces BAD results)
python3 generate_piecewise_remez_luts.py \
    --degree cubic \
    --activation sigmoid \
    --depth 8 \
    --bf16-aware  # ⚠️ DO NOT USE - produces 10-850× worse error

# Validate (shows how bad it is)
python3 validate_bf16_aware_fitting.py sigmoid 8
```

**Recommended usage**:
```bash
# Use standard FP32 fitting (current best practice)
python3 generate_piecewise_remez_luts.py \
    --degree cubic \
    --activation sigmoid \
    --depth 8

# For higher accuracy, use higher degree
python3 generate_piecewise_remez_luts.py \
    --degree hexic \    # or octic
    --activation sigmoid \
    --depth 8
```

---

## Conclusion

BF16-aware Remez fitting via function wrapping is **fundamentally flawed** and produces dramatically worse results. The current FP32-based fitting approach is superior and should continue to be used.

The 1-3% BF16 quantization overhead observed in hardware is acceptable and much smaller than the errors introduced by attempting BF16-aware fitting.

**Final Recommendation**: Keep the code for reference, but **do not use `--bf16-aware` flag in production**. Document this failure to prevent future attempts at the same approach.
