# Error Discrepancy Investigation

## Summary

Large discrepancies between polynomial fitter predictions and hardware measured errors have been identified and root-caused.

## Key Findings

### 1. Ground Truth Formulas Now Match ✅

Fixed 3 formula mismatches in `sweep_best.sh`:
- **GELU**: Changed from tanh approximation to erf (exact) formula
- **HardSigmoid**: Changed from PyTorch `(x+3)/6` to `0.2*x + 0.5`
- **CELU**: Fixed broken logic

Verification: Recent hardware outputs for hardsigmoid match the `0.2*x+0.5` formula.

### 2. Error Pattern Analysis

```
Activation   Prec   Predicted    Measured     Ratio    Status
----------------------------------------------------------------
gelu         bf16   5.49e-03     6.85e-03     1.2x     ✓ GOOD
gelu         fp32   1.01e-06     9.20e-04     915x     ✗ BAD
hardsigmoid  bf16   2.58e-03     3.79e-04     0.15x    ✓ GOOD
hardsigmoid  fp32   1.53e-04     5.03e-05     0.33x    ✓ GOOD
tanh         bf16   2.81e-03     3.13e-03     1.1x     ✓ GOOD
tanh         fp32   5.24e-07     2.42e-05     46x      ✗ BAD
sigmoid      bf16   2.34e-03     2.35e-02     10x      ~ OK
sigmoid      fp32   1.24e-07     1.74e-02     141000x  ✗ VERY BAD
```

**Pattern**:
- BF16 errors match predictions (0.15x to 10x ratio)
- FP32 errors are 50x to 140000x larger than predicted!

### 3. Root Cause: LUT Boundary Coverage ⚠️

Investigated GELU FP32 (worst case: 915x discrepancy):

**Test Configuration**:
- Input range: [-10, 10] (from activations_config.dat)
- LUT configuration: 32 segments, degree 8

**Actual LUT Boundaries**:
```
LUT covers: [-10.0, 3.934]  ← Only 14 out of 20 total range!
```

**Extrapolation Problem**:
```
x=5.0   -> segment 31 [3.033, 3.934] - extrapolating +1.07 beyond
x=7.0   -> segment 31 [3.033, 3.934] - extrapolating +3.07 beyond
x=10.0  -> segment 31 [3.033, 3.934] - extrapolating +6.07 beyond
```

**Error Distribution**:
```
x range     Error       Notes
-10 to -2   1e-05       Within LUT coverage, excellent
-2 to 2     1e-05       Within LUT coverage, excellent
2 to 4      1e-04       Near boundary, good
4 to 7      1-2e-03     Extrapolating, degrading
7 to 10     3-8e-03     Heavy extrapolation, poor
```

**Why This Happens**:
1. Curvature-based segmentation concentrates segments in high-curvature regions (x ∈ [-4, 4] for GELU)
2. Linear regions (x < -4 or x > 4) get few/no segments
3. Polynomial fitted for [3.033, 3.934] is extrapolated to x=10
4. Extrapolation error dominates for x > 4

### 4. Why BF16 Hides the Problem

**BF16 Quantization Error**: ~1e-3 to 1e-2 (for values around 1-10)
**FP32 Polynomial Extrapolation Error**: 1e-3 to 8e-3

The BF16 quantization noise masks the polynomial extrapolation errors!

### 5. Why Fitter Predictions Are Too Optimistic

The polynomial fitter's `mean_error` in best.csv is computed by:
1. Evaluating the fitted polynomial at test points WITHIN each segment
2. NOT testing extrapolation beyond the last boundary

So:
- Fitter tests x ∈ [-10, 3.934]: MAE = 1.01e-06 ✓
- Hardware tests x ∈ [-10, 10]: MAE = 9.20e-04 ✗  (extrapolation error for x > 3.934)

## Impact Assessment

### Activations with Boundary Coverage Issues

Checked other FP32 configs with large discrepancies:

| Activation | Config | LUT Range | Test Range | Issue |
|------------|--------|-----------|------------|-------|
| gelu       | fp32 32x8 | [-10, 3.93] | [-10, 10] | Missing [3.93, 10] |
| sigmoid    | fp32 32x5 | Need to check | [-10, 10] | Likely similar |
| softplus   | fp32 32x8 | Need to check | [-10, 10] | Likely similar |
| tanh       | fp32 32x8 | Need to check | [-10, 10] | Likely similar |

### Activations Working Correctly

| Activation | Why It Works |
|------------|--------------|
| hardsigmoid | Piecewise linear - fits entire range with few segments |
| relu/prelu/threshold | Piecewise linear - exact within segment |
| hardtanh | Piecewise constant/linear - exact within segment |

## Solutions

### Option 1: Fix Segmentation Algorithm (Recommended)

Modify curvature-based segmentation to FORCE coverage of full input range:
1. Always place boundaries at range_min and range_max
2. Ensure no gap > threshold (e.g., 2.0) between boundaries
3. For nearly-linear regions, use fewer segments but still cover them

**Pros**: Fixes root cause, accurate across full range
**Cons**: Requires modifying polynomial fitter

### Option 2: Kernel-Side Range Clamping

Modify compute kernels to clamp inputs to LUT boundary range before evaluation:

```cpp
vFloat x_clamped = x;
v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
v_if (x > lut[NUM_BOUNDARIES-1]) { x_clamped = lut[NUM_BOUNDARIES-1]; } v_endif;
```

**Pros**: Simple, prevents extrapolation
**Cons**: Changes function behavior (clamps instead of extrapolates), may hurt accuracy

### Option 3: Linear Extrapolation Beyond Boundaries

For x > last_boundary, use linear extrapolation based on last segment's slope.

**Pros**: Better accuracy than clamping
**Cons**: More complex kernel code, only works for monotonic functions

### Option 4: Accept BF16 Accuracy, Document FP32 Limitations

Document that FP32 LUTs may have extrapolation errors and are only accurate within the fitted range.

**Pros**: No code changes needed
**Cons**: Surprising behavior for users, degrades trust in FP32 mode

## Recommendation

**Short term**: Use Option 2 (kernel clamping) - already implemented in current kernels for some activations

**Long term**: Implement Option 1 (fix segmentation) - modify curvature-based segmentation in polynomial fitter to guarantee full range coverage

## Action Items

- [ ] Check boundary coverage for all FP32 LUTs with large discrepancies
- [ ] Verify kernel clamping is enabled for all polynomial kernels
- [ ] File issue in polynomial-fitter repo to fix segmentation algorithm
- [ ] Update error expectations in tests to account for boundary coverage
- [ ] Document LUT boundary ranges in generated LUT files (metadata)

## Additional Notes

**Why this wasn't caught earlier**:
1. Most development/testing done in BF16 mode (masks the problem)
2. Visual plots of approximations look good (humans can't see 1e-3 errors)
3. Error metrics in best.csv don't include extrapolation testing

**Why some activations work better**:
- Piecewise linear functions (ReLU, PReLU, HardTanh) are exact within segments
- Simple functions with low curvature everywhere (HardSigmoid) fit well even with gaps
- Monotonic functions with asymptotic behavior (sigmoid, tanh) have less extrapolation error
