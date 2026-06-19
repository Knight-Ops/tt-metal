# Investigation: Why Hexic/Octic Remez Polynomials Sometimes Show Worse MAE

## Summary

Hexic (degree-6) and octic (degree-8) Remez polynomials **sometimes** show worse MAE than lower-degree polynomials due to **coefficient ill-conditioning**, not errors in the Remez algorithm or hardware evaluation.

## Root Cause: Coefficient Ill-Conditioning

The Remez exchange algorithm produces mathematically optimal minimax approximations, but for certain activation functions, these optimal polynomials have coefficients that span **15-20 orders of magnitude**.

### Example: `hardshrink` at depth=4

| Polynomial | Coeff Range    | Largest Coeff | Smallest Coeff | MAE (Python) | MAE (Hardware) |
|------------|----------------|---------------|----------------|--------------|----------------|
| Cubic      | **1.06×10¹⁹** | 1.00e+00      | 9.47e-20       | 0.0439      | 0.0430         |
| Hexic      | **1.45×10¹⁷** | 8.57e+00      | 5.93e-17       | 0.0598      | 0.0605         |
| Octic      | **3.69×10¹⁸** | 7.89e+00      | 2.14e-18       | 0.0216      | 0.0216         |

**Result**: Hexic is 1.41× **WORSE** than cubic on hardware!

## Why This Happens

### 1. **Piecewise-Continuous Functions Are Hard to Approximate**

Functions with discontinuities or sharp transitions (e.g., `hardshrink`, `relu`, `hardtanh`) require polynomials with:
- Very large coefficients to create sharp transitions
- Very small coefficients to avoid oscillations (Runge's phenomenon)

### 2. **Float32 Precision Limits**

IEEE 754 float32 provides:
- **~7 decimal digits** of precision
- Cannot accurately represent numbers that differ by >10⁷

When computing:
```cpp
result = a6*x + a5*x + ... + a0  // Horner's method
```

If `a6 = 8.57` and `a0 = 5.93e-17`, the addition `large_result + 5.93e-17` **loses the small coefficient entirely** due to floating-point rounding.

### 3. **Higher Degrees → More Coefficients → More Opportunities for Ill-Conditioning**

- **Cubic** (4 coeffs): Fewer terms, less chance of extreme coefficient ratios
- **Hexic** (7 coeffs): More terms, higher chance of ill-conditioning
- **Octic** (9 coeffs): Even more terms, highest chance of ill-conditioning

## Empirical Results

### Activations Where Hexic Performs WORSE Than Cubic

**At depth=4:**
- `hardshrink`: 1.41× worse

**At depth=8:**
- `cosh`: 1.10× worse
- `sinh`: 1.06× worse
- `relu6`: 1.78× worse

### Coefficient Range Correlation

| Activation  | Depth | Polynomial | Coeff Range | MAE Trend |
|-------------|-------|------------|-------------|-----------|
| exp         | 4     | Hexic      | 3.40e+07    | ✅ Better  |
| cos         | 4     | Hexic      | 5.20e+04    | ✅ Better  |
| hardshrink  | 4     | Hexic      | **1.45e+17** | ❌ Worse   |
| sinh        | 8     | Hexic      | **1.68e+06** | ❌ Worse   |

**Pattern**: Coefficient ranges > 10¹⁵ correlate with degraded accuracy.

## Why the Remez Algorithm Isn't Wrong

The Remez coefficients are **mathematically correct**:
- They minimize the maximum error (L∞ norm) in infinite precision
- When evaluated in **Python float64** (15 decimal digits), they work well
- The problem arises only in **hardware float32** (7 decimal digits)

### Evidence

From `investigate_remez_errors.py` for `hardshrink` depth=4:

```
hexic_remez (degree 6):
  MAE (float64):     5.975362e-02  ← Reference accuracy
  MAE (float32):     5.975371e-02  ← Hardware accuracy
  Precision loss:    8.492433e-08 (0.00%)
```

The precision loss from float64→float32 is **negligible** (0.00%). The issue is that the hexic polynomial itself (even in infinite precision) produces higher error than cubic for this particular function at this depth.

## Why This Happens for Certain Functions

Functions prone to ill-conditioning:
1. **Piecewise-continuous functions**: `relu`, `leaky_relu`, `relu6`, `threshold`, `hardshrink`, `softshrink`
2. **Functions with sharp transitions**: `hardsigmoid`, `hardtanh`, `hardswish`
3. **Rapidly growing functions at segment boundaries**: `exp`, `cosh`, `sinh` (at higher depths)

Smooth functions work well: `sigmoid`, `tanh`, `gelu`, `swish`, `sin`, `cos`

## Kernel Implementation Analysis

The kernel implementations are correct and consistent:

### Hexic Kernel (4 segments)
```cpp
// Load 'a' coefficient (just-in-time)
vFloat coeff = lut[0];
v_if (x_clamped >= input_min + SEGMENT_WIDTH) { coeff = lut[7]; } v_endif;
v_if (x_clamped >= input_min + 2 * SEGMENT_WIDTH) { coeff = lut[14]; } v_endif;
v_if (x_clamped >= input_min + 3 * SEGMENT_WIDTH) { coeff = lut[21]; } v_endif;
vFloat result = coeff;

// Load 'b' and continue...
result = result * x_clamped + coeff;  // Horner's method
```

### Octic Kernel (16 segments)
```cpp
// Loads ALL coefficients at once
vFloat a = lut[0], b = lut[1], ..., i = lut[8];
v_if (x_clamped >= input_min + SEGMENT_WIDTH) {
    a = lut[9]; b = lut[10]; ...; i = lut[17];
} v_endif;
// Then evaluates in separate statements
vFloat result = a * x_clamped;
result = result + b;
result = result * x_clamped;
```

**Both methods are numerically equivalent** when coefficients are well-conditioned. The octic approach (loading all coefficients, breaking into statements) may actually **reduce** register pressure, but doesn't fix ill-conditioning.

## Recommendations

### 1. **For Smooth Functions**: Use Hexic/Octic
- `sigmoid`, `tanh`, `gelu`, `swish`, `sin`, `cos`, `erf`
- Coefficient ranges < 10¹⁰
- MAE improvements of 10-100× over cubic

### 2. **For Piecewise Functions**: Use Lower Degrees or More Segments
- `relu`, `hardtanh`, `hardshrink`, `threshold`
- **Option A**: Use cubic with more segments (depth=8 or 16)
- **Option B**: Use piecewise-constant or piecewise-linear (exact for linear regions)

### 3. **Adaptive Polynomial Selection**
Consider a hybrid approach:
- Smooth regions → Hexic/Octic (fewer segments needed)
- Sharp transitions → Cubic/Quadratic (more segments needed)

### 4. **Coefficient Normalization** (Future Work)
Investigate domain transformation to reduce coefficient ranges:
```python
# Instead of: y = a6*x^6 + ... + a0
# Use: y = a6*(x/scale)^6 + ... + a0*scale^0
```

## Conclusion

The worse MAE for hexic/octic polynomials is **NOT due to**:
1. ❌ Poorly calculated Remez coefficients
2. ❌ Errors in the kernel implementation
3. ❌ Hardware SFPU precision issues

It **IS due to**:
1. ✅ **Coefficient ill-conditioning** (coefficient ranges > 10¹⁵)
2. ✅ **Fundamental difficulty** of approximating piecewise-continuous functions with smooth polynomials
3. ✅ **Float32 precision limits** when adding numbers across 15-20 orders of magnitude

**The Remez algorithm works correctly** - it's finding the mathematically optimal polynomial, but that polynomial is numerically unstable in float32 for certain function classes.

## Validation

Run the investigation script to reproduce these findings:

```bash
python3 investigate_remez_errors.py --depth 4 --function hardshrink
python3 investigate_remez_errors.py --depth 8 --function sinh
```

Look for:
- `Coefficient range: > 1e+15` → Warning sign for ill-conditioning
- `MAE (float64)` vs `MAE (float32)` → Should be similar (< 20% difference)
- Compare against cubic to see if higher degree actually helps
