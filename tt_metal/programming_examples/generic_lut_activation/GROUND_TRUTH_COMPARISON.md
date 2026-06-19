# Ground Truth Reference Implementation Comparison

This document compares the activation function reference implementations across:
1. **sweep_best.sh** (`compute_reference()`) - Hardware validation ground truth
2. **polynomial-fitter** (`sollya_expressions.py`) - LUT generation ground truth
3. **polynomial-fitter** (`ground_truth.py`) - Original reference (minimal)

## Critical Discrepancies Found

### 1. GELU - DIFFERENT FORMULAS ⚠️
- **sweep_best.sh**: Uses **tanh approximation**
  ```python
  0.5 * x * (1 + math.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))
  ```

- **sollya_expressions.py**: Uses **erf (exact formula)**
  ```python
  'x * 0.5 * (1 + erf(x / sqrt(2)))'
  ```

**Impact**: These are different functions! The tanh approximation is a common fast GELU variant, but mathematically different from the true GELU.
- True GELU: `x * Φ(x)` where Φ is the standard normal CDF = `x * 0.5 * (1 + erf(x/√2))`
- Tanh GELU: Fast approximation with ~0.1% max error

**Status**: 🔴 **MISMATCH** - Need to determine which version the LUTs were fitted to.

---

### 2. CELU - LOGIC ERROR ⚠️
- **sweep_best.sh**: BROKEN LOGIC
  ```python
  max(0, x) + min(0, math.exp(x) - 1) if x < 0 else x
  ```
  This always returns `x` because the condition `if x < 0` makes the entire expression evaluate to `x` for `x >= 0`, and for `x < 0` it computes `max(0, x) + min(0, exp(x)-1)` which is just `min(0, exp(x)-1)` since `max(0, x)=0` when `x<0`.

- **sollya_expressions.py**: CORRECT
  ```python
  # For x < 0: alpha * (exp(x/alpha) - 1)
  # For x >= 0: x
  ```

**Impact**: sweep_best.sh has incorrect CELU implementation for negative values.

**Status**: 🔴 **BUG IN SWEEP_BEST.SH**

---

### 3. HardSigmoid - DIFFERENT FORMULAS ⚠️
- **sweep_best.sh**: PyTorch formula
  ```python
  max(0, min(1, (x + 3) / 6))
  ```
  This is: `clip((x + 3) / 6, 0, 1)`

- **sollya_expressions.py**: Different formula
  ```python
  'min(max(0.2 * x + 0.5, 0), 1)'
  ```
  This is: `clip(0.2*x + 0.5, 0, 1)`

**Impact**: These are DIFFERENT functions!
- PyTorch: slope=1/6, intercept=0.5 (transitions from 0 to 1 over range -3 to +3)
- sollya: slope=0.2 (=1/5), intercept=0.5 (transitions from 0 to 1 over range -2.5 to +2.5)

**Status**: 🔴 **MISMATCH** - Need to verify which formula was used for LUT fitting.

---

### 4. HardSwish - DIFFERENT FORMULAS ⚠️
- **sweep_best.sh**: Uses hardsigmoid with PyTorch formula
  ```python
  x * max(0, min(1, (x + 3) / 6))
  ```

- **sollya_expressions.py**: Different base formula
  ```python
  'x * min(max(x + 3, 0), 6) / 6'
  ```
  This simplifies to: `x * clip(x + 3, 0, 6) / 6`

**Impact**: Need to verify these are equivalent:
- sweep: `x * clip((x+3)/6, 0, 1)` = `x * clip(x+3, 0, 6) / 6` ✓ **EQUIVALENT**

**Status**: ✅ **MATCH** (after algebraic simplification)

---

### 5. Missing from ground_truth.py
The original `ground_truth.py` only has **5 functions**:
- sigmoid
- tanh
- gelu (erf formula)
- sinh
- cosh

**Missing**: All 26 other activations that are in sweep_best.sh and sollya_expressions.py

**Status**: 🟡 **ground_truth.py is outdated/incomplete** (but not used for LUT generation)

---

## Function-by-Function Comparison

| Activation | sweep_best.sh | sollya_expressions.py | Match? | Notes |
|------------|---------------|----------------------|--------|-------|
| cos | `math.cos(x)` | `'cos(x)'` | ✅ | Standard |
| sin | `math.sin(x)` | `'sin(x)'` | ✅ | Standard |
| tanh | `math.tanh(x)` | `'tanh(x)'` | ✅ | Standard |
| sigmoid | `1/(1+exp(-x))` | `'1/(1+exp(-x))'` | ✅ | Standard |
| gelu | tanh approx | erf exact | 🔴 | **DIFFERENT FORMULAS** |
| relu | `max(0, x)` | `'max(0, x)'` | ✅ | Standard |
| exp | `math.exp(x)` | `'exp(x)'` | ✅ | Standard |
| erf | `math.erf(x)` | `'erf(x)'` | ✅ | Standard |
| cosh | `math.cosh(x)` | `'cosh(x)'` | ✅ | Standard |
| sinh | `math.sinh(x)` | `'sinh(x)'` | ✅ | Standard |
| atanh | `math.atanh(x)` | `'atanh(x)'` | ✅ | Standard (with guards) |
| celu | BROKEN | Correct | 🔴 | **BUG IN SWEEP** |
| elu | `x if x>0 else exp(x)-1` | Piecewise | ✅ | Equivalent |
| selu | α=1.67326, λ=1.0507 | α=1.6732632..., λ=1.0507009... | 🟡 | **Low precision constants in sweep** |
| softplus | `log(1+exp(x))` | `'log(1+exp(x))'` | ✅ | Standard |
| hardsigmoid | `clip((x+3)/6, 0, 1)` | `clip(0.2*x+0.5, 0, 1)` | 🔴 | **DIFFERENT FORMULAS** |
| hardswish | Uses PyTorch hardsigmoid | Different formula (equiv) | ✅ | Algebraically equivalent |
| hardtanh | `clip(x, -1, 1)` | `'min(max(x, -1), 1)'` | ✅ | Equivalent |
| hardshrink | λ=0.5 | λ=0.5 | ✅ | Same |
| leaky_relu | α=0.01 | α=0.01 | ✅ | Same |
| prelu | α=0.25 | α=0.25 | ✅ | Same |
| relu6 | `clip(x, 0, 6)` | `'min(max(0, x), 6)'` | ✅ | Equivalent |
| mish | `x*tanh(log(1+exp(x)))` | `'x*tanh(log(1+exp(x)))'` | ✅ | Standard |
| swish | `x/(1+exp(-x))` | `'x/(1+exp(-x))'` | ✅ | Standard (silu) |
| softshrink | λ=0.5 | λ=0.5 | ✅ | Same |
| tanhshrink | `x - tanh(x)` | `'x - tanh(x)'` | ✅ | Standard |
| softsign | `x/(1+abs(x))` | `'x/(1+abs(x))'` | ✅ | Standard |
| threshold | θ=0, val=0 | θ=0 | ✅ | Same |
| logsigmoid | `-log(1+exp(-x))` | `'-log(1+exp(-x))'` | ✅ | Standard |

---

## Summary of Issues

### Critical (Must Fix) 🔴
1. **GELU**: Different formulas (tanh approx vs erf exact) - need to verify which LUTs were fitted to
2. **CELU**: Broken logic in sweep_best.sh - returns wrong values for x < 0
3. **HardSigmoid**: Different formulas (PyTorch vs different slope) - need to verify LUT fitting

### Low Priority 🟡
4. **SELU**: Low-precision constants in sweep_best.sh (should use full precision)
5. **ground_truth.py**: Outdated/incomplete (only 5 functions)

---

## Recommendations

1. **IMMEDIATE**: Fix CELU in sweep_best.sh:
   ```python
   elif activation == "celu":
       alpha = 1.0
       return x if x >= 0 else alpha * (math.exp(x / alpha) - 1)
   ```

2. **VERIFY**: Check which GELU formula was used in polynomial fitter best.csv:
   - Look at the actual LUT coefficients for GELU
   - Compare error against both formulas
   - Update sweep_best.sh or sollya_expressions.py to match

3. **VERIFY**: Check which HardSigmoid formula was used:
   - PyTorch: `(x + 3) / 6`
   - Alternative: `0.2 * x + 0.5`

4. **IMPROVE**: Update SELU constants to full precision:
   ```python
   alpha = 1.6732632423543772848170429916717
   scale = 1.0507009873554804934193349852946
   ```

5. **OPTIONAL**: Update ground_truth.py to include all activations (or deprecate it)

---

## Actions Taken

1. ✅ Verified polynomial fitter uses erf-based GELU (not tanh approximation)
2. ✅ Verified polynomial fitter uses 0.2*x+0.5 for HardSigmoid (not PyTorch (x+3)/6)
3. ✅ Fixed GELU in sweep_best.sh to use erf formula
4. ✅ Fixed HardSigmoid in sweep_best.sh to use 0.2*x+0.5 formula
5. ✅ Fixed CELU in sweep_best.sh to use proper alpha-parameterized formula
6. ✅ Added scipy.special.erf import to sweep_best.sh

## Remaining Issue

**HardSwish Inconsistency in Polynomial Fitter**: The polynomial fitter has an internal inconsistency:
- `hardsigmoid` uses: `0.2*x + 0.5`
- `hardswish` uses: `x * (x+3)/6` clipped (PyTorch-style)

This means hardswish embeds a different hardsigmoid formula than the standalone hardsigmoid function. This is a bug in sollya_expressions.py but doesn't affect our validation since hardswish has its own separate LUT fitted to its specific formula.
