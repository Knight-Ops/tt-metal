# Critical Point Fix Summary

## Problem

The kernel_bench test framework's epsilon calculation is **working as designed** - it uses very strict tolerances (epsilon=1.4e-44) when comparing against zero. This means even tiny errors like 1.83e-07 result in 0% accuracy.

**The solution is NOT to change the test framework**, but to make our kernels output **exactly the correct value** for critical points.

## Root Cause

The `generate_embedded_kernels.py` tool had a bug where critical point macros were only generated for:
- Range: exactly `[-10.0, 10.0]`
- Critical point: exactly `x=0.0`

This meant activations like atanh (range `[-0.99, 0.99]`) didn't get critical point handling, even though they have critical_x=0 in the config.

## The Fix

Modified `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded/tools/generate_embedded_kernels.py`:

### Before (Bug):
```python
def get_critical_point_macros(...):
    # Only worked for range [-10, 10] and x=0
    if input_min == -10.0 and input_max == 10.0 and critical_x == 0.0:
        critical_idx = depth // 2
        return macros
    return ""
```

### After (Fixed):
```python
def get_critical_point_macros(...):
    """
    Calculate which uniform boundary the critical point lands on.
    For uniform boundaries: boundary[i] = input_min + (input_max - input_min) * i / depth
    """
    # Calculate boundary index for ANY critical_x value
    range_span = input_max - input_min
    boundary_idx_float = (critical_x - input_min) * depth / range_span
    boundary_idx = round(boundary_idx_float)

    # Reconstruct boundary position
    boundary_position = input_min + (range_span * boundary_idx) / depth

    # If critical_x matches a boundary exactly, generate macros
    if abs(critical_x - boundary_position) < 1e-6:
        return f'''
#define HAS_CRITICAL_POINT
#define CRITICAL_IDX {boundary_idx}
#define CRITICAL_VALUE {critical_value}f
'''
    return ""
```

## What This Does

When `HAS_CRITICAL_POINT` is defined, the piecewise_cubic.cpp kernel has special handling:

```cpp
#ifdef HAS_CRITICAL_POINT
    v_if (x == lut[CRITICAL_IDX]) {
        result = CRITICAL_VALUE;  // Return exact value
    }
    v_else {
#endif
        // ... polynomial evaluation
    }
    v_endif;
#endif
```

This bypasses polynomial evaluation and returns the **exact** critical value when input equals the critical point.

## Which Activations Benefit

### ✅ Get Critical Point Handling (for 16-segment uniform):

| Activation | Critical Point | Range | Boundary Index | Value |
|------------|---------------|-------|----------------|-------|
| atanh | x=0.0 | [-0.99, 0.99] | 8 | 0.0 |
| cos | x=0.0 | [-π, π] | 8 | 1.0 |
| sin | x=0.0 | [-π, π] | 8 | 0.0 |
| cosh | x=0.0 | [-10, 10] | 8 | 1.0 |
| sinh | x=0.0 | [-10, 10] | 8 | 0.0 |
| sigmoid | x=0.0 | [-10, 10] | 8 | 0.5 |
| swish | x=0.0 | [-10, 10] | 8 | 0.0 |
| tanh | x=0.0 | [-10, 10] | 8 | 0.0 |
| tanhshrink | x=0.0 | [-10, 10] | 8 | 0.0 |
| softsign | x=0.0 | [-15, 15] | 8 | 0.0 |

### ❌ Don't Get Critical Point Handling (critical point not on uniform boundary):

| Activation | Critical Points | Why Not |
|------------|----------------|---------|
| gelu | x=±1.41 | For 16 uniform segments in [-10,10], boundaries are at ±1.25, ±2.5, etc. The critical points at ±1.41 fall between boundaries, not ON them. |

## Testing

### Before Fix:
```python
# atanh(0) without critical point
Input: 0.0
Expected: 0.0
TTNN Output: -1.829999973779195e-07
Error: 1.83e-07
Test accuracy: 0% (epsilon=1.4e-44, error ratio=1.3e+37)
```

### After Fix:
```python
# atanh(0) with critical point
Input: 0.0
Expected: 0.0
TTNN Output: 0.0
Error: 0.0
Is exactly zero: True
Test accuracy: 100%
```

## How to Regenerate

After modifying `generate_embedded_kernels.py`, regenerate kernels:

```bash
cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded

# Regenerate specific activation
python3 tools/generate_embedded_kernels.py --activation atanh --degree cubic --depth 16 --segmentation uniform

# Regenerate all activations with critical points
python3 tools/generate_embedded_kernels.py --activation atanh,sigmoid,cos,sin,sinh,cosh,tanh,tanhshrink,softsign,swish --degree cubic --depth 16 --segmentation uniform
```

Then package for kernel_bench:

```bash
cd /localdev/nkapre/kernel_bench
python3 tools/package_embedded_lut.py --activation atanh --degree cubic --depth 16 --segmentation uniform --tt-metal-path /localdev/nkapre/tt-metal
```

## Expected Impact on Pass Rates

For activations with critical points that now get exact handling:
- **Scalar zero tests**: 0% → 100% accuracy
- **Small value tests**: 80-95% → 99%+ accuracy
- **Overall pass rate**: 7-20% → 95%+ expected

For activations without critical point handling (like GELU), no change - polynomial approximation still works but may have small errors at critical points.

## Files Modified

1. `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded/tools/generate_embedded_kernels.py`
   - Function: `get_critical_point_macros()` (around line 110)
   - Change: Calculate boundary index for ANY critical point value, not just x=0 in [-10,10]

## Verification

Check that regenerated kernels have critical point macros:

```bash
grep -A3 "Critical point" kernels/compute/piecewise_cubic_embedded/atanh_cubic_16_uniform.cpp
# Should show:
# // Critical point handling: atanh(0.0) = 0.0
# #define HAS_CRITICAL_POINT
# #define CRITICAL_IDX 8
# #define CRITICAL_VALUE 0.0f
```

## Key Takeaways

1. **Test framework epsilon is correct** - it should be strict for zero comparisons
2. **Solution is in the kernel** - return exact values at critical points, don't approximate
3. **Critical points must be on boundaries** - only works for uniform boundaries where critical_x exactly matches a boundary position
4. **GELU is special** - critical points at ±1.41 don't land on uniform boundaries for standard depths
5. **Always use auto-generation** - never manually create critical point macros
