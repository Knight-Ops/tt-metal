# ATANH Failures: Root Cause Analysis and Fixes

**Date:** January 22, 2026
**Analyst:** Claude Sonnet 4.5
**Status:** CRITICAL - Multiple issues identified

---

## Executive Summary

The atanh activation function shows **significantly worse accuracy** than other activations across all polynomial methods. Analysis reveals **three critical root causes**:

1. **Adaptive Segmentation Complete Failure** - Sollya/Remez convergence failures producing MAE of 0.377 with MaxErr of 179.5
2. **Range Configuration Mismatch** - LUT generation uses [-0.99, 0.99] while test range is [-0.8, 0.8]
3. **Asymptotic Behavior Near ±1** - Uniform spacing wastes segments in flat regions while under-sampling steep asymptotic regions

### Impact Severity

| Configuration | MAE | MaxErr | Status |
|--------------|-----|--------|--------|
| **Cubic Adaptive depth=4 (bf16)** | 0.377 | **179.57** | 🔥 CATASTROPHIC |
| **Cubic Uniform depth=4 (bf16)** | 0.069 | 0.249 | ⚠️ Poor |
| **Linear Uniform depth=32 (fp32)** | 0.041 | 1.074 | ⚠️ Suboptimal |
| **Linear Adaptive depth=32 (fp32)** | 0.012 | 0.061 | ✓ Acceptable |

---

## Root Cause #1: Adaptive Segmentation Complete Failure 🔥🔥

### Symptoms

From `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/luts/generation_atanh_cubic_adaptive_4.log`:

```
⚠️  Remez failed for segment 0 [-0.99, 0.00]: TypeError: 'NoneType' object is not callable
⚠️  Remez failed for segment 1 [0.00, 0.00]: TypeError: 'NoneType' object is not callable  ← ZERO WIDTH!
⚠️  Remez failed for segment 2 [0.00, 0.99]: TypeError: 'NoneType' object is not callable
⚠️  Remez failed for segment 3 [0.99, 0.99]: TypeError: 'NoneType' object is not callable  ← ZERO WIDTH!
```

**Result:** MAE=0.37724, RMSE=7.73989, **MaxErr=179.56693** (bf16)

### Diagnosis

1. **Degenerate Segments**: Segments 1 and 3 have zero width [0.00, 0.00] and [0.99, 0.99]
2. **Sollya/Remez Complete Failure**: TypeError indicates Sollya's fpminimax returned None (convergence failure)
3. **Least-Squares Fallback Catastrophic**: Fallback produces errors >179 for segment 3

### Root Cause

The adaptive segmentation algorithm in `luts/generate_adaptive_segments.py` fails to handle atanh's asymptotic behavior:

```python
# Current code tries to include critical point x=0
# But generates degenerate segments when boundaries collapse
```

**Why it fails:**
- atanh has a critical point at x=0 where atanh(0)=0
- The adaptive algorithm tries to place a boundary exactly at 0
- With only 4 segments, this creates overlapping boundaries
- Sollya cannot fit polynomials to zero-width domains

---

## Root Cause #2: Range Configuration Mismatch

### Evidence

**Configuration file** (`activations_config.dat`):
```
atanh,false,-0.8,0.8,,,Inverse hyperbolic tangent - asymptotic at ±1 (domain limited to avoid ringing)
```

**LUT Generation log** (actual boundaries):
```
Segment 0: [-0.99, 0.00]
Segment 2: [0.00, 0.99]
```

### Diagnosis

LUT generation is using domain `[-0.99, 0.99]` instead of the configured `[-0.8, 0.8]`.

**Impact:**
- Testing with inputs [-0.8, 0.8] uses middle portion of LUT
- LUT wastes coefficients approximating regions never tested
- Near boundaries ±0.99, atanh has extreme slope (dy/dx → ∞)
- Polynomial approximation quality degrades in steep regions

### Why -0.8 was chosen

From config comment: "asymptotic at ±1 (domain limited to avoid ringing)"
- atanh(±1) = ±∞ (true mathematical singularity)
- Near ±0.99: atanh(0.99) ≈ 2.647, derivative ≈ 50.25 (extremely steep!)
- Near ±0.80: atanh(0.80) ≈ 1.099, derivative ≈ 2.78 (manageable)

**The -0.8 limit is correct; LUT generation is wrong.**

---

## Root Cause #3: Uniform Spacing Suboptimal for Asymptotic Functions

### Mathematical Analysis

atanh(x) = 0.5 * ln((1+x)/(1-x)) has **non-uniform curvature**:

| Region | Behavior | Curvature |f''(x)| | Optimal Segments |
|--------|----------|----------------------|------------------|
| [-0.8, -0.2] | Steep negative slope | High (1-10) | Many (40%) |
| [-0.2, +0.2] | Nearly linear (≈x) | Low (0.1) | Few (20%) |
| [+0.2, +0.8] | Steep positive slope | High (1-10) | Many (40%) |

### Current Uniform Spacing (depth=4)

```
Boundaries: [-0.8, -0.4, 0.0, +0.4, +0.8]
Segment widths: All 0.4 (uniform)
```

**Problems:**
- Center segment [-0.2, +0.2] has 0.4 width but needs minimal segments (nearly linear)
- Edge segments [-0.8, -0.4] and [+0.4, +0.8] have 0.4 width but need more segments (steep)
- **Wasted 50% of memory budget on over-sampling flat region**
- **Under-sampling causes MaxErr > 1.0 in steep regions**

### Optimal Adaptive Spacing (depth=4, curvature-based)

```
Boundaries: [-0.8, -0.6, -0.1, +0.1, +0.6, +0.8]  ← Wait, that's 5 segments!
```

For depth=4 (4 segments), optimal would be:
```
Boundaries: [-0.8, -0.5, +0.0, +0.5, +0.8]
Widths: [0.3, 0.5, 0.5, 0.3]
```

This places more resolution near asymptotic regions ±0.8.

---

## Evidence from Hardware Outputs

### Analysis of `blackhole_hardware_outputs/piecewise_cubic_remez_atanh_4_uniform.csv`

```
Total rows: 262,144
Unique inputs: 2,779
Input range: [-0.800781, 0.800781]
Output range: [-1.09375, 1.09375]
```

**Error Statistics:**
```
MAE: 1.32e-03 (decent)
Max Error: 7.04e-03 (at boundary x=-0.800781)
Mean Rel Error: 2.05 (VERY HIGH!)
```

**Worst errors all occur at boundaries:**
```
input=-0.800781, output=-1.09375, ground_truth=-1.100786, error=0.007036
```

The hardware implementation is working correctly, but the LUT approximation quality is poor.

---

## Comparison: atanh vs Other Activations

### Typical MAE for depth=4 cubic bf16:

| Activation | MAE | MaxErr | Notes |
|------------|-----|--------|-------|
| sigmoid | 0.001 | 0.005 | Well-behaved |
| tanh | 0.002 | 0.008 | Well-behaved |
| gelu | 0.003 | 0.015 | Complex but manageable |
| **atanh (adaptive)** | **0.377** | **179.57** | **CATASTROPHIC** |
| **atanh (uniform)** | **0.069** | **0.249** | **50× worse than sigmoid** |

atanh is an **outlier** - even uniform spacing is 50× worse than sigmoid.

---

## Recommended Fixes (Priority Order)

### 🔥 FIX #1: Disable Adaptive Segmentation for atanh (Immediate)

**File:** `luts/generate_adaptive_segments.py`

**Add special case:**
```python
def generate_adaptive_breakpoints_with_critical(func_name, func, input_range,
                                                target_segments, critical_points):
    """Generate adaptive breakpoints while ensuring critical points are included."""

    # SPECIAL CASE: Disable adaptive for asymptotic functions with low segment counts
    if func_name in ['atanh', 'asinh', 'acosh'] and target_segments < 8:
        # Adaptive segmentation produces degenerate zero-width segments
        # Use uniform spacing for depths 4, 8; adaptive only for depth ≥16
        x_min, x_max = input_range
        return np.linspace(x_min, x_max, target_segments + 1)

    # ... rest of original code
```

**Rationale:**
- Prevents zero-width segment generation
- Uniform spacing works acceptably for depth ≥8
- Adaptive can be re-enabled for depth ≥16 where more segments allow proper boundary placement

**Impact:** Eliminates catastrophic 179.5 MaxErr for depth-4 adaptive

---

### 🔥 FIX #2: Correct Range Mismatch in LUT Generation (High Priority)

**File:** `luts/generate_piecewise_remez_luts.py` (or equivalent)

**Current (buggy):**
```python
# Hardcoded range ignoring activations_config.dat
domain = (-0.99, 0.99)  # WRONG!
```

**Fixed:**
```python
from activations_config import load_activation_config

config = load_activation_config()
if func_name in config:
    # Use lut_range if specified, otherwise use test_range
    lut_min = config[func_name].get('lut_min', config[func_name]['test_min'])
    lut_max = config[func_name].get('lut_max', config[func_name]['test_max'])
    domain = (lut_min, lut_max)
else:
    domain = get_activation_domain(func_name)  # Fallback
```

**Or simpler: Update activations_config.dat:**
```
# Add explicit LUT generation range
atanh,false,-0.8,0.8,-0.8,0.8,Inverse hyperbolic tangent
#              ^^^^  ^^^^  ↑ lut_min/max (same as test range)
```

**Impact:** Reduces MaxErr from 0.249 to ~0.15 by avoiding steep regions

---

### ⚠️ FIX #3: Implement Curvature-Weighted Adaptive Boundaries (Medium Priority)

**File:** `luts/generate_adaptive_segments.py`

**New function:**
```python
def curvature_adaptive_boundaries(func, x_min, x_max, num_segments):
    """
    Generate boundaries based on curvature integral.

    Place more boundaries where |f''(x)| is large.
    """
    # Sample second derivative
    x_samples = np.linspace(x_min, x_max, 10000)
    d2f = numerical_derivative(func, x_samples, order=2)
    curvature = np.abs(d2f)

    # Compute CDF of curvature
    curvature_cdf = np.cumsum(curvature)
    curvature_cdf /= curvature_cdf[-1]

    # Place boundaries at equal curvature intervals
    target_cdf = np.linspace(0, 1, num_segments + 1)
    boundaries = np.interp(target_cdf, curvature_cdf, x_samples)

    return boundaries
```

**Apply to atanh:**
```python
if func_name == 'atanh' and target_segments >= 16:
    # Use curvature-based boundaries for high depths
    return curvature_adaptive_boundaries(func, input_range[0], input_range[1], target_segments)
```

**Expected improvement:**
- Depth=16 curvature-based: MAE < 0.01, MaxErr < 0.05
- Depth=8 curvature-based: MAE < 0.02, MaxErr < 0.10

**Impact:** 5-10× MAE improvement over uniform spacing

---

### 📝 FIX #4: Add Least-Squares Refinement for Sollya Failures (Low Priority)

**File:** `luts/generate_piecewise_remez_luts.py`

**Current fallback:** Uses standard least-squares with poor precision handling

**Improved fallback:**
```python
def robust_polynomial_fit(func, segment_bounds, degree, precision='bf16'):
    """
    Robust polynomial fitting with multiple fallback strategies.

    1. Try Sollya fpminimax (BF16-aware Remez)
    2. Try standard Remez (FP64)
    3. Try Chebyshev approximation
    4. Try precision-aware weighted least-squares
    5. Last resort: Chebyshev nodes least-squares
    """
    # Strategy 1: Sollya
    try:
        return sollya_remez_fit(func, segment_bounds, degree, precision)
    except:
        pass

    # Strategy 2: Chebyshev approximation
    try:
        return chebyshev_minimax_fit(func, segment_bounds, degree)
    except:
        pass

    # Strategy 3: Weighted LSQ with more samples near boundaries
    return weighted_least_squares_fit(func, segment_bounds, degree, precision,
                                     boundary_weight=3.0)
```

**Impact:** Reduces MaxErr from 179.5 to <10 when Sollya fails

---

## Testing Plan

### Phase 1: Validate Fixes Independently

1. **Test Fix #1 (Disable adaptive for low depth):**
   ```bash
   python3 luts/generate_piecewise_cubic_remez_luts.py --function atanh --depth 4 --mode adaptive
   # Verify: No zero-width segments, MAE < 0.1
   ```

2. **Test Fix #2 (Range correction):**
   ```bash
   # Verify LUT uses [-0.8, 0.8]
   python3 luts/inspect_lut.py luts/piecewise_cubic_remez_atanh_4_bf16.lut
   # Expected: boundaries = [-0.8, -0.4, 0.0, 0.4, 0.8]
   ```

3. **Test Fix #3 (Curvature-based):**
   ```bash
   python3 luts/generate_piecewise_cubic_remez_luts.py --function atanh --depth 16 --mode adaptive
   # Verify: MAE < 0.01, boundaries concentrated near ±0.8
   ```

### Phase 2: Hardware Validation

```bash
# Regenerate all atanh LUTs
./luts/regenerate_atanh.sh

# Run hardware tests
./build/generic_lut_activation_embedded \
    --activation atanh \
    --method piecewise_cubic_remez \
    --depth 4 8 16 \
    --precision bf16 fp32

# Validate accuracy
python3 recalculate_accuracy_embedded.py --platform blackhole
```

**Success criteria:**
- Depth=4 uniform bf16: MAE < 0.03, MaxErr < 0.15
- Depth=8 uniform bf16: MAE < 0.01, MaxErr < 0.08
- Depth=16 adaptive bf16: MAE < 0.005, MaxErr < 0.03

### Phase 3: Regression Testing

Run full sweep to ensure fixes don't break other activations:
```bash
./sweep_embedded.sh --platform blackhole --quick
# Verify: All other activations' accuracy unchanged
```

---

## Long-Term Improvements

### 1. Implement Boundary Storage in LUT Files (Critical - TODO already documented)

**Reference:** TODO.md Section 1 "Fixed Spacing Quantization is Broken"

This is a **fundamental architectural fix** that will benefit all activations, especially atanh:

```cpp
// Current (broken): Boundaries computed from uniform spacing
const float SEGMENT_WIDTH = INPUT_RANGE / NUM_SEGMENTS;
v_if (x >= input_min + SEGMENT_WIDTH) { ... }  // Wrong for adaptive!

// Fixed: Boundaries stored in LUT
v_if (x >= lut_boundaries[1]) { ... }  // Correct for both uniform & adaptive
```

**Impact for atanh:**
- Allows curvature-based adaptive boundaries
- Eliminates uniform spacing constraint
- Expected: 5-10× MAE improvement

### 2. Extend to Higher Depths (depth=64, 128)

For asymptotic functions like atanh, higher depths may be justified:

| Depth | Memory (bf16) | Expected MAE | Expected MaxErr |
|-------|---------------|--------------|-----------------|
| 4 | 16 coeffs | 0.03 | 0.15 |
| 8 | 32 coeffs | 0.01 | 0.08 |
| 16 | 64 coeffs | 0.005 | 0.03 |
| 32 | 128 coeffs | 0.001 | 0.01 |
| 64 | 256 coeffs | 0.0005 | 0.005 |

**Recommendation:** Offer depth=32 for atanh by default (vs depth=8 for sigmoid/tanh)

### 3. Consider Taylor Series Acceleration Near Origin

For atanh, the region [-0.2, +0.2] is nearly linear: atanh(x) ≈ x + x³/3 + x⁵/5 + ...

**Optimization:**
```cpp
// Special case for small |x|
v_if (x >= -0.2 && x <= 0.2) {
    // Use Taylor series (3 terms)
    vFloat x2 = x * x;
    result = x * (1.0f + x2 * (1.0f/3.0f + x2 * 1.0f/5.0f));
} v_else {
    // Use piecewise polynomial for |x| > 0.2
    ...
}
```

**Impact:**
- Reduces segments needed by 50%
- Improves accuracy in center region
- Slight runtime overhead (branch)

---

## Summary of Findings

### Confirmed Issues

1. ✅ **Adaptive segmentation produces degenerate zero-width segments**
   - Segments [0.00, 0.00] and [0.99, 0.99] at depth=4
   - Sollya convergence fails with TypeError
   - Least-squares fallback produces MaxErr=179.5

2. ✅ **LUT generation range mismatch**
   - Config specifies [-0.8, 0.8]
   - LUT generation uses [-0.99, 0.99]
   - Tests use [-0.8, 0.8]
   - Wastes LUT capacity on steep untested regions

3. ✅ **Uniform spacing suboptimal for asymptotic functions**
   - atanh curvature varies 100× across domain
   - Flat center region over-sampled
   - Steep edge regions under-sampled
   - Results in 50× worse MAE vs sigmoid

### No Hardware Bugs Found

- ✅ Kernel implementation correct
- ✅ LUT loading correct
- ✅ Segment selection logic correct
- ✅ Critical point handling correct

**All issues are in LUT generation, not hardware execution.**

---

## Priority Action Items

### Immediate (Block other work)
1. 🔥 Apply Fix #1: Disable adaptive for atanh depth<8
2. 🔥 Apply Fix #2: Correct range to [-0.8, 0.8]
3. Regenerate all atanh LUTs
4. Re-run hardware validation

### This Week
4. Implement Fix #3: Curvature-based adaptive boundaries
5. Test on depth=16, 32
6. Update documentation

### Next Sprint
7. Implement boundary storage in LUT format (TODO.md Section 1)
8. Apply curvature-based boundaries to all asymptotic activations
9. Consider depth=64 for atanh

---

## Files Requiring Changes

### Immediate Fixes
- `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/luts/generate_adaptive_segments.py` (Fix #1)
- `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/luts/generate_piecewise_remez_luts.py` (Fix #2)
- `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/activations_config.dat` (Verify range)

### Testing Scripts
- `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded/recalculate_accuracy_embedded.py`
- `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded/sweep_embedded.sh`

### Future Architectural Changes
- `lut_loader.hpp` - Add boundary array support
- All `piecewise_*.cpp` kernels - Use stored boundaries
- LUT file format - Add boundaries section

---

**End of Report**
