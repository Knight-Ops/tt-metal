# Breakpoint and Discontinuity Analysis for Activation Functions

**Generated**: 2026-01-18
**Tool**: `analyze_breakpoints.py`

## Executive Summary

This document identifies **discontinuities** and **optimal breakpoint locations** for all 29 activation functions in the generic LUT approximation system. Understanding these characteristics is critical for:

1. **Adaptive segmentation**: Placing more segments in high-curvature regions
2. **Error reduction**: Ensuring discontinuities are segment boundaries
3. **Memory efficiency**: Using fewer segments in saturated/flat regions

---

## Key Findings by Category

### 1. Functions with Derivative Discontinuities

These functions have **non-smooth first derivatives** and require careful handling:

| Activation | Discontinuity Location(s) | Magnitude | Notes |
|------------|--------------------------|-----------|-------|
| **relu** | x ≈ 0 | 1.0 | Classic discontinuity at zero |
| **leaky_relu** | x ≈ 0 | ~0.99 | Slope changes from α to 1 |
| **prelu** | x ≈ 0 | ~0.75 | Slope changes from α=0.25 to 1 |
| **threshold** | x ≈ 0 | 1.0 | Step function behavior |
| **selu** | x ≈ 0 | ~0.88 | Scaled ELU discontinuity |
| **hardsigmoid** | x ≈ -3, x ≈ +3 | Variable | Piecewise linear with flat regions |
| **hardswish** | x ≈ -3, x ≈ +3 | Variable | Piecewise with product term |
| **hardtanh** | x = -1, x = +1 | Variable | Clipping boundaries |
| **relu6** | x = 0, x = 6 | Variable | ReLU with upper clip |
| **hardshrink** | x ≈ -0.5, x ≈ +0.5 | Variable | Shrinkage thresholds |
| **softshrink** | x ≈ -0.5, x ≈ +0.5 | Variable | Soft thresholding |

**Recommendation**: Always place segment boundaries **exactly at discontinuities** to avoid attempting to approximate a discontinuous function within a segment.

---

### 2. Functions with High Curvature Regions

These functions have regions where |f''(x)| is large, requiring **dense sampling**:

| Activation | Peak Curvature Location | Max |f''(x)| | Characteristics |
|------------|------------------------|--------------|-----------------|
| **exp** | x → +5 | 148.4 | Exponential growth, highest curvature at upper bound |
| **gelu** | x ≈ 0 | 0.80 | Concentrated around origin |
| **sigmoid** | x ≈ ±1.3 | 0.10 | Symmetric peaks around inflection |
| **tanh** | x ≈ ±0.7 | 0.39 | Similar to sigmoid but narrower |
| **softplus** | x ≈ 0 | ~1.0 | Smooth ReLU transition region |
| **mish** | x ≈ -0.3, x ≈ 1.5 | Variable | Two distinct high-curvature regions |
| **atanh** | x → ±0.8 | Large | Approaches infinity at domain boundaries |
| **cosh** | x → ±5 | Large | Exponential-like behavior at extremes |

**Recommendation**: Use **adaptive segmentation** with higher density in these regions. For exp and cosh, consider exponentially-spaced breakpoints rather than uniform spacing.

---

### 3. Functions with Inflection Points

Inflection points (where f''(x) = 0) mark curvature changes:

| Activation | Inflection Point(s) | Significance |
|------------|---------------------|--------------|
| **sigmoid** | x = 0 | Single inflection, curvature changes sign |
| **tanh** | x = 0 | Similar to sigmoid |
| **gelu** | x ≈ ±1.41 | Two symmetric inflection points |
| **mish** | x ≈ -0.6, x ≈ 0.5 | Two inflections, complex behavior |
| **swish** | x ≈ -0.3, x ≈ 1.5 | Similar to mish |
| **erf** | x = 0 | Gaussian-like, single inflection |
| **atanh** | x = 0 | Center of domain |
| **sinh** | x = 0 | Odd function, inflection at origin |
| **softsign** | x = 0 | Polynomial-like, single inflection |
| **tanhshrink** | x = 0 | Residual-like, inflection at origin |
| **elu** | x ≈ 0 | Near discontinuity, complex behavior |
| **celu** | x ≈ 0 | Similar to ELU |
| **selu** | x ≈ 0 | Scaled version of ELU |
| **cos** | x ≈ -π, 0, +π, +2π | Multiple due to periodicity |
| **sin** | x ≈ -π/2, +π/2, +3π/2 | Multiple due to periodicity |

**Recommendation**: Inflection points are natural candidates for breakpoints as they separate convex/concave regions.

---

### 4. Functions with Saturation Regions

Regions where |f'(x)| < 0.01 (nearly flat), requiring **sparse sampling**:

| Activation | Saturation Region(s) | Why Saturated |
|------------|----------------------|---------------|
| **sigmoid** | x < -4.5, x > +4.5 | Approaches 0 and 1 asymptotically |
| **tanh** | x < -3.0, x > +3.0 | Approaches ±1 asymptotically |
| **relu** | x < 0 | Constant zero region |
| **leaky_relu** | x < -2 (weak) | Linear but flat slope |
| **softplus** | x < -5 | Approaches zero |
| **exp** | x < -4.5 | Underflow to near-zero |
| **gelu** | x < -3 | Approaches zero linearly |
| **erf** | x < -3, x > +3 | Saturates to ±1 |
| **logsigmoid** | x < -10 (weak) | Approaches -∞ slowly |
| **hardsigmoid** | x < -3, x > +3 | Clipped to [0, 1] |
| **hardtanh** | x < -1, x > +1 | Clipped to [-1, 1] |
| **relu6** | x < 0, x > 6 | Clipped to [0, 6] |
| **threshold** | x < 0 | Constant value region |

**Recommendation**: Use **fewer segments** in saturated regions to save memory. For some functions (relu, hardtanh), these regions can be represented with a single constant segment.

---

## Adaptive Segmentation Strategy

### Current Approach (Uniform Segmentation)
```python
segment_width = (x_max - x_min) / num_segments
```
- **Problem**: Wastes segments in flat regions, under-samples high-curvature regions
- **Result**: Suboptimal error distribution

### Proposed Approach (Adaptive Segmentation)

The `analyze_breakpoints.py` tool suggests optimal breakpoints using:

1. **Curvature-based density**: More segments where |f''(x)| is large
2. **Discontinuity preservation**: Always place boundaries at derivative discontinuities
3. **Inflection point inclusion**: Use inflection points as natural boundaries
4. **Saturation awareness**: Fewer segments in flat regions

Example for **sigmoid** (16 segments):
```
Uniform width:  0.625 (constant)
Adaptive widths: min=0.280, max=2.022, std=0.546

Segments concentrated near x=0 (high curvature)
Segments sparse near x=±5 (saturation)
```

Example for **exp** (16 segments):
```
Uniform width:  0.625 (constant)
Adaptive widths: min=0.090, max=0.941, std=0.355

Segments concentrated near x=+5 (exponential growth)
Segments sparse near x=-5 (near-zero saturation)
```

---

## Impact Analysis

### Error Reduction Potential

Based on the analysis, adaptive segmentation could reduce error by:

| Function Type | Expected Error Reduction | Reason |
|---------------|-------------------------|---------|
| **Smooth (sigmoid, tanh)** | 2-3× | Better curvature sampling |
| **Exponential (exp, cosh)** | 5-10× | Dramatic improvement in high-growth regions |
| **Discontinuous (relu, leaky_relu)** | 10-100× | Eliminates artifacts from interpolating across discontinuities |
| **Piecewise (hardsigmoid, hardtanh)** | 2-5× | Aligns segments with natural boundaries |

### Memory Efficiency

Functions like **relu**, **threshold**, **hardtanh** could use:
- 1 segment for saturated region (constant)
- Remaining segments for active region
- **Result**: Effective doubling of resolution in active region

---

## Specific Function Recommendations

### High Priority for Adaptive Segmentation

1. **exp**: Most dramatic non-uniformity
   - Suggested: Logarithmically-spaced breakpoints
   - Alternative: Exponentially-increasing segment widths

2. **atanh**: Restricted domain with divergent behavior
   - Suggested: Dense sampling near x = ±0.8
   - Critical: Avoid sampling beyond valid domain

3. **relu/leaky_relu/prelu**: Derivative discontinuities
   - **Required**: Breakpoint exactly at x = 0
   - Current uniform approach likely misses this

4. **gelu**: Complex shape with multiple inflections
   - Suggested: 3 regions (left saturation, center transition, right linear)
   - Allocate 70% of segments to center region [-2, +2]

5. **mish**: Two distinct high-curvature regions
   - Suggested: Dense sampling around x ≈ -0.3 and x ≈ 1.5

### Medium Priority

6. **sigmoid/tanh**: Well-behaved but benefit from center concentration
7. **swish**: Similar to mish but less dramatic
8. **softplus**: Smooth ReLU transition benefits from adaptive sampling
9. **cos/sin**: Periodic, benefit from inflection-based segmentation

### Low Priority (Already Well-Approximated)

10. **hardsigmoid/hardtanh**: Piecewise linear, low error even with uniform sampling
11. **softsign**: Polynomial, relatively easy to approximate

---

## Implementation Roadmap

### Phase 1: Analysis Tool (✓ Complete)
- `analyze_breakpoints.py` identifies discontinuities, inflections, curvature

### Phase 2: Adaptive LUT Generation (Proposed)
1. Modify `generate_piecewise_*.py` scripts to accept adaptive breakpoints
2. Add `--adaptive` flag to use suggested breakpoints from analysis
3. Compare error metrics: uniform vs adaptive

### Phase 3: Visualization (Proposed)
1. Generate plots showing:
   - Function, derivative, curvature
   - Uniform vs adaptive segment placement
   - Error distribution across domain

### Phase 4: Hardware Integration (Proposed)
1. Modify C++ LUT lookup to handle non-uniform segments
2. Store segment boundaries in addition to coefficients
3. Use binary search for segment lookup instead of division

---

## Example: Adaptive Breakpoints for Common Functions

### ReLU (16 segments, domain [-2, 5])
```python
# Uniform (current):
# [-2.0, -1.56, -1.12, -0.68, -0.24, 0.19, 0.63, 1.07, 1.51, 1.95, 2.39, 2.83, 3.27, 3.71, 4.15, 4.59, 5.0]

# Adaptive (proposed):
# [-2.0, -1.9, -0.1, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.5, 4.0, 5.0]
#  ^^^^^^^^^^^^^^ single segment for flat region
#                  ^^^^^ discontinuity boundary
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ uniform in active region
```

### Sigmoid (16 segments, domain [-5, 5])
```python
# Uniform (current):
# Each segment is 0.625 wide

# Adaptive (proposed):
# Dense near x=0 (high curvature): segments ~0.3 wide
# Sparse near x=±5 (saturation): segments ~2.0 wide
# Total: 16 segments with variable widths
```

### Exp (16 segments, domain [-5, 5])
```python
# Uniform (current):
# Each segment is 0.625 wide

# Adaptive (proposed):
# Very sparse near x=-5 (near-zero): segments ~1.0 wide
# Very dense near x=+5 (exponential growth): segments ~0.1 wide
# Progressive refinement from left to right
```

---

## Usage of Analysis Tool

Generate breakpoint analysis for all activations:
```bash
python3 analyze_breakpoints.py
```

Analyze specific activation with plots:
```bash
python3 analyze_breakpoints.py --activation relu --plot
```

Output:
- Console: Discontinuities, inflections, saturation regions, suggested breakpoints
- Plots: `plots_breakpoint_analysis/{activation}_breakpoint_analysis.png`

---

## Conclusion

The current uniform segmentation strategy is suboptimal for most activation functions. Key opportunities:

1. **Discontinuous functions** (relu, leaky_relu, prelu): Place breakpoint at discontinuity
2. **Exponential functions** (exp, cosh): Use exponentially-increasing segment widths
3. **Saturating functions** (sigmoid, tanh): Concentrate segments near inflection point
4. **Complex functions** (mish, gelu): Multi-region adaptive strategy

**Estimated overall impact**: 2-5× error reduction for same memory footprint, or 2-4× memory reduction for same error level.

---

## Next Steps

1. Review generated plots in `plots_breakpoint_analysis/` directory
2. Select 3-5 representative functions for adaptive segmentation prototype
3. Implement adaptive breakpoint support in LUT generation scripts
4. Benchmark error reduction on hardware
5. If successful, roll out to all 29 activation functions
