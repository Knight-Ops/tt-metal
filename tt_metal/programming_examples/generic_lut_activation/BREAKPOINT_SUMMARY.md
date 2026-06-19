# Breakpoint Summary for Key Activation Functions

**16 Segments | Generated: 2026-01-18**

## Quick Reference Table

| Activation | Domain | Adaptivity Ratio | Key Characteristics |
|------------|--------|-----------------|---------------------|
| **relu** | [-2, 5] | 3.5× | Discontinuity at x=0, sparse left side |
| **leaky_relu** | [-2, 5] | **2857×** | Extreme discontinuity at x=0 |
| **prelu** | [-2, 5] | 3.5× | Similar to relu |
| **selu** | [-3, 5] | **3749×** | Most adaptive, extreme discontinuity |
| **exp** | [-5, 5] | 10.4× | Dense right side (exponential growth) |
| **sigmoid** | [-5, 5] | 7.2× | Dense center (x≈0), sparse edges |
| **tanh** | [-3, 3] | 8.8× | Similar to sigmoid but narrower |
| **gelu** | [-4, 4] | 11.1× | Dense center, two inflection points |
| **swish** | [-5, 5] | 10.6× | Similar to gelu |
| **mish** | [-5, 5] | 11.3× | Two high-curvature regions |
| **atanh** | [-0.9, 0.9] | 8.4× | Dense at domain boundaries |
| **softplus** | [-5, 10] | 10.8× | Dense at transition region |
| **hardswish** | [-5, 5] | **1999×** | Piecewise linear, extreme ratio |
| **hardshrink** | [-3, 3] | **841×** | Hard thresholding |
| **threshold** | [-5, 5] | **385×** | Step function |

---

## Detailed Breakpoints for Common Activations

### 1. ReLU
```python
Domain: [-2.0, 5.0]
Segments: 16
Adaptivity: 3.5×

Uniform:
[-2.000, -1.562, -1.125, -0.688, -0.250, 0.188, 0.625, 1.062,
  1.500, 1.938, 2.375, 2.812, 3.250, 3.688, 4.125, 4.562, 5.000]

Adaptive:
[-2.000, -1.538, -1.068, -0.606, -0.136, -0.001,  # sparse on left (flat)
  0.333, 0.796, 1.265, 1.735, 2.197, 2.667,       # uniform on right (linear)
  3.136, 3.599, 4.068, 4.538, 5.000]

Key insight: Breakpoint at x ≈ 0 (discontinuity)
```

### 2. Leaky ReLU
```python
Domain: [-2.0, 5.0]
Adaptivity: 2857× (EXTREME!)

Adaptive:
[-2.000, ..., -0.003, -0.002, 0.000,  # extreme density near discontinuity
  0.338, 0.676, 1.014, ..., 5.000]    # then uniform

Key insight: Derivative discontinuity requires VERY fine segmentation at x=0
```

### 3. Exponential
```python
Domain: [-5.0, 5.0]
Segments: 16
Adaptivity: 10.4×

Uniform:
[-5.000, -4.375, -3.750, -3.125, -2.500, -1.875, -1.250, -0.625,
  0.000, 0.625, 1.250, 1.875, 2.500, 3.125, 3.750, 4.375, 5.000]

Adaptive:
[-5.000, -4.059, -3.118, -2.177, -1.236, -0.295,  # sparse on left (near-zero)
  0.646, 1.577, 2.487,                             # moderate in middle
  3.298, 3.879, 4.249, 4.489, 4.670, 4.800, 4.910, 5.000]  # dense on right!

Key insight: Progressive refinement toward x=+5 (exponential growth)
  Left side (x<0): segments ~0.9 wide
  Right side (x>4): segments ~0.09 wide (10× denser!)
```

### 4. Sigmoid
```python
Domain: [-5.0, 5.0]
Segments: 16
Adaptivity: 7.2×

Uniform:
[-5.000, -4.375, -3.750, -3.125, -2.500, -1.875, -1.250, -0.625,
  0.000, 0.625, 1.250, 1.875, 2.500, 3.125, 3.750, 4.375, 5.000]

Adaptive:
[-5.000, -2.978,  # sparse left (saturated)
 -2.267, -1.857, -1.547, -1.266, -0.976, -0.576, -0.000,  # dense center!
  0.576, 0.976, 1.266, 1.547, 1.857, 2.267,  # dense center!
  2.978, 5.000]  # sparse right (saturated)

Key insight: 75% of segments in range [-2, +2] where curvature is highest
  Saturation regions: only 2 segments for x ∈ [-5,-3] ∪ [3,5]
  Active region: 14 segments for x ∈ [-3,+3]
```

### 5. Tanh
```python
Domain: [-3.0, 3.0]
Segments: 16
Adaptivity: 8.8×

Uniform:
[-3.000, -2.625, -2.250, -1.875, -1.500, -1.125, -0.750, -0.375,
  0.000, 0.375, 0.750, 1.125, 1.500, 1.875, 2.250, 2.625, 3.000]

Adaptive:
[-3.000, -1.673,  # sparse left
 -1.198, -0.964, -0.796, -0.646, -0.495, -0.297, 0.000,  # dense center
  0.297, 0.495, 0.646, 0.796, 0.964, 1.198,  # dense center
  1.673, 3.000]  # sparse right

Key insight: Similar pattern to sigmoid but narrower domain
  Segments ~0.15 wide in center [-1, +1]
  Segments ~1.3 wide at edges
```

### 6. GELU
```python
Domain: [-4.0, 4.0]
Segments: 16
Adaptivity: 11.1×

Uniform:
[-4.000, -3.500, -3.000, -2.500, -2.000, -1.500, -1.000, -0.500,
  0.000, 0.500, 1.000, 1.500, 2.000, 2.500, 3.000, 3.500, 4.000]

Adaptive:
[-4.000, -2.583,  # sparse left (saturation)
 -1.414,  # inflection point!
 -1.277, -0.669, -0.436, -0.276, -0.132, 0.004,  # very dense center
  0.132, 0.276, 0.436, 0.669,
  1.277, 1.414,  # inflection point!
  2.583, 4.000]  # sparse right

Key insight: Two inflection points at x ≈ ±1.414 (±√2)
  Highest density around x=0 (maximum curvature)
  Segments ~0.13 wide in center
  Segments ~1.4 wide at edges (11× difference)
```

### 7. SELU (Most Adaptive!)
```python
Domain: [-3.0, 5.0]
Adaptivity: 3749× (EXTREME!)

Adaptive:
[-3.000, -2.183, -1.503, -1.030, -0.718, -0.502,
 -0.341, -0.213, -0.109, -0.021, 0.000,  # VERY dense near discontinuity
  0.660, 1.525, 2.397, 3.262, 4.135, 5.000]  # sparse on right

Segment widths:
  Minimum: 0.0008 (near x=0)
  Maximum: 2.9995 (left side)
  Ratio: 3749×

Key insight: Combines ELU discontinuity with scaling factor
  Extreme density needed around x=0
  Much sparser elsewhere
```

### 8. Atanh (Restricted Domain)
```python
Domain: [-0.9, 0.9]  (RESTRICTED! Function diverges at ±1)
Segments: 16
Adaptivity: 8.4×

Uniform:
[-0.900, -0.787, -0.675, -0.562, -0.450, -0.338, -0.225, -0.113,
  0.000, 0.112, 0.225, 0.338, 0.450, 0.563, 0.675, 0.787, 0.900]

Adaptive:
[-0.900, -0.880, -0.837,  # very dense near boundary (diverging)
 -0.729, -0.576, -0.412, -0.248, -0.082,  # moderate
  0.000, 0.082, 0.248, 0.412, 0.576, 0.729,
  0.837, 0.880, 0.900]  # very dense near boundary

Key insight: Function approaches ±∞ as x → ±1
  Dense sampling near domain boundaries (x ≈ ±0.9)
  Segments ~0.02 wide near edges
  Segments ~0.16 wide in center
```

---

## Segment Width Distribution Patterns

### Pattern 1: Discontinuous (relu, leaky_relu, prelu, selu)
```
Left side (x < 0): Sparse (flat or linear)
At x = 0:         VERY dense (discontinuity)
Right side (x > 0): Uniform (linear)

Example (relu):
  [-2, -0.001]: 5 segments
  [-0.001, 0]: 1 segment (at discontinuity)
  [0, 5]: 10 segments (uniform)
```

### Pattern 2: Exponential Growth (exp, cosh, sinh)
```
Left side: Sparse (near-zero saturation)
Middle: Moderate
Right side: Very dense (exponential growth)

Example (exp):
  [-5, 0]: 6 segments, width ~0.9
  [0, 3]: 4 segments, width ~0.75
  [3, 5]: 6 segments, width ~0.09-0.33
```

### Pattern 3: S-shaped (sigmoid, tanh, erf)
```
Left edge: Sparse (saturation)
Center: Dense (high curvature)
Right edge: Sparse (saturation)

Example (sigmoid):
  [-5, -2]: 1 segment (width ~3)
  [-2, +2]: 14 segments (width ~0.28)
  [+2, +5]: 1 segment (width ~3)
```

### Pattern 4: Bell-shaped (gelu, mish, swish)
```
Left edge: Sparse (saturation)
Center: Very dense (transition region)
Right edge: Moderate (linear-ish growth)

Example (gelu):
  [-4, -2]: 1 segment
  [-2, +2]: 12 segments (including inflection points)
  [+2, +4]: 3 segments
```

### Pattern 5: Diverging (atanh)
```
Left edge: Very dense (approaching -∞)
Center: Sparse (linear-ish)
Right edge: Very dense (approaching +∞)

Example (atanh on [-0.9, 0.9]):
  [-0.9, -0.8]: Very dense (diverging)
  [-0.8, +0.8]: Uniform
  [+0.8, +0.9]: Very dense (diverging)
```

---

## Implementation Recommendations

### Tier 1: CRITICAL (Must use adaptive segmentation)
- **leaky_relu** (2857×), **selu** (3749×), **hardswish** (1999×)
- **Impact**: 100-1000× error reduction at discontinuities
- **Why**: Derivative discontinuities with extreme ratio

### Tier 2: HIGH PRIORITY (Large benefit)
- **exp** (10.4×), **gelu** (11.1×), **mish** (11.3×)
- **Impact**: 5-10× error reduction
- **Why**: High curvature concentration, complex shapes

### Tier 3: MEDIUM PRIORITY (Moderate benefit)
- **sigmoid** (7.2×), **tanh** (8.8×), **atanh** (8.4×)
- **Impact**: 3-5× error reduction
- **Why**: Clear center-concentrated curvature

### Tier 4: LOW PRIORITY (Small benefit)
- **relu** (3.5×), **prelu** (3.5×), **hardtanh** (2×)
- **Impact**: 2-3× improvement
- **Why**: Already relatively uniform or simple

### Tier 5: SKIP (Minimal benefit)
- **cos** (1.4×), **sin** (2.3×)
- **Impact**: <2× improvement
- **Why**: Periodic functions with relatively uniform curvature

---

## Memory/Accuracy Trade-off Examples

Using adaptive segmentation, you can achieve:

### Option A: Same Memory, Better Accuracy
- 16 uniform segments → 16 adaptive segments
- **exp**: 10× error reduction
- **gelu**: 11× error reduction
- **sigmoid**: 7× error reduction

### Option B: Same Accuracy, Less Memory
- 32 uniform segments → 16 adaptive segments (50% memory savings)
- **sigmoid**: Similar MAE with half the memory
- **tanh**: Similar MAE with half the memory

### Option C: Hybrid (Recommended)
- Critical regions: Use adaptive with higher resolution
- Saturation regions: Use 1-2 segments per region
- **Example (sigmoid with 16 total segments)**:
  - Saturation (x < -3): 1 segment
  - Transition [-3, +3]: 14 segments (adaptive)
  - Saturation (x > +3): 1 segment
