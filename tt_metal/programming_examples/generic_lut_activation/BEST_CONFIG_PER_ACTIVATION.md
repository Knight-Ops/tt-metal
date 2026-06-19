# Best LUT Configuration Per Activation

**Date**: 2026-01-18
**Summary**: Optimal polynomial type, depth, and mode (uniform/adaptive) for minimum error

---

## Quick Reference Table

| Activation | Mode | Polynomial | Degree | Depth | Best MAE | Quality |
|------------|------|------------|--------|-------|----------|---------|
| **atanh** | Uniform | cubic_remez | 3 | 32 | 2.43e-07 | Very good |
| **celu** | Uniform | octic_remez | 8 | 8 | 2.73e-08 | Excellent |
| **cos** | Uniform | octic_remez | 8 | 8 | 8.68e-08 | Excellent |
| **cosh** | Uniform | octic_remez | 8 | 4 | 2.28e-05 | Good |
| **elu** | Uniform | octic_remez | 8 | 8 | 2.73e-08 | Excellent |
| **erf** | Uniform | hexic_remez | 6 | 8 | 3.81e-07 | Very good |
| **exp** | Uniform | octic_remez | 8 | 4 | 4.65e-04 | Good |
| **gelu** | Adaptive | hexic_remez | 6 | 16 | 2.56e-07 | Very good |
| **hardshrink** | Uniform | cubic_remez | 3 | 32 | 4.25e-03 | Fair |
| **hardsigmoid** | Adaptive | linear | 1 | 4 | 5.46e-09 | Excellent |
| **hardswish** | Adaptive | octic_remez | 8 | 8 | 1.97e-08 | Excellent |
| **hardtanh** | Uniform | hexic_remez | 6 | 32 | 3.22e-04 | Good |
| **leaky_relu** | Adaptive | quadratic_remez | 2 | 4 | **0.00e+00** | **PERFECT** ✓ |
| **logsigmoid** | Uniform | hexic_remez | 6 | 32 | 1.76e-07 | Very good |
| **mish** | Adaptive | hexic_remez | 6 | 32 | 2.02e-07 | Very good |
| **prelu** | Adaptive | linear | 1 | 4 | **0.00e+00** | **PERFECT** ✓ |
| **relu** | Adaptive | linear | 1 | 4 | **0.00e+00** | **PERFECT** ✓ |
| **relu6** | Adaptive | linear | 1 | 4 | **0.00e+00** | **PERFECT** ✓ |
| **selu** | Adaptive | octic_remez | 8 | 4 | 7.34e-08 | Excellent |
| **sigmoid** | Adaptive | hexic_remez | 6 | 32 | 5.68e-08 | Excellent |
| **sin** | Adaptive | octic_remez | 8 | 8 | 7.00e-08 | Excellent |
| **sinh** | Uniform | octic_remez | 8 | 4 | 2.40e-05 | Good |
| **softplus** | Uniform | hexic_remez | 6 | 16 | 1.75e-07 | Very good |
| **softshrink** | Uniform | octic_remez | 8 | 32 | 1.98e-04 | Good |
| **softsign** | Adaptive | hexic_remez | 6 | 32 | 8.73e-08 | Excellent |
| **swish** | Uniform | hexic_remez | 6 | 32 | 2.05e-07 | Very good |
| **tanh** | Uniform | hexic_remez | 6 | 16 | 2.57e-07 | Very good |
| **tanhshrink** | Uniform | hexic_remez | 6 | 32 | 1.72e-07 | Very good |
| **threshold** | Uniform | linear | 1 | 16 | **0.00e+00** | **PERFECT** ✓ |

---

## Summary Statistics

### Mode Distribution
- **Uniform best**: 17/29 (58.6%)
- **Adaptive best**: 12/29 (41.4%)

### Polynomial Type Distribution
1. **Hexic (deg 6)**: 11/29 (37.9%) 🏆 Most versatile
2. **Octic (deg 8)**: 10/29 (34.5%) 🥈 Best for complex functions
3. **Linear (deg 1)**: 5/29 (17.2%) - Perfect for piecewise
4. **Cubic (deg 3)**: 2/29 (6.9%)
5. **Quadratic (deg 2)**: 1/29 (3.4%)

### Depth Distribution
- **Depth 32**: 10/29 (34.5%) - Best for high accuracy
- **Depth 4**: 9/29 (31.0%) - Best for adaptive piecewise
- **Depth 8**: 6/29 (20.7%)
- **Depth 16**: 4/29 (13.8%)

---

## Recommendations by Activation Category

### 1. Piecewise/Discontinuous Activations
**Use Adaptive for PERFECT approximation!**

| Activation | Config | MAE |
|------------|--------|-----|
| relu | Adaptive linear d=4 | 0.00e+00 ✓ |
| relu6 | Adaptive linear d=4 | 0.00e+00 ✓ |
| leaky_relu | Adaptive quadratic_remez d=4 | 0.00e+00 ✓ |
| prelu | Adaptive linear d=4 | 0.00e+00 ✓ |
| hardsigmoid | Adaptive linear d=4 | 5.46e-09 ✓ |
| hardswish | Adaptive octic_remez d=8 | 1.97e-08 ✓ |

**Key Insight**: Adaptive segmentation at low depth (4) achieves perfect approximation for ReLU variants!

### 2. Smooth Activations
**Use Uniform hexic/octic for excellent accuracy**

| Activation | Config | MAE |
|------------|--------|-----|
| sigmoid | Adaptive hexic_remez d=32 | 5.68e-08 |
| tanh | Uniform hexic_remez d=16 | 2.57e-07 |
| swish | Uniform hexic_remez d=32 | 2.05e-07 |
| mish | Adaptive hexic_remez d=32 | 2.02e-07 |
| gelu | Adaptive hexic_remez d=16 | 2.56e-07 |
| softsign | Adaptive hexic_remez d=32 | 8.73e-08 |
| erf | Uniform hexic_remez d=8 | 3.81e-07 |

**Key Insight**: Hexic polynomials excel at smooth activations. Adaptive provides slight edge for some.

### 3. Exponential/Growth Activations
**Use Uniform octic at low depth or hexic at high depth**

| Activation | Config | MAE |
|------------|--------|-----|
| exp | Uniform octic_remez d=4 | 4.65e-04 |
| cosh | Uniform octic_remez d=4 | 2.28e-05 |
| sinh | Uniform octic_remez d=4 | 2.40e-05 |
| softplus | Uniform hexic_remez d=16 | 1.75e-07 |
| logsigmoid | Uniform hexic_remez d=32 | 1.76e-07 |
| atanh | Uniform cubic_remez d=32 | 2.43e-07 |
| selu | Adaptive octic_remez d=4 | 7.34e-08 |

**Key Insight**: Octic at depth=4 handles exponential growth. SELU benefits from adaptive.

### 4. Periodic Activations
**Use Uniform/Adaptive octic at low depth**

| Activation | Config | MAE |
|------------|--------|-----|
| sin | Adaptive octic_remez d=8 | 7.00e-08 |
| cos | Uniform octic_remez d=8 | 8.68e-08 |

**Key Insight**: Octic polynomials perfect for periodic functions at moderate depth.

---

## Production Deployment Strategy

### Tier 1: Perfect Approximation (MAE = 0)
Deploy these exact configurations for zero error:
```
relu, relu6, leaky_relu, prelu, threshold
→ Use adaptive linear/quadratic at depth=4/16
→ Storage: 5 activations × 1 config = 5 LUTs
```

### Tier 2: Excellent (MAE < 1e-7)
Deploy these for near-perfect accuracy:
```
celu, elu, cos, sin, sigmoid, softsign, hardsigmoid, hardswish, selu
→ Use hexic/octic at optimal depth per table above
→ Storage: 9 activations × 1 config = 9 LUTs
```

### Tier 3: Very Good (MAE < 1e-6)
Deploy these for production-quality accuracy:
```
gelu, tanh, swish, mish, erf, logsigmoid, softplus, atanh, tanhshrink
→ Use hexic/octic at depth=16-32
→ Storage: 9 activations × 1 config = 9 LUTs
```

### Tier 4: Good/Fair (MAE < 1e-2)
May need higher depth or different approach:
```
exp, cosh, sinh, softplus, hardtanh, hardshrink, softshrink
→ Consider depth=32 or alternative methods
→ Storage: 6 activations × 1 config = 6 LUTs
```

**Total Storage for Production**: 29 LUTs (vs 1,160 if we include all configurations)

---

## Key Findings

1. **Adaptive shines for piecewise functions**: 4/5 perfect approximations use adaptive
2. **Hexic is the all-rounder**: Best for 38% of activations, excellent for smooth functions
3. **Octic excels at complexity**: Best for exponential growth and periodic functions
4. **Depth=4 + adaptive = perfect for ReLU**: Zero error for all ReLU variants!
5. **Depth=32 for smooth functions**: Hexic/octic at depth=32 provides excellent accuracy for smooth activations

---

## Files

- **CSV**: `plots_adaptive_comprehensive/best_config_per_activation.csv`
- **Full Comparison**: `plots_adaptive_comprehensive/comparison_summary.csv` (580 configurations)
- **Plots**: `plots_adaptive_comprehensive/*.png` (29 comprehensive plots)

---

**Generated**: 2026-01-18
**Analysis**: Based on comprehensive comparison of 580 configurations (29 activations × 5 polynomial types × 4 depths × 2 modes)
