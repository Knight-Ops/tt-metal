# PR Submission Guide for Winning LUT-Based Activations

This document provides step-by-step instructions for submitting PRs that replace TTNN's existing SFPU-based activation implementations with superior LUT-based piecewise polynomial or rational approximations.

## Overview

When a LUT-based activation function demonstrates better ULP error and/or runtime performance than TTNN's current implementation, it becomes a "winning activation" candidate for inclusion in the main codebase.

---

## Prerequisites

Before starting a PR, ensure you have:

1. **Benchmark data** in `data/<arch>/best/<activation>.csv` showing the winning configuration
2. **ULP distribution plots** in `plots/<arch>/hardware_error_analysis/<activation>/`
3. **Error vs runtime plots** in `plots/<arch>/error_vs_runtime/<activation>/` (if available)
4. **Coefficients** either embedded or in CSV format

---

## Step 1: File a GitHub Issue

**Required before opening any PR.**

1. Go to https://github.com/tenstorrent/tt-metal/issues
2. Create a new issue with title: `[TTNN] Improve <activation> accuracy/performance with LUT-based approximation`
3. Include in the body:
   - Current TTNN implementation metrics (ULP, runtime)
   - Proposed LUT implementation metrics (ULP, runtime)
   - Hardware target (Blackhole, Wormhole)
   - Precision modes (BF16, FP32, or both)
   - Link to benchmark plots

**Example issue body:**
```markdown
## Summary
Replace TTNN's current GELU implementation with a LUT-based piecewise polynomial approximation.

## Current vs Proposed Metrics (Blackhole, BF16)
| Metric | Current TTNN | Proposed LUT | Improvement |
|--------|--------------|--------------|-------------|
| Max ULP | 4320411 | 128 | 33,753x |
| Mean ULP | ~1000 | ~5 | 200x |
| Runtime (256 tiles) | 38.2 us | 37.9 us | ~1% faster |

## Configuration
- Polynomial degree: 3
- Segments: 64
- Segmentation: uniform

## Evidence
See attached ULP distribution plots and error vs runtime Pareto analysis.
```

---

## Step 2: Understand the TTNN Eltwise API Structure

### Key Files to Modify

| File | Purpose |
|------|---------|
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | Add new UnaryOpType if needed |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Map operation to SFPU macro/init/call |
| `tt_metal/hw/inc/api/compute/eltwise_unary/<op>.h` | Public SFPU API (tile_init, tile) |
| `tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/ckernel_sfpu_<op>.h` | SFPU implementation |

### Existing Pattern (Example: GELU)

**API Header** (`tt_metal/hw/inc/api/compute/eltwise_unary/gelu.h`):
```cpp
template <bool fast_and_approx = true>
ALWI void gelu_tile_init() {
    MATH(SFPU_INIT_KERNEL_CALL(gelu, sfpu::gelu_init, fast_and_approx));
}

template <bool fast_and_approx = true>
ALWI void gelu_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_gelu, RC, fast_and_approx, idst));
}
```

**SFPU Implementation** (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h`):
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    // Polynomial evaluation using Horner's method
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        // ... computation ...
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
```

---

## Step 2.5: Drop-in Replacement Strategy (IMPORTANT)

**True drop-in replacement**: Change ONLY the SFPU implementation, keep all APIs identical.

### What Changes

Only ONE file per architecture:
```
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_<activation>.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_<activation>.h
```

### What Stays the Same

- Public API header (`tt_metal/hw/inc/api/compute/eltwise_unary/<activation>.h`)
- Dispatch code (`unary_op_utils.cpp`)
- Function signatures (`calculate_gelu<APPROXIMATION_MODE, ITERATIONS>()`)
- Customer code using `ttnn.<activation>()`

### Template Signatures Vary Per Activation (IMPORTANT)

> **WARNING: Non-Standard Signatures Are a Nightmare**
>
> The SFPU activation signatures are NOT standardized. Each activation has evolved
> independently with different template parameters, runtime parameters, and conventions.
> This makes automated tooling difficult and requires manual inspection of each
> activation before replacement. There's no good reason for this inconsistency -
> it's technical debt that should have been addressed when these were first written.

**You must match the exact existing signature.** Common patterns:

| Activation | Template Signature |
|------------|-------------------|
| GELU | `<bool APPROXIMATION_MODE, int ITERATIONS = 8>` |
| Sigmoid | `<bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>` |
| Tanh | `<bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>` |
| ELU | `<bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>` |
| Softplus | `<bool APPROXIMATION_MODE, int ITERATIONS = 8>` + `uint param0, param1, param2` |

**Before replacing, check the existing signature in:**
```
tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/ckernel_sfpu_<activation>.h
```

### Example: GELU Drop-in Replacement with Dtype-Specific Coefficients

**File:** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h`

```cpp
// REPLACE the entire calculate_gelu() implementation
// KEEP the exact same function signature!

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    // ===== Dtype-specific LUT coefficients (Option B) =====
#ifdef INP_FLOAT32
    // FP32 coefficients - may need higher degree/segments for precision
    constexpr uint32_t NUM_SEGMENTS = 64;
    constexpr uint32_t POLY_DEGREE = 4;
    constexpr float LUT_DATA[] = {
        // Boundaries (65 values)
        -10.0f, -9.6875f, /* ... */, 10.0f,
        // 64 segments × 5 coefficients (degree 4)
        /* ... FP32-optimized coefficients ... */
    };
#else
    // BF16 coefficients - can often use simpler approximation
    constexpr uint32_t NUM_SEGMENTS = 32;
    constexpr uint32_t POLY_DEGREE = 3;
    constexpr float LUT_DATA[] = {
        // Boundaries (33 values)
        -10.0f, -9.375f, /* ... */, 10.0f,
        // 32 segments × 4 coefficients (degree 3)
        /* ... BF16-optimized coefficients ... */
    };
#endif
    constexpr float LO = LUT_DATA[0];
    constexpr float HI = LUT_DATA[NUM_SEGMENTS];
    constexpr float SEG_WIDTH = (HI - LO) / NUM_SEGMENTS;
    // ======================================================

    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Segment index (clamped to valid range)
        sfpi::vFloat seg_f = (x - LO) / SEG_WIDTH;
        int seg_idx = static_cast<int>(seg_f);
        if (seg_idx < 0) seg_idx = 0;
        if (seg_idx >= NUM_SEGMENTS) seg_idx = NUM_SEGMENTS - 1;

        // Coefficient base pointer
        const float* c = &LUT_DATA[NUM_SEGMENTS + 1 + seg_idx * (POLY_DEGREE + 1)];

        // Horner evaluation: c[0] + x*(c[1] + x*(c[2] + ...))
        sfpi::vFloat result = c[POLY_DEGREE];
        for (int i = POLY_DEGREE - 1; i >= 0; i--) {
            result = result * x + c[i];
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// DELETE: Remove old calculate_gelu_chebyshev() helper function
// DELETE: Remove old POLYVAL15 macro if no longer used
```

### Gradual Rollout Option

Use the existing `APPROXIMATION_MODE` template parameter:

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        // Keep old fast approximation for fast_and_approx=True
        _calculate_gelu_<APPROXIMATION_MODE, ITERATIONS>();
    } else {
        // New LUT implementation for fast_and_approx=False (accurate mode)
        // ... LUT code with #ifdef INP_FLOAT32 here ...
    }
}
```

---

## Step 3: BF16/FP32 Handling via #ifdef (Compile-Time Selection)

**TTNN uses compile-time dtype selection via preprocessor defines.**

### How It Works

1. **Preprocessor defines are set** based on `input.dtype()`:
```cpp
// unary_program_factory.cpp
if (input.dtype() == DataType::FLOAT32) {
    unary_defines["INP_FLOAT32"] = "1";
} else {
    unary_defines["INP_FLOAT"] = "1";  // BF16
}
```

2. **Different kernels are compiled** for each dtype
3. **Kernel caching** stores compiled binaries separately

### Use #ifdef for Dtype-Specific Coefficients

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
#ifdef INP_FLOAT32
    // FP32-optimized: may need more segments or higher degree for precision
    constexpr uint32_t NUM_SEGMENTS = 64;
    constexpr uint32_t POLY_DEGREE = 4;
    constexpr float LUT_DATA[] = { /* FP32-optimized coefficients */ };
#else
    // BF16-optimized: can often use fewer segments (BF16 has less precision anyway)
    constexpr uint32_t NUM_SEGMENTS = 32;
    constexpr uint32_t POLY_DEGREE = 3;
    constexpr float LUT_DATA[] = { /* BF16-optimized coefficients */ };
#endif

    // Same evaluation logic for both - only coefficients differ
    constexpr float LO = LUT_DATA[0];
    constexpr float HI = LUT_DATA[NUM_SEGMENTS];
    constexpr float SEG_WIDTH = (HI - LO) / NUM_SEGMENTS;

    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // ... Horner evaluation (same code) ...
    }
}
```

**Benefits:**
- Both coefficient sets in same source file
- Only one compiled per kernel binary (based on dtype)
- No runtime overhead
- Can tune segments/degree independently for each precision
- BF16 can use simpler approximation (fewer segments) since it has less precision anyway

---

## Step 3.5: Parametrized Activations and LUT Implications

Some activations take runtime parameters that change the function shape. This complicates LUT-based replacement.

### Parametrized Activations

| Activation | Parameters | Example Values | LUT Strategy |
|------------|------------|----------------|--------------|
| ELU | `slope` (alpha) | 1.0 (default) | LUT only for alpha=1.0 |
| CELU | `alpha` | 1.0 (default) | LUT only for alpha=1.0 |
| SELU | `scale, alpha` | **Fixed**: 1.0507, 1.6733 | Single LUT (params are constants) |
| Leaky ReLU | `slope` | 0.01 (default) | LUT only for slope=0.01 |
| PReLU | `slope` | varies | Skip (per-channel slopes) |
| Softplus | `beta, threshold` | 1.0, 20.0 (default) | LUT only for defaults |
| Softshrink | `lambda` | 0.5 (default) | LUT only for lambda=0.5 |
| XIElu | `alpha_p, alpha_n` | varies | Skip or target common values |
| Hardtanh | `min, max` | -1.0, 1.0 (default) | **Skip** - simple clamp, no LUT needed |

### Recommended Approach

1. **Target only default parameter values** - Most users use defaults
2. **Keep original implementation as fallback** for non-default parameters:

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_elu(uint slope) {
    // Check if slope matches our LUT (alpha = 1.0)
    constexpr uint32_t ALPHA_1_0_BITS = 0x3F800000;  // 1.0f as uint32

    if (slope == ALPHA_1_0_BITS) {
        // Use optimized LUT implementation
#ifdef INP_FLOAT32
        constexpr float LUT_DATA[] = { /* FP32 coeffs for alpha=1.0 */ };
#else
        constexpr float LUT_DATA[] = { /* BF16 coeffs for alpha=1.0 */ };
#endif
        // ... LUT evaluation ...
    } else {
        // Fall back to original implementation for other alpha values
        _calculate_elu_original_<APPROXIMATION_MODE>(ITERATIONS, slope);
    }
}
```

3. **SELU is special** - Parameters are mathematical constants, single LUT works

### Combination Count

- **Non-parametrized** (GELU, Sigmoid, Tanh, etc.): 2 LUTs each (BF16 + FP32)
- **Parametrized with defaults**: 2 LUTs each (BF16 + FP32) for default params only
- **Parametrized with varied usage**: Skip or use fallback pattern

**Practical target:** ~20-30 activations × 2 dtypes = **40-60 total LUT variants**

---

## Step 3.7: SEPARABLE_LUTS - Parametric LUT Decomposition (ADVANCED)

> **EXPERIMENTAL: Requires careful error/accuracy analysis before production use.**
>
> This approach allows ONE LUT to work for ALL parameter values of a parametrized activation.
> Existing coefficients are FINE - no refitting needed. This is only needed for final PR assembly.

### Theory: Separable Structure

A parametrized activation f(x, p) is **separable** if it can be written as:
```
f(x, p) = h(p, g(x))
```
Where:
- `g(x)` = base function (parameter-independent, fitted by LUT)
- `h(p, ·)` = simple runtime operation (multiply, divide, scale input/output)

### Separability Analysis Table

| Activation | Formula | Base LUT g(x) | Runtime h(p, g) | Separable? | Needs LUT? |
|------------|---------|---------------|-----------------|------------|------------|
| **ELU** | `x>0 ? x : α(eˣ-1)` | `exp(x)-1` | `α * g(x)` | ✅ Yes | ✅ Yes |
| **CELU** | `x>0 ? x : α(e^(x/α)-1)` | `exp(u)-1` | `u=x/α, α*g(u)` | ✅ Yes | ✅ Yes |
| **SELU** | `scale * (x>0 ? x : α(eˣ-1))` | `exp(x)-1` | Fixed constants | ✅ N/A | ✅ Yes |
| **Softplus** | `(1/β)log(1+e^(βx))` | `log(1+eᵘ)` | `u=βx, g(u)/β` | ✅ Yes | ✅ Yes |
| **Leaky ReLU** | `x>0 ? x : slope*x` | None | `slope * x` | ✅ Trivial | ❌ Linear |
| **PReLU** | `x>0 ? x : slope*x` | None | `slope * x` | ✅ Trivial | ❌ Linear |
| **Hardtanh** | `clamp(x, min, max)` | None | Simple clamp | ✅ Trivial | ❌ Linear |
| **Softshrink** | `x-λ if x>λ, x+λ if x<-λ` | None | — | ❌ Boundary | ❌ Linear |
| **Hardshrink** | `0 if |x|<λ, else x` | None | — | ❌ Boundary | ❌ Linear |
| **Threshold** | `x if x>t, else v` | None | — | ❌ Boundary | ❌ Linear |
| **XIElu** | Complex | TBD | TBD | ❓ Analyze | ❓ TBD |

### Why Boundary-Dependent Activations Don't Matter

**Softshrink, Hardshrink, Threshold** have boundaries that move with the parameter:
```cpp
// Softshrink: boundary at ±λ changes with λ
v_if(v > lambda) { result = v - lambda; }
v_elseif(v < -lambda) { result = v + lambda; }
```

**BUT these are all PIECEWISE LINEAR** - just comparisons and add/subtract operations.
They don't contain transcendental functions (exp, log, etc.), so **they don't need LUTs at all!**

The separable LUT approach covers **all transcendental parametrized activations**:
- ELU, CELU → need `exp(x)-1` LUT
- Softplus → needs `log(1+exp(x))` LUT
- SELU → needs `exp(x)-1` LUT (fixed params)

The boundary-dependent ones are trivial to implement without LUTs.

### Key Insight: Coefficient Reuse

**Existing LUT coefficients work for ALL parameter values!**

- If you fitted `exp(x)-1`, those coefficients work for ELU with ANY alpha
- If you fitted `log(1+exp(x))`, those coefficients work for Softplus with ANY beta
- The runtime parameter is just a multiply/divide applied AFTER LUT evaluation

**No refitting required** - this is pure assembly/integration work.

### Kernel Modification Pattern

For piecewise kernel, add pre/post transformations:

```cpp
// ============ SEPARABLE LUT EVALUATION ============

// 1. Input transformation (CELU, Softplus)
#ifdef SEPARABLE_INPUT_TRANSFORM
    sfpi::vFloat x_eval = x * INPUT_SCALE;  // e.g., beta * x for softplus
    // or: x_eval = x / INPUT_SCALE;        // e.g., x / alpha for CELU
#else
    sfpi::vFloat x_eval = x;
#endif

// 2. Standard LUT evaluation (unchanged)
sfpi::vFloat lut_result = evaluate_piecewise_lut(x_eval, LUT_DATA, NUM_SEGMENTS, POLY_DEGREE);

// 3. Output transformation (ELU, CELU, Softplus)
#ifdef SEPARABLE_OUTPUT_TRANSFORM
    sfpi::vFloat result = lut_result * OUTPUT_SCALE;  // e.g., alpha * g(x) for ELU
    // or: result = lut_result / OUTPUT_SCALE;        // e.g., g(u) / beta for softplus
#else
    sfpi::vFloat result = lut_result;
#endif

// 4. Conditional region (most separable activations)
#ifdef SEPARABLE_POSITIVE_IDENTITY
    v_if(x > 0.0f) {
        result = x;  // Identity for x > 0
    }
    v_endif;
#endif
```

### Example: ELU with Separable LUT

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_elu(uint slope_bits) {
    sfpi::vFloat alpha = Converter::as_float(slope_bits);

    // LUT stores BASE function: g(x) = exp(x) - 1
    // These coefficients work for ANY alpha value!
#ifdef INP_FLOAT32
    constexpr float LUT_EXP_M1[] = { /* FP32 coeffs for exp(x)-1 */ };
#else
    constexpr float LUT_EXP_M1[] = { /* BF16 coeffs for exp(x)-1 */ };
#endif

    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result;

        v_if(x > 0.0f) {
            result = x;  // Identity for positive
        } v_else {
            // Evaluate base LUT: g(x) = exp(x) - 1
            sfpi::vFloat exp_m1 = evaluate_lut(x, LUT_EXP_M1);
            // Apply runtime parameter: alpha * g(x)
            result = alpha * exp_m1;
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
```

### Example: Softplus with Input/Output Scaling

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softplus(uint param0, uint param1, uint param2) {
    sfpi::vFloat beta = Converter::as_float(param0);
    sfpi::vFloat beta_recip = Converter::as_float(param1);
    sfpi::vFloat threshold = Converter::as_float(param2);

    // LUT stores: g(u) = log(1 + exp(u))  [fitted for beta=1]
    constexpr float LUT_SOFTPLUS[] = { /* coeffs */ };

    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat u = beta * x;  // Input scaling

        v_if(u < threshold) {
            sfpi::vFloat g_u = evaluate_lut(u, LUT_SOFTPLUS);
            sfpi::dst_reg[0] = beta_recip * g_u;  // Output scaling
        }
        v_endif;

        sfpi::dst_reg++;
    }
}
```

### Error Analysis Checklist

> **MUST complete before production use:**

1. **Input scaling precision**
   - `x/alpha` or `beta*x` may lose precision for extreme parameter values
   - Very small alpha → large scaled input → may exceed LUT domain
   - Test with: alpha/beta ∈ {0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0}

2. **Output scaling error propagation**
   - Error scales with parameter: `|err_result| = |alpha| * |err_lut|`
   - Large alpha amplifies LUT approximation error

3. **Domain coverage**
   - LUT domain must cover SCALED input range
   - For CELU with alpha=0.1: `x/alpha` is 10× larger
   - May need extended domain or clamping

4. **Validation tests**
   - [ ] Standard ULP sweep with default params
   - [ ] ULP sweep with param = 0.1 (small)
   - [ ] ULP sweep with param = 10.0 (large)
   - [ ] Compare against non-separable reference
   - [ ] Benchmark runtime overhead of extra ops

### When NOT to Use Separable LUTs

| Activation | Reason |
|------------|--------|
| Hardtanh | Just a clamp - no LUT needed (piecewise linear) |
| Leaky ReLU / PReLU | Linear - no LUT needed |
| Softshrink | Piecewise linear - no LUT needed |
| Hardshrink | Piecewise linear - no LUT needed |
| Threshold | Piecewise linear - no LUT needed |
| High-precision cases | Extra multiply adds error |

**Key insight:** All the "non-separable" boundary-dependent activations are **piecewise linear**
and don't benefit from LUT approximation anyway. The separable approach covers 100% of the
transcendental parametrized activations that actually need LUTs.

---

## Step 4: Extract Code for TTNN Integration

### From Generic LUT Activation to TTNN

**Source files to extract from:**
- `kernels/compute/piecewise_generic.cpp` - Horner polynomial evaluation
- `kernels/compute/piecewise_rational.cpp` - Rational approximation (if needed)
- `lut_loader.hpp` - Coefficient structure reference

**Key functions to adapt:**

1. **Segment lookup**: `get_segment_index(x, bounds, num_segments)`
2. **Horner evaluation**: `evaluate_polynomial_horner(x, coeffs, degree)`
3. **Range reduction** (if applicable): `exp_reduce()`, `trig_reduce()`, etc.

### Minimal SFPU Kernel Template

```cpp
// ckernel_sfpu_<activation>_lut.h
namespace ckernel {
namespace sfpu {

// Embedded LUT: [num_segments+1 boundaries] + [num_segments * (degree+1) coefficients]
constexpr uint32_t NUM_SEGMENTS = 64;
constexpr uint32_t POLY_DEGREE = 3;
constexpr float LUT_DATA[] = {
    // boundaries
    -10.0f, -9.6875f, ..., 10.0f,
    // segment 0 coefficients (c0, c1, c2, c3)
    0.001f, 0.025f, 0.0001f, 0.0f,
    // segment 1 coefficients
    // ...
};

template <bool APPROXIMATION_MODE>
void gelu_lut_init() {
    // No-op for embedded LUT
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu_lut() {
    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Find segment
        sfpi::vFloat seg_width = (LUT_DATA[NUM_SEGMENTS] - LUT_DATA[0]) / NUM_SEGMENTS;
        sfpi::vFloat seg_idx = sfpi::floor((x - LUT_DATA[0]) / seg_width);
        seg_idx = sfpi::clamp(seg_idx, 0.0f, (float)(NUM_SEGMENTS - 1));

        // Get coefficient base index
        uint32_t coeff_base = NUM_SEGMENTS + 1 + (uint32_t)seg_idx * (POLY_DEGREE + 1);

        // Horner evaluation: c0 + x*(c1 + x*(c2 + x*c3))
        sfpi::vFloat result = LUT_DATA[coeff_base + POLY_DEGREE];
        for (int i = POLY_DEGREE - 1; i >= 0; i--) {
            result = result * x + LUT_DATA[coeff_base + i];
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

---

## Step 5: Prepare PR Attachments

### Required Plots

1. **ULP Distribution Chart** (`plots/<arch>/hardware_error_analysis/<activation>/<activation>_<precision>_ulp.png`)
   - Shows ULP error distribution for LUT vs TTNN
   - Must include both BF16 and FP32 variants

2. **Runtime vs Error Pareto** (`plots/<arch>/error_vs_runtime/<activation>/`)
   - Shows error-latency tradeoff
   - Highlights winning configuration on Pareto frontier

### Generate Plots (if not already available)

```bash
# From generic_lut_activation directory
cd /Users/nkapre/workspace/tt-metal/tt_metal/programming_examples/generic_lut_activation

# ULP distribution
python3 plots/plot_hardware_error_analysis.py \
    --activation gelu \
    --arch blackhole \
    --precision bf16 fp32 \
    --output-dir plots/blackhole/hardware_error_analysis/gelu

# Error vs runtime
python3 plots/plot_error_vs_runtime.py \
    --activation gelu \
    --arch blackhole \
    --csv-dir ../generic_lut_activation_embedded/data/blackhole/best \
    --output-dir plots/blackhole/error_vs_runtime/gelu
```

---

## Step 6: Create the PR

### Branch Naming
```bash
git checkout -b user/<issue-number>-lut-<activation>
# Example: user/12345-lut-gelu
```

### PR Template

```markdown
### Ticket
Fixes #<issue-number>

### Problem description
TTNN's current <activation> implementation has suboptimal accuracy (max ULP: X)
and/or runtime performance. This PR replaces it with a LUT-based piecewise
polynomial approximation that achieves Y ULP with Z% runtime improvement.

### What's changed
1. Added LUT-based SFPU implementation in `ckernel_sfpu_<activation>_lut.h`
2. Updated `unary_op_utils.cpp` to use new kernel for <activation>
3. Added embedded coefficients for <N> segments, degree <D> polynomial

### Performance Comparison
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Max ULP (BF16) | X | Y | Z% |
| Mean ULP (BF16) | X | Y | Z% |
| Runtime (256 tiles) | X us | Y us | Z% |

### ULP Distribution
![ULP Chart](plots/blackhole/hardware_error_analysis/<activation>/<activation>_bf16_ulp.png)

### Checklist
- [ ] All post-commit tests pass
- [ ] New/Existing tests provide coverage for changes
- [ ] Ran clang-tidy code analysis on PR branch
```

### Build and Test

```bash
# Full build
./build_metal.sh --build-all

# Lint
cmake --preset clang-tidy
cmake --build --preset clang-tidy

# Run post-commit tests
./tests/scripts/run_python_api_unit_tests.sh
./tests/scripts/run_cpp_unit_tests.sh

# Run activation-specific tests
pytest tests/ttnn/unit_tests/operations/test_eltwise_unary.py -k "<activation>" -vvv
```

---

## Step 7: Board-Validated Agentic Workflow

For automated PR submission using Claude Code or similar agents. All validation
must run on actual hardware with profiler timing and exhaustive ULP measurement.

### 3-Step Validation Pipeline

Every drop-in replacement PR MUST pass all three validation steps:

1. **LUT Kernel Validation** (`run_csv.sh`) — Tracy profiler timing + ULP
2. **TTNN Smoke Test** (`validate_dropin.py`) — Exhaustive BF16 + dense FP32 through TTNN API
3. **Existing Tests** — TTNN unit tests must still pass

### Step 7.1: LUT Kernel Validation with `run_csv.sh`

`run_csv.sh` is the authoritative source for device-profiler timing and accuracy.
It supports both polynomial and rational CSVs (auto-detected from column names).

```bash
# Polynomial CSV (columns: c0, c1, c2, ...)
./run_csv.sh /path/to/coefficients.csv --activation <name> --precision bf16 --runs 3

# Rational CSV (columns: n0, n1, ..., d0, d1, ...)
./run_csv.sh /path/to/rational_coefficients.csv --activation <name> --precision fp32 --runs 3

# Both precisions
./run_csv.sh /path/to/coefficients.csv --activation <name> --precision both --runs 3
```

**Output format** (5 standard shapes with Tracy profiler timing + ULP):
```
Config                      DegSum          MAE       MaxErr       MaxULP      MeanULP  Prof(µs)   Host(ms)
single_tile_n9d9_s1             20     1.74e-03     1.74e-03         0.45         0.45      2.59µs
8_tiles_n9d9_s1                 20     1.74e-03     1.74e-03         0.45         0.45      2.58µs
256_tiles_n9d9_s1               20     1.79e-03     1.81e-02         1.00         0.45      5.02µs
height_sharded_n9d9_s1          20     1.78e-03     1.81e-02         1.00         0.45     54.14µs
yolov4_n9d9_s1                  20     1.78e-03     1.81e-02         1.00         0.45     27.78µs
```

**What it validates**: The LUT approximation itself — coefficients, Horner evaluation,
segment dispatch, range handling. Tracy profiler gives true on-device kernel time.

### Step 7.2: TTNN Drop-in Smoke Test with `validate_dropin.py`

After modifying `ckernel_sfpu_<activation>.h` and rebuilding, run:

```bash
# Rebuild first
./build_metal.sh

# Exhaustive BF16 (all ~33K unique values in domain) + dense FP32 (65K points)
python3 tools/validate_dropin.py <activation>

# Single precision
python3 tools/validate_dropin.py <activation> --precision bf16
```

**Output format** (with PR-ready markdown):
```
  Prec   Points        MAE     MaxErr   MaxULP  MeanULP P99 ULP  Host us
  BF16    33346   1.78e-03   1.81e-02        1     0.05       1     11.0
  FP32    65536   2.08e-07   3.81e-06      116     2.65      29     17.0
```

**What it validates**: The full TTNN integration path — kernel compilation through
TTNN dispatch, `#ifdef INP_FLOAT32` dtype selection, parameter handling (beta,
threshold, etc.), tile padding, and output correctness.

### Step 7.3: Existing Unit Tests

```bash
# Run activation-specific tests
pytest tests/ttnn/unit_tests/operations/eltwise/ -k "<activation>" -vvv

# For softplus specifically:
pytest tests/ttnn/unit_tests/operations/eltwise/test_eltwise_softplus_inf.py -vvv
```

### Single Command (pr_submission.sh)

The entire flow is automated in one script:

```bash
cd tt_metal/programming_examples/generic_lut_activation_embedded

./tools/pr_submission.sh \
    --activation softplus \
    --bf16-csv $TT_POLY_FIT_DIR/data/coefficients/softplus_n9d9_s1_uniform_rational_ulp.csv \
    --fp32-csv $TT_POLY_FIT_DIR/data/coefficients/softplus_n11d11_s2_uniform_rational_ulp.csv
```

This runs all 4 steps:

1. **Validate LUT** (`run_csv.sh`) — Tracy profiler timing + ULP for both precisions
2. **Generate + Apply** (`apply_dropin.sh`) — `generate_dropin_header.py` creates the
   ckernel header (auto-detects parity, range reduction, existing SFPU signature),
   merges upstream/main, cleans strays, syncs archs, rebuilds, clears JIT cache, validates
3. **Three-way comparison** (`compare_three_way.sh`) — original vs drop-in vs embedded
4. **Submit PR** (`submit_pr.sh`) — clean branch from upstream/main, 4 ckernel headers only

Options:
- `--no-submit` — skip PR submission (dry run)
- `--start-from apply|compare|submit` — resume from a specific step

### Example: Softplus Drop-in (Validated on Blackhole, March 2026)

**Three-way comparison** (`compare_three_way.sh`):

BF16 (n9/d9 s1, exhaustive 33K BF16 values in [-10, 10]):
| Impl | MaxULP | MeanULP | yolov4 (µs) |
|------|--------|---------|-------------|
| Original SFPU | 2.12 | 0.47 | 45.8 |
| **Drop-in** | **1.00** | **0.45** | **37.1** |
| Embedded kernel | 1.01 | 0.45 | 27.8 |

FP32 (n11/d11 s2, 262K dense FP32 values in [-10, 10]):
| Impl | MaxULP | MeanULP | yolov4 (µs) |
|------|--------|---------|-------------|
| Original SFPU | 103,235 | 27,029 | 63.1 |
| **Drop-in** | **106.66** | **2.66** | **70.6** |
| Embedded kernel | 106.66 | 2.66 | 58.6 |

**Unit test**: `test_eltwise_softplus_inf.py` — PASSED (PCC=1.0)

---

## Quick Reference: File Locations

| What | Where |
|------|-------|
| Best configs | `data/<arch>/best/<activation>.csv` |
| ULP plots | `plots/<arch>/hardware_error_analysis/<activation>/` |
| Runtime plots | `plots/<arch>/error_vs_runtime/<activation>/` |
| SFPU API headers | `tt_metal/hw/inc/api/compute/eltwise_unary/` |
| SFPU implementations | `tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/` |
| TTNN op utils | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` |
| Op type enum | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` |

---

## Common Pitfalls

1. **Post-commit tests MUST pass** - Failures result in immediate revert
2. **Use SPDX license headers** - Required on all new files
3. **Don't break existing API** - New implementations should be opt-in initially
4. **Include both BF16 and FP32 results** - Even if only one is "winning"
5. **Run clang-tidy** - PR will fail lint checks otherwise
6. **Update tests** - Add/update unit tests for modified operations

## Critical Drop-in Pitfalls (Learned the Hard Way)

### 1. Stray Header Copies Shadow Real Headers
**Symptom**: Your code changes have no effect. Old behavior persists.
**Cause**: A stray `ckernel_sfpu_<activation>.h` at repo root (from a bad `cp` command)
gets found first by the JIT compiler's `-I.` or `-I<root>` include path.
**Fix**: Always check for and delete stray copies:
```bash
find $REPO_ROOT -maxdepth 1 -name "ckernel_sfpu_*.h" -delete
```
The `apply_dropin.sh` script does this automatically.

### 2. JIT Kernel Cache Doesn't Track ckernel Header Changes
**Symptom**: After modifying a ckernel header, the old behavior persists.
**Cause**: The JIT build system hashes kernel DEFINES + source path, but the dependency
hash for `trisck.o` only tracks firmware files, NOT transitively-included ckernel headers.
**Fix**: Delete the JIT cache after header changes:
```bash
rm -rf ~/.cache/tt-metal-cache
```

### 3. Use the EXACT Same Evaluation Code as Embedded Kernels
**Symptom**: ULP is worse than run_csv.sh despite using same coefficients.
**Cause**: Hand-written Horner loops produce different rounding than the optimized
interleaved Horner with `__attribute__((always_inline))` and `#pragma unroll`.
**Fix**: Use the shared `ckernel_sfpu_piecewise_rational.h` header which contains
the exact same evaluation functions as `piecewise_rational.cpp`:
- `piecewise_rational_eval_numer_denom` (interleaved Horner, ILP)
- `piecewise_rational_unroll_segment` (recursive template, deferred reciprocal)
- `piecewise_rational_eval` (public API, single vFloat → vFloat)

### 4. INP_FLOAT32 Works in SFPU Headers
**Fact**: `INP_FLOAT32` IS defined via `defines_generated.h` and IS visible to
ckernel_sfpu headers. Use `#ifdef INP_FLOAT32` to select different coefficient
sets for BF16 vs FP32. This enables dtype-specific LUTs (e.g., simpler BF16 fit
vs higher-precision FP32 fit).

### 5. JIT Compiler Uses Source Tree, Not build_Release/libexec
**Fact**: The `-I` paths in the JIT compiler command point to the SOURCE tree
(`tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/`), not `build_Release/libexec/`.
Modifying the source file is sufficient; `build_metal.sh` copies to libexec but the
JIT compiler doesn't use that copy.

### 6. Adaptive Per-Segment Degree is CRITICAL for Polynomial Drop-ins
**Symptom**: Drop-in is 3x slower than embedded kernel with identical coefficients.
**Cause**: Embedded kernel uses `SEGMENT_DEGREES[] = {0, 11, 1, 1}` (avg degree 3.2),
but the drop-in evaluates max degree (11) for ALL segments. Segments with constant
(degree 0) or linear (degree 1) fits waste ~10 Horner FMA steps per element.
**Fix**: `generate_dropin_header.py` now detects effective per-segment degree from
coefficient values and emits `#define HAS_SEGMENT_DEGREES` + `constexpr uint32_t
SEGMENT_DEGREES[]` before the shared evaluator include.
**Impact**: ELU BF16 drop-in went from 63µs → 23µs (matching embedded 20µs).

### 7. compare_three_way.sh Cache Variable Must Be TT_METAL_CACHE
**Symptom**: Step 2 (drop-in) shows same timing as step 1 (original).
**Cause**: Script used `rm -rf "$TT_METAL_CACHE_DIR"` but the correct env var is
`TT_METAL_CACHE`. The `rm` was a no-op (empty var = `rm -rf ""`).
**Fix**: Use `TT_METAL_CACHE` consistently. Fixed in compare_three_way.sh.

### 8. TTNN Dispatch Overhead: Face Loop + ITERATIONS
**Context**: TTNN calls `calculate_<activation>` through `_llk_math_eltwise_unary_sfpu_params_`
which loops 4 faces × 8 ITERATIONS = 32 DST register evaluations per tile. The embedded
kernel has a flat loop. This adds ~3µs overhead vs embedded for large polynomial bodies.
**Mitigation**: Adaptive degree reduces per-segment cost, making the dispatch overhead
a smaller fraction. Dual-eval (processing 2 DST rows simultaneously) could close the
remaining gap but requires restructuring `calculate_<activation>`.

---

## Example: Complete Softplus LUT Drop-in PR Submission

```bash
cd tt_metal/programming_examples/generic_lut_activation_embedded

# Everything in one command:
./tools/pr_submission.sh \
    --activation softplus \
    --bf16-csv $TT_POLY_FIT_DIR/data/coefficients/softplus_n9d9_s1_uniform_rational_ulp.csv \
    --fp32-csv $TT_POLY_FIT_DIR/data/coefficients/softplus_n11d11_s2_uniform_rational_ulp.csv

# Or dry-run first:
./tools/pr_submission.sh \
    --activation softplus \
    --bf16-csv $TT_POLY_FIT_DIR/data/coefficients/softplus_n9d9_s1_uniform_rational_ulp.csv \
    --fp32-csv $TT_POLY_FIT_DIR/data/coefficients/softplus_n11d11_s2_uniform_rational_ulp.csv \
    --no-submit

# Resume from a specific step (e.g., after fixing an issue):
./tools/pr_submission.sh --activation softplus --bf16-csv ... --start-from compare
```
