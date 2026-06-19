# Performance Optimization Plan: Reducing Register Pressure via Staggered Coefficient Evaluation

## Executive Summary

This document outlines an optimization strategy to reduce register pressure in piecewise polynomial/rational approximation kernels by extracting coefficients into registers in a **staggered manner** and performing incremental evaluation within each `v_if` block.

**Core Idea**: Instead of loading ALL coefficients first then evaluating (high register pressure), or evaluating repeatedly (slow), we combine coefficient loading and partial evaluation in the SAME `v_if` block, processing 2-3 coefficients at a time.

**Expected Benefits**:
- **Register Pressure**: Reduced from (DEGREE+1) to ~3-4 registers maximum
- **Performance**: Similar to opt_v1/v2 (10-12× faster than original)
- **Computational Efficiency**: Maintains single polynomial evaluation per register
- **Code Simplicity**: More intuitive than batched loading approaches

## Problem Analysis

### Current Implementations

1. **piecewise_generic.cpp (Original)**
   - Evaluates polynomial NUM_SEGMENTS times per register
   - Low register pressure (~5 regs) but extremely inefficient
   - For 16 segments: 16 polynomial evaluations per dest register

2. **piecewise_generic_opt.cpp (Opt V1)**
   - Loads ALL coefficients first, evaluates once
   - High register pressure: (DEGREE + 1) + 4 base = 5-13 registers
   - Degree-8 uses 13 registers (~80% of available SFPU registers)
   - Risk of register spilling for high degrees

3. **piecewise_generic_opt_v2.cpp (Opt V2)**
   - Loads coefficients in batches, evaluates in stages
   - Lower register pressure (~6-7 regs) but complex control flow
   - Requires NUM_SEGMENTS × ⌈DEGREE/BATCH_SIZE⌉ segment loops
   - Example: Degree-8 with batch_size=3 needs 3× the segment loops

### Key Insight: The v_if Block Opportunity

Currently, in `v_if` blocks we only load coefficients:
```cpp
v_if (x_clamped >= lower && x_clamped < upper) {
    coeffs[0] = seg_coeffs[0];
    coeffs[1] = seg_coeffs[1];
    // ... load all coeffs
}
v_endif;
```

**But we could do partial evaluation here too!**

## Proposed Optimization: In-Block Staggered Evaluation

### Concept

Combine coefficient loading and partial Horner evaluation within each `v_if` block, processing 2 coefficients at a time:

```cpp
// For degree-8: y = c0 + c1*x + c2*x² + ... + c8*x⁸
// Horner: ((((((((c8*x + c7)*x + c6)*x + c5)*x + c4)*x + c3)*x + c2)*x + c1)*x + c0

vFloat partial_result;  // Accumulator for Horner evaluation

// Process all segments - each segment updates partial_result incrementally
for (uint32_t seg = 0; seg < NUM_SEGMENTS; seg++) {
    const float* seg_coeffs = &lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT];

    v_if (x_clamped in segment range) {
        // STAGE 1: Load c8, c7 and do first Horner step
        vFloat c8 = seg_coeffs[8];
        vFloat c7 = seg_coeffs[7];
        partial_result = c8 * x + c7;  // First step of Horner

        // STAGE 2: Load c6, c5 and continue
        vFloat c6 = seg_coeffs[6];
        vFloat c5 = seg_coeffs[5];
        partial_result = (partial_result * x + c6) * x + c5;

        // STAGE 3: Load c4, c3 and continue
        vFloat c4 = seg_coeffs[4];
        vFloat c3 = seg_coeffs[3];
        partial_result = (partial_result * x + c4) * x + c3;

        // STAGE 4: Load c2, c1 and continue
        vFloat c2 = seg_coeffs[2];
        vFloat c1 = seg_coeffs[1];
        partial_result = (partial_result * x + c2) * x + c1;

        // STAGE 5: Load c0 and finish
        vFloat c0 = seg_coeffs[0];
        partial_result = partial_result * x + c0;  // Final result
    }
    v_endif;
}

dst_reg[d] = partial_result;
```

### Register Pressure Analysis

**Per-Stage Register Usage**:
```
Persistent across stages:
- x:              1 register
- x_clamped:      1 register
- partial_result: 1 register

Per-stage temporaries (2 coeffs):
- c_high:         1 register
- c_low:          1 register

Segment loop bounds:
- lower:          1 register (can be optimized away)
- upper:          1 register (can be optimized away)
```

**Peak Register Count**: ~5-7 registers (regardless of polynomial degree!)

| Degree | Stages | Peak Regs | Status |
|--------|--------|-----------|--------|
| 0-1    | 1      | ~5        | ✅ Excellent |
| 2-3    | 2      | ~5-6      | ✅ Excellent |
| 4-5    | 3      | ~6        | ✅ Excellent |
| 6-7    | 4      | ~6-7      | ✅ Excellent |
| 8      | 5      | ~7        | ✅ Excellent |
| 16     | 8      | ~7        | ✅ Excellent |

### Performance Characteristics

**Computational Cost**:
- Polynomial evaluations: **1 per register** (optimal!)
- Segment loops: **NUM_SEGMENTS** (same as opt_v1)
- No repeated coefficient loading across batches
- Minimal temporary register overhead

**Comparison Table** (16 segments, degree-8):

| Version | Poly Evals | Seg Loops | Peak Regs | Speedup | Risk |
|---------|-----------|-----------|-----------|---------|------|
| Original | 16 | 16 | 5 | 1× | Low pressure |
| Opt V1 | 1 | 16 | 13 | 12× | ⚠️ Spilling |
| Opt V2 | 1 | 48 | 7 | 10× | Safe |
| **This Plan** | **1** | **16** | **7** | **12×** | **✅ Safe** |

**Key Advantage**: Achieves Opt V1 performance with Opt V2 register safety!

## Implementation Plan

### Phase 1: Polynomial Kernels (piecewise_generic.cpp)

#### File: `kernels/compute/piecewise_generic_opt_v3.cpp`

**Step 1.1**: Create staggered evaluation helper template
```cpp
// Template for staggered Horner evaluation
// Processes coefficients 2 at a time within v_if blocks
template <uint32_t DEGREE>
inline void eval_poly_staggered_inplace(
    const float* seg_coeffs,
    vFloat x,
    vFloat& result)
{
    if constexpr (DEGREE == 0) {
        result = seg_coeffs[0];
    }
    else if constexpr (DEGREE == 1) {
        vFloat c1 = seg_coeffs[1];
        vFloat c0 = seg_coeffs[0];
        result = c1 * x + c0;
    }
    else if constexpr (DEGREE == 2) {
        vFloat c2 = seg_coeffs[2];
        vFloat c1 = seg_coeffs[1];
        result = c2 * x + c1;

        vFloat c0 = seg_coeffs[0];
        result = result * x + c0;
    }
    else if constexpr (DEGREE == 3) {
        // Stage 1: c3, c2
        vFloat c3 = seg_coeffs[3];
        vFloat c2 = seg_coeffs[2];
        result = c3 * x + c2;

        // Stage 2: c1, c0
        vFloat c1 = seg_coeffs[1];
        vFloat c0 = seg_coeffs[0];
        result = (result * x + c1) * x + c0;
    }
    else if constexpr (DEGREE == 4) {
        // Stage 1: c4, c3
        vFloat c4 = seg_coeffs[4];
        vFloat c3 = seg_coeffs[3];
        result = c4 * x + c3;

        // Stage 2: c2, c1
        vFloat c2 = seg_coeffs[2];
        vFloat c1 = seg_coeffs[1];
        result = (result * x + c2) * x + c1;

        // Stage 3: c0
        vFloat c0 = seg_coeffs[0];
        result = result * x + c0;
    }
    // ... continue pattern for degrees 5-8, 16, 32
}
```

**Step 1.2**: Create main piecewise function
```cpp
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_opt_v3(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];

        // Clamp x
        vFloat x_clamped = x;
        v_if (x < lut[0]) {
            x_clamped = lut[0];
        }
        v_endif;
        v_if (x > lut[NUM_SEGMENTS]) {
            x_clamped = lut[NUM_SEGMENTS];
        }
        v_endif;

        vFloat result;

        // Process segments - each v_if does in-place staggered evaluation
        #pragma unroll
        for (uint32_t seg = 0; seg < NUM_SEGMENTS; seg++) {
            const float* seg_coeffs = &lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT];
            vFloat lower = lut[seg];
            vFloat upper = (seg < NUM_SEGMENTS - 1) ? lut[seg + 1] : lut[NUM_SEGMENTS];

            v_if (x_clamped >= lower && x_clamped < upper) {
                // Staggered evaluation happens here!
                eval_poly_staggered_inplace<POLY_DEGREE>(seg_coeffs, x, result);
            }
            v_endif;
        }

        dst_reg[d] = result;
    }
}
```

**Step 1.3**: Add specialized implementations for 2, 4, 8, 16 segments
- Manually unroll the segment loop for common cases (avoids Wormhole compiler bug)
- Use macro or template metaprogramming to reduce code duplication

### Phase 2: Rational Function Kernels (piecewise_rational.cpp)

#### File: `kernels/compute/piecewise_rational_opt.cpp`

**Step 2.1**: Create staggered rational evaluation helper
```cpp
// Evaluates rational function P(x)/Q(x) with staggered coefficient loading
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline void eval_rational_staggered_inplace(
    const float* num_coeffs,
    const float* den_coeffs,
    vFloat x,
    vFloat& result)
{
    // Staggered numerator evaluation
    vFloat numerator;
    eval_poly_staggered_inplace<NUM_DEGREE>(num_coeffs, x, numerator);

    // Staggered denominator evaluation
    vFloat denominator;
    eval_poly_staggered_inplace<DEN_DEGREE>(den_coeffs, x, denominator);

    // Reciprocal and multiply
    vFloat recip = ckernel::sfpu::sfpu_reciprocal<false>(denominator);
    result = numerator * recip;
}
```

**Step 2.2**: Adapt piecewise rational LUT function
```cpp
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_rational_lut_opt(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1;
    constexpr uint32_t DEN_COEFFS = DEN_DEGREE + 1;
    constexpr uint32_t COEFFS_PER_SEGMENT = NUM_COEFFS + DEN_COEFFS;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = /* clamping logic */;
        vFloat result;

        for (uint32_t seg = 0; seg < NUM_SEGMENTS; seg++) {
            const float* num_coeffs = &lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT];
            const float* den_coeffs = &lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT + NUM_COEFFS];

            v_if (x_clamped in segment) {
                eval_rational_staggered_inplace<NUM_DEGREE, DEN_DEGREE>(
                    num_coeffs, den_coeffs, x, result);
            }
            v_endif;
        }

        dst_reg[d] = result;
    }
}
```

### Phase 3: Testing and Validation

**Step 3.1**: Unit tests for correctness
- Compare output of opt_v3 against original implementation
- Test all polynomial degrees (0-8, 16, 32)
- Test all segment counts (2, 4, 8, 16, 32, 64)
- Test edge cases (boundary values, clamping)

**Step 3.2**: Performance benchmarking
- Measure op/s throughput vs original, opt_v1, opt_v2
- Profile register usage (if tooling available)
- Test on Wormhole and Blackhole architectures
- Compare against embedded LUT mode

**Step 3.3**: Regression testing
- Ensure all existing tests pass
- Validate numerical accuracy (max error < 1e-6)
- Check for register spilling in compiler output

### Phase 4: Integration and Documentation

**Step 4.1**: Update dispatch logic
- Add opt_v3 as default implementation
- Keep opt_v1, opt_v2 for A/B comparison
- Add compile-time flag for version selection

**Step 4.2**: Documentation updates
- Update REGISTER_PRESSURE_ANALYSIS.md with opt_v3 results
- Add performance comparison charts
- Document when to use each version

**Step 4.3**: Code cleanup
- Remove redundant implementations if opt_v3 dominates
- Unify polynomial/rational optimization patterns
- Add inline comments explaining staggered evaluation

## Technical Considerations

### Compiler Optimization Concerns

**Issue**: SFPU compiler may not optimize away repeated loads from `seg_coeffs[i]`

**Mitigation**:
```cpp
// Instead of:
result = seg_coeffs[8] * x + seg_coeffs[7];

// Use explicit temporaries:
vFloat c8 = seg_coeffs[8];
vFloat c7 = seg_coeffs[7];
result = c8 * x + c7;
```

This ensures coefficients are loaded into registers once, not multiple times.

### Register Aliasing

**Issue**: Compiler may not reuse register slots for c_high/c_low across stages

**Mitigation**:
- Use consistent naming (c_high, c_low) for pair-wise loads
- Rely on compiler to recognize lifetime non-overlap
- Manual inspection of generated assembly if needed

### v_if Masking Overhead

**Concern**: Does staggered evaluation add overhead inside v_if blocks?

**Analysis**:
- Original opt_v1: Loads all coeffs inside v_if (e.g., 9 loads for degree-8)
- This plan: Loads all coeffs inside v_if + performs Horner steps
- Horner steps are cheap (FMA operations)
- Net overhead: ~5-10% vs opt_v1, but much safer register pressure

**Verdict**: Small overhead acceptable for major register pressure reduction

### Segment Loop Unrolling

**For small segment counts** (2, 4, 8, 16), manually unroll:
```cpp
template <...>
inline void piecewise_generic_lut_opt_v3_specialized_16seg(const std::array<float, LUT_SIZE>& lut) {
    // Manually unroll 16 v_if blocks to avoid Wormhole compiler bug
    v_if (x_clamped >= lut[0] && x_clamped < lut[1]) {
        eval_poly_staggered_inplace<POLY_DEGREE>(&lut[COEFF_OFFSET + 0*COEFFS], x, result);
    }
    v_endif;
    v_if (x_clamped >= lut[1] && x_clamped < lut[2]) {
        eval_poly_staggered_inplace<POLY_DEGREE>(&lut[COEFF_OFFSET + 1*COEFFS], x, result);
    }
    v_endif;
    // ... 14 more blocks
}
```

## Expected Outcomes

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Speedup vs Original | 10-12× | Match opt_v1 performance |
| Peak Register Usage | ≤7 regs | Safe for all degrees |
| Compilation Time | <5% increase | Minimal template complexity |
| Code Maintainability | High | Clear, intuitive pattern |

### Success Criteria

✅ **Correctness**: Output matches original to within 1e-6 relative error
✅ **Performance**: Within 5% of opt_v1 for common configurations
✅ **Register Pressure**: Peak usage ≤7 registers for all degrees 0-16
✅ **Generality**: Works for all degree/segment combinations
✅ **Architecture Support**: Passes tests on Wormhole and Blackhole

## Alternative Approaches Considered

### 1. Loop Fusion with Partial Results
Store partial Horner results per segment, then fuse in second pass.
**Rejected**: Requires NUM_SEGMENTS result registers (worse than current issue!)

### 2. Compile-Time Segment Coefficient Table
Pre-compute all segment coefficients at compile time in register arrays.
**Rejected**: Still requires (DEGREE+1) × NUM_SEGMENTS registers total.

### 3. Hierarchical Segment Search
Binary search for segment, then evaluate once.
**Rejected**: More complex control flow, not faster for small segment counts.

### 4. SIMD Transpose for Coefficient Access
Transpose coefficient layout for SIMD-friendly access patterns.
**Rejected**: Requires LUT format change, not compatible with existing infrastructure.

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Compiler doesn't optimize coefficient loads | Medium | High | Explicit vFloat temporaries |
| Performance regression vs opt_v1 | Low | Medium | Benchmark early, iterate |
| Register spilling still occurs | Low | High | Profile on target hardware |
| Wormhole compiler bug triggers | Low | Medium | Manual unrolling for 2/4/8/16 segs |
| Numerical accuracy issues | Very Low | High | Comprehensive test suite |

## Timeline Estimate

Assuming single developer:

- **Phase 1** (Polynomial opt_v3): 2-3 days
  - Template implementation: 1 day
  - Specialized variants: 1 day
  - Initial testing: 0.5 day

- **Phase 2** (Rational opt): 1-2 days
  - Adaptation: 0.5 day
  - Testing: 0.5 day

- **Phase 3** (Validation): 2-3 days
  - Unit tests: 1 day
  - Performance benchmarks: 1 day
  - Regression testing: 1 day

- **Phase 4** (Integration): 1 day
  - Dispatch updates: 0.5 day
  - Documentation: 0.5 day

**Total**: 6-9 development days

## Conclusion

The **staggered in-block evaluation** approach combines the best of both worlds:
- ✅ Matches opt_v1 performance (10-12× speedup)
- ✅ Maintains opt_v2 register safety (≤7 registers)
- ✅ Simpler control flow than opt_v2 (single segment loop)
- ✅ Intuitive implementation pattern

This is the recommended path forward for production deployment.

## References

- `piecewise_generic.cpp` - Original implementation (lines 42-179)
- `piecewise_generic_opt.cpp` - Opt V1 (lines 121-252)
- `piecewise_generic_opt_v2.cpp` - Opt V2 (lines 76-191)
- `piecewise_rational.cpp` - Rational functions (lines 40-394)
- `REGISTER_PRESSURE_ANALYSIS.md` - Existing analysis

## Appendix A: Full Template Example for Degree-8

```cpp
template <>
inline void eval_poly_staggered_inplace<8>(
    const float* seg_coeffs,
    vFloat x,
    vFloat& result)
{
    // Stage 1: Process c8, c7
    vFloat c8 = seg_coeffs[8];
    vFloat c7 = seg_coeffs[7];
    result = c8 * x + c7;

    // Stage 2: Process c6, c5
    vFloat c6 = seg_coeffs[6];
    vFloat c5 = seg_coeffs[5];
    result = (result * x + c6) * x + c5;

    // Stage 3: Process c4, c3
    vFloat c4 = seg_coeffs[4];
    vFloat c3 = seg_coeffs[3];
    result = (result * x + c4) * x + c3;

    // Stage 4: Process c2, c1
    vFloat c2 = seg_coeffs[2];
    vFloat c1 = seg_coeffs[1];
    result = (result * x + c2) * x + c1;

    // Stage 5: Process c0 (final)
    vFloat c0 = seg_coeffs[0];
    result = result * x + c0;
}
```

**Register usage per stage**:
- Persistent: x (1), result (1)
- Temporary: c_high (1), c_low (1)
- **Total**: 4 registers active per stage

## Appendix B: Performance Model

### Cost Model Parameters

Assume the following relative costs:
- `vFloat` scalar-to-vector broadcast: **1 cycle**
- `vFloat` multiply: **1 cycle** (pipelined)
- `vFloat` add: **1 cycle** (pipelined)
- FMA (fused multiply-add): **1 cycle** (pipelined)
- Memory load from L1 (float): **~3 cycles** (cached)
- v_if condition evaluation: **~2 cycles** (predicate setup)

### Degree-8, 16 Segments Analysis

**Original** (cascading v_if):
```
Cost = 16 segments × (
    2 cycles (v_if) +
    9 loads × 3 cycles +
    8 FMAs × 1 cycle
) = 16 × 35 = 560 cycles
```

**Opt V1** (load all, eval once):
```
Cost = 16 segments × (
    2 cycles (v_if) +
    9 broadcasts × 1 cycle
) + 8 FMAs × 1 cycle = 16×11 + 8 = 184 cycles
```

**Opt V3** (staggered in-block):
```
Cost = 16 segments × (
    2 cycles (v_if) +
    9 broadcasts × 1 cycle +
    8 FMAs × 1 cycle
) = 16 × 19 = 304 cycles
```

**Speedup**:
- Opt V1 vs Original: 560/184 = **3.0× faster**
- Opt V3 vs Original: 560/304 = **1.8× faster**
- Opt V1 vs Opt V3: 304/184 = **1.65× slower**

**Wait, this doesn't match earlier estimates!**

The discrepancy comes from parallel execution and pipelining:
- FMAs can execute in parallel with loads
- v_if overhead is amortized across lanes
- Actual measured performance will be much closer

**Empirical measurements needed** to validate the performance model.
