# Dual X-Vector Evaluation Implementation Plan

## Executive Summary

Modify `eval_polynomial<>` to evaluate two independent x-vectors simultaneously, exploiting instruction-level parallelism to hide the 2-cycle SFPMAD latency. Expected speedup: ~2x for polynomial evaluation (from ~4 cycles per value to ~4 cycles per 2 values).

**Key Insight**: Horner's method has serial dependencies (result = result * x + coeff), causing pipeline stalls. By interleaving two independent Horner chains, we keep the SFPU busy while waiting for each operation to complete.

**Interface Impact**: Zero. This is a pure internal optimization to the SFPU kernel code. No changes to host code, kernel dispatch, or circular buffers.

---

## Background: Current Bottleneck

### Current Implementation (Degree 4 example)
```cpp
// piecewise_generic.cpp:128-133
vFloat result = coeffs[4];           // Load immediate
result = result * x + coeffs[3];     // Cycle 0-1: sfpmad (2 cycles)
result = result * x + coeffs[2];     // Cycle 2-3: sfpmad (waits for prev)
result = result * x + coeffs[1];     // Cycle 4-5: sfpmad (waits for prev)
result = result * x + coeffs[0];     // Cycle 6-7: sfpmad (waits for prev)
// Total: 8 cycles for 1 value
```

### Disassembly Evidence
- Each Horner step: 1 load immediate + 1 sfpmad (2 cycles)
- Serial dependency prevents pipelining
- SFPU sits idle during latency bubbles

### Proposed Solution
```cpp
// Interleaved dual evaluation
result1 = coeffs[4];
result2 = coeffs[4];
result1 = result1 * x1 + coeffs[3];  // Cycle 0-1
result2 = result2 * x2 + coeffs[3];  // Cycle 0-1 (independent, overlaps)
result1 = result1 * x1 + coeffs[2];  // Cycle 2-3
result2 = result2 * x2 + coeffs[2];  // Cycle 2-3
result1 = result1 * x1 + coeffs[1];  // Cycle 4-5
result2 = result2 * x2 + coeffs[1];  // Cycle 4-5
result1 = result1 * x1 + coeffs[0];  // Cycle 6-7
result2 = result2 * x2 + coeffs[0];  // Cycle 6-7
// Total: ~8-9 cycles for 2 values (2x speedup)
```

---

## Architecture Design

### Files to Modify
1. **`kernels/compute/piecewise_generic.cpp`**
   - Add `eval_polynomial_dual<>` template
   - Modify main loop to process pairs: `for (int d = 0; d < 32; d += 2)`
   - Add dual versions of `piecewise_generic_lut`, `piecewise_staggered_horner_lut`

2. **`kernels/compute/piecewise_generic_specialized.cpp`**
   - Add dual versions of specialized functions (4/8/16/32 segments)
   - Modify dispatcher to call dual versions

### Compilation Flag Strategy
Add a new compile-time flag: `USE_DUAL_EVAL`
- **Not defined** (default): Use existing single-evaluation code (no regression)
- **Defined**: Enable dual-evaluation optimization

This allows:
- A/B testing and performance validation
- Gradual rollout per activation function
- Easy rollback if issues arise

### Template Hierarchy
```
eval_polynomial<DEGREE>(coeffs, x) → vFloat result
    ↓ (new)
eval_polynomial_dual<DEGREE>(coeffs, x1, x2) → (vFloat result1, vFloat result2)

piecewise_generic_lut<DEGREE, SEGS, SIZE>(lut)
    ↓ (new)
piecewise_generic_lut_dual<DEGREE, SEGS, SIZE>(lut)
```

---

## Implementation Steps

### Phase 1: Core Dual Evaluation Template

**File**: `kernels/compute/piecewise_generic.cpp`

**Location**: Insert after line 316 (after existing `eval_polynomial<>`)

**Task 1.1**: Create `eval_polynomial_dual<>` template for all degrees (0-16, 32)

```cpp
// Dual polynomial evaluation using Horner's method with interleaved operations
// Returns results via reference parameters for efficiency
template <uint32_t DEGREE>
inline void eval_polynomial_dual(const float* coeffs, vFloat x1, vFloat x2, vFloat& result1, vFloat& result2) {
    if constexpr (DEGREE == 0) {
        result1 = coeffs[0];
        result2 = coeffs[0];
    } else if constexpr (DEGREE == 1) {
        result1 = coeffs[0] + coeffs[1] * x1;
        result2 = coeffs[0] + coeffs[1] * x2;
    } else if constexpr (DEGREE == 2) {
        result1 = coeffs[2];
        result2 = coeffs[2];
        result1 = result1 * x1 + coeffs[1];
        result2 = result2 * x2 + coeffs[1];
        result1 = result1 * x1 + coeffs[0];
        result2 = result2 * x2 + coeffs[0];
    } else if constexpr (DEGREE == 3) {
        result1 = coeffs[3];
        result2 = coeffs[3];
        result1 = result1 * x1 + coeffs[2];
        result2 = result2 * x2 + coeffs[2];
        result1 = result1 * x1 + coeffs[1];
        result2 = result2 * x2 + coeffs[1];
        result1 = result1 * x1 + coeffs[0];
        result2 = result2 * x2 + coeffs[0];
    } else if constexpr (DEGREE == 4) {
        result1 = coeffs[4];
        result2 = coeffs[4];
        result1 = result1 * x1 + coeffs[3];
        result2 = result2 * x2 + coeffs[3];
        result1 = result1 * x1 + coeffs[2];
        result2 = result2 * x2 + coeffs[2];
        result1 = result1 * x1 + coeffs[1];
        result2 = result2 * x2 + coeffs[1];
        result1 = result1 * x1 + coeffs[0];
        result2 = result2 * x2 + coeffs[0];
    }
    // Continue for DEGREE 5-16, 32 following same pattern
}
```

**Acceptance Criteria**:
- All degrees 0-16 and 32 implemented
- Strict interleaving: `result1` op followed immediately by `result2` op
- Identical mathematical semantics to single-eval version

---

### Phase 2: Dual Main Loop (Generic Version)

**File**: `kernels/compute/piecewise_generic.cpp`

**Location**: Insert after line 395 (after existing `piecewise_generic_lut`)

**Task 2.1**: Create `piecewise_generic_lut_dual<>` function

```cpp
#ifdef USE_DUAL_EVAL
// Dual-evaluation version: process two x-vectors per iteration
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dual(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    static_assert(LUT_SIZE == (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * COEFFS_PER_SEGMENT),
                  "LUT_SIZE must equal (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)");

    // Process pairs of destination registers (32 total → 16 iterations)
    for (int d = 0; d < 32; d += 2) {
        vFloat x_orig1 = dst_reg[d];
        vFloat x_orig2 = dst_reg[d + 1];

#if defined(RANGE_REDUCTION_EXP)
        constexpr float EXP_OVERFLOW = 88.5f;
        constexpr float EXP_UNDERFLOW = -88.5f;
        vFloat x1, x2;
        vInt k_int1, k_int2;
        exp_reduce(x_orig1, x1, k_int1);
        exp_reduce(x_orig2, x2, k_int2);
#elif defined(RANGE_REDUCTION_TRIG)
        vFloat x1, x2;
        vInt q_int1, q_int2;
        trig_reduce(x_orig1, x1, q_int1);
        trig_reduce(x_orig2, x2, q_int2);
#else
        vFloat x1 = x_orig1;
        vFloat x2 = x_orig2;
#endif

        // Clamp both x-vectors to valid range
        vFloat min_bound = lut[0];
        vFloat max_bound = lut[NUM_SEGMENTS];
        vFloat x1_clamped = x1;
        vFloat x2_clamped = x2;
        vec_min_max(min_bound, x1_clamped);
        vec_min_max(x1_clamped, max_bound);
        vec_min_max(min_bound, x2_clamped);
        vec_min_max(x2_clamped, max_bound);

        // Start with segment 0 for both vectors
        vFloat result1, result2;
        eval_polynomial_dual<POLY_DEGREE>(&lut[COEFF_OFFSET], x1, x2, result1, result2);

        // Segment selection: each vector may fall into different segments
        // NOTE: This is the tricky part - need to handle independent segment selection
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            // Check if x1 is in this segment
            v_if (x1_clamped >= lut[seg]) {
                // Only re-evaluate for x1, keep x2 result unchanged
                vFloat temp_result1 = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT], x1);
                result1 = temp_result1;
            }
            v_endif;

            // Check if x2 is in this segment
            v_if (x2_clamped >= lut[seg]) {
                vFloat temp_result2 = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT], x2);
                result2 = temp_result2;
            }
            v_endif;
        }

#if defined(RANGE_REDUCTION_EXP)
        v_if(x_orig1 > EXP_OVERFLOW) {
            result1 = std::numeric_limits<float>::infinity();
        }
        v_elseif(x_orig1 < EXP_UNDERFLOW) {
            result1 = 0.0f;
        }
        v_else {
            result1 = exp_expand(result1, k_int1);
        }
        v_endif;

        v_if(x_orig2 > EXP_OVERFLOW) {
            result2 = std::numeric_limits<float>::infinity();
        }
        v_elseif(x_orig2 < EXP_UNDERFLOW) {
            result2 = 0.0f;
        }
        v_else {
            result2 = exp_expand(result2, k_int2);
        }
        v_endif;
#elif defined(RANGE_REDUCTION_TRIG)
        result1 = trig_expand(result1, q_int1);
        result2 = trig_expand(result2, q_int2);
#endif

#ifdef HAS_CRITICAL_POINT
        v_if (x1 == lut[CRITICAL_IDX]) {
            result1 = CRITICAL_VALUE;
        }
        v_endif;
        v_if (x2 == lut[CRITICAL_IDX]) {
            result2 = CRITICAL_VALUE;
        }
        v_endif;
#endif

        dst_reg[d] = result1;
        dst_reg[d + 1] = result2;
    }
}
#endif // USE_DUAL_EVAL
```

**Acceptance Criteria**:
- Loop processes pairs: `d = 0, 2, 4, ..., 30` (16 iterations total)
- Range reduction applied independently to both x-vectors
- Segment selection works correctly when x1 and x2 fall in different segments
- Critical point handling works for both vectors
- Compiles only when `USE_DUAL_EVAL` is defined

**Critical Design Decision**: Segment selection strategy

**Problem**: After initial dual evaluation in segment 0, x1 and x2 may belong to different segments. We need to re-evaluate the correct segment for each vector independently.

**Proposed Solution**:
- Initial dual eval in segment 0 (both vectors share segment 0 polynomial)
- Then iterate through segments 1..N-1
- For each segment, check x1 and x2 independently
- If x1 is in this segment, re-evaluate using single-eval (overwrites result1)
- If x2 is in this segment, re-evaluate using single-eval (overwrites result2)

**Tradeoff**: Segment selection loops don't benefit from dual-eval (each vector might be in different segment). However, polynomial evaluation (the expensive part) still gets 2x speedup in segment 0, and subsequent segments use single-eval as fallback.

**Alternative (more complex)**: Could use dual-eval even in segment selection by buffering segment indices and batch-processing pairs that fall in the same segment. This adds significant complexity for marginal gain. **Recommendation**: Start with simple approach above.

---

### Phase 3: Dual Staggered Horner Version

**File**: `kernels/compute/piecewise_generic.cpp`

**Location**: Insert after line 491 (after existing `piecewise_staggered_horner_lut`)

**Task 3.1**: Create `piecewise_staggered_horner_lut_dual<>` function

This follows the same pattern as Phase 2 but uses the staggered Horner approach (coefficient selection via cascading v_if, then Horner steps).

**Key Difference**: The staggered Horner approach selects coefficients per-lane dynamically. With dual-eval:
- Select highest-degree coefficient for both x1 and x2
- Perform Horner steps with interleaved operations
- At each Horner step, select correct coefficient for both vectors, then dual-multiply

**Pseudo-code structure**:
```cpp
// Select highest coefficient for both vectors (separate cascades)
vFloat coeff1_highest = lut[COEFF_OFFSET + POLY_DEGREE];  // seg 0 default for x1
for (seg...) { v_if (x1_clamped >= lut[seg]) { coeff1_highest = lut[...]; } v_endif; }

vFloat coeff2_highest = lut[COEFF_OFFSET + POLY_DEGREE];  // seg 0 default for x2
for (seg...) { v_if (x2_clamped >= lut[seg]) { coeff2_highest = lut[...]; } v_endif; }

result1 = coeff1_highest;
result2 = coeff2_highest;

// Horner steps (interleaved)
for (int deg = POLY_DEGREE - 1; deg >= 0; deg--) {
    // Select coefficient for x1
    vFloat coeff1 = lut[COEFF_OFFSET + deg];
    for (seg...) { v_if (x1_clamped >= lut[seg]) { coeff1 = lut[...]; } v_endif; }

    // Select coefficient for x2
    vFloat coeff2 = lut[COEFF_OFFSET + deg];
    for (seg...) { v_if (x2_clamped >= lut[seg]) { coeff2 = lut[...]; } v_endif; }

    // Interleaved Horner step
    result1 = result1 * x1 + coeff1;
    result2 = result2 * x2 + coeff2;
}
```

**Acceptance Criteria**:
- Independent coefficient selection for x1 and x2 at each Horner step
- Interleaved Horner multiply-add operations
- Compiles only when `USE_DUAL_EVAL` is defined
- Semantically identical to single-eval staggered Horner

---

### Phase 4: Specialized Versions (4/8/16/32 segments)

**File**: `kernels/compute/piecewise_generic_specialized.cpp`

**Task 4.1**: Create dual versions of all specialized functions

For each segment count (4, 8, 16, 32):
- Copy existing specialized function
- Rename: `piecewise_generic_lut_specialized_4` → `piecewise_generic_lut_specialized_4_dual`
- Change loop: `for (int d = 0; d < 32; d++)` → `for (int d = 0; d < 32; d += 2)`
- Duplicate all variables: `x_orig`, `x`, `x_clamped`, `result` → `x_orig1/2`, `x1/2`, `x1/2_clamped`, `result1/2`
- Replace `eval_polynomial<>` with dual version (when both vectors in segment 0)
- Keep manual unrolling for segment selection (each vector checked independently)

**Example for 4-segment specialized (lines 31-117)**:

```cpp
#ifdef USE_DUAL_EVAL
template <uint32_t POLY_DEGREE, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_specialized_4_dual(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 4;
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    for (int d = 0; d < 32; d += 2) {
        vFloat x_orig1 = dst_reg[d];
        vFloat x_orig2 = dst_reg[d + 1];

        // ... range reduction for both ...

        vFloat x1_clamped = x1;
        vFloat x2_clamped = x2;
        // ... clamping for both ...

        // Initial dual evaluation in segment 0
        vFloat result1, result2;
#ifdef SEG0_DEGREE
        eval_polynomial_dual<SEG0_DEGREE>(&lut[COEFF_OFFSET], x1, x2, result1, result2);
#else
        eval_polynomial_dual<POLY_DEGREE>(&lut[COEFF_OFFSET], x1, x2, result1, result2);
#endif

        // Manual unrolling for segment selection (independent for x1 and x2)
        v_if (x1_clamped >= lut[1]) {
#ifdef SEG1_DEGREE
            result1 = eval_polynomial<SEG1_DEGREE>(&lut[COEFF_OFFSET + 1 * COEFFS_PER_SEGMENT], x1);
#else
            result1 = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + 1 * COEFFS_PER_SEGMENT], x1);
#endif
        }
        v_endif;

        v_if (x2_clamped >= lut[1]) {
#ifdef SEG1_DEGREE
            result2 = eval_polynomial<SEG1_DEGREE>(&lut[COEFF_OFFSET + 1 * COEFFS_PER_SEGMENT], x2);
#else
            result2 = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + 1 * COEFFS_PER_SEGMENT], x2);
#endif
        }
        v_endif;

        // Repeat for segments 2 and 3...

        // ... range expansion, critical point handling for both ...

        dst_reg[d] = result1;
        dst_reg[d + 1] = result2;
    }
}
#endif // USE_DUAL_EVAL
```

**Acceptance Criteria**:
- Dual versions created for 4, 8, 16, 32 segment counts
- Manual unrolling preserved (Wormhole compiler bug workaround)
- Adaptive degree optimization (SEG{i}_DEGREE macros) preserved
- All versions compile with `USE_DUAL_EVAL` flag

---

### Phase 5: Update Dispatcher

**File**: `kernels/compute/piecewise_generic_specialized.cpp`

**Location**: Lines 742-762 (dispatcher function)

**Task 5.1**: Modify `piecewise_generic_lut_dispatch<>` to call dual versions

```cpp
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dispatch(const std::array<float, LUT_SIZE>& lut) {
#ifdef USE_STAGGERED_HORNER
    // Staggered Horner variants
    #ifdef USE_DUAL_EVAL
        piecewise_staggered_horner_lut_dual<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
    #else
        piecewise_staggered_horner_lut<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
    #endif
#else
    // Standard dispatch logic
    #ifdef USE_DUAL_EVAL
        if constexpr (NUM_SEGMENTS == 4) {
            piecewise_generic_lut_specialized_4_dual<POLY_DEGREE, LUT_SIZE>(lut);
        } else if constexpr (NUM_SEGMENTS == 8) {
            piecewise_generic_lut_specialized_8_dual<POLY_DEGREE, LUT_SIZE>(lut);
        } else if constexpr (NUM_SEGMENTS == 16) {
            piecewise_generic_lut_specialized_16_dual<POLY_DEGREE, LUT_SIZE>(lut);
        } else if constexpr (NUM_SEGMENTS == 32) {
            piecewise_generic_lut_specialized_32_dual<POLY_DEGREE, LUT_SIZE>(lut);
        } else {
            piecewise_generic_lut_dual<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
        }
    #else
        // Existing single-eval logic (unchanged)
        if constexpr (NUM_SEGMENTS == 4) {
            piecewise_generic_lut_specialized_4<POLY_DEGREE, LUT_SIZE>(lut);
        } else if constexpr (NUM_SEGMENTS == 8) {
            piecewise_generic_lut_specialized_8<POLY_DEGREE, LUT_SIZE>(lut);
        } else if constexpr (NUM_SEGMENTS == 16) {
            piecewise_generic_lut_specialized_16<POLY_DEGREE, LUT_SIZE>(lut);
        } else if constexpr (NUM_SEGMENTS == 32) {
            piecewise_generic_lut_specialized_32<POLY_DEGREE, LUT_SIZE>(lut);
        } else {
            piecewise_generic_lut<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
        }
    #endif
#endif
}
```

**Acceptance Criteria**:
- Dispatcher routes to dual versions when `USE_DUAL_EVAL` is defined
- Falls back to single-eval versions when flag not defined
- Works with both `USE_STAGGERED_HORNER` and standard dispatch paths

---

## Testing Strategy

### Phase 6: Correctness Testing

**File**: `test.py` (or equivalent test harness)

**Task 6.1**: Create test configuration for dual-eval mode

Add a new test sweep that builds kernels with `-DUSE_DUAL_EVAL`:

```python
# In test.py or build configuration
def test_dual_eval_correctness():
    """
    Verify dual-eval produces identical results to single-eval
    """
    activations = ['gelu', 'sigmoid', 'tanh', 'silu', 'exp']
    degrees = [2, 3, 4, 5, 6, 7, 8]
    depths = [4, 8, 16, 32]

    for act in activations:
        for deg in degrees:
            for depth in depths:
                # Build with USE_DUAL_EVAL
                build_kernel(activation=act, degree=deg, depth=depth, flags=['-DUSE_DUAL_EVAL'])
                result_dual = run_test(...)

                # Build without USE_DUAL_EVAL (baseline)
                build_kernel(activation=act, degree=deg, depth=depth, flags=[])
                result_single = run_test(...)

                # Compare outputs (bit-exact)
                assert np.array_equal(result_dual, result_single), \
                    f"Mismatch: {act}, degree={deg}, depth={depth}"
```

**Acceptance Criteria**:
- All activation functions produce bit-exact results (single vs dual eval)
- Test covers all polynomial degrees (1-16, 32)
- Test covers all segment counts (4, 8, 16, 32, 64)
- Test includes edge cases: critical points, range reduction, overflow/underflow

**Test Locations**:
- Unit tests: Ensure kernels compile and run
- Sweep tests: Cover all degree/depth combinations
- Model tests: Run full model with dual-eval kernels (if applicable)

---

### Phase 7: Performance Benchmarking

**Task 7.1**: Add Tracy profiling markers for polynomial evaluation

If not already instrumented, add Tracy zones around the SFPU kernel execution to measure cycle counts.

**Task 7.2**: Create benchmark script

```bash
#!/bin/bash
# benchmark_dual_eval.sh

ACTIVATIONS="gelu sigmoid tanh silu swish exp"
DEGREES="3 4 5 6 7 8"
DEPTHS="8 16 32"

echo "Activation,Degree,Depth,Mode,Runtime_us,Speedup" > results.csv

for act in $ACTIVATIONS; do
    for deg in $DEGREES; do
        for depth in $DEPTHS; do
            # Baseline (single-eval)
            time_single=$(run_benchmark --activation=$act --degree=$deg --depth=$depth)

            # Dual-eval
            time_dual=$(run_benchmark --activation=$act --degree=$deg --depth=$depth --dual-eval)

            speedup=$(echo "scale=2; $time_single / $time_dual" | bc)

            echo "$act,$deg,$depth,single,$time_single,1.0" >> results.csv
            echo "$act,$deg,$depth,dual,$time_dual,$speedup" >> results.csv
        done
    done
done

# Generate summary plot
python plot_speedup.py results.csv
```

**Expected Results**:
- Polynomial evaluation: 1.7x - 2.0x speedup (approaching 2x theoretical maximum)
- End-to-end kernel: 1.3x - 1.7x speedup (accounting for segment selection overhead)
- Higher degrees should show larger speedups (more Horner steps to hide latency)

**Acceptance Criteria**:
- Geometric mean speedup across all configurations ≥ 1.5x
- No configuration shows regression (speedup < 0.95x)
- Speedup scales with polynomial degree (higher degree → better speedup)

**Task 7.3**: Disassembly verification

For one representative configuration (e.g., gelu, degree=4, depth=16):

```bash
# Dump assembly for single-eval
objdump -d kernel_single.elf | grep -A 50 "eval_polynomial" > single_asm.txt

# Dump assembly for dual-eval
objdump -d kernel_dual.elf | grep -A 50 "eval_polynomial_dual" > dual_asm.txt

# Manual inspection:
# - Verify SFPMAD instructions are interleaved (result1 op, result2 op, result1 op, ...)
# - Verify load immediates are shared between result1 and result2
# - Count SFPMAD instructions: should be ~2x in dual version but with similar cycle count
```

**Acceptance Criteria**:
- Disassembly shows interleaved SFPMAD operations
- No unexpected register spills or memory loads
- Coefficient loads are shared between result1 and result2 chains

---

## Rollout Strategy

### Phase 8: Gradual Rollout

**Option A: Per-Activation Rollout**
- Week 1: Enable `USE_DUAL_EVAL` for low-risk activations (relu, sigmoid, tanh)
- Week 2: Enable for medium-risk activations (gelu, silu, swish)
- Week 3: Enable for high-complexity activations (exp, trig, special functions)

**Option B: Per-Degree Rollout**
- Week 1: Enable for low degrees (1-4) where speedup is smaller but risk is lower
- Week 2: Enable for medium degrees (5-8)
- Week 3: Enable for high degrees (10+) where speedup is highest

**Recommendation**: Start with Option A (per-activation) since user-facing impact is clearer.

**Configuration Management**:

Edit kernel generation script to conditionally add `-DUSE_DUAL_EVAL`:

```python
# In generate_embedded_kernels.py or equivalent
DUAL_EVAL_ENABLED_ACTIVATIONS = [
    'sigmoid',  # Week 1
    'tanh',     # Week 1
    # 'gelu',   # Week 2 (commented out initially)
    # 'silu',   # Week 2
]

def get_compile_flags(activation):
    flags = ['-DEMBEDDED_LUT']
    if activation in DUAL_EVAL_ENABLED_ACTIVATIONS:
        flags.append('-DUSE_DUAL_EVAL')
    return flags
```

---

## Risk Mitigation

### Identified Risks

**Risk 1: Register Pressure**
- **Symptom**: Kernel compilation fails or generates spill code
- **Mitigation**: Monitor disassembly for register spills. If spills occur, consider:
  - Reducing local variable scope
  - Reordering operations to free registers earlier
  - Falling back to single-eval for high-degree polynomials (12+)

**Risk 2: SFPU Compiler Reordering**
- **Symptom**: Compiler doesn't preserve interleaving, no speedup observed
- **Mitigation**: Use `#pragma` hints or inline assembly to enforce ordering
- **Verification**: Check disassembly (Phase 7, Task 7.3)

**Risk 3: Segment Selection Overhead**
- **Symptom**: Dual-eval speedup is negated by doubled segment selection checks
- **Mitigation**:
  - Segment selection is already cheap (comparison + conditional move)
  - Polynomial evaluation is the bottleneck (multiple SFPMAD ops)
  - Even if segment selection doubles, polynomial speedup dominates
- **Verification**: Measure with Tracy profiling

**Risk 4: Correctness Regression**
- **Symptom**: Numerical differences between single-eval and dual-eval
- **Mitigation**: Comprehensive testing (Phase 6)
- **Rollback**: Revert to single-eval via compile flag

**Risk 5: Code Size Explosion**
- **Symptom**: Kernel size exceeds instruction cache, causing thrashing
- **Mitigation**: Dual-eval roughly doubles code size of inner loop only (not entire kernel)
- **Measurement**: Compare kernel binary sizes before/after
- **Threshold**: If kernel size > 32KB, consider disabling for that configuration

---

## Rollback Plan

If critical issues arise:

**Immediate Rollback** (same day):
```bash
# Remove USE_DUAL_EVAL flag from all kernel builds
git diff HEAD~1 kernels/compute/ > dual_eval.patch
git checkout HEAD~1 kernels/compute/
./build_metal.sh --build-programming-examples
# Verify tests pass
pytest test.py -k lut_activation
```

**Selective Rollback** (keep some activations):
```python
# In kernel generation script
DUAL_EVAL_ENABLED_ACTIVATIONS = [
    # 'gelu',  # Disabled due to issue
    'sigmoid',  # Keep working ones
    'tanh',
]
```

**Graceful Degradation**:
- If dual-eval causes issues on specific hardware (e.g., Wormhole), add hardware-specific ifdef:
```cpp
#if defined(USE_DUAL_EVAL) && !defined(ARCH_WORMHOLE)
    // Use dual-eval
#else
    // Fall back to single-eval
#endif
```

---

## Success Metrics

### Quantitative Goals
1. **Performance**: Geometric mean speedup ≥ 1.5x across all tested configurations
2. **Correctness**: 100% pass rate on regression tests (bit-exact match with baseline)
3. **Coverage**: Dual-eval enabled for ≥ 80% of activation functions by end of rollout
4. **Code Size**: Kernel binary size increase < 50% (acceptable tradeoff for 1.5x+ speedup)

### Qualitative Goals
1. **Maintainability**: Dual-eval code is well-commented and easy to understand
2. **Extensibility**: Adding new activation functions doesn't require dual-eval reimplementation
3. **Debuggability**: Disassembly and profiling clearly show performance characteristics

---

## Execution Checklist

### Pre-Implementation
- [ ] Read and understand existing code (piecewise_generic.cpp, piecewise_generic_specialized.cpp)
- [ ] Set up Tracy profiling environment (if not already available)
- [ ] Identify test harness for running activation function benchmarks
- [ ] Create feature branch: `git checkout -b feature/dual-eval-horner`

### Implementation (Phases 1-5)
- [ ] Phase 1: Implement `eval_polynomial_dual<>` for all degrees (0-16, 32)
- [ ] Phase 2: Implement `piecewise_generic_lut_dual<>` (generic version)
- [ ] Phase 3: Implement `piecewise_staggered_horner_lut_dual<>` (staggered Horner variant)
- [ ] Phase 4: Implement specialized dual versions (4/8/16/32 segments)
- [ ] Phase 5: Update dispatcher to route to dual versions

### Compilation Verification
- [ ] Compile with `USE_DUAL_EVAL` undefined (baseline) - must succeed
- [ ] Compile with `USE_DUAL_EVAL` defined - must succeed
- [ ] Verify both produce valid kernel binaries

### Testing (Phases 6-7)
- [ ] Phase 6: Run correctness tests (bit-exact comparison)
- [ ] Phase 7.1: Add Tracy profiling if needed
- [ ] Phase 7.2: Run performance benchmarks, collect results
- [ ] Phase 7.3: Dump and inspect disassembly for representative config

### Rollout (Phase 8)
- [ ] Enable dual-eval for pilot activations (sigmoid, tanh)
- [ ] Monitor for 48 hours, check for issues
- [ ] Expand to more activations incrementally
- [ ] Document final configuration in README or CLAUDE.md

### Documentation
- [ ] Update kernel documentation with dual-eval explanation
- [ ] Add comment blocks explaining interleaving strategy
- [ ] Update performance numbers in project documentation
- [ ] Create before/after comparison plots

---

## File Checklist

Files modified in this implementation:

1. **`kernels/compute/piecewise_generic.cpp`** (Lines 316-492)
   - Add: `eval_polynomial_dual<>` template (~150 lines)
   - Add: `piecewise_generic_lut_dual<>` function (~120 lines)
   - Add: `piecewise_staggered_horner_lut_dual<>` function (~100 lines)

2. **`kernels/compute/piecewise_generic_specialized.cpp`** (Lines 31-762)
   - Add: `piecewise_generic_lut_specialized_4_dual<>` (~100 lines)
   - Add: `piecewise_generic_lut_specialized_8_dual<>` (~130 lines)
   - Add: `piecewise_generic_lut_specialized_16_dual<>` (~190 lines)
   - Add: `piecewise_generic_lut_specialized_32_dual<>` (~320 lines)
   - Modify: `piecewise_generic_lut_dispatch<>` dispatcher (~30 lines changed)

3. **Test/benchmark scripts** (location depends on project structure)
   - Add: Correctness test comparing single vs dual eval
   - Add: Performance benchmark script
   - Add: Disassembly inspection script

**Total Estimated LOC**: ~1200 new lines (heavily based on existing code via copy-modify pattern)

---

## Open Questions

1. **Q**: Should we use dual-eval for segment selection too (not just polynomial eval)?
   **A**: No, start simple. Segment selection is cheap, polynomial eval is expensive. Optimize the bottleneck first.

2. **Q**: What if x1 and x2 are in the same segment most of the time?
   **A**: Then dual-eval pays off even more! Initial eval in segment 0 uses dual, and if both stay in segment 0 (no re-eval needed), we get full 2x speedup.

3. **Q**: Should we process 4 x-vectors at once (quad-eval)?
   **A**: No, register pressure would be prohibitive. Two is the sweet spot (4 result registers, 2 x registers, 2 x_clamped registers, plus range reduction state = ~12 registers). Four would need ~20+ registers.

4. **Q**: Does this work on Wormhole? Blackhole?
   **A**: Should work on both. The Wormhole SFPU compiler bug only affects `v_if` cascades in runtime loops, which we avoid via manual unrolling (existing workaround). Blackhole has the same 2-cycle SFPMAD latency, so speedup should be identical.

5. **Q**: How do we measure cycle counts accurately?
   **A**: Use Tracy profiling or hardware performance counters. Alternatively, measure end-to-end kernel runtime and compute throughput (tiles/sec).

---

## Contact for Questions

If you encounter issues or ambiguities:
- **Code owners**: Check git blame on piecewise_generic.cpp for original authors
- **Hardware experts**: For SFPU pipeline questions, consult TT-Metalium kernel team
- **Performance**: For benchmarking setup, consult performance engineering team

**This plan should be treated as a living document.** Update it as you discover new information during implementation.

---

## Appendix: Example Disassembly

### Expected Single-Eval Disassembly (Degree 4)
```asm
; result = coeffs[4]
LREG r0, 0x12345678  ; Load coeffs[4] immediate

; result = result * x + coeffs[3]
LREG r1, 0xABCDEF00  ; Load coeffs[3] immediate
SFPMAD r0, r0, r2, r1  ; r0 = r0 * r2(x) + r1, takes 2 cycles

; result = result * x + coeffs[2]
LREG r1, 0x11111111  ; Load coeffs[2] immediate (must wait for SFPMAD)
SFPMAD r0, r0, r2, r1  ; r0 = r0 * r2(x) + r1, takes 2 cycles (waits for prev)

; ... repeat for coeffs[1] and coeffs[0] ...
; Total: ~10 cycles (2 per Horner step + overhead)
```

### Expected Dual-Eval Disassembly (Degree 4)
```asm
; result1 = coeffs[4], result2 = coeffs[4]
LREG r0, 0x12345678  ; Load coeffs[4] immediate (shared)
MOV r1, r0           ; Copy to result2

; result1 = result1 * x1 + coeffs[3]
LREG r4, 0xABCDEF00  ; Load coeffs[3] immediate
SFPMAD r0, r0, r2, r4  ; r0 = r0 * r2(x1) + r4, starts cycle 0

; result2 = result2 * x2 + coeffs[3]
SFPMAD r1, r1, r3, r4  ; r1 = r1 * r3(x2) + r4, starts cycle 0 (parallel!)

; result1 = result1 * x1 + coeffs[2]
LREG r4, 0x11111111  ; Load coeffs[2] immediate
SFPMAD r0, r0, r2, r4  ; starts cycle 2 (after r0 ready)

; result2 = result2 * x2 + coeffs[2]
SFPMAD r1, r1, r3, r4  ; starts cycle 2 (after r1 ready)

; ... repeat for coeffs[1] and coeffs[0] ...
; Total: ~11 cycles for TWO values (1.8x speedup vs 20 cycles for two single-evals)
```

**Key Observation**: The two SFPMAD operations overlap in cycles 0-1, 2-3, etc., hiding each other's latency.

---

**END OF IMPLEMENTATION PLAN**
