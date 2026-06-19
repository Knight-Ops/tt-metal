# Rational Kernel Register Optimization Strategies

**Date:** 2026-02-04
**Issue:** n8_d8_s2 and n9_d8_s2 fail with SFPU register spilling error
**Root Cause:** Excessive register pressure in 2-segment high-degree rational approximations

---

## Problem Analysis

### Current Implementation

**File:** `kernels/compute/piecewise_rational.cpp`

```cpp
// Line 108-114: eval_rational function
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline vFloat eval_rational(const float* num_coeffs, const float* den_coeffs, vFloat x) {
    vFloat numerator = eval_poly<NUM_DEGREE>(num_coeffs, x);    // 9 coeffs for degree 8
    vFloat denominator = eval_poly<DEN_DEGREE>(den_coeffs, x);  // 9 coeffs for degree 8
    vFloat recip = ckernel::sfpu::sfpu_reciprocal<false>(denominator);
    return numerator * recip;
}

// Line 146-151: 2-segment evaluation
v_if (x_clamped >= lut[1]) {
    const float* seg_num = &lut[COEFF_OFFSET + 1 * COEFFS_PER_SEGMENT];
    const float* seg_den = &lut[COEFF_OFFSET + 1 * COEFFS_PER_SEGMENT + NUM_COEFFS];
    result = eval_rational<NUM_DEGREE, DEN_DEGREE>(seg_num, seg_den, x);  // FAILS HERE
}
v_endif;
```

### Register Pressure Breakdown

For **n8_d8_s2** (numerator degree 8, denominator degree 8, 2 segments):

| Phase | Live Registers | Description |
|-------|----------------|-------------|
| Segment 0 eval | ~6 | result + intermediate values from eval_rational |
| Inside v_if | ~12-15 | segment 0 result + numerator eval + denominator eval + recip + x |
| **Peak pressure** | **~15+ registers** | **EXCEEDS SFPU LIMIT (~16 registers)** |

**Why it fails:**
1. `eval_poly<8>` for numerator creates 4-5 intermediate vFloat values (Horner's method)
2. Numerator result stays alive during denominator evaluation (another 4-5 intermediates)
3. Reciprocal operation adds 1 more register
4. All this happens inside a v_if block that keeps segment 0's result alive
5. Total: **15+ vFloat registers live simultaneously → register spilling → compiler error**

---

## Optimization Strategies

### Strategy A: Sequential Evaluation with Immediate Consumption ⭐ **RECOMMENDED**

**Concept:** Evaluate denominator first, immediately compute reciprocal, then evaluate numerator.

**Why it works:** Denominator dies after reciprocal, so numerator evaluation doesn't overlap.

```cpp
// Option A1: Reorder operations to minimize overlap
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline vFloat eval_rational(const float* num_coeffs, const float* den_coeffs, vFloat x) {
    // Evaluate denominator FIRST
    vFloat denominator = eval_poly<DEN_DEGREE>(den_coeffs, x);

    // Immediately consume it via reciprocal (denominator now dead)
    vFloat recip = ckernel::sfpu::sfpu_reciprocal<false>(denominator);

    // Now evaluate numerator (doesn't overlap with denominator)
    vFloat numerator = eval_poly<NUM_DEGREE>(num_coeffs, x);

    // Final multiply (only 2 values live: numerator + recip)
    return numerator * recip;
}
```

**Register count:** Peak ~6-7 registers (down from 15+)
- Denominator eval: 4-5 intermediate + x = 5-6 registers
- Reciprocal: 1 register (denominator dead)
- Numerator eval: 4-5 intermediate + x + recip = 6-7 registers
- Final multiply: 2 registers

**Pros:**
- Simple one-line change (just reorder operations)
- No structural changes to v_if logic
- Works for all degree combinations

**Cons:**
- Still ~7 registers (improved but not optimal)
- May still fail for n9_d9 or higher degrees

**Estimated success:** ✅ **Should fix n8_d8_s2 and n9_d8_s2**

---

### Strategy B: Store-and-Reload Pattern

**Concept:** Store reciprocal to memory, evaluate numerator, reload reciprocal.

```cpp
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline vFloat eval_rational_lowreg(const float* num_coeffs, const float* den_coeffs, vFloat x) {
    // Evaluate denominator and reciprocal
    vFloat denominator = eval_poly<DEN_DEGREE>(den_coeffs, x);
    vFloat recip = ckernel::sfpu::sfpu_reciprocal<false>(denominator);

    // Store reciprocal temporarily (frees up registers)
    // Note: This requires a scratch buffer in L1 or using dst_reg cleverly
    float recip_scalar = recip[0];  // Store first element as example

    // Evaluate numerator (recip not in register)
    vFloat numerator = eval_poly<NUM_DEGREE>(num_coeffs, x);

    // Reload reciprocal and multiply
    vFloat recip_reloaded = recip_scalar;
    return numerator * recip_reloaded;
}
```

**Register count:** Peak ~5-6 registers

**Pros:**
- Minimal peak register usage

**Cons:**
- Requires memory access (slower)
- Complex to implement correctly in SFPU
- May not be worth the complexity

**Estimated success:** ✅ Works, but overkill

---

### Strategy C: JIT Coefficient Loading (Like Polynomial Fix) ⭐ **MOST ROBUST**

**Concept:** Load coefficients one at a time, evaluate on-the-fly. This is the pattern from `REGISTER_FIX_SUMMARY.md`.

```cpp
// Replace eval_rational with inline coefficient loading
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline vFloat eval_rational_jit(const float* num_coeffs, const float* den_coeffs, vFloat x) {
    // === DENOMINATOR EVALUATION WITH JIT LOADING ===
    // Load highest degree coefficient for denominator
    vFloat coeff = den_coeffs[DEN_DEGREE];
    vFloat denom_result = coeff;

    // Horner's method: load one coefficient at a time
    for (int i = DEN_DEGREE - 1; i >= 0; i--) {
        coeff = den_coeffs[i];  // Reuse register
        denom_result = denom_result * x + coeff;
    }

    // Compute reciprocal (denominator dead after this)
    vFloat recip = ckernel::sfpu::sfpu_reciprocal<false>(denom_result);

    // === NUMERATOR EVALUATION WITH JIT LOADING ===
    coeff = num_coeffs[NUM_DEGREE];  // Reuse same register!
    vFloat numer_result = coeff;

    for (int i = NUM_DEGREE - 1; i >= 0; i--) {
        coeff = num_coeffs[i];  // Reuse register
        numer_result = numer_result * x + coeff;
    }

    // Final multiply
    return numer_result * recip;
}
```

**Register count:** Peak ~3-4 registers only! ✅
- During denominator eval: denom_result + coeff + x = 3 registers
- During reciprocal: recip + x = 2 registers
- During numerator eval: numer_result + coeff + x + recip = 4 registers
- Final: 2 registers

**Pros:**
- Minimal register usage (3-4 registers vs 15+)
- Proven pattern from existing polynomial fixes
- Works for ANY degree combination (even n16_d16)
- Matches the coding style in REGISTER_FIX_SUMMARY.md

**Cons:**
- More verbose code
- Loops instead of unrolled specializations (but for-loops should unroll at compile time)

**Estimated success:** ✅ **Guaranteed to work, most future-proof**

---

### Strategy D: Split Rational into Two Helper Functions

**Concept:** Create separate low-register versions for high degrees.

```cpp
// Separate evaluation for numerator and denominator
template <uint32_t DEGREE>
inline vFloat eval_poly_to_temp(const float* coeffs, vFloat x, vFloat& temp_storage) {
    vFloat result = eval_poly<DEGREE>(coeffs, x);
    temp_storage = result;  // Store result
    return result;
}

template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline vFloat eval_rational_split(const float* num_coeffs, const float* den_coeffs, vFloat x) {
    vFloat temp;

    // Evaluate denominator, store, compute reciprocal
    eval_poly_to_temp<DEN_DEGREE>(den_coeffs, x, temp);
    vFloat recip = ckernel::sfpu::sfpu_reciprocal<false>(temp);

    // Evaluate numerator (temp now available for reuse)
    eval_poly_to_temp<NUM_DEGREE>(num_coeffs, x, temp);

    return temp * recip;
}
```

**Register count:** Peak ~6-7 registers

**Pros:**
- Minimal code changes
- Compiler might optimize better

**Cons:**
- Requires passing temp storage around
- Not as clean as other solutions

**Estimated success:** ✅ Should work

---

### Strategy E: Optimize v_if Pattern (Avoid Default Segment)

**Concept:** Instead of computing segment 0 by default, branch on both segments.

**Current pattern (line 143-151):**
```cpp
// Compute segment 0 (ALWAYS executed)
vFloat result = eval_rational<NUM_DEGREE, DEN_DEGREE>(seg0_num, seg0_den, x);

// Maybe overwrite with segment 1
v_if (x_clamped >= lut[1]) {
    result = eval_rational<NUM_DEGREE, DEN_DEGREE>(seg_num, seg_den, x);
}
v_endif;
```

**Optimized pattern:**
```cpp
// Don't compute segment 0 by default
vFloat result;
bool computed = false;

// Compute segment 1 if needed
v_if (x_clamped >= lut[1]) {
    result = eval_rational<NUM_DEGREE, DEN_DEGREE>(seg1_num, seg1_den, x);
    computed = true;
}
v_endif;

// Compute segment 0 only if not already computed
v_if (!computed) {
    result = eval_rational<NUM_DEGREE, DEN_DEGREE>(seg0_num, seg0_den, x);
}
v_endif;
```

**Register count:** Same as before, but avoids double evaluation

**Pros:**
- Avoids wasted computation when segment 1 is used

**Cons:**
- Requires additional boolean flag
- More complex control flow
- **SFPU doesn't support scalar bool flags well**

**Estimated success:** ❌ **Not practical for SFPU**

---

## Recommended Implementation Plan

### Phase 1: Quick Fix (Strategy A) - **30 minutes**

Apply Strategy A (sequential evaluation) to `eval_rational`:

```cpp
// In piecewise_rational.cpp, line 108-115
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline vFloat eval_rational(const float* num_coeffs, const float* den_coeffs, vFloat x) {
    // CHANGE: Evaluate denominator FIRST, immediately consume via reciprocal
    vFloat denominator = eval_poly<DEN_DEGREE>(den_coeffs, x);
    vFloat recip = ckernel::sfpu::sfpu_reciprocal<false>(denominator);
    // CHANGE: Now evaluate numerator (denominator is dead)
    vFloat numerator = eval_poly<NUM_DEGREE>(num_coeffs, x);
    return numerator * recip;
}
```

**Test:** Rebuild and test n8_d8_s2, n9_d8_s2

**Expected outcome:** ✅ Should fix the issue

---

### Phase 2: Robust Fix (Strategy C) - **2-3 hours**

If Phase 1 doesn't fully solve it (or for future-proofing):

1. Create `eval_rational_jit` using JIT coefficient loading
2. Add compile-time dispatch to use JIT version for high degrees:
   ```cpp
   if constexpr (NUM_DEGREE >= 8 || DEN_DEGREE >= 8) {
       return eval_rational_jit<NUM_DEGREE, DEN_DEGREE>(num_coeffs, den_coeffs, x);
   } else {
       return eval_rational<NUM_DEGREE, DEN_DEGREE>(num_coeffs, den_coeffs, x);
   }
   ```
3. Test all degree combinations

**Expected outcome:** ✅ Guaranteed to work for all degrees

---

### Phase 3: Optimization (Optional) - **1 hour**

Apply JIT loading to 4-segment and 8-segment specializations for consistency:
- `piecewise_rational_lut_4`
- `piecewise_rational_lut_8`

**Expected outcome:** Lower register pressure across the board

---

## Register Pressure Summary Table

| Strategy | Peak Registers | Complexity | Robustness | Recommended? |
|----------|----------------|------------|------------|--------------|
| **Current (broken)** | 15+ | - | ❌ Fails | ❌ |
| **A: Sequential Eval** | 6-7 | Low | ✅ Good | ⭐ **START HERE** |
| **B: Store-Reload** | 5-6 | High | ✅ Good | ❌ Overkill |
| **C: JIT Loading** | 3-4 | Medium | ✅ Excellent | ⭐ **BEST LONG-TERM** |
| **D: Split Functions** | 6-7 | Medium | ✅ Good | ⚠️ Okay |
| **E: Optimize v_if** | 7+ | High | ❌ Not practical | ❌ |

---

## Testing Plan

### Test Cases

1. **n8_d8_s2** - Primary failing case
   - Activation: sigmoid, tanh, gelu
   - Expected: Should compile and run

2. **n9_d8_s2** - Secondary failing case
   - Activation: sigmoid, tanh
   - Expected: Should compile and run

3. **n7_d7_s2** - Boundary case (currently works)
   - Verify no regression

4. **n9_d9_s1** - Single segment high degree
   - Verify no regression

### Verification Commands

```bash
# Test n8_d8_s2
./sweep_rational.sh --activation sigmoid --tiles 256 | grep "n8_d8_s2"

# Test n9_d8_s2
./sweep_rational.sh --activation sigmoid --tiles 256 | grep "n9_d8_s2"

# Full sweep
./sweep_rational.sh
```

---

## Related Documentation

- `REGISTER_FIX_SUMMARY.md` - Polynomial kernel register fixes (same pattern)
- `2026-01-18.md` - Original register pressure discoveries
- `KERNEL_BUG_ANALYSIS.md` - SFPU compiler bug details

---

## Conclusion

**Recommended approach:**

1. **Start with Strategy A** (sequential evaluation) - simple, likely sufficient
2. **If needed, implement Strategy C** (JIT loading) - robust, proven pattern
3. **Test thoroughly** with all affected configurations

**Expected outcome:** n8_d8_s2 and n9_d8_s2 should compile and run successfully, with register pressure reduced from 15+ registers to 3-7 registers.
