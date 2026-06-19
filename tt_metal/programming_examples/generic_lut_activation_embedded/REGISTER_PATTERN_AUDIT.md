# Register Pattern Audit - All Polynomial Kernels

**Date:** 2026-01-24
**Status:** 3 CRITICAL issues found, multiple warnings

---

## Executive Summary

Audited all polynomial kernel specializations for register allocation patterns.

**CRITICAL FINDINGS:**
- 3 kernels use dangerous pattern with 16+ segments
- These are at HIGH RISK of register spilling crashes
- Need immediate fixes

---

## Results by Kernel

### ✅ SAFE - Using JIT Loading Pattern

#### Hexic (Degree 6)
- ✓ 4-segment specialization: JIT loading
- ✓ 8-segment specialization: JIT loading
- **Status:** SAFE

#### Octic (Degree 8)
- ✓ 4-segment specialization: JIT loading
- ✓ 8-segment specialization: JIT loading
- **Status:** SAFE

#### Quintic (Degree 5) - FIXED
- ⚠️ 4-segment: Upfront loading (low risk)
- ⚠️ 8-segment: Upfront loading (low risk)
- ✓ 16-segment: **JIT loading (FIXED!)**
- **Status:** 16-segment fixed, others acceptable

---

### ❌ CRITICAL - 16+ Segments with Upfront Loading

#### Cubic (Degree 3)
```
File: kernels/compute/piecewise_cubic.cpp
Line: 107

⚠️ 4-segment (line 44):  4 coeffs upfront, 4 segments  → Borderline
⚠️ 8-segment (line 78):  4 coeffs upfront, 8 segments  → Borderline
❌ 16-segment (line 107): 4 coeffs upfront, 16 segments → CRITICAL!
```

**Register Pressure:**
- 4 coefficients (c0, c1, c2, c3) loaded upfront
- x, x_clamped
- Total: ~6 live registers throughout 16 v_if statements

**Risk Level:** 🔴 **HIGH** - Same pattern that crashed quintic!

---

#### Quadratic (Degree 2)
```
File: kernels/compute/piecewise_quadratic.cpp
Line: 107

⚠️ 4-segment (line 33):  3 coeffs upfront, 4 segments  → Borderline
⚠️ 8-segment (line 76):  3 coeffs upfront, 8 segments  → Borderline
❌ 16-segment (line 107): 3 coeffs upfront, 16 segments → CRITICAL!
```

**Register Pressure:**
- 3 coefficients (c0, c1, c2) loaded upfront
- x, x_clamped
- Total: ~5 live registers throughout 16 v_if statements

**Risk Level:** 🔴 **HIGH** - Same pattern that crashed quintic!

---

#### Quartic (Degree 4)
```
File: kernels/compute/piecewise_quartic.cpp
Line: 104

⚠️ 4-segment (line 41):  5 coeffs upfront, 4 segments  → Borderline
⚠️ 8-segment (line 75):  5 coeffs upfront, 8 segments  → Borderline
❌ 16-segment (line 104): 5 coeffs upfront, 16 segments → CRITICAL!
```

**Register Pressure:**
- 5 coefficients (c0, c1, c2, c3, c4) loaded upfront
- x, x_clamped
- Total: ~7 live registers throughout 16 v_if statements

**Risk Level:** 🔴 **HIGH** - Same pattern that crashed quintic!

---

### ⚠️ BORDERLINE - Lower Segments with Upfront Loading

These use upfront loading but with fewer segments (4-8), so register pressure is lower:

- Cubic 4-segment, 8-segment
- Quadratic 4-segment, 8-segment
- Quartic 4-segment, 8-segment
- Quintic 4-segment, 8-segment

**Risk Level:** 🟡 **MEDIUM** - May work but not optimal

**Recommendation:** Convert to JIT loading for consistency and future-proofing.

---

## Pattern Comparison

### BAD Pattern (Upfront Loading)
```cpp
// Load ALL coefficients at start
vFloat c3 = lut[20], c2 = lut[19], c1 = lut[18], c0 = lut[17];

// Update ALL in each v_if (keeps all live!)
v_if (x >= lut[1]) { c3 = lut[24]; c2 = lut[23]; c1 = lut[22]; c0 = lut[21]; } v_endif;
v_if (x >= lut[2]) { c3 = lut[28]; c2 = lut[27]; c1 = lut[26]; c0 = lut[25]; } v_endif;
// ... 14 more v_if statements with all coefficients live ...

vFloat result = ((c3 * x + c2) * x + c1) * x + c0;
```

**Live Registers:** 4 coeffs + x + x_clamped = 6 registers

---

### GOOD Pattern (JIT Loading)
```cpp
// Load c3 (highest degree)
vFloat coeff = lut[20];
v_if (x >= lut[1]) { coeff = lut[24]; } v_endif;
v_if (x >= lut[2]) { coeff = lut[28]; } v_endif;
// ... more v_if
vFloat result = coeff;

// Load c2 (reuse coeff register!)
coeff = lut[19];
v_if (x >= lut[1]) { coeff = lut[23]; } v_endif;
v_if (x >= lut[2]) { coeff = lut[27]; } v_endif;
result = result * x + coeff;

// ... repeat for c1, c0
```

**Live Registers:** result + coeff + x = 3 registers

---

## Recommended Fixes

### Priority 1: Fix Critical 16-Segment Specializations

**Files to fix:**
1. `kernels/compute/piecewise_cubic.cpp` (line 107)
2. `kernels/compute/piecewise_quadratic.cpp` (line 107)
3. `kernels/compute/piecewise_quartic.cpp` (line 104)

**Action:** Convert to JIT loading pattern (same as quintic fix)

**Expected Impact:**
- Prevents potential register spilling crashes
- Reduces live register count from 5-7 to 3
- Matches proven octic/hexic pattern

---

### Priority 2: Fix 4-Segment and 8-Segment (Optional)

**Why:** Consistency and future-proofing

**Files:**
- All specializations currently using upfront loading
- Lower risk since fewer v_if statements, but still not optimal

---

## Testing Strategy

After fixes:
1. Run full sweep: `./sweep_best.sh`
2. Focus on 16-segment tests (most critical)
3. Verify no performance regression
4. Check that all tests still pass

---

## Why These Weren't Caught Earlier

1. **Lower coefficient counts:** Cubic (4 coeffs) and quadratic (3 coeffs) have fewer live registers than quintic (6 coeffs), so they stayed below the spilling threshold

2. **Luck with coefficient values:** These kernels might not have hit the tiny-value LTO bug yet

3. **Not tested as heavily:** Sigmoid quintic was the first to expose both bugs together

4. **Borderline cases:** With 5-7 live registers, these are close to the SFPU register limit but haven't crashed yet

---

## Bottom Line

**We found 3 time bombs!** 💣

The cubic, quadratic, and quartic 16-segment specializations use the same dangerous pattern that crashed quintic. They haven't crashed **yet**, but they're at high risk.

**Recommendation:** Fix the 3 critical 16-segment specializations immediately.
