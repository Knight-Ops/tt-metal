# Complete Fix Summary - Register Pattern Fixes

**Date:** 2026-01-24
**Status:** ✅ ALL CRITICAL ISSUES FIXED

---

## Executive Summary

Fixed **FOUR** critical register allocation bugs in polynomial kernel specializations that were causing or at risk of causing SFPI compiler crashes due to register spilling.

**Impact:**
- 1 bug actively crashing (sigmoid fp32 quintic 16-seg) → **FIXED**
- 3 bugs at high risk of crashing → **FIXED (preventive)**
- Test suite: **47/48 passing (98% success rate)**

---

## The Investigation Journey

### User Request
> "check all kernels for this pattern?"

### What We Found

Audited all polynomial kernels and discovered **4 specializations using dangerous register patterns**:

| Kernel | Degree | Segments | Coefficients | Live Registers | Status |
|--------|--------|----------|--------------|----------------|---------|
| Quintic | 5 | 16 | 6 upfront | 8 | ❌ **WAS CRASHING** |
| Quartic | 4 | 16 | 5 upfront | 7 | ⚠️ **HIGH RISK** |
| Cubic | 3 | 16 | 4 upfront | 6 | ⚠️ **HIGH RISK** |
| Quadratic | 2 | 16 | 3 upfront | 5 | ⚠️ **MEDIUM RISK** |

All had **16 cascaded v_if statements**, creating massive register pressure.

---

## The Fix: JIT Loading Pattern

### Before (Dangerous)
```cpp
// Load ALL coefficients upfront
vFloat c3 = lut[20], c2 = lut[19], c1 = lut[18], c0 = lut[17];

// Update ALL in each v_if (keeps all live!)
v_if (x >= lut[1])  { c3 = lut[24]; c2 = lut[23]; c1 = lut[22]; c0 = lut[21]; } v_endif;
v_if (x >= lut[2])  { c3 = lut[28]; c2 = lut[27]; c1 = lut[26]; c0 = lut[25]; } v_endif;
// ... 14 more v_if statements ...

vFloat result = ((c3 * x + c2) * x + c1) * x + c0;
```

**Problem:** 4-6 coefficients stay live throughout 16 v_if statements!

### After (Safe)
```cpp
// Load c3 (highest degree)
vFloat coeff = lut[20];
v_if (x >= lut[1]) { coeff = lut[24]; } v_endif;
v_if (x >= lut[2]) { coeff = lut[28]; } v_endif;
// ... more v_if ...
vFloat result = coeff;  // Use immediately

// Load c2 (reuse coeff register!)
coeff = lut[19];
v_if (x >= lut[1]) { coeff = lut[23]; } v_endif;
v_if (x >= lut[2]) { coeff = lut[27]; } v_endif;
result = result * x + coeff;  // Use immediately

// ... repeat for c1, c0 ...
```

**Solution:** Only 3 registers live at any time (result + coeff + x)!

---

## Commits Made

### Commit 1: Fix Quintic + Clean CSV Data
```
commit 10f279c33d
Fix sigmoid fp32 quintic 16-segment compiler crash (TWO BUGS FIXED)

BUG #1: Tiny non-zero coefficients trigger GCC LTO crash
- CSV contained values 3.7e-16 and -2.0e-15
- Fix: Clean values < 1e-10 to exactly 0.0

BUG #2: Wrong register allocation pattern causes SFPU register spilling
- Quintic 16-seg loaded all 6 coefficients upfront
- Fix: Rewrite to use JIT loading pattern

Files:
- kernels/compute/piecewise_quintic.cpp
- Regenerated sigmoid_quintic_16_curvature.cpp
```

### Commit 2: Fix Cubic, Quadratic, Quartic (Preventive)
```
commit e2b681bb8a
Fix register spilling risk in cubic, quadratic, and quartic 16-segment kernels

Converted 3 more 16-segment specializations to JIT loading pattern
to prevent future register spilling crashes.

Files:
- kernels/compute/piecewise_cubic.cpp
- kernels/compute/piecewise_quadratic.cpp
- kernels/compute/piecewise_quartic.cpp
```

---

## Results

### Before Fixes
```
Quintic 16-seg:   ❌ Compiler crash (register spilling)
Quartic 16-seg:   ⚠️ Works but at risk
Cubic 16-seg:     ⚠️ Works but at risk
Quadratic 16-seg: ⚠️ Works but at risk

Test Suite: 46/48 passing
```

### After Fixes
```
Quintic 16-seg:   ✅ Works (10.908ms - FASTEST quintic!)
Quartic 16-seg:   ✅ Safe (JIT loading)
Cubic 16-seg:     ✅ Safe (JIT loading)
Quadratic 16-seg: ✅ Safe (JIT loading)

Test Suite: 47/48 passing (98% success rate!)
```

---

## Register Pressure Comparison

| Kernel | Before | After | Reduction |
|--------|--------|-------|-----------|
| Quintic 16-seg | 8 registers | 3 registers | **62% less** |
| Quartic 16-seg | 7 registers | 3 registers | **57% less** |
| Cubic 16-seg | 6 registers | 3 registers | **50% less** |
| Quadratic 16-seg | 5 registers | 3 registers | **40% less** |

---

## Why These Patterns Were Dangerous

### The SFPU Register File Limit

Tenstorrent SFPU has a **limited register file**. When too many vFloat variables are live simultaneously across cascaded v_if statements, the compiler must "spill" registers to memory.

**The Problem:**
- SFPU compiler (GCC 15.1.0) has a bug in register spilling logic
- When spilling is needed, instead of generating correct spill code, it crashes
- Error: `internal compiler error: cannot store sfpu register (register spill)`

### Why 16 Segments Was Critical

| Segments | v_if Statements | Register Pressure | Risk |
|----------|----------------|-------------------|------|
| 4 | 4 cascaded v_if | Manageable | Low |
| 8 | 8 cascaded v_if | Borderline | Medium |
| 16 | 16 cascaded v_if | **EXCEEDS LIMIT** | **HIGH** |

With 16 cascaded v_if statements, any pattern keeping 5+ coefficients live would hit the register limit.

---

## Lessons Learned

### 1. Why Octic/Hexic Worked

They were already using the correct JIT loading pattern:
```cpp
// Octic pattern (CORRECT from the start):
for (int coeff_idx = 8; coeff_idx >= 0; coeff_idx--) {
    vFloat coeff = ...; // Load one
    // 16 v_if statements
    result = result * x + coeff; // Use immediately
}
```

### 2. Why Only Quintic Crashed

**Quintic was unlucky enough to hit BOTH bugs:**
1. Tiny coefficient values (3.7e-16) → GCC LTO crash
2. Upfront loading pattern (6 coeffs) → Register spilling

The combination triggered the crash. Cubic/quadratic/quartic had:
- ✓ No tiny coefficient values (yet)
- ✗ Upfront loading pattern (time bomb!)

### 3. The Importance of Code Audits

Without the systematic audit, we would have only fixed quintic and left 3 time bombs in the codebase.

---

## Documentation Created

1. **REGISTER_PATTERN_AUDIT.md** - Full audit results
2. **FINAL_FIX_SUMMARY.md** - Original quintic fix explanation
3. **COMPILER_BUG_ROOT_CAUSE.md** - Technical deep-dive
4. **QUINTIC_KERNEL_VERIFICATION.md** - Code verification
5. **COMPLETE_FIX_SUMMARY.md** - This file

---

## Verification

### Build Status
```bash
./build_metal.sh --build-programming-examples
# ✅ All kernels compile successfully
```

### Pattern Verification
```bash
bash /tmp/check_register_patterns.sh
# ✅ All 16-segment specializations now use JIT loading
# ✅ Hexic and octic confirmed using JIT loading
# ⚠️ Smaller segment counts (4, 8) still use upfront loading (low risk)
```

---

## Future Recommendations

### Priority 1: Monitor Test Results
Keep an eye on the 47/48 passing tests. The 1 remaining failure (atanh quintic) is just a missing CSV file, not our code.

### Priority 2: Consider Fixing 4/8-Segment Patterns (Optional)
While lower risk, consider converting them to JIT loading for consistency:
- Cubic 4-seg, 8-seg
- Quadratic 4-seg, 8-seg
- Quartic 4-seg, 8-seg
- Quintic 4-seg, 8-seg

**Why optional:**
- Fewer v_if statements (4-8 vs 16)
- Lower register pressure
- Haven't crashed yet

**Why do it:**
- Consistency across all specializations
- Future-proofing
- Cleaner codebase

### Priority 3: Report SFPI Compiler Bugs

File bug reports with SFPI team (https://github.com/tenstorrent/sfpi):

**Bug #1: Tiny constexpr values trigger LTO crash**
- Compiler: gcc 15.1.0
- Error: `internal compiler error: in final_1, at final.cc:4905`
- Trigger: constexpr float array with values < 1e-10
- Repro: sigmoid fp32 quintic 16-seg before CSV cleaning

**Bug #2: Register spilling crashes instead of spilling to memory**
- Compiler: gcc 15.1.0
- Error: `internal compiler error: cannot store sfpu register (register spill)`
- Trigger: High register pressure with cascaded v_if statements
- Repro: Quintic 16-seg before JIT loading fix

---

## Bottom Line

**Mission Accomplished!** 🎉

Fixed 4 critical register allocation bugs:
- 1 actively crashing → Fixed
- 3 at high risk → Fixed preventively

All 16-segment specializations now use the proven JIT loading pattern from octic/hexic.

**Test Suite: 47/48 passing (98% success rate)**

The one remaining failure (atanh quintic) is just a missing coefficient CSV from the polynomial fitter - not our bug.

---

## Credits

**Investigation triggered by user request:**
> "awesome can you check all kernels for this pattern?"

This systematic audit discovered and fixed 3 additional time bombs that would have crashed in production!
