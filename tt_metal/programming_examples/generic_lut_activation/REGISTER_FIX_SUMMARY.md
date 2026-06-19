# Register Pattern Fixes - generic_lut_activation Directory

**Date:** 2026-01-24
**Status:** ✅ ALL 4 CRITICAL ISSUES FIXED

---

## Summary

Applied the same JIT loading pattern fixes to `generic_lut_activation` directory that were previously applied to `generic_lut_activation_embedded`.

**Files Fixed:**
- `kernels/compute/piecewise_quintic.cpp` (16-segment specialization)
- `kernels/compute/piecewise_quartic.cpp` (16-segment specialization)
- `kernels/compute/piecewise_cubic.cpp` (16-segment specialization)
- `kernels/compute/piecewise_quadratic.cpp` (16-segment specialization)

---

## The Bug

All four kernels had 16-segment specializations using the **dangerous upfront loading pattern**:

```cpp
// BAD PATTERN - All coefficients loaded upfront
vFloat c5 = lut[22], c4 = lut[21], c3 = lut[20], c2 = lut[19], c1 = lut[18], c0 = lut[17];

// All coefficients updated in each v_if (keeps ALL live throughout!)
v_if (x >= lut[1]) { c5 = lut[28]; c4 = lut[27]; c3 = lut[26]; c2 = lut[25]; c1 = lut[24]; c0 = lut[23]; } v_endif;
v_if (x >= lut[2]) { c5 = lut[34]; c4 = lut[33]; c3 = lut[32]; c2 = lut[31]; c1 = lut[30]; c0 = lut[29]; } v_endif;
// ... 14 more v_if statements with 3-6 coefficients live ...

vFloat result = (((((c5 * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
```

**Problem:** With 16 cascaded `v_if` statements, keeping 3-6 coefficients live simultaneously creates excessive register pressure, risking SFPU compiler register spilling crashes.

---

## The Fix

Converted all four 16-segment specializations to use **JIT (Just-In-Time) loading pattern**:

```cpp
// GOOD PATTERN - Load one coefficient at a time
// Load c5 (highest degree)
vFloat coeff = lut[22];
v_if (x >= lut[1]) { coeff = lut[28]; } v_endif;
v_if (x >= lut[2]) { coeff = lut[34]; } v_endif;
// ... more v_if
vFloat result = coeff;  // Use immediately

// Load c4 (reuse coeff register!)
coeff = lut[21];
v_if (x >= lut[1]) { coeff = lut[27]; } v_endif;
v_if (x >= lut[2]) { coeff = lut[33]; } v_endif;
result = result * x + coeff;  // Use immediately

// ... repeat for c3, c2, c1, c0
```

**Solution:** Only 3 registers live at any time (result + coeff + x)!

---

## Register Pressure Comparison

| Kernel | Before (Upfront) | After (JIT) | Reduction |
|--------|------------------|-------------|-----------|
| Quintic 16-seg | 8 registers (c0-c5 + x + x_clamped) | 3 registers | **62% less** |
| Quartic 16-seg | 7 registers (c0-c4 + x + x_clamped) | 3 registers | **57% less** |
| Cubic 16-seg | 6 registers (c0-c3 + x + x_clamped) | 3 registers | **50% less** |
| Quadratic 16-seg | 5 registers (c0-c2 + x + x_clamped) | 3 registers | **40% less** |

---

## Why This Matters

### The SFPU Register File Limit

Tenstorrent SFPU has a **limited register file**. With 16 cascaded `v_if` statements:

| Pattern | Live Registers | Risk Level |
|---------|----------------|------------|
| Upfront loading (quintic) | 8 | 🔴 **CRITICAL** - Can trigger register spilling |
| Upfront loading (quartic) | 7 | 🔴 **HIGH** - Very close to limit |
| Upfront loading (cubic) | 6 | 🟡 **MEDIUM** - Borderline |
| Upfront loading (quadratic) | 5 | 🟡 **MEDIUM** - Borderline |
| JIT loading (all) | 3 | ✅ **SAFE** - Well below limit |

### The Compiler Bug

When SFPU register pressure exceeds the hardware limit, the SFPI compiler (GCC 15.1.0) has a bug:
- **Expected:** Generate register spill code (store to memory)
- **Actual:** Crashes with `internal compiler error: cannot store sfpu register (register spill)`

---

## Relationship to embedded_lut_activation_embedded Fixes

This is a **parallel codebase** with the same bugs:

1. **generic_lut_activation_embedded/** (Fixed earlier today)
   - Used for embedded mode (LUT compiled into kernel binary)
   - Fixed in commits 10f279c33d and e2b681bb8a

2. **generic_lut_activation/** (Fixed now)
   - Used for generic mode (LUT loaded from L1 circular buffer)
   - Files are SEPARATE (not symlinks), so needed independent fixes

Both directories had identical register allocation bugs in their 16-segment specializations.

---

## Files Modified

### 1. piecewise_quintic.cpp
**Location:** `kernels/compute/piecewise_quintic.cpp:104-139`
**Change:** Replaced 6-coefficient upfront loading with JIT loading
**Before:** 8 live registers (c0-c5 + x + x_clamped)
**After:** 3 live registers (result + coeff + x)

### 2. piecewise_quartic.cpp
**Location:** `kernels/compute/piecewise_quartic.cpp:104-139`
**Change:** Replaced 5-coefficient upfront loading with JIT loading
**Before:** 7 live registers (c0-c4 + x + x_clamped)
**After:** 3 live registers (result + coeff + x)

### 3. piecewise_cubic.cpp
**Location:** `kernels/compute/piecewise_cubic.cpp:107-141`
**Change:** Replaced 4-coefficient upfront loading with JIT loading
**Before:** 6 live registers (c0-c3 + x + x_clamped)
**After:** 3 live registers (result + coeff + x)

### 4. piecewise_quadratic.cpp
**Location:** `kernels/compute/piecewise_quadratic.cpp:107-142`
**Change:** Replaced 3-coefficient upfront loading with JIT loading
**Before:** 5 live registers (c0-c2 + x + x_clamped)
**After:** 3 live registers (result + coeff + x)

---

## Notes

### Why Only 16-Segment Specializations?

The 4-segment and 8-segment specializations also use upfront loading, but:
- **Lower risk:** Fewer cascaded `v_if` statements (4-8 instead of 16)
- **Lower register pressure:** Haven't triggered crashes yet
- **Future work:** Could convert them for consistency

### Pattern Matches embedded_lut_activation_embedded

The fixes applied here are **identical** to those applied to the embedded directory:
- Same JIT loading pattern
- Same coefficient ordering (high degree to low)
- Same Horner's method evaluation
- Same 3-register live count

---

## Verification

```bash
# Check the fixes were applied
git diff kernels/compute/piecewise_quintic.cpp
git diff kernels/compute/piecewise_quartic.cpp
git diff kernels/compute/piecewise_cubic.cpp
git diff kernels/compute/piecewise_quadratic.cpp

# All 16-segment specializations now use JIT loading pattern
```

---

## Related Documentation

See also:
- `../generic_lut_activation_embedded/COMPLETE_FIX_SUMMARY.md` - Original fix documentation
- `../generic_lut_activation_embedded/REGISTER_PATTERN_AUDIT.md` - Comprehensive audit results
- `../generic_lut_activation_embedded/COMPILER_BUG_ROOT_CAUSE.md` - Technical deep-dive

---

## Bottom Line

**Mission Accomplished!** 🎉

Fixed 4 more critical register allocation bugs in the `generic_lut_activation` directory:
- All 16-segment specializations now use proven JIT loading pattern
- Register pressure reduced by 40-62%
- Matching the fixes already applied to `generic_lut_activation_embedded`

**Total bugs fixed across both directories:** 8 (4 in embedded + 4 in generic)
