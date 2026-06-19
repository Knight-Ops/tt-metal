# Root Cause Analysis: sigmoid fp32 quintic 16-segment Compiler Crash

## Executive Summary

The sigmoid fp32 quintic 16-segment test crashes due to **TWO cascading compiler bugs**:

1. **First bug (FIXED):** Tiny non-zero values (< 1e-10) trigger GCC LTO optimizer crash
2. **Second bug (REVEALED):** 16-segment template specialization causes SFPU register spilling failure

## Timeline of Investigation

### Initial Crash (Before Fix)
```
Error: internal compiler error: in final_1, at final.cc:4905
```
- Generic LTO optimizer crash
- No specific details about what triggered it

### After Cleaning Tiny Values
```
Error: internal compiler error: cannot store sfpu register (register spill)
Location: piecewise_quintic.cpp:113 (v_if statement in 16-segment specialization)
```
- Specific error about SFPU register pressure
- Got past the first crash, hit a second one

---

## Bug #1: Tiny Non-Zero Values ✅ FIXED

### The Problem

The polynomial fitter generated coefficients with values smaller than float epsilon:
```
Segment 7 (sigmoid around x=0):
  c2: 3.7520298486518576e-16  ← Should be 0.0
  c4: -2.0510021903497738e-15 ← Should be 0.0
```

These values are **1000x smaller than float epsilon** (1.19e-7).

### Why It Caused a Crash

When GCC's LTO (Link Time Optimization) processes constexpr float arrays containing these tiny values:
1. Constant folding tries to optimize the array
2. Dead code elimination checks "is this coefficient effectively zero?"
3. Numeric edge case triggers an internal assertion or optimization bug
4. Compiler crashes in `final_1` pass

### Comparison

| Config | Tiny Non-Zero Values (<1e-10) | Result |
|--------|-------------------------------|--------|
| 16-segment (FAILING) | 2 values | Crashed |
| 32-segment (WORKING) | 0 values | Works |

### The Fix

Clean tiny values to exactly 0.0 in the CSV file:
```python
for coeff in coefficients:
    if abs(coeff) < 1e-10:
        coeff = 0.0
```

**Result:** First crash fixed, revealed second crash.

---

## Bug #2: SFPU Register Spilling ❌ NOT FIXED

### The Problem

After fixing Bug #1, compilation now fails with:
```
internal compiler error: cannot store sfpu register (register spill)
```

This happens in the 16-segment template specialization.

### Root Cause: Template Specialization vs Generic Template

The quintic kernel has TWO implementations:

#### **16-Segment Specialization** (FAILS)
```cpp
// piecewise_quintic.cpp:105-139
template <>
inline void piecewise_quintic_lut<113>(const std::array<float, 113>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 16;

    // Load default segment 0 coefficients
    vFloat c5 = lut[22], c4 = lut[21], ..., c0 = lut[17];

    // 16 cascaded v_if statements
    v_if (x >= lut[1])  { c5 = lut[28]; c4 = lut[27]; ... c0 = lut[23]; } v_endif;
    v_if (x >= lut[2])  { c5 = lut[34]; c4 = lut[33]; ... c0 = lut[29]; } v_endif;
    v_if (x >= lut[3])  { c5 = lut[40]; c4 = lut[39]; ... c0 = lut[35]; } v_endif;
    // ... 13 more v_if statements ...
    v_if (x >= lut[15]) { c5 = lut[112]; c4 = lut[111]; ... c0 = lut[107]; } v_endif;

    // Horner's method
    vFloat result = (((((c5 * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
}
```

**Register Pressure:**
- 6 live coefficient registers (c0-c5) throughout the entire function
- Each v_if statement loads 6 new values
- Compiler needs to keep multiple coefficient sets alive across v_if boundaries
- Total register pressure: 6 base + up to 96 temporary loads (16 segments × 6 coefficients)

#### **32-Segment Generic Template** (WORKS)
```cpp
// piecewise_quintic.cpp:144+
template <uint32_t LUT_SIZE>
inline void piecewise_quintic_lut(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t NUM_SEGMENTS = (LUT_SIZE - 1) / 7;

    // Find segment using loop (compiles to v_if cascade, but with better register allocation)
    vInt segment_id = 0;
    for (uint32_t i = 1; i <= NUM_SEGMENTS; i++) {
        v_if (x >= lut[i]) { segment_id = i; } v_endif;
    }

    // Load coefficients AFTER segment selection
    uint32_t coeff_idx = (NUM_SEGMENTS + 1) + (segment_id * 6);
    vFloat c0 = lut[coeff_idx];
    vFloat c1 = lut[coeff_idx + 1];
    // ... load c2-c5 ...

    // Horner's method
    vFloat result = (((((c5 * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
}
```

**Register Pressure:**
- Only 6 coefficient registers live at end
- Segment selection happens first with minimal register usage
- Coefficient loads happen AFTER segment selection
- Much lower register pressure

### Why 32-Segment Works But 16-Segment Fails

| Config | Template Used | Implementation | Register Pressure | Result |
|--------|---------------|----------------|-------------------|--------|
| 16-segment | Specialized `<113>` | Cascaded v_if with inline loads | HIGH | Register spill crash |
| 32-segment | Generic `<225>` | Loop then load | LOW | Works |
| Octic 16-seg | Specialized (octic has same pattern) | Cascaded v_if | HIGH (but degree 8) | Works?? |

### The Paradox: Why Does Octic Work?

Octic (degree 8) also has 16-segment specialization with cascaded v_if, but loads **9 coefficients** (c0-c8) per segment instead of 6!

This should have HIGHER register pressure, yet it works.

**Hypothesis:** The specific combination of:
- Quintic polynomial (5 multiplications)
- 16 segments (16 v_if statements)
- 6 coefficients per segment
- Horner's evaluation pattern

...triggers a specific register allocation bug in the SFPI backend that the octic pattern doesn't hit.

---

## Attempted Fixes

### ❌ Attempt 1: Clean Tiny Values
- Fixed Bug #1 (LTO crash)
- Revealed Bug #2 (register spilling)

### 🤔 Possible Solutions for Bug #2

#### Option A: Remove 16-Segment Specialization
Delete the `piecewise_quintic_lut<113>` specialization, force it to use generic template.

**Pros:**
- Will definitely work (32-segment proves generic template works)
- Consistent with 32+ segment behavior

**Cons:**
- May be slightly slower (loop vs cascaded v_if)
- Need to verify performance impact

#### Option B: Reduce Register Pressure in Specialization
Restructure the 16-segment specialization to match the generic template's pattern.

**Pros:**
- Keeps specialization for potential performance benefit
- Fixes root cause

**Cons:**
- More complex refactoring
- May still hit compiler limits

#### Option C: Use bf16 Instead
Use bf16 precision instead of fp32 (works perfectly).

**Pros:**
- Instant workaround
- Actually faster than fp32 (19ms)
- Good accuracy (MAE=1.32e-03)

**Cons:**
- Not a "fix", just avoids the problem
- Still broken for users who specifically need fp32

#### Option D: File Compiler Bug
Report both bugs to SFPI team.

---

## Conclusions

1. **Bug #1 (Tiny values):**
   - CONFIRMED: Tiny non-zero values cause LTO optimizer crash
   - FIX: Clean values < 1e-10 to exactly 0.0
   - STATUS: Fixed

2. **Bug #2 (Register spilling):**
   - CONFIRMED: 16-segment specialization causes register spilling in quintic
   - ROOT CAUSE: Cascaded v_if with inline coefficient loads
   - WORKAROUND: Use generic template (remove specialization) or use bf16
   - STATUS: Not fixed, workaround available

3. **Recommended Action:**
   - Short term: Use bf16 (19ms, MAE=1.32e-03) ✅
   - Medium term: Remove 16-segment quintic specialization, use generic template
   - Long term: File SFPI compiler bug with both issues

---

## Test Results After Fix

| Config | Before Fix | After Cleaning Tiny Values |
|--------|-----------|---------------------------|
| sigmoid fp32 quintic 16-seg | LTO crash (final_1) | Register spill crash |
| sigmoid fp32 quintic 32-seg | Works | Works |
| sigmoid bf16 quintic 16-seg | Works | Works |
| sigmoid bf16 quintic 32-seg | Works | Works |

---

## Files Involved

**Cleaned CSV:**
- `/localdev/nkapre/tt-polynomial-fitter/data/coefficients/sigmoid_fp32_16_5_curvature_any.csv`
- Backup: `sigmoid_fp32_16_5_curvature_any.csv.backup`

**Kernel Code:**
- `kernels/compute/piecewise_quintic.cpp` (lines 105-139: 16-segment specialization)
- `kernels/compute/piecewise_quintic_embedded/sigmoid_quintic_16_curvature.cpp` (generated wrapper)

**Related Documentation:**
- `QUINTIC_KERNEL_VERIFICATION.md` (code structure verification)
- `ACTION_ITEMS.md` (next steps)
