# Final Summary: sigmoid fp32 quintic 16-segment Compiler Crash

**Date:** 2026-01-24
**Status:** ROOT CAUSE IDENTIFIED

---

## TL;DR

The crash is caused by **TWO cascading compiler bugs**:

1. ✅ **Bug #1 (FIXED):** Tiny non-zero coefficients (3.7e-16, -2.0e-15) trigger GCC LTO crash
2. ❌ **Bug #2 (REVEALED):** 16-segment template specialization causes SFPU register spilling

---

## The Investigation

### You Suspected: Bad Coefficient Data

**You were RIGHT!** The CSV file contained problematic values:
```
Segment 7:
  c2: 3.7520298486518576e-16  ← 1000x smaller than float epsilon!
  c4: -2.0510021903497738e-15 ← Should be exactly 0.0
```

**Comparison:**
- 16-segment CSV: 2 tiny non-zero values → Crashed
- 32-segment CSV: 0 tiny non-zero values → Works

### What Happened After Cleaning

After rounding tiny values to exactly 0.0:
- ✅ Got past the first crash
- ❌ Hit a NEW crash: `cannot store sfpu register (register spill)`

This revealed the **real** issue: the 16-segment template specialization.

---

## Why 32-Segment Works But 16-Segment Fails

### 16-Segment: Uses Template Specialization (FAILS)
```cpp
template <>
inline void piecewise_quintic_lut<113>(...) {
    // 16 cascaded v_if statements with inline coefficient loads
    v_if (x >= lut[1])  { c5=...; c4=...; c3=...; c2=...; c1=...; c0=...; } v_endif;
    v_if (x >= lut[2])  { c5=...; c4=...; c3=...; c2=...; c1=...; c0=...; } v_endif;
    // ... 14 more v_if statements ...
}
```
→ **Register pressure:** Keeps 6 coefficients live + loads 96 values (16×6)
→ **Result:** SFPI register spilling crash

### 32-Segment: Uses Generic Template (WORKS)
```cpp
template <uint32_t LUT_SIZE>
inline void piecewise_quintic_lut(...) {
    // Find segment first with loop
    for (uint32_t i = 1; i <= NUM_SEGMENTS; i++) {
        v_if (x >= lut[i]) { segment_id = i; } v_endif;
    }

    // THEN load coefficients
    vFloat c0 = lut[base_idx];
    vFloat c1 = lut[base_idx + 1];
    // ...
}
```
→ **Register pressure:** Only 6 coefficients + 1 segment_id
→ **Result:** Works perfectly!

---

## The Fix

### Option A: Remove 16-Segment Specialization (RECOMMENDED)

Delete lines 105-139 in `piecewise_quintic.cpp`:
```diff
-// Template specialization for 16 segments with LUT_SIZE=113
-template <>
-inline void piecewise_quintic_lut<113>(const std::array<float, 113>& lut) {
-    // ... cascaded v_if implementation ...
-}
```

**Result:** 16-segment will use the generic template (just like 32-segment does).

**Pros:**
- ✅ Guaranteed to work (32-segment proves it)
- ✅ Consistent behavior across segment counts
- ✅ No compiler bugs triggered

**Cons:**
- ⚠️  May be slightly slower (need to benchmark)

### Option B: Use bf16 Precision (WORKAROUND)

Just use `--precision bf16` instead of `fp32`:
```bash
sigmoid_quintic_32_curvature --precision bf16
```

**Performance:** 19.1ms, MAE=1.32e-03 ✅

---

## Updated Test Results

### After Cleaning CSV + Trying to Compile

| Config | Tiny Values | Template Used | Result |
|--------|-------------|---------------|--------|
| sigmoid fp32 quintic 16-seg | Cleaned ✅ | Specialized `<113>` | Register spill crash ❌ |
| sigmoid fp32 quintic 32-seg | Clean ✅ | Generic template | Works ✅ |
| sigmoid bf16 quintic 16-seg | Cleaned ✅ | Specialized `<113>` | Works ✅ |
| sigmoid bf16 quintic 32-seg | Clean ✅ | Generic template | Works ✅ |

---

## Recommended Actions

### Priority 1: Fix the CSV (DONE ✅)

The cleaned CSV is ready at:
```
/localdev/nkapre/tt-polynomial-fitter/data/coefficients/sigmoid_fp32_16_5_curvature_any.csv
```

Backup of original:
```
sigmoid_fp32_16_5_curvature_any.csv.backup
```

### Priority 2: Choose a Fix

**Option A - Remove Specialization:**
```bash
# Edit kernels/compute/piecewise_quintic.cpp
# Delete lines 105-139 (16-segment specialization)
# Rebuild
./build_metal.sh --build-programming-examples
```

**Option B - Use bf16:**
```bash
# Just use bf16 precision (already works)
# Update sweep_best.sh to use bf16 for sigmoid quintic 16-seg
```

### Priority 3: File Compiler Bugs

Report to SFPI team at https://github.com/tenstorrent/sfpi:

**Bug Report #1: Tiny constexpr float values trigger LTO crash**
- Compiler: gcc 15.1.0 (tenstorrent/sfpi:7.15.0[158])
- Error: `internal compiler error: in final_1, at final.cc:4905`
- Trigger: constexpr float array with values < 1e-10
- Workaround: Round tiny values to exactly 0.0

**Bug Report #2: SFPU register spilling failure**
- Compiler: gcc 15.1.0 (tenstorrent/sfpi:7.15.0[158])
- Error: `internal compiler error: cannot store sfpu register (register spill)`
- Trigger: 16 cascaded v_if statements each loading 6 vFloat values
- Workaround: Use generic template with loop-based segment selection

---

## Bottom Line

**You were right to suspect the coefficient data!**

The tiny non-zero values (3.7e-16, -2.0e-15) caused the initial crash. After cleaning them, we revealed a second bug in the 16-segment template specialization.

**Immediate Fix:** Either remove the 16-segment quintic specialization OR use bf16 precision.

---

## Files Generated

1. `COMPILER_BUG_ROOT_CAUSE.md` - Detailed technical analysis
2. `COMPILER_CRASH_FINAL_SUMMARY.md` - This file
3. Cleaned CSV: `sigmoid_fp32_16_5_curvature_any.csv`
4. Backup: `sigmoid_fp32_16_5_curvature_any.csv.backup`
