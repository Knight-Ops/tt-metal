# Kernel Bug Analysis: Hardware vs Theoretical MAE Discrepancies

## Summary of Failures (OLD CODE - Before Fixes)

### Working Well (HW ≈ Theoretical):
| Kernel      | Depths Working | Pattern |
|-------------|----------------|---------|
| Linear      | 4, 8, 16, 32   | ✓ All depths work (0.92-1.53×) |
| Quadratic   | 4, 8, 16, 32   | ✓ All depths work (1.05-1.68×) |
| Cubic       | 4, 8, 32       | ✓ Three depths work (0.98-1.06×) |
| Hexic       | 4              | ✓ Only depth 4 works (0.13×) |
| Octic       | 4              | ✓ Only depth 4 works (0.07×) |

### Catastrophic Failures (HW >> Theoretical):
| Kernel | Depth | HW MAE  | Theo MAE | Ratio | Bug Explanation |
|--------|-------|---------|----------|-------|-----------------|
| Cubic  | 16    | 0.277   | 0.001    | 263×  | Unknown - needs investigation |
| Hexic  | 8     | 1.810   | 0.007    | 266×  | v_if cascade (7×7=49 statements) |
| Hexic  | 16    | 0.500   | 0.005    | 97×   | v_if cascade (7×15=105 statements) |
| Octic  | 8     | 0.250   | 0.016    | 16×   | x_clamped bug + v_if cascade (9×7=63 statements) |
| Octic  | 16    | 0.500   | 0.003    | 188×  | x_clamped bug + v_if cascade (9×15=135 statements) |
| Octic  | 32    | 0.500   | 0.004    | 127×  | x_clamped bug + v_if cascade (9×31=279 statements) |

## Root Causes

### 1. Octic Bug: Using x_clamped for Evaluation (FIXED in commit 6e4b83366d)

**Problem:**
```cpp
// BEFORE (WRONG):
v_if (x_clamped >= lut[seg]) {
    result = lut[lut_base] * x_clamped;  // ❌ Using clamped value
    result = result + lut[lut_base + 1];
    result = result * x_clamped;         // ❌ Multiplying by clamped value
    // ...
}
```

**Impact:**
- x_clamped should ONLY be used for segment selection
- Using it for polynomial evaluation causes large errors
- Polynomials are fitted to the original x range, not clamped values
- Explains why HW errors are ~0.25-0.50 (saturation effects)

**Fix:**
```cpp
// AFTER (CORRECT):
v_if (x_clamped >= lut[seg]) {
    result = lut[lut_base] * x;          // ✓ Using original x
    result = result + lut[lut_base + 1];
    result = result * x;                 // ✓ Original x throughout
    // ...
}
```

### 2. Hexic Bug: v_if Cascade Hardware Limits (FIXED in commit 26a7aa5183)

**Problem:**
The OLD hexic generic template used 7 separate coefficient-loading cascades:

```cpp
// BEFORE (BAD): Load coefficient 'a'
vFloat coeff = lut[COEFF_OFFSET];
for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
    v_if (x_clamped >= lut[seg]) {
        coeff = lut[COEFF_OFFSET + seg * 7];
    }
    v_endif;
}
vFloat result = coeff;

// Repeat 6 more times for b, c, d, e, f, g...
```

**v_if Statement Count:**
| Depth | Segments | v_if Count (7 cascades) | Result |
|-------|----------|-------------------------|--------|
| 4     | 4        | 7 × 3 = 21              | ✓ Works |
| 8     | 8        | 7 × 7 = 49              | ❌ Fails |
| 16    | 16       | 7 × 15 = 105            | ❌ Fails |
| 32    | 32       | 7 × 31 = 217            | ❌ Would fail |

**Fix:**
Compute entire polynomial inside each v_if (like cubic generic does):

```cpp
// AFTER (GOOD): Compute full polynomial per segment
vFloat result = lut[COEFF_OFFSET] * x;
result = result + lut[COEFF_OFFSET + 1];
// ... complete polynomial for segment 0

for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
    v_if (x_clamped >= lut[seg]) {
        result = lut[lut_base] * x;
        result = result + lut[lut_base + 1];
        // ... complete polynomial inside v_if
    }
    v_endif;
}
```

Now only N-1 v_if statements instead of 7×(N-1).

### 3. Cubic Depth 16 Mystery (NEEDS INVESTIGATION)

**Observation:**
- Cubic generic template already uses the GOOD pattern
- Depth 4 (specialization): Works ✓
- Depth 8 (specialization): Works ✓
- Depth 16 (generic): **FAILS 263×** ❌
- Depth 32 (generic): Works ✓

**LUT Sizes:**
```python
Depth  4: LUT_SIZE =  21  (uses specialization)
Depth  8: LUT_SIZE =  41  (uses specialization)
Depth 16: LUT_SIZE =  81  (uses generic template)
Depth 32: LUT_SIZE = 161  (uses generic template)
```

**Possible Causes:**
1. Test execution failure specific to depth 16?
2. Segmentation type mismatch (adaptive vs uniform)?
3. LUT generation issue for depth 16?
4. Hardware-specific issue at that particular depth?

**HW MAE = 0.277** suggests this might be a different kind of failure than the v_if cascade issues (which show 0.5 saturation).

## Kernel Implementation Patterns

### Pattern A: Load Coefficients, Update via v_if (OLD - Used by specializations)
```cpp
vFloat a = lut[5], b = lut[6], c = lut[7], d = lut[8];
v_if (x_clamped >= lut[1]) { a = lut[9]; b = lut[10]; c = lut[11]; d = lut[12]; } v_endif;
v_if (x_clamped >= lut[2]) { a = lut[13]; b = lut[14]; c = lut[15]; d = lut[16]; } v_endif;
// ...
vFloat result = ((a * x + b) * x + c) * x + d;
```
- Works for small depths (4, 8)
- Fails for larger depths with more coefficients (hexic: 7 coeffs, octic: 9 coeffs)

### Pattern B: Compute Full Polynomial Inside v_if (GOOD - Used by generics)
```cpp
vFloat result = lut[COEFF_OFFSET] * x;
result = result + lut[COEFF_OFFSET + 1];
// ... default segment 0

for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
    v_if (x_clamped >= lut[seg]) {
        result = lut[lut_base] * x;
        result = result + lut[lut_base + 1];
        // ... complete polynomial
    }
    v_endif;
}
```
- Only N-1 v_if statements
- Scales to any depth
- Used by cubic/quadratic/linear generic templates

## Fixes Applied

| Commit | Fix | Impact |
|--------|-----|--------|
| 6e4b83366d | Fix octic generic to use `x` instead of `x_clamped` | Should fix octic depths 8, 16, 32 |
| 26a7aa5183 | Fix hexic generic v_if cascade (217→31 for depth 32) | Should fix hexic depths 8, 16, 32 |
| 9310d6a566 | Fix embedded octic LUT_SIZE=144 register pressure | Optimization for embedded |
| b03fd26083 | Fix embedded hexic all variants v_if cascade | Fixes embedded hexic |

## Next Steps

1. **Re-run sweep on Blackhole** with fixed kernels:
   ```bash
   cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation
   ./sweep_piecewise.sh --activation sigmoid
   ```

2. **Verify fixes:**
   - Hexic depths 8, 16, 32 should now show HW ≈ Theoretical
   - Octic depths 8, 16, 32 should now show HW ≈ Theoretical

3. **Investigate cubic depth 16:**
   - Check if issue reproduces with new code
   - Examine LUT generation for depth 16
   - Check test execution logs
   - Compare with depth 32 which works fine

## Expected Results After Fixes

| Kernel    | Depth | OLD HW MAE | Theo MAE | OLD Ratio | Expected New Ratio |
|-----------|-------|------------|----------|-----------|-------------------|
| Hexic     | 8     | 1.810      | 0.007    | 266×      | ~1-2× ✓           |
| Hexic     | 16    | 0.500      | 0.005    | 97×       | ~1-2× ✓           |
| Octic     | 8     | 0.250      | 0.016    | 16×       | ~1-2× ✓           |
| Octic     | 16    | 0.500      | 0.003    | 188×      | ~1-2× ✓           |
| Octic     | 32    | 0.500      | 0.004    | 127×      | ~1-2× ✓           |
| Cubic     | 16    | 0.277      | 0.001    | 263×      | Need to investigate |
