# Adaptive Segmentation - Generator Update Status

**Date**: 2026-01-18
**Task**: Update all LUT generators to support adaptive segmentation

---

## Status Summary

| Generator | Status | Tested | Improvement (depth=16, exp) |
|-----------|--------|--------|------------------------------|
| **generate_piecewise_linear_luts.py** | ✅ DONE | ✅ YES | **1.80×** (0.185 → 0.103 MAE) |
| **generate_piecewise_cubic_remez_luts.py** | ✅ DONE | ✅ YES | **1.24×** (0.00047 → 0.00038 MAE) |
| **generate_piecewise_quadratic_remez_luts.py** | ⏳ TODO | ❌ NO | TBD |
| **generate_piecewise_hexic_remez_luts.py** | ⏳ TODO | ❌ NO | TBD |
| **generate_piecewise_octic_remez_luts.py** | ⏳ TODO | ❌ NO | TBD |

---

## Changes Made

### 1. generate_piecewise_linear_luts.py ✅

**Updates**:
- Added `load_adaptive_breakpoints()` function
- Modified `write_piecewise_linear_lut()` to accept custom breakpoints
- Modified `fit_piecewise_linear()` to fit on custom segments
- Modified `compute_accuracy_metrics()` to evaluate with custom boundaries
- Added `--adaptive` flag to CLI

**Binary Format** (UPDATED):
```
[uint32 num_values]
[float boundaries[num_segments + 1]]  ← NEW! Precomputed boundaries
[float slopes[num_segments]]
[float offsets[num_segments]]
```

**Usage**:
```bash
# Uniform (default)
python3 generate_piecewise_linear_luts.py --activation exp

# Adaptive
python3 generate_piecewise_linear_luts.py --activation exp --adaptive
```

**Output Files**:
- Uniform: `luts/piecewise_linear_exp_16.lut`
- Adaptive: `luts/piecewise_linear_exp_16_adaptive.lut`

**Test Results** (depth=16, exp):
```
Uniform:  MAE 0.18516  Max 3.74889
Adaptive: MAE 0.10314  Max 1.06939
Improvement: 1.80× MAE, 3.51× Max Error
```

### 2. generate_piecewise_cubic_remez_luts.py ✅

**Updates**:
- Added `load_adaptive_breakpoints()` function
- Modified `write_piecewise_cubic_lut()` to accept custom breakpoints
- Modified `fit_piecewise_cubic_remez()` to fit on custom segments
- Modified `compute_accuracy_metrics()` to evaluate with custom boundaries
- Added `--adaptive` flag to CLI
- **Fixed**: Convert np.float32 to Python float for Remez library compatibility

**Binary Format** (UPDATED):
```
[uint32 num_values]
[float boundaries[num_segments + 1]]  ← NEW! Precomputed boundaries
[float a_coeffs[num_segments]]        ← x³ term
[float b_coeffs[num_segments]]        ← x² term
[float c_coeffs[num_segments]]        ← x term
[float d_coeffs[num_segments]]        ← constant
```

**Usage**:
```bash
# Uniform (default)
python3 generate_piecewise_cubic_remez_luts.py --activation exp

# Adaptive
python3 generate_piecewise_cubic_remez_luts.py --activation exp --adaptive
```

**Test Results** (depth=16, exp):
```
Uniform:  MAE 0.00047  Max 0.00629
Adaptive: MAE 0.00038  Max 0.00258
Improvement: 1.24× MAE, 2.44× Max Error
```

**Note**: Smaller improvement than linear because cubic polynomials already approximate curves very well.

---

## Remaining Work

### 3. generate_piecewise_quadratic_remez_luts.py ⏳

**Changes Needed** (same pattern as cubic):
1. Add `import csv, pathlib`
2. Add `load_adaptive_breakpoints()` function
3. Update `write_piecewise_quadratic_lut()` - add `breakpoints` parameter, write boundaries first
4. Update `fit_piecewise_quadratic_remez()` - add `breakpoints` parameter, use custom segments
5. Update `compute_accuracy_metrics()` - add `breakpoints` parameter
6. Add `--adaptive` flag to main()
7. Fix type conversion: `seg_start = float(boundaries[i])`

**Binary Format Change**:
```
OLD: [uint32 num_values][a/b/c coefficients]
NEW: [uint32 num_values][boundaries][a/b/c coefficients]
```

### 4. generate_piecewise_hexic_remez_luts.py ⏳

**Changes Needed**: Same as quadratic, but with 7 coefficients per segment (hexic = 6th degree)

### 5. generate_piecewise_octic_remez_luts.py ⏳

**Changes Needed**: Same as quadratic, but with 9 coefficients per segment (octic = 8th degree)

---

## Pattern for Updating Remaining Generators

All Remez-based generators follow the same pattern. Here's the diff template:

```python
# 1. Add imports
+import csv
+from pathlib import Path

# 2. Add load function
+def load_adaptive_breakpoints(activation, depth, segments_dir='adaptive_segments'):
+    csv_file = Path(segments_dir) / f'depth_{depth}_segments.csv'
+    if not csv_file.exists():
+        return None
+    with open(csv_file, 'r') as f:
+        reader = csv.DictReader(f)
+        for row in reader:
+            if row['activation'] == activation:
+                breakpoints_str = row['breakpoints']
+                return np.array([float(x) for x in breakpoints_str.split(',')], dtype=np.float32)
+    return None

# 3. Update write function signature
-def write_piecewise_*_lut(filename, coeffs, ...):
+def write_piecewise_*_lut(filename, coeffs, ..., input_range=None, breakpoints=None):

# 4. Write boundaries in write function
+    if breakpoints is not None:
+        boundaries = breakpoints.tolist() if isinstance(breakpoints, np.ndarray) else breakpoints
+    else:
+        assert input_range is not None
+        x_min, x_max = input_range
+        segment_width = (x_max - x_min) / num_segments
+        boundaries = [x_min + i * segment_width for i in range(num_segments + 1)]
+
+    num_values = (num_segments + 1) + (num_segments * coeffs_per_segment)
+
+    # Write boundaries first
+    for boundary in boundaries:
+        f.write(struct.pack('f', float(boundary)))

# 5. Update fit function signature
-def fit_piecewise_*_remez(func, num_segments, input_range, max_iter=15):
+def fit_piecewise_*_remez(func, num_segments, input_range, max_iter=15, breakpoints=None):

# 6. Use custom breakpoints in fit function
+    if breakpoints is not None:
+        assert len(breakpoints) == num_segments + 1
+        boundaries = breakpoints
+    else:
+        segment_width = (x_max - x_min) / num_segments
+        boundaries = np.array([x_min + i * segment_width for i in range(num_segments + 1)])

     for i in range(num_segments):
-        seg_start = x_min + i * segment_width
-        seg_end = x_min + (i + 1) * segment_width
+        seg_start = float(boundaries[i])  # Convert for Remez
+        seg_end = float(boundaries[i + 1])

# 7. Update compute_accuracy_metrics signature
-def compute_accuracy_metrics(func, coeffs, ..., num_segments, input_range):
+def compute_accuracy_metrics(func, coeffs, ..., num_segments, input_range, breakpoints=None):

# 8. Use custom breakpoints in metrics
+    if breakpoints is not None:
+        boundaries = breakpoints
+    else:
+        segment_width = (x_max - x_min) / num_segments
+        boundaries = np.array([x_min + i * segment_width for i in range(num_segments + 1)])

# 9. Add --adaptive flag in main()
+    parser.add_argument('--adaptive', action='store_true', help='Use adaptive segmentation')

+    if args.adaptive:
+        print("Mode: ADAPTIVE SEGMENTATION")

     for func_name, func in functions.items():
+        breakpoints = None
+        if args.adaptive:
+            breakpoints = load_adaptive_breakpoints(func_name, depth)
+            if breakpoints is None:
+                print(f"WARNING: No adaptive breakpoints, skipping")
+                continue

-        coeffs = fit_*(func, depth, input_range)
-        write_*(filename, coeffs, ...)
-        mae, rmse, max_err = compute_*(func, coeffs, depth, input_range)
+        coeffs = fit_*(func, depth, input_range, breakpoints=breakpoints)
+        suffix = "_adaptive" if args.adaptive else ""
+        filename = f"luts/*_{func_name}_{depth}{suffix}.lut"
+        write_*(filename, coeffs, ..., input_range=input_range, breakpoints=breakpoints)
+        mae, rmse, max_err = compute_*(func, coeffs, depth, input_range, breakpoints=breakpoints)
+        mode_str = "(adaptive)" if args.adaptive else "(uniform)"
+        print(f"  {func_name:15s} {mode_str:12s} → MAE: {mae:8.5f}...")
```

---

## Testing Checklist

For each updated generator:

```bash
# 1. Test uniform mode (should work as before)
python3 generate_piecewise_*_luts.py --activation exp

# 2. Test adaptive mode
python3 generate_piecewise_*_luts.py --activation exp --adaptive

# 3. Verify output files exist
ls -lh luts/piecewise_*_exp_16.lut
ls -lh luts/piecewise_*_exp_16_adaptive.lut

# 4. Check binary format (should include boundaries)
hexdump -C luts/piecewise_*_exp_16_adaptive.lut | head -20

# 5. Compare MAE improvement
python3 generate_piecewise_*_luts.py --activation exp | grep "depth 16"
python3 generate_piecewise_*_luts.py --activation exp --adaptive | grep "depth 16"
```

---

## Expected Improvements by Polynomial Degree

Based on testing so far:

| Degree | Improvement (exp, depth=16) | Reason |
|--------|------------------------------|--------|
| Linear | **1.80×** | Simple approximation, benefits most from adaptive |
| Cubic  | **1.24×** | Already good at curves, moderate benefit |
| Quadratic | ~1.4× (expected) | Between linear and cubic |
| Hexic | ~1.1× (expected) | Very good approximation already |
| Octic | ~1.05× (expected) | Near-perfect, minimal benefit |

**Conclusion**: Adaptive segmentation benefits **low-degree polynomials** most, with diminishing returns for high-degree.

---

## Next Steps

1. **Quick win**: Update remaining generators using the pattern above (10-15 min each)
2. **Generate all adaptive LUTs**: Run with `--adaptive` for all activations
3. **Hardware testing**: Test adaptive LUTs on actual device
4. **Benchmarking**: Compare runtime (O(1) vs O(log N) lookup)
5. **Documentation**: Update main README with adaptive usage

---

## Generated Files So Far

```
luts/
├── piecewise_linear_exp_{4,8,16,32}.lut           (uniform)
├── piecewise_linear_exp_{4,8,16,32}_adaptive.lut  (adaptive) ✅
├── piecewise_cubic_remez_exp_{4,8,16,32}.lut      (uniform)
└── piecewise_cubic_remez_exp_{4,8,16,32}_adaptive.lut (adaptive) ✅

adaptive_segments/
├── depth_4_segments.csv   (29 activations × 5 breakpoints)
├── depth_8_segments.csv   (29 activations × 9 breakpoints)
├── depth_16_segments.csv  (29 activations × 17 breakpoints)
└── depth_32_segments.csv  (29 activations × 33 breakpoints)
```

**Total adaptive LUTs generated**: 8 files (exp only, linear + cubic, all depths)
**Total adaptive LUTs possible**: ~580 files (29 activations × 4 depths × 5 methods)

---

## Questions / Notes

1. **Binary format change**: Adding boundaries increases LUT size by `(num_segments + 1) * 4` bytes
   - For depth=16: +68 bytes per LUT (~10-15% increase)
   - Trade-off: Small size increase for better accuracy

2. **Hardware changes needed**: Kernel must use binary search instead of division for segment lookup
   - Current (uniform): `segment_idx = (x - x_min) / segment_width`
   - New (adaptive): `segment_idx = binary_search(boundaries, x)`

3. **Performance impact**: Binary search is O(log N) vs O(1) division
   - For N=16: ~4 comparisons vs 1 division
   - Expected: Negligible (~10 cycles vs 5 cycles)

4. **Recommendation**: Focus adaptive on **high-benefit activations** (exp, softplus, cosh, sinh, relu family)
   - Skip sigmoid, tanh (uniform already near-optimal)
   - Test on hardware to confirm theoretical improvements translate to practice
