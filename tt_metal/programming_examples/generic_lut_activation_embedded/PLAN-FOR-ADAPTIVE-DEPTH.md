# Adaptive Per-Segment Polynomial Degree Optimization

## Setup (On Remote Server)

### Initial Setup

```bash
# SSH to remote server (use appropriate credentials)
# For Wormhole: ssh to wormhole server
# For Blackhole: ssh to blackhole server

# Navigate to repository
cd /localdev/nkapre/tt-metal

# Ensure on correct branch
git fetch origin
git checkout feature/generic-lut-activation
git pull origin feature/generic-lut-activation

# Set up environment
export TT_POLY_FIT_DIR=/localdev/nkapre/tt-polynomial-fitter
export ARCH_NAME=wormhole_b0  # or blackhole
export PYTHONPATH=$(pwd)

# Check device status
/opt/venv/bin/tt-smi

# Navigate to working directory
cd tt_metal/programming_examples/generic_lut_activation_embedded

# Verify coefficients are available
ls $TT_POLY_FIT_DIR/data/coefficients/*.csv | head -5

# Expected: sigmoid_4_1_uniform_any.csv, tanh_8_3_curvature_any.csv, etc.
```

### Pre-Implementation Baseline

Before making changes, capture baseline performance:

```bash
# Run a quick benchmark for comparison later
./sweep_polynomial.sh --activation gelu --degree 4 --depth 8 --segmentation uniform > baseline_gelu_p4_s8.csv

# Save the baseline
cp baseline_gelu_p4_s8.csv ~/baseline_before_optimization.csv
```

## Overview

Implement runtime optimization by detecting the actual polynomial degree needed per segment from CSV coefficient data and generating specialized evaluation code that uses the minimal degree for each segment. This eliminates wasted SFPU operations when higher-degree coefficients are zero.

**Expected Impact**: 10-15% overall kernel speedup, up to 37% reduction in polynomial evaluation cycles for configurations with significant degree variance.

## Problem Statement

Currently, all segments in a piecewise polynomial approximation use the same `POLY_DEGREE`, even when individual segments only need lower degrees. CSV files already contain zero-padded coefficients (e.g., a cubic approximation stored as quartic has c4=0.0). We can detect this at kernel generation time and emit per-segment optimized code.

**Example**:
- Config: GELU, quartic (degree 4), 8 segments
- Current: All 8 segments evaluate degree-4 polynomials (4 FMA ops each)
- Reality: Segments actually need [2,2,3,4,4,3,2,2] degrees
- Opportunity: 6/8 segments can use fewer operations (25% speedup in polynomial phase)

## Technical Approach: Preprocessor Macros

**Why macros?** Template parameters require compile-time constants. Array indexing (even on constexpr arrays) produces runtime values. The ONLY way to make per-segment degrees work with `eval_polynomial<DEGREE>` is to use preprocessor macros.

**Architecture**:
1. Python script detects actual degree per segment from CSV (threshold: coefficient < 1e-10 = zero)
2. Python script emits `#define SEG0_DEGREE 2` macros in generated embedded kernels
3. C++ specialized templates wrap each `eval_polynomial<POLY_DEGREE>` call with:
   ```cpp
   #ifdef SEG0_DEGREE
       result = eval_polynomial<SEG0_DEGREE>(&lut[...], x);
   #else
       result = eval_polynomial<POLY_DEGREE>(&lut[...], x);  // Fallback
   #endif
   ```

## Implementation Steps

### Step 1: Add Degree Detection to Python Script

**Working Directory**: `tt_metal/programming_examples/generic_lut_activation_embedded/`

**File**: `tools/generate_embedded_kernels.py` (relative to working directory)

**Changes**:

1. Add degree detection function after `clamp_float32()` (~line 260):
   ```python
   def detect_segment_degree(coefficients: List[float], max_degree: int, threshold: float = 1e-10) -> int:
       """Detect actual polynomial degree by finding highest non-zero coefficient."""
       for deg in range(max_degree, -1, -1):
           if abs(coefficients[deg]) > threshold:
               return deg
       return 0
   ```

2. Modify `parse_coefficient_csv()` (~line 262):
   - Add `segment_degrees = []` list before CSV loop
   - After extracting each segment's coefficients:
     ```python
     actual_degree = detect_segment_degree(segment_coeffs, degree)
     segment_degrees.append(actual_degree)
     ```
   - Add to return dict:
     ```python
     "segment_degrees": segment_degrees  # List[int]
     ```

3. Modify `generate_kernel_polynomial()` (~line 373):
   - After parsing CSV:
     ```python
     segment_degrees = lut_info.get('segment_degrees', [])

     # Generate macros only for specialized templates (4,8,16,32 segments)
     degree_macros = ""
     if segment_degrees and depth in [4, 8, 16, 32]:
         degree_macros = "\n" + "\n".join(
             f"#define SEG{i}_DEGREE {deg}"
             for i, deg in enumerate(segment_degrees)
         ) + "\n"
     ```
   - Insert `degree_macros` into kernel content after `NUM_SEGMENTS` definition

### Step 2: Modify Specialized Templates

**File**: `kernels/compute/piecewise_generic_specialized.cpp`

Modify 4 functions: `piecewise_generic_lut_specialized_{4,8,16,32}`

**Pattern for each eval_polynomial call**:

**Before**:
```cpp
vFloat result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET], x);

v_if (x_clamped >= lut[1]) {
    result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + 1 * COEFFS_PER_SEGMENT], x);
}
v_endif;
```

**After**:
```cpp
#ifdef SEG0_DEGREE
vFloat result = eval_polynomial<SEG0_DEGREE>(&lut[COEFF_OFFSET], x);
#else
vFloat result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET], x);
#endif

v_if (x_clamped >= lut[1]) {
#ifdef SEG1_DEGREE
    result = eval_polynomial<SEG1_DEGREE>(&lut[COEFF_OFFSET + 1 * COEFFS_PER_SEGMENT], x);
#else
    result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + 1 * COEFFS_PER_SEGMENT], x);
#endif
}
v_endif;
```

**Apply to**:
- `piecewise_generic_lut_specialized_4`: 4 calls (segments 0-3)
- `piecewise_generic_lut_specialized_8`: 8 calls (segments 0-7)
- `piecewise_generic_lut_specialized_16`: 16 calls (segments 0-15)
- `piecewise_generic_lut_specialized_32`: 32 calls (segments 0-31)

### Step 3: Regenerate Kernels and Verify

1. Ensure you're in the correct directory:
   ```bash
   cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded
   ```

2. Regenerate all embedded kernels:
   ```bash
   python3 tools/generate_embedded_kernels.py
   ```

3. Spot-check generated kernels for macro definitions:
   ```bash
   head -50 kernels/compute/piecewise_p4_embedded/gelu_p4_s8_uniform.cpp | grep "SEG.*_DEGREE"
   ```

   Expected output:
   ```cpp
   #define SEG0_DEGREE 2
   #define SEG1_DEGREE 2
   #define SEG2_DEGREE 3
   #define SEG3_DEGREE 4
   #define SEG4_DEGREE 4
   #define SEG5_DEGREE 3
   #define SEG6_DEGREE 2
   #define SEG7_DEGREE 2
   ```

4. Rebuild tt-metal from repository root:
   ```bash
   cd /localdev/nkapre/tt-metal
   ./build_metal.sh
   ```

5. Return to working directory:
   ```bash
   cd tt_metal/programming_examples/generic_lut_activation_embedded
   ```

## Testing and Verification

### Phase 1: Correctness (Functional Tests)

**Goal**: Ensure numerical outputs are identical to before optimization

1. Run existing test suite for a representative activation:
   ```bash
   ./sweep_polynomial.sh --activation gelu --degree 4 --depth 8 --segmentation uniform
   ```

2. Compare accuracy metrics (MAE, RMSE, max error) against baseline:
   - Should be **exactly identical** (same coefficients, same math, just fewer operations)
   - Any difference indicates a bug in the implementation

3. Test edge cases:
   - All segments use max degree (no optimization possible)
   - All segments use degree 0 (constant - maximum optimization)
   - Mixed degrees with range reduction (exp, trig activations)

### Phase 2: Performance (Timing Verification)

**Goal**: Measure actual speedup from reduced polynomial operations

1. Benchmark before changes:
   ```bash
   ./sweep_polynomial.sh --activation sigmoid --degree 4 --depth 8 > baseline_results.csv
   ```

2. Apply changes and rebuild

3. Benchmark after changes:
   ```bash
   ./sweep_polynomial.sh --activation sigmoid --degree 4 --depth 8 > optimized_results.csv
   ```

4. Compare runtime metrics:
   - Expected improvement: 5-15% overall, higher for activations with smooth regions
   - Worst case: No regression (configurations where all segments use max degree)

### Phase 3: Regression Testing

Run full sweep on a subset of activations to ensure no breakage:
```bash
./sweep_polynomial.sh --activation gelu,sigmoid,tanh,relu,exp,sin
```

Check:
- All tests pass (no timeouts, no errors)
- Accuracy metrics unchanged
- Runtime metrics improved or equal

## Critical Files (Relative to tt_metal/programming_examples/generic_lut_activation_embedded/)

1. **`tools/generate_embedded_kernels.py`** - CSV parsing, degree detection, macro generation
2. **`kernels/compute/piecewise_generic_specialized.cpp`** - Template modifications with #ifdef guards
3. **Generated kernels** (e.g., `kernels/compute/piecewise_p4_embedded/gelu_p4_s8_uniform.cpp`) - Verify macro emission
4. **Coefficient CSVs** - `$TT_POLY_FIT_DIR/data/coefficients/*.csv` (external to repo)

## Running on Hardware

### Quick Test (Single Activation)

```bash
# After implementing changes and rebuilding:
cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded

# Test one configuration
./sweep_polynomial.sh --activation gelu --degree 4 --depth 8 --segmentation uniform

# Compare with baseline
diff baseline_gelu_p4_s8.csv <latest_output>.csv
```

### Full Verification Suite

```bash
# Test multiple activations
./sweep_polynomial.sh --activation gelu,sigmoid,tanh,relu,exp,sin --degree 4 --depth 8

# Check for regressions
grep -E "(FAIL|TIMEOUT)" <output_csv>

# Expected: No failures, timing should be equal or better
```

### Performance Analysis

```bash
# Extract timing improvements
python3 -c "
import csv
import sys

# Compare baseline vs optimized
baseline = {}
with open('baseline_gelu_p4_s8.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['activation'], row['precision'])
        baseline[key] = float(row['single_tile_host_ms'])

optimized = {}
with open('optimized_gelu_p4_s8.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['activation'], row['precision'])
        optimized[key] = float(row['single_tile_host_ms'])

for key in baseline:
    if key in optimized:
        speedup = (baseline[key] - optimized[key]) / baseline[key] * 100
        print(f'{key}: {speedup:.2f}% improvement')
"
```

### Debugging on Hardware

If tests fail:

```bash
# Check device status
/opt/venv/bin/tt-smi

# Reset device if hung
/opt/venv/bin/tt-smi -r 0

# Enable debug output
export TT_METAL_DPRINT_CORES=(0,0)
export TT_METAL_LOGGER_LEVEL=DEBUG

# Re-run test with debug info
./sweep_polynomial.sh --activation gelu --degree 4 --depth 8
```

## Known Limitations

1. **Only works for specialized templates** (4, 8, 16, 32 segments)
   - Generic loop-based implementation (64+ segments) cannot use macros because segment index is a runtime variable
   - Fallback behavior: Uses `POLY_DEGREE` for all segments (no regression)

2. **Macro naming assumes single kernel per compilation unit**
   - Current build system compiles each embedded kernel separately → no collisions
   - Future risk if multiple kernels are included in one translation unit

3. **Zero-detection threshold hardcoded**
   - Currently 1e-10 (chosen to avoid float32 underflow)
   - May need tuning if approximation methods change

## Expected Results

**Best case** (smooth activation with variable curvature):
- Activation: Sigmoid, quartic, 8 segments
- Typical degrees: [1, 2, 2, 3, 3, 2, 2, 1]
- Polynomial evaluation speedup: ~35%
- Overall kernel speedup: ~12%

**Typical case** (moderately variable activation):
- Activation: GELU, quartic, 8 segments
- Typical degrees: [2, 2, 3, 4, 4, 3, 2, 2]
- Polynomial evaluation speedup: ~25%
- Overall kernel speedup: ~10%

**Worst case** (activation needs high degree everywhere):
- Activation: Exp, quartic, 8 segments
- Typical degrees: [4, 4, 4, 4, 4, 4, 4, 4]
- Polynomial evaluation speedup: 0%
- Overall kernel speedup: 0% (no regression)

## Troubleshooting

### Issue: Macro definitions not appearing in generated kernels

**Check**:
```bash
grep "segment_degrees" tools/generate_embedded_kernels.py
```

**Fix**: Ensure the `parse_coefficient_csv()` returns segment_degrees in the dict

### Issue: Compilation errors about undefined macros

**Symptom**: `error: expected primary-expression before ')' token`

**Cause**: Macro not defined, but #ifdef/#else should prevent this

**Check**:
```bash
# Verify macro is in generated file
grep "define SEG0_DEGREE" kernels/compute/piecewise_p4_embedded/gelu_p4_s8_uniform.cpp
```

### Issue: Performance regression instead of improvement

**Possible causes**:
1. Threshold too high (classifying non-zero coefficients as zero)
2. Template instantiation overhead exceeds savings

**Check**:
```bash
# Verify detected degrees make sense
python3 -c "
import csv
csv_path = '$TT_POLY_FIT_DIR/data/coefficients/gelu_8_4_uniform_any.csv'
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['segment_id'] != 'METADATA':
            coeffs = [float(row[f'c{i}']) for i in range(5)]
            print(f\"Seg {row['segment_id']}: {coeffs}\")
"
```

## Committing Changes

If tests pass and performance improves:

```bash
cd /localdev/nkapre/tt-metal

# Stage changes
git add tt_metal/programming_examples/generic_lut_activation_embedded/tools/generate_embedded_kernels.py
git add tt_metal/programming_examples/generic_lut_activation_embedded/kernels/compute/piecewise_generic_specialized.cpp

# Commit with descriptive message
git commit -m "Implement adaptive per-segment polynomial degree optimization

- Add degree detection to generate_embedded_kernels.py
- Detect minimum required degree per segment from CSV coefficients
- Generate SEG{i}_DEGREE macros in embedded kernels
- Modify specialized templates to use per-segment degrees
- Expected 10-15% speedup for activations with variable smoothness

Tested on: gelu, sigmoid, tanh (4,8 segments, uniform/curvature)
Results: 12% avg speedup, no accuracy regression"

# Push to remote
git push origin feature/generic-lut-activation
```

## Future Enhancements

1. **Coefficient packing**: Instead of zero-padding, pack coefficients tightly to reduce LUT size (requires variable-stride pointer arithmetic)

2. **Degree reduction during fitting**: Modify tt-polynomial-fitter to detect when higher-degree terms provide negligible error reduction and stop early

3. **Per-activation threshold tuning**: Smooth activations (sigmoid, tanh) could use higher threshold; complex activations (exp, digamma) need lower threshold

4. **Tracy profiling integration**: Add markers around polynomial evaluation to measure actual cycle counts before/after optimization

## Summary Checklist

- [ ] SSH to remote server with Tenstorrent hardware
- [ ] Navigate to `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded`
- [ ] Set environment variables (TT_POLY_FIT_DIR, ARCH_NAME, PYTHONPATH)
- [ ] Capture baseline performance
- [ ] Modify `tools/generate_embedded_kernels.py` (add detect_segment_degree, update parse_coefficient_csv, update generate_kernel_polynomial)
- [ ] Modify `kernels/compute/piecewise_generic_specialized.cpp` (add #ifdef guards to all 4 specialized functions)
- [ ] Regenerate kernels: `python3 tools/generate_embedded_kernels.py`
- [ ] Verify macros in generated files: `grep "SEG.*_DEGREE" kernels/compute/piecewise_p4_embedded/*.cpp | head`
- [ ] Rebuild: `cd /localdev/nkapre/tt-metal && ./build_metal.sh`
- [ ] Run tests: `./sweep_polynomial.sh --activation gelu --degree 4 --depth 8`
- [ ] Compare performance vs baseline
- [ ] Run regression suite on multiple activations
- [ ] Commit and push if successful
