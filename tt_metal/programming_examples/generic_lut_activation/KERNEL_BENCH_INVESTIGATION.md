# Kernel Bench Error Investigation

## Problem Summary

The `nachiket-adaptive-luts` kernel_bench submission for atanh shows only **7.69% pass rate** (21/273 tests) despite using correctly auto-generated kernels with proper LUT data.

## Test Failures

From `eval_results.json`:
- **Zeros tests**: 0.000000 accuracy (completely failing)
- **Valid domain tests**: 84-97% accuracy (FP32)
- **Small values tests**: 87-95% accuracy (FP32)
- **Near boundary tests**: 80-84% accuracy (FP32)

## Kernel Investigation

### ✅ LUT Data is CORRECT

The embedded kernel `atanh_cubic_16_uniform.cpp` has:
```cpp
constexpr float INPUT_MIN = -0.990000000f;  // ✅ Correct
constexpr float INPUT_MAX = 0.990000000f;   // ✅ Correct
constexpr uint32_t LUT_SIZE_FP32 = 81;      // ✅ Correct (17 boundaries + 64 coefficients)
```

Boundaries (indices 0-16):
```
[-0.99, -0.866, -0.742, -0.619, -0.495, -0.371, -0.248, -0.124,
  0.0, 0.124, 0.248, 0.371, 0.495, 0.619, 0.742, 0.866, 0.99]
```

### ✅ Algorithm is CORRECT

From `piecewise_cubic.cpp` LUT_SIZE=81 specialization (lines 144-196):
1. Clamps x to select segment: `x_clamped`
2. Uses cascade of `v_if` to select coefficients based on boundaries
3. Evaluates polynomial at ORIGINAL x: `((a * x + b) * x + c) * x + d`

Manual trace for x=0:
- Segment selected: 8 (range [0.0, 0.124))
- Coefficients: a=0.342, b=-0.00092, c=1.000, d=-1.83e-07
- Result: -1.83e-07 (effectively zero)
- Expected: 0.0
- Error: 1.83e-07 (acceptable!)

### ✅ Reader/Writer Kernels are STANDARD

Both use standard interleaved tile read/write patterns.

## Potential Issues Found

### 🔴 Issue 1: BF16 Tests Use FP32 LUT Data

In `ttnn_atanh_impl.py` (lines 114-128):
```python
compute_defines = []

if is_float32:
    # Enable 32-bit destination accumulator for fp32
    compute_config.fp32_dest_acc_en = True
    # ... other FP32 config
    compute_defines.append(("PACKER_L1_ACC", "1"))

# ❌ BUG: When is_float32=False (BF16 input), no defines are set!
# This means USE_BF16 is never defined, so BF16 tests use FP32 LUT data
```

**Impact**: BF16 tests use FP32 LUT coefficients instead of BF16-optimized coefficients.

**Fix**: Add `else` clause:
```python
else:
    # Define USE_BF16 for BF16 inputs
    compute_defines.append(("USE_BF16", "1"))
```

### 🟡 Issue 2: Compile-Time Args Mismatch

In `ttnn_atanh_impl.py` (line 90):
```python
compute_compile_args = [num_tiles, 1]  # 2 args like cosh example
```

But the compute kernel only reads 1 runtime arg:
```cpp
uint32_t n_tiles = get_arg_val<uint32_t>(0);  // Runtime arg, not compile-time
```

The compute kernel has NO compile-time args. This might not cause errors (extra args ignored?) but is confusing.

### 🟡 Issue 3: Scalar Tests (Rank0 Tensors)

Tests with shape=[] (scalar tensors) are failing most severely. Need to verify:
- Does tile calculation handle scalars correctly?
- Are scalar tensors being padded to full tiles correctly?
- Is the test framework comparing scalar results correctly?

## What Still Doesn't Make Sense

### ❓ Mystery: 0% Accuracy on "Zeros" Tests

Even though manual trace shows atanh(0) produces -1.83e-07 (essentially zero), the test reports **0% accuracy**.

Possible explanations:
1. **Test framework bug**: Maybe comparing NaN vs 0?
2. **Data format issue**: Maybe zeros aren't being represented correctly in tiles?
3. **Kernel not running**: Maybe the kernel fails to execute for certain inputs?
4. **Wrong output buffer**: Maybe reading from wrong memory location?

### ❓ Mystery: FP32 Tests Only 84-97% Accurate

Manual calculation shows tiny errors (1e-7), but tests show only 84-97% accuracy. This suggests:
1. **Large errors on some values**: Maybe specific input values cause catastrophic errors
2. **Accumulation of errors**: Maybe errors compound across tile processing
3. **Data movement corruption**: Maybe data is corrupted during read/write

## Next Steps

1. **Run simple test manually**: Create minimal test case for atanh(0) and print actual output
2. **Check intermediate values**: Add debug prints to see what values kernel receives/produces
3. **Verify tile padding**: Check how scalar/small tensors are padded to 32x32 tiles
4. **Compare with reference**: Run same test in `generic_lut_activation_embedded` directory
5. **Fix BF16 define**: Add USE_BF16 define for BF16 inputs
6. **Test compilation**: Verify kernel compiles with correct LUT_SIZE and precision selection

## Files to Check

- `/localdev/nkapre/kernel_bench/submissions/atanh/nachiket-adaptive-luts/lut_cubic_16_uniform/eval_results.json`
- `/localdev/nkapre/kernel_bench/submissions/atanh/nachiket-adaptive-luts/lut_cubic_16_uniform/result/latest/code/ttnn_atanh_impl.py`
- `/localdev/nkapre/kernel_bench/submissions/atanh/nachiket-adaptive-luts/lut_cubic_16_uniform/result/latest/code/atanh_cubic_16_uniform.cpp`
- `/localdev/nkapre/kernel_bench/submissions/atanh/nachiket-adaptive-luts/lut_cubic_16_uniform/result/latest/code/piecewise_cubic.cpp`
- Test runner: Need to find where kernel_bench executes tests

## Hypothesis

The most likely issue is in the **test framework or Python wrapper**, not the kernel itself. The kernel code and LUT data are correct, but something is going wrong in:
- Tensor preparation (padding scalars to tiles)
- Data movement (reader/writer)
- Result comparison (test framework)
- Argument passing (runtime vs compile-time args)

The BF16 issue is definitely a bug but wouldn't explain FP32 failures.
