# Fix Summary: Atanh Pass Rate Issue

## Root Cause Identified

There are **TWO SETS** of embedded kernels:

### 1. OLD Manually Created Kernels (BUGGY) ❌
- Files: `atanh_cubic_16.cpp`, `sigmoid_cubic_16.cpp`, etc. (29 files)
- LUT_SIZE: **64** (WRONG - missing boundaries!)
- INPUT_MIN/MAX: **Hardcoded wrong values**
- Format: Only coefficients, no boundaries
- Example: `kernels/compute/piecewise_cubic_embedded/atanh_cubic_16.cpp`
  ```cpp
  constexpr float INPUT_MIN = -0.800000000f;  // WRONG! Should be -0.99
  constexpr float INPUT_MAX = 0.800000000f;   // WRONG! Should be 0.99
  constexpr uint32_t LUT_SIZE = 64;           // WRONG! Should be 81
  ```

### 2. NEW Auto-Generated Kernels (CORRECT) ✅
- Files: `atanh_cubic_16_uniform.cpp`, `atanh_cubic_16_adaptive.cpp`, etc. (240 files)
- LUT_SIZE: **81** (CORRECT - includes boundaries!)
- INPUT_MIN/MAX: **Correctly loaded from config**
- Format: Boundaries + coefficients (proper format)
- Example: `kernels/compute/piecewise_cubic_embedded/atanh_cubic_16_uniform.cpp`
  ```cpp
  constexpr float INPUT_MIN = -0.990000000f;  // CORRECT!
  constexpr float INPUT_MAX = 0.990000000f;   // CORRECT!
  constexpr uint32_t LUT_SIZE_FP32 = 81;      // CORRECT!
  constexpr std::array<float, 81> LUT_DATA_FP32 = {{
      // Boundaries first [0-16]
      -0.990000010f, -0.866249979f, ..., 0.990000010f,
      // Then coefficients [17-80]
      935.087097168f, 2526.503662109f, ...
  }};
  ```

## Why You're Seeing Errors

The **old manually created kernels are still in the codebase** and may be getting built/used instead of the new correct ones. The test suite might be using the old binaries.

## File Count Analysis

```bash
$ ls kernels/compute/piecewise_cubic_embedded/*.cpp | wc -l
269 total cubic kernels

$ grep -l "LUT_SIZE = 64" kernels/compute/piecewise_cubic_embedded/*.cpp | wc -l
29 old buggy kernels (for 16-segment, all activations)

$ grep -l "LUT_SIZE_FP32 = 81" kernels/compute/piecewise_cubic_embedded/*.cpp | wc -l
~240 new correct kernels
```

## Old Kernel Naming Pattern
```
{activation}_cubic_{depth}.cpp
```
Examples:
- `atanh_cubic_16.cpp` ❌
- `sigmoid_cubic_16.cpp` ❌
- `tanh_cubic_16.cpp` ❌

## New Kernel Naming Pattern
```
{activation}_cubic_{depth}_{segmentation}.cpp
```
Examples:
- `atanh_cubic_16_uniform.cpp` ✅
- `atanh_cubic_16_adaptive.cpp` ✅
- `sigmoid_cubic_16_uniform.cpp` ✅

## Immediate Fix

### Option 1: Delete Old Kernels (RECOMMENDED)
```bash
cd kernels/compute/piecewise_cubic_embedded

# Backup first
mkdir -p ../old_buggy_kernels
mv *_cubic_4.cpp *_cubic_8.cpp *_cubic_16.cpp *_cubic_32.cpp ../old_buggy_kernels/

# List what was moved:
ls ../old_buggy_kernels/
```

This removes all old manually created kernels that don't have `_uniform` or `_adaptive` suffix.

### Option 2: Rebuild CMake Targets
The CMake configuration might be building the old binaries. Check:
```bash
cmake -B build 2>&1 | grep "atanh_cubic_16"
```

Look for whether it's building:
- `programming_examples_generic_lut_activation_embedded_atanh_cubic_16` (OLD ❌)
- `programming_examples_generic_lut_activation_embedded_atanh_cubic_16_uniform` (NEW ✅)

## Verification

After deleting old kernels, rebuild and test:

```bash
# Rebuild
cd /path/to/tt-metal
./build_metal.sh --build-programming-examples

# Test new correct binary
./build/programming_examples/programming_examples_generic_lut_activation_embedded_atanh_cubic_16_uniform \
    --tiles 256

# Verify pass rate
cd tt_metal/programming_examples/generic_lut_activation_embedded
./sweep_embedded.sh
grep "atanh.*uniform.*fp32" data/hardware_outputs/*.csv
```

Expected output after fix:
```
atanh,16,uniform,fp32: MAE=0.001-0.002, RMSE=0.002-0.003, status=PASS
```

## Root Cause Timeline

1. **Initial implementation**: Manually created kernels with hardcoded LUT data
2. **Bug introduced**: Wrong LUT_SIZE (64 instead of 81), wrong INPUT_MIN/MAX
3. **Auto-generation tool created**: `generate_lut_headers.py` + `generate_embedded_kernels.py`
4. **New kernels generated**: Correct format with boundaries, proper ranges
5. **Old kernels NOT deleted**: Both old and new coexist, causing confusion

## Long-Term Solution

1. **Delete all old manually created kernels** (those without `_uniform`/`_adaptive` suffix)
2. **Update CMakeLists.txt** to only build auto-generated kernels
3. **Add CI check** to prevent manual kernel creation:
   ```bash
   # In .github/workflows/check-kernels.yml
   - name: Check for manual kernels
     run: |
       if ls kernels/compute/piecewise_*_embedded/*_cubic_[0-9]*.cpp 2>/dev/null | grep -v "_uniform\|_adaptive"; then
         echo "ERROR: Found manually created kernels. Use generate_embedded_kernels.py instead!"
         exit 1
       fi
   ```

4. **Document the auto-generation workflow** in README_EMBEDDED.md

## Files to Delete

List of old buggy 16-segment kernels:
```bash
atanh_cubic_16.cpp          mish_cubic_16.cpp          sin_cubic_16.cpp
celu_cubic_16.cpp           prelu_cubic_16.cpp         sinh_cubic_16.cpp
cos_cubic_16.cpp            relu_cubic_16.cpp          softplus_cubic_16.cpp
cosh_cubic_16.cpp           relu6_cubic_16.cpp         softshrink_cubic_16.cpp
elu_cubic_16.cpp            selu_cubic_16.cpp          softsign_cubic_16.cpp
erf_cubic_16.cpp            sigmoid_cubic_16.cpp       swish_cubic_16.cpp
exp_cubic_16.cpp            hardshrink_cubic_16.cpp    tanh_cubic_16.cpp
gelu_cubic_16.cpp           hardsigmoid_cubic_16.cpp   tanhshrink_cubic_16.cpp
hardswish_cubic_16.cpp      hardtanh_cubic_16.cpp      threshold_cubic_16.cpp
leaky_relu_cubic_16.cpp     logsigmoid_cubic_16.cpp
```

Plus similar files for depths 4, 8, 32.

## Testing After Fix

```bash
# Run embedded sweep for atanh specifically
cd /path/to/generic_lut_activation_embedded
./sweep_embedded.sh | tee sweep_fixed.log

# Check atanh results
grep "atanh" sweep_fixed.log

# Compare with previous results
diff <(grep "atanh" sweep_old.log) <(grep "atanh" sweep_fixed.log)
```

Expected improvement:
- **Before**: High errors, possible FAIL status for FP32
- **After**: Low errors (MAE ~0.001-0.002), PASS status for both BF16 and FP32

## Key Takeaway

**ALWAYS use auto-generation tools for embedded kernels!**

Manual kernel creation is:
- ❌ Error-prone (wrong LUT_SIZE, wrong ranges)
- ❌ Hard to maintain (config changes break kernels)
- ❌ Inconsistent (different activations might have different bugs)

Auto-generation ensures:
- ✅ Correct LUT format (boundaries + coefficients)
- ✅ Correct ranges (from activations_config.dat)
- ✅ Consistency across all activations
- ✅ Easy updates (just regenerate)
