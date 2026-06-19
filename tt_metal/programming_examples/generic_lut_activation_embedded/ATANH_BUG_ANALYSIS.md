# Atanh Pass Rate Issue - Root Cause Analysis

## Problem Summary

The manually created embedded kernels (e.g., `atanh_cubic_16.cpp`) have **incorrect LUT_SIZE** causing them to read garbage data, leading to FP32 test failures.

## Root Cause

### LUT File Format (Actual)
The binary `.lut` files contain:
```
[uint32_t size_header] [17 boundaries] [64 coefficients]
Total: 82 floats for 16-segment cubic
```

For `piecewise_cubic_remez_atanh_16_bf16.lut`:
```python
[0] = header (0.0 when read as float, actually uint32 size)
[1-17] = boundaries: -0.990, -0.866, ..., 0.866, 0.990
[18-81] = 64 coefficients: a0, b0, c0, d0, a1, b1, c1, d1, ...
```

### Kernel Expectation (piecewise_cubic.cpp)
The base kernel template expects:
```cpp
// LUT Format: [boundary_0, ..., boundary_N, a_0, b_0, c_0, d_0, ...]
// For 16 segments: 17 boundaries + 64 coefficients = 81 floats
constexpr uint32_t NUM_SEGMENTS = (LUT_SIZE - 1) / 5;
constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;
```

With specialized implementations for common sizes:
- `LUT_SIZE=21` → 4 segments (5 boundaries + 16 coefficients)
- `LUT_SIZE=41` → 8 segments (9 boundaries + 32 coefficients)
- `LUT_SIZE=81` → 16 segments (17 boundaries + 64 coefficients)

### Manual Embedded Kernel (BUGGY)
The manually created `atanh_cubic_16.cpp` has:
```cpp
constexpr float INPUT_MIN = -0.800000000f;  // WRONG! Should be -0.99
constexpr float INPUT_MAX = 0.800000000f;   // WRONG! Should be 0.99
constexpr uint32_t LUT_SIZE = 64;           // WRONG! Should be 81
constexpr std::array<float, LUT_SIZE> LUT_DATA = {{
    11.090146065f, 20.952108383f, ...  // Only 64 coefficients, missing boundaries!
}};
```

## Impact

### 1. **Missing Boundaries**
- Kernel expects 17 boundaries at indices [0-16]
- But LUT_DATA only has 64 values (all coefficients)
- When kernel reads `lut[0]`, `lut[1]`, ... `lut[16]` for boundaries, it gets random coefficients instead!

### 2. **Wrong Input Range**
- Config file specifies: `atanh,false,-0.99,0.99`
- LUT was generated for range [-0.99, 0.99]
- Kernel hardcodes: INPUT_MIN = -0.8, INPUT_MAX = 0.8
- This causes clamping errors and wrong segment selection

### 3. **Wrong Coefficient Access**
```cpp
constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;  // Should be 17
// With LUT_SIZE=64: NUM_SEGMENTS = (64-1)/5 = 12.6 → 12
// So COEFF_OFFSET = 13 (WRONG!)
```

The kernel calculates NUM_SEGMENTS incorrectly, then reads coefficients from wrong positions.

## Why This Happens

### Comparison: Generic vs Embedded

**Generic LUT Activation (CB-based)** - Working Correctly:
```cpp
// kernels/compute/piecewise_cubic.cpp
void MAIN {
    constexpr uint32_t LUT_SIZE = get_compile_time_arg_val(0);
    auto p_lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);

    #ifdef TRISC_MATH
    sfpi::piecewise_cubic_lut<LUT_SIZE>(*p_lut);
    #endif
}
```
- LUT is loaded from file into L1 circular buffer at runtime
- Correct LUT_SIZE passed as compile-time arg
- Reads all 81 floats (boundaries + coefficients)

**Embedded LUT Activation** - Manually Created, BUGGY:
```cpp
// kernels/compute/piecewise_cubic_embedded/atanh_cubic_16.cpp
constexpr uint32_t LUT_SIZE = 64;  // WRONG!
constexpr std::array<float, LUT_SIZE> LUT_DATA = {{ ... }};  // Only coefficients!

#include "../piecewise_cubic.cpp"  // Uses same algorithm, but wrong data!
```
- LUT data hardcoded in kernel source
- Manually typed coefficients only (no boundaries)
- Same algorithm expects boundaries first!

## Evidence

### 1. LUT File Inspection
```bash
$ python3 -c "import struct; floats = struct.unpack('82f', open('luts/piecewise_cubic_remez_atanh_16_bf16.lut', 'rb').read()); print(f'Total: {len(floats)} floats'); print(f'Boundaries [1-17]: {floats[1:18]}'); print(f'First segment coeffs [18-21]: {floats[18:22]}')"
```
Output:
```
Total: 82 floats
Boundaries [1-17]: (-0.99, -0.86625, ..., 0.86625, 0.99)
First segment coeffs [18-21]: (0.0, 0.0, 8.877, 6.508)
```

### 2. Config File
```bash
$ grep "^atanh" activations_config.dat
atanh,false,-0.99,0.99,0,0,Inverse hyperbolic tangent - asymptotic at ±1 (tests use ±0.999),
```

### 3. Manual Kernel Constants
```bash
$ grep -A3 "INPUT_MIN\|LUT_SIZE" kernels/compute/piecewise_cubic_embedded/atanh_cubic_16.cpp
constexpr float INPUT_MIN = -0.800000000f;
constexpr float INPUT_MAX = 0.800000000f;
constexpr uint32_t LUT_SIZE = 64;
```

All three are WRONG!

## Solution

### Option 1: Use Auto-Generated Headers (RECOMMENDED)
Replace manually created kernels with auto-generated ones that use proper LUT format:

```cpp
// kernels/compute/piecewise_cubic_embedded/atanh_cubic_16_uniform.cpp
#include "../../generated_luts/atanh/cubic_depth_16_uniform.hpp"

// Import into global namespace for base kernel
constexpr float INPUT_MIN = lut::atanh::cubic_depth_16_uniform::INPUT_MIN;
constexpr float INPUT_MAX = lut::atanh::cubic_depth_16_uniform::INPUT_MAX;
constexpr uint32_t LUT_SIZE = lut::atanh::cubic_depth_16_uniform::LUT_SIZE_FP32;  // Should be 81!
constexpr auto& LUT_DATA = lut::atanh::cubic_depth_16_uniform::LUT_DATA_FP32;

#include "../piecewise_cubic.cpp"
```

Then run:
```bash
cd /path/to/generic_lut_activation_embedded
python3 tools/generate_lut_headers.py --activation atanh
python3 tools/generate_embedded_kernels.py --activation atanh
```

### Option 2: Fix Manual Kernels (NOT RECOMMENDED)
Manually extract all 81 values (boundaries + coefficients) from LUT file:

```cpp
constexpr float INPUT_MIN = -0.990000000f;
constexpr float INPUT_MAX = 0.990000000f;
constexpr uint32_t LUT_SIZE = 81;  // 17 boundaries + 64 coefficients
constexpr std::array<float, LUT_SIZE> LUT_DATA = {{
    // Boundaries [0-16]
    -0.990000f, -0.866250f, -0.742500f, ..., 0.866250f, 0.990000f,
    // Coefficients [17-80]
    0.000000f, 0.000000f, 8.877115f, 6.508295f, ...
}};
```

But this is error-prone and defeats the purpose of auto-generation.

## Recommended Fix

1. **Delete manually created embedded kernels** in `kernels/compute/piecewise_cubic_embedded/`:
   ```bash
   rm kernels/compute/piecewise_cubic_embedded/atanh_cubic_*.cpp
   ```

2. **Regenerate using auto-generation tools**:
   ```bash
   cd /path/to/generic_lut_activation_embedded
   python3 tools/generate_lut_headers.py
   python3 tools/generate_embedded_kernels.py
   python3 tools/generate_cmake_targets.py > cmake/EmbeddedTargets.cmake
   ```

3. **Rebuild and retest**:
   ```bash
   cd /path/to/tt-metal
   ./build_metal.sh --build-programming-examples
   ./build/programming_examples/programming_examples_generic_lut_activation_embedded_atanh_cubic_16_uniform --tiles 256
   ```

## Why Auto-Generation Matters

The auto-generation pipeline (`generate_lut_headers.py` → `generate_embedded_kernels.py`) ensures:
- ✅ Correct LUT_SIZE (includes boundaries)
- ✅ Correct INPUT_MIN/INPUT_MAX (from activations_config.dat)
- ✅ Correct coefficient extraction (reads from binary LUT file)
- ✅ Consistent format across all activations
- ✅ No manual transcription errors

Manual kernel creation is **fragile** and **error-prone**, as evidenced by this bug affecting atanh (and possibly other activations).

## Verification

After fixing, verify with:
```bash
# Check kernel constants
grep -A3 "INPUT_MIN\|LUT_SIZE" kernels/compute/piecewise_cubic_embedded/atanh_cubic_16_uniform.cpp

# Run hardware test
./sweep_embedded.sh

# Check pass rate
grep atanh data/hardware_outputs/wormhole_piecewise_cubic_remez_atanh_*_uniform_fp32.csv
```

Expected result: **PASS** with low MAE (~0.001-0.002)
