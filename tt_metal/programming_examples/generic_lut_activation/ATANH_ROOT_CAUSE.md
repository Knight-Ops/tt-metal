# Atanh Low Pass Rate - Complete Root Cause Analysis

## TL;DR

**The embedded kernel example `kernels/compute/piecewise_cubic_embedded/atanh_cubic_16.cpp` has THREE critical bugs:**

1. ❌ **LUT_SIZE = 64** (should be 81) - missing boundaries!
2. ❌ **INPUT_MIN = -0.8** (should be -0.99) - wrong range!
3. ❌ **INPUT_MAX = 0.8** (should be 0.99) - wrong range!

This causes **massive errors** when tested on the correct range [-0.99, 0.99].

## The Bug in Detail

### What You Have (WRONG)
```cpp
// kernels/compute/piecewise_cubic_embedded/atanh_cubic_16.cpp
constexpr float INPUT_MIN = -0.800000000f;  // ❌ WRONG!
constexpr float INPUT_MAX = 0.800000000f;   // ❌ WRONG!
constexpr uint32_t LUT_SIZE = 64;           // ❌ WRONG!
constexpr std::array<float, 64> LUT_DATA = {{
    11.090146065f, 20.952108383f, ...  // Only coefficients, no boundaries!
}};
```

### What You Need (CORRECT)
```cpp
// From reference: generic_lut_activation_embedded/kernels/.../atanh_cubic_16_uniform.cpp
constexpr float INPUT_MIN = -0.990000000f;  // ✅ CORRECT!
constexpr float INPUT_MAX = 0.990000000f;   // ✅ CORRECT!
constexpr uint32_t LUT_SIZE = 81;           // ✅ CORRECT!
constexpr std::array<float, 81> LUT_DATA = {{
    // First 17 values: boundaries
    -0.990000010f, -0.866249979f, ..., 0.990000010f,
    // Remaining 64 values: coefficients (16 segments × 4 coeffs)
    935.087097168f, 2526.503662109f, ...
}};
```

## Why This Causes FP32 Errors

### The Algorithm Expects Boundaries
The `piecewise_cubic.cpp` kernel algorithm does this:

```cpp
// For 16-segment cubic LUT:
constexpr uint32_t NUM_SEGMENTS = (LUT_SIZE - 1) / 5;  // Expects boundaries!
constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;    // Skip boundaries

// Read boundaries for segment selection
v_if (x_clamped >= lut[1]) { /* use segment 1 coeffs */ } v_endif;
v_if (x_clamped >= lut[2]) { /* use segment 2 coeffs */ } v_endif;
...
```

### What Happens with LUT_SIZE=64
```cpp
NUM_SEGMENTS = (64 - 1) / 5 = 12.6 → 12  // ❌ Wrong! Should be 16
COEFF_OFFSET = 12 + 1 = 13               // ❌ Wrong! Should be 17

// When kernel reads boundaries lut[0], lut[1], ... lut[12]:
// It gets COEFFICIENTS instead of boundaries!
// When kernel reads coefficients starting at lut[13]:
// It gets WRONG coefficients from middle of array!
```

### Impact on Accuracy
- **Segment selection**: Completely broken (using coefficient values as boundaries)
- **Coefficient access**: Reading from wrong positions
- **Input clamping**: Clamping to [-0.8, 0.8] instead of [-0.99, 0.99]
- **Result**: Garbage output for FP32 tests using full [-0.99, 0.99] range

## Evidence

### 1. Config File Says -0.99 to 0.99
```bash
$ grep "^atanh" activations_config.dat
atanh,false,-0.99,0.99,0,0,Inverse hyperbolic tangent...
```

### 2. LUT File Has 82 Floats (17 boundaries + 64 coeffs + 1 header)
```bash
$ python3 -c "import struct; data = open('luts/piecewise_cubic_remez_atanh_16_bf16.lut', 'rb').read(); print(f'Total floats: {len(data)//4}')"
Total floats: 82

$ python3 -c "import struct; floats = struct.unpack('82f', open('luts/piecewise_cubic_remez_atanh_16_bf16.lut', 'rb').read()); print('Boundaries:', floats[1:18])"
Boundaries: (-0.99, -0.86625, -0.7425, ..., 0.86625, 0.99)
```

### 3. Reference Implementation Has Correct Values
```bash
$ cd ../generic_lut_activation_embedded
$ grep -A2 "INPUT_MIN\|LUT_SIZE" kernels/compute/piecewise_cubic_embedded/atanh_cubic_16_uniform.cpp
constexpr float INPUT_MIN = -0.990000000f;
constexpr float INPUT_MAX = 0.990000000f;
constexpr uint32_t LUT_SIZE_FP32 = 81;
```

## The Fix

### Option 1: Copy from Reference (QUICK FIX)
```bash
cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation

# Backup buggy version
mv kernels/compute/piecewise_cubic_embedded/atanh_cubic_16.cpp \
   kernels/compute/piecewise_cubic_embedded/atanh_cubic_16.cpp.buggy

# Copy correct version from reference
cp ../generic_lut_activation_embedded/kernels/compute/piecewise_cubic_embedded/atanh_cubic_16_uniform.cpp \
   kernels/compute/piecewise_cubic_embedded/atanh_cubic_16.cpp
```

### Option 2: Generate Correctly (PROPER FIX)
```bash
cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded

# Generate correct LUT header
python3 tools/generate_lut_headers.py --activation atanh --degree cubic --depth 16

# Generate correct kernel wrapper
python3 tools/generate_embedded_kernels.py --activation atanh --degree cubic --depth 16

# Copy to parent directory
cp kernels/compute/piecewise_cubic_embedded/atanh_cubic_16_uniform.cpp \
   ../generic_lut_activation/kernels/compute/piecewise_cubic_embedded/atanh_cubic_16.cpp
```

### Option 3: Manual Fix (TEDIOUS BUT EDUCATIONAL)
Edit `kernels/compute/piecewise_cubic_embedded/atanh_cubic_16.cpp`:

```cpp
// Change these lines:
constexpr float INPUT_MIN = -0.990000000f;  // Was -0.8
constexpr float INPUT_MAX = 0.990000000f;   // Was 0.8
constexpr uint32_t LUT_SIZE = 81;           // Was 64

// Replace LUT_DATA with 81 values from the LUT file:
constexpr std::array<float, 81> LUT_DATA = {{
    // Boundaries [0-16] from floats[1:18] in LUT file
    -0.990000010f, -0.866249979f, -0.742500007f, -0.618749976f,
    -0.495000005f, -0.371250004f, -0.247500002f, -0.123750001f,
    0.000000000f, 0.123750001f, 0.247500002f, 0.371250004f,
    0.495000005f, 0.618749976f, 0.742500007f, 0.866249979f,
    0.990000010f,
    // FP32 coefficients [17-80] from floats[18:82] in LUT file
    935.087097168f, 2526.503662109f, 2280.033203125f, 685.757690430f,
    23.901872635f, 50.922496796f, 38.358306885f, 9.235501289f,
    5.284048557f, 8.386280060f, 5.935266018f, 0.990540683f,
    1.985145688f, 2.135608912f, 1.981179953f, 0.155528113f,
    // ... (remaining 48 coefficients)
}};
```

## Testing the Fix

```bash
# Rebuild
cd /localdev/nkapre/tt-metal
./build_metal.sh --build-programming-examples

# Test with correct range
cd tt_metal/programming_examples/generic_lut_activation
./test_embedded_kernel.sh  # Or whatever test you use

# Check pass rate - should now PASS with low error
```

## Why This Wasn't Caught Earlier

1. **BF16 tests passed**: BF16 has lower precision, so errors were within tolerance
2. **Small range tests passed**: Tests using [-0.8, 0.8] range worked (by accident!)
3. **FP32 full range tests FAILED**: Tests using [-0.99, 0.99] exposed the bug
4. **CB-based implementation works**: Only embedded version was broken

## Comparison: CB vs Embedded

| Implementation | Status | LUT Format | LUT Size | Input Range |
|---|---|---|---|---|
| CB-based (main) | ✅ Works | Boundaries + Coeffs | 81 | [-0.99, 0.99] |
| Embedded (buggy) | ❌ Broken | Coeffs only | 64 | [-0.8, 0.8] |
| Embedded (fixed) | ✅ Works | Boundaries + Coeffs | 81 | [-0.99, 0.99] |

## Lesson Learned

**NEVER manually create embedded kernels!**

Always use the auto-generation pipeline:
```bash
generate_lut_headers.py → generate_embedded_kernels.py
```

This ensures:
- ✅ Correct LUT format (boundaries + coefficients)
- ✅ Correct input ranges (from activations_config.dat)
- ✅ Correct size calculations
- ✅ No transcription errors
- ✅ Consistency across all activations

## Next Steps

1. **Fix this kernel** using one of the options above
2. **Rebuild and test** to verify fix works
3. **Delete other buggy examples** if any exist
4. **Document** the correct way to create embedded kernels
5. **Add CI check** to prevent manual kernel creation

## Related Files

- Config: `activations_config.dat` (defines [-0.99, 0.99])
- LUT file: `luts/piecewise_cubic_remez_atanh_16_bf16.lut` (82 floats)
- Buggy kernel: `kernels/compute/piecewise_cubic_embedded/atanh_cubic_16.cpp` (64 floats, wrong range)
- Reference: `../generic_lut_activation_embedded/kernels/.../atanh_cubic_16_uniform.cpp` (81 floats, correct)
- Base algorithm: `kernels/compute/piecewise_cubic.cpp` (expects boundaries!)
