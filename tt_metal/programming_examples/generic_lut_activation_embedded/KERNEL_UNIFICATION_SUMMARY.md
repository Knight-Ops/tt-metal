# Kernel Unification Summary

## Overview
Successfully unified all piecewise_*.cpp kernel files between `generic_lut_activation` and `generic_lut_activation_embedded` directories. Both directories now contain identical kernel files that support dual loading modes via the `EMBEDDED_LUT` preprocessor flag.

## Unified Files

### ✅ Fully Unified (Identical in Both Directories)
- **piecewise_linear.cpp** - Linear interpolation (2 coefficients per segment)
- **piecewise_quadratic.cpp** - 2nd order polynomial (3 coefficients per segment)  
- **piecewise_cubic.cpp** - 3rd order polynomial (4 coefficients per segment)
- **piecewise_hexic.cpp** - 6th order polynomial (7 coefficients per segment)
- **piecewise_octic.cpp** - 8th order polynomial (9 coefficients per segment)

### ✅ Copied (Now Present in Both Directories)
- **piecewise_quartic.cpp** - 4th order polynomial (generic → embedded)
- **piecewise_quintic.cpp** - 5th order polynomial (generic → embedded)

### ❌ Removed from Repository
- **piecewise_constant.cpp** - Removed per request (all files and results deleted)

### ✅ Already Identical
- **dataflow/reader.cpp** - Data movement kernel
- **dataflow/writer.cpp** - Data movement kernel
- **native_sfpu.cpp** - Native SFPU operations

## Key Unification Changes

### 1. Standardized Coefficient Naming
Changed from ad-hoc naming (a/b/c, slope/offset) to standardized ascending degree order:
- Linear: `c0` (intercept), `c1` (slope)
- Quadratic: `c0`, `c1`, `c2`
- Cubic: `c0`, `c1`, `c2`, `c3`
- Hexic: `c0`, `c1`, ..., `c6`
- Octic: `c0`, `c1`, ..., `c8`

### 2. Dual Loading Mode Support
Added `#ifdef EMBEDDED_LUT` blocks to all piecewise files:

```cpp
#ifdef EMBEDDED_LUT
    // Embedded mode: compile-time constants from generated header
    constexpr float input_min = INPUT_MIN;
    constexpr float input_max = INPUT_MAX;
    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;
#else
    // Generic mode: runtime loading from L1 circular buffer
    [[maybe_unused]] float input_min = get_arg_val<float>(1);
    [[maybe_unused]] float input_max = get_arg_val<float>(2);
    constexpr uint32_t LUT_SIZE = get_compile_time_arg_val(0);
    constexpr auto cb_lut = tt::CBIndex::c_25;
    using lut_t = std::array<float, LUT_SIZE>;
    auto p_lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);
#endif
```

### 3. Simplified Function Signatures
Removed `input_min` and `input_max` parameters from SFPU functions:

**Before:**
```cpp
inline void piecewise_quadratic_lut(const std::array<float, LUT_SIZE>& lut, float input_min, float input_max);
```

**After:**
```cpp
inline void piecewise_quadratic_lut(const std::array<float, LUT_SIZE>& lut);
```

These parameters were unused since polynomial kernels use precomputed boundaries directly from the LUT.

### 4. Removed Critical Point Logic
Removed `HAS_CRITICAL_POINT` preprocessor blocks from the unified versions. This was activation-specific logic that complicated the core polynomial evaluation. Can be re-added on a per-activation basis if needed.

## Benefits

1. **Single Source of Truth**: Only one copy of each kernel needs to be maintained
2. **Compile-Time Selection**: Same source works for both embedded and generic modes
3. **Code Consistency**: All kernels use the same naming conventions and structure
4. **Reduced Maintenance**: Changes propagate to both modes automatically
5. **Zero L1 Overhead**: Embedded mode still compiles LUTs directly into kernel code
6. **Runtime Flexibility**: Generic mode still loads LUTs from L1 memory

## File Locations

All kernel files exist in both directories with identical contents:
- `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation_embedded/kernels/`
- `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/kernels/`

## How to Use

### For Embedded LUTs (zero L1 overhead):
```bash
# Define EMBEDDED_LUT during compilation
-DEMBEDDED_LUT
```

### For Generic LUTs (runtime flexibility):
```bash
# Don't define EMBEDDED_LUT (default)
```

## Verification

Run this to verify both directories have identical kernel files:
```bash
cd tt_metal/programming_examples/
diff -r generic_lut_activation/kernels generic_lut_activation_embedded/kernels
```

Should return no differences for the unified files.
