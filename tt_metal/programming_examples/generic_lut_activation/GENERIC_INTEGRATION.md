# Integrating piecewise_generic.cpp with Compile-Time Optimization

This document describes how to integrate the unified `piecewise_generic.cpp` kernel into the sweep workflow while maintaining full compile-time optimization.

## Architecture Overview

### Compile-Time Args (NOT Build-Time!)

The key insight: `POLY_DEGREE` and `NUM_SEGMENTS` are **JIT compile-time arguments**, not build-time constants:

1. **Host binary compilation** (via `./build_metal.sh`): Happens once, produces host executable
2. **Kernel JIT compilation** (when you call `CreateKernel()`): Happens at runtime, compiles kernel with specific template parameters

The flow:
```
CMakeLists.txt defines: -DPOLY_DEGREE=2 -DNUM_SEGMENTS=32
                 ↓
generic_lut_activation.cpp (host code): constexpr uint32_t POLY_DEGREE = 2;
                 ↓
CreateKernel(..., {LUT_SIZE, POLY_DEGREE, NUM_SEGMENTS}): Pass to kernel
                 ↓
piecewise_generic.cpp (device code): constexpr uint32_t poly_degree = get_compile_time_arg_val(1);
                 ↓
Template instantiation: sfpi::piecewise_generic_lut<2, 32, 129>(*p_lut);
```

The kernel compiler sees `poly_degree=2` and `num_segments=32` as **compile-time constants**, enabling full optimization (loop unrolling, template specialization, etc.).

## Changes Made

### 1. Host Code (`generic_lut_activation.cpp`)

Added compile-time parameter definitions and passing logic:

```cpp
// Define parameters (set by CMakeLists.txt)
#ifndef POLY_DEGREE
constexpr uint32_t POLY_DEGREE = 0;  // 0 means not specified
#endif

#ifndef NUM_SEGMENTS
constexpr uint32_t NUM_SEGMENTS = 0;  // 0 means not specified
#endif

// Pass to kernel
if (kernel_variant == "piecewise_generic") {
    compute_compile_args.push_back(POLY_DEGREE);
    compute_compile_args.push_back(NUM_SEGMENTS);
}
```

### 2. CMakeLists.txt

Added targets for all degree/depth combinations using `piecewise_generic.cpp`:

```cmake
# Example: Quadratic, 32 segments
add_executable(programming_examples_generic_lut_activation_generic_2_32)
target_compile_definitions(programming_examples_generic_lut_activation_generic_2_32 PRIVATE
    KERNEL_VARIANT="piecewise_generic"
    LUT_SIZE=129      # (32+1) boundaries + 32*3 coeffs
    POLY_DEGREE=2
    NUM_SEGMENTS=32)
```

**Total targets**: 28 (7 degrees × 4 depths)

### 3. Verification Script (`verify_generic_identical.sh`)

Created automated verification to compare:
- MAE/RMSE accuracy (should be identical within FP precision ~1e-10)
- Hardware outputs (element-by-element comparison)
- Kernel execution time (should be within 5%)

## Usage

### Step 1: Build the Generic Variants

You can either:

**Option A**: Manually add generated targets to CMakeLists.txt
```bash
# Generate the CMakeLists.txt entries
python3 generate_generic_cmake_targets.py >> CMakeLists.txt

# Build
./build_metal.sh --build-programming-examples
```

**Option B**: Build specific targets individually
```bash
cd build_Release
cmake .. -G Ninja
ninja programming_examples_generic_lut_activation_generic_2_32
ninja programming_examples_generic_lut_activation_generic_3_32
# etc...
```

### Step 2: Run Verification

```bash
# Verify that generic produces identical results to specialized implementations
./tt_metal/programming_examples/generic_lut_activation/verify_generic_identical.sh
```

This script will:
1. Run both original (e.g., `piecewise_quadratic.cpp`) and generic implementations
2. Compare MAE, RMSE, and hardware outputs
3. Report if results are IDENTICAL (within 1e-10 tolerance)

Expected output:
```
✅ PASS: Generic implementation produces IDENTICAL results!
  MAE difference:   0.0e+00  ✓ IDENTICAL
  RMSE difference:  0.0e+00  ✓ IDENTICAL
  Time difference:  0.1ms (0.5%)  ✓ SIMILAR
```

### Step 3: Integration into Sweep Workflow

Once verified, you can:

**Option A**: Keep both implementations (for gradual migration)
- Current sweep scripts use original implementations
- New sweep script (`sweep_generic.sh`) uses generic implementation
- Compare results periodically

**Option B**: Replace in existing sweep scripts

Modify `sweep_piecewise.sh` to use generic binaries:
```bash
# Change from:
binary="$BUILD_DIR/programming_examples_..._quadratic_2_32"

# To:
binary="$BUILD_DIR/programming_examples_..._generic_2_32"
```

## Performance Characteristics

### Compile-Time Optimization Benefits

Because `POLY_DEGREE` and `NUM_SEGMENTS` are compile-time constants:

1. **Loop unrolling**: The segment search loop unrolls completely
   ```cpp
   #pragma unroll
   for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) { ... }
   ```
   With `NUM_SEGMENTS=32`, this becomes 32 inline comparisons (no loop overhead).

2. **Template specialization**: Each degree gets optimized code
   ```cpp
   if constexpr (poly_degree == 2) {
       // Compiler generates specialized quadratic code only
       return (coeffs[2] * x + coeffs[1]) * x + coeffs[0];
   }
   ```

3. **Dead code elimination**: Other degree paths are completely removed from binary

4. **Register optimization**: Compiler knows exact number of coefficients needed

### Expected Performance

- **Accuracy**: IDENTICAL to original (within FP precision ~1e-10)
- **Speed**: Within 1-5% of original (sometimes identical, sometimes slightly faster/slower due to compiler quirks)
- **Memory**: IDENTICAL L1 usage (same LUT size, same circular buffers)

## Deprecation Path

Once verification passes for all configurations:

### Phase 1: Parallel Implementation (Safe)
- Keep both implementations
- Use generic for new development
- Gradually migrate sweep scripts

### Phase 2: Deprecation (After Confidence)
```bash
# Mark old files as deprecated (add warnings)
# kernels/compute/piecewise_quadratic.cpp → DEPRECATED
# kernels/compute/piecewise_cubic.cpp → DEPRECATED
# etc...
```

### Phase 3: Removal (After Testing Period)
```bash
# Delete deprecated files
rm kernels/compute/piecewise_quadratic.cpp
rm kernels/compute/piecewise_cubic.cpp
rm kernels/compute/piecewise_quartic.cpp
rm kernels/compute/piecewise_quintic.cpp
rm kernels/compute/piecewise_hexic.cpp
rm kernels/compute/piecewise_octic.cpp
# Keep piecewise_linear.cpp if it has special optimizations
```

## Benefits After Migration

1. **Single source of truth**: 1 kernel file instead of 7
2. **Easy to extend**: Add degree-7 or degree-9 by just updating CMakeLists.txt
3. **Reduced maintenance**: Bug fixes in one place
4. **Consistent behavior**: All degrees use same logic
5. **Same performance**: Zero runtime overhead from unification

## Testing Checklist

Before deprecating old implementations:

- [ ] Build all 28 generic targets successfully
- [ ] Run `verify_generic_identical.sh` - all tests pass
- [ ] Run sweep on subset (e.g., sigmoid, gelu, tanh) with both implementations
- [ ] Compare full sweep results (MAE distributions should be identical)
- [ ] Test on both Wormhole and Blackhole architectures
- [ ] Verify FP32 and BF16 modes both work

## Troubleshooting

### Build Errors

**Error**: `POLY_DEGREE not defined`
- **Fix**: Ensure CMakeLists.txt has `-DPOLY_DEGREE=X` in `target_compile_definitions`

**Error**: `LUT_SIZE must equal (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)`
- **Fix**: Verify LUT_SIZE calculation is correct:
  - Linear (degree 1): `LUT_SIZE = (N+1) + N*2`
  - Quadratic (degree 2): `LUT_SIZE = (N+1) + N*3`
  - Cubic (degree 3): `LUT_SIZE = (N+1) + N*4`
  - etc.

### Results Differ

**MAE differs by > 1e-10**:
- Check if LUT files are identical
- Verify same precision mode (BF16 vs FP32)
- Ensure same test range
- Compare raw hardware outputs CSV files

**Performance differs by > 5%**:
- Normal for first run (device initialization overhead)
- Run multiple times and average
- Check if other processes are using the device

## Questions?

See `piecewise_generic.cpp` header comments for detailed implementation notes.
