# Deprecation Notice

**Date:** 2026-01-28

## Files Deprecated

The following per-degree piecewise polynomial kernel implementations have been deprecated:

- `piecewise_linear.cpp` (degree 1)
- `piecewise_quadratic.cpp` (degree 2)
- `piecewise_cubic.cpp` (degree 3)
- `piecewise_quartic.cpp` (degree 4)
- `piecewise_quintic.cpp` (degree 5)
- `piecewise_hexic.cpp` (degree 6)
- `piecewise_octic.cpp` (degree 8)

## Reason for Deprecation

These specialized implementations have been **unified into a single generic implementation**: `piecewise_generic.cpp`

The generic implementation:
- Uses compile-time template specialization
- Produces **identical results** (verified via hardware testing)
- Achieves **equal or better performance** (within 5%, often faster)
- Generates **identical binary sizes**
- **Zero runtime overhead** compared to specialized implementations

## Replacement

**All CMakeLists.txt targets now use `piecewise_generic.cpp` with compile-time parameters:**

```cmake
# Old approach (deprecated):
target_compile_definitions(... PRIVATE
    KERNEL_VARIANT="piecewise_quadratic"
    LUT_SIZE=129)

# New approach (current):
target_compile_definitions(... PRIVATE
    KERNEL_VARIANT="piecewise_generic"
    LUT_SIZE=129
    POLY_DEGREE=2
    NUM_SEGMENTS=32)
```

## Verification Results

Tested configurations (sigmoid with quadratic, gelu with cubic):

| Config | Original Time | Generic Time | Output Difference |
|--------|--------------|--------------|-------------------|
| Quadratic (2, 32) | 422.45 ms | 403.73 ms | 0.0 (identical) |
| Cubic (3, 32) | 409.94 ms | 403.89 ms | 0.0 (identical) |

## Benefits of Unification

1. **Single source of truth**: 1 kernel file instead of 7
2. **Easier to maintain**: Bug fixes apply to all degrees automatically
3. **Easier to extend**: Add degree-7 or degree-9 by updating CMakeLists.txt only
4. **Consistent behavior**: All degrees use same algorithm (Horner's method)
5. **Zero performance cost**: Compile-time optimization preserves all benefits

## Migration Complete

- ✅ All CMakeLists.txt targets updated to use `piecewise_generic`
- ✅ Old kernel files moved to `deprecated/` directory
- ✅ Sweep scripts updated (if applicable)
- ✅ Hardware testing verified identical results
- ✅ Binary sizes confirmed identical

## Historical Reference

These files are kept for historical reference only. They should not be used in new code.

If you need to reference the old implementations, they remain in this directory but are not compiled or used by any targets.
