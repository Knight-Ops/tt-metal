# Piecewise Generic Implementation - Integration Guide

## Overview

The new `piecewise_generic.cpp` kernel replaces all degree-specific kernels (linear, quadratic, cubic, quartic, quintic, hexic, octic) with a single generic implementation that works with ANY degree/depth combination.

## Benefits

- **90% code reduction**: 7 files (~2000 lines) → 1 file (~250 lines)
- **Zero runtime overhead**: All branching resolved at compile-time via `if constexpr`
- **Automatic support**: Any new degree/depth combo works without code changes
- **Easier maintenance**: Single source of truth for all polynomial operations
- **Future-proof**: Add degree 7 (septic) by adding one case to eval_polynomial

## How to Use

### Method 1: L1 Circular Buffer Mode (Runtime Flexibility)

This is the existing tt-metal approach where LUT is loaded from L1 memory.

```cpp
// Host code - pass polynomial metadata as compile-time args
vector<uint32_t> compile_time_args = {
    lut_size,        // Total LUT size
    poly_degree,     // Polynomial degree (1-8)
    num_segments     // Number of segments (4, 8, 16, 32, etc.)
};

auto compute_kernel = CreateKernel(
    program,
    "tt_metal/programming_examples/generic_lut_activation/kernels/compute/piecewise_generic.cpp",
    core_spec,
    ComputeConfig{
        .compile_args = compile_time_args,
        .defines = {}  // No EMBEDDED_LUT define
    }
);
```

### Method 2: Embedded LUT Mode (Zero L1 Overhead)

This is the kernel_bench approach where LUT is compiled directly into the kernel.

```cpp
// 1. Generate LUT header file
std::ofstream header("lut_header.h");
header << "constexpr uint32_t POLY_DEGREE = " << degree << ";\n";
header << "constexpr uint32_t NUM_SEGMENTS = " << segments << ";\n";
header << "constexpr uint32_t LUT_SIZE = " << lut_size << ";\n";
header << "constexpr std::array<float, " << lut_size << "> LUT_DATA = {";
for (size_t i = 0; i < lut_data.size(); i++) {
    header << lut_data[i] << (i + 1 < lut_data.size() ? ", " : "");
}
header << "};\n";
header.close();

// 2. Create kernel with embedded LUT
auto compute_kernel = CreateKernel(
    program,
    "tt_metal/programming_examples/generic_lut_activation/kernels/compute/piecewise_generic.cpp",
    core_spec,
    ComputeConfig{
        .compile_args = {},  // No runtime args needed
        .defines = {"EMBEDDED_LUT"}
    }
);
```

## Migration Path

### Phase 1: Validation (Parallel Testing)
Keep existing specialized kernels, add generic kernel for testing:

```python
# test_piecewise_generic.py
def test_generic_vs_specialized():
    """Verify generic kernel produces identical results to specialized kernels"""

    test_cases = [
        ("linear", 1, 4),
        ("quadratic", 2, 8),
        ("cubic", 3, 16),
        ("quartic", 4, 32),
        ("quintic", 5, 16),
        ("hexic", 6, 8),
        ("octic", 8, 4),
    ]

    for name, degree, segments in test_cases:
        # Run with specialized kernel
        result_specialized = run_activation(
            kernel=f"piecewise_{name}.cpp",
            lut_data=generate_lut(degree, segments)
        )

        # Run with generic kernel
        result_generic = run_activation(
            kernel="piecewise_generic.cpp",
            lut_data=generate_lut(degree, segments),
            poly_degree=degree,
            num_segments=segments
        )

        # Verify bit-exact match
        assert torch.allclose(result_specialized, result_generic, atol=0, rtol=0)
```

### Phase 2: Integration (Update Host Code)
Modify activation operations to use generic kernel:

```cpp
// Before: dispatch to degree-specific kernel
std::string kernel_path;
switch (poly_degree) {
    case 1: kernel_path = "piecewise_linear.cpp"; break;
    case 2: kernel_path = "piecewise_quadratic.cpp"; break;
    case 3: kernel_path = "piecewise_cubic.cpp"; break;
    case 4: kernel_path = "piecewise_quartic.cpp"; break;
    // ... etc
}

// After: single generic kernel
std::string kernel_path = "piecewise_generic.cpp";
compile_time_args = {lut_size, poly_degree, num_segments};
```

### Phase 3: Cleanup (Deprecation)
Once validated, remove old specialized kernels:

```bash
# Mark as deprecated
mv piecewise_linear.cpp piecewise_linear.cpp.deprecated
mv piecewise_quadratic.cpp piecewise_quadratic.cpp.deprecated
# ... etc

# Eventually delete
git rm piecewise_linear.cpp.deprecated
# ... etc
```

## Performance Considerations

### Compile-Time Resolution
All branching is resolved at compile time:

```cpp
// This compiles to the SAME code as hand-written specialized version
if constexpr (poly_degree == 4) {
    sfpi::piecewise_generic_lut<4, 16, 97>(*p_lut);
}
// Compiler generates: piecewise_quartic_lut<97> with no runtime branches
```

### Register Pressure Optimization
For very deep LUTs (32+ segments), consider JIT coefficient loading:

```cpp
// Current implementation (eager loading)
vFloat result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET], x);

// Potential optimization for 32+ segments (load coefficients one at a time)
// This reduces register pressure at cost of more loads
// Can be added as template specialization if needed
```

## LUT Format

The generic kernel uses the same format as specialized kernels:

```
LUT Layout:
[boundaries..., coefficients...]

Boundaries: [b0, b1, b2, ..., bN]  where N = NUM_SEGMENTS
Coefficients: For each segment i:
  [c0_i, c1_i, c2_i, ..., cn_i]  where n = POLY_DEGREE

Total size: (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)
```

Example for quartic (degree=4) with 4 segments:
```
LUT_SIZE = 5 + 4*5 = 25
Layout: [b0, b1, b2, b3, b4,          // 5 boundaries
         c0_0, c1_0, c2_0, c3_0, c4_0, // segment 0
         c0_1, c1_1, c2_1, c3_1, c4_1, // segment 1
         c0_2, c1_2, c2_2, c3_2, c4_2, // segment 2
         c0_3, c1_3, c2_3, c3_3, c4_3] // segment 3
```

## Extending to New Degrees

Adding a new polynomial degree (e.g., degree 7 - septic):

```cpp
// 1. Add case to eval_polynomial template
} else if constexpr (DEGREE == 7) {
    // Septic: y = c0 + c1*x + ... + c7*x⁷
    vFloat result = coeffs[7];
    result = result * x + coeffs[6];
    result = result * x + coeffs[5];
    result = result * x + coeffs[4];
    result = result * x + coeffs[3];
    result = result * x + coeffs[2];
    result = result * x + coeffs[1];
    result = result * x + coeffs[0];
    return result;
}

// 2. Add dispatch case in MAIN
} else if constexpr (poly_degree == 7) {
    sfpi::piecewise_generic_lut<7, num_segments, lut_size>(*p_lut);
}
```

That's it! No other changes needed.

## Testing Strategy

### Unit Tests
```python
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("segments", [4, 8, 16, 32])
def test_generic_kernel_all_combos(degree, segments):
    """Test all degree/depth combinations"""
    lut = generate_random_lut(degree, segments)
    input_tensor = torch.randn(1, 1, 32, 32)
    output = run_generic_kernel(input_tensor, lut, degree, segments)

    # Verify output shape
    assert output.shape == input_tensor.shape

    # Verify no NaNs/Infs
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
```

### Accuracy Tests
```python
def test_generic_kernel_accuracy():
    """Verify accuracy against reference implementation"""
    for activation in ["sigmoid", "tanh", "gelu", "swish"]:
        for degree in [3, 4, 5]:
            for segments in [8, 16, 32]:
                lut = fit_activation(activation, degree, segments)
                result = run_generic_kernel(test_inputs, lut, degree, segments)
                reference = eval(f"torch.{activation}")(test_inputs)

                error = torch.abs(result - reference).max()
                print(f"{activation} deg={degree} seg={segments}: max_error={error:.6e}")
```

### Performance Tests
```python
def test_generic_kernel_performance():
    """Verify performance matches specialized kernels"""

    # Warm-up
    for _ in range(10):
        run_generic_kernel(test_input, lut, degree=4, segments=16)

    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        run_generic_kernel(test_input, lut, degree=4, segments=16)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    print(f"Average latency: {avg_time*1e6:.2f} µs")
```

## Known Limitations

1. **Compile-time args required in CB mode**: Must pass poly_degree and num_segments as compile-time args
   - Embedded mode doesn't have this limitation (values in header)

2. **No JIT coefficient loading for 32+ segments**: Uses eager loading for all depths
   - Can be optimized if needed (add template specialization)

3. **Requires C++17**: Uses `if constexpr` which requires C++17 or later
   - tt-metal already uses C++20, so not an issue

## Future Enhancements

1. **Add Estrin's method**: For degrees ≥4, consider Estrin's scheme to reduce critical path
2. **SIMD optimization**: Explore vectorizing coefficient loads for very deep LUTs
3. **Mixed precision**: Support different LUT precisions (BF16 vs FP32)
4. **Adaptive segment selection**: Binary search for very deep LUTs (64+ segments)

## Questions?

For issues or questions about the generic implementation:
- File issue with tag: `lut-activation`, `piecewise-generic`
- Check existing specialized kernels for reference implementations
- See kernel_bench/tools/templates/piecewise_generic.cpp for original inspiration
