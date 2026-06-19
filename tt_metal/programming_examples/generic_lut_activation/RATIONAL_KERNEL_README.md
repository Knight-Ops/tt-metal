# Rational Approximation Kernels

Implementation of piecewise rational approximation kernels for high-accuracy activation functions with minimal segments.

## Overview

Rational approximations evaluate functions as: **y = P(x) / Q(x)**
- P(x) = numerator polynomial: n0 + n1*x + n2*x² + ... + nm*x^m
- Q(x) = denominator polynomial: d0 + d1*x + d2*x² + ... + dk*x^k

### Advantages vs Polynomial Approximations

| Metric | Polynomial (16 segments, degree 4) | Rational (2 segments, degree 4/4) |
|--------|-----------------------------------|----------------------------------|
| **MAE** | ~1e-3 | ~1e-5 |
| **Segments** | 16 | 2 (8× fewer!) |
| **Memory** | ~196 bytes | ~23 floats = 92 bytes (2× less) |
| **Compute** | 1× | ~1.5-2× (two Horner chains + division) |

**Trade-off:** Fewer segments and better accuracy at the cost of slightly more computation.

## Files

### Core Implementation
- **`lut_loader_rational.hpp`** - Parses rational CSV files with n0, n1, ..., d0, d1, ... coefficients
- **`kernels/compute/piecewise_rational.cpp`** - Rational approximation kernel (evaluates P(x)/Q(x))
- **`generic_lut_activation_rational.cpp`** - Test program for rational kernels

### Build System
- **`generate_cmake_rational.py`** - Auto-generates CMake targets for rational variants
- **`CMakeLists.txt`** - Now includes 6 rational targets (appended by generator)

### Testing and Validation
- **`test_piecewise_rational.py`** - Python reference implementation and validation
- **`sweep_rational.sh`** - Automated benchmarking across activations

## CSV Format

Rational coefficient CSV files follow this format:

```csv
segment_id,lo,hi,approximation_type,num_degree,den_degree,n0,n1,n2,n3,n4,d0,d1,d2,d3,d4,error,method,segmentation
0,-10.0,0.0,rational,4,4,0.4999,0.1665,0.0216,0.0013,0.00003,1.0,-0.1629,0.1355,-0.0123,0.0047,1.3e-04,rational_xnnpack,uniform
1,0.0,10.0,rational,4,4,0.5000,0.3145,0.0920,0.0154,0.0020,1.0,0.1291,0.1191,0.0135,0.0021,2.0e-06,rational_xnnpack,uniform
```

**Key columns:**
- `num_degree`, `den_degree`: Polynomial degrees for numerator and denominator
- `n0, n1, n2, ...`: Numerator coefficients (degree + 1 values)
- `d0, d1, d2, ...`: Denominator coefficients (degree + 1 values, d0 typically 1.0)

**Available CSV files:** `~/workspace/tt-polynomial-fitter/data/coefficients/*_rational.csv`

## Build Instructions

### 1. Generate CMake Targets

The rational targets have already been appended to `CMakeLists.txt`:

```bash
# To regenerate (if needed)
python3 generate_cmake_rational.py
```

This creates 6 targets:
- `programming_examples_generic_lut_activation_rational_linear_linear_1` (1,1) × 1 segment
- `programming_examples_generic_lut_activation_rational_linear_linear_2` (1,1) × 2 segments
- `programming_examples_generic_lut_activation_rational_quadratic_linear_1` (2,1) × 1 segment
- `programming_examples_generic_lut_activation_rational_quartic_quartic_1` (4,4) × 1 segment
- `programming_examples_generic_lut_activation_rational_quartic_quartic_2` (4,4) × 2 segments
- `programming_examples_generic_lut_activation_rational_nonic_octic_1` (9,8) × 1 segment

### 2. Build Kernels

```bash
cd build
cmake .. -G Ninja
ninja programming_examples_generic_lut_activation_rational_quartic_quartic_2
```

Or build all rational targets:

```bash
ninja | grep rational
```

## Usage

### Run Single Test

```bash
# Example: sigmoid with rational (4,4) × 2 segments
./build/programming_examples/programming_examples_generic_lut_activation_rational_quartic_quartic_2 \
    ~/workspace/tt-polynomial-fitter/data/coefficients/sigmoid_fp32_2_4_4_uniform_rational.csv \
    --activation sigmoid \
    --range-min -10 \
    --range-max 10 \
    --tiles 32
```

### Python Validation

Test rational kernel output against reference implementation:

```bash
# Compute reference output
python3 test_piecewise_rational.py \
    ~/workspace/tt-polynomial-fitter/data/coefficients/sigmoid_fp32_2_4_4_uniform_rational.csv \
    --range -10 10 \
    --points 1024

# Compare hardware output (if output.csv was generated)
python3 test_piecewise_rational.py \
    sigmoid_fp32_2_4_4_rational.csv \
    --range -10 10 \
    --hardware output.csv
```

### Automated Benchmarking

Run sweep across all available rational CSV files:

```bash
./sweep_rational.sh

# Results saved to:
# plots/rational/rational_sweep_results.csv
```

## Supported Activations

Rational CSV files are available for:
- **sigmoid** - (4,4) × 2 segments, MAE ~6e-6
- **erf** - (9,8) × 1 segment, MAE ~varies
- **atanh** - (2,1) × 1 segment, MAE ~varies
- **hardsigmoid** - (1,1) × 1 segment, MAE ~varies
- **prelu** - (4,4) × 1 segment, MAE ~varies

All CSV files are in: `~/workspace/tt-polynomial-fitter/data/coefficients/*_rational.csv`

## Implementation Details

### LUT Memory Layout

```
[b0, b1, ..., bN,
 n0_seg0, n1_seg0, ..., nm_seg0, d0_seg0, d1_seg0, ..., dk_seg0,  // segment 0
 n0_seg1, n1_seg1, ..., nm_seg1, d0_seg1, d1_seg1, ..., dk_seg1,  // segment 1
 ...]
```

**LUT Size Calculation:**
```
LUT_SIZE = (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (NUM_DEGREE + DEN_DEGREE + 2)
```

Example: sigmoid with (4,4) × 2 segments:
- Boundaries: 3 values
- Coefficients per segment: 5 numerator + 5 denominator = 10 values
- Total: 3 + 2×10 = **23 floats**

### Horner's Method for Rationals

Numerically stable evaluation:

```cpp
// Evaluate numerator: P(x) = n0 + n1*x + ... + nm*x^m
vFloat num = num_coeffs[m];
for (int i = m-1; i >= 0; i--) {
    num = num * x + num_coeffs[i];
}

// Evaluate denominator: Q(x) = d0 + d1*x + ... + dk*x^k
vFloat den = den_coeffs[k];
for (int i = k-1; i >= 0; i--) {
    den = den * x + den_coeffs[i];
}

// Single division at end
return num / den;
```

### Segment Selection

Same cascaded `v_if` pattern as polynomial kernels:

```cpp
// Start with segment 0
vFloat result = eval_rational<NUM_DEGREE, DEN_DEGREE>(num_coeffs_0, den_coeffs_0, x);

// Update for correct segment
for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
    v_if (x_clamped >= lut[seg]) {
        result = eval_rational<NUM_DEGREE, DEN_DEGREE>(num_coeffs_seg, den_coeffs_seg, x);
    }
    v_endif;
}
```

## Performance Characteristics

### Computational Complexity

- **Polynomial (degree n):** n multiplications + n additions
- **Rational (degree m/k):** (m + k) multiplications + (m + k) additions + 1 division
- **Expected slowdown:** ~1.5-2× for equal degrees (due to division and two Horner chains)

### Memory Footprint

- **Polynomial:** `(depth + 1) + depth * (degree + 1)` floats
- **Rational:** `(depth + 1) + depth * (num_deg + den_deg + 2)` floats
- **Ratio:** ~2× storage for same degree, but rational needs far fewer segments!

### Accuracy vs Performance Trade-off

Example: **sigmoid**

| Method | Segments | Degree | LUT Size | MAE | Relative Speed |
|--------|----------|--------|----------|-----|----------------|
| Polynomial | 16 | 4 | 85 floats | ~1e-3 | 1.0× |
| Rational | 2 | (4,4) | 23 floats | ~6e-6 | ~0.5-0.7× |

**Rational wins:** 4× less memory, 150× better accuracy, at cost of ~1.5-2× slower compute.

## Troubleshooting

### Build Errors

**Error:** `LUT size mismatch`
- **Cause:** CSV file has wrong number of coefficients
- **Fix:** Verify CSV format and degree values

**Error:** `Degree mismatch`
- **Cause:** Compile-time NUM_DEGREE/DEN_DEGREE doesn't match CSV
- **Fix:** Use correct target for your CSV configuration

### Runtime Errors

**Error:** `Division by zero`
- **Cause:** Denominator evaluates to 0 (should not happen with proper CSV files)
- **Fix:** Check CSV denominator coefficients, d0 should be 1.0

**Error:** `Numerical instability`
- **Cause:** High-degree polynomials can be unstable
- **Fix:** Use lower degrees or better-fitted coefficients

## Future Enhancements

### Planned Optimizations

1. **Optimized Rational Kernel** (`piecewise_rational_opt.cpp`)
   - Per-lane coefficient loading (like polynomial opt)
   - Single rational evaluation per destination register
   - Target: 10-15× speedup for high segment counts

2. **Embedded LUT Mode**
   - Compile rational LUT directly into kernel binary
   - Zero L1 memory overhead
   - Ideal for production deployments

3. **Mixed Rational/Polynomial Segments**
   - Some segments use rational, others use polynomial
   - Automatically select best approximation per segment
   - Maximize accuracy while minimizing compute

## References

- **Polynomial Kernels:** `piecewise_generic.cpp` - Original polynomial implementation
- **LUT Loader:** `lut_loader.hpp` - Polynomial CSV parser (reference)
- **Main README:** `README.md` - Full activation function documentation
- **Polynomial Fitter:** `~/workspace/tt-polynomial-fitter/` - CSV generation tool
