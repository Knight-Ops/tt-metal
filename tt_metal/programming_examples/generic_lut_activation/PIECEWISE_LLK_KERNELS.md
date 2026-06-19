# Piecewise LLK Kernel Setup for Generic LUT Activation Functions

This document describes the piecewise polynomial LUT-based activation function implementation for Wormhole and Blackhole hardware.

## Overview

The piecewise LLK kernel system enables hardware-accelerated evaluation of arbitrary activation functions using:
- **Piecewise polynomial approximation**: Function domain split into segments, each with its own polynomial
- **LUT-based coefficients**: Polynomial coefficients stored in L1 circular buffer
- **SFPU vector processing**: 32-element parallel evaluation using destination registers

## LUT Data Format

```
LUT Array: [boundaries...] [coefficients...]
           [b0, b1, ..., bN] [seg0_c0, seg0_c1, ..., seg0_cD, seg1_c0, ...]

Size Calculation:
  Boundaries:    NUM_SEGMENTS + 1 entries
  Coefficients:  NUM_SEGMENTS * (POLY_DEGREE + 1) entries
  Total:         LUT_SIZE = (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)

Example (4 segments, degree 1 linear):
  Boundaries:   5 values (b0=-4, b1=-2, b2=0, b3=2, b4=4)
  Coefficients: 8 values (4 segments * 2 coeffs each: c0, c1)
  Total:        13 floats
```

## Key Files

| Component | Path |
|-----------|------|
| **Generic Piecewise Kernel** | `kernels/compute/piecewise_generic.cpp` |
| **Specialized (4/8/16 seg)** | `kernels/compute/piecewise_generic_specialized.cpp` |
| **Rational Kernel** | `kernels/compute/piecewise_rational.cpp` |
| **Quadratic Estrin** | `kernels/compute/piecewise_quadratic_estrin.cpp` |
| **Reader** | `kernels/dataflow/reader.cpp` |
| **Writer** | `kernels/dataflow/writer.cpp` |
| **LUT Loader** | `lut_loader.hpp` |
| **Host Program** | `generic_lut_activation.cpp` |
| **LLK Polynomial Eval** | `third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_polyval.h` |

## Kernel Architecture

### Polynomial Evaluation (Horner's Method)

```cpp
// Generic template - specialized for each degree
template <uint32_t DEGREE>
inline vFloat eval_polynomial(const float* coeffs, vFloat x);

// Degree 1 (Linear): y = c0 + c1*x
template <> inline vFloat eval_polynomial<1>(const float* c, vFloat x) {
    return c[0] + c[1] * x;
}

// Degree 2 (Quadratic): y = c0 + c1*x + c2*x^2
template <> inline vFloat eval_polynomial<2>(const float* c, vFloat x) {
    return (c[2] * x + c[1]) * x + c[0];
}

// Higher degrees follow same Horner pattern
```

### Segment Selection

```cpp
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;  // After boundaries

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];

        // Clamp to valid range
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[NUM_SEGMENTS]) { x_clamped = lut[NUM_SEGMENTS]; } v_endif;

        // Evaluate first segment
        vFloat result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET], x);

        // Cascading v_if for segment selection
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            v_if (x_clamped >= lut[seg]) {
                result = eval_polynomial<POLY_DEGREE>(
                    &lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT], x);
            } v_endif;
        }

        dst_reg[d] = result;
    }
}
```

## Data Flow Pipeline

```
Host                                    Device (Tensix Core)
  |                                         |
  |-- Load CSV coefficients                 |
  |-- Build LUT: [boundaries, coeffs]       |
  |-- Write LUT to L1 (CB 25)          ---> LUT in L1 (persistent)
  |-- Write input to DRAM              ---> Input Buffer (DRAM)
  |                                         |
  |                                    Reader Kernel (RISCV_0)
  |                                         |-- Read tiles: DRAM -> CB_0
  |                                         |
  |                                    Compute Kernel (TRISC)
  |                                         |-- Wait CB_0
  |                                         |-- Load tile to dst registers
  |                                         |-- For each of 32 elements:
  |                                         |     |-- Find segment
  |                                         |     |-- Evaluate polynomial
  |                                         |-- Push to CB_16
  |                                         |
  |                                    Writer Kernel (RISCV_1)
  |                                         |-- Write tiles: CB_16 -> DRAM
  |                                         |
  |<-- Read output from DRAM           <--- Output Buffer (DRAM)
```

## Build Configuration

### CMake Targets

107 total build targets covering:
- **Polynomial**: `programming_examples_generic_lut_activation_p{deg}_s{segs}`
  - Degrees: 1, 2, 3, 4, 5, 6, 8, 12, 16, 32
  - Segments: 1, 2, 3, 4, 6, 8, 12, 16, 32
- **Rational**: `programming_examples_generic_lut_activation_n{num}_d{den}_s{segs}`

### Build Commands

```bash
cd /localdev/nkapre/tt-metal
./build_metal.sh --build-programming-examples

# Run specific configuration
./build_Release/programming_examples/programming_examples_generic_lut_activation_p2_s8 \
    path/to/coefficients.csv \
    --activation gelu --range-min -4 --range-max 4 --tiles 32
```

### Compile-Time Defines

```cpp
// Kernel variant selection
#define KERNEL_VARIANT "piecewise_generic"  // or "piecewise_rational"

// Configuration (passed via compile_args)
LUT_SIZE      // Total LUT entries
POLY_DEGREE   // Polynomial degree (0-32)
NUM_SEGMENTS  // Number of piecewise segments

// FP32 mode (for Float32 data format)
#define PACKER_L1_ACC 1
```

## CSV Coefficient Format

Input CSV from polynomial fitter:

```csv
segment_id,lo,hi,c0,c1,c2,...,error,method,segmentation
0,-4.0,-2.0,-0.0160,0.0186,...,0.00123,errordriven,curvature
1,-2.0,0.0,-0.1301,0.0658,...,0.00098,errordriven,curvature
...
```

The LUTLoader parses this into the packed format: `[boundaries, coefficients]`.

## Precision Modes

### BF16 Mode
- `math_approx_mode = true`
- Hardware approximations enabled
- Lower precision, faster execution

### FP32 Mode
```cpp
ComputeConfig{
    .fp32_dest_acc_en = true,
    .unpack_to_dest_mode = UnpackToDestFp32,
    .math_approx_mode = false
}
// Plus define: PACKER_L1_ACC=1
```

## Wormhole Compiler Workaround

The Wormhole SFPU compiler has a bug with `v_if` cascades inside runtime for-loops. The specialized kernel (`piecewise_generic_specialized.cpp`) provides manually unrolled versions for 4, 8, and 16 segments.

## Performance Notes

- **Polynomial**: O(degree) multiplies using Horner's method
- **Segment selection**: O(segments) cascading v_if comparisons
- **Throughput**: 32 elements processed in parallel per tile
- **LUT access**: L1 resident, no DRAM traffic during evaluation
- **Estrin variant**: ~1.5-2x speedup for quadratic using SFPLUT hardware

## Kernel Variants

| Variant | Best For | Notes |
|---------|----------|-------|
| `piecewise_generic` | General use | Flexible degree/segments |
| `piecewise_generic_specialized` | 4/8/16 segments | Wormhole workaround |
| `piecewise_rational` | Better approximations | P(x)/Q(x) form |
| `piecewise_quadratic_estrin` | Quadratic | SFPLUT acceleration |
| `native_sfpu` | Hardware activations | GELU, sigmoid, etc. |
