# FPU-Based Piecewise Polynomial Activation Implementation

## Overview

This document describes the FPU-based implementation of piecewise polynomial activation (`piecewise_fpu.cpp`), which uses matrix multiplication on the FPU (matrix engine) instead of Horner's method on the SFPU.

## Key Concept

Instead of evaluating each segment's polynomial separately using SFPU Horner iteration:
```
For each segment s:
    if x in segment s:
        result = c0 + c1*x + c2*x^2 + ... + cD*x^D  // Horner's method
```

We build matrices and compute all segments simultaneously using one FPU matmul:
```
For entire tile (32 values):
    P = power matrix [K×32]:     P[k,j] = x[j]^k
    C = coefficient matrix [S×K]: C[s,k] = coefficient k of segment s
    Y = C × P [S×32]:            Y[s,j] = polynomial s evaluated at x[j]
    For each lane j:
        result[j] = Y[seg_id[j], j]  // Select appropriate segment
```

## Architecture

### Data Flow

```
Input Tile (32 values in dst_reg[])
    ↓
Extract x values → Build P[K×32] and C[S×K]
    ↓
Write P and C to Circular Buffers
    ↓
FPU Matmul: Y = C × P
    ↓
Read Y from CB, select appropriate row per lane
    ↓
Write results back to dst_reg[]
```

### Four Core Helper Functions

The implementation provides four functions that encapsulate the FPU matmul operations:

#### 1. `FPU_MATMUL_load_C(const float C[32][32])`

**Purpose**: Load coefficient matrix into SrcB registers

**Implementation**:
- Converts C from float32 to bfloat16
- Writes to circular buffer CB_SCRATCH_C (index 26)
- Unpacker will later load from CB to SrcB via `llk_unpack_AB_matmul()`

**Matrix Layout**:
- Logical: S×K (segments × polynomial_coeffs)
- Physical: 32×32 tile (zero-padded)
- C[s][k] = coefficient k for segment s

#### 2. `FPU_MATMUL_load_P(const float P[32][32])`

**Purpose**: Load power matrix into SrcA registers

**Implementation**:
- Converts P from float32 to bfloat16
- Writes to circular buffer CB_SCRATCH_P (index 27)
- Unpacker will later load from CB to SrcA via `llk_unpack_AB_matmul()`

**Matrix Layout**:
- Logical: K×32 (polynomial_coeffs × lanes)
- Physical: 32×32 tile (zero-padded)
- P[k][j] = x[j]^k

#### 3. `FPU_MATMUL_compute()`

**Purpose**: Perform hardware FPU matrix multiplication

**Implementation**:
- Waits for input CBs to have data
- Calls `matmul_tiles(CB_SCRATCH_C, CB_SCRATCH_P, 0, 0, 0)`
- This internally calls:
  - `llk_unpack_AB_matmul()`: Unpacks C→SrcB, P→SrcA
  - `llk_math_matmul()`: Computes Dest = SrcB × SrcA = C × P
- Packs result from Dest to CB_SCRATCH_Y (index 28)

**Operand Mapping** (Important!):
- `in0_cb` (CB_SCRATCH_C) → SrcB
- `in1_cb` (CB_SCRATCH_P) → SrcA
- Matmul computes: Dest = SrcB × SrcA = C × P

This operand swap is handled automatically by `mm_init()`.

#### 4. `FPU_MATMUL_read_Y(float Y[32][32])`

**Purpose**: Read result matrix from circular buffer back to local memory

**Implementation**:
- Waits for result in CB_SCRATCH_Y
- Reads tile data from CB
- Converts from bfloat16 to float32
- Returns Y[S×32] where Y[s][j] = segment s polynomial evaluated at x[j]

### LLK APIs Used

| Function | LLK API (via wrappers) | RISC | Purpose |
|----------|----------------------|------|---------|
| FPU_MATMUL_load_C | llk_unpack_AB_matmul | UNPACK | Load C from CB to SrcB |
| FPU_MATMUL_load_P | llk_unpack_AB_matmul | UNPACK | Load P from CB to SrcA |
| FPU_MATMUL_compute | llk_math_matmul | MATH | Compute Dest = SrcB × SrcA |
| FPU_MATMUL_read_Y | pack_tile | PACK | Pack Dest to CB |

All LLK calls are made through high-level wrapper APIs:
- `matmul_tiles()` wraps `llk_unpack_AB_matmul()` and `llk_math_matmul()`
- `mm_init()` initializes the unpacker, math engine, and packer for matmul
- `pack_tile()` wraps LLK pack operations

## Complexity Analysis

### SFPU Horner Method (Original)

For S segments and degree D polynomial:
- **Per lane**: O(S) segment searches + O(D) Horner steps
- **Per tile**: O(32 × S × D) SFPU operations

Example: S=32 segments, D=8 degree
- Operations: 32 × 32 × 8 = 8,192 SFPU ops per tile

### FPU Matmul Method (This Implementation)

For same configuration:
- **Per tile**: Build P[9×32], Build C[32×9], 1 FPU matmul (32×9 by 9×32)
- FPU matmul is hardware-accelerated, fixed-latency
- **Effective ops**: 1 hardware matmul

Example: S=32, D=8
- Matrix build: Lightweight (computed once)
- Matmul: 1 FPU operation (hardware pipelined)

## Performance Characteristics

### When FPU is Better

- **High segment count** (S ≥ 16): SFPU cost scales with S
- **High polynomial degree** (D ≥ 6): SFPU cost scales with D
- **Throughput-sensitive**: FPU matmul is hardware pipelined

### When SFPU is Better

- **Low segment count** (S ≤ 8): Overhead of matmul setup not justified
- **Very simple polynomials** (D ≤ 2): Horner is very fast
- **Latency-sensitive**: SFPU can start immediately, no CB setup

### Numerical Considerations

- Both use bfloat16 precision for computation
- FPU matmul may have slightly different rounding behavior
- For same polynomial, results should be within acceptable FP error bounds

## Usage

### Build Configuration

The kernel supports both embedded LUT and CB-based LUT modes:

**Embedded LUT Mode**:
```cpp
#define EMBEDDED_LUT
#define LUT_SIZE 41        // (8+1) + 8*(4+1) = 41 for 8 segments, degree 4
#define POLY_DEGREE 4
#define NUM_SEGMENTS 8
#define LUT_DATA lut_constant_array
```

**CB-based LUT Mode**:
```cpp
// Pass as compile-time args:
// - lut_size
// - poly_degree
// - num_segments
```

### Circular Buffer Requirements

The kernel requires 3 additional scratch CBs beyond standard input/output:

```cpp
CB 26 (CB_SCRATCH_C): 1 tile, bfloat16 format - Coefficient matrix
CB 27 (CB_SCRATCH_P): 1 tile, bfloat16 format - Power matrix
CB 28 (CB_SCRATCH_Y): 1 tile, bfloat16 format - Result matrix
```

These must be configured in the host program when creating the kernel.

### Example Host Code Snippet

```cpp
// Configure scratch CBs for FPU matmul
CircularBufferConfig cb_scratch_c_config =
    CircularBufferConfig(1 * tile_size, {{CB_SCRATCH_C, data_format}})
        .set_page_size(CB_SCRATCH_C, tile_size);
auto cb_scratch_c = CreateCircularBuffer(program, core, cb_scratch_c_config);

// Similarly for CB_SCRATCH_P and CB_SCRATCH_Y...

// Create compute kernel
auto compute_kernel = CreateKernel(
    program,
    "kernels/compute/piecewise_fpu.cpp",
    core,
    ComputeConfig{
        .compile_args = {lut_size, poly_degree, num_segments},
        // ...
    }
);
```

## Constraints and Limitations

### Compile-Time Constraints

```cpp
static_assert(POLY_DEGREE + 1 <= 32, "Polynomial degree limited by tile size");
static_assert(NUM_SEGMENTS <= 32, "Segment count limited by tile size");
static_assert(LUT_SIZE == (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1));
```

### Runtime Constraints

- Requires 3 scratch CBs with 1 tile capacity each
- Input/output tiles must be in standard format (CB 0 for input, CB 16 for output)
- LUT CB (if used) must be CB 25

### Compatibility

- Drop-in compatible with existing SFPU LUT format
- Same boundary clamping behavior as SFPU kernel
- Same segment selection logic (binary search with >= comparison)

## Testing and Validation

### Correctness Checks

1. **Numerical Accuracy**: Results should match SFPU Horner within bfloat16 precision
2. **Boundary Behavior**: Exact match at segment boundaries
3. **Edge Cases**:
   - Constant polynomials (degree 0)
   - Linear polynomials (degree 1)
   - Single segment (no piecewise)
   - All-zero polynomials

### Performance Benchmarks

Compare against SFPU version using `kernel_bench`:
```bash
./build/programming_examples/generic_lut_activation/kernel_bench \
    --mode fpu \
    --degree 8 \
    --segments 32 \
    --tiles 3200
```

Expected: FPU version 2-3× faster for high S×D configurations.

## Implementation Notes

### Tile Format Simplification

The current implementation uses a simplified row-major interpretation of tile layout for reading/writing to CBs. The actual TT-Metal tile format is more complex (organized into 4 faces of 16×16 with specific patterns), but for the matmul use case, the hardware handles the format conversion automatically.

For production use, consider using proper tile format conversion utilities:
- `ttnn::operations::primary::write_tile()`
- `ttnn::operations::primary::read_tile()`

### Memory Layout

All matrices are stored as 32×32 physical tiles, regardless of logical dimensions:
- C[S×K]: Only rows 0..S-1 and cols 0..K-1 contain data
- P[K×32]: Only rows 0..K-1 contain data
- Y[S×32]: Only rows 0..S-1 contain valid results

Zero-padding is used for unused regions.

### Optimization Opportunities

1. **Block Multiple Tiles**: Instead of processing one tile at a time, batch multiple tiles to amortize matmul setup cost
2. **Reuse Coefficient Matrix**: If processing multiple tiles with the same LUT, C can be loaded once and reused
3. **Specialized Tile Format**: Use proper tile format for better memory efficiency
4. **FP32 Accumulation**: For higher precision, use DST_ACCUM_MODE=1 with fp32 accumulation

## References

- TT-Metal Matmul API: `tt_metal/include/compute_kernel_api/matmul.h`
- LLK Unpack API: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_unpack_AB_matmul_api.h`
- LLK Math API: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_matmul_api.h`
- Matmul Examples: `tt_metal/programming_examples/matmul/matmul_single_core/`
- SFPU Generic Kernel: `kernels/compute/piecewise_generic.cpp`
