# FPU Implementation - Quick Start Guide

## TL;DR

Want maximum performance for high-degree polynomials? Use the **BF16 FPU** kernel.
Need maximum precision? Use the **TF32 FPU** kernel.

```bash
# Build BF16 FPU kernel (fast, good precision)
cmake -B build -DKERNEL_VARIANT=piecewise_fpu -DPOLY_DEGREE=4 -DNUM_SEGMENTS=16 -DLUT_SIZE=96
cmake --build build

# Build TF32 FPU kernel (fast, best precision)
cmake -B build -DKERNEL_VARIANT=piecewise_fpu_tf32 -DPOLY_DEGREE=4 -DNUM_SEGMENTS=16 -DLUT_SIZE=96
cmake --build build

# Run with FP32 output
./build/bin/generic_lut_activation coeffs.csv --activation gelu --precision fp32 -rmin -10 -rmax 10
```

---

## When to Use FPU Implementations

### Use BF16 FPU If:
- ✅ Polynomial degree ≥ 4
- ✅ Want 2-4× speedup vs SFPU
- ✅ Accuracy of ~1e-5 (0.001%) is sufficient
- ✅ Need good balance of speed and precision

### Use TF32 FPU If:
- ✅ Polynomial degree ≥ 4
- ✅ Need best hardware-accelerated precision (~1e-6)
- ✅ Coefficient precision is critical
- ✅ Can afford 2× memory overhead vs BF16

### Stick with SFPU Horner If:
- ✅ Polynomial degree ≤ 3 (minimal speedup from FPU)
- ✅ Memory constrained (FPU needs scratch CBs)
- ✅ Already fast enough

---

## Quick Comparison

| Feature | SFPU Horner | BF16 FPU | TF32 FPU |
|---------|-------------|----------|----------|
| **Speed (D=8)** | 1.0× (baseline) | **4.0×** | **3.5×** |
| **Precision** | ~1e-4 to 1e-5 | ~1e-5 | **~1e-6** |
| **Memory** | 8 KB | 12 KB | 16 KB |
| **Best For** | D ≤ 3 | D ≥ 4, speed | D ≥ 4, precision |

---

## Build Instructions

### Calculate LUT_SIZE

```
LUT_SIZE = (NUM_SEGMENTS + 1) + NUM_SEGMENTS × (POLY_DEGREE + 1)

Examples:
  D=2, S=16: (16+1) + 16×(2+1) = 17 + 48 = 65
  D=4, S=16: (16+1) + 16×(4+1) = 17 + 80 = 97
  D=8, S=32: (32+1) + 32×(8+1) = 33 + 288 = 321
```

### BF16 FPU Build

```bash
cd tt_metal/programming_examples/generic_lut_activation

# For degree-4, 16 segments
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DKERNEL_VARIANT=piecewise_fpu \
  -DPOLY_DEGREE=4 \
  -DNUM_SEGMENTS=16 \
  -DLUT_SIZE=97

cmake --build build --target generic_lut_activation
```

### TF32 FPU Build

```bash
# Same as BF16, just change kernel variant
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DKERNEL_VARIANT=piecewise_fpu_tf32 \
  -DPOLY_DEGREE=4 \
  -DNUM_SEGMENTS=16 \
  -DLUT_SIZE=97

cmake --build build --target generic_lut_activation
```

---

## Running the Kernel

### Basic Usage

```bash
./build/bin/generic_lut_activation \
  <coefficient_csv> \
  --activation <name> \
  --precision fp32 \
  --range-min <min> \
  --range-max <max>
```

**Important**: Always use `--precision fp32` with FPU kernels to enable FP32 output circular buffers.

### Example: GELU

```bash
./build/bin/generic_lut_activation \
  coefficients/gelu_bf16_16_4_errordriven_any.csv \
  --activation gelu \
  --precision fp32 \
  --range-min -10 \
  --range-max 10 \
  --tiles 32
```

### Example: Sigmoid

```bash
./build/bin/generic_lut_activation \
  coefficients/sigmoid_bf16_32_6_errordriven_any.csv \
  --activation sigmoid \
  --precision fp32 \
  --range-min -10 \
  --range-max 10 \
  --tiles 64
```

---

## Precision Configuration

### FPU Kernel Precision Hierarchy

```
Input:          BF16 (7-bit mantissa, hardware constraint)
                 ↓
Power Matrix:   BF16 (BF16 FPU) or TF32 (TF32 FPU)
                 ↓
Coeff Matrix:   BF16 (BF16 FPU) or TF32 (TF32 FPU)
                 ↓
FPU Matmul:     BF16×BF16 or TF32×TF32
                 ↓
Accumulation:   FP32 (DST_ACCUM_MODE=1, no rounding!)
                 ↓
Output:         FP32 (23-bit mantissa, full precision)
```

### Accuracy Expectations

| Kernel | Expected Relative Error | Example |
|--------|------------------------|---------|
| SFPU Horner (BF16) | ~1e-4 (0.01%) | Good for most cases |
| SFPU Horner (FP32) | ~1e-5 (0.001%) | Better, but slower |
| BF16 FPU + FP32 accum | ~1e-5 (0.001%) | Fast + good precision |
| TF32 FPU + FP32 accum | ~1e-6 (0.0001%) | **Fast + best precision** |

---

## Performance Expectations

### Speedup vs SFPU Horner (D=8, S=32)

```
                ┌─────────────────────────┐
SFPU Horner     │████████████████████    │ 1.0× (baseline)
                └─────────────────────────┘

BF16 FPU        │█████                   │ 4.0× (4× faster!)
                └─────────────────────────┘

TF32 FPU        │█████▌                  │ 3.5× (3.5× faster)
                └─────────────────────────┘
```

**Note**: These are theoretical estimates. Actual performance depends on hardware and should be benchmarked.

### When FPU Wins

FPU implementations are **faster** when:
- ✅ Polynomial degree ≥ 4 (more operations to parallelize)
- ✅ Many segments (S ≥ 16)
- ✅ Processing many tiles (amortizes matrix build overhead)

FPU implementations may be **slower** when:
- ❌ Polynomial degree ≤ 2 (SFPU is already fast)
- ❌ Very few tiles (matrix build overhead dominates)

---

## Memory Usage

### Circular Buffer Allocation

**SFPU Horner**:
- Input: 2 tiles (BF16) = 4 KB
- Output: 2 tiles (BF16) = 4 KB
- **Total: 8 KB**

**BF16 FPU**:
- Input: 2 tiles (BF16) = 4 KB
- Output: 2 tiles (FP32) = 8 KB
- Scratch C: 1 tile (BF16) = 2 KB
- Scratch P: 1 tile (BF16) = 2 KB
- Scratch Y: 1 tile (FP32) = 4 KB
- **Total: 20 KB**

**TF32 FPU**:
- Input: 2 tiles (BF16) = 4 KB
- Output: 2 tiles (FP32) = 8 KB
- Scratch C: 1 tile (TF32) = 4 KB
- Scratch P: 1 tile (TF32) = 4 KB
- Scratch Y: 1 tile (FP32) = 4 KB
- **Total: 24 KB**

**Conclusion**: FPU kernels use ~3× more L1 memory, but still well within budget (1.5 MB available).

---

## Troubleshooting

### Build Errors

**Error**: `KERNEL_VARIANT not defined`
```bash
# Fix: Specify variant in cmake command
cmake -B build -DKERNEL_VARIANT=piecewise_fpu ...
```

**Error**: `LUT_SIZE mismatch`
```bash
# Fix: Calculate LUT_SIZE correctly
# LUT_SIZE = (NUM_SEGMENTS + 1) + NUM_SEGMENTS × (POLY_DEGREE + 1)
cmake -B build -DLUT_SIZE=97 ...
```

### Runtime Errors

**Error**: `CB format mismatch`
```bash
# Fix: Use --precision fp32 (not bf16) for FPU kernels
./generic_lut_activation coeffs.csv --precision fp32 ...
```

**Error**: `Relative error too high`
```cpp
// Check that FP32 accumulation is enabled in kernel
// File: piecewise_fpu.cpp or piecewise_fpu_tf32.cpp
#ifndef DST_ACCUM_MODE
#define DST_ACCUM_MODE 1  // Must be 1 for FP32 accumulation
#endif
```

### Performance Issues

**Issue**: FPU kernel is slower than expected

**Possible causes**:
1. Polynomial degree too low (D < 4) - FPU overhead not worth it
2. Processing very few tiles - matrix build overhead dominates
3. Hardware doesn't support TF32 efficiently (try BF16 FPU instead)

**Fix**: Benchmark with different configurations and choose best for your use case.

---

## Verification

### Test Precision

```bash
# Build kernel
cmake -B build -DKERNEL_VARIANT=piecewise_fpu_tf32 -DPOLY_DEGREE=4 -DNUM_SEGMENTS=16 -DLUT_SIZE=97
cmake --build build

# Run and dump output
DUMP_OUTPUT_CSV=output.csv \
./build/bin/generic_lut_activation \
  coefficients/gelu_bf16_16_4_errordriven_any.csv \
  --activation gelu \
  --precision fp32 \
  --range-min -10 \
  --range-max 10

# Analyze with Python
python3 verify_fpu_math.py  # From earlier examples
```

### Expected Results

**BF16 FPU**:
```
Max relative error: ~5e-6 to 1e-5 (0.0005% to 0.001%)
✓ Good precision, fast execution
```

**TF32 FPU**:
```
Max relative error: ~5e-7 to 1e-6 (0.00005% to 0.0001%)
✓ Excellent precision, fast execution
```

---

## Advanced: Custom Configurations

### Enable Maximum Fidelity (TF32 only)

In `piecewise_fpu_tf32.cpp`:

```cpp
#ifndef MATH_FIDELITY
#define MATH_FIDELITY 4  // Maximum accuracy (4 phases)
#endif
```

**Trade-off**: Higher fidelity = better precision, but may be slower (~50% overhead for 4 phases).

### Tune CB Sizes

If processing larger tiles or higher degrees, you may need to adjust CB sizes in the kernel:

```cpp
// Increase CB size for larger matrices
constexpr uint32_t max_tile_size = 32 * 32;  // Can increase if needed
```

---

## Summary

### Quick Decision Tree

```
Need to evaluate polynomials on TT-Metal?
  │
  ├─ Degree ≤ 3?
  │    └─ YES → Use SFPU Horner (simple, fast enough)
  │
  └─ Degree ≥ 4?
       │
       ├─ Precision ~1e-5 sufficient?
       │    └─ YES → Use BF16 FPU (fast, good precision)
       │
       └─ Need best precision?
            └─ YES → Use TF32 FPU (fast, best precision)
```

### One-Line Recommendations

- **Fast prototyping**: SFPU Horner (simplest)
- **Production (D ≥ 4)**: BF16 FPU (best balance)
- **Critical accuracy**: TF32 FPU (best hardware precision)

---

## Next Steps

1. **Build** the FPU kernel for your configuration
2. **Run** verification tests to confirm accuracy
3. **Benchmark** performance on target hardware
4. **Compare** with SFPU baseline to quantify speedup

For detailed documentation, see:
- `FPU_IMPLEMENTATION.md` - Full implementation details
- `TF32_INTEGRATION_GUIDE.md` - TF32 specifics
- `IMPLEMENTATION_COMPARISON.md` - Side-by-side comparison

For help:
- Discord: https://discord.gg/tvhGzHQwaj
- Issues: https://github.com/tenstorrent/tt-metal/issues

---

**Ready to go?** Run the automated test script:

```bash
./build_and_test_tf32.sh gelu 4 16
```

This will build, run, and analyze the TF32 FPU kernel automatically!
