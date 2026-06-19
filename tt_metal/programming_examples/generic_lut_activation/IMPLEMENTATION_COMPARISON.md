# Piecewise Polynomial Implementation Comparison

## Overview

Three implementations are available for piecewise polynomial activation functions on TT-Metal:

1. **SFPU Horner** - Original implementation using SFPU vector operations
2. **BF16 FPU** - Matrix multiplication using BF16 precision with FP32 accumulation
3. **TF32 FPU** - Matrix multiplication using TF32 precision with FP32 accumulation

This document compares all three approaches across multiple dimensions.

---

## Quick Comparison Table

| Feature | SFPU Horner | BF16 FPU | TF32 FPU |
|---------|-------------|----------|----------|
| **Implementation** | `piecewise_generic.cpp` | `piecewise_fpu.cpp` | `piecewise_fpu_tf32.cpp` |
| **Hardware Unit** | SFPU (vector) | FPU (matrix) | FPU (matrix) |
| **Algorithm** | Horner's method | Matrix multiply | Matrix multiply |
| **Complexity** | O(S × D) per tile | O(S × K × 32) matmul | O(S × K × 32) matmul |
| **Coefficient Precision** | BF16 (7-bit mantissa) | BF16 (7-bit mantissa) | TF32 (10-bit mantissa) |
| **Power Precision** | BF16 (7-bit mantissa) | BF16 (7-bit mantissa) | TF32 (10-bit mantissa) |
| **Accumulation** | BF16 or FP32 | FP32 (DST_ACCUM_MODE=1) | FP32 (DST_ACCUM_MODE=1) |
| **Output Format** | BF16 or FP32 | FP32 | FP32 |
| **Relative Error** | ~1e-4 to 1e-5 | ~1e-5 | ~1e-6 |
| **Performance** | Baseline | **Faster** for D≥4 | **Faster** for D≥4 |
| **Memory Usage** | Low (no scratch CBs) | Medium (3 scratch CBs) | Medium (3 scratch CBs) |
| **Best For** | Low degree (D≤3) | High degree (D≥4) | Maximum precision |

---

## Detailed Comparison

### 1. SFPU Horner Implementation

**File**: `piecewise_generic.cpp`

#### Algorithm

```
For each input x in tile:
  1. Determine segment s via boundary search
  2. Load coefficients c[0..D] for segment s
  3. Evaluate polynomial using Horner's method:
     result = c[D]
     for k = D-1 down to 0:
       result = result * x + c[k]
  4. Store result
```

#### Pros

- ✅ **Simple and proven** - Well-understood algorithm
- ✅ **Low memory** - No scratch circular buffers needed
- ✅ **Fast for low degrees** - D≤3, minimal overhead
- ✅ **Numerically stable** - Horner's method minimizes error accumulation
- ✅ **Flexible** - Easy to add custom SFPU operations (range reduction, etc.)

#### Cons

- ❌ **Slow for high degrees** - D≥8 requires many sequential operations
- ❌ **Sequential** - Cannot parallelize within a tile
- ❌ **BF16 precision** - Limited by 7-bit mantissa (unless FP32 mode)

#### Performance Characteristics

```
Degree 2:  ~32 × 2 = 64 SFPU operations per tile     (FAST)
Degree 4:  ~32 × 4 = 128 SFPU operations per tile    (OKAY)
Degree 8:  ~32 × 8 = 256 SFPU operations per tile    (SLOW)
```

For 32 segments with degree 8, total cost is dominated by polynomial evaluation.

---

### 2. BF16 FPU Implementation

**File**: `piecewise_fpu.cpp`

#### Algorithm

```
For each tile (32 values):
  1. Extract all x values to CPU domain (TRISC-MATH)
  2. Compute segment IDs for all lanes
  3. Build power matrix P[K×32] where P[k,j] = x[j]^k
     - Computed in FP32, then quantized to BF16 for FPU
  4. Build coefficient matrix C[S×K] from LUT
     - Loaded as FP32, quantized to BF16 for FPU
  5. Execute FPU matmul: Y = C × P (BF16 × BF16 → FP32)
     - Hardware computes all polynomials for all x values in parallel
     - FP32 accumulation (DST_ACCUM_MODE=1) prevents rounding errors
  6. Select results: output[j] = Y[seg_id[j], j]
  7. Write back to tile
```

#### Pros

- ✅ **Fast for high degrees** - Single matmul replaces 256 SFPU ops (D=8)
- ✅ **Parallel** - FPU computes all polynomials simultaneously
- ✅ **FP32 accumulation** - No rounding between additions
- ✅ **Predictable performance** - Matmul throughput is constant regardless of D

#### Cons

- ❌ **BF16 quantization** - Coefficients and powers limited to 7-bit mantissa
- ❌ **Memory overhead** - Requires 3 scratch circular buffers
- ❌ **CPU overhead** - Matrix building happens on TRISC-MATH (not SFPU/FPU)
- ❌ **Wasteful for low degrees** - D=2 doesn't need full matmul power

#### Performance Characteristics

```
Matrix build (CPU):  ~256 scalar multiplies (D=8, 32 lanes)
FPU matmul:         1 hardware operation (constant time)
Result selection:   ~32 array lookups

Total: Dominated by FPU matmul throughput
```

**Expected speedup**: 2-4× faster than SFPU for D≥4

---

### 3. TF32 FPU Implementation

**File**: `piecewise_fpu_tf32.cpp`

#### Algorithm

**Same as BF16 FPU**, but with higher precision storage:

```
For each tile (32 values):
  1. Extract all x values to CPU domain (TRISC-MATH)
  2. Compute segment IDs for all lanes
  3. Build power matrix P[K×32] in FP32
     - Convert to TF32 (10-bit mantissa) before writing to CB
  4. Build coefficient matrix C[S×K] in FP32
     - Convert to TF32 (10-bit mantissa) before writing to CB
  5. Execute FPU matmul: Y = C × P (TF32 × TF32 → FP32)
     - FP32 accumulation (DST_ACCUM_MODE=1)
     - MATH_FIDELITY=4 for maximum accuracy
  6. Select results: output[j] = Y[seg_id[j], j]
  7. Write back to tile as FP32
```

#### Pros

- ✅ **Maximum HW precision** - 10-bit mantissa (43% more than BF16)
- ✅ **FP32 accumulation** - No rounding between additions
- ✅ **Fast for high degrees** - Single matmul replaces 256 SFPU ops
- ✅ **Captures x² precision** - Even BF16 input x produces x² with more info than BF16!
- ✅ **Best accuracy** - ~1e-6 relative error (1000× better than pure BF16)

#### Cons

- ❌ **Memory overhead** - Scratch CBs store 32-bit TF32 values (vs 16-bit BF16)
- ❌ **Possible performance penalty** - TF32 ops may be slower than BF16 on some hardware
- ❌ **CPU overhead** - Matrix building + TF32 conversion on TRISC-MATH

#### Performance Characteristics

```
Matrix build (CPU):  ~256 scalar multiplies (D=8, 32 lanes)
FP32 → TF32:        ~2048 conversions (2 × 32×32 matrices)
FPU matmul:         1 hardware operation (may be slightly slower than BF16)
Result selection:   ~32 array lookups

Total: Dominated by FPU matmul throughput (+ possible TF32 overhead)
```

**Expected performance**: Similar to BF16 FPU (within 10-20%)

---

## Precision Analysis

### Error Sources

| Error Source | SFPU Horner | BF16 FPU | TF32 FPU |
|--------------|-------------|----------|----------|
| **Coefficient quantization** | BF16 (7-bit) | BF16 (7-bit) | **TF32 (10-bit)** ← 43% better |
| **Power quantization** | BF16 (7-bit) | BF16 (7-bit) | **TF32 (10-bit)** ← 43% better |
| **Accumulation rounding** | BF16 (many rounds) | **FP32 (no rounds)** | **FP32 (no rounds)** |
| **Output precision** | BF16 or FP32 | FP32 | FP32 |

### Expected Error Bounds

For degree-8 polynomial, 32 segments, typical activation function:

| Implementation | Max Relative Error | Accuracy vs SFPU | Notes |
|----------------|-------------------|------------------|-------|
| **SFPU Horner (BF16)** | ~1e-4 (0.01%) | Baseline | Multiple BF16 roundings |
| **SFPU Horner (FP32)** | ~1e-5 (0.001%) | 10× better | FP32 mode available |
| **BF16 FPU + FP32 accum** | ~1e-5 (0.001%) | 10× better | No accum rounding |
| **TF32 FPU + FP32 accum** | ~1e-6 (0.0001%) | **100× better** | Best HW precision |
| **Software FP32** | ~1e-7 | 1000× better | Too slow for production |

### Key Insight: Why TF32 Matters Even with BF16 Input

**Question**: If input x is BF16 (7-bit mantissa), why does TF32 (10-bit mantissa) help?

**Answer**: x² contains MORE information than x!

```
Input x (BF16):        1.234375 (exact, 7-bit mantissa)
                        ↓
Compute x² (FP32):     1.523681640625 (exact, full 23-bit mantissa)
                        ↓
Store as BF16:         1.523438 (rounded to 7 bits) ← LOSES PRECISION
Store as TF32:         1.523682 (rounded to 10 bits) ← PRESERVES MORE!
```

**The product of two BF16 values has more precision than BF16 can represent.**

TF32 captures this additional precision, leading to ~10× better accuracy than BF16.

---

## Memory Usage

### Circular Buffer Requirements

| Implementation | Input CB | Output CB | Scratch CBs | Total L1 |
|----------------|----------|-----------|-------------|----------|
| **SFPU Horner** | 2 tiles (BF16) | 2 tiles (BF16) | None | ~8 KB |
| **BF16 FPU** | 2 tiles (BF16) | 2 tiles (FP32) | 3 × 1 tile (BF16) | ~12 KB |
| **TF32 FPU** | 2 tiles (BF16) | 2 tiles (FP32) | 3 × 1 tile (TF32) | ~16 KB |

**Notes**:
- BF16 tile size: 2 KB (32×32 × 2 bytes)
- FP32 tile size: 4 KB (32×32 × 4 bytes)
- TF32 tile size: 4 KB (32×32 × 4 bytes, 19 bits padded to 32)

**Scratch CBs** (FPU implementations only):
- CB 26: Coefficient matrix C[S×K]
- CB 27: Power matrix P[K×32]
- CB 28: Result matrix Y[S×32]

All implementations fit comfortably in L1 SRAM (1.5 MB available).

---

## Performance Benchmarks

### Theoretical Analysis

**SFPU Horner**: O(S × D) sequential operations per tile
- Degree 2, 16 segments: ~512 operations
- Degree 4, 16 segments: ~512 operations
- Degree 8, 32 segments: ~1024 operations

**FPU Matmul**: O(1) parallel matmul (S×K by K×32)
- Constant time regardless of S or D
- Dominated by FPU throughput

### Expected Performance (Relative to SFPU D=2)

| Configuration | SFPU Horner | BF16 FPU | TF32 FPU |
|---------------|-------------|----------|----------|
| **D=2, S=16** | 1.0× (baseline) | 1.2× (slightly faster) | 1.1× (similar) |
| **D=4, S=16** | 1.0× | **2.0×** (2× faster) | **1.8×** (faster) |
| **D=8, S=32** | 1.0× | **4.0×** (4× faster) | **3.5×** (faster) |

**Conclusion**: FPU implementations shine for high-degree polynomials (D≥4).

### Hardware Benchmarking Required

The above estimates are **theoretical**. Real performance depends on:

1. **FPU throughput** for TF32 vs BF16 operations
2. **TRISC-MATH overhead** for matrix building
3. **Memory bandwidth** for CB read/write
4. **Pipeline efficiency** in multi-tile workloads

**Recommendation**: Benchmark on actual hardware (Wormhole/Blackhole) to measure real speedup.

---

## Use Case Recommendations

### Choose SFPU Horner When:

✅ **Low polynomial degree** (D ≤ 3)
- SFPU is simpler and uses less memory
- Performance difference is small

✅ **Memory constrained**
- No scratch CBs needed
- Minimal L1 usage

✅ **Custom SFPU operations needed**
- Range reduction (exp, trig)
- Special functions (reciprocal, sqrt)

✅ **Debugging/prototyping**
- Simpler code, easier to understand
- Well-tested baseline

### Choose BF16 FPU When:

✅ **High polynomial degree** (D ≥ 4)
- FPU matmul is much faster than sequential SFPU

✅ **Many segments** (S ≥ 16)
- Parallel evaluation of all polynomials

✅ **Good accuracy needed** (~1e-5 relative error)
- FP32 accumulation prevents error buildup

✅ **Performance critical**
- Want maximum speedup for high-degree polynomials

### Choose TF32 FPU When:

✅ **Maximum precision required** (~1e-6 relative error)
- Best hardware-accelerated accuracy

✅ **High polynomial degree** (D ≥ 4)
- Same speedup as BF16 FPU

✅ **Coefficient precision matters**
- Polynomial approximations are sensitive to coefficient errors

✅ **Willing to trade memory for accuracy**
- Accept 2× CB storage vs BF16

---

## Migration Guide

### From SFPU to BF16 FPU

**Build change**:
```bash
# Old: SFPU Horner
cmake -DKERNEL_VARIANT=piecewise_generic -DPOLY_DEGREE=4 -DNUM_SEGMENTS=16

# New: BF16 FPU
cmake -DKERNEL_VARIANT=piecewise_fpu -DPOLY_DEGREE=4 -DNUM_SEGMENTS=16
```

**Runtime change**:
```bash
# Use --precision fp32 to enable FP32 output
./generic_lut_activation coeffs.csv --activation gelu --precision fp32 ...
```

**Expected result**: 2-4× faster, ~10× better accuracy

### From BF16 FPU to TF32 FPU

**Build change**:
```bash
# Old: BF16 FPU
cmake -DKERNEL_VARIANT=piecewise_fpu -DPOLY_DEGREE=4 -DNUM_SEGMENTS=16

# New: TF32 FPU
cmake -DKERNEL_VARIANT=piecewise_fpu_tf32 -DPOLY_DEGREE=4 -DNUM_SEGMENTS=16
```

**Runtime**: Same as BF16 FPU (--precision fp32)

**Expected result**: Similar speed, ~10× better accuracy

---

## Summary

### Best Practices

1. **Start with SFPU Horner** for D ≤ 3
2. **Upgrade to BF16 FPU** for D ≥ 4 (for speed)
3. **Use TF32 FPU** when accuracy is critical (for precision)
4. **Always benchmark** on target hardware to confirm performance

### Performance Hierarchy

```
Speed (D ≥ 4):  BF16 FPU ≈ TF32 FPU >> SFPU Horner
Precision:       TF32 FPU >> BF16 FPU > SFPU Horner (FP32) > SFPU Horner (BF16)
Memory:          SFPU Horner << BF16 FPU < TF32 FPU
Simplicity:      SFPU Horner >> BF16 FPU ≈ TF32 FPU
```

### The Sweet Spot

**For most use cases**: **BF16 FPU with FP32 accumulation**
- Good balance of speed, precision, and simplicity
- 2-4× faster than SFPU for D ≥ 4
- ~1e-5 relative error is sufficient for most neural networks

**For maximum precision**: **TF32 FPU with FP32 accumulation**
- ~1e-6 relative error (best hardware-accelerated precision)
- Similar performance to BF16 FPU
- Worth the extra memory overhead for accuracy-critical applications

---

## Conclusion

All three implementations are valuable and have their place:

- **SFPU Horner**: Simple, proven, best for low degrees
- **BF16 FPU**: Fast, accurate enough, great for high degrees
- **TF32 FPU**: Maximum precision, ideal for critical applications

Choose based on your specific requirements for **speed**, **precision**, and **memory usage**.

The FPU implementations represent a significant advancement in **performance** for high-degree polynomials, while the TF32 variant provides **near-FP32 accuracy** without sacrificing hardware acceleration.

**Recommendation**: Start with BF16 FPU for production, upgrade to TF32 if precision requirements demand it.
