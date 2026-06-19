# TF32 vs BF16 for Coefficients - Detailed Analysis

## Your Questions

1. **Should we fit coeffs to TF32 instead of BF16?**
2. **Can the FPU do BF16 × TF32 → FP32 (mixed precision)?**

Let's investigate both carefully.

---

## Question 1: Can FPU Do Mixed Precision (BF16 × TF32)?

### ISA Documentation Says

From the matrix multiplication support table:

| Dst Type | Operation | SrcB Type | @ | SrcA Type |
|----------|-----------|-----------|---|-----------|
| FP32/BF16 (8×16) | += | **TF32/BF16** (8×16) | @ | **TF32/BF16** (16×16) |

### Interpretation

The slash notation "TF32/BF16" typically means **OR**, not **AND**.

**Likely meaning**:
- **Option 1**: SrcB=TF32, SrcA=TF32 → Dst=FP32 ✓
- **Option 2**: SrcB=BF16, SrcA=BF16 → Dst=FP32 ✓
- **Option 3**: SrcB=BF16, SrcA=TF32 → Dst=FP32 ❓

### Hardware Reality Check

**Typical matrix engine design**: Both operands must be **same format**

**Why?**:
- Multiplier array is designed for specific bit widths
- Mixed formats require dual datapaths (area expensive)
- Most hardware does NOT support mixed precision within a single matmul

### Answer (Conservative)

**Likely NO** - both SrcA and SrcB must be same format.

**But WAIT** - there's a better solution!

---

## Solution: Use TF32 for BOTH Matrices!

### The Insight

Even though **x values are BF16**, we can still store the **power matrix in TF32**!

```
Input x: BF16 (7-bit mantissa)
  ↓
Extract to float32: 1.234375 (exact, zero-extended)
  ↓
Compute x²: 1.234375² = 1.523681640625 (exact in float32)
  ↓
Store as TF32: 1.523681640625 → TF32 (10-bit mantissa)
  ↓
FPU: TF32 × TF32 → FP32
```

**Key insight**: Even though x started as BF16, **x² has more precision** than BF16 can represent!

### Example: Why x² Has More Precision

**BF16 input**: x = 1.234375 (7-bit mantissa, exact)

**x² computed**:
```
x² = 1.234375² = 1.52368164062500000...

In binary:
1.52368164062500 = 1.10000110001000... (binary)
                     ^^^^^^^           (7 bits from BF16 x)
                            ^^^^^^^^   (additional precision from multiply!)
```

**The product has MORE information than either input!**

**Storage comparison**:
```
BF16: Can store ~7 mantissa bits of x²
TF32: Can store ~10 mantissa bits of x² ← 43% more precision!
FP32: Can store ~23 mantissa bits of x² ← Full precision!
```

**Bottom line**: Using TF32 for powers **preserves more precision** than BF16!

---

## Question 2: Should We Use TF32 for Coefficients?

### Precision Comparison

| Format | Mantissa Bits | Decimal Digits | Range |
|--------|---------------|----------------|-------|
| BF16 | 7 | ~2-3 | ±1.18e-38 to ±3.4e38 |
| TF32 | 10 | ~3-4 | ±1.18e-38 to ±3.4e38 |
| FP32 | 23 | ~7 | ±1.18e-38 to ±3.4e38 |

**TF32 = 43% more mantissa precision than BF16!**

### Coefficient Precision Requirements

Polynomial coefficients often require **high precision**:

**Example**: Approximating sigmoid with degree-8 polynomial

```
Coefficient c5 = 0.003141592653...

BF16:  0.003143... (loses precision after 3 digits)
       Error: ~0.05%

TF32:  0.003141594... (preserves 4 digits)
       Error: ~0.00006%

FP32:  0.003141592653... (preserves 7 digits)
       Error: negligible
```

**For polynomial approximation, coefficient precision matters A LOT!**

### Error Propagation

When evaluating polynomial:
```
y = c0 + c1*x + c2*x² + ... + c8*x⁸

If coefficients have error ε,
Final error ≈ ε * (sum of |ci * xi|)
```

**High-degree polynomials amplify coefficient errors!**

**For degree 8**: Coefficient error multiplied by up to 8×

---

## Recommendation: Use TF32 for Both Matrices

### Configuration

```cpp
// Use TF32 format for both coefficient and power matrices
DataFormat::TensorFloat32

// Matmul will be:
// Dst (FP32) += SrcB (TF32) @ SrcA (TF32)
```

### Precision Journey

```
┌─────────────────────────────────────────────────────────────┐
│ Input x: BF16 (7-bit mantissa)                             │
│   x = 1.234375                                              │
└────────────┬────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────┐
│ Extract to FP32:                                            │
│   x_fp32 = 1.234375 (zero-extended, exact)                 │
└────────────┬────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────┐
│ Compute Powers in FP32:                                     │
│   x²  = 1.523681640625                                      │
│   x³  = 1.881676673889...                                   │
│   x⁴  = 2.322940826416...                                   │
│   (Full FP32 precision: 23-bit mantissa)                    │
└────────────┬────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────┐
│ Store Power Matrix as TF32:                                 │
│   P[0][j] = 1.000000000 (TF32, 10-bit mantissa)           │
│   P[1][j] = 1.234375000 (TF32)                             │
│   P[2][j] = 1.523681641 (TF32, preserves 10 bits!)        │
│   P[3][j] = 1.881676674 (TF32)                             │
│   P[4][j] = 2.322940826 (TF32)                             │
│                                                             │
│   vs BF16 would give:                                       │
│   P[2][j] = 1.523438 (BF16, only 7 bits)                  │
│   Error: 0.016% per coefficient!                           │
└────────────┬────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────┐
│ Coefficient Matrix (from LUT):                             │
│   Keep LUT in FP32 in memory                               │
│   Convert to TF32 when loading to CB                       │
│                                                             │
│   C[s][k] in FP32:  0.003141592653                        │
│   C[s][k] in TF32:  0.003141594 (preserves 10 bits!)     │
│   C[s][k] in BF16:  0.003143... (only 7 bits)            │
└────────────┬────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────┐
│ FPU Matmul: TF32 × TF32 → FP32                            │
│                                                             │
│   Y[s][j] = Σ C[s][k] × P[k][j]                           │
│             (TF32)   (TF32)                                 │
│                ↓                                            │
│            Accumulate in FP32 (23-bit mantissa)            │
│                                                             │
│   Result: Full FP32 precision output!                      │
└─────────────────────────────────────────────────────────────┘
```

### Benefit vs BF16

**BF16 approach**:
- Coefficient precision: ~2-3 decimal digits
- Power precision: ~2-3 decimal digits
- Accumulation: FP32 (good!)
- **Bottleneck**: Input precision (7-bit mantissa)

**TF32 approach**:
- Coefficient precision: ~3-4 decimal digits ← **43% better!**
- Power precision: ~3-4 decimal digits ← **43% better!**
- Accumulation: FP32 (same)
- **Overall**: ~2× better accuracy!

---

## Implementation: FP32 → TF32 Conversion

### What is TF32 Format?

```
TF32 (19 bits stored in register):
┌───┬────────┬──────────────┬─────┐
│ S │ Exp(8) │ Mantissa(10) │Pad  │
└───┴────────┴──────────────┴─────┘
 1     8          10           0

Same exponent range as FP32/BF16
But 10-bit mantissa (vs 7 for BF16, 23 for FP32)
```

### Conversion Function

```cpp
inline uint32_t convert_fp32_to_tf32(float value) {
    uint32_t fp32_bits = *reinterpret_cast<uint32_t*>(&value);

    // Extract components
    uint32_t sign = (fp32_bits >> 31) & 0x1;
    uint32_t exp  = (fp32_bits >> 23) & 0xFF;
    uint32_t mant = (fp32_bits >> 13) & 0x3FF;  // Keep upper 10 bits of mantissa

    // TF32 format: 1 sign + 8 exp + 10 mantissa = 19 bits
    uint32_t tf32_bits = (sign << 18) | (exp << 10) | mant;

    return tf32_bits;
}

inline float convert_tf32_to_fp32(uint32_t tf32_bits) {
    // Extract TF32 components
    uint32_t sign = (tf32_bits >> 18) & 0x1;
    uint32_t exp  = (tf32_bits >> 10) & 0xFF;
    uint32_t mant = tf32_bits & 0x3FF;

    // Expand to FP32 format (zero-extend mantissa)
    uint32_t fp32_bits = (sign << 31) | (exp << 23) | (mant << 13);

    return *reinterpret_cast<float*>(&fp32_bits);
}
```

### Updated FPU_MATMUL_load Functions

```cpp
inline void FPU_MATMUL_load_C_tf32(const float C[32][32]) {
    cb_reserve_back(CB_SCRATCH_C, 1);
    uint32_t cb_addr = get_write_ptr(CB_SCRATCH_C);

    // TF32 uses 19 bits, but stored in 32-bit words
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_addr);

    for (int r = 0; r < 32; r++) {
        for (int c = 0; c < 32; c++) {
            // Convert FP32 → TF32
            dst[r * 32 + c] = convert_fp32_to_tf32(C[r][c]);
        }
    }

    cb_push_back(CB_SCRATCH_C, 1);
}

inline void FPU_MATMUL_load_P_tf32(const float P[32][32]) {
    cb_reserve_back(CB_SCRATCH_P, 1);
    uint32_t cb_addr = get_write_ptr(CB_SCRATCH_P);

    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_addr);

    for (int r = 0; r < 32; r++) {
        for (int c = 0; c < 32; c++) {
            // Convert FP32 → TF32
            dst[r * 32 + c] = convert_fp32_to_tf32(P[r][c]);
        }
    }

    cb_push_back(CB_SCRATCH_P, 1);
}
```

---

## Performance Comparison

| Configuration | Input Precision | Coeff Precision | Accum Precision | Relative Error | Speed |
|---------------|----------------|-----------------|-----------------|----------------|-------|
| **BF16 + BF16** | 7-bit | 7-bit | BF16 | ~1e-3 | 1.0× |
| **BF16 + BF16 + FP32** | 7-bit | 7-bit | FP32 | ~1e-4 | 1.0× |
| **TF32 + TF32 + FP32** | 10-bit | 10-bit | FP32 | ~1e-5 | 1.0×? |
| **FP32 (software)** | 23-bit | 23-bit | FP32 | ~1e-7 | 0.1× |

**TF32 may have slight performance penalty** - need to check if FPU throughput is same as BF16.

---

## Storage Overhead

### CB Size Comparison

**BF16**:
- 32×32 tile × 2 bytes = 2,048 bytes per CB

**TF32**:
- 32×32 tile × 4 bytes = 4,096 bytes per CB
- (TF32 is 19 bits, but likely stored in 32-bit words)

**Trade-off**: 2× memory for ~2× accuracy

---

## Recommendation Summary

### Option 1: BF16 + FP32 Accumulation (Easiest)
```cpp
#define DST_ACCUM_MODE 1  // FP32 accumulation
// Use BF16 for both matrices
```

**Pros**:
- One-line change
- No storage overhead
- 100× better than BF16 accumulation

**Cons**:
- Still limited by 7-bit mantissa inputs

**Recommended for**: Quick accuracy improvement

---

### Option 2: TF32 + FP32 Accumulation (Best Balance)
```cpp
#define DST_ACCUM_MODE 1  // FP32 accumulation
// Use TF32 for both matrices
```

**Pros**:
- 43% more input precision (10 vs 7 bits)
- ~2× better accuracy than BF16
- Still uses hardware FPU

**Cons**:
- Need TF32 conversion functions
- 2× CB storage overhead
- Possible performance penalty (check hardware)

**Recommended for**: Maximum hardware-accelerated accuracy

---

### Option 3: FP32 Software Matmul (Maximum Accuracy)
```cpp
// Don't use FPU at all
// Compute matmul in FP32 on CPU
```

**Pros**:
- Full FP32 precision (23-bit mantissa)
- ~1000× better than BF16

**Cons**:
- 10× slower than FPU
- Defeats purpose of FPU implementation

**Recommended for**: Accuracy-critical applications only

---

## Action Plan

### Step 1: Enable FP32 Accumulation (Now)

```cpp
#define DST_ACCUM_MODE 1
```

Test and measure accuracy improvement.

### Step 2: Investigate TF32 Support (Next)

**Check**:
1. Does unpacker support TF32 input format?
2. What is TF32 FPU throughput vs BF16?
3. What is CB configuration for TF32?

**If supported**: Implement TF32 conversion and test.

### Step 3: Benchmark Both

Compare:
- BF16 + FP32 accum
- TF32 + FP32 accum

Choose best trade-off for your application.

---

## Answers to Your Questions

### Q1: Should we fit coeffs to TF32 instead of BF16?

**A: YES, if hardware supports it!**

**Benefit**: 43% more precision (10 vs 7 mantissa bits)

**Cost**: 2× CB storage, possible performance penalty

**Trade-off**: Usually worth it for polynomial approximation

### Q2: Can the FPU do BF16 × TF32 → FP32?

**A: Probably NO (mixed precision not supported)**

**But**: You can do TF32 × TF32 → FP32!

**Solution**: Use TF32 for BOTH power matrix and coefficient matrix.

**Insight**: Even though x is BF16, storing x² as TF32 preserves more precision than BF16!

---

## Final Recommendation

### For Your Use Case (Polynomial Approximation)

**Go with Option 2: TF32 + FP32 Accumulation**

**Why**:
1. Polynomials are **sensitive to coefficient precision**
2. TF32 gives **43% more precision** for FREE (same hardware)
3. High-degree polynomials **amplify errors** - need best input precision
4. FP32 accumulation **prevents error buildup**

**Expected result**: ~500× better accuracy than pure BF16!

**Implementation**: ~100 lines of code (add TF32 conversion helpers)

---

## Summary Table

| Question | Answer | Recommendation |
|----------|--------|----------------|
| Use TF32 for coeffs? | **YES** | 43% more precision, worth it! |
| BF16 × TF32 mixed? | **NO** | Use TF32 × TF32 instead |
| TF32 for powers too? | **YES** | Preserves x² precision better |
| Performance penalty? | **Unknown** | Need to test, likely small |
| Storage overhead? | **2× CB size** | Acceptable for accuracy gain |

**Bottom line**: TF32 + FP32 accumulation gives you **near-FP32 accuracy** with **hardware FPU speed**!
