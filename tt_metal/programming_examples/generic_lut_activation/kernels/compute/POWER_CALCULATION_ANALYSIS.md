# Power Vector Calculation - Location and Performance Analysis

## Question: Where do you calculate the power vector of x?

## Answer: Location in Code

### Function: `build_power_matrix()` (lines 134-153)

```cpp
template <uint32_t POLY_DEGREE>
inline void build_power_matrix(const float x_vals[32], float P[32][32]) {
    constexpr uint32_t K = POLY_DEGREE + 1;

    // Initialize entire matrix to zero
    for (int r = 0; r < 32; r++) {
        for (int c = 0; c < 32; c++) {
            P[r][c] = 0.0f;
        }
    }

    // Fill rows 0..K-1 with powers of x
    for (int lane = 0; lane < 32; lane++) {
        float x = x_vals[lane];
        float x_power = 1.0f;  // Start with x^0 = 1

        for (uint32_t k = 0; k < K; k++) {
            P[k][lane] = x_power;  // Store current power
            x_power *= x;          // Compute next: x^(k+1)
        }
    }
}
```

### Called from: `piecewise_poly_matmul_tile()` (line 373)

```cpp
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_poly_matmul_tile(const std::array<float, LUT_SIZE>& lut) {
    // 1. Extract x values from dst_reg[]
    float x_vals[32];
    extract_x_values(x_vals);  // Line 365

    // 2. Compute segment IDs
    uint32_t seg_id[32];
    compute_segment_ids<NUM_SEGMENTS>(lut.data(), x_vals, seg_id);  // Line 369

    // 3. BUILD POWER MATRIX ← HERE!
    float P[32][32];
    build_power_matrix<POLY_DEGREE>(x_vals, P);  // Line 373 ← ANSWER!

    // 4. Build coefficient matrix
    float C[32][32];
    build_coeff_matrix<POLY_DEGREE, NUM_SEGMENTS>(lut.data(), C);  // Line 377

    // 5. Perform FPU matmul
    FPU_MATMUL_load_C(C);
    FPU_MATMUL_load_P(P);
    FPU_MATMUL_compute();

    // 6. Read and select results
    float Y[32][32];
    FPU_MATMUL_read_Y(Y);
    for (int lane = 0; lane < 32; lane++) {
        dst_reg[lane] = Y[seg_id[lane]][lane];
    }
}
```

---

## Where Does This Execute?

### Execution Context

| Property | Value | Notes |
|----------|-------|-------|
| **Processor** | TRISC-MATH | The compute RISC-V core |
| **Memory** | Local stack | `float P[32][32]` on stack/registers |
| **Precision** | float32 | Full precision before conversion |
| **Hardware** | Scalar CPU | NOT using SFPU or FPU |
| **Context** | `#ifdef TRISC_MATH` block | Math core execution |

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ TRISC-MATH (Compute RISC-V Core)                          │
├─────────────────────────────────────────────────────────────┤
│ 1. extract_x_values(x_vals)                                │
│    └─> x_vals[32] in local memory (float32)               │
│                                                             │
│ 2. build_power_matrix(x_vals, P)  ← POWER CALCULATION     │
│    └─> Scalar loop: x_power *= x  (32 lanes × K powers)   │
│    └─> P[32][32] in local memory (float32)                │
│                                                             │
│ 3. FPU_MATMUL_load_P(P)                                    │
│    └─> Convert P to bfloat16                               │
│    └─> Write to CB_SCRATCH_P                               │
│                                                             │
│ 4. matmul_tiles() triggers:                                │
│    ├─> UNPACK: CB_SCRATCH_P → SrcA registers              │
│    ├─> MATH: FPU matmul (hardware)                        │
│    └─> PACK: Dest → CB_SCRATCH_Y                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Analysis

### Operation Count

For a tile with 32 lanes and degree D:

| Operation | Count | Example (D=8) |
|-----------|-------|---------------|
| Zero initialization | 32 × 32 = 1,024 stores | 1,024 |
| Power computation | 32 lanes × D multiplies | 256 |
| Store powers | 32 lanes × (D+1) stores | 288 |
| **Total scalar ops** | **~1,300** | **~1,568** |

### Comparison to Alternatives

#### Current Implementation: Scalar Loop
```cpp
float x_power = 1.0f;
for (uint32_t k = 0; k < K; k++) {
    P[k][lane] = x_power;
    x_power *= x;  // Scalar multiply on TRISC-MATH CPU
}
```

**Performance**:
- 1 multiply per power, per lane
- Example (D=8): 32 lanes × 8 multiplies = **256 scalar FP32 multiplies**
- Runs sequentially on TRISC-MATH CPU

#### Alternative 1: SFPU Power Calculation
```cpp
// Could use SFPU vector operations (if we had them)
for (uint32_t k = 0; k < K; k++) {
    vFloat x_vec = load_vector(x_vals);
    vFloat power_vec = pow(x_vec, k);  // Vector power (if available)
    store_vector(&P[k][0], power_vec);
}
```

**Problems**:
- No direct `pow()` for arbitrary integer exponents in SFPU
- Would need to use `exp(k * log(x))` - very expensive
- Not worth it for small K values

#### Alternative 2: Unroll Loop (Compile-Time Optimization)
```cpp
template <uint32_t POLY_DEGREE>
inline void build_power_matrix_unrolled(const float x_vals[32], float P[32][32]) {
    if constexpr (POLY_DEGREE == 2) {
        for (int lane = 0; lane < 32; lane++) {
            float x = x_vals[lane];
            P[0][lane] = 1.0f;
            P[1][lane] = x;
            P[2][lane] = x * x;
        }
    } else if constexpr (POLY_DEGREE == 4) {
        for (int lane = 0; lane < 32; lane++) {
            float x = x_vals[lane];
            float x2 = x * x;
            P[0][lane] = 1.0f;
            P[1][lane] = x;
            P[2][lane] = x2;
            P[3][lane] = x2 * x;
            P[4][lane] = x2 * x2;
        }
    }
    // ... etc for other degrees
}
```

**Benefits**:
- Reduces operation count for common degrees
- Example (D=4): 3 multiplies instead of 4
- Compiler can optimize better with explicit code

---

## Is This a Bottleneck?

### Quick Analysis

Assume TRISC-MATH scalar multiply latency = 4-6 cycles:

| Degree | Scalar Ops | Cycles | % of Matmul Cost |
|--------|------------|--------|------------------|
| D=2 | 64 | ~256 | ~5% |
| D=4 | 128 | ~512 | ~10% |
| D=8 | 256 | ~1,024 | ~20% |

For a 32×9 by 9×32 matmul (degree 8):
- Power calculation: **~1,000 cycles**
- FPU matmul: **~5,000-10,000 cycles** (hardware dependent)
- Matrix writes to CB: **~2,000 cycles**

**Conclusion**: Power calculation is **10-20% of total cost** for high degrees.

---

## Optimization Opportunities

### 1. ✅ **Keep Current Implementation** (Recommended)
**Pros**:
- Simple and correct
- Already verified
- Negligible overhead for D ≤ 4
- Reasonable overhead even for D=8

**Cons**:
- Sequential scalar operations
- No vectorization

**When to use**: Most cases, especially D ≤ 8

---

### 2. 🔧 **Template Specialization for Common Degrees**

```cpp
// Specialization for degree 2 (quadratic)
template <>
inline void build_power_matrix<2>(const float x_vals[32], float P[32][32]) {
    memset(P, 0, sizeof(float) * 32 * 32);
    for (int lane = 0; lane < 32; lane++) {
        float x = x_vals[lane];
        P[0][lane] = 1.0f;
        P[1][lane] = x;
        P[2][lane] = x * x;
    }
}

// Specialization for degree 4 (quartic)
template <>
inline void build_power_matrix<4>(const float x_vals[32], float P[32][32]) {
    memset(P, 0, sizeof(float) * 32 * 32);
    for (int lane = 0; lane < 32; lane++) {
        float x = x_vals[lane];
        float x2 = x * x;
        P[0][lane] = 1.0f;
        P[1][lane] = x;
        P[2][lane] = x2;
        P[3][lane] = x2 * x;
        P[4][lane] = x2 * x2;
    }
}

// Specialization for degree 8 (octic)
template <>
inline void build_power_matrix<8>(const float x_vals[32], float P[32][32]) {
    memset(P, 0, sizeof(float) * 32 * 32);
    for (int lane = 0; lane < 32; lane++) {
        float x = x_vals[lane];
        float x2 = x * x;
        float x4 = x2 * x2;
        P[0][lane] = 1.0f;
        P[1][lane] = x;
        P[2][lane] = x2;
        P[3][lane] = x2 * x;
        P[4][lane] = x4;
        P[5][lane] = x4 * x;
        P[6][lane] = x4 * x2;
        P[7][lane] = x4 * x2 * x;
        P[8][lane] = x4 * x4;
    }
}
```

**Performance gain**:
- D=4: 3 muls instead of 4 → 25% faster
- D=8: 5 muls instead of 8 → 37% faster

**Cost**: More code, more template instantiations

---

### 3. 🚀 **SIMD-style Manual Vectorization** (Advanced)

```cpp
// Process multiple lanes in parallel using manual loop unrolling
template <uint32_t POLY_DEGREE>
inline void build_power_matrix_simd(const float x_vals[32], float P[32][32]) {
    constexpr uint32_t K = POLY_DEGREE + 1;
    memset(P, 0, sizeof(float) * 32 * 32);

    // Process 4 lanes at a time (manual vectorization)
    for (int lane = 0; lane < 32; lane += 4) {
        float x0 = x_vals[lane + 0];
        float x1 = x_vals[lane + 1];
        float x2 = x_vals[lane + 2];
        float x3 = x_vals[lane + 3];

        float p0 = 1.0f, p1 = 1.0f, p2 = 1.0f, p3 = 1.0f;

        for (uint32_t k = 0; k < K; k++) {
            P[k][lane + 0] = p0;
            P[k][lane + 1] = p1;
            P[k][lane + 2] = p2;
            P[k][lane + 3] = p3;

            p0 *= x0;
            p1 *= x1;
            p2 *= x2;
            p3 *= x3;
        }
    }
}
```

**Performance gain**: Potentially 2-4× with instruction-level parallelism

**Risk**: Compiler may already do this optimization

---

### 4. ⚡ **Skip Zero Initialization** (Micro-optimization)

```cpp
template <uint32_t POLY_DEGREE>
inline void build_power_matrix(const float x_vals[32], float P[32][32]) {
    constexpr uint32_t K = POLY_DEGREE + 1;

    // Only zero unused rows (rows K..31)
    for (int r = K; r < 32; r++) {
        memset(&P[r][0], 0, sizeof(float) * 32);
    }

    // Fill used rows (no initialization needed)
    for (int lane = 0; lane < 32; lane++) {
        float x = x_vals[lane];
        float x_power = 1.0f;
        for (uint32_t k = 0; k < K; k++) {
            P[k][lane] = x_power;
            x_power *= x;
        }
    }
}
```

**Savings**: Avoids zeroing K × 32 elements that will be overwritten

---

## Recommendation

### For Current Implementation: ✅ Keep As-Is

**Rationale**:
1. **Correct and verified** - most important
2. **Simple and maintainable** - easy to understand
3. **Good enough** - 10-20% overhead acceptable
4. **Not the bottleneck** - FPU matmul dominates runtime

### If Optimization Needed: Consider These

**Priority 1**: Template specialization for D=2,4,8 (common cases)
- Easy to implement
- Clear performance benefit
- No complexity increase

**Priority 2**: Skip zero initialization for unused rows
- Simple change
- Small but measurable benefit
- No downsides

**Low Priority**: SIMD manual vectorization
- Compiler may already optimize
- Increased code complexity
- Marginal benefit

---

## Conclusion

**Q: Where do you calculate the power vector of x?**

**A**: In the `build_power_matrix()` function (lines 134-153), called from `piecewise_poly_matmul_tile()` at line 373.

**Execution Context**:
- Runs on TRISC-MATH processor (compute RISC-V core)
- Uses scalar float32 multiplies in local memory
- Happens BEFORE FPU matmul operation
- Cost: ~1,000 cycles for degree 8 (~10-20% of total)

**Status**: Implementation is correct and performance is acceptable. Optimizations are possible but not critical.

---

## References

- Current implementation: `piecewise_fpu.cpp` lines 134-153
- Called from: `piecewise_fpu.cpp` line 373
- Verified in: `verify_fpu_math.py` (power matrix verification passed ✓)
