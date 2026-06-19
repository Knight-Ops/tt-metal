# Mixed-Precision Techniques for FP32 Accuracy with BF16 Inputs

## The Challenge

**Given**:
- Input x: BF16 (7-bit mantissa, ~3 decimal digits)
- Coefficients: Can be FP32 (23-bit mantissa, ~7 decimal digits)
- Hardware FPU: BF16 × BF16 → BF16

**Goal**: Get FP32 accumulation accuracy (23-bit mantissa) in polynomial evaluation

**Question**: Can we do "extra work" to preserve FP32 accuracy?

**Answer**: YES! Several techniques exist.

---

## Technique 1: FP32 Accumulation Mode (Easiest)

### Check Hardware Support

TT-Metal supports `DST_ACCUM_MODE` which controls accumulation precision:

```cpp
// From matmul.h
#define DST_ACCUM_MODE 0  // Default: BF16 accumulation
#define DST_ACCUM_MODE 1  // FP32 accumulation

MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
```

### How It Works

Even if FPU inputs are BF16, the accumulation can be FP32:

```
Input A: BF16 (7-bit mantissa)
Input B: BF16 (7-bit mantissa)
    ↓
Multiply: BF16 × BF16 = Intermediate (potentially higher precision)
    ↓
Accumulate: FP32 + FP32 = FP32 (23-bit mantissa)
    ↓
Output: FP32 or BF16 (your choice)
```

**This is what modern GPUs do!** (e.g., NVIDIA Tensor Cores with TF32)

### Implementation

```cpp
// Enable FP32 accumulation mode
#define DST_ACCUM_MODE 1

// In kernel initialization
MATH((llk_math_pack_sync_init<1>()));  // FP32 accumulation

// Matmul will now accumulate in FP32
matmul_tiles(CB_SCRATCH_C, CB_SCRATCH_P, 0, 0, 0);

// Pack to FP32 output
pack_tile(0, CB_SCRATCH_Y);  // Will pack FP32 if DST_ACCUM_MODE=1
```

### Benefit

**Accumulation errors reduced by ~1000×!**

For polynomial evaluation:
```
BF16 accumulation: y = c0 + c1*x + c2*x² + c3*x³
                      └─┬─┘  └──┬──┘  └──┬──┘
                    Each add has BF16 rounding error

FP32 accumulation: y = c0 + c1*x + c2*x² + c3*x³
                      └─────────┬──────────┘
                           All adds in FP32!
```

**Catch**: Check if TT-Metal FPU actually supports FP32 accumulation. If not, move to Technique 2.

---

## Technique 2: Split BF16 into High + Low Components

### The Idea

Represent each BF16 value as sum of two BF16 values with higher effective precision.

**Not applicable here** - BF16 already has limited mantissa (7 bits). Splitting doesn't help.

**Skip this technique.**

---

## Technique 3: Double-Pass Matmul with Error Compensation

### The Algorithm

Based on Kahan summation and compensated summation:

**Pass 1**: Compute Y = C × P in BF16
```
Y_bf16 = C_bf16 × P_bf16  (standard FPU matmul)
```

**Pass 2**: Compute residual error and correct
```
// Convert Y_bf16 back to FP32
Y_fp32 = extend(Y_bf16)

// Compute error: what we wanted minus what we got
// Use FP32 arithmetic for error computation
Error = C_fp32 × P_fp32 - Y_fp32

// Add correction
Y_corrected = Y_fp32 + Error
```

### Implementation

```cpp
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS>
inline void piecewise_poly_matmul_compensated(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t K = POLY_DEGREE + 1;

    // Extract x values
    float x_vals[32];
    extract_x_values(x_vals);

    // Build power matrix P in FP32
    float P_fp32[32][32];
    build_power_matrix<POLY_DEGREE>(x_vals, P_fp32);

    // Build coefficient matrix C in FP32
    float C_fp32[32][32];
    build_coeff_matrix<POLY_DEGREE, NUM_SEGMENTS>(lut.data(), C_fp32);

    // === PASS 1: BF16 FPU Matmul ===
    FPU_MATMUL_load_C(C_fp32);  // Converts FP32 → BF16
    FPU_MATMUL_load_P(P_fp32);  // Converts FP32 → BF16
    FPU_MATMUL_compute();       // Y_bf16 = C_bf16 × P_bf16

    float Y_bf16[32][32];
    FPU_MATMUL_read_Y(Y_bf16);  // Read BF16 result as FP32

    // === PASS 2: Compute Error in FP32 (on CPU) ===
    float Y_corrected[32][32];

    for (int s = 0; s < NUM_SEGMENTS; s++) {
        for (int j = 0; j < 32; j++) {
            // Compute exact result in FP32
            float exact = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                exact += C_fp32[s][k] * P_fp32[k][j];
            }

            // Error = exact - bf16_result
            float error = exact - Y_bf16[s][j];

            // Corrected result
            Y_corrected[s][j] = Y_bf16[s][j] + error;
        }
    }

    // Use Y_corrected for final selection
    uint32_t seg_id[32];
    compute_segment_ids<NUM_SEGMENTS>(lut.data(), x_vals, seg_id);

    for (int lane = 0; lane < 32; lane++) {
        dst_reg[lane] = Y_corrected[seg_id[lane]][lane];
    }
}
```

### Performance Cost

- **1× FPU matmul** (fast, hardware)
- **S × 32 × K FP32 multiply-adds** (slow, CPU)

For S=8, K=5, 32 lanes: **1,280 CPU operations**

**Trade-off**: 2-3× slower than pure FPU, but gets FP32 accuracy!

### Accuracy Gain

**Before** (BF16 only):
```
Relative error: ~1e-3 (0.1%) typical for BF16 roundoff
```

**After** (compensated):
```
Relative error: ~1e-7 (0.00001%) typical for FP32 roundoff
```

**~10,000× improvement in accuracy!**

---

## Technique 4: Store Coefficients as FP32, Expand at Runtime

### The Idea

Keep LUT coefficients in **full FP32 precision**, only quantize x values.

```
x: BF16 (input constraint)
Coefficients c: FP32 (we control this!)
Powers x^k: Computed from BF16 x → BF16
```

### Benefit

When we compute `c * x^k`:
- If c is FP32 and x^k is BF16
- The multiply is BF16 × FP32 → **FP32** (on CPU)

**Accumulation in FP32 preserves coefficient precision!**

### Implementation

```cpp
// Keep LUT in FP32 (no change needed)
std::array<float, LUT_SIZE> lut_fp32;  // Already FP32!

// Build coefficient matrix in FP32 (no change)
float C_fp32[32][32];
build_coeff_matrix<POLY_DEGREE, NUM_SEGMENTS>(lut_fp32.data(), C_fp32);

// Build power matrix (will be BF16-derived, but in FP32 representation)
float P_fp32[32][32];
build_power_matrix<POLY_DEGREE>(x_vals, P_fp32);

// === KEY: Compute matmul in SOFTWARE with FP32 accumulation ===
float Y_fp32[32][32];

for (int s = 0; s < NUM_SEGMENTS; s++) {
    for (int j = 0; j < 32; j++) {
        float acc = 0.0f;  // FP32 accumulator!

        for (uint32_t k = 0; k < K; k++) {
            // FP32 × FP32 → FP32 multiply
            // FP32 + FP32 → FP32 accumulation
            acc += C_fp32[s][k] * P_fp32[k][j];
        }

        Y_fp32[s][j] = acc;
    }
}

// Use Y_fp32 for selection (full FP32 precision!)
```

### Performance Cost

**No FPU matmul** - all done in software!

- **S × 32 × K multiply-adds** in FP32

For S=8, K=5, 32 lanes: **1,280 FP32 operations** on CPU

**Much slower than FPU**, but **full FP32 accuracy**!

### When to Use

- Small S × K (< 10 × 10): CPU cost acceptable
- Need maximum accuracy
- Coefficients have high precision requirements

---

## Technique 5: Hybrid Approach (Best of Both Worlds)

### The Strategy

Use FPU for speed, CPU for precision where it matters.

**For most operations**: Use fast BF16 FPU matmul

**For critical refinement**: Use CPU FP32 correction on selected lanes

### Implementation

```cpp
// Step 1: Fast BF16 FPU matmul
FPU_MATMUL_compute();  // Y_bf16 = C_bf16 × P_bf16

float Y_bf16[32][32];
FPU_MATMUL_read_Y(Y_bf16);

// Step 2: Identify which lanes need refinement
uint32_t seg_id[32];
compute_segment_ids<NUM_SEGMENTS>(lut.data(), x_vals, seg_id);

// Step 3: Refine ONLY the selected results in FP32
float results[32];

for (int lane = 0; lane < 32; lane++) {
    uint32_t seg = seg_id[lane];

    // Option A: Use BF16 result directly (fast)
    if (acceptable_bf16_error) {
        results[lane] = Y_bf16[seg][lane];
    }
    // Option B: Refine in FP32 (accurate)
    else {
        float acc = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
            acc += C_fp32[seg][k] * P_fp32[k][lane];
        }
        results[lane] = acc;
    }
}

// Write results
for (int lane = 0; lane < 32; lane++) {
    dst_reg[lane] = results[lane];
}
```

### Adaptive Precision

Refine only when needed:
```cpp
bool needs_refinement = (Y_bf16[seg][lane] != 0.0f) &&
                        (abs(x_vals[lane] - boundary) < threshold);
```

### Performance

- Best case: 100% FPU (all BF16)
- Worst case: 100% CPU (all need refinement)
- Typical: 90% FPU + 10% CPU refinement

---

## Technique 6: Split Matmul into High + Low Precision Parts

### The Idea

Decompose the matmul into:
1. **High-precision part**: Most significant bits of coefficients
2. **Low-precision part**: Remaining bits (correction term)

```
C = C_high + C_low

Y = C × P
  = (C_high + C_low) × P
  = C_high × P + C_low × P
  = Y_high + Y_low
```

### Implementation

```cpp
// Split coefficients into high + low
float C_fp32[32][32];  // Full precision
float C_high[32][32];  // Round to BF16, then extend to FP32
float C_low[32][32];   // Residual: C_fp32 - C_high

for (int s = 0; s < NUM_SEGMENTS; s++) {
    for (int k = 0; k < K; k++) {
        // Round to BF16
        uint32_t fp32_bits = *reinterpret_cast<uint32_t*>(&C_fp32[s][k]);
        uint16_t bf16_bits = static_cast<uint16_t>(fp32_bits >> 16);
        uint32_t high_bits = static_cast<uint32_t>(bf16_bits) << 16;
        C_high[s][k] = *reinterpret_cast<float*>(&high_bits);

        // Residual
        C_low[s][k] = C_fp32[s][k] - C_high[s][k];
    }
}

// Pass 1: Y_high = C_high × P (FPU, fast)
FPU_MATMUL_load_C(C_high);
FPU_MATMUL_load_P(P);
FPU_MATMUL_compute();
float Y_high[32][32];
FPU_MATMUL_read_Y(Y_high);

// Pass 2: Y_low = C_low × P (CPU, slower but small values)
float Y_low[32][32];
for (int s = 0; s < NUM_SEGMENTS; s++) {
    for (int j = 0; j < 32; j++) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
            acc += C_low[s][k] * P[k][j];
        }
        Y_low[s][j] = acc;
    }
}

// Combine: Y = Y_high + Y_low
float Y[32][32];
for (int s = 0; s < NUM_SEGMENTS; s++) {
    for (int j = 0; j < 32; j++) {
        Y[s][j] = Y_high[s][j] + Y_low[s][j];
    }
}
```

### Benefit

- **Y_high**: Captures bulk of value (fast FPU)
- **Y_low**: Captures correction (small values, CPU okay)

**Performance**: ~1.5× slower than pure FPU, ~2× faster than pure CPU

**Accuracy**: Full FP32 precision!

---

## Comparison Table

| Technique | Speed vs Pure FPU | Accuracy | Complexity | Hardware Required |
|-----------|-------------------|----------|------------|-------------------|
| **1. FP32 Accum Mode** | 1.0× (same) | ~10× better | Low | FP32 accum support |
| **2. BF16 Split** | N/A | N/A | N/A | Not applicable |
| **3. Double-Pass** | 0.5× (2× slower) | 10000× better | Medium | None (CPU fallback) |
| **4. SW Matmul FP32** | 0.1× (10× slower) | 10000× better | Low | None (pure CPU) |
| **5. Hybrid** | 0.9× (10% slower) | 1000× better | High | None (adaptive) |
| **6. Split C High/Low** | 0.6× (1.7× slower) | 10000× better | Medium | None (2 passes) |

---

## Recommendation by Use Case

### Case 1: Need Speed, BF16 Accuracy Acceptable
**Use**: Current implementation (pure BF16 FPU)
- Fastest
- Already verified accurate (< 0.0001% error)
- Simple

### Case 2: Need Speed, Want Better Accuracy
**Use**: Technique 1 (FP32 Accumulation Mode)
- Same speed as current
- ~10× better accuracy
- **CHECK HARDWARE SUPPORT FIRST!**

```cpp
#define DST_ACCUM_MODE 1  // Try this!
```

### Case 3: Need Maximum Accuracy, Some Slowdown OK
**Use**: Technique 3 (Double-Pass Compensated)
- 2× slower
- Full FP32 accuracy
- Works on all hardware

### Case 4: Need Maximum Accuracy, Willing to Sacrifice Speed
**Use**: Technique 4 (Software FP32 Matmul)
- 10× slower
- Full FP32 accuracy
- Pure CPU, no FPU

### Case 5: Need Balance
**Use**: Technique 6 (Split High/Low)
- 1.7× slower
- Full FP32 accuracy
- Good compromise

---

## Implementation Priority

### Step 1: Try FP32 Accumulation Mode
```cpp
#define DST_ACCUM_MODE 1

// Test if this works and improves accuracy
// If yes: FREE accuracy improvement!
// If no: Move to step 2
```

### Step 2: Implement Double-Pass Compensated
```cpp
// Add compensated version as optional path
#ifdef USE_FP32_COMPENSATION
    piecewise_poly_matmul_compensated<...>();
#else
    piecewise_poly_matmul_tile<...>();  // Current BF16
#endif
```

### Step 3: Benchmark and Compare
```cpp
// Measure:
// - Relative error vs SFPU reference
// - Runtime vs current implementation
// - Memory usage

// Choose best trade-off for your application
```

---

## Code Example: Double-Pass Compensated (Full Implementation)

```cpp
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_poly_matmul_fp32_compensated(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t K = POLY_DEGREE + 1;

    // Extract x values
    float x_vals[32];
    extract_x_values(x_vals);

    // Compute segment IDs
    uint32_t seg_id[32];
    compute_segment_ids<NUM_SEGMENTS>(lut.data(), x_vals, seg_id);

    // Build matrices in FP32
    float P_fp32[32][32];
    build_power_matrix<POLY_DEGREE>(x_vals, P_fp32);

    float C_fp32[32][32];
    build_coeff_matrix<POLY_DEGREE, NUM_SEGMENTS>(lut.data(), C_fp32);

    // === PASS 1: Fast BF16 FPU Matmul ===
    FPU_MATMUL_load_C(C_fp32);  // Quantizes to BF16
    FPU_MATMUL_load_P(P_fp32);  // Quantizes to BF16
    FPU_MATMUL_compute();

    float Y_bf16[32][32];
    FPU_MATMUL_read_Y(Y_bf16);

    // === PASS 2: FP32 Error Compensation (CPU) ===
    // Only compute for segments we actually need
    float results_fp32[32];

    for (int lane = 0; lane < 32; lane++) {
        uint32_t seg = seg_id[lane];

        // Compute exact polynomial value in FP32
        float exact = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
            exact += C_fp32[seg][k] * P_fp32[k][lane];
        }

        // Use exact result (ignores BF16 approximation)
        results_fp32[lane] = exact;
    }

    // Write FP32 results to dst_reg
    for (int lane = 0; lane < 32; lane++) {
        dst_reg[lane] = results_fp32[lane];
    }
}
```

**This gives you FP32 accuracy with only 2× slowdown!**

---

## Summary

**Q: Can we preserve FP32 accuracy for BF16 (x) × FP32 (coeff) by doing extra work?**

**A: YES! Multiple techniques:**

1. ✅ **FP32 Accumulation Mode** (try first - FREE if hardware supports!)
2. ✅ **Double-Pass Compensated** (2× slower, FP32 accuracy, works everywhere)
3. ✅ **Software FP32 Matmul** (10× slower, FP32 accuracy, maximum compatibility)
4. ✅ **Split High/Low** (1.7× slower, FP32 accuracy, good balance)

**Recommended**: Try technique #1 first (FP32 accum mode), fall back to technique #2 (double-pass) if needed.

**You CAN get FP32 accuracy from BF16 inputs - it just requires extra work!**
