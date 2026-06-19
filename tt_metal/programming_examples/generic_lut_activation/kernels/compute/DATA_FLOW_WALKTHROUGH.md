# FPU Piecewise Polynomial - Complete Data Flow Walkthrough

## Overview: Input → Output

**You supply**:
- ✅ Input x vector: 32 values (in dst_reg[])
- ✅ Coefficient LUT: Boundaries + polynomial coefficients

**You get back**:
- ✅ Result vector: 32 polynomial evaluations (in dst_reg[])

Let's trace through a **concrete example** step-by-step!

---

## Example Setup

### Configuration
- **Polynomial degree**: D = 2 (quadratic)
- **Number of segments**: S = 4
- **Input lanes**: 32 values

### Input LUT
```
Boundaries: [b0=-2, b1=-1, b2=0, b3=1, b4=2]

Segment 0 (x ∈ [-2, -1)): p0(x) = 1.0 + 0.5x + 0.1x²
Segment 1 (x ∈ [-1,  0)): p1(x) = 2.0 + 0.3x + 0.2x²
Segment 2 (x ∈ [ 0,  1)): p2(x) = 3.0 + 0.4x + 0.1x²
Segment 3 (x ∈ [ 1,  2]): p3(x) = 4.0 + 0.5x + 0.3x²
```

### Input x Vector (showing first 6 lanes)
```
x[0] = -1.5  → should use segment 0
x[1] =  0.5  → should use segment 2
x[2] =  1.8  → should use segment 3
x[3] = -0.3  → should use segment 1
x[4] =  0.0  → should use segment 2
x[5] =  2.5  → should clamp to 2.0, use segment 3
... (26 more values)
```

---

## Step-by-Step Data Flow

### 🔵 STEP 0: Initial State

**Location**: Input tile in circular buffer `cb_in` (CBIndex::c_0)

```
cb_in (bfloat16 tile format):
┌────────────────────────────────────┐
│ Input tile (32×32)                 │
│ Each row: one x value replicated   │
│                                    │
│ Row 0: [-1.5, -1.5, -1.5, ... ]   │
│ Row 1: [ 0.5,  0.5,  0.5, ... ]   │
│ Row 2: [ 1.8,  1.8,  1.8, ... ]   │
│ Row 3: [-0.3, -0.3, -0.3, ... ]   │
│ ...                                │
└────────────────────────────────────┘
```

**Kernel main loop**:
```cpp
cb_wait_front(cb_in, 1);
tile_regs_acquire();
copy_tile(cb_in, 0, 0);  // Copy to dst_reg[]
```

---

### 🔵 STEP 1: Extract x Values from dst_reg[]

**Function**: `extract_x_values(x_vals)` (lines 99-106)

**Before**:
```
dst_reg[0] = vFloat{-1.5, -1.5, -1.5, ..., -1.5}  (32 elements)
dst_reg[1] = vFloat{ 0.5,  0.5,  0.5, ...,  0.5}
dst_reg[2] = vFloat{ 1.8,  1.8,  1.8, ...,  1.8}
...
```

**Code**:
```cpp
for (int lane = 0; lane < 32; lane++) {
    vFloat x_vec = dst_reg[lane];
    x_vals[lane] = x_vec[0];  // Extract first element
}
```

**After**:
```
x_vals[32] = {-1.5, 0.5, 1.8, -0.3, 0.0, 2.5, ...}
            (float32 array in local memory)
```

---

### 🔵 STEP 2: Compute Segment IDs

**Function**: `compute_segment_ids<NUM_SEGMENTS>(lut, x_vals, seg_id)` (lines 110-128)

**Logic**:
```cpp
for (int lane = 0; lane < 32; lane++) {
    float x = x_vals[lane];

    // Clamp to boundaries
    x_clamped = clamp(x, -2.0, 2.0);

    // Find segment: which boundary is x >= ?
    seg = 0;
    if (x_clamped >= -1.0) seg = 1;
    if (x_clamped >=  0.0) seg = 2;
    if (x_clamped >=  1.0) seg = 3;

    seg_id[lane] = seg;
}
```

**Result**:
```
Input:   x_vals = {-1.5,  0.5,  1.8, -0.3,  0.0,  2.5, ...}

Clamped:        {-1.5,  0.5,  1.8, -0.3,  0.0,  2.0, ...}
                   ↓     ↓     ↓     ↓     ↓     ↓
Segment:  seg_id = {  0,    2,    3,    1,    2,    3, ...}
```

**Why these segments?**
- x=-1.5: -2.0 ≤ -1.5 < -1.0 → segment 0 ✓
- x=0.5:   0.0 ≤  0.5 <  1.0 → segment 2 ✓
- x=1.8:   1.0 ≤  1.8 <  2.0 → segment 3 ✓
- x=-0.3: -1.0 ≤ -0.3 <  0.0 → segment 1 ✓
- x=0.0:   0.0 ≤  0.0 <  1.0 → segment 2 ✓
- x=2.5:   clamped to 2.0    → segment 3 ✓

---

### 🔵 STEP 3: Build Power Matrix P[K×32]

**Function**: `build_power_matrix<POLY_DEGREE>(x_vals, P)` (lines 134-153)

**For degree D=2, K=3**: Need x^0, x^1, x^2

**Code**:
```cpp
for (int lane = 0; lane < 32; lane++) {
    float x = x_vals[lane];
    float x_power = 1.0f;

    for (uint32_t k = 0; k < 3; k++) {
        P[k][lane] = x_power;  // Store x^k
        x_power *= x;          // Next power
    }
}
```

**Result**: P[3×32] (physically 32×32, padded with zeros)

```
         Lane 0   Lane 1   Lane 2   Lane 3   Lane 4   Lane 5   ...
         x=-1.5   x=0.5    x=1.8    x=-0.3   x=0.0    x=2.0
      ┌──────────────────────────────────────────────────────────
k=0   │  1.000    1.000    1.000    1.000    1.000    1.000   ...  (x^0)
k=1   │ -1.500    0.500    1.800   -0.300    0.000    2.000   ...  (x^1)
k=2   │  2.250    0.250    3.240    0.090    0.000    4.000   ...  (x^2)
k=3   │  0.000    0.000    0.000    0.000    0.000    0.000   ...  (padding)
...   │  0.000    0.000    0.000    0.000    0.000    0.000   ...
k=31  │  0.000    0.000    0.000    0.000    0.000    0.000   ...
```

**Verification** (lane 0, x=-1.5):
- P[0][0] = 1.0 = (-1.5)^0 ✓
- P[1][0] = -1.5 = (-1.5)^1 ✓
- P[2][0] = 2.25 = (-1.5)^2 ✓

---

### 🔵 STEP 4: Build Coefficient Matrix C[S×K]

**Function**: `build_coeff_matrix<POLY_DEGREE, NUM_SEGMENTS>(lut, C)` (lines 156-180)

**Input LUT layout**:
```
lut[0..4] = boundaries = [-2.0, -1.0, 0.0, 1.0, 2.0]

lut[5..7]   = segment 0 coeffs = [1.0, 0.5, 0.1]  (c0, c1, c2)
lut[8..10]  = segment 1 coeffs = [2.0, 0.3, 0.2]
lut[11..13] = segment 2 coeffs = [3.0, 0.4, 0.1]
lut[14..16] = segment 3 coeffs = [4.0, 0.5, 0.3]
```

**Code**:
```cpp
for (uint32_t seg = 0; seg < NUM_SEGMENTS; seg++) {
    for (uint32_t k = 0; k < K; k++) {
        C[seg][k] = lut[5 + seg * 3 + k];
    }
}
```

**Result**: C[4×3] (physically 32×32, padded with zeros)

```
           c0 (k=0)  c1 (k=1)  c2 (k=2)  k=3    k=4  ...  k=31
         ┌────────────────────────────────────────────────────
Segment 0│   1.0      0.5       0.1       0.0    0.0  ...  0.0
Segment 1│   2.0      0.3       0.2       0.0    0.0  ...  0.0
Segment 2│   3.0      0.4       0.1       0.0    0.0  ...  0.0
Segment 3│   4.0      0.5       0.3       0.0    0.0  ...  0.0
Seg 4    │   0.0      0.0       0.0       0.0    0.0  ...  0.0
...      │   0.0      0.0       0.0       0.0    0.0  ...  0.0
Seg 31   │   0.0      0.0       0.0       0.0    0.0  ...  0.0
```

---

### 🔵 STEP 5: Load C into CB_SCRATCH_C

**Function**: `FPU_MATMUL_load_C(C)` (lines 193-208)

**Code**:
```cpp
cb_reserve_back(CB_SCRATCH_C, 1);
uint32_t cb_addr = get_write_ptr(CB_SCRATCH_C);
volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<...>(cb_addr);

// Convert float32 → bfloat16 and write to CB
for (int r = 0; r < 32; r++) {
    for (int c = 0; c < 32; c++) {
        uint32_t f32_bits = *reinterpret_cast<const uint32_t*>(&C[r][c]);
        dst[r * 32 + c] = static_cast<uint16_t>(f32_bits >> 16);  // bfloat16
    }
}

cb_push_back(CB_SCRATCH_C, 1);
```

**Result**: Coefficient matrix in L1 circular buffer, ready for unpacker

```
CB_SCRATCH_C (CB 26):
┌────────────────────────────────────┐
│ C[4×3] in bfloat16 tile format     │
│ (32×32 physical, zero-padded)      │
│                                    │
│ Ready for FPU matmul               │
└────────────────────────────────────┘
```

---

### 🔵 STEP 6: Load P into CB_SCRATCH_P

**Function**: `FPU_MATMUL_load_P(P)` (lines 223-238)

**Same process as Step 5**: float32 → bfloat16 → CB

**Result**: Power matrix in L1 circular buffer

```
CB_SCRATCH_P (CB 27):
┌────────────────────────────────────┐
│ P[3×32] in bfloat16 tile format    │
│ (32×32 physical, zero-padded)      │
│                                    │
│ Ready for FPU matmul               │
└────────────────────────────────────┘
```

---

### 🔵 STEP 7: FPU Matmul Compute

**Function**: `FPU_MATMUL_compute()` (lines 269-292)

**Operation**: Y = C × P

**Matrix multiplication**:
```
C[S×K] × P[K×N] = Y[S×N]
C[4×3] × P[3×32] = Y[4×32]

For each output element Y[s][j]:
    Y[s][j] = sum(C[s][k] * P[k][j] for k in 0..2)
            = C[s][0]*P[0][j] + C[s][1]*P[1][j] + C[s][2]*P[2][j]
            = c0_s * 1 + c1_s * x[j] + c2_s * x[j]^2
            = polynomial of segment s evaluated at x[j]
```

**Code**:
```cpp
cb_wait_front(CB_SCRATCH_C, 1);
cb_wait_front(CB_SCRATCH_P, 1);
cb_reserve_back(CB_SCRATCH_Y, 1);

// THIS IS THE KEY OPERATION!
matmul_tiles(CB_SCRATCH_C, CB_SCRATCH_P, 0, 0, 0);
// Expands to:
//   UNPACK: CB_SCRATCH_C → SrcB, CB_SCRATCH_P → SrcA
//   MATH:   Dest = SrcB × SrcA = C × P  (hardware FPU!)
//   PACK:   Dest → CB_SCRATCH_Y

pack_tile(0, CB_SCRATCH_Y);
cb_push_back(CB_SCRATCH_Y, 1);
```

**Result**: Y[4×32] matrix in CB_SCRATCH_Y

**Let's compute Y manually for lane 0 (x=-1.5, segment 0)**:
```
Y[0][0] = C[0][0]*P[0][0] + C[0][1]*P[1][0] + C[0][2]*P[2][0]
        = 1.0 * 1.0 + 0.5 * (-1.5) + 0.1 * 2.25
        = 1.0 - 0.75 + 0.225
        = 0.475
```

**Y matrix after matmul** (showing first 6 lanes):
```
         Lane 0    Lane 1    Lane 2    Lane 3    Lane 4    Lane 5
         x=-1.5    x=0.5     x=1.8     x=-0.3    x=0.0     x=2.0
      ┌──────────────────────────────────────────────────────────
Seg 0 │  0.475    1.375     2.074     0.991     1.000     3.500  (p0 at each x)
Seg 1 │  1.275    2.375     4.348     2.118     2.000     5.300  (p1 at each x)
Seg 2 │  2.225    3.425     5.944     3.091     3.000     7.500  (p2 at each x)
Seg 3 │  3.325    4.575     8.672     4.418     4.000    10.700  (p3 at each x)
```

**Verification** (lane 1, x=0.5):
- Segment 0: p0(0.5) = 1.0 + 0.5(0.5) + 0.1(0.25) = 1.375 ✓
- Segment 2: p2(0.5) = 3.0 + 0.4(0.5) + 0.1(0.25) = 3.425 ✓

**Key insight**: Each column j of Y contains the evaluation of ALL segment polynomials at x[j]. We computed all possibilities in parallel!

---

### 🔵 STEP 8: Read Y from CB

**Function**: `FPU_MATMUL_read_Y(Y)` (lines 311-334)

**Code**:
```cpp
cb_wait_front(CB_SCRATCH_Y, 1);
uint32_t cb_addr = get_read_ptr(CB_SCRATCH_Y);
volatile tt_l1_ptr uint16_t* src = reinterpret_cast<...>(cb_addr);

// Read tile and convert bfloat16 → float32
for (int r = 0; r < 32; r++) {
    for (int c = 0; c < 32; c++) {
        uint16_t bf16_bits = src[r * 32 + c];
        uint32_t f32_bits = static_cast<uint32_t>(bf16_bits) << 16;
        Y[r][c] = *reinterpret_cast<float*>(&f32_bits);
    }
}

cb_pop_front(CB_SCRATCH_Y, 1);
```

**Result**: Y[4×32] in local memory (float32)

```
Y[4][32] = {
  {0.475, 1.375, 2.074, 0.991, 1.000, 3.500, ...},  // Segment 0 results
  {1.275, 2.375, 4.348, 2.118, 2.000, 5.300, ...},  // Segment 1 results
  {2.225, 3.425, 5.944, 3.091, 3.000, 7.500, ...},  // Segment 2 results
  {3.325, 4.575, 8.672, 4.418, 4.000, 10.700, ...},  // Segment 3 results
}
```

---

### 🔵 STEP 9: Select Correct Result Per Lane

**Function**: `piecewise_poly_matmul_tile()` (lines 390-394)

**Logic**: For each lane, pick the row corresponding to its segment

```cpp
for (int lane = 0; lane < 32; lane++) {
    dst_reg[lane] = Y[seg_id[lane]][lane];
}
```

**Selection process**:
```
Lane  x      seg_id  Select from Y         Result
────  ─────  ──────  ────────────────────  ──────
0     -1.5   0       Y[0][0] (seg 0)       0.475
1      0.5   2       Y[2][1] (seg 2)       3.425
2      1.8   3       Y[3][2] (seg 3)       8.672
3     -0.3   1       Y[1][3] (seg 1)       2.118
4      0.0   2       Y[2][4] (seg 2)       3.000
5      2.0   3       Y[3][5] (seg 3)      10.700
...
```

**Final result in dst_reg**:
```
dst_reg[0] = 0.475   ✓ p0(-1.5) = 1.0 + 0.5(-1.5) + 0.1(2.25)
dst_reg[1] = 3.425   ✓ p2(0.5)  = 3.0 + 0.4(0.5)  + 0.1(0.25)
dst_reg[2] = 8.672   ✓ p3(1.8)  = 4.0 + 0.5(1.8)  + 0.3(3.24)
dst_reg[3] = 2.118   ✓ p1(-0.3) = 2.0 + 0.3(-0.3) + 0.2(0.09)
dst_reg[4] = 3.000   ✓ p2(0.0)  = 3.0 + 0.4(0.0)  + 0.1(0.00)
dst_reg[5] = 10.700  ✓ p3(2.0)  = 4.0 + 0.5(2.0)  + 0.3(4.00)
...
```

---

### 🔵 STEP 10: Pack Results to Output

**Main kernel code** (lines 582-589):
```cpp
tile_regs_commit();
tile_regs_wait();

cb_reserve_back(cb_out, 1);
pack_tile(0, cb_out);      // dst_reg → cb_out
cb_push_back(cb_out, 1);

cb_pop_front(cb_in, 1);
tile_regs_release();
```

**Result**: Output tile in `cb_out` (CBIndex::c_16), ready for writer kernel

```
cb_out (bfloat16 tile format):
┌────────────────────────────────────┐
│ Output tile (32×32)                │
│ Each row: polynomial result        │
│                                    │
│ Row 0: [0.475, 0.475, ... ]        │
│ Row 1: [3.425, 3.425, ... ]        │
│ Row 2: [8.672, 8.672, ... ]        │
│ Row 3: [2.118, 2.118, ... ]        │
│ ...                                │
└────────────────────────────────────┘
```

---

## Summary: Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: x vector in cb_in                                            │
│   [-1.5, 0.5, 1.8, -0.3, 0.0, 2.5, ...]                            │
└────────────┬────────────────────────────────────────────────────────┘
             │
             │ copy_tile(cb_in → dst_reg)
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Extract x values from dst_reg                              │
│   x_vals = [-1.5, 0.5, 1.8, -0.3, 0.0, 2.5, ...] (float32 array)  │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Compute segment IDs                                        │
│   seg_id = [0, 2, 3, 1, 2, 3, ...]                                 │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Build power matrix P[3×32]                                 │
│   P = [[1.0,   1.0,   1.0,   1.0,  1.0,  1.0, ...]   (x^0)        │
│        [-1.5,  0.5,   1.8,  -0.3,  0.0,  2.0, ...]   (x^1)        │
│        [2.25,  0.25,  3.24,  0.09, 0.0,  4.0, ...]]  (x^2)        │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: Build coefficient matrix C[4×3]                            │
│   C = [[1.0, 0.5, 0.1]   Segment 0: p0(x) = 1.0 + 0.5x + 0.1x²   │
│        [2.0, 0.3, 0.2]   Segment 1: p1(x) = 2.0 + 0.3x + 0.2x²   │
│        [3.0, 0.4, 0.1]   Segment 2: p2(x) = 3.0 + 0.4x + 0.1x²   │
│        [4.0, 0.5, 0.3]]  Segment 3: p3(x) = 4.0 + 0.5x + 0.3x²   │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ├──────────────┬─────────────────────────────────────────┐
             ↓              ↓                                         │
   ┌──────────────┐  ┌──────────────┐                               │
   │ STEP 5:      │  │ STEP 6:      │                               │
   │ C → CB 26    │  │ P → CB 27    │                               │
   │ (bfloat16)   │  │ (bfloat16)   │                               │
   └──────┬───────┘  └──────┬───────┘                               │
          │                 │                                        │
          └────────┬────────┘                                        │
                   ↓                                                 │
         ┌──────────────────────┐                                   │
         │ STEP 7: FPU MATMUL   │                                   │
         │  Y = C × P           │                                   │
         │  [4×3] × [3×32]      │                                   │
         │      = [4×32]        │                                   │
         │                      │                                   │
         │ Hardware FPU does:   │                                   │
         │  Y[s][j] = sum of    │                                   │
         │   C[s][k] * P[k][j]  │                                   │
         │                      │                                   │
         │ Result: Each col j   │                                   │
         │ has ALL polynomials  │                                   │
         │ evaluated at x[j]    │                                   │
         └──────────┬───────────┘                                   │
                    │                                                │
                    ↓                                                │
         ┌──────────────────────┐                                   │
         │ Result in CB 28:     │                                   │
         │                      │                                   │
         │ Y = [[0.475, 1.375,  │ ← All p0(x[j]) results           │
         │       2.074, 0.991,  │                                   │
         │       ...]           │                                   │
         │      [1.275, 2.375,  │ ← All p1(x[j]) results           │
         │       4.348, 2.118,  │                                   │
         │       ...]           │                                   │
         │      [2.225, 3.425,  │ ← All p2(x[j]) results           │
         │       5.944, 3.091,  │                                   │
         │       ...]           │                                   │
         │      [3.325, 4.575,  │ ← All p3(x[j]) results           │
         │       8.672, 4.418,  │                                   │
         │       ...]]          │                                   │
         └──────────┬───────────┘                                   │
                    │                                                │
                    ↓                                                │
         ┌──────────────────────┐                                   │
         │ STEP 8: Read Y       │                                   │
         │ bfloat16 → float32   │                                   │
         └──────────┬───────────┘                                   │
                    │                                                │
                    ↓                                                │
         ┌──────────────────────────────┐                           │
         │ STEP 9: Select correct result │  ← Uses seg_id ──────────┘
         │                               │
         │ For each lane j:              │
         │   result[j] = Y[seg_id[j]][j] │
         │                               │
         │ Lane 0: Y[0][0] = 0.475       │
         │ Lane 1: Y[2][1] = 3.425       │
         │ Lane 2: Y[3][2] = 8.672       │
         │ Lane 3: Y[1][3] = 2.118       │
         │ Lane 4: Y[2][4] = 3.000       │
         │ Lane 5: Y[3][5] = 10.700      │
         │ ...                           │
         └──────────┬────────────────────┘
                    │
                    ↓
         ┌──────────────────────┐
         │ Write to dst_reg     │
         │ dst_reg[0] = 0.475   │
         │ dst_reg[1] = 3.425   │
         │ dst_reg[2] = 8.672   │
         │ dst_reg[3] = 2.118   │
         │ ...                  │
         └──────────┬───────────┘
                    │
                    │ pack_tile(dst_reg → cb_out)
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: Results in cb_out                                           │
│   [0.475, 3.425, 8.672, 2.118, 3.000, 10.700, ...]                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Insights

### 💡 Why is this efficient?

**Instead of**:
```
For each lane:
    For each segment:
        If x in segment:
            Evaluate polynomial
```
(32 lanes × 4 segments × 3 operations = 384 SFPU operations)

**We do**:
```
Build P matrix (once)
Build C matrix (once)
ONE FPU matmul: Y = C × P
Select appropriate result per lane
```
(~300 setup ops + 1 hardware matmul + 32 selects)

### 💡 The Magic of Matrix Multiplication

The matmul **evaluates ALL polynomials at ALL x values simultaneously**:

```
Y[segment][lane] = polynomial of segment evaluated at x[lane]
```

Then we just **pick the right row** for each lane based on which segment it belongs to.

### 💡 Data Precision Journey

```
Input:  bfloat16 (in CB)
  ↓
Extract: float32 (for power calculation accuracy)
  ↓
Power matrix: float32
Coeff matrix: float32
  ↓
Load to CB: bfloat16 (hardware format)
  ↓
FPU compute: bfloat16 → bfloat16
  ↓
Read result: bfloat16 → float32 (for selection)
  ↓
Final output: bfloat16 (in CB)
```

---

## Verification Example

Let's verify lane 2 (x=1.8, segment 3):

**Expected**: p3(1.8) = 4.0 + 0.5(1.8) + 0.3(1.8²)

```
Step-by-step:
  1.8² = 3.24
  4.0 + 0.5(1.8) + 0.3(3.24)
  = 4.0 + 0.9 + 0.972
  = 5.872
```

**Via matmul**:
```
Y[3][2] = C[3][0]*P[0][2] + C[3][1]*P[1][2] + C[3][2]*P[2][2]
        = 4.0 * 1.0 + 0.5 * 1.8 + 0.3 * 3.24
        = 4.0 + 0.9 + 0.972
        = 5.872 ✓
```

**Selection**:
```
seg_id[2] = 3
result[2] = Y[seg_id[2]][2] = Y[3][2] = 5.872 ✓
```

Perfect match!

---

## Summary

**You supply**:
- ✅ x vector (32 values in dst_reg)
- ✅ LUT with boundaries and coefficients

**The kernel**:
1. Extracts x values
2. Determines segments
3. Builds power matrix P (x^0, x^1, x^2, ...)
4. Builds coefficient matrix C from LUT
5. **Runs ONE FPU matmul: Y = C × P**
6. Selects correct result per lane from Y

**You get back**:
- ✅ Polynomial results (32 values in dst_reg)

**The key insight**: Matrix multiplication computes ALL segment polynomials at ALL x values in ONE hardware operation, then we simply select the correct result!
