# FPU Implementation - LLK API Reference

## Quick Reference: Four FPU Helper Functions

This document provides the exact LLK API mappings for the four FPU helper functions implemented in `piecewise_fpu.cpp`.

---

## 1. FPU_MATMUL_load_C

### Purpose
Load coefficient matrix C[S×K] into SrcB registers for FPU matmul.

### Implementation Pattern
```cpp
inline void FPU_MATMUL_load_C(const float C[32][32]) {
    // 1. Convert float32 to bfloat16 and write to CB
    cb_reserve_back(CB_SCRATCH_C, 1);
    uint32_t cb_addr = get_write_ptr(CB_SCRATCH_C);
    volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_addr);

    for (int r = 0; r < 32; r++) {
        for (int c = 0; c < 32; c++) {
            uint32_t f32_bits = *reinterpret_cast<const uint32_t*>(&C[r][c]);
            dst[r * 32 + c] = static_cast<uint16_t>(f32_bits >> 16);
        }
    }
    cb_push_back(CB_SCRATCH_C, 1);

    // 2. Unpacker will load from CB to SrcB when matmul_tiles() is called
}
```

### LLK APIs Used (Indirect)
- **Direct**: CB write operations (`cb_reserve_back`, `cb_push_back`)
- **Indirect**: `llk_unpack_AB_matmul(CB_SCRATCH_C, ...)` called by `matmul_tiles()`
  - Defined in: `llk_unpack_AB_matmul_api.h`
  - Unpacks tile from CB_SCRATCH_C into SrcB registers

### Data Flow
```
Local Array C[32][32] (float32)
    ↓ [Convert to bfloat16]
L1 Circular Buffer CB_SCRATCH_C
    ↓ [llk_unpack_AB_matmul - UNPACK RISC]
SrcB Registers (hardware register file)
```

---

## 2. FPU_MATMUL_load_P

### Purpose
Load power matrix P[K×32] into SrcA registers for FPU matmul.

### Implementation Pattern
```cpp
inline void FPU_MATMUL_load_P(const float P[32][32]) {
    // 1. Convert float32 to bfloat16 and write to CB
    cb_reserve_back(CB_SCRATCH_P, 1);
    uint32_t cb_addr = get_write_ptr(CB_SCRATCH_P);
    volatile tt_l1_ptr uint16_t* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_addr);

    for (int r = 0; r < 32; r++) {
        for (int c = 0; c < 32; c++) {
            uint32_t f32_bits = *reinterpret_cast<const uint32_t*>(&P[r][c]);
            dst[r * 32 + c] = static_cast<uint16_t>(f32_bits >> 16);
        }
    }
    cb_push_back(CB_SCRATCH_P, 1);

    // 2. Unpacker will load from CB to SrcA when matmul_tiles() is called
}
```

### LLK APIs Used (Indirect)
- **Direct**: CB write operations (`cb_reserve_back`, `cb_push_back`)
- **Indirect**: `llk_unpack_AB_matmul(..., CB_SCRATCH_P)` called by `matmul_tiles()`
  - Defined in: `llk_unpack_AB_matmul_api.h`
  - Unpacks tile from CB_SCRATCH_P into SrcA registers

### Data Flow
```
Local Array P[32][32] (float32)
    ↓ [Convert to bfloat16]
L1 Circular Buffer CB_SCRATCH_P
    ↓ [llk_unpack_AB_matmul - UNPACK RISC]
SrcA Registers (hardware register file)
```

---

## 3. FPU_MATMUL_compute

### Purpose
Perform hardware FPU matrix multiplication: Dest = SrcB × SrcA = C × P

### Implementation Pattern
```cpp
inline void FPU_MATMUL_compute() {
    // 1. Wait for input matrices
    cb_wait_front(CB_SCRATCH_C, 1);
    cb_wait_front(CB_SCRATCH_P, 1);

    // 2. Reserve output space
    cb_reserve_back(CB_SCRATCH_Y, 1);

    // 3. Perform FPU matmul (this is the key operation!)
    matmul_tiles(CB_SCRATCH_C, CB_SCRATCH_P, 0, 0, 0);
    //           ↑ in0_cb (SrcB) ↑ in1_cb (SrcA)

    // 4. Pack result from Dest to output CB
    pack_tile(0, CB_SCRATCH_Y);
    cb_push_back(CB_SCRATCH_Y, 1);

    // 5. Cleanup
    cb_pop_front(CB_SCRATCH_C, 1);
    cb_pop_front(CB_SCRATCH_P, 1);
}
```

### LLK APIs Used (Via Wrapper)

The `matmul_tiles()` wrapper expands to:

```cpp
// UNPACK RISC (runs on unpacker processor)
UNPACK((llk_unpack_AB_matmul(CB_SCRATCH_C, CB_SCRATCH_P, 0, 0)));

// MATH RISC (runs on math/compute processor)
MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(0)));
```

**Unpacker LLK** (`llk_unpack_AB_matmul`):
- File: `llk_unpack_AB_matmul_api.h`
- Function: `llk_unpack_AB_matmul(in0_cb, in1_cb, in0_tile_idx, in1_tile_idx)`
- Action:
  - Loads tile from `in0_cb` (CB_SCRATCH_C) to **SrcB**
  - Loads tile from `in1_cb` (CB_SCRATCH_P) to **SrcA**

**Math LLK** (`llk_math_matmul`):
- File: `llk_math_matmul_api.h`
- Function: `llk_math_matmul<FIDELITY, THROTTLE>(dst_idx)`
- Action:
  - Executes hardware FPU matmul: **Dest[dst_idx] = SrcB × SrcA**
  - Uses hardware matrix multiply unit (FPU/MVMUL)
  - Result accumulated in Dest register file

**Packer** (`pack_tile`):
- File: LLK pack APIs
- Function: Packs Dest register to output CB
- Action: Converts Dest register format to CB tile format

### Data Flow
```
CB_SCRATCH_C → [llk_unpack_AB_matmul] → SrcB Registers
CB_SCRATCH_P → [llk_unpack_AB_matmul] → SrcA Registers
                    ↓
            [llk_math_matmul - FPU Hardware]
                    ↓
              Dest Registers
                    ↓
            [pack_tile]
                    ↓
            CB_SCRATCH_Y
```

### Critical Note: Operand Mapping

⚠️ **Important**: The operand mapping in `matmul_tiles()` is swapped from intuition:

| Function Argument | Hardware Register | Our Usage |
|------------------|-------------------|-----------|
| `in0_cb` | → SrcB | CB_SCRATCH_C (coefficient matrix) |
| `in1_cb` | → SrcA | CB_SCRATCH_P (power matrix) |

So when we write:
```cpp
matmul_tiles(CB_SCRATCH_C, CB_SCRATCH_P, 0, 0, 0);
```

The hardware computes:
```
Dest = SrcB × SrcA = C × P  ✓ (what we want!)
```

This swap is intentional and documented in `matmul.h`:
> "Note: in0_cb_id and in1_cb_id are swapped here because internally, matmul maps in0 to srcB and in1 to srcA"

---

## 4. FPU_MATMUL_read_Y

### Purpose
Read result matrix Y[S×32] from Dest (via output CB) back to local memory.

### Implementation Pattern
```cpp
inline void FPU_MATMUL_read_Y(float Y[32][32]) {
    // 1. Wait for result
    cb_wait_front(CB_SCRATCH_Y, 1);

    // 2. Get CB address
    uint32_t cb_addr = get_read_ptr(CB_SCRATCH_Y);
    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_addr);

    // 3. Read and convert bfloat16 → float32
    for (int r = 0; r < 32; r++) {
        for (int c = 0; c < 32; c++) {
            uint16_t bf16_bits = src[r * 32 + c];
            uint32_t f32_bits = static_cast<uint32_t>(bf16_bits) << 16;
            Y[r][c] = *reinterpret_cast<float*>(&f32_bits);
        }
    }

    // 4. Release CB
    cb_pop_front(CB_SCRATCH_Y, 1);
}
```

### LLK APIs Used (Indirect)
- **Indirect**: Result was written by `pack_tile()` which uses LLK pack APIs
- **Direct**: CB read operations (`cb_wait_front`, `cb_pop_front`)

### Data Flow
```
Dest Registers (matmul result)
    ↓ [pack_tile - PACK RISC]
L1 Circular Buffer CB_SCRATCH_Y (bfloat16)
    ↓ [Convert to float32]
Local Array Y[32][32] (float32)
```

---

## Complete FPU Matmul Pipeline

### Full Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ MATH RISC: Build Matrices                                          │
├─────────────────────────────────────────────────────────────────────┤
│  Local Memory:                                                      │
│    float C[32][32] ← build_coeff_matrix()                          │
│    float P[32][32] ← build_power_matrix()                          │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MATH RISC: Write to CBs                                            │
├─────────────────────────────────────────────────────────────────────┤
│  FPU_MATMUL_load_C(C) → CB_SCRATCH_C (bfloat16)                    │
│  FPU_MATMUL_load_P(P) → CB_SCRATCH_P (bfloat16)                    │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ FPU_MATMUL_compute()                                                │
├─────────────────────────────────────────────────────────────────────┤
│  UNPACK RISC:                                                       │
│    llk_unpack_AB_matmul(CB_SCRATCH_C → SrcB)                       │
│    llk_unpack_AB_matmul(CB_SCRATCH_P → SrcA)                       │
│                                                                     │
│  MATH RISC:                                                         │
│    llk_math_matmul() → Dest = SrcB × SrcA = C × P                 │
│                        [Hardware FPU Matrix Multiply Unit]          │
│                                                                     │
│  PACK RISC:                                                         │
│    pack_tile(Dest → CB_SCRATCH_Y)                                  │
└────────────────────────┬────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ MATH RISC: Read Result                                             │
├─────────────────────────────────────────────────────────────────────┤
│  FPU_MATMUL_read_Y() ← CB_SCRATCH_Y                                │
│    float Y[32][32] (result matrix)                                 │
│                                                                     │
│  Select results:                                                    │
│    result[lane] = Y[seg_id[lane]][lane]                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## LLK Header Files Reference

| Component | Header File | Path |
|-----------|-------------|------|
| Unpack AB Matmul | `llk_unpack_AB_matmul_api.h` | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/` |
| Math Matmul | `llk_math_matmul_api.h` | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/` |
| Pack | `llk_pack_api.h` | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/` |
| High-Level Wrapper | `matmul.h` | `tt_metal/include/compute_kernel_api/` |

---

## Comparison: SFPU vs FPU Implementation

### SFPU Horner (piecewise_generic.cpp)

```cpp
for (int d = 0; d < 32; d++) {
    vFloat x = dst_reg[d];

    // Clamp x
    vFloat x_clamped = clamp(x, lut[0], lut[NUM_SEGMENTS]);

    // Start with segment 0
    vFloat result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET], x);

    // Check all other segments
    for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
        v_if (x_clamped >= lut[seg]) {
            result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + seg * K], x);
        }
        v_endif;
    }

    dst_reg[d] = result;
}
```

**Key Operations**:
- Per-lane SFPU vector operations
- S segment comparisons × 32 lanes
- D Horner steps × S segments × 32 lanes
- Total: O(32 × S × D) SFPU operations

### FPU Matmul (piecewise_fpu.cpp)

```cpp
// Extract all 32 x values
float x_vals[32];
extract_x_values(x_vals);

// Compute segment IDs
uint32_t seg_id[32];
compute_segment_ids<NUM_SEGMENTS>(lut, x_vals, seg_id);

// Build matrices
float P[32][32];  // P[k][j] = x[j]^k
build_power_matrix<POLY_DEGREE>(x_vals, P);

float C[32][32];  // C[s][k] = coeff k for segment s
build_coeff_matrix<POLY_DEGREE, NUM_SEGMENTS>(lut, C);

// Single FPU matmul: Y = C × P
FPU_MATMUL_load_C(C);
FPU_MATMUL_load_P(P);
FPU_MATMUL_compute();  // Hardware matmul!

// Read and select results
float Y[32][32];
FPU_MATMUL_read_Y(Y);

for (int lane = 0; lane < 32; lane++) {
    dst_reg[lane] = Y[seg_id[lane]][lane];
}
```

**Key Operations**:
- 1 hardware FPU matmul (S×K by K×32)
- Matrix building overhead (one-time per tile)
- Total: O(1) FPU matmul + O(32×K) matrix building

### Performance Comparison

| Configuration | SFPU Ops | FPU Ops | FPU Advantage |
|--------------|----------|---------|---------------|
| S=8, D=2 | 512 | 1 matmul | ~2× |
| S=16, D=4 | 2,048 | 1 matmul | ~5× |
| S=32, D=8 | 8,192 | 1 matmul | ~10×+ |

**Note**: FPU matmul is hardware-pipelined and executes in fixed time regardless of S×K dimensions (up to 32×32). SFPU cost grows linearly with S and D.

---

## Example: Segment Selection Logic

For a lane with `x = 2.5`, boundaries `[0, 1, 2, 3, 4]`, segments = 4:

### SFPU Approach
```cpp
vFloat result = eval_poly<2>(&lut[5], x);        // Segment 0
v_if (x >= 1.0) result = eval_poly<2>(&lut[8], x);   // Segment 1
v_endif;
v_if (x >= 2.0) result = eval_poly<2>(&lut[11], x);  // Segment 2 ✓
v_endif;
v_if (x >= 3.0) result = eval_poly<2>(&lut[14], x);  // Segment 3
v_endif;
```
→ Evaluates **4 polynomials** per lane (wastes 3)

### FPU Approach
```cpp
// Determine segment once
seg_id = 2;  // x=2.5 is in segment 2

// Compute ALL segments simultaneously
Y = C × P;
// Y[0][lane] = segment 0 result
// Y[1][lane] = segment 1 result
// Y[2][lane] = segment 2 result ✓ (select this one)
// Y[3][lane] = segment 3 result

result = Y[2][lane];
```
→ Computes **all 4 polynomials** in one matmul, then selects the right one

**Insight**: FPU does more work but in parallel, SFPU does serial work with predication overhead.

---

## Summary

| Aspect | SFPU Horner | FPU Matmul |
|--------|------------|------------|
| **Computation Model** | Serial per-lane evaluation | Parallel matmul for all segments |
| **Complexity** | O(S × D) per lane | O(1) matmul per tile |
| **Hardware Used** | SFPU vector unit | FPU matrix engine |
| **LLK APIs** | SFPU intrinsics | llk_unpack_AB_matmul, llk_math_matmul |
| **Best For** | Low S, low D | High S, high D |
| **Memory** | Minimal (LUT only) | Requires 3 scratch CBs |
| **Numerical** | Bfloat16 SFPU | Bfloat16 FPU |
| **Latency** | Low setup | Moderate setup (CB writes) |
| **Throughput** | Limited by S×D | Fixed (pipelined matmul) |

The FPU implementation trades increased per-tile complexity for dramatically better scalability with segment count and polynomial degree.
