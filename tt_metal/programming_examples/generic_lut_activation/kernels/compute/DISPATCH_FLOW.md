# TF32 FPU Kernel - Complete Dispatch Flow

## Your Question: How Does Dispatch Know to Use the TF32 Flow?

**Short Answer**: The TF32 flow is selected at **build time** via the `KERNEL_VARIANT` macro, which tells CMake which kernel file to compile. There's NO runtime dispatch - the kernel path is hardcoded into the binary at compile time.

---

## The Complete Dispatch Chain

### Level 1: Build-Time Kernel Selection (CMake)

```
User runs:
  cmake -DKERNEL_VARIANT=piecewise_fpu_tf32 ...
       │
       └─→ CMake passes -DKERNEL_VARIANT=piecewise_fpu_tf32 to compiler
                │
                └─→ In generic_lut_activation.cpp (line 40):
                    #define COMPUTE_KERNEL_PATH ".../" KERNEL_VARIANT ".cpp"

                    Expands to:
                    #define COMPUTE_KERNEL_PATH \
                      "generic_lut_activation/kernels/compute/piecewise_fpu_tf32.cpp"
```

**Result**: The TF32 kernel file is selected at compile time, baked into the binary.

### Level 2: Host Program Creates Kernel (Runtime)

```cpp
// In generic_lut_activation.cpp (line 514-524)

auto compute_kernel_1 = CreateKernel(
    program,
    COMPUTE_KERNEL_PATH,  // ← Points to piecewise_fpu_tf32.cpp (from build)
    core_group_1,
    ComputeConfig{
        .fp32_dest_acc_en = !use_bf16_mode,  // ← Enables FP32 accumulation
        .unpack_to_dest_mode = unpack_to_dest_modes,
        .math_approx_mode = use_bf16_mode,
        .compile_args = compute_compile_args,
        .defines = compute_defines
    }
);
```

**Key Points**:
1. ✅ `COMPUTE_KERNEL_PATH` is the TF32 kernel (decided at build time)
2. ✅ `fp32_dest_acc_en = true` enables FP32 accumulation in FPU
3. ✅ `unpack_to_dest_mode` configured for FP32 (when `--precision fp32`)

### Level 3: Kernel Dispatch Within TF32 File (Compile-Time)

```cpp
// In piecewise_fpu_tf32.cpp (lines 446-465)

for (uint32_t loop = 0; loop < compute_loop_factor; loop++) {
    #ifdef TRISC_MATH  // ← TRISC-MATH CPU executes this code

    // Compile-time dispatch based on polynomial degree
    if constexpr (poly_degree == 1) {
        sfpi::piecewise_poly_matmul_tile_tf32<1, num_segments, lut_size>(*p_lut);
    } else if constexpr (poly_degree == 2) {
        sfpi::piecewise_poly_matmul_tile_tf32<2, num_segments, lut_size>(*p_lut);
    } else if constexpr (poly_degree == 4) {
        sfpi::piecewise_poly_matmul_tile_tf32<4, num_segments, lut_size>(*p_lut);
        // ↑ This template gets instantiated for degree-4 polynomials
    }
    // ... up to degree 8

    #endif
}
```

**Key Points**:
1. ✅ `if constexpr` is **compile-time** - only ONE branch is compiled
2. ✅ Template instantiation happens at compile time based on `POLY_DEGREE`
3. ✅ TRISC-MATH CPU executes this code (not SFPU)

### Level 4: FPU Matmul Execution (Runtime, Hardware)

```cpp
// In piecewise_poly_matmul_tile_tf32() (lines 367-372)

// Convert FP32 matrices to TF32 and write to CBs
FPU_MATMUL_load_C_tf32(C);  // Converts FP32 → TF32, writes to CB 26
FPU_MATMUL_load_P_tf32(P);  // Converts FP32 → TF32, writes to CB 27

// Execute hardware matmul
FPU_MATMUL_compute_tf32();  // TF32 × TF32 → FP32 on FPU hardware
```

Inside `FPU_MATMUL_compute_tf32()` (lines 278-298):
```cpp
inline void FPU_MATMUL_compute_tf32() {
    // Wait for TF32 input tiles in CBs
    cb_wait_front(CB_SCRATCH_C, 1);  // CB 26 (TF32 coefficients)
    cb_wait_front(CB_SCRATCH_P, 1);  // CB 27 (TF32 powers)

    cb_reserve_back(CB_SCRATCH_Y, 1);  // CB 28 (FP32 results)

    // Call high-level matmul API (wraps LLK calls)
    matmul_tiles(CB_SCRATCH_C, CB_SCRATCH_P, 0, 0, 0);

    // Pack FP32 result to CB
    pack_tile(0, CB_SCRATCH_Y);
    cb_push_back(CB_SCRATCH_Y, 1);

    cb_pop_front(CB_SCRATCH_C, 1);
    cb_pop_front(CB_SCRATCH_P, 1);
}
```

**What Actually Happens in Hardware**:

The `matmul_tiles()` call dispatches to Low-Level Kernel (LLK) APIs:

1. **UNPACK**: `llk_unpack_AB_matmul(CB_SCRATCH_C, CB_SCRATCH_P, ...)`
   - Loads TF32 data from CB 26 → SrcB register
   - Loads TF32 data from CB 27 → SrcA register

2. **MATH**: `llk_math_matmul(dst_index, ...)`
   - FPU executes: **Dest = SrcB × SrcA** (TF32 × TF32 → FP32)
   - With `DST_ACCUM_MODE=1`, accumulation is in **FP32** (no rounding!)

3. **PACK**: `pack_tile(dst_index, CB_SCRATCH_Y)`
   - Writes FP32 result from Dest register → CB 28

---

## Critical Question: How Does FPU Know to Use TF32?

### The Actual Mechanism

**The FPU doesn't "know" it's using TF32 from a CB data format flag!**

Instead, the TF32 kernel **manually converts** FP32 → TF32 when writing to CBs:

```cpp
// In FPU_MATMUL_load_C_tf32() (lines 215-230)

inline void FPU_MATMUL_load_C_tf32(const float C[32][32]) {
    cb_reserve_back(CB_SCRATCH_C, 1);
    uint32_t cb_addr = get_write_ptr(CB_SCRATCH_C);

    // Cast CB memory as 32-bit words
    volatile tt_l1_ptr uint32_t* dst =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_addr);

    // Manually convert each element FP32 → TF32
    for (int r = 0; r < 32; r++) {
        for (int c = 0; c < 32; c++) {
            dst[r * 32 + c] = fp32_to_tf32(C[r][c]);  // ← Manual conversion!
        }
    }

    cb_push_back(CB_SCRATCH_C, 1);
}
```

**What's `fp32_to_tf32()`?** (lines 83-95):
```cpp
inline uint32_t fp32_to_tf32(float value) {
    uint32_t fp32_bits = *reinterpret_cast<const uint32_t*>(&value);

    // Extract FP32 components
    uint32_t sign = (fp32_bits >> 31) & 0x1;
    uint32_t exp  = (fp32_bits >> 23) & 0xFF;
    uint32_t mant = (fp32_bits >> 13) & 0x3FF;  // Upper 10 bits of mantissa

    // Pack into TF32 format (19 bits stored in 32-bit word)
    uint32_t tf32_bits = (sign << 18) | (exp << 10) | mant;

    return tf32_bits;  // 32-bit word with TF32 value in lower 19 bits
}
```

### The Key Insight

**CB data format doesn't matter for TF32!**

The CBs are just **raw memory buffers**. The TF32 kernel:
1. ✅ Writes TF32-formatted bits to CB memory (as 32-bit words)
2. ✅ FPU hardware reads those bits and interprets them as TF32
3. ✅ FPU configuration (via `DST_ACCUM_MODE=1`) determines accumulation precision

**No host-side CB configuration for TF32 is needed** - the kernel handles conversion internally.

---

## Comparison: BF16 vs TF32 Dispatch

### BF16 FPU Kernel (piecewise_fpu.cpp)

```cpp
inline void FPU_MATMUL_load_C(const float C[32][32]) {
    cb_reserve_back(CB_SCRATCH_C, 1);
    uint32_t cb_addr = get_write_ptr(CB_SCRATCH_C);

    // Cast as 16-bit words (BF16)
    volatile tt_l1_ptr uint16_t* dst =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_addr);

    // Convert FP32 → BF16 (upper 16 bits)
    for (int r = 0; r < 32; r++) {
        for (int c = 0; c < 32; c++) {
            uint32_t f32_bits = *reinterpret_cast<const uint32_t*>(&C[r][c]);
            dst[r * 32 + c] = static_cast<uint16_t>(f32_bits >> 16);  // BF16!
        }
    }

    cb_push_back(CB_SCRATCH_C, 1);
}
```

**FPU reads**: 16-bit BF16 values from CB → multiplies in BF16 → accumulates in FP32

### TF32 FPU Kernel (piecewise_fpu_tf32.cpp)

```cpp
inline void FPU_MATMUL_load_C_tf32(const float C[32][32]) {
    cb_reserve_back(CB_SCRATCH_C, 1);
    uint32_t cb_addr = get_write_ptr(CB_SCRATCH_C);

    // Cast as 32-bit words (TF32 stored in 32-bit)
    volatile tt_l1_ptr uint32_t* dst =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_addr);

    // Convert FP32 → TF32 (19 bits in 32-bit word)
    for (int r = 0; r < 32; r++) {
        for (int c = 0; c < 32; c++) {
            dst[r * 32 + c] = fp32_to_tf32(C[r][c]);  // TF32!
        }
    }

    cb_push_back(CB_SCRATCH_C, 1);
}
```

**FPU reads**: 32-bit words containing TF32 values → multiplies in TF32 → accumulates in FP32

---

## The Dispatch Decision Tree

```
┌─────────────────────────────────────────────────────┐
│ cmake -DKERNEL_VARIANT=???                          │
└────────────────┬────────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
    ↓                         ↓
KERNEL_VARIANT=              KERNEL_VARIANT=
piecewise_generic            piecewise_fpu_tf32
    │                         │
    ↓                         ↓
COMPUTE_KERNEL_PATH=         COMPUTE_KERNEL_PATH=
"...piecewise_generic.cpp"   "...piecewise_fpu_tf32.cpp"
    │                         │
    ↓                         ↓
SFPU Horner's method         FPU TF32 matmul method
    │                         │
    ↓                         ↓
┌───────────────┐         ┌────────────────────────┐
│ SFPU executes │         │ CPU builds matrices    │
│ vFloat ops    │         │ FPU executes matmul    │
│ BF16 or FP32  │         │ TF32 × TF32 → FP32    │
└───────────────┘         └────────────────────────┘
```

---

## Summary: The Dispatch Mechanism

### 1. Build-Time Selection ✅
**Where**: CMake command line
**How**: `-DKERNEL_VARIANT=piecewise_fpu_tf32`
**Result**: Selects which kernel file to compile

### 2. Compile-Time Template Instantiation ✅
**Where**: Inside TF32 kernel
**How**: `if constexpr (poly_degree == 4)`
**Result**: Only one template branch compiled for given degree

### 3. Runtime Execution ✅
**Where**: TRISC-MATH CPU on Tensix core
**How**: CPU code builds matrices, calls FPU via `matmul_tiles()`
**Result**: FPU hardware executes TF32×TF32→FP32 operation

### 4. Hardware Data Format ✅
**Where**: CB memory written by kernel
**How**: Manual FP32→TF32 conversion in kernel code
**Result**: FPU reads TF32-formatted bits from CB

---

## Why This Design?

### Advantages

1. ✅ **No host-side CB format configuration** - kernel handles conversion internally
2. ✅ **Simple build system** - just change `KERNEL_VARIANT` at compile time
3. ✅ **Explicit precision control** - conversion functions are visible in kernel code
4. ✅ **Portable** - same host program works with any kernel variant

### Disadvantages

1. ⚠️ **Manual conversion overhead** - CPU must convert FP32→TF32 for 2×1024 elements
2. ⚠️ **No CB format validation** - hardware doesn't know CB contains TF32 vs BF16
3. ⚠️ **More complex kernel code** - requires manual bit manipulation

---

## Key Takeaways

### Question: "How does dispatch know to use the TF32 flow?"

**Answer**:

1. **Build time**: You tell CMake to use TF32 kernel via `-DKERNEL_VARIANT=piecewise_fpu_tf32`
2. **Compile time**: Template instantiation selects degree-specific code path
3. **Runtime**: Kernel manually converts FP32 → TF32 when writing to CBs
4. **Hardware**: FPU reads TF32 bits from CB and executes TF32×TF32 multiply

**There is NO runtime dispatch** - the kernel variant is baked into the binary at compile time.

**There is NO CB data format configuration for TF32** - the kernel manually converts and writes raw bits.

---

## Practical Example

```bash
# Step 1: Build with TF32 kernel (compile-time selection)
cmake -B build \
  -DKERNEL_VARIANT=piecewise_fpu_tf32 \
  -DPOLY_DEGREE=4 \
  -DNUM_SEGMENTS=16 \
  -DLUT_SIZE=97

cmake --build build

# Step 2: Run (kernel is already selected, no runtime dispatch)
./build/bin/generic_lut_activation \
  coeffs.csv \
  --activation gelu \
  --precision fp32 \    # ← Enables FP32 output CB
  --range-min -10 \
  --range-max 10

# What happens:
# 1. Binary contains piecewise_fpu_tf32.cpp code (from build)
# 2. Host creates kernel with fp32_dest_acc_en=true
# 3. Kernel executes on TRISC-MATH CPU:
#    - Builds C[32×32] and P[32×32] in FP32
#    - Converts to TF32 via fp32_to_tf32()
#    - Writes TF32 bits to CB 26 and CB 27
#    - Calls matmul_tiles() → FPU executes TF32×TF32→FP32
#    - Reads FP32 result from CB 28
# 4. Output written as FP32 to CB_OUT
```

**The TF32 flow is completely determined at compile time, with manual format conversion at runtime!**
