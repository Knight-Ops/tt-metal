# AGENT_SFPU_PERF.md — SFPU Performance Optimization Guide

**Purpose**: A comprehensive checklist and reference for AI agents (or developers) optimizing SFPU kernel code on Tenstorrent hardware. Distilled from analysis of tt-metal SFPU implementations, tech reports, and the generic LUT activation project.

---

## Table of Contents

1. [Disassembly Workflow](#1-disassembly-workflow)
2. [Register Pressure Management](#2-register-pressure-management)
3. [Polynomial Evaluation Patterns](#3-polynomial-evaluation-patterns)
4. [Range Reduction Techniques](#4-range-reduction-techniques)
5. [Bit Manipulation Tricks](#5-bit-manipulation-tricks)
6. [Precision & Data Format](#6-precision--data-format)
7. [Branching & Conditionals](#7-branching--conditionals)
8. [Loop Unrolling & ILP](#8-loop-unrolling--ilp)
9. [Coefficient Loading](#9-coefficient-loading)
10. [SFPREPLAY Optimization](#10-sfpreplay-optimization)
11. [Operation Chaining](#11-operation-chaining)
12. [Performance Checklist](#12-performance-checklist)
13. [Ideas for LUT Implementation](#13-ideas-for-lut-implementation)
14. [TT-Metal Kernels That Could Benefit](#14-tt-metal-kernels-that-could-benefit)

---

## 1. Disassembly Workflow

### Using `disassemble.sh`

The disassembly script extracts SFPU instructions from compiled kernels to analyze:
- Total instruction count
- Horner step efficiency (inter-MAD spacing)
- Register pressure indicators

**Usage**:
```bash
# For LUT kernels (non-embedded)
./disassemble.sh <degree> <segments> <csv_file> [binary_args...]

# For any binary
./disassemble.sh --binary <binary_path> <csv_file> [binary_args...]
```

**Example**:
```bash
./disassemble.sh 8 16 gelu_16_8.csv --activation gelu --precision fp32
```

**Key Output Metrics**:
- `Total SFPU instrs`: Overall instruction count
- `Total sfpmad`: Number of multiply-accumulate operations
- `Inter-MAD gap distribution`: Ideally 1-2 instructions between MADs for tight Horner evaluation

### Interpreting Results

A well-optimized Horner loop shows:
```
Inter-MAD gap distribution (gap → count):
     1  ████████████████████████████████████████ 40
     2  ████████ 8
```

Gaps > 3 indicate:
- Unnecessary coefficient reloads
- Register spills
- Suboptimal instruction scheduling

### ELF Location

Compiled kernels are cached at:
```
~/.cache/tt-metal-cache/*/trisc1.elf
```

The script auto-finds the most recent `trisc1.elf` (SFPU math core).

### Objdump Command

```bash
OBJDUMP="$REPO/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
$OBJDUMP -D <path_to_trisc1.elf> > output.txt
```

---

## 2. Register Pressure Management

### SFPU Register Limits

| Resource | Wormhole/Blackhole | Grayskull |
|----------|-------------------|-----------|
| LRegs (general purpose) | 8 | 8 |
| Vector width | 32 elements | 64 elements |
| vConstFloatPrgm | 3 registers | 3 registers |

### Critical Rules

1. **No automatic spilling**: SFPU compiler does NOT spill registers. Exceeding limits causes:
   ```
   "cannot store SFPU register (register spill?) - exiting!"
   ```

2. **Minimize live variables**: Keep the number of simultaneously live `vFloat`/`vInt` variables below 8.

3. **Reload strategy**: Prefer recomputing or reloading values over keeping them live:
   ```cpp
   // BAD: Keeps 4 variables live
   vFloat a = ..., b = ..., c = ..., d = ...;
   result = a + b + c + d;

   // GOOD: Only 2 live at a time
   vFloat tmp = a + b;
   tmp = tmp + c;
   result = tmp + d;
   ```

4. **Use `vConstFloatPrgm` for frequently used constants**:
   ```cpp
   // Init function
   sfpi::vConstFloatPrgm0 = 0.69314718f;  // ln(2)
   sfpi::vConstFloatPrgm1 = 1.4426950f;   // 1/ln(2)

   // Compute function - no load instruction needed
   result = x * sfpi::vConstFloatPrgm1;
   ```

5. **Built-in constants** (zero-cost):
   - `sfpi::vConst0` (0.0f)
   - `sfpi::vConst1` (1.0f)

### Register Pressure Warning Signs

In disassembly, look for:
- Excessive `sfpstore` / `sfpload` patterns
- Functions named `*.constprop.isra.*` (GCC's failed constant propagation)
- Gaps > 4 between sfpmad instructions

### Mitigation Strategies

1. **Split complex functions**: Separate range reduction from polynomial evaluation
   ```cpp
   // Phase 1: Extract mantissa (few registers)
   inline vFloat cbrt_reduce_m(vFloat x) {
       return setexp(setsgn(x, 0), 127);
   }

   // Phase 2: Compute q, r (called after poly eval completes)
   inline void cbrt_compute_qr(vInt biased_e, vInt& q, vInt& r) { ... }
   ```

2. **Use `__attribute__((always_inline))`** on helper functions to let LTO merge them

3. **Force register reuse via scoping**:
   ```cpp
   { vFloat c = coeffs[5]; result = c; }  // c dies at scope end
   { vFloat c = coeffs[4]; result = result * x + c; }  // reuses register
   ```

---

## 3. Polynomial Evaluation Patterns

### Horner's Method (Standard)

The canonical approach for polynomial evaluation:
```cpp
// P(x) = c0 + c1*x + c2*x² + c3*x³
// Horner: ((c3*x + c2)*x + c1)*x + c0
vFloat result = coeffs[3];
result = result * x + coeffs[2];
result = result * x + coeffs[1];
result = result * x + coeffs[0];
```

**Cost**: N multiplies + N adds = N MADs for degree-N polynomial

### x²-Horner (Parity Polynomials)

For polynomials with known parity (only odd or only even coefficients):
```cpp
// Odd parity: P(x) = c1*x + c3*x³ + c5*x⁵
// = x * (c1 + c3*x² + c5*x⁴)
// = x * Horner([c1,c3,c5], x²)
vFloat x2 = x * x;
vFloat result = coeffs[5];
result = result * x2 + coeffs[3];
result = result * x2 + coeffs[1];
result = result * x;  // Final multiply by x

// Cost: ceil(N/2) MADs + 1 MUL vs N MADs — ~50% reduction
```

**Use cases**: tanh (odd), cosh (even), sin (odd), cos (even)

### Dual Evaluation (ILP)

Process two independent x values simultaneously to hide 2-cycle MAD latency:
```cpp
// Interleaved chains exploit instruction-level parallelism
{ vFloat c = coeffs[5]; r1 = c; r2 = c; }
{ vFloat c = coeffs[4]; r1 = r1 * x1 + c; r2 = r2 * x2 + c; }
{ vFloat c = coeffs[3]; r1 = r1 * x1 + c; r2 = r2 * x2 + c; }
// ...
```

**Benefits**:
- Hides MAD latency (2 cycles)
- Shares coefficient loads between chains
- ~40-60% throughput improvement on suitable workloads

**Constraints**:
- Requires 2x DST registers processed per iteration
- Increases register pressure — may not combine with range reduction

### PolynomialEvaluator::eval

tt-metal provides a variadic template helper:
```cpp
sfpi::vFloat p = PolynomialEvaluator::eval(
    x,
    c0, c1, c2, c3, c4, c5, c6, c7  // ascending order
);
```

---

## 4. Range Reduction Techniques

### Purpose

Map arbitrary input to a narrow interval where polynomial approximation is accurate, then reconstruct the full result.

### Cody-Waite Range Reduction

**Key idea**: Split constants into high + low parts for extended precision:
```cpp
// Standard: r = x - k * ln(2)  — loses precision in subtraction
// Cody-Waite: r = (x - k*LN2_HI) - k*LN2_LO — maintains precision

constexpr float NEG_LN2_HI = -0.6931152343750000f;
constexpr float NEG_LN2_LO = -3.19461832987e-05f;
vFloat r_hi = k * NEG_LN2_HI + x;  // SFPMAD: k*(-ln2_hi) + x
vFloat r = k * NEG_LN2_LO + r_hi;  // SFPMAD: k*(-ln2_lo) + r_hi
```

**Used by**: exp, sin, cos, tan

### Magic Number Rounding

Branch-free float→int conversion:
```cpp
const vFloat magic = Converter::as_float(0x4B400000U);  // 2^23 + 2^22
vFloat tmp = z + magic;
vFloat k = tmp - magic;                    // rounded float
vInt k_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(magic);  // int
```

**Advantage**: Avoids branching, single instruction sequence

### Exponent/Mantissa Extraction (log, cbrt)

```cpp
// x = 2^e × m where m ∈ [1, 2)
vInt biased_exp = exexp_nodebias(x);      // e + 127
vInt e = biased_exp - 127;                // debiased exponent
vFloat m = setexp(x, 127);                // normalized mantissa
// log(x) = e * ln(2) + log(m)
```

### Sign-Magnitude vs Two's Complement

**CRITICAL**: SFPU `int32_to_float` expects **sign-magnitude**, not two's complement:
```cpp
// Two's complement -5: 0xFFFFFFFB
// Sign-magnitude -5:   0x80000005

v_if(exp < 0) {
    exp = setsgn(~exp + 1, 1);  // Convert to sign-magnitude
}
v_endif;
vFloat expf = int32_to_float(exp, 0);
```

---

## 5. Bit Manipulation Tricks

### Zero-Cost Type Punning

```cpp
// View float bits as int (no conversion)
vInt bits = reinterpret<vInt>(float_val);

// View int bits as float
vFloat f = reinterpret<vFloat>(int_val);
```

### Exponent Operations

```cpp
// Extract debiased exponent: e = floor(log2(|x|))
vInt exp = exexp(x);  // Returns e where x = 2^e × m

// Extract biased exponent (no sign-magnitude issues)
vInt biased = exexp_nodebias(x);

// Extract mantissa (implicit bit included)
vInt man = exman8(x);  // 8-bit mantissa with implicit 1

// Set exponent (normalize to [1, 2))
vFloat normalized = setexp(x, 127);  // Sets biased exponent to 127

// Add to exponent (multiply by power of 2)
vFloat scaled = addexp(x, -23);  // x *= 2^(-23)
```

### Sign Manipulation

```cpp
// Set sign bit
vFloat neg = setsgn(x, 1);   // Force negative
vFloat pos = setsgn(x, 0);   // Force positive

// Copy sign from another value
vFloat result = setsgn(abs_result, original_input);

// Extract sign bit
vInt sign_bit = reinterpret<vInt>(x) & 0x80000000;
```

### Exponent Manipulation for ldexp

```cpp
// result = x * 2^k via exponent addition
vInt p_exp = exexp_nodebias(x);
vInt new_exp = p_exp + k;
vFloat result = setexp(x, new_exp);
```

---

## 6. Precision & Data Format

### Accumulation Precision

For high-precision operations, use FP32 accumulators:
```cpp
template <bool is_fp32_dest_acc_en>
void calculate_operation() {
    if constexpr (is_fp32_dest_acc_en) {
        // Use accurate algorithm
    } else {
        // Use faster BF16-optimized algorithm
    }
}
```

### Explicit BF16 Rounding

Avoid implicit truncation — use round-to-nearest-even:
```cpp
if constexpr (!is_fp32_dest_acc_en) {
    result = reinterpret<vFloat>(float_to_fp16b(result, 0));
}
```

### Selecting Algorithm by Precision

```cpp
template <bool is_fp32_dest_acc_en>
sfpi_inline vFloat _sfpu_sigmoid_(vFloat x) {
    vFloat exp_neg_x;
    if constexpr (is_fp32_dest_acc_en) {
        exp_neg_x = _sfpu_exp_f32_accurate_(-x);  // Cody-Waite, Taylor
        return _sfpu_reciprocal_<2>(exp_neg_x + 1.0f);  // 2 NR iterations
    } else {
        exp_neg_x = _sfpu_exp_21f_<true>(-x);  // Fast approximation
        return _sfpu_reciprocal_<1>(exp_neg_x + 1.0f);  // 1 NR iteration
    }
}
```

---

## 7. Branching & Conditionals

### Vector Predicates

```cpp
v_if(x >= threshold) {
    // Executed on lanes where condition is true
    result = large_x_path(x);
}
v_elseif(x < 0.0f) {
    result = negative_path(x);
}
v_else {
    result = default_path(x);
}
v_endif;
```

### Comparison Costs

| Operation | Cost |
|-----------|------|
| `<`, `!=`, `>=` | 1 instruction |
| `<=`, `>` | 2 instructions (inverted + extra) |

**Prefer `>=` over `>` when possible**.

### Branchless Alternatives

For simple cases, use `vec_min_max`:
```cpp
// Clamp x to [lo, hi] — branchless
sfpi::vec_min_max(lo, x);    // x = max(lo, x)
sfpi::vec_min_max(x, hi);    // x = min(x, hi)
```

### Sign-Based Selection

```cpp
// result = (x >= 0) ? pos_val : neg_val
result = setsgn(abs_result, x);  // Copies sign from x
```

---

## 8. Loop Unrolling & ILP

### Compiler Directive

```cpp
#pragma GCC unroll 8
for (int d = 0; d < ITERATIONS; d++) {
    // Loop body unrolled 8 times
}
```

### Manual Unrolling via Templates

```cpp
template <uint32_t SEG, uint32_t TOTAL>
__attribute__((always_inline))
inline void unroll_segment(const float* lut, vFloat x, vFloat& result) {
    if constexpr (SEG < TOTAL) {
        v_if(x >= lut[SEG]) {
            result = eval_polynomial<DEGREE>(&coeffs[SEG], x);
        }
        v_endif;
        unroll_segment<SEG + 1, TOTAL>(lut, x, result);  // Tail recursion
    }
}
```

### Processing Multiple DST Registers

Standard tile = 32×32 elements = 32 DST registers of 32 elements each.

```cpp
// Single evaluation (baseline)
for (int d = 0; d < 32; d++) {
    vFloat x = dst_reg[d];
    dst_reg[d] = process(x);
}

// Dual evaluation (ILP exploitation)
for (int d = 0; d < 32; d += 2) {
    vFloat x1 = dst_reg[d], x2 = dst_reg[d+1];
    vFloat r1, r2;
    process_dual(x1, x2, r1, r2);  // Interleaved ops
    dst_reg[d] = r1;
    dst_reg[d+1] = r2;
}
```

---

## 9. Coefficient Loading

### Immediate Loading (LRegs)

For LUT-based approximations with small coefficient count:
```cpp
// Init
uint imm0 = 0x1DFF;  // Packed 16-bit immediate
_sfpu_load_imm16_(0, imm0);  // Loads to LReg0

// Use
sfpi::vUInt l0 = l_reg[sfpi::LRegs::LReg0];
result = sfpi::lut(val, l0, l1, l2);  // Hardware LUT instruction
```

### Programmable Constants

```cpp
// Init (once per tile batch)
void init() {
    sfpi::vConstFloatPrgm0 = 0.69314718f;
    sfpi::vConstFloatPrgm1 = 1.4426950408889634f;
    sfpi::vConstFloatPrgm2 = 0.5f;
}

// Use (zero-cost load)
result = x * sfpi::vConstFloatPrgm1;
```

### From L1 Memory

```cpp
// Pointer to L1 circular buffer
const float* coeffs = get_pointer_to_cb_data<float[]>(cb_lut, 0);

// Direct array access (compiler optimizes to sfploadi pairs)
result = coeffs[5] * x + coeffs[4];
```

### Coefficient Sharing in Dual Eval

```cpp
// Hoist coefficient load, use for both chains
{ vFloat c = coeffs[5]; r1 = c; r2 = c; }
{ vFloat c = coeffs[4]; r1 = r1 * x1 + c; r2 = r2 * x2 + c; }
```

This halves coefficient loads compared to separate loops.

---

## 10. SFPREPLAY Optimization

### What It Does

SFPREPLAY allows RISC-V to submit up to 32 SFPU instructions atomically, improving throughput for repetitive patterns.

### Requirements

- Main loop body < 32 instructions
- Repetitive instruction sequence
- Available on Wormhole/Blackhole (not Grayskull)

### How to Enable

The compiler auto-detects suitable patterns. To maximize benefit:
1. Unroll loops to expose repetitive sequences
2. Keep loop bodies simple and uniform
3. Avoid conditionals inside tight loops

### Disabling (Debugging)

```bash
# Compiler flag
-mno-tt-tensix-optimize-replay
```

---

## 11. Operation Chaining

### Concept

Keep intermediate results in DST registers between operations:
```cpp
// Bad: Each op reads/writes L1
result1 = exp(x);
cb_write(result1);
cb_read(input2);
result2 = log(input2 + 1);

// Good: Chain in registers
exp_tile(cb_in, cb_tmp);  // exp(x) stays in DST
add_scalar(1.0f);         // DST += 1, no memory access
log_tile(cb_tmp, cb_out); // log(DST)
```

### Benefits

- Eliminates intermediate memory transfers
- Reduces L1 bandwidth usage
- Lower latency for compound functions (softplus, gelu, etc.)

### Example: softplus(x) = log(1 + exp(x))

```cpp
// Initialize all operations
exp_tile_init();
log_tile_init();

for (uint32_t tile = 0; tile < n_tiles; tile++) {
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);

    // Chain: exp → +1 → log
    exp_tile(0);
    add_scalar_tile(0, 1.0f);  // Assumes this API exists
    log_tile(0);

    pack_tile(0, cb_out);
    tile_regs_release();
}
```

---

## 12. Performance Checklist

### Pre-Implementation

- [ ] Identify mathematical properties (symmetry, parity, bounded output)
- [ ] Choose appropriate range reduction for the function
- [ ] Determine target precision (BF16 vs FP32)
- [ ] Estimate register pressure from algorithm design

### During Implementation

- [ ] Use Horner's method for all polynomial evaluations
- [ ] Exploit parity with x²-Horner when applicable
- [ ] Use `vConstFloatPrgm` for frequently used constants
- [ ] Prefer `>=` over `>` in comparisons
- [ ] Add `#pragma GCC unroll 8` to inner loops
- [ ] Keep live variable count < 8
- [ ] Use `__attribute__((always_inline))` on helpers
- [ ] Handle sign-magnitude format for negative integers

### Post-Implementation

- [ ] Run `disassemble.sh` and check:
  - [ ] Total SFPU instruction count
  - [ ] Inter-MAD gap distribution (target: mostly 1-2)
  - [ ] No `*.constprop.isra.*` symbols (register spill indicator)
- [ ] Verify accuracy against reference implementation
- [ ] Benchmark on target hardware
- [ ] Compare with native SFPU implementation (if exists)

### Register Pressure Red Flags

- [ ] Functions with > 4 intermediate vFloat/vInt variables
- [ ] Range reduction + high-degree polynomial in same scope
- [ ] Nested v_if with many branches
- [ ] Dual-eval combined with complex reconstruction

---

## 13. Ideas for LUT Implementation

### Current Implementation Strengths

The generic LUT activation already implements many best practices:
- Horner's method with compile-time unrolling
- x²-Horner for parity polynomials (`POLY_PARITY_ODD`/`POLY_PARITY_EVEN`)
- Dual evaluation for ILP
- Cody-Waite range reduction for exp/trig
- Adaptive per-segment degrees (`SEGMENT_DEGREES[]`)
- Template-recursive unrolling to avoid register spills

### Potential Improvements

1. **Coefficient Compression**:
   - Pack consecutive coefficients with similar magnitude into shared exponent format
   - Use BF16 coefficients where precision permits (halves load count)

2. **Segment Boundary Search**:
   - Current: Linear cascade of v_if statements (O(N) for N segments)
   - Possible: Binary search via bit manipulation for large segment counts
   ```cpp
   // For 16 segments, 4 comparisons vs 15
   vInt idx = 0;
   v_if(x >= lut[8]) { idx = idx | 8; } v_endif;
   v_if(x >= lut[idx | 4]) { idx = idx | 4; } v_endif;
   // ...
   ```

3. **Adaptive Dual-Eval**:
   - Currently disabled when range reduction is active
   - Could interleave range reduction too if register allocation is careful

4. **LUT Instruction for Simple Cases**:
   - Hardware `sfpi::lut()` instruction for 3-coefficient piecewise linear
   - Could use for first approximation, then refine with polynomial

5. **Precomputed x² for Consecutive Tiles**:
   - If same x values appear in multiple tiles, cache x² computation

### Benchmarking Priorities

1. Compare embedded LUT vs CB-loaded LUT overhead
2. Measure dual-eval speedup per degree
3. Profile segment count scaling (8 vs 16 vs 32 vs 64)
4. Validate range reduction overhead vs accuracy gain

---

## 14. TT-Metal Kernels That Could Benefit

Based on analysis of existing SFPU implementations, these kernels could potentially benefit from the optimization patterns documented here:

### High-Priority Candidates

| Kernel | Current Approach | Potential Improvement |
|--------|-----------------|----------------------|
| `ckernel_sfpu_gelu.h` | 15th-order Chebyshev polynomial | LUT-based piecewise polynomial with adaptive degree |
| `ckernel_sfpu_tanh.h` | Polynomial + sigmoid fallback | x²-Horner for odd parity |
| `ckernel_sfpu_sigmoid.h` | exp + reciprocal composition | Direct LUT approximation |
| `ckernel_sfpu_silu.h` | x * sigmoid(x) | Fused piecewise approximation |
| `ckernel_sfpu_softplus.h` | log(1 + exp(x)) chain | Single piecewise polynomial |

### Medium-Priority Candidates

| Kernel | Current Approach | Potential Improvement |
|--------|-----------------|----------------------|
| `ckernel_sfpu_exp.h` | Multiple algorithm variants | Unified LUT with range reduction |
| `ckernel_sfpu_log.h` | Minimax polynomial | Piecewise for wider range accuracy |
| `ckernel_sfpu_elu.h` | Conditional exp | Fused piecewise |
| `ckernel_sfpu_celu.h` | Similar to ELU | Same as ELU |
| `ckernel_sfpu_hardswish.h` | Piecewise linear | Verify optimal implementation |

### Specific Improvement Opportunities

#### GELU (`ckernel_sfpu_gelu.h`)

Current: 15th-order Chebyshev polynomial
```cpp
result = POLYVAL15(c15, c14, ..., c0, x);  // 15 MADs
```

Opportunity:
- GELU is bounded and smooth — ideal for piecewise approximation
- 8-segment degree-4 polynomial: 4 MADs × 8 segments = ~8 MADs average
- Could achieve same accuracy with ~50% fewer operations

#### tanh (`ckernel_sfpu_tanh.h`)

Current: 6th-order polynomial with Sollya coefficients
```cpp
result = PolynomialEvaluator::eval(val, c0, c1, c2, c3, c4, c5, c6);  // 6 MADs
```

Opportunity:
- tanh is odd function: only odd coefficients needed
- x²-Horner: `x * Horner([c1,c3,c5], x²)` = 3 MADs + 1 MUL
- Could also use `sfpi::lut()` hardware instruction for fast approximation path

#### Bessel Functions (`ckernel_sfpu_i0.h`, `ckernel_sfpu_i1.h`)

These are computationally expensive and could benefit significantly from:
- Piecewise polynomial approximation
- Asymptotic expansion for large arguments
- Careful range reduction

---

## Appendix: Quick Reference

### SFPU Instruction Latencies (Wormhole/Blackhole)

| Instruction | Latency (cycles) |
|-------------|-----------------|
| sfpmad | 2 |
| sfpmul | 2 |
| sfpadd | 1 |
| sfpload/sfploadi | 1 |
| sfpstore | 1 |
| sfpsetcc | 1 |

### Common Constants

| Constant | Hex (FP32) | Value |
|----------|-----------|-------|
| ln(2) | 0x3F317218 | 0.69314718... |
| 1/ln(2) | 0x3FB8AA3B | 1.44269504... |
| π | 0x40490FDB | 3.14159265... |
| 2/π | 0x3F22F983 | 0.63661977... |
| Magic (rounding) | 0x4B400000 | 2^23 + 2^22 |

### Useful SFPI Functions

```cpp
// Type reinterpretation
reinterpret<vFloat>(vInt_val)
reinterpret<vInt>(vFloat_val)

// Exponent/mantissa
exexp(x)           // Debiased exponent (sign-magnitude for negative)
exexp_nodebias(x)  // Biased exponent (always positive)
exman8(x)          // 8-bit mantissa with implicit 1
exman9(x)          // 9-bit mantissa
setexp(x, e)       // Set exponent field
addexp(x, delta)   // Add to exponent (multiply by 2^delta)

// Sign manipulation
setsgn(x, sign)    // Set sign bit (0=positive, 1=negative)
abs(x)             // Absolute value

// Conversion
int32_to_float(i, 0)        // Sign-magnitude int → float
float_to_fp16b(f, 0)        // FP32 → BF16 with rounding
Converter::as_float(bits)   // Bit pattern → float
```

---

*Last updated: 2025*
*Source: Analysis of tt-metal SFPU implementations and generic_lut_activation project*
