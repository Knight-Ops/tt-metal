// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

// Include reciprocal function for tan range reduction (tan_expand needs -1/poly)
#if defined(RANGE_REDUCTION_TAN) && defined(TRISC_MATH)
#include "ckernel_sfpu_recip.h"
#endif

namespace kutil = norm::kernel_util;

/**
 * Generic Piecewise Polynomial LUT Activation
 *
 * This kernel supports ANY polynomial degree (0-16) and ANY depth (4-64 segments).
 * Each segment is approximated as: y = c0 + c1*x + c2*x² + ... + cn*x^n
 * Evaluated using Horner's scheme for numerical stability.
 *
 * Template parameters (compile-time):
 *   POLY_DEGREE   - Polynomial degree (0=constant, 1=linear, 2=quadratic, ... 16=hexadecic)
 *   NUM_SEGMENTS  - Number of segments (4, 8, 16, 32, 64, etc.)
 *   LUT_SIZE      - Total LUT size = (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)
 *
 * LUT Format: [b0, b1, ..., bN, c0_seg0, c1_seg0, ..., cn_seg0, c0_seg1, ...]
 * where bi are boundary points and cj are polynomial coefficients
 *
 * Benefits:
 *   - Single implementation for all degree/depth combinations
 *   - Zero runtime overhead (all branching at compile-time)
 *   - Automatic support for new configurations
 *   - Reduced code duplication (7 files → 1 file)
 *
 * Supports both modes:
 *   - EMBEDDED_LUT: Coefficients compiled directly into kernel (zero L1 overhead)
 *   - CB mode: Coefficients loaded from L1 circular buffer (runtime flexibility)
 */

#ifdef TRISC_MATH

// Range reduction: Cody-Waite method for exp/trig, mantissa/exponent extraction for log/cbrt
// Only included when RANGE_REDUCTION_* or ASYMPTOTIC_FACTOR_* is defined
#if defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_TRIG) || defined(RANGE_REDUCTION_LOG) || defined(RANGE_REDUCTION_TAN) || defined(RANGE_REDUCTION_CBRT) \
    || defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR)
#include "sfpu/ckernel_sfpu_converter.h"
#endif

namespace sfpi {

#ifdef RANGE_REDUCTION_EXP
// Cody-Waite range reduction: x → (s, k_int) where exp(x) = 2^k * exp(s)
// s ∈ [-ln(2)/2, ln(2)/2] ≈ [-0.347, 0.347]
inline void exp_reduce(vFloat x, vFloat& s, vInt& k_int) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    vFloat z = x * INV_LN2;

    // Round z to nearest integer (branch-free magic-number technique from ckernel_sfpu_exp.h)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat k = tmp - c231;
    k_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);

    // Cody-Waite extended precision: s = x - k*ln2
    // Using negated constants so compiler can emit single SFPMAD instructions
    constexpr float NEG_LN2_HI = -0.6931152343750000f;
    constexpr float NEG_LN2_LO = -3.19461832987e-05f;
    vFloat s_hi = k * NEG_LN2_HI + x;
    s = k * NEG_LN2_LO + s_hi;
}

// Reconstruct: result = poly_result * 2^k via exponent bit manipulation
inline vFloat exp_expand(vFloat poly_result, vInt k_int) {
    vInt p_exp = exexp_nodebias(poly_result);
    vInt new_exp = p_exp + k_int;
    return setexp(poly_result, new_exp);
}
#endif

#ifdef RANGE_REDUCTION_TRIG
// Trig range reduction: x → (s, q_int) where f(x) = (-1)^q * f(s)
// s ∈ [-π/2, π/2], works for both sin and cos
inline void trig_reduce(vFloat x, vFloat& s, vInt& q_int) {
    constexpr float FRAC_1_PI = 0.31830988618379067f;
    vFloat z = x * FRAC_1_PI;

    // Round z to nearest integer (branch-free magic-number technique)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat q = tmp - c231;
    q_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);

    // Cody-Waite: s = x - q*π with extended precision
    constexpr float NEG_PI_HI = -3.140625f;
    constexpr float NEG_PI_LO = -0.00096765358979323846f;
    vFloat s_hi = q * NEG_PI_HI + x;
    s = q * NEG_PI_LO + s_hi;
}

// Reconstruct: sin(x + kπ) = (-1)^k * sin(x), cos(x + kπ) = (-1)^k * cos(x)
inline vFloat trig_expand(vFloat poly_result, vInt q_int) {
    v_if(q_int & 1) {
        poly_result = -poly_result;
    }
    v_endif;
    return poly_result;
}
#endif

#ifdef RANGE_REDUCTION_TAN
// Tan range reduction: x → (a, j_int) where a ∈ [-π/4, π/4]
// Uses Cody-Waite reduction matching TTNN's sfpu_tan implementation
// j = round(x / (π/2)), a = x - j * (π/2)
inline void tan_reduce(vFloat x, vFloat& a, vInt& j_int) {
    constexpr float FRAC_2_PI = 0.6366197723675814f;  // 2/π
    vFloat z = x * FRAC_2_PI;

    // Round z to nearest integer (branch-free magic-number technique)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat j = tmp - c231;
    j_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);

    // Cody-Waite: a = x - j*(π/2) with extended precision
    // Split π/2 into high and low parts for accuracy
    constexpr float NEG_PI_2_HI = -1.5703125f;           // -π/2 high bits (fits in bf16)
    constexpr float NEG_PI_2_LO = -0.0004837512969970703f;  // -π/2 remainder
    vFloat a_hi = j * NEG_PI_2_HI + x;
    a = j * NEG_PI_2_LO + a_hi;
}

// Reconstruct: tan(x) where x = a + j*(π/2)
// j even: tan(x) = tan(a) = poly(a)
// j odd:  tan(x) = -cot(a) = -1/tan(a) = -1/poly(a)
inline vFloat tan_expand(vFloat poly_result, vInt j_int) {
    v_if(j_int & 1) {
        // j is odd: result = -1/poly(a)
        // Use 3 Newton-Raphson iterations for better precision (vs 2 in sfpu_reciprocal<false>)
        poly_result = -ckernel::sfpu::sfpu_reciprocal_iter<3>(poly_result);
    }
    v_endif;
    return poly_result;
}
#endif

#if defined(RANGE_REDUCTION_LOG)
// Log range reduction: x → (m, e_int) where x = 2^e * m, m ∈ [1, 2)
// log(x) = e*ln(2) + log(m)
// Uses IEEE 754 bit manipulation: exponent extraction and mantissa normalization
inline void log_reduce(vFloat x, vFloat& m, vInt& e_int) {
    // Extract biased exponent: for x = 2^e * m, biased_exp = e + 127
    vInt biased_exp = exexp_nodebias(x);

    // Unbias: e = biased_exp - 127
    // But we want m in [1, 2), so we set exponent to 127 (bias)
    // This gives m = x / 2^e = x * 2^(-e) = x with exponent = 127
    e_int = biased_exp - 127;

    // Normalize mantissa to [1, 2) by setting exponent bits to 127
    m = setexp(x, 127);
}

// Reconstruct: log(x) = e*ln(2) + log(m)
// poly_result = log(m) where m ∈ [1, 2)
// Final result = e*ln(2) + poly_result
inline vFloat log_expand(vFloat poly_result, vInt e_int) {
    // LOG_EXPAND_CONSTANT is base-specific: ln(2) for log, 1.0 for log2, log10(2) for log10
#ifndef LOG_EXPAND_CONSTANT
    #define LOG_EXPAND_CONSTANT 0.6931471805599453f
#endif
    constexpr float EXPAND_C = LOG_EXPAND_CONSTANT;
    // int32_to_float expects SIGN-MAGNITUDE format, not two's complement.
    // Negative exponents must be converted: twos complement → sign-magnitude.
    v_if(e_int < 0) {
        e_int = setsgn(~e_int + 1, 1);
    }
    v_endif;
    vFloat e_float = int32_to_float(e_int, RoundMode::Nearest);
    return e_float * EXPAND_C + poly_result;
}
#endif

#ifdef RANGE_REDUCTION_CBRT
// Cbrt range reduction: x → (m, q, r, sign) where cbrt(x) = sign * C[r] * cbrt(m) * 2^q
//
// For x = sign(x) * |x| where |x| = m * 2^e with m ∈ [1, 2):
//   e = 3*q + r  where r ∈ {0, 1, 2}
//   cbrt(|x|) = cbrt(m * 2^r) * 2^q = cbrt(m) * cbrt(2^r) * 2^q
//   cbrt(x) = sign(x) * C[r] * cbrt(m) * 2^q
//
// where C = [1.0, cbrt(2), cbrt(4)]
//
// Split into cbrt_reduce_m (trivial, 2 ops) + cbrt_reduce_qrs (heavier)
// to avoid SFPU register spills. cbrt_reduce's many inlined temporaries
// exceed the 8 LREG limit when combined with polynomial evaluation variables.

// Scale constants: C[r] = cbrt(2^r) for r = 0, 1, 2
constexpr float CBRT_SCALE_C0 = 1.0f;
constexpr float CBRT_SCALE_C1 = 1.2599210498948732f;  // cbrt(2)
constexpr float CBRT_SCALE_C2 = 1.5874010519681994f;  // cbrt(4)

// Phase 1: Extract mantissa m ∈ [1, 2) — trivial, no register pressure
inline vFloat cbrt_reduce_m(vFloat x) {
    return setexp(setsgn(x, 0), 127);
}

// Reconstruct: cbrt(x) = sign * C[r] * poly(m) * 2^q
inline vFloat cbrt_expand(vFloat poly_result, vInt q, vInt r, vInt sign_bits) {
    // Look up scale factor C[r] = cbrt(2^r)
    vFloat scale = CBRT_SCALE_C0;  // default r=0
    v_if(r == 1) { scale = CBRT_SCALE_C1; } v_endif;
    v_if(r == 2) { scale = CBRT_SCALE_C2; } v_endif;

    // result = C[r] * poly(m)
    vFloat result = scale * poly_result;

    // Multiply by 2^q via exponent addition
    vInt result_exp = exexp_nodebias(result);
    result = setexp(result, result_exp + q);

    // Apply original sign (cbrt is an odd function)
    vInt result_bits = reinterpret<vInt>(result);
    result = reinterpret<vFloat>(result_bits | sign_bits);

    return result;
}
#endif

// ============================================================================
// Asymptotic factoring: inline Cody-Waite exp for dominant factor computation
// Used when correction polynomials are multiplied by exp-based dominant factors
// to handle extreme dynamic range in function tails (e.g., GELU, sigmoid).
// Mutually exclusive with range reduction (never combined).
// ============================================================================
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR)

// Inline Cody-Waite exp(arg) — no overflow/underflow checks needed because
// the asymptotic region has known bounded input range.
// Degree-5 Taylor for exp(r), |r| < ln(2)/2: relative error < 3.3e-9.
inline vFloat asymptotic_exp(vFloat arg) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    vFloat z = arg * INV_LN2;

    // Round to nearest integer (branch-free magic-number technique)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat k = tmp - c231;
    vInt k_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);

    // Cody-Waite extended precision: r = arg - k*ln2
    constexpr float NEG_LN2_HI = -0.6931152343750000f;
    constexpr float NEG_LN2_LO = -3.19461832987e-05f;
    vFloat r = k * NEG_LN2_HI + arg;
    r = k * NEG_LN2_LO + r;

    // Degree-5 Taylor for exp(r)
    vFloat p = 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;

    // Scale by 2^k via exponent bit manipulation
    vInt p_exp = exexp_nodebias(p);
    return setexp(p, p_exp + k_int);
}

#endif // ASYMPTOTIC_FACTOR_EXP_*

// Generic polynomial evaluation using Horner's method with compile-time optimization
template <uint32_t DEGREE>
inline vFloat eval_polynomial(const float* coeffs, vFloat x) {
    if constexpr (DEGREE == 0) {
        // Constant: y = c0
        return coeffs[0];
    } else if constexpr (DEGREE == 1) {
        // Linear: y = c0 + c1*x
        return coeffs[0] + coeffs[1] * x;
    } else if constexpr (DEGREE == 2) {
        // Quadratic: y = c0 + c1*x + c2*x²
        // Horner: (c2*x + c1)*x + c0
        return (coeffs[2] * x + coeffs[1]) * x + coeffs[0];
    } else if constexpr (DEGREE == 3) {
        // Cubic: y = c0 + c1*x + c2*x² + c3*x³
        // Horner: ((c3*x + c2)*x + c1)*x + c0
        return ((coeffs[3] * x + coeffs[2]) * x + coeffs[1]) * x + coeffs[0];
    } else if constexpr (DEGREE == 4) {
        // Quartic: y = c0 + c1*x + c2*x² + c3*x³ + c4*x⁴
        // Horner: ((((c4*x + c3)*x + c2)*x + c1)*x + c0
        vFloat result = coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 5) {
        // Quintic: y = c0 + ... + c5*x⁵
        vFloat result = coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 6) {
        // Hexic: y = c0 + ... + c6*x⁶
        vFloat result = coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 7) {
        // Septic: y = c0 + ... + c7*x⁷
        vFloat result = coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 8) {
        // Octic: y = c0 + ... + c8*x⁸
        vFloat result = coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 9) {
        // Nonic: y = c0 + ... + c9*x⁹
        vFloat result = coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 10) {
        // Decic: y = c0 + ... + c10*x¹⁰
        vFloat result = coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 11) {
        // Undecic: y = c0 + ... + c11*x¹¹
        vFloat result = coeffs[11];
        result = result * x + coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 12) {
        // Dodecic: y = c0 + ... + c12*x¹²
        vFloat result = coeffs[12];
        result = result * x + coeffs[11];
        result = result * x + coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 13) {
        // Tridecic: y = c0 + ... + c13*x¹³
        vFloat result = coeffs[13];
        result = result * x + coeffs[12];
        result = result * x + coeffs[11];
        result = result * x + coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 14) {
        // Tetradecic: y = c0 + ... + c14*x¹⁴
        vFloat result = coeffs[14];
        result = result * x + coeffs[13];
        result = result * x + coeffs[12];
        result = result * x + coeffs[11];
        result = result * x + coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 15) {
        // Pentadecic: y = c0 + ... + c15*x¹⁵
        vFloat result = coeffs[15];
        result = result * x + coeffs[14];
        result = result * x + coeffs[13];
        result = result * x + coeffs[12];
        result = result * x + coeffs[11];
        result = result * x + coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 16) {
        // Hexadecic: y = c0 + ... + c16*x¹⁶
        vFloat result = coeffs[16];
        result = result * x + coeffs[15];
        result = result * x + coeffs[14];
        result = result * x + coeffs[13];
        result = result * x + coeffs[12];
        result = result * x + coeffs[11];
        result = result * x + coeffs[10];
        result = result * x + coeffs[9];
        result = result * x + coeffs[8];
        result = result * x + coeffs[7];
        result = result * x + coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 32) {
        // Duotrigesic: y = c0 + ... + c32*x³²
        vFloat result = coeffs[32];
        for (int i = 31; i >= 0; i--) {
            result = result * x + coeffs[i];
        }
        return result;
    }
}

// x²-Horner evaluation for polynomials with parity structure.
// Halves Horner step count by evaluating in x² basis with stride-2 coefficient access.
//
// Odd parity (c0=c2=c4=...=0):  P(x) = x * Horner([c1,c3,c5,...], x²)
// Even parity (c1=c3=c5=...=0): P(x) = Horner([c0,c2,c4,...], x²)
//
// DEGREE is the nominal (max) degree — the LUT stores DEGREE+1 coefficients per segment,
// but zero-valued coefficients are skipped via stride-2 access.
#if defined(POLY_PARITY_ODD)
template <uint32_t DEGREE>
inline vFloat eval_polynomial_parity(const float* coeffs, vFloat x, vFloat x2) {
    // Odd: only odd-index coefficients are nonzero (c1,c3,c5,...)
    // TOP = highest odd index ≤ DEGREE
    constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = (TOP - 1) / 2;  // number of FMA steps after init
    vFloat result = coeffs[TOP];
#pragma GCC unroll 16
    for (int k = 1; k <= STEPS; k++)
        result = result * x2 + coeffs[TOP - 2 * k];
    return result * x;  // final *x for odd parity
}
#elif defined(POLY_PARITY_EVEN)
template <uint32_t DEGREE>
inline vFloat eval_polynomial_parity(const float* coeffs, vFloat x, vFloat x2) {
    // Even: only even-index coefficients are nonzero (c0,c2,c4,...)
    // TOP = highest even index ≤ DEGREE
    constexpr int TOP = (DEGREE % 2 == 0) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = TOP / 2;  // number of FMA steps after init
    vFloat result = coeffs[TOP];
#pragma GCC unroll 16
    for (int k = 1; k <= STEPS; k++)
        result = result * x2 + coeffs[TOP - 2 * k];
    return result;
}
#endif

#ifdef USE_DUAL_EVAL
// Dual polynomial evaluation using Horner's method with interleaved operations.
// Evaluates two independent x-vectors simultaneously to exploit SFPU instruction-level
// parallelism, hiding the 2-cycle SFPMAD latency by interleaving independent chains.
//
// Usage: eval_polynomial_dual<DEGREE>(coeffs, x1, x2, result1, result2)
// The strict alternation (result1 op, result2 op, result1 op, ...) ensures
// the compiler sees independent computation chains that can be pipelined.
template <uint32_t DEGREE>
inline void eval_polynomial_dual(const float* coeffs, vFloat x1, vFloat x2, vFloat& result1, vFloat& result2) {
    // Each coefficient is hoisted into a named vFloat so the compiler emits one sfploadi
    // pair and reuses the register as src2 for BOTH MADs, halving coefficient load count.
    if constexpr (DEGREE == 0) {
        vFloat c = coeffs[0]; result1 = c; result2 = c;
    } else if constexpr (DEGREE == 1) {
        vFloat c1 = coeffs[1]; vFloat c0 = coeffs[0];
        result1 = c0 + c1 * x1; result2 = c0 + c1 * x2;
    } else if constexpr (DEGREE == 2) {
        { vFloat c = coeffs[2]; result1 = c; result2 = c; }
        { vFloat c = coeffs[1]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 3) {
        { vFloat c = coeffs[3]; result1 = c; result2 = c; }
        { vFloat c = coeffs[2]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 4) {
        { vFloat c = coeffs[4]; result1 = c; result2 = c; }
        { vFloat c = coeffs[3]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[2]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 5) {
        { vFloat c = coeffs[5]; result1 = c; result2 = c; }
        { vFloat c = coeffs[4]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[3]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[2]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 6) {
        { vFloat c = coeffs[6]; result1 = c; result2 = c; }
        { vFloat c = coeffs[5]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[4]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[3]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[2]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 7) {
        { vFloat c = coeffs[7]; result1 = c; result2 = c; }
        { vFloat c = coeffs[6]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[5]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[4]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[3]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[2]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 8) {
        { vFloat c = coeffs[8]; result1 = c; result2 = c; }
        { vFloat c = coeffs[7]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[6]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[5]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[4]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[3]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[2]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 9) {
        { vFloat c = coeffs[9]; result1 = c; result2 = c; }
        { vFloat c = coeffs[8]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[7]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[6]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[5]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[4]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[3]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[2]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 10) {
        { vFloat c = coeffs[10]; result1 = c; result2 = c; }
        { vFloat c = coeffs[9];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[8];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[7];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[6];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[5];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[4];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[3];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[2];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 11) {
        { vFloat c = coeffs[11]; result1 = c; result2 = c; }
        { vFloat c = coeffs[10]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[9];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[8];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[7];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[6];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[5];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[4];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[3];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[2];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 12) {
        { vFloat c = coeffs[12]; result1 = c; result2 = c; }
        { vFloat c = coeffs[11]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[10]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[9];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[8];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[7];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[6];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[5];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[4];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[3];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[2];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 13) {
        { vFloat c = coeffs[13]; result1 = c; result2 = c; }
        { vFloat c = coeffs[12]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[11]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[10]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[9];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[8];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[7];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[6];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[5];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[4];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[3];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[2];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 14) {
        { vFloat c = coeffs[14]; result1 = c; result2 = c; }
        { vFloat c = coeffs[13]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[12]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[11]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[10]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[9];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[8];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[7];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[6];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[5];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[4];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[3];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[2];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 15) {
        { vFloat c = coeffs[15]; result1 = c; result2 = c; }
        { vFloat c = coeffs[14]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[13]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[12]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[11]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[10]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[9];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[8];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[7];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[6];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[5];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[4];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[3];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[2];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    } else if constexpr (DEGREE == 16) {
        { vFloat c = coeffs[16]; result1 = c; result2 = c; }
        { vFloat c = coeffs[15]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[14]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[13]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[12]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[11]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[10]; result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[9];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[8];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[7];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[6];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[5];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[4];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[3];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[2];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[1];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
        { vFloat c = coeffs[0];  result1 = result1 * x1 + c; result2 = result2 * x2 + c; }
    }
}

// Dual-eval x²-Horner for polynomials with parity structure.
// Two interleaved chains in x² basis for ILP, same stride-2 pattern.
#if defined(POLY_PARITY_ODD)
template <uint32_t DEGREE>
inline void eval_polynomial_dual_parity(const float* coeffs,
    vFloat x1, vFloat x2, vFloat x1_sq, vFloat x2_sq,
    vFloat& result1, vFloat& result2) {
    constexpr int TOP = (DEGREE % 2 == 1) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = (TOP - 1) / 2;
    { vFloat c = coeffs[TOP]; result1 = c; result2 = c; }
#pragma GCC unroll 16
    for (int k = 1; k <= STEPS; k++) {
        vFloat c = coeffs[TOP - 2 * k];
        result1 = result1 * x1_sq + c;
        result2 = result2 * x2_sq + c;
    }
    result1 = result1 * x1;
    result2 = result2 * x2;
}
#elif defined(POLY_PARITY_EVEN)
template <uint32_t DEGREE>
inline void eval_polynomial_dual_parity(const float* coeffs,
    vFloat x1, vFloat x2, vFloat x1_sq, vFloat x2_sq,
    vFloat& result1, vFloat& result2) {
    constexpr int TOP = (DEGREE % 2 == 0) ? DEGREE : DEGREE - 1;
    constexpr int STEPS = TOP / 2;
    { vFloat c = coeffs[TOP]; result1 = c; result2 = c; }
#pragma GCC unroll 16
    for (int k = 1; k <= STEPS; k++) {
        vFloat c = coeffs[TOP - 2 * k];
        result1 = result1 * x1_sq + c;
        result2 = result2 * x2_sq + c;
    }
}
#endif

#endif // USE_DUAL_EVAL


// Generic piecewise polynomial LUT for any degree/depth combination
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;  // Skip boundary points

    // Compile-time validation
    static_assert(LUT_SIZE == (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * COEFFS_PER_SEGMENT),
                  "LUT_SIZE must equal (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)");

    // Process all 32 destination registers
    for (int d = 0; d < 32; d++) {
        vFloat x_orig = dst_reg[d];

#if defined(RANGE_REDUCTION_EXP)
        // Exp range reduction: clamp extreme values, then reduce to [-ln(2)/2, ln(2)/2]
        constexpr float EXP_OVERFLOW = 88.5f;
        constexpr float EXP_UNDERFLOW = -88.5f;
        vFloat x;
        vInt k_int;
        exp_reduce(x_orig, x, k_int);
#elif defined(RANGE_REDUCTION_TRIG)
        // Trig range reduction: reduce to [-π/2, π/2]
        vFloat x;
        vInt q_int;
        trig_reduce(x_orig, x, q_int);
#elif defined(RANGE_REDUCTION_TAN)
        // Tan range reduction: reduce to [-π/4, π/4] (Cody-Waite)
        vFloat x;
        vInt j_int;
        tan_reduce(x_orig, x, j_int);
#elif defined(RANGE_REDUCTION_LOG)
        // Log range reduction: extract mantissa m ∈ [1, 2) and exponent e
        // log(x) = e*ln(2) + log(m)
        vFloat x;
        vInt e_int;
        log_reduce(x_orig, x, e_int);
#elif defined(RANGE_REDUCTION_CBRT)
        // Extract BIASED exponent (always non-negative) and sign BEFORE poly eval.
        // Use exexp_nodebias to avoid SFPU's sign-magnitude format for negative debiased values.
        vInt cbrt_biased_e = exexp_nodebias(setsgn(x_orig, 0));
        vInt cbrt_sign = reinterpret<vInt>(x_orig) & 0x80000000;
        vFloat x = cbrt_reduce_m(x_orig);
#else
        vFloat x = x_orig;
#endif

        // Clamping unnecessary: segment cascade v_if(x >= boundary) naturally selects
        // the edge segment for out-of-range inputs. Removing saves SFPU registers.
        vFloat& x_clamped = x;

        // Start with segment 0 coefficients
        vFloat result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET], x);

        // Update with correct segment using plain for-loop
        // Works correctly on all architectures without kernel bloat
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            v_if (x_clamped >= lut[seg]) {
                result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT], x);
            }
            v_endif;
        }

#if defined(RANGE_REDUCTION_EXP)
        // Expand: result = poly_result * 2^k, with overflow/underflow clamping
        v_if(x_orig > EXP_OVERFLOW) {
            result = std::numeric_limits<float>::infinity();
        }
        v_elseif(x_orig < EXP_UNDERFLOW) {
            result = 0.0f;
        }
        v_else {
            result = exp_expand(result, k_int);
        }
        v_endif;
#elif defined(RANGE_REDUCTION_TRIG)
        // Trig expand: negate result if quotient is odd
        result = trig_expand(result, q_int);
#elif defined(RANGE_REDUCTION_TAN)
        // Tan expand: j even → poly(a), j odd → -1/poly(a)
        result = tan_expand(result, j_int);
#elif defined(RANGE_REDUCTION_LOG)
        // Log expand: log(x) = e*ln(2) + log(m)
        // Handle special cases: log(0) = -inf, log(negative) = NaN
        v_if(x_orig < 0.0f) {
            result = std::numeric_limits<float>::quiet_NaN();
        }
        v_elseif(x_orig == 0.0f) {
            result = -std::numeric_limits<float>::infinity();
        }
        v_else {
            result = log_expand(result, e_int);
        }
        v_endif;
#elif defined(RANGE_REDUCTION_CBRT)
        // Compute q = floor(e/3) and r = e mod 3 using biased exponent.
        // All arithmetic stays in float to avoid SFPU's sign-magnitude int format.
        {
            constexpr float ONE_THIRD_C = 0.3333333333333333f;
            const vFloat magic = ckernel::sfpu::Converter::as_float(0x4B400000U);

            // Convert biased exponent to float and debias: e_float = biased - 127
            vFloat e_float = int32_to_float(cbrt_biased_e, RoundMode::Nearest) - 127.0f;

            // q_float ≈ e/3, then round to nearest integer
            vFloat q_approx = e_float * ONE_THIRD_C;
            vFloat q_rounded = q_approx + magic;
            vInt q = reinterpret<vInt>(q_rounded) - reinterpret<vInt>(magic);

            // Get q as float WITHOUT int32_to_float (avoids sign-magnitude)
            vFloat q_back = q_rounded - magic;

            // r = e - 3*q in float, then convert to int via magic number
            vFloat r_float = e_float - (q_back + q_back + q_back);
            vInt r = reinterpret<vInt>(r_float + magic) - reinterpret<vInt>(magic);

            // Corrections for r ∈ {0,1,2}
            v_if(r < 0) { q = q - 1; r = r + 3; } v_endif;
            v_if(r > 2) { q = q + 1; r = r - 3; } v_endif;

            // Handle x=0 (biased_exp = 0 for zero/denorm)
            v_if(cbrt_biased_e < 1) { result = 0.0f; }
            v_else { result = cbrt_expand(result, q, r, cbrt_sign); }
            v_endif;
        }
#endif

#ifdef HAS_CRITICAL_POINT
        // Override with exact value at critical point for perfect accuracy
        // (e.g., cosh(0)=1, tanh(0)=0, sigmoid(0)=0.5)
        v_if (x == lut[CRITICAL_IDX]) {
            result = CRITICAL_VALUE;
        }
        v_endif;
#endif

        dst_reg[d] = result;
    }
}

#ifdef USE_DUAL_EVAL
// Dual-evaluation version of piecewise_generic_lut.
// Processes two destination registers per iteration (d and d+1), exploiting
// SFPU instruction-level parallelism by interleaving two independent Horner chains.
//
// Segment 0 uses eval_polynomial_dual for full ILP benefit.
// Segments 1..N-1 use independent single-eval (x1 and x2 may be in different segments).
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dual(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    static_assert(LUT_SIZE == (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * COEFFS_PER_SEGMENT),
                  "LUT_SIZE must equal (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)");

    // Process pairs of destination registers (32 total → 16 iterations)
    for (int d = 0; d < 32; d += 2) {
        vFloat x_orig1 = dst_reg[d];
        vFloat x_orig2 = dst_reg[d + 1];

#if defined(RANGE_REDUCTION_EXP)
        constexpr float EXP_OVERFLOW = 88.5f;
        constexpr float EXP_UNDERFLOW = -88.5f;
        vFloat x1, x2;
        vInt k_int1, k_int2;
        exp_reduce(x_orig1, x1, k_int1);
        exp_reduce(x_orig2, x2, k_int2);
#elif defined(RANGE_REDUCTION_TRIG)
        vFloat x1, x2;
        vInt q_int1, q_int2;
        trig_reduce(x_orig1, x1, q_int1);
        trig_reduce(x_orig2, x2, q_int2);
#elif defined(RANGE_REDUCTION_TAN)
        vFloat x1, x2;
        vInt j_int1, j_int2;
        tan_reduce(x_orig1, x1, j_int1);
        tan_reduce(x_orig2, x2, j_int2);
#elif defined(RANGE_REDUCTION_LOG)
        vFloat x1, x2;
        vInt e_int1, e_int2;
        log_reduce(x_orig1, x1, e_int1);
        log_reduce(x_orig2, x2, e_int2);
#elif defined(RANGE_REDUCTION_CBRT)
        // Extract BIASED exponent and sign BEFORE poly eval
        vInt cbrt_biased_e1 = exexp_nodebias(setsgn(x_orig1, 0));
        vInt cbrt_sign1 = reinterpret<vInt>(x_orig1) & 0x80000000;
        vInt cbrt_biased_e2 = exexp_nodebias(setsgn(x_orig2, 0));
        vInt cbrt_sign2 = reinterpret<vInt>(x_orig2) & 0x80000000;
        vFloat x1 = cbrt_reduce_m(x_orig1);
        vFloat x2 = cbrt_reduce_m(x_orig2);
#else
        vFloat x1 = x_orig1;
        vFloat x2 = x_orig2;
#endif

        // Clamping unnecessary: segment cascade v_if(x >= boundary) naturally selects
        // the edge segment for out-of-range inputs. Removing saves SFPU registers.
        vFloat& x1_clamped = x1;
        vFloat& x2_clamped = x2;

        // Segment 0: dual evaluation exploits ILP for interleaved Horner chains
        vFloat result1, result2;
        eval_polynomial_dual<POLY_DEGREE>(&lut[COEFF_OFFSET], x1, x2, result1, result2);

        // Segments 1..N-1: independent selection (x1 and x2 may be in different segments)
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            v_if (x1_clamped >= lut[seg]) {
                result1 = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT], x1);
            }
            v_endif;
            v_if (x2_clamped >= lut[seg]) {
                result2 = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT], x2);
            }
            v_endif;
        }

#if defined(RANGE_REDUCTION_EXP)
        v_if(x_orig1 > EXP_OVERFLOW) {
            result1 = std::numeric_limits<float>::infinity();
        }
        v_elseif(x_orig1 < EXP_UNDERFLOW) {
            result1 = 0.0f;
        }
        v_else {
            result1 = exp_expand(result1, k_int1);
        }
        v_endif;
        v_if(x_orig2 > EXP_OVERFLOW) {
            result2 = std::numeric_limits<float>::infinity();
        }
        v_elseif(x_orig2 < EXP_UNDERFLOW) {
            result2 = 0.0f;
        }
        v_else {
            result2 = exp_expand(result2, k_int2);
        }
        v_endif;
#elif defined(RANGE_REDUCTION_TRIG)
        result1 = trig_expand(result1, q_int1);
        result2 = trig_expand(result2, q_int2);
#elif defined(RANGE_REDUCTION_TAN)
        result1 = tan_expand(result1, j_int1);
        result2 = tan_expand(result2, j_int2);
#elif defined(RANGE_REDUCTION_LOG)
        // Log expand: handle special cases and reconstruct log(x) = e*ln(2) + log(m)
        v_if(x_orig1 < 0.0f) {
            result1 = std::numeric_limits<float>::quiet_NaN();
        }
        v_elseif(x_orig1 == 0.0f) {
            result1 = -std::numeric_limits<float>::infinity();
        }
        v_else {
            result1 = log_expand(result1, e_int1);
        }
        v_endif;
        v_if(x_orig2 < 0.0f) {
            result2 = std::numeric_limits<float>::quiet_NaN();
        }
        v_elseif(x_orig2 == 0.0f) {
            result2 = -std::numeric_limits<float>::infinity();
        }
        v_else {
            result2 = log_expand(result2, e_int2);
        }
        v_endif;
#elif defined(RANGE_REDUCTION_CBRT)
        // Cbrt expand using biased exponent — all arithmetic in float
        {
            constexpr float ONE_THIRD_C = 0.3333333333333333f;
            const vFloat magic = ckernel::sfpu::Converter::as_float(0x4B400000U);

            // Result 1
            vFloat e_float1 = int32_to_float(cbrt_biased_e1, RoundMode::Nearest) - 127.0f;
            vFloat q_approx1 = e_float1 * ONE_THIRD_C;
            vFloat q_rounded1 = q_approx1 + magic;
            vInt q1 = reinterpret<vInt>(q_rounded1) - reinterpret<vInt>(magic);
            vFloat q_back1 = q_rounded1 - magic;
            vFloat r_float1 = e_float1 - (q_back1 + q_back1 + q_back1);
            vInt r1 = reinterpret<vInt>(r_float1 + magic) - reinterpret<vInt>(magic);
            v_if(r1 < 0) { q1 = q1 - 1; r1 = r1 + 3; } v_endif;
            v_if(r1 > 2) { q1 = q1 + 1; r1 = r1 - 3; } v_endif;
            v_if(cbrt_biased_e1 < 1) { result1 = 0.0f; }
            v_else { result1 = cbrt_expand(result1, q1, r1, cbrt_sign1); }
            v_endif;

            // Result 2
            vFloat e_float2 = int32_to_float(cbrt_biased_e2, RoundMode::Nearest) - 127.0f;
            vFloat q_approx2 = e_float2 * ONE_THIRD_C;
            vFloat q_rounded2 = q_approx2 + magic;
            vInt q2 = reinterpret<vInt>(q_rounded2) - reinterpret<vInt>(magic);
            vFloat q_back2 = q_rounded2 - magic;
            vFloat r_float2 = e_float2 - (q_back2 + q_back2 + q_back2);
            vInt r2 = reinterpret<vInt>(r_float2 + magic) - reinterpret<vInt>(magic);
            v_if(r2 < 0) { q2 = q2 - 1; r2 = r2 + 3; } v_endif;
            v_if(r2 > 2) { q2 = q2 + 1; r2 = r2 - 3; } v_endif;
            v_if(cbrt_biased_e2 < 1) { result2 = 0.0f; }
            v_else { result2 = cbrt_expand(result2, q2, r2, cbrt_sign2); }
            v_endif;
        }
#endif

#ifdef HAS_CRITICAL_POINT
        v_if (x1 == lut[CRITICAL_IDX]) {
            result1 = CRITICAL_VALUE;
        }
        v_endif;
        v_if (x2 == lut[CRITICAL_IDX]) {
            result2 = CRITICAL_VALUE;
        }
        v_endif;
#endif

        dst_reg[d]     = result1;
        dst_reg[d + 1] = result2;
    }
}
#endif // USE_DUAL_EVAL

// Include specialized implementations for common segment counts
// These use manual unrolling to work around Wormhole SFPU compiler bug
#include "piecewise_generic_specialized.cpp"

} // namespace sfpi
#endif

void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

#ifdef EMBEDDED_LUT
    // Embedded LUT mode: LUT is compiled directly into the kernel
    // Header must define: POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE, LUT_DATA
    // This provides zero L1 memory overhead for the LUT
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Use embedded LUT data directly from header
    constexpr uint32_t lut_size = LUT_SIZE;
    constexpr uint32_t poly_degree = POLY_DEGREE;
    constexpr uint32_t num_segments = NUM_SEGMENTS;
    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;
#else
    // Generic LUT mode: LUT is loaded from L1 circular buffer
    // This allows runtime LUT generation and sharing across cores
    [[maybe_unused]] float input_min = get_arg_val<float>(1);
    [[maybe_unused]] float input_max = get_arg_val<float>(2);
    constexpr uint32_t lut_size = get_compile_time_arg_val(0);
    constexpr uint32_t poly_degree = get_compile_time_arg_val(1);
    constexpr uint32_t num_segments = get_compile_time_arg_val(2);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_lut = tt::CBIndex::c_25;

    // Get LUT array from L1 memory as float array directly
    using lut_t = std::array<float, lut_size>;
    auto p_lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);
#endif

    init_sfpu(cb_in, cb_out);

#if defined(RANGE_REDUCTION_TAN) && defined(TRISC_MATH)
    ckernel::sfpu::sfpu_reciprocal_init<false>();
#endif

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);

        // All degree parameters are constexpr — use them directly as template args.
        // No dispatch table needed; works for ANY poly_degree automatically.
        #ifdef TRISC_MATH
        sfpi::piecewise_generic_lut_dispatch<poly_degree, num_segments, lut_size>(*p_lut);
        #endif

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }
}
