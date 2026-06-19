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

// Range reduction: Cody-Waite method and cbrt exponent decomposition
// Only included when RANGE_REDUCTION_* is defined
#if defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_TRIG) || defined(RANGE_REDUCTION_TAN) || defined(RANGE_REDUCTION_CBRT)
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
    constexpr float NEG_PI_2_HI = -1.5703125f;
    constexpr float NEG_PI_2_LO = -0.0004837512969970703f;
    vFloat a_hi = j * NEG_PI_2_HI + x;
    a = j * NEG_PI_2_LO + a_hi;
}

// Reconstruct: tan(x) = tan(a) when j is even, -1/tan(a) when j is odd
inline vFloat tan_expand(vFloat poly_result, vInt j_int) {
    v_if(j_int & 1) {
        poly_result = -ckernel::sfpu::sfpu_reciprocal<false>(poly_result);
    }
    v_endif;
    return poly_result;
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
// where C = [1.0, cbrt(2), cbrt(4)] ≈ [1.0, 1.2599, 1.5874]
//
// This achieves <1 ULP accuracy with just a degree 3 polynomial on [1, 2)!
// Much faster than ttnn's exp(log(x)/3) approach (3 FMAs vs 12+ FMAs).

// Scale constants: C[r] = cbrt(2^r) for r = 0, 1, 2
constexpr float CBRT_SCALE_C0 = 1.0f;
constexpr float CBRT_SCALE_C1 = 1.2599210498948732f;  // cbrt(2)
constexpr float CBRT_SCALE_C2 = 1.5874010519681994f;  // cbrt(4)

// Cbrt range reduction: extract (m, q, r, sign) from x
// Uses floor-based division to compute q = floor(e/3), guaranteeing r ∈ {0,1,2}
inline void cbrt_reduce(vFloat x, vFloat& m, vInt& q, vInt& r, vInt& sign_bits) {
    // Extract and save sign bit
    sign_bits = reinterpret<vInt>(x) & 0x80000000;
    vFloat abs_x = setsgn(x, 0);  // |x|

    // Extract biased exponent: for x = m * 2^e, exexp returns e (with bias already handled)
    vInt e = exexp(abs_x);

    // Normalize mantissa to [1, 2): m = setexp(|x|, 127) puts mantissa in [1, 2)
    m = setexp(abs_x, 127);

    // Compute q = floor(e / 3) using floor-based approach
    // This guarantees r = e - 3*q is always in {0, 1, 2}, no correction needed!
    vFloat e_float = int32_to_float(e, 0);
    constexpr float ONE_THIRD = 0.3333333333333333f;
    vFloat q_float = e_float * ONE_THIRD;

    // Compute floor(q_float) by working with absolute value
    vInt q_sign_bit = reinterpret<vInt>(q_float) & 0x80000000;
    vFloat abs_q = setsgn(q_float, 0);

    // Round |q_float| using magic number technique (gives round-to-nearest)
    const vFloat magic = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = abs_q + magic;
    vInt abs_q_round = reinterpret<vInt>(tmp) - reinterpret<vInt>(magic);

    // Convert round to floor: if round > |q|, subtract 1
    vFloat abs_q_round_f = int32_to_float(abs_q_round, 0);
    v_if(abs_q_round_f > abs_q) {
        abs_q_round = abs_q_round - 1;
        abs_q_round_f = abs_q_round_f - 1.0f;
    }
    v_endif;
    // Now abs_q_round = floor(|q_float|)

    // Compute floor(q_float) based on sign
    // For positive: floor(q) = floor(|q|)
    // For negative: floor(-|q|) = -ceil(|q|) = -(floor(|q|) + has_frac)
    q = abs_q_round;
    v_if(q_sign_bit != 0) {
        // q_float < 0: floor = -ceil(|q|)
        v_if(abs_q != abs_q_round_f) {
            // Has fractional part: ceil = floor + 1
            q = -(abs_q_round + 1);
        }
        v_else {
            // Integer: ceil = floor
            q = -abs_q_round;
        }
        v_endif;
    }
    v_endif;

    // r = e - 3*q is guaranteed to be in {0, 1, 2} since q = floor(e/3)
    // Note: vInt has no multiply operator; use shift+add: (q<<1)+q = 3*q
    r = e - ((q << 1) + q);
}

// Reconstruct: cbrt(x) = sign * C[r] * poly(m) * 2^q
inline vFloat cbrt_expand(vFloat poly_result, vInt q, vInt r, vInt sign_bits) {
    // Look up scale factor C[r] = cbrt(2^r)
    vFloat scale = CBRT_SCALE_C0;  // default r=0
    v_if(r == 1) {
        scale = CBRT_SCALE_C1;
    }
    v_endif;
    v_if(r == 2) {
        scale = CBRT_SCALE_C2;
    }
    v_endif;

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
        // Duodecic: y = c0 + ... + c12*x¹²
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
        // Tan range reduction: reduce to [-π/4, π/4]
        vFloat x;
        vInt j_int;
        tan_reduce(x_orig, x, j_int);
#elif defined(RANGE_REDUCTION_CBRT)
        // Cbrt range reduction: reduce to [1, 2) via exponent decomposition
        // cbrt(x) = sign * C[r] * cbrt(m) * 2^q where |x| = m * 2^(3q+r)
        vFloat x;  // Will hold mantissa m ∈ [1, 2)
        vInt cbrt_q, cbrt_r, cbrt_sign;
        cbrt_reduce(x_orig, x, cbrt_q, cbrt_r, cbrt_sign);
#else
        vFloat x = x_orig;
#endif

        // Clamp x to valid range using precomputed boundaries (branchless via vec_min_max)
        vFloat min_bound = lut[0];
        vFloat max_bound = lut[NUM_SEGMENTS];
        vFloat x_clamped = x;
        vec_min_max(min_bound, x_clamped);   // x_clamped = max(x, lut[0])
        vec_min_max(x_clamped, max_bound);   // x_clamped = min(x_clamped, lut[NUM_SEGMENTS])

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
        // Tan expand: reciprocal and negate if quadrant is odd
        result = tan_expand(result, j_int);
#elif defined(RANGE_REDUCTION_CBRT)
        // Cbrt expand: result = sign * C[r] * poly(m) * 2^q
        // Handle x=0 specially: cbrt(0) = 0
        v_if(x_orig == 0.0f) {
            result = 0.0f;
        }
        v_else {
            result = cbrt_expand(result, cbrt_q, cbrt_r, cbrt_sign);
        }
        v_endif;
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

// Staggered Horner piecewise LUT
//
// Interleaved approach: for each Horner step (from highest degree down to 0),
// select the correct per-lane coefficient via one boundary cascade, then do
// the Horner step. Only ~4 SFPU registers live at any time (result, coeff, x,
// x_clamped) — no register spill even for high degrees.
//
// Total sfpmad = POLY_DEGREE (one per Horner step).
// Boundary comparisons (v_if x >= b[seg]) are sfpsetcc/sfpmov, not sfpmad.
//
// vs original: NUM_SEGMENTS * POLY_DEGREE sfpmad (evaluates all segments redundantly)
// vs staggered: POLY_DEGREE sfpmad only
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_staggered_horner_lut(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;  // Skip boundary points

    static_assert(LUT_SIZE == (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * COEFFS_PER_SEGMENT),
                  "LUT_SIZE must equal (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)");

    for (int d = 0; d < 32; d++) {
        vFloat x_orig = dst_reg[d];

#if defined(RANGE_REDUCTION_EXP)
        constexpr float EXP_OVERFLOW = 88.5f;
        constexpr float EXP_UNDERFLOW = -88.5f;
        vFloat x;
        vInt k_int;
        exp_reduce(x_orig, x, k_int);
#elif defined(RANGE_REDUCTION_TRIG)
        vFloat x;
        vInt q_int;
        trig_reduce(x_orig, x, q_int);
#elif defined(RANGE_REDUCTION_TAN)
        vFloat x;
        vInt j_int;
        tan_reduce(x_orig, x, j_int);
#elif defined(RANGE_REDUCTION_CBRT)
        vFloat x;
        vInt cbrt_q, cbrt_r, cbrt_sign;
        cbrt_reduce(x_orig, x, cbrt_q, cbrt_r, cbrt_sign);
#else
        vFloat x = x_orig;
#endif

        vFloat min_bound = lut[0];
        vFloat max_bound = lut[NUM_SEGMENTS];
        vFloat x_clamped = x;
        vec_min_max(min_bound, x_clamped);
        vec_min_max(x_clamped, max_bound);

        // Select highest-degree coefficient via boundary cascade. Only result + x_clamped live.
        vFloat result = lut[COEFF_OFFSET + POLY_DEGREE];  // seg 0 default
        #pragma unroll
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            v_if (x_clamped >= lut[seg]) {
                result = lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT + POLY_DEGREE];
            }
            v_endif;
        }

        // Interleaved: boundary cascade per coeff slot, then ONE Horner sfpmad.
        // Only 4 registers live (result, coeff, x, x_clamped) — no spill for any degree.
        #pragma unroll
        for (int deg = (int)POLY_DEGREE - 1; deg >= 0; deg--) {
            vFloat coeff = lut[COEFF_OFFSET + deg];  // seg 0 default
            #pragma unroll
            for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
                v_if (x_clamped >= lut[seg]) {
                    coeff = lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT + deg];
                }
                v_endif;
            }
            result = result * x + coeff;
        }

#if defined(RANGE_REDUCTION_EXP)
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
        result = trig_expand(result, q_int);
#elif defined(RANGE_REDUCTION_TAN)
        result = tan_expand(result, j_int);
#elif defined(RANGE_REDUCTION_CBRT)
        v_if(x_orig == 0.0f) {
            result = 0.0f;
        }
        v_else {
            result = cbrt_expand(result, cbrt_q, cbrt_r, cbrt_sign);
        }
        v_endif;
#endif

#ifdef HAS_CRITICAL_POINT
        v_if (x == lut[CRITICAL_IDX]) {
            result = CRITICAL_VALUE;
        }
        v_endif;
#endif

        dst_reg[d] = result;
    }
}

// Include specialized implementations for common segment counts
// These use manual unrolling to work around Wormhole SFPU compiler bug
#include "piecewise_generic_specialized.cpp"

} // namespace sfpi
#endif

void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t compute_loop_factor = get_arg_val<uint32_t>(1);  // How many times to recompute each tile

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

        // COMPUTE AMPLIFICATION: Run compute multiple times on same tile data
        for (uint32_t loop = 0; loop < compute_loop_factor; loop++) {
            // All degree parameters are constexpr — use them directly as template args.
            // No dispatch table needed; works for ANY poly_degree automatically.
            #ifdef TRISC_MATH
            sfpi::piecewise_generic_lut_dispatch<poly_degree, num_segments, lut_size>(*p_lut);
            #endif
        }

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }
}
