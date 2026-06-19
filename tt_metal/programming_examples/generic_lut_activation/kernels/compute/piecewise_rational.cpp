// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

// Include reciprocal function for rational evaluation
#ifdef TRISC_MATH
#include "ckernel_sfpu_recip.h"
#endif

namespace kutil = norm::kernel_util;

/**
 * Piecewise Rational Approximation Kernel
 *
 * Evaluates: y = P(x) / Q(x) where P and Q are polynomials
 *
 * For each segment:
 *   P(x) = n0 + n1*x + n2*x² + ... (numerator polynomial)
 *   Q(x) = d0 + d1*x + d2*x² + ... (denominator polynomial)
 *   y = P(x) / Q(x)
 *
 * LUT Format: [b0, b1, ..., bN, n0_seg0, n1_seg0, ..., d0_seg0, d1_seg0, ..., n0_seg1, ...]
 *
 * Template parameters:
 *   NUM_DEGREE    - Degree of numerator polynomial
 *   DEN_DEGREE    - Degree of denominator polynomial
 *   NUM_SEGMENTS  - Number of piecewise segments
 *   LUT_SIZE      - Total LUT size
 *
 * Supports both modes:
 *   - EMBEDDED_LUT: Coefficients compiled directly into kernel (zero L1 overhead)
 *   - CB mode: Coefficients loaded from L1 circular buffer (runtime flexibility)
 */

#ifdef TRISC_MATH

// Range reduction: Cody-Waite method
// Only included when RANGE_REDUCTION_EXP or RANGE_REDUCTION_TRIG is defined by the generated kernel header
#if defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_TRIG)
#include "sfpu/ckernel_sfpu_converter.h"
#endif

namespace sfpi {

#ifdef RANGE_REDUCTION_EXP
// Cody-Waite range reduction: x → (s, k_int) where exp(x) = 2^k * exp(s)
// s ∈ [-ln(2)/2, ln(2)/2] ≈ [-0.347, 0.347]
// Heavily optimized to minimize live SFPU registers - critical for exp with 2-segment rationals
inline void exp_reduce(vFloat x, vFloat& s, vInt& k_int) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    constexpr float NEG_LN2_HI = -0.6931152343750000f;
    constexpr float NEG_LN2_LO = -3.19461832987e-05f;

    // Round-to-nearest-even: z = x/ln(2), k = round(z)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = x * INV_LN2 + c231;  // Fuse multiply and add to reduce live registers

    // Extract k_int directly, minimize temporaries
    k_int = reinterpret<vInt>(tmp) - reinterpret<vInt>(c231);

    // Compute s = x - k*ln(2) with Cody-Waite extended precision
    // Reuse tmp-c231 as k to avoid additional register
    s = (tmp - c231) * NEG_LN2_LO + ((tmp - c231) * NEG_LN2_HI + x);
}

// Inline exp_expand to minimize register pressure - avoid creating intermediate vInt variables
inline vFloat exp_expand(vFloat poly_result, vInt k_int) {
    // Inline the operations to reduce live registers
    return setexp(poly_result, exexp_nodebias(poly_result) + k_int);
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

// Polynomial evaluation using Horner's method - reused for both numerator and denominator
template <uint32_t DEGREE>
inline vFloat eval_poly(const float* coeffs, vFloat x) {
    if constexpr (DEGREE == 0) {
        return coeffs[0];
    } else if constexpr (DEGREE == 1) {
        return coeffs[0] + coeffs[1] * x;
    } else if constexpr (DEGREE == 2) {
        return (coeffs[2] * x + coeffs[1]) * x + coeffs[0];
    } else if constexpr (DEGREE == 3) {
        return ((coeffs[3] * x + coeffs[2]) * x + coeffs[1]) * x + coeffs[0];
    } else if constexpr (DEGREE == 4) {
        vFloat result = coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 5) {
        vFloat result = coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 6) {
        vFloat result = coeffs[6];
        result = result * x + coeffs[5];
        result = result * x + coeffs[4];
        result = result * x + coeffs[3];
        result = result * x + coeffs[2];
        result = result * x + coeffs[1];
        result = result * x + coeffs[0];
        return result;
    } else if constexpr (DEGREE == 7) {
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
    } else {
        // General case for higher degrees
        vFloat result = coeffs[DEGREE];
        for (int i = DEGREE - 1; i >= 0; i--) {
            result = result * x + coeffs[i];
        }
        return result;
    }
}

// JIT (Just-In-Time) coefficient loading version for high-degree rationals
// This version loads coefficients one at a time to minimize register pressure
// Uses only 3-4 vFloat registers instead of 15+ for high degrees
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline vFloat eval_rational_jit(const float* num_coeffs, const float* den_coeffs, vFloat x) {
    // === PHASE 1: Evaluate denominator with JIT loading ===
    // Load highest degree coefficient for denominator
    vFloat coeff = den_coeffs[DEN_DEGREE];
    vFloat denom_result = coeff;

    // Horner's method: load one coefficient at a time, reusing the 'coeff' register
    // CRITICAL: Do NOT unroll for low degrees with range reduction - unrolling exposes
    // all loop iterations to register allocator at once, causing spills!
    // Only unroll for high degrees where loop overhead matters more than register pressure
#if defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_TRIG)
    // With range reduction: disable unroll for degrees < 10 to avoid register spills
    #if (DEN_DEGREE >= 10)
    #pragma unroll
    #endif
#else
    // Without range reduction: safe to unroll at all degrees
    #pragma unroll
#endif
    for (int i = DEN_DEGREE - 1; i >= 0; i--) {
        coeff = den_coeffs[i];  // Reuse register - previous value no longer needed
        denom_result = denom_result * x + coeff;
    }

    // Compute reciprocal (denom_result dies after this)
    vFloat recip = ckernel::sfpu::sfpu_reciprocal<false>(denom_result);

    // === PHASE 2: Evaluate numerator with JIT loading ===
    // Reuse the same 'coeff' register for numerator coefficients
    coeff = num_coeffs[NUM_DEGREE];
    vFloat numer_result = coeff;

#if defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_TRIG)
    // With range reduction: disable unroll for degrees < 10 to avoid register spills
    #if (NUM_DEGREE >= 10)
    #pragma unroll
    #endif
#else
    // Without range reduction: safe to unroll at all degrees
    #pragma unroll
#endif
    for (int i = NUM_DEGREE - 1; i >= 0; i--) {
        coeff = num_coeffs[i];  // Reuse register
        numer_result = numer_result * x + coeff;
    }

    // Final multiply: only 2 registers live (numer_result + recip)
    return numer_result * recip;
}

// Evaluate rational function for a single segment: P(x)/Q(x)
// Dispatches to JIT version for high degrees to avoid register pressure
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
inline vFloat eval_rational(const float* num_coeffs, const float* den_coeffs, vFloat x) {
    // Use JIT loading for:
    // 1. High-degree rationals (degree >= 7) to avoid register spilling
    // 2. EMBEDDED_LUT mode - constprop optimization causes register pressure even at low degrees
    // 3. Range reduction + multi-segment - additional live registers (q_int, x_orig, etc.)
    // For n8_d8_s2: without JIT, we get 15+ live registers → compiler error
    // With JIT: only 3-4 live registers → compiles successfully
#ifdef EMBEDDED_LUT
    // Always use JIT for embedded mode to avoid constprop register pressure
    return eval_rational_jit<NUM_DEGREE, DEN_DEGREE>(num_coeffs, den_coeffs, x);
#else
    // When range reduction is active, use JIT for all degrees to avoid register pressure
    // Range reduction adds 3-4 live registers (x_orig, x, k_int/q_int, result)
    // Combined with 2-segment evaluation, even low degrees hit register limits
#if defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_TRIG)
    return eval_rational_jit<NUM_DEGREE, DEN_DEGREE>(num_coeffs, den_coeffs, x);
#else
    if constexpr (NUM_DEGREE >= 7 || DEN_DEGREE >= 7) {
        return eval_rational_jit<NUM_DEGREE, DEN_DEGREE>(num_coeffs, den_coeffs, x);
    } else {
        // For low degrees in CB mode without range reduction, use the original fast path
        vFloat numerator = eval_poly<NUM_DEGREE>(num_coeffs, x);
        vFloat denominator = eval_poly<DEN_DEGREE>(den_coeffs, x);
        vFloat recip = ckernel::sfpu::sfpu_reciprocal<false>(denominator);
        return numerator * recip;
    }
#endif
#endif
}

// Include specialized implementations using recursive template unrolling
// Replaces old hand-written piecewise_rational_lut_2/4/8/16/generic functions
#include "piecewise_rational_specialized.cpp"

} // namespace sfpi
#endif

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t compute_loop_factor = get_arg_val<uint32_t>(1);  // How many times to recompute each tile

#ifdef EMBEDDED_LUT
    // Embedded LUT mode: LUT is compiled directly into the kernel
    // Header must define: NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE, LUT_DATA
    // This provides zero L1 memory overhead for the LUT
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Use embedded LUT data directly from header
    constexpr uint32_t lut_size = LUT_SIZE;
    constexpr uint32_t num_degree = NUM_DEGREE;
    constexpr uint32_t den_degree = DEN_DEGREE;
    constexpr uint32_t num_segments = NUM_SEGMENTS;
    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;
#else
    // Generic LUT mode: LUT is loaded from L1 circular buffer
    // This allows runtime LUT generation and sharing across cores
    constexpr uint32_t lut_size = get_compile_time_arg_val(0);
    constexpr uint32_t num_degree = get_compile_time_arg_val(1);
    constexpr uint32_t den_degree = get_compile_time_arg_val(2);
    constexpr uint32_t num_segments = get_compile_time_arg_val(3);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_lut = tt::CBIndex::c_25;

    // Get LUT array from L1 memory as float array directly
    using lut_t = std::array<float, lut_size>;
    auto p_lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);
#endif

    init_sfpu(cb_in, cb_out);

#ifdef TRISC_MATH
    // Initialize reciprocal SFPU function (needed for rational evaluation)
    // Platform-independent init for reciprocal
    ckernel::sfpu::sfpu_reciprocal_init<false>();
#endif

    // Must call copy_tile_init unconditionally (matches piecewise_generic.cpp)
    copy_tile_init(cb_in);

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);

        // COMPUTE AMPLIFICATION: Run compute multiple times on same tile data
        for (uint32_t loop = 0; loop < compute_loop_factor; loop++) {
            #ifdef TRISC_MATH
            // All degree parameters are constexpr — use them directly as template args.
            // No dispatch table needed; works for ANY (num_degree, den_degree) combination.
            sfpi::piecewise_rational_lut_dispatch<num_degree, den_degree, num_segments, lut_size>(*p_lut);
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
}
