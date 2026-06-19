// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Template-recursive unrolling for piecewise polynomial evaluation.
 *
 * Compile-time unrolls the v_if / eval_polynomial / v_endif chain for ANY
 * segment count using C++ template recursion. The SFPU compiler (GCC 15.1.0)
 * handles if constexpr recursion, and __attribute__((always_inline)) ensures
 * all instantiations merge into the caller before LTO — preventing the
 * constprop.isra register spilling that plagued the old hand-written
 * specializations at high degrees.
 *
 * ADAPTIVE DEGREE: When HAS_SEGMENT_DEGREES is defined (by run_csv.sh or
 * generate_embedded_kernels.py), SEGMENT_DEGREES[SEG] provides the effective
 * degree per segment as a constexpr array. Since both the array and SEG are
 * compile-time constants, SEGMENT_DEGREES[SEG] is a valid template argument
 * for eval_polynomial<>. This skips wasted FMA ops on zero high-order coeffs.
 *
 * PARITY x²-HORNER: When POLY_PARITY_ODD or POLY_PARITY_EVEN is defined,
 * evaluation uses x² basis with stride-2 coefficient access, halving FMA count.
 * x² is precomputed once per DST register and threaded through the unroller.
 *
 * CPS (coeffs per segment) is always POLY_DEGREE+1 because LUT storage uses
 * max degree for indexing regardless of per-segment effective degree.
 */

#pragma once

// ============================================================================
// Single-eval recursive unroller
// ============================================================================

template <uint32_t SEG, uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline))
inline void unroll_segment(const std::array<float, LUT_SIZE>& lut,
                           vFloat x_clamped, vFloat x, vFloat& result
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
                           , vFloat x2
#endif
                           ) {
    if constexpr (SEG < NUM_SEGMENTS) {
        constexpr uint32_t CPS = POLY_DEGREE + 1;
        constexpr uint32_t CO = NUM_SEGMENTS + 1;
#ifdef HAS_SEGMENT_DEGREES
        constexpr uint32_t DEG = SEGMENT_DEGREES[SEG];
#else
        constexpr uint32_t DEG = POLY_DEGREE;
#endif
        v_if (x_clamped >= lut[SEG]) {
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            result = eval_polynomial_parity<DEG>(&lut[CO + SEG * CPS], x, x2);
#else
            result = eval_polynomial<DEG>(&lut[CO + SEG * CPS], x);
#endif
        }
        v_endif;
        unroll_segment<SEG + 1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut, x_clamped, x, result
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            , x2
#endif
        );
    }
}

template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_specialized_N(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

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
#elif defined(RANGE_REDUCTION_LOG)
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

#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
        vFloat x2 = x * x;
#endif

        // Segment 0 (no v_if needed)
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
    #ifdef HAS_SEGMENT_DEGREES
        vFloat result = eval_polynomial_parity<SEGMENT_DEGREES[0]>(&lut[COEFF_OFFSET], x, x2);
    #else
        vFloat result = eval_polynomial_parity<POLY_DEGREE>(&lut[COEFF_OFFSET], x, x2);
    #endif
#else
    #ifdef HAS_SEGMENT_DEGREES
        vFloat result = eval_polynomial<SEGMENT_DEGREES[0]>(&lut[COEFF_OFFSET], x);
    #else
        vFloat result = eval_polynomial<POLY_DEGREE>(&lut[COEFF_OFFSET], x);
    #endif
#endif

        // Segments 1..NUM_SEGMENTS-1 via recursive unrolling
        unroll_segment<1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut, x_clamped, x, result
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            , x2
#endif
        );

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
#elif defined(RANGE_REDUCTION_LOG)
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
            vFloat e_float = int32_to_float(cbrt_biased_e, 0) - 127.0f;

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

        // ================================================================
        // Asymptotic factoring: multiply correction polynomial by dominant
        // factor for segments in the tail region. Applied post-evaluation
        // since dominant(x) is independent of the piecewise segment.
        // Mutually exclusive with range reduction (never combined).
        // ================================================================
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
        // f(x) = exp(ASYMPTOTIC_EXP_ARG_SCALE * x²) * ASYMPTOTIC_SCALE * correction(x)
        // e.g. GELU tail: exp(-x²/2) * (-1/√(2π)) * poly(x)
        v_if (x_orig < ASYMPTOTIC_UPPER_BOUND) {
            vFloat t = x_orig * x_orig * ASYMPTOTIC_EXP_ARG_SCALE;
            result = result * asymptotic_exp(t) * ASYMPTOTIC_SCALE;
        }
        v_endif;
#elif defined(ASYMPTOTIC_FACTOR_EXP_LINEAR)
        // f(x) = exp(ASYMPTOTIC_EXP_ARG_SCALE * x) * ASYMPTOTIC_SCALE * correction(x)
        // e.g. sigmoid left tail: exp(x) * poly(x)
        v_if (x_orig < ASYMPTOTIC_UPPER_BOUND) {
            vFloat t = x_orig * ASYMPTOTIC_EXP_ARG_SCALE;
            result = result * asymptotic_exp(t) * ASYMPTOTIC_SCALE;
        }
        v_endif;
#elif defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR)
        // f(x) = x * exp(ASYMPTOTIC_EXP_ARG_SCALE * x) * ASYMPTOTIC_SCALE * correction(x)
        // e.g. silu/mish left tail: x * exp(x) * poly(x)
        v_if (x_orig < ASYMPTOTIC_UPPER_BOUND) {
            vFloat t = x_orig * ASYMPTOTIC_EXP_ARG_SCALE;
            result = result * x_orig * asymptotic_exp(t) * ASYMPTOTIC_SCALE;
        }
        v_endif;
#elif defined(ASYMPTOTIC_FACTOR_X)
        // f(x) = x * ASYMPTOTIC_SCALE * correction(x)
        // Trivial factoring (e.g., GELU right tail: x * poly(x))
        v_if (x_orig > ASYMPTOTIC_LOWER_BOUND) {
            result = result * x_orig * ASYMPTOTIC_SCALE;
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

// ============================================================================
// Dual-eval recursive unroller (processes 2 DST rows for ILP)
// ============================================================================

#ifdef USE_DUAL_EVAL

template <uint32_t SEG, uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline))
inline void unroll_segment_dual(const std::array<float, LUT_SIZE>& lut,
                                vFloat x1c, vFloat x2c, vFloat x1, vFloat x2,
                                vFloat& r1, vFloat& r2
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
                                , vFloat x1_sq, vFloat x2_sq
#endif
                                ) {
    if constexpr (SEG < NUM_SEGMENTS) {
        constexpr uint32_t CPS = POLY_DEGREE + 1;
        constexpr uint32_t CO = NUM_SEGMENTS + 1;
#ifdef HAS_SEGMENT_DEGREES
        constexpr uint32_t DEG = SEGMENT_DEGREES[SEG];
#else
        constexpr uint32_t DEG = POLY_DEGREE;
#endif
        {
            vFloat tmp1, tmp2;
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            eval_polynomial_dual_parity<DEG>(&lut[CO + SEG * CPS], x1, x2, x1_sq, x2_sq, tmp1, tmp2);
#else
            eval_polynomial_dual<DEG>(&lut[CO + SEG * CPS], x1, x2, tmp1, tmp2);
#endif
            vFloat b = lut[SEG];
            v_if (x1c >= b) { r1 = tmp1; } v_endif;
            v_if (x2c >= b) { r2 = tmp2; } v_endif;
        }
        unroll_segment_dual<SEG + 1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut, x1c, x2c, x1, x2, r1, r2
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            , x1_sq, x2_sq
#endif
        );
    }
}

template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_specialized_N_dual(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

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
        vFloat x1 = cbrt_reduce_m(x_orig1);
        vFloat x2 = cbrt_reduce_m(x_orig2);
#else
        vFloat x1 = x_orig1;
        vFloat x2 = x_orig2;
#endif

        // Dual-eval: skip clamping to avoid SFPU register spill from vec_min_max temporaries.
        // Segment boundaries still gate correctly — out-of-range x just picks the edge segment.
        vFloat& x1_clamped = x1;
        vFloat& x2_clamped = x2;

#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
        vFloat x1_sq = x1 * x1;
        vFloat x2_sq = x2 * x2;
#endif

        // Segment 0: dual evaluation for ILP
        vFloat result1, result2;
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
    #ifdef HAS_SEGMENT_DEGREES
        eval_polynomial_dual_parity<SEGMENT_DEGREES[0]>(&lut[COEFF_OFFSET], x1, x2, x1_sq, x2_sq, result1, result2);
    #else
        eval_polynomial_dual_parity<POLY_DEGREE>(&lut[COEFF_OFFSET], x1, x2, x1_sq, x2_sq, result1, result2);
    #endif
#else
    #ifdef HAS_SEGMENT_DEGREES
        eval_polynomial_dual<SEGMENT_DEGREES[0]>(&lut[COEFF_OFFSET], x1, x2, result1, result2);
    #else
        eval_polynomial_dual<POLY_DEGREE>(&lut[COEFF_OFFSET], x1, x2, result1, result2);
    #endif
#endif

        // Segments 1..NUM_SEGMENTS-1 via recursive unrolling
        unroll_segment_dual<1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut, x1_clamped, x2_clamped, x1, x2, result1, result2
#if defined(POLY_PARITY_ODD) || defined(POLY_PARITY_EVEN)
            , x1_sq, x2_sq
#endif
        );

#if defined(RANGE_REDUCTION_EXP)
        v_if(x_orig1 > EXP_OVERFLOW) { result1 = std::numeric_limits<float>::infinity(); }
        v_elseif(x_orig1 < EXP_UNDERFLOW) { result1 = 0.0f; }
        v_else { result1 = exp_expand(result1, k_int1); }
        v_endif;
        v_if(x_orig2 > EXP_OVERFLOW) { result2 = std::numeric_limits<float>::infinity(); }
        v_elseif(x_orig2 < EXP_UNDERFLOW) { result2 = 0.0f; }
        v_else { result2 = exp_expand(result2, k_int2); }
        v_endif;
#elif defined(RANGE_REDUCTION_TRIG)
        result1 = trig_expand(result1, q_int1);
        result2 = trig_expand(result2, q_int2);
#elif defined(RANGE_REDUCTION_TAN)
        result1 = tan_expand(result1, j_int1);
        result2 = tan_expand(result2, j_int2);
#elif defined(RANGE_REDUCTION_LOG)
        v_if(x_orig1 < 0.0f) {
            result1 = std::numeric_limits<float>::quiet_NaN();
        }
        v_elseif(x_orig1 == 0.0f) {
            result1 = -std::numeric_limits<float>::infinity();
        }
        v_else { result1 = log_expand(result1, e_int1); }
        v_endif;
        v_if(x_orig2 < 0.0f) {
            result2 = std::numeric_limits<float>::quiet_NaN();
        }
        v_elseif(x_orig2 == 0.0f) {
            result2 = -std::numeric_limits<float>::infinity();
        }
        v_else { result2 = log_expand(result2, e_int2); }
        v_endif;
#elif defined(RANGE_REDUCTION_CBRT)
        // Note: dual-eval is disabled for cbrt (range reduction → single-eval fallback),
        // but this must compile. Uses same biased-exponent pattern as single-eval.
        {
            vFloat tmp1 = dst_reg[d];
            vInt be1 = exexp_nodebias(setsgn(tmp1, 0));
            vInt s1 = reinterpret<vInt>(tmp1) & 0x80000000;
            constexpr float OT = 0.3333333333333333f;
            const vFloat mag = ckernel::sfpu::Converter::as_float(0x4B400000U);
            vFloat ef1 = int32_to_float(be1, 0) - 127.0f;
            vFloat qr1 = ef1 * OT + mag;
            vInt q1 = reinterpret<vInt>(qr1) - reinterpret<vInt>(mag);
            vFloat qb1 = qr1 - mag;
            vInt r1 = reinterpret<vInt>(ef1 - (qb1+qb1+qb1) + mag) - reinterpret<vInt>(mag);
            v_if(r1 < 0) { q1 = q1 - 1; r1 = r1 + 3; } v_endif;
            v_if(r1 > 2) { q1 = q1 + 1; r1 = r1 - 3; } v_endif;
            v_if(be1 < 1) { result1 = 0.0f; }
            v_else { result1 = cbrt_expand(result1, q1, r1, s1); }
            v_endif;
        }
        {
            vFloat tmp2 = dst_reg[d+1];
            vInt be2 = exexp_nodebias(setsgn(tmp2, 0));
            vInt s2 = reinterpret<vInt>(tmp2) & 0x80000000;
            constexpr float OT = 0.3333333333333333f;
            const vFloat mag = ckernel::sfpu::Converter::as_float(0x4B400000U);
            vFloat ef2 = int32_to_float(be2, 0) - 127.0f;
            vFloat qr2 = ef2 * OT + mag;
            vInt q2 = reinterpret<vInt>(qr2) - reinterpret<vInt>(mag);
            vFloat qb2 = qr2 - mag;
            vInt r2 = reinterpret<vInt>(ef2 - (qb2+qb2+qb2) + mag) - reinterpret<vInt>(mag);
            v_if(r2 < 0) { q2 = q2 - 1; r2 = r2 + 3; } v_endif;
            v_if(r2 > 2) { q2 = q2 + 1; r2 = r2 - 3; } v_endif;
            v_if(be2 < 1) { result2 = 0.0f; }
            v_else { result2 = cbrt_expand(result2, q2, r2, s2); }
            v_endif;
        }
#endif

        // Asymptotic factoring (dual-eval): apply dominant factor to both lanes
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
        v_if (x_orig1 < ASYMPTOTIC_UPPER_BOUND) {
            vFloat t1 = x_orig1 * x_orig1 * ASYMPTOTIC_EXP_ARG_SCALE;
            result1 = result1 * asymptotic_exp(t1) * ASYMPTOTIC_SCALE;
        }
        v_endif;
        v_if (x_orig2 < ASYMPTOTIC_UPPER_BOUND) {
            vFloat t2 = x_orig2 * x_orig2 * ASYMPTOTIC_EXP_ARG_SCALE;
            result2 = result2 * asymptotic_exp(t2) * ASYMPTOTIC_SCALE;
        }
        v_endif;
#elif defined(ASYMPTOTIC_FACTOR_EXP_LINEAR)
        v_if (x_orig1 < ASYMPTOTIC_UPPER_BOUND) {
            vFloat t1 = x_orig1 * ASYMPTOTIC_EXP_ARG_SCALE;
            result1 = result1 * asymptotic_exp(t1) * ASYMPTOTIC_SCALE;
        }
        v_endif;
        v_if (x_orig2 < ASYMPTOTIC_UPPER_BOUND) {
            vFloat t2 = x_orig2 * ASYMPTOTIC_EXP_ARG_SCALE;
            result2 = result2 * asymptotic_exp(t2) * ASYMPTOTIC_SCALE;
        }
        v_endif;
#elif defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR)
        v_if (x_orig1 < ASYMPTOTIC_UPPER_BOUND) {
            vFloat t1 = x_orig1 * ASYMPTOTIC_EXP_ARG_SCALE;
            result1 = result1 * x_orig1 * asymptotic_exp(t1) * ASYMPTOTIC_SCALE;
        }
        v_endif;
        v_if (x_orig2 < ASYMPTOTIC_UPPER_BOUND) {
            vFloat t2 = x_orig2 * ASYMPTOTIC_EXP_ARG_SCALE;
            result2 = result2 * x_orig2 * asymptotic_exp(t2) * ASYMPTOTIC_SCALE;
        }
        v_endif;
#elif defined(ASYMPTOTIC_FACTOR_X)
        v_if (x_orig1 > ASYMPTOTIC_LOWER_BOUND) {
            result1 = result1 * x_orig1 * ASYMPTOTIC_SCALE;
        }
        v_endif;
        v_if (x_orig2 > ASYMPTOTIC_LOWER_BOUND) {
            result2 = result2 * x_orig2 * ASYMPTOTIC_SCALE;
        }
        v_endif;
#endif

#ifdef HAS_CRITICAL_POINT
        v_if (x1 == lut[CRITICAL_IDX]) { result1 = CRITICAL_VALUE; } v_endif;
        v_if (x2 == lut[CRITICAL_IDX]) { result2 = CRITICAL_VALUE; } v_endif;
#endif

        dst_reg[d]     = result1;
        dst_reg[d + 1] = result2;
    }
}

#endif // USE_DUAL_EVAL

// ============================================================================
// Dispatcher
// ============================================================================

template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_dispatch(const std::array<float, LUT_SIZE>& lut) {
#ifdef USE_DUAL_EVAL
    // Range reduction keeps extra vInt registers live across the polynomial evaluation,
    // pushing total SFPU register pressure beyond the hardware limit.
    // Fall back to single-eval when any range reduction is active.
    #if defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_TRIG) || defined(RANGE_REDUCTION_TAN) || defined(RANGE_REDUCTION_LOG) || defined(RANGE_REDUCTION_CBRT)
    piecewise_generic_lut_specialized_N<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
    #else
    piecewise_generic_lut_specialized_N_dual<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
    #endif
#else
    piecewise_generic_lut_specialized_N<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
#endif
}
