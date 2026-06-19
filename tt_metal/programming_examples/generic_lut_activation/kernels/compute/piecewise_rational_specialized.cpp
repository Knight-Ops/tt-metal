// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Template-recursive unrolling for piecewise rational evaluation.
 *
 * Compile-time unrolls the v_if / eval_rational / v_endif chain for ANY
 * segment count using C++ template recursion. __attribute__((always_inline))
 * ensures all instantiations merge into the caller before LTO — preventing
 * constprop.isra register spilling.
 */

#pragma once

template <uint32_t SEG, uint32_t NUM_DEGREE, uint32_t DEN_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline))
inline void unroll_segment_rational(const std::array<float, LUT_SIZE>& lut,
                                    vFloat x_clamped, vFloat x, vFloat& result) {
    if constexpr (SEG < NUM_SEGMENTS) {
        constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1;
        constexpr uint32_t CPS = NUM_COEFFS + DEN_DEGREE + 1;
        constexpr uint32_t CO = NUM_SEGMENTS + 1;
        v_if (x_clamped >= lut[SEG]) {
            result = eval_rational<NUM_DEGREE, DEN_DEGREE>(
                &lut[CO + SEG * CPS], &lut[CO + SEG * CPS + NUM_COEFFS], x);
        }
        v_endif;
        unroll_segment_rational<SEG + 1, NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut, x_clamped, x, result);
    }
}

template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_rational_lut_N(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1;
    constexpr uint32_t CPS = NUM_COEFFS + DEN_DEGREE + 1;
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
#else
        vFloat x = x_orig;
#endif

        vFloat min_bound = lut[0];
        vFloat max_bound = lut[NUM_SEGMENTS];
        vFloat x_clamped = x;
        vec_min_max(min_bound, x_clamped);
        vec_min_max(x_clamped, max_bound);

        // Segment 0 (no v_if needed)
        vFloat result = eval_rational<NUM_DEGREE, DEN_DEGREE>(
            &lut[COEFF_OFFSET], &lut[COEFF_OFFSET + NUM_COEFFS], x);

        // Segments 1..NUM_SEGMENTS-1 via recursive unrolling
        unroll_segment_rational<1, NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut, x_clamped, x, result);

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
#endif

        dst_reg[d] = result;
    }
}

template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_rational_lut_dispatch(const std::array<float, LUT_SIZE>& lut) {
    piecewise_rational_lut_N<NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
}
