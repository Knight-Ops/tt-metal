// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

namespace kutil = norm::kernel_util;

/**
 * Piecewise Hexic (6th-Order Polynomial) LUT Activation
 *
 * Each segment is approximated as: y = c0 + c1*x + c2*x² + c3*x³ + c4*x⁴ + c5*x⁵ + c6*x⁶
 * Evaluated using Horner's scheme: y = (((((c6*x + c5)*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
 * LUT stores 7 coefficients per segment in ascending order: [c0, c1, c2, c3, c4, c5, c6]
 *
 * Benefits over Cubic:
 *   - 10-100× lower error for complex functions (gelu, erf, exp)
 *   - Can capture higher-order curvature
 *   - Better for functions with rapid variation
 *
 * Cost:
 *   - 3 additional multiplications: 6 muls instead of 3
 *   - 75% more LUT storage: 7 coefficients instead of 4
 *
 * Format: [boundaries..., c0_seg0, c1_seg0, ..., c6_seg0, c0_seg1, ...]
 */

#ifdef TRISC_MATH
namespace sfpi {

// Forward declaration
template <uint32_t LUT_SIZE>
inline void piecewise_hexic_lut(const std::array<float, LUT_SIZE>& lut);

// Specialization for LUT_SIZE=33 (5 boundaries + 4 segments * 7 coefficients)
// LUT format: [b0, b1, b2, b3, b4, a0, b0, c0, d0, e0, f0, g0, a1, ...]
template <>
inline void piecewise_hexic_lut<33>(const std::array<float, 33>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 4;

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[4]) { x_clamped = lut[4]; } v_endif;

        // Compute using Horner's scheme: (((((c6*x + c5)*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
        // Load coefficients just-in-time to minimize register pressure
        // Coefficients start at index 5 (after boundaries), load in REVERSE order for Horner

        // Load c6 coefficient (x^6 term - highest degree)
        vFloat coeff = lut[11];
        v_if (x_clamped >= lut[1]) { coeff = lut[18]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[25]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[32]; } v_endif;
        vFloat result = coeff;  // Start with c6

        // Load c5 coefficient (x^5 term)
        coeff = lut[10];
        v_if (x_clamped >= lut[1]) { coeff = lut[17]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[24]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[31]; } v_endif;
        result = result * x + coeff;  // c6*x + c5

        // Load c4 coefficient (x^4 term)
        coeff = lut[9];
        v_if (x_clamped >= lut[1]) { coeff = lut[16]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[23]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[30]; } v_endif;
        result = result * x + coeff;  // (c6*x + c5)*x + c4

        // Load c3 coefficient (x^3 term)
        coeff = lut[8];
        v_if (x_clamped >= lut[1]) { coeff = lut[15]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[22]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[29]; } v_endif;
        result = result * x + coeff;  // ((c6*x + c5)*x + c4)*x + c3

        // Load c2 coefficient (x^2 term)
        coeff = lut[7];
        v_if (x_clamped >= lut[1]) { coeff = lut[14]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[21]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[28]; } v_endif;
        result = result * x + coeff;  // (((c6*x + c5)*x + c4)*x + c3)*x + c2

        // Load c1 coefficient (x term)
        coeff = lut[6];
        v_if (x_clamped >= lut[1]) { coeff = lut[13]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[20]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[27]; } v_endif;
        result = result * x + coeff;  // ((((c6*x + c5)*x + c4)*x + c3)*x + c2)*x + c1

        // Load c0 coefficient (constant term)
        coeff = lut[5];
        v_if (x_clamped >= lut[1]) { coeff = lut[12]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[19]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[26]; } v_endif;
        result = result * x + coeff;  // (((((c6*x + c5)*x + c4)*x + c3)*x + c2)*x + c1)*x + c0

        dst_reg[d] = result;
    }
}

// Specialization for LUT_SIZE=65 (9 boundaries + 8 segments * 7 coefficients)
// LUT format: [b0...b8, c0_0, c1_0, ..., c6_0, c0_1, c1_1, ..., c6_1, ..., c0_7, c1_7, ..., c6_7]
template <>
inline void piecewise_hexic_lut<65>(const std::array<float, 65>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 8;

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[8]) { x_clamped = lut[8]; } v_endif;

        // Compute using Horner's scheme: (((((c6*x + c5)*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
        // Load coefficients just-in-time, in REVERSE order for Horner

        // Load c6 coefficient (x^6 term - highest)
        vFloat coeff = lut[15];
        v_if (x_clamped >= lut[1]) { coeff = lut[22]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[29]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[36]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[43]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[50]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[57]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[64]; } v_endif;
        vFloat result = coeff;  // Start with c6

        // Load c5 coefficient (x^5 term)
        coeff = lut[14];
        v_if (x_clamped >= lut[1]) { coeff = lut[21]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[28]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[35]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[42]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[49]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[56]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[63]; } v_endif;
        result = result * x + coeff;  // c6*x + c5

        // Load c4 coefficient (x^4 term)
        coeff = lut[13];
        v_if (x_clamped >= lut[1]) { coeff = lut[20]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[27]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[34]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[41]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[48]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[55]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[62]; } v_endif;
        result = result * x + coeff;  // (c6*x + c5)*x + c4

        // Load c3 coefficient (x^3 term)
        coeff = lut[12];
        v_if (x_clamped >= lut[1]) { coeff = lut[19]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[26]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[33]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[40]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[47]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[54]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[61]; } v_endif;
        result = result * x + coeff;  // ((c6*x + c5)*x + c4)*x + c3

        // Load c2 coefficient (x^2 term)
        coeff = lut[11];
        v_if (x_clamped >= lut[1]) { coeff = lut[18]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[25]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[32]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[39]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[46]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[53]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[60]; } v_endif;
        result = result * x + coeff;  // (((c6*x + c5)*x + c4)*x + c3)*x + c2

        // Load c1 coefficient (x term)
        coeff = lut[10];
        v_if (x_clamped >= lut[1]) { coeff = lut[17]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[24]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[31]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[38]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[45]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[52]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[59]; } v_endif;
        result = result * x + coeff;  // ((((c6*x + c5)*x + c4)*x + c3)*x + c2)*x + c1

        // Load c0 coefficient (constant term)
        coeff = lut[9];
        v_if (x_clamped >= lut[1]) { coeff = lut[16]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[23]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[30]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[37]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[44]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[51]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[58]; } v_endif;
        result = result * x + coeff;  // (((((c6*x + c5)*x + c4)*x + c3)*x + c2)*x + c1)*x + c0

        dst_reg[d] = result;
    }
}

// NOTE: Depth 16 specialization removed - use generic template instead
// The generic template uses the correct full-polynomial-per-v_if pattern
// which avoids v_if cascade limits (only 15 v_if statements vs 105)

// Generic template for all LUT_SIZE values (including 16 and 32 segments)
// LUT format with boundaries: LUT_SIZE = (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * 7) = 8*NUM_SEGMENTS + 1
template <uint32_t LUT_SIZE>
inline void piecewise_hexic_lut(const std::array<float, LUT_SIZE>& lut) {
    // Calculate NUM_SEGMENTS from LUT_SIZE with boundaries
    constexpr uint32_t NUM_SEGMENTS = (LUT_SIZE - 1) / 8;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    // Process all dest registers (32 registers for both faces)
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];

        // Clamp x to valid range using precomputed boundaries
        vFloat x_clamped = x;
        v_if (x < lut[0]) {
            x_clamped = lut[0];
        }
        v_endif;
        v_if (x > lut[NUM_SEGMENTS]) {
            x_clamped = lut[NUM_SEGMENTS];
        }
        v_endif;

        // Default: segment 0 - compute entire polynomial to avoid v_if cascade issues
        // Horner's method: start with c6 (highest degree) and work down to c0
        vFloat result = lut[COEFF_OFFSET + 6];      // c6
        result = result * x + lut[COEFF_OFFSET + 5];
        result = result * x + lut[COEFF_OFFSET + 4];
        result = result * x + lut[COEFF_OFFSET + 3];
        result = result * x + lut[COEFF_OFFSET + 2];
        result = result * x + lut[COEFF_OFFSET + 1];
        result = result * x + lut[COEFF_OFFSET];    // c0

        // Use cascaded v_if with precomputed boundaries
        // Compute full polynomial inside each v_if to avoid excessive v_if nesting
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            uint32_t lut_base = COEFF_OFFSET + (seg * 7);
            v_if (x_clamped >= lut[seg]) {
                result = lut[lut_base + 6];          // c6
                result = result * x + lut[lut_base + 5];
                result = result * x + lut[lut_base + 4];
                result = result * x + lut[lut_base + 3];
                result = result * x + lut[lut_base + 2];
                result = result * x + lut[lut_base + 1];
                result = result * x + lut[lut_base];  // c0
            }
            v_endif;
        }

        dst_reg[d] = result;
    }
}

} // namespace sfpi
#endif

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

#ifdef EMBEDDED_LUT
    // Embedded LUT mode: LUT is compiled directly into the kernel
    // Header must define: INPUT_MIN, INPUT_MAX, LUT_SIZE, LUT_DATA
    // This provides zero L1 memory overhead for the LUT
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Use embedded LUT data directly from header
    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;
#else
    // Generic LUT mode: LUT is loaded from L1 circular buffer
    // This allows runtime LUT generation and sharing across cores
    [[maybe_unused]] float input_min = get_arg_val<float>(1);
    [[maybe_unused]] float input_max = get_arg_val<float>(2);
    constexpr uint32_t LUT_SIZE = get_compile_time_arg_val(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_lut = tt::CBIndex::c_25;

    // Get LUT array from L1 memory as float array directly
    using lut_t = std::array<float, LUT_SIZE>;
    auto p_lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);
#endif

    init_sfpu(cb_in, cb_out);

#ifdef PACKER_L1_ACC
    copy_tile_init(cb_in);
#endif

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);

        // Call SFPU function directly
        #ifdef TRISC_MATH
#ifdef EMBEDDED_LUT
        sfpi::piecewise_hexic_lut<LUT_SIZE>(*p_lut);
#else
        sfpi::piecewise_hexic_lut<LUT_SIZE>(*p_lut);
#endif
        #endif

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
#ifdef PACKER_L1_ACC
        pack_reconfig_data_format(cb_out);
#endif
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }
}
}
