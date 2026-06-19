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
 * Piecewise Octic (8th-Order Polynomial) LUT Activation
 *
 * Each segment is approximated as: y = c0 + c1*x + c2*x² + c3*x³ + c4*x⁴ + c5*x⁵ + c6*x⁶ + c7*x⁷ + c8*x⁸
 * Evaluated using Horner's scheme: y = ((((((((c8*x + c7)*x + c6)*x + c5)*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
 * LUT stores 9 coefficients per segment in ascending order: [c0, c1, c2, c3, c4, c5, c6, c7, c8]
 *
 * Benefits over Hexic:
 *   - Potentially lower error for extremely complex functions
 *   - Can capture even higher-order curvature
 *
 * Cost:
 *   - 2 additional multiplications: 8 muls instead of 6
 *   - 29% more LUT storage: 9 coefficients instead of 7
 *
 * Expected outcome:
 *   - Diminishing returns compared to hexic
 *   - Use only if hexic accuracy is insufficient
 *
 * Format: [boundaries..., c0_seg0, c1_seg0, c2_seg0, c3_seg0, c4_seg0, c5_seg0, c6_seg0, c7_seg0, c8_seg0, c0_seg1, ...]
 */

#ifdef TRISC_MATH
namespace sfpi {

// Forward declaration
template <uint32_t LUT_SIZE>
inline void piecewise_octic_lut(const std::array<float, LUT_SIZE>& lut);

// Specialization for LUT_SIZE=41 (5 boundaries + 4 segments * 9 coefficients)
// LUT format: [b0, b1, b2, b3, b4, c0_0, c1_0, c2_0, c3_0, c4_0, c5_0, c6_0, c7_0, c8_0, ..., c0_3, c1_3, c2_3, c3_3, c4_3, c5_3, c6_3, c7_3, c8_3]
template <>
inline void piecewise_octic_lut<41>(const std::array<float, 41>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 4;

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[4]) { x_clamped = lut[4]; } v_endif;

        // Compute using Horner's scheme with just-in-time coefficient loading to reduce register pressure
        // Load c8 coefficient (x^8 term - highest degree)
        vFloat coeff = lut[13];
        v_if (x_clamped >= lut[1]) { coeff = lut[22]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[31]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[40]; } v_endif;
        vFloat result = coeff;  // Start with c8

        // Load c7 coefficient (x^7 term)
        coeff = lut[12];
        v_if (x_clamped >= lut[1]) { coeff = lut[21]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[30]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[39]; } v_endif;
        result = result * x + coeff;  // c8*x + c7

        // Load c6 coefficient (x^6 term)
        coeff = lut[11];
        v_if (x_clamped >= lut[1]) { coeff = lut[20]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[29]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[38]; } v_endif;
        result = result * x + coeff;  // (c8*x + c7)*x + c6

        // Load c5 coefficient (x^5 term)
        coeff = lut[10];
        v_if (x_clamped >= lut[1]) { coeff = lut[19]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[28]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[37]; } v_endif;
        result = result * x + coeff;  // ((c8*x + c7)*x + c6)*x + c5

        // Load c4 coefficient (x^4 term)
        coeff = lut[9];
        v_if (x_clamped >= lut[1]) { coeff = lut[18]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[27]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[36]; } v_endif;
        result = result * x + coeff;  // (((c8*x + c7)*x + c6)*x + c5)*x + c4

        // Load c3 coefficient (x^3 term)
        coeff = lut[8];
        v_if (x_clamped >= lut[1]) { coeff = lut[17]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[26]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[35]; } v_endif;
        result = result * x + coeff;  // ((((c8*x + c7)*x + c6)*x + c5)*x + c4)*x + c3

        // Load c2 coefficient (x^2 term)
        coeff = lut[7];
        v_if (x_clamped >= lut[1]) { coeff = lut[16]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[25]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[34]; } v_endif;
        result = result * x + coeff;  // (((((c8*x + c7)*x + c6)*x + c5)*x + c4)*x + c3)*x + c2

        // Load c1 coefficient (x term)
        coeff = lut[6];
        v_if (x_clamped >= lut[1]) { coeff = lut[15]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[24]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[33]; } v_endif;
        result = result * x + coeff;  // ((((((c8*x + c7)*x + c6)*x + c5)*x + c4)*x + c3)*x + c2)*x + c1

        // Load c0 coefficient (constant term)
        coeff = lut[5];
        v_if (x_clamped >= lut[1]) { coeff = lut[14]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[23]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[32]; } v_endif;
        result = result * x + coeff;  // (((((((c8*x + c7)*x + c6)*x + c5)*x + c4)*x + c3)*x + c2)*x + c1)*x + c0

        dst_reg[d] = result;
    }
}

// Specialization for LUT_SIZE=81 (9 boundaries + 8 segments * 9 coefficients)
// LUT format: [b0...b8, c0_0, c1_0, c2_0, c3_0, c4_0, c5_0, c6_0, c7_0, c8_0, ..., c0_7, c1_7, c2_7, c3_7, c4_7, c5_7, c6_7, c7_7, c8_7]
template <>
inline void piecewise_octic_lut<81>(const std::array<float, 81>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 8;

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[8]) { x_clamped = lut[8]; } v_endif;

        // Just-in-time coefficient loading to reduce register pressure
        // Load c8 (x^8 term - highest degree)
        vFloat coeff = lut[17];
        v_if (x_clamped >= lut[1]) { coeff = lut[26]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[35]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[44]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[53]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[62]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[71]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[80]; } v_endif;
        vFloat result = coeff;

        // Load c7 (x^7 term)
        coeff = lut[16];
        v_if (x_clamped >= lut[1]) { coeff = lut[25]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[34]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[43]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[52]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[61]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[70]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[79]; } v_endif;
        result = result * x + coeff;

        // Load c6 (x^6 term)
        coeff = lut[15];
        v_if (x_clamped >= lut[1]) { coeff = lut[24]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[33]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[42]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[51]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[60]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[69]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[78]; } v_endif;
        result = result * x + coeff;

        // Load c5 (x^5 term)
        coeff = lut[14];
        v_if (x_clamped >= lut[1]) { coeff = lut[23]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[32]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[41]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[50]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[59]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[68]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[77]; } v_endif;
        result = result * x + coeff;

        // Load c4 (x^4 term)
        coeff = lut[13];
        v_if (x_clamped >= lut[1]) { coeff = lut[22]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[31]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[40]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[49]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[58]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[67]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[76]; } v_endif;
        result = result * x + coeff;

        // Load c3 (x^3 term)
        coeff = lut[12];
        v_if (x_clamped >= lut[1]) { coeff = lut[21]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[30]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[39]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[48]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[57]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[66]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[75]; } v_endif;
        result = result * x + coeff;

        // Load c2 (x^2 term)
        coeff = lut[11];
        v_if (x_clamped >= lut[1]) { coeff = lut[20]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[29]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[38]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[47]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[56]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[65]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[74]; } v_endif;
        result = result * x + coeff;

        // Load c1 (x term)
        coeff = lut[10];
        v_if (x_clamped >= lut[1]) { coeff = lut[19]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[28]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[37]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[46]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[55]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[64]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[73]; } v_endif;
        result = result * x + coeff;

        // Load c0 (constant term)
        coeff = lut[9];
        v_if (x_clamped >= lut[1]) { coeff = lut[18]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[27]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[36]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[45]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[54]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[63]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[72]; } v_endif;
        result = result * x + coeff;
        dst_reg[d] = result;
    }
}

// Generic template for all LUT_SIZE values (including 16 and 32 segments)
// LUT format with boundaries: LUT_SIZE = (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * 9) = 10*NUM_SEGMENTS + 1
template <uint32_t LUT_SIZE>
inline void piecewise_octic_lut(const std::array<float, LUT_SIZE>& lut) {
    // Calculate NUM_SEGMENTS from LUT_SIZE with boundaries
    constexpr uint32_t NUM_SEGMENTS = (LUT_SIZE - 1) / 10;
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

        // Default: segment 0 (coefficients start after boundaries)
        // Load coefficients in reverse order for Horner's method: c8, c7, c6, c5, c4, c3, c2, c1, c0
        vFloat result = lut[COEFF_OFFSET + 8];          // c8
        result = result * x + lut[COEFF_OFFSET + 7];    // c7
        result = result * x + lut[COEFF_OFFSET + 6];    // c6
        result = result * x + lut[COEFF_OFFSET + 5];    // c5
        result = result * x + lut[COEFF_OFFSET + 4];    // c4
        result = result * x + lut[COEFF_OFFSET + 3];    // c3
        result = result * x + lut[COEFF_OFFSET + 2];    // c2
        result = result * x + lut[COEFF_OFFSET + 1];    // c1
        result = result * x + lut[COEFF_OFFSET];        // c0

        // Use cascaded v_if with precomputed boundaries
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            uint32_t lut_base = COEFF_OFFSET + (seg * 9);

            v_if (x_clamped >= lut[seg]) {
                // Load coefficients in reverse order: c8, c7, c6, c5, c4, c3, c2, c1, c0
                result = lut[lut_base + 8];                  // c8
                result = result * x + lut[lut_base + 7];    // c7
                result = result * x + lut[lut_base + 6];    // c6
                result = result * x + lut[lut_base + 5];    // c5
                result = result * x + lut[lut_base + 4];    // c4
                result = result * x + lut[lut_base + 3];    // c3
                result = result * x + lut[lut_base + 2];    // c2
                result = result * x + lut[lut_base + 1];    // c1
                result = result * x + lut[lut_base];        // c0
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
        sfpi::piecewise_octic_lut<LUT_SIZE>(*p_lut);
#else
        sfpi::piecewise_octic_lut<LUT_SIZE>(*p_lut);
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
