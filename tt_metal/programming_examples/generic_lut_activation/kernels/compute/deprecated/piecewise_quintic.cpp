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
 * Piecewise Quintic (5th-Order Polynomial) LUT Activation
 *
 * Each segment is approximated as: y = c0 + c1*x + c2*x² + c3*x³ + c4*x⁴ + c5*x⁵
 * Evaluated using Horner's scheme: y = (((((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
 * LUT stores 6 coefficients per segment in ascending order: [c0, c1, c2, c3, c4, c5]
 *
 * Benefits over Cubic:
 *   - 2-3× lower error for same number of segments
 *   - Better approximation for high-curvature functions
 *   - Can match value, 1st, and 2nd derivatives at boundaries
 *
 * Cost:
 *   - 2 additional multiplications: 5 muls instead of 3
 *   - 50% more LUT storage: 6 coefficients instead of 4
 *
 * Format: [boundaries..., c0_seg0, c1_seg0, c2_seg0, c3_seg0, c4_seg0, c5_seg0, c0_seg1, ...]
 */

#ifdef TRISC_MATH
namespace sfpi {

// Forward declaration
template <uint32_t LUT_SIZE>
inline void piecewise_quintic_lut(const std::array<float, LUT_SIZE>& lut);

// Specialization for LUT_SIZE=29 (5 boundaries + 4 segments * 6 coefficients)
// LUT format: [b0, b1, b2, b3, b4, c0_0, c1_0, c2_0, c3_0, c4_0, c5_0, ..., c0_3, c1_3, c2_3, c3_3, c4_3, c5_3]
template <>
inline void piecewise_quintic_lut<29>(const std::array<float, 29>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 4;

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[4]) { x_clamped = lut[4]; } v_endif;

        // Default: segment 0 (coefficients start at index 5, load in reverse for Horner)
        vFloat c5 = lut[10], c4 = lut[9], c3 = lut[8], c2 = lut[7], c1 = lut[6], c0 = lut[5];

        // Use precomputed boundaries
        v_if (x_clamped >= lut[1]) {
            c5 = lut[16]; c4 = lut[15]; c3 = lut[14]; c2 = lut[13]; c1 = lut[12]; c0 = lut[11];
        } v_endif;

        v_if (x_clamped >= lut[2]) {
            c5 = lut[22]; c4 = lut[21]; c3 = lut[20]; c2 = lut[19]; c1 = lut[18]; c0 = lut[17];
        } v_endif;

        v_if (x_clamped >= lut[3]) {
            c5 = lut[28]; c4 = lut[27]; c3 = lut[26]; c2 = lut[25]; c1 = lut[24]; c0 = lut[23];
        } v_endif;

        // Compute using Horner's scheme: (((((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
        vFloat result = (((((c5 * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
        dst_reg[d] = result;
    }
}

// Specialization for LUT_SIZE=57 (9 boundaries + 8 segments * 6 coefficients)
// LUT format: [b0...b8, c0_0, c1_0, c2_0, c3_0, c4_0, c5_0, ..., c0_7, c1_7, c2_7, c3_7, c4_7, c5_7]
template <>
inline void piecewise_quintic_lut<57>(const std::array<float, 57>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 8;

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[8]) { x_clamped = lut[8]; } v_endif;

        // Default: segment 0 (coefficients start at index 9, load in reverse)
        vFloat c5 = lut[14], c4 = lut[13], c3 = lut[12], c2 = lut[11], c1 = lut[10], c0 = lut[9];

        // Use precomputed boundaries
        v_if (x_clamped >= lut[1]) { c5 = lut[20]; c4 = lut[19]; c3 = lut[18]; c2 = lut[17]; c1 = lut[16]; c0 = lut[15]; } v_endif;
        v_if (x_clamped >= lut[2]) { c5 = lut[26]; c4 = lut[25]; c3 = lut[24]; c2 = lut[23]; c1 = lut[22]; c0 = lut[21]; } v_endif;
        v_if (x_clamped >= lut[3]) { c5 = lut[32]; c4 = lut[31]; c3 = lut[30]; c2 = lut[29]; c1 = lut[28]; c0 = lut[27]; } v_endif;
        v_if (x_clamped >= lut[4]) { c5 = lut[38]; c4 = lut[37]; c3 = lut[36]; c2 = lut[35]; c1 = lut[34]; c0 = lut[33]; } v_endif;
        v_if (x_clamped >= lut[5]) { c5 = lut[44]; c4 = lut[43]; c3 = lut[42]; c2 = lut[41]; c1 = lut[40]; c0 = lut[39]; } v_endif;
        v_if (x_clamped >= lut[6]) { c5 = lut[50]; c4 = lut[49]; c3 = lut[48]; c2 = lut[47]; c1 = lut[46]; c0 = lut[45]; } v_endif;
        v_if (x_clamped >= lut[7]) { c5 = lut[56]; c4 = lut[55]; c3 = lut[54]; c2 = lut[53]; c1 = lut[52]; c0 = lut[51]; } v_endif;

        vFloat result = (((((c5 * x + c4) * x + c3) * x + c2) * x + c1) * x + c0);
        dst_reg[d] = result;
    }
}

// Specialization for LUT_SIZE=113 (17 boundaries + 16 segments * 6 coefficients)
// LUT format: [b0...b16, c0_0, c1_0, c2_0, c3_0, c4_0, c5_0, ..., c0_15, c1_15, c2_15, c3_15, c4_15, c5_15]
template <>
inline void piecewise_quintic_lut<113>(const std::array<float, 113>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 16;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;  // Skip 17 boundaries

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[NUM_SEGMENTS]) { x_clamped = lut[NUM_SEGMENTS]; } v_endif;

        // Just-in-time coefficient loading to reduce register pressure
        // Load c5 coefficient (x^5 term - highest degree)
        vFloat coeff = lut[COEFF_OFFSET + 5];
        v_if (x_clamped >= lut[1]) { coeff = lut[28]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[34]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[40]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[46]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[52]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[58]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[64]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[70]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[76]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[82]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[88]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[94]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[100]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[106]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[112]; } v_endif;
        vFloat result = coeff;  // Start with c5

        // Load c4 coefficient (x^4 term)
        coeff = lut[COEFF_OFFSET + 4];
        v_if (x_clamped >= lut[1]) { coeff = lut[27]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[33]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[39]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[45]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[51]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[57]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[63]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[69]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[75]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[81]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[87]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[93]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[99]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[105]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[111]; } v_endif;
        result = result * x + coeff;  // c5*x + c4

        // Load c3 coefficient (x^3 term)
        coeff = lut[COEFF_OFFSET + 3];
        v_if (x_clamped >= lut[1]) { coeff = lut[26]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[32]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[38]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[44]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[50]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[56]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[62]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[68]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[74]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[80]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[86]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[92]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[98]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[104]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[110]; } v_endif;
        result = result * x + coeff;  // (c5*x + c4)*x + c3

        // Load c2 coefficient (x^2 term)
        coeff = lut[COEFF_OFFSET + 2];
        v_if (x_clamped >= lut[1]) { coeff = lut[25]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[31]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[37]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[43]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[49]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[55]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[61]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[67]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[73]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[79]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[85]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[91]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[97]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[103]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[109]; } v_endif;
        result = result * x + coeff;  // ((c5*x + c4)*x + c3)*x + c2

        // Load c1 coefficient (x^1 term)
        coeff = lut[COEFF_OFFSET + 1];
        v_if (x_clamped >= lut[1]) { coeff = lut[24]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[30]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[36]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[42]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[48]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[54]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[60]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[66]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[72]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[78]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[84]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[90]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[96]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[102]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[108]; } v_endif;
        result = result * x + coeff;  // (((c5*x + c4)*x + c3)*x + c2)*x + c1

        // Load c0 coefficient (constant term)
        coeff = lut[COEFF_OFFSET];
        v_if (x_clamped >= lut[1]) { coeff = lut[23]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[29]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[35]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[41]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[47]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[53]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[59]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[65]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[71]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[77]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[83]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[89]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[95]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[101]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[107]; } v_endif;
        result = result * x + coeff;  // ((((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0

        dst_reg[d] = result;
    }
}


// Generic template for other LUT_SIZE values (fallback for 32+ segments)
// LUT format with boundaries: LUT_SIZE = (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * 6) = 7*NUM_SEGMENTS + 1
template <uint32_t LUT_SIZE>
inline void piecewise_quintic_lut(const std::array<float, LUT_SIZE>& lut) {
    // Calculate NUM_SEGMENTS from LUT_SIZE with boundaries
    constexpr uint32_t NUM_SEGMENTS = (LUT_SIZE - 1) / 7;
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

        // Default: segment 0 (coefficients start after boundaries, load in reverse for Horner)
        // Horner's method: (((((c5*x + c4)*x + c3)*x + c2)*x + c1)*x + c0
        vFloat result = lut[COEFF_OFFSET + 5];          // c5
        result = result * x + lut[COEFF_OFFSET + 4];    // c4
        result = result * x + lut[COEFF_OFFSET + 3];    // c3
        result = result * x + lut[COEFF_OFFSET + 2];    // c2
        result = result * x + lut[COEFF_OFFSET + 1];    // c1
        result = result * x + lut[COEFF_OFFSET];        // c0

        // Use cascaded v_if with precomputed boundaries
        #pragma unroll
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            uint32_t lut_base = COEFF_OFFSET + (seg * 6);

            v_if (x_clamped >= lut[seg]) {
                result = lut[lut_base + 5];                  // c5
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
    // Note: input_min/input_max are read but not used by polynomial kernels
    // Polynomial kernels use precomputed LUT boundaries instead
    [[maybe_unused]] float input_min = get_arg_val<float>(1);
    [[maybe_unused]] float input_max = get_arg_val<float>(2);
    constexpr uint32_t LUT_SIZE = get_compile_time_arg_val(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_lut = tt::CBIndex::c_25;

    // Get LUT array from L1 memory as float array directly
    using lut_t = std::array<float, LUT_SIZE>;
    auto p_lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);

    init_sfpu(cb_in, cb_out);

#ifdef PACKER_L1_ACC
    copy_tile_init(cb_in);  // Reconfigure unpacker for fp32
#endif

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);

        // Call SFPU function directly
        #ifdef TRISC_MATH
        sfpi::piecewise_quintic_lut<LUT_SIZE>(*p_lut);
        #endif

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
#ifdef PACKER_L1_ACC
        pack_reconfig_data_format(cb_out);  // Reconfigure packer for fp32
#endif
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }
}
}
