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
 * Piecewise Quartic (4th-Order Polynomial) LUT Activation
 *
 * Each segment is approximated as: y = c0 + c1*x + c2*x² + c3*x³ + c4*x⁴
 * Evaluated using Horner's scheme: y = ((((c4*x + c3)*x + c2)*x + c1)*x + c0
 * LUT stores 5 coefficients per segment in ascending order: [c0, c1, c2, c3, c4]
 *
 * Benefits over Cubic:
 *   - 2-3× lower error for same number of segments
 *   - Better approximation for functions with high curvature
 *   - Can match value and first two derivatives at boundaries
 *
 * Cost:
 *   - 1 additional multiplication: 4 muls instead of 3
 *   - 25% more LUT storage: 5 coefficients instead of 4
 *
 * Format: [boundaries..., c0_seg0, c1_seg0, c2_seg0, c3_seg0, c4_seg0, c0_seg1, ...]
 */

#ifdef TRISC_MATH
namespace sfpi {

// Forward declaration
template <uint32_t LUT_SIZE>
inline void piecewise_quartic_lut(const std::array<float, LUT_SIZE>& lut);

// Specialization for LUT_SIZE=25 (5 boundaries + 4 segments * 5 coefficients)
// LUT format: [b0, b1, b2, b3, b4, c0_0, c1_0, c2_0, c3_0, c4_0, ..., c0_3, c1_3, c2_3, c3_3, c4_3]
template <>
inline void piecewise_quartic_lut<25>(const std::array<float, 25>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 4;

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[4]) { x_clamped = lut[4]; } v_endif;

        // Default: segment 0 (coefficients start at index 5, load in reverse for Horner)
        vFloat c4 = lut[9], c3 = lut[8], c2 = lut[7], c1 = lut[6], c0 = lut[5];

        // Use precomputed boundaries
        v_if (x_clamped >= lut[1]) {
            c4 = lut[14]; c3 = lut[13]; c2 = lut[12]; c1 = lut[11]; c0 = lut[10];
        } v_endif;

        v_if (x_clamped >= lut[2]) {
            c4 = lut[19]; c3 = lut[18]; c2 = lut[17]; c1 = lut[16]; c0 = lut[15];
        } v_endif;

        v_if (x_clamped >= lut[3]) {
            c4 = lut[24]; c3 = lut[23]; c2 = lut[22]; c1 = lut[21]; c0 = lut[20];
        } v_endif;

        // Compute using Horner's scheme: ((((c4*x + c3)*x + c2)*x + c1)*x + c0
        vFloat result = ((((c4 * x + c3) * x + c2) * x + c1) * x + c0);
        dst_reg[d] = result;
    }
}

// Specialization for LUT_SIZE=49 (9 boundaries + 8 segments * 5 coefficients)
// LUT format: [b0...b8, c0_0, c1_0, c2_0, c3_0, c4_0, ..., c0_7, c1_7, c2_7, c3_7, c4_7]
template <>
inline void piecewise_quartic_lut<49>(const std::array<float, 49>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 8;

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[8]) { x_clamped = lut[8]; } v_endif;

        // Default: segment 0 (coefficients start at index 9, load in reverse)
        vFloat c4 = lut[13], c3 = lut[12], c2 = lut[11], c1 = lut[10], c0 = lut[9];

        // Use precomputed boundaries
        v_if (x_clamped >= lut[1]) { c4 = lut[18]; c3 = lut[17]; c2 = lut[16]; c1 = lut[15]; c0 = lut[14]; } v_endif;
        v_if (x_clamped >= lut[2]) { c4 = lut[23]; c3 = lut[22]; c2 = lut[21]; c1 = lut[20]; c0 = lut[19]; } v_endif;
        v_if (x_clamped >= lut[3]) { c4 = lut[28]; c3 = lut[27]; c2 = lut[26]; c1 = lut[25]; c0 = lut[24]; } v_endif;
        v_if (x_clamped >= lut[4]) { c4 = lut[33]; c3 = lut[32]; c2 = lut[31]; c1 = lut[30]; c0 = lut[29]; } v_endif;
        v_if (x_clamped >= lut[5]) { c4 = lut[38]; c3 = lut[37]; c2 = lut[36]; c1 = lut[35]; c0 = lut[34]; } v_endif;
        v_if (x_clamped >= lut[6]) { c4 = lut[43]; c3 = lut[42]; c2 = lut[41]; c1 = lut[40]; c0 = lut[39]; } v_endif;
        v_if (x_clamped >= lut[7]) { c4 = lut[48]; c3 = lut[47]; c2 = lut[46]; c1 = lut[45]; c0 = lut[44]; } v_endif;

        vFloat result = ((((c4 * x + c3) * x + c2) * x + c1) * x + c0);
        dst_reg[d] = result;
    }
}

// Specialization for LUT_SIZE=97 (17 boundaries + 16 segments * 5 coefficients)
// LUT format: [b0...b16, c0_0, c1_0, c2_0, c3_0, c4_0, ..., c0_15, c1_15, c2_15, c3_15, c4_15]
template <>
inline void piecewise_quartic_lut<97>(const std::array<float, 97>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 16;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;  // Skip 17 boundaries

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[NUM_SEGMENTS]) { x_clamped = lut[NUM_SEGMENTS]; } v_endif;

        // Just-in-time coefficient loading to reduce register pressure
        // Load c4 coefficient (x^4 term - highest degree)
        vFloat coeff = lut[COEFF_OFFSET + 4];
        v_if (x_clamped >= lut[1]) { coeff = lut[26]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[31]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[36]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[41]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[46]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[51]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[56]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[61]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[66]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[71]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[76]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[81]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[86]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[91]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[96]; } v_endif;
        vFloat result = coeff;  // Start with c4

        // Load c3 coefficient (x^3 term)
        coeff = lut[COEFF_OFFSET + 3];
        v_if (x_clamped >= lut[1]) { coeff = lut[25]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[30]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[35]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[40]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[45]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[50]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[55]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[60]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[65]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[70]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[75]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[80]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[85]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[90]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[95]; } v_endif;
        result = result * x + coeff;  // c4*x + c3

        // Load c2 coefficient (x^2 term)
        coeff = lut[COEFF_OFFSET + 2];
        v_if (x_clamped >= lut[1]) { coeff = lut[24]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[29]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[34]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[39]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[44]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[49]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[54]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[59]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[64]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[69]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[74]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[79]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[84]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[89]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[94]; } v_endif;
        result = result * x + coeff;  // (c4*x + c3)*x + c2

        // Load c1 coefficient (x^1 term)
        coeff = lut[COEFF_OFFSET + 1];
        v_if (x_clamped >= lut[1]) { coeff = lut[23]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[28]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[33]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[38]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[43]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[48]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[53]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[58]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[63]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[68]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[73]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[78]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[83]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[88]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[93]; } v_endif;
        result = result * x + coeff;  // ((c4*x + c3)*x + c2)*x + c1

        // Load c0 coefficient (constant term)
        coeff = lut[COEFF_OFFSET];
        v_if (x_clamped >= lut[1]) { coeff = lut[22]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[27]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[32]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[37]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[42]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[47]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[52]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[57]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[62]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[67]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[72]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[77]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[82]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[87]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[92]; } v_endif;
        result = result * x + coeff;  // (((c4*x + c3)*x + c2)*x + c1)*x + c0

        dst_reg[d] = result;
    }
}


// Generic template for other LUT_SIZE values (fallback for 32+ segments)
// LUT format with boundaries: LUT_SIZE = (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * 5) = 6*NUM_SEGMENTS + 1
template <uint32_t LUT_SIZE>
inline void piecewise_quartic_lut(const std::array<float, LUT_SIZE>& lut) {
    // Calculate NUM_SEGMENTS from LUT_SIZE with boundaries
    constexpr uint32_t NUM_SEGMENTS = (LUT_SIZE - 1) / 6;
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
        // Horner's method: ((((c4*x + c3)*x + c2)*x + c1)*x + c0
        vFloat result = lut[COEFF_OFFSET + 4];          // c4
        result = result * x + lut[COEFF_OFFSET + 3];    // c3
        result = result * x + lut[COEFF_OFFSET + 2];    // c2
        result = result * x + lut[COEFF_OFFSET + 1];    // c1
        result = result * x + lut[COEFF_OFFSET];        // c0

        // Use cascaded v_if with precomputed boundaries
        #pragma unroll
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            uint32_t lut_base = COEFF_OFFSET + (seg * 5);

            v_if (x_clamped >= lut[seg]) {
                result = lut[lut_base + 4];                  // c4
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
        sfpi::piecewise_quartic_lut<LUT_SIZE>(*p_lut);
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
