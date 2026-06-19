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
 * Piecewise Quadratic (2nd-Order Polynomial) LUT Activation
 *
 * Each segment is approximated as: y = c0 + c1*x + c2*x^2
 * Evaluated using Horner's scheme: y = (c2*x + c1)*x + c0 for better numerical stability
 * LUT stores 3 coefficients per segment in ascending degree order: [c0, c1, c2]
 *
 * For NUM_SEGMENTS segments:
 *   Format: [boundaries..., c0_seg0, c1_seg0, c2_seg0, c0_seg1, c1_seg1, c2_seg1, ...]
 */

#ifdef TRISC_MATH
namespace sfpi {

// Forward declaration
template <uint32_t LUT_SIZE>
inline void piecewise_quadratic_lut(const std::array<float, LUT_SIZE>& lut);

// Specialization for LUT_SIZE=17 (5 boundaries + 4 segments * 3 coefficients)
// LUT format: [b0, b1, b2, b3, b4, c0_0, c1_0, c2_0, c0_1, c1_1, c2_1, c0_2, c1_2, c2_2, c0_3, c1_3, c2_3]
template <>
inline void piecewise_quadratic_lut<17>(const std::array<float, 17>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 4;

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[4]) { x_clamped = lut[4]; } v_endif;

        // Default: segment 0 (coefficients start at index 5)
        // Load in reverse order for Horner's method
        vFloat c2 = lut[7];  // x^2 coefficient
        vFloat c1 = lut[6];  // x coefficient
        vFloat c0 = lut[5];  // constant

        // Use precomputed boundaries
        v_if (x_clamped >= lut[1]) {
            c2 = lut[10];
            c1 = lut[9];
            c0 = lut[8];
        } v_endif;

        v_if (x_clamped >= lut[2]) {
            c2 = lut[13];
            c1 = lut[12];
            c0 = lut[11];
        } v_endif;

        v_if (x_clamped >= lut[3]) {
            c2 = lut[16];
            c1 = lut[15];
            c0 = lut[14];
        } v_endif;

        // Compute quadratic using Horner's scheme: (c2*x + c1)*x + c0
        vFloat result = (c2 * x + c1) * x + c0;
        dst_reg[d] = result;
    }
}

// Specialization for LUT_SIZE=33 (9 boundaries + 8 segments * 3 coefficients)
// LUT format: [b0...b8, c0_0, c1_0, c2_0, ..., c0_7, c1_7, c2_7]
template <>
inline void piecewise_quadratic_lut<33>(const std::array<float, 33>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 8;

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[8]) { x_clamped = lut[8]; } v_endif;

        // Default: segment 0 (coefficients start at index 9)
        vFloat c2 = lut[11];
        vFloat c1 = lut[10];
        vFloat c0 = lut[9];

        // Use precomputed boundaries
        v_if (x_clamped >= lut[1]) { c2 = lut[14]; c1 = lut[13]; c0 = lut[12]; } v_endif;
        v_if (x_clamped >= lut[2]) { c2 = lut[17]; c1 = lut[16]; c0 = lut[15]; } v_endif;
        v_if (x_clamped >= lut[3]) { c2 = lut[20]; c1 = lut[19]; c0 = lut[18]; } v_endif;
        v_if (x_clamped >= lut[4]) { c2 = lut[23]; c1 = lut[22]; c0 = lut[21]; } v_endif;
        v_if (x_clamped >= lut[5]) { c2 = lut[26]; c1 = lut[25]; c0 = lut[24]; } v_endif;
        v_if (x_clamped >= lut[6]) { c2 = lut[29]; c1 = lut[28]; c0 = lut[27]; } v_endif;
        v_if (x_clamped >= lut[7]) { c2 = lut[32]; c1 = lut[31]; c0 = lut[30]; } v_endif;

        vFloat result = (c2 * x + c1) * x + c0;
        dst_reg[d] = result;
    }
}

// Specialization for LUT_SIZE=65 (17 boundaries + 16 segments * 3 coefficients)
// LUT format: [b0...b16, c0_0, c1_0, c2_0, ..., c0_15, c1_15, c2_15]
template <>
inline void piecewise_quadratic_lut<65>(const std::array<float, 65>& lut) {
    constexpr uint32_t NUM_SEGMENTS = 16;

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[16]) { x_clamped = lut[16]; } v_endif;

        // Just-in-time coefficient loading to reduce register pressure
        // Load c2 coefficient (x^2 term - highest degree)
        vFloat coeff = lut[19];
        v_if (x_clamped >= lut[1]) { coeff = lut[22]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[25]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[28]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[31]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[34]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[37]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[40]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[43]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[46]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[49]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[52]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[55]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[58]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[61]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[64]; } v_endif;
        vFloat result = coeff;  // Start with c2

        // Load c1 coefficient (x^1 term)
        coeff = lut[18];
        v_if (x_clamped >= lut[1]) { coeff = lut[21]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[24]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[27]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[30]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[33]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[36]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[39]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[42]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[45]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[48]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[51]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[54]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[57]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[60]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[63]; } v_endif;
        result = result * x + coeff;  // c2*x + c1

        // Load c0 coefficient (constant term)
        coeff = lut[17];
        v_if (x_clamped >= lut[1]) { coeff = lut[20]; } v_endif;
        v_if (x_clamped >= lut[2]) { coeff = lut[23]; } v_endif;
        v_if (x_clamped >= lut[3]) { coeff = lut[26]; } v_endif;
        v_if (x_clamped >= lut[4]) { coeff = lut[29]; } v_endif;
        v_if (x_clamped >= lut[5]) { coeff = lut[32]; } v_endif;
        v_if (x_clamped >= lut[6]) { coeff = lut[35]; } v_endif;
        v_if (x_clamped >= lut[7]) { coeff = lut[38]; } v_endif;
        v_if (x_clamped >= lut[8]) { coeff = lut[41]; } v_endif;
        v_if (x_clamped >= lut[9]) { coeff = lut[44]; } v_endif;
        v_if (x_clamped >= lut[10]) { coeff = lut[47]; } v_endif;
        v_if (x_clamped >= lut[11]) { coeff = lut[50]; } v_endif;
        v_if (x_clamped >= lut[12]) { coeff = lut[53]; } v_endif;
        v_if (x_clamped >= lut[13]) { coeff = lut[56]; } v_endif;
        v_if (x_clamped >= lut[14]) { coeff = lut[59]; } v_endif;
        v_if (x_clamped >= lut[15]) { coeff = lut[62]; } v_endif;
        result = result * x + coeff;  // (c2*x + c1)*x + c0

        dst_reg[d] = result;
    }
}

// Generic template for other LUT_SIZE values (fallback for 32+ segments)
// LUT format with boundaries: LUT_SIZE = (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * 3) = 4*NUM_SEGMENTS + 1
template <uint32_t LUT_SIZE>
inline void piecewise_quadratic_lut(const std::array<float, LUT_SIZE>& lut) {
    // Calculate NUM_SEGMENTS from LUT_SIZE with boundaries
    constexpr uint32_t NUM_SEGMENTS = (LUT_SIZE - 1) / 4;
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

        // Initialize with first segment coefficients (after boundaries)
        // Load in reverse order for Horner's method
        vFloat c2 = lut[COEFF_OFFSET + 2];  // x^2 coefficient
        vFloat c1 = lut[COEFF_OFFSET + 1];  // x coefficient
        vFloat c0 = lut[COEFF_OFFSET];      // constant

        // Use cascaded v_if to select appropriate segment coefficients using precomputed boundaries
        #pragma unroll
        for (uint32_t seg = 1; seg < NUM_SEGMENTS; seg++) {
            uint32_t lut_base = COEFF_OFFSET + (seg * 3);

            v_if (x_clamped >= lut[seg]) {
                c2 = lut[lut_base + 2];  // x^2 coefficient
                c1 = lut[lut_base + 1];  // x coefficient
                c0 = lut[lut_base + 0];  // constant
            }
            v_endif;
        }

        // Compute quadratic using Horner's scheme: y = (c2*x + c1)*x + c0
        // Horner's method reduces operations and improves numerical stability
        vFloat result = (c2 * x + c1) * x + c0;

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
        sfpi::piecewise_quadratic_lut<LUT_SIZE>(*p_lut);
#else
        sfpi::piecewise_quadratic_lut<LUT_SIZE>(*p_lut);
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
