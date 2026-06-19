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
 * Piecewise-linear LUT activation - Unified for embedded and generic LUT loading
 *
 * For piecewise linear: LUT_SIZE = (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * 2)
 *   - (NUM_SEGMENTS + 1) precomputed boundary values
 *   - NUM_SEGMENTS * 2 coefficients (c0=intercept, c1=slope per segment)
 *
 * Each segment uses polynomial form: y = c0 + c1*x (standard ascending degree order)
 *
 * Table depths supported: 4, 8, 16, 32, 64+ segments (any power of 2)
 * LUT format: [b0, b1, ..., b_N, c0_seg0, c1_seg0, c0_seg1, c1_seg1, ..., c0_segN-1, c1_segN-1]
 * Input range: Boundaries are precomputed offline to avoid floating-point accumulation errors
 *
 * Note: Precomputed boundaries eliminate dynamic boundary calculation errors.
 * The #pragma unroll directive ensures the loop is fully unrolled at compile-time.
 */

#ifdef TRISC_MATH
namespace sfpi {

// Forward declaration
template <uint32_t NUM_SEGMENTS>
inline void piecewise_linear_activation(const std::array<float, (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * 2)>& lut);

// Specialization for NUM_SEGMENTS=4
// LUT format: [b0, b1, b2, b3, b4, c0_0, c1_0, c0_1, c1_1, c0_2, c1_2, c0_3, c1_3]
// Total: 13 values (5 boundaries + 8 coefficients)
template <>
inline void piecewise_linear_activation<4>(const std::array<float, 13>& lut) {
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[4]) { x_clamped = lut[4]; } v_endif;

        // Default: segment 0 (coefficients start at index 5)
        vFloat c0 = lut[5];  // intercept
        vFloat c1 = lut[6];  // slope

        // Use precomputed boundaries
        v_if (x_clamped >= lut[1]) {
            c0 = lut[7];
            c1 = lut[8];
        } v_endif;

        v_if (x_clamped >= lut[2]) {
            c0 = lut[9];
            c1 = lut[10];
        } v_endif;

        v_if (x_clamped >= lut[3]) {
            c0 = lut[11];
            c1 = lut[12];
        } v_endif;

        // Compute result: y = c0 + c1*x
        vFloat result = c0 + c1 * x;
        dst_reg[d] = result;
    }
}

// Specialization for NUM_SEGMENTS=8
// LUT format: [b0...b8, c0_0, c1_0, ..., c0_7, c1_7]
// Total: 25 values (9 boundaries + 16 coefficients)
template <>
inline void piecewise_linear_activation<8>(const std::array<float, 25>& lut) {
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[8]) { x_clamped = lut[8]; } v_endif;

        // Default: segment 0 (coefficients start at index 9)
        vFloat c0 = lut[9];   // intercept
        vFloat c1 = lut[10];  // slope

        // Use precomputed boundaries
        v_if (x_clamped >= lut[1]) { c0 = lut[11]; c1 = lut[12]; } v_endif;
        v_if (x_clamped >= lut[2]) { c0 = lut[13]; c1 = lut[14]; } v_endif;
        v_if (x_clamped >= lut[3]) { c0 = lut[15]; c1 = lut[16]; } v_endif;
        v_if (x_clamped >= lut[4]) { c0 = lut[17]; c1 = lut[18]; } v_endif;
        v_if (x_clamped >= lut[5]) { c0 = lut[19]; c1 = lut[20]; } v_endif;
        v_if (x_clamped >= lut[6]) { c0 = lut[21]; c1 = lut[22]; } v_endif;
        v_if (x_clamped >= lut[7]) { c0 = lut[23]; c1 = lut[24]; } v_endif;

        vFloat result = c0 + c1 * x;
        dst_reg[d] = result;
    }
}

// Specialization for NUM_SEGMENTS=16
// LUT format: [b0...b16, c0_0, c1_0, ..., c0_15, c1_15]
// Total: 49 values (17 boundaries + 32 coefficients)
template <>
inline void piecewise_linear_activation<16>(const std::array<float, 49>& lut) {
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < lut[0]) { x_clamped = lut[0]; } v_endif;
        v_if (x > lut[16]) { x_clamped = lut[16]; } v_endif;

        // Default: segment 0 (coefficients start at index 17)
        vFloat c0 = lut[17];  // intercept
        vFloat c1 = lut[18];  // slope

        // Use precomputed boundaries
        v_if (x_clamped >= lut[1]) { c0 = lut[19]; c1 = lut[20]; } v_endif;
        v_if (x_clamped >= lut[2]) { c0 = lut[21]; c1 = lut[22]; } v_endif;
        v_if (x_clamped >= lut[3]) { c0 = lut[23]; c1 = lut[24]; } v_endif;
        v_if (x_clamped >= lut[4]) { c0 = lut[25]; c1 = lut[26]; } v_endif;
        v_if (x_clamped >= lut[5]) { c0 = lut[27]; c1 = lut[28]; } v_endif;
        v_if (x_clamped >= lut[6]) { c0 = lut[29]; c1 = lut[30]; } v_endif;
        v_if (x_clamped >= lut[7]) { c0 = lut[31]; c1 = lut[32]; } v_endif;
        v_if (x_clamped >= lut[8]) { c0 = lut[33]; c1 = lut[34]; } v_endif;
        v_if (x_clamped >= lut[9]) { c0 = lut[35]; c1 = lut[36]; } v_endif;
        v_if (x_clamped >= lut[10]) { c0 = lut[37]; c1 = lut[38]; } v_endif;
        v_if (x_clamped >= lut[11]) { c0 = lut[39]; c1 = lut[40]; } v_endif;
        v_if (x_clamped >= lut[12]) { c0 = lut[41]; c1 = lut[42]; } v_endif;
        v_if (x_clamped >= lut[13]) { c0 = lut[43]; c1 = lut[44]; } v_endif;
        v_if (x_clamped >= lut[14]) { c0 = lut[45]; c1 = lut[46]; } v_endif;
        v_if (x_clamped >= lut[15]) { c0 = lut[47]; c1 = lut[48]; } v_endif;

        vFloat result = c0 + c1 * x;
        dst_reg[d] = result;
    }
}

// Generic template for other NUM_SEGMENTS values (fallback for 32+)
// LUT format: [b0...b_N, c0_0, c1_0, ..., c0_(N-1), c1_(N-1)]
// Total: (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * 2) values
template <uint32_t NUM_SEGMENTS>
inline void piecewise_linear_activation(const std::array<float, (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * 2)>& lut) {
    // Process all dest registers (32 registers for full tile)
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

        // Default: use first segment (coefficients start after all boundaries)
        constexpr uint32_t coeff_offset = NUM_SEGMENTS + 1;
        vFloat c0 = lut[coeff_offset];      // intercept
        vFloat c1 = lut[coeff_offset + 1];  // slope
        vFloat result = c0 + c1 * x;

        // Generate cascaded v_if statements for each segment using precomputed boundaries
        #pragma unroll
        for (uint32_t i = 1; i < NUM_SEGMENTS; i++) {
            uint32_t lut_idx = coeff_offset + (i * 2);  // Each segment has [c0, c1]

            v_if (x_clamped >= lut[i]) {
                c0 = lut[lut_idx];
                c1 = lut[lut_idx + 1];
                result = c0 + c1 * x_clamped;
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
    // Embedded LUTs: constants from generated header
    // Header must be included before this file and define:
    // - constexpr float INPUT_MIN
    // - constexpr float INPUT_MAX
    // - constexpr uint32_t LUT_SIZE
    // - constexpr std::array<float, LUT_SIZE> LUT_DATA
    constexpr float input_min = INPUT_MIN;
    constexpr float input_max = INPUT_MAX;
    // For piecewise linear with boundaries: LUT_SIZE = (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * 2) = 3*NUM_SEGMENTS + 1
    // Therefore: NUM_SEGMENTS = (LUT_SIZE - 1) / 3
    constexpr uint32_t NUM_SEGMENTS = (LUT_SIZE - 1) / 3;

    // Use embedded LUT data directly (zero L1 overhead)
    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;
#else
    // Generic LUT loading: read from L1 memory via circular buffer
    // Note: input_min/input_max are read but not used by polynomial kernels
    // Polynomial kernels use precomputed LUT boundaries instead
    [[maybe_unused]] float input_min = get_arg_val<float>(1);
    [[maybe_unused]] float input_max = get_arg_val<float>(2);

    // Read LUT_SIZE from compile time args
    constexpr uint32_t LUT_SIZE = get_compile_time_arg_val(0);
    // For piecewise linear with boundaries: LUT_SIZE = (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * 2) = 3*NUM_SEGMENTS + 1
    // Therefore: NUM_SEGMENTS = (LUT_SIZE - 1) / 3
    constexpr uint32_t NUM_SEGMENTS = (LUT_SIZE - 1) / 3;

    constexpr auto cb_lut = tt::CBIndex::c_25;

    // Get LUT array from L1 memory as float array
    using lut_t = std::array<float, LUT_SIZE>;
    auto p_lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);
#endif

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_in, cb_out);

#ifdef PACKER_L1_ACC
    copy_tile_init(cb_in);
#endif

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);

        // Call SFPU piecewise-linear function
        #ifdef TRISC_MATH
        sfpi::piecewise_linear_activation<NUM_SEGMENTS>(*p_lut);
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
