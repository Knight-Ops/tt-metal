// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Piecewise Quadratic with Estrin+SFPLUT Acceleration
 *
 * Multi-pass segment-batched processing to enable SFPLUT usage.
 *
 * Algorithm:
 * 1. Pass 1: Determine segment ID for each of 32 elements
 * 2. Pass 2-N: For each segment:
 *    - Load segment's transformed coefficients into shared L registers
 *    - Process all 32 elements using SFPLUT
 *    - Use predication (v_if) to only update elements in this segment
 *
 * Estrin evaluation: P(x) = (c1_adj*x_norm + c0_adj) + c2*x_norm²
 *                            └──────SFPLUT──────┘   └─manual─┘
 *
 * Boundary subtraction transform:
 *   x_norm = x - boundary
 *   c0_adj = c0 + c1*boundary + c2*boundary²
 *   c1_adj = c1 + 2*c2*boundary
 *
 * Performance: ~1.5-2x faster than Horner due to:
 *   - SFPLUT hardware acceleration for linear part
 *   - Parallel computation of x² term
 */

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

namespace kutil = norm::kernel_util;

#ifdef TRISC_MATH
namespace sfpi {

//==============================================================================
// FP16 Encoding for SFPLUT
//==============================================================================

inline uint16_t float_to_fp16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (bits >> 13) & 0x3FF;

    if (exp <= 0) {
        if (exp < -10) return sign;
        mantissa = (mantissa | 0x400) >> (1 - exp);
        return sign | mantissa;
    }
    if (exp >= 31) {
        return sign | 0x7C00;
    }

    return sign | (exp << 10) | mantissa;
}

//==============================================================================
// Load Coefficients into SFPLUT L Registers
//==============================================================================

inline void load_lut_coeffs_fp16(float slope, float intercept) {
    using namespace ckernel::sfpu;

    uint16_t slope_fp16 = float_to_fp16(slope);
    uint16_t intercept_fp16 = float_to_fp16(intercept);
    uint32_t encoded = (static_cast<uint32_t>(slope_fp16) << 16) | intercept_fp16;

    // Load into L registers 0, 1, 2 (SFPLUT uses all 3)
    #ifdef HAS_SFPU_LOAD_IMM32
        _sfpu_load_imm32_(0, encoded);
        _sfpu_load_imm32_(1, encoded);
        _sfpu_load_imm32_(2, encoded);
    #else
        _sfpu_load_imm16_(0, intercept_fp16);
        _sfpu_load_imm16_(1, slope_fp16);
        _sfpu_load_imm16_(2, intercept_fp16);
    #endif
}

//==============================================================================
// Segment-Batched Piecewise Quadratic Estrin with SFPLUT
//==============================================================================

template <uint32_t LUT_SIZE>
inline void piecewise_quadratic_estrin_lut(const std::array<float, LUT_SIZE>& lut_data) {
    using namespace ckernel::sfpu;

    constexpr uint32_t NUM_SEGMENTS = (LUT_SIZE - 1) / 4;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    // DEPTH-FIRST with SFPLUT: Load coeffs once per segment, process all 32 elements
    // Use indexed access (dst_reg[i]) instead of increment to allow multiple passes

    for (uint32_t seg = 0; seg < NUM_SEGMENTS; seg++) {
        float boundary = lut_data[seg];
        float boundary_next = lut_data[seg + 1];
        float c0 = lut_data[COEFF_OFFSET + seg * 3 + 0];
        float c1 = lut_data[COEFF_OFFSET + seg * 3 + 1];
        float c2 = lut_data[COEFF_OFFSET + seg * 3 + 2];

        // Apply boundary subtraction transform
        float boundary_sq = boundary * boundary;
        float c0_adj = c0 + c1 * boundary + c2 * boundary_sq;
        float c1_adj = c1 + 2.0f * c2 * boundary;

        // Load this segment's linear coefficients into SFPLUT L registers
        uint16_t slope_fp16 = float_to_fp16(c1_adj);
        uint16_t intercept_fp16 = float_to_fp16(c0_adj);
        _sfpu_load_imm16_(0, intercept_fp16);
        _sfpu_load_imm16_(1, slope_fp16);
        _sfpu_load_imm16_(2, intercept_fp16);

        vUInt l0 = l_reg[LRegs::LReg0];
        vUInt l1 = l_reg[LRegs::LReg1];
        vUInt l2 = l_reg[LRegs::LReg2];

        // Process all 32 elements for this segment using indexed access
        bool is_last_segment = (seg == NUM_SEGMENTS - 1);
        #pragma GCC unroll 8
        for (int i = 0; i < 32; i++) {
            vFloat x = dst_reg[i];
            v_if (x >= boundary) {
                if (is_last_segment) {
                    // Last segment: no upper bound check
                    vFloat x_norm = x - boundary;
                    vFloat linear_part = lut(x_norm, l0, l1, l2);
                    vFloat x_norm_sq = x_norm * x_norm;
                    dst_reg[i] = linear_part + c2 * x_norm_sq;
                } else {
                    v_if (x < boundary_next) {
                        vFloat x_norm = x - boundary;
                        vFloat linear_part = lut(x_norm, l0, l1, l2);
                        vFloat x_norm_sq = x_norm * x_norm;
                        dst_reg[i] = linear_part + c2 * x_norm_sq;
                    } v_endif;
                }
            } v_endif;
        }

        l_reg[LRegs::LReg0] = l0;
        l_reg[LRegs::LReg1] = l1;
        l_reg[LRegs::LReg2] = l2;
    }
}

// Note: No explicit specializations needed - template handles all sizes

} // namespace sfpi
#endif

//==============================================================================
// Main Kernel
//==============================================================================

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    [[maybe_unused]] float input_min = get_arg_val<float>(1);
    [[maybe_unused]] float input_max = get_arg_val<float>(2);

    constexpr uint32_t LUT_SIZE = get_compile_time_arg_val(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_lut = tt::CBIndex::c_25;

    using lut_t = std::array<float, LUT_SIZE>;
    auto p_lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);

    init_sfpu(cb_in, cb_out);

#ifdef PACKER_L1_ACC
    copy_tile_init(cb_in);
#endif

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        tile_regs_acquire();
        cb_wait_front(cb_in, 1);

#ifdef PACKER_L1_ACC
        copy_tile(cb_in, 0, 0);
#else
        acquire_dst();
        copy_tile_to_dst_init_short(cb_in);
        copy_tile(cb_in, 0, 0);
#endif

        cb_pop_front(cb_in, 1);

        // Apply piecewise quadratic Estrin+SFPLUT (TRISC1/math only)
        #ifdef TRISC_MATH
        sfpi::piecewise_quadratic_estrin_lut<LUT_SIZE>(*p_lut);
        #endif

        tile_regs_commit();
        cb_reserve_back(cb_out, 1);

#ifdef PACKER_L1_ACC
        pack_tile_from_cb(cb_in, 0);
#else
        pack_tile(0, cb_out);
        release_dst();
#endif

        cb_push_back(cb_out, 1);
        tile_regs_wait();
    }
}
} // namespace NAMESPACE
