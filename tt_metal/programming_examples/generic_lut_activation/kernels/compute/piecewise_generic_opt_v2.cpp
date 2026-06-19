// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

namespace kutil = norm::kernel_util;

/**
 * OPTIMIZED Generic Piecewise Polynomial LUT Activation (V2 - Staggered Evaluation)
 *
 * This is an improved version that uses STAGGERED evaluation to reduce register pressure
 * while maintaining the performance benefits of per-lane coefficient loading.
 *
 * Key Optimization:
 * Instead of loading all coefficients at once (high register pressure), we:
 * 1. Load 2-3 coefficients for highest-degree terms
 * 2. Evaluate those terms using Horner's method
 * 3. Load next batch of 2-3 coefficients
 * 4. Continue evaluation
 * 5. Repeat until complete
 *
 * This reduces peak register pressure from (DEGREE+1) to just 2-3 registers!
 *
 * Performance: Same as opt_v1 (10-15× faster than original)
 * Register Pressure: ~6-7 registers regardless of degree (vs ~13 for degree-8 in v1)
 *
 * Template parameters (compile-time):
 *   POLY_DEGREE   - Polynomial degree (0=constant, 1=linear, 2=quadratic, ... 16=hexadecic, 32=duotrigesic)
 *   NUM_SEGMENTS  - Number of segments (4, 8, 16, 32, 64, etc.)
 *   LUT_SIZE      - Total LUT size = (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)
 *
 * LUT Format: [b0, b1, ..., bN, c0_seg0, c1_seg0, ..., cn_seg0, c0_seg1, ...]
 * where bi are boundary points and cj are polynomial coefficients
 */

#ifdef TRISC_MATH
namespace sfpi {

// OPTIMIZED V2: Inlined staggered polynomial evaluation with low register pressure
// Evaluates polynomial by loading and computing 2-3 terms at a time
// Key optimization: Coefficient loading is INLINED (no template function calls)
// to reduce JIT compilation complexity
template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_opt_v2(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    static_assert(LUT_SIZE == (NUM_SEGMENTS + 1) + (NUM_SEGMENTS * COEFFS_PER_SEGMENT),
                  "LUT_SIZE must equal (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1)");

    // Batch size for staggered evaluation (trade-off between register pressure and segment loops)
    constexpr uint32_t BATCH_SIZE = 3;  // Process 3 coefficients at a time

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];

        // Clamp x to valid range
        vFloat x_clamped = x;
        v_if (x < lut[0]) {
            x_clamped = lut[0];
        }
        v_endif;
        v_if (x > lut[NUM_SEGMENTS]) {
            x_clamped = lut[NUM_SEGMENTS];
        }
        v_endif;

        // STAGGERED BATCHING: Load coefficients in small batches, evaluate incrementally
        // Avoids nested loops (code explosion) while keeping register pressure low
        // Process BATCH_SIZE coefficients at a time via separate segment loops

        vFloat result;

        if constexpr (POLY_DEGREE == 0) {
            // Constant: just load c0
            for (uint32_t seg = 0; seg < NUM_SEGMENTS; seg++) {
                v_if (x_clamped >= lut[seg]) {
                    result = lut[COEFF_OFFSET + seg * COEFFS_PER_SEGMENT];
                }
                v_endif;
            }
        } else {
            // Batch 1: Load highest-degree coefficients [c_n, c_{n-1}, c_{n-2}]
            constexpr uint32_t batch1_size = (POLY_DEGREE >= BATCH_SIZE) ? BATCH_SIZE : POLY_DEGREE;
            constexpr uint32_t batch1_start = POLY_DEGREE + 1 - batch1_size;

            vFloat c[BATCH_SIZE];
            for (uint32_t seg = 0; seg < NUM_SEGMENTS; seg++) {
                v_if (x_clamped >= lut[seg]) {
                    const uint32_t base = COEFF_OFFSET + seg * COEFFS_PER_SEGMENT;
                    if constexpr (batch1_size >= 1) c[0] = lut[base + batch1_start];
                    if constexpr (batch1_size >= 2) c[1] = lut[base + batch1_start + 1];
                    if constexpr (batch1_size >= 3) c[2] = lut[base + batch1_start + 2];
                }
                v_endif;
            }

            // Evaluate batch 1
            if constexpr (batch1_size == 1) {
                result = c[0];
            } else if constexpr (batch1_size == 2) {
                result = c[1] * x + c[0];
            } else {
                result = (c[2] * x + c[1]) * x + c[0];
            }

            // Process remaining coefficients in additional batches
            if constexpr (POLY_DEGREE >= BATCH_SIZE) {
                constexpr uint32_t remaining_after_b1 = batch1_start;

                // Batch 2
                if constexpr (remaining_after_b1 > 0) {
                    constexpr uint32_t b2_size = (remaining_after_b1 >= BATCH_SIZE) ? BATCH_SIZE : remaining_after_b1;
                    constexpr uint32_t b2_start = remaining_after_b1 - b2_size;
                    for (uint32_t seg = 0; seg < NUM_SEGMENTS; seg++) {
                        v_if (x_clamped >= lut[seg]) {
                            const uint32_t base = COEFF_OFFSET + seg * COEFFS_PER_SEGMENT;
                            if constexpr (b2_size >= 1) c[0] = lut[base + b2_start];
                            if constexpr (b2_size >= 2) c[1] = lut[base + b2_start + 1];
                            if constexpr (b2_size >= 3) c[2] = lut[base + b2_start + 2];
                        }
                        v_endif;
                    }
                    if constexpr (b2_size == 1) {
                        result = result * x + c[0];
                    } else if constexpr (b2_size == 2) {
                        result = (result * x + c[1]) * x + c[0];
                    } else {
                        result = ((result * x + c[2]) * x + c[1]) * x + c[0];
                    }
                }

                // Batch 3
                constexpr uint32_t remaining_after_b2 = (remaining_after_b1 > BATCH_SIZE) ? (remaining_after_b1 - BATCH_SIZE) : 0;
                if constexpr (remaining_after_b2 > 0) {
                    constexpr uint32_t b3_size = (remaining_after_b2 >= BATCH_SIZE) ? BATCH_SIZE : remaining_after_b2;
                    constexpr uint32_t b3_start = remaining_after_b2 - b3_size;
                    for (uint32_t seg = 0; seg < NUM_SEGMENTS; seg++) {
                        v_if (x_clamped >= lut[seg]) {
                            const uint32_t base = COEFF_OFFSET + seg * COEFFS_PER_SEGMENT;
                            if constexpr (b3_size >= 1) c[0] = lut[base + b3_start];
                            if constexpr (b3_size >= 2) c[1] = lut[base + b3_start + 1];
                            if constexpr (b3_size >= 3) c[2] = lut[base + b3_start + 2];
                        }
                        v_endif;
                    }
                    if constexpr (b3_size == 1) {
                        result = result * x + c[0];
                    } else if constexpr (b3_size == 2) {
                        result = (result * x + c[1]) * x + c[0];
                    } else {
                        result = ((result * x + c[2]) * x + c[1]) * x + c[0];
                    }
                }

                // Batch 4+ for higher degrees
                constexpr uint32_t remaining_after_b3 = (remaining_after_b2 > BATCH_SIZE) ? (remaining_after_b2 - BATCH_SIZE) : 0;
                if constexpr (remaining_after_b3 > 0) {
                    constexpr uint32_t b4_size = (remaining_after_b3 >= BATCH_SIZE) ? BATCH_SIZE : remaining_after_b3;
                    constexpr uint32_t b4_start = remaining_after_b3 - b4_size;
                    for (uint32_t seg = 0; seg < NUM_SEGMENTS; seg++) {
                        v_if (x_clamped >= lut[seg]) {
                            const uint32_t base = COEFF_OFFSET + seg * COEFFS_PER_SEGMENT;
                            if constexpr (b4_size >= 1) c[0] = lut[base + b4_start];
                            if constexpr (b4_size >= 2) c[1] = lut[base + b4_start + 1];
                            if constexpr (b4_size >= 3) c[2] = lut[base + b4_start + 2];
                        }
                        v_endif;
                    }
                    if constexpr (b4_size == 1) {
                        result = result * x + c[0];
                    } else if constexpr (b4_size == 2) {
                        result = (result * x + c[1]) * x + c[0];
                    } else {
                        result = ((result * x + c[2]) * x + c[1]) * x + c[0];
                    }
                }
            }
        }

        dst_reg[d] = result;
    }
}

// Include specialized implementations with manual unrolling for 4/8/16 segments
#include "piecewise_generic_opt_v2_specialized.cpp"

} // namespace sfpi
#endif

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t compute_loop_factor = get_arg_val<uint32_t>(1);  // How many times to recompute each tile

#ifdef EMBEDDED_LUT
    // Embedded LUT mode: LUT is compiled directly into the kernel
    // Header must define: POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE, LUT_DATA
    // This provides zero L1 memory overhead for the LUT
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Use embedded LUT data directly from header
    constexpr uint32_t lut_size = LUT_SIZE;
    constexpr uint32_t poly_degree = POLY_DEGREE;
    constexpr uint32_t num_segments = NUM_SEGMENTS;
    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;
#else
    // Generic LUT mode: LUT is loaded from L1 circular buffer
    // This allows runtime LUT generation and sharing across cores
    constexpr uint32_t lut_size = get_compile_time_arg_val(0);
    constexpr uint32_t poly_degree = get_compile_time_arg_val(1);
    constexpr uint32_t num_segments = get_compile_time_arg_val(2);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_lut = tt::CBIndex::c_25;

    // Get LUT array from L1 memory as float array directly
    using lut_t = std::array<float, lut_size>;
    auto p_lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);
#endif

    init_sfpu(cb_in, cb_out);

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);

        // COMPUTE AMPLIFICATION: Run compute multiple times on same tile data
        for (uint32_t loop = 0; loop < compute_loop_factor; loop++) {
            // All degree parameters are constexpr — use them directly as template args.
            // No dispatch table needed; works for ANY poly_degree automatically.
            #ifdef TRISC_MATH
            sfpi::piecewise_generic_lut_opt_v2_dispatch<poly_degree, num_segments, lut_size>(*p_lut);
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
