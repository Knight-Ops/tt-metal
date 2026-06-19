// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// piecewise_riscv.cpp — Pure RISC-V piecewise polynomial activation — Horner's method on all 3 TRISCs.
//
// Design notes:
//   - All 3 TRISCs (UNPACK/MATH/PACK) run the same computation and write the same
//     values to out_ptr. CB primitives (cb_wait_front, cb_reserve_back, etc.) are
//     gated internally by UNPACK()/PACK() macros so only the relevant TRISC acts.
//   - get_tile_address is an ALWI that UNPACK computes and broadcasts to MATH/PACK
//     via mailbox, so all 3 TRISCs get the correct byte address.
//   - No FPU matmul is used. The earlier FPU approach (Y = C × P where P is the
//     power matrix) was abandoned because:
//       1. fifo_wr_ptr is in 16-byte datum units, not bytes (confirmed by cb_api.h
//          line 149: address = (fifo_rd_ptr + offset) << 4). Using it directly as a
//          byte pointer writes P to the wrong L1 address.
//       2. Numerical instability: power basis evaluation (x^0, x^1, ...) causes
//          catastrophic cancellation for large |x| in BF16.
//       3. Performance: 32 matmuls/tile gives ~42× slower math_us vs SFPU Horner.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

namespace kutil = norm::kernel_util;

// L1 tile face-interleaved layout:
//   face = (row/16)*2 + (col/16); offset = face*256 + (row%16)*16 + (col%16)
inline uint32_t tile_offset(uint32_t row, uint32_t col) {
    return ((row / 16) * 2 + (col / 16)) * 256 + (row % 16) * 16 + (col % 16);
}
inline float bf16_to_fp32(uint16_t v) {
    uint32_t b = static_cast<uint32_t>(v) << 16;
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}
inline uint16_t fp32_to_bf16(float v) {
    uint32_t b;
    __builtin_memcpy(&b, &v, 4);
    return static_cast<uint16_t>(b >> 16);
}

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t lut_size     = get_compile_time_arg_val(0);
    constexpr uint32_t poly_degree  = get_compile_time_arg_val(1);
    constexpr uint32_t num_segments = get_compile_time_arg_val(2);
    constexpr uint32_t K = poly_degree + 1;
    constexpr uint32_t COEFF_OFFSET = num_segments + 1;

    constexpr auto cb_in  = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_lut = tt::CBIndex::c_25;

    using lut_t = std::array<float, lut_size>;
    auto* lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);

    // out_base_addr: byte address of page 0 of cb_out (constant across tiles).
    // get_tile_address: UNPACK computes via (fifo_rd_ptr << 4), broadcasts to
    // MATH/PACK via mailbox. All 3 TRISCs receive the same byte address.
    constexpr uint32_t OUT_TILE_BYTES = 32 * 32 * sizeof(uint16_t);
    constexpr uint32_t OUT_CB_CAPACITY = 2;
    uint32_t out_base_addr = get_tile_address(cb_out, 0);

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        // UNPACK waits for input tile; PACK waits for free output slot.
        // MATH has no-ops for both.
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        // in_ptr: byte address of input tile (all 3 TRISCs via mailbox sync).
        const volatile tt_l1_ptr uint16_t* in_ptr =
            reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(get_tile_address(cb_in, 0));

        // out_ptr: manual ring-buffer address for this tile's output slot.
        volatile tt_l1_ptr uint16_t* out_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                out_base_addr + (tile % OUT_CB_CAPACITY) * OUT_TILE_BYTES);

        // Evaluate piecewise polynomial for all 1024 elements in the tile.
        // All 3 TRISCs execute identical computation and write identical values,
        // so there are no ordering hazards.
        for (uint32_t row = 0; row < 32; row++) {
            for (uint32_t col = 0; col < 32; col++) {
                uint32_t off = tile_offset(row, col);
                float x = bf16_to_fp32(in_ptr[off]);

                // Clamp x to LUT range [lut[0], lut[num_segments]].
                if (x < (*lut)[0]) x = (*lut)[0];
                if (x > (*lut)[num_segments]) x = (*lut)[num_segments];

                // Binary-linear scan for segment: largest s such that x >= lut[s].
                uint32_t seg = 0;
                for (uint32_t s = 1; s < num_segments; s++)
                    if (x >= (*lut)[s]) seg = s;

                // Horner's method: y = c[K-1]*x^(K-1) + ... + c[1]*x + c[0]
                //   = (...((c[K-1]*x + c[K-2])*x + c[K-3])*x + ...)*x + c[0]
                const float* coeffs = &(*lut)[COEFF_OFFSET + seg * K];
                float y = coeffs[K - 1];
                for (int k = static_cast<int>(K) - 2; k >= 0; k--)
                    y = y * x + coeffs[k];

                out_ptr[off] = fp32_to_bf16(y);
            }
        }

        // PACK signals output available; UNPACK frees input slot.
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
    }
}
}
