// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/cb_api.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"

namespace kutil = norm::kernel_util;

/**
 * FPU-based Piecewise Polynomial with TF32 Precision
 * ===================================================
 *
 * Same algorithm as piecewise_riscv.cpp but:
 *   - C and P matrices stored as TF32 (4 bytes/element, 10-bit mantissa)
 *   - FPU matmul: TF32×TF32 → FP32 accumulation (DST_ACCUM_MODE=1)
 *   - Y result read as FP32 (4 bytes/element)
 *   - Output written as FP32
 *
 * TF32 format (stored in lower 19 bits of a uint32):
 *   [sign:1][exp:8][mant:10][pad:13]  →  bit layout: sign@18, exp@17..10, mant@9..0
 *
 * CB sizes (host must match):
 *   CB_SCRATCH_C, CB_SCRATCH_P: Float32 format (4096 bytes per tile)
 *   CB_SCRATCH_Y:               Float32 format (4096 bytes per tile)
 *
 * See piecewise_riscv.cpp for architecture notes.
 */

#ifndef DST_ACCUM_MODE
#define DST_ACCUM_MODE 1   // FP32 accumulation
#endif

#ifndef MATH_FIDELITY
#define MATH_FIDELITY 4    // maximum fidelity
#endif

constexpr uint32_t CB_SCRATCH_C = tt::CBIndex::c_26;
constexpr uint32_t CB_SCRATCH_P = tt::CBIndex::c_27;
constexpr uint32_t CB_SCRATCH_Y = tt::CBIndex::c_28;

//===========================================================================
// Format helpers
//===========================================================================

// TF32 tile layout: same face-organized structure but uint32 elements
// A 32×32 TF32 tile = 32×32 × 4 bytes = 4096 bytes
// face_idx = (row/16)*2 + (col/16)
// linear_index = face_idx*256 + (row%16)*16 + (col%16)
inline uint32_t tile_offset(uint32_t row, uint32_t col) {
    uint32_t face = (row / 16) * 2 + (col / 16);
    return face * 256 + (row % 16) * 16 + (col % 16);
}

// FP32 to TF32: truncate mantissa from 23 bits to 10 bits
// TF32 bit layout in uint32: sign@18, exp@17..10, mant@9..0
inline uint32_t fp32_to_tf32(float value) {
    uint32_t fp32_bits;
    __builtin_memcpy(&fp32_bits, &value, 4);
    uint32_t sign = (fp32_bits >> 31) & 0x1;
    uint32_t exp  = (fp32_bits >> 23) & 0xFF;
    uint32_t mant = (fp32_bits >> 13) & 0x3FF;  // upper 10 bits of 23-bit mantissa
    return (sign << 18) | (exp << 10) | mant;
}

// BF16 helpers for reading input (fp32 output path)
inline uint16_t fp32_to_bf16(float v) {
    uint32_t bits;
    __builtin_memcpy(&bits, &v, 4);
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_to_fp32(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

// Write float[32][32] to a CB tile as TF32 (face-organized, uint32)
inline void write_f32_to_cb_tf32(uint32_t cb_idx, const float mat[32][32]) {
    cb_reserve_back(cb_idx, 1);
    uint32_t addr = get_tile_address(cb_idx, 0);
    volatile tt_l1_ptr uint32_t* dst =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t r = 0; r < 32; r++)
        for (uint32_t c = 0; c < 32; c++)
            dst[tile_offset(r, c)] = fp32_to_tf32(mat[r][c]);
    cb_push_back(cb_idx, 1);
}

// Read FP32 CB tile into float[32][32] (face-organized uint32, then pop)
inline void read_cb_fp32_to_f32(uint32_t cb_idx, float mat[32][32]) {
    cb_wait_front(cb_idx, 1);
    uint32_t addr = get_tile_address(cb_idx, 0);
    volatile tt_l1_ptr uint32_t* src =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            uint32_t fp32_bits = src[tile_offset(r, c)];
            __builtin_memcpy(&mat[r][c], &fp32_bits, 4);
        }
    }
    cb_pop_front(cb_idx, 1);
}

//===========================================================================
// Piecewise polynomial helpers (identical to BF16 variant)
//===========================================================================

template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS>
inline void build_coeff_matrix(const float* lut, float C[32][32]) {
    constexpr uint32_t K = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;
    for (uint32_t r = 0; r < 32; r++)
        for (uint32_t c = 0; c < 32; c++)
            C[r][c] = 0.0f;
    for (uint32_t s = 0; s < NUM_SEGMENTS; s++)
        for (uint32_t k = 0; k < K; k++)
            C[s][k] = lut[COEFF_OFFSET + s * K + k];
}

template <uint32_t POLY_DEGREE>
inline void build_power_matrix(const float x_vals[32], float P[32][32]) {
    constexpr uint32_t K = POLY_DEGREE + 1;
    for (uint32_t r = 0; r < 32; r++)
        for (uint32_t c = 0; c < 32; c++)
            P[r][c] = 0.0f;
    for (uint32_t j = 0; j < 32; j++) {
        float xp = 1.0f;
        for (uint32_t k = 0; k < K; k++) {
            P[k][j] = xp;
            xp *= x_vals[j];
        }
    }
}

template <uint32_t NUM_SEGMENTS>
inline void compute_segment_ids(const float* lut, const float x_vals[32],
                                 uint32_t seg_id[32]) {
    for (uint32_t j = 0; j < 32; j++) {
        float x = x_vals[j];
        if (x < lut[0]) x = lut[0];
        if (x > lut[NUM_SEGMENTS]) x = lut[NUM_SEGMENTS];
        uint32_t seg = 0;
        for (uint32_t s = 1; s < NUM_SEGMENTS; s++)
            if (x >= lut[s]) seg = s;
        seg_id[j] = seg;
    }
}

//===========================================================================
// Per-tile FPU TF32 evaluation
//===========================================================================

template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_fpu_tile_tf32(
    const std::array<float, LUT_SIZE>& lut,
    const volatile tt_l1_ptr uint16_t* in_ptr,   // input: BF16 face-organized tile
    volatile tt_l1_ptr uint32_t* out_ptr)         // output: FP32 face-organized tile
{
    constexpr uint32_t K = POLY_DEGREE + 1;
    static_assert(LUT_SIZE == (NUM_SEGMENTS + 1) + NUM_SEGMENTS * K, "LUT_SIZE mismatch");
    static_assert(K <= 32, "POLY_DEGREE + 1 must be <= 32");
    static_assert(NUM_SEGMENTS <= 32, "NUM_SEGMENTS must be <= 32");

    // Read all 1024 x-values from BF16 input tile
    float all_x[32][32];
    for (uint32_t r = 0; r < 32; r++)
        for (uint32_t c = 0; c < 32; c++)
            all_x[r][c] = bf16_to_fp32(in_ptr[tile_offset(r, c)]);

    // Build C once
    float C[32][32];
    build_coeff_matrix<POLY_DEGREE, NUM_SEGMENTS>(lut.data(), C);

    float P[32][32];
    float Y[32][32];
    float result[32][32];

    for (uint32_t row = 0; row < 32; row++) {
        uint32_t seg_id[32];
        compute_segment_ids<NUM_SEGMENTS>(lut.data(), all_x[row], seg_id);

        build_power_matrix<POLY_DEGREE>(all_x[row], P);

        // Write C (TF32) and P (TF32) to scratch CBs
        write_f32_to_cb_tf32(CB_SCRATCH_C, C);
        write_f32_to_cb_tf32(CB_SCRATCH_P, P);

        // FPU matmul: TF32×TF32 → FP32 accumulation
        // With DST_ACCUM_MODE=1, DST holds FP32 values
        cb_wait_front(CB_SCRATCH_C, 1);
        cb_wait_front(CB_SCRATCH_P, 1);
        cb_reserve_back(CB_SCRATCH_Y, 1);
        tile_regs_acquire();
        matmul_tiles(CB_SCRATCH_C, CB_SCRATCH_P, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SCRATCH_Y);
        cb_push_back(CB_SCRATCH_Y, 1);
        cb_pop_front(CB_SCRATCH_C, 1);
        cb_pop_front(CB_SCRATCH_P, 1);
        tile_regs_release();

        // Read FP32 result from CB
        read_cb_fp32_to_f32(CB_SCRATCH_Y, Y);

        // Segment selection
        for (uint32_t j = 0; j < 32; j++)
            result[row][j] = Y[seg_id[j]][j];
    }

    // Write FP32 result directly to output CB L1
    for (uint32_t r = 0; r < 32; r++) {
        for (uint32_t c = 0; c < 32; c++) {
            uint32_t fp32_bits;
            __builtin_memcpy(&fp32_bits, &result[r][c], 4);
            out_ptr[tile_offset(r, c)] = fp32_bits;
        }
    }
}

//===========================================================================
// Kernel main
//===========================================================================

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t compute_loop_factor = get_arg_val<uint32_t>(1);

#ifdef EMBEDDED_LUT
    constexpr auto cb_in  = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t lut_size     = LUT_SIZE;
    constexpr uint32_t poly_degree  = POLY_DEGREE;
    constexpr uint32_t num_segments = NUM_SEGMENTS;
    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;
#else
    constexpr uint32_t lut_size     = get_compile_time_arg_val(0);
    constexpr uint32_t poly_degree  = get_compile_time_arg_val(1);
    constexpr uint32_t num_segments = get_compile_time_arg_val(2);
    constexpr auto cb_in  = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_lut = tt::CBIndex::c_25;
    using lut_t = std::array<float, lut_size>;
    auto p_lut = kutil::compute::memory::get_pointer_to_cb_data<lut_t>(cb_lut, 0);
#endif

    // Initialize FPU matmul: CB_SCRATCH_C → SrcB, CB_SCRATCH_P → SrcA
    ckernel::mm_init(CB_SCRATCH_C, CB_SCRATCH_P, CB_SCRATCH_Y);

    // get_tile_address uses UNPACK's fifo_rd_ptr. For cb_out, the compute kernel
    // never calls cb_pop_front, so fifo_rd_ptr never advances — get_tile_address
    // would return the same base address for every tile. Capture it once and
    // manually index into the double-buffered CB (tiles_per_cb=2 in host).
    // TF32 variant produces FP32 output: 32×32×4 = 4096 bytes per tile.
    constexpr uint32_t OUT_TILE_BYTES = 32 * 32 * sizeof(uint32_t);  // 4096 for FP32
    constexpr uint32_t OUT_CB_CAPACITY = 2;
    uint32_t out_base_addr = get_tile_address(cb_out, 0);

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        uint32_t in_addr = get_tile_address(cb_in, 0);

        // Input is BF16 (or FP32 if PACKER_L1_ACC, but treat as BF16 tile layout)
        volatile tt_l1_ptr uint16_t* in_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in_addr);
        // Output is FP32 (TF32 variant always produces FP32 output).
        // Manually cycle through CB slots since fifo_rd_ptr for cb_out never advances
        // in the compute kernel (only the writer kernel pops from cb_out).
        volatile tt_l1_ptr uint32_t* out_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                out_base_addr + (tile % OUT_CB_CAPACITY) * OUT_TILE_BYTES);

        for (uint32_t loop = 0; loop < compute_loop_factor; loop++) {
            // All degree parameters are constexpr — use them directly as template args.
            piecewise_fpu_tile_tf32<poly_degree, num_segments, lut_size>(*p_lut, in_ptr, out_ptr);
        }

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
    }
}
}
