// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/relu.h"  // For relu, relu_max
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/activations.h"  // For hardsigmoid, celu, softsign, softshrink
#include "api/compute/eltwise_unary/elu.h"
#include "api/compute/eltwise_unary/selu.h"
#include "api/compute/eltwise_unary/trigonometry.h"  // For sin, cos, cosh, sinh, atanh
#include "api/compute/eltwise_unary/erf_erfc.h"      // For erf
#include "api/compute/eltwise_unary/prelu.h"         // For prelu
#include "api/compute/eltwise_unary/hardtanh.h"      // For hardtanh
#include "api/compute/eltwise_unary/threshold.h"     // For threshold
#include "api/compute/eltwise_unary/cbrt.h"          // For cbrt
#include "api/compute/eltwise_unary/erfinv.h"        // For erfinv
#include "api/compute/eltwise_unary/identity.h"      // For identity
#include "api/compute/eltwise_unary/log1p.h"         // For log1p
#include "api/compute/eltwise_unary/sqrt.h"          // For sqrt
#include "api/compute/eltwise_unary/negative.h"      // For negative
#include "api/compute/eltwise_unary/recip.h"         // For recip
#include "api/compute/compute_kernel_api.h"  // For tanh_tile, sigmoid_tile, log_tile, abs_tile, tan_tile, atan_tile, silu_tile, exp2_tile, expm1_tile

/**
 * Native SFPU activation - uses hardware-accelerated SFPU operations
 *
 * This kernel demonstrates native SFPU operations without any LUT lookup.
 * Available operations: gelu, relu, tanh, softplus, exp, leaky_relu, elu, selu,
 *                       hardsigmoid, sin, cos, erf, cosh, sinh, atanh, celu,
 *                       prelu, softsign, softshrink, hardtanh, threshold, abs,
 *                       atan, tan, cbrt, erfinv, identity, log, log1p, log2,
 *                       log10, exp2, sigmoid, sqrt, negative, recip, silu,
 *                       expm1, relu6
 *
 * Runtime arguments:
 *   - arg[0]: n_tiles - number of tiles to process
 *   - arg[1]: activation_type - activation function ID (0-38)
 *   - arg[2]: fast_approx - 1=fast approximate mode (default), 0=precise mode (slower)
 *
 * Note: Composite functions requiring multiple register operations (mish, hardswish,
 * tanhshrink, logsigmoid, logit, hardshrink) are not included in this single-pass kernel.
 */

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t activation_type = get_arg_val<uint32_t>(1);
    uint32_t fast_approx = get_arg_val<uint32_t>(2);  // 0=precise, 1=fast approximate
    // 0=gelu, 1=relu, 2=tanh, 3=softplus, 4=exp, 5=leaky_relu, 6=elu, 7=selu,
    // 8=hardsigmoid, 9=sin, 10=cos, 11=erf, 12=cosh, 13=sinh, 14=atanh,
    // 15=celu, 16=prelu, 17=softsign, 18=softshrink, 19=hardtanh, 20=threshold,
    // 21=abs, 22=atan, 23=tan, 24=cbrt, 25=erfinv, 26=identity, 27=log,
    // 28=log1p, 29=log2, 30=log10, 31=exp2, 32=sigmoid, 33=sqrt, 34=negative,
    // 35=recip, 36=silu, 37=expm1, 38=relu6

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;

    // Initialize SFPU
    init_sfpu(cb_in, cb_out);

    // Initialize the specific operation
    switch (activation_type) {
        case 0:  // GELU
            if (fast_approx) {
                gelu_tile_init<true>();
            } else {
                gelu_tile_init<false>();
            }
            break;
        case 1:  // ReLU
            // relu_tile doesn't require init
            break;
        case 2:  // Tanh
            tanh_tile_init();
            break;
        case 3:  // Softplus
            // softplus_tile doesn't require init in this API
            break;
        case 4:  // Exp
            if (fast_approx) {
                exp_tile_init<false, true>();
            } else {
                exp_tile_init<false, false>();
            }
            break;
        case 5:  // Leaky ReLU
            // leaky_relu_tile doesn't require init
            break;
        case 6:  // ELU
            elu_tile_init();
            break;
        case 7:  // SELU
            selu_tile_init();
            break;
        case 8:  // Hardsigmoid
            hardsigmoid_tile_init();
            break;
        case 9:  // Sin
            sin_tile_init();
            break;
        case 10:  // Cos
            cos_tile_init();
            break;
        case 11:  // Erf
            if (fast_approx) {
                erf_tile_init<true>();
            } else {
                erf_tile_init<false>();
            }
            break;
        case 12:  // Cosh
            cosh_tile_init();
            break;
        case 13:  // Sinh
            sinh_tile_init();
            break;
        case 14:  // Atanh
            atanh_tile_init();
            break;
        case 15:  // CELU
            celu_tile_init();
            break;
        case 16:  // PReLU
            prelu_tile_init();
            break;
        case 17:  // Softsign
            softsign_tile_init();
            break;
        case 18:  // Softshrink
            softshrink_tile_init();
            break;
        case 19:  // Hardtanh
            hardtanh_tile_init();
            break;
        case 20:  // Threshold
            threshold_tile_init();
            break;
        case 21:  // Abs
            abs_tile_init();
            break;
        case 22:  // Atan
            atan_tile_init();
            break;
        case 23:  // Tan
            tan_tile_init();
            break;
        case 24:  // Cbrt
            cbrt_tile_init();
            break;
        case 25:  // Erfinv
            erfinv_tile_init();
            break;
        case 26:  // Identity
            // identity_tile doesn't require init
            break;
        case 27:  // Log
            if (fast_approx) {
                log_tile_init<true>();
            } else {
                log_tile_init<false>();
            }
            break;
        case 28:  // Log1p
            log1p_tile_init();
            break;
        case 29:  // Log2
            if (fast_approx) {
                log_with_base_tile_init<true>();
            } else {
                log_with_base_tile_init<false>();
            }
            break;
        case 30:  // Log10
            if (fast_approx) {
                log_with_base_tile_init<true>();
            } else {
                log_with_base_tile_init<false>();
            }
            break;
        case 31:  // Exp2
            exp2_tile_init();
            break;
        case 32:  // Sigmoid
            if (fast_approx) {
                sigmoid_tile_init<true>();
            } else {
                sigmoid_tile_init<false>();
            }
            break;
        case 33:  // Sqrt
            sqrt_tile_init();
            break;
        case 34:  // Negative
            negative_tile_init();
            break;
        case 35:  // Recip
            recip_tile_init();
            break;
        case 36:  // Silu (swish)
            silu_tile_init();
            break;
        case 37:  // Expm1
            if (fast_approx) {
                expm1_tile_init<true>();
            } else {
                expm1_tile_init<false>();
            }
            break;
        case 38:  // ReLU6
            relu_max_tile_init();
            break;
    }

    for (uint32_t tile = 0; tile < n_tiles; tile++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);

        // Apply the activation function
        switch (activation_type) {
            case 0:  // GELU
                if (fast_approx) {
                    gelu_tile<true>(0);
                } else {
                    gelu_tile<false>(0);
                }
                break;
            case 1:  // ReLU
                relu_tile(0);
                break;
            case 2:  // Tanh
                tanh_tile(0);
                break;
            case 3: {  // Softplus with beta=1.0, threshold=20.0
                // softplus(x) = 1/beta * log(1 + exp(beta * x))
                // For beta=1.0: softplus(x) = log(1 + exp(x))
                // Parameters: beta, beta_reciprocal, threshold (as uint32_t bit patterns)
                uint32_t beta = 0x3f800000;          // 1.0f as uint32
                uint32_t beta_recip = 0x3f800000;    // 1.0f as uint32
                uint32_t threshold = 0x41a00000;     // 20.0f as uint32
                softplus_tile(0, beta, beta_recip, threshold);
                break;
            }
            case 4:  // Exp
                if (fast_approx) {
                    exp_tile<false, true>(0);
                } else {
                    exp_tile<false, false>(0);
                }
                break;
            case 5: {  // Leaky ReLU with slope=0.01
                uint32_t slope = 0x3c23d70a;  // 0.01f as uint32
                leaky_relu_tile(0, slope);
                break;
            }
            case 6: {  // ELU with alpha=1.0
                uint32_t alpha = 0x3f800000;  // 1.0f as uint32
                elu_tile(0, alpha);
                break;
            }
            case 7: {  // SELU with scale=1.0507 and alpha=1.6733
                uint32_t scale = 0x3f86a09e;  // 1.0507009873554804934193349852946f as uint32
                uint32_t alpha = 0x3fd5ba6e;  // 1.6732632423543772848170429916717f as uint32
                selu_tile(0, scale, alpha);
                break;
            }
            case 8:  // Hardsigmoid
                hardsigmoid_tile(0);
                break;
            case 9:  // Sin
                sin_tile(0);
                break;
            case 10:  // Cos
                cos_tile(0);
                break;
            case 11:  // Erf
                if (fast_approx) {
                    erf_tile<true>(0);
                } else {
                    erf_tile<false>(0);
                }
                break;
            case 12:  // Cosh
                cosh_tile(0);
                break;
            case 13:  // Sinh
                sinh_tile(0);
                break;
            case 14:  // Atanh
                atanh_tile(0);
                break;
            case 15: {  // CELU with alpha=1.0
                uint32_t alpha = 0x3f800000;        // 1.0f as uint32
                uint32_t alpha_recip = 0x3f800000;  // 1.0f as uint32
                celu_tile(0, alpha, alpha_recip);
                break;
            }
            case 16: {  // PReLU with slope=0.25
                uint32_t slope = 0x3e800000;  // 0.25f as uint32
                prelu_tile(0, slope);
                break;
            }
            case 17:  // Softsign
                softsign_tile(0);
                break;
            case 18: {  // Softshrink with lambda=0.5
                uint32_t lambda = 0x3f000000;  // 0.5f as uint32
                softshrink_tile(0, lambda);
                break;
            }
            case 19: {  // Hardtanh with min=-1.0, max=1.0
                uint32_t min_val = 0xbf800000;  // -1.0f as uint32
                uint32_t max_val = 0x3f800000;  //  1.0f as uint32
                hardtanh_tile(0, min_val, max_val);
                break;
            }
            case 20: {  // Threshold with threshold=0.0, value=0.0
                uint32_t threshold = 0x00000000;  // 0.0f as uint32
                uint32_t value = 0x00000000;      // 0.0f as uint32
                threshold_tile(0, threshold, value);
                break;
            }
            case 21:  // Abs
                abs_tile(0);
                break;
            case 22:  // Atan
                atan_tile(0);
                break;
            case 23:  // Tan
                tan_tile(0);
                break;
            case 24:  // Cbrt
                cbrt_tile(0);
                break;
            case 25:  // Erfinv
                erfinv_tile(0);
                break;
            case 26:  // Identity
                identity_tile(0);
                break;
            case 27:  // Log
                if (fast_approx) {
                    log_tile<true>(0);
                } else {
                    log_tile<false>(0);
                }
                break;
            case 28:  // Log1p
                log1p_tile(0);
                break;
            case 29: {  // Log2
                uint32_t base_scale = 0x3fb8aa3b;  // 1/ln(2) as uint32
                if (fast_approx) {
                    log_with_base_tile<true>(0, base_scale);
                } else {
                    log_with_base_tile<false>(0, base_scale);
                }
                break;
            }
            case 30: {  // Log10
                uint32_t base_scale = 0x3ede5bd9;  // 1/ln(10) as uint32
                if (fast_approx) {
                    log_with_base_tile<true>(0, base_scale);
                } else {
                    log_with_base_tile<false>(0, base_scale);
                }
                break;
            }
            case 31:  // Exp2
                exp2_tile(0);
                break;
            case 32:  // Sigmoid
                if (fast_approx) {
                    sigmoid_tile<VectorMode::RC, true>(0);
                } else {
                    sigmoid_tile<VectorMode::RC, false>(0);
                }
                break;
            case 33:  // Sqrt
                sqrt_tile(0);
                break;
            case 34:  // Negative
                negative_tile(0);
                break;
            case 35:  // Recip
                recip_tile(0);
                break;
            case 36:  // Silu (swish)
                silu_tile(0);
                break;
            case 37:  // Expm1
                if (fast_approx) {
                    expm1_tile<true>(0);
                } else {
                    expm1_tile<false>(0);
                }
                break;
            case 38: {  // ReLU6 - min(max(0, x), 6)
                uint32_t max_val = 0x40c00000;  // 6.0f as uint32
                relu_max_tile(0, max_val);
                break;
            }
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
