// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_nanobind.hpp"

#include "ttnn-nanobind/bind_function.hpp"
#include "indexer_score.hpp"

namespace ttnn::operations::experimental::indexer_score::detail {

void bind_indexer_score(nb::module_& mod) {
    nb::class_<IndexerScoreProgramConfig>(mod, "IndexerScoreProgramConfig")
        .def(
            nb::init<std::size_t, std::size_t, std::size_t>(),
            nb::kw_only(),
            nb::arg("q_chunk_size") = 32,
            nb::arg("k_chunk_size") = 32,
            nb::arg("head_group_size") = 1)
        .def_rw("q_chunk_size", &IndexerScoreProgramConfig::q_chunk_size)
        .def_rw("k_chunk_size", &IndexerScoreProgramConfig::k_chunk_size)
        .def_rw("head_group_size", &IndexerScoreProgramConfig::head_group_size);

    ttnn::bind_function<"indexer_score", "ttnn.experimental.">(
        mod,
        R"doc(
        DeepSeek-V3.2 DSA / MiniMax-M3 MSA lightning-indexer scorer.

        score[b, s, t] = sum_h act(q[b,h,s,:] . k[b,t,:]) * weights[b,h,s]

        with act = relu when apply_relu=True (DeepSeek-V3.2 / GLM-5), or the
        identity when apply_relu=False (MiniMax M3 MSA: raw dot product, with
        the 1/sqrt(d) scale folded into weights). For M3's per-GQA-group
        selection, run one group per device (index head Hi=1) so the head-sum
        is a no-op and the output [B,1,Sq,T] is that group's score row.

        Args:
            q: [B, Hi, Sq, D] bf16 or bfp8_b tiled (post non-interleaved RoPE)
            k: [B, 1, T, D] bf16 or bfp8_b tiled, single shared head
            weights: optional [B, Hi, Sq, 1] bf16 tiled learned per-head gates
                (DeepSeek/GLM; scale pre-folded). Omit for MiniMax M3 (no gates):
                the op then uses a constant gate equal to `scale`.
            chunk_start_idx: global position of query row 0 (causality: key t
                visible to query s iff t <= chunk_start_idx + s)
            apply_relu: apply relu(q.kT) before the gate-multiply (default True,
                DeepSeek/GLM). Set False for the raw dot product (MiniMax M3).
            scale: constant gate value used only when weights is omitted (e.g.
                1/sqrt(d) for MiniMax M3). Ignored when weights is given.
            num_groups: 1 sums all Hi heads into one plane (DeepSeek/GLM). G>1
                partitions the heads into G groups of Hi/G and sums within each
                group -> output [B, G, Sq, T] (MiniMax M3 per-GQA-group selection,
                multiple groups on one chip). G>1 needs all heads resident
                (head_group_size 0 or Hi) and k_chunk_size >= 64.
            program_config: work-unit knobs (q_chunk_size, k_chunk_size,
                head_group_size; elements, tile-aligned). Defaults always fit
                L1; raise head_group_size (0 = all resident) for performance.
            compute_kernel_config: optional DeviceComputeKernelConfig. Only
                math_fidelity is honored (default: HiFi2, or LoFi when q and k
                are both bfloat8_b); fp32_dest_acc_en / dst_full_sync_en must
                stay false (the custom LLK is validated for bf16 DEST half-sync).

        Returns: score [B, 1, Sq, T] bf16 row-major; future/pad columns -inf.
        )doc",
        &ttnn::experimental::indexer_score,
        nb::arg("q"),
        nb::arg("k"),
        nb::arg("weights") = std::nullopt,
        nb::kw_only(),
        nb::arg("chunk_start_idx") = 0,
        nb::arg("apply_relu") = true,
        nb::arg("scale") = 1.0f,
        nb::arg("num_groups") = 1,
        nb::arg("program_config") = IndexerScoreProgramConfig{},
        nb::arg("compute_kernel_config") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::indexer_score::detail
