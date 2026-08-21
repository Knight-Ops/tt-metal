# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reusable helpers for the indexed/gather MoE decode path on tt-metal.

These wrap the small, fiddly (and easy-to-get-wrong) plumbing that every MoE that adopts the
``ttnn.sparse_matmul`` indexed/gather mode needs, so it is written once instead of copied per model.

The gather mode (see ``ttnn.sparse_matmul(..., indices=...)``) makes the kernels iterate only the
``top_k`` active experts named by an on-device index list instead of scanning all ``E`` expert slots,
and returns a COMPACT output ``[..., top_k, M, N]`` where output slot ``i`` is expert ``indices[i]``.
Because ``ttnn.topk`` returns the values ``topv`` and ids ``topi`` paired (and sorted), the routed
combine is simply ``sum_i topv[i] * down_compact[i]`` over the ``top_k`` slots — no scatter to ``E``,
no reduce over ``E``.

Everything here is pure ttnn and host-sync-free (no ``to_torch``/``.item()``), so it is safe inside a
captured decode trace. Model-specific pieces (the activation between gate/up and down, the program
configs, whether gate+up are fused or split) deliberately stay in each model.
"""
from __future__ import annotations

import ttnn


def topk_to_indices(topi, top_k):
    """Turn ``ttnn.topk`` indices into the ``indices`` operand for ``ttnn.sparse_matmul`` gather mode.

    ``topi``: ``[T, top_k]`` (decode uses ``T == 1``) expert ids from ``ttnn.topk(..., sorted=True)``.
    Returns ``[1, 1, 1, top_k]`` UINT16 ROW_MAJOR on device — the dtype/layout the op validates. The
    ids' order is preserved, so it matches the compact output slot order. No host sync.
    """
    idx = ttnn.reshape(topi, [1, 1, 1, top_k])
    # topk emits UINT16 ids for vocab/expert counts <= 65535 (the MoE case); typecast defensively so
    # an UINT32 source (very large E) is still accepted as the op requires UINT16.
    if idx.dtype != ttnn.uint16:
        idx = ttnn.typecast(idx, ttnn.uint16)
    return ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT)


def gather_combine_weight(topv, top_k, num_leading_dims=3):
    """Reshape the routing weights for folding into the COMPACT down-projection *input*.

    ``topv``: ``[T, top_k]`` renormalized routing weights (paired with ``topi``). Returns a tensor that
    broadcasts ``topv[i]`` over compact down-input slot ``i`` (shape ``[1]*num_leading_dims + [top_k,
    1, 1]``). Folding the weight into the down input (vs the output) is cheaper because the down input
    has the smaller intermediate dim; ``down`` is a bias-free linear so
    ``sum_i topv_i * down(h_i) == sum_i down(topv_i * h_i)``. Optional optimization — ``gather_combine``
    on the output is the simpler default.
    """
    return ttnn.reshape(topv, [1] * num_leading_dims + [top_k, 1, 1])


def gather_combine(down_compact, topv, hidden, top_k):
    """Combine the compact down-projection output into the routed token vector.

    ``down_compact``: the gather-mode down output, any rank with total volume ``top_k * hidden`` (slot
    ``i`` is expert ``topi[i]``). ``topv``: ``[T, top_k]`` routing weights. Returns ``[hidden]`` =
    ``sum_i topv[i] * down_compact[i]`` — a reduce over ``top_k`` (~8), not ``E`` (~256). No host sync,
    no scatter.
    """
    down = ttnn.reshape(down_compact, [top_k, hidden])
    w = ttnn.reshape(topv, [top_k, 1])
    return ttnn.sum(ttnn.multiply(down, w), dim=0)
