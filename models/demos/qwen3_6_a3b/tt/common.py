# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared tt-nn helpers for the Qwen3.6-35B-A3B demo (single-device oriented)."""
from __future__ import annotations

import os

from loguru import logger

import ttnn

# Use the tuned wide 1D mcast_in0 program configs on the decode (M<=32) matmuls that otherwise run on
# the ttnn AUTO config: MoE gate_w / se_gate_up / se_down, attention wk / wv, the fused gated-delta
# w_in_proj, and lm_head. Measured per-matmul in tests/bench_matmul_layout.py (variant D vs column A);
# see wide_1d_decode_pc below and FUTURE_OPTIMIZATIONS.md "Lever 1 AMENDED". Program-config only, so
# it cannot change numerics. QWEN36_WIDE1D_DECODE=0 reverts every site for an A/B.
WIDE1D_DECODE = os.environ.get("QWEN36_WIDE1D_DECODE", "1") != "0"
# These configs are NOT numerically neutral: in0_block_w is the K-block size, so it changes the
# matmul's accumulation ORDER, and at bf16/bf8 that shifts the low bits.
#
# `gate_w` is therefore DEFAULT-OFF, and it is the only site that is. It is worth ~0.70 ms/token (the
# largest single win in the set, AUTO 29.7 -> 12.1 us x 40 layers), but the router matmul feeds a
# `topk`, so a low-bit shift can flip which expert is 8th -- a DISCRETE change in which experts fire,
# not just a rounding difference. Measured, bisected per site with QWEN36_WIDE1D_SITES against
# tests/test_moe_sparse.py::test_moe_gather_decode (gate 0.99):
#     off              PASS
#     se_gate_up       PASS
#     se_down          PASS
#     se_gate_up,se_down PASS
#     gate_w           FAIL, PCC 0.9857   <- on its own
#     all              FAIL, PCC 0.9852
# Re-enable it with QWEN36_WIDE1D_SITES=all once there is an MMLU-Redux baseline to judge it against
# (evaluation/published_scores.json is still all null), which is this project's rule for anything that
# changes what the model computes. A cheaper alternative first: sweep gate_w's in0_block_w for a value
# that keeps most of the speed with an accumulation order closer to the ttnn heuristic's.
# `lm_head` is also omitted, for the opposite reason to gate_w: it is worth EXACTLY ZERO. Isolated it
# is 1125.9 -> 1035.7 us (-90 us); in the real 40-layer step it is 1.225 -> 1.226 ms, and the full-model
# delta is fully accounted for by the other three sites. So enabling it buys nothing and still perturbs
# the low bits of the largest matmul in the step -- strictly the wrong trade. (It stays selectable, and
# the 2026-08-22 accuracy measurements were taken WITH it on, so they bound a slightly larger
# perturbation than what now ships.) See FUTURE_OPTIMIZATIONS "Lever 1 AMENDED".
_WIDE1D_DEFAULT_SITES = "se_gate_up,se_down,wk,w_in_proj"
_WIDE1D_SITES = os.environ.get("QWEN36_WIDE1D_SITES", _WIDE1D_DEFAULT_SITES)


def wide1d_enabled(site: str) -> bool:
    """Is the tuned wide-1D decode config enabled for `site`?

    Sites: gate_w, se_gate_up, se_down, wk, w_in_proj, lm_head. `QWEN36_WIDE1D_SITES` takes a
    comma-separated subset, or "all"/"none"; the default omits `gate_w` (see above).
    `QWEN36_WIDE1D_DECODE=0` disables every site, for a single-knob A/B.
    """
    if not WIDE1D_DECODE or _WIDE1D_SITES == "none":
        return False
    return _WIDE1D_SITES == "all" or site in {t.strip() for t in _WIDE1D_SITES.split(",")}


DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG

# Warn once (not per-tensor) if the on-disk weight cache can't be used and we fall back.
_cache_fallback_warned = False


def _cache_exists(cache_file_name, dtype, layout) -> bool:
    """Whether ttnn.as_tensor's cache file for this (name, dtype, layout) is already on disk. Mirrors
    the ``_dtype_..._layout_....tensorbin`` suffix ttnn appends (ttnn/ttnn/operations/core.py)."""
    path = f"{cache_file_name}_dtype_{dtype.name}_layout_{layout.name}.tensorbin"
    return os.path.isfile(path)


def _materialize(source):
    """``source`` is a torch tensor or a zero-arg callable returning one. The callable form lets us
    defer the (potentially multi-GB) HF read until we know the weight actually has to be read."""
    return source() if callable(source) else source


def to_tt(
    source,
    mesh_device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=DRAM,
    cache_file_name=None,
):
    """Replicate a torch tensor onto the (single- or multi-) device mesh.

    ``source`` may be a torch tensor OR a zero-arg callable returning one. The callable is invoked
    only when the weight must actually be read (no cache, or a cache miss), so a cache hit never
    materializes the HF weight.

    If ``cache_file_name`` is given, the converted (quantized + tilized) tensor is cached to disk via
    ``ttnn.as_tensor``: the first build writes a ``.tensorbin``; later runs load it directly, skipping
    the block-float quantization + tilize (the bulk of the ~800s/40-layer cold load) AND the HF read.
    ttnn appends a ``_dtype_<dtype>_layout_<layout>.tensorbin`` suffix, so a dtype/layout change never
    reuses a stale file. Falls back to a plain ``from_torch`` (with one warning) if caching fails."""
    global _cache_fallback_warned
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device is not None else None
    if cache_file_name:
        try:
            if _cache_exists(cache_file_name, dtype, layout):
                import torch

                # Hit: as_tensor loads the .tensorbin and ignores this argument, so don't read the
                # (multi-GB) HF weight just to hand it one.
                src = torch.empty((), dtype=torch.bfloat16)
            else:
                src = _materialize(source)
                os.makedirs(os.path.dirname(cache_file_name), exist_ok=True)
            return ttnn.as_tensor(
                src,
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=memory_config,
                mesh_mapper=mapper,
                cache_file_name=cache_file_name,
            )
        except Exception as e:  # noqa: BLE001 - any cache failure must not be fatal
            if not _cache_fallback_warned:
                logger.warning(f"weight cache unavailable ({e!r}); loading weights without cache")
                _cache_fallback_warned = True
    return ttnn.from_torch(
        _materialize(source),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=mapper,
    )


def from_tt(tt_tensor, mesh_device=None):
    """Bring a (replicated) tt tensor back to torch, taking the first device shard."""
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0) if mesh_device is not None else None
    if composer is not None:
        t = ttnn.to_torch(tt_tensor, mesh_composer=composer)
        # replicated -> all shards identical; if concatenated along dim0 take the first copy
        n = mesh_device.get_num_devices()
        if n > 1 and t.shape[0] % n == 0:
            t = t[: t.shape[0] // n]
        return t
    return ttnn.to_torch(tt_tensor)


def build_dram_shard(w, K, N, nb=8):
    """Width-shard an already-loaded weight ``w`` [K, N] across ``nb`` DRAM banks (P150 has 8) and return
    ``(w_dram, act_mc, out_mc, program_config)`` for a T=1 DRAM-sharded decode matmul.

    At T=1 a matmul that reads its weight from interleaved DRAM is only ~25-70% bandwidth-efficient;
    width-sharding the weight across the DRAM banks + L1-width-sharding the activation ([32, K/nb]) and
    output ([32, N/nb]) runs it up to ~2.7x faster (measured — biggest gain on K-heavy / narrow-N output
    projections). ``nb``=8 (== DRAM banks) is optimal; more compute cores only add multicast overhead at
    M=1 (measured sweep). Caller resards the activation into ``act_mc`` and the output back out. See
    FUTURE_OPTIMIZATIONS.md."""
    Nt, Kt = (N + 31) // 32, (K + 31) // 32
    sn, sk = ((Nt + nb - 1) // nb) * 32, ((Kt + nb - 1) // nb) * 32
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})

    def smc(h, ww, buf):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, buf, ttnn.ShardSpec(cores, [h, ww], ttnn.ShardOrientation.ROW_MAJOR)
        )

    w_dram = ttnn.to_memory_config(w, smc(K, sn, ttnn.BufferType.DRAM))
    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=sk // 32, per_core_M=1, per_core_N=sn // 32, fused_activation=None
    )
    return w_dram, smc(32, sk, ttnn.BufferType.L1), smc(32, sn, ttnn.BufferType.L1), pc


def wide_1d_decode_pc(m, k, n, num_cores, grid_w=11, in0_block_w=None, fp32_acc=False, fused_activation=None):
    """1D (``mcast_in0``) matmul program config for a skinny (M <= 32) decode matmul on an
    **INTERLEAVED** weight — the alternative to ``build_dram_shard``'s DRAM-width-sharded form.

    Why this exists as a second option. ``build_dram_shard`` width-shards the weight across the 8 DRAM
    banks and needs the activation resharded in and the output resharded out on every call (~8 us the
    pair), plus a SECOND copy of every weight it is applied to (the interleaved original stays alive
    for prefill — ~510 MB across the model). A tuned 1D mcast config on the plain interleaved weight
    needs neither. Two independent single-P150 implementations measure the interleaved form as the
    faster one on exactly this class of matmul:

      * ``models/demos/blackhole/qwen36/tt/model_config.py:179-230`` (same architecture, measured):
        gate/up 11x4 42.8us vs 43.9 at 8x4; down 11x3 +28% vs 8x2; gdn_qkvz 11x4 +22% vs 8x5;
        attn_wo and gdn_out 11x3 +25% vs 8x4. Mechanism: for a fixed core budget a WIDE-SHORT grid
        shortens the in0 multicast column, so shape the grid width-first up to ``grid_w`` (11 on a
        P150 — a harvested part exposes 11 worker columns, not 13).
      * MuseGlimmer's decode ``down_proj``: grid 11x10 (all 110 cores), per_core_M=1, per_core_N=2,
        interleaved weight, and ``in0_block_w=4`` with the note that four K tiles "nearly halves this
        projection's device time" by fixing DRAM-reader under-utilization on Blackhole.

    ``in0_block_w`` only has to DIVIDE ``k_tiles`` (mcast_in0 streams the full K on every core), so the
    largest legal divisor is usually much bigger than ``k_tiles // grid_x``. Default: the largest
    divisor <= 8. It is an L1 bound as well as a legality bound — it sizes the in0 circular buffer,
    which competes with any resident L1 output for the same 1536 KB — so a value that wins in a
    standalone sweep can still overflow in the full model. Validate at 40 layers, never per-op.

    Returns just the program config; the caller passes the activation as ordinary L1/DRAM interleaved
    (no reshard) and reads the output back the same way.
    """
    cols = min(grid_w, num_cores)
    rows = max(1, -(-num_cores // cols))
    m_tiles, k_tiles, n_tiles = -(-m // 32), -(-k // 32), -(-n // 32)
    if in0_block_w is None:
        in0_block_w = next((d for d in range(8, 0, -1) if k_tiles % d == 0), 1)
    assert k_tiles % in0_block_w == 0, f"in0_block_w={in0_block_w} must divide k_tiles={k_tiles}"
    per_core_n = max(1, -(-n_tiles // (cols * rows)))
    cap = 4 if fp32_acc else 8  # fp32 dest-acc halves the DST subblock budget on Blackhole
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_tiles % i == 0 and i * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def as_weight(source, mesh_device, dtype=ttnn.bfloat16, cache_file_name=None):
    """Convert an nn.Linear weight (out, in) to a tt tensor laid out for ttnn.linear (in, out).

    ``source`` is a torch weight or a zero-arg callable returning one; the read + transpose run only
    on a cache miss (a hit loads the cached .tensorbin and never touches the HF weight)."""

    def build():
        return _materialize(source).t().contiguous()

    return to_tt(build, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, cache_file_name=cache_file_name)
