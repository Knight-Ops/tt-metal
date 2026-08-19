# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared tt-nn helpers for the Qwen3.6-35B-A3B demo (single-device oriented)."""
from __future__ import annotations

import os

from loguru import logger

import ttnn

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


def as_weight(source, mesh_device, dtype=ttnn.bfloat16, cache_file_name=None):
    """Convert an nn.Linear weight (out, in) to a tt tensor laid out for ttnn.linear (in, out).

    ``source`` is a torch weight or a zero-arg callable returning one; the read + transpose run only
    on a cache miss (a hit loads the cached .tensorbin and never touches the HF weight)."""

    def build():
        return _materialize(source).t().contiguous()

    return to_tt(build, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, cache_file_name=cache_file_name)
