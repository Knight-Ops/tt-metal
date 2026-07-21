# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm variants for Qwen3.6-35B-A3B.

- ``TtRMSNorm``: standard norm with the Qwen3-Next "(1 + weight)" convention. The +1 is folded
  into the stored weight at load time.
- ``TtRMSNormGated``: norm (weight ones-init) followed by gating with ``silu(gate)`` — used inside
  the Gated DeltaNet on the value head dim.
"""
from __future__ import annotations

import os

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_a3b.tt.common import to_tt

# Decode norms are a ~2.71 ms/step (~9.4%) dispatch-bound pool: the interleaved single-core
# ``ttnn.rms_norm`` is ~33 us/norm, but a WIDTH-SHARDED (8-core) rms_norm is ~3.2x cheaper (~10 us,
# PCC 1.0 vs interleaved). Sharding the full-hidden decode norms (input/post/final) is the
# tt_transformers "sharded-activation-stream" pattern — a config/memory-layout change, not a kernel.
# Isolated form (reshard in + sharded norm + reshard out) measured ~+3.8%; the reshards are net-neutral
# where the following matmul is already width-sharded. Gated by QWEN36_NORM_SHARDED (bench-gate at 40L
# before defaulting on — the reshard-vs-pipeline-overlap wash risk is real; see the se_down precedent).
# See tests/bench_norm_matmul_fusion.py and FUTURE_OPTIMIZATIONS.md.
_NORM_SHARDED = os.environ.get("QWEN36_NORM_SHARDED", "0") == "1"
_NORM_SHARD_NB = 8  # == P150 DRAM banks / optimal decode width-shard core count (matches build_dram_shard)


def _norm_shard_cfg(dim, nb=_NORM_SHARD_NB):
    """(program_config, width-sharded L1 mem_config) for a T=1 width-sharded rms_norm over `dim`,
    laid out identically to build_dram_shard's activation shard (nb cores on row 0, shard [32, dim/nb])."""
    Dt = (dim + 31) // 32
    sd = ((Dt + nb - 1) // nb) * 32  # sharded width (elems) per core, tile-aligned
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [32, sd], ttnn.ShardOrientation.ROW_MAJOR),
    )
    block_w = sd // 32
    subblock_w = 4
    while subblock_w > 1 and block_w % subblock_w:
        subblock_w -= 1
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[nb, 1], subblock_w=subblock_w, block_h=1, block_w=block_w, inplace=False
    )
    return pc, mem


class TtRMSNorm(LightweightModule):
    def __init__(self, mesh_device, weight, eps, add_unit_offset=True, dtype=ttnn.bfloat16):
        super().__init__()
        self.eps = eps
        w = weight.float()
        if add_unit_offset:
            w = 1.0 + w  # fold the (1 + weight) convention into the stored weight
        # gamma layout required by ttnn.rms_norm: [1, 1, dim//32, 32], row-major
        self.dim = w.numel()
        self.weight = to_tt(w.reshape(1, 1, self.dim // 32, 32), mesh_device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        # Sharded path precompute: only for full-hidden norms whose width tile-splits across nb cores.
        self._shard_pc = self._shard_mem = None
        if _NORM_SHARDED and self.dim % (_NORM_SHARD_NB * 32) == 0:
            self._shard_pc, self._shard_mem = _norm_shard_cfg(self.dim)

    def forward(self, x):
        # Sharded fast path for T=1 full-hidden norms: reshard interleaved residual -> width-sharded,
        # 8-core rms_norm (~3.2x cheaper), reshard back to interleaved (isolated form; returns the same
        # interleaved shape/layout as the default path so callers are unchanged). Row count must be 1.
        if self._shard_pc is not None and x.shape[-1] == self.dim and x.volume() == self.dim:
            orig_shape = list(x.shape)
            xs = ttnn.to_memory_config(ttnn.reshape(x, [1, self.dim]), self._shard_mem)
            ys = ttnn.rms_norm(
                xs, epsilon=self.eps, weight=self.weight, program_config=self._shard_pc, memory_config=self._shard_mem
            )
            return ttnn.reshape(ttnn.sharded_to_interleaved(ys), orig_shape)
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)


class TtRMSNormGated(LightweightModule):
    def __init__(self, mesh_device, weight, eps, dtype=ttnn.bfloat16):
        super().__init__()
        self.eps = eps
        dim = weight.numel()
        self.weight = to_tt(
            weight.float().reshape(1, 1, dim // 32, 32), mesh_device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    def forward(self, x, gate):
        y = ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)
        return ttnn.multiply(y, ttnn.silu(gate))
