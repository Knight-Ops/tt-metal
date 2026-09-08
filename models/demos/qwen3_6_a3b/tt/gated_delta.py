# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-nn Gated DeltaNet (linear attention) for Qwen3.6-35B-A3B.

The delta-rule recurrence runs ON DEVICE (no host round-trip). Three paths, by shape/mode:

- DECODE (T=1), default: a fused single-step tt-lang kernel (``decode_step_tt`` in ttl_delta.py)
  collapses the whole per-head step into ONE launch (QWEN36_GDN_FUSED=1). Falls back to the batched-
  matmul recurrent scan (``state`` [1,V,Dk,Dv]) with QWEN36_GDN_FUSED=0.
- PREFILL (T>1), default: the chunked delta-rule with the per-chunk recurrence run in ttnn at
  HiFi4 + fp32 accumulation (``_chunk_state_ttnn``) — numerically stable and coherent over all 40
  layers. This is what demo.py and demo/server.py use by default.
- PREFILL, EXPERIMENTAL: the fused tt-lang ``chunk_state_tt`` kernel (bf16 DST). Faster but
  numerically approximate — it drifts to incoherence over many layers (see the _STABLE_PREFILL note
  below), so it is OPT-IN only (the traced-prefill path, and QWEN36_DELTA_STABLE_PREFILL=0). NOT a
  production accuracy path. Promoting it would need fp32 DST accumulation, which exceeds the ttl
  16-tile DST capacity and thus requires per-matmul tiling (future work).

The recurrent form is mathematically identical to the chunked form (verified) and is the reference
for the kernels.
"""

from __future__ import annotations

import contextlib
import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_a3b.tt import prefill_profiler as prof
from models.demos.qwen3_6_a3b.tt import signpost as sp
from models.demos.qwen3_6_a3b.tt.common import as_weight, build_dram_shard, to_tt, wide1d_enabled, wide_1d_decode_pc
from models.demos.qwen3_6_a3b.tt.rms_norm import TtRMSNormGated
from models.demos.qwen3_6_a3b.tt.ttl_delta import (
    _BATCH_SHARED_CONST,
    chunk_state_tt,
    decode_step_batch_tt,
    decode_step_chain_tt,
    decode_step_tt,
)

# Fused chunked prefill: replace the sequential recurrent scan (O(T) dispatches) with the chunked
# delta-rule — ttnn per-chunk prep batched over heads + the fused tt-lang _chunk_state kernel (one
# launch over all heads per chunk). This is the DEFAULT (5.4-5.7x faster prefill at 40 layers, warm);
# set QWEN36_FUSED_PREFILL=0 to fall back to the sequential scan. Decode (T=1) always uses the scan.
# Intra-chunk inverse (I-L)^-1 uses numerically-stable recursive block inversion (see _chunk_prep).
# The fast T≈I+L approximation and the old doubling product were both removed: both are INACCURATE for
# this model's strong-decay heads (they explode / drift to gibberish on real L). First fused prefill
# compiles the ttl kernel once (~1 min, cached on disk thereafter).
_FUSED_PREFILL = os.environ.get("QWEN36_FUSED_PREFILL", "1") == "1"
# Run the per-chunk prep in bf16 instead of fp32: the prefill is dispatch-bound and the fp32 path adds
# ~124 typecast ops/forward (up to fp32, back to bf16) plus 2x data movement. OFF by default: the
# stable recursive inverse + cumsum/exp are precision-sensitive (bf16 prep regresses 40-layer
# coherence). QWEN36_DELTA_BF16_PREP=1 enables (perf experiments only).
_PREP_DT = ttnn.bfloat16 if os.environ.get("QWEN36_DELTA_BF16_PREP") == "1" else ttnn.float32
_CHUNK = 64  # chunk_size, matches reference torch_chunk_gated_delta_rule default
# Ragged (right-padded) incremental blocks: process the PADDED width and MASK the pad rows out of the
# recurrent state, instead of slicing the block down to `valid_len`.
#
# This is a SHAPE-SPACE fix, not a numerics one. Slicing makes every op downstream of it (this module
# and the whole MoE) run at T = valid_len = prompt_len mod chunk -- a different value every turn -- so
# each turn compiles ~90 new device programs. The program cache grows without bound and the device
# HANGS past roughly 700-850 entries: measured at 40 layers with DRAM 10.6 GB free, L1 1% used and the
# trace region 90% free, so it is not a memory limit. Masking keeps the width at the padded value, of
# which there are only four (128/256/384/512), so the cache saturates. See IMPLEMENTATION_GUIDE.md.
#
# Correctness rests on the same identity the bucketed traced path already uses: beta=0 makes a row
# contribute nothing to the state (kbeta = vbeta = 0) and g=0 (log space) makes it decay nothing, so a
# masked row is a no-op in the recurrence. The conv is handled separately -- its carried state must
# still come from the last REAL rows, see _conv_silu's `state_at`.
# QWEN36_GDN_RAGGED_MASK=0 restores the slicing path (they are PCC-gated against each other in
# tests/test_ragged_mask.py).
_RAGGED_MASK = os.environ.get("QWEN36_GDN_RAGGED_MASK", "1") != "0"
# Route the chunked PREFILL recurrence through tt-metal's native fused op
# ``ttnn.transformer.chunk_gated_delta_rule`` (new in v0.77.0) instead of the in-tree chunk prep +
# ttl/ttnn chunk-state loop. Measured on a single P150 at real GDN dims (32 v-heads, Dk=Dv=128):
# 4.4x (T=64) to 11.8x (T=512) on the recurrence in isolation, core/state PCC ~0.999998.
# DEFAULT ON (QWEN36_GDN_FUSED_OP=0 reverts): measured 40-layer eager prefill 672.3 -> 450.3 ms at
# T=256, 1277.3 -> 837.3 ms at T=512, 2489.4 -> 1607.4 ms at T=1024 (1.49-1.55x). Eager path only — the traced path keeps its
# pre-allocated buffer pool (the op allocates its own outputs), so `pool is not None` falls through;
# that costs nothing, since tracing the prefill measures only ~1.02x once this op is on.
_GDN_FUSED_OP = os.environ.get("QWEN36_GDN_FUSED_OP", "1") != "0"
# The fused op runs at chunk 32, NOT _CHUNK(64): 128 exceeds the L1 CB budget, and at 64 the WY
# matrix becomes a 2x2 tile-block whose bottom-right 32x32 sub-block can be ill-conditioned enough
# that the fp32 block inverse loses precision. At 32 each WY matrix is one tile solved by the
# 16x16-blocked inverse. Same math, different internal tiling (upstream
# models/demos/blackhole/qwen36/tt/gdn/fused_chunk.py reached the same conclusion).
_FUSED_OP_CHUNK = 32
# Minimum head dim for the fused op. It hangs silently below this (see the guard in
# _forward_prefill_chunked); 128 is the only size upstream ships, tests, or that was verified here.
_FUSED_OP_MIN_HEAD_DIM = int(os.environ.get("QWEN36_GDN_FUSED_OP_MIN_HEAD_DIM", "128"))
# Run the per-chunk prep ONCE batched over all chunks (state-independent) instead of Nc sequential
# rounds — see _forward_prefill_chunked. DEFAULT on; QWEN36_DELTA_BATCH_PREP=0 reverts (A/B fallback).
_BATCH_PREP = os.environ.get("QWEN36_DELTA_BATCH_PREP", "1") != "0"


# Keep the tiny gated-delta DECODE intermediates L1-resident instead of round-tripping interleaved
# DRAM: the decode path is many tiny dispatch-bound ops, so on-chip residency cuts the step ~16%
# (measured, PCC-identical). Default on; QWEN36_GDN_L1=0 reverts to interleaved DRAM.
_GDN_L1 = os.environ.get("QWEN36_GDN_L1", "1") != "0"
_MC = ttnn.L1_MEMORY_CONFIG if _GDN_L1 else None


@contextlib.contextmanager
def _nullctx():
    """No-op stand-in for prof.phase on the T==1 decode path.

    The GDN forward is shared between prefill and decode, and `prof.phase` SYNCS THE DEVICE -- doing
    that inside a decode step would break trace capture (see the note in moe.py forward()). The
    prefill sub-phase timers below are therefore gated on T > 1 and fall back to this. Costs nothing
    when QWEN36_PROFILE_PHASES is unset either way, since prof.phase is itself a no-op then."""
    yield


# DECODE conv as an add-chain over K-1 separate [1,conv_dim] history-row buffers instead of the
# concat+multiply+row-sum+slice path. The two dim-0 (row) ops in the stacked path — concat([state,mixed])
# and slice(xpad[1:K]) — are dispatch-bound and cost ~0.075 ms/layer EACH (measured, ~87% of the decode
# conv); the row-assembly can't be made cheaper in place (ttnn slice/concat have no output_tensor= and
# tiled tensors can't be sub-region-written). Keeping history as separate rows makes the sliding-window
# shift 3 cheap equal-shape copies and the compute 4 mul + 3 add over [1,conv_dim], removing both dim-0
# ops: measured 0.430 -> 0.318 ms/layer (~3.4 ms/token, ~10% of decode). NOT bit-identical (bf16 add-chain
# vs fp32 row-sum) but PCC-validated == baseline (0.99979 on test_gated_delta_decode; the accumulation
# matches, unlike the reverted on-core kernel's matmul-ones row-sum which drifted to 0.885). conv_rows is
# populated once from conv_state at start_decode (prefill stays on the unchanged concat/stacked path).
# Default ON (measured 40L: decode 31.5->34.6 tok/s/user, +9.8%; teacher-forced decode logit-PCC vs the
# baseline mean 0.996, stable); QWEN36_CONV_ADDCHAIN=0 reverts to the concat/row-sum path. See
# FUTURE_OPTIMIZATIONS.md Lever 3b.
_CONV_ADDCHAIN = os.environ.get("QWEN36_CONV_ADDCHAIN", "1") != "0"

# Fused single-step (T=1) DECODE kernel: collapse the whole per-head gated-delta decode step
# (l2norm(q)*scale + l2norm(k) + gate/beta + rank-1 recurrence + gated RMSNorm) into ONE ttl launch
# over all value heads, instead of ~30 tiny dispatch-bound ttnn ops. The 4 projections (w_qkv/w_z/
# w_ba/w_out) + the conv stay in ttnn. DEFAULT on; QWEN36_GDN_FUSED=0 reverts to the recurrent scan.
# The kernel compiles once per (n_v_heads, n_k_heads, head-dim) shape (~1 min, disk-cached).
_GDN_FUSED = os.environ.get("QWEN36_GDN_FUSED", "1") != "0"
# Fold the K-step speculative-verify recurrence ONTO the core (one ttl launch for the whole chain)
# instead of driving it from Python. Not a kernel-time win — it deletes 6 of the 7 ttnn ops per step
# per layer, and the verify trace is dispatch-bound. QWEN36_GDN_VERIFY_CHAIN=0 reverts to the
# per-step-launch path (both are PCC-gated in tests/test_gated_delta_verify.py). See EXPERIMENTS.md.
# Alias the fused decode kernel's S and Snew arguments to the SAME persistent buffer, so the state is
# updated in place on-core and the separate `ttnn.copy(snew_buf -> state_buf)` disappears. Safe by
# construction: core h reads only S[h*kt_t : (h+1)*kt_t] and writes exactly that slab (ttl_delta.py
# :501/:614), the slabs are disjoint across cores, and within a core read() completes (t7.wait())
# before compute() runs and write() fires. Saves 2 of the 4 MB/layer/token the state costs today.
# QWEN36_GDN_ALIAS_STATE=0 reverts to the separate-buffer + copy form.
_GDN_ALIAS_STATE = os.environ.get("QWEN36_GDN_ALIAS_STATE", "1") != "0"
# Keep the gated-delta recurrent state in FLOAT32 through decode. MEASURED (tests/probe_gate_vs_state.py,
# real layer-0 weights, 8192 steps vs an all-fp32 recurrence): the bf16 decode drifts to state PCC
# 0.9327 / |S| x1.55 through TWO mechanisms that partially cancel -- bf16 rounds any decay g>0.998046875
# to exactly 1.0 (deleting it; 49% of this layer's gate values are above that cliff), and the bf16 carry
# rounds S*g back to S. Each ALONE is worse than both together, so the cheap `d=1-g` reform is a
# REGRESSION (0.9122); only fp32 gate AND carry fix it (0.999998 / x1.000). TF32 is not enough either
# (0.9924 / x1.071) -- its cliff at g>0.999756 still deletes this layer's longest-memory head.
# It has to be an ALL-fp32 kernel: ttl 1.1.6 rejects an f32 tile argument next to any bf16 one, so the
# cheap "fp32 carry, bf16 read-outs" split is not expressible (see _get_decode_op's docstring). Hence
# the input typecasts below -- bf16 read-outs are numerically free (floor 0.999998), we just cannot ask
# for them. QWEN36_GDN_FP32_STATE=0 reverts to the bf16 state.
_FP32_STATE = os.environ.get("QWEN36_GDN_FP32_STATE", "0") != "0"
_STATE_DT = ttnn.float32 if _FP32_STATE else ttnn.bfloat16
_GDN_VERIFY_CHAIN = os.environ.get("QWEN36_GDN_VERIFY_CHAIN", "1") != "0"
# ...but NOT with an fp32 state. The chained kernel is at 32/32 DFBs and only fits because `outer`
# doubles as the step-to-step carry (EXPERIMENTS.md:64-96) -- and `outer` is a 16-tile matmul output,
# the one shape fp32 DST cannot hold. The per-step path re-uses `decode_step_tt`, which already has an
# fp32 build, for no new kernel risk. Cost is what EXPERIMENTS.md #5 measured for the chain: -1.53 ms
# on a ~50 ms speculative round (~3%), and it is not on the single-user decode path at all.
_GDN_VERIFY_CHAIN = _GDN_VERIFY_CHAIN and not _FP32_STATE
# Keep speculative-verify intermediates in L1 for K up to this (rows are tiny; 8 is far inside budget).
_VERIFY_L1_MAX_K = int(os.environ.get("QWEN36_VERIFY_L1_MAX_K", "8"))
# MEASURED DEAD END (EXPERIMENTS.md #7): a low-op-count "stacked window" form of the causal conv
# (4 slices + concat + multiply + sum = 8 ops) is 38% SLOWER than the 11-op loop form below
# (0.6552 vs 0.4752 ms/layer), because `concat` and a `sum` reduction are in the expensive
# data-movement class while small broadcast-multiplies are not. Do not "optimise" this by op count.
TILE = 32

# Batched (B>1 multi-user) fused decode kernel: one ttl launch (decode_step_batch_tt) does the whole
# gated-delta step for all B users, each core looping its value-head over the B independent users and
# streaming per-user state from the persistent cache. Measured 11-28% faster per step than the ttnn scan
# (_forward_decode_batch), widening with B: 40L B=32 -> 4.52x single-user (157.5 tok/s) vs the scan's
# 3.27x. Only used for T>1 (B>1); B=1/T=1 keeps _forward_decode_fused (the batched path's relayout +
# interleaved w_out make it slower at B=1). Default ON; QWEN36_GDN_BATCH_FUSED=0 reverts to the scan.
# See models/demos/qwen3_6_a3b/Fused_gdn_handoff.md.
_GDN_BATCH_FUSED = os.environ.get("QWEN36_GDN_BATCH_FUSED", "1") != "0"

# Fuse the 3 input projections (in_proj_qkv -> conv_dim, in_proj_z -> value_dim, in_proj_b|a -> 2V)
# into ONE [hidden, conv_dim+value_dim+2V] matmul: a single dispatch + weight read per step instead of
# three (the T=1 decode path is dispatch-bound). The output is sliced [conv_dim | value_dim | 2V] in
# forward. Also makes the input projection a single big matmul, the natural unit for DRAM sharding
# (QWEN36_GDN_DRAM_SHARD). Default on; QWEN36_GDN_FUSE_IN=0 loads the three weights separately.
_FUSE_IN = os.environ.get("QWEN36_GDN_FUSE_IN", "1") != "0"

# DRAM-shard the DECODE output projection (w_out): at T=1 the [value_dim, hidden] matmul reads the
# weight from interleaved DRAM at only ~25% BW efficiency; width-sharding it across the 8 DRAM banks +
# L1-width-sharding the activation runs it ~2.7x faster (measured, incl. the in/out reshards). Decode-
# only (prefill keeps the interleaved w_out). Default on; QWEN36_GDN_DRAM_SHARD=0 reverts.
_DRAM_SHARD = os.environ.get("QWEN36_GDN_DRAM_SHARD", "1") != "0"
_DRAM_BANKS = 8  # P150 DRAM controllers

# Numerically-stable chunked PREFILL (default on). Two independent issues made the original chunked
# prefill emit gibberish for prompts >1 chunk on this model's strong-decay heads (g down to ~-92/step,
# g_cum to ~-5888), while the recurrent scan stayed correct:
#   (1) The intra-chunk inverse (I-L)^-1. The T≈I+L approximation is wrong when L is O(1), and the
#       fp32 doubling product (I+L)(I+L^2)...(I+L^32), though exact for nilpotent L, EXPLODES in finite
#       precision (intermediate L^k have huge singular values that must telescope but don't): we
#       measured 1e6-1e11 inverses where the true inverse is ~1.0. Fixed unconditionally in _chunk_prep
#       via stable recursive block inversion (bounded, matches reference; 12 batched matmuls).
#   (2) The bf16 ttl chunk_state kernel itself: even with a correct inverse, the bf16 chunk-recurrence
#       (v_new/out/Snew matmuls) accumulates too much error over 30 layers (confirmed still gibberish).
#       Fix: run the per-chunk recurrence in ttnn at HiFi4 + fp32 accumulation (near-fp32, matching the
#       reference) — _chunk_state_ttnn. Keeps the chunked parallelism (fast, O(seq/chunk)) and is
#       numerically correct. This is the DEFAULT eager prefill.
# _STABLE_PREFILL gates ONLY the eager path (use_stable = _STABLE_PREFILL and pool is None). The TRACED
# prefill path (pool != None: forward_prefill_traced, QWEN36_SERVER_PREFILL_TRACE=1, and the
# bench/prof scripts) ALWAYS uses the bf16 ttl kernel — it needs pre-allocated buffers and the fp32
# ttnn path allocates intermediates in-graph, which trace capture forbids. So traced prefill is the
# faster-but-numerically-approximate path and is EXPERIMENTAL/opt-in for that reason; the shipping
# server prefill is eager (see demo/server.py: traced is gated on QWEN36_SERVER_PREFILL_TRACE).
_STABLE_PREFILL = os.environ.get("QWEN36_DELTA_STABLE_PREFILL", "1") != "0"
# Route the TRACED prefill recurrence through the fp32/HiFi4 ttnn path (_chunk_state_ttnn) instead of
# the bf16 ttl chunk_state kernel. DEFAULT ON: the bf16 kernel is both LESS accurate (PCC ~0.996/40L,
# drifts) AND ~1.48x SLOWER than the ttnn path (MEASURED 40L, seq256: 1114 vs 754 ms) — the ttnn
# matmuls use the full 110-core grid at HiFi4 (multi-pass ~fp32 inputs) while the ttl kernel runs one
# head/core on 32 cores at single-pass LoFi. The old belief that the fp32 path's in-graph intermediates
# are forbidden in capture is FALSE: only host writes (zeros/fills) are forbidden; matmul/eltwise
# outputs allocate in DRAM and capture fine. MEASURED single-bucket traced vs eager: PCC 1.00000,
# 1.55x @128 / 1.14x @256 / 1.01x @512 (tracing helps most at short prompts) — SINGLE-BUCKET IS SOLID.
# MULTI-BUCKET: FIXED 2026-08-22, and it was never a tt-metal bug. The old caveat here said capturing
# several bucket lengths in one process corrupts the second trace's replay and to deploy a SINGLE
# bucket. Root cause: a trace bakes its buffers' ADDRESSES, nothing pinned the in-graph transients, so
# they were freed at end_trace_capture and the NEXT bucket's persistent pool was allocated into that
# hole -- replaying the earlier trace then wrote over the later bucket's pool. Observed directly:
# bucket 512's read-only zero `S0` came back |max| 33.19 and its `out[0]` came back NaN after one
# bucket-256 replay. The earlier note's own evidence (pooling to ~40 MB and doubling trace_region_size
# both changed nothing) was right that it is not a footprint overflow, and pointed at exactly this.
# Fixed by allocating every bucket's persistent buffers BEFORE any capture (model.setup_prefill_traces
# is two-pass; see TtModel.alloc_prefill_buffers). All orders now replay at PCC 1.000000. Full
# write-up, including the checklist for avoiding this class of bug: MEMORY.md.
# Set =0 to revert to the bf16 kernel.
_TRACED_FP32 = os.environ.get("QWEN36_TRACED_FP32_RECURRENCE", "1") != "0"
# Number of alternating `Snew` state-carry buffers in the traced-prefill pool (build_trace_pool).
# 2 is all the recurrence needs; QWEN36_POOL_SNEW_ALL=1 restores the old one-per-chunk allocation as
# an A/B fallback.
_POOL_SNEW_ALL = os.environ.get("QWEN36_POOL_SNEW_ALL", "0") != "0"
# Keep the traced pool's per-chunk `out` buffers in bf16 rather than fp32. `out` is a READ-OUT, not the
# fp32 state carry (which stays fp32) -- and the path already typecasts the concatenated `core` to bf16
# on the way out, so rounding each chunk before the concat instead of the whole thing after it is
# BIT-IDENTICAL: elementwise rounding commutes with concatenation, which is pure data movement. Halves
# the dominant term of the pool at long buckets and deletes the trailing typecast.
# QWEN36_POOL_OUT_FP32=1 reverts to fp32 out buffers.
_POOL_OUT_BF16 = os.environ.get("QWEN36_POOL_OUT_FP32", "0") == "0"
_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


# Express the q/k L2 norm as an RMS norm instead of a hand-rolled reduction chain. Identity:
#     l2norm(x) = x / sqrt(sum(x^2) + eps)
#               = x / (sqrt(K) * sqrt(mean(x^2) + eps/K))
#               = rms_norm(x, epsilon=eps/K) * K**-0.5
# so 5-6 ops (multiply, sum, add, rsqrt, multiply [, multiply]) collapse to 2, with any caller scale
# folded into the single trailing multiply. Same substitution the upstream implementation uses
# (models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py:235).
#
# Worth doing because `delta.qknorm` is 24% of the gated-delta PREFILL block -- 2.65 ms/layer at
# T=1024 against the recurrence's 2.10 (see the QWEN36_PROFILE_PHASES sub-phases). The recurrence is
# no longer where gated-delta prefill spends its time; the glue around it is.
# QWEN36_GDN_L2NORM_RMS=0 restores the reduction chain.
_L2NORM_RMS = os.environ.get("QWEN36_GDN_L2NORM_RMS", "1") != "0"


def _l2norm_scale_lastdim(x, scale=None, eps=1e-6, mc=None):
    """mc: memory config for the intermediates. Callers pass _MC only for DECODE (T==1), where the
    L1 residency is a measured win. Prefill must leave it None — at T=1711 the two calls here hold
    ~42 MB of L1 whose floor collides with the fused recurrence's CBs (see _conv_silu)."""
    if _L2NORM_RMS:
        K = x.shape[-1]
        y = ttnn.rms_norm(x, epsilon=eps / K, memory_config=mc)
        f = K**-0.5 if scale is None else scale * K**-0.5
        return ttnn.multiply(y, f, memory_config=mc)
    sq = ttnn.sum(ttnn.multiply(x, x, memory_config=mc), dim=-1, keepdim=True, memory_config=mc)
    y = ttnn.multiply(x, ttnn.rsqrt(ttnn.add(sq, eps, memory_config=mc), memory_config=mc), memory_config=mc)
    return ttnn.multiply(y, scale, memory_config=mc) if scale is not None else y


class TtGatedDeltaNet(LightweightModule):
    def __init__(self, mesh_device, weights, cfg, dtype=ttnn.bfloat8_b, cache_path=None):
        super().__init__()
        self.mesh_device = mesh_device
        cn = (lambda role: f"{cache_path}/{role}") if cache_path else (lambda role: None)

        # weights[*] may be lazy thunks (production loader) or plain tensors (module tests). W()
        # materializes either; for cached weights it is only invoked on a cache miss.
        def W(k):
            v = weights[k]
            return v() if callable(v) else v

        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_k = cfg.linear_conv_kernel_dim
        self.n_rep = self.num_v_heads // self.num_k_heads
        self.eps = cfg.rms_norm_eps
        self.qk_scale = self.head_k_dim**-0.5

        # weights[*] are lazy thunks (see load_checkpoints): projections passed straight through to
        # as_weight (read only on a cache miss); tiny conv/norm/A_log/dt_bias tensors read eagerly.
        # Output slice offsets for the fused input projection: [0:conv_dim] -> qkv (conv input),
        # [conv_dim:+value_dim] -> z, [+2V] -> b|a (b first, a second).
        self._in_off = (
            self.conv_dim,
            self.conv_dim + self.value_dim,
            self.conv_dim + self.value_dim + 2 * self.num_v_heads,
        )
        if _FUSE_IN:
            # One fused matmul weight = concat(in_proj_qkv, in_proj_z, in_proj_b, in_proj_a) along the
            # output dim (nn.Linear [out, hidden]); as_weight transposes to [hidden, N_total].
            self.w_in_proj = as_weight(
                lambda: torch.cat([W("in_proj_qkv"), W("in_proj_z"), W("in_proj_b"), W("in_proj_a")], dim=0),
                mesh_device,
                dtype=dtype,
                cache_file_name=cn("w_in_proj"),
            )
        else:
            self.w_qkv = as_weight(weights["in_proj_qkv"], mesh_device, dtype=dtype, cache_file_name=cn("w_qkv"))
            self.w_z = as_weight(weights["in_proj_z"], mesh_device, dtype=dtype, cache_file_name=cn("w_z"))
            # in_proj_b and in_proj_a both map hidden -> V; fuse into one [hidden, 2V] matmul (one launch
            # instead of two), split the output b|a in forward. Both weights are [V, hidden] (nn.Linear).
            self.w_ba = as_weight(
                lambda: torch.cat([W("in_proj_b"), W("in_proj_a")], dim=0),
                mesh_device,
                dtype=dtype,
                cache_file_name=cn("w_ba"),
            )
        self.w_out = as_weight(weights["out_proj"], mesh_device, dtype=dtype, cache_file_name=cn("w_out"))
        # Tuned wide 1D mcast config for the T==1 fused input projection. It is deliberately NOT
        # DRAM-sharded (measured only 1.14-1.17x for wide-N inputs, Lever 1), so it has been running on
        # the ttnn AUTO config -- at 74% of DRAM peak, the best-tuned of the untuned matmuls, hence the
        # smallest win here: 83.9 -> 78.1 us at (33 cores, in0_block_w=1). Program-config only.
        self._pc_in_proj = None
        if _FUSE_IN and wide1d_enabled("w_in_proj"):
            self._pc_in_proj = wide_1d_decode_pc(
                32, self.w_in_proj.shape[-2], self.w_in_proj.shape[-1], 33, in0_block_w=1
            )
        self.norm = TtRMSNormGated(mesh_device, W("norm"), self.eps)

        cw = W("conv1d").reshape(self.conv_dim, self.conv_k)
        self.conv_taps = [
            to_tt(cw[:, j].reshape(1, self.conv_dim), mesh_device, dtype=ttnn.bfloat16) for j in range(self.conv_k)
        ]
        # taps stacked [conv_k, conv_dim] for the fused T==1 (decode) conv: out = sum_j xpad[j]*tap[j]
        # is then one multiply + one row-sum instead of K slices + K multiplies + (K-1) adds.
        self.conv_taps_stacked = to_tt(cw.transpose(0, 1).contiguous(), mesh_device, dtype=ttnn.bfloat16)

        # A_log, dt_bias as [1, V] device tensors for g = -exp(A_log)*softplus(a+dt_bias)
        self.neg_expA = to_tt(
            -torch.exp(W("A_log").float()).reshape(1, self.num_v_heads), mesh_device, dtype=ttnn.bfloat16
        )
        self.dt_bias = to_tt(W("dt_bias").float().reshape(1, self.num_v_heads), mesh_device, dtype=ttnn.bfloat16)

        # Fused-decode-kernel constants (built once). Per-head scalars (-exp(A_log), dt_bias) are
        # carried as uniform [V*TILE, TILE] tiles (head h's whole tile = scalar_h); the gated-norm
        # weight as [TILE, Dv] (replicated rows; same for all heads). Scratch (out/Snew) is lazy.
        if _GDN_FUSED or _GDN_BATCH_FUSED:
            V, Dv = self.num_v_heads, self.head_v_dim

            def _uniform(vals):  # torch [V] -> device [V*TILE, TILE] uniform per head
                t = torch.zeros(V * TILE, TILE)
                for h in range(V):
                    t[h * TILE : (h + 1) * TILE, :] = float(vals[h])
                return to_tt(t, mesh_device, dtype=_STATE_DT)

            self._fused_negA = _uniform(-torch.exp(W("A_log").float()).reshape(V))
            self._fused_dtb = _uniform(W("dt_bias").float().reshape(V))
            nw = W("norm").float().reshape(Dv)
            self._fused_nweight = to_tt(nw.reshape(1, Dv).expand(TILE, Dv).contiguous(), mesh_device, dtype=_STATE_DT)
            self._fused_scratch = None  # (out_buf, Snew_buf), allocated on first decode
            # batched-fused: B-keyed caches for the B-replicated per-head consts and the out/Snew scratch
            self._fused_batch_const = {}  # B -> (negA_b [B*V*TILE,TILE], dtb_b [B*V*TILE,TILE])
            self._fused_batch_scratch = {}  # B -> (out_buf [B*TILE,value_dim], snew_buf [B*V*Dk,Dv])

        # DRAM-sharded decode output projection (built from the already-loaded interleaved w_out, so no
        # extra HF read). K=value_dim (in), N=hidden (out). Prefill still uses self.w_out (interleaved).
        self._wout_dram = None
        if _DRAM_SHARD:
            K, N = self.w_out.shape[-2], self.w_out.shape[-1]
            self._wout_dram, self._wout_amc, self._wout_omc, self._wout_pc = build_dram_shard(
                self.w_out, K, N, _DRAM_BANKS
            )

    # Roll the conv window forward on commit by row SELECTION instead of a concat plus two row slices
    # (see _commit_conv_const). Two variants, because only one of them is safe:
    #
    #   _COMMIT_COPY (default ON): the FULL-ACCEPT commit (j == K == conv_k-1) needs no arithmetic at
    #     all — every new conv_state row is a conv_in row, in order — so it collapses to one copy. No
    #     matmul, no transient, and it is the commit greedy decode hits most often.
    #
    #   _COMMIT_SELECT (default OFF): the general j case as two selection matmuls plus an add. It is
    #     bit-identical (0/1 selection) and it WORKS — measured commit trace 6.18 -> 1.08 ms, round
    #     54.7 -> 49.4, greedy 1.19x -> 1.31x — but it silently destroys SAMPLED acceptance at 40
    #     layers: a1 0.727 -> 0.211, tokens/round 2.18 -> 1.00, reproducible, while
    #       * conv_state and recurrent_state are BIT-IDENTICAL to the old path at 4 layers over 8
    #         rounds, eager AND traced, with identical emitted tokens;
    #       * `test_gated_delta_verify` reports conv_state PCC 1.0 at every accept count;
    #       * 40-layer GREEDY decode is unaffected (2.267 tok/round, eager-vs-traced parity 136/136).
    #     Correct-at-4-layers / broken-at-40 in one path only is the signature of the allocation-
    #     pressure failures this file already records (the multi-bucket prefill wedge, and the
    #     trace-capture corruption in setup_mtp_decode). Removing a transient by writing the sum
    #     straight into conv_state with `add(output_tensor=)` did NOT fix it, so it is not simply
    #     transient count. Left in, default OFF, with the 5.1 ms it would buy documented rather than
    #     quietly shipped.
    _COMMIT_COPY = os.environ.get("QWEN36_GDN_COMMIT_COPY", "1") != "0"
    _COMMIT_SELECT = os.environ.get("QWEN36_GDN_COMMIT_SELECT", "0") != "0"

    # Fold the speculative-verify conv into 4 dense ops instead of 11 dispatch-bound ones. Only for
    # SMALL T (verify): the stacked form has conv_k*T rows, which is free while that stays inside one
    # 32-row tile and quadratic-ish beyond it, so prefill's T=512+ must keep the loop.
    _CONV_STACK = os.environ.get("QWEN36_GDN_VERIFY_CONV_STACK", "1") != "0"
    _CONV_STACK_MAX_T = 8

    def _conv_stack_const(self, T):
        """Constants for the stacked verify conv, cached per T.

        `out[t] = sum_j tap[j] * xpad[t+j]` is a contraction over conv_k shifts. Stack every shift into
        one tensor of conv_k*T rows and it becomes:

            Z   = S @ xpad     [conv_k*T, D]   ONE matmul instead of conv_k row slices
            Y   = Z * TAPS     [conv_k*T, D]   ONE multiply instead of conv_k multiplies
            out = A @ Y        [T, D]          ONE matmul instead of conv_k-1 adds

        with S[j*T+t, t+j] = 1 and A[t, j*T+t] = 1 (pure 0/1 selection/summation matrices). The reason
        this is a win rather than a wash: in TILE layout a conv_k*T-row tensor occupies the SAME single
        32-row tile as the T-row tensors it replaces, so widening costs nothing while the op count
        drops. MEASURED at K=3, conv_dim=8192, in isolation: 1146 us -> 383 us (3.0x). A row slice
        there is 108 us against 44 us for the equivalent selection matmul."""
        if not hasattr(self, "_conv_stack_cache"):
            self._conv_stack_cache = {}
        c = self._conv_stack_cache.get(T)
        if c is None:
            CK, D, R = self.conv_k, self.conv_dim, T + self.conv_k - 1
            Sm = torch.zeros(CK * T, R)
            Am = torch.zeros(T, CK * T)
            for j in range(CK):
                for t in range(T):
                    Sm[j * T + t, t + j] = 1.0
                    Am[t, j * T + t] = 1.0
            # TAPS[j*T+t] = tap[j], built on device from the per-tap rows so it works on both the
            # fresh-weight and weight-cache paths. Built once per T, so its cost is irrelevant.
            taps = ttnn.concat([ttnn.repeat(self.conv_taps[j], ttnn.Shape([T, 1])) for j in range(CK)], dim=0)
            # S split by SOURCE so the [conv_state | mixed] concat can be skipped entirely: the concat
            # measured 0.1242 ms/layer, 53% of the whole stacked conv, while the two half-width
            # selection matmuls plus one add come to ~0.059. (An isolated microbench had called this a
            # wash at 171 vs 168 us; in-model the selection matmuls cost 17 us, not 44, so the balance
            # flips — measure the replacement where it will run.)
            c = (
                to_tt(Sm, self.mesh_device, dtype=ttnn.bfloat16),
                to_tt(Am, self.mesh_device, dtype=ttnn.bfloat16),
                taps,
                to_tt(Sm[:, : self.conv_k - 1].contiguous(), self.mesh_device, dtype=ttnn.bfloat16),
                to_tt(Sm[:, self.conv_k - 1 :].contiguous(), self.mesh_device, dtype=ttnn.bfloat16),
            )
            self._conv_stack_cache[T] = c
        return c

    def _conv_silu(self, mixed, conv_state=None, need_state=True, state_at=None, snap_at=None, snap_dst=None):
        """mixed: [T, conv_dim]. Causal depthwise conv (kernel K) + silu. Returns
        (out[T,conv_dim], new_conv_state[K-1,conv_dim]); new_conv_state is None when need_state=False.

        `state_at` (ragged mask path): take the carried state from the rows ending at this position
        instead of at the end of the block. A masked ragged block is `valid_len` real rows followed by
        right-padding, and the conv window the NEXT block continues from is the last K-1 REAL rows --
        the padding must not enter it. The slice is a fixed [K-1, conv_dim] shape at a varying offset,
        so it costs no new op shapes.

        `need_state=False` is for the speculative-verify path, which DISCARDS it — `commit_verify`
        recomputes the conv tail from the accepted prefix instead. Skipping it is free money: that one
        slice measured **0.0868 ms/layer** (a [conv_k-1, conv_dim] sub-tile row slice, and conv_dim=8192
        means it gathers 3 of 32 rows across all 256 tile-columns)."""
        T = mixed.shape[0]
        if conv_state is None:
            pad = ttnn.zeros(
                [self.conv_k - 1, self.conv_dim], dtype=mixed.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
            )
        else:
            pad = conv_state
        stacked = self._CONV_STACK and 1 < T <= self._CONV_STACK_MAX_T
        # The stacked verify path reads the two sources directly, so it only needs `xpad` when the
        # caller also wants the rolled-forward state (verify does not — commit_verify recomputes it).
        if stacked and not need_state:
            xpad = None
        else:
            # L1 only for the tiny T==1 decode concat; prefill (large T) stays in DRAM (an L1 [T+K-1,
            # conv_dim] concat is ~33 MB at T=2048 and overflows L1 — an L1-residency decode-only win).
            xpad = ttnn.concat([pad, mixed], dim=0, memory_config=(_MC if T == 1 else None))
        if T == 1:
            # decode: xpad is exactly [K, conv_dim] and out[0] = sum_j xpad[j]*tap[j]. One elementwise
            # multiply against the stacked taps + one row-sum, vs K slices + K multiplies + (K-1) adds
            # (the 4-tap conv was the largest gated-delta decode cost — pure dispatch overhead).
            acc = ttnn.sum(
                ttnn.multiply(xpad, self.conv_taps_stacked, memory_config=_MC), dim=0, keepdim=True, memory_config=_MC
            )  # [1, conv_dim]
        elif stacked:
            # verify: one stacked contraction (see _conv_stack_const). HiFi4 so the conv_k-term tap sum
            # accumulates in fp32 — strictly better than the bf16 add chain it replaces, though it is a
            # different rounding ORDER, which the recurrence is sensitive to (hence the PCC gate in
            # tests/test_gated_delta_verify.py and the env kill-switch).
            S, A, TAPS, S_pad, S_new = self._conv_stack_const(T)
            if xpad is None:  # skip the concat: select from each source and add (see _conv_stack_const)
                Z = ttnn.add(
                    ttnn.matmul(S_pad, pad, compute_kernel_config=_HIFI4),
                    ttnn.matmul(S_new, mixed, compute_kernel_config=_HIFI4),
                )
            else:
                Z = ttnn.matmul(S, xpad, compute_kernel_config=_HIFI4)
            acc = ttnn.matmul(A, ttnn.multiply(Z, TAPS), compute_kernel_config=_HIFI4)
        else:
            acc = None
            for j in range(self.conv_k):
                xj = ttnn.slice(xpad, [j, 0], [j + T, self.conv_dim])
                term = ttnn.multiply(xj, self.conv_taps[j])
                acc = term if acc is None else ttnn.add(acc, term)
        # xpad is [conv_state (K-1) ; mixed (T)], so mixed row i is xpad row i + K - 1 and the K-1
        # rows ending at real position `end` are xpad[end : end + K - 1]. end == T is the whole block.
        end = T if state_at is None else int(state_at)
        new_state = ttnn.slice(xpad, [end, 0], [end + self.conv_k - 1, self.conv_dim]) if need_state else None
        assert not (need_state and xpad is None)
        if snap_at is not None and snap_dst is not None:
            # In-block checkpoint: the same fixed [K-1, conv_dim] slice, taken at an INTERIOR row.
            # Costs one extra sub-tile row slice per layer and introduces no new op shape.
            ttnn.copy(
                ttnn.slice(xpad, [int(snap_at), 0], [int(snap_at) + self.conv_k - 1, self.conv_dim]),
                snap_dst,
            )
        # Same T==1 gate as the concat above: at prefill T this is [T, conv_dim] (28 MB at T=1711),
        # and because ttnn defaults memory_config to the INPUT's config, an L1 result here silently
        # propagates to the q/k/v slices and everything downstream of them — pinning ~700 KB/core and
        # clashing with the fused-recurrence CBs (which occupy 1.07 MB of the 1.5 MB L1 on their cores).
        return ttnn.silu(acc, memory_config=(_MC if T == 1 else None)), new_state

    def _valid_col(self, T: int, valid_len: int, dtype):
        """[T, 1] column: 1 for the first ``valid_len`` rows, 0 for the right-padding.

        Built per call. Its SHAPE depends only on T (one of a handful of padded widths), never on
        ``valid_len``, so the varying value cannot introduce new op shapes -- which is the whole point
        of the mask path."""
        m = torch.zeros(T, 1)
        m[:valid_len] = 1.0
        return to_tt(m, self.mesh_device, dtype=dtype)

    def sync_conv_rows(self, cache):
        """Populate the decode add-chain's K-1 separate history-row buffers from the [K-1, conv_dim]
        conv_state (which prefill fills). Called once when entering decode (model.start_decode). rows[i]
        = conv_state row i = x_{t-(K-1)+i} (oldest..newest). Eager, one-time (K-1 dim-0 slices)."""
        if not (isinstance(cache, dict) and "conv_rows" in cache and "conv_state" in cache):
            return
        cs, rows = cache["conv_state"], cache["conv_rows"]
        for i in range(self.conv_k - 1):
            ttnn.copy(ttnn.slice(cs, [i, 0], [i + 1, self.conv_dim]), rows[i])

    def _conv_addchain_decode(self, mixed, rows):
        """T==1 decode conv as a bf16 add-chain over separate [1, conv_dim] history rows, then shift the
        window in place. out = sum_{i<K-1} rows[i]*tap[i] + mixed*tap[K-1], matching the stacked path's
        terms (rows[i] == conv_state row i == xpad[i]; tap[i] == conv_taps_stacked[i]). Removes the two
        dispatch-bound dim-0 ops (concat + xpad slice). rows are persistent cache buffers -> the in-place
        shift is trace-safe. Returns out [1, conv_dim]; state lives in `rows` (no copy@571)."""
        K = self.conv_k
        acc = ttnn.multiply(mixed, self.conv_taps[K - 1], memory_config=_MC)
        for i in range(K - 1):
            acc = ttnn.add(acc, ttnn.multiply(rows[i], self.conv_taps[i], memory_config=_MC), memory_config=_MC)
        out = ttnn.silu(acc, memory_config=_MC)
        for i in range(K - 2):  # slide the window: drop oldest, shift down
            ttnn.copy(rows[i + 1], rows[i])
        ttnn.copy(mixed, rows[K - 2])  # newest history row = the current token
        return out

    def _ensure_chunk_masks(self):
        """Constant [1,1,C,C] masks for the chunked prep — built once (chunk_size is fixed)."""
        if getattr(self, "_chunk_masks", None) is not None:
            return self._chunk_masks
        C = _CHUNK

        def up(t):
            return ttnn.from_torch(
                t.reshape(1, 1, C, C),
                dtype=_PREP_DT,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Per-level masks for the numerically-stable recursive block inversion of (I-L) (see _chunk_prep).
        # Level k (block size s=2^k, half h): BL selects the lower-left h-block of each s-block (the new
        # off-diagonal corner), A/D the upper-left / lower-right h-diagonal-blocks. log2(C) levels.
        idx = torch.arange(C)
        inv_levels = []
        k = 1
        while (1 << k) <= C:
            s = 1 << k
            h = s >> 1
            blk = idx // s
            off = idx % s
            lower = off >= h
            same = blk[:, None] == blk[None, :]
            BL = (lower[:, None] & (~lower)[None, :]) & same
            A = ((~lower)[:, None] & (~lower)[None, :]) & same
            D = (lower[:, None] & lower[None, :]) & same
            inv_levels.append(dict(BL=up(BL.float()), A=up(A.float()), D=up(D.float())))
            k += 1

        self._chunk_masks = dict(
            tril_incl=up(torch.tril(torch.ones(C, C))),  # cumsum/decay causal incl-diag
            strict_lower=up(torch.tril(torch.ones(C, C), diagonal=-1)),  # L strictly-lower mask
            eye=up(torch.eye(C)),  # I for (I-L)^-1
            inv_levels=inv_levels,
        )
        return self._chunk_masks

    def _chunk_prep(self, q, k, v, beta, g, masks):
        """Per-chunk delta-rule prep, batched over heads. q,k,v:[1,Vh,C,D]; beta,g:[1,Vh,C] (g=log
        decay); all fp32. Returns the terms the ttl _chunk_state kernel consumes (all [1,Vh,*,*])."""
        C = _CHUNK
        Vh, D = q.shape[1], q.shape[3]
        tril_incl, strict_lower, eye = masks["tril_incl"], masks["strict_lower"], masks["eye"]
        g_cum = ttnn.cumsum(g, dim=-1)  # [1,Vh,C]
        gc_row = ttnn.reshape(g_cum, [1, Vh, C, 1])
        gc_col = ttnn.reshape(g_cum, [1, Vh, 1, C])
        # zero the strict-upper BEFORE exp (avoid exp(large)=inf) by multiplying the pre-built tril
        # mask, NOT ttnn.tril (which builds a mask via a host write -> forbidden in trace capture).
        # exp(0)=1 in the upper is harmless: `decay` re-masks it to 0 below. PCC-identical.
        diff = ttnn.multiply(ttnn.subtract(gc_row, gc_col), tril_incl)
        decay = ttnn.multiply(ttnn.exp(diff), tril_incl)  # [1,Vh,C,C]
        egc_col = ttnn.reshape(ttnn.exp(g_cum), [1, Vh, C, 1])
        beta_col = ttnn.reshape(beta, [1, Vh, C, 1])
        vbeta = ttnn.multiply(v, beta_col)
        kbeta = ttnn.multiply(k, beta_col)
        kT = ttnn.transpose(k, -2, -1)  # [1,Vh,D,C]
        # prep matmuls (esp. the inverse) always at HiFi4 + fp32 accum: Tensix matmul inputs are bf16,
        # so default (low) fidelity caps accuracy at ~0.999 vs fp32 — compounding to gibberish over 30
        # layers. HiFi4 (multi-pass ~fp32) closes that gap. The recursive inverse below is bounded, but
        # its bf16-input matmuls still need fp32 accumulation to stay accurate.
        ck = _HIFI4
        L = ttnn.multiply(
            ttnn.multiply(ttnn.multiply(ttnn.matmul(kbeta, kT, compute_kernel_config=ck), decay), strict_lower), -1.0
        )
        # (I-L)^-1 via numerically-stable recursive block inversion (block Gaussian elimination).
        # The doubling product (I+L)(I+L^2)...(I+L^32) is mathematically exact for nilpotent L but
        # numerically EXPLODES on real data: the intermediate L^k have huge entries (large singular
        # values, despite eigenvalues=0) that must telescope but don't in finite precision -> 1e6-1e11
        # blow-up while the true inverse is ~1.0. The recursive block form never forms L^k: it merges
        # h-block inverses pairwise via corner = -D^-1 (M_B) A^-1, staying bounded (max|T|~1.0). All
        # ops are full-CxC (tile-aligned) batched matmuls + fixed mask multiplies; log2(C) levels.
        # (The old T≈I+L fast approximation was removed — it is wrong for this model's O(1) L -> gibberish.)
        M = ttnn.subtract(eye, L)  # I - L (unit lower-tri)
        T = ttnn.repeat(
            eye, ttnn.Shape([1, M.shape[1], 1, 1])
        )  # identity, batched over heads (matmul needs matching batch)
        for lv in masks["inv_levels"]:
            Dp = ttnn.multiply(T, lv["D"])  # lower-right h-block inverses
            Ap = ttnn.multiply(T, lv["A"])  # upper-left h-block inverses
            Bp = ttnn.multiply(M, lv["BL"])  # original lower-left h-blocks of M
            corner = ttnn.multiply(
                ttnn.matmul(ttnn.matmul(Dp, Bp, compute_kernel_config=ck), Ap, compute_kernel_config=ck),
                lv["BL"],
            )
            T = ttnn.add(T, ttnn.multiply(corner, -1.0))
        qg = ttnn.multiply(q, egc_col)
        w = ttnn.matmul(T, vbeta, compute_kernel_config=ck)
        kcd = ttnn.matmul(T, ttnn.multiply(kbeta, egc_col), compute_kernel_config=ck)
        glast_col = ttnn.reshape(ttnn.slice(g_cum, [0, 0, C - 1], [1, Vh, C]), [1, Vh, 1, 1])
        kg = ttnn.multiply(k, ttnn.exp(ttnn.subtract(glast_col, gc_row)))  # per-row decay
        kgt = ttnn.transpose(kg, -2, -1)  # [1,Vh,D,C]
        glast = ttnn.repeat(ttnn.exp(glast_col), ttnn.Shape([1, 1, D, D]))  # [1,Vh,D,D] (kernel: S*glast)
        return dict(q=q, kt=kT, w=w, kcd=kcd, decay=decay, qg=qg, kgt=kgt, glast=glast)

    def _chunk_state_ttnn(self, tc, S, bufs=None):
        """Numerically-stable (HiFi4 / fp32) ttnn implementation of one chunk's delta-rule recurrence,
        replacing the bf16 ttl _chunk_state kernel for prefill. Mirrors the kernel math exactly:
          v_new = w − kcd·S ;  out = qg·S + ((q·kᵀ)⊙decay)·v_new ;  Snew = S·glast + kgtᵀ·v_new.
        tc: per-chunk prep terms [1,Vh,*,*] (fp32). S: [1,Vh,Dk,Dv] (fp32). Returns (out [1,Vh,C,Dv],
        Snew [1,Vh,Dk,Dv]), fp32. All matmuls are batched over the head axis at HiFi4 + fp32 accum, so
        the chunked path matches the fp32 reference (the bf16 kernel only reached ~0.996/layer).

        bufs (traced prefill): pre-allocated output buffers {kcdS,v_new,qk,aintra,qgS,av,sglast,kgtv,
        out,Snew} written via output_tensor= instead of allocating in-graph — keeps the trace's pinned
        footprint O(1) so multi-bucket capture fits DRAM (see build_trace_pool). Every op's out buffer
        is distinct from its inputs (no aliasing)."""
        ck = _HIFI4
        if bufs is None:
            v_new = ttnn.subtract(tc["w"], ttnn.matmul(tc["kcd"], S, compute_kernel_config=ck))  # [1,Vh,C,Dv]
            aintra = ttnn.multiply(
                ttnn.matmul(tc["q"], tc["kt"], compute_kernel_config=ck), tc["decay"]
            )  # [1,Vh,C,C], causal via decay
            out = ttnn.add(
                ttnn.matmul(tc["qg"], S, compute_kernel_config=ck),  # cross-chunk: qg·S
                ttnn.matmul(aintra, v_new, compute_kernel_config=ck),  # intra-chunk
            )  # [1,Vh,C,Dv]
            Snew = ttnn.add(
                ttnn.multiply(S, tc["glast"]),  # decayed incoming state
                ttnn.matmul(tc["kgt"], v_new, compute_kernel_config=ck),  # this chunk's update
            )  # [1,Vh,Dk,Dv]
            return out, Snew
        b = bufs  # trace-safe: write every result into a pre-allocated (pinned, reused) buffer
        ttnn.matmul(tc["kcd"], S, compute_kernel_config=ck, optional_output_tensor=b["kcdS"])
        ttnn.subtract(tc["w"], b["kcdS"], output_tensor=b["v_new"])  # v_new = w − kcd·S
        ttnn.matmul(tc["q"], tc["kt"], compute_kernel_config=ck, optional_output_tensor=b["qk"])
        ttnn.multiply(b["qk"], tc["decay"], output_tensor=b["aintra"])  # aintra = (q·kᵀ)⊙decay
        ttnn.matmul(tc["qg"], S, compute_kernel_config=ck, optional_output_tensor=b["qgS"])
        ttnn.matmul(b["aintra"], b["v_new"], compute_kernel_config=ck, optional_output_tensor=b["av"])
        ttnn.add(b["qgS"], b["av"], output_tensor=b["out"])  # out = qg·S + aintra·v_new
        ttnn.multiply(S, tc["glast"], output_tensor=b["sglast"])
        ttnn.matmul(tc["kgt"], b["v_new"], compute_kernel_config=ck, optional_output_tensor=b["kgtv"])
        ttnn.add(b["sglast"], b["kgtv"], output_tensor=b["Snew"])  # Snew = S·glast + kgtᵀ·v_new
        return b["out"], b["Snew"]

    def _ensure_fused_op_consts(self):
        """Constant fp32 tiles the fused op needs (identity / lower-triangular ones / all-ones at the
        op's chunk size, plus the packed 32x32 quadrant masks). Built once, device-resident: the op's
        own internal build does a HOST UPLOAD, which is illegal inside trace capture, so the caller
        must own them."""
        if getattr(self, "_fused_op_consts", None) is not None:
            return self._fused_op_consts
        C = _FUSED_OP_CHUNK
        ii, jj = torch.arange(32).unsqueeze(1), torch.arange(32).unsqueeze(0)
        lo_i, lo_j = ii < 16, jj < 16
        # Three 32x32 quadrants packed into [32,96]: top-left, bottom-right, bottom-left.
        masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)

        def up(t):
            return ttnn.from_torch(
                t.reshape(1, 1, *t.shape),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self._fused_op_consts = (
            up(torch.eye(C)),
            up(torch.tril(torch.ones(C, C))),
            up(torch.ones(C, C)),
            up(masks),
        )
        return self._fused_op_consts

    def _forward_prefill_fused_op(self, q, k, v, g, beta, init_state=None, snap_at=None, snap_dst=None):
        """Chunked delta-rule prefill via ttnn.transformer.chunk_gated_delta_rule. Same signature and
        return contract as _forward_prefill_chunked: q,k,v [T,Vh,D]; g,beta [T,Vh] (g = log decay) ->
        (core [1,Vh,T,Dv] bf16, S_final [Vh*Dk,Dv] bf16).

        Contract differences the op imposes, all handled here:
        * it does NOT l2-normalize q/k — ours are already normalized upstream, so use_qk_l2norm stays
          False (its default);
        * `scale` DEFAULTS to Dk**-0.5, but our q already carries qk_scale, so we pass scale=1.0 or
          the scaling lands twice;
        * g/beta must be fp32;
        * T must be a multiple of the op's chunk size — pad with zeros, which is inert for the same
          reason the existing path's padding is (beta=0 -> no state update, g=0 -> no decay)."""
        Vh, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
        T = q.shape[0]
        if snap_at is not None and snap_dst is not None and 0 < int(snap_at) < T:
            # IN-BLOCK CHECKPOINT. The recurrence is exactly splittable: this op takes an
            # initial_state and returns the final one, so running [0,snap_at) then [snap_at,T) seeded
            # with its result is identical to one pass -- MEASURED core PCC 1.000000 and final-state
            # PCC 1.000000 (hangdbg/splitprobe.py). The point is that only the RECURRENCE splits:
            # attention, MoE and the norms still run once over the whole block, so the caller gets a
            # resumable checkpoint at an interior 128 boundary without paying a second block's MoE
            # sweep (127 ms, see PREFILL.md 0.9).
            j = int(snap_at)
            sl = lambda t, a, b: ttnn.slice(t, [a, 0, 0], [b, t.shape[1], t.shape[2]])
            sl2 = lambda t, a, b: ttnn.slice(t, [a, 0], [b, t.shape[1]])
            o1, s_mid = self._forward_prefill_fused_op(
                sl(q, 0, j), sl(k, 0, j), sl(v, 0, j), sl2(g, 0, j), sl2(beta, 0, j), init_state=init_state
            )
            ttnn.copy(ttnn.reshape(s_mid, [1, Vh, Dk, Dv]), snap_dst)
            o2, s_end = self._forward_prefill_fused_op(
                sl(q, j, T), sl(k, j, T), sl(v, j, T), sl2(g, j, T), sl2(beta, j, T), init_state=s_mid
            )
            return ttnn.concat([o1, o2], dim=2), s_end
        C = _FUSED_OP_CHUNK
        pad = (C - T % C) % C
        Tp = T + pad
        eye, tril, ones, masks = self._ensure_fused_op_consts()

        def bt(x, D, dt):  # [T,Vh,D] -> [1,Tp,Vh,D]
            y = ttnn.reshape(x if x.dtype == dt else ttnn.typecast(x, dt), [1, T, Vh, D])
            if pad:
                z = ttnn.zeros([1, pad, Vh, D], dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
                y = ttnn.concat([y, z], dim=1)
            return y

        def bt2(x):  # [T,Vh] -> [1,Tp,Vh] fp32
            y = ttnn.reshape(ttnn.typecast(x, ttnn.float32), [1, T, Vh])
            if pad:
                z = ttnn.zeros([1, pad, Vh], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
                y = ttnn.concat([y, z], dim=1)
            return y

        gb, bb = bt2(g), bt2(beta)
        s0 = None
        if init_state is not None:
            s0 = ttnn.reshape(init_state, [1, Vh, Dk, Dv])
            if s0.dtype != ttnn.float32:
                s0 = ttnn.typecast(s0, ttnn.float32)

        o, S = ttnn.transformer.chunk_gated_delta_rule(
            bt(q, Dk, ttnn.bfloat16),
            bt(k, Dk, ttnn.bfloat16),
            bt(v, Dv, ttnn.bfloat16),
            gb,
            bb,
            scale=1.0,
            initial_state=s0,
            output_final_state=True,
            chunk_size=C,
            output_head_major=True,  # [Vh, Tp, Dv] -> our [1,Vh,T,Dv] with no token<->head permute
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )
        core = ttnn.reshape(o, [1, Vh, Tp, Dv])
        if pad:
            core = ttnn.slice(core, [0, 0, 0, 0], [1, Vh, T, Dv])
        if core.dtype != ttnn.bfloat16:
            core = ttnn.typecast(core, ttnn.bfloat16)
        S = ttnn.reshape(S, [Vh * Dk, Dv])
        if S.dtype != _STATE_DT:  # the native op returns an fp32 final_state; keep it if the cache is fp32
            S = ttnn.typecast(S, _STATE_DT)
        return core, S

    def _forward_prefill_chunked(self, q, k, v, g, beta, pool=None, init_state=None, snap_at=None, snap_dst=None):
        """Chunked delta-rule prefill. q,k,v:[T,Vh,D]; g,beta:[T,Vh] (g=log decay). Returns
        (core [1,Vh,T,Dv], S_final [Vh*Dk,Dv]). Pads T to a multiple of chunk_size and carries the
        per-head recurrent state [Vh*Dk,Dv] (head-major) across chunks.

        init_state (incremental prefill): head-major [Vh*Dk,Dv] starting state from the cache (continue
        a prior context) instead of zeros — the natural cross-chunk carry, just seeded from the cache.
        pool (traced path only): a dict of PRE-ALLOCATED buffers {S0, out[Nc], Snew[Nc]} used instead
        of in-graph ttnn.zeros (which are forbidden during trace capture). Requires T a multiple of
        chunk_size (no padding); build it with build_trace_pool(T)."""
        # Native fused op (v0.77+). Eager only: the traced path relies on the pre-allocated buffer
        # pool, which the op does not use (it allocates its own outputs).
        # Head-dim guard: the op validates only that key_dim/val_dim are multiples of 32, but it
        # SILENTLY HANGS (no TT_FATAL, device wedged, needs `tt-smi -r`) at small head dims -- measured
        # hanging at Dk=Dv=32, the toy geometry tests/test_gated_delta.py uses. Upstream ships and
        # exercises it only at head_dim=128 (models/demos/blackhole/qwen36 defaults to 128 and there is
        # no unit test for this op at any other size in-tree), and 128 is what this model uses and what
        # was verified here (core/state PCC ~0.999998 vs the in-tree path). Anything below the verified
        # envelope falls back rather than risking a wedge.
        # EAGER ONLY (pool is None). Routing the traced path (pool != None) through the op was tried
        # and HANGS during capture/replay -- and it is not worth chasing: measured on this model,
        # tracing the prefill is worth only 1.01-1.02x (40L, 650 vs 663 ms at T=256; 1243 vs 1259 ms
        # at T=512), whereas the fused op is worth ~1.6x on the eager path. Eager+fused therefore
        # BEATS traced-unfused outright, so the traced prefill path is redundant when this is on.
        if _GDN_FUSED_OP and pool is None and min(self.head_k_dim, self.head_v_dim) >= _FUSED_OP_MIN_HEAD_DIM:
            return self._forward_prefill_fused_op(
                q, k, v, g, beta, init_state=init_state, snap_at=snap_at, snap_dst=snap_dst
            )
        # In-block checkpoints are implemented on the fused path only; supports_inblock_snapshot says
        # so, and the planner falls back to a separate snapshot block when it is False.
        assert snap_at is None or snap_dst is None, "in-block snapshot requires the fused recurrence"
        C = _CHUNK
        Vh, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
        T = q.shape[0]
        pad = (C - T % C) % C
        Tp = T + pad
        assert pool is None or pad == 0, "traced chunked prefill requires T a multiple of chunk_size"
        masks = self._ensure_chunk_masks()

        def _cast(x):  # to _PREP_DT, skipping the op when x is already that dtype (the bf16 path)
            return x if x.dtype == _PREP_DT else ttnn.typecast(x, _PREP_DT)

        def to_bvtd(x):  # [T,Vh,Dx] -> [1,Vh,Tp,Dx] in _PREP_DT, zero-padded along time
            Dx = x.shape[2]
            y = ttnn.reshape(ttnn.permute(_cast(x), [1, 0, 2]), [1, Vh, T, Dx])
            if pad:
                z = ttnn.zeros([1, Vh, pad, Dx], dtype=_PREP_DT, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
                y = ttnn.concat([y, z], dim=2)
            return y

        def to_bvt(x):  # [T,Vh] -> [1,Vh,Tp] in _PREP_DT, zero-padded (padding => g=0 => state unaffected)
            y = ttnn.reshape(ttnn.permute(_cast(x), [1, 0]), [1, Vh, T])
            if pad:
                z = ttnn.zeros([1, Vh, pad], dtype=_PREP_DT, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
                y = ttnn.concat([y, z], dim=2)
            return y

        with prof.phase(self.mesh_device, "delta.setup"):
            qb, kb, vb = to_bvtd(q), to_bvtd(k), to_bvtd(v)
            gb, betab = to_bvt(g), to_bvt(beta)
            if pool is not None and "valid" in pool:
                # zero g/beta for right-padding (bucketed trace): pad rows then contribute nothing to
                # state (beta=0 -> kbeta=vbeta=0) and no decay (g=0 -> g_cum flat). Matches eager pad.
                gb = ttnn.multiply(gb, pool["valid"])
                betab = ttnn.multiply(betab, pool["valid"])

        def hm(t):  # [1,Vh,A,B] -> head-major [Vh*A,B] bf16 (kernel input layout); cast only if needed
            r = ttnn.reshape(t, [Vh * t.shape[2], t.shape[3]])
            return r if r.dtype == ttnn.bfloat16 else ttnn.typecast(r, ttnn.bfloat16)

        # initial state: cached state (incremental), pooled zeros (traced), or fresh zeros (eager)
        if init_state is not None:
            S = init_state
        elif pool is not None:
            S = pool["S0"]
        else:
            S = ttnn.zeros([Vh * Dk, Dv], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)

        # Numerically-stable ttnn-fp32 chunk-state (eager, and traced when _TRACED_FP32). State carried
        # as [1,Vh,Dk,Dv] in fp32 for accuracy (see _chunk_state_ttnn).
        use_stable = _STABLE_PREFILL and (pool is None or _TRACED_FP32)
        fp32_pooled = use_stable and pool is not None  # traced fp32 path writes into pooled buffers
        if use_stable and not fp32_pooled:  # eager: reshape+cast the fresh bf16 S to fp32
            S = ttnn.typecast(ttnn.reshape(S, [1, Vh, Dk, Dv]), _PREP_DT)
        # (fp32_pooled: pool["S0"] is already fp32 [1,Vh,Dk,Dv], read-only zeros — used directly)

        Nc = Tp // C
        # The per-chunk prep (cumsum/decay/β-scaling/inverse T/w/kcd) does NOT depend on the recurrent
        # state S (only the ttl _chunk_state kernel consumes S), so all Nc chunks' prep is independent
        # and runs in ONE batched set of ops over the stacked [1, Nc*Vh, C, *] axis instead of Nc
        # sequential rounds (cutting prep dispatches ~Nc× and folding the fp32 matmuls into one launch
        # over the larger batch). The kernel loop below still runs sequentially to carry S. The masks
        # are [1,1,C,C] and broadcast over Nc*Vh unchanged. QWEN36_DELTA_BATCH_PREP=0 reverts to the
        # per-chunk prep (kept as an A/B fallback). Nc==1 is the per-chunk path either way.
        if _BATCH_PREP and Nc > 1:

            def _chunks_on_b(x, D):  # [1,Vh,Tp,D] -> [1, Nc*Vh, C, D] chunk-major (concat of slices)
                cs = [ttnn.slice(x, [0, 0, ci * C, 0], [1, Vh, ci * C + C, D]) for ci in range(Nc)]
                return ttnn.concat(cs, dim=1)

            def _chunks_on_b3(x):  # [1,Vh,Tp] -> [1, Nc*Vh, C] chunk-major
                cs = [ttnn.slice(x, [0, 0, ci * C], [1, Vh, ci * C + C]) for ci in range(Nc)]
                return ttnn.concat(cs, dim=1)

            with prof.phase(self.mesh_device, "delta.prep"):
                terms = self._chunk_prep(
                    _chunks_on_b(qb, Dk),
                    _chunks_on_b(kb, Dk),
                    _chunks_on_b(vb, Dv),
                    _chunks_on_b3(betab),
                    _chunks_on_b3(gb),
                    masks,
                )

            def chunk_terms(ci):  # extract chunk ci's [1,Vh,*,*] views from the batched [1,Nc*Vh,*,*]
                return {
                    k: ttnn.slice(t, [0, ci * Vh, 0, 0], [1, ci * Vh + Vh, t.shape[2], t.shape[3]])
                    for k, t in terms.items()
                }

        else:
            chunk_terms = None  # per-chunk prep inside the loop

        outs = []
        for ci in range(Nc):
            if chunk_terms is not None:
                tc = chunk_terms(ci)
            else:
                s0 = ci * C
                qc = ttnn.slice(qb, [0, 0, s0, 0], [1, Vh, s0 + C, Dk])
                kc = ttnn.slice(kb, [0, 0, s0, 0], [1, Vh, s0 + C, Dk])
                vc = ttnn.slice(vb, [0, 0, s0, 0], [1, Vh, s0 + C, Dv])
                gc = ttnn.slice(gb, [0, 0, s0], [1, Vh, s0 + C])
                bc = ttnn.slice(betab, [0, 0, s0], [1, Vh, s0 + C])
                with prof.phase(self.mesh_device, "delta.prep"):
                    tc = self._chunk_prep(qc, kc, vc, bc, gc, masks)
            if use_stable:  # ttnn fp32 chunk-state (numerically correct); S stays [1,Vh,Dk,Dv]
                with prof.phase(self.mesh_device, "delta.kernel"):
                    if fp32_pooled:  # traced: write into pre-allocated pooled buffers (multi-bucket-safe)
                        # Snew alternates (build_trace_pool holds 2): this chunk's carry-out must not be
                        # the buffer S currently points at, which is the previous chunk's.
                        bufs = {
                            **pool["recur"],
                            "out": pool["out"][ci],
                            "Snew": pool["Snew"][ci % len(pool["Snew"])],
                        }
                        out4, S = self._chunk_state_ttnn(tc, S, bufs=bufs)
                    else:
                        out4, S = self._chunk_state_ttnn(tc, S)
                outs.append(out4)
                continue
            if pool is not None:  # kernel-written output buffers: pre-allocated (no in-graph zeros)
                out, Snew = pool["out"][ci], pool["Snew"][ci % len(pool["Snew"])]  # carry alternates
            else:
                out = ttnn.zeros([Vh * C, Dv], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
                Snew = ttnn.zeros([Vh * Dk, Dv], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
            with prof.phase(self.mesh_device, "delta.kernel"):
                chunk_state_tt(
                    hm(tc["q"]),
                    hm(tc["kt"]),
                    hm(tc["w"]),
                    hm(tc["kcd"]),
                    hm(tc["decay"]),
                    hm(tc["qg"]),
                    hm(tc["kgt"]),
                    hm(tc["glast"]),
                    S,
                    out,
                    Snew,
                    n_heads=Vh,
                )
            S = Snew
            outs.append(ttnn.reshape(out, [1, Vh, C, Dv]))
        core = ttnn.concat(outs, dim=2)  # [1,Vh,Tp,Dv]
        if pad:
            core = ttnn.slice(core, [0, 0, 0, 0], [1, Vh, T, Dv])
        if use_stable:  # back to head-major state + bf16 core (the path's interface)
            # The chunked prefill already computes S in fp32; with _FP32_STATE the decode cache is fp32
            # too, so this demotion is pure loss and the typecast disappears (one fewer op per layer).
            S = ttnn.reshape(S, [Vh * Dk, Dv])
            if S.dtype != _STATE_DT:
                S = ttnn.typecast(S, _STATE_DT)
            if core.dtype != ttnn.bfloat16:
                core = ttnn.typecast(core, ttnn.bfloat16)
        return core, S

    def build_trace_pool(self, T):
        """Pre-allocate the buffers _forward_prefill_chunked would otherwise create with ttnn.zeros
        (forbidden inside trace capture). Shared across all gated-delta layers (sequential -> reusable);
        S0 stays zeros (read-only initial state); out/Snew are written per chunk. T must be a chunk mult.

        With _TRACED_FP32 (default) the traced recurrence is the fp32/HiFi4 _chunk_state_ttnn, whose
        per-chunk intermediates would otherwise be allocated IN-GRAPH — ~24 MB/chunk x Nc x 30 layers,
        all PINNED for the trace lifetime (~5.8 GB at bucket 512), which overflows DRAM once a second
        bucket is captured (the real 'bucket-256 wedge'). So we pre-allocate the recurrence intermediates
        as a small SHARED set (reused across every chunk and layer -> ~20 MB total) plus per-chunk out,
        all fp32, and _chunk_state_ttnn writes into them via output_tensor= (see `recur`).

        `Snew` is only TWO buffers, alternating, not Nc. It is purely the state carry: chunk ci writes
        Snew[ci%2] and chunk ci+1 reads it as S while writing Snew[(ci+1)%2], so a buffer is dead as
        soon as the next chunk has consumed it and input/output never collide. Only `out` needs one
        buffer per chunk, because the final `concat` reads all Nc at once. Nc buffers of [1,Vh,Dk,Dv]
        fp32 is 2 MB per chunk: MEASURED 16 MB at bucket 512 and 1.02 GB at bucket 32768, against 4 MB
        either way here."""
        C = _CHUNK
        Vh, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
        Nc = T // C
        z = lambda shp, dt=ttnn.bfloat16: ttnn.zeros(shp, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        # "valid" is a per-request [1,1,T] mask (1 for real tokens, 0 for right-padding) that the bucketed
        # traced path multiplies into g/beta so pad positions contribute nothing to the recurrent state —
        # reproducing the eager zero-padding for a fixed-shape (bucket-padded) input. Written per request.
        valid = ttnn.zeros([1, 1, T], dtype=_PREP_DT, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        if _TRACED_FP32:
            f = ttnn.float32
            return {
                "S0": z([1, Vh, Dk, Dv], f),  # fp32 [1,Vh,Dk,Dv] (matches _chunk_state_ttnn S); read-only
                # per-chunk, persisted for the final concat -- bf16 by default, see _POOL_OUT_BF16
                "out": [z([1, Vh, C, Dv], ttnn.bfloat16 if _POOL_OUT_BF16 else f) for _ in range(Nc)],
                # 2 alternating carry buffers, indexed ci % 2 — see the docstring
                "Snew": [z([1, Vh, Dk, Dv], f) for _ in range(Nc if _POOL_SNEW_ALL else min(2, Nc))],
                # shared transients — reused across all chunks and all 30 layers (each fully consumed
                # within a chunk before the next overwrites it), so the pinned footprint is O(1) not O(Nc*L)
                "recur": {
                    "kcdS": z([1, Vh, C, Dv], f),
                    "v_new": z([1, Vh, C, Dv], f),
                    "qk": z([1, Vh, C, C], f),
                    "aintra": z([1, Vh, C, C], f),
                    "qgS": z([1, Vh, C, Dv], f),
                    "av": z([1, Vh, C, Dv], f),
                    "sglast": z([1, Vh, Dk, Dv], f),
                    "kgtv": z([1, Vh, Dk, Dv], f),
                },
                "valid": valid,
            }
        return {
            "S0": z([Vh * Dk, Dv]),
            "out": [z([Vh * C, Dv]) for _ in range(Nc)],
            "Snew": [z([Vh * Dk, Dv]) for _ in range(Nc if _POOL_SNEW_ALL else min(2, Nc))],  # as above
            "valid": valid,
        }

    @property
    def supports_inblock_snapshot(self):
        """Whether forward(snap_at=...) can export the carried state at an interior position.

        Only the fused recurrence can: it takes an initial_state and returns the final one, so it
        splits exactly. The hand-written chunked loop would need its per-chunk carry surfaced, which
        is not implemented -- QWEN36_GDN_FUSED_OP=0 therefore falls back to a separate snapshot
        block."""
        return bool(_GDN_FUSED_OP) and min(self.head_k_dim, self.head_v_dim) >= _FUSED_OP_MIN_HEAD_DIM

    def forward(
        self,
        x,
        cache=None,
        pool=None,
        init_state=None,
        valid_len=None,
        decode=False,
        verify=False,
        snap_at=None,
        snap_dst=None,
    ):
        """x: [1,1,T,hidden]. If cache (dict with conv_state/recurrent_state) given, continues from it.

        verify=True marks a speculative VERIFY of T=K tokens of ONE sequence (see _forward_verify_seq):
        like prefill it is a sequence, but the recurrence runs as K chained single-step ttl launches and
        the persistent state is left UNTOUCHED so the caller can roll back to the accepted prefix.

        decode=True marks BATCHED multi-user decode: the leading dim T is B independent users, each one
        token (NOT a sequence). B=1 keeps the fused ttl kernel; B>1 runs a single batched recurrent step
        over B independent states (_forward_decode_batch). This disambiguates B>1 decode from T>1 prefill.
        Returns [1,1,T,hidden]; updates cache in place when provided. pool: pre-allocated trace buffers
        (traced prefill only). init_state: head-major [Vh*Dk,Dv] starting recurrent state for incremental
        prefill (continue a prior context); conv likewise continues from the cached conv_state.

        valid_len: for a RAGGED incremental block, the number of real LEADING tokens (the remaining
        rows are right-padding the caller added so attention gets a tile-aligned block). Gated-delta
        handles any length natively, so we drop the padding here and process only the real rows — this
        keeps conv_state/recurrent_state updated over real tokens only — then right-pad the output back
        so the residual stream stays width T_full."""
        T_full = x.shape[2]
        hidden = x.shape[3]
        ragged = valid_len is not None and valid_len < T_full
        # Mask the padding out of the state (default) or slice the block down to the real rows (the
        # old path, QWEN36_GDN_RAGGED_MASK=0). Masking keeps every downstream op at the PADDED width,
        # which is what bounds the program-cache growth -- see _RAGGED_MASK.
        mask_ragged = ragged and _RAGGED_MASK
        if ragged and not mask_ragged:
            x = ttnn.slice(x, [0, 0, 0, 0], [1, 1, valid_len, hidden])  # keep only the real tokens
        T = x.shape[2]
        x2 = ttnn.reshape(x, [T, hidden])
        # Decode (T==1) keeps the tiny intermediates L1-resident (QWEN36_GDN_L1): measured to cut the
        # gated-delta step, since the decode path is many tiny dispatch-bound ops, not DRAM-bandwidth.
        # Prefill (T>1) stays on the default (interleaved DRAM) to avoid L1 overflow at large T.
        # L1-resident intermediates for T==1 decode AND for a small-K speculative verify. Verify is
        # the same regime the T==1 win came from — many tiny dispatch-bound ops — and its tensors are
        # only K rows wide, so they fit L1 easily. Measured: GDN verify was 1.0189 ms/layer against a
        # 0.3204 ms decode step, i.e. WORSE than 3 separate decodes, largely because it ran everything
        # from DRAM. Bounded by _VERIFY_L1_MAX_K so a large K cannot overflow L1.
        mc = _MC if (T == 1 or (verify and T <= _VERIFY_L1_MAX_K)) else None
        V = self.num_v_heads

        # beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias). One fused input matmul (default)
        # produces qkv|z|b|a together; slice it. z/b/a don't depend on the conv, so slice them upfront.
        with prof.phase(self.mesh_device, "delta.in_proj") if T > 1 else _nullctx(), sp.region("delta.in_proj"):
            if _FUSE_IN:
                allp = ttnn.linear(  # [T, conv_dim+value_dim+2V]
                    x2,
                    self.w_in_proj,
                    memory_config=mc,
                    # T <= 32: one tile row, so speculative verify shares decode's config (see moe.py).
                    program_config=self._pc_in_proj if (self._pc_in_proj is not None and T <= 32) else None,
                )
                o0, o1, o2 = self._in_off
                mixed = ttnn.slice(allp, [0, 0], [T, o0], memory_config=mc)  # -> conv input
                z = ttnn.slice(allp, [0, o0], [T, o1], memory_config=mc)  # [T, value_dim]
                ba = ttnn.slice(allp, [0, o1], [T, o2], memory_config=mc)  # [T, 2V]
            else:
                mixed = ttnn.linear(x2, self.w_qkv, memory_config=mc)  # [T, conv_dim]
                z = ttnn.linear(x2, self.w_z, memory_config=mc)  # [T, value_dim]
                ba = ttnn.linear(x2, self.w_ba, memory_config=mc)  # [T, 2V]

        with prof.phase(self.mesh_device, "delta.conv") if T > 1 else _nullctx(), sp.region("delta.conv"):
            conv_in = mixed  # pre-conv projection rows; verify rollback recomputes conv_state from these
            if _CONV_ADDCHAIN and (T == 1 or decode) and not verify and cache is not None and "conv_rows" in cache:
                # decode fast path (B=1 or batched B>1): per-user add-chain over separate history rows
                # ([B, conv_dim] for batched; each row is one user's independent conv history, synced from
                # conv_state at start_decode). State maintained in-place in conv_rows, no copy-back.
                mixed = self._conv_addchain_decode(mixed, cache["conv_rows"])
            else:
                conv_state = cache.get("conv_state") if cache else None
                # verify discards new_conv_state (commit_verify rebuilds the tail from conv_in)
                mixed, new_conv_state = self._conv_silu(
                    mixed,
                    conv_state,
                    need_state=not verify,
                    state_at=valid_len if mask_ragged else None,
                    snap_at=snap_at,
                    snap_dst=(snap_dst or {}).get("conv_state"),
                )
                if verify:
                    pass  # leave conv_state alone; commit_verify() applies the accepted prefix
                elif cache is not None:
                    if conv_state is not None:
                        ttnn.copy(new_conv_state, conv_state)  # in-place: keep persistent buffer (traceable)
                    else:
                        cache["conv_state"] = new_conv_state

        with sp.region("delta.split"):
            q = ttnn.slice(mixed, [0, 0], [T, self.key_dim], memory_config=mc)
            k = ttnn.slice(mixed, [0, self.key_dim], [T, 2 * self.key_dim], memory_config=mc)
            v = ttnn.slice(mixed, [0, 2 * self.key_dim], [T, self.conv_dim], memory_config=mc)

        # Fused single-step decode: one ttl launch for the whole step (prep + recurrence + gated-norm).
        # q/k [1,key_dim], v/z [1,value_dim] feed the kernel directly (heads along columns); raw a/b are
        # expanded to per-head uniform tiles; state stays head-major in the persistent cache.
        if verify and cache is not None:
            return self._forward_verify_seq(q, k, v, z, ba, cache, hidden, T, conv_in)
        if _GDN_FUSED and T == 1 and cache is not None:
            return self._forward_decode_fused(q, k, v, z, ba, cache, hidden)  # B=1 fast path
        if _GDN_BATCH_FUSED and decode and T > 1 and cache is not None:
            # Batched (B>1) fused decode: one ttl launch for all B users (on-core per-user state).
            return self._forward_decode_batch_fused(q, k, v, z, ba, cache, hidden, T)
        if decode and cache is not None:
            # Batched multi-user decode: B (== T) independent users, one token each. Single recurrent
            # step over B independent states [B,Vh,Dk,Dv] (NOT a timestep scan). Also handles B=1 when
            # the fused kernel is disabled.
            return self._forward_decode_batch(q, k, v, z, ba, cache, hidden, T)

        with sp.region("delta.gate"):
            beta = ttnn.sigmoid(ttnn.slice(ba, [0, 0], [T, V], memory_config=mc), memory_config=mc)  # [T, V]
            a = ttnn.slice(ba, [0, V], [T, 2 * V], memory_config=mc)  # [T, V]
            g = ttnn.multiply(self.neg_expA, ttnn.softplus(ttnn.add(a, self.dt_bias)), memory_config=mc)  # [T, V]
            if mask_ragged:
                # beta=0 -> the row contributes nothing (kbeta = vbeta = 0); g=0 (log space) -> it
                # decays nothing. Same identity the bucketed traced path applies via pool["valid"].
                vm = self._valid_col(T, valid_len, beta.dtype)
                beta = ttnn.multiply(beta, vm, memory_config=mc)
                g = ttnn.multiply(g, vm, memory_config=mc)
            # exp(g) is consumed ONLY by _recurrent_scan; both default recurrence paths (the native
            # fused op and the in-tree chunked prep) take the LOG-space g. Computing it here was one
            # dead [T, V] elementwise op per linear layer per forward (30 per prefill call). Built
            # lazily in the scan branch below instead.
            g_exp = None

        with prof.phase(self.mesh_device, "delta.qknorm") if T > 1 else _nullctx(), sp.region("delta.qknorm"):
            # reshape to [T, Hk, Dk]; l2norm over Dk; scale q; repeat_interleave k,q heads to V
            Vh, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
            q = ttnn.reshape(q, [T, self.num_k_heads, Dk])
            k = ttnn.reshape(k, [T, self.num_k_heads, Dk])
            v = ttnn.reshape(v, [T, Vh, Dv])
            q = _l2norm_scale_lastdim(q, scale=self.qk_scale, mc=mc)
            k = _l2norm_scale_lastdim(k, mc=mc)
            if self.n_rep > 1:
                q = ttnn.repeat_interleave(q, self.n_rep, dim=1)
                k = ttnn.repeat_interleave(k, self.n_rep, dim=1)

        with prof.phase(self.mesh_device, "delta.recurrence") if T > 1 else _nullctx(), sp.region("delta.recurrence"):
            if _FUSED_PREFILL and T > 1:
                # --- fused chunked prefill (parallel over the sequence) ---
                core, S_final = self._forward_prefill_chunked(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    pool=pool,
                    init_state=init_state,
                    snap_at=snap_at,
                    snap_dst=(snap_dst or {}).get("recurrent_state"),
                )
                if cache is not None:
                    state_buf = cache.get("recurrent_state")
                    S_final = ttnn.reshape(S_final, [1, Vh, Dk, Dv])  # decode reads state as [1,Vh,Dk,Dv]
                    if state_buf is not None:
                        ttnn.copy(S_final, state_buf)
                    else:
                        cache["recurrent_state"] = S_final
            else:
                g_exp = ttnn.exp(g, memory_config=mc)  # decay per step, [T, V] -- scan-only
                core = self._recurrent_scan(q, k, v, g_exp, beta, cache, T, Vh, Dk, Dv)

        with prof.phase(self.mesh_device, "delta.norm_out") if T > 1 else _nullctx(), sp.region("delta.norm_out"):
            core = ttnn.transpose(core, 1, 2, memory_config=mc)  # [1, T, V, Dv]
            core = ttnn.reshape(core, [1, 1, T * Vh, Dv])
            z_r = ttnn.reshape(z, [1, 1, T * Vh, Dv])
            core = self.norm.forward(core, z_r)
            core = ttnn.reshape(core, [T, self.value_dim])
            y = ttnn.linear(core, self.w_out, memory_config=mc)
            y = ttnn.reshape(y, [1, 1, T, hidden])
        if ragged and not mask_ragged:  # slicing path: restore the padded width (pad rows unused
            # downstream; the head reads the real last token). The mask path never narrowed y.
            y = ttnn.pad(y, [(0, 0), (0, 0), (0, T_full - T), (0, 0)], value=0.0)
        return y

    def _verify_conv_buf(self, K):
        """Persistent [K, conv_dim] copy of the conv INPUT rows, allocated once per K.

        `commit_verify` recomputes the rolled-back conv_state from these, and it runs in its OWN
        captured trace — a trace cannot reference a transient produced inside a DIFFERENT trace, so
        this has to be a stable buffer.

        Keyed BY K, and that is load-bearing: a round with gamma>0 alternates between the K=1 seed
        step and K-row verifies, and a single shared attribute would hand a [K, conv_dim] buffer to a
        [1, conv_dim] copy (measured: TT_FATAL shape mismatch [1,8192] vs [3,8192]).

        BF16 whatever the state dtype: commit_verify concats it with the (bf16) conv_state, and
        ttnn.concat requires matching dtypes.

        Split out of the old `_verify_scratch`, which also built the per-step `states`/`outs` that
        ONLY the non-chain path reads (see `_verify_step_scratch`). With `_GDN_VERIFY_CHAIN` on — the
        default — those were allocated and never touched: K x ([V*Dk,Dv] + [TILE,value_dim]) per
        layer, i.e. a MEASURED 150 MB across 30 layers at gamma=2 (both K=1 and K=3 are cached)."""
        if not hasattr(self, "_verify_conv_cache"):
            self._verify_conv_cache = {}
        buf = self._verify_conv_cache.get(K)
        if buf is None:
            buf = ttnn.zeros([K, self.conv_dim], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
            self._verify_conv_cache[K] = buf
        return buf

    def _verify_step_scratch(self, K):
        """Per-STEP persistent scratch for the NON-chained verify path (`_GDN_VERIFY_CHAIN=0`): K
        state buffers [V*Dk,Dv] and K output tile-rows. Stable addresses, as trace capture requires.

        Only built when that path actually runs — the chained kernel writes one [K*nv*Dk,Dv] stack
        instead (`_verify_chain_scratch`), so allocating these too was pure waste."""
        if not hasattr(self, "_verify_scratch_cache"):
            self._verify_scratch_cache = {}
        sc = self._verify_scratch_cache.get(K)
        if sc is None:
            V, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
            z_ = lambda shp: ttnn.zeros(shp, dtype=_STATE_DT, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
            # out buffers are a FULL tile-row each (the kernel writes tile-row 0 of `out`)
            sc = ([z_([V * Dk, Dv]) for _ in range(K)], [z_([TILE, self.value_dim]) for _ in range(K)])
            self._verify_scratch_cache[K] = sc
        return sc

    def _verify_chain_scratch(self, K):
        """Persistent output/state buffers for the CHAINED verify kernel: out [K*TILE, value_dim] and
        Snew [K*nv*Dk, Dv]. `Snew` must be persistent, not a transient: `commit_verify` slices step
        j-1 out of it and runs in its OWN captured trace, which cannot reference a buffer produced
        inside the verify trace."""
        if not hasattr(self, "_verify_chain_cache"):
            self._verify_chain_cache = {}
        sc = self._verify_chain_cache.get(K)
        if sc is None:
            V, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
            z_ = lambda shp: ttnn.zeros(shp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
            sc = (z_([K * TILE, self.value_dim]), z_([K * V * Dk, Dv]))
            self._verify_chain_cache[K] = sc
        return sc

    def _verify_const(self, K):
        """negA / dt_bias replicated to the chained kernel's [K*nv*TILE, TILE] layout (cached: they are
        per-head constants, identical for every step)."""
        if not hasattr(self, "_verify_const_cache"):
            self._verify_const_cache = {}
        c = self._verify_const_cache.get(K)
        if c is None:
            rep = lambda t: ttnn.reshape(
                ttnn.repeat(ttnn.reshape(t, [1, t.shape[0], t.shape[1]]), ttnn.Shape([K, 1, 1])),
                [K * t.shape[0], t.shape[1]],
            )
            c = (rep(self._fused_negA), rep(self._fused_dtb))
            self._verify_const_cache[K] = c
        return c

    def _forward_verify_seq(self, q, k, v, z, ba, cache, hidden, K, conv_in):
        """Speculative VERIFY of K tokens of ONE sequence: K CHAINED single-step ttl launches.

        This is where the GDN amortization comes from, and it is not the recurrence: the fused ttl
        kernel is only 0.061 of the 0.319 ms GDN layer, while `w_in_proj`, the conv and `w_out` — the
        other 81% — are flat in K (per_core_M=1 for any m<=32). So running the projections ONCE over
        the K rows and looping only the kernel gives ~2.5x amortization with no new kernel. A chained
        ON-CORE kernel would save a further ~2% of a round; see MTP.md.

        The persistent `recurrent_state` and `conv_state` are NOT modified — every step writes into
        scratch — so the caller can accept any prefix and then call `commit_verify(cache, j)`."""
        V, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
        conv_buf = self._verify_conv_buf(K)
        # per-step state/out buffers exist ONLY for the non-chained path; the chained kernel writes a
        # single [K*nv*Dk,Dv] stack instead, so building them there allocated 150 MB nothing read.
        states, outs = (None, None) if _GDN_VERIFY_CHAIN else self._verify_step_scratch(K)

        # ttl indexes in TILE units, not rows: `ct = 1` is one 32-row tile, and the kernel's outer
        # product k^T @ delta contracts over ALL 32 rows of it. So each step's operand must be a tile
        # whose row 0 is the token and whose other 31 rows are ZERO. A plain `ttnn.slice(q,[t,0],...)`
        # of a [K, feat] tensor does NOT satisfy that (the pad carries the neighbouring token rows) —
        # it silently produced a wrong result while every input value compared bit-identical, and only
        # K==1 happened to work because an M=1 matmul zeroes its own tile pad. `_to_tilerows` (the same
        # helper the batched kernel uses) lays each token on its own zero-padded tile-row.
        #
        # q|k and v|z are PAIRED into one relayout each rather than four separate ones: the pads are
        # 3 ops apiece and the verify trace is dispatch-bound, so this halves that overhead while
        # staying provably identical (all four operands are still zero-padded, just in two tensors).
        if _GDN_VERIFY_CHAIN:
            # the chained kernel takes each operand as its OWN [K*TILE, feat] tensor (it computes the
            # per-head column slab itself), so no pairing here; pads measured ~free (EXPERIMENTS #4).
            qT = self._to_tilerows(q, self.key_dim, K)
            kT = self._to_tilerows(k, self.key_dim, K)
            vT = self._to_tilerows(v, self.value_dim, K)
            zT = self._to_tilerows(z, self.value_dim, K)
        else:
            qkT = self._to_tilerows(ttnn.concat([q, k], dim=1), 2 * self.key_dim, K)
            vzT = self._to_tilerows(ttnn.concat([v, z], dim=1), 2 * self.value_dim, K)

        if _GDN_VERIFY_CHAIN:
            # ---- ONE ttl launch for the whole K-step chain (see ttl_delta._get_decode_op_chain) ----
            # araw/braw must be laid out per-(t,h) as [K*nv*TILE, TILE]: `ba` is [K, 2V] with row t =
            # [b(V) | a(V)], so b and a interleave per step and cannot simply be split in half. Build
            # the two halves separately and concat, which keeps this to a handful of ops for all K.
            with sp.region("delta.recurrence"):
                bslice = ttnn.slice(ba, [0, 0], [K, V])  # [K, V] all steps' b
                aslice = ttnn.slice(ba, [0, V], [K, 2 * V])  # [K, V] all steps' a
                braw = ttnn.reshape(
                    ttnn.repeat(ttnn.reshape(bslice, [K * V, 1, 1]), ttnn.Shape([1, TILE, TILE])),
                    [K * V * TILE, TILE],
                )
                araw = ttnn.reshape(
                    ttnn.repeat(ttnn.reshape(aslice, [K * V, 1, 1]), ttnn.Shape([1, TILE, TILE])),
                    [K * V * TILE, TILE],
                )
                nA, dtbc = self._verify_const(K)
                out_buf, snew_buf = self._verify_chain_scratch(K)
                decode_step_chain_tt(
                    qT,
                    kT,
                    vT,
                    zT,
                    araw,
                    braw,
                    nA,
                    dtbc,
                    ttnn.reshape(cache["recurrent_state"], [V * Dk, Dv]),
                    self._fused_nweight,
                    out_buf,
                    snew_buf,
                    V,
                    self.num_k_heads,
                    self.qk_scale,
                    self.eps,
                    K,
                )
            # `snew_buf` is the persistent per-step state STACK; commit_verify slices step j-1 out of
            # it inside its own trace (so it must not be sliced into transients here).
            pending_states = snew_buf
        else:
            with sp.region("delta.recurrence"):
                # raw a|b -> per-(t,head) uniform [V*TILE,TILE] tiles, exactly as _forward_decode_fused
                # does for one token, but sliced per step.
                #
                # `keep` is load-bearing: ttnn enqueues asynchronously, so a per-step operand whose last
                # Python reference dies at the end of the loop iteration has its device buffer FREED
                # immediately and can be handed to a later step's allocation before the earlier ttl launch
                # has run. That corrupts the EARLIER steps (the last one is never clobbered) and is
                # non-deterministic — measured as row-0 PCC wandering 0.31..0.58 while row K-1 stayed
                # ~0.99. Holding every per-step operand alive until after the launches fixes it.
                keep = []
                # Expand the raw a|b gating scalars to per-(t,head) uniform tiles ONCE for all K steps.
                # Doing it inside the loop costs 7 ops per step per layer (slice, transpose, reshape,
                # repeat, reshape, 2 slices) = 21 ops/layer at K=3, and the verify trace is DISPATCH-bound
                # (measured 64.1 ms vs a 38.4 ms cost model, with only ~1.3 ms of host glue), so op count
                # is the thing that matters here. Hoisted this is 3 ops + 2 slices/step.
                # Layout: ba is [K, 2V] and row t is [b(V) | a(V)], so flattening row-major puts (t, j) at
                # index t*2V + j -> tile-row (t*2V + j). Hence b for step t occupies tile-rows
                # [t*2V, t*2V+V) and a occupies [t*2V+V, (t+1)*2V).
                if _FP32_STATE:
                    # all-fp32 kernel: convert the hoisted operands ONCE, outside the K loop
                    ba = ttnn.typecast(ba, ttnn.float32)
                    qkT = ttnn.typecast(qkT, ttnn.float32)
                    vzT = ttnn.typecast(vzT, ttnn.float32)
                ba_all = ttnn.repeat(ttnn.reshape(ba, [K * 2 * V, 1, 1]), ttnn.Shape([1, TILE, TILE]))
                ba_all = ttnn.reshape(ba_all, [K * 2 * V * TILE, TILE])
                keep.append(ba_all)
                for t in range(K):
                    b0 = t * 2 * V * TILE
                    braw_tile = ttnn.slice(ba_all, [b0, 0], [b0 + V * TILE, TILE])
                    araw_tile = ttnn.slice(ba_all, [b0 + V * TILE, 0], [b0 + 2 * V * TILE, TILE])
                    # step 0 reads the PERSISTENT state (read-only); later steps chain through scratch
                    S_src = ttnn.reshape(cache["recurrent_state"], [V * Dk, Dv]) if t == 0 else states[t - 1]
                    # Match _forward_decode_fused's operand residency exactly: the single-step ttl kernel
                    # is only ever exercised with L1 operands (T==1 sets mc=_MC), and feeding it the
                    # DRAM rows a T=K projection produces gives a WRONG result (row-0 PCC 0.58 vs 1.0)
                    # even though the values are bit-identical. Slice, then place in L1.
                    r0, r1 = t * TILE, (t + 1) * TILE  # tile-aligned: the zero pad is preserved
                    kd, vd = self.key_dim, self.value_dim
                    q_t = ttnn.slice(qkT, [r0, 0], [r1, kd])
                    k_t = ttnn.slice(qkT, [r0, kd], [r1, 2 * kd])
                    v_t = ttnn.slice(vzT, [r0, 0], [r1, vd])
                    z_t = ttnn.slice(vzT, [r0, vd], [r1, 2 * vd])
                    keep += [braw_tile, araw_tile, S_src, q_t, k_t, v_t, z_t]
                    decode_step_tt(
                        q_t,
                        k_t,
                        v_t,
                        z_t,
                        araw_tile,
                        braw_tile,
                        self._fused_negA,
                        self._fused_dtb,
                        S_src,
                        self._fused_nweight,
                        outs[t],
                        states[t],
                        V,
                        self.num_k_heads,
                        self.qk_scale,
                        self.eps,
                        _FP32_STATE,
                    )
            pending_states = states  # list of persistent per-step buffers
        # stash what commit_verify needs. conv_in is the PRE-conv projection (the conv's own input);
        # copy it into the persistent buffer so a separately-captured commit trace can read it.
        ttnn.copy(conv_in, conv_buf)
        self._verify_pending = (pending_states, conv_buf)
        with sp.region("delta.w_out"):
            if _GDN_VERIFY_CHAIN:
                core = self._from_tilerows(out_buf, self.value_dim, K)  # [K, value_dim]
            else:
                rows = [ttnn.slice(o, [0, 0], [1, self.value_dim]) for o in outs]
                core = rows[0] if K == 1 else ttnn.concat(rows, dim=0)
            if self._wout_dram is not None and K <= TILE:
                # The DRAM-sharded w_out is usable here after all, not "M=1-only": `build_dram_shard`
                # already sizes the activation/output shards at a FULL 32-row tile with per_core_M=1,
                # so any K <= 32 fits the same config — the decode path merely happens to call it with
                # one row. Worth ~0.06 ms/layer (interleaved measured 0.0952).
                osh = ttnn.linear(
                    ttnn.to_memory_config(core, self._wout_amc),
                    self._wout_dram,
                    program_config=self._wout_pc,
                    memory_config=self._wout_omc,
                )
                y = ttnn.to_memory_config(osh, _MC if _GDN_L1 else ttnn.DRAM_MEMORY_CONFIG)
            else:
                y = ttnn.linear(core, self.w_out)
            return ttnn.reshape(y, [1, 1, K, hidden])

    def _commit_conv_const(self, j, K):
        """Selection matrices for rolling the conv window forward by j accepted tokens, cached per (j,K).

        `new_state[i] = row (j+i) of [conv_state(conv_k-1 rows) ; conv_in(K rows)]`, so it is a pure row
        selection from the two sources — the same exact-by-construction trick the verify conv uses:

            new_state = Scs @ conv_state + Sci @ conv_in

        replacing a concat plus two sub-tile row slices (~118 + 2x108 us) with two tiny matmuls and an
        add. Returns (Scs, Sci); either is None when that source contributes no row, which is the common
        case — at K == conv_k-1 the full-accept commit reads conv_in alone."""
        if not hasattr(self, "_commit_conv_cache"):
            self._commit_conv_cache = {}
        key = (j, K)
        c = self._commit_conv_cache.get(key)
        if c is None:
            P, CK = self.conv_k - 1, self.conv_k
            cs_m, ci_m = torch.zeros(P, P), torch.zeros(P, K)
            for i in range(P):
                r = j + i
                if r < P:
                    cs_m[i, r] = 1.0
                else:
                    ci_m[i, r - P] = 1.0
            mk = lambda m: to_tt(m, self.mesh_device, dtype=_STATE_DT)
            c = (
                mk(cs_m) if bool(cs_m.any()) else None,
                mk(ci_m) if bool(ci_m.any()) else None,
                # identity fast path: every new row comes from conv_in, in order, starting at row 0
                bool(not cs_m.any()) and P == K and bool((ci_m == torch.eye(P)).all()),
            )
            self._commit_conv_cache[key] = c
        return c

    def commit_verify(self, cache, j):
        """Advance the persistent state by the j ACCEPTED tokens of the last verify (j in 0..K).

        recurrent_state <- the j-th chained scratch state. conv_state <- the last (conv_k-1) rows of
        [old conv_state | conv_in[:j]] — recomputed rather than snapshotted, because `_conv_silu`
        only ever produced the tail after ALL K tokens."""
        if j <= 0 or getattr(self, "_verify_pending", None) is None:
            return
        V, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
        pending, conv_in = self._verify_pending
        if isinstance(pending, list):  # per-step-launch path: one persistent buffer per step
            st = pending[j - 1]
        else:  # chained path: one persistent [K*nv*Dk, Dv] stack, slice step j-1 out of it HERE (this
            # runs inside the commit trace, so the slice must be built from a persistent buffer)
            st = ttnn.slice(pending, [(j - 1) * V * Dk, 0], [j * V * Dk, Dv])
        ttnn.copy(ttnn.reshape(st, [1, V, Dk, Dv]), cache["recurrent_state"])
        cs = cache.get("conv_state")
        if cs is not None:
            Scs, Sci, ident = self._commit_conv_const(j, conv_in.shape[0])
            if ident and self._COMMIT_COPY:
                # full accept: new conv_state IS conv_in (see _commit_conv_const). One copy.
                ttnn.copy(conv_in, cs)
            elif not self._COMMIT_SELECT:
                kept = ttnn.concat([cs, ttnn.slice(conv_in, [0, 0], [j, self.conv_dim])], dim=0)
                ttnn.copy(ttnn.slice(kept, [j, 0], [j + self.conv_k - 1, self.conv_dim]), cs)
            elif Scs is None or Sci is None:  # one source contributes every row: a single matmul
                ttnn.copy(ttnn.matmul(Scs or Sci, cs if Sci is None else conv_in, compute_kernel_config=_HIFI4), cs)
            else:
                # `add` accepts output_tensor= (matmul does not), so the sum lands straight in
                # conv_state: two ops, and one fewer transient inside the commit trace.
                ttnn.add(
                    ttnn.matmul(Scs, cs, compute_kernel_config=_HIFI4),
                    ttnn.matmul(Sci, conv_in, compute_kernel_config=_HIFI4),
                    output_tensor=cs,
                )

    def _forward_decode_batch(self, q, k, v, z, ba, cache, hidden, B):
        """Batched multi-user decode: B independent users, each one token. ONE recurrent step over B
        independent states [B,Vh,Dk,Dv] (no timestep scan — each user advances one step). Inputs are the
        post-conv projections: q/k [B,key_dim], v/z [B,value_dim], ba [B,2*Vh]. Updates
        cache['recurrent_state'] ([B,Vh,Dk,Dv]) in place. Same recurrence math as the single-user scan/
        fused kernel; validated per-user PCC vs B=1 (tests/test_gated_delta_batch.py)."""
        Vh, Kh, Dk, Dv = self.num_v_heads, self.num_k_heads, self.head_k_dim, self.head_v_dim
        mc = _MC
        with sp.region("delta.gate"):
            beta = ttnn.sigmoid(ttnn.slice(ba, [0, 0], [B, Vh], memory_config=mc), memory_config=mc)  # [B,Vh]
            a = ttnn.slice(ba, [0, Vh], [B, 2 * Vh], memory_config=mc)  # [B,Vh]
            g_exp = ttnn.exp(
                ttnn.multiply(self.neg_expA, ttnn.softplus(ttnn.add(a, self.dt_bias)), memory_config=mc),
                memory_config=mc,
            )  # [B,Vh]
        with sp.region("delta.qknorm"):
            q = _l2norm_scale_lastdim(ttnn.reshape(q, [B, Kh, Dk]), scale=self.qk_scale, mc=mc)
            k = _l2norm_scale_lastdim(ttnn.reshape(k, [B, Kh, Dk]), mc=mc)
            v = ttnn.reshape(v, [B, Vh, Dv])
            if self.n_rep > 1:
                q = ttnn.repeat_interleave(q, self.n_rep, dim=1)
                k = ttnn.repeat_interleave(k, self.n_rep, dim=1)
        with sp.region("delta.recurrence"):
            state_buf = cache.get("recurrent_state") if cache else None
            state = state_buf
            if state is None:
                state = ttnn.zeros([B, Vh, Dk, Dv], dtype=_STATE_DT, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
            q_row = ttnn.reshape(q, [B, Vh, 1, Dk])
            k_row = ttnn.reshape(k, [B, Vh, 1, Dk])
            k_col = ttnn.reshape(k, [B, Vh, Dk, 1])
            v_row = ttnn.reshape(v, [B, Vh, 1, Dv])
            g_t = ttnn.reshape(g_exp, [B, Vh, 1, 1])
            b_t = ttnn.reshape(beta, [B, Vh, 1, 1])
            state = ttnn.multiply(state, g_t)  # [B,Vh,Dk,Dv] decay
            kv = ttnn.matmul(k_row, state)  # [B,Vh,1,Dv]
            delta = ttnn.multiply(ttnn.subtract(v_row, kv), b_t)  # [B,Vh,1,Dv]
            outer = ttnn.matmul(k_col, delta)  # [B,Vh,Dk,Dv]
            state = ttnn.add(state, outer)
            core = ttnn.matmul(q_row, state)  # [B,Vh,1,Dv]
            if cache is not None:
                if state_buf is not None:
                    ttnn.copy(state, state_buf)  # in-place persistent buffer (trace-safe)
                else:
                    cache["recurrent_state"] = state
        with sp.region("delta.norm_out"):
            core = ttnn.reshape(core, [1, 1, B * Vh, Dv])
            z_r = ttnn.reshape(z, [1, 1, B * Vh, Dv])
            core = self.norm.forward(core, z_r)  # gated RMSNorm over Dv
            core = ttnn.reshape(core, [B, self.value_dim])
            y = ttnn.linear(core, self.w_out, memory_config=mc)
            return ttnn.reshape(y, [1, 1, B, hidden])

    def _recurrent_scan(self, q, k, v, g_exp, beta, cache, T, Vh, Dk, Dv):
        """On-device recurrent delta-rule scan (decode path, or non-fused prefill). Returns core
        [1, V, T, Dv]; updates the recurrent_state cache in place when provided. Extracted verbatim
        from forward() so the signpost region boundaries above read cleanly."""
        state_buf = cache.get("recurrent_state") if cache else None
        state = state_buf
        if state is None:
            state = ttnn.zeros([1, Vh, Dk, Dv], dtype=_STATE_DT, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        outs = []
        for t in range(T):
            q_row = ttnn.reshape(ttnn.slice(q, [t, 0, 0], [t + 1, Vh, Dk]), [1, Vh, 1, Dk])
            k_row = ttnn.reshape(ttnn.slice(k, [t, 0, 0], [t + 1, Vh, Dk]), [1, Vh, 1, Dk])
            k_col = ttnn.reshape(k_row, [1, Vh, Dk, 1])
            v_row = ttnn.reshape(ttnn.slice(v, [t, 0, 0], [t + 1, Vh, Dv]), [1, Vh, 1, Dv])
            g_t = ttnn.reshape(ttnn.slice(g_exp, [t, 0], [t + 1, Vh]), [1, Vh, 1, 1])
            b_t = ttnn.reshape(ttnn.slice(beta, [t, 0], [t + 1, Vh]), [1, Vh, 1, 1])

            # NOTE: the recurrence state intermediates ([1,Vh,Dk,Dv] ≈ 1 MB each, several per
            # layer) stay in DRAM, NOT L1. A captured decode trace pins every buffer it references
            # for the trace's lifetime (no mid-trace free/reuse), so L1-resident state would
            # accumulate across all 30 gated-delta layers and overflow L1 (~1.5 MB/core) — fine at
            # 4 layers, wedges the device at 40. Only the SMALL prep/out intermediates (conv,
            # l2norm, g/β, projections) are L1 (their 40-layer cumulative footprint is tiny).
            state = ttnn.multiply(state, g_t)
            kv_mem = ttnn.matmul(k_row, state)  # [1,V,1,Dv]
            delta = ttnn.multiply(ttnn.subtract(v_row, kv_mem), b_t)  # [1,V,1,Dv]
            outer = ttnn.matmul(k_col, delta)  # [1,V,Dk,Dv]
            state = ttnn.add(state, outer)
            out_t = ttnn.matmul(q_row, state)  # [1,V,1,Dv]
            outs.append(out_t)
        if cache is not None:
            if state_buf is not None:
                ttnn.copy(state, state_buf)  # in-place: keep persistent buffer (traceable decode)
            else:
                cache["recurrent_state"] = state
        return ttnn.concat(outs, dim=2)  # [1, V, T, Dv]

    def _forward_decode_fused(self, q, k, v, z, ba, cache, hidden):
        """Fused single-step (T=1) decode: one ttl launch (decode_step_tt) does l2norm(q)*scale +
        l2norm(k) + gate/beta + the rank-1 recurrence + the gated RMSNorm, replacing the ~30-op scan
        path above. q/k:[1,key_dim], v/z:[1,value_dim] (heads along columns, fed directly). State is
        read from / written back to the PERSISTENT recurrent_state cache (head-major view), so the
        write is in-place and trace-safe. Returns [1,1,1,hidden]."""
        V, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim

        with sp.region("delta.recurrence"):
            # raw a|b -> per-head uniform [V*TILE, TILE] tiles. transpose [1,2V]->[2V,1] moves heads to
            # rows; repeat fills each head's tile; slice splits b|a (one expansion for both, then split).
            if _FP32_STATE:
                # The all-fp32 kernel needs every argument f32. Convert as FEW tensors as possible:
                # `ba` here (tiny, [1,2V]) so both per-head tiles inherit it, and q/k/v/z below.
                ba = ttnn.typecast(ba, ttnn.float32)
            bat = ttnn.transpose(ba, 0, 1)  # [2V, 1]
            bat = ttnn.repeat(ttnn.reshape(bat, [2 * V, 1, 1]), ttnn.Shape([1, TILE, TILE]))  # [2V, TILE, TILE]
            bat = ttnn.reshape(bat, [2 * V * TILE, TILE])
            braw_tile = ttnn.slice(bat, [0, 0], [V * TILE, TILE])  # b is the first V
            araw_tile = ttnn.slice(bat, [V * TILE, 0], [2 * V * TILE, TILE])  # a is the second V

            if _FP32_STATE:
                q, k, v, z = (ttnn.typecast(t, ttnn.float32) for t in (q, k, v, z))
            state_buf = cache["recurrent_state"]  # persistent [1,V,Dk,Dv]
            S_in = ttnn.reshape(state_buf, [V * Dk, Dv])  # head-major view for the kernel
            if self._fused_scratch is None:  # persistent scratch (allocated once; reused every step/trace)
                z_ = lambda shp: ttnn.zeros(shp, dtype=_STATE_DT, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
                # With _GDN_ALIAS_STATE the kernel writes the state in place, so no Snew buffer exists.
                self._fused_scratch = (
                    z_([1, self.value_dim]),
                    None if _GDN_ALIAS_STATE else z_([V * Dk, Dv]),
                )
            out_buf, snew_buf = self._fused_scratch
            if _GDN_ALIAS_STATE:
                snew_buf = S_in  # in-place: S and Snew are the same buffer

            decode_step_tt(
                q,
                k,
                v,
                z,
                araw_tile,
                braw_tile,
                self._fused_negA,
                self._fused_dtb,
                S_in,
                self._fused_nweight,
                out_buf,
                snew_buf,
                V,
                self.num_k_heads,
                self.qk_scale,
                self.eps,
                _FP32_STATE,
            )
            if not _GDN_ALIAS_STATE:
                ttnn.copy(ttnn.reshape(snew_buf, [1, V, Dk, Dv]), state_buf)  # in-place update (trace-safe)
        with sp.region("delta.w_out"):
            if self._wout_dram is not None:
                # DRAM-sharded output projection: L1-shard the activation, matmul, reshard back to L1.
                osh = ttnn.linear(
                    ttnn.to_memory_config(out_buf, self._wout_amc),
                    self._wout_dram,
                    program_config=self._wout_pc,
                    memory_config=self._wout_omc,
                )  # [1, hidden] width-sharded
                y = ttnn.to_memory_config(osh, ttnn.L1_MEMORY_CONFIG if _GDN_L1 else ttnn.DRAM_MEMORY_CONFIG)
            elif _FP32_STATE:
                # out_buf is f32 (all-fp32 kernel) while w_out is bf16; pin the result back to bf16 so
                # the residual stream dtype is unchanged downstream.
                y = ttnn.linear(out_buf, self.w_out, memory_config=_MC, dtype=ttnn.bfloat16)
            else:
                y = ttnn.linear(out_buf, self.w_out, memory_config=_MC)  # [1, hidden]
            return ttnn.reshape(y, [1, 1, 1, hidden])

    def _to_tilerows(self, x, feat, B):
        """[B, feat] (B users packed in one tile-row) -> [B*TILE, feat] (user b on its own tile-row b,
        at element row b*TILE; the other 31 rows zero — REQUIRED because the kernel's outer product
        k^T@delta contracts over all 32 tile rows). Reshape->pad->reshape (validated)."""
        x = ttnn.reshape(x, [B, 1, feat])
        x = ttnn.pad(x, [(0, 0), (0, TILE - 1), (0, 0)], value=0.0)
        return ttnn.reshape(x, [B * TILE, feat])

    def _from_tilerows(self, x, feat, B):
        """[B*TILE, feat] -> [B, feat] (extract each user's row 0 = element row b*TILE)."""
        x = ttnn.reshape(x, [B, TILE, feat])
        x = ttnn.slice(x, [0, 0, 0], [B, 1, feat])
        return ttnn.reshape(x, [B, feat])

    def _batch_const(self, B):
        """Per-head consts (-exp(A_log), dt_bias) for the batched kernel.

        With `_BATCH_SHARED_CONST` (the default) the kernel indexes them at tile-row h, so the
        un-replicated [V*TILE,TILE] tensors go straight through and nothing is allocated here at all.
        Otherwise they are replicated to [B*V*TILE,TILE] (user-b/head-h tile at row (b*V+h)) to match
        the kernel's old (b*V+h) indexing -- 4 MB/layer at B=32 (MEASURED 120 MB over 30 layers) to
        hold V=32 scalars. Cached per B."""
        if _BATCH_SHARED_CONST:
            return self._fused_negA, self._fused_dtb
        if B not in self._fused_batch_const:
            V = self.num_v_heads

            def rep(t):
                t = ttnn.repeat(ttnn.reshape(t, [1, V * TILE, TILE]), ttnn.Shape([B, 1, 1]))  # [B, V*TILE, TILE]
                return ttnn.reshape(t, [B * V * TILE, TILE])

            self._fused_batch_const[B] = (rep(self._fused_negA), rep(self._fused_dtb))
        return self._fused_batch_const[B]

    def _batch_scratch(self, B):
        """Persistent (out, Snew) scratch for the batched kernel, allocated once per B (reused every
        step/trace, so the in-place state write points at a stable address).

        Under _GDN_ALIAS_STATE there is NO Snew buffer: the kernel updates the persistent state in
        place, exactly as the B=1 path does. That buffer is [B*nv*Dk, Dv], which is the single largest
        per-layer allocation in batched decode -- MEASURED 32 MB/layer at B=32, i.e. ~960 MB across
        the 30 gated-delta layers -- plus a same-sized copy back into the cache every step. Batched
        decode is memory-capped (that is why the ttnn scan OOMs at B>=16), so this is batch headroom.

        It also removes a latent dtype bug: `Snew` here was hardcoded BFLOAT16 while the batched
        `recurrent_state` follows `_STATE_DT`, and `decode_step_batch_tt` has no fp32 build -- so
        QWEN36_GDN_FP32_STATE=1 + batched decode mismatched on the copy. With no buffer there is
        nothing to mismatch."""
        if B not in self._fused_batch_scratch:
            V, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
            self._fused_batch_scratch[B] = (
                ttnn.zeros(
                    [B * TILE, self.value_dim], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
                ),
                None
                if _GDN_ALIAS_STATE
                else ttnn.zeros(
                    [B * V * Dk, Dv], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
                ),
            )
        return self._fused_batch_scratch[B]

    def _forward_decode_batch_fused(self, q, k, v, z, ba, cache, hidden, B):
        """Batched multi-user fused decode: ONE ttl launch (decode_step_batch_tt) does the whole
        gated-delta step (l2norm+gate+recurrence+gated-norm) for B independent users. Each core loops
        its value-head over the B users, streaming per-user state from the persistent recurrent_state
        cache ([B,V,Dk,Dv]) — so NO B x recurrence intermediates are pinned in the trace (the scan's
        OOM at B>=16 is removed). Inputs are post-conv projections q/k [B,key_dim], v/z [B,value_dim],
        ba [B,2V]. Updates the state cache in place. Returns [1,1,B,hidden]."""
        V, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim

        with sp.region("delta.recurrence"):
            # raw a|b [B,2V] -> per-(b,h) uniform tiles [B*V*TILE, TILE] (tile-row (b*V+h) = the scalar).
            braw_part = ttnn.slice(ba, [0, 0], [B, V])  # b (beta) is the first V
            araw_part = ttnn.slice(ba, [0, V], [B, 2 * V])  # a is the second V

            def _uniform_tiles(t):  # [B,V] -> [B*V*TILE, TILE], tile-row (b*V+h) uniform = t[b,h]
                t = ttnn.repeat(ttnn.reshape(t, [B * V, 1, 1]), ttnn.Shape([1, TILE, TILE]))  # [B*V, TILE, TILE]
                return ttnn.reshape(t, [B * V * TILE, TILE])

            braw_tile = _uniform_tiles(braw_part)
            araw_tile = _uniform_tiles(araw_part)
            negA_b, dtb_b = self._batch_const(B)

            # q/k/v/z [B,feat] -> [B*TILE,feat] (each user on its own tile-row)
            qk = self._to_tilerows(q, self.key_dim, B)
            kk = self._to_tilerows(k, self.key_dim, B)
            vk = self._to_tilerows(v, self.value_dim, B)
            zk = self._to_tilerows(z, self.value_dim, B)

            state_buf = cache["recurrent_state"]  # persistent [B,V,Dk,Dv]
            S_in = ttnn.reshape(state_buf, [B * V * Dk, Dv])  # head-major view over (b,h)
            out_buf, snew_buf = self._batch_scratch(B)
            if _GDN_ALIAS_STATE:
                # In-place on-core state update, same as the B=1 path. Safe by the same argument,
                # generalised over b: core h owns only slab (b*nv+h)*kt_t, so the slabs are disjoint
                # across BOTH cores (h) and loop iterations (b); read() copies S(b) and t7.wait()s it
                # before compute() can Sd.wait() it, and write() fires only after compute() produces
                # Snd(b) -- so per slab the read strictly precedes the write. block_count=2 lets
                # read() run at most two iterations ahead, and those touch OTHER b slabs, never the
                # one write() is on (ttl_delta.py:791/:907).
                snew_buf = S_in

            decode_step_batch_tt(
                qk,
                kk,
                vk,
                zk,
                araw_tile,
                braw_tile,
                negA_b,
                dtb_b,
                S_in,
                self._fused_nweight,
                out_buf,
                snew_buf,
                V,
                self.num_k_heads,
                self.qk_scale,
                self.eps,
                B,
            )
            if not _GDN_ALIAS_STATE:
                ttnn.copy(ttnn.reshape(snew_buf, [B, V, Dk, Dv]), state_buf)  # in-place (trace-safe)
        with sp.region("delta.w_out"):
            core = self._from_tilerows(out_buf, self.value_dim, B)  # [B, value_dim]
            y = ttnn.linear(core, self.w_out, memory_config=_MC)  # interleaved (DRAM-shard is a B=1 opt)
            return ttnn.reshape(y, [1, 1, B, hidden])
