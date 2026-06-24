# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-nn Gated DeltaNet (linear attention) for Qwen3.6-35B-A3B.

Phase B: the delta-rule recurrence runs ON DEVICE (no host round-trip). The per-step rank-1 state
update is expressed as batched matmuls over the head dim:
    kv_mem = k_row @ state ;  outer = k_col @ delta ;  out = q_row @ state
where ``state`` is [1, V, Dk, Dv]. Supports a persistent (conv_state, recurrent_state) cache so
decode is a single step instead of recomputing the sequence.

This is still a sequential scan (O(seq)); the further optimization is a fused chunked tt-lang
kernel (parallel over the sequence). The recurrent form here is mathematically identical to the
chunked form (verified) and is the natural reference for that kernel.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_a3b.tt import prefill_profiler as prof
from models.demos.qwen3_6_a3b.tt.common import as_weight, to_tt
from models.demos.qwen3_6_a3b.tt.rms_norm import TtRMSNormGated
from models.demos.qwen3_6_a3b.tt.ttl_delta import chunk_state_tt, decode_step_tt

# Fused chunked prefill: replace the sequential recurrent scan (O(T) dispatches) with the chunked
# delta-rule — ttnn per-chunk prep batched over heads + the fused tt-lang _chunk_state kernel (one
# launch over all heads per chunk). This is the DEFAULT (5.4-5.7x faster prefill at 40 layers, warm);
# set QWEN36_FUSED_PREFILL=0 to fall back to the sequential scan. Decode (T=1) always uses the scan.
# Intra-chunk inverse (I-L)^-1: numerically-stable recursive block inversion by default (see
# _chunk_prep). QWEN36_DELTA_IPLUSL=1 selects the fast T≈I+L approximation (INACCURATE for this
# model's strong-decay heads -> gibberish; A/B only). The old doubling product is removed (it explodes
# on real L). First fused prefill compiles the ttl kernel once (~1 min, cached on disk thereafter).
_FUSED_PREFILL = os.environ.get("QWEN36_FUSED_PREFILL", "1") == "1"
_DELTA_IPLUSL = os.environ.get("QWEN36_DELTA_IPLUSL") == "1"
# Run the per-chunk prep in bf16 instead of fp32: the prefill is dispatch-bound and the fp32 path adds
# ~124 typecast ops/forward (up to fp32, back to bf16) plus 2x data movement. OFF by default: the
# stable recursive inverse + cumsum/exp are precision-sensitive (bf16 prep regresses 40-layer
# coherence). QWEN36_DELTA_BF16_PREP=1 enables (perf experiments only).
_PREP_DT = ttnn.bfloat16 if os.environ.get("QWEN36_DELTA_BF16_PREP") == "1" else ttnn.float32
_CHUNK = 64  # chunk_size, matches reference torch_chunk_gated_delta_rule default
# Run the per-chunk prep ONCE batched over all chunks (state-independent) instead of Nc sequential
# rounds — see _forward_prefill_chunked. DEFAULT on; QWEN36_DELTA_BATCH_PREP=0 reverts (A/B fallback).
_BATCH_PREP = os.environ.get("QWEN36_DELTA_BATCH_PREP", "1") != "0"


# Keep the tiny gated-delta DECODE intermediates L1-resident instead of round-tripping interleaved
# DRAM: the decode path is many tiny dispatch-bound ops, so on-chip residency cuts the step ~16%
# (measured, PCC-identical). Default on; QWEN36_GDN_L1=0 reverts to interleaved DRAM.
_GDN_L1 = os.environ.get("QWEN36_GDN_L1", "1") != "0"
_MC = ttnn.L1_MEMORY_CONFIG if _GDN_L1 else None

# Fused single-step (T=1) DECODE kernel: collapse the whole per-head gated-delta decode step
# (l2norm(q)*scale + l2norm(k) + gate/beta + rank-1 recurrence + gated RMSNorm) into ONE ttl launch
# over all value heads, instead of ~30 tiny dispatch-bound ttnn ops. The 4 projections (w_qkv/w_z/
# w_ba/w_out) + the conv stay in ttnn. DEFAULT on; QWEN36_GDN_FUSED=0 reverts to the recurrent scan.
# The kernel compiles once per (n_v_heads, n_k_heads, head-dim) shape (~1 min, disk-cached).
_GDN_FUSED = os.environ.get("QWEN36_GDN_FUSED", "1") != "0"
TILE = 32

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
#       numerically correct. The (incoherent) bf16 ttl kernel path stays behind
#       QWEN36_DELTA_STABLE_PREFILL=0 for perf experiments only.
_STABLE_PREFILL = os.environ.get("QWEN36_DELTA_STABLE_PREFILL", "1") != "0"
_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _l2norm_scale_lastdim(x, scale=None, eps=1e-6):
    sq = ttnn.sum(ttnn.multiply(x, x, memory_config=_MC), dim=-1, keepdim=True, memory_config=_MC)
    y = ttnn.multiply(x, ttnn.rsqrt(ttnn.add(sq, eps, memory_config=_MC), memory_config=_MC), memory_config=_MC)
    return ttnn.multiply(y, scale, memory_config=_MC) if scale is not None else y


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
        if _GDN_FUSED:
            V, Dv = self.num_v_heads, self.head_v_dim

            def _uniform(vals):  # torch [V] -> device [V*TILE, TILE] uniform per head
                t = torch.zeros(V * TILE, TILE)
                for h in range(V):
                    t[h * TILE : (h + 1) * TILE, :] = float(vals[h])
                return to_tt(t, mesh_device, dtype=ttnn.bfloat16)

            self._fused_negA = _uniform(-torch.exp(W("A_log").float()).reshape(V))
            self._fused_dtb = _uniform(W("dt_bias").float().reshape(V))
            nw = W("norm").float().reshape(Dv)
            self._fused_nweight = to_tt(
                nw.reshape(1, Dv).expand(TILE, Dv).contiguous(), mesh_device, dtype=ttnn.bfloat16
            )
            self._fused_scratch = None  # (out_buf, Snew_buf), allocated on first decode

    def _conv_silu(self, mixed, conv_state=None):
        """mixed: [T, conv_dim]. Causal depthwise conv (kernel K) + silu. Returns (out[T,conv_dim], new_conv_state[K-1,conv_dim])."""
        T = mixed.shape[0]
        if conv_state is None:
            pad = ttnn.zeros(
                [self.conv_k - 1, self.conv_dim], dtype=mixed.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
            )
        else:
            pad = conv_state
        # L1 only for the tiny T==1 decode concat; prefill (large T) stays in DRAM (an L1 [T+K-1,
        # conv_dim] concat is ~33 MB at T=2048 and overflows L1 — the _MC residency is a decode-only win).
        xpad = ttnn.concat([pad, mixed], dim=0, memory_config=(_MC if T == 1 else None))  # [T+K-1, conv_dim]
        if T == 1:
            # decode: xpad is exactly [K, conv_dim] and out[0] = sum_j xpad[j]*tap[j]. One elementwise
            # multiply against the stacked taps + one row-sum, vs K slices + K multiplies + (K-1) adds
            # (the 4-tap conv was the largest gated-delta decode cost — pure dispatch overhead).
            acc = ttnn.sum(
                ttnn.multiply(xpad, self.conv_taps_stacked, memory_config=_MC), dim=0, keepdim=True, memory_config=_MC
            )  # [1, conv_dim]
        else:
            acc = None
            for j in range(self.conv_k):
                xj = ttnn.slice(xpad, [j, 0], [j + T, self.conv_dim])
                term = ttnn.multiply(xj, self.conv_taps[j])
                acc = term if acc is None else ttnn.add(acc, term)
        new_state = ttnn.slice(xpad, [T, 0], [T + self.conv_k - 1, self.conv_dim])
        return ttnn.silu(acc, memory_config=_MC), new_state

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
        if _DELTA_IPLUSL:
            T = ttnn.add(eye, L)  # T ≈ I+L (fast escape; only valid when L tiny; wrong for real data)
        else:
            # (I-L)^-1 via numerically-stable recursive block inversion (block Gaussian elimination).
            # The doubling product (I+L)(I+L^2)...(I+L^32) is mathematically exact for nilpotent L but
            # numerically EXPLODES on real data: the intermediate L^k have huge entries (large singular
            # values, despite eigenvalues=0) that must telescope but don't in finite precision -> 1e6-1e11
            # blow-up while the true inverse is ~1.0. The recursive block form never forms L^k: it merges
            # h-block inverses pairwise via corner = -D^-1 (M_B) A^-1, staying bounded (max|T|~1.0). All
            # ops are full-CxC (tile-aligned) batched matmuls + fixed mask multiplies; log2(C) levels.
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

    def _chunk_state_ttnn(self, tc, S):
        """Numerically-stable (HiFi4 / fp32) ttnn implementation of one chunk's delta-rule recurrence,
        replacing the bf16 ttl _chunk_state kernel for prefill. Mirrors the kernel math exactly:
          v_new = w − kcd·S ;  out = qg·S + ((q·kᵀ)⊙decay)·v_new ;  Snew = S·glast + kgtᵀ·v_new.
        tc: per-chunk prep terms [1,Vh,*,*] (fp32). S: [1,Vh,Dk,Dv] (fp32). Returns (out [1,Vh,C,Dv],
        Snew [1,Vh,Dk,Dv]), fp32. All matmuls are batched over the head axis at HiFi4 + fp32 accum, so
        the chunked path matches the fp32 reference (the bf16 kernel only reached ~0.996/layer)."""
        ck = _HIFI4
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

    def _forward_prefill_chunked(self, q, k, v, g, beta, pool=None, init_state=None):
        """Chunked delta-rule prefill. q,k,v:[T,Vh,D]; g,beta:[T,Vh] (g=log decay). Returns
        (core [1,Vh,T,Dv], S_final [Vh*Dk,Dv]). Pads T to a multiple of chunk_size and carries the
        per-head recurrent state [Vh*Dk,Dv] (head-major) across chunks.

        init_state (incremental prefill): head-major [Vh*Dk,Dv] starting state from the cache (continue
        a prior context) instead of zeros — the natural cross-chunk carry, just seeded from the cache.
        pool (traced path only): a dict of PRE-ALLOCATED buffers {S0, out[Nc], Snew[Nc]} used instead
        of in-graph ttnn.zeros (which are forbidden during trace capture). Requires T a multiple of
        chunk_size (no padding); build it with build_trace_pool(T)."""
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

        # Numerically-stable ttnn-fp32 chunk-state (eager path only; the traced/pool path keeps the ttl
        # kernel). State carried as [1,Vh,Dk,Dv] in fp32 for accuracy (see _chunk_state_ttnn).
        use_stable = _STABLE_PREFILL and pool is None
        if use_stable:
            S = ttnn.typecast(ttnn.reshape(S, [1, Vh, Dk, Dv]), _PREP_DT)

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
                    out4, S = self._chunk_state_ttnn(tc, S)
                outs.append(out4)
                continue
            if pool is not None:  # kernel-written output buffers: pre-allocated (no in-graph zeros)
                out, Snew = pool["out"][ci], pool["Snew"][ci]
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
        if use_stable:  # back to bf16 head-major state + bf16 core (the path's interface)
            S = ttnn.typecast(ttnn.reshape(S, [Vh * Dk, Dv]), ttnn.bfloat16)
            if core.dtype != ttnn.bfloat16:
                core = ttnn.typecast(core, ttnn.bfloat16)
        return core, S

    def build_trace_pool(self, T):
        """Pre-allocate the buffers _forward_prefill_chunked would otherwise create with ttnn.zeros
        (forbidden inside trace capture). Shared across all gated-delta layers (sequential -> reusable);
        S0 stays zeros (read-only initial state); out/Snew are kernel-written. T must be a chunk multiple."""
        C = _CHUNK
        Vh, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
        Nc = T // C
        z = lambda shp: ttnn.zeros(shp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        # "valid" is a per-request [1,1,T] mask (1 for real tokens, 0 for right-padding) that the bucketed
        # traced path multiplies into g/beta so pad positions contribute nothing to the recurrent state —
        # reproducing the eager zero-padding for a fixed-shape (bucket-padded) input. Written per request.
        return {
            "S0": z([Vh * Dk, Dv]),
            "out": [z([Vh * C, Dv]) for _ in range(Nc)],
            "Snew": [z([Vh * Dk, Dv]) for _ in range(Nc)],
            "valid": ttnn.zeros([1, 1, T], dtype=_PREP_DT, layout=ttnn.TILE_LAYOUT, device=self.mesh_device),
        }

    def forward(self, x, cache=None, pool=None, init_state=None):
        """x: [1,1,T,hidden]. If cache (dict with conv_state/recurrent_state) given, continues from it.
        Returns [1,1,T,hidden]; updates cache in place when provided. pool: pre-allocated trace buffers
        (traced prefill only). init_state: head-major [Vh*Dk,Dv] starting recurrent state for incremental
        prefill (continue a prior context); conv likewise continues from the cached conv_state."""
        T = x.shape[2]
        hidden = x.shape[3]
        x2 = ttnn.reshape(x, [T, hidden])
        # Decode (T==1) keeps the tiny intermediates L1-resident (QWEN36_GDN_L1): measured to cut the
        # gated-delta step, since the decode path is many tiny dispatch-bound ops, not DRAM-bandwidth.
        # Prefill (T>1) stays on the default (interleaved DRAM) to avoid L1 overflow at large T.
        mc = _MC if T == 1 else None

        mixed = ttnn.linear(x2, self.w_qkv, memory_config=mc)  # [T, conv_dim]
        conv_state = cache.get("conv_state") if cache else None
        mixed, new_conv_state = self._conv_silu(mixed, conv_state)
        if cache is not None:
            if conv_state is not None:
                ttnn.copy(new_conv_state, conv_state)  # in-place: keep persistent buffer (traceable)
            else:
                cache["conv_state"] = new_conv_state

        q = ttnn.slice(mixed, [0, 0], [T, self.key_dim], memory_config=mc)
        k = ttnn.slice(mixed, [0, self.key_dim], [T, 2 * self.key_dim], memory_config=mc)
        v = ttnn.slice(mixed, [0, 2 * self.key_dim], [T, self.conv_dim], memory_config=mc)
        z = ttnn.linear(x2, self.w_z, memory_config=mc)  # [T, value_dim]

        # beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias). b|a from one fused matmul.
        V = self.num_v_heads
        ba = ttnn.linear(x2, self.w_ba, memory_config=mc)  # [T, 2V]

        # Fused single-step decode: one ttl launch for the whole step (prep + recurrence + gated-norm).
        # q/k [1,key_dim], v/z [1,value_dim] feed the kernel directly (heads along columns); raw a/b are
        # expanded to per-head uniform tiles; state stays head-major in the persistent cache.
        if _GDN_FUSED and T == 1 and cache is not None:
            return self._forward_decode_fused(q, k, v, z, ba, cache, hidden)

        beta = ttnn.sigmoid(ttnn.slice(ba, [0, 0], [T, V], memory_config=mc), memory_config=mc)  # [T, V]
        a = ttnn.slice(ba, [0, V], [T, 2 * V], memory_config=mc)  # [T, V]
        g = ttnn.multiply(self.neg_expA, ttnn.softplus(ttnn.add(a, self.dt_bias)), memory_config=mc)  # [T, V]
        g_exp = ttnn.exp(g, memory_config=mc)  # decay per step, [T, V]

        # reshape to [T, Hk, Dk]; l2norm over Dk; scale q; repeat_interleave k,q heads to V
        Vh, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
        q = ttnn.reshape(q, [T, self.num_k_heads, Dk])
        k = ttnn.reshape(k, [T, self.num_k_heads, Dk])
        v = ttnn.reshape(v, [T, Vh, Dv])
        q = _l2norm_scale_lastdim(q, scale=self.qk_scale)
        k = _l2norm_scale_lastdim(k)
        if self.n_rep > 1:
            q = ttnn.repeat_interleave(q, self.n_rep, dim=1)
            k = ttnn.repeat_interleave(k, self.n_rep, dim=1)

        if _FUSED_PREFILL and T > 1:
            # --- fused chunked prefill (parallel over the sequence) ---
            core, S_final = self._forward_prefill_chunked(q, k, v, g, beta, pool=pool, init_state=init_state)
            if cache is not None:
                state_buf = cache.get("recurrent_state")
                S_final = ttnn.reshape(S_final, [1, Vh, Dk, Dv])  # decode reads state as [1,Vh,Dk,Dv]
                if state_buf is not None:
                    ttnn.copy(S_final, state_buf)
                else:
                    cache["recurrent_state"] = S_final
        else:
            # --- on-device recurrent scan (decode path, or non-fused prefill) ---
            state_buf = cache.get("recurrent_state") if cache else None
            state = state_buf
            if state is None:
                state = ttnn.zeros(
                    [1, Vh, Dk, Dv], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
                )
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
            core = ttnn.concat(outs, dim=2)  # [1, V, T, Dv]

        core = ttnn.transpose(core, 1, 2, memory_config=mc)  # [1, T, V, Dv]
        core = ttnn.reshape(core, [1, 1, T * Vh, Dv])
        z_r = ttnn.reshape(z, [1, 1, T * Vh, Dv])
        core = self.norm.forward(core, z_r)
        core = ttnn.reshape(core, [T, self.value_dim])
        y = ttnn.linear(core, self.w_out, memory_config=mc)
        return ttnn.reshape(y, [1, 1, T, hidden])

    def _forward_decode_fused(self, q, k, v, z, ba, cache, hidden):
        """Fused single-step (T=1) decode: one ttl launch (decode_step_tt) does l2norm(q)*scale +
        l2norm(k) + gate/beta + the rank-1 recurrence + the gated RMSNorm, replacing the ~30-op scan
        path above. q/k:[1,key_dim], v/z:[1,value_dim] (heads along columns, fed directly). State is
        read from / written back to the PERSISTENT recurrent_state cache (head-major view), so the
        write is in-place and trace-safe. Returns [1,1,1,hidden]."""
        V, Dk, Dv = self.num_v_heads, self.head_k_dim, self.head_v_dim

        # raw a|b -> per-head uniform [V*TILE, TILE] tiles. transpose [1,2V]->[2V,1] moves heads to
        # rows; repeat fills each head's tile; slice splits b|a (one expansion for both, then split).
        bat = ttnn.transpose(ba, 0, 1)  # [2V, 1]
        bat = ttnn.repeat(ttnn.reshape(bat, [2 * V, 1, 1]), ttnn.Shape([1, TILE, TILE]))  # [2V, TILE, TILE]
        bat = ttnn.reshape(bat, [2 * V * TILE, TILE])
        braw_tile = ttnn.slice(bat, [0, 0], [V * TILE, TILE])  # b is the first V
        araw_tile = ttnn.slice(bat, [V * TILE, 0], [2 * V * TILE, TILE])  # a is the second V

        state_buf = cache["recurrent_state"]  # persistent [1,V,Dk,Dv]
        S_in = ttnn.reshape(state_buf, [V * Dk, Dv])  # head-major view for the kernel
        if self._fused_scratch is None:  # persistent scratch (allocated once; reused every step/trace)
            self._fused_scratch = (
                ttnn.zeros([1, self.value_dim], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device),
                ttnn.zeros([V * Dk, Dv], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device),
            )
        out_buf, snew_buf = self._fused_scratch

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
        )
        ttnn.copy(ttnn.reshape(snew_buf, [1, V, Dk, Dv]), state_buf)  # in-place state update (trace-safe)
        y = ttnn.linear(out_buf, self.w_out, memory_config=_MC)  # [1, hidden]
        return ttnn.reshape(y, [1, 1, 1, hidden])
