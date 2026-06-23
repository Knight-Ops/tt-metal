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
from models.demos.qwen3_6_a3b.tt.ttl_delta import chunk_state_tt

# Fused chunked prefill: replace the sequential recurrent scan (O(T) dispatches) with the chunked
# delta-rule — ttnn per-chunk prep batched over heads + the fused tt-lang _chunk_state kernel (one
# launch over all heads per chunk). This is the DEFAULT (5.4-5.7x faster prefill at 40 layers, warm);
# set QWEN36_FUSED_PREFILL=0 to fall back to the sequential scan. Decode (T=1) always uses the scan.
# Inverse: T≈I+L by default (fastest, realistic L is tiny -> matches the fp32 doubling product within
# PCC); set QWEN36_DELTA_IPLUSL=0 for the fp32 doubling-product inverse. First fused prefill compiles
# the ttl kernel once (~1 min, cached on disk thereafter).
_FUSED_PREFILL = os.environ.get("QWEN36_FUSED_PREFILL", "1") == "1"
_DELTA_IPLUSL = os.environ.get("QWEN36_DELTA_IPLUSL", "1") == "1"
# Run the per-chunk prep in bf16 instead of fp32: the prefill is dispatch-bound and the fp32 path adds
# ~124 typecast ops/forward (up to fp32, back to bf16) plus 2x data movement. The kernel already runs
# bf16 and the I+L inverse (default) needs no fp32. OFF by default until a 40-layer coherence check
# confirms cumsum/exp precision; QWEN36_DELTA_BF16_PREP=1 enables. (Keep fp32 if using the opt-in
# QWEN36_DELTA_IPLUSL=0 doubling-product inverse, which is precision-sensitive.)
_PREP_DT = ttnn.bfloat16 if os.environ.get("QWEN36_DELTA_BF16_PREP") == "1" else ttnn.float32
_CHUNK = 64  # chunk_size, matches reference torch_chunk_gated_delta_rule default


# Keep the tiny gated-delta DECODE intermediates L1-resident instead of round-tripping interleaved
# DRAM: the decode path is many tiny dispatch-bound ops, so on-chip residency cuts the step ~16%
# (measured, PCC-identical). Default on; QWEN36_GDN_L1=0 reverts to interleaved DRAM.
_GDN_L1 = os.environ.get("QWEN36_GDN_L1", "1") != "0"
_MC = ttnn.L1_MEMORY_CONFIG if _GDN_L1 else None


def _l2norm_scale_lastdim(x, scale=None, eps=1e-6):
    sq = ttnn.sum(ttnn.multiply(x, x, memory_config=_MC), dim=-1, keepdim=True, memory_config=_MC)
    y = ttnn.multiply(x, ttnn.rsqrt(ttnn.add(sq, eps, memory_config=_MC), memory_config=_MC), memory_config=_MC)
    return ttnn.multiply(y, scale, memory_config=_MC) if scale is not None else y


class TtGatedDeltaNet(LightweightModule):
    def __init__(self, mesh_device, weights, cfg, dtype=ttnn.bfloat8_b):
        super().__init__()
        self.mesh_device = mesh_device
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

        self.w_qkv = as_weight(weights["in_proj_qkv"], mesh_device, dtype=dtype)
        self.w_z = as_weight(weights["in_proj_z"], mesh_device, dtype=dtype)
        # in_proj_b and in_proj_a both map hidden -> V; fuse into one [hidden, 2V] matmul (one launch
        # instead of two), split the output b|a in forward. Both weights are [V, hidden] (nn.Linear).
        self.w_ba = as_weight(torch.cat([weights["in_proj_b"], weights["in_proj_a"]], dim=0), mesh_device, dtype=dtype)
        self.w_out = as_weight(weights["out_proj"], mesh_device, dtype=dtype)
        self.norm = TtRMSNormGated(mesh_device, weights["norm"], self.eps)

        cw = weights["conv1d"].reshape(self.conv_dim, self.conv_k)
        self.conv_taps = [
            to_tt(cw[:, j].reshape(1, self.conv_dim), mesh_device, dtype=ttnn.bfloat16) for j in range(self.conv_k)
        ]
        # taps stacked [conv_k, conv_dim] for the fused T==1 (decode) conv: out = sum_j xpad[j]*tap[j]
        # is then one multiply + one row-sum instead of K slices + K multiplies + (K-1) adds.
        self.conv_taps_stacked = to_tt(cw.transpose(0, 1).contiguous(), mesh_device, dtype=ttnn.bfloat16)

        # A_log, dt_bias as [1, V] device tensors for g = -exp(A_log)*softplus(a+dt_bias)
        self.neg_expA = to_tt(
            -torch.exp(weights["A_log"].float()).reshape(1, self.num_v_heads), mesh_device, dtype=ttnn.bfloat16
        )
        self.dt_bias = to_tt(weights["dt_bias"].float().reshape(1, self.num_v_heads), mesh_device, dtype=ttnn.bfloat16)

    def _conv_silu(self, mixed, conv_state=None):
        """mixed: [T, conv_dim]. Causal depthwise conv (kernel K) + silu. Returns (out[T,conv_dim], new_conv_state[K-1,conv_dim])."""
        T = mixed.shape[0]
        if conv_state is None:
            pad = ttnn.zeros(
                [self.conv_k - 1, self.conv_dim], dtype=mixed.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
            )
        else:
            pad = conv_state
        xpad = ttnn.concat([pad, mixed], dim=0, memory_config=_MC)  # [T+K-1, conv_dim]
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

        self._chunk_masks = dict(
            tril_incl=up(torch.tril(torch.ones(C, C))),  # cumsum/decay causal incl-diag
            strict_lower=up(torch.tril(torch.ones(C, C), diagonal=-1)),  # L strictly-lower mask
            eye=up(torch.eye(C)),  # I for (I-L)^-1
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
        L = ttnn.multiply(ttnn.multiply(ttnn.multiply(ttnn.matmul(kbeta, kT), decay), strict_lower), -1.0)
        if _DELTA_IPLUSL:
            T = ttnn.add(eye, L)  # T ≈ I+L (L tiny)
        else:
            acc = ttnn.add(eye, L)
            p = L  # fp32 doubling product
            for _ in range(5):  # C=64 -> L^64=0, 5 squarings
                p = ttnn.matmul(p, p)
                acc = ttnn.matmul(acc, ttnn.add(eye, p))
            T = acc
        qg = ttnn.multiply(q, egc_col)
        w = ttnn.matmul(T, vbeta)
        kcd = ttnn.matmul(T, ttnn.multiply(kbeta, egc_col))
        glast_col = ttnn.reshape(ttnn.slice(g_cum, [0, 0, C - 1], [1, Vh, C]), [1, Vh, 1, 1])
        kg = ttnn.multiply(k, ttnn.exp(ttnn.subtract(glast_col, gc_row)))  # per-row decay
        kgt = ttnn.transpose(kg, -2, -1)  # [1,Vh,D,C]
        glast = ttnn.repeat(ttnn.exp(glast_col), ttnn.Shape([1, 1, D, D]))  # [1,Vh,D,D] (kernel: S*glast)
        return dict(q=q, kt=kT, w=w, kcd=kcd, decay=decay, qg=qg, kgt=kgt, glast=glast)

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
        outs = []
        for ci in range(Tp // C):
            s0 = ci * C
            qc = ttnn.slice(qb, [0, 0, s0, 0], [1, Vh, s0 + C, Dk])
            kc = ttnn.slice(kb, [0, 0, s0, 0], [1, Vh, s0 + C, Dk])
            vc = ttnn.slice(vb, [0, 0, s0, 0], [1, Vh, s0 + C, Dv])
            gc = ttnn.slice(gb, [0, 0, s0], [1, Vh, s0 + C])
            bc = ttnn.slice(betab, [0, 0, s0], [1, Vh, s0 + C])
            with prof.phase(self.mesh_device, "delta.prep"):
                terms = self._chunk_prep(qc, kc, vc, bc, gc, masks)
            if pool is not None:  # kernel-written output buffers: pre-allocated (no in-graph zeros)
                out, Snew = pool["out"][ci], pool["Snew"][ci]
            else:
                out = ttnn.zeros([Vh * C, Dv], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
                Snew = ttnn.zeros([Vh * Dk, Dv], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
            with prof.phase(self.mesh_device, "delta.kernel"):
                chunk_state_tt(
                    hm(terms["q"]),
                    hm(terms["kt"]),
                    hm(terms["w"]),
                    hm(terms["kcd"]),
                    hm(terms["decay"]),
                    hm(terms["qg"]),
                    hm(terms["kgt"]),
                    hm(terms["glast"]),
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
