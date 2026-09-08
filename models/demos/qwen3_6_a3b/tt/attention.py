# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-nn full (softmax) attention for Qwen3.6-35B-A3B.

Matches the HF ``Qwen3_5MoeAttention``:
  - ``q_proj`` outputs ``n_heads * head_dim * 2``; per head it splits into a query and a gate.
    We pre-split the weight at load time into ``wq`` and ``wgate`` so the on-device path is clean.
  - per-head RMSNorm on q and k (head_dim), partial RoPE (rotary_dim = head_dim * 0.25),
  - GQA softmax attention (SDPA), output gated by ``sigmoid(gate)``, then ``o_proj``.

Implements prefill (single-shot, chunked/incremental) and cached decode: a fixed-shape KV cache
[1, n_kv, max_seq, head_dim] written in place via paged_update_cache, read by sdpa_decode. See the
forward_prefill / forward_prefill_incremental / forward_decode methods.
"""
from __future__ import annotations

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_a3b.tt import signpost as sp
from models.demos.qwen3_6_a3b.tt.common import as_weight, build_dram_shard, to_tt, wide1d_enabled, wide_1d_decode_pc
from models.demos.qwen3_6_a3b.tt.rms_norm import TtRMSNorm

# DRAM-shard the DECODE output projection (o_proj / wo): K=n_heads*head_dim, N=hidden — the K-heavy
# narrow-N shape that DRAM sharding speeds up ~2.7x at T=1 (measured; incl. reshards). Decode-only.
# Default on; QWEN36_ATTN_DRAM_SHARD=0 reverts. See FUTURE_OPTIMIZATIONS.md.
_ATTN_DRAM_SHARD = os.environ.get("QWEN36_ATTN_DRAM_SHARD", "1") != "0"
# DRAM-shard the DECODE input projections wq/wgate (K=hidden, N=n_heads*head_dim). Measured 1.71-1.73x
# per op at T=1 (bench_matmul_layout.py). Decode-only; prefill keeps interleaved. QWEN36_ATTN_DRAM_SHARD_IN=0 reverts.
_ATTN_DRAM_SHARD_IN = os.environ.get("QWEN36_ATTN_DRAM_SHARD_IN", "1") != "0"
# Speculative VERIFY: emit all K rows from ONE paged sdpa_decode over a K-row page table instead of
# looping one sdpa_decode per row. See Attention._verify_page_table for why the paged op can do what
# the flat one cannot, and forward_verify for the measured numbers. QWEN36_VERIFY_BATCH_SDPA=0 reverts
# to the row loop (bit-identical KV, PCC-equal output -- the loop is the reference).
_VERIFY_BATCH_SDPA = os.environ.get("QWEN36_VERIFY_BATCH_SDPA", "1") != "0"
# Largest k_chunk_size the CHUNKED-prefill SDPA may use. The shipped value pins it at 64, justified as
# "bounds L1 at long context (head_dim=256 is large)" -- a worst-case choice that was never A/B'd,
# because the whole prefill profile in PREFILL.md was taken at T<=1024, where forward() goes SINGLE-SHOT
# and never touches this kernel at all. Muse Glimmer runs the same kernel with a 512/256/128/64 ladder at
# head_dim=128 -- the same per-chunk L1 as 256 would be here -- which is what prompted testing it.
# `forward_prefill_incremental` takes the largest candidate <= this that divides the chunk offset.
_PREFILL_KCHUNK_MAX = int(os.environ.get("QWEN36_PREFILL_KCHUNK", "64"))
# Verify SDPA core grid. `sdpa_decode_pc` hardcodes CoreCoord(8,8) -- 64 of this card's 110 usable
# Tensix. Its k_chunk_size=128 is deliberate and measured (64 is up to 1.38x WORSE at 128K bf8); the
# GRID was never examined, and widening it to the device's full grid is worth 0.89-0.95x of the verify
# SDPA once there is enough work to fill it. MEASURED (tests/probe_packed_verify.py --pcsweep, us per
# attention layer, bf16, 110 cores vs 64):
#     K=3   pos 1024 1.03x  2048 1.07x  4096 0.99x  8192 0.95x  32768 0.91x
#     K=16  pos 1024 0.95x  2048 0.93x  4096 0.91x  8192 0.90x  32768 0.89x
# So it is not a free win: below the crossover the wider grid LOSES (each core's slice gets too small to
# amortise). Work per call scales as K*cur_pos and a single threshold on that product predicts every
# measured point, including the K=1 decode rows. Deliberately NOT applied to forward_decode: there it is
# only ~1% at 128K and negative at typical lengths, and it perturbs plain-decode numerics (a different
# grid changes the flash reduction order) -- that needs an MMLU baseline first, like `gate_w` in
# FUTURE_OPTIMIZATIONS.
#
# DEFAULT OFF, because the end-to-end number does not justify the plumbing. The per-op win is real and
# the cost model predicted it to within 5%, but the SDPA is a small slice of the verify (the projections,
# MoE and GDN dominate), so on the 40-layer model at an 8192-token prompt it is only:
#     K=3   42.709 -> 42.617 ms  (-0.092, -0.22%)   <- the shipping gamma=2
#     K=16 126.807 -> 126.041 ms (-0.766, -0.60%)
# extrapolating to ~1.2% (K=3) / ~2.4% (K=16) at 32K. FUTURE_OPTIMIZATIONS omits `lm_head` from the
# wide-1D default on exactly this reasoning -- "the default does not spend an accumulation-order change
# for nothing" -- and there is no 40-layer numerics gate here, only small-config PCC. Set
# QWEN36_VERIFY_SDPA_WIDE=1 for a long-context speculative workload where ~1-2% is worth having.
_VERIFY_SDPA_WIDE = os.environ.get("QWEN36_VERIFY_SDPA_WIDE", "0") != "0"
_VERIFY_SDPA_WIDE_WORK = int(os.environ.get("QWEN36_VERIFY_SDPA_WIDE_WORK", "12288"))


def _prefill_k_chunk(chunk_start):
    """Largest k_chunk_size <= _PREFILL_KCHUNK_MAX that the cache offset allows.

    chunked_scaled_dot_product_attention wants chunk_start aligned to the k tile, so the ladder is not
    cosmetic: with QWEN36_PREFILL_CHUNK=512 every full chunk offset divides all four candidates, but a
    RAGGED final block leaves chunk_start a multiple of only 128 (see prefill_long), and a fixed 256
    would then be illegal. Same structure as Muse Glimmer's `_key_chunk_size`."""
    for candidate in (512, 256, 128, 64):
        if candidate <= _PREFILL_KCHUNK_MAX and chunk_start % candidate == 0:
            return candidate
    return 64


def precompute_rope(seq_len, rotary_dim, theta, device, dtype=ttnn.bfloat16, start_pos=0):
    """Return (cos, sin) tt tensors of shape [1, 1, seq_len, rotary_dim] for the default RoPE."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    pos = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float)
    freqs = torch.outer(pos, inv_freq)  # (S, rotary_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (S, rotary_dim)
    cos = emb.cos()[None, None]
    sin = emb.sin()[None, None]
    return to_tt(cos, device, dtype=dtype), to_tt(sin, device, dtype=dtype)


def _rotate_half(x, rotary_dim):
    half = rotary_dim // 2
    x1 = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], half])
    x2 = ttnn.slice(x, [0, 0, 0, half], [x.shape[0], x.shape[1], x.shape[2], rotary_dim])
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def apply_rope(x, cos, sin, rotary_dim):
    """x: [1, n_heads, S, head_dim]. Apply RoPE to the first rotary_dim dims, pass the rest through."""
    head_dim = x.shape[-1]
    x_rot = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], rotary_dim])
    emb = ttnn.add(ttnn.multiply(x_rot, cos), ttnn.multiply(_rotate_half(x_rot, rotary_dim), sin))
    if rotary_dim == head_dim:
        return emb
    x_pass = ttnn.slice(x, [0, 0, 0, rotary_dim], [x.shape[0], x.shape[1], x.shape[2], head_dim])
    return ttnn.concat([emb, x_pass], dim=-1)


class TtAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        weights,
        n_heads,
        n_kv_heads,
        head_dim,
        rotary_dim,
        eps,
        dtype=ttnn.bfloat8_b,
        cache_path=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.scale = head_dim**-0.5
        # Incremental prefill hands chunked-SDPA the KV offset. Passing it as an int bakes it into
        # the program, so every new offset is a new program -- and P grows forever in a chat, so that
        # cost never amortises (measured: a ragged block at a NEW offset took 1.63 s in the serving
        # loop against 0.27 s at a known one, which is what made a finer snapshot grain a
        # regression). The tensor form reads the offset at runtime instead: one program for all
        # offsets, bit-identical output (hangdbg/flex.py: +1 program then 0, and 0.3 ms per call).
        # Allocated HERE, at build time, because anything allocated lazily on first use is allocated
        # under a live decode trace and can be corrupted by a replay -- see
        # TtModel.prewarm_prefill_shapes.
        self._csi = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
        )
        self._csi_host = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        cn = (lambda role: f"{cache_path}/{role}") if cache_path else (lambda role: None)

        # weights[*] may be lazy thunks (production loader) or plain tensors (module tests). W()
        # materializes either; for cached weights it is only invoked on a cache miss.
        def W(k):
            v = weights[k]
            return v() if callable(v) else v

        # Split the fused q_proj weight (rows: per head [query(head_dim) | gate(head_dim)]) inside
        # thunks so the read + split only run on a cache miss.
        def _wq():
            wqg = W("q_proj").reshape(n_heads, 2 * head_dim, -1)
            return wqg[:, :head_dim, :].reshape(n_heads * head_dim, -1).contiguous()

        def _wgate():
            wqg = W("q_proj").reshape(n_heads, 2 * head_dim, -1)
            return wqg[:, head_dim:, :].reshape(n_heads * head_dim, -1).contiguous()

        self.wq = as_weight(_wq, mesh_device, dtype=dtype, cache_file_name=cn("wq"))
        self.wgate = as_weight(_wgate, mesh_device, dtype=dtype, cache_file_name=cn("wgate"))
        self.wk = as_weight(weights["k_proj"], mesh_device, dtype=dtype, cache_file_name=cn("wk"))
        self.wv = as_weight(weights["v_proj"], mesh_device, dtype=dtype, cache_file_name=cn("wv"))
        self.wo = as_weight(weights["o_proj"], mesh_device, dtype=dtype, cache_file_name=cn("wo"))
        # DRAM-sharded decode output projection (built from the interleaved wo; prefill keeps interleaved).
        self._wo_dram = None
        if _ATTN_DRAM_SHARD:
            self._wo_dram, self._wo_amc, self._wo_omc, self._wo_pc = build_dram_shard(
                self.wo, self.wo.shape[-2], self.wo.shape[-1]
            )
        # DRAM-sharded decode input projections wq/wgate (identical [hidden, n_heads*head_dim] shape ->
        # one shared act/out/program config; two weight tensors). Prefill keeps the interleaved wq/wgate.
        self._wq_dram = None
        if _ATTN_DRAM_SHARD_IN:
            Kq, Nq = self.wq.shape[-2], self.wq.shape[-1]
            self._wq_dram, self._wqg_amc, self._wqg_omc, self._wqg_pc = build_dram_shard(self.wq, Kq, Nq)
            self._wgate_dram, _, _, _ = build_dram_shard(self.wgate, Kq, Nq)

        # wk/wv were skipped by the DRAM-shard work as "too small" (FUTURE_OPTIMIZATIONS Lever 1b),
        # which left them on the ttnn AUTO config -- and that is the worst-tuned matmul in the block.
        # Measured (tests/bench_matmul_layout.py, [2048, 512] bf8, traced us/call):
        #   AUTO 34.4  ->  DRAM-shard incl. reshards 17.8  ->  wide 1D mcast 12.7  at (8 cores, bw=16)
        # i.e. 2.7x, and the wide-1D form needs no reshard and no duplicate weight copy. Decode only
        # (per_core_M=1); prefill keeps the heuristic.
        self._pc_kv = (
            wide_1d_decode_pc(32, self.wk.shape[-2], self.wk.shape[-1], 8, in0_block_w=16)
            if wide1d_enabled("wk")
            else None
        )
        self.q_norm = TtRMSNorm(mesh_device, W("q_norm"), eps, add_unit_offset=True)
        self.k_norm = TtRMSNorm(mesh_device, W("k_norm"), eps, add_unit_offset=True)
        # Bounded K/V chunking so flash-decode L1 use stays within budget at long context
        # (default config overflows L1 past ~256 cache len). k_chunk_size=128 fits comfortably.
        self.sdpa_decode_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=128,
        )
        self._verify_pts = {}  # K -> [K,1] int32 page table; see _verify_page_table
        self._verify_wide_pc = None  # built on demand; see _verify_sdpa_pc
        # paged_update_cache (decode KV write) wants the new K/V height-sharded on B(=1) cores with
        # the kv-heads dim padded to TILE_HEIGHT. Precompute that 1-core height-sharded mem config.
        self._kv_update_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                [ttnn.TILE_SIZE, head_dim],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def _qkv(self, x, cos, sin):
        S = x.shape[2]
        with sp.region("attn.qkv"):
            if S == 1 and self._wq_dram is not None:
                # Decode: DRAM-sharded wq/wgate. Reshard x once (shared K), run both, reshard outputs
                # back to interleaved DRAM and restore the [1,1,1,N] contract the interleaved path gives.
                N = self.n_heads * self.head_dim
                x2 = ttnn.to_memory_config(ttnn.reshape(x, [1, self.wq.shape[-2]]), self._wqg_amc)
                q = ttnn.linear(x2, self._wq_dram, program_config=self._wqg_pc, memory_config=self._wqg_omc)
                gate = ttnn.linear(x2, self._wgate_dram, program_config=self._wqg_pc, memory_config=self._wqg_omc)
                q = ttnn.reshape(ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG), [1, 1, 1, N])
                gate = ttnn.reshape(ttnn.to_memory_config(gate, ttnn.DRAM_MEMORY_CONFIG), [1, 1, 1, N])
            else:
                q = ttnn.linear(x, self.wq)
                gate = ttnn.linear(x, self.wgate)
            # S <= 32: one tile row, so the K-row verify path uses the same config as decode (see the
            # note in moe.py _shared_and_router -- keeping the two paths on one config matters for
            # spec-vs-plain token agreement, not just for speed).
            kv_pc = self._pc_kv if (self._pc_kv is not None and S <= 32) else None
            k = ttnn.linear(x, self.wk, program_config=kv_pc)
            v = ttnn.linear(x, self.wv, program_config=kv_pc)
        with sp.region("attn.qk_norm"):
            q = self.q_norm.forward(ttnn.reshape(q, [1, S, self.n_heads, self.head_dim]))
            k = self.k_norm.forward(ttnn.reshape(k, [1, S, self.n_kv_heads, self.head_dim]))
            v = ttnn.reshape(v, [1, S, self.n_kv_heads, self.head_dim])
        with sp.region("attn.rope"):
            q = apply_rope(ttnn.transpose(q, 1, 2), cos, sin, self.rotary_dim)  # [1, nh, S, hd]
            k = apply_rope(ttnn.transpose(k, 1, 2), cos, sin, self.rotary_dim)  # [1, n_kv, S, hd]
            v = ttnn.transpose(v, 1, 2)  # [1, n_kv, S, hd]
        return q, k, v, gate

    @staticmethod
    def _to_cache_dtype(t, cache):
        """Cast a K/V tensor to the cache's dtype before a *fill* (prefill) write.

        The projections emit bf16 while the cache may be BFP8/BFP4 (ModelArgs.kv_cache_dtype,
        QWEN36_KV_DTYPE), so the fill path needs an explicit cast: `ttnn.fill_cache` hard-requires a
        match (`update_cache_device_operation.cpp:72: input_tensor.dtype() == cache_tensor.dtype()`),
        and while `paged_fill_cache`'s check is a permissive disjunction (:34-35) its program factory
        derives the CB format from the INPUT (`paged_fill_cache_program_factory.cpp:84`), so matching
        the two is the only unambiguously safe call. A no-op at the bf16 default; ~1 MB at S=1024.

        Deliberately NOT applied on the decode path -- see the note in _kv_update_input."""
        return t if t.dtype == cache.dtype else ttnn.typecast(t, cache.dtype)

    def _out(self, attn, gate, S):
        with sp.region("attn.out"):
            attn = ttnn.transpose(attn, 1, 2)  # [1, S, nh, hd]
            attn = ttnn.reshape(attn, [1, 1, S, self.n_heads * self.head_dim])
            attn = ttnn.multiply(attn, ttnn.sigmoid(gate))
            return ttnn.linear(attn, self.wo)

    def forward(self, x, cos, sin):
        """No-cache causal prefill (used by unit tests). x: [1,1,S,hidden]."""
        S = x.shape[2]
        q, k, v, gate = self._qkv(x, cos, sin)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return self._out(attn, gate, S)

    def forward_prefill(self, x, cos, sin, kv_cache, fill_page_table=None):
        """Causal prefill that also fills a fixed-shape KV cache (kv_cache=[k_cache,v_cache])."""
        S = x.shape[2]
        q, k, v, gate = self._qkv(x, cos, sin)
        with sp.region("attn.kv_write"):
            k_w = self._to_cache_dtype(k, kv_cache[0])
            v_w = self._to_cache_dtype(v, kv_cache[1])
            if fill_page_table is None:
                ttnn.fill_cache(kv_cache[0], k_w, 0)
                ttnn.fill_cache(kv_cache[1], v_w, 0)
            else:
                # Paged: one call per user -- the kernel reads batch_idx_ptr[0] for all positions, so a
                # batched prefill must loop (tt_transformers attention.py:1022). B==1 here.
                ttnn.experimental.paged_fill_cache(kv_cache[0], k_w, fill_page_table, batch_idx=0)
                ttnn.experimental.paged_fill_cache(kv_cache[1], v_w, fill_page_table, batch_idx=0)
        with sp.region("attn.sdpa"):
            attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return self._out(attn, gate, S)

    def forward_prefill_incremental(
        self, x, cos, sin, kv_cache, page_table, chunk_start, q_chunk=None, fill_page_table=None
    ):
        """Incremental prefill: ingest the new tokens (x: [1,1,M,hidden]) at absolute offset
        chunk_start (P) and attend over the ACCUMULATED KV cache [0:P+M]. cos/sin are RoPE at offset P.
        The flat cache [1,n_kv,max_seq,head_dim] is used directly as a single-block paged cache
        (max_blocks=1, block_s=max_seq) with a trivial page_table.
        Eager (int chunk_start); the traced variant needs paged_fill_cache + chunk_start_idx_tensor.

        chunked-SDPA requires chunk_start (P) to be a multiple of q_chunk_size. For full chunks
        q_chunk_size defaults to M (P is a multiple of the chunk size). For a RAGGED final block the
        caller passes q_chunk (=128): P is always a multiple of the chunk (=> of 128) and M is padded
        to a multiple of 128, so both alignment constraints hold."""
        M = x.shape[2]
        qc = q_chunk or M
        q, k, v, gate = self._qkv(x, cos, sin)
        with sp.region("attn.kv_write"):
            k_w = self._to_cache_dtype(k, kv_cache[0])
            v_w = self._to_cache_dtype(v, kv_cache[1])
            if fill_page_table is None:
                ttnn.fill_cache(kv_cache[0], k_w, 0, update_idx=chunk_start)  # new K/V at rows P..P+M-1
                ttnn.fill_cache(kv_cache[1], v_w, 0, update_idx=chunk_start)
            else:
                # Paged: `chunk_start` is expressed by the SLICE of the page table handed in, so this
                # writes the new block(s) rather than seeking within a flat tensor. The caller passes the
                # sub-table covering logical blocks [chunk_start/block, ...) -- see TtModel._kv_pager.
                ttnn.experimental.paged_fill_cache(kv_cache[0], k_w, fill_page_table, batch_idx=0)
                ttnn.experimental.paged_fill_cache(kv_cache[1], v_w, fill_page_table, batch_idx=0)
        # q_chunk_size=qc (M for full chunks; 128 for ragged). k_chunk_size defaults to 64, which bounds
        # chunked-SDPA L1 at long context (head_dim=256 is large) and lets the offset P be a multiple of
        # 64 rather than 128; QWEN36_PREFILL_KCHUNK raises the ceiling -- see _prefill_k_chunk.
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=qc,
            k_chunk_size=_prefill_k_chunk(chunk_start),
        )
        with sp.region("attn.sdpa"):
            # Runtime offset, not a compile-time one: see self._csi in __init__.
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor([chunk_start], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                ),
                self._csi,
            )
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                kv_cache[0],
                kv_cache[1],
                page_table,
                chunk_start_idx_tensor=self._csi,
                scale=self.scale,
                program_config=pc,
            )
        return self._out(attn, gate, M)

    def _verify_sdpa_pc(self, K, pos_hint):
        """Program config for the batched verify SDPA: the device's full grid once K*pos is big enough.

        `pos_hint` is a HOST int, and it has to be: a program config is baked into the trace at CAPTURE
        time, so the choice cannot depend on the device-side `positions` tensor that varies per replay.
        A trace captured at a short prompt therefore keeps the narrow grid for that whole request even
        if generation runs long. That is the same trade Muse Glimmer makes ("Program selection is baked
        into this request's trace. Use the prompt boundary..."), and it is the right one here because
        MTP traces are captured per request, after prefill, so the hint is the prompt length.

        `pos_hint=None` (module tests, any caller that has no host position) keeps the shipped config.
        """
        if not _VERIFY_SDPA_WIDE or pos_hint is None or K * int(pos_hint) < _VERIFY_SDPA_WIDE_WORK:
            return self.sdpa_decode_pc
        if self._verify_wide_pc is None:
            # The device grid, not a hardcoded 11x10: a differently harvested P150 exposes a different
            # worker grid (see ROOFLINE.md "the machine").
            self._verify_wide_pc = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=self.sdpa_decode_pc.k_chunk_size,
            )
        return self._verify_wide_pc

    def _verify_page_table(self, K):
        """A [K, 1] page table of zeros: the K verify rows presented as K decode "users" that all read
        the SAME cache. This is the whole trick, and it rests on two facts.

        (1) `sdpa_decode` takes its output batch from the KV cache's batch dim, so a batch-1 cache
            returns ONE row -- which is why forward_verify looped. The PAGED op takes it from the PAGE
            TABLE instead (`sdpa_decode_device_operation.cpp:196`):
                const auto B = page_table_tensor.is_sharded() ? q_shape[1]
                                                              : page_table_tensor.padded_shape()[0];
            so K identical table rows give K independently-masked outputs over one sequence, each
            bounded by its own `cur_pos`. No extra KV memory, no batch-K cache.
        (2) The FLAT `[1, n_kv, max_seq, head_dim]` cache already IS a paged cache with exactly one
            block of `max_seq` tokens. So the table is `zeros([K, 1])` and this needs no paging: it
            drops into the shipping flat-cache verify path unchanged. This half is not new --
            `forward_prefill_incremental` has always fed the flat cache to `chunked_scaled_dot_product_
            attention` the same way. What is new is the K ROWS: one table row per query.

        Requires `max_seq % k_chunk_size == 0` so the kernel's chunk-rounded read range stays inside
        block 0 (`paged_kv.read_granularity_default` documents the rounding; the FLAT sdpa enforces the
        same divisibility itself, so any config that runs today already satisfies it).

        Built lazily on first use, which is safe for exactly the reason
        `TtModel.capture_mtp_verify_trace` documents: it runs `_mtp_verify_graph()` EAGERLY once before
        `begin_trace_capture` precisely so every lazily-built constant exists before capture, because
        building one DURING capture is a host write and fatals. This table is one more such constant.
        """
        pt = self._verify_pts.get(K)
        if pt is None:
            pt = ttnn.from_torch(
                torch.zeros(K, 1, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
            )
            self._verify_pts[K] = pt
        return pt

    def _kv_update_mc_for(self, B):
        """Height-sharded L1 config for the paged_update_cache decode input: B users -> B shards of
        [TILE_SIZE, head_dim], one user per core (generalizes the B=1 single-core config to B cores)."""
        if B == 1:
            return self._kv_update_mc
        if not hasattr(self, "_kv_update_mcs"):
            self._kv_update_mcs = {}
        mc = self._kv_update_mcs.get(B)
        if mc is None:
            g = self.mesh_device.compute_with_storage_grid_size()
            cores = ttnn.num_cores_to_corerangeset(B, g, row_wise=True)
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, [ttnn.TILE_SIZE, self.head_dim], ttnn.ShardOrientation.ROW_MAJOR),
            )
            self._kv_update_mcs[B] = mc
        return mc

    def _kv_update_input(self, new, B):
        """new: [1, n_kv, B, head_dim] (from _qkv) -> [1, B, TILE_SIZE, head_dim] height-sharded (kv-heads
        padded to a tile, one user per shard), the layout paged_update_cache expects. B==1 keeps the
        single-core config; B>1 shards one user per core."""
        t = ttnn.transpose(new, 1, 2)  # [1, B, n_kv, head_dim]
        pad = ttnn.TILE_SIZE - self.n_kv_heads
        if pad > 0:
            t = ttnn.pad(t, padding=[(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)  # [1,B,TILE_SIZE,hd]
        # NOTE: no dtype cast here. The two cache-write families have OPPOSITE dtype contracts:
        #   fill_cache / paged_fill_cache  -- require input.dtype == cache.dtype (prefill; see
        #     update_cache_device_operation.cpp:72), hence _to_cache_dtype at those call sites;
        #   paged_update_cache             -- requires the INPUT be fp32/bf16 and converts on the way in
        #     (paged_update_cache_device_operation.cpp:295), while accepting a bf16/bf8/BFP4 cache
        #     (:45-46). Casting the input to bf8 here FATALs.
        return ttnn.to_memory_config(t, self._kv_update_mc_for(B))

    def forward_decode(self, x, cos, sin, kv_cache, current_pos, page_table=None):
        """Single-token decode for B users (x: [1,1,B,hidden]), fully traceable. KV cache
        [B, n_kv, max_seq, head_dim] updated IN PLACE per user at its dynamic position via
        paged_update_cache with a tensor index — O(1) traffic. current_pos: int32 [B] (per-user
        positions); cos/sin: [1,1,B,rotary_dim] (per-user RoPE). B=1 is the single-user path unchanged."""
        B = x.shape[2]
        q, k, v, gate = self._qkv(x, cos, sin)  # k,v: [1, n_kv, B, head_dim]; q: [1, nh, B, head_dim]
        # page_table None -> the flat [1, n_kv, max_seq, hd] cache (the default); a tensor -> a PAGED
        # [num_blocks, n_kv, block, hd] cache. MEASURED free at decode: paged SDPA is 0.99-1.00x flat at
        # every position and the write costs +0.3-0.8 us (tests/probe_paged_kv_perf.py, tt/paged_kv.py).
        with sp.region("attn.kv_write"):
            ttnn.experimental.paged_update_cache(
                kv_cache[0], self._kv_update_input(k, B), update_idxs_tensor=current_pos, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                kv_cache[1], self._kv_update_input(v, B), update_idxs_tensor=current_pos, page_table=page_table
            )
        with sp.region("attn.sdpa"):
            q = ttnn.transpose(q, 1, 2)  # [1, B, nh, hd] for sdpa_decode (users along dim 1)
            sdpa = (
                ttnn.transformer.scaled_dot_product_attention_decode
                if page_table is None
                else ttnn.transformer.paged_scaled_dot_product_attention_decode
            )
            kw = {} if page_table is None else {"page_table_tensor": page_table}
            attn = sdpa(
                q,
                kv_cache[0],
                kv_cache[1],
                cur_pos_tensor=current_pos,
                scale=self.scale,
                **kw,
                program_config=self.sdpa_decode_pc,
            )  # [1, B, nh, hd]
        return self._out_decode(attn, gate, B)

    def forward_verify(self, x, cos, sin, kv_cache, positions, pos_hint=None):
        """Speculative VERIFY of K tokens of ONE sequence: x [1,1,K,hidden]; `positions` is a DEVICE
        int32 tensor of shape [K] holding the K absolute positions (contiguous). Returns [1,1,K,hidden]
        — row j is the output for position j, attending 0..positions[j] inclusive (exact causal).

        `positions` must be a device tensor, not a python list: this runs inside a captured trace, and
        building an index tensor per step with `to_tt` would be a host write during capture
        ("Writes are not supported during trace capture"). Slicing a persistent [K] buffer is a pure
        device op, so the caller just refreshes that one buffer between replays.

        K>1 takes `_verify_batched`: ONE paged sdpa_decode whose batch comes from a K-row page table,
        which is 0.40-0.67x this loop's SDPA time (see that method for the measurements). The loop below
        is the K=1 path and the reference implementation — QWEN36_VERIFY_BATCH_SDPA=0 restores it for
        every K, and `test_attention_verify` gates the two against each other.

        The loop was originally written because the FLAT `sdpa_decode` derives its output batch from the
        KV cache's batch dim, so a batch-1 cache always returns ONE row (`share_cache=True` does not
        change that — it has no in-tree users and cannot express this), and a batch-K cache costs K x
        the KV memory. That reasoning holds for the flat op; the PAGED op takes its batch from the page
        table instead, which is the loophole `_verify_page_table` documents."""
        K = x.shape[2]
        assert positions.shape[-1] == K, (positions.shape, K)
        # projections + RoPE + q/k norms over all K rows at once (M=K is free: per_core_M=1)
        q, k, v, gate = self._qkv(x, cos, sin)  # q [1,nh,K,hd]; k,v [1,n_kv,K,hd]
        if _VERIFY_BATCH_SDPA and K > 1:
            return self._verify_batched(q, k, v, gate, kv_cache, positions, K, pos_hint)
        nkv, hd = self.n_kv_heads, self.head_dim
        rows = []
        for j in range(K):
            idx = ttnn.slice(positions, [j], [j + 1])  # device-side; no host write
            with sp.region("attn.kv_write"):
                # token j must be visible to rows j..K-1, so write it BEFORE row j's sdpa; rows
                # already-computed are unaffected (they read a shorter cur_pos window).
                kj = ttnn.slice(k, [0, 0, j, 0], [1, nkv, j + 1, hd])
                vj = ttnn.slice(v, [0, 0, j, 0], [1, nkv, j + 1, hd])
                ttnn.experimental.paged_update_cache(
                    kv_cache[0], self._kv_update_input(kj, 1), update_idxs_tensor=idx, page_table=None
                )
                ttnn.experimental.paged_update_cache(
                    kv_cache[1], self._kv_update_input(vj, 1), update_idxs_tensor=idx, page_table=None
                )
            with sp.region("attn.sdpa"):
                qj = ttnn.transpose(ttnn.slice(q, [0, 0, j, 0], [1, self.n_heads, j + 1, hd]), 1, 2)
                rows.append(
                    ttnn.transformer.scaled_dot_product_attention_decode(
                        qj,
                        kv_cache[0],
                        kv_cache[1],
                        cur_pos_tensor=idx,
                        scale=self.scale,
                        program_config=self.sdpa_decode_pc,
                    )  # [1, 1, nh, hd]
                )
        attn = rows[0] if K == 1 else ttnn.concat(rows, dim=1)  # [1, K, nh, hd]
        return self._out_decode(attn, gate, K)

    def _verify_batched(self, q, k, v, gate, kv_cache, positions, K, pos_hint=None):
        """One paged sdpa_decode for all K rows, instead of K flat ones.

        The row loop this replaces cost 3K ops per attention layer (2 writes + 1 sdpa per row); this
        costs 2K + 1. The K/V writes stay looped ON PURPOSE -- see the note below -- so the saving is
        the K-1 redundant SDPA launches, each of which re-read the whole cache prefix.

        MEASURED, whole verify trace, PAIRED A/B on the real 40-layer model (`prof_mtp.py --sweep`,
        ms per traced replay, QWEN36_VERIFY_BATCH_SDPA=0 vs 1):

              K   pos~30 loop -> batched          pos~2048 loop -> batched
              1     27.11 -> 27.11   +0.00           27.40 -> 27.40   -0.00   <- K=1 keeps the loop
              3     41.97 -> 41.18   -0.79 (1.9%)    42.83 -> 41.66   -1.17 (2.7%)  <- shipping gamma=2
              4     45.60 -> 44.48   -1.12 (2.5%)    46.73 -> 45.04   -1.69 (3.6%)
              8     69.22 -> 66.50   -2.72 (3.9%)    71.49 -> 67.50   -3.99 (5.6%)
             16    125.51 ->119.45   -6.06 (4.8%)   130.05 ->121.36   -8.69 (6.7%)

        The fit is the point, because it says WHICH cost went away:

              pos~30    loop 20.55 + 6.560*K  ->  batched 20.95 + 6.156*K   marginal -6.2%
              pos~2048  loop 20.55 + 6.843*K  ->  batched 21.13 + 6.264*K   marginal -8.5%

        The INTERCEPT is unchanged and the SLOPE drops: this removes a per-speculative-token cost, not a
        fixed one, so it compounds with K -- the axis a block drafter would grow. It also closes part of
        the gap `prof_mtp` prints against the Phase-A cost model (`MTP.md`: 24.31 + 4.683*K): marginal
        was 1.40x the model at short context and 1.46x at 2048, now 1.31x / 1.34x. That gap existed
        because the model charges attention 0.0016 ms per extra token (`attn = 0.295 + 0.0016(K-1)`,
        which is the SDPA op's own growth) while the loop was really paying a whole extra SDPA launch --
        plus a q `slice` + `transpose` per row and a K-row `concat` at the end, all of which this path
        also deletes. Per-op numbers per attention layer are in tests/probe_packed_verify.py's header.

        WHY THE WRITES STAY LOOPED. `paged_update_cache` also batches -- one op, K users, the same
        K-row table -- and it is much faster (K=16 @1024: 252.7 -> 136.9 us). It is also WRONG: the
        program factory dispatches one user per core, and K consecutive positions share a 32-row tile,
        so K cores read-modify-write the same tile. MEASURED: max |written - intended| on the new rows
        is 0.88-1.05 against a ~0.25 value scale (i.e. garbage), while the SAME batched write with the
        rows spread 32 apart -- one tile each -- is bit-exact. Muse Glimmer hits this too and pays for
        it with a hand-written NOC kernel whose comment is explicit that each core must serialize its
        own row writes. That kernel is the remaining ~1.2 ms/verify at K=16; it is not this change.
        """
        nkv, hd = self.n_kv_heads, self.head_dim
        # The [K,1] all-zero table says "every row's logical block 0 is physical block 0", which is only
        # the truth for the FLAT single-block cache. A genuinely PAGED cache would need slot 0's real
        # table row repeated K times (a device ttnn.repeat of PagedKV.table()); the verify path has never
        # been wired for paging, so assert rather than silently read block 0 for the whole sequence.
        assert kv_cache[0].shape[0] == 1, (
            f"batched verify assumes the flat single-block cache, got {list(kv_cache[0].shape)}; "
            f"a paged cache needs the slot's page-table row repeated K times"
        )
        # Block 0 is the whole cache, so the kernel's chunk-rounded read must land inside max_seq.
        max_seq, kchunk = kv_cache[0].shape[2], self.sdpa_decode_pc.k_chunk_size
        assert max_seq % kchunk == 0, (
            f"batched verify needs max_seq ({max_seq}) divisible by k_chunk_size ({kchunk}); "
            f"set QWEN36_VERIFY_BATCH_SDPA=0 or round max_seq up"
        )
        # Every row is written BEFORE the single sdpa, which is safe because causality comes from
        # cur_pos, not from write order: row j attends 0..positions[j] and the kernel masks the rest of
        # the chunk it read. (The row loop had to interleave write and sdpa; this does not.)
        with sp.region("attn.kv_write"):
            for j in range(K):
                idx = ttnn.slice(positions, [j], [j + 1])  # device-side; no host write
                kj = ttnn.slice(k, [0, 0, j, 0], [1, nkv, j + 1, hd])
                vj = ttnn.slice(v, [0, 0, j, 0], [1, nkv, j + 1, hd])
                ttnn.experimental.paged_update_cache(
                    kv_cache[0], self._kv_update_input(kj, 1), update_idxs_tensor=idx, page_table=None
                )
                ttnn.experimental.paged_update_cache(
                    kv_cache[1], self._kv_update_input(vj, 1), update_idxs_tensor=idx, page_table=None
                )
        with sp.region("attn.sdpa"):
            attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                ttnn.transpose(q, 1, 2),  # [1, K, nh, hd]: the K rows ARE the batch
                kv_cache[0],
                kv_cache[1],
                cur_pos_tensor=positions,
                page_table_tensor=self._verify_page_table(K),
                scale=self.scale,
                program_config=self._verify_sdpa_pc(K, pos_hint),
            )  # [1, K, nh, hd]
        return self._out_decode(attn, gate, K)

    def _out_decode(self, attn, gate, B):
        """Shared tail of decode/verify: attn [1,B,nh,hd] (already head-minor, unlike the prefill
        `_out` which takes [1,nh,S,hd]) -> gate -> o_proj -> [1,1,B,hidden]."""
        with sp.region("attn.out"):
            attn = ttnn.reshape(attn, [1, 1, B, self.n_heads * self.head_dim])
            attn = ttnn.multiply(attn, ttnn.sigmoid(gate))
            if self._wo_dram is not None and B == 1:  # DRAM-shard is a B=1/M=1 latency opt only
                attn2 = ttnn.to_memory_config(ttnn.reshape(attn, [1, self.n_heads * self.head_dim]), self._wo_amc)
                osh = ttnn.linear(attn2, self._wo_dram, program_config=self._wo_pc, memory_config=self._wo_omc)
                return ttnn.reshape(ttnn.to_memory_config(osh, ttnn.DRAM_MEMORY_CONFIG), [1, 1, 1, self.wo.shape[-1]])
            return ttnn.linear(attn, self.wo)  # [1, 1, B, hidden]
