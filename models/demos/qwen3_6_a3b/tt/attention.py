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

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_a3b.tt.common import as_weight, to_tt
from models.demos.qwen3_6_a3b.tt.rms_norm import TtRMSNorm


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
        q = ttnn.linear(x, self.wq)
        gate = ttnn.linear(x, self.wgate)
        k = ttnn.linear(x, self.wk)
        v = ttnn.linear(x, self.wv)
        q = self.q_norm.forward(ttnn.reshape(q, [1, S, self.n_heads, self.head_dim]))
        k = self.k_norm.forward(ttnn.reshape(k, [1, S, self.n_kv_heads, self.head_dim]))
        v = ttnn.reshape(v, [1, S, self.n_kv_heads, self.head_dim])
        q = apply_rope(ttnn.transpose(q, 1, 2), cos, sin, self.rotary_dim)  # [1, nh, S, hd]
        k = apply_rope(ttnn.transpose(k, 1, 2), cos, sin, self.rotary_dim)  # [1, n_kv, S, hd]
        v = ttnn.transpose(v, 1, 2)  # [1, n_kv, S, hd]
        return q, k, v, gate

    def _out(self, attn, gate, S):
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

    def forward_prefill(self, x, cos, sin, kv_cache):
        """Causal prefill that also fills a fixed-shape KV cache (kv_cache=[k_cache,v_cache])."""
        S = x.shape[2]
        q, k, v, gate = self._qkv(x, cos, sin)
        ttnn.fill_cache(kv_cache[0], k, 0)
        ttnn.fill_cache(kv_cache[1], v, 0)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return self._out(attn, gate, S)

    def forward_prefill_incremental(self, x, cos, sin, kv_cache, page_table, chunk_start, q_chunk=None):
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
        ttnn.fill_cache(kv_cache[0], k, 0, update_idx=chunk_start)  # write new K/V at rows P..P+M-1
        ttnn.fill_cache(kv_cache[1], v, 0, update_idx=chunk_start)
        # k_chunk_size=64 bounds chunked-SDPA L1 at long context (head_dim=256 is large); also lets the
        # offset P be a multiple of 64 (vs 128). q_chunk_size=qc (M for full chunks; 128 for ragged).
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), exp_approx_mode=False, q_chunk_size=qc, k_chunk_size=64
        )
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q, kv_cache[0], kv_cache[1], page_table, chunk_start_idx=chunk_start, scale=self.scale, program_config=pc
        )
        return self._out(attn, gate, M)

    def _kv_update_input(self, new):
        """new: [1, n_kv, 1, head_dim] (from _qkv) -> [1, B=1, TILE_SIZE, head_dim] height-sharded,
        the layout paged_update_cache expects (kv-heads padded to a tile, height-sharded on 1 core)."""
        t = ttnn.transpose(new, 1, 2)  # [1, 1, n_kv, head_dim]
        pad = ttnn.TILE_SIZE - self.n_kv_heads
        if pad > 0:
            t = ttnn.pad(t, padding=[(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)  # [1,1,TILE_SIZE,hd]
        return ttnn.to_memory_config(t, self._kv_update_mc)

    def forward_decode(self, x, cos, sin, kv_cache, current_pos):
        """Single-token decode, fully traceable. KV cache (fixed-shape [1, n_kv, max_seq, head_dim])
        updated IN PLACE at the dynamic position via ttnn.experimental.paged_update_cache with a
        tensor index — O(1) traffic (vs the old O(max_seq) masked write). current_pos: int32 [1]."""
        q, k, v, gate = self._qkv(x, cos, sin)  # S==1; k,v: [1, n_kv, 1, head_dim]
        ttnn.experimental.paged_update_cache(
            kv_cache[0], self._kv_update_input(k), update_idxs_tensor=current_pos, page_table=None
        )
        ttnn.experimental.paged_update_cache(
            kv_cache[1], self._kv_update_input(v), update_idxs_tensor=current_pos, page_table=None
        )
        q = ttnn.transpose(q, 1, 2)  # [1, 1(b), nh, hd] for sdpa_decode
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            cur_pos_tensor=current_pos,
            scale=self.scale,
            program_config=self.sdpa_decode_pc,
        )
        attn = ttnn.reshape(attn, [1, 1, 1, self.n_heads * self.head_dim])
        attn = ttnn.multiply(attn, ttnn.sigmoid(gate))
        return ttnn.linear(attn, self.wo)
