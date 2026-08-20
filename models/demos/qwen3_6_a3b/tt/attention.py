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
from models.demos.qwen3_6_a3b.tt.common import as_weight, build_dram_shard, to_tt
from models.demos.qwen3_6_a3b.tt.rms_norm import TtRMSNorm

# DRAM-shard the DECODE output projection (o_proj / wo): K=n_heads*head_dim, N=hidden — the K-heavy
# narrow-N shape that DRAM sharding speeds up ~2.7x at T=1 (measured; incl. reshards). Decode-only.
# Default on; QWEN36_ATTN_DRAM_SHARD=0 reverts. See FUTURE_OPTIMIZATIONS.md.
_ATTN_DRAM_SHARD = os.environ.get("QWEN36_ATTN_DRAM_SHARD", "1") != "0"
# DRAM-shard the DECODE input projections wq/wgate (K=hidden, N=n_heads*head_dim). Measured 1.71-1.73x
# per op at T=1 (bench_matmul_layout.py). Decode-only; prefill keeps interleaved. QWEN36_ATTN_DRAM_SHARD_IN=0 reverts.
_ATTN_DRAM_SHARD_IN = os.environ.get("QWEN36_ATTN_DRAM_SHARD_IN", "1") != "0"


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
            k = ttnn.linear(x, self.wk)
            v = ttnn.linear(x, self.wv)
        with sp.region("attn.qk_norm"):
            q = self.q_norm.forward(ttnn.reshape(q, [1, S, self.n_heads, self.head_dim]))
            k = self.k_norm.forward(ttnn.reshape(k, [1, S, self.n_kv_heads, self.head_dim]))
            v = ttnn.reshape(v, [1, S, self.n_kv_heads, self.head_dim])
        with sp.region("attn.rope"):
            q = apply_rope(ttnn.transpose(q, 1, 2), cos, sin, self.rotary_dim)  # [1, nh, S, hd]
            k = apply_rope(ttnn.transpose(k, 1, 2), cos, sin, self.rotary_dim)  # [1, n_kv, S, hd]
            v = ttnn.transpose(v, 1, 2)  # [1, n_kv, S, hd]
        return q, k, v, gate

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

    def forward_prefill(self, x, cos, sin, kv_cache):
        """Causal prefill that also fills a fixed-shape KV cache (kv_cache=[k_cache,v_cache])."""
        S = x.shape[2]
        q, k, v, gate = self._qkv(x, cos, sin)
        with sp.region("attn.kv_write"):
            ttnn.fill_cache(kv_cache[0], k, 0)
            ttnn.fill_cache(kv_cache[1], v, 0)
        with sp.region("attn.sdpa"):
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
        with sp.region("attn.kv_write"):
            ttnn.fill_cache(kv_cache[0], k, 0, update_idx=chunk_start)  # write new K/V at rows P..P+M-1
            ttnn.fill_cache(kv_cache[1], v, 0, update_idx=chunk_start)
        # k_chunk_size=64 bounds chunked-SDPA L1 at long context (head_dim=256 is large); also lets the
        # offset P be a multiple of 64 (vs 128). q_chunk_size=qc (M for full chunks; 128 for ragged).
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), exp_approx_mode=False, q_chunk_size=qc, k_chunk_size=64
        )
        with sp.region("attn.sdpa"):
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                kv_cache[0],
                kv_cache[1],
                page_table,
                chunk_start_idx=chunk_start,
                scale=self.scale,
                program_config=pc,
            )
        return self._out(attn, gate, M)

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
        return ttnn.to_memory_config(t, self._kv_update_mc_for(B))

    def forward_decode(self, x, cos, sin, kv_cache, current_pos):
        """Single-token decode for B users (x: [1,1,B,hidden]), fully traceable. KV cache
        [B, n_kv, max_seq, head_dim] updated IN PLACE per user at its dynamic position via
        paged_update_cache with a tensor index — O(1) traffic. current_pos: int32 [B] (per-user
        positions); cos/sin: [1,1,B,rotary_dim] (per-user RoPE). B=1 is the single-user path unchanged."""
        B = x.shape[2]
        q, k, v, gate = self._qkv(x, cos, sin)  # k,v: [1, n_kv, B, head_dim]; q: [1, nh, B, head_dim]
        with sp.region("attn.kv_write"):
            ttnn.experimental.paged_update_cache(
                kv_cache[0], self._kv_update_input(k, B), update_idxs_tensor=current_pos, page_table=None
            )
            ttnn.experimental.paged_update_cache(
                kv_cache[1], self._kv_update_input(v, B), update_idxs_tensor=current_pos, page_table=None
            )
        with sp.region("attn.sdpa"):
            q = ttnn.transpose(q, 1, 2)  # [1, B, nh, hd] for sdpa_decode (users along dim 1)
            attn = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                kv_cache[0],
                kv_cache[1],
                cur_pos_tensor=current_pos,
                scale=self.scale,
                program_config=self.sdpa_decode_pc,
            )  # [1, B, nh, hd]
        return self._out_decode(attn, gate, B)

    def forward_verify(self, x, cos, sin, kv_cache, positions):
        """Speculative VERIFY of K tokens of ONE sequence: x [1,1,K,hidden]; `positions` is a DEVICE
        int32 tensor of shape [K] holding the K absolute positions (contiguous). Returns [1,1,K,hidden]
        — row j is the output for position j, attending 0..positions[j] inclusive (exact causal).

        `positions` must be a device tensor, not a python list: this runs inside a captured trace, and
        building an index tensor per step with `to_tt` would be a host write during capture
        ("Writes are not supported during trace capture"). Slicing a persistent [K] buffer is a pure
        device op, so the caller just refreshes that one buffer between replays.

        Why not one batched sdpa_decode: `sdpa_decode` derives its output batch from the KV cache's
        batch dim, so a batch-1 cache always returns ONE row (`share_cache=True` does not change
        that — it has no in-tree users and cannot express this). A batch-K cache DOES work (verified
        PCC 1.0) but costs K x the KV memory. It is unnecessary: attention's cost is ~0.27 ms of
        PROJECTIONS vs ~0.022 ms of SDPA per layer, so running `_qkv`/`wo` ONCE over the K rows and
        looping only the (kv-write, sdpa) pair is already flat in the expensive part — no extra
        memory, and correct at any context length."""
        K = x.shape[2]
        assert positions.shape[-1] == K, (positions.shape, K)
        # projections + RoPE + q/k norms over all K rows at once (M=K is free: per_core_M=1)
        q, k, v, gate = self._qkv(x, cos, sin)  # q [1,nh,K,hd]; k,v [1,n_kv,K,hd]
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
