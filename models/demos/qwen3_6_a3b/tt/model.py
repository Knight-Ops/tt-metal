# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-nn full model assembly for Qwen3.6-35B-A3B (text-only, prefill path).

Embedding -> 40 hybrid decoder layers -> final RMSNorm -> lm_head. Layers are built one at a time
via the streaming ``CheckpointLoader`` so host RAM never holds the full 72 GB checkpoint.

Phase A: prefill only (no KV/state cache). Generation in the demo re-runs the growing sequence per
token. KV/state caches + the fused delta-rule kernel come in Phase B.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_a3b.tt import prefill_profiler as prof
from models.demos.qwen3_6_a3b.tt.attention import precompute_rope
from models.demos.qwen3_6_a3b.tt.common import as_weight, from_tt, to_tt
from models.demos.qwen3_6_a3b.tt.decoder import TtDecoderLayer
from models.demos.qwen3_6_a3b.tt.rms_norm import TtRMSNorm


class TtModel(LightweightModule):
    def __init__(self, mesh_device, args, loader, num_layers=None):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.n_layers = num_layers if num_layers is not None else args.n_layers

        cp = args.weight_cache_path
        # loader.embed_tokens()/lm_head() return lazy thunks (the big-weight read is deferred so a
        # weight-cache hit skips it); final_norm() returns the tiny tensor directly.
        self.embed_weight = to_tt(
            loader.embed_tokens(),
            mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=f"{cp}/embed_tokens" if cp else None,
        )
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(TtDecoderLayer(mesh_device, args, loader, i))
        self.final_norm = TtRMSNorm(mesh_device, loader.final_norm(), args.norm_eps, add_unit_offset=True)
        # lm_head is the biggest single decode op and is DRAM-bandwidth-bound (~70% peak), so the only
        # lever is fewer weight bytes: BFP4 ~halves its read (~1.47 -> ~0.75 ms). Accuracy-gated; set
        # QWEN36_LMHEAD_BF4=0 to fall back to BFP8 if greedy generation degrades.
        lm_dtype = ttnn.bfloat8_b if os.environ.get("QWEN36_LMHEAD_BF4") == "0" else ttnn.bfloat4_b
        # as_weight transposes (out,in)->(in,out) lazily inside the thunk, so a cache hit skips both
        # the HF read and the transpose.
        self.lm_head_w = as_weight(
            loader.lm_head(),
            mesh_device,
            dtype=lm_dtype,
            cache_file_name=f"{cp}/lm_head" if cp else None,
        )
        # On-device RoPE tables: cos/sin for every position, indexed by position with ttnn.embedding
        # during decode (no per-token host recompute). Row-major so they act as embedding weights.
        self.cos_table, self.sin_table = self._build_rope_tables()
        # Token selection is greedy (on-device argmax) by default. enable_sampling() swaps in the
        # on-device temperature/top-k/top-p sampler; None here means "greedy, zero added cost".
        self.sampling = None
        # Captured decode trace (one at a time). Re-captured per request; the PRIOR trace is released
        # first (capture_decode_trace) so traces never accumulate — that accumulation, together with
        # per-request cache reallocation, was the leak that OOMed the server.
        self.trace_id = None

    def _build_rope_tables(self):
        """[max_seq_len, rotary_dim] cos/sin tables (default RoPE), as ROW_MAJOR embedding weights."""
        rd, theta, max_seq = self.args.rotary_dim, self.args.rope_theta, self.args.max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, rd, 2, dtype=torch.float) / rd))
        pos = torch.arange(0, max_seq, dtype=torch.float)
        emb = torch.cat((torch.outer(pos, inv_freq),) * 2, dim=-1)  # [max_seq, rd]
        cos = to_tt(emb.cos(), self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        sin = to_tt(emb.sin(), self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        return cos, sin

    def _embed(self, input_ids, T):
        ids = to_tt(input_ids.to(torch.int32), self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.embedding(ids, self.embed_weight, layout=ttnn.TILE_LAYOUT)  # [1, T, hidden]
        return ttnn.reshape(x, [1, 1, T, self.args.dim])

    def _head(self, x, T):
        # Prefill generation only needs the LAST token's logits (demo + decode use logits[0, -1]).
        # Slice to the last position BEFORE the 248k-vocab matmul: ~T× less lm_head compute and a
        # ~T× smaller D2H transfer (e.g. ~127 MB -> ~0.5 MB at T=256). Returns [1, 1, vocab].
        x = ttnn.slice(x, [0, 0, T - 1, 0], [1, 1, T, self.args.dim])
        x = self.final_norm.forward(x)
        x = ttnn.reshape(x, [1, self.args.dim])
        logits = ttnn.linear(x, self.lm_head_w)  # [1, vocab]
        return from_tt(logits, self.mesh_device).reshape(1, 1, self.args.vocab_size)

    def _alloc_cache(self, layer):
        if layer.is_linear:
            # Persistent gated-delta state buffers. Pre-allocating (rather than deferring to first-use,
            # which stores the transient prefill output as the cache) keeps the decode's in-place state
            # writes pointed at a STABLE address. Required for trace capture: a reclaimed/reallocated
            # transient buffer would force a host write on the in-place copy-into, which trace capture
            # forbids (TT_FATAL "Writes are not supported during trace capture"). Without this, the
            # eager-prefill→traced-decode path fataled only at high layer counts (memory pressure
            # reclaims the ~1 MB-per-layer transients); the traced-prefill path already pre-allocated.
            a = self.args
            return {
                "conv_state": ttnn.zeros(
                    [a.conv_kernel_size - 1, a.lin_conv_dim],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                ),
                "recurrent_state": ttnn.zeros(
                    [1, a.lin_num_v_heads, a.lin_head_k_dim, a.lin_head_v_dim],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                ),
            }
        # fixed-shape KV cache for an attention layer: [1, n_kv, max_seq, head_dim]
        shape = [1, self.args.n_kv_heads, self.max_seq, self.args.head_dim]
        k = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        v = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        return [k, v]

    def _reset_linear_state(self):
        """Zero the gated-delta conv left-pad in place so a REUSED cache behaves like a freshly
        allocated one for a fresh (non-continuation) prefill. Only conv_state needs it: the fused
        prefill starts the recurrent state at S=0 internally and overwrites recurrent_state
        (gated_delta.py), and the KV cache is filled 0..T-1 and read only up to self.pos. In-place
        (no allocation), so it is safe to call while a captured trace is live."""
        for c in self.caches:
            if isinstance(c, dict) and "conv_state" in c:
                ttnn.multiply(c["conv_state"], 0.0, output_tensor=c["conv_state"])

    # Long prompts are prefilled in chunks via prefill_long (single-shot can't hold full-sequence
    # activations/attention past ~1-2K). forward() routes there above this length; QWEN36_PREFILL_CHUNK
    # sets the chunk size. The threshold stays above the chunk so the per-chunk single-shot calls don't recurse.
    _PREFILL_CHUNK = int(os.environ.get("QWEN36_PREFILL_CHUNK", "512"))
    _LONG_PREFILL_THRESHOLD = int(os.environ.get("QWEN36_LONG_PREFILL_THRESHOLD", "2048"))

    def forward(self, input_ids: torch.Tensor):
        """Prefill. input_ids: torch [1, T]. Returns last-token logits [1,1,vocab]. Long prompts
        (T > threshold) are streamed in chunks via prefill_long; short prompts go single-shot."""
        if input_ids.shape[1] > self._LONG_PREFILL_THRESHOLD:
            return self.prefill_long(input_ids)
        return self._prefill_single(input_ids)

    def _prefill_single(self, input_ids: torch.Tensor):
        """Single-shot prefill (bounded to short prompts: materializes the full [T,*] activations).
        Allocates per-layer caches and sets self.pos."""
        T = input_ids.shape[1]
        self.max_seq = self.args.max_seq_len
        # Allocate per-layer caches ONCE and reuse them across prefills (the same _pf_caches_ready
        # guard the traced-prefill path uses). Reallocating per request was the OOM: the decode
        # trace pins the caches it references, so the old ones could never be freed.
        if getattr(self, "caches", None) is None or not getattr(self, "_pf_caches_ready", False):
            self.caches = [self._alloc_cache(layer) for layer in self.layers]
            self._pf_caches_ready = True
        else:
            self._reset_linear_state()  # fresh prefill: clear the gated-delta conv left-pad
        self.pos = T
        prof.reset()
        with prof.phase(self.mesh_device, "embed"):
            x = self._embed(input_ids, T)
        with prof.phase(self.mesh_device, "rope"):
            cos, sin = precompute_rope(T, self.args.rotary_dim, self.args.rope_theta, self.mesh_device)
        for layer, cache in zip(self.layers, self.caches):
            x = layer.forward_prefill(x, cos, sin, cache)
        with prof.phase(self.mesh_device, "head"):
            out = self._head(x, T)
        prof.report()
        return out

    def prefill_long(self, input_ids: torch.Tensor, chunk: int | None = None):
        """Streamed (chunked) prefill for long prompts — supports the model's full context (up to
        max_seq_len, e.g. 256K) with memory bounded by the chunk size, not the prompt length. Ingests
        the prompt in `chunk`-token blocks: the first block via _prefill_single (allocates caches), the
        rest via forward_incremental (gated-delta carries its O(1) recurrent+conv state; attention fills
        the KV cache and attends the new block over the accumulated context with chunked SDPA). Leaves
        caches + self.pos correct for start_decode. Returns the last block's logits [1,1,vocab].

        chunk (default QWEN36_PREFILL_CHUNK=512) must be a multiple of 128 (fill_cache/SDPA alignment);
        with a multiple of 128 the running offset P is always a multiple of both chunk and k_chunk_size=64,
        satisfying chunked_scaled_dot_product_attention's chunk_start_idx alignment. T need NOT be a
        multiple of chunk: the leftover final block (T mod chunk) is ingested as a ragged
        forward_incremental block (right-padded to a 128-multiple internally for attention; gated-delta
        processes only the real tokens via valid_len). Every offset P passed to a block is still a
        multiple of chunk, so the chunk-start alignment holds."""
        M = chunk or self._PREFILL_CHUNK
        T = input_ids.shape[1]
        assert M % 128 == 0, f"prefill chunk {M} must be a multiple of 128"
        assert T <= self.args.max_seq_len, f"prompt {T} exceeds max_seq_len {self.args.max_seq_len}"
        if T <= M:
            return self._prefill_single(input_ids)
        logits = self._prefill_single(input_ids[:, :M])  # first block: allocates caches, self.pos=M
        n_full = (T // M) * M  # tokens covered by whole M-token chunks
        for P in range(M, n_full, M):
            logits = self.forward_incremental(input_ids[:, P : P + M])  # carries state; self.pos -> P+M
        if T > n_full:  # ragged final block (T mod M tokens); forward_incremental pads it internally
            logits = self.forward_incremental(input_ids[:, n_full:T], ragged=True)
        return logits

    # ---- Traced prefill (additive; mirrors capture_decode_trace). Prefill is ~54% host-dispatch over
    # ~11k tiny ops; replaying a captured graph removes that per-op launch latency. Shape-static, so we
    # capture one trace per sequence length T (a serving deployment buckets prompts by padded length and
    # reuses the trace across requests). The mesh MUST be opened with trace_region_size>0. Decode is
    # untouched. The chunked gated-delta loop is fixed-shape for a fixed T, so it unrolls cleanly. ----
    def _prefill_graph(self, T):
        """Self-contained traceable prefill body: embed (from the persistent id buffer) -> layers,
        writing the per-layer caches in place, then copy the layers' output into the persistent output
        buffer. final_norm + lm_head stay eager (the last-token slice index is data-dependent on T)."""
        x = ttnn.embedding(self._pf_ids[T], self.embed_weight, layout=ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, [1, 1, T, self.args.dim])
        for layer, cache in zip(self.layers, self.caches):
            x = layer.forward_prefill_traced(x, self._pf_cos[T], self._pf_sin[T], cache, self._pf_pool[T])
        ttnn.copy(x, self._pf_out[T])  # land the result at a stable address for the eager head

    def capture_prefill_trace(self, T):
        """Allocate persistent buffers for length T, run one eager compile pass (builds kernels +
        creates the per-layer cache entries), then record the trace. PRECONDITION: mesh opened with
        trace_region_size>0."""
        if not hasattr(self, "_pf_traces"):
            self._pf_traces, self._pf_ids, self._pf_cos, self._pf_sin, self._pf_out = {}, {}, {}, {}, {}
            self._pf_pool = {}
        # Caches are length-independent (KV is [1,n_kv,max_seq,hd]; gated-delta state is [1,Vh,Dk,Dv]),
        # so allocate them ONCE and share across trace lengths.
        if getattr(self, "caches", None) is None or not getattr(self, "_pf_caches_ready", False):
            self.max_seq = self.args.max_seq_len
            # _alloc_cache now pre-allocates PERSISTENT gated-delta conv/recurrent state (so the
            # in-place decode writes hit a stable address — trace-safe; see _alloc_cache).
            self.caches = [self._alloc_cache(layer) for layer in self.layers]
            self._pf_caches_ready = True
        self._pf_ids[T] = to_tt(
            torch.zeros(1, T, dtype=torch.int32), self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self._pf_cos[T], self._pf_sin[T] = precompute_rope(
            T, self.args.rotary_dim, self.args.rope_theta, self.mesh_device
        )
        self._pf_out[T] = ttnn.zeros(
            [1, 1, T, self.args.dim], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
        )
        # Pre-allocated gated-delta scratch/state pool (shared across linear layers; replaces the
        # per-chunk in-graph ttnn.zeros that capture forbids). Built from the first linear layer's mixer.
        lin = next(l for l in self.layers if l.is_linear)
        self._pf_pool[T] = lin.mixer.build_trace_pool(T)
        # Zero source to reset each layer's conv_state before a replay: prefill is always fresh (no
        # prior context), but the conv reads conv_state as its left-pad, and that buffer persists
        # across replays. (recurrent_state is output-only — the chunked path starts S=0 internally;
        # KV is written fresh and prefill attends fresh k/v, so only conv_state needs resetting.)
        if not hasattr(self, "_pf_conv_zero"):
            self._pf_conv_zero = ttnn.zeros(
                [self.args.conv_kernel_size - 1, self.args.lin_conv_dim],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
            )
        self._prefill_graph(T)  # eager compile pass (must precede capture: capture cannot JIT)
        ttnn.synchronize_device(self.mesh_device)
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._prefill_graph(T)
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        self._pf_traces[T] = tid

    def _default_prefill_buckets(self):
        """Powers of two that are chunk-multiples, up to (and including) max_seq_len — the model's hard
        prompt limit (fixed-shape KV cache). So every prompt the model can run maps to a bucket."""
        mx = self.args.max_seq_len
        bs = [b for b in (128, 256, 512, 1024, 2048, 4096, 8192) if b <= mx]
        if not bs or bs[-1] != mx:
            bs.append(mx)  # ensure coverage up to the KV-cache limit
        return sorted(set(bs))

    def setup_prefill_traces(self, buckets=None):
        """Pre-capture one prefill trace per bucket (call once at startup so capture never happens
        mid-serving). Prompts are padded up to the smallest bucket >= their length at replay."""
        self._pf_buckets = sorted(buckets) if buckets else self._default_prefill_buckets()
        for B in self._pf_buckets:
            self.capture_prefill_trace(B)

    def _write_valid_mask(self, B, T):
        """Write the per-request [1,1,B] valid mask (1 for the real T tokens, 0 for right-padding) into
        each layer's shared pool, so the gated-delta zeros pad positions out of the recurrent state."""
        m = torch.zeros(1, 1, B)
        m[:, :, :T] = 1.0
        valid = self._pf_pool[B]["valid"]
        host = ttnn.from_torch(m, dtype=valid.dtype, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, valid)

    def forward_prefill_traced(self, input_ids: torch.Tensor):
        """Traced prefill with bucketing. Pads the prompt up to the smallest captured bucket >= T,
        replays that trace, and runs the eager head at the REAL last token. Falls back to eager
        forward() if T exceeds the largest bucket (only possible if buckets don't reach max_seq_len;
        a prompt > max_seq_len can't run at all — fixed KV cache). Leaves caches + self.pos for decode."""
        T = input_ids.shape[1]
        buckets = getattr(self, "_pf_buckets", None) or sorted(getattr(self, "_pf_traces", {}).keys())
        B = next((b for b in buckets if b >= T), None)
        if B is None:
            return self.forward(input_ids)  # no bucket covers T -> eager fallback (always correct)
        if not hasattr(self, "_pf_traces") or B not in self._pf_traces:
            self.capture_prefill_trace(B)
        padded = torch.zeros(1, B, dtype=torch.int32)
        padded[0, :T] = input_ids[0].to(torch.int32)
        host_ids = ttnn.from_torch(padded, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_ids, self._pf_ids[B])
        self._write_valid_mask(B, T)  # zero the gated-delta state contribution of the pad tokens
        for cache in self.caches:  # fresh prefill: reset conv left-pad to zeros before replay
            if isinstance(cache, dict) and "conv_state" in cache:
                ttnn.copy(self._pf_conv_zero, cache["conv_state"])
        ttnn.execute_trace(self.mesh_device, self._pf_traces[B], cq_id=0, blocking=False)
        self.pos = T
        return self._head(self._pf_out[B], T)  # head slices the REAL last token (T-1), not the bucket

    def forward_incremental(self, input_ids: torch.Tensor, ragged: bool = False):
        """Incremental (from-cache) prefill of M NEW tokens continuing from the existing caches
        (self.pos = current context length P), instead of re-prefilling the whole P+M context.
        Requires a prior forward()/forward_incremental() to have populated the caches. P and M must be
        multiples of 128 (chunked-SDPA + fill_cache alignment). Updates caches + self.pos; returns the
        last NEW token's logits [1,1,vocab]. Eager.

        ragged=True (the final block of a long prompt, size M not 128-aligned): the block is right-
        padded to a multiple of 128 so attention's fill_cache/SDPA stay aligned, gated-delta is told
        the real length (valid_len) so it updates state over real tokens only, and chunked-SDPA uses
        q_chunk_size=128 (P is a multiple of the chunk => of 128, so P % q_chunk_size == 0 holds; the
        block width M_pad is also a multiple of 128). self.pos advances by the REAL M and the head
        reads the real last token, so the padded KV rows (past self.pos) are never read."""
        M = input_ids.shape[1]
        P = self.pos
        if ragged:
            M_pad = ((M + 127) // 128) * 128  # minimal 128-multiple padding (bounds extra KV rows)
            valid = M if M_pad != M else None  # tell gated-delta the real length when we padded
            q_chunk = 128  # q_chunk_size that divides P (multiple of the chunk) and M_pad
            assert P + M_pad <= self.max_seq, (
                f"ragged block overflows the KV cache: P({P}) + padded_M({M_pad}) > max_seq({self.max_seq}); "
                f"raise QWEN36_MAX_SEQ to a multiple of 128 >= {((P + M + 127) // 128) * 128}"
            )
            if M_pad != M:
                padded = torch.zeros(1, M_pad, dtype=input_ids.dtype)
                padded[0, :M] = input_ids[0]
                input_ids = padded
        else:
            M_pad, valid, q_chunk = M, None, None  # full chunk: already aligned, unchanged path
        if not hasattr(self, "_inc_page_table"):  # trivial single-block page table (built once)
            self._inc_page_table = to_tt(
                torch.zeros(1, 32, dtype=torch.int32), self.mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        x = self._embed(input_ids, M_pad)
        cos, sin = precompute_rope(M_pad, self.args.rotary_dim, self.args.rope_theta, self.mesh_device, start_pos=P)
        for layer, cache in zip(self.layers, self.caches):
            x = layer.forward_prefill_incremental(
                x, cos, sin, cache, self._inc_page_table, P, valid_len=valid, q_chunk=q_chunk
            )
        self.pos = P + M  # advance by the REAL token count (not the padded width)
        return self._head(x, M)  # slice the real last token (index M-1)

    # Sampling tail constants. ttnn.topk is only multicore (~0.2 ms) for power-of-2 widths <= 32768;
    # over the full 248k vocab it falls back to single-core (~38 ms). So we chunk the vocab into
    # SAMP_CHUNK-wide power-of-2 pieces, topk each, then merge. ttnn.sampling is fixed to 32 users.
    SAMP_CHUNK = 32768
    SAMP_K = 32  # max_top_k (device limit); also ttnn.sampling's cap
    SAMP_USERS = 32
    SAMP_NEG = -1.0e4  # pad fill for the last chunk; below any real logit, so never selected

    def enable_sampling(
        self, temperature: float, top_k: int = 0, top_p: float = 1.0, seed: int = 0, presence_penalty: float = 0.0
    ):
        """Switch token selection from greedy argmax to on-device temperature/top-k/top-p sampling.

        ``temperature == 0`` restores greedy (the default, fastest path: plain ttnn.argmax). Otherwise
        the decode tail runs chunked ``ttnn.topk`` + ``ttnn.sampling`` fully on device and in-trace
        (see ``_select_token``), adding ~0.5 ms/token. With ``presence_penalty > 0`` a vocab-sized
        presence mask (tokens generated so far) is subtracted from the logits before topk
        (``logits -= presence_penalty * mask``); the mask is scattered/accumulated on device each step,
        adding ~0.5 ms/token. Call BEFORE start_decode / trace capture so the chosen tail is recorded.

        ttnn.sampling's ``temp`` is 1/T and ``k`` must be in (0, 32]; we clamp accordingly.
        """
        if temperature == 0:
            self.sampling = None  # greedy
            return
        self.sampling = True
        V, CW, K, B = self.args.vocab_size, self.SAMP_CHUNK, self.SAMP_K, self.SAMP_USERS
        nc = (V + CW - 1) // CW  # number of power-of-2 chunks (last is logit-padded to CW)
        self._presence_penalty = max(presence_penalty, 0.0)

        def _t(vals, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
            return ttnn.from_torch(vals, device=self.mesh_device, dtype=dtype, layout=layout)

        # Static (param-independent) buffers, built once.
        if getattr(self, "_samp_nc", None) != nc:
            self._samp_nc = nc
            # per-chunk global-index offset (chunk i -> +i*CW), broadcast across the K kept indices
            off = (torch.arange(nc, dtype=torch.int32).view(1, 1, nc, 1).expand(1, 1, nc, K).contiguous()) * CW
            self._samp_offsets = _t(off, ttnn.int32, ttnn.TILE_LAYOUT)
            # 0..CW-1 indices for topk, replicated per chunk row
            li = torch.arange(CW, dtype=torch.int32).view(1, 1, 1, CW).expand(1, 1, nc, CW).contiguous()
            self._samp_local_idx = _t(li, ttnn.uint16, ttnn.TILE_LAYOUT)
            self._samp_uids = _t(torch.arange(B, dtype=torch.int32), ttnn.uint32)
            self.t_tok32 = _t(torch.zeros(1, 1, 1, B, dtype=torch.int32), ttnn.uint32)  # rank-4 sampling output
            # 0..V-1 vocab indices, used to one-hot the sampled token into the presence mask (see below)
            self._samp_iota = _t(torch.arange(V, dtype=torch.int32).view(1, 1, 1, V), ttnn.int32, ttnn.TILE_LAYOUT)

        k = top_k if 0 < top_k <= K else K  # k in (0, 32]; <=0 ("all") or >32 -> 32
        p = min(max(top_p, 0.0), 1.0)
        self.t_k = _t(torch.full((B,), k, dtype=torch.int32), ttnn.uint32)
        self.t_p = _t(torch.full((B,), p), ttnn.bfloat16)
        self.t_temp = _t(torch.full((B,), 1.0 / temperature), ttnn.bfloat16)  # ttnn.sampling scales by 1/T
        # Per-step RNG seed, advanced on device each decode step (plus_one) so a static trace still
        # draws fresh randomness every replay; deterministic for a fixed starting `seed`.
        self.t_seed = _t(torch.full((B,), seed, dtype=torch.int32), ttnn.uint32)
        # Presence mask over the vocab (generated tokens), reset per request, accumulated on device.
        self.t_presence = _t(torch.zeros(1, 1, 1, V), ttnn.bfloat16, ttnn.TILE_LAYOUT)

    def start_decode(self, first_token_id: int):
        """Seed the device-resident decode state from the prefill's first token. After this, drive
        generation with decode_step_eager() (compiles) and/or decode_step_traced()."""
        self.t_tok = to_tt(
            torch.tensor([first_token_id], dtype=torch.int32),
            self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.t_curpos = to_tt(
            torch.tensor([self.pos], dtype=torch.int32),
            self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.t_ropepos = to_tt(
            torch.tensor([self.pos], dtype=torch.int32),
            self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def _rope_for(self, rope_pos):
        """Index the RoPE tables at the current position -> (cos, sin) of [1,1,1,rotary_dim]."""
        rd = self.args.rotary_dim
        cos = ttnn.reshape(ttnn.embedding(rope_pos, self.cos_table, layout=ttnn.TILE_LAYOUT), [1, 1, 1, rd])
        sin = ttnn.reshape(ttnn.embedding(rope_pos, self.sin_table, layout=ttnn.TILE_LAYOUT), [1, 1, 1, rd])
        return cos, sin

    def _decode_graph(self):
        """Self-contained, traceable decode step: read device-resident token + position, run
        embed -> layers -> norm -> lm_head -> argmax, write the next token back into self.t_tok, and
        advance the position tensors on device. No host work / no full-logit readback."""
        cos, sin = self._rope_for(self.t_ropepos)
        x = ttnn.embedding(self.t_tok, self.embed_weight, layout=ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, [1, 1, 1, self.args.dim])
        for layer, cache in zip(self.layers, self.caches):
            x = layer.forward_decode(x, cos, sin, cache, self.t_curpos)
        x = self.final_norm.forward(x)
        x = ttnn.reshape(x, [1, self.args.dim])
        self._select_token(x)  # lm_head -> argmax (greedy) or topk+sampling; writes self.t_tok
        ttnn.plus_one(self.t_curpos)  # advance position on device (for KV write + sdpa)
        ttnn.plus_one(self.t_ropepos)  # advance RoPE-table index on device

    def _select_token(self, x):
        """Run lm_head on the post-norm hidden state and write the next token id into self.t_tok.

        Greedy (self.sampling is None) is the default and unchanged: the single-core argmax over the
        248k vocab is ~8.4 ms (the lm_head matmul itself is only ~1.5 ms); the multicore argmax needs
        a ROW_MAJOR input and runs in ~0.06 ms, so converting layout first (~0.1 ms) cuts the head
        stage ~9.9 -> ~1.6 ms/token.

        Sampling replaces the argmax tail with chunked on-device topk + ttnn.sampling. topk over the
        full 248k vocab is single-core (~38 ms); chunking into power-of-2 (<=32768) pieces keeps every
        topk multicore (~0.2 ms). The expensive lm_head + topk stay 1-row; only the tiny merged
        candidate set (nc*32 values) is replicated to the 32 users ttnn.sampling requires, and we read
        the token from row 0. The seed is advanced on device so a captured trace still varies per step."""
        if self.sampling is None:  # greedy
            logits = ttnn.linear(x, self.lm_head_w)  # device [1, vocab]
            logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
            tok = ttnn.argmax(logits, dim=-1, keepdim=False, use_multicore=True)  # [1] uint32
            ttnn.copy(ttnn.to_layout(tok, ttnn.ROW_MAJOR_LAYOUT), self.t_tok)  # next token, in place
            return
        V, CW, K, B, nc = self.args.vocab_size, self.SAMP_CHUNK, self.SAMP_K, self.SAMP_USERS, self._samp_nc
        logits = ttnn.linear(x, self.lm_head_w)  # [1, vocab]  (1-row: lm_head stays cheap)
        logits = ttnn.reshape(logits, [1, 1, 1, V])
        if self._presence_penalty > 0:  # discourage repeats: subtract penalty from already-seen tokens
            logits = ttnn.sub(logits, ttnn.multiply(self.t_presence, self._presence_penalty))
        logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, nc * CW - V)], value=self.SAMP_NEG)
        chunks = ttnn.reshape(logits, [1, 1, nc, CW])  # power-of-2 width per row -> multicore topk
        vals, idxs = ttnn.topk(chunks, k=K, dim=-1, sorted=False, indices_tensor=self._samp_local_idx)
        gidx = ttnn.add(ttnn.typecast(idxs, ttnn.int32), self._samp_offsets, dtype=ttnn.int32)  # global ids
        cand_v = ttnn.repeat(ttnn.reshape(vals, [1, 1, 1, nc * K]), ttnn.Shape([1, 1, B, 1]))  # [1,1,32,nc*K]
        cand_i = ttnn.repeat(ttnn.reshape(gidx, [1, 1, 1, nc * K]), ttnn.Shape([1, 1, B, 1]))
        cand_i = ttnn.untilize(ttnn.to_memory_config(cand_i, ttnn.DRAM_MEMORY_CONFIG), use_multicore=True)  # RM int32
        ttnn.manual_seed(seeds=self.t_seed, user_ids=self._samp_uids)
        ttnn.sampling(cand_v, cand_i, k=self.t_k, p=self.t_p, temp=self.t_temp, output_tensor=self.t_tok32)
        idx4 = ttnn.reshape(ttnn.slice(self.t_tok32, [0, 0, 0, 0], [1, 1, 1, 1]), [1, 1, 1, 1])  # sampled token
        ttnn.copy(ttnn.reshape(idx4, [1]), self.t_tok)  # next token, in place
        if self._presence_penalty > 0:  # mark the new token present: OR a one-hot into the mask.
            # ``ttnn.scatter`` over the [1,1,1,vocab] mask is ~2 ms (it scans the 32x tile-padded
            # vocab); an elementwise one-hot (eq vs the iota index buffer) + maximum is ~0.4 ms.
            idx_t = ttnn.to_layout(ttnn.typecast(idx4, ttnn.int32), ttnn.TILE_LAYOUT)
            onehot = ttnn.typecast(ttnn.eq(self._samp_iota, idx_t), ttnn.bfloat16)
            ttnn.maximum(self.t_presence, onehot, output_tensor=self.t_presence)  # in-place OR
        ttnn.plus_one(self.t_seed)  # fresh RNG next step (trace-safe; validated)

    def decode_step_eager(self):
        """Run one decode step eagerly (compiles kernels for trace capture). Returns the next token."""
        self._decode_graph()
        self.pos += 1
        return int(from_tt(self.t_tok, self.mesh_device).flatten()[0])

    def _state_tensors(self):
        ts = []
        for c in self.caches:
            if isinstance(c, list):  # attention KV [k, v]
                ts += c
            else:  # gated-delta state dict
                ts += [v for v in c.values()]
        ts += [self.t_tok, self.t_curpos, self.t_ropepos]  # generation state advances on device
        if self.sampling is not None:
            ts.append(self.t_seed)  # RNG seed advances each step; snapshot/restore around capture
            if self._presence_penalty > 0:
                ts.append(self.t_presence)  # presence mask accumulates each step
        return ts

    def capture_decode_trace(self):
        """Record the self-contained decode graph as a trace. PRECONDITION: at least one
        decode_step_eager() has run so all kernels are compiled (capture must not JIT). The decode
        state (caches + token + positions) is snapshotted and restored so recording the dummy step
        doesn't perturb generation. After this, drive every real step through decode_step_traced().

        Releases any PRIOR decode trace first: traces are captured per request, and without the
        release the device's trace region accumulates one trace per request (and pins the buffers
        each references) — that, with per-request cache reallocation, was the OOM. Persistent caches
        (reused across requests) + this release keep device memory flat."""
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.trace_id)
        snap = [ttnn.clone(t) for t in self._state_tensors()]
        self.trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._decode_graph()
        ttnn.end_trace_capture(self.mesh_device, self.trace_id, cq_id=0)
        for orig, s in zip(self._state_tensors(), snap):
            ttnn.copy(s, orig)  # undo the mutation done while recording

    def decode_step_traced(self):
        """Replay the captured decode trace for one real step. Host work: kick off the trace and
        read back only the single next-token id (no per-token input rebuild, no full-logit D2H)."""
        ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=False)
        self.pos += 1
        return int(from_tt(self.t_tok, self.mesh_device).flatten()[0])
