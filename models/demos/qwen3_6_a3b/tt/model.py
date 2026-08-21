# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-nn full model assembly for Qwen3.6-35B-A3B (text-only).

Embedding -> 40 hybrid decoder layers -> final RMSNorm -> lm_head. Layers are built one at a time
via the streaming ``CheckpointLoader`` so host RAM never holds the full 72 GB checkpoint.

Implements prefill (single-shot, chunked long-prompt, and traced) plus cached decode with per-layer
KV / gated-delta state caches and an optional captured decode trace. Decode selection is on-device
(greedy argmax by default; on-device sampling via enable_sampling); a logits-returning decode
(decode_forward_logits) supports host-side sampling for the vLLM path.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_a3b.tt import prefill_profiler as prof
from models.demos.qwen3_6_a3b.tt import signpost as sp
from models.demos.qwen3_6_a3b.tt.attention import precompute_rope
from models.demos.qwen3_6_a3b.tt.common import as_weight, from_tt, to_tt
from models.demos.qwen3_6_a3b.tt.decoder import TtDecoderLayer
from models.demos.qwen3_6_a3b.tt.gated_delta import _CONV_ADDCHAIN
from models.demos.qwen3_6_a3b.tt.rms_norm import TtRMSNorm


class TtModel(LightweightModule):
    def __init__(self, mesh_device, args, loader, num_layers=None):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.loader = loader  # kept for lazily-built extras (e.g. the MTP draft head)
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
        with sp.region("head.norm"):
            x = ttnn.slice(x, [0, 0, T - 1, 0], [1, 1, T, self.args.dim])
            x = self.final_norm.forward(x)
            x = ttnn.reshape(x, [1, self.args.dim])
        with sp.region("head.lm_head"):
            logits = ttnn.linear(x, self.lm_head_w)  # [1, vocab]
        with sp.region("head.d2h"):
            return from_tt(logits, self.mesh_device).reshape(1, 1, self.args.vocab_size)

    def _head_all(self, x, T, last_n=None):
        """Like _head but returns logits for EVERY position (or the last ``last_n``), not just the last
        token. Used by eval loglikelihood scoring (evaluation/), which needs the model's distribution at
        each continuation position. ``last_n`` slices to the final ``last_n`` positions BEFORE the
        248k-vocab matmul so the compute + D2H transfer scale with the (short) continuation, not the full
        prompt. Returns torch [S, vocab] where S = min(T, last_n) (or T if last_n is None)."""
        S = T if last_n is None or last_n >= T else last_n
        if S != T:
            x = ttnn.slice(x, [0, 0, T - S, 0], [1, 1, T, self.args.dim])
        x = self.final_norm.forward(x)
        x = ttnn.reshape(x, [S, self.args.dim])
        logits = ttnn.linear(x, self.lm_head_w)  # [S, vocab]
        return from_tt(logits, self.mesh_device).reshape(S, self.args.vocab_size)

    def forward_prefill_all_logits(self, input_ids: torch.Tensor, last_n: int | None = None):
        """Prefill returning per-position logits [S, vocab] (S = min(T, last_n), or T). Additive eval
        entry point — existing callers (forward/_prefill_single) are unchanged. Single-shot only, so the
        prompt must fit a single-shot prefill (bounded by max_seq_len; MMLU-style prompts are short).
        Each call is self-contained: caches are (re)allocated and the gated-delta conv left-pad reset, so
        successive independent sequences don't leak state into one another."""
        T = input_ids.shape[1]
        self.max_seq = self.args.max_seq_len
        if getattr(self, "caches", None) is None or not getattr(self, "_pf_caches_ready", False):
            self.caches = [self._alloc_cache(layer) for layer in self.layers]
            self._pf_caches_ready = True
        else:
            self._reset_linear_state()  # fresh sequence: clear the gated-delta conv left-pad
        self.pos = T
        x = self._embed(input_ids, T)
        cos, sin = precompute_rope(T, self.args.rotary_dim, self.args.rope_theta, self.mesh_device)
        for layer, cache in zip(self.layers, self.caches):
            x = layer.forward_prefill(x, cos, sin, cache)
        return self._head_all(x, T, last_n=last_n)

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
            cache = {
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
            if _CONV_ADDCHAIN:
                # decode add-chain history: K-1 persistent [1, conv_dim] row buffers (stable address ->
                # in-place shift is trace-safe). Populated from conv_state at start_decode; see gated_delta.
                cache["conv_rows"] = [
                    ttnn.zeros(
                        [1, a.lin_conv_dim], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
                    )
                    for _ in range(a.conv_kernel_size - 1)
                ]
            return cache
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
    # Set by a caller that intends to prime the MTP head (see prime_mtp) BEFORE it calls a prefill;
    # the prefill then leaves the per-position post-final-norm hidden in _prefill_hidden.
    _keep_prefill_hidden = False
    _prefill_hidden = None
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
        with prof.phase(self.mesh_device, "embed"), sp.region("embed"):
            x = self._embed(input_ids, T)
        with prof.phase(self.mesh_device, "rope"), sp.region("rope"):
            cos, sin = precompute_rope(T, self.args.rotary_dim, self.args.rope_theta, self.mesh_device)
        for layer, cache in zip(self.layers, self.caches):
            x = layer.forward_prefill(x, cos, sin, cache)
        if self._keep_prefill_hidden:
            # The per-position POST-final-norm hidden, for prime_mtp. This is the only moment it
            # exists: _head slices to the last row BEFORE the norm (a T-fold lm_head + D2H saving),
            # so nothing downstream can reconstruct it. Opt-in because it is dead weight (one T-row
            # RMSNorm and a [1,1,T,dim] tensor) for any request that will not speculate.
            with sp.region("prefill.keep_hidden"):
                self._prefill_hidden = self.final_norm.forward(x)
        with prof.phase(self.mesh_device, "head"), sp.region("head"):
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
        # _prefill_single stashed a hidden for the FIRST CHUNK only, and the later chunks' hiddens were
        # never materialised — priming off that would feed the head a truncated prompt while claiming
        # the whole one. Drop it: prime_mtp then no-ops and long prompts keep today's behaviour.
        self._prefill_hidden = None
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
        a prompt > max_seq_len can't run at all — fixed KV cache). Leaves caches + self.pos for decode.

        ACCURACY: the traced gated-delta prefill now runs the SAME fp32/HiFi4 ttnn recurrence as eager
        (QWEN36_TRACED_FP32_RECURRENCE=1, default) — trace capture only forbids host writes (zeros/
        fills), not the recurrence's in-graph matmul/eltwise outputs, so no bf16 kernel is needed.
        MEASURED single-bucket traced vs eager: PCC 1.00000 (was the old bf16-kernel caveat).
        MULTI-BUCKET CAVEAT: capturing several bucket lengths can corrupt the larger trace once the
        pinned per-trace intermediate footprint exceeds free DRAM (the real "bucket-256 wedge"); use a
        single bucket, or pool the recurrence intermediates (follow-up), until that lands. Opt-in in
        demo/server.py via QWEN36_SERVER_PREFILL_TRACE."""
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
        if self._keep_prefill_hidden:
            # Same stash as _prefill_single, sliced to the REAL length first: rows T..B-1 are padding.
            # Kept in step with the eager path so enabling prefill tracing cannot silently turn
            # MTP priming off.
            real = ttnn.slice(self._pf_out[B], [0, 0, 0, 0], [1, 1, T, self.args.dim])
            self._prefill_hidden = self.final_norm.forward(real)
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
        # Speculative decode samples on the HOST (from verify's per-row candidates), so it needs the
        # same seed and a fresh generated-token set per request. Reset both here — enable_sampling is
        # the per-request entry point — for the greedy tail too, so a mode switch cannot leak state.
        self._samp_seed = int(seed)
        self._mtp_rng = torch.Generator().manual_seed(int(seed))
        self._mtp_gen = set()
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
            self._samp_offsets_f = None  # rebuilt lazily from the new offsets (see _samp_offsets_f32)
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
        # Host-side copy of the truncation params: speculative decode applies them on the host (over
        # verify's candidates) rather than in `ttnn.sampling`, so it needs the python values.
        self._samp_params = (float(temperature), int(k), float(p), float(self._presence_penalty))
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
        if _CONV_ADDCHAIN:
            # seed the decode conv add-chain's history rows from the prefill-filled conv_state (once,
            # eager). After this, each decode step reads/shifts conv_rows in place; conv_state is unused.
            for layer, cache in zip(self.layers, self.caches):
                if layer.is_linear:
                    layer.mixer.sync_conv_rows(cache)

    def setup_decode_batch(self, first_token_ids):
        """Give the single-user-prefilled caches a batch axis (B slots) and seed B-user decode state,
        then drive with decode_forward_logits() (returns [B, vocab]; host samples per user). This
        SYNCHRONIZED-batch path replicates the current prefilled state into B identical slots (all users
        continue the same prefill) — the validation/first-cut path. Continuous batching (per-slot prefill,
        independent arrival) is a follow-on that writes each slot from its own prefill instead of repeat.

        first_token_ids: length-B list of the per-user first decode token ids. Positions all start at
        self.pos. Caches: KV -> [B,n_kv,max_seq,hd], recurrent_state -> [B,Vh,Dk,Dv], conv_rows ->
        [B,conv_dim]. B=1 reproduces the single-user state."""
        B = len(first_token_ids)
        if _CONV_ADDCHAIN:  # sync add-chain rows from prefilled conv_state BEFORE batching (as start_decode)
            for layer, cache in zip(self.layers, self.caches):
                if layer.is_linear:
                    layer.mixer.sync_conv_rows(cache)
        for cache in self.caches:
            if isinstance(cache, list):  # attention KV [k, v]
                cache[0] = ttnn.repeat(cache[0], ttnn.Shape([B, 1, 1, 1]))
                cache[1] = ttnn.repeat(cache[1], ttnn.Shape([B, 1, 1, 1]))
            else:  # gated-delta state
                cache["recurrent_state"] = ttnn.repeat(cache["recurrent_state"], ttnn.Shape([B, 1, 1, 1]))
                if "conv_rows" in cache:
                    cache["conv_rows"] = [ttnn.repeat(r, ttnn.Shape([B, 1])) for r in cache["conv_rows"]]
        mk = lambda t, dt: to_tt(t, self.mesh_device, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.t_tok = mk(torch.tensor(list(first_token_ids), dtype=torch.int32), ttnn.uint32)
        self.t_curpos = mk(torch.full((B,), self.pos, dtype=torch.int32), ttnn.int32)
        self.t_ropepos = mk(torch.full((B,), self.pos, dtype=torch.int32), ttnn.uint32)

    # ── vLLM CONTINUOUS BATCHING (V2, per-slot) ───────────────────────────────────
    def alloc_batch_caches(self, batch_size):
        """Allocate [B,...] per-slot caches for continuous batching, plus a [1,...] scratch used to run
        the validated single-request prefill before copying its state into a slot. Rows are decode slots
        (0..B-1); dead slots (no active request) just hold zeros. Call once at batch setup."""
        a = self.args
        self.batch_size = batch_size
        self.max_seq = a.max_seq_len
        self._scratch_caches = [self._alloc_cache(layer) for layer in self.layers]
        self._pf_caches_ready = True  # scratch is the prefill target; _prefill_single reuses + resets it
        z = lambda shape: ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        self.caches = []
        for layer in self.layers:
            if layer.is_linear:
                c = {"recurrent_state": z([batch_size, a.lin_num_v_heads, a.lin_head_k_dim, a.lin_head_v_dim])}
                if _CONV_ADDCHAIN:
                    c["conv_rows"] = [z([batch_size, a.lin_conv_dim]) for _ in range(a.conv_kernel_size - 1)]
                self.caches.append(c)
            else:
                shape = [batch_size, a.n_kv_heads, self.max_seq, a.head_dim]
                self.caches.append([z(shape), z(shape)])
        self.slot_pos = [0] * batch_size

    def prefill_into_slot(self, input_ids, slot):
        """Prefill ONE request (torch [1,T]) into batch row `slot`, reusing the validated B=1 prefill:
        run it into the [1,...] scratch, then copy K/V + GDN state into row `slot` of the [B,...] caches.
        Returns last-token host logits [1,1,vocab] (the plugin host-samples the first token). Positions
        for this slot recorded in self.slot_pos[slot].

        VERIFY-ON-DEVICE: the per-slot writes below. KV uses ttnn.fill_cache(dst[B,...], src[1,...], slot)
        (its native batch-index write). The GDN recurrent_state + conv_rows sliced writes are the ones to
        confirm/fix on-device (fill_cache may not accept the [B,Vh,Dk,Dv] / [B,conv_dim] shapes)."""
        batch = self.caches
        self.caches = self._scratch_caches  # _prefill_single writes here (+ resets scratch linear state)
        logits = self._prefill_single(input_ids)  # fills scratch [1,...], sets self.pos = T
        T = self.pos
        if _CONV_ADDCHAIN:  # sync scratch conv_state -> scratch conv_rows [1,conv_dim] (as start_decode does)
            for layer, sc in zip(self.layers, self._scratch_caches):
                if layer.is_linear:
                    layer.mixer.sync_conv_rows(sc)
        self.caches = batch
        for bl, sc in zip(self.caches, self._scratch_caches):
            if isinstance(bl, list):  # attention KV: native fill_cache batch-index write
                ttnn.fill_cache(bl[0], sc[0], slot)
                ttnn.fill_cache(bl[1], sc[1], slot)
            else:  # GDN recurrent + conv history — VERIFY these device ops on-device
                ttnn.fill_cache(bl["recurrent_state"], sc["recurrent_state"], slot)
                if "conv_rows" in bl:
                    D = self.args.lin_conv_dim
                    for i in range(self.args.conv_kernel_size - 1):
                        dst = ttnn.reshape(bl["conv_rows"][i], [self.batch_size, 1, 1, D])
                        src = ttnn.reshape(sc["conv_rows"][i], [1, 1, 1, D])
                        ttnn.fill_cache(dst, src, slot)
        self.slot_pos[slot] = T
        return logits

    def setup_batch_decode(self, first_token_ids):
        """Seed batched decode from the per-slot prefills (self.caches already [B,...] populated by
        prefill_into_slot). first_token_ids: length-B list of each slot's host-sampled first token
        (dead slots pass any id, e.g. 0). Per-slot positions come from self.slot_pos. Allocates the
        stable-address device state tensors; after this, drive each step with set_decode_state([B],[B])
        + decode_forward_logits() -> [B, vocab] (host sampling)."""
        B = self.batch_size
        mk = lambda t, dt: to_tt(t, self.mesh_device, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.t_tok = mk(torch.tensor(list(first_token_ids), dtype=torch.int32), ttnn.uint32)
        self.t_curpos = mk(torch.tensor(self.slot_pos, dtype=torch.int32), ttnn.int32)
        self.t_ropepos = mk(torch.tensor(self.slot_pos, dtype=torch.int32), ttnn.uint32)
        # Per-slot presence mask for batched repetition control, built once when the sampling tail is
        # active so the captured graph includes the presence ops. t_presence_b accumulates generated
        # tokens per slot (reset on slot reuse); the penalty strength is a uniform baked scalar (below).
        self._presence_on_batch = self.sampling is not None and os.environ.get("QWEN36_BATCH_PRESENCE", "1") != "0"
        if self._presence_on_batch:
            V = self.args.vocab_size
            # Per-SLOT presence mask (each request's own generated tokens). The penalty STRENGTH is a
            # uniform scalar baked into the trace — per-request strength would need a W-broadcast tensor
            # multiply (unreliable in TILE); a server-wide default is simpler and fixes the small-model
            # repetition loops. Default 1.5 matches demo/server; QWEN36_PRESENCE_PENALTY overrides (0=off).
            self._presence_penalty_batch = float(os.environ.get("QWEN36_PRESENCE_PENALTY", "1.5"))
            if self._presence_penalty_batch <= 0:
                self._presence_on_batch = False  # disabled -> skip the presence ops entirely
            else:
                self.t_presence_b = ttnn.zeros(
                    [1, 1, B, V], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
                )
                # FAST marking path (opt-in): a per-row iota [1,1,B,V] lets us build the one-hot with a
                # SINGLE column-broadcast eq (iota_full[...,V] eq token[...,1] — only the last dim
                # broadcasts, the standard/safe ttnn pattern) instead of the B-way eq loop + concat. This
                # is NOT the mutual/outer broadcast that deadlocks the device. Default OFF until verified.
                self._presence_fast = os.environ.get("QWEN36_PRESENCE_FAST", "0") != "0"
                if self._presence_fast:
                    iota = torch.arange(V, dtype=torch.int32).view(1, 1, 1, V).expand(1, 1, B, V).contiguous()
                    self._samp_iota_b = to_tt(iota, self.mesh_device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT)

    def set_decode_state(self, token_ids, positions):
        """Overwrite the batched decode state (token + absolute position, per slot) from the plugin's
        per-step inputs, IN THIS MODEL'S SLOT ORDER. Makes batched decode fully plugin-driven: no
        reliance on internal position advancing, so a slot reused by a new request mid-stream gets that
        request's fresh position (not a stale advanced one), and vLLM's compaction is irrelevant.
        token_ids / positions: length-B lists (dead slots pass anything)."""
        mk = lambda t, dt: to_tt(t, self.mesh_device, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT)
        tok = torch.as_tensor(list(token_ids), dtype=torch.int32).flatten()
        pos = torch.as_tensor(list(positions), dtype=torch.int32).flatten()
        ttnn.copy(mk(tok, ttnn.uint32), self.t_tok)  # in-place into the stable-address decode buffers
        ttnn.copy(mk(pos, ttnn.int32), self.t_curpos)
        ttnn.copy(mk(pos, ttnn.uint32), self.t_ropepos)

    def set_sampling_params_batch(self, top_ks, top_ps, temps, seeds):
        """Write PER-SLOT sampling params in place into the 32-lane sampling buffers (trace-safe: the
        captured graph reads these exact addresses each replay, so per-request params take effect with
        no re-capture — the batched analogue of set_decode_state). All lists are length nslot (lane b =
        slot b); lanes past nslot are padded harmlessly. temp<=0 is greedy-equivalent (top_k=1, temp=1
        -> argmax over the topk candidates). `seeds` are the ABSOLUTE per-slot values for THIS step (the
        caller folds the position in so each request's RNG differs AND advances per step; this replaces
        the single global seed)."""
        B, nslot, K = self.SAMP_USERS, self.batch_size, self.SAMP_K
        ks, ps, ts, ss = [], [], [], []
        for i in range(nslot):
            t = temps[i]
            if t is None or float(t) <= 0.0:  # greedy-equivalent lane
                ks.append(1)
                ts.append(1.0)
            else:
                k = int(top_ks[i]) if top_ks[i] else K
                ks.append(k if 0 < k <= K else K)
                ts.append(1.0 / float(t))  # ttnn.sampling scales by 1/T
            p = top_ps[i]
            ps.append(min(max(float(p) if p is not None else 1.0, 0.0), 1.0))
            ss.append(int(seeds[i]) & 0x7FFFFFFF)
        ks += [K] * (B - nslot)
        ps += [1.0] * (B - nslot)
        ts += [1.0] * (B - nslot)
        ss += [0] * (B - nslot)
        mk = lambda arr, td, dt: to_tt(
            torch.tensor(arr, dtype=td), self.mesh_device, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn.copy(mk(ks, torch.int32, ttnn.uint32), self.t_k)
        ttnn.copy(mk(ps, torch.float32, ttnn.bfloat16), self.t_p)
        ttnn.copy(mk(ts, torch.float32, ttnn.bfloat16), self.t_temp)
        ttnn.copy(mk(ss, torch.int32, ttnn.uint32), self.t_seed)

    def reset_presence_slots(self, slots):
        """Clear the presence-mask rows for `slots` (a new request took the slot -> empty generated-token
        history). Done EAGERLY outside the trace via an in-place multiply by a full-width keep-mask (0 for
        the reused rows, 1 elsewhere). Full [1,1,nslot,V] (not a [.,1] W-broadcast, which HANGS the device
        here) — occasional (only on slot reuse), so materializing the host tensor is fine."""
        if getattr(self, "t_presence_b", None) is None or not slots:
            return
        nslot, V = self.batch_size, self.args.vocab_size
        if getattr(self, "_presence_fast", False):  # cheap [.,1] column-broadcast keep-mask
            keep = torch.ones(nslot, dtype=torch.float32)
            for s in slots:
                if 0 <= s < nslot:
                    keep[s] = 0.0
            keep_t = to_tt(keep.view(1, 1, nslot, 1), self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        else:  # SAFE (default): full-width [1,1,nslot,V] keep-mask (no W-broadcast)
            keep = torch.ones(1, 1, nslot, V, dtype=torch.float32)
            for s in slots:
                if 0 <= s < nslot:
                    keep[0, 0, s, :] = 0.0
            keep_t = to_tt(keep, self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.multiply(self.t_presence_b, keep_t, output_tensor=self.t_presence_b)

    def _rope_for(self, rope_pos):
        """Index the RoPE tables at the current position(s) -> (cos, sin) of [1,1,B,rotary_dim].
        rope_pos: [B] per-user positions (B=1 single-user). The B users become B positions along the
        SDPA/mixer S axis, so per-user RoPE is exactly prefill's per-token cos/sin generalized."""
        rd = self.args.rotary_dim
        B = rope_pos.shape[0]
        cos = ttnn.reshape(ttnn.embedding(rope_pos, self.cos_table, layout=ttnn.TILE_LAYOUT), [1, 1, B, rd])
        sin = ttnn.reshape(ttnn.embedding(rope_pos, self.sin_table, layout=ttnn.TILE_LAYOUT), [1, 1, B, rd])
        return cos, sin

    def _decode_hidden(self):
        """Shared decode body: read the device-resident token(s) + position(s), run
        embed -> layers -> final norm, and return the post-norm hidden state [B, dim]. Used by both
        the self-contained on-device-select path (_decode_graph) and the logits-returning path
        (decode_forward_logits). Does NOT advance positions — the caller does, after consuming x."""
        B = self.t_tok.shape[0]  # decode batch (B independent users, one token each; B=1 single-user)
        with sp.region("dec.rope_idx"):
            cos, sin = self._rope_for(self.t_ropepos)
        with sp.region("dec.embed"):
            x = ttnn.embedding(self.t_tok, self.embed_weight, layout=ttnn.TILE_LAYOUT)
            x = ttnn.reshape(x, [1, 1, B, self.args.dim])
        for layer, cache in zip(self.layers, self.caches):
            x = layer.forward_decode(x, cos, sin, cache, self.t_curpos)
        with sp.region("dec.final_norm"):
            x = self.final_norm.forward(x)
        return ttnn.reshape(x, [B, self.args.dim])

    def _decode_graph(self):
        """Self-contained, traceable decode step: read device-resident token + position, run
        embed -> layers -> norm -> lm_head -> argmax, write the next token back into self.t_tok, and
        advance the position tensors on device. No host work / no full-logit readback."""
        x = self._decode_hidden()
        with sp.region("dec.select"):
            self._select_token(x)  # lm_head -> argmax (greedy) or topk+sampling; writes self.t_tok
        with sp.region("dec.pos_advance"):
            ttnn.plus_one(self.t_curpos)  # advance position on device (for KV write + sdpa)
            ttnn.plus_one(self.t_ropepos)  # advance RoPE-table index on device

    # ================================================================= MTP / speculative decode
    # Round structure (gamma drafts -> K = gamma+1 verify rows), see MTP.md:
    #   state: the backbone has consumed positions < pos; `t_tok` is the token AT `pos`, and `h_prev`
    #   is the backbone hidden that predicted it.
    #   draft : d_j = MTP(h, embed(tok)) for j=1..gamma, chaining the head's PRE-mtp.norm output.
    #   verify: run the backbone over [t_tok, d_1..d_gamma] at positions [pos .. pos+gamma] -> K rows
    #           of logits. Row i's argmax a_i is the TRUE token following the first i+1 of them.
    #   accept: a_0 is always correct; additionally accept while d_{i+1} == a_i. n accepted tokens.
    #   commit: advance the recurrent state by n (commit_verify) and pos by n. Rejected attention KV
    #           and MTP KV sit past the new pos and are overwritten next round.

    def build_mtp_head(self, gamma=None):
        """Build the MTP draft head + its own KV cache. Requires a checkpoint that ships `mtp.*`."""
        from models.demos.qwen3_6_a3b.tt.mtp import TtMtpHead

        if not self.loader.has_mtp():
            raise RuntimeError("checkpoint has no mtp.* head; speculative decode unavailable")
        self.mtp_gamma = int(os.environ.get("QWEN36_MTP_GAMMA", "2")) if gamma is None else int(gamma)
        if getattr(self, "mtp", None) is not None:
            # gamma-only change: never rebuild the head (845M params) just to sweep gamma
            self.mtp_stats = {"rounds": 0, "tokens": 0, "accepted": 0, "drafted": 0}
            self._mtp_hidden = None
            return self.mtp
        self.mtp = TtMtpHead(self.mesh_device, self.args, self.loader, self.embed_weight, self.lm_head_w)
        # self.max_seq is only set by a prefill; fall back so the head can be built up front
        max_seq = getattr(self, "max_seq", self.args.max_seq_len)
        shape = [1, self.args.n_kv_heads, max_seq, self.args.head_dim]
        self.mtp_kv = [
            ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device) for _ in range(2)
        ]
        self._mtp_hidden = None  # backbone hidden that predicted the pending token
        self.mtp_stats = {"rounds": 0, "tokens": 0, "accepted": 0, "drafted": 0}
        return self.mtp

    def prime_mtp(self, input_ids: torch.Tensor, first_id: int) -> int:
        """Prime the MTP head's own KV cache over the prompt. Returns the number of primed rows
        (0 = did not run). Requires ``_keep_prefill_hidden`` set before the prefill, and the head built.

        MEASURED RESULT FIRST: priming makes drafting WORSE, so it is default-off in the server
        (QWEN36_MTP_PRIME=0). Paired ABBA A/B at 40L, gamma=2, 1024-token prompt, 250 rounds per arm,
        one model and one captured trace set shared by both arms:

            cold cache   2.608 tok/round   19.60 ms/token
            primed       2.252 tok/round   23.41 ms/token     -0.356 tok/round, 0.837x

        Three runs (40/250 rounds, 40/992 primed rows, both orderings) all put priming behind, and the
        ABBA control ruled out run-order drift -- the LAST block, cold, was the best of the four, so
        ordering had been masking part of the effect rather than manufacturing it. Notably the deficit
        did not grow with prompt length: 992 primed rows cost the same as 40, which is the opposite of
        what "the head needs more context" predicts.

        WHY IT LOSES -- and why the obvious reasoning had it backwards. The tempting argument is that a
        cold cache leaves the first draft attending over zero rows, and that those rows steal softmax
        mass from the one genuine signal (the ``(h_i, t_{i+1})`` pair arriving through ``fc``). The
        dilution is real; the CONSEQUENCE is the opposite of harmful. Every K/V row below `pos` is
        EXACTLY zero, so every score is exactly zero and the softmax spreads near-uniformly over ~pos
        zero-valued V rows: the attention output is attenuated toward zero, roughly as 1/pos. A cold
        head therefore collapses to ``fc(h_i, t_{i+1}) + MoE`` with its attention branch effectively
        switched off -- and that drafts BETTER than the head with real context. Which makes sense once
        stated: h_i is the post-final-norm backbone hidden, so it already carries the entire prompt
        through 40 layers. The head's single attention layer has little left to add, and evidently adds
        noise. Priming switches that branch back on, and costs 16%.

        Corollary worth knowing even though priming is off: the cold head is not a position-invariant
        baseline either, since its attention attenuation scales with pos. It behaves differently at
        pos=10 than at pos=1000.

        SO WHAT IS THIS STILL DOING HERE. It is the reproduction harness for the finding
        (bench_mtp.py --prime-ab) and the answer to an obvious "did you try priming the head?". It is
        also safe to leave in: draft quality is a THROUGHPUT property only -- a rejected draft is
        discarded and the emitted token comes from the backbone's own distribution -- so this can never
        change what the model emits, and it is eager, prefill-time, outside every captured trace.

        MECHANICS. The head is one full-attention layer with a PRIVATE cache (``self.mtp_kv``,
        allocated as ``ttnn.zeros`` in build_mtp_head) that nothing else ever writes -- the backbone's
        prefill fills the backbone's caches. This runs ``TtMtpHead.forward_prefill`` over the prompt for
        its KV-cache side effect alone.

        POSITIONS — the part that is easy to get wrong. Per tt/mtp.py the head's row i consumes
        ``(h_i, embed(t_{i+1}))`` and predicts t_{i+2}. But the ROW INDEX and the KV/RoPE POSITION the
        decode path gives it differ by one: ``_m_pos_*`` is shared with verify (which needs true
        absolute backbone positions), and the draft chain slices entry j of it, so a draft whose pair is
        contract row ``pos-1+j`` is stored at head position ``pos+j``. That offset is invisible today
        because a cold cache has no other rows for it to be relative to — but the moment anything else
        is in the cache, priming must adopt the same convention or every relative distance is off by
        one. So contract row i is written at head position i+1, and head position 0 gets a duplicate of
        contract row 0 as filler (one real-but-repeated row, versus a zero row that would attract
        softmax mass).

        We can prime contract rows 0..T-1, i.e. head positions 1..T:
          * rows 0..T-2 come entirely from the prompt;
          * row T-1 is ``(h_{T-1}, t_T)`` where t_T is the token the prefill just produced — passed in
            as `first_id`. Priming it matters: the head's first write of its own is at position pos+1
            = T+1 (round 1 is a K=1 backbone seed step that does not run the head at all), so skipping
            row T-1 would leave a zero row at position T, immediately adjacent to the first query,
            where RoPE weights it most. With it, primed rows and drafted rows are contiguous.

        SAFETY. Draft quality is a THROUGHPUT property only — a rejected draft is discarded and the
        emitted token comes from the backbone's own distribution — so this cannot change what the model
        emits, only how fast. It is eager and prefill-time, outside every captured trace, so it needs no
        re-capture and cannot interact with the trace-capture hazard. Stale rows past T are left alone,
        exactly as before: decode overwrites them as it advances.
        """
        h, self._prefill_hidden = self._prefill_hidden, None  # consume: never prime off a stale hidden
        if h is None or getattr(self, "mtp", None) is None:
            return 0
        T, dim = int(input_ids.shape[1]), self.args.dim
        S = T + 1  # filler row + contract rows 0..T-1
        if T < 1 or S > int(self.mtp_kv[0].shape[2]):
            return 0
        with sp.region("mtp.prime"):
            # hidden: [h_0, h_0, h_1, ..., h_{T-1}] -- h_0 duplicated as the position-0 filler
            hs = ttnn.concat([ttnn.slice(h, [0, 0, 0, 0], [1, 1, 1, dim]), h], dim=2)
            # tokens: [t_1, t_1, ..., t_{T-1}, t_T] -- the SHIFTED half, ending in the prefill's output
            tok = torch.cat(
                [
                    input_ids[:, 1:2] if T > 1 else torch.tensor([[first_id]]),
                    input_ids[:, 1:T],
                    torch.tensor([[first_id]]),
                ],
                dim=1,
            )
            emb = self._embed(tok, S)
            cos, sin = precompute_rope(S, self.args.rotary_dim, self.args.rope_theta, self.mesh_device)
            # Output discarded: forward_prefill is run for its KV-cache side effect alone (its logits
            # would predict t_2..t_{T+1}, and we already know all but the last).
            self.mtp.forward_prefill(hs, emb, cos, sin, self.mtp_kv)
        self.mtp_primed = S
        return S

    def setup_mtp_decode(self):
        """Allocate the persistent, stable-address buffers a captured speculative round needs.

        Trace capture forbids HOST writes, not device allocation — so everything the host has to
        refresh between rounds (the K candidate tokens, their positions, the seed hidden) must live in
        a buffer written via `ttnn.copy` OUTSIDE the trace, and everything the round produces must land
        in a buffer the host can read after `execute_trace`.

        IDEMPOTENT for a given K, and that is load-bearing: the captured traces bake in these exact
        buffer ADDRESSES, so re-allocating them (e.g. once per served request) would both leak device
        memory and silently invalidate every MTP trace. A repeat call with the same K therefore only
        resets the per-request state and keeps the traces; a call with a DIFFERENT K (a gamma sweep)
        releases the traces first, because they are shaped for the old K."""
        K = self.mtp_gamma + 1
        dim = self.args.dim
        if getattr(self, "_m_tok", None) is not None:
            if self._m_tok.shape[0] == K:
                self._mtp_cur = None
                self._mtp_gen = set()
                self._mtp_rng = torch.Generator().manual_seed(int(getattr(self, "_samp_seed", 0)))
                return K
            self.release_mtp_traces()  # K changed: the old traces are the wrong shape
        z = lambda shp, dt: ttnn.zeros(shp, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)
        self._m_tok = z([K], ttnn.uint32)  # slot 0 = pending token, 1..gamma = drafts
        self._m_pos_i32 = z([K], ttnn.int32)  # absolute positions (attention / kv writes)
        self._m_pos_u32 = z([K], ttnn.uint32)  # same, for the RoPE table lookup
        self._m_argmax = z([K], ttnn.uint32)  # verify's per-row argmax (what the host reads back)
        self._m_hidden = ttnn.zeros(
            [1, 1, 1, dim], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
        )
        self._m_vhidden = ttnn.zeros(
            [1, 1, K, dim], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
        )
        if self.mtp_gamma > 0:
            self._m_draft = z([self.mtp_gamma], ttnn.uint32)  # the gamma drafted ids
            self._m_dtok = z([1], ttnn.uint32)  # chain's current token
            self._m_dhidden = ttnn.zeros(
                [1, 1, 1, dim], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
            )
        self.mtp_verify_trace = None
        # Which tail the captured verify graph records: "greedy" (argmax only) or "sampling" (argmax +
        # per-row candidates). Fixed for as long as the trace lives, and there is exactly ONE verify
        # trace, because MEASURED: keeping a second verify trace resident while capturing another
        # CORRUPTS the second trace's output buffers -- the sampling tail's candidate rows came back as
        # ~1e9 garbage, which then fed an out-of-vocab id into ttnn.embedding and HUNG the device.
        # (Same family as the "multi-bucket prefill trace wedge" in forward_prefill_traced, and the
        # reason capture_decode_trace always releases the prior trace before capturing.)
        # The A/B that established it: greedy-capture-then-sampling-capture FAILS; the same sequence
        # with release_mtp_traces() in between, or with no greedy capture at all, PASSES.
        self.mtp_verify_tail = None
        self.mtp_draft_trace = None
        self.mtp_commit_traces = None
        self._mtp_cur = None  # re-seed the python-tracked pending token from t_tok on the next round
        # Speculative-SAMPLING state (unused on the greedy path). `_mtp_gen` is the sequence's
        # generated-token set for the presence penalty; the RNG is host-side and seeded so a given
        # `seed` reproduces a run exactly, as the on-device sampler's `t_seed` does for plain decode.
        self._mtp_gen = set()
        self._mtp_rng = torch.Generator().manual_seed(int(getattr(self, "_samp_seed", 0)))
        return K

    def release_mtp_traces(self):
        """Release every captured MTP trace (verify per tail, draft, commit 1..K) and forget them.

        The MTP traces are captured ONCE and reused across requests — unlike the plain decode trace,
        which `capture_decode_trace` re-captures per request — because their graphs read only the
        persistent `_m_*` buffers and the shared caches. So the only times they must go are a gamma
        change (wrong shape) and shutdown. Not releasing on a gamma change is the same
        trace-accumulation leak that OOMed the server."""
        if getattr(self, "mtp_verify_trace", None) is not None:
            ttnn.release_trace(self.mesh_device, self.mtp_verify_trace)
        self.mtp_verify_trace = None
        self.mtp_verify_tail = None
        if getattr(self, "mtp_draft_trace", None) is not None:
            ttnn.release_trace(self.mesh_device, self.mtp_draft_trace)
        self.mtp_draft_trace = None
        for tid in (getattr(self, "mtp_commit_traces", None) or {}).values():
            ttnn.release_trace(self.mesh_device, tid)
        self.mtp_commit_traces = None

    def _mtp_tail(self):
        """The tail the NEXT capture should record. Follows `enable_sampling`; once captured it is
        pinned in `self.mtp_verify_tail` (see setup_mtp_decode for why there is only ever one)."""
        return "sampling" if self.sampling is not None else "greedy"

    def mtp_tail_ready(self):
        """Can the captured verify trace serve the CURRENT sampling mode?

        A sampling round needs the candidate rows, so a greedy-tail trace cannot serve it (it would
        replay fine and then read a buffer the graph never wrote). A greedy round is happy with either
        tail: the sampling tail writes `_m_argmax` too, it just also pays the candidate topk. Returns
        False when the round must fall back to the (correct, slower) eager verify."""
        # getattr: the eager-only entry points (spec_decode_step straight after build_mtp_head, as the
        # correctness tests do) never call setup_mtp_decode, so these attributes may not exist.
        if getattr(self, "mtp_verify_trace", None) is None:
            return False
        return getattr(self, "mtp_verify_tail", None) == "sampling" or self.sampling is None

    def _mtp_cand_buffers(self, K):
        """Persistent candidate buffer the verify graph writes and the host reads after `execute_trace`.

        ONE float32 `[1, 1, 2K, nc*SAMP_K]` tensor: rows 0..K-1 hold each verify row's candidate
        logits, rows K..2K-1 hold their global vocab indices. Indices are < 2^24 so float32 represents
        them EXACTLY (vocab 248320, max offset+local 262143), and keeping a single dtype is deliberate:
        an int32 index tensor written from inside the CAPTURED graph came back holding float32 bit
        patterns, even though the identical int32 typecast/add/concat/copy chain is correct when run
        eagerly (verified model-free). One dtype also halves the readback to a single D2H.

        Allocated on the WARM run — `ttnn.zeros` is a host write, which trace capture forbids — and
        cached, exactly like gated_delta's `_verify_scratch`."""
        assert self.sampling is not None, "call enable_sampling() before capturing a sampling verify"
        w = self._samp_nc * self.SAMP_K
        key = (K, w)
        if getattr(self, "_mtp_cand_cache", None) is None:
            self._mtp_cand_cache = {}
        buf = self._mtp_cand_cache.get(key)
        if buf is None:
            buf = ttnn.zeros([1, 1, 2 * K, w], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
            self._mtp_cand_cache[key] = buf
        self._m_cand = buf
        return buf

    def _h2d(self, values, dtype, dst):
        """Write a small python/torch vector into an EXISTING device buffer.

        `copy_host_to_device_tensor` writes into the already-allocated `dst`; the obvious
        `ttnn.copy(to_tt(...), dst)` instead allocates a fresh device buffer per call and frees it
        again. A speculative round does ~9 of these, so at 40 layers that allocation churn is a real
        share of the round's host time — the same reason gemma4 keeps persistent traced inputs."""
        t = torch.as_tensor(list(values), dtype=torch.int32).flatten()
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT), dst)

    def _mtp_write_round(self, tokens, positions):
        """Host -> device refresh of the round's inputs. Runs BETWEEN trace replays, never inside."""
        self._h2d(tokens, ttnn.uint32, self._m_tok)
        self._h2d(positions, ttnn.int32, self._m_pos_i32)
        self._h2d(positions, ttnn.uint32, self._m_pos_u32)

    def sync_mtp_state(self):
        """Push the python-tracked pending token/position back into the device decode state, so a caller
        can hand off from speculative rounds to plain `decode_step_*` without a stale `t_tok`.

        ALSO re-seeds the decode conv add-chain. The two paths keep the gated-delta conv history in
        DIFFERENT places: verify reads `cache["conv_state"]` as its left-pad and `commit_verify` rolls
        that forward, while plain decode reads/shifts the `conv_rows` row buffers (`_CONV_ADDCHAIN`,
        default on). Without this, a spec -> plain-decode handoff silently runs on the conv window as
        it stood when `start_decode` last ran, i.e. it drops every token the spec rounds emitted."""
        if getattr(self, "_mtp_cur", None) is not None:
            self.set_decode_state([self._mtp_cur], [self.pos])
        if _CONV_ADDCHAIN:
            for layer, cache in zip(self.layers, self.caches):
                if layer.is_linear:
                    layer.mixer.sync_conv_rows(cache)

    def refresh_conv_state(self):
        """The inverse of `sync_conv_rows`: fold the decode add-chain's `conv_rows` back into
        `conv_state`, which is what the VERIFY conv reads as its left-pad. Needed when entering
        speculative rounds after plain decode steps have advanced `conv_rows` (after a prefill the two
        already agree, so this is a no-op there). One concat + copy per linear layer, once."""
        if not _CONV_ADDCHAIN:
            return
        for layer, cache in zip(self.layers, self.caches):
            if layer.is_linear and isinstance(cache, dict) and "conv_rows" in cache and "conv_state" in cache:
                ttnn.copy(ttnn.concat(cache["conv_rows"], dim=0), cache["conv_state"])

    def _mtp_verify_graph(self):
        """The traceable verify body: reads the persistent token/position buffers, writes the
        persistent argmax + hidden buffers. No host writes, no return value."""
        K = self._m_tok.shape[0]
        cos, sin = self._rope_for(self._m_pos_u32)
        x = ttnn.reshape(
            ttnn.embedding(self._m_tok, self.embed_weight, layout=ttnn.TILE_LAYOUT), [1, 1, K, self.args.dim]
        )
        for layer, cache in zip(self.layers, self.caches):
            x = layer.forward_verify(x, cos, sin, cache, self._m_pos_i32)
        x = self.final_norm.forward(x)
        ttnn.copy(x, self._m_vhidden)
        logits = ttnn.linear(ttnn.reshape(x, [K, self.args.dim]), self.lm_head_w)  # [K, vocab]
        am = ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1)
        ttnn.copy(ttnn.reshape(am, [K]), self._m_argmax)
        # The tail is decided by what this capture is FOR, not by the live sampling flag: a greedy
        # request replaying a sampling-tail trace must execute the very same graph.
        if (self.mtp_verify_tail or self._mtp_tail()) == "sampling":
            self._verify_candidate_tail(logits, K)

    def _verify_candidate_tail(self, logits, K):
        """Speculative SAMPLING needs the target DISTRIBUTION per verify row, not just its argmax.

        Emits each row's top-(nc*SAMP_K) (value, global index) pair -- the same candidate set the plain
        decode tail feeds to `ttnn.sampling` -- into two persistent buffers the host reads after the
        replay. The host then applies presence/temperature/top-k/top-p and runs the acceptance rule
        (`tt/mtp_sampling.py`), which is why the truncation parameters are NOT baked into the trace: a
        request can change temperature/top_k/top_p with no re-capture.

        Two things are deliberate:
          * the PER-ROW loop (slice -> pad -> reshape -> topk), not a single `[1,1,K*nc,CW]` reshape:
            that cross-row reshape in TILE layout is what scrambled candidates in the batched decode
            path, and it is the exact hazard `_select_token_batch` documents;
          * float32 END TO END, values and indices alike. The logits are bf16 out of the matmul so
            this adds no accuracy, but an int32 index tensor produced inside the CAPTURED graph came
            back holding float32 bit patterns (the same typecast/add/concat/copy chain is correct
            eagerly), and float32 holds every index exactly -- see `_mtp_cand_buffers`.
        """
        V, CW, Kk, nc = self.args.vocab_size, self.SAMP_CHUNK, self.SAMP_K, self._samp_nc
        cand = self._mtp_cand_buffers(K)
        off = self._samp_offsets_f32()
        lg = ttnn.reshape(logits, [1, 1, K, V])
        vrows, irows = [], []
        with sp.region("mtp.verify.topk"):
            for i in range(K):
                row = ttnn.reshape(ttnn.slice(lg, [0, 0, i, 0], [1, 1, i + 1, V]), [1, 1, 1, V])
                row = ttnn.pad(row, [(0, 0), (0, 0), (0, 0), (0, nc * CW - V)], value=self.SAMP_NEG)
                chunks = ttnn.reshape(row, [1, 1, nc, CW])  # power-of-2 width per row -> multicore topk
                vals, idxs = ttnn.topk(chunks, k=Kk, dim=-1, sorted=False, indices_tensor=self._samp_local_idx)
                gidx = ttnn.add(ttnn.typecast(idxs, ttnn.float32), off)  # local -> global vocab index
                vrows.append(ttnn.typecast(ttnn.reshape(vals, [1, 1, 1, nc * Kk]), ttnn.float32))
                irows.append(ttnn.reshape(gidx, [1, 1, 1, nc * Kk]))
        # one [1,1,2K,W] block: K value rows then K index rows -> a single copy and a single readback
        ttnn.copy(ttnn.concat(vrows + irows, dim=2), cand)

    def _samp_offsets_f32(self):
        """`_samp_offsets` as float32, built once. Keeps the [1,1,nc,SAMP_K] shape of `ttnn.topk`'s
        index output -- the add is elementwise per (chunk, rank), so flattening it first is a
        broadcasting error, not a reshape."""
        if getattr(self, "_samp_offsets_f", None) is None:
            self._samp_offsets_f = ttnn.typecast(self._samp_offsets, ttnn.float32)
        return self._samp_offsets_f

    def read_verify_candidates(self, K=None):
        """Host copy of the last verify's per-row candidates: (values [K, nc*SAMP_K] float32,
        indices [K, nc*SAMP_K] int64). Additive -- `verify_step_traced`'s return value is unchanged."""
        K = self._m_tok.shape[0] if K is None else K
        t = from_tt(self._m_cand, self.mesh_device).reshape(2 * K, -1).float()
        return t[:K], t[K:].round().to(torch.int64)

    def _mtp_draft_graph(self):
        """Traceable gamma-step draft chain, fully on device.

        Everything the chain needs per step already lives in a persistent buffer, so no host write and
        no per-step readback: the token is fed back with `ttnn.copy` into `_m_dtok`, the hidden with a
        copy into `_m_dhidden` (the PRE-`mtp.norm` state, which is what A5 measured), and the position
        is a slice of the round's `_m_pos_*` buffers. The gamma argmax results are concatenated once at
        the end into `_m_draft` — ttnn has no slice-assign, so accumulating into one buffer per step is
        not expressible; a single concat+copy is."""
        gamma, dim = self.mtp_gamma, self.args.dim
        drafts = []
        for j in range(gamma):
            pos_u = ttnn.slice(self._m_pos_u32, [j], [j + 1])
            pos_i = ttnn.slice(self._m_pos_i32, [j], [j + 1])
            cos = ttnn.reshape(
                ttnn.embedding(pos_u, self.cos_table, layout=ttnn.TILE_LAYOUT), [1, 1, 1, self.args.rotary_dim]
            )
            sin = ttnn.reshape(
                ttnn.embedding(pos_u, self.sin_table, layout=ttnn.TILE_LAYOUT), [1, 1, 1, self.args.rotary_dim]
            )
            out, pre = self.mtp.forward_decode(self._m_dhidden, self._m_dtok, cos, sin, self.mtp_kv, pos_i)
            logits = ttnn.linear(ttnn.reshape(out, [1, dim]), self.lm_head_w)  # [1, vocab]
            am = ttnn.reshape(ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1), [1])
            drafts.append(am)
            if j < gamma - 1:  # feed back for the next step; the last step needs neither
                ttnn.copy(am, self._m_dtok)
                ttnn.copy(pre, self._m_dhidden)
        ttnn.copy(drafts[0] if gamma == 1 else ttnn.concat(drafts, dim=0), self._m_draft)

    def capture_mtp_draft_trace(self):
        """Capture the draft chain. Prime `_m_dtok`/`_m_dhidden`/`_m_pos_*` first (as with verify): the
        warm+capture runs write the MTP head's OWN KV cache, so those writes must be idempotent with
        the first replay — they land at positions >= pos, which the next round overwrites anyway."""
        if self.mtp_draft_trace is not None:
            ttnn.release_trace(self.mesh_device, self.mtp_draft_trace)
            self.mtp_draft_trace = None
        self._mtp_draft_graph()  # warm: compile + build constants
        ttnn.synchronize_device(self.mesh_device)
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._mtp_draft_graph()
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        self.mtp_draft_trace = tid
        return tid

    def draft_step_traced(self, hidden, cur_token):
        """Replay the draft chain. `hidden` is [1,1,1,dim] (verify's accepted-prefix row); positions
        come from the already-written `_m_pos_*`. Returns the gamma drafted ids."""
        ttnn.copy(hidden, self._m_dhidden)
        self._h2d([cur_token], ttnn.uint32, self._m_dtok)
        ttnn.execute_trace(self.mesh_device, self.mtp_draft_trace, cq_id=0, blocking=False)
        return from_tt(self._m_draft, self.mesh_device).reshape(-1).tolist()

    def capture_mtp_verify_trace(self, tokens, positions):
        """Capture the verify pass. Verify is ~90% of a round's op count, so this is the bulk of the
        eager->traced win.

        `tokens`/`positions` MUST be the first round's real inputs. Two reasons:
          * a warm run before capture is load-bearing — it forces every lazily-built constant (MoE
            ROWSEL/sparsity, the ttl op compile, sharded configs) to exist BEFORE capture, since
            building one DURING capture is a host write and fatals;
          * the warm and capture runs both WRITE the attention KV cache, and the KV cache is
            deliberately outside `_trace_snapshot_tensors()` (snapshotting it would transiently double
            KV memory). So those writes must be IDEMPOTENT with the first real replay — same tokens at
            the same positions. Capturing with the buffers still zeroed writes token 0 at position 0
            and silently corrupts the prompt's first KV row, which then perturbs every verify row.
        """
        tail = self._mtp_tail()
        if self.mtp_verify_trace is not None:  # never keep two resident (see setup_mtp_decode)
            ttnn.release_trace(self.mesh_device, self.mtp_verify_trace)
            self.mtp_verify_trace = None
        self.mtp_verify_tail = tail  # the graph below records THIS tail
        self._mtp_write_round(tokens, positions)
        self._mtp_verify_graph()  # warm: compile + build constants, and prime the KV rows
        ttnn.synchronize_device(self.mesh_device)
        snap = [ttnn.clone(t) for t in self._trace_snapshot_tensors()]
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._mtp_verify_graph()
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        for src, dst in zip(snap, self._trace_snapshot_tensors()):
            ttnn.copy(src, dst)  # undo the warm/capture runs' effect on the accumulators
        self.mtp_verify_trace = tid
        self.mtp_verify_tail = tail
        return tid

    def verify_step_traced(self, tokens, positions):
        """Replay the verify trace for one round. Returns the K per-row argmax token ids (host)."""
        self._mtp_write_round(tokens, positions)
        ttnn.execute_trace(self.mesh_device, self.mtp_verify_trace, cq_id=0, blocking=False)
        return from_tt(self._m_argmax, self.mesh_device).reshape(-1).tolist()

    def _pos_tensor(self, positions, dtype=ttnn.uint32):
        return to_tt(
            torch.tensor(list(positions), dtype=torch.int32),
            self.mesh_device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def verify_forward(self, token_ids, positions):
        """Run the backbone over K tokens of ONE sequence at `positions`. Returns
        (logits [K, vocab], hidden [1,1,K,dim]). Leaves every recurrent state untouched — call
        `commit_verify(n)` afterwards with the accepted count."""
        K = len(token_ids)
        assert K == len(positions), (K, len(positions))
        ids = self._pos_tensor(token_ids)
        pos_i32 = self._pos_tensor(positions, dtype=ttnn.int32)
        cos, sin = self._rope_for(self._pos_tensor(positions))
        x = ttnn.reshape(ttnn.embedding(ids, self.embed_weight, layout=ttnn.TILE_LAYOUT), [1, 1, K, self.args.dim])
        for layer, cache in zip(self.layers, self.caches):
            x = layer.forward_verify(x, cos, sin, cache, pos_i32)
        x = self.final_norm.forward(x)
        logits = ttnn.linear(ttnn.reshape(x, [K, self.args.dim]), self.lm_head_w)  # [K, vocab]
        return logits, x

    def commit_verify(self, n):
        """Accept the first n verify tokens: advance the recurrent states and `pos` by n.

        Uses a captured trace when one exists. `n` varies per round and a trace is a fixed op sequence,
        so there is one trace PER possible accept count (1..K) rather than one parameterised trace —
        `n` cannot be a runtime input without device-side dynamic indexing."""
        tr = getattr(self, "mtp_commit_traces", None)
        if tr is not None and n in tr:
            ttnn.execute_trace(self.mesh_device, tr[n], cq_id=0, blocking=False)
        else:
            for layer, cache in zip(self.layers, self.caches):
                layer.commit_verify(cache, n)
        self.pos += n

    def _mtp_commit_graph(self, n):
        for layer, cache in zip(self.layers, self.caches):
            layer.commit_verify(cache, n)

    def capture_mtp_commit_traces(self):
        """Capture one commit trace per accept count 1..K (~120 device copies each at 40 layers, which
        is 12-24 ms of eager dispatch per round — worth capturing).

        Each capture MUTATES the recurrent/conv state, and successive captures would stack on top of
        each other, so every one is bracketed by a snapshot/restore of `_trace_snapshot_tensors()`
        (which covers recurrent_state, conv_state and conv_rows — but not the KV caches, by design).
        Requires a verify to have run first, so `_verify_pending` and the persistent conv buffer exist."""
        K = self._m_tok.shape[0]
        for tid in (getattr(self, "mtp_commit_traces", None) or {}).values():
            ttnn.release_trace(self.mesh_device, tid)
        self.mtp_commit_traces = {}
        # ONE snapshot for the whole loop (~30 MB at 40 layers): every warm run and every capture run
        # mutates the state, so each is followed by a restore from the same baseline.
        snap = [ttnn.clone(t) for t in self._trace_snapshot_tensors()]

        def restore():
            for src, dst in zip(snap, self._trace_snapshot_tensors()):
                ttnn.copy(src, dst)

        for n in range(1, K + 1):
            self._mtp_commit_graph(n)  # warm/compile
            restore()
            ttnn.synchronize_device(self.mesh_device)
            tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            self._mtp_commit_graph(n)
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
            restore()
            self.mtp_commit_traces[n] = tid
        return self.mtp_commit_traces

    def setup_mtp_traces(self):
        """Run the warm eager rounds a capture needs, then capture verify + draft + commit.

        Returns the token ids the warm rounds emitted — they are REAL generated output, so the caller
        must keep them (this is why capture cannot simply be hidden inside the first replay).

        Releases every MTP trace FIRST, so the whole set is captured from a clean slate. MEASURED: a
        verify trace captured while the previous round's draft + commit traces are still resident comes
        back CORRUPT (its candidate rows read as ~1e9 garbage, which then hangs the device by feeding
        an out-of-vocab id to ttnn.embedding); the identical sequence with everything released first is
        correct. Capturing draft and commit AFTER verify is fine -- it is capturing the big verify trace
        under pinned company that breaks. Same failure family as forward_prefill_traced's multi-bucket
        wedge, where the pinned per-trace intermediate footprint is what corrupts the larger trace.

        TWO warm rounds, not one: round 1 is the K=1 seed step (no draft, and a 1-row verify), and
        round 2 is the first true K-row verify, which is what creates the K-shaped verify scratch the
        commit traces slice their per-step state out of. Capture is then primed with real POSITIONS
        (>= self.pos): the warm and capture runs write the attention KV cache, which is deliberately
        outside `_trace_snapshot_tensors()`, and rows at positions >= pos are rewritten by the very
        next verify before anything reads them. Priming with the zeroed buffers instead would write
        position 0 and corrupt the prompt's first KV row, which nothing ever repairs."""
        self.release_mtp_traces()  # clean slate: nothing may be resident while verify is captured
        emitted = list(self.spec_decode_step())
        emitted += list(self.spec_decode_step())
        K = self.mtp_gamma + 1
        toks = [self._mtp_cur] * K
        positions = [self.pos + i for i in range(K)]
        self.capture_mtp_verify_trace(toks, positions)
        if self.mtp_gamma > 0:
            self._mtp_write_round(toks, positions)  # the traced draft chain slices these positions
            self.capture_mtp_draft_trace()
        self.capture_mtp_commit_traces()
        return emitted

    def set_mtp_sampler(self, sampler):
        """Override the acceptance rule in process (the env is the normal route). Used by bench_mtp to
        sweep rules without rebuilding the model: the captured verify trace is rule-INDEPENDENT — it
        only emits candidates — so only this host-side object changes."""
        self._mtp_sampler_cache = (getattr(self, "_samp_params", None), sampler)
        return sampler

    def mtp_divergence(self):
        """How far this run's emitted stream strayed from the target distribution.

        Returns (mean TV per emitted token, mean p of ACCEPTED drafts, fraction of accepted drafts the
        target rated < 0.1), or None on the greedy path. The exact rule must report TV == 0."""
        st = getattr(self, "mtp_stats", None) or {}
        if not st.get("accept_tests"):
            return None
        n_tok = max(st.get("tokens", 0), 1)
        n_acc = max(st.get("acc_p_n", 0), 1)
        return (
            st.get("tv_sum", 0.0) / n_tok,
            st.get("acc_p_sum", 0.0) / n_acc,
            st.get("acc_p_low", 0) / n_acc,
        )

    def mtp_acceptance(self):
        """(tokens_per_round, accepted/drafted, mean accept probability) since the stats were reset.
        Mean accept probability is None on the greedy path (no probabilities are computed there)."""
        st = getattr(self, "mtp_stats", None) or {}
        rounds = max(st.get("rounds", 0), 1)
        tests = st.get("accept_tests", 0)
        return (
            st.get("tokens", 0) / rounds,
            st.get("accepted", 0) / max(st.get("drafted", 0), 1),
            (st.get("accept_prob_sum", 0.0) / tests) if tests else None,
        )

    def mtp_draft(self, hidden, first_token, gamma):
        """Chain `gamma` draft steps from the backbone hidden `hidden` ([1,1,1,dim]) and the pending
        token `first_token`. Each step feeds back the head's PRE-`mtp.norm` output (what A5 measured).
        Returns the list of drafted token ids."""
        drafts, h, tok = [], hidden, first_token
        for j in range(gamma):
            p = self.pos + j
            cos, sin = self._rope_for(self._pos_tensor([p]))
            out, pre = self.mtp.forward_decode(
                h, self._pos_tensor([tok]), cos, sin, self.mtp_kv, self._pos_tensor([p], dtype=ttnn.int32)
            )
            tok = int(
                from_tt(ttnn.linear(ttnn.reshape(out, [1, self.args.dim]), self.lm_head_w), self.mesh_device)
                .reshape(-1)
                .argmax()
            )
            drafts.append(tok)
            h = pre
        return drafts

    def _mtp_sampler(self):
        """The acceptance rule + truncation config for this request (see tt/mtp_sampling.py). Cached
        per parameter tuple so a long generation does not rebuild it every round."""
        from models.demos.qwen3_6_a3b.tt.mtp_sampling import sampler_from_env

        params = getattr(self, "_samp_params", None)
        assert params is not None, "enable_sampling() must run before sampled speculative decode"
        cache = getattr(self, "_mtp_sampler_cache", None)
        if cache is None or cache[0] != params:
            self._mtp_sampler_cache = (params, sampler_from_env(*params))
        return self._mtp_sampler_cache[1]

    def _mtp_uniform(self):
        return float(torch.rand((), generator=self._mtp_rng))

    def _mtp_candidates(self, K, logits=None):
        """Per-row (values, indices) candidates for the acceptance rule.

        Traced verify (`logits is None`) reads the persistent candidate buffers the graph wrote; the
        eager path derives the SAME candidate width from the full-vocab logits with `torch.topk`, so
        both paths hand the rule identical inputs and the eager path stays a valid reference."""
        if logits is None:
            v, i = self.read_verify_candidates(K)
        else:
            lg = from_tt(logits, self.mesh_device).reshape(K, -1).float()
            w = min(self._samp_nc * self.SAMP_K, lg.shape[-1])
            v, idx = torch.topk(lg, w, dim=-1)
            i = idx.to(torch.int64)
        return [(v[r], i[r]) for r in range(K)]

    def spec_decode_step(self):
        """One speculative round. Returns the list of newly emitted token ids (1..gamma+1 of them).
        `self.t_tok` is left holding the last emitted token so the next round continues seamlessly.

        GREEDY (`self.sampling is None`) accepts a draft iff it equals the verify row's argmax.
        SAMPLING runs speculative sampling instead: the draft is accepted with probability p(draft)
        under the row's truncated target distribution and a rejection emits from the residual, which
        makes the emitted stream EXACTLY target-distributed (proof + host tests in tt/mtp_sampling.py).
        Both rules produce the same accept-count contract, so `commit_verify` is untouched."""
        gamma = self.mtp_gamma
        # The pending token is tracked in PYTHON. Reading it back off `t_tok` every round costs a
        # blocking D2H sync, and nothing in the verify path consumes t_tok/t_curpos (verify reads the
        # `_m_*` buffers). `sync_mtp_state()` pushes it back if the caller returns to plain decode.
        if getattr(self, "_mtp_cur", None) is None:
            self._mtp_cur = int(from_tt(self.t_tok, self.mesh_device).reshape(-1)[0])
        cur = self._mtp_cur
        if self._mtp_hidden is None:
            # first round: no backbone hidden yet for the pending token, so take one plain decode step.
            # Always eager — the captured trace is shaped for K = gamma+1 rows, not 1.
            logits, hid = self.verify_forward([cur], [self.pos])
            if self.sampling is None:
                a0 = int(from_tt(logits, self.mesh_device).reshape(1, -1)[0].argmax())
            else:  # sample it, exactly as a plain sampled decode step would
                sm = self._mtp_sampler()
                cv, ci = self._mtp_candidates(1, logits=logits)[0]
                pr, ids = sm.probs(cv, ci, penalised=self._mtp_gen)
                a0 = sm.sample(pr, ids, self._mtp_uniform())
                self._mtp_gen.add(a0)
            self.commit_verify(1)
            self._mtp_hidden = hid
            self._mtp_cur = a0
            self.mtp_stats["rounds"] += 1
            self.mtp_stats["tokens"] += 1
            return [a0]

        if getattr(self, "mtp_draft_trace", None) is not None:
            # positions must already be on device for the traced chain to slice them
            self._mtp_write_round([cur] * (gamma + 1), [self.pos + i for i in range(gamma + 1)])
            drafts = self.draft_step_traced(self._mtp_hidden, cur)
        else:
            drafts = self.mtp_draft(self._mtp_hidden, cur, gamma)
        toks = [cur] + drafts
        positions = [self.pos + i for i in range(len(toks))]
        # Gate on the TAIL, not just "a trace exists": a sampling round replaying a greedy-tail trace
        # would read a candidate buffer the graph never wrote. Falling back to eager verify is slow but
        # always correct.
        logits = None
        if self.mtp_tail_ready():
            # traced verify: bit-exact vs the eager path (test_mtp_trace) and ~90% of the round's ops
            a = self.verify_step_traced(toks, positions)
            hid = self._m_vhidden
        else:
            logits, hid = self.verify_forward(toks, positions)
            a = from_tt(logits, self.mesh_device).reshape(len(toks), -1).argmax(dim=-1).tolist()

        info = None
        if self.sampling is None:  # greedy: accept while the draft matches the row's argmax
            n = 1
            for i in range(gamma):
                if drafts[i] == a[i]:
                    n += 1
                else:
                    break
            emitted = a[:n]
        else:  # speculative SAMPLING (exact by default; see tt/mtp_sampling.py)
            sm = self._mtp_sampler()
            cands = self._mtp_candidates(len(toks), logits=logits)
            # kept for inspection: the per-row target candidates this round's decisions were made
            # from. Cheap (K x 256 host floats) and the only way a test can check "the emitted token
            # was in the row's support" on the EAGER path, where no candidate buffer is written.
            self._mtp_last_cands, self._mtp_last_drafts = cands, list(drafts)
            emitted, info = sm.resolve_round(cands, drafts, self._mtp_uniform, generated=self._mtp_gen)
            n = len(emitted)
            self._mtp_gen.update(emitted)
        self.commit_verify(n)
        # seed the next round from the accepted prefix's last hidden and last emitted token
        self._mtp_hidden = ttnn.slice(hid, [0, 0, n - 1, 0], [1, 1, n, self.args.dim])
        self._mtp_cur = emitted[-1]
        st = self.mtp_stats
        st["rounds"] += 1
        st["tokens"] += n
        st["accepted"] += n - 1
        st["drafted"] += gamma
        if info is not None:  # sampled-mode observability: how peaked the target actually was
            st["accept_prob_sum"] = st.get("accept_prob_sum", 0.0) + sum(info["accept_probs"])
            st["accept_tests"] = st.get("accept_tests", 0) + len(info["accept_probs"])
            st["bonus"] = st.get("bonus", 0) + int(info["bonus"])
            st["fallbacks"] = st.get("fallbacks", 0) + int(not info["bonus"])
            # Divergence bookkeeping: the exact per-position TV between what the rule emitted and the
            # target (0 for the exact rule by construction), plus how often a rule force-emitted a
            # draft the target rated UNLIKELY — the concrete thing a non-exact rule trades away.
            st["tv_sum"] = st.get("tv_sum", 0.0) + sum(info["tvs"])
            acc_p = info["accept_probs"][: info["accepted"]]  # loop breaks at the first rejection
            st["acc_p_sum"] = st.get("acc_p_sum", 0.0) + sum(acc_p)
            st["acc_p_n"] = st.get("acc_p_n", 0) + len(acc_p)
            st["acc_p_low"] = st.get("acc_p_low", 0) + sum(1 for x in acc_p if x < 0.1)
        return emitted

    def set_decode_tokens(self, token_ids):
        """Host->device write of the next input token(s) into the device-resident decode state.
        The continuous-batching counterpart to the on-device token write _select_token does for the
        standalone greedy/sampling path: under vLLM the host samples from the returned logits and
        feeds the chosen token(s) back here before the next decode_forward_logits() call.
        token_ids: an int or a length-B sequence of ints."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        t = torch.as_tensor(list(token_ids), dtype=torch.int32).flatten()
        src = to_tt(t, self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy(src, self.t_tok)  # in place, stable address (trace-safe)

    def decode_forward_logits(self, read_from_device: bool = True):
        """Run one decode step and return logits [B, vocab] WITHOUT selecting a token on device.
        This is the vLLM continuous-batching contract: vLLM's sampler consumes the logits (per-request
        temperature/top-p/penalties/logprobs/stops) and writes the chosen token(s) back via
        set_decode_tokens() before the next call. Positions still advance on device, exactly as in the
        greedy path. (B==1 today; A3 threads the batch axis.)

        read_from_device=True (default) returns HOST logits (D2H here). read_from_device=False returns
        the on-DEVICE ttnn logits tensor and defers the readback to the caller — the vLLM async-decode
        path (read_decode_output does an async cpu() so the D2H overlaps the next step's host work)."""
        x = self._decode_hidden()
        with sp.region("dec.lm_head"):
            logits = ttnn.linear(x, self.lm_head_w)  # [B, vocab]  (all rows kept; no last-token slice)
        with sp.region("dec.pos_advance"):
            ttnn.plus_one(self.t_curpos)  # advance position on device (for KV write + sdpa)
            ttnn.plus_one(self.t_ropepos)  # advance RoPE-table index on device
        self.pos += 1
        if not read_from_device:
            return logits  # on-device ttnn tensor; caller reads it (async D2H) later
        with sp.region("dec.d2h"):
            return from_tt(logits, self.mesh_device)

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
            with sp.region("head.lm_head"):
                logits = ttnn.linear(x, self.lm_head_w)  # device [1, vocab]
            with sp.region("head.argmax"):
                logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
                tok = ttnn.argmax(logits, dim=-1, keepdim=False)  # [1] uint32
                ttnn.copy(ttnn.to_layout(tok, ttnn.ROW_MAJOR_LAYOUT), self.t_tok)  # next token, in place
            return
        V, CW, K, B, nc = self.args.vocab_size, self.SAMP_CHUNK, self.SAMP_K, self.SAMP_USERS, self._samp_nc
        with sp.region("head.lm_head"):
            logits = ttnn.linear(x, self.lm_head_w)  # [1, vocab]  (1-row: lm_head stays cheap)
        logits = ttnn.reshape(logits, [1, 1, 1, V])
        with sp.region("head.samp.topk"):
            if self._presence_penalty > 0:  # discourage repeats: subtract penalty from already-seen tokens
                logits = ttnn.sub(logits, ttnn.multiply(self.t_presence, self._presence_penalty))
            logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, nc * CW - V)], value=self.SAMP_NEG)
            chunks = ttnn.reshape(logits, [1, 1, nc, CW])  # power-of-2 width per row -> multicore topk
            vals, idxs = ttnn.topk(chunks, k=K, dim=-1, sorted=False, indices_tensor=self._samp_local_idx)
        with sp.region("head.samp.sample"):
            gidx = ttnn.add(ttnn.typecast(idxs, ttnn.int32), self._samp_offsets, dtype=ttnn.int32)  # global ids
            cand_v = ttnn.repeat(ttnn.reshape(vals, [1, 1, 1, nc * K]), ttnn.Shape([1, 1, B, 1]))  # [1,1,32,nc*K]
            cand_i = ttnn.repeat(ttnn.reshape(gidx, [1, 1, 1, nc * K]), ttnn.Shape([1, 1, B, 1]))
            cand_i = ttnn.untilize(
                ttnn.to_memory_config(cand_i, ttnn.DRAM_MEMORY_CONFIG), use_multicore=True
            )  # RM int32
            ttnn.manual_seed(seeds=self.t_seed, user_ids=self._samp_uids)
            ttnn.sampling(cand_v, cand_i, k=self.t_k, p=self.t_p, temp=self.t_temp, output_tensor=self.t_tok32)
            idx4 = ttnn.reshape(ttnn.slice(self.t_tok32, [0, 0, 0, 0], [1, 1, 1, 1]), [1, 1, 1, 1])  # sampled token
            ttnn.copy(ttnn.reshape(idx4, [1]), self.t_tok)  # next token, in place
        if self._presence_penalty > 0:  # mark the new token present: OR a one-hot into the mask.
            with sp.region("head.samp.penalty"):
                # ``ttnn.scatter`` over the [1,1,1,vocab] mask is ~2 ms (it scans the 32x tile-padded
                # vocab); an elementwise one-hot (eq vs the iota index buffer) + maximum is ~0.4 ms.
                idx_t = ttnn.to_layout(ttnn.typecast(idx4, ttnn.int32), ttnn.TILE_LAYOUT)
                onehot = ttnn.typecast(ttnn.eq(self._samp_iota, idx_t), ttnn.bfloat16)
                ttnn.maximum(self.t_presence, onehot, output_tensor=self.t_presence)  # in-place OR
        ttnn.plus_one(self.t_seed)  # fresh RNG next step (trace-safe; validated)

    # ── vLLM CONTINUOUS BATCHING: per-slot ON-DEVICE token selection (removes host sampling) ──────
    def _select_token_batch(self, x):
        """Batched on-device token selection over ALL B slots (B = self.batch_size <= SAMP_USERS=32) —
        the B>1 generalization of _select_token (which is row-0 only). Writes the next token id into
        every row of self.t_tok [B]. Greedy = per-row argmax; sampling = per-row chunked topk +
        ttnn.sampling across the 32 user lanes. A per-slot presence mask (t_presence_b) times a uniform
        baked penalty discourages repeats when enabled.

        VERIFY-ON-DEVICE: the per-row reshapes/pads and the 32-lane ttnn.sampling with B<32 real rows."""
        Bm = self.batch_size
        if self.sampling is None:  # greedy: per-row argmax over [B, vocab]
            with sp.region("head.lm_head"):
                logits = ttnn.linear(x, self.lm_head_w)  # [B, vocab]
            with sp.region("head.argmax"):
                logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
                tok = ttnn.argmax(logits, dim=-1, keepdim=False)  # [B]
                ttnn.copy(ttnn.to_layout(tok, ttnn.ROW_MAJOR_LAYOUT), self.t_tok)
            return
        V, CW, K, U, nc = self.args.vocab_size, self.SAMP_CHUNK, self.SAMP_K, self.SAMP_USERS, self._samp_nc
        with sp.region("head.lm_head"):
            logits = ttnn.linear(x, self.lm_head_w)  # [B, vocab]
        logits = ttnn.reshape(logits, [1, 1, Bm, V])
        if getattr(self, "_presence_on_batch", False):  # subtract per-slot presence mask * uniform penalty
            logits = ttnn.sub(logits, ttnn.multiply(self.t_presence_b, self._presence_penalty_batch))
        # Per-row chunked topk via the VALIDATED single-row path (avoids the TILE-layout cross-row
        # reshape that scrambled candidates). B slices + topks unroll into the trace (static B).
        cand_v_rows, cand_i_rows = [], []
        with sp.region("head.samp.topk"):
            for b in range(Bm):
                row = ttnn.reshape(ttnn.slice(logits, [0, 0, b, 0], [1, 1, b + 1, V]), [1, 1, 1, V])  # row b logits
                row = ttnn.pad(row, [(0, 0), (0, 0), (0, 0), (0, nc * CW - V)], value=self.SAMP_NEG)
                chunks = ttnn.reshape(row, [1, 1, nc, CW])
                vals, idxs = ttnn.topk(chunks, k=K, dim=-1, sorted=False, indices_tensor=self._samp_local_idx)
                gidx = ttnn.add(ttnn.typecast(idxs, ttnn.int32), self._samp_offsets, dtype=ttnn.int32)  # global ids
                cand_v_rows.append(ttnn.reshape(vals, [1, 1, 1, nc * K]))
                cand_i_rows.append(ttnn.reshape(gidx, [1, 1, 1, nc * K]))
        with sp.region("head.samp.sample"):
            cand_v = ttnn.concat(cand_v_rows, dim=2)  # [1,1,B,nc*K]  each row = its own candidates
            cand_i = ttnn.concat(cand_i_rows, dim=2)
            if Bm < U:  # ttnn.sampling wants 32 lanes; pad the unused rows
                cand_v = ttnn.pad(cand_v, [(0, 0), (0, 0), (0, U - Bm), (0, 0)], value=self.SAMP_NEG)
                cand_i = ttnn.pad(cand_i, [(0, 0), (0, 0), (0, U - Bm), (0, 0)], value=0)
            cand_i = ttnn.untilize(ttnn.to_memory_config(cand_i, ttnn.DRAM_MEMORY_CONFIG), use_multicore=True)
            ttnn.manual_seed(seeds=self.t_seed, user_ids=self._samp_uids)
            ttnn.sampling(cand_v, cand_i, k=self.t_k, p=self.t_p, temp=self.t_temp, output_tensor=self.t_tok32)
            toks = ttnn.reshape(ttnn.slice(self.t_tok32, [0, 0, 0, 0], [1, 1, 1, Bm]), [Bm])  # first B lanes
            ttnn.copy(toks, self.t_tok)
        if getattr(self, "_presence_on_batch", False):  # mark each slot's sampled token present (in-place OR)
            with sp.region("head.samp.presence"):
                if getattr(self, "_presence_fast", False):
                    # FAST: one COLUMN-broadcast eq — iota_full[1,1,Bm,V] eq tokens[1,1,Bm,1] (only the
                    # last dim broadcasts; matches the safe bias/scale pattern, unlike the mutual/outer
                    # broadcast that hangs). tokens reshaped to a column in ROW-MAJOR first (avoids the
                    # TILE cross-row reshape scramble). No per-row loop, no concat.
                    idx = ttnn.reshape(ttnn.slice(self.t_tok32, [0, 0, 0, 0], [1, 1, 1, Bm]), [1, 1, Bm, 1])
                    idx = ttnn.to_layout(ttnn.typecast(idx, ttnn.int32), ttnn.TILE_LAYOUT)
                    onehot = ttnn.typecast(ttnn.eq(self._samp_iota_b, idx), ttnn.bfloat16)  # [1,1,Bm,V]
                    ttnn.maximum(self.t_presence_b, onehot, output_tensor=self.t_presence_b)
                else:
                    # SAFE (default): per-row single-row-broadcast eq (iota[1,1,1,V] eq token[1,1,1,1]) +
                    # concat + maximum. Verified working on-device; slower (B-way eq loop + concat).
                    onehots = []
                    for b in range(Bm):
                        tb = ttnn.to_layout(
                            ttnn.typecast(ttnn.slice(self.t_tok32, [0, 0, 0, b], [1, 1, 1, b + 1]), ttnn.int32),
                            ttnn.TILE_LAYOUT,
                        )
                        onehots.append(ttnn.typecast(ttnn.eq(self._samp_iota, tb), ttnn.bfloat16))  # [1,1,1,V]
                    onehot = ttnn.concat(onehots, dim=2)  # [1,1,Bm,V]
                    ttnn.maximum(self.t_presence_b, onehot, output_tensor=self.t_presence_b)
        ttnn.plus_one(self.t_seed)  # fresh RNG next step

    def _decode_graph_batch(self):
        """Traceable batched decode step: embed -> layers -> norm -> batched on-device select (writes
        t_tok [B]) -> advance positions. Batched analogue of _decode_graph."""
        x = self._decode_hidden()
        with sp.region("dec.select"):
            self._select_token_batch(x)
        with sp.region("dec.pos_advance"):
            ttnn.plus_one(self.t_curpos)
            ttnn.plus_one(self.t_ropepos)

    def capture_decode_trace_batch(self):
        """Capture the batched device-select decode graph (analogue of capture_decode_trace). PRECONDITION:
        one decode_step_eager_batch() has run (kernels compiled). Snapshots/restores decode state around
        the dummy recording step."""
        if getattr(self, "batch_trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self.batch_trace_id)
        snap = [ttnn.clone(t) for t in self._trace_snapshot_tensors()]
        self.batch_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._decode_graph_batch()
        ttnn.end_trace_capture(self.mesh_device, self.batch_trace_id, cq_id=0)
        for orig, s in zip(self._trace_snapshot_tensors(), snap):
            ttnn.copy(s, orig)

    def decode_step_eager_batch(self):
        """One eager batched device-select decode step (compiles kernels); returns [B] host token ids."""
        self._decode_graph_batch()
        return from_tt(self.t_tok, self.mesh_device).flatten()[: self.batch_size]

    def decode_step_traced_batch(self):
        """Replay the batched device-select decode trace; returns [B] host token ids (only the single
        [B]-token readback per step — no full-vocab D2H, no host sampling)."""
        ttnn.execute_trace(self.mesh_device, self.batch_trace_id, cq_id=0, blocking=False)
        return from_tt(self.t_tok, self.mesh_device).flatten()[: self.batch_size]

    def decode_step_eager(self, read_from_device: bool = True):
        """Run one decode step eagerly (compiles kernels for trace capture). Returns the next token id
        (int) when read_from_device=True, else the on-DEVICE token tensor (cloned) for the vLLM async
        path to read later."""
        self._decode_graph()
        self.pos += 1
        if not read_from_device:
            return ttnn.clone(self.t_tok)  # device token; async read happens later
        return int(from_tt(self.t_tok, self.mesh_device).flatten()[0])

    def _state_tensors(self):
        ts = []
        for c in self.caches:
            if isinstance(c, list):  # attention KV [k, v]
                ts += c
            else:  # gated-delta state dict (conv_rows is itself a list of row buffers -> flatten)
                for v in c.values():
                    ts += v if isinstance(v, list) else [v]
        ts += [self.t_tok, self.t_curpos, self.t_ropepos]  # generation state advances on device
        if self.sampling is not None:
            ts.append(self.t_seed)  # RNG seed advances each step; snapshot/restore around capture
            if self._presence_penalty > 0:
                ts.append(self.t_presence)  # presence mask accumulates each step
        return ts

    def _trace_snapshot_tensors(self):
        """State the dummy trace-capture step mutates and that must be restored, EXCLUDING the
        position-indexed KV caches. The dummy decode writes K/V only at the CURRENT position (which is
        restored), and the first traced replay rewrites that exact position with the same token → the
        KV cache ends up identical either way, so it needs no snapshot. Cloning it would transiently
        DOUBLE KV memory (e.g. +7.5 GB at pool=3 × 128K → OOM in capture). The gated-delta
        recurrent_state + conv rows ARE running accumulators (not position-indexed) so they must be
        restored, along with the on-device generation state (token/positions/seed/presence)."""
        ts = []
        for c in self.caches:
            if isinstance(c, dict):  # gated-delta running state (recurrent_state + conv_rows); NOT KV
                for v in c.values():
                    ts += v if isinstance(v, list) else [v]
        ts += [self.t_tok, self.t_curpos, self.t_ropepos]
        if self.sampling is not None:
            ts.append(self.t_seed)
            if self._presence_penalty > 0:
                ts.append(self.t_presence)
            if getattr(self, "t_presence_b", None) is not None:  # batched presence mask accumulates in-graph
                ts.append(self.t_presence_b)
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
        snap = [ttnn.clone(t) for t in self._trace_snapshot_tensors()]
        self.trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._decode_graph()
        ttnn.end_trace_capture(self.mesh_device, self.trace_id, cq_id=0)
        for orig, s in zip(self._trace_snapshot_tensors(), snap):
            ttnn.copy(s, orig)  # undo the mutation done while recording

    def decode_step_traced(self, read_from_device: bool = True):
        """Replay the captured decode trace for one real step. Host work: kick off the trace and
        read back only the single next-token id (no per-token input rebuild, no full-logit D2H).
        read_from_device=False returns the on-DEVICE token tensor (cloned so the next replay can't
        overwrite it before the vLLM async path reads it)."""
        ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=False)
        self.pos += 1
        if not read_from_device:
            return ttnn.clone(self.t_tok)
        return int(from_tt(self.t_tok, self.mesh_device).flatten()[0])

    def capture_decode_logits_trace(self):
        """Record the LOGITS-returning decode step (embed -> layers -> norm -> lm_head) as a trace,
        for the vLLM host-sampling path: replay computes logits into a persistent buffer and the host
        samples from them (per-request temperature/top-p/penalties/logprobs stay in vLLM's sampler).
        Unlike capture_decode_trace, this does NOT select a token on device — no _select_token — so the
        chosen token must be written back via set_decode_tokens() before each decode_step_logits_traced().

        Mirrors capture_decode_trace: releases any prior logits trace, snapshots/restores the decode
        state around the dummy recording step (the recorded step advances positions on device, so it
        would perturb generation otherwise). PRECONDITION: one eager decode_forward_logits() has run so
        all kernels are compiled (capture must not JIT)."""
        if getattr(self, "logits_trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self.logits_trace_id)
        snap = [ttnn.clone(t) for t in self._trace_snapshot_tensors()]
        self.logits_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        x = self._decode_hidden()
        with sp.region("dec.lm_head"):
            self._traced_logits = ttnn.linear(x, self.lm_head_w)  # persistent output buffer (read after replay)
        with sp.region("dec.pos_advance"):
            ttnn.plus_one(self.t_curpos)  # advance position on device (for KV write + sdpa)
            ttnn.plus_one(self.t_ropepos)  # advance RoPE-table index on device
        ttnn.end_trace_capture(self.mesh_device, self.logits_trace_id, cq_id=0)
        for orig, s in zip(self._trace_snapshot_tensors(), snap):
            ttnn.copy(s, orig)  # undo the mutation done while recording

    def decode_step_logits_traced(self, read_from_device: bool = True):
        """Replay the captured logits trace for one real step. The 40-layer compute is replayed from
        the trace (no per-op host dispatch); the only per-step host work is the full-vocab readback.

        read_from_device=True returns HOST logits [B, vocab] (blocking D2H here). read_from_device=False
        returns the on-DEVICE persistent logits buffer and defers the readback to the vLLM async path
        (read_decode_output does an async cpu() so the D2H overlaps the next step's scheduling/detok)."""
        ttnn.execute_trace(self.mesh_device, self.logits_trace_id, cq_id=0, blocking=False)
        self.pos += 1
        if not read_from_device:
            # The trace always writes the SAME persistent buffer, so under vLLM async decode the next
            # step's replay would overwrite it before this step's deferred read. Hand back an
            # independent clone (cheap ~0.5 MB D2D on cq 0, ordered before the next replay) so the
            # async cpu() reads a stable tensor. No clone needed on the sync path (read below is immediate).
            return ttnn.clone(self._traced_logits)
        return from_tt(self._traced_logits, self.mesh_device)

    def process_output_decode(self, tt_out, B=1, S=1, is_tokens=False, is_log_probs=False, **kwargs):
        """Convert a decode output from a (host) ttnn tensor to a host torch tensor. Called by the
        inherited Generator.process_decode_output_host on the vLLM async-decode path (after
        read_decode_output's async cpu()). is_tokens=True -> [B] token ids (on-device sampling path);
        else -> [B, S, vocab] logits (host-sampling path). On-device log-probs are not produced."""
        assert not is_log_probs, "on-device log-probs unsupported (host sampling only)"
        if is_tokens:
            return ttnn.to_torch(tt_out).to(torch.int64).reshape(-1)[:B]  # [B] sampled token ids
        out = ttnn.to_torch(tt_out).float()
        return out.reshape(B, S, -1)[..., : self.args.vocab_size]
