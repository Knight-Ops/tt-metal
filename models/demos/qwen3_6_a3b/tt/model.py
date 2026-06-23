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

    def forward(self, input_ids: torch.Tensor):
        """Prefill. input_ids: torch [1, T]. Allocates per-layer caches; returns last-token logits
        [1, 1, vocab]."""
        T = input_ids.shape[1]
        self.max_seq = self.args.max_seq_len
        self.caches = [self._alloc_cache(layer) for layer in self.layers]
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

    def forward_incremental(self, input_ids: torch.Tensor):
        """Incremental (from-cache) prefill for multi-turn / long context: ingest M NEW tokens
        continuing from the existing caches (self.pos = current context length P), instead of
        re-prefilling the whole P+M context. Requires a prior forward()/forward_incremental() to have
        populated the caches. M and P must be multiples of 128 (chunked-SDPA + fill_cache alignment).
        Updates the caches + self.pos; returns the last NEW token's logits [1,1,vocab]. Eager."""
        M = input_ids.shape[1]
        P = self.pos
        if not hasattr(self, "_inc_page_table"):  # trivial single-block page table (built once)
            self._inc_page_table = to_tt(
                torch.zeros(1, 32, dtype=torch.int32), self.mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        x = self._embed(input_ids, M)
        cos, sin = precompute_rope(M, self.args.rotary_dim, self.args.rope_theta, self.mesh_device, start_pos=P)
        for layer, cache in zip(self.layers, self.caches):
            x = layer.forward_prefill_incremental(x, cos, sin, cache, self._inc_page_table, P)
        self.pos = P + M
        return self._head(x, M)

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
        logits = ttnn.linear(x, self.lm_head_w)  # device [1, vocab]
        # Greedy argmax. The default single-core argmax over the 248k vocab is ~8.4 ms (the lm_head
        # matmul itself is only ~1.5 ms); the multicore argmax needs a ROW_MAJOR input and runs in
        # ~0.06 ms, so converting layout first (~0.1 ms) cuts the head stage ~9.9 -> ~1.6 ms/token.
        logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        tok = ttnn.argmax(logits, dim=-1, keepdim=False, use_multicore=True)  # greedy, [1] uint32
        ttnn.copy(ttnn.to_layout(tok, ttnn.ROW_MAJOR_LAYOUT), self.t_tok)  # next token, in place
        ttnn.plus_one(self.t_curpos)  # advance position on device (for KV write + sdpa)
        ttnn.plus_one(self.t_ropepos)  # advance RoPE-table index on device

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
        return ts + [self.t_tok, self.t_curpos, self.t_ropepos]  # generation state advances on device

    def capture_decode_trace(self):
        """Record the self-contained decode graph as a trace. PRECONDITION: at least one
        decode_step_eager() has run so all kernels are compiled (capture must not JIT). The decode
        state (caches + token + positions) is snapshotted and restored so recording the dummy step
        doesn't perturb generation. After this, drive every real step through decode_step_traced()."""
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
