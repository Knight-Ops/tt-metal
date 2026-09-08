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
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_a3b.tt import prefill_profiler as prof
from models.demos.qwen3_6_a3b.tt import signpost as sp
from models.demos.qwen3_6_a3b.tt.attention import precompute_rope
from models.demos.qwen3_6_a3b.tt.common import as_weight, from_tt, to_tt, wide1d_enabled, wide_1d_decode_pc
from models.demos.qwen3_6_a3b.tt.decoder import TtDecoderLayer
from models.demos.qwen3_6_a3b.tt.gated_delta import _CONV_ADDCHAIN, _STATE_DT
from models.demos.qwen3_6_a3b.tt.paged_kv import PagedKV
from models.demos.qwen3_6_a3b.tt.rms_norm import TtRMSNorm


def common_prefix_len(a, b) -> int:
    """Length of the longest common prefix of two token-id sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def pick_checkpoint(positions, prefix_len: int, min_reuse: int = 0) -> int:
    """Deepest checkpoint position that the new prompt still agrees with, or 0 for "prefill fresh".

    ``positions`` are context lengths whose gated-delta state we hold (see TtModel.snapshot_gdn_state);
    ``prefix_len`` is how far the new prompt matches the one the caches were built from. A checkpoint is
    usable iff it lies at or before that divergence point, and worth using iff it saves at least
    ``min_reuse`` positions."""
    best = max((p for p in positions if p <= prefix_len), default=0)
    return best if best >= min_reuse else 0


class ConversationSlots:
    """Which parked conversation an incoming prompt belongs to, and which one to evict.

    Slot identity is the LONGEST COMMON PREFIX between the prompt and each slot's remembered
    history. That is not a heuristic stand-in for a session id, it is the only thing that is
    actually load-bearing: reuse is correct exactly as far as the token prefixes agree, so a slot
    whose history shares 6272 tokens with this prompt IS this conversation for reuse purposes, and
    one that shares 3 is not, whatever the client thinks. It also handles the cases a session id
    would get wrong -- a re-rendered transcript, a regenerated turn, a branch of an earlier message
    all land on the right slot at the right depth.

    Eviction is least-recently-used. A slot that is claimed but yields no reuse is still claimed:
    that is the point, since its FIRST turn cannot reuse anything and its second is where the win
    is. `min_lcp` keeps unrelated short prompts from stealing a long conversation's slot on the
    strength of a shared system prompt -- below it the prompt is treated as a new conversation.
    """

    def __init__(self, n, min_lcp):
        self.n = n
        self.min_lcp = min_lcp
        self.hist = [None] * n  # remembered prompt per slot (list[int]); None = free
        self.used = [0] * n  # LRU stamp
        self.streak = [0] * n  # consecutive reuses, for QWEN36_PREFIX_REUSE_MAX_TURNS
        self.clock = 0

    def match(self, ids):
        """(slot, lcp, evicted) for `ids`. `evicted` is the slot whose conversation was discarded."""
        self.clock += 1
        best, best_lcp = None, 0
        for i, h in enumerate(self.hist):
            if not h:
                continue
            lcp = common_prefix_len(h, ids)
            if lcp > best_lcp:
                best, best_lcp = i, lcp
        if best is not None and best_lcp >= self.min_lcp:
            self.used[best] = self.clock
            return best, best_lcp, None
        free = next((i for i, h in enumerate(self.hist) if not h), None)
        if free is not None:
            self.used[free] = self.clock
            return free, 0, None
        victim = min(range(self.n), key=lambda i: self.used[i])
        self.used[victim] = self.clock
        return victim, 0, victim

    def commit(self, slot, ids, reused):
        self.hist[slot] = ids
        self.streak[slot] = self.streak[slot] + 1 if reused else 0

    def forget(self, slot):
        self.hist[slot] = None
        self.streak[slot] = 0

    def lru(self, exclude=()):
        """Least-recently-used OCCUPIED slot, or None. For freeing KV blocks before a prefill that
        would otherwise exhaust the pool -- `exclude` is the conversation about to run."""
        live = [i for i, h in enumerate(self.hist) if h and i not in exclude]
        return min(live, key=lambda i: self.used[i]) if live else None

    def describe(self):
        return " ".join(f"{i}:{len(h) if h else '-'}" for i, h in enumerate(self.hist))


# Chunked-SDPA q_chunk_size for the ragged remainder. 128 is what the shipped ragged path uses and
# the only value regression-tested at an offset the block width does not divide (test_ragged_prefill:
# P=512, padded width 384).
_RAGGED_Q_CHUNK = 128

# The set of padded widths a prefill block may run at. EVERY distinct width costs ~90-110 device
# programs, one time, and those stay live for the process -- so this ladder IS the model's steady-state
# program working set, and it is the thing to keep small.
#
# This was {128, 512} for a while, on the reasoning that four widths take ~384 programs of a ~550
# working set "and the device starts hanging around there". That hang was NOT program pressure -- it
# was the per-width-constant corruption fixed by prewarm_prefill_shapes (section 5 of
# IMPLEMENTATION_GUIDE.md), and a synthetic loop has since reached 801 cached programs with a
# replaying trace and 21.5 GB of ballast without complaint. So the ladder is back to every
# 128-multiple, because the padding it saves is now the dominant prefill cost:
#
#   a reuse turn re-ingests ~130-260 tokens. At {128,512} that pads to 512 and costs ~614 ms; at
#   {128,256,384,512} it pads to 256 and costs ~370 ms (block cost = 127 ms + 0.95 ms/token,
#   PREFILL.md 0.9). MEASURED end to end, 40 layers, 10-turn chat: 0.34 s per reuse turn against
#   0.56 s, and the worst turn 0.62 s against 0.88 s.
#
# It still pays to keep this SHORT -- every width is ~90-110 one-time programs, and they are all
# built up front by prewarm_prefill_shapes -- so add a width only for a length range that actually
# recurs. QWEN36_PREFILL_WIDTHS overrides; "128,512" restores the narrow ladder.
_PREFILL_WIDTHS = sorted(int(x) for x in os.environ.get("QWEN36_PREFILL_WIDTHS", "128,256,384,512").split(","))


def pad_width(m: int, widths=None, limit: int | None = None) -> int:
    """Smallest allowed padded width >= m (rounding up to a multiple of the largest for long blocks).

    ``limit`` caps the result -- the room left in the KV cache. Padding writes real KV rows, so a block
    near the end of the context cannot be widened past ``max_seq``; there the ladder is abandoned for
    the minimal 128-multiple. That costs a couple of extra shapes, but only for prompts within one
    width of the context limit, which is rare and bounded.

    Pure, so the ladder's effect on the shape space is testable without a device."""
    widths = _PREFILL_WIDTHS if widths is None else sorted(widths)
    big = widths[-1]
    if m % big == 0 and (limit is None or m <= limit):
        return m  # already a whole number of full-width blocks; nothing to pad
    for w in widths:
        if m <= w and (limit is None or w <= limit):
            return w
    minimal = ((m + _RAGGED_Q_CHUNK - 1) // _RAGGED_Q_CHUNK) * _RAGGED_Q_CHUNK
    return minimal  # may still exceed `limit`; the caller's own bounds check reports that properly


# Debug: synchronize the device and log after every layer of an incremental prefill (and inside the
# head), so a device HANG localizes to one op instead of surfacing at the next readback -- which is
# wherever the queue happens to drain, and told us nothing while chasing the prefix-reuse hang.
# Costs a full pipeline drain per layer, so it is strictly a diagnostic. QWEN36_PREFILL_LAYER_SYNC=1.
_LAYER_SYNC = os.environ.get("QWEN36_PREFILL_LAYER_SYNC") == "1"


def plan_prefill_blocks(
    T: int,
    start: int = 0,
    chunk: int = 512,
    single_shot_max: int = 2048,
    snap: int = _RAGGED_Q_CHUNK,
    inblock: bool = False,
):
    """Block schedule for prefill_from: ``(offset, width, kind, q_chunk)`` ingesting positions
    [start, T) in order, where kind is "single" (single-shot, allocates the caches), "block" (a full
    incremental block) or "ragged" (the right-padded remainder).

    Every offset is a multiple of ``chunk`` and every full block is exactly ``chunk`` wide, which is
    NOT a stylistic choice -- it is the entire safety argument. Those are the only shapes the shipped
    chunked prefill (prefill_long) has ever run, so a resumed walk emits nothing new: full blocks are
    ``chunk`` wide at ``chunk``-multiple offsets, and the remainder is a ragged block at a
    ``chunk``-multiple offset with ``_RAGGED_Q_CHUNK``, which is test_ragged_prefill's case.

    ``snap`` is the SNAPSHOT granularity, and it is decoupled from ``chunk`` on purpose: the coarse
    512 structure (the single-shot head and the full blocks) is kept because a block width has to come
    off the padding ladder or every prompt length compiles ~90 new programs (section 2), while the
    TAIL is split so a snapshot lands on a ``snap`` multiple. A turn then re-ingests at most
    ``snap - 1`` tokens it could have skipped instead of ``chunk - 1``.

    HISTORY, worth keeping: this used to emit only ``chunk``-wide blocks at ``chunk``-multiple offsets,
    because two configurations that satisfy chunked-SDPA's stated rule (``chunk_start % q_chunk == 0``)
    hung the device outright:
        P=640, width=256, q_chunk=128
        P=768, width=256, q_chunk=256
    while P=512/640/768 at width=128 were all fine. Note the pattern: both failures are width **256**
    and every pass is width **128** -- keyed on the WIDTH, not the offset. That is the signature of the
    lazily-built per-width constants fixed by prewarm_prefill_shapes (section 5), and with that fix
    both configurations now pass with a decode trace resident (hangdbg/envelope.py). So the envelope
    was a workaround for that bug, not a real alignment rule, and the tail can be split.

    The one rule that IS real: chunked-SDPA requires ``chunk_start % q_chunk_size == 0``, and
    q_chunk_size defaults to the block width. So a block whose width does not divide its offset must
    be handed ``_RAGGED_Q_CHUNK`` explicitly -- every offset here is a ``snap`` multiple, so 128
    always divides it."""
    assert 0 <= start < T, f"start {start} must be in [0, {T})"
    assert chunk % snap == 0, f"chunk {chunk} must be a multiple of the snapshot grain {snap}"
    assert snap % _RAGGED_Q_CHUNK == 0, f"snapshot grain {snap} must be a multiple of {_RAGGED_Q_CHUNK}"
    assert start % snap == 0, f"resume offset {start} must be a multiple of the snapshot grain {snap}"

    def _block(P, m, kind):
        # q_chunk=None lets attention use the width itself; legal only when the width divides the
        # offset. Otherwise pin it to 128, which every snap-multiple offset is divisible by.
        return (P, m, kind, _RAGGED_Q_CHUNK if (kind == "ragged" or P % m) else None)

    coarse = (T // chunk) * chunk  # the 512 structure, unchanged
    snap_end = (T // snap) * snap  # last position a snapshot may be taken at
    plan, P = [], start
    if inblock:
        # The recurrence can export its carried state at an interior row (see
        # GatedDelta.supports_inblock_snapshot), so the tail does NOT need its own block to land a
        # snapshot on the grain: prefill_from asks the last block for a checkpoint at snap_end. Full
        # blocks run while a whole chunk still fits, which keeps the remainder below `chunk` and so
        # on the padding ladder.
        if P == 0:
            if coarse == 0:
                return [(0, T, "single", None)]
            plan.append(_block(0, coarse if coarse <= single_shot_max else chunk, "single"))
            P = plan[-1][1]
        while P + chunk <= T:
            plan.append(_block(P, chunk, "block"))
            P += chunk
        if T > P:
            plan.append(_block(P, T - P, "ragged"))
        return plan
    if P == 0:
        if coarse == 0:  # T < chunk: nothing coarse to align to
            if snap_end == 0:  # and under one snap grain: nothing to snapshot either
                return [(0, T, "single", None)]
            plan.append(_block(0, snap_end, "single"))
            P = snap_end
        else:
            # Keep the head as wide as forward() would have taken in one shot, and on a CHUNK
            # multiple so its padded width stays on the ladder.
            plan.append(_block(0, coarse if coarse <= single_shot_max else chunk, "single"))
            P = plan[-1][1]
    while P + chunk <= snap_end:
        plan.append(_block(P, chunk, "block"))
        P += chunk
    if P < snap_end:  # one narrower block so a snapshot lands on the snap grain
        plan.append(_block(P, snap_end - P, "block"))
        P = snap_end
    if T > P:  # the sub-snap remainder; no snapshot is taken here
        plan.append(_block(P, T - P, "ragged"))
    return plan


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
        # Tuned wide 1D mcast config for the lm_head. It has been on the ttnn AUTO config because it
        # CANNOT be DRAM-sharded (its 248K-wide output OOMs L1-sharded, bank_manager fatal), which left
        # the biggest single matmul in the step untuned. Measured (tests/bench_matmul_layout.py,
        # [2048, 248320] bf4, traced): AUTO 1125.9 us -> 1035.7 us at (110 cores, in0_block_w=4),
        # i.e. 58.8% -> 63.9% of DRAM peak. per_core_M=1 covers a full 32-row tile, so it is valid for
        # every <=32-row call (decode B<=32, MTP verify K, and prefill's last-token slice) -- _lmh()
        # gates on that and falls back to AUTO for the S-row eval path in _head_all. Not numerically
        # neutral (in0_block_w changes the K accumulation order); accuracy-gated like any dtype change.
        self._lm_head_pc = (
            wide_1d_decode_pc(32, self.args.dim, self.args.vocab_size, 110, in0_block_w=4)
            if wide1d_enabled("lm_head")
            else None
        )
        # On-device RoPE tables: cos/sin for every position, indexed by position with ttnn.embedding
        # during decode (no per-token host recompute). Row-major so they act as embedding weights.
        self.cos_table, self.sin_table = self._build_rope_tables()
        # Token selection is greedy (on-device argmax) by default. enable_sampling() swaps in the
        # on-device temperature/top-k/top-p sampler; None here means "greedy, zero added cost".
        self.sampling = None
        # Paged KV cache. DEFAULT OFF: the flat [1, n_kv, max_seq, hd] cache is unchanged unless
        # QWEN36_PAGED_KV=1, so demo.py / demo/server.py are byte-identical until this is flipped.
        # Paging is a MEMORY change, not a speedup -- measured free at decode (tt/paged_kv.py) -- and it
        # only reclaims anything if the pool is over-subscribed via QWEN36_KV_POOL_TOKENS. It is also
        # what unblocks traced incremental prefill (PREFILL.md §4a) and vLLM continuous batching, which
        # supplies its OWN page table and can simply overwrite `self.kv_pager.table`.
        self.paged_kv = os.environ.get("QWEN36_PAGED_KV") == "1"
        self.kv_pager = None
        # PARKED CONVERSATIONS. How many independent conversations keep their KV blocks and their
        # gated-delta checkpoints resident at once, so an interleaved request from another
        # conversation does not destroy this one's prefix. 1 = the historical single-slot behaviour.
        #
        # WHY THIS IS NEEDED AT ALL, and why it cannot live in the serving adapter: prefix reuse
        # restores the gated-delta state at position P and recomputes only [P, T). That is correct
        # ONLY if the KV rows below P still belong to this sequence. With one flat cache they do not
        # -- any other request's prefill writes rows [0, its T) -- so an adapter that merely
        # remembered several histories would restore state against another conversation's KV and
        # emit plausible garbage. MEASURED in a real deployment: a chat UI's title/tag side-requests
        # (209..409 tokens) interleave with the user's 6.3K conversation, and every single turn fell
        # back to a full prefill. Paging is what makes parking free: each conversation owns its own
        # blocks out of one pool, so nothing overlaps and nothing is copied.
        self.n_conv_slots = 1
        self._conv_active = 0
        # Captured decode trace (one at a time). Re-captured per request; the PRIOR trace is released
        # first (capture_decode_trace) so traces never accumulate — that accumulation, together with
        # per-request cache reallocation, was the leak that OOMed the server.
        self.trace_id = None

    def _lmh(self, x):
        """The lm_head matmul. Uses the tuned wide-1D config when the activation is <= 32 rows (one
        tile row, what per_core_M=1 covers); otherwise the ttnn heuristic. See self._lm_head_pc."""
        if self._lm_head_pc is not None and x.shape[-2] <= 32:
            return ttnn.linear(x, self.lm_head_w, program_config=self._lm_head_pc)
        return ttnn.linear(x, self.lm_head_w)  # heuristic: config off, or more rows than per_core_M=1

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
        if _LAYER_SYNC:
            ttnn.synchronize_device(self.mesh_device)
            logger.info(f"[inc] head.norm ok (T={T})")
        with sp.region("head.lm_head"):
            logits = self._lmh(x)  # [1, vocab]
        if _LAYER_SYNC:
            ttnn.synchronize_device(self.mesh_device)
            logger.info(f"[inc] head.lm_head ok (T={T})")
        with sp.region("head.d2h"):
            # ROW_MAJOR before the readback. A [1, vocab] TILE tensor is PHYSICALLY [32, vocab] =
            # 15.9 MB at vocab=248320, so a tiled D2H moves 32x the bytes the caller actually uses.
            # The relayout is an on-device op; measured worth ~14 ms of a T=512 prefill (PREFILL.md).
            logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
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
        logits = self._lmh(x)  # [S, vocab]
        # ROW_MAJOR before the readback: TILE pads S up to a multiple of 32, so a tiled D2H moves
        # ceil(S/32)*32 rows of a 248k-wide tensor. Same fix as _head.
        logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
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

    def _kv_advance(self, slot=None):
        """Map any KV blocks the NEXT decode step will touch. Call from the Python decode loop, never
        from inside a captured graph: it may issue a host->device write. It is a no-op on the ~63 of
        every 64 steps that do not cross a block boundary, so the steady-state step stays host-free.

        `slot=None` means the conversation currently activated (activate_conversation), which is what
        every decode caller wants: it is decoding whatever prefill_reuse last ran."""
        if self.paged_kv and self.kv_pager is not None:
            self.kv_pager.ensure(self.pos, self._conv_active if slot is None else slot)

    def _kv_fill_table_at(self, P, M, slot=0):
        """Fill table for an INCREMENTAL chunk writing M tokens at offset P.

        `paged_fill_cache` has no position argument -- the offset is expressed by WHICH logical blocks
        the table names, so hand it the sub-table starting at block P/block_size (the same trick
        tt_transformers' _chunk_prefill_page_table uses). P is a multiple of QWEN36_PREFILL_CHUNK (>=128)
        and the block size is 64, so P is always block-aligned."""
        if not self.paged_kv or self.kv_pager is None:
            return None
        pg = self.kv_pager
        assert P % pg.block_size == 0, f"incremental offset {P} must be a multiple of block {pg.block_size}"
        pg.ensure(P + M - 1, slot)
        b0 = P // pg.block_size
        nb = -(-M // pg.block_size)
        row = pg._host[slot : slot + 1, b0 : b0 + nb].contiguous()
        return to_tt(row, self.mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _kv_read_table(self, slot=0):
        """The page table chunked-SDPA READS through during incremental prefill.

        Per-SLOT, unlike decode's: chunked prefill runs one sequence at a time, so the kernel treats row
        0 of whatever table it is handed as that sequence's mapping. Passing the full [B, blocks] decode
        table would silently make every slot read slot 0's pages. Returns the flat path's trivial
        single-block table when not paging."""
        if self.paged_kv and self.kv_pager is not None:
            return self.kv_pager.fill_page_table_for(slot)
        if not hasattr(self, "_inc_page_table"):  # trivial single-block page table (built once)
            self._inc_page_table = to_tt(
                torch.zeros(1, 32, dtype=torch.int32),
                self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        return self._inc_page_table

    def _kv_fill_table(self, upto, slot=0):
        """Reserve KV blocks covering positions [0, upto) and return the `[1, blocks]` table
        `paged_fill_cache` wants -- or None on the flat path, where the callee then uses `fill_cache`.

        Host-side, and deliberately NOT inside any captured graph: it can issue a
        copy_host_to_device_tensor, which is illegal under trace capture (MEMORY.md §1)."""
        if not self.paged_kv or self.kv_pager is None:
            return None
        self.kv_pager.ensure(max(0, upto - 1), slot)
        return self.kv_pager.fill_page_table_for(slot)

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
                # dtype follows QWEN36_GDN_FP32_STATE: the recurrent state is a feedback accumulator
                # and bf16 drifts measurably over a long decode (see gated_delta._FP32_STATE).
                "recurrent_state": ttnn.zeros(
                    [1, a.lin_num_v_heads, a.lin_head_k_dim, a.lin_head_v_dim],
                    dtype=_STATE_DT,
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
        if self.paged_kv:
            if self.kv_pager is None:  # one pool + one shared page table for all attention layers
                self.kv_pager = PagedKV(
                    self.mesh_device,
                    self.args.n_kv_heads,
                    self.args.head_dim,
                    # batch_size is set by alloc_batch_caches before it builds the pool; 1 for the
                    # single-user path. The page table is trace-baked, so its row count is fixed here.
                    max_seq=self.max_seq,
                    batch=max(1, getattr(self, "batch_size", 1)),
                    dtype=self.args.kv_cache_dtype,
                    # Extra HOST mapping rows for parked conversations. The device table keeps its
                    # `batch` rows (and therefore its baked address), so this costs nothing but a
                    # few int32s of host memory -- see PagedKV.activate.
                    nslots=self.n_conv_slots,
                )
                logger.info(f"[qwen36] paged KV: {self.kv_pager}")
            return self.kv_pager.alloc_layer_cache()
        # fixed-shape KV cache for an attention layer: [1, n_kv, max_seq, head_dim]
        shape = [1, self.args.n_kv_heads, self.max_seq, self.args.head_dim]
        kvdt = self.args.kv_cache_dtype
        k = ttnn.zeros(shape, dtype=kvdt, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        v = ttnn.zeros(shape, dtype=kvdt, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
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

    # Escape hatch, OFF by default. This existed only to bound the program cache while the
    # long-conversation wedge was unexplained; that wedge is now root-caused (lazily-built per-width
    # constants corrupted under a live decode trace -- see prewarm_prefill_shapes and
    # IMPLEMENTATION_GUIDE.md section 5) and the cache was never the problem:
    #   * the whole cache is ~31-42 MiB of kernel binaries (~70-79 KiB/program) against >10 GB free;
    #   * a bare op loop reached 801 cached programs with 21.5 GB of DRAM ballast AND a resident
    #     trace being replayed, with no trouble.
    # Leaving it enabled is actively harmful once the ladder is four widths wide: the steady-state
    # working set is ~544 entries, so a 450 limit fires mid-conversation, drops every trace and every
    # binary, and the next request recompiles -- MEASURED as 0.55 s and 0.63 s prefills against
    # 0.34 s on the turns either side of it. If you ever do need it, set it ABOVE the working set;
    # setting it below is a cure worse than the disease. QWEN36_PROGRAM_CACHE_LIMIT=<n> enables.
    _PROGRAM_CACHE_LIMIT = int(os.environ.get("QWEN36_PROGRAM_CACHE_LIMIT", "0"))

    _CKPT_DEPTH = int(os.environ.get("QWEN36_STATE_CKPT_DEPTH", "4"))

    # Per-conversation BAND depth once several conversations are parked (n_conv_slots > 1). The ring
    # is partitioned statically, band `c` = ring slots [c*band, (c+1)*band), so one conversation can
    # never evict another's checkpoint -- which a shared LRU ring would do constantly under the
    # traffic this exists for (a 6.3K conversation interleaved with 200-400 token side requests
    # would lose its only checkpoint to them every turn). Static partitioning also makes the owner of
    # a ring slot derivable from its index, so no slot needs an owner tag.
    #
    # 2 rather than _CKPT_DEPTH: at 40 layers a slot is ~45 MiB, so 4 conversations x 4 = 720 MiB
    # against ~10.6 GB free, and the extra depth only buys tolerance for edits to EARLIER turns
    # (append-only chat, the case that matters, needs exactly one). QWEN36_CKPT_BAND overrides.
    _CKPT_BAND_ENV = int(os.environ.get("QWEN36_CKPT_BAND", "0"))

    @property
    def CKPT_BAND(self):
        """Checkpoints held per parked conversation."""
        if self._CKPT_BAND_ENV:
            return max(1, self._CKPT_BAND_ENV)
        return self._CKPT_DEPTH if self.n_conv_slots <= 1 else 2

    # Snapshot / resume grain, now decoupled from the prefill chunk. It USED to be forced to the
    # chunk (512) because resuming at a 128-multiple that is not a chunk-multiple was measured to
    # hang the device -- that turned out to be the per-width-constant corruption fixed by
    # prewarm_prefill_shapes, and both offending configurations now pass (see plan_prefill_blocks'
    # HISTORY note). So 128 is available. It is NOT the default, because it was measured to be a
    # wash, 40 layers, 10-turn growing chat, prefill seconds on the reuse turns:
    #
    #     grain 512: 0.33 0.55 0.88 0.33 0.55 0.61 0.88 0.33 0.56   mean 0.558  worst 0.88
    #     grain 128: 0.56 0.32 0.82 0.54 0.55 0.55 0.56 0.56 0.55   mean 0.557  worst 0.82
    #
    # Identical mean. The reason is the measured block cost, 127 ms fixed + 0.95 ms/token: the finer
    # grain re-ingests up to 383 fewer tokens (~364 ms) but always needs a SECOND block to land the
    # snapshot on the grain (+127 ms) and pays padding on both, which cancels it. It does flatten the
    # sawtooth, and it costs branch-reuse coverage: at 4 ring slots, 512 spans 2048 tokens of history
    # and 128 spans only 512, so an edit to an earlier turn is likelier to miss. QWEN36_CKPT_ALIGN=128
    # selects it. The real win needs ONE block, i.e. snapshotting at 128 boundaries INSIDE the block
    # (the gated-delta chunked recurrence already iterates in 128-token chunks) -- predicted ~370 ms
    # flat against 0.55 s now.
    # Resolved in the property, not here: _PREFILL_CHUNK is defined further down the class body.
    _CKPT_ALIGN_ENV = int(os.environ.get("QWEN36_CKPT_ALIGN", "0"))

    @property
    def CKPT_ALIGN(self):
        """Context lengths a snapshot may be taken at, and therefore the resume grain.

        With in-block checkpoints the fine grain is free -- the tail is still ONE block, so there is
        no second MoE sweep to pay for -- and it cuts the tokens a turn re-ingests from up to
        chunk-1 to up to 127. Without them the fine grain needs its own block and measures as a wash
        (see _CKPT_ALIGN_ENV), so fall back to the chunk."""
        if self._CKPT_ALIGN_ENV:
            return self._CKPT_ALIGN_ENV
        return _RAGGED_Q_CHUNK if self.supports_inblock_snapshot else self._PREFILL_CHUNK

    def _gdn_state_keys(self):
        """(cache index, key) for every gated-delta tensor a PREFILL reads or writes.

        `conv_rows` is deliberately absent: it is the decode add-chain's own copy of the conv window,
        re-seeded from conv_state by start_decode (sync_conv_rows), so restoring conv_state is enough
        and a snapshot taken during prefill never has to know about it."""
        return [
            (i, k)
            for i, c in enumerate(getattr(self, "caches", []) or [])
            if isinstance(c, dict)
            for k in ("recurrent_state", "conv_state")
        ]

    # ---------------------------------------------------------------- parked conversations
    def alloc_conversation_slots(self, n: int):
        """Park up to `n` conversations at once. Call BEFORE the caches are allocated.

        Ordering matters twice over. The KV pager is built inside the first `_alloc_cache`, and its
        host mapping row count is fixed there; the checkpoint ring is sized from `n` as well. Both
        are read from `self.n_conv_slots`, so this must run before `alloc_state_checkpoints()` and
        before the model builds its caches -- i.e. immediately after construction.

        REQUIRES paged KV. This is not a policy choice, it is the whole mechanism: with the flat
        `[1, n_kv, max_seq, hd]` cache every conversation writes the same rows starting at 0, so two
        of them cannot coexist no matter how much state the caller remembers. Raising here is
        deliberate -- silently falling back to one slot would leave a server advertising a feature
        that quietly does nothing, which is exactly the failure this whole change exists to fix.
        """
        n = max(1, int(n))
        if n > 1 and not self.paged_kv:
            raise RuntimeError(
                f"alloc_conversation_slots({n}) requires paged KV (QWEN36_PAGED_KV=1): parked "
                "conversations each need their own KV blocks, and the flat cache has one set of rows"
            )
        if getattr(self, "caches", None) is not None or getattr(self, "_ckpt_ring", None) is not None:
            raise RuntimeError(
                "alloc_conversation_slots must run before the caches and the checkpoint ring are "
                f"allocated (n_conv_slots is baked into both); found n_conv_slots={self.n_conv_slots}"
            )
        self.n_conv_slots = n
        self._conv_active = 0

    def activate_conversation(self, conv: int):
        """Make `conv` the conversation that prefill and decode read and write.

        Switches the device page table to that conversation's block mapping -- CONTENTS only, at the
        table's baked address, which is the same in-place rewrite `PagedKV.ensure` already does on
        block boundaries and is therefore safe with a decode trace resident (MEMORY.md §1). Never
        call this from inside a captured graph: it can issue a host->device write.
        """
        if not 0 <= conv < self.n_conv_slots:
            raise IndexError(f"conversation {conv} out of range (n_conv_slots={self.n_conv_slots})")
        self._conv_active = conv
        if self.paged_kv and self.kv_pager is not None:
            self.kv_pager.activate(conv)
        return conv

    def release_conversation(self, conv: int):
        """Age a conversation out: return its KV blocks to the pool and forget its checkpoints.

        The ring BUFFERS stay allocated (they are pre-allocated once, before any trace capture, and
        must never be freed and re-made -- see alloc_state_checkpoints); only the positions they
        advertise are cleared, so the band is immediately reusable by the next conversation.

        `PagedKV.release` deliberately leaves the DEVICE page table alone, so releasing the
        conversation that is currently shown leaves stale entries in it. That is safe here and only
        here: the caller releases a conversation either (a) to prefill a new one into the same slot,
        and that prefill's `ensure` remaps from block 0 and rewrites the table before anything reads
        it, or (b) to free blocks for a DIFFERENT slot, which is not the shown one. Do not add a
        decode between the release and the prefill."""
        self.drop_state_checkpoints(conv=conv)
        if self.paged_kv and self.kv_pager is not None:
            self.kv_pager.release(conv)

    def conversation_tokens(self, conv: int):
        """Positions of `conv` that currently have KV blocks mapped (0 when not paging)."""
        if self.paged_kv and self.kv_pager is not None:
            return self.kv_pager.mapped_tokens(conv)
        return 0

    def kv_headroom_tokens(self):
        """Tokens of KV the pool can still hand out, across all conversations.

        Parked conversations hold their blocks until they are aged out, so a caller that parks
        several long ones has to check this BEFORE a prefill: the pool has no eviction of its own
        and `ensure` raises "block pool exhausted" mid-prefill, which is a failed request rather
        than a slow one. Returns a large number when not paging (the flat cache is per-position and
        bounded by max_seq, which prefill already asserts)."""
        if not (self.paged_kv and self.kv_pager is not None):
            return self.args.max_seq_len
        _, free = self.kv_pager.slot_usage()
        return free * self.kv_pager.block_size

    def alloc_state_checkpoints(self):
        """Allocate the snapshot ring from ``args`` alone, WITHOUT needing the caches to exist.

        Call this immediately after building the model, before any prefill and before any trace
        capture. That ordering is the whole point, and it is a MEASURED requirement, not hygiene:
        allocating the ring the lazy way -- cloning the caches once they exist, i.e. after the first
        prefill's transients have been allocated and freed -- drops 180 MiB (40 layers) into that
        freed hole and fragments the region the traces and later prefill transients share. The 40-layer
        server then served its first request and HUNG on the second, inside an ordinary
        `ttnn.concat` in the gated-delta conv, needing a `tt-smi -r`. Same server with
        QWEN36_PREFIX_REUSE=0 (no ring at all) served the same conversation fine, which is what pinned
        it on the allocation rather than on anything prefix reuse actually does -- reuse had not even
        engaged, the prompts being shorter than one chunk. See MEMORY.md §1-3 and
        tests/probe_prefix_reuse_hang.py."""
        if getattr(self, "_ckpt_ring", None) is not None:
            return
        a = self.args
        z = lambda shape, dt: ttnn.zeros(shape, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        spec = []  # (cache index, key, shape, dtype) -- mirrors _alloc_cache's linear-layer branch
        for i, layer in enumerate(self.layers):
            if layer.is_linear:
                spec.append(
                    (i, "recurrent_state", [1, a.lin_num_v_heads, a.lin_head_k_dim, a.lin_head_v_dim], _STATE_DT)
                )
                spec.append((i, "conv_state", [a.conv_kernel_size - 1, a.lin_conv_dim], ttnn.bfloat16))
        self._ckpt_ring = [{(i, k): z(sh, dt) for i, k, sh, dt in spec} for _ in range(self._ckpt_slots())]
        self._ckpt_reset_bands()
        self._log_ckpt_ring("pre-allocated")

    def _ensure_ckpt_ring(self):
        """Allocate the snapshot ring (all slots at once -- see the class note). Requires caches.

        FALLBACK for callers that did not call alloc_state_checkpoints() first -- prefer that one,
        which allocates before any prefill has churned the allocator. This variant clones the caches,
        so it can only run once they exist.

        MUST run before the first trace capture, which is why prefill_from calls it as soon as the
        caches exist rather than waiting for the first snapshot. Allocating it later -- e.g. lazily on
        the first prompt long enough to checkpoint, by which time warmup has captured a decode trace
        -- lands 30-126 MiB in the hole that capture's freed transients left, and MEASURED that
        wedged the device on the NEXT request (`allocator.cpp:123`: "Allocating device buffers is
        unsafe due to the existence of an active trace"). Same failure mode as the multi-bucket
        prefill wedge; see MEMORY.md §1-3. The warning below is the canary for a regression."""
        if getattr(self, "_ckpt_ring", None) is not None:
            return
        if getattr(self, "trace_id", None) is not None:
            logger.warning(
                "[qwen36] allocating the gated-delta checkpoint ring while a decode trace is LIVE -- "
                "this is the trace/allocator aliasing hazard (MEMORY.md §1-3) and can corrupt the "
                "trace. It should have been allocated with the caches; please file this."
            )
        keys = self._gdn_state_keys()
        self._ckpt_ring = [{k: ttnn.clone(self.caches[k[0]][k[1]]) for k in keys} for _ in range(self._ckpt_slots())]
        self._ckpt_reset_bands()
        self._log_ckpt_ring("cloned from caches")

    # ---- ring geometry. Slot i belongs to conversation i // CKPT_BAND; see _CKPT_BAND_ENV. -----
    def _ckpt_slots(self):
        return max(1, self.n_conv_slots) * max(1, self.CKPT_BAND)

    def _ckpt_reset_bands(self):
        self._ckpt_pos = [None] * len(self._ckpt_ring)
        self._ckpt_next = [0] * max(1, self.n_conv_slots)  # per-band round-robin cursor

    def _ckpt_span(self, conv=None):
        """`range` of ring slot indices owned by `conv` (the active conversation by default)."""
        conv = self._conv_active if conv is None else conv
        band = max(1, self.CKPT_BAND)
        return range(conv * band, min((conv + 1) * band, len(self._ckpt_ring or ())))

    def _ckpt_cursor(self, conv=None):
        """Absolute ring index the next snapshot for `conv` will be written into."""
        conv = self._conv_active if conv is None else conv
        span = self._ckpt_span(conv)
        return span[self._ckpt_next[conv] % len(span)]

    def _log_ckpt_ring(self, how):
        mib = sum(t.volume() * t.element_size() for t in self._ckpt_ring[0].values()) / 2**20
        n = len(self._ckpt_ring)
        logger.info(
            f"[qwen36] gated-delta state checkpoints: {n} x {mib:.1f} MiB ({n * mib:.1f} MiB total, "
            f"{how}) = {self.n_conv_slots} conversation slot(s) x {self.CKPT_BAND} checkpoint(s)"
        )

    def snapshot_gdn_state(self, pos: int) -> bool:
        """Record the gated-delta state as it stands at context length ``pos`` (a CKPT_ALIGN multiple),
        overwriting the oldest ring slot. In-place copies only; no allocation after the first call."""
        if pos <= 0 or pos % self.CKPT_ALIGN != 0:
            return False
        self._ensure_ckpt_ring()
        conv = self._conv_active
        slot = self._ckpt_cursor(conv)
        for key, dst in self._ckpt_ring[slot].items():
            ttnn.copy(self.caches[key[0]][key[1]], dst)
        self._ckpt_pos[slot] = pos
        self._ckpt_next[conv] += 1
        return True

    @property
    def supports_inblock_snapshot(self):
        """True when every linear mixer can export its state at an interior row of a block."""
        return all(l.mixer.supports_inblock_snapshot for l in self.layers if l.is_linear)

    def _begin_inblock_snapshot(self, pos: int):
        """Claim the ring slot an in-block checkpoint at ``pos`` will be written into, and return it.

        Unlike snapshot_gdn_state, which COPIES the caches after a block, this hands the destination
        buffers to the layers so they write the interior state as they compute it -- there is no
        moment afterwards when the caches still hold the state at ``pos``. The slot's position is
        published only by _commit_inblock_snapshot, after the block has actually run, so a failure
        part-way cannot leave a slot advertising a state it does not hold."""
        if pos <= 0 or pos % self.CKPT_ALIGN != 0:
            return None
        self._ensure_ckpt_ring()
        return self._ckpt_ring[self._ckpt_cursor()]

    def _commit_inblock_snapshot(self, pos: int):
        """Publish the slot claimed by _begin_inblock_snapshot and advance the active band."""
        conv = self._conv_active
        self._ckpt_pos[self._ckpt_cursor(conv)] = pos
        self._ckpt_next[conv] += 1

    def checkpoint_positions(self, conv=None):
        """Context lengths the gated-delta state can currently be rewound to, ascending.

        Scoped to ONE conversation (the active one by default): a checkpoint from another
        conversation describes a different token sequence and a different set of KV blocks, so it
        must never appear in this one's resume search. Identical to the flat list when only one
        conversation slot exists."""
        if not getattr(self, "_ckpt_pos", None):
            return []
        return sorted(p for i in self._ckpt_span(conv) if (p := self._ckpt_pos[i]) is not None)

    def restore_gdn_state(self, pos: int, conv=None) -> bool:
        """Rewind the gated-delta state to context length ``pos`` and set self.pos. False if no
        snapshot for ``pos`` is held.

        Nothing else needs rewinding: KV rows at and beyond ``pos`` are overwritten by the resumed
        prefill (and any left over past its end are masked by causality / `cur_pos`), and `conv_rows`
        is rebuilt from the restored conv_state when start_decode runs."""
        ring = getattr(self, "_ckpt_ring", None)
        if ring is None:
            return False
        slot = next((i for i in self._ckpt_span(conv) if self._ckpt_pos[i] == pos), None)
        if slot is None:
            return False
        for key, src in ring[slot].items():
            ttnn.copy(src, self.caches[key[0]][key[1]])
        self.pos = pos
        return True

    def drop_state_checkpoints(self, above: int | None = None, conv=None):
        """Invalidate snapshots (buffers stay allocated). ``above`` keeps only positions <= it, which is
        how a prefill that SHORTENS the sequence discards snapshots belonging to the longer one -- a
        stale snapshot at a position the new sequence also reaches would otherwise restore the wrong
        context's state. ``None`` invalidates all.

        ``conv`` scopes it to ONE conversation's band; ``None`` (the default) spans every band, which
        is what a warmup or a global reset wants. Note the asymmetry, because it is the bug this
        whole change fixes: ``drop_state_checkpoints(above=T)`` unscoped is exactly what let a
        200-token side request wipe a 6.3K conversation's only checkpoint, so every per-request call
        in prefill_reuse passes its own ``conv``."""
        if getattr(self, "_ckpt_pos", None) is None:
            return
        span = range(len(self._ckpt_pos)) if conv is None else self._ckpt_span(conv)
        for i in span:
            p = self._ckpt_pos[i]
            if above is None or (p is not None and p > above):
                self._ckpt_pos[i] = None
        if above is None:  # the band(s) are empty: restart their round-robin at the bottom
            for c in range(len(self._ckpt_next)) if conv is None else (conv,):
                self._ckpt_next[c] = 0

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

    def _prefill_single(self, input_ids: torch.Tensor, slot: int = 0, valid_len: int | None = None):
        """Single-shot prefill (bounded to short prompts: materializes the full [T,*] activations).
        Allocates per-layer caches and sets self.pos.

        `slot` selects which PAGED KV row this prompt's blocks come from (continuous batching); it is
        ignored on the flat path, where the caller instead copies row-wise after the fact. See
        prefill_into_slot."""
        self._release_traces_for_prefill(widths={int(input_ids.reshape(1, -1).shape[1])})
        T = input_ids.shape[1]  # the (possibly bucket-padded) width every op runs at
        real = T if valid_len is None else int(valid_len)  # the real token count
        self.max_seq = self.args.max_seq_len
        # Allocate per-layer caches ONCE and reuse them across prefills (the same _pf_caches_ready
        # guard the traced-prefill path uses). Reallocating per request was the OOM: the decode
        # trace pins the caches it references, so the old ones could never be freed.
        if getattr(self, "caches", None) is None or not getattr(self, "_pf_caches_ready", False):
            self.caches = [self._alloc_cache(layer) for layer in self.layers]
            self._pf_caches_ready = True
        else:
            self._reset_linear_state()  # fresh prefill: clear the gated-delta conv left-pad
        self.pos = real
        prof.reset()
        with prof.phase(self.mesh_device, "embed"), sp.region("embed"):
            x = self._embed(input_ids, T)
        with prof.phase(self.mesh_device, "rope"), sp.region("rope"):
            cos, sin = precompute_rope(T, self.args.rotary_dim, self.args.rope_theta, self.mesh_device)
        fpt = self._kv_fill_table(T, slot)
        for layer, cache in zip(self.layers, self.caches):
            x = layer.forward_prefill(x, cos, sin, cache, fill_page_table=fpt, valid_len=valid_len)
        if self._keep_prefill_hidden:
            # The per-position POST-final-norm hidden, for prime_mtp. This is the only moment it
            # exists: _head slices to the last row BEFORE the norm (a T-fold lm_head + D2H saving),
            # so nothing downstream can reconstruct it. Opt-in because it is dead weight (one T-row
            # RMSNorm and a [1,1,T,dim] tensor) for any request that will not speculate.
            with sp.region("prefill.keep_hidden"):
                self._prefill_hidden = self.final_norm.forward(x)
        with prof.phase(self.mesh_device, "head"), sp.region("head"):
            out = self._head(x, real)  # the REAL last token, not the padding
        prof.report()
        return out

    def prefill_long(self, input_ids: torch.Tensor, chunk: int | None = None, slot: int = 0):
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
        self._release_traces_for_prefill()
        M = chunk or self._PREFILL_CHUNK
        T = input_ids.shape[1]
        assert M % 128 == 0, f"prefill chunk {M} must be a multiple of 128"
        assert T <= self.args.max_seq_len, f"prompt {T} exceeds max_seq_len {self.args.max_seq_len}"
        if T <= M:
            return self._prefill_single(input_ids, slot)
        logits = self._prefill_single(input_ids[:, :M], slot)  # first block: allocates caches, self.pos=M
        n_full = (T // M) * M  # tokens covered by whole M-token chunks
        for P in range(M, n_full, M):
            logits = self.forward_incremental(input_ids[:, P : P + M], slot=slot)  # state carries; pos -> P+M
        if T > n_full:  # ragged final block (T mod M tokens); forward_incremental pads it internally
            logits = self.forward_incremental(input_ids[:, n_full:T], ragged=True, slot=slot)
        # _prefill_single stashed a hidden for the FIRST CHUNK only, and the later chunks' hiddens were
        # never materialised — priming off that would feed the head a truncated prompt while claiming
        # the whole one. Drop it: prime_mtp then no-ops and long prompts keep today's behaviour.
        self._prefill_hidden = None
        return logits

    def prefill_from(self, input_ids: torch.Tensor, start: int = 0, slot: int = 0):
        """Prefill ``input_ids`` (torch [1, T]) computing only positions [start, T), and snapshot the
        gated-delta state at every block boundary crossed so the NEXT turn can resume in turn.

        ``start`` is 0 (from scratch) or a CKPT_ALIGN multiple for which restore_gdn_state has just
        run. Returns the last token's logits [1, 1, vocab], like forward().

        Block structure, and why it differs from prefill_long: the prompt is ingested up to
        ``floor(T / CKPT_ALIGN) * CKPT_ALIGN`` and its remainder handled as one ragged block, so a
        snapshot always exists at the last CKPT_ALIGN multiple -- which is where the next turn's
        divergence point (typically T-2) floors to. For a prompt short enough that forward() would
        have gone single-shot, the first block is kept that wide, so the only added work versus
        today's path is one extra MoE pass over the (<= CKPT_ALIGN-1 token) remainder. That pass is
        the price of the next turn's reuse; callers that will never resume should use forward()."""
        ids = input_ids.reshape(1, -1)
        T = int(ids.shape[1])
        assert T <= self.args.max_seq_len, f"prompt {T} exceeds max_seq_len {self.args.max_seq_len}"
        plan = plan_prefill_blocks(
            T,
            start=start,
            chunk=self._PREFILL_CHUNK,  # coarse structure: keeps block widths on the padding ladder
            snap=self.CKPT_ALIGN,  # one source of truth: the plan's grain IS the snapshot's
            inblock=self.supports_inblock_snapshot,
            single_shot_max=self._LONG_PREFILL_THRESHOLD,
        )
        # The padded widths this prefill will touch; whether any is new decides whether the resident
        # decode trace has to be dropped first (see _release_traces_for_prefill).
        self._release_traces_for_prefill(
            widths={
                pad_width(m, limit=self.args.max_seq_len)
                if kind == "single"
                else pad_width(m, limit=max(self.args.max_seq_len - P, _RAGGED_Q_CHUNK))
                for P, m, kind, _q in plan
            }
        )
        logits = None
        if getattr(self, "_pf_caches_ready", False):
            self._ensure_ckpt_ring()  # before any capture -- see _ensure_ckpt_ring
        for P, m, kind, q_chunk in plan:
            block = ids[:, P : P + m]
            logger.debug(f"[prefill_from] block {kind} P={P} m={m} q_chunk={q_chunk} (T={T}, start={start})")
            _pc0 = self.mesh_device.num_program_cache_entries()
            if kind == "single":
                # Bucket a short prompt up to a _RAGGED_Q_CHUNK multiple and mask the padding, exactly
                # as the ragged tail does. Without this, every prompt under one chunk runs at its RAW
                # length and compiles ~90 new device programs that will never be reused -- the common
                # case in chat, and the last unbounded source feeding the program-cache ceiling.
                w = pad_width(m, limit=self.args.max_seq_len)
                if w != m:
                    padded = torch.zeros(1, w, dtype=block.dtype)
                    padded[0, :m] = block[0]
                    logits = self._prefill_single(padded, slot, valid_len=m)
                else:
                    logits = self._prefill_single(block, slot)  # allocates caches, resets the conv left-pad
                self._ensure_ckpt_ring()  # the caches now exist, and no trace can have been captured yet
            else:
                # In-block checkpoint for the LAST block: the deepest grain multiple this prompt
                # reaches usually falls INSIDE the final block, and exporting the recurrence state
                # there costs one extra slice per linear layer instead of a whole extra block.
                snap_at = None
                if self.supports_inblock_snapshot:
                    grain = self.CKPT_ALIGN
                    snap_end = (T // grain) * grain
                    if P < snap_end < P + m:
                        snap_at = snap_end - P
                logits = self.forward_incremental(
                    block, ragged=kind == "ragged", slot=slot, q_chunk=q_chunk, snap_at=snap_at
                )
            _pc1 = self.mesh_device.num_program_cache_entries()
            logger.debug(f"[prefill_from] {kind} P={P} m={m} compiled {_pc1 - _pc0} new programs (total {_pc1})")
            if kind != "ragged":  # the remainder does not land on a CKPT_ALIGN multiple
                self.snapshot_gdn_state(P + m)
        # Never valid for a multi-block prefill: only the FIRST block's hidden was materialised (see
        # prefill_long), so priming the MTP head off it would claim a prompt it never saw.
        self._prefill_hidden = None
        return logits

    # Prefill must never allocate device buffers while a decode trace is resident. tt-metal warns
    # about exactly this once per device ("Allocating device buffers is unsafe due to the existence
    # of an active trace. These buffers may be corrupted once a trace is executed.",
    # allocator.cpp:123) and then goes quiet, so the hazard is silent from then on.
    #
    # What actually gets corrupted: the read-only per-WIDTH constants the prefill path builds lazily
    # (GatedDelta._conv_stack_const, MoE._rowsel). First use of a width happens under the resident
    # trace, a replay stomps them, and being write-once they are never repaired -- so the wedge lands
    # on the next turn that REUSES that width (measured 6/6 at turn 5: width 256 built at turn 2,
    # reused at turn 5). prewarm_prefill_shapes() builds them all before any capture and is the real
    # fix; with it the trace stays resident for the whole conversation and this release never fires.
    # This remains as the backstop for a width that was never pre-warmed.
    # See IMPLEMENTATION_GUIDE.md sections 4b and 5.
    _RELEASE_TRACE_BEFORE_PREFILL = os.environ.get("QWEN36_RELEASE_TRACE_BEFORE_PREFILL", "1") != "0"

    def prewarm_prefill_shapes(self, widths=None):
        """Force every LAZY per-shape device constant into existence before any trace is captured.

        The prefill path caches read-only device constants keyed by block width and builds them on
        first use of that width -- `GatedDelta._conv_stack_const(T)` (five 0/1 selection/summation
        matrices per width, per linear layer) and `MoE._rowsel(K)` are the ones in this repo, and the
        set is not closed: any future per-shape constant inherits the same hazard. After warmup, the
        first use of a width always happens with the decode trace resident, so those constants are
        allocated under an active trace -- exactly what allocator.cpp warns about ("These buffers may
        be corrupted once a trace is executed") -- and, being read-only, they are never rewritten, so
        a single stomp is permanent. That is the same fault MEMORY.md 1-3 fixed for the prefill
        buckets by allocating every bucket's buffers before any capture.

        Running one real prefill per ladder width here fills those caches through the actual code
        paths, so nothing has to enumerate them. Costs a few seconds of startup, once."""
        if getattr(self, "trace_id", None) is not None:
            logger.warning(
                "[qwen36] prewarm_prefill_shapes ran with a decode trace already live -- it must run "
                "BEFORE the first capture or it defeats its own purpose; please file this."
            )
        chunk = self._PREFILL_CHUNK
        widths = sorted({int(w) for w in (widths or _PREFILL_WIDTHS)} | {_RAGGED_Q_CHUNK})
        ids = torch.zeros(1, chunk + max(widths) + 8, dtype=torch.long)
        for W in widths:
            self._prefill_single(ids[:, :chunk])
            # ragged=True/q_chunk=128 is the tail shape; ragged=False with q_chunk=None is the full
            # "block" shape, whose chunked-SDPA q_chunk_size is the WIDTH. They are different
            # programs, and with a runtime chunk_start the first of each costs a ~1.4 s compile --
            # which showed up as a 2.36 s request when only the tail shape had been warmed. Warm both.
            self.forward_incremental(ids[:, chunk : chunk + W], ragged=True, q_chunk=_RAGGED_Q_CHUNK)
            self._prefill_single(ids[:, :chunk])
            self.forward_incremental(ids[:, chunk : chunk + W], ragged=False)
        self.drop_state_checkpoints()
        self.pos = 0
        self._warmed_widths = set(widths)
        logger.info(
            f"[qwen36] pre-warmed per-shape prefill constants for widths {widths} " f"(before any trace capture)"
        )

    def release_traces(self):
        """Release every captured trace. They are re-captured on demand, so this is always safe at a
        quiescent point; it must NEVER run while a trace is being replayed."""
        if getattr(self, "trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self.trace_id)
            self.trace_id = None
            self._trace_sig = None
        if getattr(self, "logits_trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self.logits_trace_id)
            self.logits_trace_id = None
            self._logits_trace_sig_v = None
        for tid in list(getattr(self, "_pf_traces", {}).values()):
            ttnn.release_trace(self.mesh_device, tid)
        if hasattr(self, "_pf_traces"):
            self._pf_traces.clear()
        if getattr(self, "mtp_trace_ids", None) or getattr(self, "_mtp_traces", None):
            try:
                self.release_mtp_traces()
            except Exception as e:  # never let cleanup take the server down
                logger.warning(f"[qwen36] trace release: MTP trace release failed ({e})")

    def _release_traces_for_prefill(self, widths=None):
        """Drop the DECODE-side traces so the prefill about to run cannot allocate into space they own.

        ``widths`` are the padded block widths this prefill will use. If every one of them has already
        had its per-shape constants built (prewarm_prefill_shapes, or an earlier prefill that paid for
        it), there is nothing left to allocate and the traces can stay resident -- which is the whole
        point, since re-capturing the decode trace costs ~0.33 s. ``widths=None`` means "caller cannot
        tell us", and is treated conservatively.

        Deliberately leaves ``_pf_traces`` alone: those are prefill's own traces, they are captured
        after alloc_prefill_buffers has placed every bucket's persistent buffers (which is what fixed
        the multi-bucket aliasing), and an eager fallback prefill must not tear them down.
        """
        if not self._RELEASE_TRACE_BEFORE_PREFILL:
            return
        if widths is not None:
            warmed = getattr(self, "_warmed_widths", None)
            if warmed is not None and all(w in warmed for w in widths):
                return  # constants already exist: this prefill allocates nothing new
            # About to build constants for a new width. Do it with no trace resident, then remember
            # the width so no later turn pays this again.
            if warmed is None:
                self._warmed_widths = warmed = set()
            warmed.update(widths)
        if (
            getattr(self, "trace_id", None) is None
            and getattr(self, "logits_trace_id", None) is None
            and not getattr(self, "mtp_trace_ids", None)
            and not getattr(self, "_mtp_traces", None)
        ):
            return  # nothing resident; keep the common path free of work
        if getattr(self, "trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self.trace_id)
            self.trace_id = None
            self._trace_sig = None
        if getattr(self, "logits_trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self.logits_trace_id)
            self.logits_trace_id = None
            self._logits_trace_sig_v = None
        if getattr(self, "mtp_trace_ids", None) or getattr(self, "_mtp_traces", None):
            try:
                self.release_mtp_traces()
            except Exception as e:  # never let cleanup take the server down
                logger.warning(f"[qwen36] pre-prefill trace release: MTP release failed ({e})")
        logger.debug("[qwen36] released decode-side traces before prefill (allocator/trace aliasing)")

    def maybe_recycle_programs(self) -> bool:
        """Drop the device program cache if it has grown past ``_PROGRAM_CACHE_LIMIT``. True if it did.

        MUST be called at a quiescent point -- between requests, before a prefill -- and never while a
        trace is being replayed: every captured trace references the program binaries this frees, so
        the traces are released first and re-captured on demand. Checkpoints and caches are DATA, not
        programs, so they survive untouched and prefix reuse keeps working across a recycle."""
        limit = self._PROGRAM_CACHE_LIMIT
        if not limit:
            return False
        n = self.mesh_device.num_program_cache_entries()
        if n < limit:
            return False
        self.release_traces()
        if os.environ.get("QWEN36_RECYCLE_TRACES_ONLY") == "1":
            # Diagnostic: release the traces but KEEP the program cache. Separates "freeing the kernel
            # binaries" from "re-capturing the traces" -- the recycle does both, and only one of them
            # may be what actually prevents the hang.
            logger.info(f"[qwen36] traces released at {n} entries; program cache KEPT (diagnostic)")
            return True
        # The DRAM delta across the clear IS the kernel-binary bytes the cache was holding -- the one
        # number that distinguishes "too many programs" from "too many program BYTES".
        from models.demos.qwen3_6_a3b.tt import memstat

        before = memstat.region(self.mesh_device, ttnn.BufferType.DRAM).allocated
        self.mesh_device.clear_program_cache()
        after = memstat.region(self.mesh_device, ttnn.BufferType.DRAM).allocated
        logger.info(
            f"[qwen36] program cache recycled at {n} entries (limit {limit}); "
            f"freed {(before - after) / 2**20:.1f} MiB of kernel binaries "
            f"({(before - after) / max(n, 1) / 1024:.1f} KiB/program); next request recompiles"
        )
        return True

    def prefill_reuse(
        self, input_ids: torch.Tensor, hist_ids=None, min_reuse: int | None = None, slot: int = 0, conv=None
    ):
        """Prefill with multi-turn PREFIX REUSE: continue from the deepest gated-delta checkpoint the
        new prompt still agrees with instead of recomputing the whole context.

        ``hist_ids``: the token ids the caches currently represent, i.e. the prompt this model
        prefilled last (in order). The resume point is the longest common prefix of ``hist_ids`` and
        ``input_ids``, floored to a checkpoint we still hold. Returns ``(logits, reused)`` where
        ``reused`` is the number of prompt positions that were NOT recomputed (0 = full prefill).

        Correctness rests on comparing tokens, not on any assumption about the chat template: if the
        client re-renders history differently -- dropped reasoning, normalised whitespace,
        re-serialised tool arguments, a compacted transcript -- the common prefix simply lands
        earlier and everything after it is recomputed. The caller owns ``hist_ids`` and must set it to
        the prompt it just passed here; handing over a stale list is the one way to get this wrong,
        which is why nothing infers it from device state.

        ``conv`` selects which PARKED CONVERSATION this request belongs to (see
        alloc_conversation_slots). It activates that conversation's KV block mapping and scopes the
        checkpoint search, the resume and both invalidations to its own band, so an interleaved
        request from another conversation cannot see, restore or destroy this one's state. It also
        defaults ``slot`` -- on the single-sequence path the KV slot IS the conversation -- so
        callers pass one number, not two that must agree."""
        conv = self._conv_active if conv is None else self.activate_conversation(conv)
        if slot == 0 and conv:
            slot = conv  # the conversation's own KV blocks; see the docstring
        self.maybe_recycle_programs()  # quiescent point: before any prefill work for this request
        ids = input_ids.reshape(1, -1)
        T = int(ids.shape[1])
        start = 0
        if hist_ids is not None and len(hist_ids) and getattr(self, "caches", None) is not None:
            lcp = common_prefix_len(hist_ids, ids[0].tolist())
            start = pick_checkpoint(
                self.checkpoint_positions(conv), min(lcp, T - 1), self.CKPT_ALIGN if min_reuse is None else min_reuse
            )
        logger.debug(f"[prefill_reuse] conv={conv} T={T} start={start} checkpoints={self.checkpoint_positions(conv)}")
        if start and self.restore_gdn_state(start, conv):
            logits = self.prefill_from(ids, start=start, slot=slot)
        else:
            # Fresh prefill: every checkpoint THIS conversation holds describes a different token
            # sequence at those positions, so none of them may survive into its next turn's search.
            # Other conversations' bands are untouched -- their checkpoints still match their own
            # histories and their own KV blocks.
            self.drop_state_checkpoints(conv=conv)
            start = 0
            logits = self.prefill_from(ids, start=0, slot=slot)
        # Discard checkpoints past the end of THIS sequence (this conversation's previous prompt may
        # have been longer): the next turn's prefix can only reach T, but a stale snapshot at a
        # position <= T would restore an older revision of the same conversation at a position this
        # one also has. Scoped to `conv` -- unscoped, this line WAS the bug.
        self.drop_state_checkpoints(above=T, conv=conv)
        return logits, start

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

    def alloc_prefill_buffers(self, T):
        """Allocate the persistent buffers bucket ``T``'s traced prefill reads and writes: the input id
        buffer, its RoPE tables, the output landing buffer, and the gated-delta pool. Idempotent.

        MUST happen for EVERY bucket BEFORE the FIRST capture, which is why it is split out of
        capture_prefill_trace. Reason, MEASURED 2026-08-22 (t4_smoking_gun.py):

          A trace bakes the ADDRESSES of the buffers it touches, and nothing pins its in-graph
          transients — so they are freed the moment `end_trace_capture` returns. Allocating the NEXT
          bucket's persistent buffers then hands them exactly that freed hole. Replaying the earlier
          trace afterwards writes its recorded addresses, which now belong to the later bucket's pool:
          observed bucket 512's read-only zero `S0` come back with |max| 33.19 and its `out[0]` come
          back NaN after a single bucket-256 replay. That is the "multi-bucket prefill wedge" — the
          later-captured bucket replayed at PCC ~0 (-0.0495), the earlier one always fine, whichever
          way round the two were captured.

          It was never a footprint problem (each bucket retains only ~18-21 MB against ~11 GB free,
          which is why raising trace_region_size never helped) and never specific to bucket 512.

        Allocating every bucket's persistent buffers up front puts them all BELOW the region the
        captures then use for transients, so no trace's recorded writes can land on another bucket's
        state. The transient region is still shared between traces, which is harmless: every transient
        is written before it is read within a replay, so stale bytes there are overwritten, unlike the
        read-only `S0` that used to be corrupted. tt-metal warns about exactly this hazard in
        allocator.cpp ("buffers allocated when a trace is active may be corrupted once a trace is
        executed")."""
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

    def record_prefill_trace(self, T):
        """Eager compile pass (capture cannot JIT) then record bucket ``T``'s trace. PRECONDITION:
        alloc_prefill_buffers has run for EVERY bucket that will be captured — see its docstring."""
        self._prefill_graph(T)
        ttnn.synchronize_device(self.mesh_device)
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._prefill_graph(T)
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        self._pf_traces[T] = tid

    def capture_prefill_trace(self, T):
        """Allocate + record ONE bucket. Safe on its own; for several buckets use
        setup_prefill_traces, which allocates them all before capturing any (see
        alloc_prefill_buffers for why interleaving the two corrupts the later bucket)."""
        self.alloc_prefill_buffers(T)
        self.record_prefill_trace(T)

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
        # TWO passes, and the order is load-bearing: every bucket's persistent buffers must exist
        # before the first capture, or the later buckets' pools get allocated into the hole left by an
        # earlier capture's freed transients and are then overwritten by that earlier trace's replay.
        # See alloc_prefill_buffers for the measurement. This is the multi-bucket wedge fix.
        for B in self._pf_buckets:
            self.alloc_prefill_buffers(B)
        for B in self._pf_buckets:
            self.record_prefill_trace(B)

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
        MULTI-BUCKET: works as of 2026-08-22. It was never a footprint problem — it was trace address
        aliasing (an earlier trace's replay writing into a later bucket's pool, because the earlier
        capture's transients were freed and the later pool was allocated into the hole). Fixed by
        allocating every bucket's buffers before any capture; see alloc_prefill_buffers and MEMORY.md.
        Opt-in in demo/server.py via QWEN36_SERVER_PREFILL_TRACE."""
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
        # Fresh prefill: reset the conv left-pad. The conv reads conv_state as its left-pad and that
        # buffer persists across replays, so it has to be cleared — but IN PLACE, via the same
        # multiply-by-zero _reset_linear_state uses, not by copying from a zero buffer. This runs
        # eagerly before execute_trace, so it is an ordinary op; the old form kept a whole
        # [conv_k-1, conv_dim] zero source alive and paid a DRAM->DRAM copy per linear layer per
        # replay to read it. (recurrent_state is output-only — the chunked path starts S=0 internally;
        # KV is written fresh and prefill attends fresh k/v, so only conv_state needs resetting.)
        self._reset_linear_state()
        ttnn.execute_trace(self.mesh_device, self._pf_traces[B], cq_id=0, blocking=False)
        self.pos = T
        if self._keep_prefill_hidden:
            # Same stash as _prefill_single, sliced to the REAL length first: rows T..B-1 are padding.
            # Kept in step with the eager path so enabling prefill tracing cannot silently turn
            # MTP priming off.
            real = ttnn.slice(self._pf_out[B], [0, 0, 0, 0], [1, 1, T, self.args.dim])
            self._prefill_hidden = self.final_norm.forward(real)
        return self._head(self._pf_out[B], T)  # head slices the REAL last token (T-1), not the bucket

    def forward_incremental(
        self, input_ids: torch.Tensor, ragged: bool = False, slot: int = 0, q_chunk=None, snap_at: int | None = None
    ):
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
        reads the real last token, so the padded KV rows (past self.pos) are never read.

        `slot` selects the PAGED KV row (continuous batching); ignored on the flat path, which has one
        cache per model. See _kv_read_table for why the read table must be per-slot.

        `q_chunk` overrides chunked-SDPA's q_chunk_size. It defaults to the block width M, which is
        only legal while P is a multiple of M -- true of the from-scratch chunked walk, but NOT of a
        prefill RESUMED from a state checkpoint, whose offset is only a CKPT_ALIGN multiple. Callers
        on that path pass 128 (see prefill_from)."""
        self._release_traces_for_prefill(
            widths={
                pad_width(int(input_ids.reshape(1, -1).shape[1]), limit=max(self.max_seq - self.pos, _RAGGED_Q_CHUNK))
            }
        )
        M = input_ids.shape[1]
        P = self.pos
        if ragged:
            # a width from the ladder, NOT the minimal 128-multiple (see _PREFILL_WIDTHS), capped by
            # the room left in the KV cache -- padding writes real rows and must not run past max_seq
            M_pad = pad_width(M, limit=self.max_seq - P)
            valid = M if M_pad != M else None  # tell gated-delta the real length when we padded
            q_chunk = q_chunk or 128  # q_chunk_size that divides P (multiple of the chunk) and M_pad
            assert P + M_pad <= self.max_seq, (
                f"ragged block overflows the KV cache: P({P}) + padded_M({M_pad}) > max_seq({self.max_seq}); "
                f"raise QWEN36_MAX_SEQ to a multiple of 128 >= {((P + M + 127) // 128) * 128}"
            )
            if M_pad != M:
                padded = torch.zeros(1, M_pad, dtype=input_ids.dtype)
                padded[0, :M] = input_ids[0]
                input_ids = padded
        else:
            M_pad, valid = M, None  # full chunk: already aligned; q_chunk None -> the kernel uses M
        inc_fpt = self._kv_fill_table_at(P, M_pad, slot)
        read_pt = self._kv_read_table(slot)
        x = self._embed(input_ids, M_pad)
        if _LAYER_SYNC:
            ttnn.synchronize_device(self.mesh_device)
            logger.info(f"[inc] P={P} M={M_pad} embed ok")
        cos, sin = precompute_rope(M_pad, self.args.rotary_dim, self.args.rope_theta, self.mesh_device, start_pos=P)
        if _LAYER_SYNC:
            ttnn.synchronize_device(self.mesh_device)
            logger.info(f"[inc] P={P} M={M_pad} rope ok")
        # In-block checkpoint: hand each LINEAR layer the ring slot it should write its state into
        # `snap_at` rows into this block. That leaves a resumable snapshot at P + snap_at without
        # splitting the block, so the block's MoE sweep (127 ms, PREFILL.md 0.9) is paid once.
        snap_slot = self._begin_inblock_snapshot(P + snap_at) if snap_at else None
        for li, (layer, cache) in enumerate(zip(self.layers, self.caches)):
            x = layer.forward_prefill_incremental(
                x,
                cos,
                sin,
                cache,
                read_pt,  # READ table for chunked SDPA over this slot's accumulated cache
                P,
                valid_len=valid,
                q_chunk=q_chunk,
                fill_page_table=inc_fpt,
                snap_at=snap_at,
                snap_dst=None
                if snap_slot is None
                else {k: snap_slot[(li, k)] for k in ("recurrent_state", "conv_state") if (li, k) in snap_slot},
            )
            if _LAYER_SYNC:
                ttnn.synchronize_device(self.mesh_device)
                logger.info(
                    f"[inc] P={P} M={M_pad} valid={valid} layer {li} " f"({'linear' if layer.is_linear else 'attn'}) ok"
                )
        if snap_slot is not None:
            self._commit_inblock_snapshot(P + snap_at)
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
        # These five are written IN PLACE at stable addresses, not reallocated per request. They are
        # read by the captured decode graph, so a fresh allocation moved their addresses and silently
        # invalidated the trace -- which is why a re-capture was needed every request (see
        # _decode_state_buf for what that cost). The batched path already did it this way in
        # set_sampling_params_batch; this is the single-user analogue.
        self._h2d([k] * B, ttnn.uint32, self._decode_state_buf("t_k", ttnn.uint32, B))
        self._h2d([p] * B, ttnn.bfloat16, self._decode_state_buf("t_p", ttnn.bfloat16, B), torch.float32)
        # ttnn.sampling scales by 1/T
        self._h2d(
            [1.0 / temperature] * B, ttnn.bfloat16, self._decode_state_buf("t_temp", ttnn.bfloat16, B), torch.float32
        )
        # Per-step RNG seed, advanced on device each decode step (plus_one) so a static trace still
        # draws fresh randomness every replay; deterministic for a fixed starting `seed`.
        self._h2d([seed] * B, ttnn.uint32, self._decode_state_buf("t_seed", ttnn.uint32, B))
        # Presence mask over the vocab (generated tokens), accumulated on device across a request.
        # Allocated once; the per-request RESET is now explicit. It used to come free from
        # reallocating the tensor, and losing it would leak one request's repetition penalties into
        # the next -- the in-place multiply-by-zero is the same trick _reset_linear_state uses (a
        # device op, no host write, so it stays legal next to a live trace).
        if getattr(self, "t_presence", None) is None or self.t_presence.shape[-1] != V:
            self.t_presence = ttnn.zeros(
                [1, 1, 1, V], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
            )
        else:
            ttnn.multiply(self.t_presence, 0.0, output_tensor=self.t_presence)

    def _decode_state_buf(self, name, dtype, n=1):
        """Persistent [n] decode-state buffer, allocated once and thereafter REUSED at a stable address.

        The address is the point. A captured decode trace bakes in the addresses of the buffers it
        reads, so reallocating `t_tok`/`t_curpos`/`t_ropepos` per request — which `start_decode` used to
        do — silently invalidated the trace and forced a re-capture every request. That re-capture is
        not cheap: demo/server.py measures the 16 steps after one at 43-46 ms/token against a settled
        ~29, i.e. ~225 ms of drained throughput per request, on top of a full eager decode step and a
        ~35 MB clone snapshot. Reused in place, one trace serves every request with the same tail."""
        buf = getattr(self, name, None)
        if buf is None or tuple(buf.shape) != (n,) or buf.dtype != dtype:
            buf = ttnn.zeros([n], dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)
            setattr(self, name, buf)
        return buf

    def start_decode(self, first_token_id: int):
        """Seed the device-resident decode state from the prefill's first token. After this, drive
        generation with decode_step_eager() (compiles) and/or decode_step_traced().

        Writes the three state buffers IN PLACE (see _decode_state_buf) so a decode trace captured for
        an earlier request stays valid — `decode_trace_ready()` is what decides whether it is reused."""
        self._h2d([first_token_id], ttnn.uint32, self._decode_state_buf("t_tok", ttnn.uint32))
        self._h2d([self.pos], ttnn.int32, self._decode_state_buf("t_curpos", ttnn.int32))
        self._h2d([self.pos], ttnn.uint32, self._decode_state_buf("t_ropepos", ttnn.uint32))
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
        if self.paged_kv:
            # ONE pool shared by every slot AND every attention layer, so the KV budget is set by
            # concurrent demand (QWEN36_KV_POOL_TOKENS) instead of batch_size * max_seq. This is the
            # case paging actually pays for -- see the note on blocks_per_seq in tt/paged_kv.py: within
            # a single sequence paging is memory-neutral, ACROSS sequences it is the whole point.
            if self.kv_pager is None:
                self.kv_pager = PagedKV(
                    self.mesh_device,
                    a.n_kv_heads,
                    a.head_dim,
                    self.max_seq,
                    batch=batch_size,
                    dtype=a.kv_cache_dtype,
                )
                logger.info(f"[qwen36] paged KV (B={batch_size}): {self.kv_pager}")
            elif self.kv_pager.batch < batch_size:
                raise RuntimeError(
                    f"KV pager was built for batch={self.kv_pager.batch} but alloc_batch_caches asked "
                    f"for {batch_size}; the page table is a trace-baked address and cannot be regrown"
                )
        z = lambda shape: ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        self.caches = []
        for layer in self.layers:
            if layer.is_linear:
                c = {
                    "recurrent_state": ttnn.zeros(
                        [batch_size, a.lin_num_v_heads, a.lin_head_k_dim, a.lin_head_v_dim],
                        dtype=_STATE_DT,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                    )
                }
                if _CONV_ADDCHAIN:
                    c["conv_rows"] = [z([batch_size, a.lin_conv_dim]) for _ in range(a.conv_kernel_size - 1)]
                self.caches.append(c)
            elif self.paged_kv:
                self.caches.append(self.kv_pager.alloc_layer_cache())  # a view onto the shared pool
            else:
                # Flat per-slot KV. NOTE the dtype: it follows ModelArgs.kv_cache_dtype like every other
                # cache allocation (_alloc_cache), NOT an unconditional bf16 -- otherwise QWEN36_KV_DTYPE
                # would silently apply to single-user serving and not to continuous batching.
                shape = [batch_size, a.n_kv_heads, self.max_seq, a.head_dim]
                mkz = lambda: ttnn.zeros(
                    shape, dtype=a.kv_cache_dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
                )
                self.caches.append([mkz(), mkz()])
        self.slot_pos = [0] * batch_size
        # Prefill scratch. The GDN layers genuinely need a [1,...] staging buffer (their state is
        # [B,Vh,Dk,Dv] and _prefill_single writes a single row). The ATTENTION layers do not: paged
        # prefill writes straight into the target slot's pages via its page table, and allocating a
        # scratch there would duplicate the whole pool per layer. So the scratch aliases the real
        # caches for attention and only stages GDN state.
        self._scratch_caches = [
            self._alloc_cache(layer) if (layer.is_linear or not self.paged_kv) else batch_cache
            for layer, batch_cache in zip(self.layers, self.caches)
        ]
        self._pf_caches_ready = True  # scratch is the prefill target; _prefill_single reuses + resets it

    def prefill_into_slot(self, input_ids, slot):
        """Prefill ONE request (torch [1,T]) into batch row `slot`, reusing the validated B=1 prefill:
        run it into the [1,...] scratch, then copy K/V + GDN state into row `slot` of the [B,...] caches.
        Returns last-token host logits [1,1,vocab] (the plugin host-samples the first token). Positions
        for this slot recorded in self.slot_pos[slot]. Long prompts go through the CHUNKED prefill
        (prefill_long), so a slot is not limited to what a single-shot prefill can materialize -- which
        is what makes the paged pool's "one request may use the whole pool" claim real.

        VERIFY-ON-DEVICE: the per-slot writes below. KV uses ttnn.fill_cache(dst[B,...], src[1,...], slot)
        (its native batch-index write). The GDN recurrent_state + conv_rows sliced writes are the ones to
        confirm/fix on-device (fill_cache may not accept the [B,Vh,Dk,Dv] / [B,conv_dim] shapes)."""
        batch = self.caches
        self.caches = self._scratch_caches  # _prefill_single writes here (+ resets scratch linear state)
        # Paged: the attention entries of `_scratch_caches` ARE the shared pool, and `slot` picks the
        # page-table row, so K/V lands in this request's own pages with no post-hoc copy. Flat: the
        # scratch is a [1,...] staging cache and the row copy below moves it.
        logits = self.prefill_long(input_ids, slot=slot)  # dispatches to _prefill_single when T <= chunk
        T = self.pos
        if _CONV_ADDCHAIN:  # sync scratch conv_state -> scratch conv_rows [1,conv_dim] (as start_decode does)
            for layer, sc in zip(self.layers, self._scratch_caches):
                if layer.is_linear:
                    layer.mixer.sync_conv_rows(sc)
        self.caches = batch
        for bl, sc in zip(self.caches, self._scratch_caches):
            if isinstance(bl, list):  # attention KV
                if self.paged_kv:
                    continue  # already written into slot `slot`'s pages by the prefill above
                ttnn.fill_cache(bl[0], sc[0], slot)  # native fill_cache batch-index write
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

    def release_slot(self, slot):
        """Return a finished request's KV blocks to the pool. No-op on the flat path (a slot's rows are
        simply overwritten by the next request). Call this when the serving layer retires a slot --
        without it the pool leaks and a long-running server eventually raises "block pool exhausted"
        even though only a few requests are live."""
        if self.paged_kv and self.kv_pager is not None:
            self.kv_pager.release(slot)

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

    def set_decode_state(self, token_ids, positions, active_slots=None):
        """Overwrite the batched decode state (token + absolute position, per slot) from the plugin's
        per-step inputs, IN THIS MODEL'S SLOT ORDER. Makes batched decode fully plugin-driven: no
        reliance on internal position advancing, so a slot reused by a new request mid-stream gets that
        request's fresh position (not a stale advanced one), and vLLM's compaction is irrelevant.
        token_ids / positions: length-B lists (dead slots pass anything). `active_slots` (paged KV
        only) names the slots that actually hold a live request; without it every slot -- including dead
        ones parked at position 0 -- would be handed a physical block that nothing ever releases."""
        mk = lambda t, dt: to_tt(t, self.mesh_device, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT)
        tok = torch.as_tensor(list(token_ids), dtype=torch.int32).flatten()
        pos = torch.as_tensor(list(positions), dtype=torch.int32).flatten()
        # Map any block THIS step will write, for every live slot. Host-side and outside any capture
        # (it may issue a copy_host_to_device_tensor), which is exactly why it belongs here rather than
        # in the traced graph -- same contract as the single-user _kv_advance.
        if self.paged_kv and self.kv_pager is not None:
            live = range(len(pos)) if active_slots is None else active_slots
            for s_i in live:
                self.kv_pager.ensure(int(pos[s_i]), int(s_i))
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
            x = layer.forward_decode(
                x,
                cos,
                sin,
                cache,
                self.t_curpos,
                # The page table is a PERSISTENT buffer at a stable address, so baking it into the
                # decode trace is safe; its CONTENTS are refreshed between replays by
                # _kv_advance() (see MEMORY.md §1 -- addresses are baked, contents are not).
                page_table=self.kv_pager.table if (self.paged_kv and self.kv_pager) else None,
            )
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
        # Follows ModelArgs.kv_cache_dtype like every other KV allocation: the head reuses the SAME
        # TtAttention (so Attention._to_cache_dtype already handles the write side), and at 262144 this
        # single-layer cache is 0.52 GB -- worth halving alongside the backbone's, not left behind.
        self.mtp_kv = [
            ttnn.zeros(shape, dtype=self.args.kv_cache_dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
            for _ in range(2)
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
        cached, exactly like gated_delta's `_verify_conv_buf`."""
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

    def _h2d(self, values, dtype, dst, src_dtype=torch.int32):
        """Write a small python/torch vector into an EXISTING device buffer.

        `copy_host_to_device_tensor` writes into the already-allocated `dst`; the obvious
        `ttnn.copy(to_tt(...), dst)` instead allocates a fresh device buffer per call and frees it
        again. A speculative round does ~9 of these, so at 40 layers that allocation churn is a real
        share of the round's host time — the same reason gemma4 keeps persistent traced inputs.

        Keeping `dst`'s ADDRESS stable is the other reason, and it is what lets a captured decode trace
        outlive the request that recorded it (see _decode_trace_sig). `src_dtype` is the torch dtype of
        the staging tensor: int32 for token/position/seed vectors, float32 for the bf16 sampling params."""
        t = torch.as_tensor(list(values), dtype=src_dtype).flatten()
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
            # host-side position: the SDPA program config is baked at CAPTURE time, so it cannot
            # read the device positions tensor. See Attention._verify_sdpa_pc.
            x = layer.forward_verify(x, cos, sin, cache, self._m_pos_i32, pos_hint=self.pos)
        x = self.final_norm.forward(x)
        ttnn.copy(x, self._m_vhidden)
        logits = self._lmh(ttnn.reshape(x, [K, self.args.dim]))  # [K, vocab]
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
            logits = self._lmh(ttnn.reshape(out, [1, dim]))  # [1, vocab]
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
            x = layer.forward_verify(x, cos, sin, cache, pos_i32, pos_hint=self.pos)
        x = self.final_norm.forward(x)
        logits = self._lmh(ttnn.reshape(x, [K, self.args.dim]))  # [K, vocab]
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
            tok = int(from_tt(self._lmh(ttnn.reshape(out, [1, self.args.dim])), self.mesh_device).reshape(-1).argmax())
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
            logits = self._lmh(x)  # [B, vocab]  (all rows kept; no last-token slice)
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
                logits = self._lmh(x)  # device [1, vocab]
            with sp.region("head.argmax"):
                logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
                tok = ttnn.argmax(logits, dim=-1, keepdim=False)  # [1] uint32
                ttnn.copy(ttnn.to_layout(tok, ttnn.ROW_MAJOR_LAYOUT), self.t_tok)  # next token, in place
            return
        V, CW, K, B, nc = self.args.vocab_size, self.SAMP_CHUNK, self.SAMP_K, self.SAMP_USERS, self._samp_nc
        with sp.region("head.lm_head"):
            logits = self._lmh(x)  # [1, vocab]  (1-row: lm_head stays cheap)
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
                logits = self._lmh(x)  # [B, vocab]
            with sp.region("head.argmax"):
                logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
                tok = ttnn.argmax(logits, dim=-1, keepdim=False)  # [B]
                ttnn.copy(ttnn.to_layout(tok, ttnn.ROW_MAJOR_LAYOUT), self.t_tok)
            return
        V, CW, K, U, nc = self.args.vocab_size, self.SAMP_CHUNK, self.SAMP_K, self.SAMP_USERS, self._samp_nc
        with sp.region("head.lm_head"):
            logits = self._lmh(x)  # [B, vocab]
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
        self._kv_advance()  # map any block this step will write into (no-op unless a boundary is crossed)
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

    def _decode_trace_sig(self):
        """Everything about the CURRENT config that changes the shape of the recorded decode graph.

        Two configs with the same signature produce the same op sequence over the same (now stable)
        buffer addresses, so one capture serves both. The tail (greedy argmax vs chunked topk +
        ttnn.sampling) and the presence-penalty ops are both decided at graph-BUILD time in
        _select_token, and the decode batch B unrolls into the graph, so all three belong here."""
        # The presence penalty is the one sampling parameter that is a baked python SCALAR rather than
        # a device tensor -- `_select_token` emits `sub(logits, multiply(t_presence, penalty))`, so the
        # value is frozen into the graph at capture. top_k / top_p / temperature / seed are all read
        # from tensors (t_k / t_p / t_temp / t_seed) and therefore DO pick up per-request writes, which
        # is what makes reuse worthwhile at all. Recording only "penalty > 0" here would let a request
        # with penalty 0.8 replay a trace baked with 1.5 -- harmless for demo/server.py, which uses one
        # server-wide default, but wrong for the vLLM adapter, which passes the request's own value.
        return (
            self.sampling is not None,
            float(getattr(self, "_presence_penalty", 0.0)) if self.sampling is not None else 0.0,
            int(self.t_tok.shape[0]) if getattr(self, "t_tok", None) is not None else 0,
            getattr(self, "_samp_nc", None),
        )

    # Reuse a live decode trace across requests when the tail matches. QWEN36_TRACE_REUSE=0 forces the
    # old per-request re-capture (A/B fallback).
    _TRACE_REUSE = os.environ.get("QWEN36_TRACE_REUSE", "1") != "0"

    def decode_trace_ready(self):
        """Can the live decode trace serve the current config? If so the caller can skip BOTH the
        re-capture and the eager warm-up step that precedes it, and go straight to decode_step_traced().

        This is only sound because the buffers the graph reads are now reused in place rather than
        reallocated per request (see _decode_state_buf / enable_sampling). Before that, every request
        moved t_tok/t_curpos/t_ropepos and the sampling params, so no trace could ever be reused."""
        if not self._TRACE_REUSE:
            return False
        return self.trace_id is not None and getattr(self, "_trace_sig", None) == self._decode_trace_sig()

    def capture_decode_trace(self, force=False):
        """Record the self-contained decode graph as a trace. PRECONDITION: at least one
        decode_step_eager() has run so all kernels are compiled (capture must not JIT). The decode
        state (caches + token + positions) is snapshotted and restored so recording the dummy step
        doesn't perturb generation. After this, drive every real step through decode_step_traced().

        NO-OP when the live trace already matches the current config (decode_trace_ready), which is the
        common case across requests: same model, same sampling tail. That skip is worth having — the
        capture costs a ~35 MB clone snapshot plus a drained pipeline that demo/server.py measures at
        43-46 ms/token for 16 steps against a settled ~29. Pass force=True to re-record regardless.

        Otherwise releases any PRIOR decode trace first: without the release the device's trace region
        accumulates one trace per capture (and pins the buffers each references) — that, with
        per-request cache reallocation, was the OOM. Persistent caches + this release keep memory flat."""
        if not force and self.decode_trace_ready():
            return self.trace_id
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.trace_id)
        snap = [ttnn.clone(t) for t in self._trace_snapshot_tensors()]
        # HIGH-WATER BALLAST. A capture's in-graph transients are freed the moment end_trace_capture
        # returns, but every REPLAY still writes their recorded addresses. Captured on a quiet
        # allocator (warmup prefills a 1-token prompt) those addresses sit LOW -- exactly where a real
        # prefill's much larger transients will later be allocated. That collision is the
        # long-conversation hang: it needs a captured trace (with the decode trace off, the same
        # conversation that hangs on turn 3 runs clean), it is not the program cache (801 cached
        # programs synthetically is fine; the model's cache is 31 MiB) and no memory metric moves.
        #
        # Holding a ballast across the capture pushes the trace's transients ABOVE the high-water mark
        # a prefill can reach, so prefill allocations can never land on them. Same fix, same reason, as
        # alloc_prefill_buffers' "allocate every bucket before any capture" (MEMORY.md §1-3).
        # MEASURED which region actually collides. A probe that samples both allocators after every
        # heavy op during the real prefills (hangdbg/peak_probe.py) gives prefill's peak footprint
        # above the quiescent post-warmup baseline:
        #
        #     DRAM  +233..252 MiB   (flat across turns)
        #     L1    +9.45 MiB       (11.2 MiB total, ~88 KiB of the 1.5 MB per core)
        #
        # So DRAM is NOT the colliding region: a 3 GB DRAM ballast is 12x prefill's whole DRAM
        # working set, and it still only bought one extra turn. L1 is where the aliasing happens --
        # which also explains why QWEN36_GDN_L1=0 (moving gated-delta off L1) shifted the failure by
        # exactly one turn. Ballast BOTH, sized from the measurement with margin, and the decode
        # trace's transients land above anything a prefill can reach.
        #
        # QWEN36_TRACE_BALLAST_L1_KB is per BANK (L1 banks == worker cores), because that is the
        # dimension the allocator hands out; QWEN36_TRACE_BALLAST_MB is a DRAM total.
        ballast = []
        mb = int(os.environ.get("QWEN36_TRACE_BALLAST_MB", "0"))
        if mb:
            ballast.append(
                ttnn.zeros(
                    [1, 1, 32, mb * 1024 * 1024 // 2 // 32],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                )
            )
        l1_kb = int(os.environ.get("QWEN36_TRACE_BALLAST_L1_KB", "0"))
        if l1_kb:
            from models.demos.qwen3_6_a3b.tt import memstat

            banks = memstat.region(self.mesh_device, ttnn.BufferType.L1).banks
            want = l1_kb * 1024 * banks
            # Several modest tensors rather than one huge one: an interleaved L1 tensor has to fit
            # every bank, and a single request for the whole amount is the one most likely to fail.
            per = max(1, want // 8)
            got = 0
            for _ in range(8):
                try:
                    ballast.append(
                        ttnn.zeros(
                            [1, 1, 32, max(32, per // 2 // 32)],
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=self.mesh_device,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                        )
                    )
                    got += per
                except Exception as e:  # out of L1 -- keep what we got, the capture still needs room
                    logger.warning(f"[qwen36] L1 capture ballast stopped at {got / 2**20:.1f} MiB ({e})")
                    break
            logger.info(
                f"[qwen36] capture ballast: DRAM {mb} MiB + L1 {got / 2**20:.1f} MiB "
                f"({l1_kb} KiB/bank x {banks} banks requested)"
            )
        self.trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._decode_graph()
        ttnn.end_trace_capture(self.mesh_device, self.trace_id, cq_id=0)
        for b in ballast:
            ttnn.deallocate(b)  # the hole it leaves is BELOW the trace's transients: safe to reuse
        for orig, s in zip(self._trace_snapshot_tensors(), snap):
            ttnn.copy(s, orig)  # undo the mutation done while recording
        self._trace_sig = self._decode_trace_sig()
        return self.trace_id

    def decode_step_traced(self, read_from_device: bool = True):
        """Replay the captured decode trace for one real step. Host work: kick off the trace and
        read back only the single next-token id (no per-token input rebuild, no full-logit D2H).
        read_from_device=False returns the on-DEVICE token tensor (cloned so the next replay can't
        overwrite it before the vLLM async path reads it)."""
        self._kv_advance()  # BEFORE the replay, and outside it: it may write the page table from host
        ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=False)
        self.pos += 1
        if not read_from_device:
            return ttnn.clone(self.t_tok)
        return int(from_tt(self.t_tok, self.mesh_device).flatten()[0])

    def _logits_trace_sig(self):
        """What changes the LOGITS decode graph's shape. Only the decode batch: this graph is
        embed -> layers -> norm -> lm_head with no sampling tail, so none of the sampling params
        (baked or tensor-borne) can affect it -- unlike _decode_trace_sig."""
        return (int(self.t_tok.shape[0]) if getattr(self, "t_tok", None) is not None else 0,)

    def logits_trace_ready(self):
        """Can the live logits trace serve the current config? If so the caller can skip both the
        re-capture and the eager warm-up call, and go straight to decode_step_logits_traced().

        Sound for the same reason as decode_trace_ready: the buffers this graph reads (t_tok,
        t_curpos, t_ropepos) are now written in place at stable addresses rather than reallocated per
        request, so a trace outlives the request that recorded it."""
        if not self._TRACE_REUSE:
            return False
        return getattr(self, "logits_trace_id", None) is not None and (
            getattr(self, "_logits_trace_sig_v", None) == self._logits_trace_sig()
        )

    def capture_decode_logits_trace(self, force=False):
        """Record the LOGITS-returning decode step (embed -> layers -> norm -> lm_head) as a trace,
        for the vLLM host-sampling path: replay computes logits into a persistent buffer and the host
        samples from them (per-request temperature/top-p/penalties/logprobs stay in vLLM's sampler).
        Unlike capture_decode_trace, this does NOT select a token on device — no _select_token — so the
        chosen token must be written back via set_decode_tokens() before each decode_step_logits_traced().

        Mirrors capture_decode_trace: releases any prior logits trace, snapshots/restores the decode
        state around the dummy recording step (the recorded step advances positions on device, so it
        would perturb generation otherwise). PRECONDITION: one eager decode_forward_logits() has run so
        all kernels are compiled (capture must not JIT).

        NO-OP when the live logits trace already matches (logits_trace_ready) — the same
        across-requests reuse capture_decode_trace does. force=True re-records regardless."""
        if not force and self.logits_trace_ready():
            return self.logits_trace_id
        if getattr(self, "logits_trace_id", None) is not None:
            ttnn.release_trace(self.mesh_device, self.logits_trace_id)
        snap = [ttnn.clone(t) for t in self._trace_snapshot_tensors()]
        self.logits_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        x = self._decode_hidden()
        with sp.region("dec.lm_head"):
            self._traced_logits = self._lmh(x)  # persistent output buffer (read after replay)
        with sp.region("dec.pos_advance"):
            ttnn.plus_one(self.t_curpos)  # advance position on device (for KV write + sdpa)
            ttnn.plus_one(self.t_ropepos)  # advance RoPE-table index on device
        ttnn.end_trace_capture(self.mesh_device, self.logits_trace_id, cq_id=0)
        for orig, s in zip(self._trace_snapshot_tensors(), snap):
            ttnn.copy(s, orig)  # undo the mutation done while recording
        self._logits_trace_sig_v = self._logits_trace_sig()
        return self.logits_trace_id

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
