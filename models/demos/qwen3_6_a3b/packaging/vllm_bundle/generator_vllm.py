"""
vLLM generator adapter for Qwen3.6-35B-A3B (hybrid GDN + full-attention MoE) on a single
Blackhole P150 — **MILESTONE V1: single-sequence (B=1)**.

Ships inside a tt-kernel `--backend vllm` bundle; `tt-kernel serve` launches vLLM with the
Tenstorrent plugin, which resolves this class from the HF `architectures` name
(`Qwen3_5MoeForConditionalGeneration`).

DESIGN (why this diverges from the Llama/Qwen delegation template)
------------------------------------------------------------------
The 35B-A3B is HYBRID: 30 Gated-DeltaNet layers carry a recurrent + conv state, 10 layers
are full attention. Per the plugin's constraints (it has no Mamba/recurrent KV-cache spec —
`model_runner._validate_kv_cache_groups` rejects any non-AttentionSpec), GDN state must be
**model-owned**: we inherit plain `Generator`, define **no** `get_kv_cache_spec`, and keep
both the attention KV and the GDN conv/recurrent state in the model's own device buffers.

Unlike the off-branch dense Qwen (which implements the tt_transformers standard decode
contract prepare_inputs_decode/ttnn_decode_forward/process_output_decode and lets
`super().decode_forward` drive it), THIS model (`models/demos/qwen3_6_a3b`) has its own
self-contained decode loop (`start_decode` → `decode_forward_logits` → `set_decode_tokens`).
So V1 **overrides** prefill_forward/decode_forward to drive that loop directly, avoiding a
model-internals refactor. Trace is OFF in V1 (eager decode) for simplicity; continuous
batching and traced decode are V2 (see packaging/VLLM_CONTINUOUS_BATCHING.md).

PREFIX REUSE (B=1): a turn is prefilled by CONTINUING from the previous turn's device caches
rather than recomputing the whole conversation -- see the `_PREFIX_REUSE` block below and
`TtModel.prefill_reuse`. This is the fix for multi-turn TTFT growing with the transcript; it is
model-internal because vLLM's own prefix caching cannot represent gated-delta state.

PAGED KV (2026-08-23): wired, and ON by default for `--max-num-seqs > 1`. One block pool
shared across slots and across the 10 attention layers, so the KV budget follows concurrent
demand (`QWEN36_KV_POOL_TOKENS`) rather than `max_num_seqs x max_model_len` — which is what
removes the per-slot context cap the contiguous arrangement needed. It stays MODEL-owned
(see allocate_kv_cache); `QWEN36_PAGED_KV=0` restores the contiguous path. Gated by
`tests/test_paged_cb_kv.py`.

⚠️  VERIFY-ON-DEVICE (this is a hardware-unverified draft — every item below is an assumption
    about the plugin contract that must be confirmed against a running P150 + TT vLLM plugin):
      1. allocate_kv_cache: RESOLVED — the plugin accepts an empty kv_cache (confirmed
         on-device 2026-07-18), and it stays empty now that the model pages its own KV, since
         the caches are model-bound by necessity (no AttentionSpec exists for the 30 GDN layers).
      2. decode_forward ignores `start_pos`/`page_table` (the model tracks positions
         internally). Valid ONLY for B=1 single-sequence continuing from its own prefill.
      3. prefill_forward returns logits `[B,1,vocab]`. If the runner unpacks `(logits,
         rope_deltas)` for this arch (mrope), switch to the tuple return (see below).
      4. warmup_model_* are no-ops in V1 (we don't use the base's traced-decode machinery);
         first request pays kernel-compile. A light warm can be added later.
      5. Registration: the plugin must resolve `Qwen3_5MoeForConditionalGeneration` →
         this class via EXTRA_MODELS_DIR. Confirm the installed plugin honors it.
"""

from __future__ import annotations

import os
import random
import time

import torch
from loguru import logger

# vLLM multimodal machinery. The HF arch name `Qwen3_5MoeForConditionalGeneration` is
# multimodal-shaped, so vLLM requires the model class to register a multimodal processor
# (a `_processor_factory`) even for a text-only port. We reuse vLLM's own Qwen3.5-MoE
# processing classes but advertise zero image/video (see TT_Qwen3_5MoeProcessingInfo).
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5MoeProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

# Model primitives (present in the serving env's tt-metal checkout).
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import ConversationSlots, TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs
from models.tt_transformers.tt.generator import Generator


class TT_Qwen3_5MoeProcessingInfo(Qwen3_5MoeProcessingInfo):
    """Text-only port: advertise zero image/video so vLLM's multimodal registry is satisfied
    (the *ForConditionalGeneration arch forces mm registration) without expecting media."""

    def get_supported_mm_limits(self):
        return {"image": 0, "video": 0}


def _row0(v, default=0.0):
    """First per-row value of a TTSamplingParams field (list / tensor / scalar) as a python scalar.
    The plugin passes per-request fields as length-B lists; at B=1 we read row 0."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return v
    try:
        x = v[0]
    except (TypeError, IndexError, KeyError):
        return default
    return x.item() if hasattr(x, "item") else x


def _rowval(v, j, default):
    """Per-row value j of a TTSamplingParams field (list/tensor/scalar) as a python scalar, or default
    if the field is absent or the row is None (unset)."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return v
    try:
        x = v[j]
    except (TypeError, IndexError, KeyError):
        return default
    if x is None:
        return default
    return x.item() if hasattr(x, "item") else x


def _resolve_weights(hf_config) -> str:
    """MODEL_WEIGHTS_DIR (tt-inference-server convention) → HF_MODEL → hf_config._name_or_path,
    resolving a bare hub id to a local snapshot dir."""
    name_or_path = (
        os.environ.get("MODEL_WEIGHTS_DIR") or os.environ.get("HF_MODEL") or getattr(hf_config, "_name_or_path", None)
    )
    if name_or_path and not os.path.isdir(os.path.expanduser(name_or_path)):
        from huggingface_hub import snapshot_download

        offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
        name_or_path = snapshot_download(name_or_path, local_files_only=offline)
    return os.path.expanduser(name_or_path)


# ── Multi-turn prefix reuse (B=1) ──────────────────────────────────────────────
# Without this, EVERY turn re-prefills the whole conversation: vLLM's own prefix caching is
# unavailable here (the plugin disables it unless `supports_prefix_caching`, and it could not help
# anyway -- it reuses blocks in vLLM's KV pool, while this model owns its KV *and* 30 layers of
# gated-delta recurrent state that has no block representation). So the reuse is done here, against
# the model's own caches, via TtModel.prefill_reuse.
#
# What makes it cheap: a turn's prompt is nearly always an extension of the previous turn's, because
# the chat template re-renders older history identically. Measured on Qwen3.6's template, the two
# token streams first differ 2 tokens before the end of the previous prompt (the generation prompt's
# "<think>\n") in plain chat, and NOT AT ALL in a tool-call loop -- there the template retains the
# reasoning of every assistant turn since the last real user query, so the cache is an exact prefix.
# Either way the per-turn ingest is the last answer's visible text plus the new message, not the
# whole context. Thinking mode needs no special handling: dropped <think> blocks move the divergence
# point, and the reuse point is derived by COMPARING TOKENS, never by trusting the template.
#
# Validated in tests/test_prefix_reuse.py (resumed vs full prefill PCC 0.9988, and 0.9994 after four
# chained turns, so it does not drift) and end to end on a 5-turn conversation: 83.5 s of prefill
# without it against 11.3 s with it.
#
# Measured at 40 layers: 47-87% of each prompt reused across a 10-turn conversation out to 2586
# tokens. Two supporting fixes are what make it safe, and neither is optional -- the prefill shape
# space is bucketed so the program cache saturates instead of growing ~90 entries a turn, and the
# cache is recycled past QWEN36_PROGRAM_CACHE_LIMIT. Without them the device hangs a few turns in.
# See IMPLEMENTATION_GUIDE.md and tests/probe_prefix_reuse_hang.py.
#
#   QWEN36_PREFIX_REUSE=0            disable (every turn re-prefills in full, the old behaviour)
#   QWEN36_PREFIX_REUSE_MIN          positions a checkpoint must save to be worth using (default 512)
#   QWEN36_PREFIX_REUSE_MAX_TURNS    force a full re-prefill after N consecutive reuses (0 = never).
#                                    A resumed turn builds on a RESTORED state rather than one
#                                    recomputed from the prompt, so numerical divergence from a full
#                                    prefill compounds across turns; this bounds it. Single-hop
#                                    incremental prefill is PCC >= 0.999 (PREFILL.md §3.5) but the
#                                    many-hop case is unmeasured -- see tests/test_prefix_reuse.py.
#   QWEN36_STATE_CKPT_DEPTH          checkpoint ring depth (tt/model.py, default 4 x 31.4 MiB). Also
#                                    how far back a client that rewrites OLD history can resume from.
#   QWEN36_CONV_SLOTS                parked conversations at B=1 (default 4; 1 = the old single-slot
#                                    behaviour). See the MULTI-CONVERSATION note below.
#
# MULTI-CONVERSATION (2026-09-08). One remembered history is not enough, and the failure is silent.
# Real clients interleave prompt streams: a chat UI runs the user's conversation AND short
# auto-title / auto-tag completions against the same engine. MEASURED in the shipped container --
# a 6.3K conversation growing +56 tokens a turn, interleaved with 209/263/318/409-token side
# requests, and EVERY prefill logged "reused 0 (0%) ... [full prefill]". Two single-slot pieces of
# state destroyed each other:
#
#   * this adapter kept ONE `_reuse_hist`, so each side request overwrote the conversation's
#     history and the next real turn's common prefix collapsed to ~0;
#   * `prefill_reuse` ended with an UNSCOPED `drop_state_checkpoints(above=T)`, so a 209-token
#     request discarded every checkpoint the 6.3K conversation held (`checkpoints=[]` in its logs).
#
# So the fix is two-sided, and the adapter half alone would be worse than useless -- restoring a
# gated-delta state against another conversation's KV rows produces plausible garbage rather than a
# slow answer. `TtModel.alloc_conversation_slots` gives each conversation its own KV blocks (paged;
# see tt/paged_kv.py activate()) and its own band of the checkpoint ring; `ConversationSlots`
# (tt/model.py, next to the rest of the reuse policy) picks which one a prompt belongs to. There is
# no conversation id to key on -- the plugin hands prefill_forward only tokens -- so identity comes
# from the longest common prefix, which is the same comparison prefix reuse already trusts, and
# slots age out least-recently-used.
_PREFIX_REUSE = os.environ.get("QWEN36_PREFIX_REUSE", "1") != "0"
_PREFIX_REUSE_MIN = int(os.environ.get("QWEN36_PREFIX_REUSE_MIN", "512"))
_PREFIX_REUSE_MAX_TURNS = int(os.environ.get("QWEN36_PREFIX_REUSE_MAX_TURNS", "0"))
_CONV_SLOTS = max(1, int(os.environ.get("QWEN36_CONV_SLOTS", "4")))


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=TT_Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen36ForCausalLM(Generator, SupportsMultiModal):
    """V1 single-sequence vLLM wrapper for Qwen3.6-35B-A3B on Blackhole P150."""

    # ACTIVE: ON-DEVICE sampling for speed (~30 tok/s, server.py parity) via sample_on_device_mode=
    # "decode_only" (set in vllm_metadata.json's launch --additional-config). decode_forward is
    # dual-path: requests the plugin CAN sample on device (temperature/top_k/top_p/presence/seed) go
    # the fast on-device trace; requests needing host-only features (guided/structured decoding,
    # logprobs, min_p, frequency_penalty, logit_bias, ...) are auto-routed by the plugin to host
    # sampling (sampling_params=None) and get the slower ~16 tok/s logits path — so those features are
    # NOT lost, they just run slower.
    #
    # ASYNC decode ON (retry): now viable because on-device sampling keeps the token on-device (the
    # earlier async attempt paired it with HOST sampling, so the sampled token had to round-trip
    # host→device on the autoregressive critical path — which broke it). With device sampling,
    # decode_forward returns the deferred on-device token and the inherited Generator.read_decode_output
    # async-cpu()s it, letting vLLM overlap per-step CPU work with the next device forward. (The
    # host-fallback logits path's async return remains the still-unverified one — see
    # VLLM_CONTINUOUS_BATCHING.md; flip this to False for the known-good sync path.)
    # Per-slot ON-DEVICE sampling (sample_on_device_mode=decode_only in the launch config): the batched
    # decode selects each slot's token in-graph (_select_token_batch) and returns tokens, so there's no
    # full-vocab D2H + CPU sampling per step — the throughput unlock for B>1. Presence penalty is
    # omitted on the batched path (v1); requests needing host-only features still route to host sampling.
    model_capabilities = {
        # Stays False, and that is not a TODO: vLLM's prefix caching reuses blocks in ITS KV pool,
        # which this model bypasses entirely (allocate_kv_cache returns []), and 30 of 40 layers carry
        # gated-delta recurrent state that has no block representation to reuse. Cross-turn reuse is
        # instead done against the model's own caches in _prefill_single_seq / TtModel.prefill_reuse,
        # which is invisible to vLLM: we are always handed the full prompt and simply decline to
        # recompute the part we already hold.
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
    }

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> int:
        # Per-user KV token capacity × decode slots. V1 is B=1 so max_num_seqs defaults to 1.
        if max_model_len is not None:
            return int(max_model_len) * int(max_num_seqs or 1)
        return super().get_max_tokens_all_users(
            model_name=model_name, num_devices=num_devices, tt_data_parallel=tt_data_parallel, **kwargs
        )

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        # V2 PROBE: B>1 is not correctly served yet (per-slot prefill unbuilt) — but we lift the
        # single-sequence assert so the plugin will actually exercise the B>1 path and we can log the
        # single-device continuous-batching contract (what prefill_forward/decode_forward receive).
        # Output at B>1 is garbage until per-slot prefill lands; this is purely to capture the contract.
        cls._max_batch_size = int(max_batch_size)
        max_seq_len = int(max_seq_len)
        if cls._max_batch_size > 1:
            # KV for B slots. Two arrangements, chosen by QWEN36_PAGED_KV (default ON for B>1):
            #
            #  PAGED (default). One block pool shared by all slots and all 10 attention layers, handed
            #  out 64 tokens at a time as each sequence grows (tt/paged_kv.py). The footprint is set by
            #  the AGGREGATE budget QWEN36_KV_POOL_TOKENS, not by B x max_model_len, so no per-slot cap
            #  is needed: one request may use the whole pool, or B requests may share it. The default
            #  budget below is deliberately the same number of tokens the contiguous arrangement used to
            #  pin (B x 8192), so enabling paging changes flexibility, not memory -- see the
            #  "blocks_per_seq" note in tt/paged_kv.py for why that distinction matters.
            #
            #  CONTIGUOUS (QWEN36_PAGED_KV=0). B slots x max_seq each, so the full 262K context x B
            #  slots does not fit a 32 GB P150 and the per-slot cap is mandatory.
            paged = os.environ.get("QWEN36_PAGED_KV", "1") != "0"
            os.environ["QWEN36_PAGED_KV"] = "1" if paged else "0"
            if paged:
                per_slot = int(os.environ.get("QWEN36_CB_TOKENS_PER_SLOT", "8192"))
                # FLOOR: the pool must hold at least ONE max-length request. The pool IS the
                # per-sequence cap (tt/paged_kv.py: blocks_per_seq == num_blocks), so a pool below
                # max_model_len makes the server advertise a context it cannot actually serve -- a long
                # prompt would fail with "block pool exhausted" rather than being merely slow. At the
                # shipped 262144 / bf8 that floor is 2.66 GiB, which is what makes "any one request may
                # use the whole pool" a real guarantee instead of a slogan.
                pool = int(
                    os.environ.get(
                        "QWEN36_KV_POOL_TOKENS",
                        str(max(cls._max_batch_size * per_slot, max_seq_len)),
                    )
                )
                if pool < max_seq_len:
                    logger.warning(
                        f"[qwen36-vllm][cb] QWEN36_KV_POOL_TOKENS={pool} is below max_model_len="
                        f"{max_seq_len}: requests longer than {pool} tokens will fail with "
                        f"'block pool exhausted'. Raise the pool or lower --max-model-len."
                    )
                os.environ["QWEN36_KV_POOL_TOKENS"] = str(pool)
                logger.info(
                    f"[qwen36-vllm][cb] B={cls._max_batch_size}: paged KV, pool={pool} tokens shared "
                    f"across slots (any single request may use up to the whole pool); "
                    f"~{pool * 10880 / 2**30:.2f} GiB at bf8, ~{pool * 20480 / 2**30:.2f} GiB at bf16"
                )
            else:
                cap = int(os.environ.get("QWEN36_MAX_SEQ", "8192"))
                max_seq_len = min(max_seq_len, cap)
                logger.warning(
                    f"[qwen36-vllm][cb] B={cls._max_batch_size}: QWEN36_PAGED_KV=0, capping per-slot "
                    f"context to {max_seq_len} tokens (contiguous KV)"
                )
        conv_slots = 1
        if cls._max_batch_size == 1 and _PREFIX_REUSE and _CONV_SLOTS > 1:
            # PARKED CONVERSATIONS need paged KV -- see the MULTI-CONVERSATION note at the top of
            # this file and TtModel.alloc_conversation_slots. This is the one place the single-
            # sequence path turns paging on, and it is memory-NEUTRAL: the pool defaults to
            # max_model_len, exactly the tokens the flat cache reserved anyway, and the parked
            # conversations SHARE it instead of each getting their own. Explicitly opting out
            # (QWEN36_PAGED_KV=0) keeps the flat cache and drops back to one conversation.
            if os.environ.get("QWEN36_PAGED_KV") == "0":
                logger.warning(
                    "[qwen36-vllm] QWEN36_PAGED_KV=0: parked conversations need paged KV, so prefix "
                    "reuse falls back to ONE conversation. Interleaved prompt streams (a chat UI's "
                    "auto-title requests, say) will each force a full prefill."
                )
            else:
                os.environ["QWEN36_PAGED_KV"] = "1"
                pool = int(os.environ.get("QWEN36_KV_POOL_TOKENS", str(max_seq_len)))
                os.environ["QWEN36_KV_POOL_TOKENS"] = str(pool)
                conv_slots = _CONV_SLOTS
                logger.info(
                    f"[qwen36-vllm] paged KV, pool={pool} tokens shared by {conv_slots} parked "
                    f"conversation slots (~{pool * 10880 / 2**30:.2f} GiB at bf8). Any one "
                    f"conversation may use the whole pool; slots age out least-recently-used."
                )
        ckpt = _resolve_weights(hf_config)
        os.environ.setdefault("QWEN36_EXPERT_DTYPE", "bf4")  # only bf4 (~17.5 GB) fits a 32 GB P150
        os.environ.setdefault("TT_CACHE_PATH", os.path.join(ckpt, "tt_weight_cache"))
        logger.info(f"[qwen36-vllm] building model from {ckpt} (max_seq_len={max_seq_len})")
        args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_batch_size=int(max_batch_size), max_seq_len=max_seq_len)
        loader = CheckpointLoader(ckpt)
        model = TtModel(mesh_device, args, loader, num_layers=n_layers)
        # BEFORE the caches and the checkpoint ring: the conversation-slot count is baked into both
        # (the KV pager's host mapping rows, and the ring's per-conversation bands), and neither can
        # be regrown afterwards -- the page table is an address a decode trace bakes, and the ring
        # must not be reallocated once a trace exists (IMPLEMENTATION_GUIDE.md §1).
        if conv_slots > 1:
            model.alloc_conversation_slots(conv_slots)
        # Persistent buffers FIRST, before any prefill or trace capture -- biggest first. The
        # gated-delta checkpoint ring is 180 MiB at 40 layers and fragments the trace region if it is
        # allocated later; see TtModel.alloc_state_checkpoints for the measurement.
        if _PREFIX_REUSE:
            model.alloc_state_checkpoints()
        # Then every per-WIDTH prefill constant, still before any capture. GatedDelta caches five
        # read-only selection matrices per block width and MoE a per-K one-hot, both built on FIRST
        # use of that width -- which, once a trace exists, means "allocated under an active trace".
        # A replay stomps them and, being write-once, they are never repaired, so the next request
        # that reuses that width wedges the device (measured 6/6 at 40 layers). See
        # TtModel.prewarm_prefill_shapes.
        if os.environ.get("QWEN36_PREWARM_SHAPES", "1") != "0":
            model.prewarm_prefill_shapes()
        # Allocate the on-device sampling tail's persistent buffers NOW, before any prefill or trace
        # capture. decode_forward otherwise allocates them on the first request that samples, which
        # may be after an earlier request captured a trace -- allocating device buffers while a trace
        # is live is the aliasing hazard in MEMORY.md §1-3 and can wedge the device on a LATER
        # request (see the same fix, and the bisection, in demo/server.py's warmup). enable_sampling(0)
        # restores greedy; the buffers persist and every later call just rewrites them in place.
        model.enable_sampling(1.0, top_k=20, top_p=0.95, seed=0, presence_penalty=0.0)
        model.enable_sampling(0.0)
        model._vllm_decode_started = False  # bespoke prefill→decode handoff flag (V1)
        return cls([model], [args], mesh_device)

    @property
    def cache_path(self):
        wcp = getattr(self.model_args[0], "weight_cache_path", None)
        if callable(wcp):
            wcp = wcp()
        return wcp or os.environ.get("TT_CACHE_PATH", "")

    # ── KV cache ────────────────────────────────────────────────────────────────
    def allocate_kv_cache(self, *args, **kwargs):
        """Attention KV and GDN state are MODEL-owned, so vLLM's paged manager stays bypassed and this
        returns an empty structure.

        This is true even now that the model pages its KV (QWEN36_PAGED_KV): the pool, the free list and
        the device page table all live in `tt/paged_kv.py`, and vLLM's block ids are used only as STABLE
        REQUEST KEYS (`page_table[i][0]`) to map a request to one of our slots. The two page tables are
        never required to agree -- ours indexes our pool. The reason it has to work this way is the
        hybrid architecture: 30 of 40 layers carry GDN recurrent+conv state that has no AttentionSpec,
        and `model_runner._validate_kv_cache_groups` rejects any non-AttentionSpec group, so we cannot
        hand vLLM a spec for this model at all (see the DESIGN note at the top of this file).

        CONFIRMED on-device 2026-07-18: the plugin accepts an empty kv_cache here."""
        return []

    # ── prefill (model-owned) ─────────────────────────────────────────────────────
    def prefill_forward(self, tokens, page_table=None, kv_cache=None, prompt_lens=None, **kwargs):
        """Eager prefill; returns last-token host logits [B,1,vocab].

        At B=1 this goes through _prefill_single_seq, which continues from the previous turn's cached
        prefix when the new prompt extends it (QWEN36_PREFIX_REUSE) and otherwise prefills fresh --
        a fresh prefill re-zeros the gated-delta conv left-pad (_reset_linear_state in
        _prefill_single), so an unrelated request never inherits state. page_table/kv_cache are
        accepted but unused (model-bound caches)."""
        model = self.model[0]
        # ── V2 continuous batching: prefill each newly-scheduled request into a free slot ──
        # Each request gets a STABLE model slot keyed by its first KV block id (page_table[i][0]); the
        # slot is freed when that key disappears from the decode batch. Keying by block id (not row
        # order) makes us robust to vLLM's slot_remap compaction.
        if getattr(type(self), "_max_batch_size", 1) > 1:
            if not getattr(model, "_cb_ready", False):
                # Slot pool = max_num_seqs + margin. vLLM never has more than max_num_seqs requests
                # ALIVE, but a request that finished at decode#N only disappears from the wire batch at
                # decode#N+1 — while its replacement is prefilled in between (a 1-step transient where
                # the model briefly holds max_num_seqs+k requests). The decode-time reconcile frees the
                # finished slot one step later, so a couple of spare slots absorb the transient and the
                # occupancy-based assignment uses them automatically — no slot collision. Capped at the
                # 32 ttnn.sampling lanes. The decode graph runs at pool width (wire batch B <= pool).
                margin = int(os.environ.get("QWEN36_CB_SLOT_MARGIN", "1"))
                pool = min(32, int(type(self)._max_batch_size) + margin)
                logger.info(
                    f"[qwen36-vllm][cb] slot pool = {pool} (max_num_seqs={type(self)._max_batch_size} + margin={pool - int(type(self)._max_batch_size)})"
                )
                model.alloc_batch_caches(pool)
                model._cb_ready = True
                model._cb_slot_of_key = {}  # key(block id) -> slot; a slot is free iff not a value here
                model._cb_last_active = set()
                model._cb_ever_decoded = set()
                model._cb_setup_done = False
                model._cb_seed_base = {}  # key -> per-request base RNG seed (request seed, or random if unset)
                model._cb_slot_key = {}  # slot -> key currently occupying it (to detect reuse -> presence reset)
            n = int(tokens.shape[0])
            pt = torch.as_tensor(page_table)
            outs = []
            assigned_this_call = set()  # keys given a slot in THIS prefill batch (protect from eviction)
            for i in range(n):
                Ti = int(prompt_lens[i]) if prompt_lens is not None else int(tokens.shape[1])
                key = int(pt[i, 0])
                slot = model._cb_slot_of_key.get(key)
                if slot is None:
                    # A slot is free iff no key currently maps to it (single source of truth).
                    used = set(model._cb_slot_of_key.values())
                    free = [s for s in range(model.batch_size) if s not in used]
                    if free:
                        slot = free[0]
                    else:
                        # No free slot. With decode-time reconciliation + slot margin this only happens
                        # if a burst of requests finished AND got replaced within one step. Evict a key
                        # that is NOT in the last decode's active set and was NOT just assigned in this
                        # same prefill batch (never steal a live or sibling-new request's slot).
                        last = getattr(model, "_cb_last_active", set())
                        victim = next(
                            (k for k in model._cb_slot_of_key if k not in last and k not in assigned_this_call),
                            None,
                        )
                        if victim is not None:
                            model._cb_ever_decoded.discard(victim)
                            slot = model._cb_slot_of_key.pop(victim)
                            model.release_slot(slot)  # the evicted request's blocks return to the pool
                            logger.warning(f"[qwen36-vllm][cb] over capacity: evicting key={victim} for key={key}")
                        else:
                            # Truly over pool (burst of finishes > margin). Never clobber a SIBLING just
                            # assigned in this batch (guaranteed corruption); the remaining choice among
                            # stale-live keys is unavoidable -> raise QWEN36_CB_SLOT_MARGIN to prevent this.
                            sib = {model._cb_slot_of_key[k] for k in assigned_this_call if k in model._cb_slot_of_key}
                            free_of_sib = [s for s in range(model.batch_size) if s not in sib]
                            slot = free_of_sib[0] if free_of_sib else 0
                            logger.error(
                                f"[qwen36-vllm][cb] pool of {model.batch_size} exhausted; forcing slot {slot} (raise QWEN36_CB_SLOT_MARGIN or lower --max-num-seqs)"
                            )
                    model._cb_slot_of_key[key] = slot
                assigned_this_call.add(key)
                logger.debug(f"[qwen36-vllm][cb] prefill req {i} key={key} ({Ti} tok) -> slot {slot}")
                li = model.prefill_into_slot(tokens[i : i + 1, :Ti], slot)
                outs.append(torch.as_tensor(li).reshape(1, 1, -1).float())
            return torch.cat(outs, dim=0), torch.zeros(n, dtype=torch.long)  # [n,1,vocab], rope_deltas

        model._vllm_decode_started = False  # new sequence → reseat the decode handoff
        T = int(prompt_lens[0]) if prompt_lens is not None else int(tokens.shape[1])
        ids = tokens[:, :T] if tokens.shape[1] > T else tokens
        logits = self._prefill_single_seq(model, ids, T)  # host logits [1, 1, vocab]
        logits = torch.as_tensor(logits).reshape(1, 1, -1).float()
        # The TT plugin's submit_prefill unpacks (logits, rope_deltas); rope_deltas are zeros
        # for this text-only port (no mrope position shift). CONFIRMED on-device 2026-07-18.
        rope_deltas = torch.zeros(logits.shape[0], dtype=torch.long)
        return logits, rope_deltas

    def _prefill_single_seq(self, model, ids, T):
        """B=1 prefill, continuing from the previous turn's cached prefix where possible.

        The history this compares against is the prompt we prefilled LAST -- not the tokens the model
        went on to generate. That is deliberate and sufficient: every checkpoint sits at or before the
        end of a prefill, and the next turn's divergence point (the newest assistant turn) is inside
        the previous prompt, so decode-time checkpointing would buy nothing and would have to deal
        with the conv add-chain (`conv_rows`) that a prefill-time snapshot can ignore."""
        if not _PREFIX_REUSE:
            logger.info(f"[qwen36-vllm] prefill user up to {T} tokens")
            return model.forward(ids)
        toks = ids[0].tolist()
        slots = self._conv_slots
        conv, lcp, evicted = slots.match(toks)
        if evicted is not None:
            # Aged out: hand the conversation's KV blocks back to the pool and forget its
            # checkpoints, so the incoming prompt starts from a clean slot rather than resuming
            # against a stranger's cache.
            logger.info(
                f"[qwen36-vllm] conversation slot {conv} aged out "
                f"({len(slots.hist[conv] or ())} tok) to make room for a {T}-token prompt"
            )
            slots.forget(conv)
            model.release_conversation(conv)
        # A slot being FREE is not the same as the pool having room for it. Parked conversations keep
        # their blocks until aged out, and the pool has no eviction of its own -- `ensure` raises
        # "block pool exhausted" partway through a prefill, which fails the request outright. So make
        # room first, least-recently-used, and only ever from conversations other than this one.
        need = T - model.conversation_tokens(conv)
        while need > 0 and model.kv_headroom_tokens() < need:
            victim = slots.lru(exclude=(conv,))
            if victim is None:
                break  # nothing left to give; let the pool raise with its own actionable message
            logger.info(
                f"[qwen36-vllm] KV pool headroom {model.kv_headroom_tokens()} < {need} needed: "
                f"aging out conversation slot {victim} ({len(slots.hist[victim] or ())} tok)"
            )
            slots.forget(victim)
            model.release_conversation(victim)
        # Escape hatch: drop the history every N reuses so the context is periodically recomputed
        # from the prompt rather than extended from a restored state.
        streak = slots.streak[conv]
        hist = None if (_PREFIX_REUSE_MAX_TURNS and streak >= _PREFIX_REUSE_MAX_TURNS) else slots.hist[conv]
        t0 = time.time()
        logits, reused = model.prefill_reuse(ids, hist_ids=hist, min_reuse=_PREFIX_REUSE_MIN, conv=conv)
        slots.commit(conv, toks, reused)
        dt = time.time() - t0
        new_tok = T - reused
        logger.info(
            f"[qwen36-vllm] prefill {T} tok in {dt:.2f}s: reused {reused} ({100.0 * reused / max(T, 1):.0f}%), "
            f"computed {new_tok}"
            + (f" = {new_tok / dt:.0f} tok/s" if dt > 0 else "")
            + (
                f" [conv {conv}/{slots.n} lcp {lcp}"
                + (f" streak {slots.streak[conv]}]" if reused else " full prefill]")
            )
            + (f" slots {slots.describe()}" if slots.n > 1 else "")
        )
        return logits

    @property
    def _conv_slots(self):
        """The parked-conversation table, built on first use.

        Sized to what the MODEL actually allocated, not to QWEN36_CONV_SLOTS: if the model came up
        with one slot (paging off, or an older build) the table must be one slot too, or the adapter
        would hand prefill_reuse a conversation index the model has no KV blocks for."""
        t = getattr(self, "_conv_slots_tbl", None)
        if t is None:
            n = max(1, int(getattr(self.model[0], "n_conv_slots", 1)))
            t = self._conv_slots_tbl = ConversationSlots(n, _PREFIX_REUSE_MIN)
            logger.info(f"[qwen36-vllm] prefix reuse across {n} parked conversation slot(s)")
        return t

    # ── decode (dual path: on-device sampling → tokens; host sampling → logits) ────────
    def decode_forward(
        self,
        tokens,
        start_pos=None,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        **kwargs,
    ):
        """One decode step, driving THIS model's self-contained decode loop.

        Chosen by whether the plugin sampled on device for this step (sample_on_device_mode=
        "decode_only"):
        - sampling_params is not None → ON-DEVICE sampling: map the per-request params to the model's
          enable_sampling and let the decode trace select the token in-graph (server.py's ~30 tok/s
          path); return the sampled token id(s) as a host torch [B] int tensor.
        - sampling_params is None → HOST sampling: the plugin routed this request to its own sampler
          (a feature we don't do on device — guided decoding / logprobs / min_p / frequency_penalty /
          logit_bias). Return logits [B,1,vocab]; vLLM samples and feeds the token back via `tokens`.

        The model tracks positions internally (start_pos/page_table ignored — valid at B=1). The decode
        trace is captured on the first step of each request and replayed after (sampling params bake
        into the trace → one capture per request). Extra plugin kwargs (rope_deltas_all_users,
        page_tables_per_layer, reset_batch, …) are absorbed by **kwargs."""
        model = self.model[0]
        # ── V2 continuous batching: one batched host-sampling step, driven by the plugin's inputs ──
        # Map each plugin decode row -> our stable slot by page-table key, drive the model in slot
        # order, free finished slots, and scatter the logits back to the plugin's row order.
        if getattr(type(self), "_max_batch_size", 1) > 1:
            B = int(tokens.shape[0])
            toks = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
            spos = (
                torch.as_tensor(start_pos).reshape(-1).to(torch.int64)
                if start_pos is not None
                else torch.zeros(B, dtype=torch.int64)
            )
            pt = torch.as_tensor(page_table)
            nslot = model.batch_size
            tok_by_slot, pos_by_slot, slot_of_row, active = [0] * nslot, [0] * nslot, [0] * B, set()
            # Per-slot sampling params (defaults fill dead slots). reset_slots = slots that just got a
            # NEW request this step (-> clear their presence mask).
            temp_slot, topk_slot, topp_slot, seed_slot = (
                [0.6] * nslot,
                [20] * nslot,
                [0.95] * nslot,
                [0] * nslot,
            )
            reset_slots = []
            sp_ = sampling_params
            for j in range(B):
                if int(spos[j]) < 0:
                    continue  # dead / padding row
                key = int(pt[j, 0])
                slot = model._cb_slot_of_key.get(key)
                if slot is None:
                    continue  # a request we never prefilled (shouldn't happen)
                active.add(key)
                slot_of_row[j] = slot
                tok_by_slot[slot] = int(toks[j])
                pos_by_slot[slot] = int(spos[j])
                # per-request sampling params (plugin row j -> our slot)
                temp_slot[slot] = float(_rowval(getattr(sp_, "temperature", None), j, 0.6))
                topk_slot[slot] = int(_rowval(getattr(sp_, "top_k", None), j, 20))
                topp_slot[slot] = float(_rowval(getattr(sp_, "top_p", None), j, 0.95))
                # per-request RNG base (its seed if given, else random once), folded with the position so
                # each request's stream both DIFFERS and advances per step
                base = model._cb_seed_base.get(key)
                if base is None:
                    rq = _rowval(getattr(sp_, "seed", None), j, None)
                    base = int(rq) if rq is not None else random.randrange(1, 1 << 31)
                    model._cb_seed_base[key] = base
                seed_slot[slot] = (base + int(spos[j])) & 0x7FFFFFFF
                if model._cb_slot_key.get(slot) != key:  # slot reused by a new request -> reset its mask
                    reset_slots.append(slot)
                    model._cb_slot_key[slot] = key
            # Reconcile slot ownership against the plugin's ground truth. The decode batch IS the full
            # set of running requests, so any key we still hold that (a) has decoded at least once and
            # (b) is no longer in `active` is a FINISHED request → free its slot NOW. Freeing here (not
            # deferred to prefill) keeps _cb_slot_of_key == exactly the live set, so prefill always finds
            # a genuine free slot and can never collide. A just-prefilled key that hasn't decoded yet is
            # not in _cb_ever_decoded, so it's protected from being freed before its first step.
            ever = getattr(model, "_cb_ever_decoded", set())
            for k in [k for k in list(model._cb_slot_of_key) if k in ever and k not in active]:
                freed = model._cb_slot_of_key.pop(k, None)
                ever.discard(k)
                model._cb_seed_base.pop(k, None)  # forget the finished request's RNG base
                # Hand the request's KV blocks back to the shared pool. Without this the pool leaks one
                # sequence's worth of blocks per completed request and a long-running server eventually
                # raises "block pool exhausted" with only a few requests live. No-op when not paged.
                if freed is not None:
                    model.release_slot(freed)
            model._cb_last_active = set(active)
            model._cb_ever_decoded = ever | active
            if not getattr(model, "_cb_setup_done", False):
                # Force the on-device SAMPLING tail (never argmax) so per-request temp/top_k/top_p apply
                # per slot. The params are TENSORS refreshed in place each step (set_sampling_params_batch),
                # so they take effect with NO re-capture; greedy requests (temp<=0) map to top_k=1 per lane.
                # This enable_sampling call only allocates the 32-lane buffers + presence mask.
                model.enable_sampling(1.0, top_k=0, top_p=1.0, seed=0)
                model.setup_batch_decode([0] * nslot)  # allocate stable decode-state + presence buffers once
                model._cb_setup_done = True
                # One-time EFFECTIVE-config line so a deploy can confirm what's actually active (env applied).
                logger.info(
                    f"[qwen36-vllm][cb] EFFECTIVE decode config: pool(width)={nslot} active_now={len(active)} "
                    f"presence={'ON' if model._presence_on_batch else 'OFF'} "
                    f"presence_fast={getattr(model, '_presence_fast', False)} "
                    f"presence_penalty={getattr(model, '_presence_penalty_batch', 0.0) if model._presence_on_batch else 0.0}"
                )
            model.reset_presence_slots(reset_slots)  # clear presence for slots that got a new request
            model.set_sampling_params_batch(topk_slot, topp_slot, temp_slot, seed_slot)  # per-request
            # active_slots: paged KV maps blocks only for live slots, so a dead slot parked at position
            # 0 never takes a block the pool can't get back (it has no request to retire and release it).
            model.set_decode_state(
                tok_by_slot,
                pos_by_slot,  # plugin-driven per-slot token + position
                active_slots=sorted({model._cb_slot_of_key[k] for k in active if k in model._cb_slot_of_key}),
            )
            # Trace-on-first-step: the batched device-select decode graph reads the stable-address
            # [B,...] caches + t_tok/t_curpos + sampling buffers, so it's captured once and replayed;
            # new prefills write the SAME buffers in place, so the trace stays valid. Returns [B] tokens
            # (no full-vocab D2H, no host sampling) — the throughput unlock.
            if not getattr(model, "_cb_traced", False):
                toks_dev = model.decode_step_eager_batch()  # eager compile (first step) -> [B] tokens
                model.capture_decode_trace_batch()
                model._cb_traced = True
            else:
                toks_dev = model.decode_step_traced_batch()  # traced replay -> [B] tokens
            toks_slot = torch.as_tensor(toks_dev).to(torch.int64)  # [nslot] in slot order
            return toks_slot[torch.tensor(slot_of_row, dtype=torch.long)]  # [B] device-sampled tokens, plugin row order

        tok = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
        device_sampling = sampling_params is not None

        # First decode step of a new request (prefill_forward reset _vllm_decode_started). For
        # on-device sampling, configure the sampling tail BEFORE start_decode/capture — the model
        # bakes temperature/top_k/top_p/seed/presence into the captured trace.
        if not getattr(model, "_vllm_decode_started", False):
            if device_sampling:
                sp = sampling_params
                _t = float(_row0(sp.temperature, 1.0))
                _tk = int(_row0(sp.top_k, 0))
                _tp = float(_row0(sp.top_p, 1.0))
                _pp = float(_row0(getattr(sp, "presence_penalty", 0.0), 0.0))
                # DIAGNOSTIC: the effective per-request sampling. temp==0 -> GREEDY, which Qwen3 warns
                # against in thinking mode (repetition/degradation). If OpenCode forces temp=0 this line
                # shows it, and no server default can override an explicit request value.
                logger.info(
                    f"[qwen36-vllm] sampling: temp={_t} top_k={_tk} top_p={_tp} presence={_pp} "
                    f"mode={'GREEDY(argmax)' if _t == 0 else 'sample'}"
                )
                model.enable_sampling(_t, top_k=_tk, top_p=_tp, seed=int(_row0(sp.seed, 0) or 0), presence_penalty=_pp)
            else:
                # sampling_params is None -> plugin routed this request to HOST sampling because it uses
                # features on-device sampling can't do (repetition/frequency penalty, min_p, logprobs,
                # guided decoding). Repetition penalty, if OpenCode sends it, lands here.
                logger.info("[qwen36-vllm] sampling: HOST path (rep/freq penalty / min_p / logprobs / guided)")
            model.start_decode(int(tok[0]))  # seed from the host-sampled first token (prefill)
            model._vllm_decode_started = True
            model._vllm_traced = False
        elif not device_sampling:
            model.set_decode_tokens(tok)  # host path: feed the host-sampled token back
        # on-device non-first steps: the model already wrote the sampled token into t_tok in-trace.

        if device_sampling:
            # on-device sampling: the trace runs lm_head + topk/argmax + ttnn.sampling and writes the
            # next token into t_tok; decode_step_* returns that id. Capture on step 1, replay after.
            # read_from_device=False (async path) returns the on-device token tensor, deferred.
            rfd = read_from_device
            if enable_trace:
                if not model._vllm_traced:
                    if model.decode_trace_ready():
                        # An earlier request already captured this exact tail, and the buffers the
                        # graph reads (token/positions, and the sampling params) are written in place
                        # at stable addresses rather than reallocated -- so replay it directly and skip
                        # BOTH the eager warm-up step and the capture. Per-request top_k / top_p /
                        # temperature / seed still take effect: they are device tensors the graph
                        # reads. presence_penalty is the exception (a baked scalar) and is part of the
                        # readiness signature, so changing it re-captures rather than silently
                        # reusing the previous request's value.
                        out = model.decode_step_traced(read_from_device=rfd)
                    else:
                        out = model.decode_step_eager(read_from_device=rfd)  # compile
                        model.capture_decode_trace()
                    model._vllm_traced = True
                else:
                    out = model.decode_step_traced(read_from_device=rfd)  # fast replay (~30 tok/s)
            else:
                out = model.decode_step_eager(read_from_device=rfd)
            if not rfd:
                # per-DP-rank list of (device token tensor, logprobs=None); inherited
                # read_decode_output async-cpu()s it, process_output_decode(is_tokens=True) -> [B] ids.
                return [(out, None)]
            return torch.tensor([int(out)], dtype=torch.int64)  # [B] sampled token ids (sync)

        # host-sampling path: return logits for vLLM's sampler. read_from_device=False is the PARKED
        # async path (returns on-device logits list); default True returns host logits [B,1,vocab].
        rfd = read_from_device
        if enable_trace:
            if not model._vllm_traced:
                if model.logits_trace_ready():
                    out = model.decode_step_logits_traced(read_from_device=rfd)  # reuse; see above
                else:
                    out = model.decode_forward_logits(read_from_device=rfd)
                    model.capture_decode_logits_trace()
                model._vllm_traced = True
            else:
                out = model.decode_step_logits_traced(read_from_device=rfd)
        else:
            out = model.decode_forward_logits(read_from_device=rfd)
        if not rfd:
            return [out]  # parked async path
        logits = torch.as_tensor(out)
        B = logits.shape[0]
        return logits.reshape(B, 1, -1).float()

    # ── warmup (no-op in V1) ──────────────────────────────────────────────────────
    def warmup_model_prefill(self, *args, **kwargs):
        # V1 runs eager prefill; the base's traced-prefill warmup would call standard-contract
        # methods this model doesn't implement. No-op; first request pays the compile.
        return

    def warmup_model_decode(self, *args, **kwargs):
        # V1 runs eager decode via our own loop, not the base's traced-decode machinery.
        return
