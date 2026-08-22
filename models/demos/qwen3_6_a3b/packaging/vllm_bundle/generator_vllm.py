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
batching, paged KV, and traced decode are V2 (see packaging/VLLM_CONTINUOUS_BATCHING.md).

⚠️  VERIFY-ON-DEVICE (this is a hardware-unverified draft — every item below is an assumption
    about the plugin contract that must be confirmed against a running P150 + TT vLLM plugin):
      1. allocate_kv_cache: V1 bypasses vLLM's paged manager (caches are model-bound). The
         plugin MAY require a non-empty kv_cache of a specific shape — if so, allocate a dummy
         of `kv_cache_shape` or bring paged KV forward from V2.
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
from models.demos.qwen3_6_a3b.tt.model import TtModel
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
            # Contiguous per-slot KV can't hold the full 262K context × B slots on a 32 GB P150 (OOM).
            # Cap per-slot context to QWEN36_MAX_SEQ (default 8192) for the batched build; real paged KV
            # (V2 follow-on) removes this. Requests longer than the cap will overflow their slot.
            cap = int(os.environ.get("QWEN36_MAX_SEQ", "8192"))
            max_seq_len = min(max_seq_len, cap)
            logger.warning(
                f"[qwen36-vllm][cb] B={cls._max_batch_size}: capping per-slot context to {max_seq_len} tokens (contiguous KV)"
            )
        ckpt = _resolve_weights(hf_config)
        os.environ.setdefault("QWEN36_EXPERT_DTYPE", "bf4")  # only bf4 (~17.5 GB) fits a 32 GB P150
        os.environ.setdefault("TT_CACHE_PATH", os.path.join(ckpt, "tt_weight_cache"))
        logger.info(f"[qwen36-vllm] building model from {ckpt} (max_seq_len={max_seq_len})")
        args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_batch_size=int(max_batch_size), max_seq_len=max_seq_len)
        loader = CheckpointLoader(ckpt)
        model = TtModel(mesh_device, args, loader, num_layers=n_layers)
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
        """V1: attention KV and GDN state are model-bound (allocated at model build); vLLM's
        paged manager is bypassed for single-sequence serving. Returns an empty structure.
        VERIFY: if the plugin requires a non-empty kv_cache of `kv_cache_shape`, allocate a
        dummy here or bring paged KV forward from V2 (gap #D)."""
        return []

    # ── prefill (model-owned) ─────────────────────────────────────────────────────
    def prefill_forward(self, tokens, page_table=None, kv_cache=None, prompt_lens=None, **kwargs):
        """Eager prefill via TtModel.forward(); returns last-token host logits [B,1,vocab].

        forward() re-zeros the GDN state for a fresh sequence (_reset_linear_state in
        _prefill_single), so each new request starts clean. page_table/kv_cache are accepted
        but unused (model-bound caches)."""
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
        logger.info(f"[qwen36-vllm] prefill user up to {T} tokens")
        logits = model.forward(ids)  # host logits [1, 1, vocab]
        logits = torch.as_tensor(logits).reshape(1, 1, -1).float()
        # The TT plugin's submit_prefill unpacks (logits, rope_deltas); rope_deltas are zeros
        # for this text-only port (no mrope position shift). CONFIRMED on-device 2026-07-18.
        rope_deltas = torch.zeros(logits.shape[0], dtype=torch.long)
        return logits, rope_deltas

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
                model._cb_slot_of_key.pop(k, None)
                ever.discard(k)
                model._cb_seed_base.pop(k, None)  # forget the finished request's RNG base
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
            model.set_decode_state(tok_by_slot, pos_by_slot)  # plugin-driven per-slot token + position
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
