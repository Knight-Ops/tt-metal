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
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
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
        assert int(max_batch_size) == 1, (
            f"V1 adapter is single-sequence (B=1); got max_batch_size={max_batch_size}. "
            "Continuous batching is Milestone V2 (see packaging/VLLM_CONTINUOUS_BATCHING.md)."
        )
        ckpt = _resolve_weights(hf_config)
        os.environ.setdefault("QWEN36_EXPERT_DTYPE", "bf4")  # only bf4 (~17.5 GB) fits a 32 GB P150
        os.environ.setdefault("TT_CACHE_PATH", os.path.join(ckpt, "tt_weight_cache"))
        logger.info(f"[qwen36-vllm] building model from {ckpt} (max_seq_len={max_seq_len})")
        args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_batch_size=int(max_batch_size), max_seq_len=int(max_seq_len))
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
        tok = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
        device_sampling = sampling_params is not None

        # First decode step of a new request (prefill_forward reset _vllm_decode_started). For
        # on-device sampling, configure the sampling tail BEFORE start_decode/capture — the model
        # bakes temperature/top_k/top_p/seed/presence into the captured trace.
        if not getattr(model, "_vllm_decode_started", False):
            if device_sampling:
                sp = sampling_params
                model.enable_sampling(
                    float(_row0(sp.temperature, 1.0)),
                    top_k=int(_row0(sp.top_k, 0)),
                    top_p=float(_row0(sp.top_p, 1.0)),
                    seed=int(_row0(sp.seed, 0) or 0),
                    presence_penalty=float(_row0(getattr(sp, "presence_penalty", 0.0), 0.0)),
                )
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
