# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
vLLM wrapper SCAFFOLD for Qwen3.6-35B-A3B (tt-inference-server integration).

STATUS: skeleton only. This file makes the tt-inference-server / vLLM contract
explicit so it can be filled in later, but the underlying ``TtModel``
(``models/demos/qwen3_6_a3b/tt/model.py``) does not yet satisfy that contract.
``initialize_vllm_model`` builds the model; the forward / kv-cache bridge methods
raise ``NotImplementedError`` with the specific gap each one needs closed.

Why a scaffold and not a working wrapper (the gaps):
  1. Batched / multi-user. ``TtModel`` is batch-1 (input ``[1, T]``); vLLM drives
     ``max_num_seqs > 1``. A batch dimension must be threaded through prefill+decode.
  2. Paged KV cache. Attention layers use a contiguous ``[1, n_kv, max_seq, hd]``
     cache updated at a scalar position (``paged_update_cache``). vLLM owns a paged
     cache and passes per-layer ``page_table`` / block tables that the model must consume.
  3. Linear-attention state has no KV-cache spec. ``config.layer_types`` contains
     ``"linear_attention"`` (Gated DeltaNet, recurrent state — not attention KV). The
     hybrid base ``get_kv_cache_spec`` only accepts ``sliding_attention`` /
     ``full_attention`` and will raise on ``linear_attention``. Gated-delta needs a
     recurrent ("Mamba-style") state spec that the TT vLLM plugin must support — this
     is likely the single biggest blocker; confirm plugin support first.
  4. Sampling. Decode bakes greedy argmax on-device and returns only the next token
     id; vLLM expects host-side sampling from logits (or a verified on-device-sample
     contract via ``supports_sample_on_device``). A logits-returning decode is needed.
  5. Generator-base mismatch. The stock wrappers delegate to
     ``prefill_forward_text`` / ``decode_forward`` on a ``tt_transformers.Transformer``;
     ``TtModel`` implements neither, so the bridge must be written against this model.

Registration (in the separate tt-inference-server repo) once the above is done:
  - tt-vllm-plugin ``register_models()``:
        ModelRegistry.register_model(
            "TTQwen3_5MoeForConditionalGeneration",   # arch from config.json + "TT" prefix
            "models.demos.qwen3_6_a3b.demo.generator_vllm:Qwen36ForCausalLM",
        )
  - workflows ``model_spec``: an ``ImplSpec`` (code_path models/demos/qwen3_6_a3b) plus a
    Blackhole/P150 ``ModelSpec`` (InferenceEngine.VLLM,
    override_tt_config={"optimizations": "performance"}).
"""

import ttnn  # noqa: F401  (kept for parity with sibling wrappers / future use)
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs
from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM

_GAP = (
    "Qwen36ForCausalLM is a scaffold: the bespoke TtModel does not yet support "
    "vLLM's batched + paged-KV + recurrent-state contract. See module docstring "
    "for the gap list before implementing this method."
)


class Qwen36ForCausalLM(HybridAttentionForCausalLM):
    """vLLM wrapper for Qwen3.6-35B-A3B (hybrid: gated-delta + full attention).

    Subclasses ``HybridAttentionForCausalLM`` because the model mixes
    ``linear_attention`` and ``full_attention`` layers. NOTE: the base
    ``get_kv_cache_spec`` rejects ``linear_attention`` layer types and must be
    overridden to emit a recurrent-state spec for those layers (gap #3).
    """

    # Conservative defaults; flip on only once each path is implemented + verified.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,  # decode argmax is on-device today
    }

    def __init__(self, model, model_args, mesh_device):
        # Mirrors the (tt_model, model_args, mesh_device) shape the stock wrappers use.
        # NOTE: the Generator base may expect ``model``/``model_args`` to be per-DP
        # lists; wrap accordingly when wiring data-parallel.
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
    ):
        """Build the bespoke ``TtModel`` and wrap it.

        Deliberately does NOT call ``initialize_vllm_text_transformer`` — that
        constructs a ``tt_transformers.Transformer``, a different model than this
        demo's hand-written ``TtModel``.
        """
        if max_batch_size and max_batch_size > 1:
            raise NotImplementedError(f"max_batch_size={max_batch_size}: {_GAP} (gap #1: batched prefill/decode)")
        ckpt_dir = getattr(hf_config, "_name_or_path", None)
        args = ModelArgs(mesh_device, ckpt_dir=ckpt_dir, max_seq_len=max_seq_len)
        loader = CheckpointLoader(args.ckpt_dir)
        model = TtModel(mesh_device, args, loader, num_layers=n_layers)
        return cls(model, args, mesh_device)

    @property
    def cache_path(self):
        ma = self.model_args[0] if isinstance(self.model_args, (list, tuple)) else self.model_args
        return ma.weight_cache_path

    def get_kv_cache_spec(self, vllm_config):  # noqa: D401
        # Base impl only handles sliding/full attention. Qwen3.6 also has
        # linear_attention (recurrent) layers needing a Mamba-style state spec.
        raise NotImplementedError(f"{_GAP} (gap #3: linear_attention needs a recurrent-state KV spec)")

    def prefill_forward(self, *args, **kwargs):
        raise NotImplementedError(f"{_GAP} (gaps #1/#2/#5: batched, paged-KV prefill bridge)")

    def decode_forward(self, *args, **kwargs):
        raise NotImplementedError(f"{_GAP} (gaps #1/#2/#4/#5: batched, paged-KV, logits-returning decode)")

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        raise NotImplementedError(f"{_GAP} (gap #2/#3: per-layer paged KV + recurrent state allocation)")
