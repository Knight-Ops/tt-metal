# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
vLLM wrapper SCAFFOLD for Qwen3.6-35B-A3B (tt-inference-server integration).

STATUS: contract-only. ``initialize_vllm_model`` builds the model; the forward / kv-cache bridge
methods raise ``NotImplementedError`` with the specific gap each one needs closed. It is NOT wired up
because completing AND verifying the bridge requires the tt-inference-server + vLLM runtime (a
tt-vllm plugin fork), which is not part of this repo's python_env — an unrunnable bridge would be
guesswork, so we ship an accurate map instead of unverified code.

ARCHITECTURE (corrected). This model is HYBRID: 30 Gated-DeltaNet (GDN, "linear_attention") layers
carrying a recurrent + conv state, and 10 full-attention layers with a KV cache. The natural
temptation is to inherit ``HybridAttentionForCausalLM`` and emit a recurrent KV-cache spec for the
GDN layers from ``get_kv_cache_spec``. That is a DEAD END:
  - ``HybridAttentionForCausalLM.get_kv_cache_spec`` (models/tt_transformers/tt/generator_vllm.py)
    only accepts ``sliding_attention`` / ``full_attention`` and RAISES on any other layer type; and
  - vLLM's KV-cache manager has no recurrent/Mamba spec that the TT plugin consumes for GDN.

So the correct design (the one Tenstorrent's own MoE build uses — see
``models/demos/blackhole/qwen36/tt/qwen36_vllm.py`` on branch ``kyamaguchi/qwen35moe-35b-a3b``) is:
  * Inherit plain ``Generator`` (NOT the hybrid class) — vLLM does not manage GDN state at all.
  * Bind BOTH the attention KV caches AND the GDN recurrent/conv state to the MODEL, in fixed-address
    device buffers, so a position-0 decode trace stays valid across steps (the recurrent scan is
    non-idempotent, so every new sequence must re-zero the GDN buffers before its first token).
  * Prefill is model-owned: pad the prompt up to a fixed bucket and mask the GDN to the EXACT
    ``valid_len`` (GDN forbids token-padding inside its scan); replay a chunk-outer trace for long
    prompts. Return ``(logits, rope_deltas)`` — rope_deltas are zeros for this text-only port.
  * Decode is the inherited traced decode (logits out, host samples, token fed back).

IMPLEMENTATION MAP (this model already has the needed primitives):
  gap #A allocate_kv_cache  -> model builds per-layer caches today in TtModel._alloc_cache /
        self.caches (KV [k,v] pairs for attn; {conv_state, recurrent_state} dicts for GDN). Expose a
        model.allocate_kv_caches(...) that (re)allocates these at fixed addresses and returns the attn
        KV pairs, mirroring TT's model.allocate_kv_caches. GDN state stays model-bound (not returned).
  gap #B prefill_forward    -> TtModel.forward() (eager, fp32-stable) already returns last-token
        logits and leaves caches + self.pos correct for decode; wrap it to return (logits,
        rope_deltas). For the fast/bucketed path, TtModel.forward_prefill_traced() exists (NOTE its
        numerical caveat: it uses the bf16 ttl chunk kernel — see gated_delta.py; prefer eager until
        that path is fp32). A per-new-sequence GDN reset already happens in _prefill_single via
        _reset_linear_state().
  gap #C decode_forward     -> TtModel already implements the logits-returning decode contract:
        start_decode(first_id) once, then decode_forward_logits() returns host logits [B, vocab] and
        advances positions on device, and set_decode_tokens(ids) writes the host-sampled token(s)
        back before the next step. Wire these to Generator's decode loop.
  gap #D paged vs contiguous KV. This model's attention uses a CONTIGUOUS [1,n_kv,max_seq,hd] cache
        updated at an absolute position, NOT vLLM's paged blocks. Either accept the model-owned
        contiguous cache at B=1 (simplest; TT effectively does single-sequence serving) or port the
        attention to paged_update_cache + page_table. B=1 single-sequence is the supported scope.

Batched (max_batch_size>1) is out of scope: TtModel is B=1 (input [1,T]); TT's vLLM path is also B=1
(max_concurrency=1). Thread a batch axis through prefill+decode to lift it.

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

import os

import ttnn  # noqa: F401  (kept for parity with sibling wrappers / future use)
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs
from models.tt_transformers.tt.generator import Generator

_GAP = (
    "Qwen36ForCausalLM is a scaffold: the bespoke TtModel's model-bound KV + GDN state bridge is not "
    "wired to vLLM yet, and completing it requires the tt-inference-server + vLLM runtime to verify. "
    "See the module docstring's IMPLEMENTATION MAP before implementing this method."
)


class Qwen36ForCausalLM(Generator):
    """vLLM wrapper for Qwen3.6-35B-A3B (hybrid: gated-delta + full attention).

    Inherits plain ``Generator`` (NOT ``HybridAttentionForCausalLM``): vLLM does not manage the
    Gated-DeltaNet recurrent state, so this model owns BOTH its attention KV caches and its GDN
    conv/recurrent state in fixed device buffers. See the module docstring for the corrected
    architecture and the per-method implementation map.
    """

    # Conservative defaults; flip on only once each path is implemented + verified.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,  # decode argmax/sampling is on-device today
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
        """All-user KV-cache token capacity. B=1 single-sequence serving, so the whole cache belongs
        to one user and its capacity is exactly the served context length (max_model_len)."""
        if max_model_len is not None:
            return int(max_model_len) * int(max_num_seqs or 1)
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            max_num_seqs=max_num_seqs,
            **kwargs,
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
        optimizations: str = "performance",
    ):
        """Build the bespoke ``TtModel`` and wrap it. Passes model/args as single-element lists to
        match ``Generator.__init__(model, model_args, mesh_device)`` (its data-parallel convention)."""
        if max_batch_size and max_batch_size > 1:
            raise NotImplementedError(
                f"max_batch_size={max_batch_size}: {_GAP} (batched prefill/decode is out of scope)"
            )
        ckpt_dir = (
            os.environ.get("MODEL_WEIGHTS_DIR")
            or os.environ.get("QWEN36_CKPT")
            or getattr(hf_config, "_name_or_path", None)
        )
        args = ModelArgs(mesh_device, ckpt_dir=ckpt_dir, max_seq_len=max_seq_len)
        loader = CheckpointLoader(args.ckpt_dir)
        model = TtModel(mesh_device, args, loader, num_layers=n_layers)
        return cls([model], [args], mesh_device)

    @property
    def cache_path(self):
        ma = self.model_args[0] if isinstance(self.model_args, (list, tuple)) else self.model_args
        return ma.weight_cache_path

    def allocate_kv_cache(self, *args, **kwargs):
        # gap #A/#D: allocate the model-bound attention KV pairs at fixed addresses (mirror TtModel
        # ._alloc_cache / self.caches) + re-zero the GDN conv/recurrent state; return the attn KV pairs.
        raise NotImplementedError(f"{_GAP} (gap #A/#D: model-bound KV allocation + GDN state binding)")

    def prefill_forward(self, *args, **kwargs):
        # gap #B: wrap TtModel.forward() (eager, fp32-stable -> (logits, rope_deltas=0)); the model
        # already leaves caches + self.pos correct for decode and resets GDN state per new sequence.
        raise NotImplementedError(f"{_GAP} (gap #B: model-owned masked-bucket prefill bridge)")

    def decode_forward(self, *args, **kwargs):
        # gap #C: drive TtModel.start_decode / decode_forward_logits / set_decode_tokens (the
        # logits-returning decode contract this model already implements).
        raise NotImplementedError(f"{_GAP} (gap #C: logits-returning decode bridge)")
