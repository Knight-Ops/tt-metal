# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Model + device configuration for Qwen3.6-35B-A3B on Blackhole.

Wraps the architecture config (parsed from the HF ``config.json`` via the standalone reference
``Qwen35MoeConfig``) and adds tt-nn / device-specific settings. Default target is a single
Blackhole P150 (32 GB) which forces BFP4 expert/MLP weights (the 35B model is ~17.5 GB in BFP4
vs ~35 GB in BFP8).
"""
from __future__ import annotations

import os

import ttnn
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig


class ModelArgs:
    """Holds architecture dims (from HF config) plus tt-nn device/precision config."""

    # Default context. Only the 10 full_attention layers hold a KV cache (the 30 linear_attention
    # layers carry a fixed-size recurrent state), so KV costs 10 * 2(K,V) * n_kv_heads(2) *
    # head_dim(256) * 2 B = 20 KiB/token -- a quarter of the 80 KiB/token a dense 40-layer model of
    # these dims would need. 32K context is therefore ~0.64 GB of KV against ~17.5 GB of BFP4
    # weights; even the checkpoint's native 262144 is only ~5.0 GB. See README ("Memory").
    DEFAULT_MAX_SEQ_LEN = 32768

    def __init__(
        self,
        mesh_device,
        ckpt_dir: str | None = None,
        max_batch_size: int = 1,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ):
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.ckpt_dir = ckpt_dir or os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
        self.config = Qwen35MoeConfig.from_hf_config(self.ckpt_dir)

        # On-disk cache of converted (quantized + tilized) weights. The first build writes one
        # .tensorbin per weight; later runs load them directly and skip the block-float quantization
        # (the bulk of cold-load time). Root defaults to $TT_CACHE_PATH or <ckpt>/tt_weight_cache.
        # QWEN36_WEIGHT_CACHE=0 disables it (always convert from the HF checkpoint).
        if os.environ.get("QWEN36_WEIGHT_CACHE") == "0":
            self.weight_cache_path = None
        else:
            root = os.environ.get("TT_CACHE_PATH") or os.path.join(self.ckpt_dir, "tt_weight_cache")
            self.weight_cache_path = root

        # --- convenience aliases (architecture) ---
        c = self.config
        self.dim = c.hidden_size
        self.n_layers = c.num_hidden_layers
        self.n_heads = c.num_attention_heads
        self.n_kv_heads = c.num_key_value_heads
        self.head_dim = c.head_dim
        self.vocab_size = c.vocab_size
        self.norm_eps = c.rms_norm_eps
        self.rope_theta = c.rope_theta
        self.rotary_dim = int(c.head_dim * c.partial_rotary_factor)  # 64
        self.layer_types = c.layer_types
        # Fidelity guards for the two architecture facts the tt-nn modules hardcode rather than
        # read: the per-head attention output gate (tt/attention.py splits q_proj into query and
        # gate halves and always applies sigmoid(gate)), and the absence of attention bias. Both
        # hold for this checkpoint; a variant that flipped either would produce silently wrong
        # output, so fail loudly at build time instead of generating garbage.
        assert c.attn_output_gate, "attn_output_gate=False is not implemented (tt/attention.py always gates)"
        assert not c.attention_bias, "attention_bias=True is not implemented (q/k/v/o are bias-free)"

        # linear-attention (Gated DeltaNet) dims
        self.lin_num_k_heads = c.linear_num_key_heads
        self.lin_num_v_heads = c.linear_num_value_heads
        self.lin_head_k_dim = c.linear_key_head_dim
        self.lin_head_v_dim = c.linear_value_head_dim
        self.lin_key_dim = c.linear_key_head_dim * c.linear_num_key_heads
        self.lin_value_dim = c.linear_value_head_dim * c.linear_num_value_heads
        self.lin_conv_dim = self.lin_key_dim * 2 + self.lin_value_dim
        self.conv_kernel_size = c.linear_conv_kernel_dim

        # MoE dims
        self.num_experts = c.num_experts
        self.num_experts_per_tok = c.num_experts_per_tok
        self.moe_intermediate_size = c.moe_intermediate_size
        self.shared_expert_intermediate_size = c.shared_expert_intermediate_size

        # MTP (multi-token prediction) head — one full-attention+MoE decoder layer shipped under the
        # checkpoint's `mtp.*` prefix, sharing embed_tokens and lm_head with the backbone. Only used
        # when speculative decode is enabled; the backbone ignores it. See MTP.md.
        self.mtp_num_hidden_layers = c.mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = c.mtp_use_dedicated_embeddings

        # --- device config ---
        self.num_devices = mesh_device.get_num_devices() if mesh_device is not None else 1
        self.is_blackhole = mesh_device is not None and ttnn.device.is_blackhole(mesh_device)

        # --- precision (single-card memory budget) ---
        # Routed experts dominate the parameter count -> BFP4. Attention/linear-attn projections
        # and norms stay higher precision for accuracy.
        # QWEN36_EXPERT_DTYPE={bf4,bf8} overrides the routed-expert + shared-MLP precision. Default
        # bf4 (the 35B model is ~17.5 GB in BFP4 vs ~35 GB in BFP8, and only BFP4 fits a 32 GB P150).
        # bf8 is used by the accuracy sweep in evaluation/ to measure the BFP4 accuracy cost (it needs
        # more DRAM, so pair it with fewer QWEN36_LAYERS if it OOMs on a single card).
        _expert_dtype = {"bf4": ttnn.bfloat4_b, "bf8": ttnn.bfloat8_b}.get(
            os.environ.get("QWEN36_EXPERT_DTYPE", "bf4").lower(), ttnn.bfloat4_b
        )
        self.expert_weight_dtype = _expert_dtype
        self.mlp_weight_dtype = _expert_dtype
        # Mixed-precision recipe (AesSedai/Ubergarm): pin the more-sensitive routed-expert down_proj
        # to bf8 while gate_up stays bf4. Costs ~+5GB (still fits a 32GB P150), unlike uniform bf8
        # experts (~+16GB -> OOM). QWEN36_EXPERT_DOWN_BF8=1 enables; default = same as expert dtype.
        self.expert_down_weight_dtype = (
            ttnn.bfloat8_b if os.environ.get("QWEN36_EXPERT_DOWN_BF8") == "1" else self.expert_weight_dtype
        )
        self.attn_weight_dtype = ttnn.bfloat8_b
        self.linear_attn_weight_dtype = ttnn.bfloat8_b
        self.activation_dtype = ttnn.bfloat16
        # Decode MoE: dense (no host sync, traceable, measured faster at batch=1) is the default.
        # gather-top-k sparse path (a host readback per call) is opt-in via QWEN36_SPARSE_DECODE=1.
        self.sparse_moe_decode = os.environ.get("QWEN36_SPARSE_DECODE", "0") == "1"

        self.compute_kernel_lofi = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        self.compute_kernel_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.compute_kernel_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def is_linear_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "linear_attention"

    def reference_config(self) -> Qwen35MoeConfig:
        return self.config

    @property
    def model_name(self) -> str:
        return "Qwen3.6-35B-A3B"
