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

    def __init__(self, mesh_device, ckpt_dir: str | None = None, max_batch_size: int = 1, max_seq_len: int = 4096):
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.ckpt_dir = ckpt_dir or os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
        self.config = Qwen35MoeConfig.from_hf_config(self.ckpt_dir)

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

        # --- device config ---
        self.num_devices = mesh_device.get_num_devices() if mesh_device is not None else 1
        self.is_blackhole = mesh_device is not None and ttnn.device.is_blackhole(mesh_device)

        # --- precision (single-card memory budget) ---
        # Routed experts dominate the parameter count -> BFP4. Attention/linear-attn projections
        # and norms stay higher precision for accuracy.
        self.expert_weight_dtype = ttnn.bfloat4_b
        self.mlp_weight_dtype = ttnn.bfloat4_b
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
