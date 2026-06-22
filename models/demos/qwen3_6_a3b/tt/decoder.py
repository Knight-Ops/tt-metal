# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-nn decoder layer for Qwen3.6-35B-A3B (hybrid: linear_attention or full_attention + MoE)."""
from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_a3b.tt import prefill_profiler as prof
from models.demos.qwen3_6_a3b.tt.attention import TtAttention
from models.demos.qwen3_6_a3b.tt.gated_delta import TtGatedDeltaNet
from models.demos.qwen3_6_a3b.tt.moe import TtMoE
from models.demos.qwen3_6_a3b.tt.rms_norm import TtRMSNorm


class TtDecoderLayer(LightweightModule):
    def __init__(self, mesh_device, args, loader, layer_idx):
        super().__init__()
        cfg = args.config
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.is_linear = args.is_linear_layer(layer_idx)

        nw = loader.norm_weights(layer_idx)
        self.input_norm = TtRMSNorm(mesh_device, nw["input_layernorm"], args.norm_eps, add_unit_offset=True)
        self.post_norm = TtRMSNorm(mesh_device, nw["post_attention_layernorm"], args.norm_eps, add_unit_offset=True)

        if self.is_linear:
            self.mixer = TtGatedDeltaNet(
                mesh_device, loader.gated_delta_weights(layer_idx), cfg, dtype=args.linear_attn_weight_dtype
            )
        else:
            self.mixer = TtAttention(
                mesh_device,
                loader.attention_weights(layer_idx),
                args.n_heads,
                args.n_kv_heads,
                args.head_dim,
                args.rotary_dim,
                args.norm_eps,
                dtype=args.attn_weight_dtype,
            )
        self.moe = TtMoE(
            mesh_device,
            loader.moe_weights(layer_idx),
            args.num_experts,
            args.num_experts_per_tok,
            expert_dtype=args.expert_weight_dtype,
            dtype=args.activation_dtype,
            sparse_decode=args.sparse_moe_decode,
            compute_kernel_config=args.compute_kernel_lofi,
        )

    def forward_prefill(self, x, cos, sin, cache):
        """cache: for linear layers a dict (gated-delta state); for attn layers [k_cache, v_cache]."""
        with prof.phase(self.mesh_device, "norm"):
            h = self.input_norm.forward(x)
        with prof.phase(self.mesh_device, "delta" if self.is_linear else "attn"):
            if self.is_linear:
                h = self.mixer.forward(h, cache=cache)
            else:
                h = self.mixer.forward_prefill(h, cos, sin, cache)
        with prof.phase(self.mesh_device, "norm"):
            x = ttnn.add(x, h)
            h = self.post_norm.forward(x)
        with prof.phase(self.mesh_device, "moe"):
            out = ttnn.add(x, self.moe.forward(h))
        return out

    def forward_prefill_traced(self, x, cos, sin, cache, pool):
        """Trace-safe prefill (no in-graph ttnn.zeros): linear mixer uses the pre-allocated pool; MoE
        is already trace-safe. Attention is unchanged for now (fill_cache trace-safety is a follow-up)."""
        h = self.input_norm.forward(x)
        if self.is_linear:
            h = self.mixer.forward(h, cache=cache, pool=pool)
        else:
            h = self.mixer.forward_prefill(h, cos, sin, cache)
        x = ttnn.add(x, h)
        h = self.post_norm.forward(x)
        return ttnn.add(x, self.moe.forward(h))

    def forward_prefill_incremental(self, x, cos, sin, cache, page_table, P):
        """Incremental prefill of new tokens continuing from the cache (offset P). Gated-delta starts
        its recurrent state from the cached state (and conv from the cached conv tail); attention
        attends the new tokens over the accumulated KV cache. MoE is per-token (unchanged)."""
        h = self.input_norm.forward(x)
        if self.is_linear:
            m = self.mixer
            init = ttnn.reshape(cache["recurrent_state"], [m.num_v_heads * m.head_k_dim, m.head_v_dim])
            h = m.forward(h, cache=cache, init_state=init)
        else:
            h = self.mixer.forward_prefill_incremental(h, cos, sin, cache, page_table, P)
        x = ttnn.add(x, h)
        h = self.post_norm.forward(x)
        return ttnn.add(x, self.moe.forward(h))

    def forward_decode(self, x, cos, sin, cache, current_pos):
        h = self.input_norm.forward(x)
        if self.is_linear:
            h = self.mixer.forward(h, cache=cache)  # single-step on-device scan from state cache
        else:
            h = self.mixer.forward_decode(h, cos, sin, cache, current_pos)
        x = ttnn.add(x, h)
        h = self.post_norm.forward(x)
        return ttnn.add(x, self.moe.forward(h))
