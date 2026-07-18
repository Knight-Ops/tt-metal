# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-nn decoder layer for Qwen3.6-35B-A3B (hybrid: linear_attention or full_attention + MoE)."""
from __future__ import annotations

from ttrunner_qwen36._vendor.lightweightmodule import LightweightModule
from ttrunner_qwen36.tt import prefill_profiler as prof
from ttrunner_qwen36.tt import signpost as sp
from ttrunner_qwen36.tt.attention import TtAttention
from ttrunner_qwen36.tt.gated_delta import TtGatedDeltaNet
from ttrunner_qwen36.tt.moe import TtMoE
from ttrunner_qwen36.tt.rms_norm import TtRMSNorm

import ttnn


class TtDecoderLayer(LightweightModule):
    def __init__(self, mesh_device, args, loader, layer_idx):
        super().__init__()
        cfg = args.config
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.is_linear = args.is_linear_layer(layer_idx)
        # Per-layer weight-cache prefix (None disables caching for this layer's big linear weights).
        cache_path = f"{args.weight_cache_path}/layer_{layer_idx}" if args.weight_cache_path else None

        nw = loader.norm_weights(layer_idx)
        self.input_norm = TtRMSNorm(mesh_device, nw["input_layernorm"], args.norm_eps, add_unit_offset=True)
        self.post_norm = TtRMSNorm(mesh_device, nw["post_attention_layernorm"], args.norm_eps, add_unit_offset=True)

        if self.is_linear:
            self.mixer = TtGatedDeltaNet(
                mesh_device,
                loader.gated_delta_weights(layer_idx),
                cfg,
                dtype=args.linear_attn_weight_dtype,
                cache_path=cache_path,
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
                cache_path=cache_path,
            )
        self.moe = TtMoE(
            mesh_device,
            loader.moe_weights(layer_idx),
            args.num_experts,
            args.num_experts_per_tok,
            expert_dtype=args.expert_weight_dtype,
            down_dtype=args.expert_down_weight_dtype,
            dtype=args.activation_dtype,
            sparse_decode=args.sparse_moe_decode,
            compute_kernel_config=args.compute_kernel_lofi,
            cache_path=cache_path,
            # Config dims so a weight-cache hit needn't read expert weights to learn their shapes.
            hidden=args.dim,
            inter=args.moe_intermediate_size,
            se_inter=args.shared_expert_intermediate_size,
        )

    def forward_prefill(self, x, cos, sin, cache):
        """cache: for linear layers a dict (gated-delta state); for attn layers [k_cache, v_cache]."""
        with prof.phase(self.mesh_device, "norm"), sp.region("layer.input_norm"):
            h = self.input_norm.forward(x)
        with prof.phase(self.mesh_device, "delta" if self.is_linear else "attn"), sp.region("layer.mixer"):
            if self.is_linear:
                h = self.mixer.forward(h, cache=cache)
            else:
                h = self.mixer.forward_prefill(h, cos, sin, cache)
        with prof.phase(self.mesh_device, "norm"), sp.region("layer.post_norm"):
            with sp.region("layer.residual1"):
                x = ttnn.add(x, h)
            h = self.post_norm.forward(x)
        with prof.phase(self.mesh_device, "moe"):
            with sp.region("layer.moe"):
                m = self.moe.forward(h)
            with sp.region("layer.residual2"):
                out = ttnn.add(x, m)
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

    def forward_prefill_incremental(self, x, cos, sin, cache, page_table, P, valid_len=None, q_chunk=None):
        """Incremental prefill of new tokens continuing from the cache (offset P). Gated-delta starts
        its recurrent state from the cached state (and conv from the cached conv tail); attention
        attends the new tokens over the accumulated KV cache. MoE is per-token (unchanged).

        valid_len (ragged final block): the real leading-token count. Gated-delta processes only those
        (state stays correct); attention runs the full padded block (causal + the caller reads only the
        real last token, and decode never reads the padded KV rows past self.pos). q_chunk: chunked-SDPA
        q_chunk_size override for the ragged block (so P stays a multiple of it)."""
        h = self.input_norm.forward(x)
        if self.is_linear:
            m = self.mixer
            init = ttnn.reshape(cache["recurrent_state"], [m.num_v_heads * m.head_k_dim, m.head_v_dim])
            h = m.forward(h, cache=cache, init_state=init, valid_len=valid_len)
        else:
            h = self.mixer.forward_prefill_incremental(h, cos, sin, cache, page_table, P, q_chunk=q_chunk)
        x = ttnn.add(x, h)
        h = self.post_norm.forward(x)
        return ttnn.add(x, self.moe.forward(h))

    def forward_decode(self, x, cos, sin, cache, current_pos):
        with sp.region("layer.input_norm"):
            h = self.input_norm.forward(x)
        with sp.region("layer.mixer"):
            if self.is_linear:
                # decode=True: B (=leading dim) independent users, one token each. B=1 keeps the fused
                # ttl kernel; B>1 runs the batched single-step recurrence (_forward_decode_batch).
                h = self.mixer.forward(h, cache=cache, decode=True)
            else:
                h = self.mixer.forward_decode(h, cos, sin, cache, current_pos)
        with sp.region("layer.residual1"):
            x = ttnn.add(x, h)
        with sp.region("layer.post_norm"):
            h = self.post_norm.forward(x)
        with sp.region("layer.moe"):
            m = self.moe.forward(h)
        with sp.region("layer.residual2"):
            return ttnn.add(x, m)
