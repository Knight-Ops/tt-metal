# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""tt-nn multi-token-prediction (MTP) head for Qwen3.6-35B-A3B — the draft model for speculative decode.

The checkpoint ships this head under the top-level ``mtp.`` prefix (``mtp_num_hidden_layers: 1``,
``mtp_use_dedicated_embeddings: false``). It is 19 tensors: two pre-fc RMSNorms, an ``fc`` that halves
a ``[2*hidden]`` concat back to ``[hidden]``, ONE decoder layer that is dimensionally identical to a
main full-attention layer (``q_proj [8192,2048]`` = Q+gate fused, ``q_norm``/``k_norm`` over
head_dim=256, a 256-expert MoE with a 512-wide shared expert), and a final norm. ``embed_tokens`` and
``lm_head`` are SHARED with the backbone and are therefore NOT owned here.

Contract: at position i the backbone produces hidden ``h_i`` and predicts token ``t_{i+1}``; this head
consumes ``(h_i, t_{i+1})`` and predicts ``t_{i+2}``.

Two contract details were resolved by measurement (MTP.md, probe A5) — do not "fix" them:
  * concat order is ``fc([norm_emb(embed(t_{i+1})), norm_hidden(h_i)])`` (token first). The other
    order scores alpha = EXACTLY 0.0000, so this is not a judgement call.
  * the backbone hidden fed in is the POST-final-norm state, i.e. ``TtModel._decode_hidden()``'s
    return value (alpha 0.9115 vs 0.8854 for the pre-norm state).
  * for a multi-step draft (gamma > 1) the head runs autoregressively on ITS OWN output, and the state
    fed back is the layer output BEFORE ``mtp.norm`` — hence ``forward_decode`` returns both.
"""
from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_a3b.tt import signpost as sp
from models.demos.qwen3_6_a3b.tt.attention import TtAttention
from models.demos.qwen3_6_a3b.tt.common import as_weight
from models.demos.qwen3_6_a3b.tt.moe import TtMoE
from models.demos.qwen3_6_a3b.tt.rms_norm import TtRMSNorm


class TtMtpHead(LightweightModule):
    def __init__(self, mesh_device, args, loader, embed_weight, lm_head_w):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        # shared with the backbone (mtp_use_dedicated_embeddings: false)
        self.embed_weight = embed_weight
        self.lm_head_w = lm_head_w
        cache_path = f"{args.weight_cache_path}/mtp" if args.weight_cache_path else None

        hw = loader.mtp_head_weights()
        self.pre_fc_norm_embedding = TtRMSNorm(
            mesh_device, hw["pre_fc_norm_embedding"], args.norm_eps, add_unit_offset=True
        )
        self.pre_fc_norm_hidden = TtRMSNorm(mesh_device, hw["pre_fc_norm_hidden"], args.norm_eps, add_unit_offset=True)
        # fc: [hidden, 2*hidden] on disk -> as_weight transposes to [2*hidden, hidden] for ttnn.linear
        self.fc = as_weight(
            hw["fc"],
            mesh_device,
            dtype=args.attn_weight_dtype,
            cache_file_name=(f"{cache_path}/fc" if cache_path else None),
        )
        self.norm = TtRMSNorm(mesh_device, hw["norm"], args.norm_eps, add_unit_offset=True)

        nw = loader.mtp_norm_weights()
        self.input_norm = TtRMSNorm(mesh_device, nw["input_layernorm"], args.norm_eps, add_unit_offset=True)
        self.post_norm = TtRMSNorm(mesh_device, nw["post_attention_layernorm"], args.norm_eps, add_unit_offset=True)

        self.attn = TtAttention(
            mesh_device,
            loader.mtp_attention_weights(),
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
            loader.mtp_moe_weights(),
            args.num_experts,
            args.num_experts_per_tok,
            expert_dtype=args.expert_weight_dtype,
            down_dtype=args.expert_down_weight_dtype,
            dtype=args.moe_shared_weight_dtype,  # see decoder.py: this sets router/shared precision
            sparse_decode=args.sparse_moe_decode,
            compute_kernel_config=args.compute_kernel_lofi,
            cache_path=cache_path,
            hidden=args.dim,
            inter=args.moe_intermediate_size,
            se_inter=args.shared_expert_intermediate_size,
        )

    # ------------------------------------------------------------------ pieces
    def combine(self, hidden, token_embeds):
        """fc(concat([norm_emb(embed(t_{i+1})), norm_hidden(h_i)])) -> [1, 1, B, hidden].

        hidden / token_embeds: [1, 1, B, hidden]. Token half FIRST — see the module docstring."""
        e = self.pre_fc_norm_embedding.forward(token_embeds)
        h = self.pre_fc_norm_hidden.forward(hidden)
        return ttnn.linear(ttnn.concat([e, h], dim=-1), self.fc)

    def embed(self, token_ids):
        """token_ids: device uint32 [B] -> [1, 1, B, hidden] (the SHARED backbone embedding)."""
        e = ttnn.embedding(token_ids, self.embed_weight, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(e, [1, 1, token_ids.shape[0], self.args.dim])

    # ------------------------------------------------------------------ forward
    def forward_decode(self, hidden, token_ids, cos, sin, kv_cache, current_pos):
        """One draft step. hidden: [1,1,B,dim] backbone (or previous draft) state; token_ids: [B].

        Returns (out, pre) — ``out`` is post-``mtp.norm`` and feeds the SHARED lm_head; ``pre`` is the
        layer output before ``mtp.norm`` and is what a gamma>1 chain feeds back as the next `hidden`."""
        with sp.region("mtp.combine"):
            x = self.combine(hidden, self.embed(token_ids))
        with sp.region("mtp.mixer"):
            h = self.attn.forward_decode(self.input_norm.forward(x), cos, sin, kv_cache, current_pos)
        x = ttnn.add(x, h)
        with sp.region("mtp.moe"):
            x = ttnn.add(x, self.moe.forward(self.post_norm.forward(x)))
        with sp.region("mtp.norm"):
            return self.norm.forward(x), x

    def forward_prefill(self, hidden, token_embeds, cos, sin, kv_cache):
        """Teacher-forced pass over a whole sequence — used to PRIME the head's own KV cache over the
        prompt so the first draft step has real context. hidden/token_embeds: [1,1,T,dim]."""
        x = self.combine(hidden, token_embeds)
        h = self.attn.forward_prefill(self.input_norm.forward(x), cos, sin, kv_cache)
        x = ttnn.add(x, h)
        x = ttnn.add(x, self.moe.forward(self.post_norm.forward(x)))
        return self.norm.forward(x), x
