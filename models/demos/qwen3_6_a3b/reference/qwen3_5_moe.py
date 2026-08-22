# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Standalone PyTorch reference for the TEXT path of Qwen3.6-35B-A3B
(HF architecture ``Qwen3_5MoeForConditionalGeneration`` / text config ``qwen3_5_moe_text``).

This is a faithful, dependency-light port of the HuggingFace ``transformers`` modeling code
(``qwen3_next`` + ``qwen3_5`` + ``qwen3_vl_moe``). It deliberately depends only on ``torch`` so
that PCC unit tests for the tt-nn implementation do not require ``transformers>=5.2`` to be
installed in the tt-metal python_env.

Scope: text-only (no vision tower). Implements the no-cache prefill path and the
single-step (seq_len==1) decode path with explicit conv/recurrent state, which is all the tt-nn
PCC tests need. Numerics mirror the HF torch fallback kernels exactly (fp32 internal compute in
the gated-delta rule, l2-norm on q/k, etc.).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Qwen35MoeConfig:
    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    # Per-head output gate on full attention (q_proj emits n_heads*head_dim*2; the second half gates
    # the attention output through sigmoid). The tt-nn attention implements this unconditionally, so
    # ModelArgs asserts it is set — see Qwen35MoeAttention / tt/attention.py.
    attn_output_gate: bool = True
    # rope
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25
    mrope_section: tuple = (11, 11, 10)
    mrope_interleaved: bool = True
    max_position_embeddings: int = 262144
    # linear attention (Gated DeltaNet)
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    # moe
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    # hybrid layout
    full_attention_interval: int = 4
    layer_types: list = field(default_factory=list)
    # multi-token prediction (MTP) head — one full-attention+MoE decoder layer shipped in the
    # checkpoint under the `mtp.*` prefix. Ignored by the backbone; consumed by Qwen35MoeMTP.
    mtp_num_hidden_layers: int = 0
    mtp_use_dedicated_embeddings: bool = False

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = [
                "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]

    @classmethod
    def from_hf_config(cls, config_path: str | Path) -> "Qwen35MoeConfig":
        with open(Path(config_path) / "config.json") as f:
            raw = json.load(f)
        tc = raw.get("text_config", raw)
        rope = tc.get("rope_parameters", tc.get("rope_scaling", {})) or {}
        kw = dict(
            vocab_size=tc["vocab_size"],
            hidden_size=tc["hidden_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
            hidden_act=tc.get("hidden_act", "silu"),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            attention_bias=tc.get("attention_bias", False),
            attn_output_gate=tc.get("attn_output_gate", True),
            rope_theta=rope.get("rope_theta", tc.get("rope_theta", 1e7)),
            partial_rotary_factor=rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)),
            mrope_section=tuple(rope.get("mrope_section", [11, 11, 10])),
            mrope_interleaved=rope.get("mrope_interleaved", True),
            max_position_embeddings=tc.get("max_position_embeddings", 262144),
            linear_conv_kernel_dim=tc["linear_conv_kernel_dim"],
            linear_key_head_dim=tc["linear_key_head_dim"],
            linear_value_head_dim=tc["linear_value_head_dim"],
            linear_num_key_heads=tc["linear_num_key_heads"],
            linear_num_value_heads=tc["linear_num_value_heads"],
            num_experts=tc["num_experts"],
            num_experts_per_tok=tc["num_experts_per_tok"],
            moe_intermediate_size=tc["moe_intermediate_size"],
            shared_expert_intermediate_size=tc["shared_expert_intermediate_size"],
            full_attention_interval=tc.get("full_attention_interval", 4),
            layer_types=tc.get("layer_types", []),
            mtp_num_hidden_layers=tc.get("mtp_num_hidden_layers", 0),
            mtp_use_dedicated_embeddings=tc.get("mtp_use_dedicated_embeddings", False),
        )
        return cls(**kw)


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------
class Qwen3NextRMSNorm(nn.Module):
    """Note: stored weight is "0-centered": output uses (1 + weight)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# RoPE (partial + text MRoPE == default)
# ---------------------------------------------------------------------------
class Qwen35TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen35MoeConfig, device=None):
        super().__init__()
        self.config = config
        dim = int(config.head_dim * config.partial_rotary_factor)  # 64
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (batch, seq). For text-only MRoPE, all 3 sections share these ids,
        # so the embedding reduces to the default RoPE on the rotary fraction of head_dim.
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, n_kv, slen, hd = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv, n_rep, slen, hd)
    return hidden_states.reshape(batch, n_kv * n_rep, slen, hd)


# ---------------------------------------------------------------------------
# Full attention (with fused output gate)
# ---------------------------------------------------------------------------
class Qwen35MoeAttention(nn.Module):
    def __init__(self, config: Qwen35MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[..., : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# Gated DeltaNet (linear attention)
# ---------------------------------------------------------------------------
def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        attn = attn.masked_fill(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen35MoeGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = nn.SiLU()
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        # Qwen3.5 uses SEPARATE projections (unlike Qwen3-Next's combined in_proj_qkvz/in_proj_ba).
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, cache=None):
        """cache: optional dict with 'conv_state' (B, conv_dim, K-1) and 'recurrent_state'.
        If provided and seq_len==1, runs single-step decode; else prefill (optionally with prior state)."""
        batch_size, seq_len, _ = hidden_states.shape
        use_cache = cache is not None and cache.get("recurrent_state") is not None

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # (B, conv_dim, S)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_cache and seq_len == 1:
            conv_state = cache["conv_state"]
            hidden_new = torch.cat([conv_state, mixed_qkv], dim=-1)
            cache["conv_state"] = hidden_new[:, :, -(self.conv_kernel_size - 1) :].clone()
            out = F.conv1d(hidden_new, self.conv1d.weight, None, padding=0, groups=self.conv_dim)
            mixed_qkv = F.silu(out[:, :, -seq_len:])
        else:
            if cache is not None:
                cache["conv_state"] = (
                    F.pad(mixed_qkv, (self.conv_kernel_size - 1 - mixed_qkv.shape[-1], 0))[
                        :, :, -(self.conv_kernel_size - 1) :
                    ]
                    if mixed_qkv.shape[-1] < self.conv_kernel_size - 1
                    else mixed_qkv[:, :, -(self.conv_kernel_size - 1) :].clone()
                )
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        init_state = cache["recurrent_state"] if use_cache else None
        if use_cache and seq_len == 1:
            core_attn_out, last_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=init_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=init_state,
                output_final_state=cache is not None,
                use_qk_l2norm_in_kernel=True,
            )
        if cache is not None:
            cache["recurrent_state"] = last_state

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        return self.out_proj(core_attn_out)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------
class Qwen35MoeMLP(nn.Module):
    def __init__(self, config: Qwen35MoeConfig, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen35MoeTopKRouter(nn.Module):
    def __init__(self, config: Qwen35MoeConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        return router_logits, router_top_value.to(router_logits.dtype), router_indices


class Qwen35MoeExperts(nn.Module):
    """Expert weights as stacked 3D params, matching HF packing."""

    def __init__(self, config: Qwen35MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            cur = self.act_fn(gate) * up
            cur = F.linear(cur, self.down_proj[expert_idx])
            cur = cur * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, cur.to(final.dtype))
        return final


class Qwen35MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen35MoeConfig):
        super().__init__()
        self.gate = Qwen35MoeTopKRouter(config)
        self.experts = Qwen35MoeExperts(config)
        self.shared_expert = Qwen35MoeMLP(config, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        bsz, seq, hidden = hidden_states.shape
        hs = hidden_states.view(-1, hidden)
        shared = self.shared_expert(hs)
        _, routing_weights, selected = self.gate(hs)
        expert_out = self.experts(hs, selected, routing_weights)
        shared = torch.sigmoid(self.shared_expert_gate(hs)) * shared
        out = expert_out + shared
        return out.reshape(bsz, seq, hidden)


# ---------------------------------------------------------------------------
# Decoder layer + model
# ---------------------------------------------------------------------------
class Qwen35MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen35MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen35MoeGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen35MoeAttention(config, layer_idx)
        self.mlp = Qwen35MoeSparseMoeBlock(config)
        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, cache=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, cache=cache)
        else:
            hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen35MoeTextModel(nn.Module):
    def __init__(self, config: Qwen35MoeConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen35MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen35TextRotaryEmbedding(config)

    def forward(self, input_ids, position_ids=None, caches=None):
        bsz, seq = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = torch.full((seq, seq), float("-inf"), device=input_ids.device).triu(1)[None, None]
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            hidden_states = layer(hidden_states, position_embeddings, causal_mask, cache)
        return self.norm(hidden_states)


class Qwen35MoeForCausalLM(nn.Module):
    def __init__(self, config: Qwen35MoeConfig):
        super().__init__()
        self.config = config
        self.model = Qwen35MoeTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids=None, caches=None):
        hidden_states = self.model(input_ids, position_ids, caches)
        return self.lm_head(hidden_states)


# ---------------------------------------------------------------------------
# Multi-token prediction (MTP) head
# ---------------------------------------------------------------------------
class Qwen35MoeMTP(nn.Module):
    """The `mtp.*` head shipped in the Qwen3.6-35B-A3B checkpoint (`mtp_num_hidden_layers: 1`).

    Structure (19 tensors, verified against model.safetensors.index.json): two pre-FC RMSNorms, an
    fc that halves a [2*hidden] concat back to [hidden], ONE ordinary decoder layer that is
    dimensionally identical to a main full-attention layer (q_proj [8192,2048] = Q+gate fused,
    q_norm/k_norm over head_dim=256, 256-expert MoE with a 512-wide shared expert), and a final
    norm. `mtp_use_dedicated_embeddings: false`, so embed_tokens and lm_head are SHARED with the
    backbone and are NOT part of this module.

    Contract: at position i the backbone produces hidden h_i and predicts token t_{i+1}; this head
    consumes (h_i, t_{i+1}) and predicts t_{i+2}.

    `concat_order` selects which half of `mtp.fc` sees which operand. The two prior in-tree MTP
    attempts disagreed, and getting it wrong yields garbage (the halves multiply distinct column
    blocks), so it is a constructor knob to be resolved empirically -- see
    tests/probe_mtp_acceptance.py. "token_first" matches the DeepSeek-V3 MTP convention.
    """

    def __init__(self, config: Qwen35MoeConfig, concat_order: str = "token_first"):
        super().__init__()
        assert concat_order in ("token_first", "hidden_first"), concat_order
        self.config = config
        self.concat_order = concat_order
        self.pre_fc_norm_embedding = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        # one full-attention decoder layer, whatever the backbone's layout says at that index
        lcfg = replace(config, layer_types=["full_attention"])
        self.layers = nn.ModuleList([Qwen35MoeDecoderLayer(lcfg, 0)])
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen35TextRotaryEmbedding(config)

    def combine(self, hidden, token_embeds):
        """fc(concat(norm_emb(embed(t_{i+1})), norm_hidden(h_i))) -> [*, hidden]."""
        e = self.pre_fc_norm_embedding(token_embeds)
        h = self.pre_fc_norm_hidden(hidden)
        parts = (e, h) if self.concat_order == "token_first" else (h, e)
        return self.fc(torch.cat(parts, dim=-1))

    def forward(self, hidden, token_embeds, position_ids=None, return_prenorm=False):
        """hidden/token_embeds: [B, S, hidden] (teacher-forced over a whole captured sequence, so the
        head's own attention builds its KV over the sequence exactly as it would incrementally).
        Returns the pre-lm_head hidden [B, S, hidden] -- apply the SHARED lm_head to get logits.

        return_prenorm also returns the layer output BEFORE `mtp.norm`, which is what a multi-step
        (gamma>1) draft chain feeds back as the next step's `hidden` -- the head predicts exactly ONE
        token ahead (mtp_num_hidden_layers=1), so deeper drafts run it on its own output."""
        x = self.combine(hidden, token_embeds)
        bsz, seq = x.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(seq, device=x.device).unsqueeze(0).expand(bsz, -1)
        pe = self.rotary_emb(x, position_ids)
        causal_mask = torch.full((seq, seq), float("-inf"), device=x.device).triu(1)[None, None]
        for layer in self.layers:
            x = layer(x, pe, causal_mask, None)
        out = self.norm(x)
        return (out, x) if return_prenorm else out
