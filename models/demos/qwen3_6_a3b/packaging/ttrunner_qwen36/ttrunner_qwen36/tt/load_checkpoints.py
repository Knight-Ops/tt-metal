# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Streaming HF checkpoint loader for Qwen3.6-35B-A3B.

The 72 GB checkpoint cannot fit in this box's RAM, so tensors are read lazily one at a time from
their safetensors shard (``safe_open``) and converted to tt tensors immediately by the caller, who
then drops the host copy. Builds per-module weight dicts whose keys match the tt module constructors.

The big-weight builders (attention/gated-delta/MoE projections, embed, lm_head) return zero-arg
THUNKS (``lambda: self.get(name)``) rather than tensors, so the read is deferred to ``to_tt`` /
``as_weight``, which skip it entirely on an on-disk weight-cache hit. The tiny always-read weights
(``norm_weights``, ``final_norm``) stay eager and return tensors directly.
"""
from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open

PREFIX = "model.language_model"


class CheckpointLoader:
    def __init__(self, ckpt_dir: str):
        self.dir = Path(ckpt_dir)
        with open(self.dir / "model.safetensors.index.json") as f:
            self.weight_map = json.load(f)["weight_map"]
        self._handles: dict[str, object] = {}

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(self.dir / shard, framework="pt", device="cpu")
        return self._handles[shard]

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str):
        """Load a single tensor (torch, cpu) from its shard."""
        shard = self.weight_map[name]
        return self._handle(shard).get_tensor(name)

    def _fn(self, name: str):
        """A zero-arg thunk that reads ``name`` when called (deferred so a cache hit can skip it)."""
        return lambda: self.get(name)

    # --- per-module weight-dict builders (keys match tt module constructors) ---
    # Big cacheable weights are returned as thunks; see module docstring.
    def layer_prefix(self, i: int) -> str:
        return f"{PREFIX}.layers.{i}"

    def attention_weights(self, i: int) -> dict:
        p = f"{self.layer_prefix(i)}.self_attn"
        return {
            "q_proj": self._fn(f"{p}.q_proj.weight"),
            "k_proj": self._fn(f"{p}.k_proj.weight"),
            "v_proj": self._fn(f"{p}.v_proj.weight"),
            "o_proj": self._fn(f"{p}.o_proj.weight"),
            "q_norm": self._fn(f"{p}.q_norm.weight"),
            "k_norm": self._fn(f"{p}.k_norm.weight"),
        }

    def gated_delta_weights(self, i: int) -> dict:
        p = f"{self.layer_prefix(i)}.linear_attn"
        return {
            "in_proj_qkv": self._fn(f"{p}.in_proj_qkv.weight"),
            "in_proj_z": self._fn(f"{p}.in_proj_z.weight"),
            "in_proj_b": self._fn(f"{p}.in_proj_b.weight"),
            "in_proj_a": self._fn(f"{p}.in_proj_a.weight"),
            "out_proj": self._fn(f"{p}.out_proj.weight"),
            "conv1d": self._fn(f"{p}.conv1d.weight"),
            "norm": self._fn(f"{p}.norm.weight"),
            "A_log": self._fn(f"{p}.A_log"),
            "dt_bias": self._fn(f"{p}.dt_bias"),
        }

    def moe_weights(self, i: int) -> dict:
        p = f"{self.layer_prefix(i)}.mlp"
        return {
            "gate": self._fn(f"{p}.gate.weight"),
            "gate_up_proj": self._fn(f"{p}.experts.gate_up_proj"),
            "down_proj": self._fn(f"{p}.experts.down_proj"),
            "se_gate_proj": self._fn(f"{p}.shared_expert.gate_proj.weight"),
            "se_up_proj": self._fn(f"{p}.shared_expert.up_proj.weight"),
            "se_down_proj": self._fn(f"{p}.shared_expert.down_proj.weight"),
            "se_router": self._fn(f"{p}.shared_expert_gate.weight"),
        }

    def norm_weights(self, i: int) -> dict:
        # Tiny (dim floats) and never cached -> read eagerly, return tensors.
        p = self.layer_prefix(i)
        return {
            "input_layernorm": self.get(f"{p}.input_layernorm.weight"),
            "post_attention_layernorm": self.get(f"{p}.post_attention_layernorm.weight"),
        }

    def embed_tokens(self):
        return self._fn(f"{PREFIX}.embed_tokens.weight")

    def final_norm(self):
        return self.get(f"{PREFIX}.norm.weight")  # tiny, eager

    def lm_head(self):
        return self._fn("lm_head.weight")
