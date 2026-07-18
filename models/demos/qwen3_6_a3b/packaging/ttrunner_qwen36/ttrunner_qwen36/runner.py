# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
tt-kernel *dispatch* runner for Qwen3.6-35B-A3B (single Blackhole P150).

Self-contained wheel variant. Identical to the reference runner except it drives the
renamespaced, FastAPI-free ``Qwen36Engine`` shipped inside this package (``.engine``)
rather than importing the model out of a tt-metal checkout. See
tt-kernel-package-manager/docs/authoring_runners.md for the contract.

Contract surface:
    __init__(self, model_path, device, **kwargs)
    attributes: _tokenizer, _listed, _community
    generate(prompt, max_new_tokens=50, temperature=1.0, chat=True) -> str
    generate_stream(prompt, max_new_tokens=50, temperature=1.0, chat=True)
        -> yields str deltas; FINAL yielded item is a dict
           {"finish_reason": str, "prompt_tokens": int, "completion_tokens": int}
    benchmark(prompt, n_tokens=50) -> (tokens_per_second: float, text: str)

We open our own 1x1 mesh (MANAGES_OWN_DEVICE = True; ``device`` arrives as None) and
tear it down cleanly on both atexit and SIGTERM — an ungraceful close leaks the device
mutex (/dev/shm/tt_device_*) and wedges the card until ``tt-smi -r``.

Serving-env prerequisites (co-versioned with the tt-metal build used to push the bundle):
``ttnn`` and ``ttl`` (tt-lang 1.1.3) present; ``transformers==4.53.0``; Blackhole
firmware >= 19.5.0. Only bf4 experts fit a 32 GB P150 (bf8 OOMs) — pinned via setdefault.
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
from typing import Iterator, Optional, Union

import torch
from ttrunner_qwen36.engine import Qwen36Engine

import ttnn


class Qwen36Runner:
    """Single-batch Qwen3.6-35B-A3B runner over a self-owned 1x1 Blackhole mesh."""

    MANAGES_OWN_DEVICE = True

    # Optional auto-discovery hooks (tt_models.runners entry point). The explicit
    # --runner-spec always works regardless; these only help hf_config-based selection.
    # NOTE: verify this string against the real config.json (it may live under
    # config.text_config.model_type) before relying on auto-discovery — claims() checks both.
    supported_model_types = {"qwen3_5_moe"}

    # Qwen3 "thinking mode" sampling aux knobs (the contract passes only temperature).
    _SAMPLING_AUX = {"top_k": 20, "top_p": 0.95, "presence_penalty": 1.5, "seed": None}

    @classmethod
    def claims(cls, hf_config) -> bool:
        mt = getattr(hf_config, "model_type", None)
        tc = getattr(hf_config, "text_config", None)
        if tc is not None:
            mt = getattr(tc, "model_type", None) or (tc.get("model_type") if isinstance(tc, dict) else None) or mt
        return mt in cls.supported_model_types

    def __init__(
        self,
        model_path: str,
        device,
        max_seq: int = 8192,
        trace_region_size: Optional[int] = None,
        use_trace: bool = True,
        enable_thinking: bool = True,
        **kwargs,
    ):
        # (a) Env defaults BEFORE ModelArgs/TtModel build (ModelArgs reads env at build time).
        os.environ.setdefault("QWEN36_EXPERT_DTYPE", "bf4")  # only bf4 (~17.5 GB) fits a 32 GB P150
        os.environ.setdefault("TT_CACHE_PATH", os.path.join(model_path, "tt_weight_cache"))

        # (b) Open + own the 1x1 mesh. Decode tracing (the OOM fix) needs a nonzero trace region.
        self.use_trace = use_trace
        trs = trace_region_size if trace_region_size is not None else (200 * 1024 * 1024 if use_trace else 0)
        if use_trace and trs == 0:
            trs = 200 * 1024 * 1024
        self.mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=trs)

        self._closed = False
        atexit.register(self._close)
        self._install_sigterm_handler()

        try:
            # (c) Build the full-model engine (n_layers=None => all 40 layers).
            self._engine = Qwen36Engine(self.mesh, ckpt=model_path, n_layers=None, max_seq=max_seq, use_trace=use_trace)
        except Exception:
            self._close()
            raise
        self._enable_thinking = enable_thinking

        # (d) Contract attributes.
        self._tokenizer = self._engine.tokenizer
        self._listed = True
        self._community = False

        # (e) Warm both decode tails (greedy + sampling).
        self._engine.warmup()

    # ── device lifecycle ────────────────────────────────────────────────────────
    def _close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            ttnn.close_mesh_device(self.mesh)
        except Exception:
            pass

    def _install_sigterm_handler(self):
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            prev = signal.getsignal(signal.SIGTERM)

            def _handler(signum, frame):
                self._close()
                if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
                    prev(signum, frame)
                else:
                    raise SystemExit(0)

            signal.signal(signal.SIGTERM, _handler)
        except (ValueError, OSError):
            pass

    # ── encoding / sampling helpers ──────────────────────────────────────────────
    def _encode(self, prompt: str, chat: bool) -> torch.Tensor:
        tok = self._tokenizer
        if chat and getattr(tok, "chat_template", None):
            out = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=bool(self._enable_thinking),
            )
            ids = (out["input_ids"] if not isinstance(out, torch.Tensor) else out).squeeze(0)
        else:
            ids = tok.encode(prompt, return_tensors="pt").squeeze(0)
        return ids

    def _sampling(self, temperature: float) -> dict:
        s = dict(self._SAMPLING_AUX)
        s["temperature"] = temperature
        return s

    # ── contract methods ─────────────────────────────────────────────────────────
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, chat: bool = True) -> str:
        ids = self._encode(prompt, chat)
        with self._engine.lock:
            text, _n, _secs, _tps, _fr = self._engine.generate(ids, max_new_tokens, **self._sampling(temperature))
        return text

    def generate_stream(
        self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, chat: bool = True
    ) -> Iterator[Union[str, dict]]:
        ids = self._encode(prompt, chat)
        prompt_tokens = int(ids.shape[0])
        with self._engine.lock:
            for delta in self._engine.stream_text(ids, max_new_tokens, **self._sampling(temperature)):
                yield delta
            yield {
                "finish_reason": self._engine._finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": self._engine._n_generated,
            }

    def benchmark(self, prompt: str, n_tokens: int = 50) -> tuple[float, str]:
        ids = self._encode(prompt, chat=False)
        with self._engine.lock:
            text, _n, _secs, tok_s, _fr = self._engine.generate(ids, n_tokens, **self._sampling(temperature=0.6))
        return (tok_s, text)
