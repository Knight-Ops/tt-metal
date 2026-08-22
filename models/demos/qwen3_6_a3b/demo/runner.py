# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
tt-kernel *dispatch* runner for Qwen3.6-35B-A3B (single Blackhole P150).

This is the class the dispatch serving layer constructs and drives. It implements the
runner contract from tt-kernel-package-manager/docs/authoring_runners.md by wrapping the
already-validated ``Qwen36Engine`` generation loop from ``server.py`` — same prefill /
traced-decode / on-device-sampling path the OpenAI HTTP server uses, minus the FastAPI/
tool-calling layer.

Contract surface:
    __init__(self, model_path, device, **kwargs)
    attributes: _tokenizer, _listed, _community
    generate(prompt, max_new_tokens=50, temperature=1.0, chat=True) -> str
    generate_stream(prompt, max_new_tokens=50, temperature=1.0, chat=True)
        -> yields str deltas; FINAL yielded item is a dict
           {"finish_reason": str, "prompt_tokens": int, "completion_tokens": int}
    benchmark(prompt, n_tokens=50) -> (tokens_per_second: float, text: str)

We open our own 1x1 mesh (dispatch doesn't open a Blackhole 1x1 topology for us), so
MANAGES_OWN_DEVICE = True and ``device`` arrives as None. Clean teardown is mandatory:
an ungraceful close leaks the device mutex (/dev/shm/tt_device_*) and wedges the card
until ``tt-smi -r``. We register BOTH an atexit hook and a SIGTERM handler because
atexit does not fire on SIGTERM (how a serving layer typically swaps models).

Serving-env prerequisites (co-versioned with the tt-metal build used to push a bundle):
``ttnn`` and ``ttl`` (tt-lang 1.1.3) present; ``transformers==4.53.0``; Blackhole
firmware >= 19.5.0 (older FW deadlocks the first SDPA-decode). Only bf4 experts fit a
32 GB P150 (bf8 OOMs) — pinned below via setdefault.
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
from typing import Iterator, Optional, Union

import torch

import ttnn

# Reference-runner mode: the model tree is importable in the serving env, so we reuse
# the production engine directly (the Phase-2 wheel instead ships a renamespaced copy).
from models.demos.qwen3_6_a3b.demo.server import Qwen36Engine


class Qwen36Runner:
    """Single-batch Qwen3.6-35B-A3B runner over a self-owned 1x1 Blackhole mesh."""

    # We open/close the device ourselves (dispatch passes device=None).
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
        max_seq: int = 32768,
        trace_region_size: Optional[int] = None,
        use_trace: bool = True,
        enable_thinking: bool = True,
        **kwargs,
    ):
        # (a) Env defaults BEFORE ModelArgs/TtModel build (ModelArgs reads env at build time).
        #     setdefault so an operator override still wins.
        os.environ.setdefault("QWEN36_EXPERT_DTYPE", "bf4")  # only bf4 (~17.5 GB) fits a 32 GB P150
        os.environ.setdefault("TT_CACHE_PATH", os.path.join(model_path, "tt_weight_cache"))
        # Leave the fast-path flags (QWEN36_GDN_FUSED / QWEN36_FUSED_PREFILL / QWEN36_SPARSE_DECODE)
        # unset — they already default to the validated fast path.

        # (b) Open + own the 1x1 mesh. Decode tracing (the OOM fix) needs a nonzero trace region;
        #     guard the case where the serving layer passes 0 alongside use_trace=True.
        self.use_trace = use_trace
        trs = trace_region_size if trace_region_size is not None else (200 * 1024 * 1024 if use_trace else 0)
        if use_trace and trs == 0:
            trs = 200 * 1024 * 1024
        self.mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=trs)

        # Idempotent clean teardown via atexit AND SIGTERM (atexit does not fire on SIGTERM).
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
        self._listed = True  # known / validated model
        self._community = False  # first-party / verified

        # (e) Warm both decode tails (greedy + sampling) so the first request/benchmark is representative.
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
        # signal.signal only works on the main thread; fall back to atexit-only otherwise.
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
            pass  # atexit still covers a normal exit

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
        # Pass the caller's temperature through unmodified (temperature==0 => greedy on device).
        s = dict(self._SAMPLING_AUX)
        s["temperature"] = temperature
        return s

    # ── contract methods ─────────────────────────────────────────────────────────
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, chat: bool = True) -> str:
        ids = self._encode(prompt, chat)
        with self._engine.lock:  # device decode state is single-user
            text, _n, _secs, _tps, _fr = self._engine.generate(ids, max_new_tokens, **self._sampling(temperature))
        return text

    def generate_stream(
        self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, chat: bool = True
    ) -> Iterator[Union[str, dict]]:
        ids = self._encode(prompt, chat)
        prompt_tokens = int(ids.shape[0])  # 1-D after squeeze(0)
        with self._engine.lock:
            for delta in self._engine.stream_text(ids, max_new_tokens, **self._sampling(temperature)):
                yield delta  # str deltas, one per step
            # FINAL item MUST be the usage dict; _finish_reason / _n_generated are final now.
            yield {
                "finish_reason": self._engine._finish_reason,  # "stop" | "length"
                "prompt_tokens": prompt_tokens,
                "completion_tokens": self._engine._n_generated,
            }

    def benchmark(self, prompt: str, n_tokens: int = 50) -> tuple[float, str]:
        ids = self._encode(prompt, chat=False)  # raw encode for a clean throughput measure
        with self._engine.lock:
            text, _n, _secs, tok_s, _fr = self._engine.generate(ids, n_tokens, **self._sampling(temperature=0.6))
        return (tok_s, text)
