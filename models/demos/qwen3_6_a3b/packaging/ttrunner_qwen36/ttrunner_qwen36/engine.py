# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.6-35B-A3B generation engine (single Blackhole, batch-1).

Extracted verbatim from ``demo/server.py``'s ``Qwen36Engine`` and decoupled from the
FastAPI / pydantic / tool-calling layer: only the prefill + traced-decode + on-device
sampling loop remains. The dispatch runner (``runner.py``) drives this class. The
message/tool helpers (``ids_from_messages`` / ``_to_template_msg`` / ``_select_tools``)
are intentionally dropped — the runner does its own plain-string encoding.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Iterator, List, Optional

import torch
from loguru import logger
from ttrunner_qwen36.tt.load_checkpoints import CheckpointLoader
from ttrunner_qwen36.tt.model import TtModel
from ttrunner_qwen36.tt.model_config import ModelArgs


def _norm_stop(stop) -> List[str]:
    """Normalize the OpenAI ``stop`` field (str | list[str] | None) to a list of non-empty strings."""
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [s for s in stop if isinstance(s, str) and s]


def _earliest_stop(text: str, stops: List[str]) -> Optional[int]:
    """Index of the earliest occurrence of any stop string in ``text``, or None."""
    cut: Optional[int] = None
    for s in stops:
        i = text.find(s)
        if i != -1 and (cut is None or i < cut):
            cut = i
    return cut


class Qwen36Engine:
    """Single-batch Qwen3.6 wrapper. One request at a time (device decode state is
    single-user), so the HTTP layer serializes generation via ``self.lock``."""

    def __init__(self, mesh_device, ckpt: str, n_layers: int, max_seq: int, use_trace: bool):
        from transformers import AutoTokenizer

        self.mesh_device = mesh_device
        self.max_seq = max_seq
        self.use_trace = use_trace
        # Decode tracing (captured once, replayed per request) is the OOM fix and is on with use_trace.
        # Prefill tracing is a separate, EXPERIMENTAL opt-in (QWEN36_SERVER_PREFILL_TRACE=1): the
        # bucketed traced-prefill path is faster and allocation-free, but currently wedges the device
        # at larger buckets (e.g. 256) — under investigation. Default off: the server uses reliable
        # eager prefill (which reuses the persistent caches, so it does NOT leak/OOM either).
        self.use_prefill_trace = use_trace and os.environ.get("QWEN36_SERVER_PREFILL_TRACE", "0") == "1"

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.eos_ids = self._resolve_eos_ids(ckpt)

        args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_seq_len=max_seq)
        loader = CheckpointLoader(ckpt)
        logger.info(f"Building {n_layers}-layer model ({args.model_name})...")
        t0 = time.time()
        self.model = TtModel(mesh_device, args, loader, num_layers=n_layers)
        logger.info(f"Qwen3.6 model built in {time.time() - t0:.1f}s")
        self.lock = threading.Lock()

    def warmup(self):
        """Warm up both decode tails (their kernels differ): the first forward JIT-compiles every op
        and captures the decode trace, so the first real request reports representative tok/s. The
        default request samples, so without warming the sampling tail the first sampling request
        would pay the sampling-kernel compile (~seconds). Each generate() captures (and releases the
        prior) decode trace; persistent caches + that release keep device memory flat across
        requests (the OOM fix)."""
        warm_ids = self.tokenizer.encode("Hello", return_tensors="pt").squeeze(0)
        self.generate(warm_ids, 4)  # greedy tail
        self.generate(warm_ids, 4, temperature=1.0, top_k=20, top_p=0.95, presence_penalty=1.5)  # sampling tail

    def _resolve_eos_ids(self, ckpt: str) -> set:
        ids = set()
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            ids.add(int(eos))
        # generation_config.json may list extra stop ids (e.g. <|im_end|>)
        try:
            with open(os.path.join(ckpt, "generation_config.json")) as f:
                gc = json.load(f)
            g_eos = gc.get("eos_token_id")
            if isinstance(g_eos, int):
                ids.add(g_eos)
            elif isinstance(g_eos, list):
                ids.update(int(x) for x in g_eos)
        except Exception:
            pass
        return ids

    def _generate_ids(
        self,
        ids: torch.Tensor,
        max_new: int,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
    ) -> Iterator[int]:
        """Core generator: prefill then yield one token id per decode step.

        ``temperature == 0`` is greedy argmax (on device); ``> 0`` enables the on-device
        temperature/top-k/top-p (+ presence_penalty) sampling tail. ``min_p`` and
        ``repetition_penalty`` are accepted but not implemented (no-ops at their default
        0.0 / 1.0). Mirrors demo.py's loop. Stops on an EOS id, on ``max_new``, or when the
        KV / state cache would overflow ``max_seq``. Holds no lock itself — callers serialize.
        """
        if min_p > 0 or repetition_penalty != 1.0:
            logger.warning(f"min_p={min_p} / repetition_penalty={repetition_penalty} are not implemented; ignoring.")
        ids = ids.reshape(1, -1)
        prompt_len = int(ids.shape[1])
        if prompt_len >= self.max_seq:
            raise ValueError(
                f"prompt length {prompt_len} >= max_seq_len {self.max_seq}; raise QWEN36_MAX_SEQ or shorten the prompt"
            )
        # Finish reason for this generation, refined as we go: "length" unless we hit an EOS id
        # ("stop") or, in the text layer, a user stop string ("stop"). Read by callers after the
        # generator is exhausted (generation is serialized under self.lock).
        self._finish_reason = "length"
        # leave room: prefill consumes prompt_len positions, each decode step consumes one
        budget = min(max_new, self.max_seq - prompt_len - 1)
        if budget <= 0:
            return

        # Select greedy vs sampling for this request's decode steps (must precede start_decode
        # so the captured trace records the right tail). temperature==0 -> greedy.
        self.model.enable_sampling(temperature, top_k, top_p, seed or 0, presence_penalty)

        # Prefill: eager by default (reuses persistent caches -> no leak); bucketed traced replay
        # only when the experimental QWEN36_SERVER_PREFILL_TRACE flag is set.
        if self.use_prefill_trace:
            logits = self.model.forward_prefill_traced(ids)  # prefill -> [1, 1, vocab] host logits
        else:
            logits = self.model.forward(ids)
        next_id = int(logits[0, -1].argmax())  # first token is greedy argmax
        self.model.start_decode(next_id)

        # Capture the decode trace on this request's first step, then replay it for the rest. The
        # capture RELEASES the prior request's trace (see capture_decode_trace) so traces don't
        # accumulate; combined with persistent caches, device memory stays flat (no OOM).
        traced = False
        for step in range(budget):
            if next_id in self.eos_ids:
                self._finish_reason = "stop"
                return
            yield next_id
            if self.use_trace and not traced:
                next_id = self.model.decode_step_eager()  # warmup compiles kernels
                self.model.capture_decode_trace()
                traced = True
            elif self.use_trace:
                next_id = self.model.decode_step_traced()
            else:
                next_id = self.model.decode_step_eager()

    def _generate_text(self, ids: torch.Tensor, max_new: int, stop=None, **sampling) -> Iterator[str]:
        """Drive ``_generate_ids`` and yield the cumulative decoded text after each step.

        Applies user ``stop`` strings: if one appears, the text is truncated just before it,
        the generation ends, and ``self._finish_reason`` becomes "stop". While streaming, a tail
        of up to ``maxlen(stop) - 1`` chars is withheld so a stop string spanning a token
        boundary is never emitted; the final yield flushes it. Also tracks ``self._n_generated``
        (tokens produced). ``self._finish_reason`` is otherwise set by ``_generate_ids``
        ("stop" on EOS, "length" on the token budget)."""
        stops = _norm_stop(stop)
        hold = max((len(s) for s in stops), default=1) - 1  # withheld tail guards boundary-spanning stops
        out_ids: List[int] = []
        self._n_generated = 0
        text = ""
        t_start = time.time()  # progress logging: prove the device is decoding even if the client renders nothing
        for tid in self._generate_ids(ids, max_new, **sampling):
            out_ids.append(tid)
            self._n_generated = len(out_ids)
            if len(out_ids) % 128 == 0:
                dt = time.time() - t_start
                logger.info(f"[gen] {len(out_ids)} tok, {len(out_ids) / dt:.1f} tok/s (still generating)")
            text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            if stops:
                cut = _earliest_stop(text, stops)
                if cut is not None:
                    self._finish_reason = "stop"
                    yield text[:cut]
                    return
                yield text[: max(0, len(text) - hold)]  # hold back a possible partial stop string
            else:
                yield text
        yield text  # flush any withheld tail (no-op when nothing was held back)

    def generate(self, ids: torch.Tensor, max_new: int, stop=None, **sampling):
        """Drain ``_generate_text`` into a full completion.

        ``sampling`` = temperature/top_k/top_p/seed (forwarded to ``_generate_ids``).
        Returns (text, completion_tokens, gen_seconds, tokens_per_second, finish_reason).
        """
        t0 = time.time()
        text = ""
        for text in self._generate_text(ids, max_new, stop=stop, **sampling):
            pass
        secs = time.time() - t0
        n_tok = self._n_generated
        tok_s = n_tok / secs if secs > 0 else 0.0
        return text, n_tok, secs, tok_s, self._finish_reason

    def stream_text(self, ids: torch.Tensor, max_new: int, stop=None, **sampling) -> Iterator[str]:
        """Yield incremental decoded text deltas (handles multi-byte tokens and stop strings by
        re-decoding the running id list and emitting only the new suffix)."""
        prev = ""
        for full in self._generate_text(ids, max_new, stop=stop, **sampling):
            if len(full) > len(prev):
                yield full[len(prev) :]
                prev = full
