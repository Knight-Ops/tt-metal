# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shape gates for the server's SSE streams — no device, no model, no HTTP.

These exist because the stream's *timing* is part of its contract and got it wrong twice over:

  * the assistant-role chunk was emitted BEFORE generation started, so any client timing "first
    chunk" measured the HTTP round-trip rather than time-to-first-token. llama-benchy read 24 ms on
    a 4096-token prefill and derived 324k tok/s prefill from it;
  * no `usage` was ever sent, so benchmark tools silently fell back to re-tokenizing the text.

Both are invisible to a functional test that only checks the decoded text, which is why they shipped.
A stub engine is enough to assert them, so this runs anywhere in milliseconds.
"""
from __future__ import annotations

import json

import torch

from models.demos.qwen3_6_a3b.demo import server as srv


class _StubEngine:
    """Minimum surface _stream_chat / _stream_completion touch."""

    def __init__(self, pieces):
        self._pieces = pieces
        self._finish_reason = "stop"
        self._n_generated = len(pieces)
        self.started = False  # set when generation is first advanced (i.e. after prefill)

        class _NullLock:
            def __enter__(self_):
                return None

            def __exit__(self_, *a):
                return False

        self.lock = _NullLock()

    def _generate_text(self, ids, max_new, stop=None, **sampling):
        self.started = True  # the real one has finished prefill by the time it yields
        text = ""
        for p in self._pieces:  # the real one yields the CUMULATIVE text each step
            text += p
            yield text

    def stream_text(self, ids, max_new, stop=None, **sampling):
        self.started = True
        yield from self._pieces  # the real one yields incremental deltas


def _events(chunks):
    """SSE lines -> parsed JSON payloads (dropping the [DONE] sentinel)."""
    out = []
    for c in chunks:
        body = c[len("data: ") :].strip()
        if body != "[DONE]":
            out.append(json.loads(body))
    return out


def _chat(pieces, prompt_len=7):
    ids = torch.zeros(1, prompt_len, dtype=torch.long)
    return _events(list(srv._stream_chat(_StubEngine(pieces), ids, 16, "m", expect_thinking=False)))


def test_no_chunk_is_emitted_before_generation_starts():
    """The property that actually matters: nothing reaches the client until generation has produced
    something. The role chunk may still LEAD the stream — it just must not be sent ahead of prefill,
    or a client timing "first chunk" measures the HTTP round-trip and derives a nonsense prefill rate.
    Asserted on TIMING, not chunk shape, so it holds however the role ends up packaged."""
    eng = _StubEngine(["Hello", " world"])
    stream = srv._stream_chat(eng, torch.zeros(1, 7, dtype=torch.long), 16, "m", expect_thinking=False)
    first = next(stream)
    assert eng.started, f"a chunk was emitted before generation began: {first!r}"
    stream.close()


def test_role_is_still_announced_first():
    """Wire format is unchanged: a role-only chunk still leads, it just arrives with the first token."""
    ev = _chat(["Hi"])
    roles = [i for i, e in enumerate(ev) if e["choices"][0]["delta"].get("role") == "assistant"]
    assert roles == [0] or (roles and roles[0] <= 1), f"role chunk missing or late: {ev}"


def test_stream_reports_usage():
    ev = _chat(["a", "b", "c"], prompt_len=11)
    usage = [e["usage"] for e in ev if e.get("usage")]
    assert usage, "no chunk carried usage; tools will re-tokenize locally"
    assert usage[-1] == {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}, usage[-1]


def test_terminal_chunk_has_finish_reason():
    ev = _chat(["x"])
    assert ev[-1]["choices"][0]["finish_reason"] == "stop", ev[-1]
    assert all(e["choices"][0]["finish_reason"] is None for e in ev[:-1]), "early finish_reason"


def test_empty_generation_is_still_well_formed():
    """A request that emits nothing must still produce role + a terminal finish chunk."""
    ev = _chat([])
    assert ev[0]["choices"][0]["delta"].get("role") == "assistant", ev
    assert ev[-1]["choices"][0]["finish_reason"] == "stop", ev


def test_completion_stream_reports_usage():
    ids = torch.zeros(1, 5, dtype=torch.long)
    ev = _events(list(srv._stream_completion(_StubEngine(["p", "q"]), ids, 16, "m")))
    assert ev[0]["choices"][0]["text"] == "p", "first completion chunk should be real text"
    assert ev[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}, ev[-1]
