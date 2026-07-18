# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Env-gated Tracy signposts for per-component profiling of prefill AND decode (QWEN36_SIGNPOST=1).

Off by default and a COMPLETE no-op when off (no f-string, no call, no import cost), so it never
touches normal runs — the point of gating: on our default Tracy-enabled build a bare ``signpost()``
still records a host-timeline message per call even with no capture attached, and there are hundreds
of calls per decode step across 40 layers. With the flag set, each ``region(name)`` emits a
start/end pair of Tracy messages (``name`` … ``name/end``) that the device profiler records; the
companion ``tests/signpost_report.py`` slices the op-perf CSV between them into a per-component
device-kernel-time breakdown.

Unlike ``prefill_profiler.phase`` these markers do NOT ``synchronize_device`` — a signpost is a pure
host-side Tracy message that enqueues nothing on the device command queue — so they are trace-safe
and (unlike phase) usable on the decode path. NOTE: host signposts fire during trace CAPTURE, not
during ``execute_trace`` replay, so profile the EAGER path (decode_step_eager / eager forward); the
per-op DEVICE KERNEL DURATION there is identical to traced replay (same kernels).

Name regions hierarchically ("layer.moe", "moe.router", …): consecutive nested start/end pairs let
the report build a tree and aggregate a name across all 40 layers in one step. Reuse the same names
the prefill ``prof.phase`` brackets already use where they overlap (moe.*, delta.*), so the two
instruments line up.
"""
from __future__ import annotations

import os

ENABLED = os.environ.get("QWEN36_SIGNPOST") == "1"

if ENABLED:
    try:
        from tracy import signpost as _sp
    except Exception:  # non-Tracy build / tracy not importable — degrade to a no-op

        def _sp(header, message=None):
            pass

else:

    def _sp(header, message=None):
        pass


def mark(name):
    """Emit a single instantaneous signpost (e.g. the literal "start"/"stop" the tt-perf-report and
    models/perf has_signposts=True slicers key off). No-op unless QWEN36_SIGNPOST=1."""
    if ENABLED:
        _sp(name)


class region:
    """Context manager emitting paired ``name`` / ``name/end`` signposts. No-op unless
    QWEN36_SIGNPOST=1. Nestable; signpost_report.py pairs them via a stack and sums the
    DEVICE KERNEL DURATION of the ops in between (self-time = region minus nested children)."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if ENABLED:
            _sp(self.name)
        return self

    def __exit__(self, *exc):
        if ENABLED:
            _sp(self.name + "/end")
        return False
