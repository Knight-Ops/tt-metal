# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Env-gated per-phase device-synced timers for the prefill path (QWEN36_PROFILE_PHASES=1).

Off by default and a complete no-op when off, so it never touches the decode path or normal runs.
When on, ``phase(mesh, name)`` brackets ``ttnn.synchronize_device`` + ``time.perf_counter`` so the
measured time is real device-completion time, not Python dispatch-return time.

CAVEAT: the per-phase sync serializes host and device, so the SUM of phase timers will exceed the
unsynced end-to-end wall clock (you lose the host/device pipelining that normally hides dispatch).
That is intentional: this instrument answers "where does DEVICE-BUSY time go" (delta vs attn vs MoE
vs head), not "what is the latency". For the kernel-vs-dispatch-gap split use the Tracy device
profiler instead. Report the breakdown at the end of a prefill via ``report()``.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict

import ttnn

ENABLED = os.environ.get("QWEN36_PROFILE_PHASES") == "1"

_totals: dict[str, float] = defaultdict(float)
_counts: dict[str, int] = defaultdict(int)


def reset():
    _totals.clear()
    _counts.clear()


class phase:
    """Context manager: time a named phase, device-synced. No-op unless QWEN36_PROFILE_PHASES=1."""

    def __init__(self, mesh, name):
        self.mesh = mesh
        self.name = name

    def __enter__(self):
        if ENABLED:
            ttnn.synchronize_device(self.mesh)
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if ENABLED:
            ttnn.synchronize_device(self.mesh)
            _totals[self.name] += time.perf_counter() - self._t0
            _counts[self.name] += 1
        return False


def report():
    if not ENABLED or not _totals:
        return
    total = sum(_totals.values())
    print(
        "\n[qwen36-prefill-profile] device-synced per-phase breakdown "
        "(sum > wall clock by design; see module docstring):"
    )
    for name in sorted(_totals, key=lambda k: -_totals[k]):
        ms = _totals[name] * 1000.0
        print(
            f"  {name:<18} {ms:9.1f} ms  ({100 * _totals[name] / total:5.1f}%)  "
            f"x{_counts[name]}  ({ms / _counts[name]:.2f} ms/call)"
        )
    print(f"  {'TOTAL (synced)':<18} {total * 1000.0:9.1f} ms")
