# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-memory measurement for the Qwen3.6-35B-A3B demo.

Every memory claim in this model should be a measured number, not a computed one. This wraps
``ttnn.get_memory_view`` into something cheap enough to call from inside a forward pass.

WHY IT IS SAFE TO CALL ANYWHERE (including mid-forward and inside trace capture): the allocator is
a HOST-side book-keeper. ttnn enqueues work asynchronously but allocates synchronously, so the
allocator's state at the moment of a python call is exactly the set of buffers currently live — no
``synchronize_device``, no device round-trip, no perturbation of the thing being measured. That is
what makes ``probe()`` usable as a peak tracker inside a hot loop.

WHAT "ALLOCATED" MEANS HERE. ``MemoryView`` reports per BANK; DRAM on a P150 has 8 banks and L1 has
one per Tensix core, so this module multiplies back up to a whole-device figure. A tensor is counted
from ``ttnn.from_torch``/op-output until its last python reference dies (or ``ttnn.deallocate``), so
"allocated" is the live set, and ``peak`` is the high-water mark across the probes you placed.

    from models.demos.qwen3_6_a3b.tt import memstat

    memstat.report(mesh, "after build")             # one-shot line
    with memstat.track(mesh, "prefill") as t:       # before/after + peak over probes
        model.forward(ids)
    print(t)

Set ``QWEN36_MEMSTAT=1`` to arm the in-model ``probe()`` calls (they are a no-op otherwise, so
leaving them in the hot path costs one module-level bool test).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field

from loguru import logger

import ttnn

# Arms the probe() calls placed inside model code. Off by default: probing is cheap but not free
# (a few hundred ns of allocator query), and a decode step is latency-critical.
ENABLED = os.environ.get("QWEN36_MEMSTAT", "0") != "0"

MB = 1024.0 * 1024.0


@dataclass(frozen=True)
class Region:
    """Whole-device totals for one allocator region (banks already multiplied back in)."""

    name: str
    banks: int
    total: int
    allocated: int
    free: int
    largest_free_per_bank: int

    def __str__(self) -> str:
        return (
            f"{self.name} {self.allocated / MB:8.1f}/{self.total / MB:.0f} MB used, "
            f"{self.free / MB:8.1f} MB free (largest contiguous block/bank {self.largest_free_per_bank / MB:.1f} MB)"
        )


def region(mesh_device, buffer_type=ttnn.BufferType.DRAM, name=None) -> Region:
    """Snapshot one allocator region. Host-side only; does not synchronize the device."""
    v = ttnn.get_memory_view(mesh_device, buffer_type)
    n = int(v.num_banks)
    return Region(
        name=name or str(buffer_type).rsplit(".", 1)[-1],
        banks=n,
        total=int(v.total_bytes_per_bank) * n,
        allocated=int(v.total_bytes_allocated_per_bank) * n,
        free=int(v.total_bytes_free_per_bank) * n,
        largest_free_per_bank=int(v.largest_contiguous_bytes_free_per_bank),
    )


def dram(mesh_device) -> Region:
    return region(mesh_device, ttnn.BufferType.DRAM, "DRAM")


def l1(mesh_device) -> Region:
    return region(mesh_device, ttnn.BufferType.L1, "L1")


def report(mesh_device, label: str = "", to_logger: bool = False) -> tuple[Region, Region]:
    """Print (or log) both regions. Returns them so a caller can keep the numbers."""
    d, one = dram(mesh_device), l1(mesh_device)
    tag = f"[mem] {label}: " if label else "[mem] "
    for r in (d, one):
        line = f"{tag}{r}"
        logger.info(line) if to_logger else print(line)
    return d, one


# --------------------------------------------------------------------------- peak tracking


@dataclass
class Tracker:
    """Before/after DRAM plus a high-water mark over the probes hit while it is active.

    ``peak`` only sees the points where ``probe()`` is actually called, so it is a lower bound on the
    true peak — an unprobed allocation between two probes is invisible. Place probes at the points a
    footprint claim is about (e.g. just before the down-projection matmul in the MoE, where the live
    set is largest) rather than trusting a coarse before/after difference.
    """

    mesh_device: object
    label: str = ""
    start: int = 0
    end: int = 0
    peak: int = 0
    peak_at: str = ""
    marks: list[tuple[str, int]] = field(default_factory=list)

    def mark(self, where: str = "") -> int:
        """Sample now; update the high-water mark. Returns allocated bytes."""
        a = dram(self.mesh_device).allocated
        self.marks.append((where, a))
        if a > self.peak:
            self.peak, self.peak_at = a, where
        return a

    @property
    def delta(self) -> int:
        """Net DRAM retained across the tracked region (leak detector)."""
        return self.end - self.start

    def __str__(self) -> str:
        s = (
            f"[mem] {self.label or 'region'}: {self.start / MB:.1f} -> {self.end / MB:.1f} MB "
            f"(net {self.delta / MB:+.1f} MB)"
        )
        if self.peak > max(self.start, self.end):
            s += f", peak {self.peak / MB:.1f} MB (+{(self.peak - self.start) / MB:.1f} transient"
            s += f" at {self.peak_at})" if self.peak_at else ")"
        return s


# The tracker probe() reports into. A single module-level slot (not a stack): the interesting
# question is always "what did THIS forward pass peak at", and nesting trackers would double-count.
_active: Tracker | None = None


@contextmanager
def track(mesh_device, label: str = "", arm: bool = True):
    """Measure a block: DRAM before/after, plus the peak over any probe() inside it.

    ``arm=True`` turns probes on for the duration even if QWEN36_MEMSTAT is unset, so a benchmark can
    measure peaks without the env var while normal runs pay nothing.
    """
    global _active, ENABLED
    t = Tracker(mesh_device, label)
    t.start = t.peak = dram(mesh_device).allocated
    prev_active, prev_enabled = _active, ENABLED
    _active = t
    ENABLED = ENABLED or arm
    try:
        yield t
    finally:
        t.end = dram(mesh_device).allocated
        if t.end > t.peak:
            t.peak, t.peak_at = t.end, "exit"
        _active, ENABLED = prev_active, prev_enabled


def probe(mesh_device, where: str = "") -> None:
    """Sample the live set into the active tracker. No-op unless armed — safe in the hot path."""
    if ENABLED and _active is not None:
        _active.mark(where)
