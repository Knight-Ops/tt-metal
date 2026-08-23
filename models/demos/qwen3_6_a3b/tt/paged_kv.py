# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Paged KV cache: block pool, on-demand allocation, and the device page table.

WHY. `TtModel._alloc_cache` allocates a FLAT `[1, n_kv, max_seq, head_dim]` cache per attention layer,
so every slot pays `max_seq` whether it uses it or not. MEASURED: a 512-token prefill at
`QWEN36_MAX_SEQ=262144` allocates **5.23 GB** of cache it never touches (weights 20.3 GB -> 25.4 GB,
leaving 6.3 GB free). Paging makes that proportional to the tokens actually present.

WHAT IT COSTS AT DECODE: nothing measurable. `tests/probe_paged_kv_perf.py`, this model's attention
dims, block=64:

    pos      KV MB/step   flat SDPA   paged SDPA   ratio    write flat   write paged
    1024        2.0        36.3 us      36.0 us    0.99       7.1 us       7.3 us
    8192       16.0        83.5         83.3       1.00       7.0          7.4
    32768      64.0       227.9        226.3       0.99       7.0          7.4
    131072    256.0       794.9        798.7       1.00       6.9          7.7

i.e. paged SDPA-decode is free (a 64-token block is 64 KB, big enough that the page-table indirection
amortises) and the write costs +0.3-0.8 us, which over 10 attention layers is +3-8 us on a ~25 ms step.
So this is a MEMORY and FLEXIBILITY change, not a speedup. Read `ROOFLINE.md` before treating it as one.

THE ONE THING THAT ACTUALLY SAVES MEMORY. A page table is only a win if the pool is OVER-SUBSCRIBED:
`tt_transformers` builds a static permutation covering `max_num_blocks` and partitions it across batch
slots, which at B=1 with a pool sized to `max_seq` is exactly as big as the flat cache. The saving comes
from decoupling the POOL BUDGET from the per-sequence cap -- allow 256K context but budget, say, 32K of
blocks, and hand them out as the sequence grows. That is what `pool_tokens` is, and why it defaults to
`max_seq` (behaviour-preserving) rather than to something clever.

TRACE SAFETY. The page table is allocated ONCE at a stable address and rewritten in place with
`ttnn.copy_host_to_device_tensor`, because a captured decode trace bakes the ADDRESSES of the buffers it
reads (`MEMORY.md` §1). Reallocating it per request would silently invalidate the trace; growing it in
place does not, and `PagedKV.ensure()` is a no-op on the steady-state steps that do not cross a block
boundary, so the common decode step still copies nothing from host.
"""
from __future__ import annotations

import os

import torch

import ttnn

UNMAPPED = -1  # page-table entry for a logical block with no physical block yet (tt_transformers uses -1)


def read_granularity_default() -> int:
    """The k-range granularity `sdpa_decode` actually READS, in tokens.

    This is NOT the same as the block size, and conflating them is a correctness bug rather than a
    tuning choice. `paged_scaled_dot_product_attention_decode` processes K in `k_chunk_size` pieces and
    rounds the range it touches UP to a whole chunk, so at cur_pos=128 with k_chunk_size=128 it reads
    positions 0..255 -- four 64-token blocks -- even though only three cover cur_pos. Any logical block
    in that rounded range must be MAPPED, or the kernel dereferences an UNMAPPED (-1) page-table entry
    and attends over whatever that address holds.

    MEASURED (tests/test_paged_cb_kv.py --slots 4): mapping only up to cur_pos made exactly the slots
    whose rounded range crossed a block boundary emit garbage -- pos 128 (needs 4 blocks, had 3) and
    pos 256 (needs 6, had 5) failed, while pos 64 and 192 (rounded range lands exactly on the mapped
    prefix) were correct. Must match Attention._sdpa_pc's k_chunk_size (attention.py:141)."""
    return int(os.environ.get("QWEN36_KV_READ_GRAN", "128"))


def block_size_default() -> int:
    """Tokens per block. 64 is what `blackhole/qwen36` and `tt_transformers` ship, and the probe above
    measured the indirection as free at that size; smaller blocks shrink the reads the SDPA kernel
    issues and would start to show."""
    return int(os.environ.get("QWEN36_KV_BLOCK", "64"))


class PagedKV:
    """One block pool + one shared device page table for ALL attention layers.

    Each layer gets its own `[num_blocks, n_kv, block_size, head_dim]` cache pair, and they all share
    this page table: logical block `i` of slot `b` lives at physical block `table[b, i]` WITHIN that
    layer's own tensor. That is the `tt_transformers` arrangement (one table, per-layer caches) and it
    keeps the table small -- a 32K pool at block 64 is 512 int32s.
    """

    def __init__(
        self,
        mesh_device,
        n_kv_heads,
        head_dim,
        max_seq,
        batch=1,
        block_size=None,
        pool_tokens=None,
        dtype=ttnn.bfloat16,
    ):
        self.mesh_device = mesh_device
        self.n_kv_heads, self.head_dim = n_kv_heads, head_dim
        self.dtype = dtype  # see ModelArgs.kv_cache_dtype
        self.block_size = block_size or block_size_default()
        self.read_granularity = read_granularity_default()
        self.batch = batch
        self.max_seq = max_seq
        # Pool budget, in tokens, across ALL slots. Defaults to batch*max_seq so enabling paging alone
        # changes nothing about footprint -- see the module docstring. Lower it to actually reclaim.
        pool = pool_tokens or int(os.environ.get("QWEN36_KV_POOL_TOKENS", str(batch * max_seq)))
        # Round the pool UP to a whole read-granularity window. Otherwise the top of the pool is
        # unusable: ensure() must map the kernel's rounded read range, so a pool of (say) 1300 tokens
        # at gran=128 can only serve positions below 1280 and the remainder is a silent cliff. With
        # gran=128 and block=64 this just means an even block count.
        gran_blocks = max(1, self.read_granularity // self.block_size)
        nb = max(1, -(-pool // self.block_size))
        self.num_blocks = -(-nb // gran_blocks) * gran_blocks
        # Per-slot logical capacity == the POOL, not max_seq. `paged_update_cache` enforces
        #     page_table.padded_shape()[1] <= cache_tensor.padded_shape()[0]
        # (paged_update_cache_device_operation.cpp:194), i.e. a slot's table may not address more
        # logical blocks than the pool physically has. So a single sequence CANNOT be over-subscribed:
        # measured, a 4096-entry table against a 128-block pool is rejected at the first decode step,
        # after prefill had already silently succeeded.
        #
        # This is the crux of what paging does and does not buy, and it is worth stating plainly:
        #   * ACROSS sequences it is a real win -- B slots share one pool, so the pool is sized by
        #     concurrent demand rather than by B * max_seq. That is the continuous-batching case.
        #   * WITHIN one sequence it saves nothing. Supporting 256K for a single user still needs 256K
        #     of blocks, because the user may actually reach it. Accepting a lower cap saves memory on
        #     the FLAT path too, just by lowering QWEN36_MAX_SEQ.
        # So for the single-user server this is memory-NEUTRAL. Its value here is that it unblocks
        # traced incremental prefill (PREFILL.md §4a) and vLLM continuous batching (README barrier #1).
        self.blocks_per_seq = self.num_blocks
        self.max_reachable = self.num_blocks * self.block_size

        assert self.max_reachable >= 1
        self._host = torch.full((batch, self.blocks_per_seq), UNMAPPED, dtype=torch.int32)
        # High-water mark: how many logical blocks of each slot are mapped. The mapping is always a
        # contiguous prefix (`ensure` fills 0..need-1), so this is exact, and it keeps `ensure` O(new
        # blocks) instead of O(all blocks mapped so far). That matters at long context: chunked prefill
        # calls ensure() once per chunk, so a rescan-from-zero made a 256K prompt O(n^2) -- 1.05M torch
        # scalar indexes across 512 chunks, measured 1-5 s of pure Python.
        self._mapped = [0] * batch
        self._free = list(range(self.num_blocks))  # simple free list; no eviction (see `ensure`)
        self._table = ttnn.from_torch(self._host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)

    # ---- geometry -------------------------------------------------------------------------------
    def cache_shape(self):
        return [self.num_blocks, self.n_kv_heads, self.block_size, self.head_dim]

    def alloc_layer_cache(self):
        """The `[k, v]` pair for one attention layer."""
        return [
            ttnn.zeros(self.cache_shape(), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
            for _ in range(2)
        ]

    def bytes_per_layer(self):
        bpe = 2.0 if self.dtype == ttnn.bfloat16 else 1.0625
        return int(2 * self.num_blocks * self.n_kv_heads * self.block_size * self.head_dim * bpe)

    # ---- mapping --------------------------------------------------------------------------------
    @property
    def table(self):
        return self._table

    def ensure(self, upto_pos, slot=0):
        """Map logical blocks so positions [0, upto_pos] are addressable for `slot`.

        Returns True if the device table was rewritten. Called every decode step; it is a pure no-op
        except on the ~1-in-`block_size` steps that cross a boundary, which is what keeps the steady
        state free of host->device traffic."""
        if upto_pos >= self.max_reachable:
            raise RuntimeError(
                f"position {upto_pos} exceeds the KV pool's reach ({self.max_reachable} tokens = "
                f"{self.num_blocks} blocks x {self.block_size}). The pool IS the per-sequence cap -- see "
                f"the note on blocks_per_seq. Raise QWEN36_KV_POOL_TOKENS."
            )
        # Round to what the kernel READS, not to what the position needs -- see read_granularity_default.
        reach = (upto_pos // self.read_granularity + 1) * self.read_granularity
        need = -(-reach // self.block_size)
        if need > self.blocks_per_seq:  # cannot happen now that the pool is gran-aligned; not silent if it does
            raise RuntimeError(
                f"position {upto_pos} needs {need} logical blocks (sdpa_decode reads up to {reach} "
                f"tokens at granularity {self.read_granularity}) but a slot may address only "
                f"{self.blocks_per_seq}. Raise QWEN36_KV_POOL_TOKENS."
            )
        changed = False
        for i in range(self._mapped[slot], need):
            if not self._free:
                raise RuntimeError(
                    f"KV block pool exhausted: {self.num_blocks} blocks of {self.block_size} tokens "
                    f"= {self.num_blocks * self.block_size} tokens, needed block {i} of slot {slot} "
                    f"(pos {upto_pos}). Raise QWEN36_KV_POOL_TOKENS, or lower QWEN36_MAX_SEQ."
                )
            self._host[slot, i] = self._free.pop(0)
            self._mapped[slot] = i + 1
            changed = True
        if changed:  # in place, at a stable address -- see the trace-safety note above
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(self._host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), self._table
            )
        return changed

    def release(self, slot=0):
        """Return a slot's blocks to the pool (end of request). Does NOT touch the device table: the
        next `ensure` rewrites the entries it needs, and a stale entry is never read past `cur_pos`."""
        row = self._host[slot]
        self._free.extend(int(b) for b in row[: self._mapped[slot]].tolist())
        row.fill_(UNMAPPED)
        self._mapped[slot] = 0

    def fill_page_table_for(self, slot=0):
        """The `[1, blocks_per_seq]` table `paged_fill_cache` wants for one user (it reads
        `batch_idx_ptr[0]`, so prefill is one call per user -- see tt_transformers attention.py:1022)."""
        return ttnn.from_torch(
            self._host[slot : slot + 1].contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
        )

    def __repr__(self):
        used = self.num_blocks - len(self._free)
        return (
            f"PagedKV(block={self.block_size}, pool={self.num_blocks} blocks = "
            f"{self.num_blocks * self.block_size} tokens, used={used}, "
            f"per-layer {self.bytes_per_layer() / 2**20:.0f} MB)"
        )
