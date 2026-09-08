# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Cache-topology gate for the hybrid model.

The whole memory story of this model rests on ONE invariant: attention KV is allocated only for the
``full_attention`` layers (10 of 40), and the ``linear_attention`` layers (30 of 40) carry a
Gated-DeltaNet state whose size does NOT depend on the sequence length. That gives ~20 KiB/token of
KV instead of the ~80 KiB/token a dense 40-layer model would need — the 4x saving that makes long
context and B=32 fit a 32 GB P150 at all.

Nothing asserted it. ``layer_types`` is read straight from the checkpoint's ``config.json``, so a
variant checkpoint (different ``full_attention_interval``, or a fully-dense one) silently changes
both the footprint and which cache shape each layer gets. These tests pin it.

Checkpoint-free in the expensive sense: ``ModelArgs`` reads only ``config.json``, and ``_alloc_cache``
is exercised on a bare ``TtModel`` instance, so no safetensors and no weights are touched.
Run: ./python_env/bin/python -m pytest models/demos/qwen3_6_a3b/tests/test_cache_topology.py
"""
import os

import pytest

import ttnn
from models.demos.qwen3_6_a3b.tt.gated_delta import _CONV_ADDCHAIN
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs
from models.demos.qwen3_6_a3b.tt.paged_kv import block_size_default


class _StubLayer:
    """``_alloc_cache`` reads exactly one attribute off the layer."""

    def __init__(self, is_linear):
        self.is_linear = is_linear


def _allocator(mesh_device, args, max_seq, paged=False):
    """A TtModel carrying only the attributes ``_alloc_cache`` reads — no layers, no weights.

    Keep this in sync with `_alloc_cache`: it grew `paged_kv`/`kv_pager` when paged KV landed, and
    because the stub bypasses `__init__` entirely a missing attribute shows up as an AttributeError in
    THIS test rather than anywhere near the real cause."""
    m = TtModel.__new__(TtModel)
    m.mesh_device = mesh_device
    m.args = args
    m.max_seq = max_seq
    m.paged_kv = paged
    m.kv_pager = None
    m.n_conv_slots = 1  # parked conversations: the pager's host mapping rows (TtModel.__init__)
    return m


def _args_or_skip(mesh_device, max_seq):
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    if not os.path.exists(os.path.join(ckpt, "config.json")):
        pytest.skip(f"no config.json at {ckpt} (set QWEN36_CKPT)")
    return ModelArgs(mesh_device, ckpt_dir=ckpt, max_seq_len=max_seq)


def _free(cache):
    for t in cache.values() if isinstance(cache, dict) else cache:
        for x in t if isinstance(t, list) else [t]:
            ttnn.deallocate(x)


def test_layer_types_split(mesh_device):
    """The config's own layer_types is the source of truth, and it is honoured (not re-derived)."""
    args = _args_or_skip(mesh_device, 256)
    lt = args.layer_types
    assert len(lt) == args.n_layers, f"layer_types has {len(lt)} entries for {args.n_layers} layers"
    assert set(lt) <= {"linear_attention", "full_attention"}, sorted(set(lt))

    n_full = lt.count("full_attention")
    interval = args.config.full_attention_interval
    assert n_full == args.n_layers // interval, f"{n_full} full-attention layers, interval {interval}"
    # is_linear_layer must agree with the list for every index
    for i, t in enumerate(lt):
        assert args.is_linear_layer(i) == (t == "linear_attention"), i


def test_cache_type_per_layer(mesh_device):
    """Every layer gets the cache its type calls for — and only that."""
    max_seq = 256
    args = _args_or_skip(mesh_device, max_seq)
    m = _allocator(mesh_device, args, max_seq)

    for idx in range(args.n_layers):
        is_lin = args.is_linear_layer(idx)
        cache = m._alloc_cache(_StubLayer(is_lin))
        try:
            if is_lin:
                assert isinstance(cache, dict), f"layer {idx} (linear) must not get an attention KV cache"
                assert list(cache["recurrent_state"].shape) == [
                    1,
                    args.lin_num_v_heads,
                    args.lin_head_k_dim,
                    args.lin_head_v_dim,
                ]
                assert list(cache["conv_state"].shape) == [args.conv_kernel_size - 1, args.lin_conv_dim]
                if _CONV_ADDCHAIN:
                    assert len(cache["conv_rows"]) == args.conv_kernel_size - 1
                    for r in cache["conv_rows"]:
                        assert list(r.shape) == [1, args.lin_conv_dim]
            else:
                assert isinstance(cache, list) and len(cache) == 2, f"layer {idx} (full attn) needs [k, v]"
                for t in cache:
                    assert list(t.shape) == [1, args.n_kv_heads, max_seq, args.head_dim]
        finally:
            _free(cache)


def test_gdn_state_is_sequence_length_independent(mesh_device):
    """The invariant that earns the 4x: doubling max_seq doubles KV and leaves GDN state untouched."""
    args_a = _args_or_skip(mesh_device, 256)
    args_b = _args_or_skip(mesh_device, 512)

    def shapes(args, is_linear):
        m = _allocator(mesh_device, args, args.max_seq_len)
        cache = m._alloc_cache(_StubLayer(is_linear))
        try:
            if is_linear:
                out = {
                    k: [list(r.shape) for r in v] if isinstance(v, list) else list(v.shape) for k, v in cache.items()
                }
            else:
                out = [list(t.shape) for t in cache]
        finally:
            _free(cache)
        return out

    assert shapes(args_a, True) == shapes(args_b, True), "GDN state must not scale with max_seq"

    kv_a, kv_b = shapes(args_a, False), shapes(args_b, False)
    assert [s[2] for s in kv_a] == [256, 256] and [s[2] for s in kv_b] == [512, 512], (kv_a, kv_b)


def test_paged_cache_topology(mesh_device):
    """Paged: every attention layer draws block-shaped tensors from ONE pool, and what it costs is the
    pool's TOKEN BUDGET, not max_seq per layer. This is the whole difference from the flat arrangement,
    and the thing a continuous-batching deploy sizes against."""
    max_seq = 256
    args = _args_or_skip(mesh_device, max_seq)
    m = _allocator(mesh_device, args, max_seq, paged=True)
    block = block_size_default()

    caches = []
    try:
        for idx in range(args.n_layers):
            if args.is_linear_layer(idx):
                continue
            c = m._alloc_cache(_StubLayer(False))
            caches.append(c)
            assert isinstance(c, list) and len(c) == 2
            for t in c:
                # [num_blocks, n_kv, block, head_dim] -- NOT [1, n_kv, max_seq, head_dim]
                assert list(t.shape) == [m.kv_pager.num_blocks, args.n_kv_heads, block, args.head_dim]
        # one pool, one page table, shared by every attention layer
        assert m.kv_pager is not None
        assert m.kv_pager.num_blocks * block >= max_seq, "pool must reach the default budget (max_seq)"
        # the constraint the kernel enforces (paged_update_cache_device_operation.cpp:194): a slot may
        # not address more logical blocks than the pool physically has
        assert m.kv_pager.blocks_per_seq <= m.kv_pager.num_blocks
    finally:
        for c in caches:
            _free(c)


def test_kv_bytes_per_token(mesh_device):
    """Pin the per-token KV cost, and that it is 4x below the dense-equivalent. This is the number
    capacity planning depends on, so a config or dtype change should have to update it here."""
    args = _args_or_skip(mesh_device, 256)
    n_full = args.layer_types.count("full_attention")
    # K+V at bfloat16. QWEN36_KV_DTYPE=bf8 halves this (1.0625 B/elem -> ~10.9 KiB/token); the bf16
    # figure is pinned here because it is the default and what capacity planning quotes.
    per_layer = 2 * args.n_kv_heads * args.head_dim * 2
    per_token = n_full * per_layer
    dense = args.n_layers * per_layer

    assert per_layer == 2048, per_layer
    assert per_token == 20 * 1024, per_token  # 20 KiB/token
    assert dense == 4 * per_token, (dense, per_token)
