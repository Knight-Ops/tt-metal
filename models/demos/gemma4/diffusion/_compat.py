# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shims for running DiffusionGemma on the released ``gemma4`` package.

DiffusionGemma was developed against the ``gemma4_cody`` fork, which carries a
handful of helpers that are not present in this released ``models.demos.gemma4``
package. Rather than patching the released core modules, the diffusion subdir
vendors those helpers here so it stays self-contained and drop-in.

Every symbol below is copied verbatim from ``gemma4_cody`` (only the few internal
``gemma4_cody`` import paths inside ``_patch_for_mock_mode_if_needed`` are
retargeted to ``gemma4``); provenance is noted per block.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from loguru import logger

import ttnn


# ── from gemma4_cody/utils/general_utils.py ──────────────────────────────────
def get_tensor_cache_file_path(cache_file_name, dtype, layout):
    if not cache_file_name:
        return None
    return Path(f"{cache_file_name}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")


def cached_tensor_placeholder(cache_file_name, dtype, layout):
    """Return a cheap placeholder when a TTNN tensor cache exists.

    ttnn.as_tensor() does not inspect the source torch tensor on cache hits, so
    this avoids materializing large HF tensors just to load the cached .tensorbin.
    """
    cache_path = get_tensor_cache_file_path(cache_file_name, dtype, layout)
    if cache_path is None or not cache_path.is_file():
        if cache_path is not None:
            logger.info(f"TTNN tensor cache not found at {cache_path}; loading HF tensor")
        return None
    logger.info(f"Using cached TTNN tensor {cache_path}; skipping HF tensor load")
    return torch.empty((), dtype=torch.bfloat16)


# ── from gemma4_cody/tt/model.py (host-side RoPE tables for arbitrary offsets) ─
def _default_rope_inv_freq(head_dim, theta):
    exponent = np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)
    return (1.0 / (np.float32(theta) ** exponent)).astype(np.float32, copy=False)


def _proportional_rope_inv_freq(head_dim, rope_params):
    theta = np.float32(rope_params["rope_theta"])
    factor = np.float32(rope_params.get("factor", 1.0))
    rope_proportion = rope_params.get("partial_rotary_factor", 1.0)
    rope_angles = int(rope_proportion * head_dim // 2)

    rotated_exponent = np.arange(0, 2 * rope_angles, 2, dtype=np.float32) / np.float32(head_dim)
    inv_freq = 1.0 / (theta**rotated_exponent)

    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        inv_freq = np.concatenate((inv_freq, np.zeros(nope_angles, dtype=np.float32)), axis=0)

    return (inv_freq / factor).astype(np.float32, copy=False)


def _create_rope_cache_tensors(hf_config, max_seq_len, layer_type):
    """Create Gemma4 RoPE caches without invoking HF's Torch rotary forward path."""
    rope_params = hf_config.rope_parameters[layer_type]
    rope_type = rope_params["rope_type"]

    if rope_type == "default":
        head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
        inv_freq = _default_rope_inv_freq(head_dim, rope_params["rope_theta"])
    elif rope_type == "proportional":
        head_dim = getattr(hf_config, "global_head_dim", None) or getattr(
            hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
        )
        inv_freq = _proportional_rope_inv_freq(head_dim, rope_params)
    else:
        raise ValueError(f"Unsupported Gemma4 RoPE type for offline cache generation: {rope_type!r}")

    positions = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1).astype(np.float32, copy=False)
    cos_np = np.ascontiguousarray(np.cos(emb).astype(np.float32, copy=False).reshape(1, max_seq_len, emb.shape[-1]))
    sin_np = np.ascontiguousarray(np.sin(emb).astype(np.float32, copy=False).reshape(1, max_seq_len, emb.shape[-1]))
    cos = torch.from_numpy(cos_np)
    sin = torch.from_numpy(sin_np)
    return cos, sin


# ── from gemma4_cody/tt/ccl.py ────────────────────────────────────────────────
def make_block_sharded_matmul_config(m, k_local, n_local, grid=(12, 8), with_out_mem=False):
    """L1-block-sharded matmul config for packed-verify dense matmuls.

    Measured at 512x5376x5376/bf4 on tf1: 0.51 ms (DRAM interleaved auto) →
    0.28 ms (105 TF/dev) — the dense matmul ceiling on Blackhole is mcast
    bandwidth, not FLOPs; block-sharded in0 cuts mcast volume 4×.

    Constraints: in0_block_w must divide K-shard tile count (K/32/gx).
    Returns (program_config, memory_config) — call x.to_memory_config(mem)
    first; output stays block-sharded.
    """
    gx, gy = grid
    if (k_local // 32) % gx:
        gx = 8  # e.g. o_proj K=2048 (64 tiles): 12 cols don't divide; 8 do
    mt, kt, nt = max(1, m // 32), max(1, k_local // 32), max(1, n_local // 32)
    per_core_m = -(-mt // gy)
    per_core_n = -(-nt // gx)
    shard_kt = -(-kt // gx)
    # in0_block_w = full K shard maximizes reuse, but circular buffers are
    # statically allocated: in0/in1 (double-buffered) + output + the L1
    # in0 shard must fit a core's 1.46 MB. Every M=512 shape fits at full
    # shard depth; the 2048 prefill bucket o_proj (K=2048 → 8-col grid,
    # 21 N-tiles/core) overflows — shrink in0_block_w to the largest
    # divisor of the K-shard that fits (kw14 → kw7 costs ~9% at M=512).
    tile_bytes = 2048
    budget = 1_400_000
    if with_out_mem:
        # gate/up chain keeps two L1-sharded outputs resident next to the CBs.
        budget -= 2 * per_core_m * per_core_n * tile_bytes
    kw = shard_kt
    while kw > 1:
        cb_bytes = (
            2 * per_core_m * kw  # in0 CB (double-buffered)
            + 2 * kw * per_core_n  # in1 CB (double-buffered)
            + 2 * per_core_m * per_core_n  # out CB + partials
            + per_core_m * shard_kt  # in0 L1 shard
        ) * tile_bytes
        if cb_bytes <= budget:
            break
        kw = max((d for d in range(1, kw) if shard_kt % d == 0), default=1)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kw,
        out_subblock_h=1,
        out_subblock_w=max(d for d in (8, 7, 4, 2, 1) if per_core_n % d == 0),
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        out_block_w=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )
    mem = ttnn.create_sharded_memory_config(
        (per_core_m * 32, shard_kt * 32),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    if not with_out_mem:
        return pc, mem
    # L1 block-sharded output: saves the DRAM write when the consumer is the
    # next matmul on the same grid (gate/up → mul → down_proj).
    out_mem = ttnn.create_sharded_memory_config(
        (per_core_m * 32, per_core_n * 32),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return pc, mem, out_mem


# ── from gemma4_cody/tt/shared_mlp.py ─────────────────────────────────────────
_LOFI_KCFG = None


def _mlp_kernel_config(mesh_device):
    global _LOFI_KCFG
    if _LOFI_KCFG is None:
        _LOFI_KCFG = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    return _LOFI_KCFG


# ── from gemma4_cody/demo/generate_weight_cache.py ────────────────────────────
def _parse_mesh_shape(value):
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("mesh shape must be ROWSxCOLS, for example 1x8")
    rows, cols = (int(part) for part in parts)
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError("mesh shape dimensions must be positive")
    return rows, cols


def _patch_for_mock_mode_if_needed():
    """When running against a mock cluster descriptor, skip device-side
    scratch-buffer allocations that would fail without a real command queue.

    The only such allocation during ``create_tt_model(create_kv_cache=False,
    create_rope_cache=False)`` is the fused matmul-reduce-scatter persistent
    buffer pair created inside ``Gemma4Attention.__init__`` and
    ``SharedMLP.__init__``. They're scratch space for the inference-time
    fused CCL kernel, not weight tensors — cache generation produces no
    tensorbins for them, so returning ``(None, None)`` is a clean no-op
    for the cache-gen path and harmless on real hardware (because we only
    patch when mock mode is detected).
    """
    if not os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        return
    import models.demos.gemma4.tt.attention as _attn_mod
    from models.demos.gemma4.tt import ccl as _ccl_mod
    from models.demos.gemma4.tt import model as _model_mod
    from models.demos.gemma4.tt import shared_mlp as _mlp_mod

    def _stub_buffers(*_args, **_kwargs):
        return None, None

    _ccl_mod.make_reduce_scatter_persistent_buffers = _stub_buffers
    _attn_mod.make_reduce_scatter_persistent_buffers = _stub_buffers
    _mlp_mod.make_reduce_scatter_persistent_buffers = _stub_buffers

    # SamplingGenerator allocates per-user k/p/temp/seed device tensors at
    # __init__ via uncached ttnn.from_torch. Cache generation never invokes
    # the sampling op, so replace the type with a marker that satisfies the
    # ``self.sampling is not None`` checks downstream — we don't need any
    # methods because nothing calls them in cache-gen.
    class _StubSampling:
        def __init__(self, *_args, **_kwargs):
            pass

    _model_mod.SamplingGenerator = _StubSampling

    logger.info(
        "Mock cluster descriptor detected (TT_METAL_MOCK_CLUSTER_DESC_PATH); "
        "stubbing make_reduce_scatter_persistent_buffers and SamplingGenerator "
        "for cache generation."
    )


# ── lazy-state-dict substate (restores gemma4_cody behavior) ─────────────────
# The released gemma4 utils/substate.substate() does
#   {k[plen:]: v for k, v in state.items() if k.startswith(prefix)}
# which (a) ignores a custom .substate() method and (b) realizes EVERY value via
# state.items() just to filter by prefix. With diffusion's lazy state dict (and
# this multi-GB MoE checkpoint on a small-RAM host) that materializes the whole
# checkpoint per call -> OOM. gemma4_cody's substate delegated to .substate()
# (a lazy sub-view) and otherwise fetched only the matched keys; restore it.
def _lazy_substate(state, key):
    if hasattr(state, "substate"):
        return state.substate(key)
    prefix = f"{key}."
    plen = len(prefix)
    # iterate keys, fetch values only for matches (no full .items() realization)
    return {k[plen:]: state[k] for k in state if k.startswith(prefix)}


def _lazy_has_substate(state, key):
    prefix = f"{key}."
    return any(k.startswith(prefix) for k in state)


def patch_lazy_substate():
    """Swap the released gemma4 substate()/has_substate() for the lazy-aware
    versions in every module that imported the names (each ``from ... import
    substate`` binds its own copy). Safe for plain dicts too — same result."""
    import importlib

    targets = [
        "models.demos.gemma4.utils.substate",
        "models.demos.gemma4.tt.model",
        "models.demos.gemma4.tt.layer",
        "models.demos.gemma4.tt.moe",
        "models.demos.gemma4.tt.router",
    ]
    for modname in targets:
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(m, "substate"):
            m.substate = _lazy_substate
        if hasattr(m, "has_substate"):
            m.has_substate = _lazy_has_substate


# ── expert weight loader (restores gemma4_cody behavior) ─────────────────────
# The released gemma4 load_expert_weights ALWAYS reads the HF gate_up_proj /
# down_proj from safetensors to unfuse them in torch — even on a cache hit —
# and caches the experts as bfloat16 (553 MB/proj/layer). On a small-RAM host
# over NFS that makes every model build re-read ~45 GB of HF tensors and pull
# ~1.7 GB/layer of bf16 cache. gemma4_cody's loader uses cached_tensor_placeholder
# to SKIP the HF read on a cache hit and caches as bfloat4_b (155 MB/proj/layer),
# matching the existing cache. Diffusion only uses experts.weights as the source
# for its own bf4 _gate_cat (see tt_model._build_moe_concat), so bf4 is correct.
_EXPERT_TILE = 32


def _cody_load_expert_weights(
    mesh_device,
    config,
    state_dict,
    mesh_config=None,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
):
    from models.demos.gemma4.tt.experts.weights import ExpertWeights
    from models.demos.gemma4.utils.general_utils import get_cache_file_name

    # Force bf4 to match the existing cache (and what diffusion re-quantizes to).
    weight_dtype = ttnn.bfloat4_b

    num_experts = config.num_experts
    hidden_size = config.hidden_size
    intermediate_size = config.moe_intermediate_size
    tp = mesh_config.tp if mesh_config else 1
    tp_suffix = f"_tp{tp}" if tp > 1 else ""
    gate_cache_name = get_cache_file_name(tensor_cache_path, f"gate_proj{tp_suffix}")
    up_cache_name = get_cache_file_name(tensor_cache_path, f"up_proj{tp_suffix}")
    down_cache_name = get_cache_file_name(tensor_cache_path, f"down_proj{tp_suffix}")

    if state_dict and "gate_up_proj" in state_dict:
        gate_proj = cached_tensor_placeholder(gate_cache_name, weight_dtype, ttnn.TILE_LAYOUT)
        up_proj = cached_tensor_placeholder(up_cache_name, weight_dtype, ttnn.TILE_LAYOUT)
        down_proj = cached_tensor_placeholder(down_cache_name, weight_dtype, ttnn.TILE_LAYOUT)

        if gate_proj is None or up_proj is None:
            fused = state_dict["gate_up_proj"]  # [E, 2*I, H]
            if gate_proj is None:
                gate_proj = fused[:, :intermediate_size, :].transpose(-2, -1).unsqueeze(0)
            if up_proj is None:
                up_proj = fused[:, intermediate_size:, :].transpose(-2, -1).unsqueeze(0)
        if down_proj is None:
            down_proj = state_dict["down_proj"].transpose(-2, -1).unsqueeze(0)

        if tp > 1:
            per_device = intermediate_size // tp
            padded_per_device = ((per_device + _EXPERT_TILE - 1) // _EXPERT_TILE) * _EXPERT_TILE
            pad_amount = padded_per_device * tp - intermediate_size
            if pad_amount > 0:
                if gate_proj.dim() > 0:
                    gate_proj = torch.nn.functional.pad(gate_proj, (0, pad_amount))
                if up_proj.dim() > 0:
                    up_proj = torch.nn.functional.pad(up_proj, (0, pad_amount))
                if down_proj.dim() > 0:
                    down_proj = torch.nn.functional.pad(down_proj, (0, 0, 0, pad_amount))
    else:
        gate_proj = up_proj = down_proj = None

    if tp > 1:
        per_device_intermediate = ((intermediate_size // tp + _EXPERT_TILE - 1) // _EXPERT_TILE) * _EXPERT_TILE
    else:
        per_device_intermediate = intermediate_size

    is_mesh = hasattr(mesh_device, "shape")
    if tp > 1 and mesh_config is not None:
        col_mapper = mesh_config.column_parallel(mesh_device)
        row_mapper = mesh_config.row_parallel(mesh_device)
    else:
        col_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
        row_mapper = col_mapper

    def _as(t, mapper, cache_name):
        return ttnn.as_tensor(
            t,
            device=mesh_device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,
            cache_file_name=cache_name,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return ExpertWeights(
        gate_proj=_as(gate_proj, col_mapper, gate_cache_name),
        up_proj=_as(up_proj, col_mapper, up_cache_name),
        down_proj=_as(down_proj, row_mapper, down_cache_name),
        intermediate_size_per_device=per_device_intermediate,
    )


def patch_expert_loader():
    """Replace the released gemma4 load_expert_weights with the cache-skipping
    bf4 version in every module that bound the name (experts package + weights)."""
    import importlib

    for modname in ("models.demos.gemma4.tt.experts", "models.demos.gemma4.tt.experts.weights"):
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(m, "load_expert_weights"):
            m.load_expert_weights = _cody_load_expert_weights


# ── from gemma4_cody/tests/test_factory.py ───────────────────────────────────
# Vendored so demo.py / server.py do not import the released gemma4
# tests/test_factory, which runs ``is_moe_model()`` (an AutoConfig.from_pretrained
# call) at *module import* time. That eager call dies on the gemma-diff checkpoint
# because its model_type is unknown to the venv's transformers — the very reason
# DiffusionGemma parses config.json by hand instead of using AutoConfig.
def parametrize_mesh_with_fabric(mesh_shapes=None):
    """Universal mesh parametrization with FABRIC_1D.

    Generates paired mesh_device + device_params parametrization for tests at
    any TP factor. Only includes mesh shapes that fit on the current system.
    Fabric is enabled (FABRIC_1D) for multi-device shapes and disabled for (1, 1).
    """
    import pytest

    num_devices = ttnn.get_num_devices()

    if mesh_shapes is None:
        all_shapes = [(1, 1), (1, 2), (1, 4), (1, 8), (1, 32)]
        mesh_shapes = [s for s in all_shapes if s[0] * s[1] <= num_devices]
    else:
        # User-provided shapes: still filter to those that fit, so an explicit
        # mesh_shapes=[(1,8)] decorator gracefully skips on smaller systems.
        mesh_shapes = [s for s in mesh_shapes if s[0] * s[1] <= num_devices]

    # CI mode: pick only the largest fitting shape so that one yaml entry can
    # target multiple SKUs and let each runner select the appropriate mesh.
    if os.getenv("CI") == "true" and len(mesh_shapes) > 1:
        mesh_shapes = [max(mesh_shapes, key=lambda s: s[0] * s[1])]

    if not mesh_shapes:
        params = [
            pytest.param(
                (1, 1),
                {"fabric_config": None},
                id="1x1",
                marks=pytest.mark.skip(reason="Not enough devices"),
            )
        ]
    else:
        params = [
            pytest.param(
                s,
                {"fabric_config": None if s == (1, 1) else ttnn.FabricConfig.FABRIC_1D},
                id=f"{s[0]}x{s[1]}",
            )
            for s in mesh_shapes
        ]

    def decorator(func):
        return pytest.mark.parametrize("mesh_device, device_params", params, indirect=True)(func)

    return decorator
