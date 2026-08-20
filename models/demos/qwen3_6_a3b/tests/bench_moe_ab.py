# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""A/B benchmark: tt-metal ``ttnn.experimental.moe_compute`` vs. the Qwen3.6 custom
``ttnn.sparse_matmul`` MoE path, single Blackhole P150, at Qwen3.6-35B-A3B MoE dims.

WHY: the model's MoE expert block currently runs on a *custom* ``ttnn.sparse_matmul`` addition to
tt-metal. tt-metal ships ``ttnn.experimental.moe_compute`` (a fused selective-tilize → w0/w1 matmul →
activation → w2 matmul, BFP4) that might be faster and would let us drop the custom op. The open
question is purely timing — is it faster or slower? — measured here head-to-head in one process.

KEY ARCHITECTURAL FACT (verified against live source, not any prior notes):
``moe_compute(compute_only=True)`` on a *single card* has NO combine. In ComputeOnly mode its final
output (slot 4) is ``[num_cores, DOUBLE_BUFFER_SIZE(=2), TOKEN_SIZE, hidden]`` — a 2-expert rolling
double-buffer window, not a per-token combined MoE output
(``moe_compute_device_operation.cpp:307-342``). The fused ``SelectiveReduceCombine`` only exists on the
non-ComputeOnly (CCL / multi-device) path. The op still *streams all 256 experts*, so its measured
latency IS the full expert compute — a valid quantity to A/B — but it is NOT a drop-in single-card
replacement without extra combine work. Hence the only sound single-card metric is **pre-combine
expert-compute latency**, and the combine is reported SEPARATELY (see ``combine_cost``).

WHAT IS COMPARED (all pre-combine expert compute, traced ms/call), across token counts {1,32,64,96,128,256}:
  * moe_compute  — ``ttnn.experimental.moe_compute(compute_only=True)``            [candidate]
  * sparse       — custom sparse_matmul: ``forward_sparse_decode`` (T=1, indexed/gather)
                   / ``forward_sparse_prefill`` (T>1, per-tile union)               [current custom op]
  * dense_xo     — dense batched matmul over all E experts, experts-only (router/shared/combine excluded)
  * dense_full   — ``moe.forward(x)`` (production full block: router+experts+shared+combine) — baseline-to-beat

Timing is value-independent for these ops (dense matmuls don't depend on weight values; the MoE always
activates exactly top_k experts), so synthetic random weights are used — no checkpoint load.

Tenstorrent best-practices baked in: all weights TILE + BFP4; token counts tile-aligned (T=1 padded to
a 32-tile for moe_compute — it SIGFPEs on a sub-tile buffer, so it is NEVER called at T=1); moe_compute
gets its required ROW_MAJOR dispatched sparse-buffer/indices/scores while the sparse/dense paths get
TILE inputs — each leg builds its own inputs. ``output_height_shard_dim=4`` (the only validated value).

Run (needs a real P150; grid must present the unharvested 11x10 Blackhole grid):
    ./python_env/bin/python models/demos/qwen3_6_a3b/tests/bench_moe_ab.py --iters 100
    ./python_env/bin/python -m pytest -q \
        models/demos/qwen3_6_a3b/tests/bench_moe_ab.py::test_bench_moe_ab \
        -o log_cli=true --log-cli-level=INFO
Env: QWEN36_MOE_AB_ITERS (pytest iters, default 50), QWEN36_MOE_AB_DEEPSEEK=1 (also time the
     deepseek_moe_fast_reduce_nc_fused combine primitive standalone — default OFF; see combine_cost).
"""
from __future__ import annotations

import argparse
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common import moe_gather
from models.demos.qwen3_6_a3b.tt.moe import _DENSE_TMAX, TtMoE

# ---------------------------------------------------------------------------- Qwen3.6 MoE dims
QWEN_E = 256  # num_experts (all resident on one card)
QWEN_H = 2048  # hidden_size
QWEN_I = 512  # moe_intermediate (per-expert FFN dim)
QWEN_SE = 512  # shared_expert_intermediate
QWEN_TOPK = 8  # num_experts_per_tok

TOKENS = [1, 32, 64, 96, 128, 256]  # decode / batched / prefill regimes (all >1 are tile multiples)
MC_TOKENS = [32, 64, 96, 128, 256]  # moe_compute is NEVER called at T=1 (sub-tile SIGFPE); use T=32
_MC_HEIGHT_SHARD = 4  # output_height_shard_dim: the only validated value (h=1 wrong, h=8 doesn't help)


# ---------------------------------------------------------------------------- timing primitive
def time_traced(mesh_device, make_call, iters=100, warmup=3):
    """ms/op around ttnn.execute_trace replays. Identical primitive to bench_decode.py:44 and
    probe_union_moe.py:34 (defined locally to avoid importing the heavy TtModel chain for 15 lines)."""
    for _ in range(warmup):  # eager compile (capture must not JIT)
        make_call()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    make_call()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)  # warm replay
    ttnn.synchronize_device(mesh_device)
    t0 = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    dt = time.time() - t0
    ttnn.release_trace(mesh_device, tid)
    return dt / iters * 1e3  # ms/op


def _try_time(mesh_device, make_call, iters, warmup):
    """Time a leg; on any failure sync the device and return (None, short_reason). L1-clash /
    grid failures fire on the first (warmup) call — before any trace is open — so the device stays
    clean and later legs are unaffected."""
    try:
        return time_traced(mesh_device, make_call, iters, warmup), None
    except Exception as e:  # noqa: BLE001 — a failing leg must never abort the whole run
        reason = f"{type(e).__name__}: {str(e)[:60]}"
        try:
            ttnn.synchronize_device(mesh_device)
        except Exception:  # noqa: BLE001
            pass
        return None, reason


# ---------------------------------------------------------------------------- current path (TtMoE)
def build_moe(mesh_device, expert_dtype=ttnn.bfloat4_b):
    """Standalone random-weight ``TtMoE`` at Qwen dims (BFP4 experts, matching production). The weight
    dict is built directly from torch — no reference module / checkpoint (timing is value-independent).
    Weight shapes verified against reference/qwen3_5_moe.py; construction mirrors tests/test_moe.py."""
    torch.manual_seed(2026)
    E, H, I, SE = QWEN_E, QWEN_H, QWEN_I, QWEN_SE
    weights = {
        "gate": torch.randn(E, H) * 0.04,  # [E, hidden]
        "gate_up_proj": torch.randn(E, 2 * I, H) * 0.04,  # [E, 2*inter, hidden]
        "down_proj": torch.randn(E, H, I) * 0.04,  # [E, hidden, inter]
        "se_gate_proj": torch.randn(SE, H) * 0.04,  # [se_inter, hidden]
        "se_up_proj": torch.randn(SE, H) * 0.04,  # [se_inter, hidden]
        "se_down_proj": torch.randn(H, SE) * 0.04,  # [hidden, se_inter]
        "se_router": torch.randn(1, H) * 0.04,  # [1, hidden]
    }
    return TtMoE(
        mesh_device,
        weights,
        E,
        QWEN_TOPK,
        expert_dtype=expert_dtype,
        dtype=ttnn.bfloat16,
        hidden=H,
        inter=I,
        se_inter=SE,
    )


def build_sparse_inputs(mesh_device, moe, T):
    """Build the (x, x2, routing, sparsity, topv, indices) operands for the current-path legs at T
    tokens. Runs the real router once (eager) then fabricates the sparse operands exactly as
    TtMoE.forward does (moe.py:368-380). Returns device-resident tensors reused inside the trace.

    routing is [T, E] and is what forward_sparse_prefill / dense consume directly. The [1,1,1,E]
    sparsity + gather indices are ONLY meaningful for the T==1 decode path (forward_sparse_decode);
    reshaping [T,E]->[1,1,1,E] is a volume error for T>1, so they are built only at T==1 (None otherwise)."""
    H = QWEN_H
    x = ttnn.from_torch(
        torch.randn(1, 1, T, H) * 0.5,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x2 = ttnn.reshape(x, [T, H])
    _shared, topv, topi, probs = moe._shared_and_router(x2)
    routing = ttnn.scatter(ttnn.multiply(probs, 0.0), 1, topi, topv)  # [T, E]
    sparsity = indices = None
    if T == 1:
        sparsity = ttnn.to_layout(ttnn.reshape(routing, [1, 1, 1, QWEN_E]), ttnn.ROW_MAJOR_LAYOUT)  # [1,1,1,E]
        indices = moe_gather.topk_to_indices(topi, QWEN_TOPK)  # [1,1,1,top_k] uint16 ROW_MAJOR
    ttnn.synchronize_device(mesh_device)
    return x, x2, routing, sparsity, topv, indices


def dense_experts_only(moe, x2, routing):
    """Dense batched-matmul expert compute, experts-only — mirrors TtMoE.forward's dense branch
    (moe.py:398-434) MINUS the router/shared/combine, so it is the fair pre-combine dense number.
    Returns the routed [T, hidden] output (before adding the shared expert)."""
    E, H, I = moe.num_experts, moe.hidden, moe.inter
    T = x2.shape[0]
    gate_up_w = ttnn.reshape(moe.gate_up_sp, [E, H, 2 * I])
    down_w = ttnn.reshape(moe.down_sp, [E, I, H])
    outs = []
    for t0 in range(0, T, _DENSE_TMAX):
        t1 = min(t0 + _DENSE_TMAX, T)
        tc = t1 - t0
        xe = ttnn.repeat(ttnn.reshape(ttnn.slice(x2, [t0, 0], [t1, H]), [1, tc, H]), ttnn.Shape([E, 1, 1]))
        hgu = ttnn.matmul(
            xe,
            gate_up_w,
            program_config=moe._dense_expert_pc(tc, 2 * I),
            compute_kernel_config=moe.compute_kernel_config,
        )
        gate = ttnn.slice(hgu, [0, 0, 0], [E, tc, I])
        up = ttnn.slice(hgu, [0, 0, I], [E, tc, 2 * I])
        h = ttnn.multiply(ttnn.silu(gate), up)
        wc = ttnn.reshape(ttnn.transpose(ttnn.slice(routing, [t0, 0], [t1, E]), 0, 1), [E, tc, 1])
        h = ttnn.multiply(h, wc)
        ye = ttnn.matmul(
            h, down_w, program_config=moe._dense_expert_pc(tc, H), compute_kernel_config=moe.compute_kernel_config
        )
        outs.append(ttnn.sum(ye, dim=0))
    return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=0)


# ---------------------------------------------------------------------------- candidate (moe_compute)
def build_moe_compute_common(mesh_device):
    """Build the token-count-INDEPENDENT moe_compute state once (the big BFP4 ring-packed weights +
    the shard dims). Returns a dict, or (None, reason) if the grid gate fails / the op's helper modules
    aren't importable / weight packing fails. Ports the proven setup from
    test_moe_compute_qwen36_probe.py:270-384 (weights half)."""
    # moe_compute-specific imports are lazy so the current-path legs still run if these are unavailable.
    try:
        from ttnn.experimental.moe_compute_utils import (
            auto_output_width_shard_dim,
            effective_matmul_ring_size,
            get_weight_core_shard_maps,
            get_weight_mem_configs,
            prepare_w0_w1_tensor_for_moe_compute,
            prepare_w2_tensor_for_moe_compute,
        )

        from tests.nightly.tg.ccl.moe.test_moe_compute_6U import create_torch_w0, create_torch_w1, create_torch_w2
    except Exception as e:  # noqa: BLE001
        return None, f"import {type(e).__name__}: {str(e)[:60]}"

    # --- grid gate: moe_compute hard-requires the unharvested 11x10 BH grid (7x10 on WH) ---
    arch = mesh_device.arch()
    grid = mesh_device.compute_with_storage_grid_size()
    if arch == ttnn.device.Arch.BLACKHOLE:
        if grid.y < 10 or grid.x < 11:
            return None, f"grid {grid.x}x{grid.y}<11x10"
    elif arch == ttnn.device.Arch.WORMHOLE_B0:
        if grid.y < 10 or grid.x < 7:
            return None, f"grid {grid.x}x{grid.y}<7x10"
    else:
        return None, f"arch {arch} unsupported"

    H, N = QWEN_H, QWEN_I
    num_layers, experts_per_device = 1, QWEN_E
    ring_n = effective_matmul_ring_size(mesh_device)
    owsd = auto_output_width_shard_dim(H, matmul_ring_size=ring_n)
    try:
        w0_w1_shard_map, w2_shard_map, dram_core_range_set = get_weight_core_shard_maps(mesh_device, H, N)
        torch_w0 = create_torch_w0(num_layers, experts_per_device, H, N)
        torch_w1 = create_torch_w1(num_layers, experts_per_device, H, N)
        torch_w2 = create_torch_w2(num_layers, experts_per_device, N, H)
        w0_w1_mem_config, w2_mem_config, _, _ = get_weight_mem_configs(
            num_layers, experts_per_device, H, N, w0_w1_shard_map, w2_shard_map, dram_core_range_set, has_bias=False
        )
        tt_w0_w1 = ttnn.from_torch(
            prepare_w0_w1_tensor_for_moe_compute(
                torch_w0, torch_w1, num_layers, experts_per_device, H, N, w0_w1_shard_map
            ),
            dtype=ttnn.bfloat4_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w0_w1_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt_w2 = ttnn.from_torch(
            prepare_w2_tensor_for_moe_compute(
                torch_w2, num_layers, experts_per_device, N, H, w2_shard_map, w0_w1_shard_map
            ),
            dtype=ttnn.bfloat4_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    except Exception as e:  # noqa: BLE001
        return None, f"weights {type(e).__name__}: {str(e)[:60]}"
    return {"tt_w0_w1": tt_w0_w1, "tt_w2": tt_w2, "owsd": owsd, "ring_n": ring_n}, None


def build_moe_compute_call(mesh_device, common, T):
    """Build the per-T moe_compute inputs (dispatched sparse buffer + indices + scores + mapping) and
    return a zero-arg ``make_call`` closure, or (None, reason). Ports the token-dependent setup from
    test_moe_compute_qwen36_probe.py:287-408. Never call at T=1 (sub-tile SIGFPE — caller must use T>=32)."""
    try:
        from ttnn.operations.ccl import MoEActivationFunction

        from tests.nightly.tg.ccl.moe.test_moe_compute_6U import (
            create_sharded_memory_config,
            gen_expert_mapping,
            gen_sparse_buffer_and_indices,
            tt_to_torch_dtype,
        )
    except Exception as e:  # noqa: BLE001
        return None, f"import {type(e).__name__}: {str(e)[:60]}"

    H, N, k = QWEN_H, QWEN_I, QWEN_TOPK
    dtype = ttnn.bfloat16
    mesh_shape = (1, 1)
    cluster_axis = None
    num_devices = 1
    total_tokens = T  # single dispatch device
    experts = QWEN_E
    experts_per_cluster = experts  # num_replicated_devices == 1
    experts_per_device = QWEN_E
    ohsd, owsd = _MC_HEIGHT_SHARD, common["owsd"]

    try:
        drain = ttnn.experimental.get_moe_tilize_drain_core(mesh_device, ohsd, owsd, H)
        tilize_drain_core = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(drain.x, drain.y), ttnn.CoreCoord(drain.x, drain.y))}
        )
        expert_mapping = gen_expert_mapping(
            num_devices, 1, cluster_axis, experts, experts_per_cluster, experts_per_device
        )
        tt_expert_mapping = ttnn.from_torch(
            expert_mapping,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        indices_mem_config = create_sharded_memory_config(tilize_drain_core, [total_tokens, k], ttnn.uint16)
        scores_mem_config = create_sharded_memory_config(tilize_drain_core, [total_tokens, k], dtype)
        sparse_buffer, expert_indices, expert_scores, _ = gen_sparse_buffer_and_indices(
            T, H, experts, k, mesh_shape, cluster_axis, dtype=tt_to_torch_dtype(dtype)
        )
        tt_sparse_buffer = ttnn.from_torch(
            sparse_buffer,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_expert_indices = ttnn.from_torch(
            expert_indices.reshape(total_tokens, k).unsqueeze(0).repeat(num_devices, 1, 1),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=indices_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_expert_scores = ttnn.from_torch(
            expert_scores.reshape(total_tokens, k).unsqueeze(0).repeat(num_devices, 1, 1),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=scores_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
    except Exception as e:  # noqa: BLE001
        return None, f"inputs {type(e).__name__}: {str(e)[:60]}"

    tt_w0_w1, tt_w2 = common["tt_w0_w1"], common["tt_w2"]

    def _call():
        # kwargs mirror the validated probe closure (test_moe_compute_qwen36_probe.py:388-408).
        return ttnn.experimental.moe_compute(
            tt_sparse_buffer,
            tt_expert_indices,
            tt_expert_scores,
            tt_expert_mapping,
            tt_w0_w1,
            tt_w2,
            layer_id=0,
            output_height_shard_dim=ohsd,
            intermediate_size=N,
            has_bias=False,
            cluster_axis=None,
            topology=None,
            num_links=None,
            mux_core_range_set=None,
            optional_output_tensor=None,
            optional_cross_device_semaphore=None,
            activation_type=MoEActivationFunction.SILU,
            compute_only=True,
        )

    return _call, None


# ---------------------------------------------------------------------------- combine (reported separately)
def combine_cost(mesh_device, iters, warmup):
    """Time the routed combine SEPARATELY from the expert compute (see module docstring: moe_compute
    stops before combine and can't be combined on one card). Two numbers:
      * gather_combine  — the weighted top_k reduce the custom sparse decode path uses today (always run).
      * deepseek_fused  — ttnn.experimental.deepseek_moe_fast_reduce_nc_fused, a CANDIDATE fused combine
        for the *sparse/gather* layout (NOT wired to moe_compute output — layout-incompatible). Opt-in
        (QWEN36_MOE_AB_DEEPSEEK=1, default OFF: it is a fabric/CCL op and best-effort on a single card)."""
    H, k = QWEN_H, QWEN_TOPK
    out = {}

    # --- gather_combine (decode): sum_i topv[i] * down_compact[i] over top_k slots -> [hidden] ---
    down_compact = ttnn.from_torch(
        torch.randn(k, H) * 0.5,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    topv = ttnn.from_torch(
        torch.rand(1, k),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ms, reason = _try_time(mesh_device, lambda: moe_gather.gather_combine(down_compact, topv, H, k), iters, warmup)
    out["gather_combine"] = ms if ms is not None else reason

    if os.environ.get("QWEN36_MOE_AB_DEEPSEEK") == "1":
        out["deepseek_fused"] = _combine_deepseek(mesh_device, iters, warmup)
    else:
        out["deepseek_fused"] = "off (set QWEN36_MOE_AB_DEEPSEEK=1)"
    return out


def _combine_deepseek(mesh_device, iters, warmup, T=32):
    """Best-effort standalone timing of deepseek_moe_fast_reduce_nc_fused at the gather layout it
    validates. Single-card cluster_axis=0 -> num_replicated=1 -> split_size=hidden, 1 output. Any
    failure (this is a fabric op) -> a reason string; never aborts the run."""
    try:
        H, k, E = QWEN_H, QWEN_TOPK, QWEN_E
        act = ttnn.from_torch(
            torch.rand(k, 1, T, H, dtype=torch.bfloat16) - 0.5,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        sc = torch.rand(T, 1, 1, k, dtype=torch.bfloat16)
        sc = sc / sc.sum(dim=-1, keepdim=True)
        scores = ttnn.from_torch(
            sc,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        idx = ttnn.from_torch(
            torch.randint(0, E, (T, 1, 1, k), dtype=torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        mapping = ttnn.from_torch(
            torch.zeros(1, E, dtype=torch.int32),  # all experts on device 0 (single card)
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        def _call():
            return ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                act,
                idx,
                mapping,
                reduce_dim=0,
                split_size=H,
                cluster_axis=0,
                output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                scores_tensor=scores,
                num_shared_experts=0,
                shared_expert_scale=1.0,
            )

        ms, reason = _try_time(mesh_device, _call, iters, warmup)
        return ms if ms is not None else reason
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:60]}"


# ---------------------------------------------------------------------------- driver
def _verdict(cand, base):
    if not isinstance(cand, (int, float)) or not isinstance(base, (int, float)):
        return "n/a"
    return f"moe_compute FASTER x{base / cand:.2f}" if cand < base else f"moe_compute SLOWER x{cand / base:.2f}"


def _cell(v):
    return f"{v:8.3f}" if isinstance(v, (int, float)) else f"{str(v)[:8]:>8}"


def run(mesh_device, iters=100, warmup=3):
    logger.info(f"[moe-ab] building random-weight TtMoE (BFP4 experts) at Qwen dims E={QWEN_E} H={QWEN_H} ...")
    moe = build_moe(mesh_device)
    mc_common, mc_reason = build_moe_compute_common(mesh_device)
    if mc_common is None:
        logger.warning(f"[moe-ab] moe_compute unavailable: {mc_reason} (current-path legs still run)")

    res = {}  # (leg, T) -> ms(float) | reason(str)
    for T in TOKENS:
        logger.info(f"[moe-ab] T={T}: building inputs + timing legs ...")
        x, x2, routing, sparsity, topv, indices = build_sparse_inputs(mesh_device, moe, T)

        # --- current custom sparse_matmul path: decode (T=1) vs per-tile prefill (T>1) ---
        if T == 1:
            ms, r = _try_time(
                mesh_device, lambda: moe.forward_sparse_decode(x2, sparsity, topv, indices), iters, warmup
            )
        else:
            ms, r = _try_time(mesh_device, lambda: moe.forward_sparse_prefill(x2, routing), iters, warmup)
        res[("sparse", T)] = ms if ms is not None else r

        # --- dense experts-only (pre-combine); dense full block (production baseline) ---
        if T > 1:
            ms, r = _try_time(mesh_device, lambda: dense_experts_only(moe, x2, routing), iters, warmup)
            res[("dense_xo", T)] = ms if ms is not None else r
        else:
            res[("dense_xo", T)] = "n/a(T=1)"
        # traced=True: this leg deliberately times the DENSE block, and the gathered path's host
        # readback is illegal under the trace capture that _try_time performs.
        ms, r = _try_time(mesh_device, lambda: moe.forward(x, traced=True), iters, warmup)
        res[("dense_full", T)] = ms if ms is not None else r

        # --- candidate moe_compute (never at T=1) ---
        if T in MC_TOKENS:
            if mc_common is None:
                res[("moe_compute", T)] = mc_reason
            else:
                mc_call, r = build_moe_compute_call(mesh_device, mc_common, T)
                if mc_call is None:
                    res[("moe_compute", T)] = r
                else:
                    ms, r = _try_time(mesh_device, mc_call, iters, warmup)
                    res[("moe_compute", T)] = ms if ms is not None else r
        else:
            res[("moe_compute", T)] = "->T=32"
        ttnn.synchronize_device(mesh_device)

    combine = combine_cost(mesh_device, iters, warmup)
    _report(res, combine, iters)


def _report(res, combine, iters):
    print("\n" + "=" * 96)
    print(f" MoE expert-compute A/B  —  moe_compute vs custom sparse_matmul  (Qwen3.6, single P150, {iters} replays)")
    print(
        f" dims: E={QWEN_E} H={QWEN_H} inter={QWEN_I} top_k={QWEN_TOPK}, BFP4 experts. Pre-combine ms/call (lower=faster)."
    )
    print("=" * 96)
    print(f"  {'T':>4} | {'moe_compute':>10} | {'sparse':>10} | {'dense_xo':>10} | {'dense_full':>10}   (regime)")
    print("  " + "-" * 76)
    regime = {1: "decode", 32: "batched", 64: "prefill", 96: "prefill", 128: "prefill", 256: "prefill"}
    for T in TOKENS:
        print(
            f"  {T:>4} | {_cell(res[('moe_compute', T)])} | {_cell(res[('sparse', T)])} | "
            f"{_cell(res[('dense_xo', T)])} | {_cell(res[('dense_full', T)])}   ({regime[T]})"
        )

    print("\n  --- verdict (candidate moe_compute vs best current pre-combine path) ---")
    # decode: sparse(T=1) vs moe_compute measured at T=32 (1 real token padded to a full 32-tile)
    if ("sparse", 1) in res and ("moe_compute", 32) in res:
        print(
            f"  decode  T=1 : sparse={_cell(res[('sparse', 1)])}  vs  moe_compute(@T=32, 1 real tok in a 32-tile)"
            f"={_cell(res[('moe_compute', 32)])}  ->  {_verdict(res.get(('moe_compute', 32)), res.get(('sparse', 1)))}"
        )
    for T in (32, 64, 96):
        if ("moe_compute", T) not in res:
            continue
        cur = [v for v in (res.get(("sparse", T)), res.get(("dense_xo", T))) if isinstance(v, (int, float))]
        best = min(cur) if cur else "n/a"
        tag = "batched" if T == 32 else "prefill"
        print(
            f"  {tag:<7} T={T:>3}: best_current={_cell(best)}  vs  moe_compute={_cell(res[('moe_compute', T)])}"
            f"  ->  {_verdict(res.get(('moe_compute', T)), best)}"
        )

    print("\n  --- combine cost (reported SEPARATELY; moe_compute has no single-card combine) ---")
    print(f"  gather_combine (decode, current path) : {_cell(combine['gather_combine'])} ms")
    print(
        f"  deepseek_moe_fast_reduce_nc_fused     : {_cell(combine['deepseek_fused'])}"
        f"   [candidate for the sparse/gather layout; NOT wired to moe_compute]"
    )

    print("\n  caveats:")
    print("   * moe_compute streams all 256 experts but retains no combinable output on one card")
    print("     (double-buffer, combine is CCL/multi-device only) -> a faster compute number is")
    print("     necessary-but-not-sufficient to replace the custom op; the combine is the follow-up.")
    print("   * moe_compute T=1 is measured at T=32: it pays a full 32-row tile for one real decode token.")
    print("   * moe_compute FAIL at T=128/256 is the expected L1 clash (prefill max-fit ~96 tokens).")
    print("   * weight ring-pack + TtMoE weight upload are one-time host cost, excluded from ms/call.")
    print("=" * 96 + "\n")


# ---------------------------------------------------------------------------- entry points
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 90_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_bench_moe_ab(mesh_device, mesh_shape):
    """Pytest entry (single-card, COL dispatch, trace region). QWEN36_MOE_AB_ITERS controls replays."""
    run(mesh_device, iters=int(os.environ.get("QWEN36_MOE_AB_ITERS", "50")), warmup=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        trace_region_size=90_000_000,
        dispatch_core_config=ttnn.DispatchCoreConfig(axis=ttnn.DispatchCoreAxis.COL),
    )
    try:
        run(mesh, iters=a.iters, warmup=3)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
