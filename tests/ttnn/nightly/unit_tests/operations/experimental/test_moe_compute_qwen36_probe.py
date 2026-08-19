# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Stage-0 feasibility probe: can ``ttnn.experimental.moe_compute`` (compute_only=True)
replace the custom ``sparse_matmul`` indexed/gather path used by the Qwen3.6-35B-A3B
MoE (models/demos/qwen3_6_a3b)?

This does NOT build a full A/B benchmark. It only answers, at Qwen's MoE dims on a
single Blackhole P150, the go/no-go gates from the plan
(~/.claude/plans/i-need-you-to-async-rain.md):

  1. Grid gate  - the P150 exposes the >=11x10 unharvested BH grid moe_compute needs.
  2. Runs       - no TT_FATAL at 256 experts / N=512 (moe_compute is tuned for <=8
                  experts/device on DeepSeek/GPT; Qwen single-card = 256).
  3. Packs      - the ring-aware weight packer accepts N=512 (16 tiles) / H=2048 (64 tiles).
  4. Correct    - matmul-output PCC vs the torch golden clears the helper's floor (~0.984).
  5. Timing     - BONUS, non-blocking: rough ms/call vs the sparse_matmul baselines.

The correctness gates (1-4) reuse ``_run_moe_compute_single_card_test`` VERBATIM - same
synthetic weights, goldens, and validators as the DeepSeek/GPT single-card test - so this
file is just a thin Qwen-dim parametrization. MoE op correctness/timing is value-independent,
so no checkpoint load is needed.

Qwen3.6 MoE dims: hidden_size=2048, moe_intermediate (N)=512, num_experts=256, top_k=8,
BFP4 experts, activation = silu(gate)*up (no bias) => MoEActivationFunction.SILU, has_bias=False.

Baselines to interpret the bonus timing against (prior measurements, per MoE layer):
  - Decode T=1        : sparse_matmul indexed/gather ~0.32 ms/layer
  - Batched decode B32: dense ttnn.matmul ~3.0 ms/step (~0.094 ms/token)
  - Prefill (T large) : dense ttnn.matmul ~3.6 ms/layer

MEASURED on a P150 (Blackhole 11x10) 2026-07-21 at Qwen dims, compute_only=True:
  - Grid gate ...................... PASS (P150 is exactly 11x10)
  - Batched decode, T=32, h=4 ...... PASS, matmul PCC ~0.99  (tile-aligned token count)
  - Single-token decode, T=1 ....... SIGFPE (div-by-zero) at ANY output_height_shard_dim.
      Root cause: total_tokens=1 is SUB-TILE. On TT everything is 32x32-tile-granular, so a
      decode token must be padded up to a 32-row tile (exactly what the current sparse_matmul
      decode path does implicitly via TILE_LAYOUT, moe.py:243). moe_compute does NOT auto-pad a
      sub-tile ROW_MAJOR sparse buffer -> the caller must pass a tile-aligned (>=32) token count.
      NOT a fundamental limit; skipped below so it doesn't core-dump / dirty the device.
  - output_height_shard_dim=1 ...... FAIL (matmul output wrong). Use h=4 (the validated value).
  - Prefill T=64/T=96, h=4 ......... PASS, matmul PCC ~0.99
  - Prefill T=128, h=4 AND h=8 ..... FAIL: static CBs clash with L1 buffers. Max-fit chunk is
      ~96 tokens (3 tiles); more token-cores (h=8) does NOT fix it (clash is the weight/CB
      region, not per-core token buffers).
  => Viable & correct for TILE-ALIGNED token counts in [32, 96] at Qwen dims; the open question
     is TIMING vs sparse_matmul/dense (batched-decode A/B), and the active-expert count at low
     real-token counts (B=1 padded). Prefill is L1-capped at ~96-token chunks.

Run (needs a real P150; dispatch_core_axis=COL is set via device_params):
  ./python_env/bin/python -m pytest -q \
    tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_compute_qwen36_probe.py \
    -o log_cli=true --log-cli-level=INFO
"""

import time

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.ccl import MoEActivationFunction
from ttnn.experimental.moe_compute_utils import (
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w2_tensor_for_moe_compute,
    get_weight_core_shard_maps,
    get_weight_mem_configs,
    auto_output_width_shard_dim,
    effective_matmul_ring_size,
)

# Reuse the single-card test body verbatim for the correctness gates. This is the whole
# point of the probe: never duplicate the compute/validation logic - same goldens, same PCC.
from tests.ttnn.nightly.unit_tests.operations.experimental.test_moe_compute_single_card import (
    _run_moe_compute_single_card_test,
)

# Setup helpers shared with the 6U/single-card tests (used only by the bonus timing case).
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import (
    create_torch_w0,
    create_torch_w1,
    create_torch_w2,
    create_sharded_memory_config,
    gen_expert_mapping,
    gen_sparse_buffer_and_indices,
    tt_to_torch_dtype,
)

# ----------------------------------------------------------------------------------------
# Qwen3.6-35B-A3B MoE dimensions (see models/demos/qwen3_6_a3b/tt/moe.py & model_config.py).
# ----------------------------------------------------------------------------------------
QWEN_HIDDEN_SIZE = 2048  # H
QWEN_N = 512  # moe_intermediate (per-expert FFN dim)
QWEN_EXPERTS = 256  # num_experts (all resident on one card => experts_per_device)
QWEN_TOP_K = 8  # num_experts_per_tok
QWEN_ACTIVATION = MoEActivationFunction.SILU  # silu(gate) * up, no bias

# One token-count per regime the user asked to cover. T=1 may need tile padding; kept as
# its own case so a failure there still lets the batched/prefill cases report.
#   1   -> decode (single-user)     32  -> batched decode (B=32)
#   128 -> prefill chunk            256 -> larger prefill chunk
REGIME_TOKENS = [
    # Sub-tile T=1 SIGFPEs (core-dumps) because total_tokens=1 is below one 32-row tile and
    # moe_compute doesn't auto-pad a sub-tile ROW_MAJOR buffer. On TT a decode token must be
    # padded to a 32-tile anyway (the current sparse_matmul path does this via TILE_LAYOUT);
    # so the correct decode representation is the tile-aligned T=32 case below. Skipped so the
    # probe never core-dumps / dirties the device.
    pytest.param(
        1,
        id="decode_T1",
        marks=pytest.mark.skip(
            reason="sub-tile total_tokens=1 SIGFPEs; pad decode token to a 32-tile (use T=32). See module docstring."
        ),
    ),
    pytest.param(32, id="batched_B32"),  # one tile -> the correct tile-aligned decode/batch representation
    pytest.param(128, id="prefill_T128"),  # measured: FAILS (L1 clash); prefill max-fit ~96 tokens
    pytest.param(256, id="prefill_T256"),  # measured: FAILS (L1 clash)
]

# dispatch_core_axis=COL matches the single-card / 6U test setup (keeps all 10 functional
# rows available so the op's tilize/combine core placement fits). See the single-card test.
_DEVICE_PARAMS_CORRECTNESS = {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 500000}
_DEVICE_PARAMS_TIMING = {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 90_000_000}


# ========================================================================================
# GATES 1-4: does moe_compute run correctly at Qwen dims? (thin wrapper, verbatim reuse)
# ========================================================================================
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS_CORRECTNESS], indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("tokens_per_device", REGIME_TOKENS)
def test_moe_compute_qwen36_feasibility(mesh_device, mesh_shape, tokens_per_device):
    """compute_only=True on a 1x1 mesh at Qwen3.6 MoE dims (256 experts, H=2048, N=512, top_k=8).

    A SKIP means the grid gate failed (P150 harvested below 11x10) => moe_compute is out.
    A FAIL (TT_FATAL / assert) pinpoints the blocker (256-expert scale, N=512 packing, or
    T=1 padding). A PASS means moe_compute is viable at this regime; check the paired timing
    case for the ballpark ms/call.
    """
    ring_n = effective_matmul_ring_size(mesh_device)
    output_width_shard_dim = auto_output_width_shard_dim(QWEN_HIDDEN_SIZE, matmul_ring_size=ring_n)

    logger.info(
        f"[qwen36-probe] feasibility: tokens_per_device={tokens_per_device} experts={QWEN_EXPERTS} "
        f"H={QWEN_HIDDEN_SIZE} N={QWEN_N} top_k={QWEN_TOP_K} ring_n={ring_n} "
        f"output_width_shard_dim={output_width_shard_dim} dtype=bf16 weights=bf4"
    )

    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=QWEN_EXPERTS,
        tokens_per_device=tokens_per_device,
        selected_experts_k=QWEN_TOP_K,
        N=QWEN_N,
        hidden_size=QWEN_HIDDEN_SIZE,
        output_height_shard_dim=4,
        output_width_shard_dim=output_width_shard_dim,
        dtype=ttnn.bfloat16,
        activation_type=QWEN_ACTIVATION,
        has_bias=False,
    )


# ========================================================================================
# FIX INVESTIGATION: can moe_compute be made to cover decode (T=1) and prefill (large T)?
#
# Feasibility run found: T=1 -> SIGFPE (div-by-zero), T=128 -> TT_FATAL (static CBs clash
# with L1). These cases probe candidate fixes. Each SIGFPE aborts the whole process, so run
# ONE id at a time:  pytest ...::test_moe_compute_qwen36_fix -k "decode_T1_h1"
#
#   T=1 SIGFPE hypothesis: output_height_shard_dim=4 > total_tokens=1 makes a per-shard token
#     count go to 0 -> later modulo divides by zero. output_height_shard_dim=1 should fix it.
#   Prefill L1 clash: sweep token counts to find the max that fits L1 at output_height_shard_dim=4
#     (tells us the moe_compute prefill chunk size), and retry with height=1.
# ========================================================================================
FIX_CASES = [
    pytest.param(1, 1, id="decode_T1_h1"),  # single-token decode with 1 token-core
    pytest.param(1, 2, id="decode_T1_h2"),
    pytest.param(32, 1, id="decode_T32_h1"),  # sanity: height=1 works at a full tile
    pytest.param(64, 4, id="prefill_T64_h4"),  # max-fit sweep for the L1 clash
    pytest.param(96, 4, id="prefill_T96_h4"),
    pytest.param(128, 8, id="prefill_T128_h8"),  # more token-cores -> smaller per-core L1
]


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS_CORRECTNESS], indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("tokens_per_device, output_height_shard_dim", FIX_CASES)
def test_moe_compute_qwen36_fix(mesh_device, mesh_shape, tokens_per_device, output_height_shard_dim):
    """Probe candidate fixes for the T=1 SIGFPE and the prefill L1 clash at Qwen dims."""
    ring_n = effective_matmul_ring_size(mesh_device)
    output_width_shard_dim = auto_output_width_shard_dim(QWEN_HIDDEN_SIZE, matmul_ring_size=ring_n)
    logger.info(
        f"[qwen36-probe] fix: tokens_per_device={tokens_per_device} "
        f"output_height_shard_dim={output_height_shard_dim} output_width_shard_dim={output_width_shard_dim}"
    )
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=QWEN_EXPERTS,
        tokens_per_device=tokens_per_device,
        selected_experts_k=QWEN_TOP_K,
        N=QWEN_N,
        hidden_size=QWEN_HIDDEN_SIZE,
        output_height_shard_dim=output_height_shard_dim,
        output_width_shard_dim=output_width_shard_dim,
        dtype=ttnn.bfloat16,
        activation_type=QWEN_ACTIVATION,
        has_bias=False,
    )


# ========================================================================================
# GATE 5 (BONUS, non-blocking): rough ms/call for moe_compute at Qwen dims.
# Reuses the same input construction as the single-card test, minus goldens/validation,
# and times the op via a traced replay (time_traced primitive from bench_decode.py:44-61).
# On any setup error it SKIPs (logs the reason) so it never muddies the correctness signal.
# ========================================================================================
def _time_traced(mesh_device, make_call, iters=50, warmup=3):
    """Wall-clock ms/op around ttnn.execute_trace replays (see bench_decode.py:44-61)."""
    for _ in range(warmup):
        make_call()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    make_call()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    t0 = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    dt = time.time() - t0
    ttnn.release_trace(mesh_device, tid)
    return dt / iters * 1e3


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS_TIMING], indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("tokens_per_device", REGIME_TOKENS)
def test_moe_compute_qwen36_timing(mesh_device, mesh_shape, tokens_per_device):
    """BONUS: rough per-call ms for moe_compute(compute_only=True) at Qwen dims.

    Non-authoritative - the correctness test above is the real go/no-go. This just logs a
    ballpark to compare against the sparse_matmul baselines in the module docstring.
    """
    torch.manual_seed(2003)

    experts_per_device = QWEN_EXPERTS
    hidden_size = QWEN_HIDDEN_SIZE
    N = QWEN_N
    selected_experts_k = QWEN_TOP_K
    dtype = ttnn.bfloat16
    num_layers = 1
    layer_id = 0

    # Single device, no CCL (mirrors _run_moe_compute_single_card_test derivations).
    cluster_axis = None
    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = num_devices
    num_replicated_devices = num_devices // num_dispatch_devices
    total_tokens = tokens_per_device * num_dispatch_devices
    experts = experts_per_device * num_devices
    experts_per_cluster = experts // num_replicated_devices

    ring_n = effective_matmul_ring_size(mesh_device)
    output_height_shard_dim = 4
    output_width_shard_dim = auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n)

    try:
        # --- Grid gate (same check the correctness helper does; skip cleanly if unmet) ---
        arch = mesh_device.arch()
        grid = mesh_device.compute_with_storage_grid_size()
        if arch == ttnn.device.Arch.BLACKHOLE:
            if grid.y < 10 or grid.x < 11:
                pytest.skip(f"BH grid {grid.x}x{grid.y} < 11x10 required by moe_compute (grid gate).")
        elif arch == ttnn.device.Arch.WORMHOLE_B0:
            if grid.y < 10 or grid.x < 7:
                pytest.skip(f"WH grid {grid.x}x{grid.y} < 7x10 required by moe_compute (grid gate).")
        else:
            pytest.skip(f"moe_compute timing probe: arch {arch} unsupported (WH/BH only).")

        # --- Tilize input tensors (dispatched sparse buffer + indices + scores) ---
        drain = ttnn.experimental.get_moe_tilize_drain_core(
            mesh_device, output_height_shard_dim, output_width_shard_dim, hidden_size
        )
        tilize_drain_core = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(drain.x, drain.y), ttnn.CoreCoord(drain.x, drain.y))}
        )

        expert_mapping = gen_expert_mapping(
            num_devices, num_replicated_devices, cluster_axis, experts, experts_per_cluster, experts_per_device
        )
        tt_expert_mapping = ttnn.from_torch(
            expert_mapping,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        indices_mem_config = create_sharded_memory_config(
            tilize_drain_core, [total_tokens, selected_experts_k], ttnn.uint16
        )
        scores_mem_config = create_sharded_memory_config(tilize_drain_core, [total_tokens, selected_experts_k], dtype)

        sparse_buffer, expert_indices, expert_scores, _ = gen_sparse_buffer_and_indices(
            tokens_per_device,
            hidden_size,
            experts,
            selected_experts_k,
            mesh_shape,
            cluster_axis,
            dtype=tt_to_torch_dtype(dtype),
        )

        tt_sparse_buffer = ttnn.from_torch(
            sparse_buffer,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        expert_indices_replicated = (
            expert_indices.reshape(total_tokens, selected_experts_k).unsqueeze(0).repeat(num_devices, 1, 1)
        )
        tt_expert_indices = ttnn.from_torch(
            expert_indices_replicated,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=indices_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        expert_scores_replicated = (
            expert_scores.reshape(total_tokens, selected_experts_k).unsqueeze(0).repeat(num_devices, 1, 1)
        )
        tt_expert_scores = ttnn.from_torch(
            expert_scores_replicated,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=scores_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        # --- Weights (BFP4, ring-aware packed) ---
        w0_w1_shard_map, w2_shard_map, dram_core_range_set = get_weight_core_shard_maps(mesh_device, hidden_size, N)
        torch_w0 = create_torch_w0(num_layers, experts_per_device, hidden_size, N)
        torch_w1 = create_torch_w1(num_layers, experts_per_device, hidden_size, N)
        torch_w2 = create_torch_w2(num_layers, experts_per_device, N, hidden_size)

        w0_w1_mem_config, w2_mem_config, _, _ = get_weight_mem_configs(
            num_layers,
            experts_per_device,
            hidden_size,
            N,
            w0_w1_shard_map,
            w2_shard_map,
            dram_core_range_set,
            has_bias=False,
        )

        torch_w0_w1_reordered = prepare_w0_w1_tensor_for_moe_compute(
            torch_w0, torch_w1, num_layers, experts_per_device, hidden_size, N, w0_w1_shard_map
        )
        tt_w0_w1 = ttnn.from_torch(
            torch_w0_w1_reordered,
            dtype=ttnn.bfloat4_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w0_w1_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        torch_w2_reordered = prepare_w2_tensor_for_moe_compute(
            torch_w2, num_layers, experts_per_device, N, hidden_size, w2_shard_map, w0_w1_shard_map
        )
        tt_w2 = ttnn.from_torch(
            torch_w2_reordered,
            dtype=ttnn.bfloat4_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    except Exception as e:  # setup issue -> not the timing signal we care about; log & skip.
        pytest.skip(f"timing setup failed at tokens={tokens_per_device}: {type(e).__name__}: {e}")

    def _call():
        return ttnn.experimental.moe_compute(
            tt_sparse_buffer,
            tt_expert_indices,
            tt_expert_scores,
            tt_expert_mapping,
            tt_w0_w1,
            tt_w2,
            layer_id=layer_id,
            output_height_shard_dim=output_height_shard_dim,
            intermediate_size=N,
            has_bias=False,
            cluster_axis=None,
            topology=None,
            num_links=None,
            mux_core_range_set=None,
            optional_output_tensor=None,
            optional_cross_device_semaphore=None,
            activation_type=QWEN_ACTIVATION,
            compute_only=True,
        )

    ms = _time_traced(mesh_device, _call)
    per_token_ms = ms / tokens_per_device
    logger.info(
        f"[qwen36-probe] TIMING tokens_per_device={tokens_per_device}: "
        f"{ms:.3f} ms/call ({per_token_ms:.4f} ms/token) "
        f"| sparse_matmul baselines: decode ~0.32 ms/layer, batched-dense ~3.0 ms/step (~0.094 ms/token), "
        f"prefill-dense ~3.6 ms/layer"
    )
