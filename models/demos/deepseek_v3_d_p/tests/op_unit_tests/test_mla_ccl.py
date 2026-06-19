# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the all-gather and reduce-scatter CCLs in the chunked-prefill MLA forward
(models/demos/deepseek_v3_d_p/tt/mla/mla.py).

The MLA forward runs four CCLs, all along the TP axis (tp_factor=4):

    | id          | op             | dim | per-device in -> out          | mla.py |
    |-------------|----------------|-----|-------------------------------|--------|
    | q_a_proj_rs | reduce_scatter |  3  | [1,1,S,1536] -> [1,1,S,384]   | :718   |
    | q_ag        | all_gather     |  3  | [1,1,S,384]  -> [1,1,S,1536]  | :729   |
    | kv_ag       | all_gather     |  1  | [1,1,S,576]  -> [1,4,S,576]   | :797   |
    | o_proj_rs   | reduce_scatter |  3  | [1,1,S,7168] -> [1,1,S,1792]  | :912   |

S is the PER-DEVICE sequence length (seq_len_local). On the 8x4 Galaxy with a 5120-token
chunk (sp=8), each chip sees chunk_size_global / sp = 5120 / 8 = 640 tokens. These tests fix
S=640 so that running on a 2x4 mesh reproduces exactly the per-device CCL shapes seen on 8x4
(the CCLs are on the TP axis, which is 4 on both meshes; only SP differs, and SP just adds
independent replicas of the same per-device op).

The op call signatures mirror mla.py exactly: DRAM interleaved, the model's semaphore layout
(all_gather -> [2 sems] + barrier; reduce_scatter -> [3 sems] + barrier, persistent_output_buffers
=None), and ccl_num_links = 2 on Blackhole else 1.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config as Cfg
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# MLA feature dims (kimi_k2_6 == deepseek_v3 for all of these; only num_heads differs and it
# does not enter the CCL widths).
Q_LORA_RANK = Cfg.Q_LORA_RANK  # 1536
KVPE_DIM = Cfg.KV_LORA_RANK + Cfg.QK_ROPE_HEAD_DIM  # 512 + 64 = 576
HIDDEN_SIZE = Cfg.EMB_SIZE  # 7168
TP_FACTOR = 4  # production TP; tp_axis size on both 8x4 and 2x4
SEQ_LOCAL = 640  # per-device seq: chunk_size_global(5120) / sp(8) on the 8x4 Galaxy

# (id, kind, dim, per-device-input feature size on the gathered/scattered dim)
#   rs: feat is the FULL width each device holds; output is feat // tp.
#   ag dim=3: feat is the PER-DEVICE width; output is feat * tp.
#   ag dim=1: feat is the channel width (576); the gather is over the head dim (dim1, 1 -> tp).
MLA_CCL_OPS = [
    ("q_a_proj_rs", "rs", 3, Q_LORA_RANK),  # mla.py:718  all-reduce part 1
    ("q_ag", "ag", 3, Q_LORA_RANK // TP_FACTOR),  # mla.py:729  all-reduce part 2
    ("kv_ag", "ag", 1, KVPE_DIM),  # mla.py:797  gather-then-reduce
    ("o_proj_rs", "rs", 3, HIDDEN_SIZE),  # mla.py:912  output reduce-scatter
]

# Fabric/topology variants, shared by all CCL tests in this file. Each entry is
# (device_params, ttnn.Topology); device_params is consumed by the mesh_device fixture (indirect).
DEVICE_PARAMS_TOPOLOGY = [
    (
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        },
        ttnn.Topology.Linear,
    ),
    (
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        },
        ttnn.Topology.Ring,
    ),
    (
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        },
        ttnn.Topology.Linear,
    ),
]
DEVICE_PARAMS_TOPOLOGY_IDS = ["line", "ring", "fabric2d"]


def _make_global_semaphores(mesh_device, cores, n):
    return [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(n)]


def _run_mla_ccl(mesh_device, kind, dim, feat, topology, num_iters=1, pcc_threshold=0.999):
    tp_axis = 1
    sp, tp = list(mesh_device.shape)
    # Production TP is 4; tp>TP_FACTOR (e.g. 1x8) is allowed for CCL connectivity/perf experiments.
    # The golden uses the live `tp`, so the op stays self-consistent as long as the scattered dim is
    # tile-aligned after the split.
    if kind == "rs":
        assert (feat // tp) % 32 == 0, f"rs scatter width {feat}//{tp} must be tile-aligned"
    num_links = 2 if is_blackhole() else 1

    # --- sub-device + semaphores (same scaffolding the model's TT_CCL uses) ---
    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    barrier_sem = ttnn.create_global_semaphore(mesh_device, ccl_crs, 0)

    # --- inputs: each of the tp devices holds an independent [1,1,S,feat] tensor, laid out on
    #     dim1 so a tp-shard of dim1 hands each device its own slice. SP replicates (independent
    #     copies of the same per-device op). ---
    torch_in = torch.randn(1, tp, SEQ_LOCAL, feat, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        torch_in,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, 1]),
    )

    try:
        for i in range(num_iters):
            logger.info(f"{kind} dim={dim} feat={feat}: iteration {i + 1}/{num_iters}")
            if kind == "ag":
                mdgs = _make_global_semaphores(mesh_device, ccl_crs, 2)
                tt_out = ttnn.experimental.all_gather_async(
                    tt_in,
                    dim=dim,
                    multi_device_global_semaphore=mdgs,
                    barrier_semaphore=barrier_sem,
                    num_links=num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=topology,
                    cluster_axis=tp_axis,
                )
            else:  # "rs"
                mdgs = _make_global_semaphores(mesh_device, ccl_crs, 3)
                tt_out = ttnn.experimental.reduce_scatter_minimal_async(
                    tt_in,
                    persistent_output_buffers=None,
                    dim=dim,
                    multi_device_global_semaphore=mdgs,
                    barrier_semaphore=barrier_sem,
                    num_links=num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=topology,
                    cluster_axis=tp_axis,
                )
            ttnn.synchronize_device(mesh_device)
            out_torch = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(sp, tp), dims=(0, dim)),
            )[
                0:1
            ]  # SP replicas are identical; keep the first

            # --- golden ---
            if kind == "rs":
                # reduce over the tp devices (dim1), then the concat-over-tp readback on `dim`
                # reassembles the full scattered sum.
                golden = torch_in.to(torch.float32).sum(dim=1, keepdim=True)
                out_torch = out_torch.to(torch.float32)
            elif dim == 3:
                # all_gather along width: every device ends up with the full concat; tp copies are
                # identical, so take the first one back out of the concat readback.
                golden = torch.cat([torch_in[:, d : d + 1] for d in range(tp)], dim=3)
                out_torch = out_torch[:, :, :, : feat * tp]
            else:  # ag dim == 1 (head gather): output is the tp slices stacked on dim1
                golden = torch_in
                out_torch = out_torch[:, :tp]

            logger.info(f"{kind} dim={dim} feat={feat}: in {list(torch_in.shape)} -> out {list(out_torch.shape)}")
            passed, msg = comp_pcc(out_torch, golden, pcc_threshold)
            logger.info(f"PCC: {msg}")
            assert passed, f"{kind} dim={dim} feat={feat} iter {i + 1}/{num_iters} FAILED: {msg}"
    finally:
        mesh_device.reset_sub_device_stall_group()


@pytest.mark.parametrize("ccl_id, kind, dim, feat", MLA_CCL_OPS, ids=[c[0] for c in MLA_CCL_OPS])
@pytest.mark.parametrize(
    "device_params, topology", DEVICE_PARAMS_TOPOLOGY, indirect=["device_params"], ids=DEVICE_PARAMS_TOPOLOGY_IDS
)
@pytest.mark.parametrize(
    "mesh_device", [(1, 4), (1, 8), (2, 4), (8, 4)], ids=["1x4", "1x8", "2x4", "8x4"], indirect=True
)
@pytest.mark.parametrize("num_iters", [1], ids=lambda n: f"iters{n}")
@pytest.mark.timeout(0)
def test_mla_ccl(mesh_device, device_params, topology, ccl_id, kind, dim, feat, num_iters):
    """Each chunked-MLA all-gather / reduce-scatter at its per-device shape (seq_local=640),
    reproducing the 8x4 per-device load on a 2x4 mesh. Runs the op `num_iters` times in a loop."""
    _run_mla_ccl(mesh_device, kind, dim, feat, topology, num_iters=num_iters)


# ---------------------------------------------------------------------------
# Ring-attention all-gather CCL (the gather inside ring_mla / ring_joint_sdpa)
# ---------------------------------------------------------------------------
# ring_mla (mla.py:627, chunked prefill) and ring_joint_scaled_dot_product_attention (mla.py:857,
# single-shot) internally ring-all-gather the K/V over the sequence dim across the SP axis (the
# ring-parallel axis; sp_axis=0, tp_axis=1 in mla.py), landing the full prefix in a persistent output
# buffer on every ring device. ring_mla gathers the single combined latent KV (kvpe, width
# KV_LORA_RANK + QK_ROPE_HEAD_DIM = 576; mla.py TT_CCL.get_mla_chunked_kv_buffer), replicated over TP.
#
# This exercises that gather in isolation via ttnn.experimental.ring_attention_all_gather_async
# (same op the ring attention kernels use; see tests/nightly/t3000/ccl/test_ring_attention_all_gather.py)
# with the model's sub-device / semaphore scaffolding and per-device shape (seq_local=640). The gather
# is on the SP axis, so unlike test_mla_ccl the ring length scales with the mesh's SP dim (1/1/2/8 on
# 1x4 / 1x8 / 2x4 / 8x4) rather than being fixed at the production value.
RING_SP_AXIS = 0  # mla.py sp_axis: the ring-parallel axis the K/V gather runs on
RING_SEQUENCE_INDEX = 2  # gather dim (sequence)
RING_HEAD_INDEX = 1


def _run_mla_ring_attention_ag(
    mesh_device, topology, num_iters=1, kvpe_dim=KVPE_DIM, dtype=ttnn.bfloat8_b, pcc_threshold=0.999
):
    sp, tp = list(mesh_device.shape)
    rp_axis = RING_SP_AXIS
    num_links = 2 if is_blackhole() else 1

    # Per-device gather slice stays SEQ_LOCAL (the 8x4 chunk/sp load); the full gathered sequence is
    # SEQ_LOCAL * sp. ring_mla's gathered KV is [1, 1, seq, kvpe_dim], replicated over TP (head=1).
    seq_global = SEQ_LOCAL * sp
    seq_per_device = SEQ_LOCAL
    ag_output_shape = [1, 1, seq_global, kvpe_dim]

    # --- sub-device + semaphores (same scaffolding the model's TT_CCL uses) ---
    grid = mesh_device.compute_with_storage_grid_size()
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    worker_sub_device = ttnn.SubDevice([ccl_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    # --- input: full (gathered) KV; shard the sequence over the SP/ring axis, replicate over TP. ---
    torch_kv = torch.rand(ag_output_shape).bfloat16()
    input_dims = [None, None]
    input_dims[rp_axis] = RING_SEQUENCE_INDEX  # shard sequence across the ring; TP replicated
    tt_in = ttnn.from_torch(
        torch_kv,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=input_dims),
    )

    output_dims = [None, None]
    output_dims[rp_axis] = RING_SEQUENCE_INDEX  # concat the ring shards back along the sequence
    output_dims[1 - rp_axis] = RING_HEAD_INDEX  # TP replicas land on the (size-1) head dim

    try:
        for it in range(num_iters):
            logger.info(f"ring_mla AG kvpe={kvpe_dim} seq={seq_global}: iteration {it + 1}/{num_iters}")
            # persistent output buffer holds the FULL gather on every device (replicated over the mesh).
            persistent_out = ttnn.from_torch(
                torch.zeros(ag_output_shape),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[None, None]),
            )
            mdgs = _make_global_semaphores(mesh_device, ccl_crs, 2)
            tt_out = ttnn.experimental.ring_attention_all_gather_async(
                [tt_in],
                persistent_output_buffer=[persistent_out],
                dim=RING_SEQUENCE_INDEX,
                multi_device_global_semaphore=mdgs,
                cluster_axis=rp_axis,
                mesh_device=mesh_device,
                num_links=num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
                subdevice_id=worker_sub_device_id,
            )
            ttnn.synchronize_device(mesh_device)

            # readback: concat the ring (SP) shards along the sequence, collapse TP replicas (head=1).
            tt_ag_out = ttnn.to_torch(
                tt_out[0],
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(sp, tp), dims=output_dims),
            )[:, :1]
            ring_chunks = torch.chunk(tt_ag_out, sp, dim=RING_SEQUENCE_INDEX)
            for ring_idx, tt_ring in enumerate(ring_chunks):
                # AG does not write a device's own local slice, so zero that window on both sides
                # before comparing the rest.
                tt_check = tt_ring.clone()
                torch.narrow(tt_check, RING_SEQUENCE_INDEX, ring_idx * seq_per_device, seq_per_device).zero_()
                golden = torch_kv.clone()
                torch.narrow(golden, RING_SEQUENCE_INDEX, ring_idx * seq_per_device, seq_per_device).zero_()

                passed, msg = comp_pcc(tt_check, golden, pcc_threshold)
                logger.info(f"ring AG iter {it + 1} ring {ring_idx}: PCC {msg}")
                assert passed, f"ring_mla AG iter {it + 1}/{num_iters} ring {ring_idx} FAILED: {msg}"
    finally:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize(
    "device_params, topology", DEVICE_PARAMS_TOPOLOGY, indirect=["device_params"], ids=DEVICE_PARAMS_TOPOLOGY_IDS
)
@pytest.mark.parametrize(
    "mesh_device", [(1, 4), (1, 8), (2, 4), (8, 4)], ids=["1x4", "1x8", "2x4", "8x4"], indirect=True
)
@pytest.mark.parametrize("num_iters", [1], ids=lambda n: f"iters{n}")
@pytest.mark.timeout(0)
def test_mla_ring_attention_ccl(mesh_device, device_params, topology, num_iters):
    """Ring all-gather CCL inside chunked-prefill ring_mla (mla.py:627), in isolation, at the
    per-device kvpe shape (seq_local=640). Gather runs on the SP axis. Runs the op `num_iters` times
    in a loop."""
    _run_mla_ring_attention_ag(mesh_device, topology, num_iters=num_iters)
