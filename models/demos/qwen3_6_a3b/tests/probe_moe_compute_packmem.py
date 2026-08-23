# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""How much HOST RAM does the moe_compute weight packer need at Qwen dims, as a function of E?

`tests/bench_moe_ab.py --legs moe_compute` is OOM-killed (rc=137) on a 15 GB box before it runs a
single op, even with no TtMoE in the process. That blocks the FullLocal A/B, which is the gating
experiment for 40% of the decode step (see FUTURE_OPTIMIZATIONS "Lever 2b"). rc=137 is silent -- the
log just stops -- so this probe exists to turn it into a number: pack at increasing E and report peak
RSS, so the requirement can be extrapolated and the growth term identified.

Run:  ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_moe_compute_packmem.py
"""
import resource

import ttnn

H, N = 2048, 512  # Qwen3.6 hidden / moe_intermediate_size


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0


def main():
    from ttnn.experimental.moe_compute_utils import (
        effective_matmul_ring_size,
        get_weight_core_shard_maps,
        prepare_w0_w1_tensor_for_moe_compute,
    )

    from tests.nightly.tg.ccl.moe.test_moe_compute_6U import create_torch_w0, create_torch_w1

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        ring_n = effective_matmul_ring_size(mesh)
        w0_w1_shard_map, _, _ = get_weight_core_shard_maps(mesh, H, N)
        print(f"H={H} N={N}  matmul ring size {ring_n}   baseline peak RSS {rss_gb():.2f} GB\n")
        print(f"{'E':>5}{'w0+w1 GB':>11}{'packed GB':>11}{'peak RSS GB':>13}{'GB per expert':>15}")
        for E in (8, 16, 32, 64):
            w0 = create_torch_w0(1, E, H, N)
            w1 = create_torch_w1(1, E, H, N)
            raw = (w0.numel() + w1.numel()) * w0.element_size() / 2**30
            packed = prepare_w0_w1_tensor_for_moe_compute(w0, w1, 1, E, H, N, w0_w1_shard_map)
            pk = packed.numel() * packed.element_size() / 2**30
            peak = rss_gb()
            print(f"{E:>5}{raw:>11.3f}{pk:>11.3f}{peak:>13.2f}{peak / E:>15.3f}")
            del w0, w1, packed
        print("\nExtrapolating the per-expert slope to E=256 gives the requirement for the real model.")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
