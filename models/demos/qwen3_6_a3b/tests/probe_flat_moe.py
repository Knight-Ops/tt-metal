# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prefill MoE reformulation: BATCHED per-expert matmul vs a FLAT 2D matmul over a merged
`expert x channel` axis. Correctness (PCC vs the batched form) and traced ms/call, at real dims.

THE IDEA. `_dense_experts` computes `[E, tc, H] @ [E, H, 2I]`, which forces an E-fold broadcast of the
activation (`ttnn.repeat`, 143 MB at tc=256) and an E-fold output that then has to be reduced
(`ttnn.sum`, another 143 MB read). Merge the expert axis into the channel axis and both disappear:

    batched:  xe = repeat(x, E);  hgu = xe @ Wgu[E,H,2I];  gate,up = slice;  h = silu(g)*u*w
              ye = h @ Wdn[E,I,H];  y = sum(ye, dim=0)
    flat:     hgu = x[tc,H] @ Wgu[H, 2*E*I]     <- in0 is multicast, not materialised E times
              gate,up = two CONTIGUOUS slices;  h = silu(g)*u*w
              y   = h[tc,E*I] @ Wdn[E*I, H]     <- the sum over E IS this matmul's K reduction

Same arithmetic, ~2.3x fewer bytes (`ROOFLINE.md` §3), and ~18 ops per chunk collapse to ~5.
`Wdn` needs NO relayout: `down_sp` is already `[E, I, H]` and `[E*I, H]` is the same tile order, so the
reshape is a free view -- asserted here by PCC rather than assumed. `Wgu` needs a one-time host
permute, and for `minimal_matmul(fuse_swiglu=True)` a tile-pair-interleaved `[gate|up]` order on top
(`models.tt_dit.utils.tensor.prepare_for_fused_swiglu`), which also folds the two slices and the
silu+multiply into the matmul.

Run:  ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_flat_moe.py --tcs 256,512
"""
from __future__ import annotations

import argparse
import gc
import time

import torch

import ttnn

E, H, I = 256, 2048, 512
LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
)


def time_traced(mesh, fn, iters=20, warmup=2):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.time() - t0) / iters * 1e3
    ttnn.release_trace(mesh, tid)
    return dt


def largest_divisor(n, cap):
    return max(d for d in range(1, cap + 1) if n % d == 0)


def dense_pc(grid, m, n, bw=8):
    mt, nt = (m + 31) // 32, (n + 31) // 32
    sw = largest_divisor(nt, 4)
    sh = largest_divisor(mt, max(1, 8 // sw))
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        in0_block_w=bw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=mt,
        per_core_N=nt,
    )


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcs", default="256,512")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    tcs = [int(t) for t in a.tcs.split(",")]

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        g = mesh.compute_with_storage_grid_size()
        grid = (g.x, g.y)
        print(f"grid {grid}  E={E} H={H} I={I}  E*I={E*I}\n")
        torch.manual_seed(0)

        # ---- weights: build once in bf16, upload, free the host copy immediately ----
        t_gu = torch.randn(E, H, 2 * I, dtype=torch.bfloat16) * 0.02  # [E, H, gate(I)|up(I)]
        t_dn = torch.randn(E, I, H, dtype=torch.bfloat16) * 0.02  # [E, I, H]
        W_gu_b = ttnn.from_torch(t_gu, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh)
        W_dn_b = ttnn.from_torch(t_dn, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh)
        # flat gate_up: [H, gate(E*I) | up(E*I)]
        flat = torch.cat(
            [t_gu[:, :, :I].permute(1, 0, 2).reshape(H, E * I), t_gu[:, :, I:].permute(1, 0, 2).reshape(H, E * I)],
            dim=-1,
        )
        W_gu_f = ttnn.from_torch(flat, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh)
        from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

        W_gu_s = ttnn.from_torch(
            prepare_for_fused_swiglu(flat, ndev=1, gate_is_first=True),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        del flat, t_gu, t_dn
        gc.collect()
        # the claim under test: [E, I, H] -> [E*I, H] is a free view in TILE layout
        W_dn_f = ttnn.reshape(W_dn_b, [E * I, H])

        for tc in tcs:
            print(f"--- tc = {tc} ---")
            x2 = ttnn.from_torch(
                torch.randn(tc, H, dtype=torch.bfloat16) * 0.5,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
            )
            rt = torch.zeros(tc, E, dtype=torch.bfloat16)
            for r in range(tc):  # a realistic top-8 routing
                rt[r, torch.randperm(E)[:8]] = 0.125
            routing = ttnn.from_torch(rt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            wc = ttnn.reshape(ttnn.transpose(routing, 0, 1), [E, tc, 1])  # batched form
            rflat = ttnn.repeat_interleave(routing, I, dim=1)  # flat form [tc, E*I]
            rflat = ttnn.typecast(rflat, ttnn.bfloat8_b)

            gu_b = ttnn.reshape(W_gu_b, [E, H, 2 * I])
            dn_b = ttnn.reshape(W_dn_b, [E, I, H])

            def batched():
                xe = ttnn.repeat(ttnn.reshape(x2, [1, tc, H]), ttnn.Shape([E, 1, 1]))
                hgu = ttnn.matmul(xe, gu_b, program_config=dense_pc(grid, tc, 2 * I), compute_kernel_config=LOFI)
                gate = ttnn.slice(hgu, [0, 0, 0], [E, tc, I])
                up = ttnn.slice(hgu, [0, 0, I], [E, tc, 2 * I])
                h = ttnn.multiply(ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), wc)
                ye = ttnn.matmul(h, dn_b, program_config=dense_pc(grid, tc, H), compute_kernel_config=LOFI)
                return ttnn.sum(ye, dim=0)

            def flat():
                hgu = ttnn.matmul(x2, W_gu_f, compute_kernel_config=LOFI)
                gate = ttnn.slice(hgu, [0, 0], [tc, E * I])
                up = ttnn.slice(hgu, [0, E * I], [tc, 2 * E * I])
                h = ttnn.multiply(ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), rflat)
                return ttnn.matmul(h, W_dn_f, compute_kernel_config=LOFI)

            def flat_mm():
                h = ttnn.experimental.minimal_matmul(x2, W_gu_s, fuse_swiglu=True, compute_kernel_config=LOFI)
                h = ttnn.multiply(h, rflat)
                return ttnn.experimental.minimal_matmul(h, W_dn_f, compute_kernel_config=LOFI)

            def hybrid():
                """Batched gate_up (NO new weight -- [1,E,H,2I] is a free view of the sparse decode
                layout) + FLAT down (also a free view: [E,I,H] -> [E*I,H]). Costs one permute of `h`
                and removes the [E,tc,H] `ye` write, its read, and the sum over E.

                Why this variant exists: the flat gate_up needs an E-major -> H-major retile, which is
                NOT a view, so it is a SECOND 302 MB/layer copy = 12 GB across 40 layers -- and only
                ~9.6 GB is free. This form needs no extra weight memory at all. It also targets the
                right constraint: `_dense_experts`' own docstring says it is the DOWN projection
                (per_core_N = Nt = 64 at N=hidden) that overflows L1 past tc~320."""
                xe = ttnn.repeat(ttnn.reshape(x2, [1, tc, H]), ttnn.Shape([E, 1, 1]))
                hgu = ttnn.matmul(xe, gu_b, program_config=dense_pc(grid, tc, 2 * I), compute_kernel_config=LOFI)
                gate = ttnn.slice(hgu, [0, 0, 0], [E, tc, I])
                up = ttnn.slice(hgu, [0, 0, I], [E, tc, 2 * I])
                h = ttnn.multiply(ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), wc)
                hf = ttnn.reshape(ttnn.permute(h, (1, 0, 2)), [tc, E * I])  # [E,tc,I] -> [tc, E*I]
                return ttnn.matmul(hf, W_dn_f, compute_kernel_config=LOFI)

            def batched_nsplit(ns=4):
                """Current batched form, but every expert matmul split into `ns` column chunks.

                `MatmulMultiCoreReuseProgramConfig` hard-requires per_core_N == Nt, so each core holds
                the FULL [tc, N] output block and L1 use grows with tc -- that, not bytes, is what pins
                _DENSE_TMAX at 256. Splitting N cuts the block by `ns` without touching the weight
                LAYOUT (unlike the flat form, whose E-major -> H-major retile costs a second 302 MB/layer
                copy = 12 GB across the model, against ~9.6 GB free). The weight halves are sliced per
                call here so the measurement includes that cost; a real implementation would slice once
                at load (memory-neutral: the chunks partition the weight)."""
                xe = ttnn.repeat(ttnn.reshape(x2, [1, tc, H]), ttnn.Shape([E, 1, 1]))
                hs = []
                for j in range(ns):
                    w = ttnn.slice(gu_b, [0, 0, j * 2 * I // ns], [E, H, (j + 1) * 2 * I // ns])
                    hs.append(
                        ttnn.matmul(xe, w, program_config=dense_pc(grid, tc, 2 * I // ns), compute_kernel_config=LOFI)
                    )
                hgu = hs[0] if ns == 1 else ttnn.concat(hs, dim=-1)
                gate = ttnn.slice(hgu, [0, 0, 0], [E, tc, I])
                up = ttnn.slice(hgu, [0, 0, I], [E, tc, 2 * I])
                h = ttnn.multiply(ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), wc)
                ys = []
                for j in range(ns):
                    w = ttnn.slice(dn_b, [0, 0, j * H // ns], [E, I, (j + 1) * H // ns])
                    ys.append(ttnn.matmul(h, w, program_config=dense_pc(grid, tc, H // ns), compute_kernel_config=LOFI))
                ye = ys[0] if ns == 1 else ttnn.concat(ys, dim=-1)
                return ttnn.sum(ye, dim=0)

            ref = None
            for name, fn in (
                ("batched (current)", batched),
                ("batched Nsplit=4", batched_nsplit),
                ("flat", flat),
                ("flat+minimal_mm", flat_mm),
            ):
                try:
                    out = fn()
                    y = ttnn.to_torch(out).reshape(tc, H).float()
                    if ref is None:
                        ref, p = y, 1.0
                    else:
                        p = pcc(ref, y)
                    ms = time_traced(mesh, fn, iters=a.iters)
                    print(f"    {name:20s} {ms:8.3f} ms   PCC vs batched {p:.6f}")
                except Exception as e:  # noqa: BLE001
                    print(f"    {name:20s} FAILED  {type(e).__name__}: {str(e)[:100]}")
            print()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
