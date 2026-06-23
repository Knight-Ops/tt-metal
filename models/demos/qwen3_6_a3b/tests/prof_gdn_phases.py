# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-synced phase timing of the Gated-DeltaNet DECODE step (reliable; profiler-independent).

Replicates the mixer's decode forward (tt/gated_delta.py forward, T=1) but with a
ttnn.synchronize_device + perf_counter checkpoint between each sub-phase, so each phase's wall-clock
(device compute + dispatch) is isolated. Use it to size the removable cost before refactoring.

Run: QWEN36_LAYERS=4 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/prof_gdn_phases.py
"""
import os
import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt import gated_delta as gd
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs


@torch.no_grad()
def main():
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
    iters = int(os.environ.get("QWEN36_ITERS", "100"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=128)
        loader = CheckpointLoader(ckpt)
        model = TtModel(mesh, args, loader, num_layers=n_layers)
        torch.manual_seed(0)
        prompt = torch.randint(0, args.vocab_size, (1, 32))
        logits = model.forward(prompt)  # prefill: allocates per-layer caches
        model.start_decode(int(logits[0, -1].argmax()))
        model.decode_step_eager()  # warmup: compile decode kernels + steady-state caches
        ttnn.synchronize_device(mesh)

        lin_idx = next(i for i, l in enumerate(model.layers) if l.is_linear)
        m = model.layers[lin_idx].mixer
        cache = model.caches[lin_idx]
        x = ttnn.from_torch(
            torch.randn(1, 1, 1, args.dim) * 0.5,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        m.forward(x, cache)  # populate caches + compile
        ttnn.synchronize_device(mesh)

        Vh, Dk, Dv = m.num_v_heads, m.head_k_dim, m.head_v_dim
        conv_state = cache.get("conv_state")
        state0 = cache.get("recurrent_state")
        phases = [
            "qkv_proj",
            "conv_silu",
            "prep_slices_proj",
            "prep_qk_l2norm",
            "prep_repeat_interleave",
            "recurrence",
            "out_proj",
        ]
        acc = {p: 0.0 for p in phases}

        def ck(mesh, t0):
            ttnn.synchronize_device(mesh)
            return time.perf_counter(), time.perf_counter() - t0

        for _ in range(iters):
            x2 = ttnn.reshape(x, [1, args.dim])
            ttnn.synchronize_device(mesh)
            t = time.perf_counter()

            mixed = ttnn.linear(x2, m.w_qkv)
            t, dt = ck(mesh, t)
            acc["qkv_proj"] += dt

            mixed_c, _ = m._conv_silu(mixed, conv_state)
            t, dt = ck(mesh, t)
            acc["conv_silu"] += dt

            q = ttnn.slice(mixed_c, [0, 0], [1, m.key_dim])
            k = ttnn.slice(mixed_c, [0, m.key_dim], [1, 2 * m.key_dim])
            v = ttnn.slice(mixed_c, [0, 2 * m.key_dim], [1, m.conv_dim])
            z = ttnn.linear(x2, m.w_z)
            ba = ttnn.linear(x2, m.w_ba)
            beta = ttnn.sigmoid(ttnn.slice(ba, [0, 0], [1, Vh]))
            a = ttnn.slice(ba, [0, Vh], [1, 2 * Vh])
            g_exp = ttnn.exp(ttnn.multiply(m.neg_expA, ttnn.softplus(ttnn.add(a, m.dt_bias))))
            v = ttnn.reshape(v, [1, Vh, Dv])
            t, dt = ck(mesh, t)
            acc["prep_slices_proj"] += dt

            q = gd._l2norm_scale_lastdim(ttnn.reshape(q, [1, m.num_k_heads, Dk]), scale=m.qk_scale)
            k = gd._l2norm_scale_lastdim(ttnn.reshape(k, [1, m.num_k_heads, Dk]))
            t, dt = ck(mesh, t)
            acc["prep_qk_l2norm"] += dt

            if m.n_rep > 1:
                q = ttnn.repeat_interleave(q, m.n_rep, dim=1)
                k = ttnn.repeat_interleave(k, m.n_rep, dim=1)
            t, dt = ck(mesh, t)
            acc["prep_repeat_interleave"] += dt

            q_row = ttnn.reshape(ttnn.slice(q, [0, 0, 0], [1, Vh, Dk]), [1, Vh, 1, Dk])
            k_row = ttnn.reshape(ttnn.slice(k, [0, 0, 0], [1, Vh, Dk]), [1, Vh, 1, Dk])
            k_col = ttnn.reshape(k_row, [1, Vh, Dk, 1])
            v_row = ttnn.reshape(ttnn.slice(v, [0, 0, 0], [1, Vh, Dv]), [1, Vh, 1, Dv])
            g_t = ttnn.reshape(g_exp, [1, Vh, 1, 1])
            b_t = ttnn.reshape(beta, [1, Vh, 1, 1])
            state = ttnn.multiply(state0, g_t)
            kv_mem = ttnn.matmul(k_row, state)
            delta = ttnn.multiply(ttnn.subtract(v_row, kv_mem), b_t)
            outer = ttnn.matmul(k_col, delta)
            state = ttnn.add(state, outer)
            out_t = ttnn.matmul(q_row, state)
            t, dt = ck(mesh, t)
            acc["recurrence"] += dt

            core = ttnn.transpose(out_t, 1, 2)
            core = ttnn.reshape(core, [1, 1, Vh, Dv])
            z_r = ttnn.reshape(z, [1, 1, Vh, Dv])
            core = m.norm.forward(core, z_r)
            core = ttnn.reshape(core, [1, m.value_dim])
            y = ttnn.linear(core, m.w_out)
            t, dt = ck(mesh, t)
            acc["out_proj"] += dt

        tot = sum(acc.values()) / iters * 1000
        print(f"\n[gdn phase timing] {iters} eager iters, conv_dim={m.conv_dim}, Vh={Vh}, Dk={Dk}, Dv={Dv}")
        print(f"  {'phase':12s} {'ms/step':>9s} {'%':>6s}")
        for p in phases:
            ms = acc[p] / iters * 1000
            print(f"  {p:12s} {ms:9.3f} {100*ms/tot:5.1f}%")
        print(f"  {'TOTAL':12s} {tot:9.3f}  (note: per-phase syncs inflate vs the 0.77 ms traced step)")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
