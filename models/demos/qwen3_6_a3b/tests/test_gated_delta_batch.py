# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validate batched multi-user GDN decode (_forward_decode_batch) == B independent single-user runs.

Batched decode advances B independent users one token each in a single call, with a B axis on the
recurrent state [B,Vh,Dk,Dv] and per-user conv history rows [B,conv_dim]. Each user's recurrence is
independent, so batched must be per-user-identical to running each user alone. We force the batched
matmul path at B=1 too (QWEN36_GDN_FUSED=0) so the reference and the batched run share code and the only
difference is the batch grouping -> expect PCC ~1.0.

Run: QWEN36_LAYERS=4 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/test_gated_delta_batch.py
"""
import os

os.environ["QWEN36_GDN_FUSED"] = "0"  # MUST precede any qwen36 import (read at module load)
os.environ["QWEN36_GDN_BATCH_FUSED"] = "0"  # this test validates the SCAN batching path specifically

import torch

import ttnn
from models.demos.qwen3_6_a3b.tests.bench_decode import build_model


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    if a.numel() != b.numel():
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
        n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
        model, args = build_model(mesh, n_layers, ckpt, int(os.environ.get("QWEN36_MAX_SEQ", "128")), seq=8)
        m = next(l for l in model.layers if l.is_linear).mixer
        Vh, Kh, Dk, Dv, cd, K = m.num_v_heads, m.num_k_heads, m.head_k_dim, m.head_v_dim, m.conv_dim, m.conv_k
        dim = args.dim
        B = 4
        torch.manual_seed(0)
        xh = torch.randn(1, 1, B, dim) * 0.5  # per-user decode hidden
        rs = torch.randn(B, Vh, Dk, Dv) * 0.1  # per-user initial recurrent state
        cr = [torch.randn(B, cd) * 0.1 for _ in range(K - 1)]  # per-user conv history rows

        def tt(t):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)

        # --- reference: each user independently (B=1, same batched-matmul code path) ---
        ref_out, ref_state = [], []
        for i in range(B):
            cache = {
                "recurrent_state": tt(rs[i : i + 1].clone()),
                "conv_rows": [tt(cr[j][i : i + 1].clone()) for j in range(K - 1)],
            }
            x = tt(xh[:, :, i : i + 1, :].clone())  # [1,1,1,dim]
            o = m.forward(x, cache=cache, decode=True)  # -> [1,1,1,dim]
            ref_out.append(ttnn.to_torch(o).reshape(dim))
            ref_state.append(ttnn.to_torch(cache["recurrent_state"]).reshape(Vh, Dk, Dv))

        # --- batched: all B users at once ---
        cacheB = {
            "recurrent_state": tt(rs.clone()),  # [B,Vh,Dk,Dv]
            "conv_rows": [tt(cr[j].clone()) for j in range(K - 1)],  # [B,conv_dim]
        }
        oB = m.forward(tt(xh.clone()), cache=cacheB, decode=True)  # [1,1,B,dim]
        oB_t = ttnn.to_torch(oB).reshape(B, dim)
        stB = ttnn.to_torch(cacheB["recurrent_state"]).reshape(B, Vh, Dk, Dv)

        print(f"\n=== batched GDN decode B={B} vs {B} independent single-user runs ===")
        ok = True
        for i in range(B):
            po, ps = pcc(ref_out[i], oB_t[i]), pcc(ref_state[i], stB[i])
            flag = "OK" if (po > 0.999 and ps > 0.999) else "FAIL"
            ok = ok and po > 0.999 and ps > 0.999
            print(f"  user {i}: out PCC {po:.5f}   state PCC {ps:.5f}   [{flag}]")
        print(f"\nRESULT: {'PASS' if ok else 'FAIL'} (batched GDN decode is per-user-identical)")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
