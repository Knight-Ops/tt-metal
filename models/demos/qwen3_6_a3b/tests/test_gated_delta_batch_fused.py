# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validate the BATCHED FUSED GDN decode kernel (_forward_decode_batch_fused / decode_step_batch_tt).

Two checks on a real mixer, B=4:
  (A) batched-fused == 4 independent single-user FUSED runs (QWEN36_GDN_FUSED default on, so B=1 uses
      the shipped fused ttl kernel). Validates the batched grouping + integration (tile-row relayout,
      araw/braw build, head-major state view, w_out) against the shipped single-user kernel.
  (B) batched-fused == batched SCAN (_forward_decode_batch, the golden reference from the handoff) on
      the identical batched input — obtained by monkeypatching the routed method to the scan (same
      signature). This is the direct fused-vs-scan PCC gate.

Each user's recurrence is independent, so all comparisons must be per-user-identical -> PCC ~1.0.

Run: QWEN36_LAYERS=4 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/test_gated_delta_batch_fused.py
"""
import os

os.environ["QWEN36_GDN_BATCH_FUSED"] = "1"  # MUST precede any qwen36 import (read at module load)

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

        def mk_cache(sl):
            return {
                "recurrent_state": tt(rs[sl].clone()),
                "conv_rows": [tt(cr[j][sl].clone()) for j in range(K - 1)],
            }

        # --- reference A: each user independently (B=1 fused ttl kernel) ---
        ref_out, ref_state = [], []
        for i in range(B):
            cache = mk_cache(slice(i, i + 1))
            o = m.forward(tt(xh[:, :, i : i + 1, :].clone()), cache=cache, decode=True)  # -> [1,1,1,dim]
            ref_out.append(ttnn.to_torch(o).reshape(dim))
            ref_state.append(ttnn.to_torch(cache["recurrent_state"]).reshape(Vh, Dk, Dv))

        # --- batched-fused: all B users at once (routes to _forward_decode_batch_fused) ---
        cacheB = mk_cache(slice(None))
        oB = m.forward(tt(xh.clone()), cache=cacheB, decode=True)  # [1,1,B,dim]
        oB_t = ttnn.to_torch(oB).reshape(B, dim)
        stB = ttnn.to_torch(cacheB["recurrent_state"]).reshape(B, Vh, Dk, Dv)

        # --- golden scan on the SAME batched input: route the branch to the scan method ---
        m._forward_decode_batch_fused = m._forward_decode_batch  # same signature; forward now calls scan
        cacheS = mk_cache(slice(None))
        oS = m.forward(tt(xh.clone()), cache=cacheS, decode=True)  # [1,1,B,dim] via scan
        oS_t = ttnn.to_torch(oS).reshape(B, dim)
        stS = ttnn.to_torch(cacheS["recurrent_state"]).reshape(B, Vh, Dk, Dv)

        ok = True
        print(f"\n=== (A) batched-fused B={B} vs {B} independent single-user FUSED runs ===")
        for i in range(B):
            po, ps = pcc(ref_out[i], oB_t[i]), pcc(ref_state[i], stB[i])
            good = po > 0.999 and ps > 0.999
            ok = ok and good
            print(f"  user {i}: out PCC {po:.5f}   state PCC {ps:.5f}   [{'OK' if good else 'FAIL'}]")

        print(f"\n=== (B) batched-fused vs batched SCAN (golden reference), same input ===")
        for i in range(B):
            po, ps = pcc(oS_t[i], oB_t[i]), pcc(stS[i], stB[i])
            good = po > 0.999 and ps > 0.999
            ok = ok and good
            print(f"  user {i}: out PCC {po:.5f}   state PCC {ps:.5f}   [{'OK' if good else 'FAIL'}]")

        print(f"\nRESULT: {'PASS' if ok else 'FAIL'} (batched fused GDN decode kernel matches both references)")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
