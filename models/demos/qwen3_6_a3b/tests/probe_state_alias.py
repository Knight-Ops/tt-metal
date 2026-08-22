# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Can the fused decode kernel's S and Snew be the SAME buffer? (QWEN36_GDN_ALIAS_STATE)

Today `_forward_decode_fused` moves 4 MB per gated-delta layer per token: the kernel reads S (1 MB)
and writes Snew (1 MB), then `ttnn.copy(reshape(snew_buf), state_buf)` reads and writes 1 MB more
each. That copy exists only because S and Snew are separate arguments.

It should not need to. Core h reads only `S[h*kt_t : (h+1)*kt_t]` (ttl_delta.py:501) and writes
exactly that slab (:614); the slabs are disjoint across cores, and within a core `read()` completes
(`t7.wait()`) before `compute()` runs and `write()` fires. So aliasing is safe by construction --
IF ttl / ttnn.generic_op accepts the same tensor as two arguments. That is what this probe settles.

Worth 2 MB/layer/token in bf16 today, and it is what makes an fp32 state DRAM-NEGATIVE versus the
current code (4 MB aliased-fp32 vs 4 MB separate-bf16), so it is a prerequisite for the fp32 work
as well as a standalone win.

Checks, at REAL dims (V=32, Dk=Dv=128) on one real layer:
  1. does it even run (does ttl reject a duplicated argument)?
  2. is the resulting state bit-identical to the separate-buffer path?
  3. what does it cost / save per launch?

    ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_state_alias.py
    ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_state_alias.py --iters 500
Env: QWEN36_CKPT. --random-weights runs without a checkpoint (fine here -- this is an exactness
test, not a drift test, so the weight values do not matter).
"""

from __future__ import annotations

import argparse
import os
import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig, Qwen35MoeGatedDeltaNet
from models.demos.qwen3_6_a3b.tt.common import from_tt
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.ttl_delta import decode_step_tt

TILE = 32


def _weights(ckpt, layer, cfg, random_weights):
    if random_weights:
        ref = Qwen35MoeGatedDeltaNet(cfg, layer_idx=0).eval()
        torch.manual_seed(0)
        for p in ref.parameters():
            p.normal_(0, 0.05)
        ref.A_log.data.uniform_(0, 16).log_()
        ref.dt_bias.data.uniform_(0.5, 1.5)
        return {"A_log": ref.A_log.data, "dt_bias": ref.dt_bias.data, "norm": ref.norm.weight.data}
    w = {k: (v() if callable(v) else v) for k, v in CheckpointLoader(ckpt).gated_delta_weights(layer).items()}
    return {"A_log": w["A_log"].float(), "dt_bias": w["dt_bias"].float(), "norm": w["norm"].float()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--ckpt", default=os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36")))
    args = ap.parse_args()

    cfg = Qwen35MoeConfig() if args.random_weights else Qwen35MoeConfig.from_hf_config(args.ckpt)
    V, Kh = cfg.linear_num_value_heads, cfg.linear_num_key_heads
    Dk, Dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    key_dim, value_dim = Kh * Dk, V * Dv
    w = _weights(args.ckpt, args.layer, cfg, args.random_weights)
    scale, eps = Dk**-0.5, cfg.rms_norm_eps

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:

        def tt(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)

        torch.manual_seed(0)
        q = tt(torch.randn(1, key_dim) * 0.1)
        k = tt(torch.randn(1, key_dim) * 0.1)
        v = tt(torch.randn(1, value_dim) * 0.1)
        z = tt(torch.randn(1, value_dim) * 0.1)
        # per-head uniform tiles, as _forward_decode_fused builds them
        rep = lambda t: tt(t.reshape(V, 1, 1).expand(V, TILE, TILE).reshape(V * TILE, TILE).contiguous())
        araw = rep(torch.randn(V) * 0.5)
        braw = rep(torch.randn(V) * 0.5)
        negA = rep(-w["A_log"].exp())
        dtb = rep(w["dt_bias"])
        nweight = tt(w["norm"].reshape(1, Dv).expand(TILE, Dv).contiguous())

        S0 = torch.randn(V * Dk, Dv) * 0.05  # a non-trivial starting state

        # ---- A: separate buffers (what ships today) ----
        S_a = tt(S0)
        out_a = tt(torch.zeros(1, value_dim))
        snew_a = tt(torch.zeros(V * Dk, Dv))
        decode_step_tt(q, k, v, z, araw, braw, negA, dtb, S_a, nweight, out_a, snew_a, V, Kh, scale, eps)
        ttnn.synchronize_device(dev)
        ref_state = from_tt(snew_a, dev).clone()
        ref_out = from_tt(out_a, dev).clone()
        print("A (separate S/Snew): ran OK")

        # ---- B: aliased -- S and Snew are the same tensor ----
        S_b = tt(S0)
        out_b = tt(torch.zeros(1, value_dim))
        try:
            decode_step_tt(q, k, v, z, araw, braw, negA, dtb, S_b, nweight, out_b, S_b, V, Kh, scale, eps)
            ttnn.synchronize_device(dev)
        except Exception as e:
            print(f"\nB (aliased): REJECTED -> {type(e).__name__}: {e}")
            print("\nVERDICT: NO-GO. ttl/generic_op will not take one tensor as two arguments; keep the")
            print("         separate buffer + ttnn.copy. The fp32 work must budget 8 MB/layer/token.")
            return
        got_state = from_tt(S_b, dev).clone()
        got_out = from_tt(out_b, dev).clone()
        print("B (aliased S==Snew):  ran OK")

        ds = (got_state.float() - ref_state.float()).abs().max().item()
        do = (got_out.float() - ref_out.float()).abs().max().item()
        exact = ds == 0.0 and do == 0.0
        print(f"\nmax|state_aliased - state_separate| = {ds:.3e}   {'(BIT-IDENTICAL)' if ds == 0 else ''}")
        print(f"max|out_aliased   - out_separate|   = {do:.3e}   {'(BIT-IDENTICAL)' if do == 0 else ''}")

        # ---- timing: per-launch, both forms ----
        def bench(alias):
            Sx = tt(S0)
            ox = tt(torch.zeros(1, value_dim))
            sx = Sx if alias else tt(torch.zeros(V * Dk, Dv))
            for _ in range(20):
                decode_step_tt(q, k, v, z, araw, braw, negA, dtb, Sx, nweight, ox, sx, V, Kh, scale, eps)
                if not alias:
                    ttnn.copy(sx, Sx)
            ttnn.synchronize_device(dev)
            t0 = time.time()
            for _ in range(args.iters):
                decode_step_tt(q, k, v, z, araw, braw, negA, dtb, Sx, nweight, ox, sx, V, Kh, scale, eps)
                if not alias:
                    ttnn.copy(sx, Sx)
            ttnn.synchronize_device(dev)
            return (time.time() - t0) / args.iters * 1e6

        us_sep = bench(False)
        us_ali = bench(True)
        print(f"\nper step (kernel + copy), {args.iters} iters, EAGER so host dispatch dominates:")
        print(f"  separate + ttnn.copy : {us_sep:8.1f} us")
        print(
            f"  aliased (no copy)    : {us_ali:8.1f} us   ({us_sep - us_ali:+.1f} us, "
            f"{100 * (us_sep - us_ali) / us_sep:+.1f}%)"
        )
        print("  (eager numbers include ~190 us/op of host dispatch -- the traced saving is the DRAM")
        print("   traffic, 2 of 4 MB per layer per token. Confirm with tests/bench_decode.py.)")

        if exact:
            print("\nVERDICT: GO. Aliasing is accepted and BIT-IDENTICAL. Delete the copy")
            print("         (QWEN36_GDN_ALIAS_STATE=1, now the default) and re-run bench_decode.py.")
        else:
            print("\nVERDICT: NO-GO. Aliasing runs but changes the result -- the read/write overlap is")
            print("         NOT safe as reasoned. Revert to the separate buffer.")
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
