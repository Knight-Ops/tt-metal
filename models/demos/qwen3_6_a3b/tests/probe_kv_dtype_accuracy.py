# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What does a BFP8 KV cache cost in accuracy? Paired, teacher-forced, one model build.

WHY NOT MMLU. `evaluation/run_mmlu_bench.py` cannot see this at all. Single-shot prefill runs SDPA on
the LIVE q/k/v tensors and only FILLS the cache (`attention.py:forward_prefill`), so a prefill-scored
eval never READS it. Any KV-precision gate has to decode.

WHY IT COULD PLAUSIBLY HURT, unlike a weight dtype. The cache is an ACCUMULATOR THAT IS READ MANY
TIMES: a token written at position p is re-read by every subsequent step, so a rounding error committed
once is then attended over for the rest of the generation, at every layer. That is a different risk
shape from quantising a weight, which is re-read identically each time.

DESIGN. One model build, two arms, same steps: prefill a real prompt, then teacher-force the SAME token
stream through both a bf16 and a bf8 cache and compare per-step logits. The caches are reallocated
in-process (`_alloc_cache` reads the dtype at call time), so nothing else differs -- same weights, same
prompt, same tokens. Reports per-step logit PCC, argmax agreement, and the top1-top2 margin at any
disagreement, plus whether error ACCUMULATES over the run (the failure mode that matters here).

RESULT (2026-08-23, 40 layers, prefill 1024, 48 steps): logit PCC mean 0.995024 / median 0.996008 /
min 0.977164, STABLE (median 1st half 0.995533 -> 2nd half 0.996215, i.e. no accumulation -- the risk
above did not materialise), argmax agreement 44/48 with all four flips on near-ties (mean top1-top2
margin 0.2344 against an all-step 3.4941). On par with the shipped conv add-chain's own 0.996 gate.
See EXPERIMENTS.md "BFP8 KV cache" and FUTURE_OPTIMIZATIONS.md "Lever 6" for the perf side.

ONE PREREQUISITE THIS PROBE UNCOVERED. The cache-write ops have OPPOSITE dtype contracts, so a
quantized cache needs `Attention._to_cache_dtype` on the fill path and must NOT have it on the decode
path: `ttnn.fill_cache` requires `input.dtype == cache.dtype`, while `paged_update_cache` requires a
fp32/bf16 INPUT and converts on the way in. Without the former this probe fatals in its second arm.

Run:  QWEN36_LAYERS=40 ./python_env/bin/python \
          models/demos/qwen3_6_a3b/tests/probe_kv_dtype_accuracy.py --prefill 1024 --steps 48
"""
from __future__ import annotations

import argparse
import os
import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs

PROMPT = "The Tenstorrent Blackhole architecture is interesting because"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run_arm(model, ids, steps, kvdt, forced=None):
    """Reallocate the KV caches at `kvdt`, prefill, then run `steps` decode steps. If `forced` is
    given, teacher-force those tokens instead of the argmax. Returns (tokens, [logits...])."""
    model.args.kv_cache_dtype = kvdt
    model.max_seq = model.args.max_seq_len  # normally set by _prefill_single; we allocate before it
    model.kv_pager = None  # so a paged run rebuilds its pool at the new dtype too
    model.caches = [model._alloc_cache(layer) for layer in model.layers]
    model._pf_caches_ready = True

    logits0 = model.forward(ids)
    first = int(logits0[0, -1].argmax())
    model.start_decode(first)
    toks, lgs = [], []
    for i in range(steps):
        lg = torch.as_tensor(model.decode_forward_logits()).reshape(-1).float()
        t = int(forced[i]) if forced is not None else int(lg.argmax())
        lgs.append(lg)
        toks.append(t)
        model.set_decode_tokens(t)
    return toks, lgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", type=int, default=1024, help="prompt length: more context = more cache reads")
    ap.add_argument("--steps", type=int, default=48)
    a = ap.parse_args()

    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=max(4096, a.prefill + a.steps + 64))
        t0 = time.time()
        model = TtModel(mesh, args, CheckpointLoader(ckpt), num_layers=n_layers)
        print(
            f"[kvdt] {n_layers}-layer model built in {time.time()-t0:.0f}s; "
            f"prefill {a.prefill} tokens, {a.steps} teacher-forced steps",
            flush=True,
        )
        torch.manual_seed(0)
        ids = torch.randint(0, args.vocab_size, (1, a.prefill))

        base_toks, base_lgs = run_arm(model, ids, a.steps, ttnn.bfloat16)
        print(f"[kvdt] bf16 arm done ({len(base_toks)} steps)", flush=True)
        _, bf8_lgs = run_arm(model, ids, a.steps, ttnn.bfloat8_b, forced=base_toks)

        ps, same, flips = [], 0, []
        for i, (b, q) in enumerate(zip(base_lgs, bf8_lgs)):
            ps.append(pcc(b, q))
            if int(b.argmax()) == int(q.argmax()):
                same += 1
            else:
                t2 = b.topk(2).values
                flips.append((i, float(t2[0] - t2[1])))
        p = torch.tensor(ps)
        h1, h2 = p[: len(p) // 2].median().item(), p[len(p) // 2 :].median().item()
        allm = torch.tensor([float(b.topk(2).values[0] - b.topk(2).values[1]) for b in base_lgs])
        print(f"\n[kvdt] logit PCC  mean {p.mean():.6f}  median {p.median():.6f}  min {p.min():.6f}")
        print(f"[kvdt] median 1st half {h1:.6f} -> 2nd half {h2:.6f}  (delta {h2 - h1:+.2e})")
        print(
            f"[kvdt]   -> {'ACCUMULATING (the failure mode that matters for a cache)' if h2 - h1 < -1e-3 else 'STABLE: no accumulation over the run'}"
        )
        print(f"[kvdt] argmax agreement {same}/{len(ps)}")
        if flips:
            fm = torch.tensor([g for _, g in flips])
            print(f"[kvdt] {len(flips)} flips at steps {[i for i, _ in flips]}")
            print(
                f"[kvdt]   their top1-top2 margins: mean {fm.mean():.4f} vs all-step mean {allm.mean():.4f}"
                f"  -> {'near-ties' if fm.mean() < allm.mean() else 'NOT tie-driven, investigate'}"
            )
        print("\n[kvdt] compare against what this codebase already ships on: the conv add-chain was")
        print("[kvdt] accepted at teacher-forced decode logit PCC 0.996 over 96 steps.")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
