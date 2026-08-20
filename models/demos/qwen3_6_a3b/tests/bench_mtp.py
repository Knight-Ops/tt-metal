# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Speculative-decode (MTP) benchmark on the REAL Qwen3.6-35B-A3B model.

Reports the two things that decide whether MTP pays:
  * E[tokens/round] — the ACCEPTANCE, which only means anything at the full layer count (the MTP head
    was trained against the 40-layer backbone's hidden states, so a shallow backbone gives alpha ~ 0),
  * the round cost, decomposed into draft / verify / host glue.

Both are measured EAGER here (tracing is a follow-up), so the absolute tok/s is dispatch-bound and not
the shipping number. The projected traced throughput is computed from the measured E[tokens/round] and
the Phase-A verify cost model (see MTP.md).

    QWEN36_LAYERS=40 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/bench_mtp.py --rounds 24
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

CKPT = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
PROMPTS = [
    "Explain why a mixture-of-experts layer is harder to batch efficiently than a dense feed-forward "
    "layer on an accelerator with a fixed tile size.",
    "Write a Python function that merges two sorted lists without using sorted(), with a docstring.",
]

# Phase-A calibrated per-layer costs (MTP.md); used only to PROJECT traced throughput.
BASE_STEP_MS = 29.01
N_MOE, N_GDN, N_ATTN = 40, 30, 10
MOE_FLAT, MOE_C0, MOE_SLOPE = 0.2165, 0.0277, 0.0818
GDN_FLAT, GDN_KERN1, GDN_DELTA = 0.2576, 0.0614, 0.0465
ATTN_MS, ATTN_D, HEAD_MS, GAP_MS, DRAFT_MS, GLUE_MS = 0.295, 0.0016, 1.226, 2.21, 2.03, 0.5


def projected_verify_ms(K):
    return (
        N_MOE * (MOE_FLAT + MOE_C0 + MOE_SLOPE * K)
        + N_GDN * (GDN_FLAT + GDN_KERN1 + GDN_DELTA * (K - 1))
        + N_ATTN * (ATTN_MS + ATTN_D * (K - 1))
        + HEAD_MS
        + GAP_MS
    )


def _prompt(tok, text):
    enc = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    if not torch.is_tensor(ids):
        ids = torch.tensor(ids)
    return ids.reshape(1, -1).to(torch.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--gammas", type=str, default="1,2,3")
    ap.add_argument("--baseline-steps", type=int, default=16)
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        args = ModelArgs(mesh, ckpt_dir=CKPT, max_seq_len=int(os.environ.get("QWEN36_MAX_SEQ", "512")))
        loader = CheckpointLoader(CKPT)
        assert loader.has_mtp(), "checkpoint has no mtp.* head"
        t0 = time.time()
        model = TtModel(mesh, args, loader, num_layers=n_layers)
        print(f"[bench_mtp] built {n_layers}-layer model in {time.time()-t0:.0f}s", flush=True)
        model.build_mtp_head(gamma=1)
        print(f"[bench_mtp] built MTP head", flush=True)

        # ---- eager baseline decode, for an apples-to-apples eager comparison ----
        ids = _prompt(tok, PROMPTS[0])
        lg = model.forward(ids)
        model.start_decode(int(lg[0, -1].argmax()))
        model.decode_step_eager()  # warm/compile
        t0 = time.time()
        for _ in range(a.baseline_steps):
            model.decode_step_eager()
        base_ms = (time.time() - t0) / a.baseline_steps * 1e3
        print(f"\n  eager baseline decode: {base_ms:.1f} ms/token ({1000/base_ms:.1f} tok/s)")
        print(f"  (traced baseline for reference: {BASE_STEP_MS:.2f} ms/token " f"= {1000/BASE_STEP_MS:.1f} tok/s)\n")

        hdr = (
            f"{'gamma':>6}{'K':>3}{'mode':>8}{'tok/round':>11}{'accept':>8}"
            f"{'ms/round':>10}{'ms/token':>10}{'tok/s':>8}{'vs base':>9}{'proj':>8}"
        )
        print(hdr)
        print("-" * len(hdr))
        for gamma in [int(g) for g in a.gammas.split(",")]:
            model.build_mtp_head(gamma=gamma)
            streams = {}
            for mode in ("eager", "verify", "full"):
                model.mtp_verify_trace = None
                model.mtp_draft_trace = None
                model.mtp_commit_traces = None
                ids = _prompt(tok, PROMPTS[0])
                lg = model.forward(ids)
                model.start_decode(int(lg[0, -1].argmax()))
                model._mtp_hidden = None
                # Same number of warm rounds in EVERY mode: the traced modes need two eager rounds
                # before capture (K=1 seed, then a K-row verify so `_verify_pending` holds the K-shaped
                # scratch), so eager must burn the same two or the timed streams start at different
                # positions and the parity check compares offset sequences.
                model.setup_mtp_decode()
                model.spec_decode_step()
                model.spec_decode_step()
                if mode != "eager":
                    cur = int(ttnn.to_torch(model.t_tok).reshape(-1)[0])
                    positions = [model.pos + i for i in range(gamma + 1)]
                    model.capture_mtp_verify_trace([cur] * (gamma + 1), positions)
                    if mode == "full":
                        model._mtp_write_round([cur] * (gamma + 1), positions)
                        model.capture_mtp_draft_trace()
                        model.capture_mtp_commit_traces()
                model.spec_decode_step()  # warm
                st0 = dict(model.mtp_stats)
                t0 = time.time()
                emitted, stream = 0, []
                for _ in range(a.rounds):
                    e = model.spec_decode_step()
                    emitted += len(e)
                    stream.extend(e)
                dt = time.time() - t0
                streams[mode] = stream
                st = model.mtp_stats
                tpr = emitted / a.rounds
                acc = (st["accepted"] - st0["accepted"]) / max(st["drafted"] - st0["drafted"], 1)
                ms_round = dt / a.rounds * 1e3
                mspt = ms_round / tpr
                proj = (projected_verify_ms(gamma + 1) + gamma * DRAFT_MS + GLUE_MS) / tpr
                print(
                    f"{gamma:>6}{gamma+1:>3}{mode:>8}{tpr:>11.3f}{acc:>8.2f}{ms_round:>10.1f}"
                    f"{mspt:>10.2f}{1000/mspt:>8.1f}{BASE_STEP_MS/mspt:>8.2f}x"
                    f"{BASE_STEP_MS/proj:>7.2f}x",
                    flush=True,
                )
            # parity: every trace is bit-exact vs its eager counterpart, so the token STREAMS must
            # match. A wrong commit trace (the accept-count rollback) shows up here and nowhere else.
            for m in ("verify", "full"):
                a_, b_ = streams["eager"], streams.get(m, [])
                n_ = min(len(a_), len(b_))
                div = next((i for i in range(n_) if a_[i] != b_[i]), n_)
                print(
                    f"         parity eager vs {m:>6}: identical for {div}/{n_} tokens"
                    f"{'' if div == n_ else '  <-- MISMATCH'}",
                    flush=True,
                )
        # ---- decompose the traced round: replay each trace in isolation ----
        if model.mtp_verify_trace is not None:

            def rep(tid, iters=30):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                t0 = time.time()
                for _ in range(iters):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                return (time.time() - t0) / iters * 1e3

            v = rep(model.mtp_verify_trace)
            d = rep(model.mtp_draft_trace) if model.mtp_draft_trace else float("nan")
            c = rep(model.mtp_commit_traces[1]) if model.mtp_commit_traces else float("nan")
            print(
                f"\n  per-trace replay (pure device):  verify {v:.2f} ms   draft(gamma) {d:.2f} ms"
                f"   commit {c:.2f} ms   sum {v+d+c:.2f} ms"
            )
            print(
                f"  cost-model verify({gamma+1}) = {projected_verify_ms(gamma+1):.2f} ms,"
                f" draft = {gamma*DRAFT_MS:.2f} ms -> model round {projected_verify_ms(gamma+1)+gamma*DRAFT_MS+GLUE_MS:.2f} ms"
            )

        print("\n  'vs base' is measured against the 29.01 ms traced baseline decode step; 'proj' is the")
        print("  Phase-A cost model ceiling (fully traced round: draft + verify + commit all captured).")
        print("  Only VERIFY is traced here — draft and commit are still eager, so the gap between the")
        print("  measured and projected columns is what capturing those two would recover.")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
