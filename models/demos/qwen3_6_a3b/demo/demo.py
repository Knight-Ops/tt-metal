# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Runnable text demo / perf harness for Qwen3.6-35B-A3B on a single Blackhole.

Usage (tt-metal python_env):
    QWEN36_LAYERS=40 python models/demos/qwen3_6_a3b/demo/demo.py --seq 64 --gen 8
    QWEN36_LAYERS=4  python models/demos/qwen3_6_a3b/demo/demo.py --prompt "Hello, world"

Notes:
  - Phase A: prefill-only path; "generation" re-runs the growing sequence per token (no caches),
    so decode is O(seq^2). Use it to sanity-check coherence + measure prefill throughput.
  - Real perf needs Phase B (fused on-device delta-rule scan + sparse MoE). See plan/status.
"""

import argparse
import os
import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs


def maybe_tokenizer(ckpt):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(ckpt)
    except Exception as e:
        print(f"[demo] tokenizer unavailable ({type(e).__name__}); using random token ids")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="The capital of France is")
    ap.add_argument("--seq", type=int, default=0, help="if >0, ignore prompt and use this many random tokens")
    ap.add_argument("--gen", type=int, default=0, help="number of tokens to generate")
    ap.add_argument("--trace", action="store_true", help="use captured trace for decode (fast path)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default); >0 = on-device sampling")
    ap.add_argument("--top-k", type=int, default=0, help="top-k for sampling (0 or >32 = 32, the device max)")
    ap.add_argument("--top-p", type=float, default=1.0, help="top-p (nucleus) for sampling")
    ap.add_argument("--presence-penalty", type=float, default=0.0, help="subtract from logits of seen tokens")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for sampling (deterministic per seed)")
    args_cli = ap.parse_args()

    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))

    # Trace capture (greedy or sampling) needs a nonzero trace region; eager-only can use 0.
    trace_region = 200 * 1024 * 1024 if args_cli.trace else 0
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=trace_region)
    try:
        max_seq = int(os.environ.get("QWEN36_MAX_SEQ", "512"))
        args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=max_seq)
        loader = CheckpointLoader(ckpt)
        print(f"[demo] building {n_layers}-layer model ({args.model_name}) on {args.num_devices} device(s)...")
        t0 = time.time()
        model = TtModel(mesh, args, loader, num_layers=n_layers)
        print(f"[demo] model built in {time.time() - t0:.1f}s")

        tok = maybe_tokenizer(ckpt)
        if args_cli.seq > 0 or tok is None:
            T = args_cli.seq or 32
            ids = torch.randint(0, args.vocab_size, (1, T))
        else:
            ids = torch.tensor([tok.encode(args_cli.prompt)], dtype=torch.long)
        T = ids.shape[1]

        t0 = time.time()
        logits = model.forward(ids)
        dt = time.time() - t0
        print(f"[demo] prefill {T} tokens: {dt:.2f}s -> {T / dt:.2f} tok/s (TTFT {dt * 1000:.0f} ms)")

        # Greedy by default (temperature 0). With --temperature>0 the decode tail samples on device
        # (temperature/top-k/top-p); the first token (from prefill) stays greedy argmax.
        if args_cli.temperature > 0:
            model.enable_sampling(
                args_cli.temperature, args_cli.top_k, args_cli.top_p, args_cli.seed, args_cli.presence_penalty
            )
            print(
                f"[demo] sampling: temperature={args_cli.temperature} top_k={args_cli.top_k or 32} "
                f"top_p={args_cli.top_p} presence_penalty={args_cli.presence_penalty} seed={args_cli.seed}"
            )

        generated = []
        dec_times = []
        next_id = int(logits[0, -1].argmax())  # prefill's first token (greedy)
        if args_cli.trace and args_cli.gen:
            # token + positions live on device; decode runs embed->...->argmax->next-token in one
            # traced graph. one eager step compiles the kernels, then record-only capture + replay.
            generated.append(next_id)
            model.start_decode(next_id)
            next_id = model.decode_step_eager()  # warmup (compiles); produces the 2nd token
            t0 = time.time()
            model.capture_decode_trace()
            print(f"[demo] trace captured in {time.time() - t0:.1f}s")
            for step in range(args_cli.gen - 1):
                generated.append(next_id)
                t0 = time.time()
                next_id = model.decode_step_traced()  # only the 1 next-token id comes back to host
                dec_times.append(time.time() - t0)
        else:
            model.start_decode(next_id)
            for step in range(args_cli.gen):
                generated.append(next_id)
                t0 = time.time()
                next_id = model.decode_step_eager()
                dec_times.append(time.time() - t0)
        if dec_times:
            avg = sum(dec_times) / len(dec_times)
            tag = "traced" if args_cli.trace else "eager"
            print(f"[demo] decode ({tag}): {avg * 1000:.0f} ms/token -> {1 / avg:.2f} tok/s/user")
        if tok is not None and generated:
            print(f"[demo] generated: {tok.decode(generated)!r}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
