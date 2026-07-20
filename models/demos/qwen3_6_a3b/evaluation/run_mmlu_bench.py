# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Self-contained MMLU-Redux accuracy + perf benchmark for Qwen3.6-35B-A3B on a single Blackhole P150.

Purpose: put a NUMBER on the BFP4 quantization the single-card build ships with. Per-module PCC gates
prove each op is faithful, but nothing measures the end-to-end accuracy cost of BFP4 experts. Tenstorrent
explicitly moved their 27B off bf4/bf8 to bf16 for accuracy, so this is the missing datapoint.

Scoring is standard 0-shot MMLU letter log-prob: the prompt ends with "Answer:" and we compare the
next-token log-probabilities of "A"/"B"/"C"/"D" (last-token logits only — no generation needed). Also
reports prefill TTFT / throughput binned by prompt length. Decode throughput is NOT measured here (an
eager per-sample step is the cold/compile path, ~10x slower than real decode) — use bench_decode.py,
which times the traced steady-state decode (the demo/server path).

Depends only on ``datasets`` / ``huggingface_hub`` (both already in the tt-metal python_env) — NOT on
lm-eval. For the general lm-evaluation-harness path, see ``lm_eval_wrapper.py``.

Usage (tt-metal python_env):
    # default: 100 samples, BFP4 experts (as shipped)
    QWEN36_LAYERS=40 python -u models/demos/qwen3_6_a3b/evaluation/run_mmlu_bench.py --samples 100

    # BFP4-vs-BFP8 accuracy sweep (the deliverable):
    QWEN36_LAYERS=40 QWEN36_EXPERT_DTYPE=bf4 python -u .../run_mmlu_bench.py --samples 200 --output /tmp/mmlu_bf4.json
    QWEN36_LAYERS=40 QWEN36_EXPERT_DTYPE=bf8 python -u .../run_mmlu_bench.py --samples 200 --output /tmp/mmlu_bf8.json
    #   (bf8 needs ~2x expert DRAM; drop QWEN36_LAYERS if it OOMs on a single card.)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs

LETTERS = ["A", "B", "C", "D"]
SYSTEM_PROMPT = "The following are multiple choice questions (with answers) about {subject_fmt}.\n\n"
SEQ_BINS = [(0, 64), (64, 96), (96, 128), (128, 10**9)]
SEQ_BIN_LABELS = ["0-64", "64-96", "96-128", "128+"]


def load_mmlu_redux(num_samples: int, seed: int):
    """Download MMLU-Redux test CSVs and return a random pandas sample (subject, question, choices, answer)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files("edinburgh-dawg/mmlu-redux", repo_type="dataset")
    csv_files = sorted(f for f in files if f.endswith("test.csv"))
    rows = []
    for csv_file in csv_files:
        subject = csv_file.split("/")[0]
        path = hf_hub_download("edinburgh-dawg/mmlu-redux", csv_file, repo_type="dataset")
        df = pd.read_csv(path, header=None, skiprows=1)
        df.columns = ["question", "choices", "answer", "error_type", "source", "correct_answer", "potential_reason"]
        df["subject"] = subject
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    sample = all_df.sample(n=min(num_samples, len(all_df)), random_state=seed)
    return sample.reset_index(drop=True)


def format_mmlu_prompt(question: str, choices: list[str], subject: str) -> str:
    prompt = SYSTEM_PROMPT.format(subject_fmt=subject.replace("_", " "))
    prompt += f"{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{LETTERS[i]}. {choice}\n"
    prompt += "Answer:"
    return prompt


def get_seq_bin(token_len: int) -> int:
    for i, (lo, hi) in enumerate(SEQ_BINS):
        if lo <= token_len < hi:
            return i
    return len(SEQ_BINS) - 1


def main():
    ap = argparse.ArgumentParser(description="MMLU-Redux accuracy+perf benchmark (single P150)")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="/tmp/qwen36_mmlu_result.json")
    ap.add_argument(
        "--trace",
        action="store_true",
        help="use the traced (bucketed) prefill path — faster (strips host dispatch), but it runs the "
        "bf16 ttl GDN chunk kernel, NOT the fp32-stable eager path, so accuracy reflects the TRACED "
        "path (may be lower). Default (eager) matches the shipping/demo/server prefill.",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))
    max_seq = int(os.environ.get("QWEN36_MAX_SEQ", "512"))

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    # First token of each answer letter (leading-space variants handled by the "Answer:" suffix).
    letter_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in LETTERS]
    print(
        f"[config] layers={n_layers} max_seq={max_seq} prefill={'traced(bf16)' if args.trace else 'eager(fp32)'} "
        f"expert_dtype={os.environ.get('QWEN36_EXPERT_DTYPE', 'bf4')} "
        f"expert_down_bf8={os.environ.get('QWEN36_EXPERT_DOWN_BF8', '0')} "
        f"lmhead_bf4={os.environ.get('QWEN36_LMHEAD_BF4', '1')} letter_ids={dict(zip(LETTERS, letter_ids))}"
    )

    print("[data] loading MMLU-Redux...")
    t0 = time.time()
    sample_df = load_mmlu_redux(args.samples, args.seed)
    prompts = []
    for _, row in sample_df.iterrows():
        choices = ast.literal_eval(row["choices"])
        if len(choices) != 4:
            continue  # letter-scoring assumes A-D
        text = format_mmlu_prompt(row["question"], choices, row["subject"])
        ids = tokenizer.encode(text, add_special_tokens=False)
        prompts.append(
            {"input_ids": ids, "token_len": len(ids), "answer_idx": int(row["answer"]), "subject": row["subject"]}
        )
    print(
        f"[data] {len(prompts)} usable samples in {time.time() - t0:.1f}s "
        f"(token_len min/mean/max = {min(p['token_len'] for p in prompts)}/"
        f"{sum(p['token_len'] for p in prompts) // len(prompts)}/{max(p['token_len'] for p in prompts)})"
    )

    # Traced prefill needs a trace region big enough for one full-sequence prefill trace PER captured
    # bucket (× n_layers) — ~300MB for the default buckets at max_seq=512, and it grows with max_seq.
    # Default generous (768MB); override with QWEN36_TRACE_REGION_MB if a large max_seq overflows it.
    # Eager needs no trace region.
    trace_region_mb = int(os.environ.get("QWEN36_TRACE_REGION_MB", "768"))
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1), trace_region_size=trace_region_mb * 1024 * 1024 if args.trace else 0
    )
    try:
        model_args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=max_seq)
        loader = CheckpointLoader(ckpt)
        print(f"[model] building {n_layers}-layer model ({model_args.model_name})...")
        t0 = time.time()
        model = TtModel(mesh, model_args, loader, num_layers=n_layers)
        print(f"[model] built in {time.time() - t0:.1f}s")

        # Prefill path. Eager = fp32-stable (accuracy default). Traced = faster (bucketed replay, no
        # host dispatch) BUT runs the bf16 ttl GDN kernel — accuracy is for the traced path, not the
        # fp32 eager default. setup_prefill_traces() pre-captures one trace per bucket (compiles the
        # ttl kernel once, ~70s); prompts pad up to the smallest bucket >= their length at replay.
        if args.trace:
            print(
                "[prefill] WARNING: --trace uses the bf16 ttl GDN kernel (numerically approximate); "
                "accuracy reflects the TRACED path, not the fp32 eager default. Capturing prefill trace..."
            )
            # SINGLE-bucket traced prefill: capture exactly ONE bucket sized to cover the longest prompt
            # and pad every prompt to it. Multi-bucket capture (one trace per length class) is fragile in
            # this batch loop — juggling several resident traces + the per-call eager head at 40 layers
            # deadlocks after a couple of samples. One resident trace of one fixed shape is robust; short
            # prompts pad up (a little wasted compute, but still ~15x faster than eager). forward_prefill_
            # traced pads T up to the captured bucket and masks the GDN to the real valid_len.
            default_b = model._default_prefill_buckets()
            max_len = min(max(p["token_len"] for p in prompts), max_seq)
            bucket = next((b for b in default_b if b >= max_len), default_b[-1])
            print(
                f"[prefill] single-bucket traced prefill: padding all prompts to {bucket} "
                f"(trace region {trace_region_mb}MB)"
            )
            t0 = time.time()
            model.setup_prefill_traces([bucket])
            print(f"[prefill] trace captured in {time.time() - t0:.1f}s")
        prefill = model.forward_prefill_traced if args.trace else model.forward

        results, correct = [], 0
        t_eval = time.time()
        for idx, p in enumerate(prompts):
            ids = torch.tensor([p["input_ids"]], dtype=torch.long)
            if ids.shape[1] > max_seq:
                ids = ids[:, -max_seq:]

            t_pf = time.perf_counter()
            logits = prefill(ids)  # [1, 1, vocab] (host); eager fp32 or traced bf16 per --trace
            if args.trace:
                # forward_prefill_traced runs execute_trace(blocking=False) and writes SHARED persistent
                # input buffers (_pf_ids, conv reset) each call. At high layer counts the trace runs long
                # enough that the next iteration's buffer writes race the still-executing trace, filling
                # the 2-deep command queue and deadlocking (hangs after ~2 samples, no error). Drain the
                # trace before the next iteration reuses those buffers. (Eager needs no sync — no trace.)
                ttnn.synchronize_device(mesh)
            ttft = time.perf_counter() - t_pf

            last = logits[0, -1][: model_args.vocab_size].float()
            lp = F.log_softmax(last, dim=-1)
            answer_lps = [lp[tid].item() for tid in letter_ids]
            pred_idx = max(range(4), key=lambda i: answer_lps[i])
            is_correct = pred_idx == p["answer_idx"]
            correct += int(is_correct)

            # NOTE: decode throughput is deliberately NOT measured here. MMLU letter-scoring only needs
            # prefill logits, and a per-sample eager decode step measures the cold/compile-warmup eager
            # path (~10x slower than the real traced decode). For decode tok/s use bench_decode.py, which
            # captures the decode trace and times decode_step_traced (the demo/server steady-state path).
            results.append(
                {
                    "idx": idx,
                    "subject": p["subject"],
                    "token_len": p["token_len"],
                    "seq_bin": get_seq_bin(p["token_len"]),
                    "correct": is_correct,
                    "pred": LETTERS[pred_idx],
                    "ref": LETTERS[p["answer_idx"]],
                    "ttft_ms": ttft * 1000,
                    "prefill_tps": p["token_len"] / ttft if ttft else 0,
                }
            )
            acc = correct / len(results) * 100
            print(
                f"  [{idx + 1:>3}/{len(prompts)}] {'OK' if is_correct else 'X '} "
                f"pred={LETTERS[pred_idx]} ref={LETTERS[p['answer_idx']]} acc={acc:.0f}% "
                f"len={p['token_len']:>3} ttft={ttft * 1000:.0f}ms",
                flush=True,
            )

        total_time = time.time() - t_eval
        total = len(results)

        # bin aggregation
        bins = defaultdict(lambda: {"correct": 0, "total": 0, "ttft_ms": [], "prefill_tps": []})
        for r in results:
            b = bins[r["seq_bin"]]
            b["total"] += 1
            b["correct"] += int(r["correct"])
            b["ttft_ms"].append(r["ttft_ms"])
            b["prefill_tps"].append(r["prefill_tps"])

        def avg(xs):
            return sum(xs) / len(xs) if xs else 0.0

        print(f"\n{'=' * 74}")
        print(f" MMLU-Redux — Qwen3.6-35B-A3B (experts={model_args.expert_weight_dtype}) on single P150")
        print(f"{'=' * 74}")
        print(
            f" Overall: {total} samples, Accuracy = {correct / total * 100:.1f}%  "
            f"({total_time:.0f}s, {total_time / total:.1f}s/sample)\n"
        )
        print(f" {'Seq Len':>10}  {'N':>5}  {'Acc':>7}  {'TTFT(ms)':>10}  {'Prefill TPS':>11}")
        print(f" {'-' * 10}  {'-' * 5}  {'-' * 7}  {'-' * 10}  {'-' * 11}")
        for i, label in enumerate(SEQ_BIN_LABELS):
            b = bins[i]
            if not b["total"]:
                continue
            print(
                f" {label:>10}  {b['total']:>5}  {b['correct'] / b['total'] * 100:>6.1f}%  "
                f"{avg(b['ttft_ms']):>10.0f}  {avg(b['prefill_tps']):>11.1f}"
            )
        print(f" {'-' * 10}  {'-' * 5}  {'-' * 7}  {'-' * 10}  {'-' * 11}")
        print(
            f" {'ALL':>10}  {total:>5}  {correct / total * 100:>6.1f}%  "
            f"{avg([r['ttft_ms'] for r in results]):>10.0f}  "
            f"{avg([r['prefill_tps'] for r in results]):>11.1f}"
        )
        print(" (decode throughput: use bench_decode.py — traced steady-state, not the eager per-sample step)")
        print(f"{'=' * 74}")

        output = {
            "model": model_args.model_name,
            "n_layers": n_layers,
            "expert_weight_dtype": str(model_args.expert_weight_dtype),
            "expert_down_weight_dtype": str(model_args.expert_down_weight_dtype),
            "lmhead_dtype": str(model.lm_head_w.dtype),
            "prefill_mode": "traced_bf16" if args.trace else "eager_fp32",
            "dataset": "MMLU-Redux",
            "num_samples": total,
            "seed": args.seed,
            "overall_accuracy": correct / total * 100,
            "avg_ttft_ms": avg([r["ttft_ms"] for r in results]),
            "avg_prefill_tps": avg([r["prefill_tps"] for r in results]),
            "total_eval_time_s": total_time,
            "per_sample": results,
        }
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {args.output}")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
