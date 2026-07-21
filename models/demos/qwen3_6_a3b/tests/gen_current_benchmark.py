# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One command → CURRENT_BENCHMARK.md, carrying three complementary views on a single Blackhole:

  1. End-to-end THROUGHPUT (real prompts, no profiler)         — true ms/token (tests/benchmark.py).
  2. TRACED per-component breakdown (real, un-inflated)        — where the traced decode step's time
     actually goes, timed by wall-clock replays (bench_decode), NOT the profiler → stable at 40 layers.
  3. Fine-grained SIGNPOST proportions (prefill + decode)      — the sub-component split within
     MoE/attention/gated-delta, from a Tracy signpost run.

Why (2) and (3) are separate: the device profiler that feeds the fine-grained signpost breakdown
aborts above ~4-8 layers (device-data readback overflow — a known tt-metal profiler+scale limit), and
even when it runs it inflates absolute µs ~1.8×. So the *real* per-component times come from the
un-profiled traced replays in (2) at the full 40 layers, while (3) supplies the finer split from a
small-layer run — valid because per-layer op shapes (hence per-component proportions) are identical at
any depth.

This script never opens the device itself (holds no chip locks); it runs two device sub-runs as
subprocesses and assembles the Markdown.

Run (tt-metal python_env; ~10-15 min: one 40-layer load for 1+2, one small-layer profiled run for 3):
    QWEN36_LAYERS=40 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/gen_current_benchmark.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(MODEL_DIR, "..", "..", ".."))
SCRATCH = os.environ.get("TMPDIR", "/tmp")
THROUGHPUT_JSON = os.path.join(SCRATCH, "qwen36_throughput.json")

sys.path.insert(0, HERE)
import signpost_report as sr  # pandas-only; safe here (no ttnn, no device)


def _run(cmd, env_extra, log_path):
    env = {**os.environ, **env_extra}
    print(f"[gen] $ {' '.join(cmd)}\n[gen]   env={env_extra} -> {log_path}", flush=True)
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"[gen]   returncode={rc}")
    return rc


def run_throughput(n_layers, max_seq, gen, prefill_iters, breakdown_iters):
    rc = _run(
        [
            sys.executable,
            "models/demos/qwen3_6_a3b/tests/benchmark.py",
            "--gen",
            str(gen),
            "--prefill-iters",
            str(prefill_iters),
            "--breakdown-iters",
            str(breakdown_iters),
            "--json-out",
            THROUGHPUT_JSON,
        ],
        {"QWEN36_LAYERS": str(n_layers), "QWEN36_MAX_SEQ": str(max_seq)},
        os.path.join(SCRATCH, "qwen36_throughput.log"),
    )
    if rc != 0:
        raise SystemExit(f"[gen] throughput sub-run failed; see {SCRATCH}/qwen36_throughput.log")
    with open(THROUGHPUT_JSON) as f:
        return json.load(f)


def run_breakdown(breakdown_layers, seq, max_seq, op_support):
    """Profiled signpost run at a REDUCED layer count (the device profiler is unreliable at 40).
    Returns (csv, prefill_agg, decode_agg); (None, None, None) if the profiler run fails so the doc
    still assembles from throughput + the real traced breakdown."""
    rc = _run(
        [
            sys.executable,
            "-m",
            "tracy",
            "-r",
            "-v",
            "--op-support-count",
            str(op_support),
            "models/demos/qwen3_6_a3b/tests/prof_decode.py",
        ],
        {
            "QWEN36_LAYERS": str(breakdown_layers),
            "QWEN36_SEQ": str(seq),
            "QWEN36_MAX_SEQ": str(max_seq),
            "QWEN36_SIGNPOST": "1",
            "QWEN36_PROF_EAGER_ONLY": "1",
        },
        os.path.join(SCRATCH, "qwen36_breakdown.log"),
    )
    if rc != 0:
        print("[gen] WARNING: profiled breakdown run failed — fine-grained section will be omitted.")
        return None, None, None
    csvs = glob.glob(os.path.join(REPO, "generated/profiler/reports/*/ops_perf_results_*.csv"))
    csv = max(csvs, key=os.path.getmtime) if csvs else None
    try:
        prefill = sr.aggregate(csv, between=["prefill_start", "prefill_stop"])
        decode = sr.aggregate(csv, between=["eager_start", "eager_stop"])
        if prefill["total_ns"] == 0 or decode["total_ns"] == 0:
            raise ValueError("empty signpost window (profiler likely truncated the capture)")
    except Exception as e:
        print(f"[gen] WARNING: could not slice signpost windows ({e}); omitting fine-grained section.")
        return None, None, None
    print(f"[gen] breakdown CSV: {csv}")
    return csv, prefill, decode


# --------------------------------------------------------------------------- markdown assembly
def _config_section(cfg, breakdown_layers, seq):
    return "\n".join(
        [
            "## Configuration\n",
            f"- **Model**: {cfg['model_name']}, **{cfg['n_layers']} layers** (30 gated-delta + 10 full-attention), "
            f"{cfg['num_experts']} experts / top-{cfg['num_experts_per_tok']} MoE",
            f"- **Weights**: experts `{cfg['expert_dtype']}` (down `{cfg['expert_down_dtype']}`), "
            f"attention/linear `{cfg['attn_dtype']}`, activations `{cfg['activation_dtype']}`",
            f"- **Device**: 1× Blackhole   •   **Decode**: greedy, traced   •   **max_seq_len**: {cfg['max_seq']}",
            f"- **Fine-grained breakdown** profiled at **{breakdown_layers} layers**, {seq}-token prefill "
            "(proportions are layer-count-independent; see §3)\n",
        ]
    )


def _throughput_section(data):
    rows = data["rows"]
    n = len(rows) or 1
    mean = lambda k: sum(r[k] for r in rows) / n
    L = [
        "## 1. End-to-end throughput (real prompts, no profiler)\n",
        "Real wall-clock on the production path (chat-templated greedy prefill + traced decode).\n",
        "| Prompt | Prompt tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s/user) |",
        "|---|--:|--:|--:|--:|",
    ]
    for r in rows:
        L.append(
            f"| {r['label']} | {r['prompt_tokens']} | {r['ttft_ms']:.1f} | "
            f"{r['prefill_tps']:.1f} | {r['decode_tps']:.2f} |"
        )
    L.append(
        f"| **mean** | — | **{mean('ttft_ms'):.1f}** | **{mean('prefill_tps'):.1f}** | "
        f"**{mean('decode_tps'):.2f}** |\n"
    )
    L.append("<details><summary>Sample greedy generations</summary>\n")
    for r in rows:
        snip = r["text"][:160] + ("…" if len(r["text"]) > 160 else "")
        L.append(f"- **{r['label']}**: {snip}")
    L.append("\n</details>\n")
    return "\n".join(L)


def _traced_breakdown_section(b):
    if not b:
        return ""
    step = b["full_model_ms"]
    L = [
        "## 2. Traced decode — real per-component breakdown (the optimization map)\n",
        f"Where the traced decode step's time goes, by major component. Timed by **real "
        f"`execute_trace` wall-clock replays (no profiler)** — these are the actual traced "
        f"kernel+gap times at the full {b.get('iters','?')}-iter average, **not inferred from the "
        f"eager run and not profiler-inflated**. Per-token layer counts: 30 gated-delta, 10 "
        f"attention, 40 MoE, 1 head.\n",
        f"Full traced step: **{step:.2f} ms/token → {b['full_model_tok_s']:.1f} tok/s/user** "
        f"({b['full_model_dev_ms']:.2f} ms pure `execute_trace`).\n",
        "| Component | ms/op | × layers | total ms | % of step |",
        "|---|--:|--:|--:|--:|",
    ]
    for key, name in [
        ("gated-delta", "Gated-DeltaNet"),
        ("attention", "Attention"),
        ("moe", "MoE"),
        ("lm_head", "lm_head"),
    ]:
        ms = b["component_ms"].get(key, 0.0)
        tot = b["component_totals_ms"].get(key, 0.0)
        L.append(f"| {name} | {ms:.3f} | {b['layer_counts'][key]} | {tot:.2f} | {100*tot/step:.1f} |")
    L.append(f"| **component sum** | — | — | **{b['sum_ms']:.2f}** | **{100*b['sum_ms']/step:.1f}** |")
    L.append(
        f"\n_MoE split (per op): router+shared `{b['moe_router_ms']:.3f}` ms, sparse experts "
        f"`{b['moe_sparse_ms']:.3f}` ms. The ~{100*(step-b['sum_ms'])/step:.0f}% gap between the "
        "component sum and the full step is on-device inter-op dispatch latency (what op-fusion "
        "can still attack)._\n"
    )
    return "\n".join(L)


def _signpost_section(prefill, decode, breakdown_layers, seq, csv):
    L = ["## 3. Fine-grained signpost proportions (prefill + decode)\n"]
    if not decode or not prefill:
        L.append(
            "_Not available: the device profiler run failed at this scale. Reproduce manually with "
            "the commands in §How-to; see the profiler-scale note there._\n"
        )
        return "\n".join(L)
    L.append(
        f"Sub-component split from the Tracy signpost regions (`tt/signpost.py`), profiled at "
        f"**{breakdown_layers} layers** (device profiler is unreliable at 40). **Read these as "
        "PROPORTIONS, not absolute ms** — the profiler inflates absolutes ~1.8× and this is a "
        "reduced-depth run; the real ms live in §2. `self` = time in ops directly in the region "
        "(the thing to optimize); `incl` = region + nested children.\n"
    )
    L.append(f"### 3a. Decode step — within-layer split (eager, {breakdown_layers} layers)\n")
    L.append(f"**Top self-time targets:** {_headroom(decode)}.\n")
    L.append(
        f"> The once-per-step `dec.select`/`head.lm_head` is **over-weighted at {breakdown_layers} "
        "layers** (it does not scale with depth); §2 shows lm_head is only ~4% of the real 40-layer "
        "step. Use this table for the *within-layer* ranking (`moe.*`, `delta.*`, `attn.*`) and §2 "
        "for the true cross-component share.\n"
    )
    L.append(sr.markdown_table(decode))
    L.append(f"\n### 3b. Prefill — {seq}-token prompt ({breakdown_layers} layers)\n")
    L.append(f"**Top self-time targets:** {_headroom(prefill)}.\n")
    L.append(sr.markdown_table(prefill))
    L.append(f"\n<sub>signpost CSV: `{os.path.relpath(csv, REPO)}`</sub>\n")
    return "\n".join(L)


def _headroom(agg, k=5):
    total = agg["total_ns"] or 1
    items = sorted(agg["self"].items(), key=lambda kv: -kv[1])[:k]
    return ", ".join(f"`{n}` ({100*v/total:.0f}%)" for n, v in items if v > 0)


def _howto_section(cfg, breakdown_layers, seq, gen, op_support):
    return "\n".join(
        [
            "## How to regenerate & expectations\n",
            "```bash",
            "# One command — throughput + real traced breakdown + fine-grained signposts:",
            f"QWEN36_LAYERS={cfg['n_layers']} ./python_env/bin/python \\",
            "    models/demos/qwen3_6_a3b/tests/gen_current_benchmark.py \\",
            f"    --breakdown-layers {breakdown_layers} --seq {seq} --gen {gen}",
            "```\n",
            "**What to expect when running it**",
            "- **Runtime ~10–15 min**: one 40-layer model load (~110 s warm from the `.tensorbin` cache, "
            "longer cold) drives both §1 and §2; a second, small-layer load drives the §3 profiled run. "
            "Model-load time is excluded from all reported figures.",
            "- **Two device sub-runs** (see `$TMPDIR/qwen36_throughput.log` and `qwen36_breakdown.log`). "
            "The script itself never opens the device.",
            f"- **§3 is profiled at {breakdown_layers} layers on purpose.** The tt-metal device profiler "
            "aborts during device-data readback above ~4–8 layers (buffer overflow at scale), so a full "
            "40-layer signpost capture is currently not possible. Per-layer op shapes are identical at any "
            "depth, so the *proportions* in §3 hold for 40 layers; the real 40-layer per-component ms are "
            "in §2 (which uses un-profiled wall-clock replays and is stable at 40).",
            "- **Trust ratios, not absolutes, in §3** (profiler inflates ~1.8×). §1 and §2 are real " "wall-clock.",
            "- **When numbers move**: re-run after any change to kernels, dtypes (`tt/model_config.py`), "
            "layer config, or the tt-metal/ttnn version. Decode tok/s/user is the headline single-user "
            "metric; watch the §2 component that dominates (today: MoE).\n",
            "**Piecewise / deeper profiling**",
            "```bash",
            "# throughput + real traced breakdown only (40 layers):",
            "QWEN36_LAYERS=40 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/benchmark.py --gen 32",
            "# fine-grained signposts only (keep layers small — see the §3 profiler-scale note;",
            "# and NO -p flag: it truncates the capture), then slice each window:",
            f"QWEN36_LAYERS={breakdown_layers} QWEN36_SIGNPOST=1 QWEN36_SEQ={seq} QWEN36_PROF_EAGER_ONLY=1 \\",
            f"    ./python_env/bin/python -m tracy -r -v --op-support-count {op_support} \\",
            "    models/demos/qwen3_6_a3b/tests/prof_decode.py",
            "python models/demos/qwen3_6_a3b/tests/signpost_report.py --between eager_start eager_stop",
            "python models/demos/qwen3_6_a3b/tests/signpost_report.py --between prefill_start prefill_stop",
            "```\n",
            "Signpost regions are documented in `DEBUGGING.md` §1.5.1; markers are env-gated "
            "(`QWEN36_SIGNPOST`) and cost nothing in normal runs.\n",
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seq",
        type=int,
        default=128,
        help="prefill length for the fine-grained breakdown "
        "(kept modest: the profiler capture truncates the measured window at larger prefills)",
    )
    ap.add_argument("--gen", type=int, default=32, help="tokens/prompt for the throughput section")
    ap.add_argument("--prefill-iters", type=int, default=3)
    ap.add_argument("--breakdown-iters", type=int, default=100, help="traced-replay iters for §2")
    ap.add_argument("--breakdown-layers", type=int, default=4, help="layer count for the §3 profiled run")
    ap.add_argument("--op-support", type=int, default=20000, help="tracy --op-support-count for §3")
    ap.add_argument("--skip-throughput", action="store_true", help="reuse the existing throughput JSON")
    ap.add_argument("--skip-breakdown", action="store_true", help="reuse the newest profiler CSV for §3")
    ap.add_argument("--out", default=os.path.join(MODEL_DIR, "CURRENT_BENCHMARK.md"))
    a = ap.parse_args()
    n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))
    max_seq = int(os.environ.get("QWEN36_MAX_SEQ", "2048"))

    if a.skip_throughput:
        with open(THROUGHPUT_JSON) as f:
            data = json.load(f)
    else:
        data = run_throughput(n_layers, max_seq, a.gen, a.prefill_iters, a.breakdown_iters)

    if a.skip_breakdown:
        csvs = glob.glob(os.path.join(REPO, "generated/profiler/reports/*/ops_perf_results_*.csv"))
        csv = max(csvs, key=os.path.getmtime) if csvs else None
        prefill = sr.aggregate(csv, between=["prefill_start", "prefill_stop"]) if csv else None
        decode = sr.aggregate(csv, between=["eager_start", "eager_stop"]) if csv else None
    else:
        csv, prefill, decode = run_breakdown(a.breakdown_layers, a.seq, max_seq, a.op_support)

    cfg = data["config"]
    date = time.strftime("%Y-%m-%d %H:%M %Z")
    md = "\n".join(
        [
            "# Qwen3.6-35B-A3B — Current Benchmark\n",
            f"_Generated by `models/demos/qwen3_6_a3b/tests/gen_current_benchmark.py` on {date}._\n",
            "Single Blackhole, single user. Three views: **(1)** real-prompt throughput, **(2)** the real "
            "traced per-component breakdown (the optimization map), and **(3)** fine-grained signpost "
            "proportions within each component, for prefill and decode.\n",
            _config_section(cfg, a.breakdown_layers, a.seq),
            _throughput_section(data),
            _traced_breakdown_section(data.get("breakdown")),
            _signpost_section(prefill, decode, a.breakdown_layers, a.seq, csv),
            _howto_section(cfg, a.breakdown_layers, a.seq, a.gen, a.op_support),
        ]
    )
    with open(a.out, "w") as f:
        f.write(md)
    print(f"\n[gen] wrote {a.out}")


if __name__ == "__main__":
    main()
