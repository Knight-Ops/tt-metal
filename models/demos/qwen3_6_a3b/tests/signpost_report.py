# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Turn a Tracy op-perf CSV into a per-component DEVICE-KERNEL-TIME breakdown for Qwen3.6.

Companion to the QWEN36_SIGNPOST=1 region markers in tt/ (see tt/signpost.py). Those markers emit
nested ``name`` / ``name/end`` signpost pairs around each component; this script walks the op-perf
CSV in timeline order, pairs the markers with a stack, and sums the DEVICE KERNEL DURATION of the ops
inside each region — aggregated across ALL occurrences of a name (e.g. one ``layer.moe`` row is the
MoE cost summed over all 40 layers in the profiled step). Reports both INCLUSIVE time (region + its
nested children) and SELF time (ops directly in the region, not in a nested child).

Workflow (profiler-enabled build, ENABLE_TRACY=ON):
    QWEN36_LAYERS=40 python -m tracy -r -p -v --op-support-count 6000 \
        models/demos/qwen3_6_a3b/tests/prof_decode.py          # or prof_prefill.py
    python models/demos/qwen3_6_a3b/tests/signpost_report.py   # reads the newest ops_perf CSV

The prof_*.py harnesses bracket the measured EAGER step with flat markers (eager_start/eager_stop,
prefill_eager_start/prefill_eager_stop); by default this script restricts to the first such window it
finds so warmup/compile and the traced replay are excluded. Pass --between START STOP to pick a
different window, --all for the whole file, or --csv PATH for a specific report.

Decode note: host signposts fire during trace CAPTURE, not execute_trace replay, so this breaks down
the EAGER decode step. Per-op DEVICE KERNEL DURATION there equals traced replay (same kernels); only
the op-to-op gaps differ. Cross-check totals against bench_decode.py::bench_components.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import pandas as pd

# Known flat (non-region) markers the prof_*.py harnesses emit; used as window bounds, never regions.
_FLAT_MARKERS = {
    "start",
    "stop",
    "eager_start",
    "eager_stop",
    "trace_start",
    "trace_stop",
    "prefill_start",
    "prefill_stop",
    "prefill_eager_start",
    "prefill_eager_stop",
    "prefill_traced_start",
    "prefill_traced_stop",
}
# Preferred eager windows to auto-select, in priority order (start, stop).
_DEFAULT_WINDOWS = [("eager_start", "eager_stop"), ("prefill_eager_start", "prefill_eager_stop"), ("start", "stop")]


def _find_csv(explicit):
    if explicit:
        return explicit
    from tracy.process_model_log import get_latest_ops_log_filename

    return get_latest_ops_log_filename("")


def _duration_col(df):
    if "DEVICE KERNEL DURATION [ns]" in df.columns:
        return "DEVICE KERNEL DURATION [ns]"
    for c in df.columns:  # tolerate minor header drift across profiler versions
        if "KERNEL DURATION" in c.upper():
            return c
    raise SystemExit(f"no kernel-duration column found in CSV; columns = {list(df.columns)}")


def _is_signpost(op_type):
    return isinstance(op_type, str) and "signpost" in op_type.lower()


def _slice_window(df, between, use_all):
    """Return (df_window, label). Restrict rows to a [start, stop) marker window unless --all."""
    if use_all:
        return df, "whole file"
    codes = df["OP CODE"].astype(str)
    types = df["OP TYPE"].astype(str)
    sp_mask = types.map(_is_signpost)
    candidates = [tuple(between)] if between else _DEFAULT_WINDOWS
    for start, stop in candidates:
        s_idx = df.index[sp_mask & (codes == start)]
        e_idx = df.index[sp_mask & (codes == stop)]
        if len(s_idx) and len(e_idx):
            s, e = s_idx[0], e_idx[e_idx > s_idx[0]][0] if len(e_idx[e_idx > s_idx[0]]) else e_idx[-1]
            return df.loc[s + 1 : e - 1], f"{start}..{stop}"
    if between:
        raise SystemExit(f"window markers {between} not found in CSV")
    return df, "whole file (no eager markers found)"


def aggregate(csv_path, between=None, use_all=False):
    """Walk the CSV timeline, pair name/name-end signposts via a stack, and sum DEVICE KERNEL
    DURATION per region (inclusive + self), aggregated across all occurrences. Returns a dict:
    {incl, self, counts, total_ns, unbracketed, window, csv} — no printing (importable)."""
    df = pd.read_csv(csv_path)
    if "OP CODE" not in df.columns or "OP TYPE" not in df.columns:
        raise SystemExit(f"unexpected CSV format; columns = {list(df.columns)}")
    dur_col = _duration_col(df)
    win, win_label = _slice_window(df, between, use_all)

    # Region base-names = codes that have a matching "/end"; only those open a region (everything else
    # that isn't a "/end" is a flat marker and is ignored for nesting).
    codes = win["OP CODE"].astype(str)
    region_names = {c[:-4] for c in codes if c.endswith("/end")}

    incl = defaultdict(float)  # name -> inclusive ns (region + nested children)
    self_ns = defaultdict(float)  # name -> self ns (ops directly in region)
    counts = defaultdict(int)  # name -> number of occurrences (opens)
    stack = []  # open region names, outermost first
    total_ns = 0.0
    unbracketed = 0.0

    for _, row in win.iterrows():
        op_type = row["OP TYPE"]
        code = str(row["OP CODE"])
        if _is_signpost(op_type):
            if code.endswith("/end"):
                name = code[:-4]
                if name in stack:  # pop to the matching open (tolerate a dangling inner region)
                    while stack and stack[-1] != name:
                        stack.pop()
                    if stack:
                        stack.pop()
            elif code in region_names:
                stack.append(code)
                counts[code] += 1
            # else: flat marker inside the window — ignore
            continue
        dur = pd.to_numeric(row.get(dur_col), errors="coerce")
        if pd.isna(dur):
            continue
        dur = float(dur)
        total_ns += dur
        if stack:
            self_ns[stack[-1]] += dur
            for nm in set(stack):  # inclusive: credit every open ancestor once
                incl[nm] += dur
        else:
            unbracketed += dur

    return {
        "incl": dict(incl),
        "self": dict(self_ns),
        "counts": dict(counts),
        "total_ns": total_ns,
        "unbracketed": unbracketed,
        "window": win_label,
        "csv": str(csv_path),
    }


def analyze(csv_path, between=None, use_all=False):
    agg = aggregate(csv_path, between=between, use_all=use_all)
    _print(agg["csv"], agg["window"], agg["total_ns"], agg["incl"], agg["self"], agg["counts"], agg["unbracketed"])
    return agg


def markdown_table(agg, top=None, only_prefix=None):
    """Render an aggregate() result as a Markdown table sorted by inclusive time. `top` limits rows;
    `only_prefix` (e.g. "moe.") keeps only matching component names."""
    incl, self_ns, counts, total = agg["incl"], agg["self"], agg["counts"], agg["total_ns"]
    if total <= 0:
        return "_no device-op kernel durations found in this window._\n"
    names = sorted(incl, key=lambda n: -incl[n])
    if only_prefix:
        names = [n for n in names if n.startswith(only_prefix)]
    if top:
        names = names[:top]
    pct = lambda ns: 100.0 * ns / total
    out = [
        f"_Window total device-kernel time: **{total/1e6:.3f} ms**. incl = region + nested children; "
        "self = ops directly in the region; n = occurrences (summed across all layers)._\n",
        "| Component | incl ms | incl % | self ms | self % | n | incl/occ ms |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for n in names:
        c = counts.get(n, 0)
        per = (incl[n] / 1e6 / c) if c else 0.0
        out.append(
            f"| `{n}` | {incl[n]/1e6:.3f} | {pct(incl[n]):.1f} | {self_ns.get(n,0)/1e6:.3f} | "
            f"{pct(self_ns.get(n,0)):.1f} | {c} | {per:.4f} |"
        )
    if agg.get("unbracketed"):
        out.append(
            f"| `<unbracketed>` | — | — | {agg['unbracketed']/1e6:.3f} | {pct(agg['unbracketed']):.1f} | — | — |"
        )
    return "\n".join(out) + "\n"


def _print(csv_path, win_label, total_ns, incl, self_ns, counts, unbracketed):
    tot_ms = total_ns / 1e6
    print(f"\n[qwen36-signpost] {csv_path}")
    print(f"  window: {win_label}   total device-kernel time: {tot_ms:.3f} ms   ({total_ns/1e3:.1f} us)\n")
    if total_ns == 0:
        print("  no device-op kernel durations in the window — did the profiler run with -r on a Tracy build?")
        return
    pct = lambda ns: 100.0 * ns / total_ns
    names = sorted(incl, key=lambda n: -incl[n])
    w = max((len(n) for n in names), default=12)
    print(
        f"  {'component':<{w}}  {'incl ms':>9} {'incl%':>6}  {'self ms':>9} {'self%':>6}  {'n':>4}  {'incl/occ ms':>11}"
    )
    print(f"  {'-'*w}  {'-'*9} {'-'*6}  {'-'*9} {'-'*6}  {'-'*4}  {'-'*11}")
    for n in names:
        c = counts.get(n, 0)
        per = (incl[n] / 1e6 / c) if c else 0.0
        print(
            f"  {n:<{w}}  {incl[n]/1e6:9.3f} {pct(incl[n]):6.1f}  "
            f"{self_ns.get(n, 0)/1e6:9.3f} {pct(self_ns.get(n, 0)):6.1f}  {c:4d}  {per:11.4f}"
        )
    if unbracketed:
        print(f"  {'<unbracketed>':<{w}}  {'':>9} {'':>6}  {unbracketed/1e6:9.3f} {pct(unbracketed):6.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--csv", default=None, help="ops_perf_results CSV (default: newest under generated/profiler/reports)"
    )
    ap.add_argument(
        "--between", nargs=2, metavar=("START", "STOP"), default=None, help="restrict to this marker window"
    )
    ap.add_argument("--all", action="store_true", help="analyze the whole file (ignore eager-window markers)")
    args = ap.parse_args(argv)
    csv_path = _find_csv(args.csv)
    analyze(csv_path, between=args.between, use_all=args.all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
