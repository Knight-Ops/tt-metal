"""Score one rung: correctness (max abs err vs reference) + static analysis.

Usage: score.py <rung> <dump_csv> [<trisc_obj>]
Prints: "<fma> <sfpu_insns|NA> <max_abs_err>"
"""
import sys
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TUT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from static_analysis import fma_count, count_sfpu_insns  # noqa: E402

rung = sys.argv[1]
dump = sys.argv[2]
obj = sys.argv[3] if len(sys.argv) > 3 else ""

is_rational = rung.startswith("r")

# seg degrees + LUT from the generated header
hdr_name = "bench_rational_lut.h" if is_rational else "bench_lut.h"
hdr_path = os.path.join(TUT, "kernels/common", hdr_name)
hdr = ""
try:
    hdr = open(hdr_path).read()
    sd = [int(x) for x in re.search(r"BENCH_SEGMENT_DEGREES\[\d+\]\s*=\s*\{([^}]*)\}", hdr).group(1).split(",")]
except Exception:
    sd = [8]


def _gt_evaluator():
    """Build an exact segment-aware piecewise-polynomial evaluator from bench_lut.h.

    The tutorial's "reference" IS the polynomial, so we evaluate it directly at
    each measured x instead of looking up a precomputed CSV. This is robust to
    sampling-grid offset between the device input ramp and the reference grid,
    and to the segment-boundary discontinuities of the piecewise function (a
    CSV nearest-key lookup straddling a discontinuity reports a spurious error).
    """
    num = int(re.search(r"BENCH_NUM_SEGMENTS\s*=\s*(\d+)", hdr).group(1))
    maxd = int(re.search(r"BENCH_MAX_DEGREE\s*=\s*(\d+)", hdr).group(1))
    lut = [float(v.strip().rstrip("f")) for v in re.search(r"BENCH_LUT\s*=\s*\{\{([^}]*)\}\}", hdr).group(1).split(",")]
    bounds = lut[: num + 1]
    coeffs = lut[num + 1 :]

    def gt(x):
        # segment s = last boundary <= x, clamped to [0, num-1]
        s = 0
        for i in range(num):
            if x >= bounds[i]:
                s = i
        base = s * (maxd + 1)
        acc = 0.0
        for d in range(maxd, -1, -1):
            acc = acc * x + coeffs[base + d]
        return acc

    return gt


# Correctness: max abs err of measured output vs the exact piecewise polynomial.
err = 9.99
try:
    gt = _gt_evaluator()
    rows = list(csv.reader(open(dump)))[1:]
    e = 0.0
    n = 0
    for a, b in rows:
        e = max(e, abs(float(b) - gt(float(a))))
        n += 1
    err = e if n else 9.99
except Exception:
    # Fallback: brittle rounded-key CSV lookup (legacy path).
    ref_csv = os.path.join(TUT, "bench_rational_reference.csv" if is_rational else "bench_reference.csv")
    ref = {}
    try:
        for a, b in list(csv.reader(open(ref_csv)))[1:]:
            ref[round(float(a), 5)] = float(b)
        e = 0.0
        n = 0
        for a, b in list(csv.reader(open(dump)))[1:]:
            x = round(float(a), 5)
            if x in ref:
                e = max(e, abs(float(b) - ref[x]))
                n += 1
        err = e if n else 9.99
    except Exception:
        err = 9.99

adaptive = "adaptive" in rung
# parity rungs evaluate in the x^2 basis; the adaptive rung is parity+adaptive.
parity = ("parity" in rung) or ("adaptive" in rung)
degs = sd if adaptive else [max(sd)] * len(sd)
fma = fma_count(degs, parity=parity)
insns = count_sfpu_insns(obj)
print(f"{fma} {insns if insns is not None else 'NA'} {err:.3e}")
