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
ref_csv = os.path.join(TUT, "bench_rational_reference.csv" if is_rational else "bench_reference.csv")

# reference (rounded-input lookup)
ref = {}
try:
    for a, b in list(csv.reader(open(ref_csv)))[1:]:
        ref[round(float(a), 5)] = float(b)
except Exception:
    ref = {}

err = 9.99
try:
    rows = list(csv.reader(open(dump)))[1:]
    e = 0.0
    n = 0
    for a, b in rows:
        x = round(float(a), 5)
        if x in ref:
            e = max(e, abs(float(b) - ref[x]))
            n += 1
    err = e if n else 9.99
except Exception:
    err = 9.99

# seg degrees from the generated header
hdr_name = "bench_rational_lut.h" if is_rational else "bench_lut.h"
hdr_path = os.path.join(TUT, "kernels/common", hdr_name)
try:
    hdr = open(hdr_path).read()
    sd = [int(x) for x in re.search(r"BENCH_SEGMENT_DEGREES\[\d+\]\s*=\s*\{([^}]*)\}", hdr).group(1).split(",")]
except Exception:
    sd = [8]

adaptive = "adaptive" in rung
parity = "parity" in rung
degs = sd if adaptive else [max(sd)] * len(sd)
fma = fma_count(degs, parity=parity)
insns = count_sfpu_insns(obj)
print(f"{fma} {insns if insns is not None else 'NA'} {err:.3e}")
