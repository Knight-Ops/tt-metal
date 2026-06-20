#!/bin/bash
# =============================================================================
# sweep_best_native_vs_embedded.sh — Batch NATIVE-vs-OURS over a best*.csv
#
# Resolves every (activation, precision) row of a tt-polynomial-fitter best*.csv
# to its best-ULP coefficient CSV, then runs compare_native_vs_embedded.sh for
# each — producing a full ULP + runtime table of TTNN native vs our LUT kernel.
#
# Works with best.csv, best95.csv (cheapest within 95% of peak accuracy),
# best99.csv, etc. — the selection policy lives in the CSV, this just drives it.
#
# Resolution is by HEADER NAME (not fixed column indices) and uses the
# best_ulp_source_metric column for the filename suffix, so it stays correct
# across best.csv schema changes (the bug that broke pr_submission.sh).
#
# Usage:
#   ./sweep_best_native_vs_embedded.sh --best-csv $TT_POLY_FIT_DIR/best95.csv \
#       [--precision both|bf16|fp32] [--activations tanh,asin,gelu] [--tiles N]
#
# Output: a table to stdout + /tmp/sweep_<best-csv-stem>_<precision>.txt
#
# Requires: python_env (ttnn), /usr/bin/python3 + numpy, TT_POLY_FIT_DIR,
#           the adhoc target built (see README "Setup & Build").
# =============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_POLY_FIT_DIR="${TT_POLY_FIT_DIR:-/localdev/$USER/tt-polynomial-fitter}"
COEFF_DIR="$TT_POLY_FIT_DIR/data/coefficients"

BEST_CSV=""; PREC_FILTER="both"; ACT_FILTER=""; TILES=256
while [[ $# -gt 0 ]]; do
  case "$1" in
    --best-csv|-b)    BEST_CSV="$2"; shift 2 ;;
    --precision|-p)   PREC_FILTER="$2"; shift 2 ;;
    --activations|-a) ACT_FILTER="$2"; shift 2 ;;
    --tiles|-t)       TILES="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
[[ -z "$BEST_CSV" ]] && BEST_CSV="$TT_POLY_FIT_DIR/best.csv"
[[ ! -f "$BEST_CSV" ]] && { echo "Error: best csv not found: $BEST_CSV"; exit 1; }

stem="$(basename "${BEST_CSV%.csv}")"
WL="$(mktemp /tmp/sweep_wl_XXXX.txt)"
OUT="/tmp/sweep_${stem}_${PREC_FILTER}.txt"
trap 'rm -f "$WL"' EXIT

# Resolve best_ulp coeff filename per row, by header name + source_metric.
python3 - "$BEST_CSV" "$COEFF_DIR" "$PREC_FILTER" "$ACT_FILTER" > "$WL" <<'PYEOF'
import csv, os, sys
best_csv, coeff_dir, prec_filter, act_filter = sys.argv[1:5]
acts = set(a for a in act_filter.split(",") if a) if act_filter else None
rows = list(csv.reader(open(best_csv))); h = rows[0]; idx = {n: i for i, n in enumerate(h)}
need = ["best_ulp_degree","best_ulp_num_segments","best_ulp_segmentation","best_ulp_fitting","best_ulp_source_metric"]
if any(n not in idx for n in need):
    sys.stderr.write("best csv missing best_ulp_* columns (regenerate with best_all.sh)\n"); sys.exit(1)
hit = miss = 0
for r in rows[1:]:
    act, prec = r[0], r[1]
    if prec_filter != "both" and prec != prec_filter: continue
    if acts and act not in acts: continue
    deg = r[idx["best_ulp_degree"]]; segs = r[idx["best_ulp_num_segments"]]
    seg = r[idx["best_ulp_segmentation"]]; fit = r[idx["best_ulp_fitting"]]
    sm = r[idx["best_ulp_source_metric"]]
    approx = f"n{deg.replace('/','d')}" if "/" in deg else f"p{deg}"
    fname = f"{act}_{approx}_s{segs}_{seg}_{fit}_{sm}.csv"
    if os.path.exists(os.path.join(coeff_dir, fname)):
        print(f"{act},{prec},{fname}"); hit += 1
    else:
        sys.stderr.write(f"  MISS {act} {prec} -> {fname}\n"); miss += 1
sys.stderr.write(f"resolved {hit} / {hit+miss}\n")
PYEOF

echo "=== sweep $stem ($PREC_FILTER) :: $(grep -c . "$WL") configs ==="
: > "$OUT"
n=0; total=$(grep -c . "$WL")
while IFS=, read -r act prec csv; do
  [[ -z "$act" ]] && continue
  n=$((n+1)); echo "[$n/$total] $act $prec" >&2
  "$SCRIPT_DIR/compare_native_vs_embedded.sh" --activation "$act" --precision "$prec" \
     --csv "$COEFF_DIR/$csv" --tiles "$TILES" 2>/dev/null | grep -E '\| NATIVE' | tee -a "$OUT"
done < "$WL"
echo "=== done: $n configs -> $OUT ==="
