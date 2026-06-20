#!/bin/bash
# =============================================================================
# compare_native_vs_embedded.sh — Safe two-way accuracy comparison
#
# Compares, over the SAME input range, for one activation:
#   1. NATIVE   — TTNN's built-in op:  ttnn.<activation>(x)
#   2. EMBEDDED — our LUT kernel via run_csv.sh (coefficients from a CSV)
#
# Reports MAE + MaxULP for both (extract_accuracy.py ground truth) plus the
# embedded Tracy kernel time. NO ckernel-header surgery — unlike
# compare_three_way.sh this never touches tt_metal/hw/ckernels, so it is safe
# to run on stock activations that have no drop-in installed.
#
# Usage:
#   ./compare_native_vs_embedded.sh --activation tanh \
#       --csv $TT_POLY_FIT_DIR/data/coefficients/tanh_n8d8_s1_uniform_rational_ulp.csv \
#       [--precision fp32|bf16|both]
#
# Batch (loop in your shell):
#   while IFS=, read -r a p c; do
#     ./compare_native_vs_embedded.sh --activation "$a" --precision "$p" \
#         --csv "$TT_POLY_FIT_DIR/data/coefficients/$c"
#   done < worklist.csv
#
# Requires:
#   - python_env with ttnn (./create_venv.sh) for the NATIVE run
#   - /usr/bin/python3 with numpy for extract_accuracy.py (ground truth / ULP)
#   - TT_POLY_FIT_DIR pointing at the tt-polynomial-fitter checkout
#   - The adhoc target built (see README "Setup & Build")
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"
WORK_DIR="$REPO_ROOT/tt_metal/programming_examples/generic_lut_activation_embedded"
TT_POLY_FIT_DIR="${TT_POLY_FIT_DIR:-/localdev/$USER/tt-polynomial-fitter}"
ACCURACY_SCRIPT="$TT_POLY_FIT_DIR/extract_accuracy.py"
SYSPY="/usr/bin/python3"        # numpy + correct BF16 ULP spacing (python_env torch is wrong)
VENV="$REPO_ROOT/python_env/bin/activate"

ACTIVATION=""; CSV_FILE=""; PRECISION="fp32"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --activation|-a) ACTIVATION="$2"; shift 2 ;;
    --csv|-c)        CSV_FILE="$2"; shift 2 ;;
    --precision|-p)  PRECISION="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
[[ -z "$ACTIVATION" ]] && { echo "Error: --activation required"; exit 1; }
[[ -z "$CSV_FILE"   ]] && { echo "Error: --csv required"; exit 1; }
[[ ! -f "$CSV_FILE" ]] && { echo "Error: CSV not found: $CSV_FILE"; exit 1; }
[[ ! -f "$VENV"     ]] && { echo "Error: python_env missing — run ./create_venv.sh"; exit 1; }
[[ ! -f "$ACCURACY_SCRIPT" ]] && { echo "Error: extract_accuracy.py not found; set TT_POLY_FIT_DIR"; exit 1; }

# Native runner: ttnn.<op> over [lo,hi]; dense fp32 / exhaustive bf16. Dumps input,output CSV.
NATIVE_PY="$(mktemp /tmp/native_run_XXXX.py)"
cat > "$NATIVE_PY" << 'PYEOF'
import sys, torch, ttnn, numpy as np
act, prec, lo, hi, out_csv = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
dev = ttnn.open_device(device_id=0)
is_bf16 = prec == "bf16"
dt_tt = ttnn.bfloat16 if is_bf16 else ttnn.float32
dt_t  = torch.bfloat16 if is_bf16 else torch.float32
if is_bf16:
    bits = np.arange(65536, dtype=np.uint16)
    vals = np.frombuffer((bits.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
    m = np.isfinite(vals) & (vals >= lo) & (vals <= hi)
    x = torch.from_numpy(np.sort(vals[m])).bfloat16()
else:
    x = torch.linspace(lo, hi, 262144, dtype=torch.float32)
n = len(x)
if n == 0:
    open(out_csv, 'w').write('input,output\n'); ttnn.close_device(dev); print("native_ok n=0"); sys.exit(0)
pad = ((n + 1023) // 1024) * 1024
xp = torch.zeros(pad, dtype=dt_t); xp[:n] = x
xt = ttnn.from_torch(xp.reshape(1, 1, 1, -1), device=dev, layout=ttnn.TILE_LAYOUT, dtype=dt_tt)
hw = ttnn.to_torch(getattr(ttnn, act)(xt)).squeeze().float().numpy()[:n]
xn = x.float().numpy()
with open(out_csv, 'w') as f:
    f.write('input,output\n')
    for i in range(n): f.write(f'{xn[i]},{hw[i]}\n')
ttnn.close_device(dev); print(f"native_ok n={n}")
PYEOF
trap 'rm -f "$NATIVE_PY"' EXIT

run_one() {
  local prec="$1"
  echo "=== $ACTIVATION ($prec) :: $(basename "$CSV_FILE") ==="
  # --- EMBEDDED (ours) via run_csv.sh ---
  local emb_dump="/tmp/emb_${ACTIVATION}_${prec}.csv"
  local embout
  embout="$("$WORK_DIR/run_csv.sh" "$CSV_FILE" --activation "$ACTIVATION" --precision "$prec" \
              --tiles 256 --runs 1 --dump-csv "$emb_dump" 2>&1)"
  local lo hi row ours_mae ours_ulp ours_prof
  lo="$(echo "$embout" | grep -oP 'DETECTED_RANGE_MIN=\K[-0-9.eE]+' | head -1)"
  hi="$(echo "$embout" | grep -oP 'DETECTED_RANGE_MAX=\K[-0-9.eE]+' | head -1)"
  row="$(echo "$embout" | grep -E '^custom_' | tail -1)"
  ours_mae="$(echo "$row" | awk '{print $3}')"; ours_ulp="$(echo "$row" | awk '{print $5}')"
  ours_prof="$(echo "$row" | awk '{print $7}')"
  [[ -z "$ours_mae" ]] && ours_mae="FAIL"

  # --- NATIVE (ttnn) over the SAME range ---
  local nat_mae="-" nat_ulp="-" npts="-"
  if [[ -n "$lo" && -n "$hi" ]]; then
    local nat_dump="/tmp/nat_${ACTIVATION}_${prec}.csv" natrun acc
    natrun="$(source "$VENV" && python3 "$NATIVE_PY" "$ACTIVATION" "$prec" "$lo" "$hi" "$nat_dump" 2>/dev/null)"
    npts="$(echo "$natrun" | grep -oP 'native_ok n=\K[0-9]+' | head -1)"
    if [[ -s "$nat_dump" ]]; then
      acc="$("$SYSPY" "$ACCURACY_SCRIPT" "$ACTIVATION" "$nat_dump" 2>/dev/null | tail -1)"
      nat_mae="$(echo "$acc" | cut -d, -f1)"; nat_ulp="$(echo "$acc" | cut -d, -f5)"
      [[ -z "$nat_mae" ]] && nat_mae="NAT_FAIL"
    else nat_mae="NAT_FAIL"; fi
  else nat_mae="NO_RANGE"; fi

  printf '%-14s %-5s | native MAE %-12s ULP %-12s | ours MAE %-12s ULP %-10s | ours %-9s (range [%s,%s], n=%s)\n' \
    "$ACTIVATION" "$prec" "$nat_mae" "$nat_ulp" "$ours_mae" "$ours_ulp" "$ours_prof" "${lo:-?}" "${hi:-?}" "${npts:-?}"
}

if [[ "$PRECISION" == "both" ]]; then
  run_one bf16
  run_one fp32
else
  run_one "$PRECISION"
fi
