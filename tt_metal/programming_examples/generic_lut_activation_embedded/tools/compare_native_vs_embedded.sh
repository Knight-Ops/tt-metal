#!/bin/bash
# =============================================================================
# compare_native_vs_embedded.sh — Safe two-way ULP + runtime comparison
#
# For one activation, over the SAME input range and SAME tensor shape:
#   1. NATIVE   — TTNN's built-in op:  ttnn.<activation>(x)
#   2. EMBEDDED — our LUT kernel via run_csv.sh (coefficients from a CSV)
#
# Reports, for BOTH: MaxULP / MeanULP (extract_accuracy.py ground truth) and
# Tracy DEVICE compute time. Timing uses the SAME profiler path + extractor as
# run_csv.sh (TT_METAL_DEVICE_PROFILER + extract_profiler_compute_time), so the
# two µs numbers are apples-to-apples — nothing here is reinvented.
#
# NO ckernel-header surgery (unlike compare_three_way.sh) — safe on stock ops.
#
# Usage:
#   ./compare_native_vs_embedded.sh --activation tanh \
#       --csv $TT_POLY_FIT_DIR/data/coefficients/tanh_n8d8_s1_uniform_rational_ulp.csv \
#       [--precision fp32|bf16|both] [--tiles N]
#
# Requires: python_env with ttnn (./create_venv.sh); /usr/bin/python3 + numpy
#   for extract_accuracy.py; TT_POLY_FIT_DIR; the adhoc target built.
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"
WORK_DIR="$REPO_ROOT/tt_metal/programming_examples/generic_lut_activation_embedded"
TT_POLY_FIT_DIR="${TT_POLY_FIT_DIR:-/localdev/$USER/tt-polynomial-fitter}"
ACCURACY_SCRIPT="$TT_POLY_FIT_DIR/extract_accuracy.py"
SYSPY="/usr/bin/python3"
VENV="$REPO_ROOT/python_env/bin/activate"

# Reuse the existing Tracy extractor (extract_profiler_compute_time) — do not reinvent.
source "$WORK_DIR/profiler_helpers.sh"

ACTIVATION=""; CSV_FILE=""; PRECISION="fp32"; TILES=256
while [[ $# -gt 0 ]]; do
  case "$1" in
    --activation|-a) ACTIVATION="$2"; shift 2 ;;
    --csv|-c)        CSV_FILE="$2"; shift 2 ;;
    --precision|-p)  PRECISION="$2"; shift 2 ;;
    --tiles|-t)      TILES="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
[[ -z "$ACTIVATION" ]] && { echo "Error: --activation required"; exit 1; }
[[ -z "$CSV_FILE"   ]] && { echo "Error: --csv required"; exit 1; }
[[ ! -f "$CSV_FILE" ]] && { echo "Error: CSV not found: $CSV_FILE"; exit 1; }
[[ ! -f "$VENV"     ]] && { echo "Error: python_env missing — run ./create_venv.sh"; exit 1; }
[[ ! -f "$ACCURACY_SCRIPT" ]] && { echo "Error: extract_accuracy.py not found; set TT_POLY_FIT_DIR"; exit 1; }

# Native runner: accuracy dump over [lo,hi] + a profiled run on a TILES-sized
# (32*sqrt-ish) tensor so DEVICE_PROFILER captures the op's compute time.
NATIVE_PY="$(mktemp /tmp/native_run_XXXX.py)"
cat > "$NATIVE_PY" << 'PYEOF'
import sys, torch, ttnn, numpy as np
act, prec, lo, hi, out_csv, tiles = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], int(sys.argv[6])
dev = ttnn.open_device(device_id=0)
is_bf16 = prec == "bf16"
dt_tt = ttnn.bfloat16 if is_bf16 else ttnn.float32
dt_t  = torch.bfloat16 if is_bf16 else torch.float32
# --- accuracy dump over [lo,hi] ---
if is_bf16:
    bits = np.arange(65536, dtype=np.uint16)
    vals = np.frombuffer((bits.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
    m = np.isfinite(vals) & (vals >= lo) & (vals <= hi)
    x = torch.from_numpy(np.sort(vals[m])).bfloat16()
else:
    x = torch.linspace(lo, hi, 262144, dtype=torch.float32)
n = len(x)
with open(out_csv, 'w') as f:
    f.write('input,output\n')
    if n:
        pad = ((n + 1023) // 1024) * 1024
        xp = torch.zeros(pad, dtype=dt_t); xp[:n] = x
        xt = ttnn.from_torch(xp.reshape(1,1,1,-1), device=dev, layout=ttnn.TILE_LAYOUT, dtype=dt_tt)
        hw = ttnn.to_torch(getattr(ttnn, act)(xt)).squeeze().float().numpy()[:n]
        xn = x.float().numpy()
        for i in range(n): f.write(f'{xn[i]},{hw[i]}\n')
# --- profiled timing run: TILES tiles (32x32 each), shape 32 x (32*TILES) ---
W = 32 * tiles
xt_t = ttnn.from_torch(torch.rand(1,1,32,W,dtype=dt_t)*(hi-lo)+lo, device=dev, layout=ttnn.TILE_LAYOUT, dtype=dt_tt)
fn = getattr(ttnn, act)
for _ in range(3): fn(xt_t)          # warmup
ttnn.synchronize_device(dev)
fn(xt_t)                              # profiled
ttnn.synchronize_device(dev)
ttnn.close_device(dev); print(f"native_ok n={n}")
PYEOF
trap 'rm -f "$NATIVE_PY"' EXIT

run_one() {
  local prec="$1"
  # --- EMBEDDED (ours) via run_csv.sh: ULP + Tracy Prof at TILES ---
  local emb_dump="/tmp/emb_${ACTIVATION}_${prec}.csv" embout lo hi row ours_mae ours_ulp ours_meanulp ours_prof
  embout="$("$WORK_DIR/run_csv.sh" "$CSV_FILE" --activation "$ACTIVATION" --precision "$prec" \
              --tiles "$TILES" --runs 3 --dump-csv "$emb_dump" 2>&1)"
  lo="$(echo "$embout" | grep -oP 'DETECTED_RANGE_MIN=\K[-0-9.eE]+' | head -1)"
  hi="$(echo "$embout" | grep -oP 'DETECTED_RANGE_MAX=\K[-0-9.eE]+' | head -1)"
  row="$(echo "$embout" | grep -E '^custom_' | tail -1)"
  ours_mae="$(echo "$row" | awk '{print $3}')"; ours_ulp="$(echo "$row" | awk '{print $5}')"
  ours_meanulp="$(echo "$row" | awk '{print $6}')"; ours_prof="$(echo "$row" | awk '{print $7}')"
  [[ -z "$ours_ulp" ]] && ours_ulp="FAIL"

  # --- NATIVE (ttnn) over SAME range + same TILES, same profiler/extractor ---
  local nat_ulp="-" nat_meanulp="-" nat_us="-"
  if [[ -n "$lo" && -n "$hi" ]]; then
    local nat_dump="/tmp/nat_${ACTIVATION}_${prec}.csv" prof_dir="/tmp/nat_prof_${ACTIVATION}_${prec}"
    rm -rf "$prof_dir"; mkdir -p "$prof_dir"
    ( source "$VENV"; TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$prof_dir" \
        python3 "$NATIVE_PY" "$ACTIVATION" "$prec" "$lo" "$hi" "$nat_dump" "$TILES" >/dev/null 2>&1 )
    if [[ -s "$nat_dump" ]]; then
      local acc; acc="$("$SYSPY" "$ACCURACY_SCRIPT" "$ACTIVATION" "$nat_dump" 2>/dev/null | tail -1)"
      nat_ulp="$(echo "$acc" | cut -d, -f5)"; nat_meanulp="$(echo "$acc" | cut -d, -f6)"
      [[ -z "$nat_ulp" ]] && nat_ulp="NAT_FAIL"
    else nat_ulp="NAT_FAIL"; fi
    local pcsv="$prof_dir/.logs/profile_log_device.csv"
    [[ -f "$pcsv" ]] && nat_us="$(extract_profiler_compute_time "$pcsv" "$WORK_DIR")"
  fi

  printf '%-14s %-5s | NATIVE MaxULP %-12s %-9s | OURS MaxULP %-12s %-9s | range [%s,%s] %dt\n' \
    "$ACTIVATION" "$prec" "$nat_ulp" "${nat_us}us" "$ours_ulp" "$ours_prof" "${lo:-?}" "${hi:-?}" "$TILES"
}

echo "=== $ACTIVATION :: $(basename "$CSV_FILE") (NATIVE ttnn vs OURS embedded, ${TILES} tiles) ==="
if [[ "$PRECISION" == "both" ]]; then run_one bf16; run_one fp32; else run_one "$PRECISION"; fi
