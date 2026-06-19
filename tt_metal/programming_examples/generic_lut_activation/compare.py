#!/usr/bin/env python3
"""
Hardware-measured TTNN vs LUT comparison table.

Produces side-by-side BF16/FP32 tables comparing TTNN (native SFPU precise mode)
against LUT (piecewise polynomial/rational from sweep_best_all.sh) using actual
hardware-measured accuracy and timing data.

Data sources:
  - LUT:  generic_lut_activation_embedded/data/{arch}/best/{activation}.csv
  - TTNN: generic_lut_activation/data/{arch}/native_sfpu_results.csv
  - Ranges: sweep_config.dat ACTIVATION_RANGES

Usage:
    python3 compare.py                          # All activations, all metrics
    python3 compare.py --metric ulp             # Only BEST BY ULP table
    python3 compare.py --activation gelu        # Single activation
    python3 compare.py --arch wormhole          # Different arch
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path


# ── ANSI colors ──────────────────────────────────────────────────────────────

GREEN = '\033[32m'
YELLOW = '\033[33m'
RED = '\033[31m'
GREY = '\033[90m'
BOLD = '\033[1m'
RESET = '\033[0m'
BG_PURPLE = '\033[48;2;50;30;70m'
BG_YELLOW = '\033[48;2;60;55;30m'  # faint dark-yellow tint (subtle on dark terminals)
BG_RESET = '\033[49m'

_ANSI_RE = re.compile(r'\033\[[^m]*m')


# ── Data loading ─────────────────────────────────────────────────────────────

def parse_activation_ranges(config_path):
    """Parse ACTIVATION_RANGES from sweep_config.dat -> {activation: (lo, hi)}."""
    ranges = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('ACTIVATION_RANGES='):
                entries = line.split('=', 1)[1]
                for entry in entries.split(','):
                    parts = entry.split(':')
                    if len(parts) == 3:
                        ranges[parts[0]] = (float(parts[1]), float(parts[2]))
    return ranges


def load_ttnn_results(results_csv, shape='yolov4', sfpu_mode='precise'):
    """Load TTNN native SFPU results from single CSV -> dict keyed by (activation, precision)."""
    data = {}
    if not os.path.exists(results_csv):
        return data
    with open(results_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['sfpu_mode'] != sfpu_mode or row['shape'] != shape:
                continue
            if row['status'] != 'pass':
                continue
            key = (row['activation'], row['precision'])
            data[key] = {
                'mae': float(row['mae']),
                'max_error': float(row['max_error']),
                'max_ulp': float(row['max_ulp_error']),
                'time_us': float(row['time_profiler_us']),
            }
    return data


def load_ttnn_results_from_dir(native_sfpu_dir, shape='yolov4', sfpu_mode='precise'):
    """Load TTNN native SFPU results from native_sfpu/*.csv -> dict keyed by (activation, precision)."""
    data = {}
    if not os.path.isdir(native_sfpu_dir):
        return data
    for csv_file in sorted(Path(native_sfpu_dir).glob('*.csv')):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['sfpu_mode'] != sfpu_mode or row['shape'] != shape:
                    continue
                if row['status'] != 'pass':
                    continue
                key = (row['activation'], row['precision'])
                data[key] = {
                    'mae': float(row['mae']),
                    'max_error': float(row['max_error']),
                    'max_ulp': float(row['max_ulp_error']),
                    'time_us': float(row['time_profiler_us']),
                }
    return data


def load_lut_best_results(best_dir, shape='yolov4'):
    """Load LUT best results from per-activation CSVs -> dict keyed by (activation, precision, metric).

    CSV format (new): activation,precision,metric,depth,degree,segmentation,shape,...
    CSV format (old): activation,precision,depth,degree,segmentation,shape,...
    """
    data = {}
    if not os.path.isdir(best_dir):
        return data
    for csv_file in sorted(Path(best_dir).glob('*.csv')):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['shape'] != shape:
                    continue
                if row['status'] != 'pass':
                    continue
                act = row['activation']
                prec = row['precision']
                metric = row.get('metric', 'unknown')  # old CSVs lack metric column
                key = (act, prec, metric)
                data[key] = {
                    'mae': float(row['mae']),
                    'max_error': float(row['max_error']),
                    'max_ulp': float(row['max_ulp']),
                    'time_us': float(row['time_profiler_us']),
                    'config': f"{row['depth']}s/d{row['degree']}/{row['segmentation'][:3]}",
                }
    return data


# ── Table rendering ──────────────────────────────────────────────────────────

# Column indices within one side (0-indexed)
#  0: Activation   1: Range
#  2: TTNN MAE     3: TTNN Max     4: TTNN ULP     5: TTNN us
#  6: LUT MAE      7: LUT Max      8: LUT ULP      9: LUT us     10: Config
COL_TTNN_ULP = 4
COL_LUT_ULP = 8
ULP_COLS = {COL_TTNN_ULP, COL_LUT_ULP}
COL_TTNN_US = 5
COL_LUT_US = 9
RUNTIME_COLS = {COL_TTNN_US, COL_LUT_US}

SIDE_W = [13, 14, 10, 10, 10, 9, 10, 10, 10, 9, 16]
SIDE_HEADERS = [
    "Activation", "Range",
    "TTNN MAE", "TTNN Max", "TTNN ULP", "TTNN \u00b5s",
    "LUT MAE", "LUT Max", "LUT ULP", "LUT \u00b5s", "Config",
]
SIDE_ALIGNS = ['l', 'l'] + ['r'] * 9

GAP_W = 3  # gap between BF16 and FP32 sides


def _vlen(s):
    """Visible length after stripping ANSI codes."""
    return len(_ANSI_RE.sub('', s))


def _cell(val, width, align='r'):
    """Format a cell: 1-space padding each side."""
    content_w = width - 2
    vl = _vlen(val)
    if vl > content_w:
        val = val[:content_w]
        vl = content_w
    pad = content_w - vl
    if align == 'r':
        return ' ' + ' ' * pad + val + ' '
    else:
        return ' ' + val + ' ' * pad + ' '


def _side_hline(widths, fill='─', mid='─'):
    """Build horizontal line segment for one side."""
    return mid.join(fill * w for w in widths)


def _full_hline(left, mid, right, fill='─'):
    """Build full-width horizontal line string for disconnected tables."""
    side = _side_hline(SIDE_W, fill, mid)
    gap = ' ' * GAP_W
    return left + side + right + gap + left + side + right


def _side_row(values, aligns=None):
    """Build one side's cell string with background coloring on ULP and runtime cols."""
    if aligns is None:
        aligns = SIDE_ALIGNS
    # For ULP columns, use foreground-only reset to preserve background
    FG_RESET = '\033[39m\033[22m'
    parts = []
    for i, (val, w) in enumerate(zip(values, SIDE_W)):
        a = aligns[i] if i < len(aligns) else 'r'
        cell_str = _cell(val, w, a)
        if i in ULP_COLS:
            # Replace full RESET with foreground-only reset to keep purple BG
            cell_str = cell_str.replace(RESET, FG_RESET)
            cell_str = f"{BG_PURPLE}{cell_str}{BG_RESET}"
        elif i in RUNTIME_COLS:
            # Replace full RESET with foreground-only reset to keep yellow BG
            cell_str = cell_str.replace(RESET, FG_RESET)
            cell_str = f"{BG_YELLOW}{cell_str}{BG_RESET}"
        else:
            # Fully reset all attributes (FG, BG, bold) to prevent color bleed from previous cells
            cell_str = f"{RESET}{cell_str}"
        parts.append(cell_str)

    # Join with separators, applying BG to separators between special cols
    result = ''
    for i, part in enumerate(parts):
        result += part
        if i < len(parts) - 1:
            # Separator between col i and col i+1
            # Only color separator if BOTH adjacent columns share the same special BG
            if i in ULP_COLS and (i + 1) in ULP_COLS:
                sep = f"{BG_PURPLE}│{BG_RESET}"
            elif i in RUNTIME_COLS and (i + 1) in RUNTIME_COLS:
                sep = f"{BG_YELLOW}│{BG_RESET}"
            else:
                sep = f"{BG_RESET}│"
            result += sep
    return result


def _full_row(bf16_vals, fp32_vals, aligns=None):
    """Print a full row: |BF16 side|   |FP32 side| (disconnected tables)."""
    bf16 = _side_row(bf16_vals, aligns)
    fp32 = _side_row(fp32_vals, aligns)
    gap = ' ' * GAP_W
    print('│' + bf16 + '│' + gap + '│' + fp32 + '│')


def _grey_sep():
    """Print grey separator line with yellow BG on runtime column positions (disconnected)."""
    def grey_side():
        parts = []
        for i, w in enumerate(SIDE_W):
            seg = f"{GREY}{'─' * w}{RESET}"
            if i in RUNTIME_COLS:
                seg = f"{BG_YELLOW}{seg}{BG_RESET}"
            parts.append(seg)
        # Join with separators - only color if BOTH adjacent columns share same BG
        result = ''
        for i, part in enumerate(parts):
            result += part
            if i < len(parts) - 1:
                if i in RUNTIME_COLS and (i + 1) in RUNTIME_COLS:
                    sep = f"{BG_YELLOW}┼{BG_RESET}"
                else:
                    sep = '┼'
                result += sep
        return result
    gap = ' ' * GAP_W
    print('├' + grey_side() + '┤' + gap + '├' + grey_side() + '┤')


def _hline_with_bg(left, mid, right, fill='─'):
    """Horizontal line with purple BG on ULP columns, yellow BG on runtime columns (disconnected)."""
    def side():
        parts = []
        for i, w in enumerate(SIDE_W):
            seg = fill * w
            if i in ULP_COLS:
                seg = f"{BG_PURPLE}{seg}{BG_RESET}"
            elif i in RUNTIME_COLS:
                seg = f"{BG_YELLOW}{seg}{BG_RESET}"
            parts.append(seg)
        result = ''
        for i, part in enumerate(parts):
            result += part
            if i < len(parts) - 1:
                # Only color separator if BOTH adjacent columns share the same special BG
                if i in ULP_COLS and (i + 1) in ULP_COLS:
                    sep = f"{BG_PURPLE}{mid}{BG_RESET}"
                elif i in RUNTIME_COLS and (i + 1) in RUNTIME_COLS:
                    sep = f"{BG_YELLOW}{mid}{BG_RESET}"
                else:
                    sep = mid
                result += sep
        return result
    gap = ' ' * GAP_W
    return left + side() + right + gap + left + side() + right


# ── Formatting & coloring ───────────────────────────────────────────────────

def fmt_val(val):
    """Format a numeric value for display."""
    if val is None or val == float('inf'):
        return '--'
    return f"{val:.2e}"


def fmt_time(val):
    """Format timing value in microseconds."""
    if val is None:
        return '--'
    if val < 10:
        return f"{val:.2f}"
    elif val < 100:
        return f"{val:.1f}"
    else:
        return f"{val:.0f}"


def fmt_range(lo, hi):
    """Format range as compact string."""
    def f(v):
        if v == int(v):
            return str(int(v))
        s = f"{v:.4f}".rstrip('0').rstrip('.')
        return s
    return f"[{f(lo)},{f(hi)}]"


def color_vs_ttnn(lut_val, ttnn_val, lut_str):
    """Color LUT value vs TTNN: green=better, yellow=similar(2x), red=worse."""
    if lut_val is None or ttnn_val is None or lut_val == float('inf') or ttnn_val == float('inf'):
        return lut_str
    if ttnn_val == 0:
        return f"{GREEN}{lut_str}{RESET}" if lut_val == 0 else f"{RED}{lut_str}{RESET}"
    ratio = lut_val / ttnn_val
    if ratio <= 0.5:
        return f"{GREEN}{BOLD}{lut_str}{RESET}"
    elif ratio <= 1.0:
        return f"{GREEN}{lut_str}{RESET}"
    elif ratio <= 2.0:
        return f"{YELLOW}{lut_str}{RESET}"
    else:
        return f"{RED}{lut_str}{RESET}"


def color_time_vs(lut_us, ttnn_us, lut_str):
    """Color LUT timing vs TTNN: green=faster, yellow=similar, red=slower."""
    if lut_us is None or ttnn_us is None:
        return lut_str
    if ttnn_us == 0:
        return lut_str
    ratio = lut_us / ttnn_us
    if ratio <= 0.75:
        return f"{GREEN}{BOLD}{lut_str}{RESET}"
    elif ratio <= 1.0:
        return f"{GREEN}{lut_str}{RESET}"
    elif ratio <= 1.5:
        return f"{YELLOW}{lut_str}{RESET}"
    else:
        return f"{RED}{lut_str}{RESET}"


def is_win(our_val, their_val):
    """Check if our_val <= their_val (win or tie)."""
    if our_val is None or their_val is None:
        return False
    if our_val == float('inf') or their_val == float('inf'):
        return False
    if their_val == 0:
        return our_val == 0
    return our_val / their_val <= 1.0


def is_tie(val1, val2, tolerance=0.05):
    """Check if two values are within tolerance of each other."""
    if val1 is None or val2 is None:
        return False
    if val1 == float('inf') or val2 == float('inf'):
        return False
    if val1 == 0 and val2 == 0:
        return True
    if val1 == 0 or val2 == 0:
        return False
    ratio = val1 / val2
    return (1.0 - tolerance) <= ratio <= (1.0 + tolerance)


# ── Main table generation ───────────────────────────────────────────────────

def print_comparison_table(title, sort_key, metric_name, activations, ranges,
                           ttnn_data, lut_data):
    """Print one comparison table with BF16 (left) | gap | FP32 (right) sides."""
    print("=" * 100)
    print(f"  COMPARISON: {title} (Hardware-Measured)")
    print("=" * 100)
    print(f"  Legend: {GREEN}LUT better than TTNN{RESET} | "
          f"{YELLOW}Similar (within 2x){RESET} | "
          f"{RED}LUT worse than TTNN{RESET}")
    print(f"  Data: yolov4 shape | TTNN sfpu_mode=precise")
    print()

    # Print BF16 / FP32 labels centered over each side
    side_w = sum(SIDE_W) + len(SIDE_W) - 1
    bf16_pad = (side_w - 4) // 2
    fp32_pad = (side_w - 4) // 2
    print(' ' + ' ' * bf16_pad + f"{BOLD}BF16{RESET}"
          + ' ' * (side_w - bf16_pad - 4)
          + '   '
          + ' ' * fp32_pad + f"{BOLD}FP32{RESET}")

    # Top border
    print(_hline_with_bg('┌', '┬', '┐'))

    # Header row
    _full_row(SIDE_HEADERS, SIDE_HEADERS, aligns=SIDE_ALIGNS)

    # Header separator
    print(_hline_with_bg('├', '┼', '┤'))

    # Win/tie counters per precision side
    # indices 0-2 = TTNN wins (mae/max/ulp), 3-5 = LUT wins (mae/max/ulp)
    bf16_wins = [0] * 6;  bf16_ties = [0] * 6;  bf16_totals = [0] * 6
    fp32_wins = [0] * 6;  fp32_ties = [0] * 6;  fp32_totals = [0] * 6

    for act in activations:
        range_str = '--'
        if act in ranges:
            lo, hi = ranges[act]
            range_str = fmt_range(lo, hi)

        sides = []
        for prec, win_arr, tie_arr, tot_arr in [
            ('bf16', bf16_wins, bf16_ties, bf16_totals),
            ('fp32', fp32_wins, fp32_ties, fp32_totals),
        ]:
            ttnn = ttnn_data.get((act, prec))
            # Try metric-specific key first, fall back to 'unknown' (old CSV format)
            lut = lut_data.get((act, prec, metric_name))
            if lut is None:
                lut = lut_data.get((act, prec, 'unknown'))

            ttnn_mae = ttnn['mae'] if ttnn else None
            ttnn_max = ttnn['max_error'] if ttnn else None
            ttnn_ulp = ttnn['max_ulp'] if ttnn else None
            ttnn_us = ttnn['time_us'] if ttnn else None

            lut_mae = lut['mae'] if lut else None
            lut_max = lut['max_error'] if lut else None
            lut_ulp = lut['max_ulp'] if lut else None
            lut_us = lut['time_us'] if lut else None
            config = lut['config'] if lut else '--'

            # Format raw values
            ttnn_mae_s = fmt_val(ttnn_mae)
            ttnn_max_s = fmt_val(ttnn_max)
            ttnn_ulp_s = fmt_val(ttnn_ulp)
            ttnn_us_s = fmt_time(ttnn_us)

            lut_mae_s = fmt_val(lut_mae)
            lut_max_s = fmt_val(lut_max)
            lut_ulp_s = fmt_val(lut_ulp)
            lut_us_s = fmt_time(lut_us)

            # Color LUT values vs TTNN
            lut_mae_d = color_vs_ttnn(lut_mae, ttnn_mae, lut_mae_s)
            lut_max_d = color_vs_ttnn(lut_max, ttnn_max, lut_max_s)
            lut_ulp_d = color_vs_ttnn(lut_ulp, ttnn_ulp, lut_ulp_s)
            lut_us_d = color_time_vs(lut_us, ttnn_us, lut_us_s)

            # Count wins/ties
            pairs = [(lut_mae, ttnn_mae), (lut_max, ttnn_max), (lut_ulp, ttnn_ulp)]
            for i, (lut_v, ttnn_v) in enumerate(pairs):
                if lut_v is not None and ttnn_v is not None:
                    tot_arr[i] += 1
                    if is_win(ttnn_v, lut_v):
                        win_arr[i] += 1
                    if is_tie(ttnn_v, lut_v):
                        tie_arr[i] += 1
                    tot_arr[i + 3] += 1
                    if is_win(lut_v, ttnn_v):
                        win_arr[i + 3] += 1
                    if is_tie(lut_v, ttnn_v):
                        tie_arr[i + 3] += 1

            sides.append([
                act, range_str,
                ttnn_mae_s, ttnn_max_s, ttnn_ulp_s, ttnn_us_s,
                lut_mae_d, lut_max_d, lut_ulp_d, lut_us_d, config,
            ])

        _full_row(sides[0], sides[1])

    # Summary separator
    print(_hline_with_bg('├', '┼', '┤'))

    def fmt_count(w, t, threshold=50):
        if t == 0:
            return '--'
        pct = 100 * w / t
        return f"{GREEN}{w}/{t}{RESET}" if pct >= threshold else f"{w}/{t}"

    def fmt_tie_count(ti, t):
        if t == 0:
            return '--'
        return f"{YELLOW}{ti}/{t}{RESET}" if ti > 0 else f"{ti}/{t}"

    def fmt_sole(w, ti, t):
        if t == 0:
            return '--'
        sole = w - ti
        pct = 100 * sole / t if t > 0 else 0
        return f"{GREEN}{BOLD}{sole}/{t}{RESET}" if pct >= 50 else f"{sole}/{t}"

    for label in ['WINS', 'TIES', 'SOLE WINS']:
        bf16_row = fp32_row = None
        for arr_w, arr_t, arr_ti in [
            (bf16_wins, bf16_totals, bf16_ties),
            (fp32_wins, fp32_totals, fp32_ties),
        ]:
            row = [label, '']
            if label == 'WINS':
                for i in range(3):
                    row.append(fmt_count(arr_w[i], arr_t[i]))
                row.append('')  # TTNN us col
                for i in range(3, 6):
                    row.append(fmt_count(arr_w[i], arr_t[i]))
                row.extend(['', ''])  # LUT us, config
            elif label == 'TIES':
                for i in range(3):
                    row.append(fmt_tie_count(arr_ti[i], arr_t[i]))
                row.append('')
                for i in range(3, 6):
                    row.append(fmt_tie_count(arr_ti[i], arr_t[i]))
                row.extend(['', ''])
            else:
                for i in range(3):
                    row.append(fmt_sole(arr_w[i], arr_ti[i], arr_t[i]))
                row.append('')
                for i in range(3, 6):
                    row.append(fmt_sole(arr_w[i], arr_ti[i], arr_t[i]))
                row.extend(['', ''])
            if bf16_row is None:
                bf16_row = row
            else:
                fp32_row = row
        _full_row(bf16_row, fp32_row)

    # Bottom border
    print(_hline_with_bg('└', '┴', '┘'))
    print()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Hardware-measured TTNN vs LUT comparison table')
    parser.add_argument('--metric', '-m', choices=['max', 'mae', 'ulp'],
                        help='Only show one metric table (default: all 3)')
    parser.add_argument('--activation', '-a',
                        help='Filter to single activation')
    parser.add_argument('--arch', default=None,
                        help='Architecture (default: auto-detect from data/)')
    args = parser.parse_args()

    # Locate directories — resolve() first to follow symlinks (embedded -> non-embedded)
    script_dir = Path(__file__).resolve().parent
    embedded_dir = script_dir.parent / 'generic_lut_activation_embedded'
    ttnn_dir = script_dir.parent / 'generic_lut_activation'  # TTNN data always from non-embedded
    config_path = script_dir / 'sweep_config.dat'

    # Use blackhole by default
    arch = args.arch if args.arch else 'blackhole'
    print(f"  Arch: {arch}", file=sys.stderr)

    # Load activation ranges
    ranges = {}
    if config_path.exists():
        ranges = parse_activation_ranges(config_path)

    # Load data
    ttnn_sfpu_dir = ttnn_dir / 'data' / arch / 'native_sfpu'
    lut_best_dir = embedded_dir / 'data' / arch / 'best'

    ttnn_data = load_ttnn_results_from_dir(str(ttnn_sfpu_dir), shape='yolov4')
    lut_data = load_lut_best_results(str(lut_best_dir), shape='yolov4')

    print(f"  TTNN entries: {len(ttnn_data)} (from {ttnn_sfpu_dir})", file=sys.stderr)
    print(f"  LUT  entries: {len(lut_data)} (from {lut_best_dir})", file=sys.stderr)

    if not ttnn_data and not lut_data:
        print("Error: No data found. Run sweep_native_sfpu.sh and sweep_best_all.sh first.",
              file=sys.stderr)
        sys.exit(1)

    # Build activation list from union of TTNN + LUT keys
    # TTNN keys: (act, prec), LUT keys: (act, prec, metric)
    all_acts = sorted(set(
        k[0] for k in list(ttnn_data.keys()) + list(lut_data.keys())
    ))

    if args.activation:
        if args.activation not in all_acts:
            print(f"Error: activation '{args.activation}' not found in data",
                  file=sys.stderr)
            sys.exit(1)
        all_acts = [args.activation]

    print(f"  Activations: {len(all_acts)}", file=sys.stderr)
    print(file=sys.stderr)

    tables = [
        ('max', 'max_error', 'BEST BY MAX ERROR'),
        ('mae', 'mae', 'BEST BY MAE'),
        ('ulp', 'max_ulp', 'BEST BY ULP'),
    ]
    if args.metric:
        tables = [t for t in tables if t[0] == args.metric]

    for table_type, sort_key, title in tables:
        print_comparison_table(title, sort_key, table_type, all_acts, ranges,
                               ttnn_data, lut_data)

    # Print recommended activations to replace TTNN
    print_recommended_replacements(all_acts, ranges, ttnn_data, lut_data)


def print_recommended_replacements(activations, ranges, ttnn_data, lut_data):
    """Print activations where LUT is better than TTNN on ULP with acceptable runtime.

    Criteria:
      - LUT ULP better than TTNN (ratio < 1.0)
      - BF16: runtime overhead < 50% (LUT_us / TTNN_us <= 1.5)
      - FP32: runtime overhead <= 2x (LUT_us / TTNN_us <= 2.0)
    """
    print("=" * 100)
    print(f"  {BOLD}RECOMMENDED ACTIVATIONS TO REPLACE TTNN{RESET}")
    print("=" * 100)
    print(f"  Criteria: LUT ULP better than TTNN")
    print(f"            BF16: runtime gap < 50% (LUT <= 1.5x TTNN)")
    print(f"            FP32: runtime cap 2x (LUT <= 2x TTNN)")
    print()

    # Collect recommendations per precision
    bf16_recs = []
    fp32_recs = []

    for act in activations:
        for prec, recs, runtime_cap in [('bf16', bf16_recs, 1.5), ('fp32', fp32_recs, 2.0)]:
            ttnn = ttnn_data.get((act, prec))
            # Use ULP-optimized LUT results
            lut = lut_data.get((act, prec, 'ulp'))
            if lut is None:
                lut = lut_data.get((act, prec, 'unknown'))

            if ttnn is None or lut is None:
                continue

            ttnn_ulp = ttnn['max_ulp']
            lut_ulp = lut['max_ulp']
            ttnn_us = ttnn['time_us']
            lut_us = lut['time_us']

            # Skip if either has invalid values
            if ttnn_ulp is None or lut_ulp is None or ttnn_ulp == float('inf'):
                continue
            if ttnn_us is None or lut_us is None or ttnn_us == 0:
                continue

            runtime_ratio = lut_us / ttnn_us

            # Skip if errors match but LUT is slower - no benefit to replacement
            if ttnn_ulp == lut_ulp and runtime_ratio > 1.0:
                continue

            # Skip if TTNN is already perfect (ULP = 0) - can't improve
            if ttnn_ulp == 0:
                continue

            ulp_ratio = lut_ulp / ttnn_ulp

            # Criteria: ULP better AND runtime within cap
            if ulp_ratio < 1.0 and runtime_ratio <= runtime_cap:
                recs.append({
                    'activation': act,
                    'ttnn_ulp': ttnn_ulp,
                    'lut_ulp': lut_ulp,
                    'ulp_ratio': ulp_ratio,
                    'ttnn_us': ttnn_us,
                    'lut_us': lut_us,
                    'runtime_ratio': runtime_ratio,
                    'config': lut['config'],
                })

    # Sort by ULP improvement (best first)
    bf16_recs.sort(key=lambda x: x['ulp_ratio'])
    fp32_recs.sort(key=lambda x: x['ulp_ratio'])

    def fmt_ulp_impr(ratio):
        """Format ULP improvement ratio nicely."""
        if ratio <= 0:
            return "inf"
        impr = 1.0 / ratio
        if impr >= 1e9:
            return f"{impr:.0e}x"
        elif impr >= 1e6:
            return f"{impr/1e6:.0f}Mx"
        elif impr >= 1e3:
            return f"{impr/1e3:.0f}Kx"
        elif impr >= 10:
            return f"{impr:.0f}x"
        else:
            return f"{impr:.1f}x"

    def fmt_rt_pct(ratio):
        """Format runtime as percentage change: negative = faster, positive = slower."""
        pct = (ratio - 1.0) * 100
        if abs(pct) < 1:
            pct_str = f"{pct:+.1f}%"
        else:
            pct_str = f"{pct:+.0f}%"
        if pct < 0:
            return f"{GREEN}{pct_str}{RESET}"
        elif pct > 0:
            return f"{RED}{pct_str}{RESET}"
        return pct_str

    # Print BF16 recommendations
    print(f"  {BOLD}BF16 Recommendations{RESET} ({len(bf16_recs)} activations):")
    if bf16_recs:
        print(f"  {'Activation':<15} {'TTNN ULP':>12} {'LUT ULP':>12} {'ULP Impr':>10} {'TTNN µs':>10} {'LUT µs':>10} {'RT Δ%':>10} {'Config':<16}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*16}")
        for r in bf16_recs:
            ulp_impr = fmt_ulp_impr(r['ulp_ratio'])
            rt_pct = fmt_rt_pct(r['runtime_ratio'])
            print(f"  {r['activation']:<15} {fmt_val(r['ttnn_ulp']):>12} {GREEN}{fmt_val(r['lut_ulp']):>12}{RESET} "
                  f"{GREEN}{ulp_impr:>10}{RESET} {fmt_time(r['ttnn_us']):>10} {fmt_time(r['lut_us']):>10} "
                  f"{rt_pct:>19} {r['config']:<16}")
    else:
        print(f"  {GREY}No activations meet the criteria{RESET}")
    print()

    # Print FP32 recommendations
    print(f"  {BOLD}FP32 Recommendations{RESET} ({len(fp32_recs)} activations):")
    if fp32_recs:
        print(f"  {'Activation':<15} {'TTNN ULP':>12} {'LUT ULP':>12} {'ULP Impr':>10} {'TTNN µs':>10} {'LUT µs':>10} {'RT Δ%':>10} {'Config':<16}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*16}")
        for r in fp32_recs:
            ulp_impr = fmt_ulp_impr(r['ulp_ratio'])
            rt_pct = fmt_rt_pct(r['runtime_ratio'])
            print(f"  {r['activation']:<15} {fmt_val(r['ttnn_ulp']):>12} {GREEN}{fmt_val(r['lut_ulp']):>12}{RESET} "
                  f"{GREEN}{ulp_impr:>10}{RESET} {fmt_time(r['ttnn_us']):>10} {fmt_time(r['lut_us']):>10} "
                  f"{rt_pct:>19} {r['config']:<16}")
    else:
        print(f"  {GREY}No activations meet the criteria{RESET}")
    print()

    # Summary line
    total_recs = len(set(r['activation'] for r in bf16_recs) | set(r['activation'] for r in fp32_recs))
    print(f"  {BOLD}Summary:{RESET} {total_recs} unique activations recommended for replacement")
    print()

    # Collect activations where error is better but runtime exceeds bounds
    bf16_exceeds = []
    fp32_exceeds = []

    for act in activations:
        for prec, exceeds, runtime_cap in [('bf16', bf16_exceeds, 1.5), ('fp32', fp32_exceeds, 2.0)]:
            ttnn = ttnn_data.get((act, prec))
            lut = lut_data.get((act, prec, 'ulp'))
            if lut is None:
                lut = lut_data.get((act, prec, 'unknown'))

            if ttnn is None or lut is None:
                continue

            ttnn_ulp = ttnn['max_ulp']
            lut_ulp = lut['max_ulp']
            ttnn_us = ttnn['time_us']
            lut_us = lut['time_us']

            if ttnn_ulp is None or lut_ulp is None or ttnn_ulp == float('inf'):
                continue
            if ttnn_us is None or lut_us is None or ttnn_us == 0:
                continue

            runtime_ratio = lut_us / ttnn_us

            # Skip if TTNN is already perfect
            if ttnn_ulp == 0:
                continue

            ulp_ratio = lut_ulp / ttnn_ulp

            # Criteria: ULP better BUT runtime exceeds cap
            if ulp_ratio < 1.0 and runtime_ratio > runtime_cap:
                exceeds.append({
                    'activation': act,
                    'ttnn_ulp': ttnn_ulp,
                    'lut_ulp': lut_ulp,
                    'ulp_ratio': ulp_ratio,
                    'ttnn_us': ttnn_us,
                    'lut_us': lut_us,
                    'runtime_ratio': runtime_ratio,
                    'config': lut['config'],
                })

    # Sort by ULP improvement (best first)
    bf16_exceeds.sort(key=lambda x: x['ulp_ratio'])
    fp32_exceeds.sort(key=lambda x: x['ulp_ratio'])

    if bf16_exceeds or fp32_exceeds:
        print("=" * 100)
        print(f"  {BOLD}BETTER ERROR BUT RUNTIME EXCEEDS BOUNDS{RESET}")
        print("=" * 100)
        print(f"  Note: These have better ULP but runtime exceeds the cap")
        print(f"        BF16 cap: 1.5x | FP32 cap: 2.0x")
        print()

        if bf16_exceeds:
            print(f"  {BOLD}BF16{RESET} ({len(bf16_exceeds)} activations):")
            print(f"  {'Activation':<15} {'TTNN ULP':>12} {'LUT ULP':>12} {'ULP Impr':>10} {'TTNN µs':>10} {'LUT µs':>10} {'RT Δ%':>10} {'Config':<16}")
            print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*16}")
            for r in bf16_exceeds:
                ulp_impr = fmt_ulp_impr(r['ulp_ratio'])
                rt_pct = fmt_rt_pct(r['runtime_ratio'])
                print(f"  {r['activation']:<15} {fmt_val(r['ttnn_ulp']):>12} {GREEN}{fmt_val(r['lut_ulp']):>12}{RESET} "
                      f"{GREEN}{ulp_impr:>10}{RESET} {fmt_time(r['ttnn_us']):>10} {fmt_time(r['lut_us']):>10} "
                      f"{rt_pct:>19} {r['config']:<16}")
            print()

        if fp32_exceeds:
            print(f"  {BOLD}FP32{RESET} ({len(fp32_exceeds)} activations):")
            print(f"  {'Activation':<15} {'TTNN ULP':>12} {'LUT ULP':>12} {'ULP Impr':>10} {'TTNN µs':>10} {'LUT µs':>10} {'RT Δ%':>10} {'Config':<16}")
            print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*16}")
            for r in fp32_exceeds:
                ulp_impr = fmt_ulp_impr(r['ulp_ratio'])
                rt_pct = fmt_rt_pct(r['runtime_ratio'])
                print(f"  {r['activation']:<15} {fmt_val(r['ttnn_ulp']):>12} {GREEN}{fmt_val(r['lut_ulp']):>12}{RESET} "
                      f"{GREEN}{ulp_impr:>10}{RESET} {fmt_time(r['ttnn_us']):>10} {fmt_time(r['lut_us']):>10} "
                      f"{rt_pct:>19} {r['config']:<16}")
            print()


if __name__ == '__main__':
    main()
