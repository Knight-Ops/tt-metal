#!/usr/bin/env python3
"""Shared utilities for result comparison and display."""

import csv
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[32m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls):
        """Disable colors for non-terminal output."""
        cls.GREEN = cls.RED = cls.YELLOW = cls.RESET = ''


@dataclass
class ResultRow:
    """Generic result row from any CSV."""
    activation: str
    precision: str
    mae: float
    rmse: float
    max_error: float
    mean_rel_error: float
    max_ulp: float
    mean_ulp: float
    runtime_us: float  # Using 256_tiles as representative
    config: str
    status: str


def detect_result_type(csv_file: str) -> str:
    """
    Detect if CSV is polynomial, rational, or SFPU results.

    Args:
        csv_file: Path to results CSV

    Returns:
        'polynomial', 'rational', 'sfpu', or 'unknown'
    """
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                if 'num_degree' in reader.fieldnames and 'den_degree' in reader.fieldnames:
                    return 'rational'
                elif 'degree' in reader.fieldnames and 'depth' in reader.fieldnames:
                    return 'polynomial'
                elif 'sfpu_mode' in reader.fieldnames:
                    return 'sfpu'
    except FileNotFoundError:
        return 'unknown'
    return 'unknown'


def read_results(
    csv_file: str,
    result_type: Optional[str] = None,
    status_filter: str = 'pass'
) -> Dict[Tuple[str, str], ResultRow]:
    """
    Read results CSV and return dict keyed by (activation, precision).
    Automatically finds best config per activation for polynomial/rational.

    Args:
        csv_file: Path to results CSV
        result_type: 'polynomial', 'rational', 'sfpu', or None (auto-detect)
        status_filter: Only include rows with this status (default: 'pass')

    Returns:
        Dict mapping (activation, precision) to best ResultRow
    """
    if result_type is None:
        result_type = detect_result_type(csv_file)

    results = {}

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row['status'] != status_filter:
                continue

            # Row-per-shape format: only use 256_tiles rows as representative
            if 'shape' in row and row.get('shape') != '256_tiles':
                continue

            activation = row['activation']
            precision = row.get('precision', 'fp32')
            key = (activation, precision)

            # Extract metrics - row-per-shape uses generic names, legacy uses shape-suffixed names
            mae = float(row.get('mae', row.get('mae_256_tiles', 0)))
            rmse = float(row.get('rmse', row.get('rmse_256_tiles', 0)))
            max_error = float(row.get('max_error', row.get('max_error_256_tiles', 0)))
            mean_rel = float(row.get('mean_rel_error', row.get('mean_rel_error_256_tiles', 0)))
            max_ulp = float(row.get('max_ulp', row.get('max_ulp_error_256_tiles', 0)))
            mean_ulp = float(row.get('mean_ulp', row.get('mean_ulp_error_256_tiles', 0)))

            # Timing: row-per-shape uses time_profiler_us, legacy uses time_profiler_256_tiles_us
            runtime_us = float(row.get('time_profiler_us', row.get('time_profiler_256_tiles_us', 0)))

            # Build config string based on result type
            if result_type == 'polynomial':
                degree = int(row['degree'])
                depth = int(row['depth'])
                config = f"p{degree}_s{depth}"
            elif result_type == 'rational':
                num_deg = int(row['num_degree'])
                den_deg = int(row['den_degree'])
                segs = int(row['segments'])
                config = f"n{num_deg}_d{den_deg}_s{segs}"
            elif result_type == 'sfpu':
                sfpu_mode = row.get('sfpu_mode', 'fast')
                config = f"sfpu_{sfpu_mode}"
            else:
                config = "unknown"

            result_row = ResultRow(
                activation=activation,
                precision=precision,
                mae=mae,
                rmse=rmse,
                max_error=max_error,
                mean_rel_error=mean_rel,
                max_ulp=max_ulp,
                mean_ulp=mean_ulp,
                runtime_us=runtime_us,
                config=config,
                status=row['status']
            )

            # Keep best MAE for each (activation, precision)
            if key not in results or mae < results[key].mae:
                results[key] = result_row

    return results


def format_error(val: Optional[float], precision: int = 2) -> str:
    """Format error value in scientific notation."""
    if val is None:
        return "N/A"
    if val == 0:
        return "0.00e+00"
    return f"{val:.{precision}e}"


def format_ratio(val: Optional[float], ref: Optional[float], width: int = 8) -> str:
    """
    Format ratio as Nx with color coding.

    Args:
        val: Value to compare (lower is better for errors)
        ref: Reference value
        width: Display width for alignment

    Returns:
        Colored ratio string (green if val < ref, red otherwise)
    """
    if val is None or ref is None:
        return "-".center(width)
    if ref < 1e-15 or val < 1e-15:
        return "-".center(width)

    ratio = ref / val  # >1 means val is better (lower error)
    ratio_int = int(round(ratio))

    if ratio >= 1:
        # val is better
        color_len = len(Colors.GREEN) + len(Colors.RESET)
        return f"{Colors.GREEN}{ratio_int}×{Colors.RESET}".rjust(width + color_len)
    else:
        # ref is better
        inv_ratio = int(round(1/ratio))
        if inv_ratio == 1:
            return "~1×".rjust(width)
        color_len = len(Colors.RED) + len(Colors.RESET)
        return f"{Colors.RED}1/{inv_ratio}×{Colors.RESET}".rjust(width + color_len)


def print_box_row(cells: List[str], widths: List[int], borders: str = "│││"):
    """
    Print a row with box-drawing characters.

    Args:
        cells: Cell contents
        widths: Width for each cell
        borders: Three characters: left, middle, right separators
    """
    left, mid, right = borders

    row_parts = []
    for cell, width in zip(cells, widths):
        # Calculate visible length (excluding ANSI codes)
        visible_len = len(cell)
        for code in [Colors.GREEN, Colors.RED, Colors.YELLOW, Colors.RESET]:
            visible_len -= cell.count(code) * len(code)

        padding = width - visible_len
        row_parts.append(f" {cell}{' ' * max(0, padding)} ")

    print(left + mid.join(row_parts) + right)


def print_comparison_table(
    title: str,
    headers: List[str],
    rows: List[List[str]],
    widths: List[int]
):
    """
    Print a formatted comparison table with Unicode box-drawing.

    Args:
        title: Table title
        headers: Column headers
        rows: Data rows
        widths: Column widths
    """
    print(f"\n{title}")
    print("=" * sum(widths))
    print()

    # Top border
    print("┌" + "┬".join("─" * (w + 2) for w in widths) + "┐")

    # Header
    print_box_row(headers, widths, "│││")

    # Header separator
    print("├" + "┼".join("─" * (w + 2) for w in widths) + "┤")

    # Data rows
    for row in rows:
        print_box_row(row, widths, "│││")

    # Bottom border
    print("└" + "┴".join("─" * (w + 2) for w in widths) + "┘")
    print()
