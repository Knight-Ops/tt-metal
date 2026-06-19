# Unified Comparison Script Design

## Unified Script: `compare_results.py`

### Design Goal
Single script that intelligently handles all comparison scenarios:
- 2-way: Polynomial vs SFPU
- 2-way: Rational vs SFPU
- 3-way: Rational vs Polynomial vs SFPU
- Future: Any approximation method vs baselines

### Usage

```bash
# Polynomial vs SFPU (2-way)
python3 compare_results.py polynomial_results.csv --sfpu sfpu_results.csv

# Rational vs Polynomial vs SFPU (3-way)
python3 compare_results.py rational_results.csv --polynomial polynomial_results.csv --sfpu sfpu_results.csv

# Rational vs SFPU only (2-way)
python3 compare_results.py rational_results.csv --sfpu sfpu_results.csv

# Polynomial only (no comparison, just summary)
python3 compare_results.py polynomial_results.csv
```

### Auto-detection Logic

Script automatically detects result type from CSV headers:
- **Polynomial**: Has columns `degree`, `depth`, `segmentation`
- **Rational**: Has columns `num_degree`, `den_degree`, `segments`
- **SFPU**: Has columns `sfpu_mode`

### Table Generation

Based on what files are provided:
1. **Main results table**: Show primary results (polynomial or rational)
2. **MAE comparison table**: If SFPU and/or polynomial provided
3. **Max error comparison table**: If SFPU and/or polynomial provided
4. **Top improvements table**: If baseline provided
5. **Summary statistics**: Always shown

### Command-line Interface

```python
#!/usr/bin/env python3
"""
Compare activation function approximation results.

Usage:
    compare_results.py <primary_results.csv> [OPTIONS]

Options:
    --sfpu FILE           SFPU baseline results for comparison
    --polynomial FILE     Polynomial results for comparison (when primary is rational)
    --format {table,csv}  Output format (default: table)
    --no-color           Disable colored output
    --metric {mae,max}   Primary metric for ranking (default: mae)
    --precision PREC     Filter to specific precision (bf16 or fp32)
    --activation ACT     Filter to specific activation
"""
```

### Comparison Logic

```python
class ComparisonReport:
    def __init__(self, primary_file, sfpu_file=None, polynomial_file=None):
        self.primary_type = detect_result_type(primary_file)
        self.primary_data = read_results(primary_file)

        self.sfpu_data = read_results(sfpu_file) if sfpu_file else None
        self.poly_data = read_results(polynomial_file) if polynomial_file else None

    def generate(self):
        # Print MAE comparison
        self.print_mae_comparison()

        # Print max error comparison (if different best configs)
        if self.has_different_max_error_winners():
            self.print_max_error_comparison()

        # Print top improvements
        if self.has_baseline():
            self.print_top_improvements()

        # Print summary stats
        self.print_summary()
```

### Shared Utilities Module: `comparison_utils.py`

```python
"""Shared utilities for result comparison and display."""

import csv
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ANSI color codes
class Colors:
    GREEN = '\033[32m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls):
        cls.GREEN = cls.RED = cls.YELLOW = cls.RESET = ''

@dataclass
class ResultRow:
    """Generic result row from any CSV."""
    activation: str
    precision: str
    mae: float
    max_error: float
    mean_ulp: float
    max_ulp: float
    runtime_us: float
    config: str
    status: str

def read_results(csv_file: str, result_type: str = None) -> Dict[Tuple[str, str], ResultRow]:
    """
    Read results CSV and return dict keyed by (activation, precision).
    Auto-detects result type if not provided.
    """
    pass

def detect_result_type(csv_file: str) -> str:
    """Detect if CSV is polynomial, rational, or SFPU results."""
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            if 'num_degree' in reader.fieldnames and 'den_degree' in reader.fieldnames:
                return 'rational'
            elif 'degree' in reader.fieldnames and 'depth' in reader.fieldnames:
                return 'polynomial'
            elif 'sfpu_mode' in reader.fieldnames:
                return 'sfpu'
    return 'unknown'

def format_error(val: float, precision: int = 2) -> str:
    """Format error value in scientific notation."""
    if val == 0:
        return "0.00e+00"
    return f"{val:.{precision}e}"

def format_ratio(val: float, ref: float, width: int = 8) -> str:
    """
    Format ratio as Nx with color coding.
    Green if val is better (ratio > 1), red if worse (ratio < 1).
    """
    if val is None or ref is None or ref < 1e-15 or val < 1e-15:
        return "-".center(width)

    ratio = ref / val  # >1 means val is better

    if ratio >= 1:
        ratio_int = int(round(ratio))
        return f"{Colors.GREEN}{ratio_int}×{Colors.RESET}".rjust(width + len(Colors.GREEN) + len(Colors.RESET))
    else:
        inv_ratio = int(round(1/ratio))
        if inv_ratio == 1:
            return "~1×".rjust(width)
        return f"{Colors.RED}1/{inv_ratio}×{Colors.RESET}".rjust(width + len(Colors.RED) + len(Colors.RESET))

def print_table_header(title: str, width: int = 115):
    """Print table title with border."""
    print(f"\n{title}")
    print("=" * width)
    print()

def print_box_row(cells: List[str], widths: List[int], borders: str = "│││"):
    """Print a row with box-drawing characters."""
    left, mid, right = borders
    row = left
    for cell, width in zip(cells, widths):
        # Handle ANSI color codes in width calculation
        visible_len = len(cell)
        for code in [Colors.GREEN, Colors.RED, Colors.YELLOW, Colors.RESET]:
            visible_len -= cell.count(code) * len(code)
        padding = width - visible_len
        row += f" {cell}{' ' * padding} {mid}"
    print(row[:-1] + right)

def print_comparison_table(
    title: str,
    headers: List[str],
    rows: List[List[str]],
    widths: List[int]
):
    """Print a formatted comparison table with box-drawing."""
    print_table_header(title)

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
```

### Main Script: `compare_results.py`

```python
#!/usr/bin/env python3
"""Compare activation function approximation results."""

import argparse
import sys
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

from comparison_utils import (
    read_results, detect_result_type, format_error, format_ratio,
    print_table_header, print_comparison_table, Colors, ResultRow
)

class ComparisonReport:
    def __init__(
        self,
        primary_file: str,
        sfpu_file: Optional[str] = None,
        polynomial_file: Optional[str] = None,
        metric: str = 'mae',
        precision_filter: Optional[str] = None,
        activation_filter: Optional[str] = None
    ):
        self.primary_type = detect_result_type(primary_file)
        self.metric = metric
        self.precision_filter = precision_filter
        self.activation_filter = activation_filter

        # Read all result sets
        self.primary_data = read_results(primary_file, self.primary_type)
        self.sfpu_data = read_results(sfpu_file, 'sfpu') if sfpu_file else {}
        self.poly_data = read_results(polynomial_file, 'polynomial') if polynomial_file else {}

        # Filter results
        self._apply_filters()

    def _apply_filters(self):
        """Apply precision and activation filters."""
        if self.precision_filter or self.activation_filter:
            for data in [self.primary_data, self.sfpu_data, self.poly_data]:
                keys_to_remove = []
                for key in list(data.keys()):
                    activation, precision = key
                    if self.precision_filter and precision != self.precision_filter:
                        keys_to_remove.append(key)
                    elif self.activation_filter and activation != self.activation_filter:
                        keys_to_remove.append(key)
                for key in keys_to_remove:
                    del data[key]

    def print_mae_comparison(self):
        """Print MAE comparison table."""
        title = "MAE COMPARISON"
        if self.poly_data:
            title += f": {self.primary_type.title()} vs Polynomial vs SFPU"
        elif self.sfpu_data:
            title += f": {self.primary_type.title()} vs SFPU"
        else:
            title += f": {self.primary_type.title()} Results"

        headers = ["Activation", "Prec", f"{self.primary_type.title()}", "HW MAE"]
        widths = [12, 6, 12, 10]

        if self.poly_data:
            headers.extend(["Poly Cfg", "Poly MAE", "vs Poly"])
            widths.extend([10, 10, 14])

        if self.sfpu_data:
            headers.extend(["SFPU MAE", "vs SFPU"])
            widths.extend([10, 14])

        rows = []
        for key in sorted(self.primary_data.keys()):
            activation, precision = key
            primary = self.primary_data[key]

            row = [
                activation,
                precision,
                primary.config,
                format_error(primary.mae)
            ]

            if self.poly_data:
                poly = self.poly_data.get(key)
                if poly:
                    row.extend([
                        poly.config,
                        format_error(poly.mae),
                        format_ratio(primary.mae, poly.mae, 14)
                    ])
                else:
                    row.extend(["-", "-", "-"])

            if self.sfpu_data:
                sfpu = self.sfpu_data.get(key)
                if sfpu:
                    row.extend([
                        format_error(sfpu.mae),
                        format_ratio(primary.mae, sfpu.mae, 14)
                    ])
                else:
                    row.extend(["-", "-"])

            rows.append(row)

        print_comparison_table(title, headers, rows, widths)

        # Legend
        print(f"Legend: ratio=Ref÷{self.primary_type.title()} "
              f"({Colors.GREEN}green{Colors.RESET}={self.primary_type} wins, "
              f"{Colors.RED}red{Colors.RESET}=ref wins)")
        print()

    def print_summary(self):
        """Print summary statistics."""
        total = len(self.primary_data)

        print("Summary:")
        print(f"  Total activations tested: {total}")

        if self.poly_data:
            beats_poly = sum(
                1 for k in self.primary_data
                if k in self.poly_data and self.primary_data[k].mae < self.poly_data[k].mae
            )
            compared = len([k for k in self.primary_data if k in self.poly_data])
            if compared > 0:
                print(f"  {self.primary_type.title()} beats Polynomial (MAE): "
                      f"{beats_poly}/{compared} ({100*beats_poly/compared:.0f}%)")

        if self.sfpu_data:
            beats_sfpu = sum(
                1 for k in self.primary_data
                if k in self.sfpu_data and self.primary_data[k].mae < self.sfpu_data[k].mae
            )
            compared = len([k for k in self.primary_data if k in self.sfpu_data])
            if compared > 0:
                print(f"  {self.primary_type.title()} beats SFPU (MAE): "
                      f"{beats_sfpu}/{compared} ({100*beats_sfpu/compared:.0f}%)")
        print()

    def generate(self):
        """Generate complete comparison report."""
        self.print_mae_comparison()
        # Could add max_error comparison, top improvements, etc.
        self.print_summary()

def main():
    parser = argparse.ArgumentParser(
        description="Compare activation function approximation results"
    )
    parser.add_argument(
        "primary_results",
        help="Primary results CSV (polynomial or rational)"
    )
    parser.add_argument(
        "--sfpu",
        help="SFPU baseline results CSV"
    )
    parser.add_argument(
        "--polynomial",
        help="Polynomial results CSV (for rational comparison)"
    )
    parser.add_argument(
        "--metric",
        choices=["mae", "max"],
        default="mae",
        help="Primary metric for ranking (default: mae)"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp32"],
        help="Filter to specific precision"
    )
    parser.add_argument(
        "--activation",
        help="Filter to specific activation"
    )

    args = parser.parse_args()

    if args.no_color:
        Colors.disable()

    report = ComparisonReport(
        args.primary_results,
        sfpu_file=args.sfpu,
        polynomial_file=args.polynomial,
        metric=args.metric,
        precision_filter=args.precision,
        activation_filter=args.activation
    )

    report.generate()

if __name__ == '__main__':
    main()
```

### Updated Sweep Script Usage

```bash
# In sweep_polynomial.sh
if [[ $processed -gt 0 ]]; then
    python3 "$WORK_DIR/python_helpers/compare_results.py" \
        "$RESULTS_FILE" \
        --sfpu "$SFPU_RESULTS_FILE"
fi

# In sweep_rational.sh
if [[ $processed -gt 0 ]]; then
    python3 "$WORK_DIR/python_helpers/compare_results.py" \
        "$RESULTS_FILE" \
        --polynomial "$POLY_RESULTS_FILE" \
        --sfpu "$SFPU_RESULTS_FILE"
fi
```

## Benefits of Unified Approach

1. **Single source of truth** for comparison logic
2. **Consistent formatting** across all comparison reports
3. **Easier to extend** (add new metrics, comparison types)
4. **Better tested** (one script to test thoroughly)
5. **More flexible** (can compare any combination)
6. **Simpler maintenance** (fix bugs in one place)

## File Structure

```
python_helpers/
├── __init__.py
├── compute_accuracy_from_csv.py    # 95 lines (accuracy computation)
├── compare_results.py              # 200 lines (unified comparison)
└── comparison_utils.py             # 150 lines (shared utilities)
```

**Total**: ~445 lines of well-organized Python
**Replaces**: ~709 lines of inline Python
**Net reduction**: ~264 lines (7.4%)
