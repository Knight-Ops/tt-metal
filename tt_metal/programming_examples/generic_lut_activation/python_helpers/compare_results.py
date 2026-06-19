#!/usr/bin/env python3
"""
Compare activation function approximation results.

Unified comparison script that handles:
- 2-way: Polynomial vs SFPU
- 2-way: Rational vs SFPU
- 3-way: Rational vs Polynomial vs SFPU
"""

import argparse
import sys
import os
from typing import Optional
from collections import defaultdict

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

from comparison_utils import (
    read_results, detect_result_type, format_error, format_ratio,
    print_comparison_table, Colors, ResultRow
)


class ComparisonReport:
    """Generate comparison reports between different approximation methods."""

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
        try:
            self.primary_data = read_results(primary_file, self.primary_type)
        except FileNotFoundError:
            print(f"ERROR: Primary results file not found: {primary_file}", file=sys.stderr)
            sys.exit(1)

        self.sfpu_data = {}
        if sfpu_file and os.path.exists(sfpu_file):
            self.sfpu_data = read_results(sfpu_file, 'sfpu')

        self.poly_data = {}
        if polynomial_file and os.path.exists(polynomial_file):
            self.poly_data = read_results(polynomial_file, 'polynomial')

        # Apply filters
        self._apply_filters()

    def _apply_filters(self):
        """Apply precision and activation filters to all datasets."""
        if not (self.precision_filter or self.activation_filter):
            return

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
        # Build title
        title = "MAE COMPARISON"
        if self.poly_data and self.sfpu_data:
            title += f": {self.primary_type.title()} vs Polynomial vs SFPU"
        elif self.poly_data:
            title += f": {self.primary_type.title()} vs Polynomial"
        elif self.sfpu_data:
            title += f": {self.primary_type.title()} vs SFPU"
        else:
            title += f": {self.primary_type.title()} Results"

        # Build headers and widths
        headers = ["Activation", "Prec", f"{self.primary_type.title()}", "HW MAE"]
        widths = [12, 6, 12, 10]

        if self.poly_data:
            headers.extend(["Poly Cfg", "Poly MAE", "vs Poly"])
            widths.extend([10, 10, 14])

        if self.sfpu_data:
            headers.extend(["SFPU MAE", "vs SFPU"])
            widths.extend([10, 14])

        # Build rows
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

            # Add polynomial comparison
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

            # Add SFPU comparison
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

        # Print legend
        print(f"Legend: ratio=Ref÷{self.primary_type.title()} "
              f"({Colors.GREEN}green{Colors.RESET}={self.primary_type} wins, "
              f"{Colors.RED}red{Colors.RESET}=ref wins)")
        print()

    def print_max_error_comparison(self):
        """Print max error comparison table (if different from MAE winners)."""
        # Check if any configs have different MAE vs max_error winners
        has_different = False
        for key in self.primary_data.keys():
            primary = self.primary_data[key]

            # Check against polynomial
            if key in self.poly_data:
                poly = self.poly_data[key]
                mae_winner = primary.mae < poly.mae
                max_winner = primary.max_error < poly.max_error
                if mae_winner != max_winner:
                    has_different = True
                    break

            # Check against SFPU
            if key in self.sfpu_data:
                sfpu = self.sfpu_data[key]
                mae_winner = primary.mae < sfpu.mae
                max_winner = primary.max_error < sfpu.max_error
                if mae_winner != max_winner:
                    has_different = True
                    break

        if not has_different:
            return  # Skip if MAE and max_error have same winners

        # Build title
        title = "MAX ERROR COMPARISON"
        if self.poly_data and self.sfpu_data:
            title += f": {self.primary_type.title()} vs Polynomial vs SFPU"
        elif self.poly_data:
            title += f": {self.primary_type.title()} vs Polynomial"
        elif self.sfpu_data:
            title += f": {self.primary_type.title()} vs SFPU"

        # Build headers and widths
        headers = ["Activation", "Prec", f"{self.primary_type.title()}", "HW Max"]
        widths = [12, 6, 12, 10]

        if self.poly_data:
            headers.extend(["Poly Cfg", "Poly Max", "vs Poly"])
            widths.extend([10, 10, 14])

        if self.sfpu_data:
            headers.extend(["SFPU Max", "vs SFPU"])
            widths.extend([10, 14])

        # Build rows
        rows = []
        for key in sorted(self.primary_data.keys()):
            activation, precision = key
            primary = self.primary_data[key]

            row = [
                activation,
                precision,
                primary.config,
                format_error(primary.max_error)
            ]

            if self.poly_data:
                poly = self.poly_data.get(key)
                if poly:
                    row.extend([
                        poly.config,
                        format_error(poly.max_error),
                        format_ratio(primary.max_error, poly.max_error, 14)
                    ])
                else:
                    row.extend(["-", "-", "-"])

            if self.sfpu_data:
                sfpu = self.sfpu_data.get(key)
                if sfpu:
                    row.extend([
                        format_error(sfpu.max_error),
                        format_ratio(primary.max_error, sfpu.max_error, 14)
                    ])
                else:
                    row.extend(["-", "-"])

            rows.append(row)

        print_comparison_table(title, headers, rows, widths)

        print(f"Legend: ratio=Ref÷{self.primary_type.title()} "
              f"({Colors.GREEN}green{Colors.RESET}={self.primary_type} wins, "
              f"{Colors.RED}red{Colors.RESET}=ref wins)")
        print()

    def print_summary(self):
        """Print summary statistics."""
        total = len(self.primary_data)

        print("Summary:")
        print(f"  Total activations tested: {total}")

        # Compare against polynomial
        if self.poly_data:
            beats_poly_mae = sum(
                1 for k in self.primary_data
                if k in self.poly_data and self.primary_data[k].mae < self.poly_data[k].mae
            )
            compared = len([k for k in self.primary_data if k in self.poly_data])
            if compared > 0:
                pct = 100 * beats_poly_mae / compared
                print(f"  {self.primary_type.title()} beats Polynomial (MAE): "
                      f"{beats_poly_mae}/{compared} ({pct:.0f}%)")

            beats_poly_max = sum(
                1 for k in self.primary_data
                if k in self.poly_data and self.primary_data[k].max_error < self.poly_data[k].max_error
            )
            if compared > 0:
                pct = 100 * beats_poly_max / compared
                print(f"  {self.primary_type.title()} beats Polynomial (MaxErr): "
                      f"{beats_poly_max}/{compared} ({pct:.0f}%)")

        # Compare against SFPU
        if self.sfpu_data:
            beats_sfpu_mae = sum(
                1 for k in self.primary_data
                if k in self.sfpu_data and self.primary_data[k].mae < self.sfpu_data[k].mae
            )
            compared = len([k for k in self.primary_data if k in self.sfpu_data])
            if compared > 0:
                pct = 100 * beats_sfpu_mae / compared
                print(f"  {self.primary_type.title()} beats SFPU (MAE): "
                      f"{beats_sfpu_mae}/{compared} ({pct:.0f}%)")

            beats_sfpu_max = sum(
                1 for k in self.primary_data
                if k in self.sfpu_data and self.primary_data[k].max_error < self.sfpu_data[k].max_error
            )
            if compared > 0:
                pct = 100 * beats_sfpu_max / compared
                print(f"  {self.primary_type.title()} beats SFPU (MaxErr): "
                      f"{beats_sfpu_max}/{compared} ({pct:.0f}%)")

        print()

    def generate(self):
        """Generate complete comparison report."""
        self.print_mae_comparison()
        self.print_max_error_comparison()
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Compare activation function approximation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Polynomial vs SFPU
  compare_results.py polynomial_results.csv --sfpu sfpu_results.csv

  # Rational vs Polynomial vs SFPU
  compare_results.py rational_results.csv --polynomial poly.csv --sfpu sfpu.csv

  # Filter to specific precision
  compare_results.py results.csv --sfpu sfpu.csv --precision bf16

  # Disable colors for piping to file
  compare_results.py results.csv --sfpu sfpu.csv --no-color > report.txt
        """
    )

    parser.add_argument(
        "primary_results",
        help="Primary results CSV (polynomial or rational)"
    )
    parser.add_argument(
        "--sfpu",
        help="SFPU baseline results CSV for comparison"
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
