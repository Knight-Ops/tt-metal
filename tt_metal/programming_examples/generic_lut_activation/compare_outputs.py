#!/usr/bin/env python3
"""
Sample-by-sample comparison between hardware and theoretical LUT outputs.

Usage:
    python3 compare_outputs.py <hardware_csv> <theoretical_csv>
    python3 compare_outputs.py <hardware_csv> <theoretical_csv> --top 20
    python3 compare_outputs.py <hardware_csv> <theoretical_csv> --threshold 0.1

This helps debug discrepancies between hardware execution and theoretical validation.
"""

import numpy as np
import sys
from pathlib import Path
import argparse


def load_csv(filename):
    """Load input,output CSV file."""
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except Exception as e:
        print(f"ERROR: Could not load {filename}: {e}", file=sys.stderr)
        sys.exit(1)


def compare_outputs(hw_file, theo_file, top_n=10, threshold=None):
    """
    Compare hardware and theoretical outputs sample-by-sample.

    Args:
        hw_file: Hardware output CSV path
        theo_file: Theoretical output CSV path
        top_n: Number of worst samples to show
        threshold: Only show samples with error > threshold
    """
    print("=" * 80)
    print("Sample-by-Sample LUT Output Comparison")
    print("=" * 80)

    # Load data
    print(f"\nLoading hardware output:    {hw_file}")
    hw_data = load_csv(hw_file)
    print(f"Loading theoretical output: {theo_file}")
    theo_data = load_csv(theo_file)

    # Validate same number of samples
    if len(hw_data) != len(theo_data):
        print(f"\nERROR: Sample count mismatch!")
        print(f"  Hardware:    {len(hw_data):,} samples")
        print(f"  Theoretical: {len(theo_data):,} samples")
        sys.exit(1)

    num_samples = len(hw_data)
    print(f"\n✓ Both files have {num_samples:,} samples")

    # Check if inputs match
    input_diff = np.abs(hw_data[:, 0] - theo_data[:, 0])
    max_input_diff = np.max(input_diff)

    if max_input_diff > 1e-5:
        print(f"\n⚠️  WARNING: Input values don't match exactly!")
        print(f"  Max input difference: {max_input_diff:.6e}")
        print(f"  This suggests different test point generation strategies.")
    else:
        print(f"\n✓ Input values match (max diff: {max_input_diff:.6e})")

    # Compute output differences
    hw_outputs = hw_data[:, 1]
    theo_outputs = theo_data[:, 1]
    output_diff = np.abs(hw_outputs - theo_outputs)

    # Overall statistics
    print("\n" + "=" * 80)
    print("Overall Statistics")
    print("=" * 80)
    print(f"Mean absolute difference:   {np.mean(output_diff):.6e}")
    print(f"Median absolute difference: {np.median(output_diff):.6e}")
    print(f"Max absolute difference:    {np.max(output_diff):.6e}")
    print(f"Std dev of difference:      {np.std(output_diff):.6e}")

    # Compute relative error (where hardware output is significant)
    mask = np.abs(hw_outputs) > 1e-6
    if np.any(mask):
        rel_error = output_diff[mask] / np.abs(hw_outputs[mask])
        print(f"\nRelative error (where |hw_output| > 1e-6):")
        print(f"  Mean:   {np.mean(rel_error):.6e}")
        print(f"  Median: {np.median(rel_error):.6e}")
        print(f"  Max:    {np.max(rel_error):.6e}")

    # Find worst samples
    if threshold is not None:
        # Show all samples above threshold
        bad_indices = np.where(output_diff > threshold)[0]
        print(f"\n" + "=" * 80)
        print(f"Samples with error > {threshold}")
        print("=" * 80)
        print(f"Found {len(bad_indices):,} samples above threshold ({100*len(bad_indices)/num_samples:.2f}%)")
    else:
        # Show top N worst samples
        bad_indices = np.argsort(output_diff)[-top_n:][::-1]  # Descending order
        print(f"\n" + "=" * 80)
        print(f"Top {top_n} Worst Samples")
        print("=" * 80)

    if len(bad_indices) == 0:
        print("No samples exceed the threshold!")
        return 0

    # Print header
    print(f"\n{'Index':<10} {'Input':<15} {'Hardware':<15} {'Theoretical':<15} {'Abs Diff':<15} {'Rel Err':<12}")
    print("-" * 90)

    # Print worst samples
    for idx in bad_indices[:100]:  # Limit to first 100 even if more
        inp = hw_data[idx, 0]
        hw_out = hw_outputs[idx]
        theo_out = theo_outputs[idx]
        diff = output_diff[idx]

        # Compute relative error
        if abs(hw_out) > 1e-10:
            rel_err = diff / abs(hw_out)
            rel_err_str = f"{rel_err:.6e}"
        else:
            rel_err_str = "N/A"

        print(f"{idx:<10} {inp:<15.6f} {hw_out:<15.6e} {theo_out:<15.6e} {diff:<15.6e} {rel_err_str:<12}")

    if len(bad_indices) > 100:
        print(f"... ({len(bad_indices) - 100} more samples not shown)")

    # Distribution analysis
    print(f"\n" + "=" * 80)
    print("Error Distribution")
    print("=" * 80)

    percentiles = [50, 75, 90, 95, 99, 99.9, 100]
    print(f"{'Percentile':<15} {'Error':<15}")
    print("-" * 30)
    for p in percentiles:
        val = np.percentile(output_diff, p)
        print(f"{p:<15.1f} {val:<15.6e}")

    return 0


def main():
    parser = argparse.ArgumentParser(description='Compare hardware vs theoretical LUT outputs')
    parser.add_argument('hardware_csv', type=str, help='Hardware output CSV file')
    parser.add_argument('theoretical_csv', type=str, help='Theoretical output CSV file')
    parser.add_argument('--top', type=int, default=10, help='Number of worst samples to show (default: 10)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Show all samples with error > threshold (overrides --top)')

    args = parser.parse_args()

    # Validate files exist
    if not Path(args.hardware_csv).exists():
        print(f"ERROR: Hardware CSV not found: {args.hardware_csv}", file=sys.stderr)
        return 1

    if not Path(args.theoretical_csv).exists():
        print(f"ERROR: Theoretical CSV not found: {args.theoretical_csv}", file=sys.stderr)
        return 1

    return compare_outputs(args.hardware_csv, args.theoretical_csv, args.top, args.threshold)


if __name__ == "__main__":
    sys.exit(main())
