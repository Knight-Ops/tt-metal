#!/usr/bin/env python3
"""
Test and validate piecewise rational approximation kernels

This script:
1. Loads rational LUT from CSV file
2. Generates test input data
3. Computes reference output using Python implementation
4. Optionally compares against hardware output from CSV
5. Reports accuracy metrics (MAE, RMSE, max error)
"""

import numpy as np
import csv
import argparse
from pathlib import Path


def load_rational_csv(csv_path):
    """
    Load rational approximation LUT from CSV file

    CSV format:
    segment_id,lo,hi,approximation_type,num_degree,den_degree,n0,n1,n2,...,d0,d1,d2,...,error,method,segmentation

    Returns:
        dict with keys: lut_data, num_segments, num_degree, den_degree, lut_size, boundaries
    """
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        segments = []
        for row in reader:
            # Skip METADATA rows
            if row.get('segment_id', '').startswith('METADATA'):
                continue

            segment = {
                'lo': float(row['lo']),
                'hi': float(row['hi']),
                'num_degree': int(row.get('num_degree', 0)),
                'den_degree': int(row.get('den_degree', 0)),
                'num_coeffs': [],
                'den_coeffs': []
            }

            # Parse numerator coefficients (n0, n1, n2, ...)
            i = 0
            while f'n{i}' in row:
                segment['num_coeffs'].append(float(row[f'n{i}']))
                i += 1

            # Parse denominator coefficients (d0, d1, d2, ...)
            i = 0
            while f'd{i}' in row:
                segment['den_coeffs'].append(float(row[f'd{i}']))
                i += 1

            segments.append(segment)

    if not segments:
        raise ValueError(f"No segments found in {csv_path}")

    # Validate consistent degrees
    num_degree = segments[0]['num_degree']
    den_degree = segments[0]['den_degree']
    for seg in segments:
        if seg['num_degree'] != num_degree or seg['den_degree'] != den_degree:
            raise ValueError(f"Inconsistent degrees across segments in {csv_path}")

    # Build LUT: [boundaries...] [num_coeffs_seg0, den_coeffs_seg0, ...]
    num_segments = len(segments)
    boundaries = [segments[0]['lo']] + [seg['hi'] for seg in segments]

    lut_data = boundaries.copy()
    for seg in segments:
        lut_data.extend(seg['num_coeffs'])
        lut_data.extend(seg['den_coeffs'])

    lut_size = (num_segments + 1) + num_segments * (num_degree + den_degree + 2)

    return {
        'lut_data': np.array(lut_data, dtype=np.float32),
        'num_segments': num_segments,
        'num_degree': num_degree,
        'den_degree': den_degree,
        'lut_size': lut_size,
        'boundaries': boundaries,
        'segments': segments
    }


def eval_polynomial_horner(coeffs, x):
    """
    Evaluate polynomial using Horner's method: c[0] + c[1]*x + c[2]*x^2 + ... + c[n]*x^n

    Args:
        coeffs: array of coefficients [c0, c1, c2, ..., cn]
        x: input value or array

    Returns:
        polynomial value at x
    """
    if len(coeffs) == 0:
        return 0.0
    elif len(coeffs) == 1:
        return coeffs[0]

    # Horner: start with highest degree, multiply by x, add next coefficient
    result = coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        result = result * x + coeffs[i]

    return result


def reference_rational_eval(lut_info, x):
    """
    Evaluate piecewise rational approximation (reference implementation)

    Args:
        lut_info: dict from load_rational_csv()
        x: input value or array

    Returns:
        output value or array
    """
    x = np.asarray(x, dtype=np.float32)
    is_scalar = x.ndim == 0
    x = np.atleast_1d(x)

    boundaries = lut_info['boundaries']
    num_segments = lut_info['num_segments']
    num_degree = lut_info['num_degree']
    den_degree = lut_info['den_degree']
    lut_data = lut_info['lut_data']

    # Clamp x to valid range
    x_clamped = np.clip(x, boundaries[0], boundaries[-1])

    # Initialize output
    output = np.zeros_like(x_clamped)

    # Coefficient layout: [boundaries...] [num_coeffs_seg0, den_coeffs_seg0, ...]
    coeff_offset = num_segments + 1
    coeffs_per_seg = num_degree + den_degree + 2

    # Evaluate for each segment
    for seg_idx in range(num_segments):
        # Find elements in this segment
        if seg_idx == num_segments - 1:
            # Last segment includes upper boundary
            mask = (x_clamped >= boundaries[seg_idx]) & (x_clamped <= boundaries[seg_idx + 1])
        else:
            mask = (x_clamped >= boundaries[seg_idx]) & (x_clamped < boundaries[seg_idx + 1])

        if not np.any(mask):
            continue

        # Extract coefficients for this segment
        base = coeff_offset + seg_idx * coeffs_per_seg
        num_coeffs = lut_data[base:base + num_degree + 1]
        den_coeffs = lut_data[base + num_degree + 1:base + coeffs_per_seg]

        # Evaluate rational function: P(x) / Q(x)
        x_seg = x_clamped[mask]
        numerator = eval_polynomial_horner(num_coeffs, x_seg)
        denominator = eval_polynomial_horner(den_coeffs, x_seg)

        # Division with zero protection
        denominator = np.where(denominator == 0.0, 1e-10, denominator)

        output[mask] = numerator / denominator

    return output[0] if is_scalar else output


def compute_accuracy_metrics(reference, hardware):
    """
    Compute accuracy metrics between reference and hardware outputs

    Args:
        reference: reference output array
        hardware: hardware output array

    Returns:
        dict with keys: mae, rmse, max_error, median_error
    """
    error = np.abs(reference - hardware)

    metrics = {
        'mae': np.mean(error),
        'rmse': np.sqrt(np.mean(error ** 2)),
        'max_error': np.max(error),
        'median_error': np.median(error),
        'mean_rel_error': np.mean(error / (np.abs(reference) + 1e-10))
    }

    return metrics


def test_rational_lut(csv_path, test_range=(-10, 10), num_points=1024, hardware_csv=None, verbose=True):
    """
    Test rational LUT kernel

    Args:
        csv_path: path to rational coefficient CSV
        test_range: (min, max) test range
        num_points: number of test points
        hardware_csv: optional path to hardware output CSV
        verbose: print detailed results

    Returns:
        dict with accuracy metrics
    """
    # Load LUT
    if verbose:
        print(f"Loading rational LUT from: {csv_path}")
    lut_info = load_rational_csv(csv_path)

    if verbose:
        print(f"  Segments: {lut_info['num_segments']}")
        print(f"  Numerator degree: {lut_info['num_degree']}")
        print(f"  Denominator degree: {lut_info['den_degree']}")
        print(f"  LUT size: {lut_info['lut_size']}")
        print(f"  Boundaries: {lut_info['boundaries']}")

    # Generate test input
    x_test = np.linspace(test_range[0], test_range[1], num_points, dtype=np.float32)

    # Compute reference output
    if verbose:
        print(f"\nComputing reference output for {num_points} points...")
    y_reference = reference_rational_eval(lut_info, x_test)

    if verbose:
        print(f"  Reference output range: [{np.min(y_reference):.6f}, {np.max(y_reference):.6f}]")
        print(f"  Reference output mean: {np.mean(y_reference):.6f}")

    # Compare with hardware output if provided
    if hardware_csv:
        if verbose:
            print(f"\nLoading hardware output from: {hardware_csv}")
        y_hardware = np.loadtxt(hardware_csv, delimiter=',', dtype=np.float32)

        if len(y_hardware) != len(y_reference):
            print(f"WARNING: Length mismatch - reference: {len(y_reference)}, hardware: {len(y_hardware)}")
            # Truncate to shorter length
            min_len = min(len(y_reference), len(y_hardware))
            y_reference = y_reference[:min_len]
            y_hardware = y_hardware[:min_len]
            x_test = x_test[:min_len]

        # Compute accuracy metrics
        metrics = compute_accuracy_metrics(y_reference, y_hardware)

        if verbose:
            print(f"\nAccuracy Metrics:")
            print(f"  MAE:              {metrics['mae']:.6e}")
            print(f"  RMSE:             {metrics['rmse']:.6e}")
            print(f"  Max Error:        {metrics['max_error']:.6e}")
            print(f"  Median Error:     {metrics['median_error']:.6e}")
            print(f"  Mean Rel Error:   {metrics['mean_rel_error']:.6e}")

            # Find worst-case points
            error = np.abs(y_reference - y_hardware)
            worst_idx = np.argmax(error)
            print(f"\nWorst-case error:")
            print(f"  Input:     {x_test[worst_idx]:.6f}")
            print(f"  Reference: {y_reference[worst_idx]:.6f}")
            print(f"  Hardware:  {y_hardware[worst_idx]:.6f}")
            print(f"  Error:     {error[worst_idx]:.6e}")

        return metrics
    else:
        # No hardware output - just return reference
        if verbose:
            print("\nNo hardware output provided - reference computed successfully")

        return {
            'reference': y_reference,
            'x_test': x_test,
            'lut_info': lut_info
        }


def main():
    parser = argparse.ArgumentParser(
        description='Test and validate piecewise rational approximation kernels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute reference output
  python test_piecewise_rational.py sigmoid_fp32_2_4_4_rational.csv --range -10 10

  # Compare hardware output
  python test_piecewise_rational.py sigmoid_fp32_2_4_4_rational.csv \\
      --range -10 10 --hardware output.csv

  # Test multiple activations
  for csv in data/coefficients/*_rational.csv; do
      python test_piecewise_rational.py "$csv" --range -10 10
  done
        """
    )

    parser.add_argument('csv_file', help='Rational coefficient CSV file')
    parser.add_argument('--range', '-r', nargs=2, type=float, default=[-10, 10],
                        help='Test range (min max)')
    parser.add_argument('--points', '-n', type=int, default=1024,
                        help='Number of test points')
    parser.add_argument('--hardware', '-hw', help='Hardware output CSV file for comparison')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode (only print summary)')

    args = parser.parse_args()

    test_range = tuple(args.range)
    verbose = not args.quiet

    result = test_rational_lut(
        args.csv_file,
        test_range=test_range,
        num_points=args.points,
        hardware_csv=args.hardware,
        verbose=verbose
    )

    if args.hardware:
        # Print summary for CSV parsing
        metrics = result
        print(f"\nSUMMARY: MAE={metrics['mae']:.6e}, RMSE={metrics['rmse']:.6e}, MaxError={metrics['max_error']:.6e}")

        # Exit code based on accuracy
        if metrics['mae'] > 1e-2:
            print("FAILED: MAE exceeds threshold (1e-2)")
            return 1
        else:
            print("PASSED")
            return 0
    else:
        print("PASSED (reference computed)")
        return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
