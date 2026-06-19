#!/usr/bin/env python3
"""
Hardware Output CSV to Error CSV Converter

Converts hardware output CSVs (input,output) to error CSVs with computed metrics.
Caches results in {original}.err.npz files to avoid recomputation.

Usage:
    # Convert single file
    python3 hwoutcsv2errcsv.py input.csv --activation gelu --precision bf16

    # Convert all CSVs in directory
    python3 hwoutcsv2errcsv.py data/hardware_outputs/blackhole/gelu/ --activation gelu --precision bf16

    # Force recompute (ignore cache)
    python3 hwoutcsv2errcsv.py input.csv --activation gelu --precision bf16 --force

As a library:
    from hwoutcsv2errcsv import load_or_compute_errors, get_error_cache_path
    errors = load_or_compute_errors('hw_output.csv', 'gelu', 'bf16')
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add tt-polynomial-fitter to path for shared accuracy computation
TT_POLY_FIT_DIR = Path.home() / 'workspace' / 'tt-polynomial-fitter'
if TT_POLY_FIT_DIR.exists():
    sys.path.insert(0, str(TT_POLY_FIT_DIR))
    sys.path.insert(0, str(TT_POLY_FIT_DIR / 'competition'))
    from extract_accuracy import compute_error_arrays
    from precision_utils import to_bf16
    from ground_truth import compute_ground_truth
else:
    # Try alternate paths
    for alt_path in ['/localdev/nkapre/tt-polynomial-fitter', '/proj_sw/user_dev/nkapre/tt-polynomial-fitter']:
        if os.path.isdir(alt_path):
            sys.path.insert(0, alt_path)
            sys.path.insert(0, os.path.join(alt_path, 'competition'))
            from extract_accuracy import compute_error_arrays
            from precision_utils import to_bf16
            from ground_truth import compute_ground_truth
            break
    else:
        raise FileNotFoundError(f"tt-polynomial-fitter not found")

# Load activation functions dynamically from tt-polynomial-fitter/activations/*.json
import json
import importlib

def _load_activation_functions():
    """Load activation functions from JSON config files."""
    activations = {}

    # Special cases where PyTorch doesn't have a direct function
    special_cases = {
        'cbrt': lambda x: torch.sign(x) * torch.abs(x).pow(1.0/3.0),
        'identity': lambda x: x,
        'swish': lambda x: x * torch.sigmoid(x),  # Same as silu
    }

    # Find tt-polynomial-fitter activations directory
    search_paths = [
        Path.home() / 'workspace' / 'tt-polynomial-fitter' / 'activations',
        Path('/localdev/nkapre/tt-polynomial-fitter/activations'),
        Path('/proj_sw/user_dev/nkapre/tt-polynomial-fitter/activations'),
    ]

    activations_dir = None
    for p in search_paths:
        if p.exists():
            activations_dir = p
            break

    if activations_dir is None:
        # Fallback to minimal hardcoded set
        return {
            'gelu': lambda x: torch.nn.functional.gelu(x),
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'relu': torch.relu,
            **special_cases,
        }

    # Load each JSON file
    for json_file in activations_dir.glob('*.json'):
        try:
            with open(json_file) as f:
                config = json.load(f)

            name = config['name']

            # Check special cases first
            if name in special_cases:
                activations[name] = special_cases[name]
                continue

            pytorch_config = config.get('pytorch', {})
            module_name = pytorch_config.get('module', 'torch')
            func_name = pytorch_config.get('function', name)

            # Get the module and function
            module = importlib.import_module(module_name)
            func = getattr(module, func_name, None)

            if func is not None:
                activations[name] = func
        except Exception:
            pass  # Skip invalid JSON files

    return activations

ACTIVATION_FUNCTIONS = _load_activation_functions()


def load_hardware_outputs(csv_path):
    """Load hardware outputs from CSV file (input,output format)."""
    # Use numpy for fast CSV loading (skip invalid lines)
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1, ndmin=2, invalid_raise=False)
    if data.size == 0:
        return np.array([]), np.array([])
    # Filter out rows with NaN (from invalid lines)
    valid_mask = ~np.isnan(data).any(axis=1)
    data = data[valid_mask]
    if data.size == 0:
        return np.array([]), np.array([])
    return data[:, 0], data[:, 1]


def compute_pytorch_baseline(inputs, activation):
    """Compute PyTorch baseline for given inputs."""
    if activation not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation: {activation}. Available: {list(ACTIVATION_FUNCTIONS.keys())}")

    x = torch.tensor(inputs, dtype=torch.float32)
    with torch.no_grad():
        y = ACTIVATION_FUNCTIONS[activation](x)
    return y.numpy()


def downcast_to_precision(values, precision):
    """Downcast values to target precision."""
    if precision in ['bf16', 'bfloat16']:
        return to_bf16(values, flush_to_zero=True, rounding_mode='rne')
    elif precision in ['fp16', 'half', 'float16']:
        return np.asarray(values, dtype=np.float16).astype(np.float32)
    return np.asarray(values, dtype=np.float32)


def is_normal_number(values, dtype='bf16'):
    """Check if values are normal (not subnormal) numbers."""
    values = np.asarray(values, dtype=np.float32)
    if dtype in ['bf16', 'bfloat16']:
        min_normal = 2.0 ** -126
    elif dtype in ['fp16', 'half', 'float16']:
        min_normal = 2.0 ** -14
    else:
        min_normal = 2.0 ** -126
    return np.abs(values) >= min_normal


def get_error_cache_path(csv_path):
    """Get error cache file path: {original}.err.npz"""
    csv_path = Path(csv_path)
    return csv_path.parent / f'{csv_path.stem}.err.npz'


def is_cache_valid(csv_path, cache_path=None):
    """Check if cache exists and is newer than source CSV."""
    if cache_path is None:
        cache_path = get_error_cache_path(csv_path)
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False
    csv_mtime = os.path.getmtime(csv_path)
    cache_mtime = os.path.getmtime(cache_path)
    return cache_mtime > csv_mtime


def load_error_cache(cache_path):
    """Load errors from cache .npz file.

    Returns:
        dict with 'inputs', 'abs_errors', 'rel_errors', 'ulp_errors' arrays
        and summary statistics
    """
    # Use numpy binary format for instant loading
    data = np.load(cache_path)

    inputs = data['inputs']
    abs_arr = data['abs_errors']
    rel_arr = data['rel_errors']
    ulp_arr = data['ulp_errors']

    return {
        'inputs': inputs,
        'abs_errors': abs_arr,
        'rel_errors': rel_arr,
        'ulp_errors': ulp_arr,
        # Summary statistics
        'max_abs': np.nanmax(abs_arr),
        'mean_abs': np.nanmean(abs_arr),
        'max_rel': np.nanmax(rel_arr),
        'mean_rel': np.nanmean(rel_arr),
        'max_ulp': np.nanmax(ulp_arr),
        'mean_ulp': np.nanmean(ulp_arr),
        'median_ulp': np.nanmedian(ulp_arr),
    }


def save_error_cache(cache_path, inputs, abs_errors, rel_errors, ulp_errors):
    """Save errors to cache .npz file (compressed)."""
    np.savez_compressed(cache_path, inputs=inputs, abs_errors=abs_errors,
                        rel_errors=rel_errors, ulp_errors=ulp_errors)


def compute_errors(csv_path, activation, precision, filter_subnormals=True):
    """Compute all error metrics for a hardware output CSV.

    Args:
        csv_path: Path to hardware output CSV (input,output format)
        activation: Activation function name
        precision: 'bf16' or 'fp32'
        filter_subnormals: If True, filter out subnormal values

    Returns:
        dict with 'inputs', 'abs_errors', 'rel_errors', 'ulp_errors' arrays
        and summary statistics
    """
    inputs, outputs = load_hardware_outputs(csv_path)
    # Use FP64 ground truth from ground_truth.py (same as extract_accuracy.py)
    # This ensures hardware and simulated error analysis use identical baselines
    baseline = compute_ground_truth(activation, inputs)

    if filter_subnormals:
        # Filter to normal numbers
        baseline_downcast = downcast_to_precision(baseline, precision)
        x_normal = is_normal_number(inputs, dtype=precision)
        y_normal = is_normal_number(baseline_downcast, dtype=precision)
        normal_mask = x_normal & y_normal

        inputs = inputs[normal_mask]
        outputs = outputs[normal_mask]
        baseline = baseline[normal_mask]

    # Compute all error metrics
    abs_errors, rel_errors, ulp_errors, valid_mask = compute_error_arrays(
        baseline, outputs, precision=precision, inputs=inputs
    )
    abs_errors = np.where(valid_mask, abs_errors, np.nan)
    rel_errors = np.where(valid_mask, rel_errors, np.nan)
    ulp_errors = np.where(valid_mask, ulp_errors, np.nan)

    # Sort by input
    sort_idx = np.argsort(inputs)
    inputs = inputs[sort_idx]
    abs_errors = abs_errors[sort_idx]
    rel_errors = rel_errors[sort_idx]
    ulp_errors = ulp_errors[sort_idx]

    return {
        'inputs': inputs,
        'abs_errors': abs_errors,
        'rel_errors': rel_errors,
        'ulp_errors': ulp_errors,
        # Summary statistics
        'max_abs': np.nanmax(abs_errors),
        'mean_abs': np.nanmean(abs_errors),
        'max_rel': np.nanmax(rel_errors),
        'mean_rel': np.nanmean(rel_errors),
        'max_ulp': np.nanmax(ulp_errors),
        'mean_ulp': np.nanmean(ulp_errors),
        'median_ulp': np.nanmedian(ulp_errors),
    }


def load_or_compute_errors(csv_path, activation, precision, force=False, verbose=False):
    """Load errors from cache or compute and save.

    Args:
        csv_path: Path to hardware output CSV
        activation: Activation function name
        precision: 'bf16' or 'fp32'
        force: If True, recompute even if cache exists
        verbose: If True, print cache status

    Returns:
        dict with error arrays and statistics, or None on failure
    """
    cache_path = get_error_cache_path(csv_path)

    # Try to load from cache
    if not force and is_cache_valid(csv_path, cache_path):
        try:
            result = load_error_cache(cache_path)
            if verbose:
                print(f"  [cached] {Path(csv_path).name}")
            return result
        except Exception:
            pass  # Cache corrupted, recompute

    # Compute errors
    try:
        error_data = compute_errors(csv_path, activation, precision)

        # Save to cache
        save_error_cache(
            cache_path,
            error_data['inputs'],
            error_data['abs_errors'],
            error_data['rel_errors'],
            error_data['ulp_errors']
        )

        return error_data
    except Exception as e:
        print(f"  Warning: Failed to compute errors for {csv_path}: {e}")
        return None


def process_file(csv_path, activation, precision, force=False, verbose=True):
    """Process a single hardware output CSV file.

    Returns:
        Path to error cache file, or None on failure
    """
    cache_path = get_error_cache_path(csv_path)
    was_cached = not force and is_cache_valid(csv_path, cache_path)

    error_data = load_or_compute_errors(csv_path, activation, precision, force)
    if error_data is None:
        return None

    if verbose:
        status = "cached" if was_cached else "computed"
        print(f"  [{status}] {Path(csv_path).name} -> max_ulp={error_data['max_ulp']:.2f}")

    return cache_path


def _process_file_worker(args):
    """Worker for parallel file processing."""
    csv_path, activation, precision, force = args
    cache_path = get_error_cache_path(csv_path)
    was_cached = not force and is_cache_valid(csv_path, cache_path)

    error_data = load_or_compute_errors(csv_path, activation, precision, force)
    if error_data is None:
        return ('failed', csv_path, None)

    status = "cached" if was_cached else "computed"
    return (status, csv_path, error_data['max_ulp'])


def process_directory(dir_path, activation, precision, force=False, pattern='*.csv', parallel=1):
    """Process all hardware output CSVs in a directory.

    Args:
        dir_path: Directory containing CSV files
        activation: Activation function name
        precision: 'bf16' or 'fp32'
        force: If True, recompute even if cache exists
        pattern: Glob pattern for CSV files
        parallel: Number of parallel workers (default: 1)

    Returns:
        tuple (processed_count, cached_count, failed_count)
    """
    dir_path = Path(dir_path)
    csv_files = list(dir_path.glob(pattern))

    # Filter out .err.npz files
    csv_files = [f for f in csv_files if not f.name.endswith('.err.npz')]

    if not csv_files:
        print(f"No CSV files found in {dir_path}")
        return 0, 0, 0

    # Separate already-cached vs needs-computation
    needs_compute = []
    already_cached = []
    for f in csv_files:
        if not force and is_cache_valid(str(f)):
            already_cached.append(f)
        else:
            needs_compute.append(f)

    print(f"Processing {len(csv_files)} CSV files in {dir_path} "
          f"({len(already_cached)} cached, {len(needs_compute)} to compute)")

    processed = len(already_cached)
    cached = len(already_cached)
    failed = 0

    if not needs_compute:
        print(f"Summary: {processed} processed ({cached} from cache), {failed} failed")
        return processed, cached, failed

    if parallel > 1 and len(needs_compute) > 1:
        from multiprocessing import Pool
        args = [(str(f), activation, precision, force) for f in needs_compute]
        n_workers = min(parallel, len(needs_compute))
        print(f"Computing {len(needs_compute)} files with {n_workers} workers...")
        with Pool(n_workers) as pool:
            for i, (status, path, max_ulp) in enumerate(
                    pool.imap_unordered(_process_file_worker, args)):
                if status == 'failed':
                    failed += 1
                    print(f"  [{i+1}/{len(needs_compute)}] FAILED {Path(path).name}")
                else:
                    processed += 1
                    print(f"  [{i+1}/{len(needs_compute)}] [{status}] "
                          f"{Path(path).name} -> max_ulp={max_ulp:.2f}")
    else:
        for csv_path in needs_compute:
            result = process_file(csv_path, activation, precision, force, verbose=True)
            if result is None:
                failed += 1
            else:
                processed += 1

    print(f"Summary: {processed} processed ({cached} from cache), {failed} failed")
    return processed, cached, failed


def main():
    parser = argparse.ArgumentParser(
        description='Convert hardware output CSV to error CSV with caching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert single file
    python3 hwoutcsv2errcsv.py gelu_bf16_p5_s64_tiles1600.csv --activation gelu

    # Convert all CSVs in directory
    python3 hwoutcsv2errcsv.py data/hardware_outputs/blackhole/gelu/ --activation gelu

    # Force recompute
    python3 hwoutcsv2errcsv.py input.csv --activation gelu --force
"""
    )
    parser.add_argument('path', type=str, help='CSV file or directory to process')
    parser.add_argument('--activation', '-a', type=str, required=True,
                        help=f'Activation function: {", ".join(ACTIVATION_FUNCTIONS.keys())}')
    parser.add_argument('--precision', '-p', type=str, default='bf16',
                        choices=['bf16', 'fp32'], help='Precision (default: bf16)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force recompute even if cache exists')
    parser.add_argument('--pattern', type=str, default='*.csv',
                        help='Glob pattern for directory processing (default: *.csv)')
    parser.add_argument('--parallel', '-j', type=int, default=1,
                        help='Parallel workers for directory processing (default: 1)')

    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        result = process_file(path, args.activation, args.precision, args.force)
        if result:
            print(f"Output: {result}")
        else:
            sys.exit(1)
    elif path.is_dir():
        processed, cached, failed = process_directory(
            path, args.activation, args.precision, args.force, args.pattern,
            parallel=args.parallel
        )
        if failed > 0:
            sys.exit(1)
    else:
        print(f"Error: {path} is not a file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
