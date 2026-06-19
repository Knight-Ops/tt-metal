#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Composite Operation Runner

This script runs composite activation functions using TTNN operations,
generating output compatible with sweep_native_sfpu.sh infrastructure.

Usage:
    python ttnn_composite_runner.py <activation_name> \
        --precision bf16|fp32 \
        --range-min -10 \
        --range-max 10 \
        [--tiles N] \
        [--fast-approx | --no-fast-approx]

Fast/Precise Modes:
    - Operations that support fast_and_approximate_mode: exp, erf, gelu, rsqrt, erfc
    - Other composite operations ignore this flag (always use default mode)
    - Both modes are tested for consistency with native SFPU sweeps

Environment:
    DUMP_OUTPUT_CSV: If set, writes output CSV to specified path

Output Format (compatible with native kernel):
    - TIMING_* lines for profiling
    - "✓ Test PASSED" on success
    - CSV output when DUMP_OUTPUT_CSV is set
"""

import torch
import ttnn
import argparse
import sys
import os
import csv
import time
import numpy as np
from exhaustive_bf16_inputs import generate_exhaustive_bf16_torch, should_use_exhaustive_bf16


# ============================================================================
# Multi-op composites (can't be expressed as a single ttnn call)
# ============================================================================

def _tanhshrink(x):
    """Tanhshrink: x - tanh(x)"""
    return ttnn.sub(x, ttnn.tanh(x))


def _hardshrink(x, lambda_val=0.5):
    """Hardshrink: x if |x| > λ else 0"""
    mask = ttnn.gt(ttnn.abs(x), ttnn.full_like(x, lambda_val))
    return ttnn.where(mask, x, ttnn.zeros_like(x))


def _logit(x, eps=1e-7):
    """Logit: log(x / (1 - x)) with clamping"""
    x_clamped = ttnn.clip(x, eps, 1.0 - eps)
    one_minus_x = ttnn.sub(ttnn.ones_like(x_clamped), x_clamped)
    return ttnn.log(ttnn.div(x_clamped, one_minus_x))


def _relu_max(x, max_val=6.0):
    """ReLU with maximum: min(max(0, x), max_val)"""
    return ttnn.minimum(ttnn.relu(x), ttnn.full_like(x, max_val))


def _relu_min(x, min_val=0.0):
    """ReLU with minimum: max(relu(x), min_val)"""
    return ttnn.maximum(ttnn.relu(x), ttnn.full_like(x, min_val))


# HARDCODED PARAMETER: polygamma requires an order k (positional arg).
# The sweep framework has no mechanism to pass per-activation extra args,
# so we hardcode k=1 (trigamma). Change this if you need a different order.
_POLYGAMMA_K = 1

def _polygamma(x, fast=True):
    """polygamma with hardcoded order k (see _POLYGAMMA_K above)"""
    print(f"  [NOTE] polygamma using hardcoded k={_POLYGAMMA_K} (trigamma)")
    return ttnn.polygamma(x, _POLYGAMMA_K)


# Name aliases: activation name -> ttnn function name (only where they differ)
_NAME_ALIASES = {
    "hardtanh": "clip",
    "identity": "clone",
    "negative": "neg",
    "recip": "reciprocal",
    "logsigmoid": "log_sigmoid",
    "swish": "silu",
}

# Multi-op composites that can't be expressed as a single ttnn call
# (includes ops that need extra positional args the generic wrapper can't provide)
_COMPOSITE_FNS = {
    "tanhshrink": _tanhshrink,
    "hardshrink": _hardshrink,
    "logit": _logit,
    "relu_max": _relu_max,
    "relu_min": _relu_min,
    "polygamma": _polygamma,
}

# Default kwargs for operations that need them (activation_name -> kwargs dict)
_DEFAULT_KWARGS = {
    "celu": {"alpha": 1.0},
    "elu": {"alpha": 1.0},
    "leaky_relu": {"negative_slope": 0.01},
    "prelu": {"weight": 0.25},
    "softplus": {"beta": 1.0, "threshold": 20.0},
    "softshrink": {"lambda_param": 0.5},
    "threshold": {"threshold": 0.0, "value": 0.0},
    "clip": {"min": -1.0, "max": 1.0},  # used by hardtanh alias
}


def _resolve_ttnn_fn(name):
    """Resolve a ttnn function by name, trying the name directly then underscore variant."""
    fn = getattr(ttnn, name, None)
    if fn is not None and callable(fn):
        return fn
    # Try underscore-separated variant (e.g., sigmoid_accurate -> sigmoid_accurate)
    underscore_name = name.replace("-", "_")
    if underscore_name != name:
        fn = getattr(ttnn, underscore_name, None)
        if fn is not None and callable(fn):
            return fn
    return None


def get_activation_fn(name):
    """
    Dynamically resolve an activation function by name.
    Returns a callable(x, fast=True) or None if not found.
    """
    # 1. Multi-op composites (can't be a single ttnn call)
    if name in _COMPOSITE_FNS:
        return _COMPOSITE_FNS[name]

    # 2. Resolve the ttnn function (direct name or alias)
    ttnn_name = _NAME_ALIASES.get(name, name)
    fn = _resolve_ttnn_fn(ttnn_name)
    if fn is None:
        return None

    # 3. Build the wrapper with appropriate kwargs
    kwargs = _DEFAULT_KWARGS.get(ttnn_name, _DEFAULT_KWARGS.get(name, {}))

    # Check if this is a backward op (ends with _bw)
    is_backward_op = name.endswith("_bw")

    def wrapper(x, fast=True, _fn=fn, _kwargs=kwargs, _name=name, _is_bw=is_backward_op):
        call_kwargs = dict(_kwargs)

        if _is_bw:
            # Backward ops take (grad_tensor, input_tensor) and return a list
            # Create grad_tensor of all 1s so output = derivative(input)
            grad_tensor = ttnn.ones_like(x)
            # Try with approximate kwarg for gelu_bw, tanh_bw etc
            try:
                result = _fn(grad_tensor, x, approximate="none", **call_kwargs)
            except TypeError:
                result = _fn(grad_tensor, x, **call_kwargs)
            # Return first element of the list (input gradient)
            return result[0] if isinstance(result, (list, tuple)) else result

        # Forward ops: try fast_and_approximate_mode — if the op doesn't accept it, retry without
        try:
            return _fn(x, fast_and_approximate_mode=fast, **call_kwargs)
        except TypeError:
            return _fn(x, **call_kwargs)

    return wrapper


class _DynamicOperationMap:
    """Dict-like object that resolves activations dynamically from ttnn."""

    def __contains__(self, name):
        return get_activation_fn(name) is not None

    def __getitem__(self, name):
        fn = get_activation_fn(name)
        if fn is None:
            raise KeyError(name)
        return fn

    def keys(self):
        return list(_COMPOSITE_FNS.keys()) + list(_NAME_ALIASES.keys())


COMPOSITE_OPERATIONS = _DynamicOperationMap()


# ============================================================================
# Main Runner
# ============================================================================

def run_composite_activation(
    activation_name,
    range_min=-10.0,
    range_max=10.0,
    rows=32,
    cols=32,
    precision="bf16",
    fast_approx=True
):
    """
    Run a composite activation function using TTNN.

    Args:
        rows: Number of rows (must be multiple of 32)
        cols: Number of columns (must be multiple of 32)

    Returns: (success, timings_dict)
    """

    # Check if operation is available
    if activation_name not in COMPOSITE_OPERATIONS:
        print(f"ERROR: Unknown activation: {activation_name} (not found in ttnn or composite ops)", file=sys.stderr)
        return False, {}

    # Configuration
    use_bf16 = (precision == "bf16")
    dtype = ttnn.bfloat16 if use_bf16 else ttnn.float32
    num_elements = rows * cols

    # Timing dictionary
    timings = {}

    # ==================================================================
    # STEP 1: Device Initialization
    # ==================================================================
    t_start = time.perf_counter()
    device = ttnn.open_device(device_id=0)
    t_end = time.perf_counter()
    timings['device_init'] = (t_end - t_start) * 1000  # Convert to ms

    try:
        # ==================================================================
        # STEP 2: Program Creation (N/A for TTNN, but included for compatibility)
        # ==================================================================
        t_start = time.perf_counter()
        # No program creation needed for TTNN
        t_end = time.perf_counter()
        timings['program_creation'] = (t_end - t_start) * 1000

        # ==================================================================
        # STEP 3: Buffer Allocation (handled by TTNN internally)
        # ==================================================================
        t_start = time.perf_counter()
        # Buffer allocation is implicit in TTNN
        t_end = time.perf_counter()
        timings['buffer_alloc'] = (t_end - t_start) * 1000

        # ==================================================================
        # STEP 4: Data Preparation
        # ==================================================================
        t_start = time.perf_counter()

        # Generate input data
        if should_use_exhaustive_bf16(precision, num_elements):
            # Use exhaustive BF16 generation (all possible BF16 values in range)
            torch_input = generate_exhaustive_bf16_torch(range_min, range_max, num_elements, dtype=torch.float32)
        else:
            # Use linear spacing for FP32 (exhaustive not feasible - 2^32 values)
            torch_input = torch.linspace(range_min, range_max, num_elements, dtype=torch.float32)

        # Reshape to tile-friendly dimensions (rows and cols already multiples of 32)
        torch_input = torch_input.reshape(1, 1, rows, cols)

        t_end = time.perf_counter()
        timings['data_prep'] = (t_end - t_start) * 1000

        # ==================================================================
        # STEP 5: Host to Device Transfer
        # ==================================================================
        t_start = time.perf_counter()

        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )

        t_end = time.perf_counter()
        timings['host_to_device'] = (t_end - t_start) * 1000

        # ==================================================================
        # STEP 6: Kernel Creation (N/A for TTNN, but included for compatibility)
        # ==================================================================
        t_start = time.perf_counter()
        # No kernel creation needed for TTNN - operations are pre-compiled
        t_end = time.perf_counter()
        timings['kernel_creation'] = (t_end - t_start) * 1000

        # ==================================================================
        # STEP 7: Kernel Execution (Apply Activation)
        # ==================================================================
        t_start = time.perf_counter()

        activation_fn = COMPOSITE_OPERATIONS[activation_name]

        # All operations now accept fast parameter (ignored if not applicable)
        # Try calling with fast parameter, fall back to no parameter if TypeError
        try:
            ttnn_output = activation_fn(ttnn_input, fast=fast_approx)
        except TypeError:
            # Composite operations that don't take fast parameter
            ttnn_output = activation_fn(ttnn_input)

        # Ensure computation completes
        ttnn.device.synchronize_device(device)

        t_end = time.perf_counter()
        timings['kernel_exec'] = (t_end - t_start) * 1000

        # ==================================================================
        # STEP 8: Device to Host Transfer
        # ==================================================================
        t_start = time.perf_counter()

        torch_output = ttnn.to_torch(ttnn_output)

        t_end = time.perf_counter()
        timings['device_to_host'] = (t_end - t_start) * 1000

        # ==================================================================
        # STEP 9: Export CSV if requested
        # ==================================================================
        output_csv = os.environ.get('DUMP_OUTPUT_CSV')
        if output_csv:
            # Flatten and extract actual data (remove padding)
            # Convert to float32 first since numpy doesn't support bfloat16
            input_np = torch_input.float().numpy().flatten()[:num_elements]
            output_np = torch_output.float().numpy().flatten()[:num_elements]

            # Write CSV (match native SFPU format: 10 decimal places)
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            with open(output_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['input', 'output'])
                for inp, out in zip(input_np, output_np):
                    # Format with 17 significant figures in scientific notation to preserve BF16 precision
                    writer.writerow([f'{float(inp):.17e}', f'{float(out):.17e}'])

        return True, timings

    finally:
        ttnn.close_device(device)


def main():
    parser = argparse.ArgumentParser(
        description="Run TTNN composite operations (compatible with sweep_native_sfpu.sh)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("activation", help="Activation function name")
    parser.add_argument("--precision", "-p", choices=["bf16", "fp32"], default="bf16",
                       help="Precision mode (default: bf16)")
    parser.add_argument("--range-min", "-rmin", type=float, required=True,
                       help="Minimum input range value")
    parser.add_argument("--range-max", "-rmax", type=float, required=True,
                       help="Maximum input range value")
    parser.add_argument("--rows", type=int, default=32,
                       help="Number of rows (must be multiple of 32, default: 32)")
    parser.add_argument("--cols", type=int, default=32,
                       help="Number of columns (must be multiple of 32, default: 32)")
    parser.add_argument("--fast-approx", action="store_true",
                       help="Use fast approximate mode (for exp, erf, gelu that support it)")
    parser.add_argument("--no-fast-approx", action="store_true",
                       help="Use precise mode (for exp, erf, gelu that support it)")

    args = parser.parse_args()

    # Validate activation name
    if args.activation not in COMPOSITE_OPERATIONS:
        print(f"ERROR: Unknown activation: {args.activation} (not found in ttnn or composite ops)", file=sys.stderr)
        return 1

    # Validate range
    if args.range_min >= args.range_max:
        print(f"ERROR: range-min ({args.range_min}) must be less than range-max ({args.range_max})", file=sys.stderr)
        return 1

    # Print header
    print("=" * 60)
    print("TTNN Composite Activation Function")
    print("=" * 60)
    print(f"Activation: {args.activation}\n")

    print(f"Activation: {args.activation} | Test range: [{args.range_min}, {args.range_max}]")
    print(f"Precision mode: {args.precision.upper()}")
    print(f"SFPU mode: {'Fast approximate' if args.fast_approx else 'Precise (slower)'}")

    # Validate shape (must be multiples of 32)
    if args.rows % 32 != 0 or args.cols % 32 != 0:
        print(f"ERROR: rows ({args.rows}) and cols ({args.cols}) must be multiples of 32", file=sys.stderr)
        return 1

    # Run the activation
    success, timings = run_composite_activation(
        args.activation,
        args.range_min,
        args.range_max,
        args.rows,
        args.cols,
        args.precision,
        args.fast_approx
    )

    if not success:
        return 1

    # Print timing results in format expected by sweep script
    print(f"TIMING_DEVICE_INIT: {timings['device_init']:.6f}")
    print(f"TIMING_PROGRAM_CREATION: {timings['program_creation']:.6f}")
    print(f"TIMING_BUFFER_ALLOCATION: {timings['buffer_alloc']:.6f}")
    print(f"TIMING_DATA_PREPARATION: {timings['data_prep']:.6f}")
    print(f"TIMING_HOST_TO_DEVICE: {timings['host_to_device']:.6f}")
    print(f"TIMING_KERNEL_CREATION: {timings['kernel_creation']:.6f}")
    print(f"TIMING_KERNEL_EXECUTION: {timings['kernel_exec']:.6f}")
    print(f"TIMING_DEVICE_TO_HOST: {timings['device_to_host']:.6f}")

    # Print sample output
    output_csv = os.environ.get('DUMP_OUTPUT_CSV')
    if output_csv and os.path.exists(output_csv):
        print("\nResults (samples across input range):")
        print("-" * 60)
        print(f"{'Index':>5} {'Input':>12} {'Output':>12}")
        print("-" * 60)

        # Read and display samples
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            indices = [0, len(rows)//4, len(rows)//2, 3*len(rows)//4, len(rows)-1]
            for i in indices:
                if i < len(rows):
                    row = rows[i]
                    print(f"{i:5d} {float(row['input']):12.6f} {float(row['output']):12.6f}")

        print("-" * 60)
        print(f"\n✓ Output data dumped to: {output_csv}")

    # Calculate tile count
    n_tiles = (args.rows // 32) * (args.cols // 32)
    print(f"\nProcessed shape ({args.rows}, {args.cols}) = {n_tiles} tiles")

    # Final success message (required by sweep script)
    print("\n" + "=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
