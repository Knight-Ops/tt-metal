#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Tracy Profiling Runner

This script runs TTNN composite operations with Tracy profiling markers,
compatible with sweep_native_sfpu.sh when launched via `python -m tracy`.

Usage:
    python -m tracy -r -p --no-runtime-analysis -o <output_dir> -n <name> \
        ttnn_tracy_runner.py <iterations> <precision> <activation> <tiles> \
        <range_min> <range_max> [--fast-approx]

Arguments:
    iterations: Number of benchmark iterations
    precision: bf16 or fp32
    activation: Activation function name
    tiles: Number of tiles to process
    range_min: Minimum input range value
    range_max: Maximum input range value
    --fast-approx: Use fast approximate mode (optional)

Tracy Integration:
    - Uses signpost markers to define profiling regions
    - Launched via `python -m tracy` for profiling
    - Generates ops_perf_results CSV with hardware timing
"""

import torch
import ttnn
import sys
import os

# Import Tracy signpost for profiling markers
try:
    from tracy import signpost
except ImportError:
    # Fallback if Tracy not available (shouldn't happen when launched via tracy)
    def signpost(msg):
        pass

# Import operation definitions from composite runner
from ttnn_runner import COMPOSITE_OPERATIONS
from exhaustive_bf16_inputs import generate_exhaustive_bf16_torch, should_use_exhaustive_bf16


def run_tracy_benchmark(
    activation_name,
    iterations,
    range_min,
    range_max,
    rows,
    cols,
    precision,
    fast_approx=True
):
    """
    Run TTNN activation with Tracy profiling markers.

    Args:
        rows: Number of rows (must be multiple of 32)
        cols: Number of columns (must be multiple of 32)

    Returns: success (bool)
    """

    # Configuration
    use_bf16 = (precision == "bf16")
    dtype = ttnn.bfloat16 if use_bf16 else ttnn.float32
    num_elements = rows * cols

    # Initialize device
    signpost("DEVICE_INIT_START")
    device = ttnn.open_device(device_id=0)
    signpost("DEVICE_INIT_END")

    try:
        # Prepare data
        signpost("DATA_PREP_START")
        use_exhaustive = should_use_exhaustive_bf16(precision, num_elements)
        print(f"DEBUG: precision={precision}, num_elements={num_elements}, use_exhaustive={use_exhaustive}")
        if use_exhaustive:
            # Use exhaustive BF16 generation (all possible BF16 values in range)
            print(f"DEBUG: Using exhaustive BF16 generation for range [{range_min}, {range_max}]")
            torch_input = generate_exhaustive_bf16_torch(range_min, range_max, num_elements, dtype=torch.float32)
            print(f"DEBUG: Generated {len(torch_input)} elements, unique: {len(torch.unique(torch_input))}")
        else:
            # Use linear spacing for FP32 (exhaustive not feasible - 2^32 values)
            print(f"DEBUG: Using linear spacing (FP32 or non-BF16)")
            torch_input = torch.linspace(range_min, range_max, num_elements, dtype=torch.float32)

        # Reshape to tile-friendly dimensions (rows and cols already multiples of 32)
        torch_input = torch_input.reshape(1, 1, rows, cols)
        signpost("DATA_PREP_END")

        # Transfer to device
        signpost("HOST_TO_DEVICE_START")
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device
        )
        signpost("HOST_TO_DEVICE_END")

        # Get activation function
        activation_fn = COMPOSITE_OPERATIONS[activation_name]

        # Warm-up run (not profiled)
        try:
            _ = activation_fn(ttnn_input, fast=fast_approx)
        except TypeError:
            _ = activation_fn(ttnn_input)
        ttnn.device.synchronize_device(device)

        # Benchmark loop with Tracy markers
        signpost("BENCHMARK_START")
        for i in range(iterations):
            signpost(f"ITERATION_{i}_START")

            try:
                ttnn_output = activation_fn(ttnn_input, fast=fast_approx)
            except TypeError:
                ttnn_output = activation_fn(ttnn_input)

            # Ensure computation completes
            ttnn.device.synchronize_device(device)

            signpost(f"ITERATION_{i}_END")
        signpost("BENCHMARK_END")

        # Transfer results back (for validation, not timed in benchmark)
        signpost("DEVICE_TO_HOST_START")
        torch_output = ttnn.to_torch(ttnn_output)
        signpost("DEVICE_TO_HOST_END")

        # Optional: Write output CSV if environment variable set
        output_csv = os.environ.get('DUMP_OUTPUT_CSV')
        if output_csv:
            import csv
            input_np = torch_input.float().numpy().flatten()[:num_elements]
            output_np = torch_output.float().numpy().flatten()[:num_elements]

            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            with open(output_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['input', 'output'])
                for inp, out in zip(input_np, output_np):
                    writer.writerow([f'{float(inp):.17e}', f'{float(out):.17e}'])

        return True

    finally:
        ttnn.close_device(device)


def main():
    if len(sys.argv) < 8:
        print("Usage: ttnn_tracy_runner.py <iterations> <precision> <activation> <rows> <cols> <range_min> <range_max> [--fast-approx]", file=sys.stderr)
        return 1

    iterations = int(sys.argv[1])
    precision = sys.argv[2]
    activation_name = sys.argv[3]
    rows = int(sys.argv[4])
    cols = int(sys.argv[5])
    range_min = float(sys.argv[6])
    range_max = float(sys.argv[7])
    fast_approx = "--fast-approx" in sys.argv

    # Validate activation
    if activation_name not in COMPOSITE_OPERATIONS:
        print(f"ERROR: Unknown activation: {activation_name} (not found in ttnn or composite ops)", file=sys.stderr)
        return 1

    # Validate precision
    if precision not in ["bf16", "fp32"]:
        print(f"ERROR: Invalid precision: {precision}", file=sys.stderr)
        return 1

    # Validate shape
    if rows % 32 != 0 or cols % 32 != 0:
        print(f"ERROR: rows ({rows}) and cols ({cols}) must be multiples of 32", file=sys.stderr)
        return 1

    n_tiles = (rows // 32) * (cols // 32)
    print(f"Tracy Profiling: {activation_name} ({precision}, shape ({rows}, {cols}) = {n_tiles} tiles, {iterations} iterations)")
    print(f"Range: [{range_min}, {range_max}]")
    print(f"SFPU mode: {'fast' if fast_approx else 'precise'}")

    success = run_tracy_benchmark(
        activation_name,
        iterations,
        range_min,
        range_max,
        rows,
        cols,
        precision,
        fast_approx
    )

    if success:
        print("✓ Tracy profiling completed")
        return 0
    else:
        print("❌ Tracy profiling failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
