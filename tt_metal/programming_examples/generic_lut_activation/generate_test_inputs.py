#!/usr/bin/env python3
"""
Generate domain-aware test inputs for activation function benchmarks.

This utility demonstrates how to use activation domain specifications to
generate appropriate test inputs for hardware benchmarking and accuracy testing.

Usage:
    python3 generate_test_inputs.py <activation_name> <num_samples>

Examples:
    python3 generate_test_inputs.py sigmoid 1024
    python3 generate_test_inputs.py atanh 1024
    python3 generate_test_inputs.py sinh 1024 --random
"""

import numpy as np
import argparse
import os
import sys

# Add tt-polynomial-fitter to path for ground_truth module
_poly_fit_dir = os.environ.get('TT_POLY_FIT_DIR', '/localdev/nkapre/tt-polynomial-fitter')
sys.path.insert(0, _poly_fit_dir)
from ground_truth import get_activation_domain, get_activation_info, ACTIVATION_FUNCTIONS


def is_bf16_subnormal(bf16_bits):
    """
    Check if a BF16 bit pattern represents a subnormal (denormal) value.

    BF16 format: 1 sign + 8 exponent + 7 mantissa
    Subnormal: exponent = 0x00 (all zeros) AND mantissa != 0x00 (non-zero)

    Args:
        bf16_bits: uint16 BF16 bit pattern

    Returns:
        True if subnormal, False otherwise
    """
    exponent = bf16_bits & 0x7F80  # Mask exponent bits (bits 7-14)
    mantissa = bf16_bits & 0x007F  # Mask mantissa bits (bits 0-6)
    return (exponent == 0x0000) and (mantissa != 0x0000)


def generate_exhaustive_bf16_inputs(activation_name):
    """
    Generate ALL possible BF16 values within activation domain, EXCLUDING subnormals.

    CRITICAL: BF16 has only 2^16 = 65,536 total possible values.
    Linear sampling with np.linspace() only hits ~8% of BF16 values,
    missing worst-case errors that occur at specific BF16 bit patterns.

    This function generates EVERY possible BF16 value in the activation
    domain, ensuring true worst-case error measurement.

    SUBNORMAL FILTERING: Hardware flushes output subnormals to zero (FTZ mode).
    We filter input subnormals to match hardware behavior and avoid testing
    values that would be flushed during computation.

    Args:
        activation_name: Name of activation function

    Returns:
        numpy array of all normal BF16 values in domain (typically ~33K values for [-10, 10])
    """
    min_val, max_val = get_activation_domain(activation_name)

    # Handle infinite bounds
    if min_val == -np.inf:
        min_val = -10.0
    if max_val == np.inf:
        max_val = 10.0

    # Generate all 2^16 = 65,536 possible BF16 bit patterns
    all_bf16_bits = np.arange(2**16, dtype=np.uint16)

    # Filter out subnormals (hardware uses FTZ mode)
    normal_mask = ~np.array([is_bf16_subnormal(bits) for bits in all_bf16_bits])
    normal_bf16_bits = all_bf16_bits[normal_mask]

    # Convert BF16 bits to FP32 (BF16 occupies upper 16 bits of FP32)
    all_bf16_float32 = np.zeros(len(normal_bf16_bits), dtype=np.float32)
    for i, bits in enumerate(normal_bf16_bits):
        # BF16 to FP32: shift left 16 bits
        fp32_bits = np.uint32(bits) << 16
        all_bf16_float32[i] = np.frombuffer(fp32_bits.tobytes(), dtype=np.float32)[0]

    # Filter to activation domain
    in_range = (all_bf16_float32 >= min_val) & (all_bf16_float32 <= max_val)
    bf16_values = all_bf16_float32[in_range]

    # Sort and return
    return np.sort(bf16_values)


def generate_linearly_spaced_inputs(activation_name, num_samples):
    """
    Generate linearly spaced test inputs across the specified range.

    Args:
        activation_name: Name of activation function
        num_samples: Number of input samples to generate

    Returns:
        numpy array of input values
    """
    min_val, max_val = get_activation_domain(activation_name)

    # Handle infinite bounds
    if min_val == -np.inf:
        min_val = -10.0
    if max_val == np.inf:
        max_val = 10.0

    return np.linspace(min_val, max_val, num_samples)


def generate_random_uniform_inputs(activation_name, num_samples, seed=None):
    """
    Generate uniformly distributed random test inputs.

    Args:
        activation_name: Name of activation function
        num_samples: Number of input samples to generate
        seed: Random seed for reproducibility (None for random)

    Returns:
        numpy array of input values
    """
    if seed is not None:
        np.random.seed(seed)

    min_val, max_val = get_activation_domain(activation_name)

    # Handle infinite bounds
    if min_val == -np.inf:
        min_val = -10.0
    if max_val == np.inf:
        max_val = 10.0

    return np.random.uniform(min_val, max_val, num_samples)


def generate_normal_distribution_inputs(activation_name, num_samples,
                                       sigma_scale=0.3, seed=None):
    """
    Generate normally distributed test inputs centered around 0.

    This mimics typical activation distributions in neural networks where
    inputs are often centered around zero with moderate variance.

    Args:
        activation_name: Name of activation function
        num_samples: Number of input samples to generate
        sigma_scale: Standard deviation as fraction of range (default: 0.3)
        seed: Random seed for reproducibility (None for random)

    Returns:
        numpy array of input values, clipped to valid range
    """
    if seed is not None:
        np.random.seed(seed)

    min_val, max_val = get_activation_domain(activation_name)

    # Handle infinite bounds
    if min_val == -np.inf:
        min_val = -10.0
    if max_val == np.inf:
        max_val = 10.0

    # Compute standard deviation from range
    range_width = max_val - min_val
    sigma = range_width * sigma_scale

    # Generate normal distribution centered at 0
    samples = np.random.normal(0, sigma, num_samples)

    # Clip to valid range
    return np.clip(samples, min_val, max_val)


def generate_hybrid_inputs(activation_name, num_samples, seed=None):
    """
    Generate hybrid test inputs combining multiple strategies.

    Splits samples into:
    - 50% linearly spaced (coverage)
    - 25% uniform random (edge cases)
    - 25% normal distribution (realistic)

    Args:
        activation_name: Name of activation function
        num_samples: Number of input samples to generate (will be rounded to nearest multiple of 4)
        seed: Random seed for reproducibility (None for random)

    Returns:
        numpy array of input values
    """
    # Round to nearest multiple of 4 for clean splitting
    num_samples = (num_samples // 4) * 4

    n_linear = num_samples // 2
    n_uniform = num_samples // 4
    n_normal = num_samples - n_linear - n_uniform

    linear = generate_linearly_spaced_inputs(activation_name, n_linear)
    uniform = generate_random_uniform_inputs(activation_name, n_uniform, seed)
    normal = generate_normal_distribution_inputs(activation_name, n_normal, seed=seed+1 if seed else None)

    # Combine and shuffle
    combined = np.concatenate([linear, uniform, normal])
    if seed is not None:
        np.random.seed(seed + 2)
    np.random.shuffle(combined)

    return combined


def main():
    parser = argparse.ArgumentParser(
        description='Generate domain-aware test inputs for activation functions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1024 inputs for sigmoid (linearly spaced)
  python3 generate_test_inputs.py sigmoid 1024

  # Generate random inputs for atanh (respects restricted domain)
  python3 generate_test_inputs.py atanh 1024 --random

  # Generate hybrid inputs for sinh
  python3 generate_test_inputs.py sinh 1024 --hybrid

  # Show domain information
  python3 generate_test_inputs.py logsigmoid --info
        """
    )

    parser.add_argument('activation', help='Activation function name')
    parser.add_argument('num_samples', type=int, nargs='?', default=1024,
                       help='Number of samples to generate (default: 1024)')
    parser.add_argument('--random', action='store_true',
                       help='Generate uniform random inputs instead of linearly spaced')
    parser.add_argument('--normal', action='store_true',
                       help='Generate normally distributed inputs (centered at 0)')
    parser.add_argument('--hybrid', action='store_true',
                       help='Generate hybrid inputs (50%% linear, 25%% random, 25%% normal)')
    parser.add_argument('--exhaustive-bf16', action='store_true',
                       help='Generate ALL possible BF16 values in domain (~33K for typical range)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--info', action='store_true',
                       help='Show domain information and exit')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')

    args = parser.parse_args()

    # Validate activation name
    if args.activation.lower() not in ACTIVATION_FUNCTIONS:
        available = ', '.join(sorted(ACTIVATION_FUNCTIONS.keys()))
        print(f"ERROR: Unknown activation '{args.activation}'")
        print(f"Available: {available}")
        return 1

    # Show info mode
    if args.info:
        info = get_activation_info(args.activation)
        print(f"\nActivation: {args.activation}")
        print("=" * 70)
        print(f"Description: {info['description']}")
        print()
        print(f"Range: {info['range']}")
        print()
        return 0

    # Generate inputs based on mode
    if args.exhaustive_bf16:
        inputs = generate_exhaustive_bf16_inputs(args.activation)
        mode = 'exhaustive-bf16'
        print(f"WARNING: Ignoring num_samples={args.num_samples}, using ALL {len(inputs)} BF16 values",
              file=sys.stderr if args.output else sys.stdout)
    elif args.hybrid:
        inputs = generate_hybrid_inputs(args.activation, args.num_samples, args.seed)
        mode = 'hybrid'
    elif args.normal:
        inputs = generate_normal_distribution_inputs(args.activation, args.num_samples, seed=args.seed)
        mode = 'normal'
    elif args.random:
        inputs = generate_random_uniform_inputs(args.activation, args.num_samples, args.seed)
        mode = 'random'
    else:
        inputs = generate_linearly_spaced_inputs(args.activation, args.num_samples)
        mode = 'linear'

    # Print summary
    min_val, max_val = get_activation_domain(args.activation)
    print(f"Generated {len(inputs)} inputs for '{args.activation}'", file=sys.stderr if args.output else sys.stdout)
    print(f"  Mode: {mode}", file=sys.stderr if args.output else sys.stdout)
    print(f"  Domain range: {(min_val, max_val)}", file=sys.stderr if args.output else sys.stdout)
    print(f"  Actual range: [{inputs.min():.4f}, {inputs.max():.4f}]", file=sys.stderr if args.output else sys.stdout)
    print(f"  Mean: {inputs.mean():.4f}, Std: {inputs.std():.4f}", file=sys.stderr if args.output else sys.stdout)

    # Output
    if args.output:
        np.savetxt(args.output, inputs, fmt='%.10f')
        print(f"  Saved to: {args.output}", file=sys.stderr)
    else:
        print()
        # Print first 10 and last 10 samples
        print("First 10 samples:")
        for i in range(min(10, len(inputs))):
            print(f"  {i:4d}: {inputs[i]:12.6f}")
        if len(inputs) > 20:
            print("  ...")
            print("Last 10 samples:")
            for i in range(len(inputs) - 10, len(inputs)):
                print(f"  {i:4d}: {inputs[i]:12.6f}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
