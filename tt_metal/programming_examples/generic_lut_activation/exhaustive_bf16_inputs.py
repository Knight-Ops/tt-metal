#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Exhaustive BF16 Input Generation

This module provides utilities for generating exhaustive BF16 test inputs by iterating
all possible BF16 bit patterns. This ensures we test worst-case bit patterns and can
detect catastrophic ULP errors that might not appear with linearly-spaced inputs.

For example, linear spacing in range [-10, 10] with spacing ~6e-06 misses tiny normal
values near 2^-126 (≈1.175e-38) where polynomials can have ULP errors of 1e+36.
"""

import numpy as np
import struct
import torch


def is_bf16_subnormal(bits: int) -> bool:
    """
    Check if a BF16 bit pattern represents a subnormal number.

    Subnormal: exponent == 0 AND mantissa != 0

    Args:
        bits: 16-bit unsigned integer representing BF16 bit pattern

    Returns:
        True if subnormal, False otherwise
    """
    exponent = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    return (exponent == 0x00) and (mantissa != 0x00)


def bf16_bits_to_float(bits: int) -> float:
    """
    Convert BF16 bit pattern to float32 value.

    BF16 is stored in upper 16 bits of FP32, so we shift left by 16.

    Args:
        bits: 16-bit unsigned integer representing BF16 bit pattern

    Returns:
        Float32 value corresponding to the BF16 bit pattern
    """
    fp32_bits = bits << 16
    return struct.unpack('f', struct.pack('I', fp32_bits))[0]


def generate_exhaustive_bf16_values(range_min: float, range_max: float) -> np.ndarray:
    """
    Generate all possible BF16 values in the specified range, excluding subnormals.

    This function iterates through all 2^16 = 65,536 possible BF16 bit patterns,
    filters out subnormals (which hardware treats as zero in FTZ mode), and
    returns only values within the specified test range.

    Args:
        range_min: Minimum value of test range (inclusive)
        range_max: Maximum value of test range (inclusive)

    Returns:
        Sorted numpy array of unique BF16 values in range [range_min, range_max]

    Example:
        >>> inputs = generate_exhaustive_bf16_values(-10.0, 10.0)
        >>> print(f"Found {len(inputs)} unique BF16 values")
        >>> print(f"Smallest: {inputs[0]:.10e}, Largest: {inputs[-1]:.10e}")
    """
    bf16_values = []

    # Iterate all 2^16 = 65,536 possible BF16 bit patterns
    for bits in range(65536):
        # Skip subnormals (hardware uses FTZ mode)
        if is_bf16_subnormal(bits):
            continue

        # Convert to float
        val = bf16_bits_to_float(bits)

        # Filter to test range and check for valid values
        if np.isfinite(val) and range_min <= val <= range_max:
            bf16_values.append(val)

    # Return as sorted numpy array for reproducibility
    return np.array(sorted(bf16_values), dtype=np.float32)


def generate_exhaustive_bf16_torch(
    range_min: float,
    range_max: float,
    num_elements: int,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generate exhaustive BF16 inputs as a PyTorch tensor, tiling to desired length.

    This generates all unique BF16 values in the range, then tiles/repeats them
    to fill the requested number of elements. This ensures comprehensive testing
    across all representable BF16 values.

    Args:
        range_min: Minimum value of test range
        range_max: Maximum value of test range
        num_elements: Number of elements to generate (will tile exhaustive set)
        dtype: PyTorch dtype for output tensor (default: torch.float32)

    Returns:
        PyTorch tensor of shape (num_elements,) containing exhaustive BF16 inputs

    Example:
        >>> inputs = generate_exhaustive_bf16_torch(-10.0, 10.0, 1000000)
        >>> print(f"Generated {len(inputs)} elements from {len(torch.unique(inputs))} unique values")
    """
    # Generate exhaustive BF16 values
    unique_values = generate_exhaustive_bf16_values(range_min, range_max)

    if len(unique_values) == 0:
        raise ValueError(f"No BF16 values found in range [{range_min}, {range_max}]")

    # Tile/repeat to fill requested number of elements
    num_unique = len(unique_values)
    num_repeats = (num_elements + num_unique - 1) // num_unique

    # Create tiled array
    tiled = np.tile(unique_values, num_repeats)[:num_elements]

    # Convert to torch tensor
    return torch.from_numpy(tiled).to(dtype)


def should_use_exhaustive_bf16(precision: str, num_elements: int) -> bool:
    """
    Determine whether to use exhaustive BF16 generation or linear spacing.

    Exhaustive BF16 is only useful for BF16 precision. For FP32, it's not
    feasible (2^32 values), so we fall back to linear spacing.

    Args:
        precision: Precision mode ("bf16" or "fp32")
        num_elements: Number of elements requested

    Returns:
        True if exhaustive BF16 should be used, False for linear spacing
    """
    return precision.lower() in ['bf16', 'bfloat16']


if __name__ == "__main__":
    # Self-test / demo
    print("Exhaustive BF16 Input Generation - Self Test")
    print("=" * 80)
    print()

    # Test range [-10, 10] (typical for activations like GELU)
    test_min, test_max = -10.0, 10.0

    print(f"Generating exhaustive BF16 values in range [{test_min}, {test_max}]...")
    unique_values = generate_exhaustive_bf16_values(test_min, test_max)

    print(f"  Found {len(unique_values)} unique BF16 values")
    print(f"  Smallest positive: {min(v for v in unique_values if v > 0):.10e}")
    print(f"  Largest negative: {max(v for v in unique_values if v < 0):.10e}")
    print(f"  Smallest: {unique_values.min():.6f}")
    print(f"  Largest: {unique_values.max():.6f}")
    print()

    # Test torch tensor generation
    num_elements = 1000000
    print(f"Generating {num_elements} elements (tiling exhaustive set)...")
    torch_tensor = generate_exhaustive_bf16_torch(test_min, test_max, num_elements)

    print(f"  Tensor shape: {torch_tensor.shape}")
    print(f"  Unique values in tensor: {len(torch.unique(torch_tensor))}")
    print(f"  Min: {torch_tensor.min().item():.6f}")
    print(f"  Max: {torch_tensor.max().item():.6f}")
    print()

    # Show smallest positive normal BF16 value
    smallest_normal = 2.0 ** -126
    print(f"Smallest normal BF16 value: 2^-126 = {smallest_normal:.10e}")
    smallest_in_set = min(v for v in unique_values if v > 0)
    print(f"Smallest positive in set: {smallest_in_set:.10e}")
    print(f"Match? {abs(smallest_in_set - smallest_normal) < 1e-40}")
    print()

    print("✓ Self-test complete")
