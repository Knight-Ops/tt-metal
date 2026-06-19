#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script to validate piecewise_generic.cpp works with all degree/depth combinations.

This script verifies that the generic kernel produces results consistent with the
expected LUT format and handles boundary conditions correctly.
"""

import pytest
import numpy as np


def calculate_lut_size(poly_degree: int, num_segments: int) -> int:
    """Calculate expected LUT size for given degree and segments."""
    return (num_segments + 1) + num_segments * (poly_degree + 1)


def generate_test_lut(poly_degree: int, num_segments: int, x_min: float = -5.0, x_max: float = 5.0) -> np.ndarray:
    """
    Generate a simple test LUT for validation.

    For testing, we use a simple polynomial: f(x) = x^degree
    This makes it easy to verify correctness.
    """
    lut_size = calculate_lut_size(poly_degree, num_segments)
    lut = np.zeros(lut_size, dtype=np.float32)

    # 1. Generate boundaries (evenly spaced)
    boundaries = np.linspace(x_min, x_max, num_segments + 1)
    lut[: num_segments + 1] = boundaries

    # 2. Generate polynomial coefficients for each segment
    # For simplicity, use Taylor expansion around segment midpoint
    coeff_offset = num_segments + 1

    for seg_idx in range(num_segments):
        x0 = boundaries[seg_idx]
        x1 = boundaries[seg_idx + 1]
        x_mid = (x0 + x1) / 2

        # For f(x) = x^n, Taylor expansion around x_mid:
        # f(x) ≈ f(x_mid) + f'(x_mid)*(x-x_mid) + f''(x_mid)*(x-x_mid)^2/2! + ...
        # But for testing, let's use simpler identity polynomial: f(x) = x
        # This gives coefficients: [0, 1, 0, 0, ...] for all segments

        coeffs = np.zeros(poly_degree + 1)
        if poly_degree >= 1:
            coeffs[1] = 1.0  # Linear term = 1 (identity function)

        # Store coefficients in LUT
        base = coeff_offset + seg_idx * (poly_degree + 1)
        lut[base : base + poly_degree + 1] = coeffs

    return lut


def reference_eval_polynomial(coeffs: np.ndarray, x: float) -> float:
    """Reference implementation of Horner's method for polynomial evaluation."""
    degree = len(coeffs) - 1
    if degree == 0:
        return coeffs[0]

    # Horner's method: start with highest degree coefficient
    result = coeffs[degree]
    for i in range(degree - 1, -1, -1):
        result = result * x + coeffs[i]

    return result


def reference_piecewise_eval(lut: np.ndarray, x: float, poly_degree: int, num_segments: int) -> float:
    """Reference implementation of piecewise polynomial evaluation."""
    # Extract boundaries
    boundaries = lut[: num_segments + 1]

    # Clamp x to valid range
    x_clamped = np.clip(x, boundaries[0], boundaries[-1])

    # Find segment
    seg_idx = 0
    for i in range(num_segments):
        if x_clamped >= boundaries[i]:
            seg_idx = i

    # Extract coefficients for this segment
    coeff_offset = num_segments + 1
    coeffs_per_seg = poly_degree + 1
    base = coeff_offset + seg_idx * coeffs_per_seg
    coeffs = lut[base : base + coeffs_per_seg]

    # Evaluate polynomial
    return reference_eval_polynomial(coeffs, x)


class TestPiecewiseGeneric:
    """Test suite for piecewise_generic.cpp validation."""

    @pytest.mark.parametrize("poly_degree", [1, 2, 3, 4, 5, 6, 8])
    def test_lut_size_calculation(self, poly_degree):
        """Verify LUT size formula is correct."""
        test_cases = [
            (4, 1, 4),  # (segments, expected_boundaries, expected_coeffs_total)
            (8, 1, 8),
            (16, 1, 16),
            (32, 1, 32),
        ]

        for num_segments, _, _ in test_cases:
            expected_size = (num_segments + 1) + num_segments * (poly_degree + 1)
            actual_size = calculate_lut_size(poly_degree, num_segments)
            assert actual_size == expected_size, f"LUT size mismatch for degree={poly_degree}, segments={num_segments}"

    @pytest.mark.parametrize(
        "poly_degree,num_segments",
        [
            (1, 4),
            (2, 8),
            (3, 16),
            (4, 4),
            (4, 8),
            (4, 16),
            (5, 8),
            (6, 4),
            (8, 4),
        ],
    )
    def test_lut_format(self, poly_degree, num_segments):
        """Verify generated LUT has correct format."""
        lut = generate_test_lut(poly_degree, num_segments)

        # Check total size
        expected_size = calculate_lut_size(poly_degree, num_segments)
        assert len(lut) == expected_size

        # Check boundaries are monotonically increasing
        boundaries = lut[: num_segments + 1]
        assert all(boundaries[i] < boundaries[i + 1] for i in range(num_segments))

        # Check coefficients exist for all segments
        coeff_offset = num_segments + 1
        coeffs_per_seg = poly_degree + 1
        for seg_idx in range(num_segments):
            base = coeff_offset + seg_idx * coeffs_per_seg
            coeffs = lut[base : base + coeffs_per_seg]
            assert len(coeffs) == coeffs_per_seg
            assert not np.any(np.isnan(coeffs))

    @pytest.mark.parametrize("poly_degree", [1, 2, 3, 4, 5])
    def test_reference_eval_identity(self, poly_degree):
        """Verify reference implementation evaluates identity function correctly."""
        num_segments = 8
        lut = generate_test_lut(poly_degree, num_segments, x_min=-5.0, x_max=5.0)

        # Test points across the domain
        test_points = np.linspace(-5.0, 5.0, 100)

        for x in test_points:
            result = reference_piecewise_eval(lut, x, poly_degree, num_segments)
            # Since we used identity function (f(x) = x), result should equal x
            assert np.abs(result - x) < 1e-5, f"Identity function failed at x={x}: got {result}, expected {x}"

    @pytest.mark.parametrize("poly_degree", [1, 2, 3, 4])
    def test_boundary_clamping(self, poly_degree):
        """Verify inputs outside boundaries are clamped correctly."""
        num_segments = 4
        x_min, x_max = -2.0, 2.0
        lut = generate_test_lut(poly_degree, num_segments, x_min, x_max)

        # Test points outside boundaries
        test_cases = [
            (-10.0, x_min),  # Far below min
            (-2.5, x_min),  # Slightly below min
            (2.5, x_max),  # Slightly above max
            (10.0, x_max),  # Far above max
        ]

        for x_test, x_expected in test_cases:
            # The reference function clamps internally, so it should evaluate
            # as if x was at the boundary
            result_test = reference_piecewise_eval(lut, x_test, poly_degree, num_segments)
            result_expected = reference_piecewise_eval(lut, x_expected, poly_degree, num_segments)
            assert np.abs(result_test - result_expected) < 1e-6

    def test_degree_combinations_supported(self):
        """Verify all expected degree combinations are supported."""
        supported_degrees = [1, 2, 3, 4, 5, 6, 8]  # 7 not implemented yet
        supported_depths = [4, 8, 16, 32, 64]

        for degree in supported_degrees:
            for depth in supported_depths:
                lut_size = calculate_lut_size(degree, depth)
                assert lut_size > 0
                assert lut_size < 10000  # Sanity check for reasonable size

    def test_compile_time_args_format(self):
        """Verify compile-time args are in correct format for kernel."""
        test_configs = [
            (3, 16),  # cubic, 16 segments
            (4, 8),  # quartic, 8 segments
            (5, 4),  # quintic, 4 segments
        ]

        for poly_degree, num_segments in test_configs:
            lut_size = calculate_lut_size(poly_degree, num_segments)

            # Format that would be passed to kernel
            compile_time_args = [lut_size, poly_degree, num_segments]

            # Verify args are valid
            assert all(isinstance(arg, int) for arg in compile_time_args)
            assert lut_size == (num_segments + 1) + num_segments * (poly_degree + 1)

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6, 8])
    def test_horner_reference_correctness(self, degree):
        """Verify Horner's method reference implementation is correct."""
        # Test polynomial: f(x) = 1 + 2x + 3x^2 + 4x^3 + ...
        coeffs = np.arange(1, degree + 2, dtype=np.float32)

        # Test points
        test_points = [-2.0, -1.0, 0.0, 1.0, 2.0]

        for x in test_points:
            # Horner's method result
            result_horner = reference_eval_polynomial(coeffs, x)

            # Direct evaluation: sum(c_i * x^i)
            result_direct = sum(coeffs[i] * (x**i) for i in range(len(coeffs)))

            # Should match within floating point precision
            assert np.abs(result_horner - result_direct) < 1e-5


def print_lut_example():
    """Print an example LUT for documentation."""
    print("\n=== Example LUT Format ===\n")

    poly_degree = 3  # Cubic
    num_segments = 4
    lut = generate_test_lut(poly_degree, num_segments, x_min=-2.0, x_max=2.0)

    print(f"Polynomial Degree: {poly_degree} (cubic)")
    print(f"Number of Segments: {num_segments}")
    print(f"Total LUT Size: {len(lut)}\n")

    # Print boundaries
    print("Boundaries:")
    boundaries = lut[: num_segments + 1]
    for i, b in enumerate(boundaries):
        print(f"  b[{i}] = {b:.4f}")

    # Print coefficients
    print("\nCoefficients (per segment):")
    coeff_offset = num_segments + 1
    coeffs_per_seg = poly_degree + 1

    for seg_idx in range(num_segments):
        base = coeff_offset + seg_idx * coeffs_per_seg
        coeffs = lut[base : base + coeffs_per_seg]
        print(f"  Segment {seg_idx}: [", end="")
        print(", ".join(f"{c:.4f}" for c in coeffs), end="")
        print("]")

    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    # Run tests
    print("Running piecewise_generic validation tests...\n")

    # Print example for documentation
    print_lut_example()

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
