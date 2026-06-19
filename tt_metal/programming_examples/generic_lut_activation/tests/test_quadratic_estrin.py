#!/usr/bin/env python3
"""
Test for piecewise_quadratic_estrin.cpp kernel

Tests the Estrin+SFPLUT implementation against:
1. Reference numpy implementation
2. Original Horner implementation (piecewise_quadratic.cpp)

Uses a simple quadratic function: f(x) = x^2 for validation
"""

import numpy as np

try:
    import pytest
    import torch
except ImportError:
    pass  # Not needed for standalone execution


def reference_function(x):
    """Simple quadratic: f(x) = x^2"""
    return x * x


def fit_quadratic_segment(x_seg, y_seg):
    """Fit degree-2 polynomial to segment data"""
    # Returns [c0, c1, c2] for y = c0 + c1*x + c2*x^2
    coeffs_high_to_low = np.polyfit(x_seg, y_seg, 2)
    return coeffs_high_to_low[::-1]  # [c0, c1, c2]


def generate_quadratic_lut_4seg():
    """Generate LUT for x^2 with 4 segments over [-2, 2]"""
    boundaries = [-2.0, -1.0, 0.0, 1.0, 2.0]

    lut = boundaries.copy()

    # Fit each segment
    for i in range(4):
        x_seg = np.linspace(boundaries[i], boundaries[i+1], 100)
        y_seg = reference_function(x_seg)
        coeffs = fit_quadratic_segment(x_seg, y_seg)
        lut.extend(coeffs)

    return np.array(lut, dtype=np.float32)


def eval_piecewise_quadratic_horner(x, lut):
    """Evaluate using Horner's method (matches piecewise_quadratic.cpp)"""
    num_boundaries = 5  # 4 segments
    num_segments = 4
    boundaries = lut[:num_boundaries]

    # Clamp x to range
    x_clamped = np.clip(x, boundaries[0], boundaries[-1])

    # Find segment (last boundary where x >= boundary)
    seg = 0
    for i in range(1, num_segments):
        if x_clamped >= boundaries[i]:
            seg = i

    # Get coefficients [c0, c1, c2]
    coeff_start = num_boundaries + seg * 3
    c0, c1, c2 = lut[coeff_start:coeff_start+3]

    # Horner: (c2*x + c1)*x + c0
    return (c2 * x + c1) * x + c0


def eval_piecewise_quadratic_estrin(x, lut, _unused=None):
    """Evaluate using Estrin's method with boundary subtraction"""
    num_boundaries = 5
    num_segments = 4
    boundaries = lut[:num_boundaries]

    # Clamp x to range
    x_clamped = np.clip(x, boundaries[0], boundaries[-1])

    # Find segment and boundary
    seg = 0
    boundary = boundaries[0]
    for i in range(1, num_segments):
        if x_clamped >= boundaries[i]:
            seg = i
            boundary = boundaries[i]

    # Get coefficients [c0, c1, c2]
    coeff_start = num_boundaries + seg * 3
    c0, c1, c2 = lut[coeff_start:coeff_start+3]

    # Boundary subtraction transform
    x_norm = x - boundary
    boundary_sq = boundary * boundary
    c0_adj = c0 + c1 * boundary + c2 * boundary_sq
    c1_adj = c1 + 2.0 * c2 * boundary

    # Estrin: (c1_adj*x_norm + c0_adj) + c2*x_norm^2
    linear_part = c1_adj * x_norm + c0_adj
    x_norm_sq = x_norm * x_norm
    return linear_part + c2 * x_norm_sq


def test_quadratic_estrin_accuracy():
    """Test Estrin accuracy matches Horner"""
    print("\n" + "="*80)
    print("Testing Quadratic Estrin vs Horner - f(x) = x^2")
    print("="*80)

    # Generate LUT
    lut = generate_quadratic_lut_4seg()
    print(f"\nLUT size: {len(lut)} floats (5 boundaries + 4 segments * 3 coeffs)")
    print(f"Boundaries: {lut[:5]}")

    # Test over range
    x_test = np.linspace(-2.0, 2.0, 1000)
    y_ref = reference_function(x_test)

    y_horner = np.array([eval_piecewise_quadratic_horner(x, lut) for x in x_test])
    y_estrin = np.array([eval_piecewise_quadratic_estrin(x, lut, lut[0]) for x in x_test])

    # Compute errors
    error_horner = np.abs(y_horner - y_ref)
    error_estrin = np.abs(y_estrin - y_ref)
    error_diff = np.abs(y_estrin - y_horner)

    print(f"\nAccuracy Results:")
    print(f"  Horner max error:  {error_horner.max():.6e}")
    print(f"  Estrin max error:  {error_estrin.max():.6e}")
    print(f"  Estrin vs Horner:  {error_diff.max():.6e}")

    # Both should match reference well
    assert error_horner.max() < 1e-5, f"Horner error too large: {error_horner.max()}"
    assert error_estrin.max() < 1e-5, f"Estrin error too large: {error_estrin.max()}"

    # Estrin should match Horner (same polynomial, different eval method)
    assert error_diff.max() < 1e-6, f"Estrin vs Horner mismatch: {error_diff.max()}"

    print(f"\n✓ Test passed! Estrin matches Horner within numerical precision")


def test_boundary_subtraction():
    """Test boundary subtraction math is correct"""
    print("\n" + "="*80)
    print("Testing Boundary Subtraction Transform")
    print("="*80)

    # Simple polynomial: P(x) = 3 + 2x + 1x^2
    c0, c1, c2 = 3.0, 2.0, 1.0
    boundary = 1.0

    # Original evaluation at x=2.5
    x = 2.5
    y_direct = c0 + c1*x + c2*x*x

    # Boundary subtraction
    x_norm = x - boundary
    boundary_sq = boundary * boundary
    c0_adj = c0 + c1*boundary + c2*boundary_sq
    c1_adj = c1 + 2.0*c2*boundary

    y_transformed = c0_adj + c1_adj*x_norm + c2*x_norm*x_norm

    print(f"\nOriginal coefficients: c0={c0}, c1={c1}, c2={c2}")
    print(f"Boundary: {boundary}")
    print(f"Test point: x={x}")
    print(f"\nTransformed coefficients: c0_adj={c0_adj:.6f}, c1_adj={c1_adj:.6f}, c2={c2}")
    print(f"x_norm = {x_norm}")
    print(f"\nDirect evaluation:      {y_direct:.6f}")
    print(f"Transformed evaluation: {y_transformed:.6f}")
    print(f"Difference:             {abs(y_direct - y_transformed):.6e}")

    assert abs(y_direct - y_transformed) < 1e-10, "Boundary subtraction math incorrect"
    print(f"\n✓ Boundary subtraction is mathematically correct")


def test_edge_cases():
    """Test edge cases for quadratic Estrin"""
    print("\n" + "="*80)
    print("Testing Edge Cases")
    print("="*80)

    lut = generate_quadratic_lut_4seg()

    # Test at boundaries
    test_points = [-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 1.5]

    print(f"\n{'x':<10} {'Reference':<12} {'Estrin':<12} {'Error':<12}")
    print("-" * 46)

    max_error = 0.0
    for x in test_points:
        y_ref = reference_function(x)
        y_estrin = eval_piecewise_quadratic_estrin(x, lut, lut[0])
        error = abs(y_ref - y_estrin)
        max_error = max(max_error, error)
        print(f"{x:<10.2f} {y_ref:<12.6f} {y_estrin:<12.6f} {error:<12.6e}")

    assert max_error < 1e-5, f"Edge case error too large: {max_error}"
    print(f"\n✓ All edge cases passed (max error: {max_error:.6e})")


if __name__ == "__main__":
    print("╔" + "═"*78 + "╗")
    print("║" + " QUADRATIC ESTRIN TEST ".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    test_boundary_subtraction()
    test_quadratic_estrin_accuracy()
    test_edge_cases()

    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80)
    print("\nNext step: Compile and run on hardware")
    print("  1. Build: cd /Users/nkapre/workspace/tt-metal && ./build_metal.sh")
    print("  2. Test kernel integration (TBD)")
