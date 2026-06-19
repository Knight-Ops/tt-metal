#!/usr/bin/env python3
"""
Mathematical Verification of FPU Piecewise Polynomial Implementation

This script simulates both SFPU (Horner) and FPU (matmul) methods and verifies
they produce identical results for piecewise polynomial evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# ============================================================================
# Configuration
# ============================================================================

# Test configuration
POLY_DEGREE = 4
NUM_SEGMENTS = 8
NUM_LANES = 32  # Tile width

# Generate test LUT for a simple piecewise polynomial
# Example: Approximating sigmoid with piecewise polynomials
def generate_test_lut(degree: int, num_segments: int) -> Tuple[np.ndarray, List[float]]:
    """
    Generate a test LUT for piecewise polynomial approximation.

    Returns:
        boundaries: Array of segment boundaries [b0, b1, ..., bN]
        coefficients: Flat list of coefficients [c0_seg0, c1_seg0, ..., c0_seg1, ...]
    """
    # Define boundaries (evenly spaced for simplicity)
    x_min, x_max = -4.0, 4.0
    boundaries = np.linspace(x_min, x_max, num_segments + 1)

    # Generate polynomial coefficients for each segment
    # For testing, use simple polynomials based on segment index
    coefficients = []
    for seg in range(num_segments):
        # Center of segment
        x_center = (boundaries[seg] + boundaries[seg + 1]) / 2

        # Create polynomial coefficients (c0, c1, c2, ..., cD)
        # Simple pattern: make polynomial roughly match segment position
        seg_coeffs = []
        for d in range(degree + 1):
            # Create interesting patterns
            if d == 0:
                coeff = x_center / 4.0  # Constant term based on position
            elif d == 1:
                coeff = 0.5  # Linear term
            elif d == 2:
                coeff = 0.1 * (seg % 3 - 1)  # Quadratic varies by segment
            else:
                coeff = 0.01 * ((-1) ** d) * d  # Higher order terms
            seg_coeffs.append(coeff)

        coefficients.extend(seg_coeffs)

    return boundaries, coefficients


# ============================================================================
# SFPU Method: Horner's Scheme (Ground Truth)
# ============================================================================

def eval_polynomial_horner(coeffs: List[float], x: float) -> float:
    """
    Evaluate polynomial using Horner's method.

    For polynomial: c0 + c1*x + c2*x^2 + ... + cN*x^N
    Horner's method: ((...((cN*x + c(N-1))*x + c(N-2))*x + ...) + c1)*x + c0
    """
    degree = len(coeffs) - 1
    result = coeffs[degree]
    for i in range(degree - 1, -1, -1):
        result = result * x + coeffs[i]
    return result


def sfpu_method(x_values: np.ndarray, boundaries: np.ndarray, coefficients: List[float],
                degree: int, num_segments: int) -> np.ndarray:
    """
    Simulate SFPU piecewise polynomial evaluation.

    This is the ground truth method.
    """
    results = np.zeros_like(x_values)
    coeffs_per_segment = degree + 1

    for i, x in enumerate(x_values):
        # Clamp x to valid range
        x_clamped = np.clip(x, boundaries[0], boundaries[-1])

        # Find segment (binary search in practice, linear for clarity)
        seg_id = 0
        for seg in range(1, num_segments):
            if x_clamped >= boundaries[seg]:
                seg_id = seg

        # Get coefficients for this segment
        coeff_start = seg_id * coeffs_per_segment
        seg_coeffs = coefficients[coeff_start:coeff_start + coeffs_per_segment]

        # Evaluate polynomial using Horner's method
        results[i] = eval_polynomial_horner(seg_coeffs, x)

    return results


# ============================================================================
# FPU Method: Matrix Multiplication
# ============================================================================

def build_power_matrix(x_values: np.ndarray, degree: int) -> np.ndarray:
    """
    Build power matrix P[K x N] where K = degree + 1, N = num_lanes.

    P[k, j] = x[j]^k for k = 0..degree

    Returns 32x32 matrix (padded with zeros)
    """
    K = degree + 1
    N = len(x_values)

    # Create 32x32 matrix (padded)
    P = np.zeros((32, 32), dtype=np.float32)

    # Fill the K x N portion
    for k in range(K):
        for j in range(N):
            P[k, j] = x_values[j] ** k

    return P


def build_coeff_matrix(boundaries: np.ndarray, coefficients: List[float],
                       degree: int, num_segments: int) -> np.ndarray:
    """
    Build coefficient matrix C[S x K] where S = num_segments, K = degree + 1.

    C[s, k] = coefficient k for segment s

    Returns 32x32 matrix (padded with zeros)
    """
    K = degree + 1
    S = num_segments

    # Create 32x32 matrix (padded)
    C = np.zeros((32, 32), dtype=np.float32)

    # Fill the S x K portion
    for s in range(S):
        for k in range(K):
            C[s, k] = coefficients[s * K + k]

    return C


def compute_segment_ids(x_values: np.ndarray, boundaries: np.ndarray,
                        num_segments: int) -> np.ndarray:
    """
    Compute segment IDs for each x value.
    """
    seg_ids = np.zeros(len(x_values), dtype=np.int32)

    for j, x in enumerate(x_values):
        # Clamp x
        x_clamped = np.clip(x, boundaries[0], boundaries[-1])

        # Find segment
        seg_id = 0
        for seg in range(1, num_segments):
            if x_clamped >= boundaries[seg]:
                seg_id = seg

        seg_ids[j] = seg_id

    return seg_ids


def fpu_method(x_values: np.ndarray, boundaries: np.ndarray, coefficients: List[float],
               degree: int, num_segments: int) -> np.ndarray:
    """
    Simulate FPU piecewise polynomial evaluation using matrix multiplication.

    Algorithm:
    1. Build power matrix P[K x 32]: P[k, j] = x[j]^k
    2. Build coefficient matrix C[S x K]: C[s, k] = coeff k for segment s
    3. Compute Y = C @ P (matrix multiply)
    4. Select Y[seg_id[j], j] for each lane j
    """
    # Step 1: Build power matrix
    P = build_power_matrix(x_values, degree)

    # Step 2: Build coefficient matrix
    C = build_coeff_matrix(boundaries, coefficients, degree, num_segments)

    # Step 3: Compute Y = C @ P (matmul)
    Y = np.matmul(C, P)  # Result is S x 32 (logical), 32 x 32 (physical)

    # Step 4: Compute segment IDs
    seg_ids = compute_segment_ids(x_values, boundaries, num_segments)

    # Step 5: Select appropriate result for each lane
    results = np.zeros(len(x_values), dtype=np.float32)
    for j in range(len(x_values)):
        results[j] = Y[seg_ids[j], j]

    return results


# ============================================================================
# Verification and Testing
# ============================================================================

def verify_implementation(degree: int, num_segments: int, num_lanes: int):
    """
    Verify that FPU method produces identical results to SFPU method.
    """
    print("=" * 80)
    print(f"Verification: Degree={degree}, Segments={num_segments}, Lanes={num_lanes}")
    print("=" * 80)

    # Generate test LUT
    boundaries, coefficients = generate_test_lut(degree, num_segments)

    print(f"\nLUT Configuration:")
    print(f"  Boundaries: {boundaries}")
    print(f"  Num coefficients: {len(coefficients)}")
    print(f"  Expected: {num_segments * (degree + 1)}")

    # Generate random test inputs spanning the boundary range
    x_min, x_max = boundaries[0], boundaries[-1]
    np.random.seed(42)

    # Test set 1: Random values across range
    x_values = np.random.uniform(x_min - 1, x_max + 1, num_lanes).astype(np.float32)

    # Test set 2: Values exactly at boundaries (critical test)
    x_boundaries = boundaries[:num_lanes].astype(np.float32)

    # Test set 3: Values slightly above/below boundaries
    x_edge_cases = []
    for b in boundaries[:num_segments]:
        x_edge_cases.append(b - 0.001)
        x_edge_cases.append(b + 0.001)
    x_edge_cases = np.array(x_edge_cases[:num_lanes], dtype=np.float32)

    test_sets = [
        ("Random values", x_values),
        ("Boundary values", x_boundaries),
        ("Edge cases", x_edge_cases)
    ]

    all_passed = True

    for test_name, x_test in test_sets:
        print(f"\n{'-' * 80}")
        print(f"Test Set: {test_name}")
        print(f"{'-' * 80}")

        # Pad to num_lanes if needed
        if len(x_test) < num_lanes:
            x_test = np.pad(x_test, (0, num_lanes - len(x_test)),
                          constant_values=x_min)

        print(f"Input x values (first 10): {x_test[:10]}")

        # Compute using SFPU method (ground truth)
        sfpu_results = sfpu_method(x_test, boundaries, coefficients, degree, num_segments)

        # Compute using FPU method
        fpu_results = fpu_method(x_test, boundaries, coefficients, degree, num_segments)

        # Compare results
        abs_diff = np.abs(fpu_results - sfpu_results)
        rel_diff = np.abs((fpu_results - sfpu_results) / (np.abs(sfpu_results) + 1e-10))

        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_abs_diff = np.mean(abs_diff)

        print(f"\nResults comparison:")
        print(f"  SFPU results (first 10): {sfpu_results[:10]}")
        print(f"  FPU results (first 10):  {fpu_results[:10]}")
        print(f"\nError metrics:")
        print(f"  Max absolute error: {max_abs_diff:.2e}")
        print(f"  Max relative error: {max_rel_diff:.2e}")
        print(f"  Mean absolute error: {mean_abs_diff:.2e}")

        # Check if results match (within floating point tolerance)
        tolerance = 1e-6
        if max_abs_diff < tolerance:
            print(f"  ✓ PASS: Results match within tolerance {tolerance}")
        else:
            print(f"  ✗ FAIL: Results differ by more than {tolerance}")
            all_passed = False

            # Show first mismatch
            mismatch_idx = np.argmax(abs_diff)
            print(f"\n  First mismatch at index {mismatch_idx}:")
            print(f"    x = {x_test[mismatch_idx]}")
            print(f"    SFPU result = {sfpu_results[mismatch_idx]}")
            print(f"    FPU result = {fpu_results[mismatch_idx]}")
            print(f"    Difference = {abs_diff[mismatch_idx]}")

    return all_passed


def visualize_comparison(degree: int, num_segments: int):
    """
    Create visualization comparing SFPU and FPU methods.
    """
    print("\n" + "=" * 80)
    print("Generating Visualization")
    print("=" * 80)

    # Generate test LUT
    boundaries, coefficients = generate_test_lut(degree, num_segments)

    # Generate dense x values for plotting
    x_min, x_max = boundaries[0], boundaries[-1]
    x_dense = np.linspace(x_min - 0.5, x_max + 0.5, 500).astype(np.float32)

    # Evaluate using both methods
    sfpu_dense = np.zeros(len(x_dense))
    fpu_dense = np.zeros(len(x_dense))

    # Process in batches of NUM_LANES
    for i in range(0, len(x_dense), NUM_LANES):
        end_idx = min(i + NUM_LANES, len(x_dense))
        batch_size = end_idx - i

        x_batch = x_dense[i:end_idx]
        if batch_size < NUM_LANES:
            x_batch = np.pad(x_batch, (0, NUM_LANES - batch_size),
                           constant_values=x_min)

        sfpu_batch = sfpu_method(x_batch, boundaries, coefficients, degree, num_segments)
        fpu_batch = fpu_method(x_batch, boundaries, coefficients, degree, num_segments)

        sfpu_dense[i:end_idx] = sfpu_batch[:batch_size]
        fpu_dense[i:end_idx] = fpu_batch[:batch_size]

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Both methods
    ax = axes[0]
    ax.plot(x_dense, sfpu_dense, 'b-', linewidth=2, label='SFPU (Horner)', alpha=0.7)
    ax.plot(x_dense, fpu_dense, 'r--', linewidth=2, label='FPU (Matmul)', alpha=0.7)

    # Mark boundaries
    for b in boundaries:
        ax.axvline(b, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y = f(x)', fontsize=12)
    ax.set_title(f'Piecewise Polynomial Evaluation: Degree={degree}, Segments={num_segments}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Absolute difference
    ax = axes[1]
    diff = np.abs(fpu_dense - sfpu_dense)
    ax.plot(x_dense, diff, 'g-', linewidth=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('|FPU - SFPU|', fontsize=12)
    ax.set_title('Absolute Difference between FPU and SFPU Methods',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Add statistics
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    ax.text(0.02, 0.98, f'Max diff: {max_diff:.2e}\nMean diff: {mean_diff:.2e}',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10)

    plt.tight_layout()

    # Save plot
    output_path = 'fpu_verification_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    return max_diff


def test_matrix_construction(degree: int, num_segments: int):
    """
    Detailed verification of matrix construction logic.
    """
    print("\n" + "=" * 80)
    print("Matrix Construction Verification")
    print("=" * 80)

    # Generate test LUT
    boundaries, coefficients = generate_test_lut(degree, num_segments)

    # Test with simple x values
    x_values = np.array([0.0, 1.0, 2.0, -1.0, 0.5, 1.5], dtype=np.float32)
    x_values = np.pad(x_values, (0, NUM_LANES - len(x_values)), constant_values=0.0)

    print(f"\nTest x values (first 6): {x_values[:6]}")

    # Build power matrix
    P = build_power_matrix(x_values, degree)
    K = degree + 1

    print(f"\nPower Matrix P[{K} x {NUM_LANES}] (showing first 6 columns):")
    print(f"Shape: {P.shape}")
    for k in range(K):
        print(f"  P[{k},:6] (x^{k}): {P[k, :6]}")

    # Verify power matrix manually for first few values
    print(f"\nManual verification of power matrix:")
    for j in range(min(6, NUM_LANES)):
        x = x_values[j]
        print(f"  x[{j}] = {x:.2f}:")
        for k in range(K):
            expected = x ** k
            actual = P[k, j]
            match = "✓" if np.isclose(expected, actual) else "✗"
            print(f"    P[{k},{j}] = {actual:.4f}, expected {expected:.4f} {match}")

    # Build coefficient matrix
    C = build_coeff_matrix(boundaries, coefficients, degree, num_segments)

    print(f"\nCoefficient Matrix C[{num_segments} x {K}]:")
    print(f"Shape: {C.shape}")
    for s in range(num_segments):
        coeffs_str = ", ".join(f"{C[s, k]:.3f}" for k in range(K))
        print(f"  Segment {s}: [{coeffs_str}]")

    # Verify coefficient matrix
    print(f"\nManual verification of coefficient matrix:")
    for s in range(num_segments):
        print(f"  Segment {s}:")
        for k in range(K):
            expected = coefficients[s * K + k]
            actual = C[s, k]
            match = "✓" if np.isclose(expected, actual) else "✗"
            print(f"    C[{s},{k}] = {actual:.4f}, expected {expected:.4f} {match}")

    # Perform matmul and show result
    Y = np.matmul(C, P)
    print(f"\nResult Matrix Y = C @ P:")
    print(f"Shape: {Y.shape}")
    print(f"First 6 results for each segment:")
    for s in range(num_segments):
        print(f"  Y[{s},:6]: {Y[s, :6]}")

    # Manual verification: compute Y[0, 0] step by step
    print(f"\nManual computation of Y[0, 0] (segment 0, lane 0):")
    print(f"  Y[0, 0] = sum(C[0, k] * P[k, 0] for k in range({K}))")
    manual_sum = 0
    for k in range(K):
        term = C[0, k] * P[k, 0]
        manual_sum += term
        print(f"    k={k}: C[0,{k}] * P[{k},0] = {C[0, k]:.4f} * {P[k, 0]:.4f} = {term:.4f}")
    print(f"  Manual sum: {manual_sum:.4f}")
    print(f"  Matrix Y[0, 0]: {Y[0, 0]:.4f}")
    print(f"  Match: {'✓' if np.isclose(manual_sum, Y[0, 0]) else '✗'}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FPU Piecewise Polynomial Implementation - Mathematical Verification")
    print("=" * 80)

    # Test 1: Matrix construction
    test_matrix_construction(POLY_DEGREE, NUM_SEGMENTS)

    # Test 2: Functional verification
    all_passed = verify_implementation(POLY_DEGREE, NUM_SEGMENTS, NUM_LANES)

    # Test 3: Visualization
    max_diff = visualize_comparison(POLY_DEGREE, NUM_SEGMENTS)

    # Test 4: Different configurations
    print("\n" + "=" * 80)
    print("Testing Different Configurations")
    print("=" * 80)

    test_configs = [
        (1, 4, 32),   # Linear, few segments
        (2, 8, 32),   # Quadratic, moderate segments
        (4, 16, 32),  # Quartic, many segments
        (8, 32, 32),  # High degree, max segments
    ]

    config_results = []
    for degree, segments, lanes in test_configs:
        print(f"\n{'=' * 80}")
        passed = verify_implementation(degree, segments, lanes)
        config_results.append((degree, segments, passed))

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\nConfiguration Test Results:")
    for degree, segments, passed in config_results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Degree={degree:2d}, Segments={segments:2d}: {status}")

    all_configs_passed = all(result[2] for result in config_results)

    print("\n" + "=" * 80)
    if all_configs_passed:
        print("✓ ALL TESTS PASSED - FPU implementation is mathematically correct!")
    else:
        print("✗ SOME TESTS FAILED - Review implementation")
    print("=" * 80)

    print(f"\nVisualization saved to: fpu_verification_plot.png")
    print("\nVerification complete!")
