#!/usr/bin/env python3
"""
Test all activation functions with Sollya to identify which ones are unsupported.
This script does NOT use any piecewise workarounds - it tests direct Sollya support.
"""

import sys
from pathlib import Path

# Add tt-polynomial-fitter to path for ground_truth module
_poly_fit_dir = os.environ.get('TT_POLY_FIT_DIR', '/localdev/nkapre/tt-polynomial-fitter')
sys.path.insert(0, _poly_fit_dir)
from ground_truth import get_all_activations
ALL_ACTIVATIONS = get_all_activations()

# Add luts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "luts"))

try:
    from sollya_bf16_remez import SollyaBF16Fitter
    SOLLYA_AVAILABLE = True
except ImportError:
    SOLLYA_AVAILABLE = False
    print("ERROR: Sollya not available!")
    sys.exit(1)

def test_activation(fitter, activation, test_range=(-5, 5)):
    """
    Test if an activation can be approximated by Sollya.

    Returns:
        (success, error_message) tuple
    """
    try:
        # Try to fit a single segment (degree 3, simple test)
        coeffs, max_err = fitter.fit_segment(activation, degree=3,
                                             seg_start=test_range[0],
                                             seg_end=test_range[1])
        return (True, None)
    except Exception as e:
        return (False, str(e))

def main():
    print("=" * 80)
    print("Testing All Activations with Sollya (NO piecewise workarounds)")
    print("=" * 80)
    print()

    if not SOLLYA_AVAILABLE:
        print("ERROR: Sollya not available!")
        return 1

    # Initialize Sollya fitter
    print("Initializing Sollya BF16 fitter...")
    fitter = SollyaBF16Fitter(bf16_mantissa_bits=7)
    print()

    # Test all activations
    supported = []
    unsupported = []

    print(f"Testing {len(ALL_ACTIVATIONS)} activations...")
    print("-" * 80)

    for activation in ALL_ACTIVATIONS:
        sys.stdout.write(f"Testing {activation:20s} ... ")
        sys.stdout.flush()

        success, error = test_activation(fitter, activation)

        if success:
            supported.append(activation)
            print("✓ SUPPORTED")
        else:
            unsupported.append((activation, error))
            print(f"✗ UNSUPPORTED: {error}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Supported:   {len(supported)}/{len(ALL_ACTIVATIONS)}")
    print(f"Unsupported: {len(unsupported)}/{len(ALL_ACTIVATIONS)}")
    print()

    if supported:
        print("Supported activations:")
        for act in supported:
            print(f"  ✓ {act}")
        print()

    if unsupported:
        print("Unsupported activations (require custom implementation):")
        for act, error in unsupported:
            print(f"  ✗ {act}")
            print(f"      Error: {error}")
        print()

    # Generate list for easy copying
    if unsupported:
        print("Python list of unsupported activations:")
        unsupported_names = [act for act, _ in unsupported]
        print(f"UNSUPPORTED_ACTIVATIONS = {unsupported_names}")
        print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
