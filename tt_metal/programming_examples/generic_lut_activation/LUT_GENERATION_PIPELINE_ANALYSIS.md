# LUT Generation Pipeline Analysis

**Date**: 2026-01-19
**Status**: ✅ All critical points now loaded from CSV (single source of truth)

## Overview

This document analyzes the LUT generation pipeline for generic activation functions, including HPP generation, embedded generation, and critical point handling.

## Pipeline Architecture

```
activations_config.csv (SINGLE SOURCE OF TRUTH)
    │
    ├──> activations_config.py
    │       │
    │       ├──> load_critical_points_from_csv()
    │       │       │
    │       │       ├──> generate_adaptive_segments.py
    │       │       ├──> generate_piecewise_quadratic_remez_luts.py
    │       │       ├──> generate_piecewise_cubic_remez_luts.py
    │       │       ├──> generate_piecewise_hexic_remez_luts.py
    │       │       └──> generate_piecewise_octic_remez_luts.py
    │       │
    │       └──> ALL_ACTIVATIONS, NATIVE_SFPU_ACTIVATIONS, PIECEWISE_DEPTHS
    │
    └──> activation_config.hpp (C++ loader, reads CSV at runtime)
            └──> generic_lut_activation.cpp (test harness)

LUT Generation Flow:
    1. generate_luts.sh (orchestrator)
    2. generate_adaptive_segments.py → adaptive_segments/*.csv
    3. Remez scripts → luts/*.lut (uniform + adaptive)
```

## Critical Points Configuration

### Purpose
Critical points ensure exact function values at specific x-coordinates, preventing numerical drift in polynomial approximations.

### Current Critical Points (activations_config.csv)

| Activation | Critical X | Critical Value | Reason |
|-----------|-----------|----------------|---------|
| **cosh**  | 0.0 | 1.0 | cosh(0) = 1 exactly |
| **sin**   | 0.0 | 0.0 | sin(0) = 0 exactly |
| **sinh**  | 0.0 | 0.0 | sinh(0) = 0 exactly |
| **softsign** | 0.0 | 0.0 | softsign(0) = 0 exactly |
| **swish** | 0.0 | 0.0 | swish(0) = x·sigmoid(x) = 0 |
| **tanh**  | 0.0 | 0.0 | tanh(0) = 0 exactly |
| **tanhshrink** | 0.0 | 0.0 | tanhshrink(0) = x - tanh(x) = 0 |

### Implementation

**CSV Format**:
```csv
activation,has_native_sfpu,range_min,range_max,critical_point_x,critical_point_value,notes
cosh,false,-10.0,10.0,0.0,1.0,Exponential growth; cosh(0) = 1
swish,true,-10.0,10.0,0.0,0.0,x * sigmoid(x); swish(0) = 0
```

**Python Loader** (`activations_config.py:10-32`):
```python
def load_critical_points_from_csv(csv_path='activations_config.csv'):
    """Load critical points from activations_config.csv."""
    critical_points_map = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            activation = row['activation'].strip()
            if row['critical_point_x'].strip() and row['critical_point_value'].strip():
                critical_x = float(row['critical_point_x'])
                critical_value = float(row['critical_point_value'])
                critical_points_map[activation] = {critical_x: critical_value}
    return critical_points_map
```

**Usage in Generation Scripts**:
```python
from activations_config import load_critical_points_from_csv

# Load once at module level
CRITICAL_POINTS = load_critical_points_from_csv()

# Use in fitting
def fit_piecewise_cubic_remez(..., func_name=None):
    critical_points = None
    if func_name and func_name in CRITICAL_POINTS:
        critical_points = CRITICAL_POINTS[func_name]
        print(f"Using constrained fitting with critical points: {critical_points}")

    # Pass to constrained Remez fitting
    return fit_piecewise_cubic_constrained(..., critical_points=critical_points)
```

## LUT Generation Scripts

### 1. Adaptive Segment Generation
**Script**: `generate_adaptive_segments.py`
**Purpose**: Generate non-uniform breakpoints that concentrate segments in high-curvature regions

**Process**:
1. Load critical points from CSV
2. Identify discontinuities (hardcoded for piecewise functions like ReLU, hardsigmoid)
3. Add CSV-defined critical points
4. Generate adaptive breakpoints using curvature analysis
5. Output to `adaptive_segments/depth_{4,8,16,32}_segments.csv`

**Critical Point Integration**:
```python
# Load from CSV at module level
CRITICAL_POINTS_FROM_CSV = load_critical_points_from_csv()

def identify_critical_points(func_name, func, input_range):
    """Identify critical points from CSV + hardcoded discontinuities."""
    critical = [x_min, x_max]  # Domain boundaries

    # Add CSV-defined critical points
    if func_name in CRITICAL_POINTS_FROM_CSV:
        critical_x = list(CRITICAL_POINTS_FROM_CSV[func_name].keys())[0]
        if x_min <= critical_x <= x_max:
            critical.append(critical_x)
            print(f"  → Added CSV critical point for {func_name}: x={critical_x}")

    # Add hardcoded discontinuities (ReLU, hardsigmoid, etc.)
    # ...

    return sorted(set(critical))
```

### 2. Remez LUT Generation (Uniform + Adaptive)
**Scripts**:
- `generate_piecewise_quadratic_remez_luts.py` (degree 2)
- `generate_piecewise_cubic_remez_luts.py` (degree 3)
- `generate_piecewise_hexic_remez_luts.py` (degree 6)
- `generate_piecewise_octic_remez_luts.py` (degree 8)

**Process**:
1. Load critical points from CSV
2. For each activation and depth:
   - Load adaptive breakpoints (if `--adaptive` flag)
   - Fit minimax polynomial using Remez algorithm
   - Apply critical point constraints via `fit_constrained_remez.py`
   - Write binary LUT file
3. Output to `luts/piecewise_{method}_remez_{activation}_{depth}[_adaptive].lut`

**Critical Point Enforcement** (`fit_constrained_remez.py`):
```python
def fit_piecewise_constrained_remez(..., critical_points=None):
    """
    Fit piecewise polynomial with constraints:
    1. C0 continuity at segment boundaries
    2. Exact values at critical points
    """
    # Step 1: Fit unconstrained Remez polynomial
    # Step 2: Adjust constant terms for boundary continuity
    # Step 3: Enforce exact values at critical points
    if critical_points is not None:
        for critical_x, expected_value in critical_points.items():
            segment_idx = find_segment(boundaries, critical_x)
            if segment_idx is not None:
                current_value = evaluate_polynomial(coeffs, segment_idx, critical_x)
                error = expected_value - current_value
                constant_coeffs[segment_idx] += error
                print(f"✓ Enforced constraint: f({critical_x}) = {expected_value}")
```

### 3. Orchestration Script
**Script**: `generate_luts.sh`
**Purpose**: Parallel LUT generation for all activations

**Modes**:
- `./generate_luts.sh` - Generate both uniform and adaptive LUTs
- `./generate_luts.sh --uniform-only` - Generate only uniform LUTs
- `./generate_luts.sh --adaptive-only` - Generate only adaptive LUTs

**Process**:
1. Generate adaptive breakpoints (unless `--uniform-only`)
2. Launch parallel processes (one per activation)
3. Each process generates all methods (constant, linear, quadratic, cubic, hexic, octic)
4. Log output to `luts/generation_{activation}.log`

## HPP Generation (activation_config.hpp)

**Purpose**: C++ header for runtime loading of activation configurations

**Key Features**:
1. **CSV Loader**: Reads `activations_config.csv` at runtime (NOT compile-time)
2. **Path Search**: Tries multiple locations (current dir, repo root, build dir)
3. **Activation Extraction**: Parses LUT filenames to extract activation name
4. **Range Retrieval**: Returns test range for each activation

**Usage in C++ Tests**:
```cpp
#include "activation_config.hpp"

// Load configurations from CSV
auto configs = ActivationConfigLoader::load_from_csv();

// Extract activation name from LUT filename
std::string activation = ActivationConfigLoader::extract_activation_name(lut_filename);

// Get test range
auto [min, max] = ActivationConfigLoader::get_test_range(lut_filename, configs);
```

**Note**: HPP does NOT currently read critical points from CSV (only range_min, range_max, has_native_sfpu). Critical points are only used during LUT generation, not during runtime testing.

## Embedded Generation

**Status**: Not yet implemented. Planned feature for embedding LUTs directly in C++ source.

**Proposed Flow**:
```
luts/*.lut → embedded_lut_generator.py → kernels/embedded_luts.hpp
```

## Verification and Testing

### 1. Critical Points Consistency Check
```bash
# Verify all scripts load identical critical points
python3 -c "
from generate_piecewise_quadratic_remez_luts import CRITICAL_POINTS as quad
from generate_piecewise_cubic_remez_luts import CRITICAL_POINTS as cubic
from generate_piecewise_hexic_remez_luts import CRITICAL_POINTS as hexic
from generate_piecewise_octic_remez_luts import CRITICAL_POINTS as octic

assert quad == cubic == hexic == octic
print('✓ All scripts have identical critical points')
"
```

### 2. Critical Point Enforcement Check
```bash
# Generate a single LUT and verify critical point
python3 generate_piecewise_cubic_remez_luts.py --activation swish
# Should print: "Using constrained fitting with critical points: {0.0: 0.0}"
```

### 3. Adaptive Segment Generation Check
```bash
# Verify adaptive breakpoints include critical points
python3 generate_adaptive_segments.py --depth 16
grep "swish" adaptive_segments/depth_16_segments.csv
# Should show 0.0 in the breakpoints list
```

## Issues Resolved

### ✅ Issue 1: Hardcoded Critical Points (RESOLVED)
**Problem**: Critical points were hardcoded in 4 generation scripts, creating duplication and inconsistency.

**Solution**: Replaced all hardcoded `CRITICAL_POINTS` dictionaries with calls to `load_critical_points_from_csv()`.

**Changed Files**:
- `generate_piecewise_quadratic_remez_luts.py`
- `generate_piecewise_cubic_remez_luts.py`
- `generate_piecewise_hexic_remez_luts.py`
- `generate_piecewise_octic_remez_luts.py`

### ✅ Issue 2: Missing swish Critical Point (RESOLVED)
**Problem**: CSV defined swish(0) = 0, but no generation script included it.

**Solution**: Loading from CSV automatically includes swish critical point.

**Verification**: All scripts now enforce swish(0) = 0 during LUT generation.

### ✅ Issue 3: Missing cosh in Quadratic Script (RESOLVED)
**Problem**: Quadratic script didn't include cosh(0) = 1.

**Solution**: Loading from CSV automatically includes all 7 critical points.

## Best Practices

### Adding New Critical Points
1. **Edit only activations_config.csv**:
   ```csv
   newactivation,false,-10.0,10.0,2.0,5.0,newactivation(2) = 5 exactly
   ```

2. **Verify critical point is correct**:
   ```bash
   python3 -c "
   from ground_truth import ACTIVATION_FUNCTIONS
   func = ACTIVATION_FUNCTIONS['newactivation']
   print(f'newactivation(2.0) = {func(2.0)}')  # Should be 5.0
   "
   ```

3. **Regenerate LUTs**:
   ```bash
   ./generate_luts.sh --activation newactivation
   ```

### Modifying Activation Ranges
1. **Edit activations_config.csv** (range_min, range_max columns)
2. **Regenerate adaptive segments** (uses updated ranges)
3. **Regenerate LUTs** (uses updated ranges)

### Testing Configuration Changes
```bash
# Quick test with single activation and depth
python3 generate_adaptive_segments.py --depth 4
python3 generate_piecewise_cubic_remez_luts.py --activation sigmoid

# Full regeneration
./generate_luts.sh
```

## Summary

**Single Source of Truth**: `activations_config.csv`
- ✅ All critical points defined in CSV
- ✅ All generation scripts load from CSV
- ✅ No hardcoded critical points in Python scripts
- ✅ Consistent critical points across all methods (quadratic, cubic, hexic, octic)
- ✅ HPP reads activation ranges from CSV (not critical points)

**Critical Points Coverage**: 7 activations
- cosh, sin, sinh, softsign, swish, tanh, tanhshrink

**LUT Generation**: Fully automated
- Adaptive segmentation respects critical points
- Remez fitting enforces critical point constraints
- Parallel generation via `generate_luts.sh`

**Status**: ✅ Pipeline validated and production-ready
