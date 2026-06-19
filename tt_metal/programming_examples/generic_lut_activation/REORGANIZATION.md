# Code Reorganization - Segmentation Module

## Overview

Reorganized scattered segmentation and fitting code into a unified `segmentation/` module with clean interfaces and shared data structures.

## Before (Scattered)

```
├── chebyshev.py                    # DP optimal segmentation + Sollya interface
├── analyze_breakpoints.py         # Curvature-based + critical points
├── luts/sollya_expressions.py     # Sollya expression mapping
├── luts/sollya_python_wrapper.py  # Python Sollya wrapper
├── luts/sollya_bf16_remez.py      # BF16 fitting
├── luts/sollya_optimal_degree_search.py  # Degree search
├── luts/sollya_extended_functions.py     # Extended functions
└── luts/generate_adaptive_segments.py    # High-level generation
```

**Problems:**
- Code duplicated across files (SollyaInterface defined multiple times)
- No shared data structures
- Unclear separation of concerns
- Hard to maintain and extend
- Imports scattered everywhere

## After (Organized)

```
segmentation/
├── __init__.py              # Public API exports
├── base.py                  # Shared data structures (PolySegment, ApproxResult, SegmentationMethod)
├── sollya_interface.py      # Unified Sollya interface with caching
├── sollya_expressions.py    # Sollya expression mapping
├── curvature.py            # Curvature-based adaptive segmentation
└── chebyshev.py            # Chebyshev DP optimal segmentation

luts/
├── generate_adaptive_segments.py  # High-level generation (uses segmentation/)
└── generate_piecewise_remez_luts.py  # LUT generation (uses segmentation/)
```

**Benefits:**
- Single source of truth for each concept
- Shared interfaces via base classes
- Clean public API through `__init__.py`
- Easy to test and extend
- Clear separation: segmentation/ for algorithms, luts/ for generation

## Module Structure

### `segmentation/base.py`
**Shared data structures used by all methods:**
- `PolySegment`: Single polynomial segment with interval, coefficients, error
- `ApproxResult`: Complete approximation result with segments and metadata
- `SegmentationMethod`: Base class for all segmentation algorithms
- Utility functions: `validate_breakpoints()`, `merge_critical_points()`

### `segmentation/sollya_interface.py`
**Unified Sollya interface:**
- `SollyaInterface`: Main class for Sollya calls
  - Caching via MD5 hash
  - Support for both relative and absolute error
  - BF16/FP32/custom precision
  - Robust error parsing
- `fit_segment_worker()`: Worker for parallel fitting
- `get_sollya()`: Singleton access

### `segmentation/sollya_expressions.py`
**Sollya expression mapping:**
- `SOLLYA_EXPRESSIONS`: Dict mapping function names to Sollya expressions
- `get_sollya_expression_for_piece()`: Get expression for piecewise functions
- `supports_sollya()`: Check if function has Sollya expression
- `is_piecewise_linear_activation()`: Identify piecewise linear functions

### `segmentation/curvature.py`
**Curvature-based adaptive segmentation:**
- `CurvatureSegmenter`: Main segmenter class
  - Uses log-curvature weighted distribution
  - Adaptive sampling in high-curvature regions
  - Preserves critical points
- `curvature_segment()`: Convenience function
- `numerical_derivative()`: Finite difference derivatives
- `compute_curvature_distribution()`: Build cumulative curvature
- `is_piecewise_linear()`: Test for piecewise linearity

### `segmentation/chebyshev.py`
**Chebyshev DP optimal segmentation:**
- `ChebyshevSegmenter`: Main segmenter class
  - Dynamic programming over discretized grid
  - Parallel Sollya precomputation
  - Minimax error minimization
  - Returns full `ApproxResult` with coefficients
- `chebyshev_segment()`: Convenience function

### `segmentation/__init__.py`
**Public API:**
```python
# Import everything through clean interface
from segmentation import (
    # Base types
    PolySegment, ApproxResult, SegmentationMethod,

    # Sollya
    SollyaInterface, SOLLYA_EXPRESSIONS,

    # Curvature method
    CurvatureSegmenter, curvature_segment,

    # Chebyshev method
    ChebyshevSegmenter, chebyshev_segment
)
```

## Usage Examples

### Before (Old Code)
```python
# Had to import from multiple places
from chebyshev import OptimalSegmenter
from analyze_breakpoints import suggest_adaptive_breakpoints, numerical_derivative
from luts.sollya_expressions import SOLLYA_EXPRESSIONS

# Data structures duplicated
@dataclass
class PolySegment:  # Defined in chebyshev.py
    lo: float
    hi: float
    coeffs: List[float]
    error: float

# Sollya interface also duplicated
class SollyaInterface:  # Defined in multiple files
    ...
```

### After (New Code)
```python
# Single import
from segmentation import (
    CurvatureSegmenter,
    ChebyshevSegmenter,
    PolySegment,
    ApproxResult
)

# Clean API
curvature_seg = CurvatureSegmenter()
chebyshev_seg = ChebyshevSegmenter(grid_size=100)

# Or use convenience functions
from segmentation import curvature_segment, chebyshev_segment

breakpoints = curvature_segment(np.tanh, (-5, 5), num_segments=16)
result = chebyshev_segment('tanh(x)', (-5, 5), num_segments=16, degree=3)
```

## Migration Guide

### For `luts/generate_adaptive_segments.py`

**Old imports:**
```python
from analyze_breakpoints import (
    suggest_adaptive_breakpoints,
    numerical_derivative
)
from sollya_expressions import SOLLYA_EXPRESSIONS
from chebyshev import OptimalSegmenter
```

**New imports:**
```python
from segmentation import (
    curvature_segment,
    chebyshev_segment,
    ChebyshevSegmenter,
    SOLLYA_EXPRESSIONS,
    numerical_derivative
)
```

### For new code using segmentation

```python
# Curvature-based segmentation
from segmentation import curvature_segment

breakpoints = curvature_segment(
    func=my_function,
    domain=(-5, 5),
    num_segments=16,
    critical_points=[0.0]  # Optional
)

# Chebyshev DP segmentation
from segmentation import chebyshev_segment

result = chebyshev_segment(
    func_expr='1/(1+exp(-x))',  # Sollya expression
    domain=(-5, 5),
    num_segments=16,
    degree=3,
    grid_size=100,
    precision='single'
)

# Access result
print(f"Max error: {result.max_error}")
print(f"Breakpoints: {result.breakpoints}")
print(f"Segment widths: {result.segment_widths}")
```

## Inheritance Hierarchy

```
SegmentationMethod (base class)
├── CurvatureSegmenter
│   ├── Fast, lightweight
│   ├── Returns breakpoints only
│   └── Works with Python functions
└── ChebyshevSegmenter
    ├── Optimal minimax
    ├── Returns full ApproxResult
    └── Requires Sollya expressions
```

Both implement:
```python
def segment(self, func_name, domain, num_segments, degree, **kwargs) -> Result
```

## Testing

### Test imports
```bash
cd segmentation
python3 -c "
from segmentation import (
    PolySegment, ApproxResult,
    CurvatureSegmenter, ChebyshevSegmenter,
    SollyaInterface, SOLLYA_EXPRESSIONS
)
print('✓ All imports successful')
"
```

### Test generate_adaptive_segments
```bash
python3 -m py_compile luts/generate_adaptive_segments.py
# Should compile without errors
```

### Test functionality
```bash
# Test curvature method
python3 -c "
from segmentation import curvature_segment
import numpy as np
bps = curvature_segment(np.tanh, (-5, 5), 16)
print(f'Generated {len(bps)} breakpoints')
"

# Test Chebyshev method (requires Sollya)
python3 -c "
from segmentation import chebyshev_segment
result = chebyshev_segment('tanh(x)', (-5, 5), 16, 3)
print(f'Max error: {result.max_error:.6e}')
"
```

## Benefits Summary

1. **Single Source of Truth**
   - `PolySegment`, `ApproxResult` defined once in `base.py`
   - `SollyaInterface` defined once in `sollya_interface.py`
   - No code duplication

2. **Clean Separation**
   - `segmentation/`: Core algorithms and interfaces
   - `luts/`: High-level generation scripts that use segmentation
   - `analyze_breakpoints.py`: Legacy file, can be deprecated

3. **Easy to Extend**
   - Add new segmentation method: inherit from `SegmentationMethod`
   - Add new Sollya expression: update `SOLLYA_EXPRESSIONS` dict
   - Add new utility: add to appropriate module

4. **Testable**
   - Each module can be tested independently
   - Clear interfaces make mocking easy
   - No circular dependencies

5. **Discoverable**
   - `from segmentation import *` shows all public API
   - Clear module names indicate purpose
   - Docstrings on all public functions

## Files Status

### New Files (Keep)
- ✅ `segmentation/__init__.py`
- ✅ `segmentation/base.py`
- ✅ `segmentation/sollya_interface.py`
- ✅ `segmentation/sollya_expressions.py`
- ✅ `segmentation/curvature.py`
- ✅ `segmentation/chebyshev.py`

### Modified Files
- ✅ `luts/generate_adaptive_segments.py` - Updated to use segmentation module

### Legacy Files (Can be deprecated)
- ⚠️ `chebyshev.py` (root) - Replaced by `segmentation/chebyshev.py`
- ⚠️ `luts/sollya_expressions.py` - Replaced by `segmentation/sollya_expressions.py`
- ⚠️ Parts of `analyze_breakpoints.py` - Curvature code moved to `segmentation/curvature.py`

### Keep But Update
- `analyze_breakpoints.py` - Still needed for `identify_critical_points()` and analysis/plotting
- `luts/sollya_bf16_remez.py` - Specific to BF16 LUT generation
- `luts/sollya_optimal_degree_search.py` - Specific utility
- `luts/generate_piecewise_remez_luts.py` - High-level LUT generation

## Next Steps

1. ✅ Create segmentation module structure
2. ✅ Move code to appropriate modules
3. ✅ Update imports in generate_adaptive_segments.py
4. ✅ Test compilation
5. ⏭️ Update other dependent scripts (if any)
6. ⏭️ Add unit tests for segmentation module
7. ⏭️ Update documentation to reference new structure

## Backward Compatibility

To maintain backward compatibility during transition:

```python
# In chebyshev.py (root)
"""Legacy compatibility - use segmentation.chebyshev instead"""
from segmentation.chebyshev import *
import warnings
warnings.warn("Use 'from segmentation import chebyshev_segment' instead", DeprecationWarning)
```

This allows old code to keep working while encouraging migration.
