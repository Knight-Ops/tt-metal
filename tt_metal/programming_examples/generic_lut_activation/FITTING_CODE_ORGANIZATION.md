# Fitting Code Organization

## Reorganization Summary

Moved polynomial fitting code from `luts/` to `segmentation/` where it belongs.

### Before (Incorrect)
```
luts/
├── sollya_bf16_remez.py              # ❌ Polynomial fitting (misplaced)
├── sollya_optimal_degree_search.py   # ❌ Degree search (misplaced)
└── generate_luts_from_coefficient_table.py  # ✓ LUT generation (correct place)

segmentation/
├── chebyshev.py                      # ✓ DP segmentation
└── curvature.py                      # ✓ Curvature segmentation
```

### After (Correct)
```
segmentation/                         # Polynomial fitting and segmentation
├── sollya_remez.py                   # ✓ Sollya Remez/fpminimax fitting
├── optimal_degree_search.py          # ✓ Optimal degree per segment
├── chebyshev.py                      # ✓ Chebyshev DP segmentation
├── curvature.py                      # ✓ Curvature-based segmentation
└── generate_coefficients.py          # ✓ High-level coefficient generation

luts/                                 # Binary LUT file generation
└── generate_luts_from_coefficient_table.py  # ✓ CSV → .LUT converter
```

## Clear Separation of Concerns

### segmentation/ - Polynomial Approximation
**Purpose**: Generate optimal polynomial coefficients

**Contents**:
- Segmentation strategies (Chebyshev DP, curvature-based)
- Polynomial fitting (Sollya Remez, fpminimax)
- Degree optimization (search for best degree per segment)
- Coefficient generation (output CSV)

**Input**: Function expression, domain, parameters
**Output**: CSV with polynomial coefficients

### luts/ - Hardware LUT Generation
**Purpose**: Convert coefficients to hardware-specific binary format

**Contents**:
- Binary .LUT file generation
- Precision conversion (FP32 → BF16, etc.)
- Hardware-specific formatting

**Input**: CSV with coefficients
**Output**: Binary .LUT files for hardware

## Moved Files

| Old Location | New Location | Purpose |
|--------------|--------------|---------|
| `luts/sollya_bf16_remez.py` | `segmentation/sollya_remez.py` | Sollya Remez fitting |
| `luts/sollya_optimal_degree_search.py` | `segmentation/optimal_degree_search.py` | Degree optimization |

## Updated Exports

`segmentation/__init__.py` now exports:
```python
from segmentation import (
    # Core segmentation
    ChebyshevSegmenter,
    CurvatureSegmenter,

    # Sollya fitting
    SollyaBF16Fitter,
    OptimalDegreeSearchFitter,

    # Data structures
    PolySegment,
    ApproxResult
)
```

## Optimal Degree Search

### Feature Added
```bash
# Fixed degree (current default)
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" \
    --domain -5 5 \
    --segments 16 \
    --degree 3

# Variable degree optimization (coming soon)
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" \
    --domain -5 5 \
    --segments 16 \
    --degree 8 \
    --optimize-degree  # ← Try degrees 1-8, pick best per segment
```

### How Optimal Degree Works

With `--optimize-degree`:
1. For each segment, try degrees 1, 2, 3, ..., up to `--degree`
2. Evaluate max error (L∞) in target precision (BF16/FP32)
3. Select degree with lowest error for that segment
4. Different segments can use different degrees (variable degree)

**Why this matters:**
- Higher degree ≠ always better (quantization, instability)
- Degree 3 might beat degree 6 for some segments
- Reduces memory without sacrificing accuracy

### Current Status

- `--optimize-degree` flag recognized
- Integration with Chebyshev DP **coming soon**
- Works with activation-based workflow already
- Need to adapt for Sollya expression interface

### Implementation Note

`OptimalDegreeSearchFitter` currently expects activation function names (maps to Python functions via `ACTIVATION_FUNCTIONS`). The standalone `generate_coefficients.py` uses raw Sollya expressions. Need to:
1. Adapt OptimalDegreeSearchFitter to work with Sollya expressions, OR
2. Create expression → activation name mapping, OR
3. Use both interfaces (expressions for standalone, names for integrated)

## Benefits of Reorganization

1. **Logical Structure**: Fitting code with fitting code, generation code with generation code
2. **Clear Dependencies**: `luts/` depends on `segmentation/`, not vice versa
3. **Easier to Find**: Looking for fitting? → `segmentation/`. Looking for LUT generation? → `luts/`
4. **Better Testing**: Can test fitting independently of LUT generation
5. **Cleaner Imports**: `from segmentation import OptimalDegreeSearchFitter`

## Migration for Existing Code

### If you import from luts/
```python
# Old (broken)
from luts.sollya_bf16_remez import SollyaBF16Fitter
from luts.sollya_optimal_degree_search import OptimalDegreeSearchFitter

# New (correct)
from segmentation.sollya_remez import SollyaBF16Fitter
from segmentation.optimal_degree_search import OptimalDegreeSearchFitter

# Or through __init__
from segmentation import SollyaBF16Fitter, OptimalDegreeSearchFitter
```

### Check for Broken Imports

```bash
# Find files that import from old locations
grep -r "from luts.sollya_bf16_remez" .
grep -r "from luts.sollya_optimal_degree_search" .

# These need updating to new locations
```

## File Status

### ✅ Moved and Updated
- `segmentation/sollya_remez.py` (was `luts/sollya_bf16_remez.py`)
- `segmentation/optimal_degree_search.py` (was `luts/sollya_optimal_degree_search.py`)
- `segmentation/__init__.py` (updated exports)
- `segmentation/generate_coefficients.py` (added `--optimize-degree`)

### ✅ Kept in luts/
- `luts/generate_luts_from_coefficient_table.py` (correct location)
- `luts/generate_piecewise_remez_luts.py` (high-level LUT generation)

### ⚠️ May Need Import Updates
- Any scripts importing from old `luts/sollya_*.py` locations
- Check activation config generation scripts
- Check benchmark/validation scripts

## Testing

```bash
# Test imports work
python3 -c "
from segmentation import (
    SollyaBF16Fitter,
    OptimalDegreeSearchFitter,
    ChebyshevSegmenter
)
print('✓ All imports successful')
"

# Test generate_coefficients with new flag
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" \
    --domain -5 5 \
    --segments 16 \
    --degree 8 \
    --optimize-degree

# Should show: "⚠️ NOTE: --optimize-degree recognized but not yet integrated..."
```

## Next Steps

1. **Adapt OptimalDegreeSearchFitter for Sollya Expressions**
   - Currently expects activation names
   - Need direct Sollya expression support

2. **Full Integration**
   - Wire `--optimize-degree` to call OptimalDegreeSearchFitter
   - Output CSV with variable degree per segment
   - Update CSV format to include `degree` column

3. **Testing**
   - Compare fixed vs variable degree accuracy
   - Benchmark memory savings
   - Validate with hardware

4. **Documentation**
   - Add examples with `--optimize-degree`
   - Benchmark results (accuracy vs memory)
   - Best practices for degree selection

## Summary

✅ **Fitting code moved to `segmentation/`** - Logical organization
✅ **`--optimize-degree` flag added** - Recognized, integration pending
✅ **Clean separation** - `segmentation/` → coefficients, `luts/` → binary files
✅ **Updated exports** - Easy imports via `segmentation/__init__.py`
⏭️ **Full integration coming** - Optimal degree search with Sollya expressions
