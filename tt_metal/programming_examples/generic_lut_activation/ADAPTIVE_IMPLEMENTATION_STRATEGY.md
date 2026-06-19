# Adaptive Segmentation Implementation Strategy

## Problem Statement

How should we supply adaptive segment breakpoints to the LUT generation process?

### Requirements
1. Support multiple segment counts (4, 8, 16, 32)
2. Easy to inspect and modify
3. Version controlled
4. Reproducible builds
5. Minimal complexity for users

---

## Option Analysis

### Option 1: Store in `activations_config.csv`

```csv
activation,has_native_sfpu,range_min,range_max,breakpoints_4,breakpoints_8,breakpoints_16,breakpoints_32
sigmoid,yes,-5.0,5.0,"[-5,0,5]","[-5,-2,0,2,5]","[-5,-3,-2,-1,0,1,2,3,5]","..."
```

**Pros:**
- Single source of truth
- Easy to version control
- Human-readable

**Cons:**
- ❌ CSV format poor for arrays (escaping, parsing complexity)
- ❌ Different segment counts need different columns (gets messy)
- ❌ Very wide table (32-segment breakpoints = ~100 characters)
- ❌ Manual maintenance nightmare for 29 activations × 4 depths

**Verdict:** ❌ **NOT RECOMMENDED** - CSV format not suitable for arrays

---

### Option 2: Compute On-The-Fly

```python
# In generate_piecewise_*_luts.py
if args.adaptive:
    breakpoints = suggest_adaptive_breakpoints(func_name, func, input_range, depth)
else:
    breakpoints = np.linspace(x_min, x_max, depth + 1)
```

**Pros:**
- ✅ Always optimal for any segment count
- ✅ No manual maintenance
- ✅ Algorithm already implemented (`analyze_breakpoints.py`)
- ✅ Easy to experiment with different strategies

**Cons:**
- ⚠️ Adds ~0.1-1 second per activation (curvature computation)
- ⚠️ Need to ensure dependencies available (mpmath for some functions)
- ⚠️ Slightly harder to debug (non-deterministic if algorithm changes)

**Verdict:** ✅ **RECOMMENDED** - Clean, maintainable, flexible

---

### Option 3: Pre-generated Breakpoint Files

```
luts/
  breakpoints/
    sigmoid_4.breakpoints
    sigmoid_8.breakpoints
    sigmoid_16.breakpoints
    ...
```

Each file contains just the breakpoint array:
```
-5.000000
-2.978263
-2.267481
-1.857143
...
5.000000
```

**Pros:**
- ✅ Fast lookup (no computation)
- ✅ Reproducible (exact same breakpoints every time)
- ✅ Easy to inspect individual files
- ✅ Can be version controlled

**Cons:**
- ⚠️ 29 activations × 4 depths = 116 files to manage
- ⚠️ Need generation script to create them
- ⚠️ Hard to modify (need to regenerate all files)

**Verdict:** ⚠️ **ACCEPTABLE** - Good for production, poor for experimentation

---

### Option 4: JSON/YAML Configuration

```yaml
# adaptive_breakpoints.yaml
sigmoid:
  domain: [-5.0, 5.0]
  segments:
    4: [-5.0, -0.5, 0.0, 0.5, 5.0]
    8: [-5.0, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 5.0]
    16: [-5.0, -2.978, -2.267, ..., 5.0]
    32: [...]
```

**Pros:**
- ✅ Structured format (better than CSV)
- ✅ Single file (better than 116 separate files)
- ✅ Human-readable
- ✅ Easy to parse

**Cons:**
- ⚠️ Still manual maintenance for 29 activations
- ⚠️ Large file (~1000+ lines)
- ⚠️ Need YAML parser dependency

**Verdict:** ⚠️ **ACCEPTABLE** - Good middle ground

---

## Recommended Approach: Hybrid Strategy

**Combine Options 2 and 3:**

### Phase 1: Development (compute on-the-fly)
```bash
# Generate LUTs with adaptive segmentation
python3 generate_piecewise_linear_luts.py --adaptive --depth 16

# Internally calls:
breakpoints = suggest_adaptive_breakpoints(func_name, func, input_range, depth)
```

### Phase 2: Production (cached breakpoints)
```bash
# Pre-generate breakpoint cache
python3 cache_adaptive_breakpoints.py --output breakpoints_cache.json

# Use cached breakpoints
python3 generate_piecewise_linear_luts.py --adaptive --cache breakpoints_cache.json
```

---

## Detailed Implementation Plan

### 1. Add Adaptive Flag to LUT Generators

Modify each `generate_piecewise_*_luts.py`:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adaptive', action='store_true',
                       help='Use adaptive segmentation (curvature-weighted)')
    parser.add_argument('--breakpoint-cache', type=str,
                       help='Path to cached breakpoints JSON (optional)')
    args = parser.parse_args()

    for depth in table_depths:
        for func_name, func in functions.items():
            input_range = get_activation_domain(func_name)

            # Get breakpoints (adaptive or uniform)
            if args.adaptive:
                if args.breakpoint_cache:
                    # Load from cache
                    breakpoints = load_cached_breakpoints(
                        args.breakpoint_cache, func_name, depth
                    )
                else:
                    # Compute on-the-fly
                    breakpoints = suggest_adaptive_breakpoints(
                        func_name, func, input_range, depth
                    )
            else:
                # Uniform (current behavior)
                breakpoints = np.linspace(input_range[0], input_range[1], depth + 1)

            # Generate LUT using breakpoints
            coeffs = fit_polynomial_segments(func, breakpoints, degree=1)
            write_lut(filename, coeffs)
```

### 2. Create Breakpoint Cache Generator

```python
#!/usr/bin/env python3
"""
cache_adaptive_breakpoints.py - Pre-generate adaptive breakpoints for all activations
"""

import json
import numpy as np
from ground_truth import ACTIVATION_FUNCTIONS, get_activation_domain
from analyze_breakpoints import suggest_adaptive_breakpoints

def generate_breakpoint_cache(depths=[4, 8, 16, 32], output_file='breakpoints_cache.json'):
    """Generate all adaptive breakpoints and save to JSON."""

    cache = {
        'metadata': {
            'generated': '2026-01-18',
            'algorithm': 'curvature_weighted_adaptive',
            'version': '1.0'
        },
        'breakpoints': {}
    }

    activations = [name for name in sorted(ACTIVATION_FUNCTIONS.keys())
                  if name not in ['glu', 'geglu', 'swiglu', 'reglu']]

    for func_name in activations:
        func = ACTIVATION_FUNCTIONS[func_name]
        input_range = get_activation_domain(func_name)

        cache['breakpoints'][func_name] = {
            'domain': input_range,
            'segments': {}
        }

        for depth in depths:
            breakpoints = suggest_adaptive_breakpoints(func_name, func, input_range, depth)
            # Convert to list for JSON serialization
            cache['breakpoints'][func_name]['segments'][str(depth)] = breakpoints.tolist()

    with open(output_file, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"Generated breakpoint cache: {output_file}")
    print(f"  Activations: {len(activations)}")
    print(f"  Depths: {depths}")
    print(f"  Total entries: {len(activations) * len(depths)}")

if __name__ == "__main__":
    generate_breakpoint_cache()
```

### 3. Modify Polynomial Fitting Functions

Current fitting functions assume uniform segments. Need to generalize:

**Before (uniform only):**
```python
def fit_piecewise_linear(func, num_segments, input_range):
    x_min, x_max = input_range
    segment_width = (x_max - x_min) / num_segments

    for i in range(num_segments):
        seg_start = x_min + i * segment_width
        seg_end = x_min + (i + 1) * segment_width
        # ... fit polynomial
```

**After (supports both uniform and adaptive):**
```python
def fit_piecewise_linear(func, breakpoints_or_num_segments, input_range=None):
    """
    Fit piecewise linear approximation.

    Args:
        func: Function to approximate
        breakpoints_or_num_segments: Either:
            - int: number of segments (uniform) - requires input_range
            - array: explicit breakpoint positions
        input_range: (min, max) tuple, required if breakpoints_or_num_segments is int
    """
    # Handle both calling conventions
    if isinstance(breakpoints_or_num_segments, int):
        # Uniform segmentation (backward compatible)
        num_segments = breakpoints_or_num_segments
        x_min, x_max = input_range
        breakpoints = np.linspace(x_min, x_max, num_segments + 1)
    else:
        # Adaptive segmentation (new)
        breakpoints = np.asarray(breakpoints_or_num_segments)
        num_segments = len(breakpoints) - 1

    slopes = np.zeros(num_segments, dtype=np.float32)
    offsets = np.zeros(num_segments, dtype=np.float32)

    for i in range(num_segments):
        seg_start = breakpoints[i]
        seg_end = breakpoints[i + 1]

        # Sample densely within segment for least-squares fit
        x_samples = np.linspace(seg_start, seg_end, 100, dtype=np.float32)
        y_samples = func(x_samples)

        # Fit line: y = slope * x + offset
        X = np.vstack([x_samples, np.ones_like(x_samples)]).T
        coeffs = np.linalg.lstsq(X, y_samples, rcond=None)[0]

        slopes[i] = coeffs[0]
        offsets[i] = coeffs[1]

    return slopes, offsets
```

### 4. Update Hardware Lookup (C++)

Current lookup assumes uniform segments:

**Before:**
```cpp
// Uniform segment lookup (fast - just division)
float segment_width = (x_max - x_min) / num_segments;
int segment_idx = (x - x_min) / segment_width;
```

**After (adaptive):**
```cpp
// Adaptive segment lookup (binary search)
// Breakpoints stored as: [x0, x1, x2, ..., xN]

int find_segment_binary_search(float x, const float* breakpoints, int num_segments) {
    // Binary search to find segment containing x
    int left = 0;
    int right = num_segments;

    while (left < right) {
        int mid = (left + right) / 2;
        if (x < breakpoints[mid]) {
            right = mid;
        } else if (x >= breakpoints[mid + 1]) {
            left = mid + 1;
        } else {
            return mid;  // x is in segment [breakpoints[mid], breakpoints[mid+1])
        }
    }

    return left;
}
```

**Performance impact:** Binary search is O(log N) vs O(1) for division
- 16 segments: ~4 comparisons vs 1 division
- On modern hardware: ~10 cycles vs 5 cycles (negligible)

### 5. LUT File Format Changes

Need to store breakpoints in addition to coefficients.

**Option A: Separate header**
```
File format:
[uint32_t format_version]  // 1 = uniform, 2 = adaptive
[uint32_t num_segments]
[float32 x_min, x_max]     // Only if format_version == 1

// If adaptive (format_version == 2):
[float32 breakpoint_0]
[float32 breakpoint_1]
...
[float32 breakpoint_N]

// Then coefficients as before:
[float32 coeff_0_0]
[float32 coeff_0_1]
...
```

**Option B: Keep uniform, add adaptive variants**
```
luts/
  piecewise_linear_sigmoid_16.lut          # Uniform (existing)
  piecewise_linear_adaptive_sigmoid_16.lut # Adaptive (new)
```

**Recommendation:** Option B (cleaner, backward compatible)

---

## Usage Examples

### Development Workflow
```bash
# Generate uniform LUTs (current behavior)
python3 generate_piecewise_linear_luts.py --activation sigmoid

# Generate adaptive LUTs (compute on-the-fly)
python3 generate_piecewise_linear_luts.py --activation sigmoid --adaptive

# Compare accuracy
python3 plot_theoretical_error.py --activation sigmoid
```

### Production Workflow
```bash
# 1. Pre-generate breakpoint cache (one-time)
python3 cache_adaptive_breakpoints.py --output breakpoints_cache.json

# 2. Generate all adaptive LUTs using cache (fast)
python3 generate_piecewise_linear_luts.py --adaptive --cache breakpoints_cache.json
python3 generate_piecewise_cubic_remez_luts.py --adaptive --cache breakpoints_cache.json

# 3. Verify cache is used
ls -lh breakpoints_cache.json
# Should be ~50-100 KB for all activations
```

### CI/CD Integration
```bash
# In GitHub Actions / CI pipeline
- name: Generate breakpoint cache
  run: python3 cache_adaptive_breakpoints.py --output breakpoints_cache.json

- name: Generate adaptive LUTs
  run: |
    python3 generate_piecewise_linear_luts.py --adaptive --cache breakpoints_cache.json
    python3 generate_piecewise_cubic_remez_luts.py --adaptive --cache breakpoints_cache.json

- name: Commit cache if changed
  run: |
    git add breakpoints_cache.json luts/
    git diff --staged --quiet || git commit -m "Update adaptive breakpoints"
```

---

## Recommended File Structure

```
generic_lut_activation/
├── activations_config.csv           # Keep simple: name, domain, has_sfpu
├── breakpoints_cache.json           # Pre-generated adaptive breakpoints
├── analyze_breakpoints.py           # Analysis tool (curvature, discontinuities)
├── cache_adaptive_breakpoints.py    # Generate breakpoints_cache.json
├── generate_piecewise_linear_luts.py      # Modified to support --adaptive
├── generate_piecewise_cubic_remez_luts.py # Modified to support --adaptive
└── luts/
    ├── piecewise_linear_sigmoid_16.lut          # Uniform
    ├── piecewise_linear_adaptive_sigmoid_16.lut # Adaptive
    ├── piecewise_cubic_remez_sigmoid_16.lut
    └── piecewise_cubic_remez_adaptive_sigmoid_16.lut
```

---

## Migration Path

### Phase 1: Add Adaptive Support (Week 1)
- [x] Implement `analyze_breakpoints.py` ✅ Done
- [ ] Add `cache_adaptive_breakpoints.py`
- [ ] Modify LUT generators to accept `--adaptive` flag
- [ ] Update fitting functions to accept explicit breakpoints

### Phase 2: Testing (Week 2)
- [ ] Generate adaptive LUTs for all activations
- [ ] Benchmark accuracy improvement (MAE, max error)
- [ ] Profile performance impact of binary search vs division

### Phase 3: Hardware Integration (Week 3)
- [ ] Implement adaptive segment lookup in C++
- [ ] Update LUT loader to handle adaptive format
- [ ] Run hardware benchmarks

### Phase 4: Optimization (Week 4)
- [ ] Tune adaptive algorithm parameters
- [ ] A/B test different breakpoint strategies
- [ ] Document results in tech report

---

## Decision: Recommendation

**Use Option 2 (Compute On-The-Fly) + JSON Cache**

**Implementation:**
1. Add `--adaptive` flag to all LUT generators
2. Compute breakpoints using `suggest_adaptive_breakpoints()`
3. Optional: Use `--cache breakpoints_cache.json` for production builds
4. Store adaptive LUTs with `_adaptive` suffix to avoid confusion

**Rationale:**
- ✅ Clean, maintainable code
- ✅ Easy to experiment with different algorithms
- ✅ No manual CSV wrangling
- ✅ Reproducible via cache file
- ✅ Fast iteration during development
- ✅ Production-ready with caching

**activations_config.csv stays simple:**
```csv
activation,has_native_sfpu,range_min,range_max,notes
sigmoid,yes,-5.0,5.0,"Standard sigmoid, S-shaped curve"
relu,yes,-2.0,5.0,"Piecewise linear, discontinuity at x=0"
exp,yes,-5.0,5.0,"Exponential growth, high curvature at right boundary"
```

No need to store breakpoint arrays - compute them when needed!
