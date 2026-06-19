# Chebyshev DP-Based Optimal Segmentation

This document describes the integration of Chebyshev's dynamic programming approach for optimal piecewise polynomial segmentation into the adaptive segments workflow.

## Overview

The adaptive segments generation now supports two methods for determining breakpoint placement:

1. **Curvature-based method** (default): Uses log-curvature weighted cumulative distribution
2. **Chebyshev DP method**: Uses dynamic programming with Sollya's `fpminimax` to find optimal segment boundaries

## Method Comparison

### Curvature-Based (Default)
- Fast and lightweight
- Places segments based on estimated curvature
- Works with Python functions directly
- No external dependencies beyond NumPy

### Chebyshev DP
- Optimal in minimax sense for a given degree
- Precomputes errors for all candidate segments
- Requires Sollya installation
- Slower but guarantees optimality
- Works best with smooth functions that have Sollya expressions

## Usage

### Basic Usage

Generate adaptive segments using the default curvature-based method:
```bash
cd luts
python3 generate_adaptive_segments.py --depth 16
```

Generate using Chebyshev DP method:
```bash
python3 generate_adaptive_segments.py --depth 16 --chebyshev-splitting --degree 3
```

### Comparison Mode

Run both methods and compare results:
```bash
python3 generate_adaptive_segments.py --depth 16 --compare --degree 3
```

This generates two sets of CSV files:
- `adaptive_segments_curvature/depth_16_segments.csv`
- `adaptive_segments_chebyshev/depth_16_segments.csv`

### Options

- `--chebyshev-splitting`: Enable Chebyshev DP-based optimal segmentation
- `--degree N`: Polynomial degree for Chebyshev method (default: 3)
  - Higher degrees = better fit per segment but more computation
  - Typical values: 2 (quadratic), 3 (cubic), 4 (quartic)
- `--compare`: Run both methods side-by-side for comparison
- `--activation NAME`: Process only specific activation (e.g., `sigmoid`)
- `--depth N`: Generate only specific depth (e.g., `16`)

## How It Works

### 1. Sollya Expression Mapping

Functions are mapped to Sollya-compatible expressions in `sollya_expressions.py`:
```python
SOLLYA_EXPRESSIONS = {
    'sigmoid': '1/(1+exp(-x))',
    'tanh': 'tanh(x)',
    'gelu': 'x * 0.5 * (1 + erf(x / sqrt(2)))',
    # ...
}
```

### 2. Piecewise Function Handling

For piecewise functions (ELU, SELU, CELU), the Chebyshev method:
1. Splits function at critical points (discontinuities)
2. Allocates segment budget to each piece based on curvature
3. Runs Chebyshev DP on each piece separately
4. Concatenates results

Example for ELU:
- Piece 1: `x < 0` → expression `exp(x) - 1` gets N₁ segments
- Piece 2: `x ≥ 0` → expression `x` gets N₂ segments
- Total: N₁ + N₂ = target_segments

### 3. Padding/Trimming

If Chebyshev DP converges with fewer segments than requested:
- **Padding**: Split largest high-curvature segments until target depth
- **Trimming**: Remove points creating smallest segments (preserving critical points)

### 4. Critical Point Preservation

Critical points (discontinuities, piecewise boundaries) are always included:
- For piecewise linear functions: discontinuities are segment boundaries
- For point/value pairs: only x-coordinate is used (split point)
- Ensures mathematical correctness across all methods

## Supported Activations

### Fully Supported (Sollya expressions available)
- sigmoid, tanh, silu/swish, gelu, mish, softplus
- exp, log, sin, cos, sinh, cosh, atanh, erf
- Piecewise smooth: elu, selu, celu (handled as multiple pieces)

### Fallback to Curvature Method
- Piecewise linear: relu, leaky_relu, hardtanh, hardswish (Sollya not beneficial)
- Complex functions: digamma (no simple Sollya expression)

## Testing

Run the integration test:
```bash
python3 test_chebyshev_integration.py
```

This tests both methods on sigmoid with depth=16 and validates:
- Correct number of breakpoints generated
- Boundary preservation
- Monotonicity
- Comparison statistics

## Performance Considerations

### Chebyshev Method
- **Precomputation**: O(G² × N) Sollya calls where G = grid_size, N = num_segments
- **DP solving**: O(G² × N) time
- **Grid size**: Default 50 for coarse implementation (can tune for accuracy)
- **Parallelization**: Uses ProcessPoolExecutor with 4 workers

### Typical Runtime
- Single activation, depth=16: ~5-30 seconds (depending on grid_size)
- Full sweep (all activations, depths 4,8,16,32): ~5-10 minutes

### Cache
- Sollya results are cached in `.sollya_cache_<activation>/`
- Rerunning with same parameters is much faster (seconds)
- Delete cache to force recomputation

## Integration Points

The integration touches these files:

1. **`luts/sollya_expressions.py`** (new)
   - Maps activation names to Sollya expressions
   - Handles piecewise function pieces

2. **`luts/generate_adaptive_segments.py`** (modified)
   - Added `generate_chebyshev_breakpoints()`
   - Added `pad_chebyshev_breakpoints()`
   - Added `ensure_critical_points_included()`
   - Updated CLI with `--chebyshev-splitting`, `--degree`, `--compare`

3. **`chebyshev.py`** (completed)
   - Fixed incomplete `solve()` method
   - Now returns proper `ApproxResult` with segments

## Future Enhancements

Potential improvements:
1. Adaptive grid sizing based on function complexity
2. Hybrid method: Chebyshev for high-priority regions, curvature elsewhere
3. Degree selection per segment (variable degree)
4. Integration with LUT generation pipeline for end-to-end optimization
5. Parallel processing across activations, not just segments

## References

- `chebyshev.py`: Implementation of DP-based optimal segmenter
- `analyze_breakpoints.py`: Curvature-based adaptive algorithm
- Sollya documentation: http://sollya.gforge.inria.fr/
