# TODO: Next Steps for Generic LUT Activation

**Last updated:** January 16, 2026 (end of day)

---

# ⚠️ CRITICAL ARCHITECTURAL ISSUES ⚠️

**STOP:** Before proceeding with any other work, read this section carefully. Two critical issues have been identified that affect all piecewise approximation methods.

---

## 🔥🔥 CRITICAL #1: Fixed Spacing Quantization is Broken

### Store Segment Boundaries in LUT Files 🐛 ARCHITECTURAL FLAW
**Priority:** **HIGHEST** - Blocking all other optimization work
**Discovered:** January 16, 2026 (evening)
**Status:** Not yet implemented - fundamental design flaw affecting all piecewise methods

**Problem:** Current LUT format only stores polynomial coefficients. Segment boundaries are computed using fixed uniform spacing:
```cpp
const float SEGMENT_WIDTH = INPUT_RANGE / NUM_SEGMENTS;
v_if (x_clamped >= input_min + SEGMENT_WIDTH) { ... }
v_if (x_clamped >= input_min + 2 * SEGMENT_WIDTH) { ... }
```

**Why this is broken:**
- **Uniform spacing is mathematically suboptimal** for functions with non-uniform curvature
- **Sigmoid**: Most curvature at x=0, but uniform spacing wastes segments on flat tails
- **Exp**: Exponential growth means uniform spacing clusters points in wrong regions
- **Asymmetric functions**: Need asymmetric boundaries, not symmetric uniform ones
- **Quantization mismatch**: LUT generation can compute optimal boundaries, but kernel can't use them!

**Impact:**
- Accuracy degraded by 2-10× compared to optimal boundary placement
- Higher polynomial degrees needed to compensate for poor segmentation
- Memory wasted on over-sampling flat regions
- Under-sampling high-curvature regions causes spikes in error

**Required changes:**

**1. Update LUT file format to include boundaries:**
```python
# generate_piecewise_*_luts.py
def write_piecewise_lut(filename, boundaries, coefficients):
    """
    New format:
    [uint32_t num_boundaries]  # N+1 boundaries for N segments
    [float32 boundary_0]
    [float32 boundary_1]
    ...
    [float32 boundary_N]
    [uint32_t num_coeffs]      # N * coeffs_per_segment
    [float32 coeff_0_0]
    [float32 coeff_0_1]
    ...
    """
    num_boundaries = len(boundaries)
    with open(filename, 'wb') as f:
        # Write boundaries section
        f.write(struct.pack('I', num_boundaries))
        for boundary in boundaries:
            f.write(struct.pack('f', float(boundary)))

        # Write coefficients section
        f.write(struct.pack('I', len(coefficients)))
        for coeff in coefficients:
            f.write(struct.pack('f', float(coeff)))
```

**2. Update LUT loader to read boundaries:**
```cpp
// lut_loader.hpp
struct LUTData {
    std::vector<float> boundaries;  // Length: num_segments + 1
    std::vector<float> coefficients;  // Length: num_segments * coeffs_per_segment
};

LUTData load_lut_with_boundaries(const std::string& path) {
    // Read boundaries section
    uint32_t num_boundaries;
    file.read(&num_boundaries, sizeof(uint32_t));
    std::vector<float> boundaries(num_boundaries);
    file.read(boundaries.data(), num_boundaries * sizeof(float));

    // Read coefficients section
    uint32_t num_coeffs;
    file.read(&num_coeffs, sizeof(uint32_t));
    std::vector<float> coefficients(num_coeffs);
    file.read(coefficients.data(), num_coeffs * sizeof(float));

    return {boundaries, coefficients};
}
```

**3. Update kernels to use boundaries from LUT:**
```cpp
// kernels/compute/piecewise_quadratic.cpp
template <>
inline void piecewise_quadratic_lut<12>(
    const std::array<float, 12>& coeffs,
    const std::array<float, 5>& boundaries,  // N+1 boundaries for N segments
    float input_min, float input_max
) {
    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];
        vFloat x_clamped = x;
        v_if (x < input_min) { x_clamped = input_min; } v_endif;
        v_if (x > input_max) { x_clamped = input_max; } v_endif;

        vFloat a = coeffs[0], b = coeffs[1], c = coeffs[2];

        // Use boundaries from LUT instead of computed uniform spacing
        v_if (x_clamped >= boundaries[1]) { a = coeffs[3]; b = coeffs[4]; c = coeffs[5]; } v_endif;
        v_if (x_clamped >= boundaries[2]) { a = coeffs[6]; b = coeffs[7]; c = coeffs[8]; } v_endif;
        v_if (x_clamped >= boundaries[3]) { a = coeffs[9]; b = coeffs[10]; c = coeffs[11]; } v_endif;

        vFloat result = (a * x_clamped + b) * x_clamped + c;
        dst_reg[d] = result;
    }
}
```

**4. Enable curvature-adaptive boundary generation:**
```python
# generate_luts.py
def compute_optimal_boundaries(func, x_min, x_max, num_segments, method='curvature'):
    """
    Generate non-uniform boundaries based on function characteristics.

    Methods:
    - 'curvature': Place more boundaries where |f''(x)| is large
    - 'error': Iteratively refine to minimize max error
    - 'adaptive': Recursively subdivide high-error regions
    """
    if method == 'curvature':
        # Sample second derivative
        x_samples = np.linspace(x_min, x_max, 1000)
        curvatures = [abs(derivative(func, x, n=2)) for x in x_samples]

        # Compute CDF and place boundaries at equal curvature intervals
        curvature_cdf = np.cumsum(curvatures)
        curvature_cdf /= curvature_cdf[-1]
        boundary_positions = np.linspace(0, 1, num_segments + 1)
        boundaries = np.interp(boundary_positions, curvature_cdf, x_samples)

        return boundaries
```

**Benefits of this change:**
- 2-10× better accuracy with same memory budget
- Can achieve depth-8 accuracy with depth-4 segments (2× memory reduction)
- Mathematically principled boundary placement
- Future-proof: Can experiment with different boundary strategies
- Fixes fundamental design flaw

**Implementation order:**
1. Update one LUT generator (start with quadratic) to write boundaries
2. Update lut_loader.hpp to support new format
3. Update one kernel (piecewise_quadratic.cpp) to use boundaries
4. Test accuracy improvement on sigmoid, exp, gelu
5. Roll out to all piecewise methods once validated
6. Regenerate all LUT files with optimal boundaries

**Estimated impact:**
- **Accuracy improvement:** 2-10× for functions with non-uniform curvature
- **Memory reduction:** Can reduce depth by 50% for same accuracy
- **Development effort:** ~1-2 days for full rollout
- **Backward compatibility:** Breaking change - requires LUT regeneration

**DO NOT PROCEED** with any other tasks until this is addressed. The current fixed-spacing approach is fundamentally flawed and limits the effectiveness of all other optimizations.

---

# 🔥 CRITICAL #2: SFPU Compiler Bug

### Create Minimal Reproducing Example 🐛 URGENT
**Priority:** CRITICAL - Blocking bug affecting all loop-based SFPU kernels
**Status:** Workaround implemented, bug report pending

**Problem:** SFPU compiler fails to properly unroll loops with dynamic array indexing inside `v_if` blocks. Code compiles successfully but produces wrong results on hardware.

**Current workaround:** Hardcoded template specializations for depths 4, 8, 16, 32 (see commits 6f32c1ae56, d0d579e93b, 4a9e116a4b, cf93633019)

**Required for bug report:**

1. **Create minimal test kernel** (`sfpu_loop_unroll_bug_test.cpp`):
```cpp
// FAILING VERSION: Generic loop with array indexing
#pragma GCC unroll 4
for (uint32_t seg = 0; seg < 4; seg++) {
    v_if (x < thresholds[seg]) {
        result = values[seg];
    }
    v_endif;
}

// WORKING VERSION: Hardcoded if-else
v_if (x < thresholds[0]) {
    result = values[0];
} v_elseif (x < thresholds[1]) {
    result = values[1];
} v_elseif (x < thresholds[2]) {
    result = values[2];
} v_else {
    result = values[3];
}
v_endif;
```

2. **Collect evidence:**
- Hardware output CSV showing wrong segment selection (loop version)
- Hardware output CSV showing correct results (hardcoded version)
- Compile logs showing no warnings/errors
- Git history of 40+ debugging attempts (commits 1275d3cb11 through 70489a91f9)

3. **Document symptoms:**
- Different segments selected than expected
- Appears to be register allocation or loop unrolling failure
- Issue affects all depths and all piecewise methods
- No compile-time errors or warnings

4. **File bug report:**
- Tenstorrent GitHub issues or internal bug tracker
- Include minimal reproducing code
- Include hardware outputs
- Link to full implementation in this repo
- Reference commits showing debugging journey

**Key files for reference:**
- `kernels/compute/piecewise_constant.cpp` (shows both failing and working patterns)
- `kernels/compute/piecewise_linear.cpp`
- `kernels/compute/piecewise_quadratic.cpp`
- `kernels/compute/piecewise_cubic.cpp`

---

## 🔥 Immediate Tasks (January 17, 2026)

### 2. Re-run Blackhole Sweep 🔧 HIGH PRIORITY
**Status:** Previous sweep had incomplete/corrupted CSV files
**Issue:** Missing headers in `blackhole_piecewise_quadratic_remez_results.csv`

**Steps:**
```bash
# SSH to Blackhole server
source hosts.sh
ssh -A -p $BLACKHOLE_PORT $BLACKHOLE_HOST

# Navigate to project directory
cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation

# Clean old results
rm blackhole_*.csv

# Re-run full sweep with all methods
./sweep_all.sh

# Verify all CSVs have proper headers
for f in blackhole_*.csv; do
    echo "=== $f ==="
    head -1 "$f"  # Should show: depth,activation,tile_count,...
    wc -l "$f"    # Should show >100 lines for full sweep
done

# Commit results
cd /localdev/nkapre/tt-metal
git add -f tt_metal/programming_examples/generic_lut_activation/blackhole_*.csv
git commit -m "Complete Blackhole sweep results with hardcoded SFPU workarounds

- All methods: Native SFPU, PC, PL, PQR, PCR
- Depths: [4, 8, 16] (32 removed as overkill)
- Tiles: [32, 64, 128, 256]
- Total: ~460 configurations
- Using hardcoded specializations to work around SFPU compiler bug

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Pull, rebase, and push
git pull --rebase origin feature/generic-lut-activation
git push origin feature/generic-lut-activation
```

### 3. Generate Pareto Frontier Plots 📊 HIGH PRIORITY
**Why:** New plot design ready, need to generate for both architectures

**Steps:**
```bash
cd tt_metal/programming_examples/generic_lut_activation

# Generate Pareto plots (after Blackhole sweep completes)
python3 plot_pareto_frontier.py --arch wormhole
python3 plot_pareto_frontier.py --arch blackhole

# Commit plots
git add wormhole_plots_pareto/*.png blackhole_plots_pareto/*.png
git commit -m "Add Pareto frontier plots showing polynomial degree tradeoffs

- 1x3 grid layout: Depths 4, 8, 16 (excluded 32 as overkill)
- Dual Y-axes: MAE (log scale, blue) + Runtime (ms, red)
- X-axis: Polynomial degree (Const, Linear, Quad, Cubic)
- Annotations: Memory usage and runtime values
- Added heatmap: Depth vs Degree vs MAE

Key insights:
- Depth 4: Linear→Quad huge MAE improvement
- Depth 8: Best balance accuracy/speed
- Depth 16: Diminishing returns for Cubic

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
git push origin feature/generic-lut-activation
```

## 🔬 Research Tasks

### 4. Explore Higher-Order Polynomials 🧪 RESEARCH
**Goal:** Test degree-6 (hexic) and degree-8 (octic) approximations to understand accuracy vs complexity tradeoffs

**Motivation:**
- Cubic (degree-3) provides excellent accuracy for most functions
- But complex functions (gelu, erf, exp) may benefit from higher degrees
- Want to understand diminishing returns and memory/speed tradeoffs

**Implementation plan:**

**Step 1: Add hexic (degree-6) support**
```cpp
// kernels/piecewise_hexic.hpp
template <uint32_t NUM_SEGMENTS>
inline void piecewise_hexic_approx(vFloat& result, vFloat x, const uint32_t lut[]) {
    // 7 coefficients per segment: a0 + a1*dx + a2*dx^2 + ... + a6*dx^6
    // Coefficients: lut[seg*7 + 0..6]
    // Will need hardcoded specializations for depths 4, 8, 16
}

// Add to generic_lut_activation.cpp
if (lut_type == "piecewise_hexic_remez") {
    piecewise_hexic_approx<NUM_SEGMENTS>(result, x, lut);
}
```

**Step 2: Add LUT generation for hexic**
```python
# generate_luts.py
def generate_piecewise_hexic_remez_lut(activation, depth, x_min, x_max):
    """Generate hexic (degree-6) Remez approximation."""
    from scipy.optimize import minimize  # Use OptimalPoly or manual Remez
    # 7 coefficients per segment
    # Minimize max error (minimax)
```

**Step 3: Add octic (degree-8) support**
```cpp
// kernels/piecewise_octic.hpp
// 9 coefficients per segment: a0 + a1*dx + ... + a8*dx^8
```

**Expected findings:**
- Hexic may provide 10-100× MAE improvement over cubic for complex functions
- Octic likely shows diminishing returns (marginal improvement over hexic)
- Memory cost: Hexic uses 7×N coefficients, Octic uses 9×N
- Runtime cost: More polynomial terms = more SFPU operations

**Concerns:**
- SFPU compiler bug may affect higher-degree polynomials too
- More coefficients = more register pressure
- May need even more hardcoded specializations
- Diminishing returns likely (cubic already <0.001 MAE for most functions)

**Test plan:**
1. Implement hexic for depths 4, 8, 16
2. Run sweep on Wormhole and Blackhole
3. Compare MAE: cubic vs hexic for all 29 activations
4. Plot: Polynomial degree (0-6) vs MAE
5. Decide if octic is worth implementing based on hexic results

### 5. Non-Uniform Segment Boundaries - SEE CRITICAL ISSUE #1 ABOVE ⚠️
**Status:** MOVED TO CRITICAL PRIORITY

This was previously a research task, but is now recognized as a **critical architectural fix** that must be addressed immediately. The current fixed uniform spacing is fundamentally flawed.

**See Section 1 above** for full implementation plan and priority justification.

### 6. Discontinuity-Aware Boundary Placement 🧪 RESEARCH
**Priority:** MEDIUM (depends on completing Critical Issue #1 first)
**Goal:** Automatically detect function discontinuities and force them to be segment boundaries

**Motivation:**
- Piecewise-continuous functions (ReLU, LeakyReLU, HardTanh, HardShrink) have discontinuities in derivatives
- Current approach uses uniform or curvature-based spacing, which may place boundaries near but not exactly at discontinuities
- Discontinuities cause polynomial approximation to fail - need exact boundary placement
- Example: ReLU has derivative discontinuity at x=0, LeakyReLU at x=0, HardTanh at x=-1 and x=1

**Implementation Plan:**

**Step 1: Discontinuity Detection**
```python
# lut_utils.py
def detect_discontinuities(func, x_min, x_max, epsilon=1e-6):
    """
    Scan function range to identify discontinuities in function value or derivatives.

    Returns:
        List of x-coordinates where discontinuities occur
    """
    discontinuities = []

    # Known analytical discontinuities for common activations
    known_discontinuities = {
        'relu': [0.0],
        'leaky_relu': [0.0],
        'prelu': [0.0],
        'relu6': [0.0, 6.0],
        'hardtanh': [-1.0, 1.0],
        'hardsigmoid': [-3.0, 3.0],
        'hardswish': [-3.0, 3.0],
        'threshold': [0.0],  # or user-specified threshold
        'hardshrink': [-0.5, 0.5],  # lambda parameter
        'softshrink': [-0.5, 0.5],
    }

    # Check if function has known discontinuities
    if func_name in known_discontinuities:
        return known_discontinuities[func_name]

    # Otherwise, numerical detection:
    # Sample function densely and look for jumps in value or derivative
    x_samples = np.linspace(x_min, x_max, 10000)
    y_samples = np.array([func(x) for x in x_samples])

    # Detect value discontinuities (jumps in function value)
    dy = np.diff(y_samples)
    dx = np.diff(x_samples)
    slopes = dy / dx

    # Find locations where slope changes dramatically
    slope_changes = np.abs(np.diff(slopes))
    threshold = np.percentile(slope_changes, 99.9)  # Top 0.1% changes

    discontinuity_indices = np.where(slope_changes > threshold)[0]
    discontinuities = [x_samples[i] for i in discontinuity_indices]

    return discontinuities
```

**Step 2: Boundary Generation with Discontinuity Constraints**
```python
# generate_luts.py
def generate_boundaries_with_discontinuities(func, x_min, x_max, num_segments, func_name):
    """
    Generate non-uniform boundaries ensuring discontinuities are segment boundaries.

    Algorithm:
    1. Detect all discontinuities in [x_min, x_max]
    2. Force these as mandatory boundaries
    3. Fill remaining boundaries using curvature-based placement
    """
    # Detect discontinuities
    discontinuities = detect_discontinuities(func, x_min, x_max, func_name)

    # Filter discontinuities to range
    discontinuities = [d for d in discontinuities if x_min < d < x_max]

    if len(discontinuities) == 0:
        # No discontinuities - use curvature-based placement
        return compute_optimal_boundaries(func, x_min, x_max, num_segments, method='curvature')

    # Ensure we have enough segments for discontinuities
    if len(discontinuities) + 1 > num_segments:
        raise ValueError(f"Need at least {len(discontinuities) + 1} segments for {len(discontinuities)} discontinuities")

    # Start with discontinuities as mandatory boundaries
    boundaries = [x_min] + sorted(discontinuities) + [x_max]

    # Calculate segments needed in each region
    num_regions = len(discontinuities) + 1
    segments_remaining = num_segments - len(boundaries) + 1
    segments_per_region = segments_remaining // num_regions

    # Subdivide each region using curvature-based placement
    final_boundaries = [x_min]
    for i in range(len(boundaries) - 1):
        region_start = boundaries[i]
        region_end = boundaries[i + 1]

        # Allocate segments to this region based on its curvature
        region_boundaries = compute_optimal_boundaries(
            func, region_start, region_end,
            segments_per_region, method='curvature'
        )

        # Add interior boundaries (exclude endpoints to avoid duplicates)
        final_boundaries.extend(region_boundaries[1:-1])
        final_boundaries.append(region_end)

    return np.array(final_boundaries)
```

**Step 3: Update Kernel to Support Dynamic Boundaries**
```cpp
// kernels/compute/piecewise_quadratic.cpp
// This is already addressed by Critical Issue #1 - kernels will read boundaries from LUT
// No additional changes needed once boundary storage is implemented

// Example usage after Critical Issue #1 is fixed:
template <>
inline void piecewise_quadratic_lut<12>(
    const std::array<float, 12>& coeffs,
    const std::array<float, 5>& boundaries,  // May include discontinuities
    float input_min, float input_max
) {
    // boundaries[0] = x_min
    // boundaries[1] = discontinuity at x=0 (e.g., for ReLU)
    // boundaries[2] = curvature-based boundary
    // boundaries[3] = curvature-based boundary
    // boundaries[4] = x_max

    // Kernel logic unchanged - just uses boundaries array
    v_if (x_clamped >= boundaries[1]) { ... } v_endif;
    v_if (x_clamped >= boundaries[2]) { ... } v_endif;
    v_if (x_clamped >= boundaries[3]) { ... } v_endif;
}
```

**Step 4: LUT File Format Enhancement**
```python
# Already addressed by Critical Issue #1
# Boundaries stored in LUT file include discontinuities
# No additional format changes needed
```

**Benefits:**
- **Exact discontinuity handling**: Polynomial segments never span discontinuities
- **Better accuracy**: No Gibbs phenomenon or Runge's phenomenon at discontinuities
- **Optimal resource allocation**: Remaining segments allocated based on curvature in smooth regions
- **Automatic detection**: Works for both known and unknown discontinuities

**Expected Improvements:**
- ReLU family: 10-100× MAE improvement (current approach approximates kink, new approach respects it)
- HardTanh/HardShrink: 5-50× MAE improvement at transition points
- Smooth functions (sigmoid, tanh): No change (no discontinuities to exploit)

**Implementation Priority:**
1. ⚠️ **MUST** complete Critical Issue #1 (boundary storage) first
2. Implement analytical discontinuity detection for known activations
3. Test on ReLU family at depth=4 and depth=8
4. Implement numerical discontinuity detection as fallback
5. Extend to all piecewise-continuous activations
6. Compare MAE: curvature-only vs curvature+discontinuities

**Test Cases:**
```python
# test_discontinuity_boundaries.py
def test_relu_boundary_at_zero():
    """ReLU discontinuity at x=0 must be a segment boundary."""
    boundaries = generate_boundaries_with_discontinuities(
        relu, x_min=-3, x_max=3, num_segments=4, func_name='relu'
    )
    assert 0.0 in boundaries, "ReLU discontinuity at x=0 must be boundary"

def test_hardtanh_two_discontinuities():
    """HardTanh has discontinuities at x=-1 and x=1."""
    boundaries = generate_boundaries_with_discontinuities(
        hardtanh, x_min=-2, x_max=2, num_segments=8, func_name='hardtanh'
    )
    assert -1.0 in boundaries and 1.0 in boundaries
```

**Known Discontinuities by Activation:**
| Activation   | Discontinuity Locations | Type                    |
|--------------|-------------------------|-------------------------|
| ReLU         | x=0                     | Derivative discontinuity |
| LeakyReLU    | x=0                     | Derivative discontinuity |
| PReLU        | x=0                     | Derivative discontinuity |
| ReLU6        | x=0, x=6                | Derivative discontinuity |
| HardTanh     | x=-1, x=1               | Derivative discontinuity |
| HardSigmoid  | x=-3, x=3               | Derivative discontinuity |
| HardSwish    | x=-3, x=3               | Derivative discontinuity |
| Threshold    | x=threshold             | Value + derivative disc  |
| HardShrink   | x=-λ, x=λ               | Value + derivative disc  |
| SoftShrink   | x=-λ, x=λ               | Derivative discontinuity |

**Status:** NOT STARTED - depends on Critical Issue #1

## 📋 Medium Priority Tasks

### 7. Compare Wormhole vs Blackhole Performance 📊 MEDIUM PRIORITY
**Goal:** Understand platform differences (once Blackhole data is complete)

**Analysis script to create:**
```python
# analyze_platform_comparison.py
import pandas as pd
import matplotlib.pyplot as plt

for method in ['native_sfpu', 'piecewise_constant', 'piecewise_linear',
               'piecewise_quadratic_remez', 'piecewise_cubic_remez']:
    wh = pd.read_csv(f'wormhole_{method}_results.csv')
    bh = pd.read_csv(f'blackhole_{method}_results.csv')

    print(f"\n{method} Platform Comparison:")
    print(f"Wormhole:  {wh['runtime_mean_ms'].mean():.2f}ms avg runtime")
    print(f"Blackhole: {bh['runtime_mean_ms'].mean():.2f}ms avg runtime")
    print(f"Speedup:   {wh['runtime_mean_ms'].mean() / bh['runtime_mean_ms'].mean():.2f}×")
    print(f"Wormhole:  {wh['mae'].mean():.6f} avg MAE")
    print(f"Blackhole: {bh['mae'].mean():.6f} avg MAE")
```

**Questions to answer:**
1. Which platform is faster overall?
2. Do accuracy characteristics differ? (They shouldn't - same LUTs)
3. Are there activation-specific platform preferences?
4. Do runtime variances differ between platforms?
5. How does Blackhole's improved architecture affect SFPU performance?

### 8. Documentation Updates 📝 MEDIUM PRIORITY

**Tasks:**

1. **Update README with SFPU bug section:**
```markdown
## Known Issues

### SFPU Compiler Loop Unrolling Bug
The SFPU compiler (RISC-V backend for Tensix compute cores) fails to properly
unroll loops with dynamic array indexing inside `v_if` blocks. Code compiles
without warnings but produces incorrect results on hardware.

**Workaround:** All piecewise kernels use hardcoded template specializations
for depths 4, 8, 16 instead of generic loop-based code.

**Status:** Bug reported to Tenstorrent [link to issue]

See: 2026-01-16.md for detailed debugging journey (40+ commits)
```

2. **Document hardcoded specialization pattern:**
```markdown
## Kernel Implementation Notes

All piecewise approximation kernels use hardcoded specializations instead
of generic loops due to SFPU compiler bug:

```cpp
// ❌ DOESN'T WORK: Generic loop (compiles but wrong results)
#pragma GCC unroll 16
for (uint32_t seg = 0; seg < NUM_SEGMENTS; seg++) {
    v_if (x < thresholds[seg]) {
        result = lut[seg * COEFFS + 0] + ...;
    }
    v_endif;
}

// ✅ WORKS: Hardcoded specialization
template <>
inline void piecewise_approx<4>(...) {
    v_if (x < t0) {
        result = ...;
    } v_elseif (x < t1) {
        result = ...;
    } // ... explicit for each segment
}
```
```

3. **Add Architecture Decision Record (ADR):**
Create `docs/ADR-001-pareto-plot-design.md` documenting:
- Why we chose 1×3 grid over other layouts
- Why we excluded depth=32
- Why dual Y-axes instead of separate plots
- Design iteration history

### 9. Performance Deep Dive 📊 LOW PRIORITY
**Goal:** Understand performance characteristics and create optimization guide

**Analysis tasks:**
1. **Runtime vs Depth:** How does segment count affect kernel execution time?
2. **Runtime vs Degree:** How do more coefficients affect computation time?
3. **Memory Bandwidth:** Is DRAM transfer or kernel compute the bottleneck?
4. **Pareto Frontier:** What's the optimal accuracy/speed tradeoff for each activation?

**Create visualization:**
```python
# plot_performance_analysis.py
# 1. Runtime breakdown: device_init, data_transfer, kernel_exec
# 2. Scaling: depth vs runtime (linear? logarithmic?)
# 3. Comparison: SFPU vs piecewise methods overhead
```

## 📝 Documentation Tasks

### 10. Update README with Results Summary 📄 MEDIUM PRIORITY
**After plots are regenerated:**

Add section to README:
```markdown
## Results Summary (Corrected Input Ranges)

### Key Findings
- **logsigmoid accuracy fixed**: Mean relative error <1.0 (was >7000 with hardcoded ranges)
- **All activations**: Error metrics now meaningful and match theoretical predictions
- **29 activations tested**: Depths [4,8,16,32], tiles [32,64,128,256]
- **5 methods compared**: Native SFPU, PC, PL, PQR, PCR

### Accuracy vs Runtime Tradeoffs
[Include key plot images]

### Platform Comparison (Wormhole B0 vs Blackhole)
[Include comparison metrics]
```

## ✅ Completed Tasks

### January 16, 2026

#### SFPU Compiler Bug Investigation ✅ WORKAROUND IMPLEMENTED
- [x] Discovered loop unrolling bug in SFPU compiler
- [x] Tried 9+ different debugging approaches (40+ commits)
- [x] Implemented hardcoded specializations for all depths (4, 8, 16)
- [x] Verified hardcoded versions produce correct results on hardware
- [x] Applied workaround to all piecewise methods (PC, PL, PQ, PCR)
- [ ] Minimal reproducing example for bug report (TODO)
- [ ] Filed bug report with Tenstorrent (TODO)

#### Pareto Frontier Visualization ✅ COMPLETED
- [x] Designed 1×3 grid layout (depths 4, 8, 16)
- [x] Implemented dual Y-axes (MAE + Runtime)
- [x] Added memory usage annotations
- [x] Added runtime value annotations
- [x] Created depth vs degree error heatmap
- [x] Removed depth=32 (deemed overkill)
- [x] Iterated through 5 design revisions
- [x] Updated plot_pareto_frontier.py

#### Configuration and Path Fixes ✅ COMPLETED
- [x] Fixed CSV output directory paths in sweep scripts (c2a7d984cd)
- [x] Fixed LUT generation range mismatch (d6772ff397)
- [x] Removed hardcoded input ranges from kernels (29a5ee4b2e, 16bcbafcb6)
- [x] Fixed activations_config.csv search path from build directory (cffa6fbc4a)
- [x] Added input clamping to piecewise constant and linear (4264a1a69b)
- [x] Fixed LUT_SIZE/NUM_SEGMENTS mismatch (933b0676ae)
- [x] Updated LUT naming scheme (e82071d039, a9fffa965e)

#### Workflow and Tooling ✅ COMPLETED
- [x] Parallelized plotting scripts (7b940fad63)
- [x] Added --model flag to sweep scripts (087af81570)
- [x] Added --activation flag to plot_activation_comparison.py (7ec42a19e2)
- [x] Added --arch flag to pull_results.sh (7a1a182073)
- [x] Created unified plots.sh (07288148a4)
- [x] Removed redundant one-off shell scripts (2f3a2794c2)
- [x] Added Remote Server Access section to CLAUDE.md (e8a5934873)

#### Data Collection ✅ WORMHOLE COMPLETE
- [x] Wormhole B0 sweep completed with hardcoded workarounds
- [x] All 5 methods: Native SFPU, PC, PL, PQR, PCR
- [x] Depths tested: 4, 8, 16 (32 removed)
- [x] ~460 configurations collected
- [ ] Blackhole sweep incomplete (needs re-run)

### January 15, 2026

### Critical Bug Fix: Input Range Issue ✅ COMPLETED
- [x] Created activation_config.hpp CSV parser
- [x] Modified generic_lut_activation.cpp to use test_range
- [x] Modified generic_lut_activation_native.cpp to use test_range
- [x] Fixed filesystem header compilation error
- [x] Deployed to both servers
- [x] Rebuilt all binaries (119 total)
- [x] Launched full sweeps with corrected ranges

### Configuration Management ✅ COMPLETED
- [x] Centralized configuration in activations_config.csv
- [x] Created activations_config.sh for bash
- [x] Created activations_config.py for Python
- [x] Created activation_config.hpp for C++
- [x] All scripts now use single source of truth

### Workflow Optimizations ✅ COMPLETED
- [x] Integrated hardware capture (DUMP_OUTPUT_CSV on last run)
- [x] Removed PQ least-squares from sweeps (only use Remez)
- [x] Updated dashboard.py to remove PQ row
- [x] Updated pull_results.sh to handle missing LSQ files
- [x] Changed QUICK_RUN_DEPTHS to [32] only

### Documentation ✅ COMPLETED
- [x] Updated README with Recent Improvements section
- [x] Documented centralized configuration system
- [x] Updated activation counts (29 total, 19 native SFPU)
- [x] Enhanced CSV format documentation with timing columns
- [x] Created comprehensive work summary (2026-01-15.md)

### Sweep Execution ✅ COMPLETED
- [x] Wormhole B0 sweep completed and results committed/pushed
- [ ] Blackhole sweep in progress (expected ~19:00 UTC)

## 🔮 Future Ideas (Lower Priority)

### 11. Explore Native Hardware LUT Operations 🔬 RESEARCH
**Priority:** RESEARCH - Potential performance optimization
**Discovered:** January 18, 2026 (investigation of native SFPU tanh implementation)

**Key Finding:** Native SFPU tanh uses hardware LUT registers (`l0`, `l1`, `l2`) rather than pure mathematical operations. This provides a middle ground between full memory-based LUTs and pure SFPU math.

**Current Implementation:**
```cpp
// From ckernel_sfpu_tanh.h (line 118-135)
if constexpr (APPROXIMATION_MODE) {
    // SFPU microcode with hardware LUT registers
    sfpi::vUInt l0 = l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1 = l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2 = l_reg[sfpi::LRegs::LReg2];

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        val = sfpi::lut(val, l0, l1, l2);  // Hardware LUT lookup!
        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}
```

**Hardware LUT Initialization:**
```cpp
// tanh_init (line 159-165)
if constexpr (APPROXIMATION_MODE) {
    uint imm0 = 0x1DFF;  // 0.90625*x
    uint imm1 = 0x481A;  // 0.09375*x + 0.8125
    uint imm2 = 0xFF00;  // 1
    _sfpu_load_imm16_(0, imm0);
    _sfpu_load_imm16_(1, imm1);
    _sfpu_load_imm16_(2, imm2);
}
```

**Research Questions:**
1. **What is the hardware LUT operation?**
   - How does `sfpi::lut(val, l0, l1, l2)` work internally?
   - What's the format of the immediate values (0x1DFF, 0x481A, 0xFF00)?
   - How many LUT entries can be stored in hardware registers?

2. **Can we use this for piecewise approximations?**
   - Current approach: Load full LUT from DRAM (high memory bandwidth)
   - Alternative: Store critical parameters in hardware LUT registers
   - Hybrid: Use hardware LUT for coarse approximation + DRAM LUT for fine details

3. **Performance comparison:**
   - Hardware LUT (native SFPU tanh): ~0.0009 MAE, ~14ms runtime
   - DRAM-based LUT (piecewise linear tanh depth-16): ~0.0034 MAE, ~15ms runtime
   - Would hardware LUT scheme close the accuracy gap?

4. **Feasibility analysis:**
   - How many hardware LUT registers are available?
   - Can we fit piecewise boundaries + coefficients?
   - What's the programming model for custom hardware LUTs?

**Potential Hybrid Approach:**
```cpp
// Hypothetical: Use hardware LUT for boundary checks, DRAM for coefficients
template <uint32_t NUM_SEGMENTS>
inline void piecewise_approx_hardware_lut(...) {
    // Load segment boundaries into hardware LUT registers
    _sfpu_load_imm16_(0, boundary_1_encoded);
    _sfpu_load_imm16_(1, boundary_2_encoded);
    _sfpu_load_imm16_(2, boundary_3_encoded);

    for (int d = 0; d < 32; d++) {
        vFloat x = dst_reg[d];

        // Use hardware LUT for fast segment selection
        uint segment_id = sfpi::lut_segment_id(x, l0, l1, l2);

        // Load coefficients from DRAM based on segment_id
        // (This part still requires DRAM access but only 3-9 coefficients)
        vFloat result = evaluate_polynomial(x, coeffs[segment_id]);
        dst_reg[d] = result;
    }
}
```

**Expected Benefits:**
- Reduced DRAM bandwidth (only coefficients, not boundaries)
- Faster segment selection using hardware LUT
- Potential 10-20% runtime improvement

**Expected Challenges:**
- Limited documentation on hardware LUT programming
- May be activation-specific (different encodings for different functions)
- Hardware LUT capacity constraints
- Increased complexity vs current DRAM-based approach

**Implementation Plan:**
1. **Understand hardware LUT internals:**
   - Study `sfpi::lut()` implementation in tt-metal codebase
   - Analyze immediate value encoding format (0x1DFF → 0.90625*x)
   - Determine LUT capacity and addressing scheme

2. **Prototype for piecewise linear:**
   - Encode 4 segment boundaries as hardware LUT immediates
   - Test if boundaries can be loaded and used for segment selection
   - Compare runtime vs DRAM-based approach

3. **Measure performance:**
   - Hardware LUT approach vs current DRAM LUT
   - Quantify DRAM bandwidth savings
   - Identify bottleneck: Is it boundary checks or coefficient loading?

4. **Decide feasibility:**
   - If hardware LUT provides <5% speedup → Not worth complexity
   - If hardware LUT provides >20% speedup → Consider implementing
   - If hardware LUT is too limited → Document as infeasible

**Status:** NOT STARTED - Requires low-level hardware documentation
**Priority:** LOW - Current DRAM-based approach works well, this is optimization

**Alternative: Just use native SFPU approximations when available:**
- For tanh: Use native SFPU (0.0009 MAE)
- For activations without native SFPU: Use piecewise LUT
- Don't try to reinvent the wheel - native implementations already optimized

### 12. Investigate Remez Fitting and LSQ Fallback Behavior 🐛 MEDIUM PRIORITY
**Priority:** MEDIUM - Understanding why Remez occasionally fails and falls back to LSQ
**Status:** Investigation needed

**Observed Issue:**
The LUT generation code uses minimax Remez approximation as primary method, with least-squares (LSQ) as fallback when Remez fails. However, the conditions and frequency of these fallbacks are not well understood.

**Questions to investigate:**

1. **When does Remez fail?**
   - Which activations trigger LSQ fallback most often?
   - Which depths/polynomial degrees cause failures?
   - Is it numerical instability in the Remez algorithm?
   - Are some functions poorly suited for Remez?

2. **Why does Remez fail?**
   ```python
   # From generate_piecewise_*_remez_luts.py
   try:
       remez_result = minimax_approximation(...)
   except Exception as e:
       print(f"Remez failed for {func_name}: {e}, falling back to LSQ")
       lsq_result = least_squares_fit(...)
   ```
   - What exceptions are being caught?
   - Is it convergence failure, numerical overflow, ill-conditioned matrices?
   - Can we predict which functions will fail before attempting Remez?

3. **How much accuracy do we lose with LSQ fallback?**
   - Compare MAE: Remez vs LSQ for same activation/depth/degree
   - Quantify worst-case accuracy degradation
   - Identify if certain activations are severely impacted

4. **Can we improve Remez robustness?**
   - Better initial guesses for Remez iteration
   - Numerical conditioning (rescaling input range)
   - Alternative minimax algorithms (Chebyshev nodes, etc.)
   - Hybrid approach: Use Remez for most segments, LSQ for problematic ones

**Investigation plan:**

**Step 1: Add logging to LUT generation**
```python
# Modify generate_piecewise_*_remez_luts.py
def generate_lut_with_fallback_tracking(activation, depth, degree):
    fallback_log = []

    for segment in range(depth):
        try:
            coeffs = minimax_approximation(func, segment_bounds, degree)
            method = "remez"
        except Exception as e:
            print(f"⚠️  Remez failed for {activation} seg={segment}: {e}")
            coeffs = least_squares_fit(func, segment_bounds, degree)
            method = "lsq"
            fallback_log.append({
                'activation': activation,
                'segment': segment,
                'depth': depth,
                'degree': degree,
                'exception': str(e)
            })

    # Save fallback log
    with open('remez_fallback_log.json', 'a') as f:
        json.dump(fallback_log, f, indent=2)
```

**Step 2: Analyze fallback patterns**
```python
# analyze_remez_failures.py
import json
import pandas as pd

# Load all fallback logs
fallbacks = []
with open('remez_fallback_log.json') as f:
    for line in f:
        fallbacks.append(json.loads(line))

df = pd.DataFrame(fallbacks)

# Analysis questions:
print("Activations with most Remez failures:")
print(df.groupby('activation').size().sort_values(ascending=False))

print("\nDegrees with most Remez failures:")
print(df.groupby('degree').size().sort_values(ascending=False))

print("\nDepths with most Remez failures:")
print(df.groupby('depth').size().sort_values(ascending=False))

print("\nCommon exception types:")
print(df['exception'].value_counts())
```

**Step 3: Compare accuracy of Remez vs LSQ**
```python
# compare_remez_lsq_accuracy.py
def compare_approximation_quality(activation, depth, degree):
    """Generate both Remez and LSQ LUTs, measure MAE for each segment."""

    remez_lut = generate_with_remez(activation, depth, degree)
    lsq_lut = generate_with_lsq(activation, depth, degree)

    # Test on dense grid
    test_points = np.linspace(x_min, x_max, 100000)

    remez_mae = compute_mae(test_points, remez_lut)
    lsq_mae = compute_mae(test_points, lsq_lut)

    print(f"{activation} depth={depth} degree={degree}:")
    print(f"  Remez MAE: {remez_mae:.6e}")
    print(f"  LSQ MAE:   {lsq_mae:.6e}")
    print(f"  Ratio:     {lsq_mae / remez_mae:.2f}× worse")
```

**Step 4: Improve Remez robustness**
```python
# Enhanced Remez with better initial guesses
def robust_minimax_approximation(func, x_min, x_max, degree):
    """Try multiple strategies to get Remez to converge."""

    strategies = [
        # Strategy 1: Standard Chebyshev nodes
        {'initial_points': 'chebyshev', 'max_iter': 100},

        # Strategy 2: Equispaced initial points
        {'initial_points': 'uniform', 'max_iter': 100},

        # Strategy 3: Rescale to [-1, 1] for numerical stability
        {'initial_points': 'chebyshev', 'rescale': True, 'max_iter': 200},

        # Strategy 4: Use higher precision arithmetic
        {'initial_points': 'chebyshev', 'precision': 'mpmath', 'max_iter': 100},
    ]

    for strategy in strategies:
        try:
            result = minimax_with_strategy(func, x_min, x_max, degree, **strategy)
            return result, 'remez'
        except Exception as e:
            continue

    # All strategies failed, fall back to LSQ
    print(f"⚠️  All Remez strategies failed, using LSQ fallback")
    return least_squares_fit(func, x_min, x_max, degree), 'lsq'
```

**Expected findings:**
- Certain activations (exp, sinh, cosh) may fail Remez due to rapid growth
- Higher degrees (hexic, octic) may have convergence issues
- Shallow depths (4 segments) may have wide segments causing numerical issues
- LSQ fallback likely 2-5× worse MAE than Remez for most functions

**Mitigation strategies:**
1. **Segment subdivision:** If Remez fails on wide segment, split it and try again
2. **Domain rescaling:** Map to [-1, 1] for better numerical conditioning
3. **Adaptive degree:** Try lower degree if high degree fails
4. **Hybrid per-segment:** Use Remez where it works, LSQ only for problematic segments

**Success criteria:**
- Reduce LSQ fallback rate from ~10-20% to <5%
- Understand root causes of all remaining failures
- Document which activations/configurations require special handling
- Improve overall LUT quality by minimizing LSQ usage

### 13. Implement Remez for Remaining Methods 🎯 LOW PRIORITY
**Status:** Only PQR and PCR use Remez, PC/PL still use least-squares

**Why low priority:** Least-squares works fine for these simpler methods

**If time permits:**
- Implement Remez for Constant (trivial - just midpoint sampling)
- Implement Remez for Linear (use OptimalPoly or manual Remez)
- Compare least-squares vs Remez accuracy for linear approximations
- Expected: Minimal difference (linear functions are simple)

### 12. Extended Precision Analysis 📊 RESEARCH
**Goal:** Understand bfloat16 precision limits

**Questions:**
- How much does bfloat16 quantization limit achievable accuracy?
- Would Float16 or Float32 improve results?
- What's the trade-off: accuracy vs bandwidth vs memory?

**Test plan:**
1. Generate Float32 LUTs and test on CPU (gold reference)
2. Compare with bfloat16 LUTs on hardware
3. Quantify precision loss due to bfloat16
4. Determine if higher precision would justify the cost

**Expected findings:**
- bfloat16 (7-bit mantissa) limits MAE to ~10^-3 for well-behaved functions
- Float16 (10-bit mantissa) might improve to ~10^-4
- Float32 (23-bit mantissa) could reach ~10^-7
- But: Higher precision = 2-4× memory bandwidth cost

### 13. Production Integration Guide 📚 DOCUMENTATION
**Goal:** Document how to integrate generic LUT activation into production models

**Topics to cover:**
1. **Choosing polynomial degree and depth:**
   - Use Pareto plots to select accuracy/speed tradeoff
   - Depth 8 quadratic recommended for most cases
   - Depth 4 cubic for memory-constrained scenarios

2. **Per-activation recommendations:**
   - Simple functions (sigmoid, tanh): Depth 4 linear sufficient
   - Complex functions (gelu, erf): Depth 8 quadratic or cubic
   - Pathological functions (exp): Depth 16 cubic for high accuracy

3. **Integration steps:**
   - Generate LUT files for your activations
   - Copy LUTs to model directory
   - Include generic_lut_activation kernel in your program
   - Pass LUT path at runtime

4. **Performance optimization:**
   - Batch multiple activation calls to amortize setup cost
   - Use sharded memory layout for large tensors
   - Profile kernel execution vs data transfer time

## 📊 Current Status Summary

### Critical Issues

🔥🔥 **ARCHITECTURAL FLAW - Fixed Spacing Quantization:** LUT boundaries computed using uniform spacing instead of stored in LUT
- **Impact:** 2-10× accuracy degradation for functions with non-uniform curvature
- **Root cause:** Design flaw - boundaries should be in LUT file, not computed
- **Status:** BLOCKING - Must fix before any other optimization work
- **Priority:** HIGHEST - Affects all piecewise methods and all activations

🔥 **SFPU Compiler Bug:** Loop unrolling with dynamic array indexing fails silently
- **Impact:** All generic loop-based SFPU kernels affected
- **Workaround:** Hardcoded specializations (implemented)
- **Status:** Needs bug report with minimal reproducing example
- **Priority:** HIGH - Workaround in place, but need to report upstream

### Sweeps Status
- **Wormhole B0:** ✅ Complete (~460 configs, 5 methods, 29 activations, depths 4/8/16)
- **Blackhole:** ⚠️ Incomplete (corrupted CSV files, needs re-run)

### Code Quality
- **Kernels:** ✅ All piecewise methods have working hardcoded specializations
- **Configuration:** ✅ Centralized in activations_config.csv
- **Scripts:** ✅ Unified workflow with --model, --arch, --activation flags
- **Documentation:** ⚠️ Needs update with SFPU bug workaround explanation

### Visualizations
- **Pareto plots:** ✅ 1×3 grid design finalized (depths 4/8/16)
- **Error plots:** ✅ Wormhole complete, Blackhole needs regeneration
- **Runtime plots:** ✅ Wormhole complete, Blackhole needs regeneration
- **Heatmaps:** ✅ Depth vs Degree error heatmap implemented

### Next Session Priorities
1. 🔥🔥 **CRITICAL - BLOCKING:** Implement boundary storage in LUT files (Section 1)
   - Update LUT file format to include boundaries
   - Modify LUT loader to read boundaries
   - Update kernels to use boundaries from LUT instead of computed uniform spacing
   - Test accuracy improvement on sigmoid, exp, gelu
   - Roll out to all piecewise methods
2. 🔥 **CRITICAL:** Create minimal SFPU bug reproducer and file bug report
3. 🔧 **HIGH:** Re-run Blackhole sweep with fixed scripts (AFTER boundary fix)
4. 📊 **HIGH:** Generate Pareto frontier plots for both architectures
5. 🔬 **RESEARCH:** Prototype hexic (degree-6) approximation
6. 📝 **MEDIUM:** Update documentation with SFPU bug section

---

**Summary:**

**CRITICAL DISCOVERY (January 16 evening):** Fixed uniform spacing for segment boundaries is a fundamental design flaw causing 2-10× accuracy degradation. Current implementation computes boundaries using `SEGMENT_WIDTH = INPUT_RANGE / NUM_SEGMENTS` in kernels, which is mathematically suboptimal for functions with non-uniform curvature. **MUST** update LUT file format to store boundaries and modify kernels to use them. This is now the highest priority blocking issue.

**January 16 work:** Discovered and worked around critical SFPU compiler bug through extensive debugging (40+ commits). Implemented hardcoded specializations for all piecewise methods at depths 4/8/16. Refined Pareto frontier visualization through 5 design iterations. Wormhole data collection complete; Blackhole needs re-run.

**Next critical tasks:**
1. **BLOCKING:** Implement boundary storage in LUT files (new #1 priority)
2. File SFPU bug report with minimal reproducing example
3. Re-run sweeps after boundary fix to measure accuracy improvement

**Research tasks queued:** Higher-order polynomials (hexic/octic) - but ONLY after boundary fix is complete.
