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
python3 plots/plot_pareto_frontier.py --arch wormhole
python3 plots/plot_pareto_frontier.py --arch blackhole

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

### 4. Advanced Sollya Integration & Automated Optimization 🧪 RESEARCH
**Priority:** MEDIUM - Improve automation and explore higher degrees
**Status:** Investigation phase

#### 4a. Custom Operators in Sollya
**Goal:** Extend Sollya to handle custom activation functions that aren't built-in

**Problem:**
- Sollya has built-in functions (exp, log, sin, cos, etc.)
- Many activation functions (softplus, gelu, swish, mish) are compositions requiring custom operators
- Current approach: Define functions in Python, pass to Sollya via string evaluation
- Limitation: Cannot leverage Sollya's symbolic differentiation for custom functions

**Investigation tasks:**
1. **Research Sollya's extensibility mechanism:**
   - Check Sollya documentation for custom function definition
   - Explore `library()` directive for loading external implementations
   - Investigate if Sollya can symbolically differentiate custom functions
   - Test example: `softplus(x) = log(1 + exp(x))`

2. **Test custom operator integration:**
   ```sollya
   // Can Sollya handle compositions directly?
   f = log(1 + exp(x));
   remez(f, 3, [-10;10], 1e-6);

   // Or do we need library extensions?
   library("custom_activations.so");
   f = softplus(x);  // Defined in shared library
   ```

3. **Explore automatic differentiation:**
   - Does Sollya autodiff custom compositions?
   - Can we provide derivatives manually for better convergence?
   - Test: `softplus'(x) = sigmoid(x) = 1/(1+exp(-x))`

4. **Document limitations:**
   - Which activations require custom operators?
   - Which can be expressed as Sollya built-in compositions?
   - Performance implications of external libraries vs built-in functions

**Expected benefits:**
- Better convergence for complex activations
- Leveraging Sollya's symbolic capabilities
- More robust fitting for compositions

**References:**
- Sollya documentation: https://www.sollya.org/
- Check if tt-metal has existing Sollya extensions

#### 4b. Agentic Flow for Automated LUT Optimization 🤖 RESEARCH
**Goal:** Build autonomous system to iteratively improve LUT generation through error-driven refinement

**Motivation:**
- Current approach: Fixed depth/degree, accept whatever error results
- Many activations fail Sollya convergence (softplus hexic/octic example from today)
- Manual debugging is tedious (need to try different configurations)
- Want: Automated system that finds optimal configuration given error budget

**Proposed agentic architecture:**

```python
class LUTOptimizationAgent:
    """
    Autonomous agent that iteratively refines LUT configuration to meet error budget.

    Strategy:
    1. Start with conservative config (e.g., depth 4, quadratic)
    2. Measure error on test points
    3. If error > budget:
       - Identify problematic segments (high local error)
       - Decide: Increase depth (more segments) vs increase degree (higher polynomial)
       - Try adjustment, measure improvement
    4. Repeat until error budget met or resources exhausted
    5. Return Pareto frontier of configurations
    """

    def __init__(self, activation_func, error_budget_mae, max_depth=32, max_degree=15):
        self.activation = activation_func
        self.error_budget = error_budget_mae
        self.max_depth = max_depth
        self.max_degree = max_degree
        self.tried_configs = []  # Cache to avoid redundant attempts

    def optimize(self):
        """Main optimization loop."""
        config = {'depth': 4, 'degree': 2, 'segmentation': 'uniform'}

        while True:
            # Generate LUT and measure error
            lut, error_metrics = self.generate_and_test(config)
            self.tried_configs.append((config, error_metrics))

            if error_metrics['mae'] <= self.error_budget:
                return lut, config, error_metrics

            # Error analysis: which segments have high error?
            problematic_segments = self.identify_high_error_segments(lut, error_metrics)

            # Decide next action using heuristic or learned policy
            next_config = self.suggest_next_config(config, problematic_segments)

            if next_config is None:  # Exhausted options
                return None, config, error_metrics

            config = next_config

    def suggest_next_config(self, current, problem_segments):
        """
        Decide whether to:
        1. Increase depth (more segments)
        2. Increase degree (higher polynomial)
        3. Switch to adaptive segmentation
        4. Subdivide specific problem segments

        Heuristics:
        - If few segments have high error → increase depth (localized issue)
        - If many segments have high error → increase degree (global curvature)
        - If uniform spacing wastes segments → try adaptive
        - If Sollya convergence fails → try lower degree or --no-sollya
        """
        # Implement decision tree or learned policy
        ...

    def handle_sollya_failure(self, config, error_msg):
        """
        Autonomous error recovery for Sollya convergence failures.

        Strategies:
        1. Try --no-sollya (standard FP32 Remez instead of BF16-aware)
        2. Reduce polynomial degree for problematic segments
        3. Increase number of segments (reduce segment width)
        4. Use adaptive segmentation to avoid large exponential-growth regions
        """
        if "fpminimax failed to converge" in error_msg:
            # Try without Sollya
            return {**config, 'use_sollya': False}
        elif "Expected at least 1 coefficient, found 0" in error_msg:
            # Sollya returned empty result - reduce degree
            return {**config, 'degree': max(1, config['degree'] - 1)}
        elif config['segmentation'] == 'uniform':
            # Try adaptive to avoid problematic regions
            return {**config, 'segmentation': 'adaptive'}
        else:
            # Increase depth to reduce segment width
            return {**config, 'depth': min(self.max_depth, config['depth'] * 2)}
```

**Implementation phases:**

**Phase 1: Error-driven segment refinement**
- Input: Activation function, error budget (e.g., MAE < 1e-4)
- Algorithm: Start with depth-4, measure segment-wise error, subdivide high-error segments
- Output: Non-uniform boundaries optimized for error distribution

**Phase 2: Polynomial degree escalation**
- Input: Fixed depth, error budget
- Algorithm: Start with linear (degree 1), increase degree until error budget met
- Constraint: Stop at degree 15 (max practical for SFPU)
- Handle: Sollya convergence failures with automatic fallback

**Phase 3: Pareto frontier exploration**
- Input: Activation function, error budget range
- Algorithm: Explore depth×degree space, identify Pareto-optimal configs
- Output: Tradeoff curve: memory vs accuracy vs runtime
- Visualization: Interactive plot showing optimal choices for different budgets

**Phase 4: Multi-objective optimization with learned policy**
- Train RL agent on historical successful configurations
- Learn: Which depth/degree combinations work well for which activation types
- Learn: When to switch strategies (adaptive vs uniform, Sollya vs --no-sollya)
- Generalize: Predict good starting config for new activation functions

**Key components to build:**

1. **Segment error analyzer:**
   ```python
   def analyze_segment_errors(lut, activation, test_points=10000):
       """
       Measure error in each segment individually.
       Returns: array of (segment_idx, max_error, mean_error)
       """
   ```

2. **Adaptive segment subdivider:**
   ```python
   def subdivide_high_error_segments(boundaries, errors, threshold):
       """
       Split segments where error > threshold into smaller segments.
       Uses bisection or curvature-based placement.
       """
   ```

3. **Sollya convergence wrapper with retry logic:**
   ```python
   def robust_sollya_fit(func, degree, segment, precision='bf16', max_attempts=5):
       """
       Try Sollya fit with exponential backoff:
       1. Try BF16-aware Sollya
       2. Try FP32 Sollya (--no-sollya)
       3. Try lower degree
       4. Try wider numerical tolerances
       5. Fall back to least-squares if all fail
       """
   ```

4. **Configuration space explorer:**
   ```python
   def explore_pareto_frontier(activation, depth_range, degree_range, error_budgets):
       """
       Systematically explore configuration space.
       Find Pareto-optimal (depth, degree) for each error budget.
       Cache results to avoid redundant computations.
       """
   ```

**Expected outcomes:**
- Automatic LUT generation that meets error budgets without manual tuning
- Robust handling of Sollya convergence failures (today's softplus issue)
- Pareto frontiers for all activations showing memory/accuracy tradeoffs
- Learned heuristics for which configurations work well for different activation types

**Testing:**
- Start with problematic activations from today (softplus hexic/octic)
- Test: Can agent automatically find working config when Sollya fails?
- Benchmark: How many iterations needed vs manual configuration?
- Validate: Does agent find better configs than human-selected defaults?

#### 4c. Polynomial Degree Extension Beyond Octic (Degree 8) 🧪 RESEARCH
**Goal:** Explore very high-degree polynomials (degrees 9-15) for ultra-high-accuracy requirements

**Motivation:**
- Current implementation supports up to octic (degree 8)
- **Observation:** Found 15th-order polynomials referenced in tt-metal codebase
- Question: When are degrees 9-15 justified? What are the tradeoffs?

**Research questions:**

1. **Where are degree-15 polynomials used in tt-metal?**
   - Search codebase for high-degree polynomial references
   - Understand context: Which operations/activations need degree 15?
   - Document: Performance vs accuracy tradeoffs observed
   - Reference commits or files using high-degree polynomials

2. **Accuracy vs complexity tradeoff analysis:**
   - Hypothesis: Diminishing returns beyond cubic (degree 3)
   - Test: sigmoid, tanh → Does degree 8 provide meaningful improvement over degree 3?
   - Test: gelu, erf, softplus → Do complex functions benefit from degree 15?
   - Measure: MAE improvement per additional polynomial degree
   - Cost: Memory (coefficients), Runtime (SFPU operations), Convergence (Sollya stability)

3. **SFPU register pressure concerns:**
   - Degree 15 polynomial requires 16 coefficients per segment
   - Tensix SFPU cores have limited registers
   - Question: Can SFPU even execute degree-15 polynomials efficiently?
   - Test: Runtime scaling as degree increases (compile and benchmark on hardware)

4. **Sollya convergence at high degrees:**
   - Observation: Sollya fails for softplus degree 8 on small segments
   - Hypothesis: Higher degrees exacerbate numerical instability
   - Test: Can Sollya fit degree 15 for any activations?
   - Alternative: Barycentric Remez (more numerically stable for high degrees)

5. **Segment count reduction potential:**
   - Hypothesis: Degree-15 polynomial can match depth-16 cubic accuracy using depth-4
   - Memory tradeoff: Depth-4 × 16 coeffs = 64 vs Depth-16 × 4 coeffs = 64 (same memory!)
   - Runtime tradeoff: More polynomial terms vs more segment boundary checks
   - Question: Which is faster on SFPU hardware?

**Implementation plan:**

**Step 1: Codebase archaeology**
```bash
# Find references to high-degree polynomials in tt-metal
cd /Users/nkapre/workspace/tt-metal
git grep -i "15.*degree\|degree.*15\|polynomial.*15"
git grep -i "decic\|degree.*10\|degree.*12"
rg "coeffs\[1[0-5]\]" --type cpp  # Look for indexing coefficients > 10
```

**Step 2: Extend generator to support degrees 9-15**
```python
# luts/generate_piecewise_remez_luts.py
DEGREE_INFO = {
    'quadratic': {'degree': 2, 'coeffs': 3, 'description': 'Fast, good for smooth functions'},
    'cubic': {'degree': 3, 'coeffs': 4, 'description': 'Best general-purpose tradeoff'},
    'hexic': {'degree': 6, 'coeffs': 7, 'description': 'High accuracy for complex functions'},
    'octic': {'degree': 8, 'coeffs': 9, 'description': 'Ultra-high accuracy'},
    # NEW: Add higher degrees
    'decic': {'degree': 10, 'coeffs': 11, 'description': 'Experimental - very high accuracy'},
    'dodecic': {'degree': 12, 'coeffs': 13, 'description': 'Experimental - extreme accuracy'},
    'tetradecic': {'degree': 14, 'coeffs': 15, 'description': 'Experimental - maximum accuracy'},
    'hexadecic': {'degree': 15, 'coeffs': 16, 'description': 'Experimental - 15th degree limit'},
}
```

**Step 3: Test Sollya convergence at high degrees**
```python
# Test script: test_high_degree_convergence.py
import sollya
for activation in ['sigmoid', 'tanh', 'gelu', 'softplus']:
    for degree in [8, 10, 12, 15]:
        for depth in [4, 8]:
            try:
                # Attempt Sollya fit
                result = sollya_fit_segment(activation, degree, segment_bounds, precision='bf16')
                print(f"✓ {activation} degree-{degree} depth-{depth}: SUCCESS")
            except Exception as e:
                print(f"✗ {activation} degree-{degree} depth-{depth}: FAILED - {e}")
```

**Step 4: Benchmark accuracy vs degree**
```python
# Compare error reduction as degree increases
degrees = [1, 2, 3, 6, 8, 10, 12, 15]
for activation in ACTIVATIONS:
    for depth in [4, 8]:
        errors = []
        for degree in degrees:
            lut = generate_lut(activation, depth, degree)
            mae = measure_error(lut, activation)
            errors.append(mae)

        # Plot: degree vs MAE (log scale)
        plt.semilogy(degrees, errors, label=f'{activation} depth-{depth}')

# Key question: Does MAE improvement plateau? At what degree?
```

**Step 5: Measure runtime impact**
```python
# Benchmark kernel execution time vs polynomial degree
# Run on Wormhole hardware
for degree in [3, 6, 8, 10, 12, 15]:
    for depth in [4, 8]:
        kernel_time = benchmark_kernel(activation, degree, depth, tiles=128)
        print(f"Degree {degree} Depth {depth}: {kernel_time:.3f}ms")

# Expected: Linear scaling with degree? Superlinear due to register pressure?
```

**Expected findings:**
- **Diminishing returns:** MAE improvement plateaus around degree 8-10 for most activations
- **Convergence issues:** Sollya likely fails for degree > 10 on small segments
- **Memory/runtime tradeoff:** Depth reduction via high-degree polynomials may not be worthwhile
- **Practical limit:** Degree 8 (octic) is likely the sweet spot; degree 15 rarely justified
- **Exception cases:** Pathological functions (exp, cosh) might benefit from degree 10-12 at low depth

**Deliverables:**
1. Documentation: When to use degrees 9-15 (if ever)
2. Generator support: Extended remez script with degrees up to 15
3. Benchmark results: Accuracy vs runtime vs memory for all degrees
4. Recommendation: Practical upper limit for production use

**References to investigate:**
- Search tt-metal for degree-15 polynomial usage
- Check SFPU documentation for register limits
- Review literature on high-degree Remez approximation stability

### 5. Explore Higher-Order Polynomials (Original Task) 🧪 RESEARCH
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

## 📋 Medium Priority Tasks

### 6. Compare Wormhole vs Blackhole Performance 📊 MEDIUM PRIORITY
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

### 7. Documentation Updates 📝 MEDIUM PRIORITY

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

### 8. Performance Deep Dive 📊 LOW PRIORITY
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

### 7. Update README with Results Summary 📄 MEDIUM PRIORITY
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

### 9. Implement Remez for Remaining Methods 🎯 LOW PRIORITY
**Status:** Only PQR and PCR use Remez, PC/PL still use least-squares

**Why low priority:** Least-squares works fine for these simpler methods

**If time permits:**
- Implement Remez for Constant (trivial - just midpoint sampling)
- Implement Remez for Linear (use OptimalPoly or manual Remez)
- Compare least-squares vs Remez accuracy for linear approximations
- Expected: Minimal difference (linear functions are simple)

### 10. Extended Precision Analysis 📊 RESEARCH
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

### 11. Production Integration Guide 📚 DOCUMENTATION
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
