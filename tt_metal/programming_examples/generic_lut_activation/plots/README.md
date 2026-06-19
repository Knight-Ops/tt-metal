# Plotting Scripts

Visualization tools for analyzing activation function approximations.

## Active Scripts

### Error Analysis
- **plot_theoretical_error.py** - Theoretical error analysis from .LUT files
- **compare_bf16_fp32_error.py** - Compare BF16 vs FP32 precision effects
- **plot_error_vs_runtime.py** - Error vs runtime tradeoff analysis
- **plot_degree_vs_mae_grid.py** - Degree vs MAE grid visualization

### Activation Visualization
- **plot_activation.py** - General activation function plotting
- **plot_activation_comparison.py** - Compare multiple activations
- **plot_breakpoint_curves.py** - Tufte-style visualization of polynomial segments from CSV ✓ standalone

### Optimal Degree Analysis
- **plot_optimal_degree_selection.py** - Visualize degree selection per segment
- **plot_optimal_degree_summary.py** - Summary of optimal degree results
- **plot_degree_selection_patterns.py** - Patterns in degree selection

### Performance Analysis
- **plot_runtime_comparison.py** - Runtime comparison across configurations
- **plot_pareto_frontier.py** - Error/memory Pareto frontier

## Deprecated Scripts (plots/deprecated/)

Moved to `deprecated/` as they reference obsolete workflows:

- **compare_sollya_optimalpoly.py** - Compared OptimalPoly vs Sollya (OptimalPoly removed)
- **compare_uniform_vs_adaptive.py** - Old adaptive segmentation workflow
- **plot_adaptive_comparison_comprehensive.py** - Old adaptive comparison
- **plot_boundary_optimization.py** - Iterative boundary optimization (superseded by Chebyshev DP)
- **compare_segmentation.py** - Old curvature-based segmentation

These scripts are kept for reference but are no longer maintained with the new workflow.

## New Workflow

The current polynomial fitting workflow uses:
- **Chebyshev DP splitting** for optimal boundary placement (~/tt-polynomial-fitter/)
- **Sollya fpminimax** for precision-aware fitting
- **Standalone coefficient generation** (no activations_config.dat dependency)

For new visualizations of the standalone workflow, see `~/tt-polynomial-fitter/`.

## Dependencies

Most active scripts require:
- `ground_truth` module (activation reference implementations)
- Hardware benchmark CSVs in `data/`
- .LUT files in `luts/`

## Usage Examples

```bash
# Plot theoretical error for sigmoid
python3 plots/plot_theoretical_error.py --activation sigmoid

# Compare BF16 vs FP32
python3 plots/compare_bf16_fp32_error.py

# Visualize optimal degree selection
python3 plots/plot_optimal_degree_selection.py --activation tanh

# Error vs runtime tradeoff
python3 plots/plot_error_vs_runtime.py

# Visualize polynomial segments from CSV (new workflow)
python3 plots/plot_breakpoint_curves.py coefficients/sigmoid_bf16_16_3_chebyshev.csv
python3 plots/plot_breakpoint_curves.py coefficients/tanh_fp32_32_5_fpminimax.csv --output plots/output/tanh.png
```
