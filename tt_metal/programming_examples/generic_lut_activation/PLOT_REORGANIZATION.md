# Plot Directory Reorganization

## Summary

All plotting scripts have been updated to output plots to architecture-specific directories within the `plots/` folder:
- `plots/wormhole/` - Wormhole architecture plots
- `plots/blackhole/` - Blackhole architecture plots
- `plots/output/` - Default output for scripts without architecture specified

## Updated Scripts

### Scripts with existing `--arch-prefix` support (Updated paths):

1. **plot_degree_vs_mae_grid.py**
   - Old output: `{arch_prefix}_plots_error/`
   - New output: `plots/{arch_prefix}/`
   - CSV paths: Updated to use `data/` directory

2. **plot_error_vs_runtime.py**
   - Old output: `{arch_prefix}_plots_error_{segmentation}/` and `{arch_prefix}_plots_error_comparison/`
   - New output: `plots/{arch_prefix}/{segmentation}/` and `plots/{arch_prefix}/comparison/`
   - CSV paths: Updated to use `data/` directory

3. **plot_pareto_frontier.py**
   - Old output: `{arch_prefix}_plots_pareto/`
   - New output: `plots/{arch_prefix}/`
   - CSV paths: Updated to use `data/` directory

4. **plot_runtime_comparison.py**
   - Old output: `{arch_prefix}_plots_runtime/`
   - New output: `plots/{arch_prefix}/`
   - CSV paths: Updated to use `data/` directory

### Scripts updated with new `--arch` parameter:

5. **compare_segmentation.py**
   - Old output: `plots_breakpoint_analysis/`
   - New output: `plots/{arch}/` (or `plots/output/` if no arch specified)
   - New parameter: `--arch <wormhole|blackhole>`

6. **plot_adaptive_comparison_comprehensive.py**
   - Old output: `plots_adaptive_comprehensive/`
   - New output: `plots/{arch}/` (or `plots/output/` if no arch specified)
   - New parameter: `--arch <wormhole|blackhole>`
   - Existing `--output-dir` parameter overrides `--arch`

7. **plot_theoretical_error.py**
   - Old output: `plots_error_theoretical/`
   - New output: `plots/{arch}/` (or `plots/output/` if no arch specified)
   - New parameter: `--arch <wormhole|blackhole>`

8. **plot_activation.py**
   - Already outputs to `plots_output/` - kept as is for now
   - Could be updated to use `--arch` parameter if needed

## Directory Structure

```
plots/
├── wormhole/              # Wormhole architecture plots
│   ├── uniform/           # Uniform segmentation plots
│   ├── adaptive/          # Adaptive segmentation plots
│   ├── comparison/        # Comparison plots
│   └── *.png              # Various plot files
├── blackhole/             # Blackhole architecture plots
│   ├── uniform/           # Uniform segmentation plots
│   ├── adaptive/          # Adaptive segmentation plots
│   ├── comparison/        # Comparison plots
│   └── *.png              # Various plot files
└── output/                # Legacy plot collections and default output
    ├── plots_*/           # Moved existing plot directories here
    └── *.png              # Default output plots
```

## Usage Examples

### With architecture-specific output:
```bash
# Generate plots for Wormhole
python3 plots/plot_pareto_frontier.py --arch-prefix wormhole
python3 plots/plot_degree_vs_mae_grid.py --arch-prefix wormhole
python3 plots/plot_error_vs_runtime.py --arch-prefix wormhole
python3 plots/plot_runtime_comparison.py --arch-prefix wormhole

# Generate plots for Blackhole
python3 plots/plot_pareto_frontier.py --arch-prefix blackhole
python3 plots/plot_theoretical_error.py --arch blackhole --activation sigmoid
python3 plots/compare_segmentation.py --arch blackhole --activation gelu
```

### Without architecture (uses plots/output/):
```bash
python3 plots/plot_theoretical_error.py --activation sigmoid
python3 plots/compare_segmentation.py --activation gelu
```

## Data Directory Integration

All scripts now read CSV result files from the `data/` directory:
- `data/wormhole_*.csv`
- `data/blackhole_*.csv`

This ensures consistency with the data organization established by `pull_results.sh`.
