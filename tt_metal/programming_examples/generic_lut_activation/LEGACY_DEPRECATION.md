# Legacy Code Deprecation

## Removed

- ✅ `generate_luts_legacy.sh` - Replaced by `generate_coefficients.sh` + `generate_luts.sh`
- ✅ `segmentation/generate_adaptive_segments.py` - Replaced by Chebyshev DP optimal boundaries
- ✅ `luts/compare_with_production_libs.py` - Static benchmark comparison (removed)
- ✅ `luts/compare_all_approaches.py` - Runtime benchmark with tt-metal results (removed)
- ✅ `luts/sollya_expressions.py` - Duplicate (use segmentation/sollya_expressions.py)

## Deprecated (Still Present for Sweep Scripts)

The following files are deprecated but kept for backward compatibility with sweep/test scripts:

### Old Per-Degree Generators (luts/)
- `generate_piecewise_constant_luts.py`
- `generate_piecewise_linear_luts.py`
- `generate_piecewise_remez_luts.py`

**Replacement:** Use unified workflow:
```bash
./generate_coefficients.sh --activation sigmoid --degree 3
./generate_luts.sh
```

### Still Used By (TODO: Migrate)
- `sweep_piecewise_constant.sh`
- `sweep_piecewise.sh`
- `luts/test_unified_remez.sh`
- `plots/plot_boundary_optimization.py`
- `plots/compare_sollya_optimalpoly.py`
- `compare_fixed_vs_optimal_degree.py`

## Migration Path

Old per-degree generators will be removed after sweep scripts are migrated to new workflow.

## New Workflow

**Polynomial Fitting:** `~/tt-polynomial-fitter/`
- Core algorithms: chebyshev_splitting, sollya_fpminimax, optimal_degree_search
- Standalone, no project dependencies

**LUT Generation:**
```bash
# Step 1: Generate coefficients CSV
./generate_coefficients.sh --activation sigmoid --precision bf16

# Step 2: Convert to binary LUT
./generate_luts.sh --input-dir coefficients/
```

**File Naming:** `{activation}_{precision}_{segments}_{degree}_{method}.[csv|lut]`
