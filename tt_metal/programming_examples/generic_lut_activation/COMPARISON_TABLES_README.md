# Sweep Comparison Tables

The `sweep_all.sh` script now generates **two separate comparison tables** when run with the `--compare` flag:

## 1. MAE Comparison Table

Shows Mean Absolute Error (MAE) across different configurations:

- **Theory Config / Theory MAE**: Theoretical best configuration from tt-polynomial-fitter
- **Best Config / Best MAE**: Best configuration from sweep_best.sh
- **Empir Config / Empir MAE**: Empirically best configuration from all sweep combinations
- **SFPU MAE**: Native SFPU baseline (if available)

## 2. Max Error Comparison Table

Shows Maximum Error across different configurations:

- **Theory Config / Theory MaxErr**: Theoretical best configuration
- **Best Config / Best MaxErr**: Best configuration from sweep_best.sh
- **Empir Config / Empir MaxErr**: Empirically best configuration (minimizing max error)
- **SFPU MaxErr**: Native SFPU baseline (if available)

## Key Features

- **Separate tables**: MAE and MaxErr are shown in separate tables for clarity
- **SFPU baseline**: Includes SFPU metrics when available
- **Indicators**: `*` marks activations where empirical best differs from sweep_best config
- **Summary statistics**: Shows how often theoretical configs match empirical best
- **Comparison plots**: Automatically generates visual comparison plots

## Usage

```bash
# Run sweep and show comparison
./sweep_all.sh --compare

# Skip running tests, just show comparison from existing results
./sweep_all.sh --skip-run --compare

# Filter by activation
./sweep_all.sh --activation sigmoid --compare
```

## Output Files

- **Tables**: Printed to console
- **Plots**: Saved to `plots/${PLATFORM_PREFIX}comparison/`
  - `${PLATFORM_PREFIX}comparison_mae.png`
  - `${PLATFORM_PREFIX}comparison_max_error.png`

## Example Output

```
MAE COMPARISON: Theory vs Best Config vs Empirical Best vs SFPU
┌─────────────┬────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Activation  │Prec│ Theory Config│ Theory MAE   │ Best Config  │ Best MAE     │ Empir Config │ Empir MAE    │ SFPU MAE     │
├─────────────┼────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ sigmoid     │bf16 │     d22 deg2 │     6.98e-04 │     d22 deg2 │     0.001344 │     d32 deg2 │     0.001417 │          N/A │*
│ sigmoid     │fp32 │     d32 deg6 │     3.00e-08 │     d32 deg6 │     2.31e-08 │     d32 deg6 │     2.31e-08 │          N/A │ 
...

MAX ERROR COMPARISON: Theory vs Best Config vs Empirical Best vs SFPU
┌─────────────┬────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Activation  │Prec│ Theory Config│ Theory MaxErr│ Best Config  │ Best MaxErr  │ Empir Config │ Empir MaxErr │ SFPU MaxErr  │
├─────────────┼────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ sigmoid     │bf16 │     d22 deg2 │       0.0041 │     d22 deg2 │       0.0055 │     d32 deg1 │       0.0066 │          N/A │*
│ sigmoid     │fp32 │     d32 deg6 │     2.05e-07 │     d32 deg6 │     2.07e-07 │     d32 deg8 │     1.85e-07 │          N/A │*
...
```

## Notes

- SFPU data is loaded from `data/${PLATFORM_PREFIX}native_sfpu_results.csv`
- Empirical best for MAE may differ from empirical best for MaxErr (different configurations optimize different metrics)
- `N/A` indicates data not available for that configuration
