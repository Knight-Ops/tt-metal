# sweep_all.sh Refactor Summary

## Overview

Refactored `sweep_all.sh` to be consistent with `sweep_best.sh` while exploring ALL combinations instead of just the best configurations.

## Key Changes

### Before (Old sweep_all.sh)
- **Orchestrator script**: Called other sweep scripts (sweep_native_sfpu.sh, sweep_piecewise.sh) with different parameters
- **Remote execution management**: Handled SSH to remote servers and tmux session management
- **High-level coordination**: Didn't directly run tests, delegated to sub-scripts

### After (New sweep_all.sh)
- **Direct test execution**: Follows the same structure as `sweep_best.sh`, running tests directly
- **Programmatic combination generation**: Generates all combinations from parameter space:
  - Activations (from activations_config_loader.sh)
  - Precisions: bf16, fp32
  - Depths: 4, 8, 16, 32
  - Degrees: 1 (linear), 2 (quadratic), 3 (cubic), 6 (hexic), 8 (octic)
  - Segmentation: uniform, adaptive
- **Consistent with sweep_best.sh**: Same CSV format, timing collection, accuracy metrics, result tracking

## Architecture Comparison

### sweep_best.sh (reference)
```bash
Read best.csv → For each config → Run test → Save results
```

### sweep_all.sh (old)
```bash
Orchestrate → Call sweep_native_sfpu.sh
           → Call sweep_piecewise.sh --degree linear
           → Call sweep_piecewise.sh --degree quadratic
           → etc.
```

### sweep_all.sh (new)
```bash
Generate all combinations → For each config → Run test → Save results
```

## Features

### Filters (same as sweep_best.sh)
- `--activation <name>`: Test single activation
- `--precision <type>`: bf16 or fp32 (default: both)
- `--depth <depths>`: Comma-separated depths (default: all)
- `--degree <type>`: linear|quadratic|cubic|hexic|octic (default: all)
- `--segmentation <type>`: uniform, adaptive, or curvature (default: all)
- `--compare`: Show detailed comparison statistics

### Output Format
Saves to `data/${PLATFORM_PREFIX}all_results.csv` with columns:
```
activation,precision,depth,degree,segmentation,fitting,tile_count,
device_init_ms,program_creation_ms,buffer_alloc_ms,data_prep_ms,
host_to_device_ms,kernel_creation_ms,kernel_exec_ms,device_to_host_ms,
runtime_mean_ms,runtime_min_ms,runtime_stddev_ms,mae,rmse,max_error,
mean_rel_error,status
```

### Summary Statistics (--compare flag)
Groups results by:
- **Degree**: Shows median MAE and runtime per polynomial degree
- **Segmentation**: Compares uniform vs adaptive segmentation
- **Depth**: Shows impact of LUT depth on accuracy and performance

## Configuration Space

Full sweep explores (without filters):
- ~30 activations × 2 precisions × 4 depths × 5 degrees × 3 segmentations
- **= ~3,600 configurations total**

⚠️ **Warning**: Full sweep is very large! Use filters to reduce scope.

## Usage Examples

### Comprehensive Sweep (Warning: Very Long!)
```bash
./sweep_all.sh
```

### Targeted Sweeps
```bash
# Single activation, all configs
./sweep_all.sh --activation sigmoid

# Single degree, all activations
./sweep_all.sh --degree cubic

# Specific depth only
./sweep_all.sh --depth 32

# Combined filters
./sweep_all.sh --activation tanh --degree cubic --depth 32

# With summary statistics
./sweep_all.sh --activation sigmoid --compare
```

### Precision and Segmentation Filters
```bash
# BF16 only, all other parameters
./sweep_all.sh --precision bf16

# Uniform segmentation only
./sweep_all.sh --segmentation uniform

# Curvature segmentation only
./sweep_all.sh --segmentation curvature

# BF16 uniform configs only
./sweep_all.sh --precision bf16 --segmentation uniform
```

## Consistency with sweep_best.sh

Both scripts now share:
1. **Same initialization logic**: Repository detection, architecture detection, directory setup
2. **Same result tracking**: CSV headers, timing collection, accuracy metrics
3. **Same Python accuracy computation**: Identical compute_reference() implementations
4. **Same output format**: Consistent CSV structure for downstream analysis
5. **Same execution pattern**: Loop → Test → Collect timing → Compute accuracy → Save results

## Differences from sweep_best.sh

| Feature | sweep_best.sh | sweep_all.sh |
|---------|--------------|--------------|
| **Input source** | Reads best.csv | Generates combinations programmatically |
| **Scope** | 1 config per activation (~30 tests) | All combinations (~3,600 tests) |
| **Output file** | `best_results.csv` | `all_results.csv` |
| **Use case** | Validate best theoretical configs | Explore full parameter space |
| **Runtime** | Fast (~5-10 min) | Very long (~12-18 hours full sweep) |
| **Summary stats** | Theoretical vs hardware comparison | Group by degree/segmentation/depth |

## Binary and LUT Naming

### Binary Pattern
```
programming_examples_generic_lut_activation_{degree_name}_{degree_num}_{depth}
```
Example: `programming_examples_generic_lut_activation_cubic_3_32`

### LUT Pattern
```
{activation}_{precision}_{depth}_{degree_num}_{segmentation}_{fitting}.lut
```
Example: `sigmoid_bf16_32_3_adaptive_remez.lut`

### Fitting Method Selection
- **Linear (degree 1)**: Uses `lsq` (least squares) fitting
- **All others**: Uses `remez` (minimax) fitting

### Segmentation Types
- **uniform**: Equal-width segments across the input range
- **adaptive**: Adaptive segmentation based on function characteristics
- **curvature**: Curvature-based segmentation that places more breakpoints in high-curvature regions

## Migration Notes

The old orchestrator functionality of `sweep_all.sh` (remote SSH, tmux management) has been removed. If needed, that functionality can be:
1. Added to a separate `orchestrate_sweeps.sh` script
2. Integrated into the deployment/CI system
3. Handled by the user manually for distributed execution

The new design focuses on being a direct, single-server sweep script that's consistent with `sweep_best.sh`.

## Testing

Verified with:
```bash
# Syntax check
bash -n sweep_all.sh

# Help display
GROUND_TRUTH="$HOME/workspace/tt-polynomial-fitter" ./sweep_all.sh --help

# Dry run (will skip tests if LUTs/binaries not found)
GROUND_TRUTH="$HOME/workspace/tt-polynomial-fitter" ./sweep_all.sh --activation sigmoid --depth 32
```

## Benefits

1. **Consistency**: Same structure as sweep_best.sh makes both scripts easier to understand and maintain
2. **Flexibility**: Fine-grained filters allow targeted exploration of parameter space
3. **Comprehensive**: Can explore every combination to find optimal configs
4. **Analysis-friendly**: Same CSV format enables direct comparison with best.csv results
5. **Statistics**: Built-in summary statistics with --compare flag

## Future Enhancements

Potential additions:
- Progress bar for long sweeps
- Checkpoint/resume functionality
- Parallel execution across multiple devices
- Real-time plotting of results
- Integration with polynomial fitter to compare theoretical vs hardware
