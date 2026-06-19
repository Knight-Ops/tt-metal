# Depth/Degree Sweep Scripts

Two complementary scripts for measuring runtime trends across polynomial degrees and tile shapes.

## Scripts

### 1. `sweep_depth_degree_simple.sh` - Device Profiler (Tracy-like)
- **Measures**: Pure SFPU compute time (µs)
- **Overhead**: ~10-20% from device profiler
- **Accuracy**: Isolates compute from data movement
- **Use for**: Analyzing SFPU computational complexity

### 2. `sweep_depth_degree_host.sh` - Host Timing Only
- **Measures**: Total kernel execution time (ms)
- **Overhead**: Minimal (no profiler)
- **Accuracy**: Includes data movement + compute
- **Use for**: End-to-end performance, CI/CD benchmarks

## Features

✅ **Both FP32 and BF16 precisions** tested
✅ **5 tile shapes** (1, 8, 256, 1600, 3200 tiles)
✅ **5 polynomial degrees** (1, 2, 4, 6, 8)
✅ **2D grid results** showing FP32 vs BF16 comparison
✅ **Tile scaling analysis** showing compute trends

## Usage

### Run Profiler Sweep (SFPU timing)
```bash
./sweep_depth_degree_simple.sh
```

**Output**: `depth_degree_profiler_results/depth_degree_profiler_summary.csv`

### Run Host Timing Sweep (end-to-end)
```bash
./sweep_depth_degree_host.sh
```

**Output**: `depth_degree_host_results/depth_degree_host_summary.csv`

### View Results
```bash
# View profiler results
./show_depth_degree_results.sh

# View host timing results
./show_depth_degree_results.sh depth_degree_host_results/depth_degree_host_summary.csv
```

## Example Output

```
================================================================================
2D GRID: SFPU Compute Time (µs) @ 256 tiles
================================================================================

Degree               FP32           BF16          Ratio
-------------------------------------------------------
p1_s16           29.04µs       28.33µs        0.976×
p2_s16           38.59µs       38.23µs        0.991×
p4_s16           57.53µs       57.33µs        0.997×
p6_s16           81.42µs       81.67µs        1.003×
p8_s16           111.9µs      112.04µs        1.001×
```

## Key Findings

### FP32 vs BF16 Compute Time
- **Nearly identical** (~1.0× ratio)
- Precision affects accuracy, not compute time
- Both use same SFPU operations

### Degree Scaling (256 tiles)
- **Linear scaling**: ~13-14 µs per degree increase
- Degree 1: 29 µs
- Degree 8: 112 µs (3.9× increase)

### Tile Scaling (Degree 6, FP32)
- **Near-linear** with tile count
- 1 tile: 35.21 µs
- 3200 tiles: 1015.35 µs (28.8× increase)
- Fixed overhead: 1 and 8 tiles show identical timing

## CSV Format

### Profiler CSV
```csv
degree,segments,precision,shape,tiles,profiler_compute_us,host_kernel_exec_ms,profiler_runs,host_runs
6,16,fp32,256_tiles,256,81.42,8.361109,3,3
```

### Host Timing CSV
```csv
degree,segments,precision,shape,tiles,host_kernel_exec_ms,runs
6,16,fp32,256_tiles,256,8.361109,3
```

## Extending

### Add More Degrees
Edit `TEST_CONFIGS` array in the scripts:

```bash
TEST_CONFIGS=(
    # ... existing configs ...
    "10:16:fp32:gelu_fp32_16_10_curvature_any.csv"
    "12:16:fp32:gelu_fp32_16_12_curvature_any.csv"
)
```

**Note**: Requires corresponding CSV coefficient files and binaries.

### Add More Tile Shapes
Edit `TEST_SHAPES` array:

```bash
TEST_SHAPES=(
    # ... existing shapes ...
    "custom_shape:1024:256"  # 1024×256 (8192 tiles)
)
```

### Adjust Run Count
Modify `NUM_RUNS` variable (default: 3):

```bash
NUM_RUNS=5  # More runs = better statistical confidence
```

## Analysis Examples

### Plot Degree Scaling
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('depth_degree_profiler_results/depth_degree_profiler_summary.csv')
df_256_fp32 = df[(df['tiles'] == 256) & (df['precision'] == 'fp32')]

plt.plot(df_256_fp32['degree'], df_256_fp32['profiler_compute_us'], 'o-')
plt.xlabel('Polynomial Degree')
plt.ylabel('SFPU Compute Time (µs)')
plt.title('SFPU Compute Scaling vs Degree (256 tiles, FP32)')
plt.grid(True)
plt.show()
```

### Compare FP32 vs BF16
```python
df_256 = df[df['tiles'] == 256]
pivot = df_256.pivot(index='degree', columns='precision', values='profiler_compute_us')

pivot.plot(kind='bar')
plt.ylabel('SFPU Compute Time (µs)')
plt.title('FP32 vs BF16 Compute Time (256 tiles)')
plt.legend(title='Precision')
plt.show()
```

### Tile Scaling Analysis
```python
df_deg6_fp32 = df[(df['degree'] == 6) & (df['precision'] == 'fp32')]

plt.plot(df_deg6_fp32['tiles'], df_deg6_fp32['profiler_compute_us'], 'o-')
plt.xlabel('Number of Tiles')
plt.ylabel('SFPU Compute Time (µs)')
plt.xscale('log')
plt.yscale('log')
plt.title('SFPU Compute Scaling vs Tile Count (Degree 6, FP32)')
plt.grid(True)
plt.show()
```

## Performance Insights

### Why 1 tile and 8 tiles have identical timing?
- **Fixed dispatch overhead** dominates for small tile counts
- Likely kernel launch + setup costs (~35 µs baseline)
- Once you exceed this baseline, scaling becomes linear

### Why BF16 ≈ FP32 compute time?
- **Same SFPU operations** regardless of precision
- Precision only affects:
  - Numerical accuracy
  - Memory bandwidth (slightly)
- SFPU ALU operates at same speed for both

### Linear degree scaling
- Each additional polynomial degree adds ~13-14 µs
- **Constant per-coefficient cost** due to Horner's method
- No branching or conditional overhead

## Comparison with Original Scripts

| Feature | Original `sweep_host_timing.sh` | New `sweep_depth_degree_*.sh` |
|---------|--------------------------------|------------------------------|
| **Tile shapes** | Single (256) | 5 shapes (1-3200 tiles) |
| **Precisions** | One at a time | Both FP32 & BF16 |
| **Output** | Amp 1× vs 100× | Tile scaling |
| **Display** | CSV only | 2D grid + CSV |
| **Use case** | Amplification validation | Scaling analysis |

## Files Generated

```
depth_degree_profiler_results/
├── depth_degree_profiler_summary.csv       # Main results
├── p1_s16_fp32_single_tile/
│   ├── profiler_run1/.logs/profile_log_device.csv
│   ├── stdout_run1.txt
│   └── stderr_run1.txt
├── p1_s16_fp32_8_tiles/
│   └── ...
└── ... (50 test directories total)

depth_degree_host_results/
├── depth_degree_host_summary.csv           # Main results
├── p1_s16_fp32_single_tile/
│   ├── stdout_run1.txt
│   └── stderr_run1.txt
└── ... (50 test directories total)
```

## Troubleshooting

### Script hangs during CSV discovery
- **Cause**: Too many CSV files in `coefficients.bak` directory (5814 files)
- **Solution**: Scripts now use hardcoded config list (degrees 1-8)

### "Error: --precision is required"
- **Cause**: Missing precision parameter
- **Solution**: Fixed in current scripts (v2)

### Regex matching fails (empty degree/segments)
- **Cause**: zsh vs bash shell incompatibility
- **Solution**: Run with explicit `bash` command or use hardcoded configs

### Tests timeout
- **Symptom**: Large tile counts (3200) timeout after 60s
- **Solution**: Increase timeout in script (line: `timeout 60`)

## Future Enhancements

- [ ] Parallel execution across configs
- [ ] Automatic plot generation
- [ ] Memory bandwidth analysis
- [ ] Support for rational approximations (n/d format)
- [ ] JSON output format
- [ ] Integration with CI/CD
