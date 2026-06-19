# Depth/Degree Profiler Sweep

## Overview

`sweep_depth_degree_profiler.sh` is a modified version of `sweep_host_timing.sh` that uses device profiler (Tracy-like) to measure accurate SFPU compute timing instead of just host timing.

## Motivation

The original `sweep_host_timing.sh` measured only host-side `TIMING_KERNEL_EXECUTION`, which includes data movement overhead. For analyzing SFPU compute trends across different polynomial degrees and segment counts, we need **pure compute timing** isolated from data movement.

## Key Changes from sweep_host_timing.sh

### 1. Profiler Integration (from sweep_best.sh)
- **Added**: Device profiler mode (`TT_METAL_DEVICE_PROFILER=1`)
- **Added**: Profiler helper functions (`profiler_helpers.sh`)
- **Added**: Extract SFPU compute time from `profile_log_device.csv`

### 2. Multiple Runs per Configuration
- **Changed**: Now runs each test **5 times** (configurable via `NUM_RUNS`)
- **Changed**: Reports **minimum timing** across runs (reduces variance)
- **Added**: Tracks number of successful runs for each config

### 3. Dual Timing Columns
- **Profiler timing**: SFPU compute time in **microseconds** (µs)
- **Host timing**: Total kernel execution in **milliseconds** (ms)
- **Ratio**: Shows amplification factor (amp=100 / amp=1) for both metrics

### 4. Enhanced CSV Output

**New CSV format**:
```csv
degree,segments,amp,profiler_compute_us,host_kernel_exec_ms,profiler_runs,host_runs
```

**Example row**:
```
6,16,1,1114.23,1.45,5,5
6,16,100,111423.45,144.83,5,5
```

This shows:
- Degree 6, 16 segments, amp=1: 1.114ms compute time (from 5 successful runs)
- Degree 6, 16 segments, amp=100: 111.4ms compute time (from 5 successful runs)
- Amplification ratio: 100× (as expected)

## Usage

```bash
cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation

# Run the sweep (tests all available p{degree}_s{segments} binaries)
./sweep_depth_degree_profiler.sh
```

## Output Structure

```
depth_degree_profiler_results/
├── depth_degree_profiler_summary.csv          # Main results CSV
├── p1_s1_amp1/
│   ├── profiler_run1/
│   │   └── .logs/profile_log_device.csv       # Profiler data (run 1)
│   ├── profiler_run2/
│   │   └── .logs/profile_log_device.csv       # Profiler data (run 2)
│   ...
│   ├── stdout_run1.txt                        # Host timing (run 1)
│   └── stderr_run1.txt
├── p1_s1_amp100/
│   ├── profiler_run1/
│   │   └── .logs/profile_log_device.csv
│   ...
├── p2_s2_amp1/
├── p2_s2_amp100/
...
```

## Console Output Example

```
[INFO] Testing p6_s16...
  amp=1:   Prof=1114.23µs  Host=1.45ms
  amp=100: Prof=111423.45µs  Host=144.83ms
  Profiler Ratio: 100.00×  Host Ratio: 99.88×

[INFO] Testing p8_s16...
  amp=1:   Prof=1523.67µs  Host=1.89ms
  amp=100: Prof=152367.12µs  Host=188.92ms
  Profiler Ratio: 100.00×  Host Ratio: 99.99×
```

## Analysis Use Cases

### 1. Compute Scaling Verification
Check that compute amplification is linear (ratio ≈ 100):
```bash
awk -F',' 'NR>1 {if ($3==1) amp1=$4; if ($3==100) print $1,$2,amp1,$4,$4/amp1}' \
    depth_degree_profiler_results/depth_degree_profiler_summary.csv
```

### 2. Degree Complexity Trends
Plot compute time vs polynomial degree:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('depth_degree_profiler_results/depth_degree_profiler_summary.csv')
df_amp100 = df[df['amp'] == 100]

plt.figure(figsize=(10, 6))
for segs in df_amp100['segments'].unique():
    data = df_amp100[df_amp100['segments'] == segs]
    plt.plot(data['degree'], data['profiler_compute_us'], marker='o', label=f'{segs} segments')

plt.xlabel('Polynomial Degree')
plt.ylabel('SFPU Compute Time (µs)')
plt.title('SFPU Compute Time vs Degree (amp=100, 256 tiles)')
plt.legend()
plt.grid(True)
plt.show()
```

### 3. Segment Complexity Trends
Plot compute time vs segment count:
```python
for deg in df_amp100['degree'].unique():
    data = df_amp100[df_amp100['degree'] == deg]
    plt.plot(data['segments'], data['profiler_compute_us'], marker='o', label=f'degree {deg}')
```

## Comparison with Original sweep_host_timing.sh

| Feature | sweep_host_timing.sh | sweep_depth_degree_profiler.sh |
|---------|---------------------|-------------------------------|
| **Timing Source** | Host only (ms) | Profiler (µs) + Host (ms) |
| **Metric** | Total kernel exec | Pure SFPU compute |
| **Runs per Config** | 1 | 5 (min reported) |
| **Overhead** | Low | Higher (profiler) |
| **Accuracy** | Includes data movement | SFPU only |
| **Use Case** | Quick estimates | Detailed analysis |

## Known Limitations

1. **Profiler Overhead**: Device profiler adds ~10-20% timing overhead
2. **Disk I/O**: Each run writes ~1-2MB of profiler data (can fill disk on large sweeps)
3. **Slower Execution**: ~5× slower than host-only timing (5 runs + profiler)
4. **Device Resets**: Timeouts trigger device resets (adds delay)

## Recommendations

### When to Use This Script
- ✅ Analyzing SFPU compute scaling trends
- ✅ Isolating compute from data movement
- ✅ Comparing degree/segment complexity
- ✅ Validating compute amplification

### When to Use Original sweep_host_timing.sh
- ✅ Quick sanity checks
- ✅ Total kernel timing (compute + data movement)
- ✅ Large sweeps (minimize disk usage)
- ✅ CI/CD regression testing

## Future Enhancements

Possible improvements:
1. **Selective profiling**: Only profile amp=100 (skip amp=1 for speed)
2. **Parallel execution**: Run multiple configs in parallel
3. **Memory tracking**: Extract DRAM bandwidth from profiler
4. **Automated plotting**: Generate trend graphs from CSV
5. **Comparison mode**: Compare against theoretical FLOPs
