# TTNN Composite Operations Integration

This directory contains tools to run composite activation functions using TTNN operations, fully integrated with the existing sweep infrastructure.

## Overview

**Problem:** Some composite activation functions (mish, hardswish, tanhshrink, logsigmoid, logit, hardshrink) are difficult to implement in native SFPU kernels because they require:
- Multiple DST registers
- Binary operations between registers
- Temporary circular buffers

**Solution:** Use TTNN Python API to call these operations directly, generating output compatible with the sweep infrastructure.

## Files

### 1. `ttnn_composite_runner.py`
Python script that:
- Takes same arguments as native SFPU binary
- Uses TTNN operations to execute composite activations
- Outputs timing data in same format (`TIMING_*` lines)
- Generates CSV output when `DUMP_OUTPUT_CSV` is set
- Compatible with sweep script expectations

### 2. `sweep_composite_ttnn.sh`
Sweep script that:
- Tests all composite activations
- Integrates with existing sweep infrastructure
- Uses same output format as `sweep_native_sfpu.sh`
- Generates results CSV compatible with analysis tools

### 3. `ttnn_activation_wrapper.py`
Standalone wrapper for testing any TTNN operation (includes both simple and composite).

## Quick Start

### Test a Single Composite Activation

```bash
# Using the runner directly
python3 ttnn_composite_runner.py mish \
    --precision bf16 \
    --range-min -10 \
    --range-max 10 \
    --tiles 256

# With CSV output
DUMP_OUTPUT_CSV="output/mish_output.csv" \
python3 ttnn_composite_runner.py mish \
    --precision bf16 \
    --range-min -10 \
    --range-max 10 \
    --tiles 256
```

### Run Full Composite Sweep

```bash
# Run all composite activations
./sweep_composite_ttnn.sh

# Run specific activation
./sweep_composite_ttnn.sh --activation mish

# Run specific precision
./sweep_composite_ttnn.sh --precision bf16

# Dry run (show what would be tested)
./sweep_composite_ttnn.sh --dry-run

# Skip run (show existing results)
./sweep_composite_ttnn.sh --skip-run
```

## Fast/Precise Mode Support

TTNN operations support `fast_and_approximate_mode` for certain operations:

**Operations with Fast/Precise Support:**
- `exp` - Exponential function
- `erf` - Error function
- `gelu` - GELU activation
- `rsqrt` - Reciprocal square root
- `erfc` - Complementary error function

**Behavior:**
- `--fast-approx`: Uses `fast_and_approximate_mode=True` (faster, slightly less accurate)
- `--no-fast-approx`: Uses `fast_and_approximate_mode=False` (slower, more accurate)
- Other operations: Flag is accepted but ignored (always use default mode)

**Example:**
```bash
# Fast mode (for exp, erf, gelu)
python3 ttnn_composite_runner.py exp --precision bf16 --range-min -10 --range-max 10 --fast-approx

# Precise mode (for exp, erf, gelu)
python3 ttnn_composite_runner.py exp --precision bf16 --range-min -10 --range-max 10 --no-fast-approx

# Composite ops (flag accepted but ignored)
python3 ttnn_composite_runner.py mish --precision bf16 --range-min -10 --range-max 10 --fast-approx
```

## Available Composite Operations

| Operation | Formula | Description | Fast/Precise Support |
|-----------|---------|-------------|---------------------|
| `silu` | `x * sigmoid(x)` | SiLU/Swish activation | No (always default) |
| `mish` | `x * tanh(softplus(x))` | Mish activation | No (always default) |
| `hardswish` | `x * hardsigmoid(x)` | Hardswish activation | No (always default) |
| `logsigmoid` | `log(sigmoid(x))` | Log-sigmoid | No (always default) |
| `tanhshrink` | `x - tanh(x)` | Tanh shrinkage | No (always default) |
| `hardshrink` | `x if |x| > λ else 0` | Hard shrinkage | No (always default) |
| `logit` | `log(x / (1-x))` | Logit with clamping | No (always default) |
| `relu_max` | `min(relu(x), max)` | ReLU with maximum | No (always default) |
| `relu_min` | `max(relu(x), min)` | ReLU with minimum | No (always default) |
| `exp` | `e^x` | Exponential | **Yes** ✅ |
| `erf` | `erf(x)` | Error function | **Yes** ✅ |
| `gelu` | `GELU(x)` | GELU activation | **Yes** ✅ |

## Output Format

### Timing Output
```
TIMING_DEVICE_INIT: 123.456
TIMING_PROGRAM_CREATION: 0.001
TIMING_BUFFER_ALLOCATION: 0.002
TIMING_DATA_PREPARATION: 45.678
TIMING_HOST_TO_DEVICE: 12.345
TIMING_KERNEL_CREATION: 0.003
TIMING_KERNEL_EXECUTION: 8.901
TIMING_DEVICE_TO_HOST: 23.456
```

### CSV Output Format (Identical to Native SFPU)
```csv
input,output
-10.0000000000,0.0000000000
-9.9000000000,0.0000500000
...
```

**Format Specifications:**
- Header: `input,output`
- Precision: 10 decimal places (matches native SFPU output)
- Delimiter: comma
- Encoding: UTF-8

### Results CSV (in `data/` directory)
```
precision,activation,sfpu_mode,tile_count,device_init_ms,program_creation_ms,...,mae,rmse,max_error,...,status
bf16,mish,fast,256,123.4,0.0,0.0,45.6,12.3,0.0,8.9,23.4,65.4,64.2,1.2,0.00012,0.00034,0.00156,0.0012,45.2,12.3,pass
```

## Integration with Existing Infrastructure

### Architecture Detection

The sweep script automatically detects your hardware architecture from the `ARCH_NAME` environment variable (e.g., `blackhole`, `wormhole_b0`). All output paths include the architecture:

```bash
# Architecture is automatically detected
export ARCH_NAME=blackhole  # or wormhole_b0

# Run sweep
./sweep_composite_ttnn.sh

# Outputs go to architecture-specific directories:
# Results:     data/blackhole/composite_ttnn_results.csv
# Hardware:    data/hardware_outputs/blackhole/mish/composite_ttnn_mish_bf16_fast.csv
```

**Example paths for different architectures:**

```
# Blackhole
data/blackhole/composite_ttnn_results.csv
data/hardware_outputs/blackhole/mish/composite_ttnn_mish_bf16_fast.csv

# Wormhole
data/wormhole/composite_ttnn_results.csv
data/hardware_outputs/wormhole/mish/composite_ttnn_mish_bf16_fast.csv
```

### Directory Structure
```
data/hardware_outputs/<arch>/
└── <activation>/
    ├── native_sfpu_<activation>_bf16_fast.csv      # From sweep_native_sfpu.sh
    ├── native_sfpu_<activation>_bf16_precise.csv
    ├── composite_ttnn_<activation>_bf16_fast.csv   # From sweep_composite_ttnn.sh ✨ NEW
    └── ...
```

For example:
```
data/hardware_outputs/blackhole/mish/composite_ttnn_mish_bf16_fast.csv
data/hardware_outputs/wormhole/hardswish/composite_ttnn_hardswish_bf16_fast.csv
```

### Results Files (per architecture)
```
data/
├── blackhole/
│   ├── native_sfpu_results.csv       # Native SFPU results
│   ├── composite_ttnn_results.csv    # Composite TTNN results ✨ NEW
│   └── polynomial_results.csv        # Polynomial LUT results
└── wormhole/
    ├── native_sfpu_results.csv
    ├── composite_ttnn_results.csv    # ✨ NEW
    └── polynomial_results.csv
```

## Complete Coverage

With both native SFPU and composite TTNN, you now have **complete coverage** of all 48 activation functions:

### Native SFPU Kernel (39 functions)
- IDs 0-38 in `native_sfpu.cpp`
- Single-pass, maximum performance
- Run with: `sweep_native_sfpu.sh`

### Composite TTNN (9 functions)
- mish, hardswish, tanhshrink, logsigmoid, logit, hardshrink, relu_max, relu_min, silu
- Multi-operation TTNN calls
- Run with: `sweep_composite_ttnn.sh`

## Performance Comparison

After running both sweeps, compare performance:

```bash
# Run native SFPU sweep
./sweep_native_sfpu.sh

# Run composite TTNN sweep
./sweep_composite_ttnn.sh

# View results (architecture-specific paths)
cat data/$ARCH_NAME/native_sfpu_results.csv
cat data/$ARCH_NAME/composite_ttnn_results.csv
```

Expected characteristics:
- **Native SFPU**: Lower latency (single kernel launch), best for simple operations
- **Composite TTNN**: Higher latency (multiple kernel launches), necessary for complex operations

## Usage in Combined Sweeps

To test **all** activations (native + composite):

```bash
# Method 1: Use the provided wrapper script (recommended)
./sweep_all_complete.sh

# With specific options
./sweep_all_complete.sh --precision bf16
./sweep_all_complete.sh --activation gelu  # Only runs if in native SFPU list

# Method 2: Run both sweeps separately
./sweep_native_sfpu.sh
./sweep_composite_ttnn.sh
```

The `sweep_all_complete.sh` script:
- Runs native SFPU sweep (39 activations)
- Runs composite TTNN sweep (9 activations)
- Reports final output paths with correct architecture
- Total: 48 activation functions covered

## Format Compatibility

### CSV Output Format Verification

Both native SFPU and composite TTNN produce **identical CSV format**:

**Native SFPU (C++):**
```cpp
csv_file << std::fixed << std::setprecision(10);
csv_file << "input,output\n";
csv_file << input_value << "," << output_value << "\n";
```

**Composite TTNN (Python):**
```python
writer.writerow(['input', 'output'])
writer.writerow([f'{float(inp):.10f}', f'{float(out):.10f}'])
```

**Output (both produce identical format):**
```csv
input,output
-10.0000000000,0.0000450000
-9.9000000000,0.0000500000
```

### Results CSV Format Verification

Both sweeps write to the **same schema**:
```
precision,activation,sfpu_mode,tile_count,device_init_ms,program_creation_ms,
buffer_alloc_ms,data_prep_ms,host_to_device_ms,kernel_creation_ms,
kernel_exec_ms,device_to_host_ms,runtime_mean_ms,runtime_min_ms,
runtime_stddev_ms,mae,rmse,max_error,mean_rel_error,max_ulp_error,
mean_ulp_error,status
```

This ensures:
- ✅ Same plotting tools work for both
- ✅ Same accuracy extraction (`extract_accuracy.py`)
- ✅ Same analysis scripts
- ✅ Direct performance comparison

## Accuracy Comparison

Both approaches use the same accuracy computation:

```bash
# Extract accuracy from generated CSVs (works for both native SFPU and composite TTNN)
python3 ~/workspace/tt-polynomial-fitter/extract_accuracy.py \
    mish \
    data/hardware_outputs/<arch>/mish/composite_ttnn_mish_bf16_fast.csv
```

The `sweep_composite_ttnn.sh` script automatically calls `extract_accuracy.py` just like `sweep_native_sfpu.sh`.

## Troubleshooting

### Runner Script Not Found
```bash
# Ensure script is executable
chmod +x ttnn_composite_runner.py

# Verify path in sweep script
ls -la ttnn_composite_runner.py
```

### TTNN Import Error
```bash
# Ensure virtual environment is activated
source python_env/bin/activate

# Verify TTNN is installed
python3 -c "import ttnn; print(ttnn.__version__)"
```

### CSV Not Generated
```bash
# Check environment variable
echo $DUMP_OUTPUT_CSV

# Manually set and run
DUMP_OUTPUT_CSV="test.csv" python3 ttnn_composite_runner.py mish \
    --precision bf16 --range-min -10 --range-max 10
```

### Device Timeout
```bash
# Reset device
tt-smi -r 0

# Increase timeout
export TIMEOUT_SECONDS=120
./sweep_composite_ttnn.sh
```

## Implementation Details

### TTNN Composite Operation Examples

#### Mish (Already in TTNN)
```python
def ttnn_mish(x):
    return ttnn.mish(x)  # Native TTNN implementation
```

#### Tanhshrink (2 operations)
```python
def ttnn_tanhshrink(x):
    tanh_x = ttnn.tanh(x)
    return ttnn.sub(x, tanh_x)
```

#### Hardshrink (4 operations)
```python
def ttnn_hardshrink(x, lambda_val=0.5):
    abs_x = ttnn.abs(x)
    lambda_tensor = ttnn.full_like(abs_x, lambda_val)
    mask = ttnn.gt(abs_x, lambda_tensor)
    zeros = ttnn.zeros_like(x)
    return ttnn.where(mask, x, zeros)
```

#### Logit (5 operations)
```python
def ttnn_logit(x, eps=1e-7):
    x_clamped = ttnn.clip(x, eps, 1.0 - eps)
    one = ttnn.ones_like(x_clamped)
    one_minus_x = ttnn.sub(one, x_clamped)
    ratio = ttnn.div(x_clamped, one_minus_x)
    return ttnn.log(ratio)
```

## Future Work

### Potential Optimizations
1. **Fused Kernels**: TTNN team could add fused implementations for composite ops
2. **Caching**: Reuse device/program between runs
3. **Batching**: Process multiple activations in single device session

### Additional Composite Operations
If more composite operations are needed:

1. Add to `COMPOSITE_OPERATIONS` dict in `ttnn_composite_runner.py`
2. Implement using TTNN operations
3. Add to `COMPOSITE_ACTIVATIONS` array in `sweep_composite_ttnn.sh`
4. Run sweep

Example:
```python
# In ttnn_composite_runner.py
def ttnn_my_custom_activation(x):
    # Implement using TTNN operations
    return ttnn.custom_op(x)

COMPOSITE_OPERATIONS["my_custom"] = ttnn_my_custom_activation
```

## Summary

✅ **Complete activation coverage**: All 48 functions from `~/workspace/tt-polynomial-fitter/activations/`
✅ **Sweep integration**: Compatible with existing infrastructure
✅ **CSV output**: Same format for plotting and analysis
✅ **Timing metrics**: Same profiling format as native kernel
✅ **Accuracy comparison**: Uses same ground truth computation

**Result:** You can now run sweeps for ALL activation functions and compare:
- Native SFPU (fast, single-pass)
- Composite TTNN (flexible, multi-operation)
- Polynomial LUT (configurable accuracy/performance)
- Rational function approximation
