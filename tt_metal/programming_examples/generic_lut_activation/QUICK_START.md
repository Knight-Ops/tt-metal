# Quick Start: Standalone LUT Generation

## Two-Step Workflow

### Step 1: Generate Polynomial Coefficients
```bash
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" \
    --domain -5 5 \
    --segments 16 \
    --degree 3 \
    --output tanh_coeffs.csv
```

### Step 2: Generate Hardware LUT File
```bash
# FP32 precision (default)
python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_coeffs.csv \
    --output tanh_fp32.lut

# BF16 precision
python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_coeffs.csv \
    --output tanh_bf16.lut \
    --precision bf16
```

## Common Examples

### Sigmoid
```bash
# Generate coefficients
python3 segmentation/generate_coefficients.py \
    --function "1/(1+exp(-x))" \
    --domain -8 8 \
    --segments 16 \
    --degree 3 \
    --output sigmoid_coeffs.csv

# Generate both FP32 and BF16 LUTs
python3 luts/generate_luts_from_coefficient_table.py \
    --input sigmoid_coeffs.csv \
    --output sigmoid_fp32.lut \
    --precision fp32

python3 luts/generate_luts_from_coefficient_table.py \
    --input sigmoid_coeffs.csv \
    --output sigmoid_bf16.lut \
    --precision bf16
```

### GELU
```bash
python3 segmentation/generate_coefficients.py \
    --function "x * 0.5 * (1 + erf(x / sqrt(2)))" \
    --domain -5 5 \
    --segments 32 \
    --degree 4 \
    --output gelu_coeffs.csv

python3 luts/generate_luts_from_coefficient_table.py \
    --input gelu_coeffs.csv \
    --output gelu_bf16.lut \
    --precision bf16
```

### SiLU/Swish
```bash
python3 segmentation/generate_coefficients.py \
    --function "x / (1 + exp(-x))" \
    --domain -5 5 \
    --segments 16 \
    --degree 3 \
    --output silu_coeffs.csv

python3 luts/generate_luts_from_coefficient_table.py \
    --input silu_coeffs.csv \
    --output silu_bf16.lut \
    --precision bf16
```

## Input Parameters

### generate_coefficients.py
| Parameter | Description | Example |
|-----------|-------------|---------|
| `--function` | Sollya expression | `"tanh(x)"` |
| `--domain` | Min and max bounds | `--domain -5 5` |
| `--segments` | Number of segments | `--segments 16` |
| `--degree` | Polynomial degree | `--degree 3` |
| `--precision` | Number precision | `fp32` (default), `bf16`, `fp16`, `double` |
| `--critical` | Critical breakpoints | `--critical 0.0` (can repeat) |
| `--grid-size` | DP grid resolution | `--grid-size 100` (default) |
| `--output` | Output CSV file | `--output coeffs.csv` |

### generate_luts_from_coefficient_table.py
| Parameter | Description | Example |
|-----------|-------------|---------|
| `--input` | Coefficient CSV | `--input coeffs.csv` |
| `--output` | Output LUT file | `--output activation.lut` |
| `--precision` | Number format | `fp32` or `bf16` (default: fp32) |
| `--verify` | Verify output | `--verify` (optional) |

## Precision Options

Both scripts accept the same precision names for consistency:

| Precision | Aliases | Mantissa Bits | Bytes/Value | Description |
|-----------|---------|---------------|-------------|-------------|
| `fp32` | `single`, `float32` | 23 | 4 | Standard single precision (default) |
| `bf16` | `bfloat16` | 7 | 2 | Google BFloat16, ML/AI optimized |
| `fp16` | `half`, `float16` | 10 | 2 | IEEE half precision |
| `double` | `fp64`, `float64` | 52 | 8 | Double precision |

**Advanced**: You can also specify mantissa bit count directly (e.g., `--precision 7` for BF16)

## Typical Workflow

```bash
# 1. Generate coefficients with BF16 precision for your hardware
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" \
    --domain -5 5 \
    --segments 16 \
    --degree 3 \
    --precision bf16 \
    --output tanh_coeffs.csv

# 2. Generate LUT file in BF16 format
python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_coeffs.csv \
    --output tanh.lut \
    --precision bf16

# 3. Deploy tanh.lut to hardware accelerator
```

## Output Files

### Coefficient CSV (`*.csv`)
Human-readable text format:
```csv
segment_id,lo,hi,c0,c1,c2,c3,error
0,-5.0000000000,-4.0000000000,1.0e+00,3.4e-02,8.9e-04,1.2e-05,2.3e-07
1,-4.0000000000,-3.0000000000,9.9e-01,4.5e-02,2.1e-03,4.8e-05,1.8e-06
...
```

### LUT Binary (`*.lut`)
Hardware-optimized binary format:
- Header (32 bytes): metadata
- Segment bounds
- Polynomial coefficients
- Error table

## Performance Comparison

| Configuration | File Size | Max Error | Use Case |
|---------------|-----------|-----------|----------|
| 8 seg, deg 2, BF16 | ~200 bytes | ~1e-4 | Low accuracy, small footprint |
| 16 seg, deg 3, BF16 | ~336 bytes | ~1e-6 | Balanced (recommended) |
| 32 seg, deg 4, BF16 | ~768 bytes | ~1e-8 | High accuracy |
| 16 seg, deg 3, FP32 | ~608 bytes | ~1e-7 | Higher precision needed |

## Troubleshooting

### Sollya not found
```bash
# Install Sollya from https://sollya.gforge.inria.fr/
# macOS:
brew install sollya

# Ubuntu:
sudo apt-get install sollya
```

### Invalid function expression
```bash
# Use Sollya syntax:
✓ "tanh(x)"
✓ "1/(1+exp(-x))"
✓ "x * 0.5 * (1 + erf(x / sqrt(2)))"

✗ "np.tanh(x)"  # Wrong: Python syntax
✗ "sigmoid(x)"  # Wrong: no predefined sigmoid
```

### Domain too large/small
```bash
# Adjust --grid-size for better precision
python3 segmentation/generate_coefficients.py \
    --function "exp(x)" \
    --domain -10 10 \
    --segments 32 \
    --degree 4 \
    --grid-size 200  # ← Increase for large domains
```

## Next Steps

- See `STANDALONE_WORKFLOW.md` for detailed documentation
- See `segmentation/` for algorithm details
- See `REORGANIZATION_FINAL.md` for code organization
