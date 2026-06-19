# Standalone LUT Generation Workflow

## Overview

Clean, standalone pipeline for generating hardware LUT files:

```
1. generate_coefficients.py    →  2. generate_luts_from_coefficient_table.py  →  3. .LUT files
   (segmentation/)                   (luts/)

   Input:                            Input:                                       Output:
   - Function expression             - Coefficient CSV                            - Binary .LUT files
   - Domain bounds                   - Format (FP32/BF16)                        - Hardware-ready
   - # segments, degree
   - Critical points                 Output:
                                     - Binary LUT file
   Output:
   - CSV with polynomial coeffs
```

## Step 1: Generate Polynomial Coefficients

### Script: `segmentation/generate_coefficients.py`

**Purpose**: Generate optimal polynomial coefficients for piecewise approximation

**Inputs**:
- `--function`: Sollya expression (e.g., `"tanh(x)"`, `"1/(1+exp(-x))"`)
- `--domain`: Min/max bounds (e.g., `--domain -5 5`)
- `--segments`: Number of piecewise segments
- `--degree`: Polynomial degree per segment
- `--critical`: Optional critical breakpoints (can specify multiple)
- `--method`: Segmentation method (`chebyshev` - optimal minimax)
- `--precision`: Sollya precision (`single`, `double`, or mantissa bits like `7` for BF16)
- `--grid-size`: Discretization for DP (default: 100)
- `--output`: Output CSV file

**Output CSV Format**:
```csv
segment_id,lo,hi,c0,c1,c2,c3,error
0,-5.0000000000,-4.0000000000,1.0e+00,3.4e-02,8.9e-04,1.2e-05,2.3e-07
1,-4.0000000000,-3.0000000000,9.9e-01,4.5e-02,2.1e-03,4.8e-05,1.8e-06
...
```

**Examples**:

```bash
# Basic usage: Sigmoid, 16 segments, degree 3
python3 segmentation/generate_coefficients.py \
    --function "1/(1+exp(-x))" \
    --domain -5 5 \
    --segments 16 \
    --degree 3 \
    --output sigmoid_coeffs.csv

# Tanh with critical breakpoint at x=0
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" \
    --domain -5 5 \
    --segments 16 \
    --degree 3 \
    --critical 0.0 \
    --output tanh_coeffs.csv

# BF16 precision
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" \
    --domain -5 5 \
    --segments 16 \
    --degree 3 \
    --precision bf16 \
    --output tanh_bf16_coeffs.csv

# GELU with higher precision grid
python3 segmentation/generate_coefficients.py \
    --function "x * 0.5 * (1 + erf(x / sqrt(2)))" \
    --domain -5 5 \
    --segments 32 \
    --degree 4 \
    --grid-size 200 \
    --output gelu_coeffs.csv
```

**Output**:
```
================================================================================
POLYNOMIAL COEFFICIENT GENERATION
================================================================================
Function: tanh(x)
Domain: [-5.0, 5.0]
Segments: 16
Degree: 3
Method: chebyshev

Using Chebyshev DP method...
  Precomputing 1225 segment errors...
    100/1225
    200/1225
    ...
  Done precomputing.
  Running DP for 16 segments...
  Optimal solution: 16 segments, max error = 1.234567e-06

Function: tanh(x)
Method: chebyshev
Domain: [-5.0000, 5.0000]
Segments: 16
Degree: 3
Max Error: 1.234567e-06
Segment Widths: min=0.3125, max=0.6250, mean=0.4688
Segment Errors: min=2.345e-07, max=1.234e-06, mean=6.789e-07

✓ Coefficients written to: tanh_coeffs.csv

================================================================================
✓ COMPLETE
================================================================================
Output: tanh_coeffs.csv
Segments: 16
Max error: 1.234567e-06

Next step:
  python3 luts/generate_luts_from_coefficient_table.py --input tanh_coeffs.csv
```

## Step 2: Generate Binary LUT Files

### Script: `luts/generate_luts_from_coefficient_table.py`

**Purpose**: Convert CSV coefficients to hardware-specific binary .LUT format

**Inputs**:
- `--input`: CSV file from Step 1
- `--output`: Output .LUT file name
- `--precision`: Number precision (`fp32`/`single`/`float32` or `bf16`/`bfloat16`)
- `--verify`: Optional verification (reads back and checks)

**Output .LUT Binary Format**:
```
Header (32 bytes):
  - Magic: 0x4C555400 ("LUT\0")
  - Version: uint32
  - Format: uint32 (0=FP32, 1=BF16)
  - Degree: uint32
  - Num segments: uint32
  - Reserved: 12 bytes

Segment Table:
  - [lo, hi] for each segment

Coefficient Table:
  - All coefficients [c0, c1, ..., cn] for seg0
  - All coefficients [c0, c1, ..., cn] for seg1
  - ...

Error Table:
  - Max error for each segment
```

**Examples**:

```bash
# Generate FP32 LUT
python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_coeffs.csv \
    --output tanh_fp32.lut \
    --precision fp32

# Generate BF16 LUT
python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_coeffs.csv \
    --output tanh_bf16.lut \
    --precision bf16

# Generate both formats
python3 luts/generate_luts_from_coefficient_table.py \
    --input sigmoid_coeffs.csv \
    --output sigmoid_fp32.lut \
    --precision fp32

python3 luts/generate_luts_from_coefficient_table.py \
    --input sigmoid_coeffs.csv \
    --output sigmoid_bf16.lut \
    --precision bf16
```

**Output**:
```
Loading coefficients from: tanh_coeffs.csv
✓ Loaded 16 segments, degree 3

Generating FP32 LUT...
================================================================================
LUT FILE GENERATED
================================================================================
Output: tanh_fp32.lut
Format: FP32
File size: 608 bytes

Segments: 16
Degree: 3
Domain: [-5.000000, 5.000000]

Segment Statistics:
  Width: min=0.312500, max=0.625000, mean=0.468750
  Error: min=2.345e-07, max=1.234e-06, mean=6.789e-07

Memory Layout:
  Header: 32 bytes
  Segment bounds: 128 bytes
  Coefficients: 256 bytes
  Errors: 64 bytes
  Total: 608 bytes
================================================================================
```

## Complete Example: Sigmoid Activation

### Step-by-Step

```bash
# 1. Generate coefficients (16 segments, cubic polynomials)
python3 segmentation/generate_coefficients.py \
    --function "1/(1+exp(-x))" \
    --domain -8 8 \
    --segments 16 \
    --degree 3 \
    --precision single \
    --output sigmoid_16_3.csv

# 2a. Generate FP32 LUT
python3 luts/generate_luts_from_coefficient_table.py \
    --input sigmoid_16_3.csv \
    --output sigmoid_16_3_fp32.lut \
    --precision fp32

# 2b. Generate BF16 LUT
python3 luts/generate_luts_from_coefficient_table.py \
    --input sigmoid_16_3.csv \
    --output sigmoid_16_3_bf16.lut \
    --precision bf16
```

### Output Files
```
sigmoid_16_3.csv           # Intermediate coefficients (text)
sigmoid_16_3_fp32.lut      # FP32 binary LUT (608 bytes)
sigmoid_16_3_bf16.lut      # BF16 binary LUT (336 bytes)
```

## Comparison: Different Configurations

### Accuracy vs. Size Trade-off

```bash
# Low accuracy, small size: 8 segments, degree 2
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" --domain -5 5 \
    --segments 8 --degree 2 --output tanh_8_2.csv

python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_8_2.csv --output tanh_8_2_bf16.lut --precision bf16
# Result: 200 bytes, error ~1e-4

# Medium accuracy: 16 segments, degree 3
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" --domain -5 5 \
    --segments 16 --degree 3 --output tanh_16_3.csv

python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_16_3.csv --output tanh_16_3_bf16.lut --precision bf16
# Result: 336 bytes, error ~1e-6

# High accuracy, large size: 32 segments, degree 5
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" --domain -5 5 \
    --segments 32 --degree 5 --output tanh_32_5.csv

python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_32_5.csv --output tanh_32_5_bf16.lut --precision bf16
# Result: 1024 bytes, error ~1e-9
```

## Sollya Expression Reference

Common activation functions:

```
# Sigmoid
"1/(1+exp(-x))"

# Tanh
"tanh(x)"

# GELU (exact)
"x * 0.5 * (1 + erf(x / sqrt(2)))"

# GELU (tanh approximation)
"x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"

# SiLU / Swish
"x / (1 + exp(-x))"

# Mish
"x * tanh(log(1 + exp(x)))"

# Softplus
"log(1 + exp(x))"

# ELU (x < 0 piece)
"exp(x) - 1"

# SELU (x < 0 piece, alpha=1.67, scale=1.05)
"1.0507 * 1.6732 * (exp(x) - 1)"

# Exponential
"exp(x)"

# Sine/Cosine
"sin(x)"
"cos(x)"

# Hyperbolic
"sinh(x)"
"cosh(x)"
"atanh(x)"

# Error function
"erf(x)"
```

## Critical Breakpoints

For piecewise functions, specify critical points where derivative is discontinuous:

```bash
# ReLU-like functions: breakpoint at x=0
python3 segmentation/generate_coefficients.py \
    --function "exp(x) - 1" \  # ELU negative piece
    --domain -5 0 \
    --segments 8 \
    --degree 3 \
    --critical 0.0

# HardSwish: breakpoints at x=-3 and x=3
python3 segmentation/generate_coefficients.py \
    --function "x * (x + 3) / 6" \  # Middle piece
    --domain -3 3 \
    --segments 12 \
    --degree 3 \
    --critical -3.0 --critical 3.0

# Custom piecewise: multiple critical points
python3 segmentation/generate_coefficients.py \
    --function "..." \
    --domain -5 5 \
    --segments 16 \
    --degree 3 \
    --critical -2.0 --critical 0.0 --critical 2.0
```

## Batch Generation Script

For generating multiple activations:

```bash
#!/bin/bash
# generate_all_luts.sh

ACTIVATIONS=(
  "tanh(x):-5:5:tanh"
  "1/(1+exp(-x)):-8:8:sigmoid"
  "x * 0.5 * (1 + erf(x / sqrt(2))):-5:5:gelu"
  "x / (1 + exp(-x)):-5:5:silu"
)

for act in "${ACTIVATIONS[@]}"; do
  IFS=':' read -r expr min max name <<< "$act"

  echo "Generating $name..."

  # Generate coefficients
  python3 segmentation/generate_coefficients.py \
    --function "$expr" \
    --domain $min $max \
    --segments 16 \
    --degree 3 \
    --output "${name}_coeffs.csv"

  # Generate FP32 LUT
  python3 luts/generate_luts_from_coefficient_table.py \
    --input "${name}_coeffs.csv" \
    --output "luts/${name}_fp32.lut" \
    --precision fp32

  # Generate BF16 LUT
  python3 luts/generate_luts_from_coefficient_table.py \
    --input "${name}_coeffs.csv" \
    --output "luts/${name}_bf16.lut" \
    --precision bf16
done
```

## Integration with Hardware

The generated .LUT files can be loaded by hardware accelerators:

```cpp
// C++ example
#include <fstream>
#include <cstdint>

struct LUTHeader {
    uint32_t magic;       // 0x4C555400
    uint32_t version;
    uint32_t format;      // 0=FP32, 1=BF16
    uint32_t degree;
    uint32_t num_segments;
    uint32_t reserved[3];
};

void load_lut(const char* filename) {
    std::ifstream f(filename, std::ios::binary);

    LUTHeader header;
    f.read((char*)&header, sizeof(header));

    // Verify magic
    assert(header.magic == 0x4C555400);

    // Read segment boundaries
    std::vector<float> boundaries(header.num_segments * 2);
    f.read((char*)boundaries.data(), boundaries.size() * sizeof(float));

    // Read coefficients
    size_t num_coeffs = header.num_segments * (header.degree + 1);
    std::vector<float> coeffs(num_coeffs);
    f.read((char*)coeffs.data(), num_coeffs * sizeof(float));

    // Use LUT...
}
```

## Advantages of This Workflow

1. **Standalone**: Each step is independent and testable
2. **Flexible**: Easy to change segmentation parameters
3. **Reusable**: Same CSV can generate multiple .LUT formats
4. **Inspectable**: CSV format is human-readable for debugging
5. **Version Control Friendly**: Text CSV files diff well in git
6. **Reproducible**: Complete parameters in CSV metadata

## File Organization

```
project/
├── segmentation/
│   ├── generate_coefficients.py        # Step 1: Generate coefficients
│   ├── base.py, chebyshev.py, etc.     # Core algorithms
│   └── ...
├── luts/
│   ├── generate_luts_from_coefficient_table.py  # Step 2: Generate .LUT files
│   ├── sigmoid_fp32.lut                         # Generated LUT files
│   ├── sigmoid_bf16.lut
│   └── ...
└── coefficients/                        # Store intermediate CSVs
    ├── sigmoid_16_3.csv
    ├── tanh_16_3.csv
    └── ...
```

## Next Steps

1. **Verification**: Add `--verify` to read LUT back and check accuracy
2. **Visualization**: Plot approximation error across domain
3. **Benchmarking**: Compare different segment/degree configurations
4. **Hardware Integration**: Load .LUT files into accelerator
5. **Optimization**: Tune grid_size for accuracy vs. speed trade-off
