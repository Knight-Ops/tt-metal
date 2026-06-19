# Precision Guide

## Consistent Precision Interface

Both `generate_coefficients.py` and `generate_luts_from_coefficient_table.py` now use the same `--precision` parameter with identical naming conventions.

## Supported Precisions

| Precision | Aliases | Mantissa Bits | Bytes/Value | Use Case |
|-----------|---------|---------------|-------------|----------|
| **FP32** | `fp32`, `single`, `float32` | 23 | 4 | Standard precision (default) |
| **BF16** | `bf16`, `bfloat16` | 7 | 2 | ML/AI accelerators, reduced memory |
| **FP16** | `fp16`, `half`, `float16` | 10 | 2 | GPU-optimized, mobile devices |
| **FP64** | `double`, `fp64`, `float64` | 52 | 8 | High accuracy scientific computing |

**Advanced**: Direct mantissa bit specification (e.g., `--precision 7` for BF16)

## Usage Examples

### Step 1: Generate Coefficients

```bash
# FP32 precision (default)
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" \
    --domain -5 5 \
    --segments 16 \
    --degree 3 \
    --output tanh_fp32_coeffs.csv

# BF16 precision (recommended for hardware accelerators)
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" \
    --domain -5 5 \
    --segments 16 \
    --degree 3 \
    --precision bf16 \
    --output tanh_bf16_coeffs.csv

# FP16 precision (mobile/embedded)
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" \
    --domain -5 5 \
    --segments 16 \
    --degree 3 \
    --precision fp16 \
    --output tanh_fp16_coeffs.csv

# Double precision (high accuracy required)
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" \
    --domain -5 5 \
    --segments 16 \
    --degree 3 \
    --precision double \
    --output tanh_fp64_coeffs.csv
```

### Step 2: Generate LUT Files

```bash
# FP32 LUT (default)
python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_fp32_coeffs.csv \
    --output tanh_fp32.lut

# BF16 LUT (matching precision from Step 1)
python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_bf16_coeffs.csv \
    --output tanh_bf16.lut \
    --precision bf16

# FP16 LUT
python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_fp16_coeffs.csv \
    --output tanh_fp16.lut \
    --precision fp16
```

## Precision Matching

### Best Practice: Match precision across both steps

```bash
# ✓ GOOD: BF16 throughout
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" --domain -5 5 --segments 16 --degree 3 \
    --precision bf16 --output tanh_coeffs.csv

python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_coeffs.csv --output tanh.lut --precision bf16


# ✓ ALSO VALID: FP32 coefficients → BF16 LUT (higher precision → lower)
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" --domain -5 5 --segments 16 --degree 3 \
    --precision fp32 --output tanh_coeffs.csv

python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_coeffs.csv --output tanh.lut --precision bf16


# ⚠️ SUBOPTIMAL: BF16 coefficients → FP32 LUT (can't add precision)
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" --domain -5 5 --segments 16 --degree 3 \
    --precision bf16 --output tanh_coeffs.csv

python3 luts/generate_luts_from_coefficient_table.py \
    --input tanh_coeffs.csv --output tanh.lut --precision fp32
# Note: FP32 LUT won't be more accurate than BF16 coefficients
```

## Precision Trade-offs

### Memory vs. Accuracy

| Config | Coeff File | LUT Size | Max Error | Notes |
|--------|-----------|----------|-----------|-------|
| FP64 → FP32 | Large | 608 B | ~1e-7 | Development/validation |
| FP32 → FP32 | Medium | 608 B | ~1e-7 | Production (good accuracy) |
| FP32 → BF16 | Medium | 336 B | ~1e-6 | Production (recommended) |
| BF16 → BF16 | Small | 336 B | ~1e-6 | Production (memory constrained) |
| FP16 → FP16 | Small | 336 B | ~1e-5 | Mobile/embedded |

*Assuming 16 segments, degree 3*

### Recommendations by Use Case

**ML/AI Accelerators (TPU, GPU)**
```bash
--precision bf16
```
- Most ML accelerators optimize for BF16
- 2× memory savings vs FP32
- Accuracy sufficient for neural networks

**High-Precision Scientific Computing**
```bash
--precision fp32  # or double
```
- When accuracy matters more than memory
- Validation and testing
- Reference implementations

**Mobile/Embedded Devices**
```bash
--precision fp16
```
- ARM NEON optimizations
- Mobile GPU support
- Battery-efficient

**Memory-Constrained Systems**
```bash
--precision bf16  # or fp16
```
- 50% memory reduction
- Faster memory transfers
- Lower power consumption

## Internal Precision Mapping

The scripts internally convert user-friendly names to Sollya/binary formats:

### generate_coefficients.py → Sollya
```python
fp32/single/float32  → "single"
bf16/bfloat16        → "7"      (7 mantissa bits)
fp16/half/float16    → "10"     (10 mantissa bits)
double/fp64/float64  → "double"
```

### generate_luts_from_coefficient_table.py → Binary Format
```python
fp32/single/float32  → 32-bit IEEE 754 float
bf16/bfloat16        → 16-bit BFloat16 (truncated FP32)
```

## Error Analysis

### Precision Impact on Approximation Error

```bash
# Test different precisions on same function
for prec in fp32 bf16 fp16; do
    python3 segmentation/generate_coefficients.py \
        --function "tanh(x)" --domain -5 5 --segments 16 --degree 3 \
        --precision $prec --output tanh_${prec}_coeffs.csv
done

# Compare errors in CSV files
# FP32: max_error ~ 1e-7
# BF16: max_error ~ 1e-6 (10× higher, still good for ML)
# FP16: max_error ~ 1e-5 (100× higher, acceptable for mobile)
```

### When Precision Matters

**High Precision Needed:**
- Scientific computing (physics, chemistry simulations)
- Financial calculations
- Medical imaging
- Cumulative operations over many steps

**Lower Precision Acceptable:**
- Neural network activations (BF16)
- Image/video processing (FP16)
- Audio synthesis (FP16)
- Game graphics (FP16)

## Validation

### Test Precision Conversion
```bash
# Verify precision names work correctly
python3 -c "
from segmentation.generate_coefficients import precision_to_sollya

tests = ['fp32', 'bf16', 'fp16', 'double']
for t in tests:
    print(f'{t} → {precision_to_sollya(t)}')
"

# Output:
# fp32 → single
# bf16 → 7
# fp16 → 10
# double → double
```

### Compare Outputs
```bash
# Generate with different precisions
python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" --domain -5 5 --segments 8 --degree 2 \
    --precision fp32 --output tanh_fp32.csv

python3 segmentation/generate_coefficients.py \
    --function "tanh(x)" --domain -5 5 --segments 8 --degree 2 \
    --precision bf16 --output tanh_bf16.csv

# Compare error columns in CSV files
python3 -c "
import csv

def print_errors(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        errors = [float(row['error']) for row in reader]
    print(f'{filename}: max={max(errors):.6e}, mean={sum(errors)/len(errors):.6e}')

print_errors('tanh_fp32.csv')
print_errors('tanh_bf16.csv')
"
```

## Summary

✅ **Consistent Interface**: Both scripts use `--precision` with same naming
✅ **User-Friendly**: No need to remember mantissa bit counts
✅ **Flexible**: Supports all common precision formats
✅ **Backward Compatible**: Direct bit counts still work (e.g., `--precision 7`)
✅ **Hardware Optimized**: BF16/FP16 for ML accelerators, FP32 for general purpose

**Recommended Default**: `--precision bf16` for production ML workloads
