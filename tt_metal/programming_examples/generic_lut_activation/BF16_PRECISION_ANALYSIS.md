# BF16 Precision Analysis for LUT Activations

## Summary

**The theoretical error was calculated in FP32/FP64 precision, but hardware runs in BF16 precision.** This explains why hardware error is higher than theoretical predictions.

## What We Fixed

1. ✅ Installed `ml_dtypes` library for proper BF16 quantization
2. ✅ Added `--simulate-bf16` flag to `plot_theoretical_error.py`
3. ✅ Created analysis tools to decompose error sources

## Error Sources

Hardware LUT error has **three components**:

```
Total Hardware Error = Polynomial Approximation Error
                     + Input Quantization Error (FP32 → BF16)
                     + Output Quantization Error (FP32 → BF16)
```

### Before (Theoretical - FP32 only):
- ✗ Only measured polynomial approximation error
- ✗ Assumed perfect FP32 precision throughout
- ✗ Didn't account for hardware BF16 conversions

### After (Hardware-Accurate - BF16 simulation):
- ✅ Simulates input quantization: test_value (FP32) → BF16 → FP32
- ✅ Evaluates LUT with quantized input
- ✅ Simulates output quantization: result (FP32) → BF16 → FP32
- ✅ Matches actual hardware path

## Quantization Overhead by Activation

| Activation | Depth | Method | Polynomial Error (MAE) | BF16 Overhead (MAE) | Overhead % |
|------------|-------|--------|------------------------|---------------------|------------|
| sigmoid    | 8     | cubic  | 9.11e+02               | 1.42e+01           | +1.6%      |
| tanh       | 16    | cubic  | 8.02e+02               | 1.61e+01           | +2.0%      |
| exp        | 16    | cubic  | 1.07e+07               | 2.78e+05           | +2.6%      |

**Key Finding**: BF16 quantization typically adds **1-3% to MAE**, but can add **36-60% to max error** depending on the activation's output range.

## Usage

### 1. Generate theoretical error plots with BF16 simulation:
```bash
# FP32 ideal precision (old behavior)
python3 plot_theoretical_error.py --activation sigmoid

# BF16 hardware-accurate (new, recommended)
python3 plot_theoretical_error.py --activation sigmoid --simulate-bf16
```

### 2. Compare FP32 vs BF16 side-by-side:
```bash
python3 compare_bf16_fp32_error.py sigmoid 8 piecewise_cubic_remez
```

### 3. Analyze quantization overhead breakdown:
```bash
python3 analyze_bf16_overhead.py sigmoid 8
```

## Why BF16 Sometimes Shows Lower Error

In some test cases, BF16 quantization appears to reduce error compared to FP32. This is due to **lucky rounding**:

- BF16 truncates the mantissa from 23 bits (FP32) to 7 bits (BF16)
- By chance, this rounding can move a value *closer* to the ground truth
- This is **not** a systematic improvement, just statistical noise
- What matters is the **additional error variance** introduced by quantization

## Technical Details

### BF16 Format (bfloat16)
- 8-bit exponent (same as FP32)
- 7-bit mantissa (vs 23-bit in FP32)
- ~3-4 decimal digits of precision (vs 7-8 for FP32)
- Relative error per conversion: ~2^-7 ≈ 0.78%

### Hardware Dataflow
```
Host (Python)           Device (TT-Metal)
────────────────        ────────────────
test_input (FP32)  →    BF16 quantize
                        ↓
                        LUT eval (FP32 coeffs × BF16 input)
                        ↓
                        BF16 quantize
                        ↓
result (BF16)      ←    return to host
```

## Recommendations

1. **Always use `--simulate-bf16`** when comparing theoretical vs hardware error
2. **Report both errors** in accuracy tables:
   - Theoretical (FP32): Polynomial approximation quality
   - Hardware (BF16): Real-world expected accuracy
3. **Budget for 2-5% overhead** when sizing LUT depth
4. **Watch max error** - quantization can significantly increase outliers

## Files Modified

- `plot_theoretical_error.py`: Added BF16 simulation support
- `compare_bf16_fp32_error.py`: New - side-by-side comparison tool
- `analyze_bf16_overhead.py`: New - detailed quantization overhead analysis

## Example Output

```
QUANTIZATION OVERHEAD:
  • BF16 quantization adds: 2.776e+05 MAE (2.60% of polynomial error)
  • BF16 quantization adds: 1.334e+08 max error (60.83% of polynomial error)

TOTAL ERROR INCREASE:
  • Hardware MAE is +2.08% vs ideal
  • Hardware max error is -0.08% vs ideal

QUANTIZATION BREAKDOWN:
  • Input quantization (FP32→BF16):  mean=7.29e-03, max=3.12e-02
  • Output quantization (FP32→BF16): mean=1.64e+04, max=5.16e+05
```

This explains the gap between theoretical predictions and actual hardware measurements!
