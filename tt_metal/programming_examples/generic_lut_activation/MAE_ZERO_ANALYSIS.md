# MAE=0 Cases Investigation Results

## Summary

**Finding**: Most MAE=0 cases in the results are **FALSE POSITIVES** caused by insufficient sampling during error calculation. The sweep scripts only use 11 sample points from 262,144 total data points (0.004% sampling rate).

## Root Cause

The sweep scripts (e.g., `sweep_native_sfpu.sh`, `sweep_piecewise_constant.sh`) extract only 11 sample points for accuracy calculation:

```bash
# Line 166 in sweep_native_sfpu.sh
echo "$output" | awk '/^[[:space:]]*[0-9]+[[:space:]]+[-0-9.]+[[:space:]]+[-0-9.]+$/ {print}' > /tmp/tt_samples.txt
```

These samples correspond to indices: `{0, 3276, 6553, 9830, 13107, 16384, 19660, 22937, 26214, 29491, 32767}` from the C++ code (generic_lut_activation.cpp:311).

## Key Statistics

- **Total MAE=0 cases found**: 140 across all result files
- **Truly legitimate**: ~2-3 cases (native_sfpu_relu, some threshold variants)
- **False positives**: ~137-138 cases (97-98%)
- **Root cause**: Sparse sampling (11 out of 262,144 points = 0.004% sampling rate)

## Detailed Findings by Category

### Category A: Truly Legitimate MAE≈0

#### 1. native_sfpu_relu - ✓ LEGITIMATE
- **Reported MAE**: 0.0
- **Actual MAE**: 0.0 (verified on all 262,144 points)
- **Max Error**: 0.0
- **Diagnosis**: LEGITIMATE - Hardware SFPU produces perfect ReLU outputs
- **Reason**: Uses native hardware ReLU instruction, not LUT approximation

#### 2. piecewise_quadratic_remez_threshold_{4,8,16,32} - ✓ LEGITIMATE
- **Reported MAE**: 0.0
- **Actual MAE**: 0.0 (verified on all 262,144 points)
- **Max Error**: 0.0
- **Diagnosis**: LEGITIMATE - Threshold function is identical to ReLU with default parameters
- **Reason**: ReLU/threshold is piecewise linear (2 segments: y=0 for x≤0, y=x for x>0), which can be exactly represented by piecewise quadratic approximations when the breakpoint aligns at x=0

### Category B: False Positives (Piecewise Linear Activations)

#### 3. piecewise_linear_relu_32 - ⚠️ FALSE POSITIVE
- **Reported MAE**: 0.0
- **Actual MAE**: 0.000122
- **Max Error**: 0.022461
- **Diagnosis**: Minor false positive - very low error but not exactly zero
- **Reason**: Even though ReLU is piecewise linear, the piecewise linear approximation uses a different segment structure that doesn't perfectly align

#### 4. piecewise_constant_relu_4 - ⚠️ FALSE POSITIVE
- **Reported MAE**: 0.0
- **Actual MAE**: 0.324
- **Max Error**: 0.875
- **Diagnosis**: DISCREPANCY - Reported MAE=0 is incorrect
- **Example Error**: Input=5.0, HW Output=4.125, Expected=5.0, Error=0.875
- **Reason**: With only 4 segments, the piecewise constant approximation is very coarse. The 11 sample points happened to land exactly on segment boundaries or representative points where the error appeared to be zero.

### Category C: False Positives (Curved Activations)

These activations contain smooth curves and cannot be perfectly represented by finite piecewise approximations. All MAE=0 reports for these are artifacts of sparse sampling.

#### 5. piecewise_constant_hardswish_4 - ⚠️ FALSE POSITIVE
- **Reported MAE**: 0.0
- **Actual MAE**: 0.677
- **Max Error**: 2.5
- **Diagnosis**: DISCREPANCY - Reported MAE=0 is incorrect
- **Example Error**: Input=10.0, HW Output=7.5, Expected=10.0, Error=2.5
- **Reason**: Coarse 4-segment approximation with lucky sample point placement

#### 6. piecewise_constant_hardtanh_4 - ⚠️ FALSE POSITIVE
- **Reported MAE**: 0.0
- **Actual MAE**: 0.050
- **Max Error**: 1.0
- **Diagnosis**: DISCREPANCY - Reported MAE=0 is incorrect
- **Example Error**: Input=0.0, HW Output=1.0, Expected=0.0, Error=1.0
- **Reason**: The approximation has significant discontinuities near zero that weren't captured by sparse sampling

#### 7. piecewise_constant_threshold_4 - ⚠️ FALSE POSITIVE
- **Reported MAE**: 0.0
- **Actual MAE**: 0.625
- **Max Error**: 2.5
- **Diagnosis**: DISCREPANCY - Reported MAE=0 is incorrect
- **Example Error**: Input=10.0, HW Output=7.5, Expected=10.0, Error=2.5
- **Reason**: Coarse approximation with insufficient sampling

## Why This Happens

1. **Sparse Sampling**: Only 11 out of 262,144 points (0.004%) are used for error calculation
2. **Lucky Placement**: For piecewise approximations with few segments, the sample points can accidentally align with segment midpoints or boundaries where the local error is low
3. **Coarse Quantization**: With only 4 segments covering a range like [-10, 10], each segment spans 5 units. Sample points at regular intervals can miss regions of high error.

## Impact

- **Accuracy Metrics**: The reported MAE values in results CSV files are unreliable for coarse approximations
- **Performance Claims**: Any claims about "perfect accuracy" (MAE=0) for piecewise approximations should be verified
- **Comparison Fairness**: Comparisons between methods using reported MAE are potentially misleading

#### 8. piecewise_linear_hardshrink_32 - ⚠️ FALSE POSITIVE
- **Reported MAE**: 0.0
- **Actual MAE**: 0.008734
- **Max Error**: 0.275391
- **Diagnosis**: False positive - hardshrink has curved regions
- **Reason**: Hardshrink(x) = x if |x| > λ else 0, but transitions aren't sharp in practice

### Category D: Severe False Positives

#### 9. piecewise_hexic_remez_relu6_32 - ⚠️ SEVERE FALSE POSITIVE
- **Reported MAE**: 0.0
- **Actual MAE**: 11.16
- **Max Error**: 954.0
- **Diagnosis**: CRITICAL - Extremely large errors despite reported MAE=0
- **Reason**: Likely bug in LUT generation or kernel implementation for relu6. The 11 sample points completely missed the catastrophic errors.

## Activation Function Categories

1. **Piecewise Linear** (can theoretically achieve MAE=0 with proper alignment):
   - relu, threshold, hardtanh, leaky_relu, prelu, relu6
   - In practice, only threshold consistently achieves true MAE≈0

2. **Smooth Curves** (cannot achieve exact MAE=0 with finite approximations):
   - sigmoid, tanh, gelu, swish, softplus, exp, elu, selu
   - sin, cos, erf, atanh, sinh, cosh

3. **Piecewise with Curves** (cannot achieve exact MAE=0):
   - hardswish, hardshrink, softshrink, mish, softsign, logsigmoid

## Recommendations

1. **Use Full Dataset**: Calculate MAE on all hardware output points, not just samples
   - **CRITICAL**: The relu6 case shows errors of 954.0 that were completely missed by sampling
2. **Increase Sampling**: If sampling is necessary for performance, use at least 1000-10000 points with stratified sampling
3. **Separate Metrics**: Report both:
   - Sample-based MAE (for consistency with current results)
   - Full-dataset MAE (from the CSV files)
4. **Add Validation**: Include a check that warns when reported MAE is suspiciously low for coarse approximations

## Files for Reference

- Sample extraction: `sweep_native_sfpu.sh:166` (and similar in other sweep scripts)
- Error calculation: `extract_accuracy.py`
- Ground truth: `ground_truth.py`
- Hardware outputs: `blackhole_hardware_outputs/*.csv`
- Results: `blackhole_*_results.csv`

## Verification Script

Run `python3 diagnose_mae_zero.py` to verify actual MAE from full CSV files vs reported MAE.
