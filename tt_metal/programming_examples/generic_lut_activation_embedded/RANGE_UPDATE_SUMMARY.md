# Range Value Update Summary

**Date:** January 16, 2026
**File Updated:** `activations_config.csv`

## What Changed

Updated the `nn_range_min` and `nn_range_max` columns in `activations_config.csv` to use wider, more realistic ranges based on:
1. **kernel_bench** test generators (from `/Users/nkapre/workspace/kernel_bench/benchmark/`)
2. **ground_truth.py** `ACTIVATION_DOMAINS` dictionary
3. Best practices for neural network activation functions

## Key Changes

| Activation | Old Range | New Range | Change |
|------------|-----------|-----------|--------|
| **Most activations** | (-1.0, 1.0) | (-10.0, 10.0) | **10x wider** |
| **atanh** | (-0.99, 0.99) | (-0.9, 0.9) | **Safer margin** |
| **cos, sin** | (-1.0, 1.0) | (-6.28, 6.28) | **Full 2π period** |
| **tanh, sinh, cosh** | (-1.0, 1.0) | (-5.0, 5.0) | **5x wider** |
| **erf, hardsigmoid** | (-1.0, 1.0) | (-3.0, 3.0) | **3x wider** |

## How These Ranges Are Used

The `nn_range_min` and `nn_range_max` columns map to `test_min` and `test_max` in the C++ code:

```cpp
// activation_config.hpp lines 100-101
config.test_min = std::stof(tokens[2]);  // Reads nn_range_min
config.test_max = std::stof(tokens[3]);  // Reads nn_range_max
```

These values are used for:

1. **Test Input Generation** (generic_lut_activation.cpp:215-221)
   ```cpp
   float test_range = test_max - test_min;
   for (size_t i = 0; i < input_data.size(); i++) {
       float val = test_min + test_range * (i / float(input_data.size()));
       input_data[i] = bfloat16(val);
   }
   ```

2. **Kernel Runtime Args** (generic_lut_activation.cpp:281)
   ```cpp
   SetRuntimeArgs(program, compute, core, {n_tiles, test_min, test_max});
   ```

3. **Kernel Segment Computation** (all piecewise kernels)
   ```cpp
   const float INPUT_RANGE = input_max - input_min;
   const float SEGMENT_WIDTH = INPUT_RANGE / NUM_SEGMENTS;
   ```

## Impact

### Positive Effects

1. **✅ Better Coverage**: LUTs now cover the full range of values seen in real neural networks
2. **✅ Matches kernel_bench**: Aligns with standard benchmark expectations
3. **✅ More General-Purpose**: LUTs work for pre-normalization, residual connections, attention mechanisms
4. **✅ Captures Full Behavior**: Tests periodic functions over complete periods, exponential functions over growth regions

### Expected Changes

1. **Slightly Higher Errors**: Wider ranges mean more area to approximate with same number of segments
   - **Previous**: Max error ~1e-3 over [-1, 1] (2-unit range)
   - **Expected**: Max error ~1e-2 over [-10, 10] (20-unit range, 10x wider)
   - **Still acceptable** for most neural network applications (tolerance ~1e-2 to 1e-3)

2. **Unchanged Runtime**: Range width doesn't affect kernel performance (same number of v_if statements)

3. **LUTs Need Regeneration**: All existing `.lut` files must be regenerated with new ranges

## Next Steps

1. **Regenerate All LUTs** with new ranges:
   ```bash
   ./generate_luts.sh
   ```

2. **Re-run Sweeps** on both architectures:
   ```bash
   # Wormhole
   ./sweep_all.sh

   # Blackhole (on remote server)
   ssh -A -p $BLACKHOLE_PORT $BLACKHOLE_HOST
   cd /localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation
   ./sweep_all.sh
   ```

3. **Verify Accuracy**: Check that errors are still acceptable (<1e-2 for most activations)

4. **Compare Results**: Plot new error metrics vs old to quantify impact

## Special Cases

### atanh: Safer Boundary
- **Old**: -0.99 to 0.99 (too close to asymptotes at ±1)
- **New**: -0.9 to 0.9 (matches kernel_bench safety margin)
- **Reason**: atanh(x) diverges at x=±1, so we need margin for numerical stability

### Periodic Functions (sin, cos)
- **Old**: -1.0 to 1.0 (less than half period)
- **New**: -6.28 to 6.28 (one full 2π period)
- **Reason**: Periodic functions need at least one full period for meaningful LUT approximation

### Exponential Functions (exp, sinh, cosh)
- **Old**: -1.0 to 1.0 (too narrow to see exponential behavior)
- **New**: -5.0 to 5.0 (captures exponential growth without overflow)
- **Reason**: Float32 overflows beyond ±88, but interesting behavior is in ±5 range

## Validation

You can verify the changes are correct by:

1. **Check CSV diff**:
   ```bash
   git diff activations_config.csv
   ```

2. **Test one activation**:
   ```bash
   # Regenerate sigmoid LUT
   python3 generate_piecewise_quadratic_remez_luts.py --activations sigmoid --depths 8

   # Build and run
   ./build_metal.sh
   build_Release/programming_examples/generic_lut_activation \
       luts/piecewise_quadratic_remez_sigmoid_8.lut 32

   # Should print: "Test range: [-10.0, 10.0]" (not [-1.0, 1.0])
   ```

3. **Compare with ground_truth.py**:
   ```python
   from ground_truth import get_activation_domain
   print(get_activation_domain('sigmoid', 'typical_range'))  # (-5.0, 5.0)
   # Our choice of (-10.0, 10.0) is even wider, which is good
   ```

## References

- **kernel_bench**: `/Users/nkapre/workspace/kernel_bench/benchmark/*/` test generators
- **ground_truth.py**: Lines 645-882 (ACTIVATION_DOMAINS dictionary)
- **activation_config.hpp**: Lines 100-101 (CSV parsing)
- **generic_lut_activation.cpp**: Lines 120, 215-221, 281 (range usage)

## Commit Message

```
Update activation ranges in config to match realistic NN distributions

- Change nn_range from narrow (-1, 1) to wider typical ranges
- Most activations now use (-10, 10) to match kernel_bench
- Periodic functions (sin, cos) use full 2π period (-6.28, 6.28)
- Exponential functions (tanh, sinh, cosh) use (-5, 5)
- atanh uses safer (-0.9, 0.9) margin away from asymptotes

This makes LUTs more general-purpose and suitable for:
- Pre-normalization inputs
- Residual connections
- Attention mechanisms
- All stages of training

Matches ranges used in kernel_bench test generators.
```
