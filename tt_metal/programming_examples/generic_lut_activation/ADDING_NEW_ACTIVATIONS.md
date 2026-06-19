# Adding New Activation Functions to Sollya LUT Generation

This guide explains how to add support for new activation functions to the Sollya-based polynomial LUT generation system.

## Quick Reference

**16 activations currently unsupported** (out of 30 total):
- Convergence failures: `atanh`, `erf`, `sinh`
- Missing definitions: `digamma`, `elu`, `hardshrink`, `hardsigmoid`, `hardswish`, `hardtanh`, `leaky_relu`, `prelu`, `relu`, `relu6`, `selu`, `softshrink`, `threshold`

## Table of Contents

1. [Overview](#overview)
2. [Method 1: Piecewise Definitions (Recommended)](#method-1-piecewise-definitions-recommended)
3. [Method 2: Direct Sollya Expressions](#method-2-direct-sollya-expressions)
4. [Method 3: External C Functions](#method-3-external-c-functions-advanced)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The Sollya LUT generation system supports three methods for adding activation functions:

| Method | Use Case | Difficulty | Example |
|--------|----------|-----------|---------|
| **Piecewise Definitions** | Functions with different behaviors in different regions | Easy | ReLU, ELU, HardTanh |
| **Direct Expressions** | Smooth functions using Sollya built-ins | Easy | Sigmoid, Tanh, GELU |
| **External C** | Functions not available in Sollya | Advanced | Digamma, custom functions |

---

## Method 1: Piecewise Definitions (Recommended)

### When to Use
- Functions that are linear/constant in some regions
- Functions with discontinuities or kinks
- Functions that are piecewise-defined by design (ReLU, thresholding, etc.)

### Step-by-Step Guide

#### Step 1: Define Breakpoints and Expressions

Identify the regions where your function has different behavior. For example, **ReLU** is defined as:

```
f(x) = {  0    if x ≤ 0
       {  x    if x > 0
```

#### Step 2: Add Entry to `activations_config.dat`

Edit `tt_metal/programming_examples/generic_lut_activation/activations_config.dat`:

```csv
activation,has_native_sfpu,range_min,range_max,critical_point_x,critical_point_value,notes,piecewise_breakpoints
relu,true,-10.0,10.0,,,Works on hardware - symmetric range,"-10,0,10"
```

**Format explanation:**
- `activation`: Function name (must match Python/TTNN name)
- `has_native_sfpu`: Whether hardware has native support (true/false)
- `range_min,range_max`: Input domain for the function
- `critical_point_x,critical_point_value`: Critical points (comma-separated for multiple)
- `notes`: Description/notes about the function
- `piecewise_breakpoints`: Comma-separated breakpoints for piecewise functions

#### Step 3: Verify the Definition

Run the test script to verify Sollya can process your definition:

```bash
cd tt_metal/programming_examples/generic_lut_activation
python3 luts/sollya_extended_functions.py
```

#### Step 4: Generate LUTs

```bash
cd tt_metal/programming_examples/generic_lut_activation
./generate_luts.sh --activation your_activation_name
```

### Examples

#### Example 1: Leaky ReLU

```python
# f(x) = { 0.01*x  if x ≤ 0
#        {   x     if x > 0
```

```csv
leaky_relu,-10,10,,-10|0|10,0.01 * x|x
```

#### Example 2: Hard Sigmoid

```python
# f(x) = {  0          if x < -3
#        {  (x+3)/6    if -3 ≤ x ≤ 3
#        {  1          if x > 3
```

```csv
hardsigmoid,-10,10,,-10|-3|3|10,0|(x + 3.0) / 6.0|1
```

#### Example 3: Hard Tanh

```python
# f(x) = { -1   if x < -1
#        {  x   if -1 ≤ x ≤ 1
#        {  1   if x > 1
```

```csv
hardtanh,-10,10,,-10|-1|1|10,-1|x|1
```

---

## Method 2: Direct Sollya Expressions

### When to Use
- Smooth functions everywhere (no discontinuities)
- Functions that can be expressed using Sollya's built-in functions

### Sollya Built-in Functions

```
exp(x), log(x), sin(x), cos(x), tan(x)
sinh(x), cosh(x), tanh(x)
atan(x), asin(x), acos(x)
sqrt(x), abs(x)
erf(x)  # May have convergence issues on large domains
```

### Step-by-Step Guide

#### Step 1: Express Function in Sollya Syntax

Write your function using Sollya's syntax. For example:

```python
# Sigmoid
'1 / (1 + exp(-x))'

# Swish
'x / (1 + exp(-x))'

# GELU
'0.5 * x * (1 + erf(x / sqrt(2)))'

# Mish
'x * tanh(log(1 + exp(x)))'
```

#### Step 2: Add to `sollya_bf16_remez.py`

Edit `luts/sollya_bf16_remez.py` and add your function to the `func_map` dictionary (line ~128):

```python
func_map = {
    # Existing functions...
    'sigmoid': '1 / (1 + exp(-x))',

    # Add your new function:
    'your_function': 'your_sollya_expression_here',
}
```

#### Step 3: Add to Activation List

Add your function name to `activations_config.py`:

```python
ALL_ACTIVATIONS = [
    'atanh', 'celu', 'cos', # ... existing activations
    'your_function',  # Add here
]
```

#### Step 4: Add Range to Config

Add an entry to `activations_config.dat` with just the range:

```csv
your_function,true,min_x,max_x,,,Custom function description,
```

Leave `critical_point_x`, `critical_point_value`, and `piecewise_breakpoints` empty for smooth functions.

### Example: Custom Activation

Let's add a "Soft ReLU" activation: `f(x) = log(1 + exp(x))` (also known as Softplus)

```python
# 1. Add to sollya_bf16_remez.py
'soft_relu': 'log(1 + exp(x))'

# 2. Add to activations_config.py
ALL_ACTIVATIONS = [..., 'soft_relu']

# 3. Add to activations_config.dat
soft_relu,-10,10,,,
```

---

## Method 3: External C Functions (Advanced)

### When to Use
- Functions not available in Sollya (digamma, bessel functions, etc.)
- Functions requiring specialized numerical methods
- Functions with domain restrictions that Sollya cannot handle

### Prerequisites

```bash
# Install Sollya development headers
sudo apt-get install libsollya-dev  # Ubuntu/Debian
# OR
brew install sollya  # macOS
```

### Step-by-Step Guide

#### Step 1: Implement C Function

Create your function in `luts/sollya_custom_functions.c`:

```c
#include <sollya.h>
#include <math.h>

int sollya_lib_your_function(sollya_obj_t result, sollya_obj_t x, ...) {
    double x_val, result_val;
    sollya_mpfi_t x_interval, result_interval;

    // Handle constant case
    if (sollya_lib_obj_is_constant(x)) {
        sollya_lib_get_constant_as_double(&x_val, x);

        // YOUR IMPLEMENTATION HERE
        result_val = your_math_function(x_val);

        *result = sollya_lib_constant_from_double(result_val);
        return 1;  // Success
    }

    // Handle interval case (for Remez algorithm)
    sollya_lib_init_mpfi(x_interval);
    sollya_lib_init_mpfi(result_interval);

    if (sollya_lib_get_interval_from_obj(x_interval, x)) {
        double x_low = sollya_lib_get_inf_as_double(x_interval);
        double x_high = sollya_lib_get_sup_as_double(x_interval);

        // Compute conservative bounds
        double result_low = your_math_function(x_low);
        double result_high = your_math_function(x_high);

        sollya_lib_set_interval_from_doubles(result_interval, result_low, result_high);
        *result = sollya_lib_obj_from_interval(result_interval);

        sollya_lib_clear_mpfi(x_interval);
        sollya_lib_clear_mpfi(result_interval);
        return 1;
    }

    sollya_lib_clear_mpfi(x_interval);
    sollya_lib_clear_mpfi(result_interval);
    return 0;  // Failure
}
```

#### Step 2: Build Shared Library

```bash
cd luts/
bash build_sollya_custom.sh
# This creates libsollya_custom_functions.so
```

#### Step 3: Create Sollya Wrapper

Create a Sollya script `your_function.sollya`:

```sollya
// Load external library
bashexecute("cd luts && ./build_sollya_custom.sh");

// Bind external function
externalproc(your_function_impl, "./luts/libsollya_custom_functions.so", (constant) -> constant);

// Create function object
f = function(your_function_impl);

// Test
print("Testing:", f(1.0));

// Use with fpminimax
p = fpminimax(f, 3, [|7,7,7,7|], [0; 5]);
print("Polynomial:", p);
```

#### Step 4: Integrate with Python

Extend `sollya_bf16_remez.py` to support external procedures:

```python
def fit_external_procedure(self, lib_path, func_name, degree, seg_start, seg_end):
    """Fit external C function using Sollya."""

    script = f"""
    externalproc({func_name}_impl, "{lib_path}", (constant) -> constant);
    f = function({func_name}_impl);
    p = fpminimax(f, {degree}, [|{','.join([str(self.mantissa_bits)]*(degree+1))}|], [{seg_start};{seg_end}]);
    print("POLYNOMIAL_START");
    print("POLY:", p);
    print("POLYNOMIAL_END");
    quit;
    """

    # Execute and parse...
```

---

## Testing

### Test Individual Activation

```bash
# Generate LUTs for one activation
cd tt_metal/programming_examples/generic_lut_activation
./generate_luts.sh --activation your_function

# Check output
ls -lh luts/piecewise_*_your_function_*.lut
```

### Test with Hardware

```bash
# Run hardware tests
cd tt_metal/programming_examples/generic_lut_activation_embedded
./sweep_embedded.sh

# Check results
cat blackhole_piecewise_cubic_results.csv | grep your_function
```

### Verify Sollya Compatibility

```bash
# Test if Sollya can process your function
cd tt_metal/programming_examples/generic_lut_activation
python3 tools/test_sollya_activations.py
```

---

## Troubleshooting

### Issue: "Sollya fpminimax failed to converge"

**Cause:** Domain too large or function too complex

**Solution:**
1. Use piecewise definitions to break into smaller segments
2. Reduce input range in `activations_config.dat`
3. Lower polynomial degree

### Issue: "Unknown function 'xxx'"

**Cause:** Function not in Sollya's `func_map`

**Solution:**
1. Add expression to `sollya_bf16_remez.py`
2. OR use piecewise definition
3. OR implement as external C function

### Issue: External library not loading

**Cause:** Sollya headers not installed or incorrect paths

**Solution:**
```bash
# Check Sollya installation
sollya --version

# Check for headers
ls /usr/include/sollya.h
ls /usr/local/include/sollya.h

# Reinstall if missing
sudo apt-get install --reinstall libsollya-dev
```

### Issue: LUT generation hangs

**Cause:** Sollya timeout on complex expression

**Solution:**
1. Simplify expression (use piecewise)
2. Increase timeout in `sollya_bf16_remez.py` (line ~180)
3. Use smaller segments

---

## Complete Example: Adding Soft Clamp

Let's add a new activation: **Soft Clamp** = `tanh(x) * 5` (output range ≈ [-5, 5])

### Step 1: Choose Method

Since `tanh(x)` is available in Sollya, use **Method 2** (Direct Expression).

### Step 2: Add to sollya_bf16_remez.py

```python
# In luts/sollya_bf16_remez.py, line ~158
func_map = {
    # ... existing functions
    'soft_clamp': 'tanh(x) * 5',
}
```

### Step 3: Add to activations_config.py

```python
# In activations_config.py
ALL_ACTIVATIONS = [
    'atanh', 'celu', # ... existing
    'soft_clamp',
]
```

### Step 4: Add to activations_config.dat

```csv
soft_clamp,-10,10,,,
```

### Step 5: Generate and Test

```bash
# Generate LUTs
./generate_luts.sh --activation soft_clamp --degree cubic --depth 16

# Verify output
ls -lh luts/piecewise_cubic_remez_soft_clamp_16_*.lut

# Test on hardware
./sweep_embedded.sh
```

Done! Your new activation is now supported.

---

## Summary

| Task | Method | Files to Edit |
|------|--------|--------------|
| Add piecewise function | Method 1 | `activations_config.dat` |
| Add Sollya-compatible function | Method 2 | `sollya_bf16_remez.py`, `activations_config.py`, `activations_config.dat` |
| Add custom C function | Method 3 | `sollya_custom_functions.c`, `sollya_bf16_remez.py` |

**Recommendation:** Start with Method 1 (piecewise) for most activations. It's the simplest and most reliable approach.

---

## References

- [Sollya Documentation](https://www.sollya.org/)
- [Sollya externalproc](https://www.sollya.org/sollya-8.0/help.php?name=externalproc)
- [Sollya fpminimax](https://www.sollya.org/sollya-8.0/help.php?name=fpminimax)
- `ground_truth.py` - PyTorch activation implementations
- `sollya_bf16_remez.py` - Sollya BF16 fitting interface

---

**Questions?** Check the existing implementations in `activations_config.dat` or `sollya_bf16_remez.py` for examples.
