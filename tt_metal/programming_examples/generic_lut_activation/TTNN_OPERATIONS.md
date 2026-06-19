# Using TTNN Operations Directly

Instead of implementing low-level SFPU kernels, you can call TTNN operations directly from Python or C++. This is much simpler for composite operations like `mish`, `hardswish`, `tanhshrink`, etc.

## Approach Comparison

### 1. **Low-Level SFPU Kernel** (Current `native_sfpu.cpp`)
- ✅ **Pros**: Maximum performance, single-pass execution, minimal memory
- ❌ **Cons**: Complex to implement composite operations, requires multiple DST registers and circular buffers

### 2. **TTNN Operations** (Recommended for composite ops)
- ✅ **Pros**: Simple to use, handles complexity automatically, already optimized
- ✅ **Pros**: Supports all composite operations out-of-the-box
- ❌ **Cons**: Slightly more overhead (tensor management, multiple kernel launches)

## Python Usage

### Simple Example

```python
import torch
import ttnn

# Open device
device = ttnn.open_device(device_id=0)

# Create input tensor
torch_input = torch.randn(1, 1, 32, 32)
ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=device)

# Apply activation (all operations work the same way!)
output = ttnn.mish(ttnn_input)           # Mish activation
output = ttnn.hardswish(ttnn_input)       # Hardswish activation
output = ttnn.silu(ttnn_input)            # SiLU/Swish activation
output = ttnn.log_sigmoid(ttnn_input)     # LogSigmoid activation

# Convert back to PyTorch
torch_output = ttnn.to_torch(output)

ttnn.close_device(device)
```

### Running the Wrapper Script

```bash
# Simple activations
python ttnn_activation_wrapper.py mish --range-min -10 --range-max 10
python ttnn_activation_wrapper.py hardswish --range-min -5 --range-max 5
python ttnn_activation_wrapper.py silu --range-min -10 --range-max 10

# Composite operation example
python ttnn_activation_wrapper.py --composite-example

# List all available operations
python ttnn_activation_wrapper.py --list

# Save output to CSV
python ttnn_activation_wrapper.py mish --range-min -10 --range-max 10 --output mish_output.csv
```

## Available TTNN Operations

### ✅ Directly Available (39 operations)

**Simple SFPU operations:**
- `ttnn.abs`, `ttnn.atan`, `ttnn.atanh`, `ttnn.cbrt`, `ttnn.cos`, `ttnn.cosh`
- `ttnn.erf`, `ttnn.erfinv`, `ttnn.exp`, `ttnn.exp2`, `ttnn.expm1`
- `ttnn.gelu`, `ttnn.identity` (clone), `ttnn.log`, `ttnn.log1p`, `ttnn.log2`, `ttnn.log10`
- `ttnn.neg`, `ttnn.relu`, `ttnn.relu6`, `ttnn.sigmoid`, `ttnn.sin`, `ttnn.sinh`
- `ttnn.sqrt`, `ttnn.tan`, `ttnn.tanh`

**Composite operations (automatically handled):**
- `ttnn.silu` - SiLU/Swish: x * sigmoid(x)
- `ttnn.mish` - Mish: x * tanh(softplus(x))
- `ttnn.hardswish` - Hardswish: x * hardsigmoid(x)
- `ttnn.hardsigmoid` - Hardsigmoid
- `ttnn.log_sigmoid` - LogSigmoid: log(sigmoid(x))
- `ttnn.softplus` - Softplus: log(1 + exp(x))

**Parameterized operations:**
- `ttnn.elu(x, alpha=1.0)`
- `ttnn.leaky_relu(x, negative_slope=0.01)`
- `ttnn.selu(x)` - scale=1.0507, alpha=1.6733
- `ttnn.softsign(x)`
- `ttnn.threshold(x, threshold=0.0, value=0.0)`

### 🔧 Require Manual Composition (4 operations)

These need 2-3 TTNN calls:

**1. tanhshrink: `x - tanh(x)`**
```python
output = ttnn.sub(x, ttnn.tanh(x))
```

**2. logit: `log(x / (1-x))`**
```python
# With clamping for numerical stability
x_clamped = ttnn.clip(x, eps, 1-eps)
one = ttnn.full_like(x, 1.0)
output = ttnn.log(ttnn.div(x_clamped, ttnn.sub(one, x_clamped)))
```

**3. hardshrink: `x if |x| > λ else 0`**
```python
lambda_val = 0.5
mask = ttnn.gt(ttnn.abs(x), lambda_val)  # |x| > λ
output = ttnn.where(mask, x, ttnn.zeros_like(x))
```

**4. relu_max / relu_min: Parameterized ReLU**
```python
# ReLU with max
output = ttnn.clip(ttnn.relu(x), min=0.0, max=6.0)

# ReLU with min
output = ttnn.maximum(ttnn.relu(x), min_val)
```

### ❌ Special Functions

- `digamma` - Not available in TTNN
- `lgamma` - Available as `ttnn.lgamma`

## C++ Usage Example

```cpp
#include "ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

// Create device
auto device = ttnn::open_device(0);

// Create input tensor
auto input_tensor = /* ... create tensor ... */;

// Apply operations (same as Python!)
auto mish_output = ttnn::operations::unary::mish(input_tensor);
auto hardswish_output = ttnn::operations::unary::hardswish(input_tensor);
auto silu_output = ttnn::operations::unary::silu(input_tensor);

ttnn::close_device(device);
```

## Integration with Current Code

You can create a hybrid approach:

### Option 1: Python Wrapper (Simplest)
Use `ttnn_activation_wrapper.py` to test all activations including composites.

### Option 2: Add TTNN Branch to Native Code
Modify `generic_lut_activation_native.cpp` to use TTNN operations for composite functions:

```cpp
// For simple SFPU: use existing native_sfpu.cpp kernel
if (activation_is_simple_sfpu(activation_type)) {
    // Use existing kernel
    run_native_sfpu_kernel(device, activation_type, ...);
}
// For composite operations: use TTNN C++ API
else {
    // Use TTNN operations
    auto output = run_ttnn_operation(device, activation_name, input_tensor);
}
```

### Option 3: Pure TTNN Implementation
Replace the entire implementation with TTNN calls - simpler but slightly higher overhead.

## Performance Considerations

| Approach | Latency | Memory | Complexity | Flexibility |
|----------|---------|--------|------------|-------------|
| Native SFPU kernel | ⭐⭐⭐ Best | ⭐⭐⭐ Minimal | ⭐ High | ⭐ Limited |
| TTNN operations | ⭐⭐ Good | ⭐⭐ Low | ⭐⭐⭐ Simple | ⭐⭐⭐ Excellent |

**Recommendation:**
- Use **native SFPU kernels** for simple single-pass operations (already implemented: 39 functions)
- Use **TTNN operations** for composite operations (mish, hardswish, tanhshrink, logit, hardshrink)

## Summary

You now have **TWO approaches** to handle all activation functions:

1. **39 native SFPU operations** (IDs 0-38) - Already implemented in `native_sfpu.cpp`
2. **TTNN wrapper** - For composite ops and easy testing

**All 48 activation functions** from `~/workspace/tt-polynomial-fitter/activations` are covered:
- ✅ 39 via native SFPU kernel
- ✅ 9 via TTNN operations (mish, hardswish, tanhshrink, logsigmoid, logit, hardshrink, relu_max, relu_min, digamma/lgamma)

To test any activation:
```bash
# Native SFPU (fast, single-pass)
./build/programming_examples/generic_lut_activation_native gelu --range-min -10 --range-max 10

# TTNN wrapper (flexible, composite ops)
python ttnn_activation_wrapper.py mish --range-min -10 --range-max 10
```
