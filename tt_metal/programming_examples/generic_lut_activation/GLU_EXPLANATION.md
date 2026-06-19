# GLU (Gated Linear Unit) Variants - Why They Don't Use LUTs

## TL;DR

**GLU variants (glu, geglu, swiglu, reglu) are NOT suitable for 1D LUT approximation** because they:
1. Require 2× input dimension (splitting)
2. Perform element-wise multiplication (gating)
3. Are architectural components, not scalar activation functions

They need **kernel implementations** instead.

---

## Standard Activations (LUT-Compatible)

### Scalar Functions
Standard activations are **univariate functions**: `f: ℝ → ℝ`

```python
# Example: Sigmoid
input:  x = 3.5
output: y = σ(3.5) = 1/(1 + e^(-3.5)) = 0.9706

# Example: GELU (scalar)
input:  x = 1.5
output: y = GELU(1.5) = 1.5 * Φ(1.5) ≈ 1.404
```

### LUT Representation
```python
# Piecewise-linear LUT with 4 segments for sigmoid
segments = [
    {'range': (-10, -5), 'slope': 0.0001, 'offset': 0.0},
    {'range': (-5, 0),   'slope': 0.25,   'offset': 0.5},
    {'range': (0, 5),    'slope': 0.25,   'offset': 0.5},
    {'range': (5, 10),   'slope': 0.0001, 'offset': 1.0}
]

# Lookup: Given x, find segment, evaluate polynomial
def lut_sigmoid(x):
    seg_idx = find_segment(x)
    return slopes[seg_idx] * x + offsets[seg_idx]
```

**Why this works**: One input → one output. Simple table lookup.

---

## GLU Variants (NOT LUT-Compatible)

### What is a Gated Linear Unit?

GLU applies a **gating mechanism** where one half of the input controls (gates) the other half:

```python
def glu(x):
    """
    Gated Linear Unit

    Input: x of shape [..., 2*n]
    Output: y of shape [..., n]
    """
    values, gates = split(x, dim=-1)  # Split into two equal parts
    return values * sigmoid(gates)     # Element-wise product
```

### Examples

#### GeGLU (GELU Gated Linear Unit)
```python
# Used in: Transformer FFN layers (GLU variants paper, 2020)
input:  x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # Shape: (6,)

# Step 1: Split in half
values = [1.0, 2.0, 3.0]                     # First half
gates  = [4.0, 5.0, 6.0]                     # Second half

# Step 2: Apply GELU to gates
activated = GELU([4.0, 5.0, 6.0]) = [3.999, 4.999, 5.999]

# Step 3: Element-wise multiply
output = values * activated = [3.999, 9.998, 17.997]  # Shape: (3,)
```

#### SwiGLU (Swish Gated Linear Unit)
```python
# Used in: LLaMA, PaLM models
input:  x = [2.0, -1.0, 3.0, 0.5, 1.5, -0.5]  # Shape: (6,)

values = [2.0, -1.0, 3.0]    # First half
gates  = [0.5, 1.5, -0.5]    # Second half

activated = Swish(gates) = [0.311, 1.350, -0.189]
output = [0.622, -1.350, -0.567]  # Shape: (3,)
```

---

## Why GLU Can't Use 1D LUTs

### Problem 1: Not a Scalar Function

```python
# Standard activation (LUT works):
sigmoid(x) : ℝ → ℝ
Input:  Single number
Output: Single number

# GLU variant (LUT doesn't work):
glu(x) : ℝ^(2n) → ℝ^n
Input:  Vector of even length
Output: Vector of half the length
```

**You can't represent a vector-to-vector operation with a scalar lookup table.**

### Problem 2: Context-Dependent Output

```python
# Same gate value, different outputs based on paired value:

# Case 1:
values = [10.0], gates = [2.0]
output = 10.0 * sigmoid(2.0) = 10.0 * 0.88 = 8.8

# Case 2:
values = [5.0], gates = [2.0]  # Same gate!
output = 5.0 * sigmoid(2.0) = 5.0 * 0.88 = 4.4

# Problem: Same input (gate=2.0) gives different outputs
#          depending on the paired value!
```

A 1D LUT must return the **same output for the same input**. GLU violates this because output depends on TWO inputs (value AND gate).

### Problem 3: Requires Splitting

```python
# Standard activation:
y = activation(x)  # Just apply function

# GLU:
x_split = split_tensor(x, dim=-1)     # Must split first
values, gates = x_split[0], x_split[1]
y = values * activation(gates)         # Then multiply
```

**Splitting is a tensor operation, not a mathematical function.** You can't encode "split this tensor" in a LUT.

### Problem 4: Element-wise Multiplication

```python
# GLU requires element-wise product:
output[i] = values[i] * activation(gates[i])

# This means:
# - For each position i, you need BOTH values[i] AND gates[i]
# - The activation is only applied to gates
# - Then you multiply, which is a separate operation

# A LUT can only represent: x → f(x)
# Not: (x, y) → x * f(y)
```

---

## Correct Implementation: Kernel

GLU variants require **kernel implementations** because they need:
1. **Tensor splitting** - Divide input into values and gates
2. **Selective activation** - Apply activation only to gates
3. **Element-wise operations** - Multiply values by activated gates

### Example TTNN Kernel Structure

```cpp
// Pseudo-code for GeGLU kernel
__global__ void geglu_kernel(float* input, float* output, int n) {
    int idx = get_global_id();

    // Split: first half are values, second half are gates
    float value = input[idx];
    float gate = input[idx + n];

    // Apply GELU to gate only
    float activated_gate = gelu(gate);

    // Multiply
    output[idx] = value * activated_gate;
}
```

---

## Why We Added GLU to ground_truth.py

Even though GLU can't use LUTs, we added them for:
1. **Completeness** - Catalog all activation functions
2. **Reference implementation** - Show correct mathematical definition
3. **Future work** - May implement as native TTNN operations

**Result**: GLU generation fails (expected), only constant LUT created.

---

## Summary Table

| Activation Type | Scalar | LUT Compatible | Implementation |
|-----------------|--------|----------------|----------------|
| sigmoid, tanh, ReLU, GELU | Yes | ✅ Yes | LUT approximation |
| GLU, GeGLU, SwiGLU, ReGLU | No (vector) | ❌ No | Kernel required |
| Attention, LayerNorm | No (tensor op) | ❌ No | Kernel required |

---

## References

- **GLU Variants Paper**: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (2020)
- **SwiGLU in LLaMA**: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (2023)
- **PaLM**: [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) (2022)

---

## LUT Generation Results

```bash
$ ls luts/*glu*.lut
luts/geglu_1024.lut    # Only constant LUT (1D lookup)
luts/glu_1024.lut      # Piecewise methods fail (expected)
luts/reglu_1024.lut
luts/swiglu_1024.lut
```

**Status**: Working as intended. GLU variants need different implementation approach.
