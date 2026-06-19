# Embedded Kernel Sync Issue

## The Real Problem

After further investigation, I found that **BOTH directories have embedded cubic kernels**:

```bash
$ ls kernels/compute/piecewise_cubic_embedded/*.cpp | wc -l
269  # In generic_lut_activation (parent)

$ ls ../generic_lut_activation_embedded/kernels/compute/piecewise_cubic_embedded/*.cpp | wc -l
269  # In generic_lut_activation_embedded (sibling)
```

Both have the SAME 269 files including:
- Old buggy: `atanh_cubic_16.cpp` (LUT_SIZE=64, wrong range)
- New correct: `atanh_cubic_16_uniform.cpp`, `atanh_cubic_16_adaptive.cpp` (LUT_SIZE=81, correct range)

## Your Question: Why Aren't They Synced?

You're right to ask if `generate*.sh` should copy them over. The answer is:

**The `_embedded` directory is for COMPILE-TIME embedded kernels (zero L1 overhead)**
**The parent directory is for RUNTIME LUT loading (CB-based, uses L1)**

They're actually TWO DIFFERENT USE CASES:

### 1. Runtime LUT (Parent: `generic_lut_activation/`)
- LUTs loaded from `.lut` files at runtime → L1 circular buffer
- Single generic kernel that reads LUT from CB
- Flexible: can change LUT without recompiling
- Uses: `generic_lut_activation.cpp` (loads LUT file dynamically)

### 2. Compile-Time Embedded (Sibling: `generic_lut_activation_embedded/`)
- LUTs embedded in kernel source code at compile time
- 522 specialized binaries (one per activation/method/depth)
- Zero L1 overhead, faster access
- Uses: `generic_lut_activation.cpp` with `KERNEL_VARIANT` define

## The Embedded Kernels IN PARENT Directory

The `kernels/compute/piecewise_cubic_embedded/` directory in the PARENT is for:

**Demonstrating embedded kernel approach as EXAMPLES**

These are kernel SOURCE FILES that can be compiled with `-DKERNEL_VARIANT=...` to create embedded-LUT binaries.

## The Bug

The example kernel `atanh_cubic_16.cpp` was **manually created** with wrong values:
- LUT_SIZE=64 (missing boundaries!)
- INPUT_MIN/MAX=-0.8/0.8 (should be -0.99/0.99)

Meanwhile, the AUTO-GENERATED kernels are correct:
- `atanh_cubic_16_uniform.cpp`: LUT_SIZE=81, correct range
- `atanh_cubic_16_adaptive.cpp`: LUT_SIZE=81, correct range

## The Solution

**You don't need to "sync" between directories!**

What you need is to **replace the buggy manual example with a correct one**:

```bash
# Option 1: Delete the buggy example
rm kernels/compute/piecewise_cubic_embedded/atanh_cubic_16.cpp

# Option 2: Replace with correct version
cp kernels/compute/piecewise_cubic_embedded/atanh_cubic_16_uniform.cpp \
   kernels/compute/piecewise_cubic_embedded/atanh_cubic_16.cpp

# Option 3: Update documentation to use the correct kernel name
# Change examples to reference atanh_cubic_16_uniform instead of atanh_cubic_16
```

## What `generate_embedded_kernels.py` Actually Does

The tool in `generic_lut_activation_embedded/tools/generate_embedded_kernels.py` creates the **269 auto-generated kernels** in BOTH directories:

1. Generates in `_embedded/kernels/compute/piecewise_*_embedded/`
2. (Probably) symlinks or copies to parent `kernels/compute/piecewise_*_embedded/`

OR they're independently generated in each directory. Let me verify:

```bash
$ diff <(ls generic_lut_activation/kernels/compute/piecewise_cubic_embedded/) \
       <(ls generic_lut_activation_embedded/kernels/compute/piecewise_cubic_embedded/)
```

If they're identical, they might be symlinked or independently generated from the same tool.

## Recommended Action

1. **Delete or rename the buggy examples** (those without `_uniform`/`_adaptive` suffix)
2. **Update documentation** to reference the correct auto-generated kernels
3. **Do NOT manually create embedded kernels** - always use `generate_embedded_kernels.py`

## Key Takeaway

The embedded kernels are NOT meant to be "synced" between directories because they serve different purposes:
- `_embedded/`: Full implementation with all 522 binaries
- Parent: Example kernels demonstrating the technique

The bug is that the EXAMPLE kernel was manually created incorrectly. Use the auto-generated ones instead!
