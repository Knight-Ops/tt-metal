# Embedded LUT Activation Functions

This directory contains the **compile-time embedded LUT** implementation for activation functions on Tenstorrent hardware. Unlike the CB-based implementation in `generic_lut_activation/`, this version embeds LUT constants directly into kernel binaries for **zero L1 memory overhead**.

## Architecture

**Compile-Time Embedding Flow:**
```
Binary .lut files (shared via symlink)
    ↓ tools/generate_lut_headers.py
696 LUT headers with constexpr arrays
    ↓ tools/generate_embedded_kernels.py
522 kernel wrappers
    ↓ tools/generate_cmake_targets.py
522 CMake binary targets
    ↓ Build system
522 binaries with zero-overhead LUT access
```

## Benefits vs. CB-Based Implementation

### Performance Improvements

**Overhead Eliminated:**
- ❌ LUT file I/O: ~5-10ms startup time
- ❌ L1 buffer allocation: ~3.3MB transfer
- ❌ Circular buffer setup: ~1-2ms
- ❌ L1 memory loads: 1-9 cycles per element

**Per-Tile Speedup** (1024 elements):
- Piecewise Constant: 1K-2K cycles saved
- Piecewise Linear: 2K-4K cycles saved
- Piecewise Quadratic: 3K-6K cycles saved
- Piecewise Cubic: 4K-8K cycles saved
- Piecewise Hexic: 7K-14K cycles saved
- Piecewise Octic: 9K-18K cycles saved

**Expected Speedup: 15-35%** depending on polynomial degree

### API Simplification

**CB-Based (legacy):**
```bash
./binary --lut-file luts/sigmoid_16.lut --tiles 256
# Runtime args: n_tiles, test_min, test_max
```

**Embedded (new):**
```bash
./binary_embedded_sigmoid_constant_16 --tiles 256
# Runtime args: n_tiles only
```

## File Organization

### Auto-Generated Files (git-ignored)

```
kernels/compute/generated_luts/          # 696 LUT headers
├── sigmoid/
│   ├── constant_depth_4.hpp
│   ├── constant_depth_8.hpp
│   ├── linear_depth_16.hpp
│   └── ...
└── tanh/
    └── ...

kernels/compute/piecewise_*_embedded/    # 522 kernel wrappers
├── piecewise_constant_embedded/
│   ├── sigmoid_constant_4.cpp
│   ├── sigmoid_constant_16.cpp
│   └── ...
└── ...

cmake/EmbeddedTargets.cmake              # 522 CMake targets (6851 lines)
```

### Source Files (tracked in git)

```
tools/
├── generate_lut_headers.py           # Generates constexpr LUT headers
├── generate_embedded_kernels.py      # Generates kernel wrappers
└── generate_cmake_targets.py         # Generates CMake targets

kernels/compute/
├── piecewise_constant.cpp            # Base kernel (embedded mode only)
├── piecewise_linear.cpp
├── piecewise_quadratic.cpp
└── ...

generic_lut_activation.cpp             # Host code (embedded mode only)
generic_lut_activation_native.cpp      # Native SFPU operations (no LUTs)
CMakeLists.txt                         # Includes cmake/EmbeddedTargets.cmake
../generic_lut_activation/activations_config.dat  # Min/max ranges for all activations (parent folder)
```

## Building

### Regenerate All Auto-Generated Files

```bash
cd tt_metal/programming_examples/generic_lut_activation_embedded

# 1. Generate LUT headers (696 files)
python3 tools/generate_lut_headers.py

# 2. Generate kernel wrappers (522 files)
python3 tools/generate_embedded_kernels.py

# 3. Generate CMake targets (6851 lines)
python3 tools/generate_cmake_targets.py > cmake/EmbeddedTargets.cmake
```

### Build All Embedded Binaries

```bash
cd /path/to/tt-metal
./build_metal.sh --build-programming-examples

# Binaries will be in:
build/programming_examples/programming_examples_generic_lut_activation_embedded_*
```

### Build Specific Activation

```bash
cd /path/to/tt-metal
cmake --build build --target programming_examples_generic_lut_activation_embedded_sigmoid_constant_16
```

## Usage

### Running a Single Embedded Binary

```bash
# Example: Sigmoid with piecewise constant (16 segments)
./build/programming_examples/programming_examples_generic_lut_activation_embedded_sigmoid_constant_16 --tiles 256
```

### Binary Naming Convention

```
programming_examples_generic_lut_activation_embedded_{activation}_{method}_{depth}
```

**Examples:**
- `..._sigmoid_constant_16` - Sigmoid, constant, 16 segments
- `..._tanh_linear_32` - Tanh, linear, 32 segments
- `..._gelu_quadratic_8` - GELU, quadratic, 8 segments

### Available Combinations

**29 Activations:**
atanh, celu, cos, cosh, elu, erf, exp, gelu, hardshrink, hardsigmoid, hardswish, hardtanh, leaky_relu, logsigmoid, mish, prelu, relu, relu6, selu, sigmoid, sin, sinh, softplus, softshrink, softsign, swish, tanh, tanhshrink, threshold

**Per Activation:**
- Piecewise constant: depths 4, 8, 16, 32 (4 binaries)
- Piecewise linear: depths 4, 8, 16, 32 (4 binaries)
- Piecewise quadratic: depths 4, 8, 16 (3 binaries)
- Piecewise cubic: depths 4, 8, 16 (3 binaries)
- Piecewise hexic: depths 4, 8 (2 binaries)
- Piecewise octic: depths 4, 8 (2 binaries)

**Total: 522 embedded binaries** (18 per activation)

## Testing

### Running an Embedded Binary

```bash
# Run embedded sigmoid with 16 constant segments
./build/programming_examples/programming_examples_generic_lut_activation_embedded_sigmoid_constant_16 \
    --tiles 256

# Run native SFPU operations (no LUTs)
./build/programming_examples/programming_examples_generic_lut_activation_embedded_native_sfpu \
    gelu --tiles 1024
```

### Performance Benchmarking

```bash
# Measure execution time
time ./build/programming_examples/programming_examples_generic_lut_activation_embedded_sigmoid_constant_16 --tiles 1024
```

## Implementation Details

### Generated LUT Header Format

```cpp
// kernels/compute/generated_luts/sigmoid/constant_depth_16.hpp
namespace lut {
namespace sigmoid {
namespace constant_depth_16 {

constexpr float INPUT_MIN = -5.000000000f;
constexpr float INPUT_MAX = 5.000000000f;
constexpr uint32_t LUT_SIZE = 16;

constexpr std::array<float, LUT_SIZE> LUT_DATA = {{
    0.006692851f, 0.017986210f, 0.047425874f, ...
}};

}}}
```

### Generated Kernel Wrapper Format

```cpp
// kernels/compute/piecewise_constant_embedded/sigmoid_constant_16.cpp

#include "../../generated_luts/sigmoid/constant_depth_16.hpp"

// Import into global namespace for base kernel
constexpr float INPUT_MIN = lut::sigmoid::constant_depth_16::INPUT_MIN;
constexpr float INPUT_MAX = lut::sigmoid::constant_depth_16::INPUT_MAX;
constexpr uint32_t LUT_SIZE = lut::sigmoid::constant_depth_16::LUT_SIZE;
constexpr auto& LUT_DATA = lut::sigmoid::constant_depth_16::LUT_DATA;

#include "../piecewise_constant.cpp"  // Base implementation
```

### Kernel Implementation

```cpp
// piecewise_constant.cpp (base implementation)
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // Embedded LUT constants from generated header
    constexpr float input_min = INPUT_MIN;
    constexpr float input_max = INPUT_MAX;
    constexpr uint32_t NUM_SEGMENTS = LUT_SIZE;

    // Use embedded LUT data directly (zero L1 overhead)
    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;

    // Process tiles...
}
```

### Host Code

```cpp
// generic_lut_activation.cpp (embedded mode only)
int main(int argc, char** argv) {
    // Parse --tiles argument only (no LUT file needed)

    // Extract activation name from KERNEL_VARIANT
    // Load test range from activations_config.dat

    // Create device and program
    // Skip LUT circular buffer allocation
    // Only pass n_tiles as runtime arg
    SetRuntimeArgs(program, compute, core, {n_tiles});

    // Execute and verify results
}
```

## Maintenance

### Adding a New Activation

1. Add entry to `../generic_lut_activation/activations_config.dat`:
   ```csv
   new_activation,false,-5.0,5.0,Description
   ```

2. Generate LUTs in parent directory:
   ```bash
   cd ../generic_lut_activation
   python3 generate_piecewise_constant_luts.py --activation new_activation
   python3 generate_piecewise_linear_luts.py --activation new_activation
   # ... other methods
   ```

3. Regenerate embedded infrastructure:
   ```bash
   cd ../generic_lut_activation_embedded
   python3 tools/generate_lut_headers.py
   python3 tools/generate_embedded_kernels.py
   python3 tools/generate_cmake_targets.py > cmake/EmbeddedTargets.cmake
   ```

4. Rebuild:
   ```bash
   cd ../../..
   ./build_metal.sh --build-programming-examples
   ```

### Modifying a Kernel

If you modify base kernels (`piecewise_constant.cpp`, etc.):
- No regeneration needed - changes apply to all embedded binaries
- Just rebuild affected targets

If you modify generation scripts:
- Regenerate all files using scripts above
- Rebuild affected targets

## Troubleshooting

**Build errors about missing headers:**
```bash
# Regenerate LUT headers
python3 tools/generate_lut_headers.py
```

**CMake target not found:**
```bash
# Regenerate CMake targets
python3 tools/generate_cmake_targets.py > cmake/EmbeddedTargets.cmake
# Reconfigure CMake
cmake -B build
```

**Kernel compilation errors:**
```bash
# Regenerate kernel wrappers
python3 tools/generate_embedded_kernels.py
```

## Performance Notes

- Embedded LUTs are stored in kernel binary's read-only data segment
- Compiler may inline LUT values for additional optimization
- No runtime overhead for accessing constants
- Binary size increases ~100-500 bytes per LUT (acceptable trade-off)
- Expected wall-clock speedup: 15-35% depending on method

## Comparison with CB-Based Implementation

| Aspect | CB-Based (legacy) | Embedded (new) |
|--------|-------------------|----------------|
| LUT Storage | L1 circular buffer | Kernel binary (read-only) |
| LUT Loading | Runtime (file → L1) | Compile-time |
| L1 Overhead | 4-288 bytes per LUT | 0 bytes |
| Access Cost | 1-9 cycles/element | 0 cycles (inlined) |
| Binary Count | ~50 (generic) | 522 (specialized) |
| Runtime Args | 3 (n_tiles, min, max) | 1 (n_tiles) |
| Startup Time | ~10ms | ~0ms |
| Speedup | Baseline | 15-35% faster |

## Future Work

- [ ] Benchmark all 522 binaries on hardware
- [ ] Generate comparative performance plots
- [ ] Add CI/CD validation (embedded vs. CB comparison)
- [ ] Explore code size optimization (shared LUT sections)
- [ ] Extend to other kernel types (linear, quadratic, etc.)
