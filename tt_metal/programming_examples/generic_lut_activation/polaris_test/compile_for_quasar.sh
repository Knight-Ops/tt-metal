#!/bin/bash
# Manual kernel compilation for Quasar architecture
#
# This script compiles the piecewise_generic compute kernel for Quasar
# without requiring a device or simulator to be available.
#
# The fundamental problem: tt-metal's JIT requires device initialization
# before compiling kernels. Quasar has no physical hardware and no ttsim
# support, so we need to compile manually.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="/localdev/nkapre/tt-metal"
OUT_DIR="$SCRIPT_DIR/quasar_elfs"

# Compiler
GXX="$TT_METAL_ROOT/runtime/sfpi/compiler/bin/riscv-tt-elf-g++"

# Check if compiler exists
if [[ ! -f "$GXX" ]]; then
    echo "ERROR: Compiler not found at $GXX"
    echo "Make sure tt-metal is built with SFPI toolchain"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "=============================================="
echo "Manual Quasar Kernel Compilation"
echo "=============================================="
echo "Compiler: $GXX"
echo "Output: $OUT_DIR"
echo ""

# Common include paths for quasar
INCLUDES=(
    "-I$TT_METAL_ROOT/tt_metal/hw/ckernels/blackhole/metal/common"
    "-I$TT_METAL_ROOT/tt_metal/hw/ckernels/blackhole/metal/llk_io"
    "-I$TT_METAL_ROOT/tt_metal/hw/ckernels/blackhole/metal/llk_api"
    "-I$TT_METAL_ROOT/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"
    "-I$TT_METAL_ROOT/tt_metal/hw/inc/internal/tt-2xx"
    "-I$TT_METAL_ROOT/tt_metal/hw/inc/internal/tt-2xx/quasar"
    "-I$TT_METAL_ROOT/tt_metal/hw/inc/internal/tt-2xx/quasar/quasar_defines"
    "-I$TT_METAL_ROOT/tt_metal/hw/inc/internal/tt-2xx/quasar/noc"
    "-I$TT_METAL_ROOT/tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc"
    "-I$TT_METAL_ROOT/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib"
    "-I$TT_METAL_ROOT/tt_metal/hw/firmware/src/tt-2xx"
    "-I$TT_METAL_ROOT/tt_metal/hw/inc"
    "-I$TT_METAL_ROOT/tt_metal/api/tt-metalium"
    "-I$TT_METAL_ROOT/runtime/sfpi/include"
    "-I$TT_METAL_ROOT/tt_metal/include"
    "-I$TT_METAL_ROOT/tt_metal/include/compute_kernel_api"
)

# Common defines for quasar
DEFINES=(
    "-DARCH_QUASAR"
    "-DTENSIX_FIRMWARE"
    "-DLOCAL_MEM_EN=0"
    "-DDEBUG_PRINT_ENABLED=0"
    "-DUCK_CHLKC_MATH=0"
    "-DUCK_CHLKC_PACK=0"
    "-DUCK_CHLKC_UNPACK=0"
)

# Compiler flags
CFLAGS=(
    "-std=c++17"
    "-fno-exceptions"
    "-fno-rtti"
    "-ffreestanding"
    "-ffunction-sections"
    "-fdata-sections"
    "-Os"
    "-g"
)

# Linker flags for compute kernel (trisc)
TRISC_LDFLAGS=(
    "-T$TT_METAL_ROOT/runtime/hw/toolchain/quasar/kernel_trisc.ld"
    "-nostartfiles"
    "-nostdlib"
    "-Wl,--gc-sections"
    "-Wl,--emit-relocs"
)

# Print compiler version
echo "Compiler version:"
$GXX --version | head -1
echo ""

# The piecewise_generic compute kernel source
KERNEL_SRC="$TT_METAL_ROOT/tt_metal/programming_examples/generic_lut_activation/kernels/compute/piecewise_generic.cpp"

if [[ ! -f "$KERNEL_SRC" ]]; then
    echo "ERROR: Kernel source not found: $KERNEL_SRC"
    exit 1
fi

echo "Compiling kernel: $KERNEL_SRC"
echo ""

# Compile command (just show it first)
echo "Compile command:"
echo "$GXX ${CFLAGS[*]} ${DEFINES[*]} ${INCLUDES[*]} -c -o $OUT_DIR/piecewise_generic.o $KERNEL_SRC"
echo ""

# Try to compile
echo "Attempting compilation..."
$GXX ${CFLAGS[*]} ${DEFINES[*]} ${INCLUDES[*]} -c -o "$OUT_DIR/piecewise_generic.o" "$KERNEL_SRC" 2>&1 || {
    echo ""
    echo "Compilation failed. This is expected - the kernel requires many tt-metal"
    echo "runtime definitions that are only available during JIT compilation."
    echo ""
    echo "The issue is that tt-metal's kernel compilation is tightly integrated with"
    echo "the runtime and requires HAL initialization which needs a device."
    echo ""
    echo "Options to get quasar ELFs:"
    echo "1. Request ttsim quasar support: https://github.com/tenstorrent/ttsim/issues"
    echo "2. Use polaris RTL test data (requires Tailscale access)"
    echo "3. Modify tt-metal to add compile-only mode for quasar"
    exit 1
}

echo "Compilation successful!"
echo "Object file: $OUT_DIR/piecewise_generic.o"
