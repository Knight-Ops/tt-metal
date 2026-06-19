#!/bin/bash
# Mockup script to test LUT activation kernels on Polaris/ttsim for Quasar
#
# This script demonstrates how to:
# 1. Compile kernels for Quasar architecture
# 2. Extract ELF files from tt-metal JIT compilation
# 3. Configure and run polaris ttsim for Tensix Neo simulation
#
# Prerequisites:
# - tt-metal built with programming examples
# - polaris repository at /localdev/nkapre/polaris
# - Conda environment with polaris dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="/localdev/nkapre/tt-metal"
POLARIS_ROOT="/localdev/nkapre/polaris/polaris"
WORK_DIR="$SCRIPT_DIR"

# Create output directories
mkdir -p "$WORK_DIR/elfs"
mkdir -p "$WORK_DIR/config"
mkdir -p "$WORK_DIR/output"

echo "=============================================="
echo "Polaris/ttsim Quasar Test for LUT Activation"
echo "=============================================="
echo ""

# Step 1: Check if we have ELF files
# NOTE: tt-metal JIT compiles kernels at runtime. To get ELFs:
# 1. Run with TT_METAL_SKIP_DELETING_BUILT_CACHE=1
# 2. Find ELFs in the cache directory
# 3. Or compile manually using riscv-tt-elf-g++

echo "Step 1: Checking for ELF files..."
echo "-------------------------------------------"

# The ELF files should be for Quasar architecture
# Expected files:
# - thread_0.elf (data movement kernel - brisc)
# - thread_1.elf (data movement kernel - ncrisc)
# - thread_2.elf (compute kernel - trisc0/unpack)
# - thread_3.elf (compute kernel - trisc1/math) - for quasar with 4 threads

if [[ ! -f "$WORK_DIR/elfs/thread_0.elf" ]]; then
    echo "WARNING: ELF files not found in $WORK_DIR/elfs/"
    echo ""
    echo "To generate ELF files for Quasar, you need to either:"
    echo ""
    echo "Option A: Extract from tt-metal JIT cache (if running on actual hardware)"
    echo "  export TT_METAL_SKIP_DELETING_BUILT_CACHE=1"
    echo "  export TT_METAL_CACHE=/tmp/quasar-cache"
    echo "  # Run your program, then find ELFs in the cache"
    echo ""
    echo "Option B: Compile manually using the SFPI toolchain"
    echo "  COMPILER=$TT_METAL_ROOT/runtime/sfpi/compiler/bin/riscv-tt-elf-g++"
    echo "  # Compile with quasar-specific flags"
    echo ""
    echo "Option C: Use pre-compiled test ELFs from polaris RTL test data"
    echo "  See: $POLARIS_ROOT/tools/run_rtl_neosim_correlation.py"
    echo ""
    echo "For now, creating placeholder config..."
fi

# Step 2: Create polaris input configuration
echo ""
echo "Step 2: Creating polaris input configuration..."
echo "-------------------------------------------"

cat > "$WORK_DIR/config/gelu_linear_4_inputcfg.json" << 'EOF'
{
  "arch": "ttqs",
  "llkVersionTag": "sep23",
  "debug": 15,
  "numTCores": 1,
  "input": {
    "syn": 0,
    "name": "gelu-linear-4-lut-activation",
    "tc0": {
      "numThreads": 4,
      "startFunction": "main",
      "th0Elf": "thread_0.elf",
      "th0Path": "WORK_DIR_PLACEHOLDER/elfs",
      "th1Elf": "thread_1.elf",
      "th1Path": "WORK_DIR_PLACEHOLDER/elfs",
      "th2Elf": "thread_2.elf",
      "th2Path": "WORK_DIR_PLACEHOLDER/elfs",
      "th3Elf": "thread_3.elf",
      "th3Path": "WORK_DIR_PLACEHOLDER/elfs"
    }
  },
  "description": {
    "kernel": "gelu_linear_4",
    "activation": "gelu",
    "degree": "linear (p=1)",
    "segments": 4,
    "source": "tt-metal generic_lut_activation_embedded"
  }
}
EOF

# Replace placeholder with actual path
sed -i "s|WORK_DIR_PLACEHOLDER|$WORK_DIR|g" "$WORK_DIR/config/gelu_linear_4_inputcfg.json"

echo "Created: $WORK_DIR/config/gelu_linear_4_inputcfg.json"

# Step 3: Show how to run polaris
echo ""
echo "Step 3: Running polaris ttsim..."
echo "-------------------------------------------"

echo "To run the simulation:"
echo ""
echo "  cd $POLARIS_ROOT"
echo "  conda activate polaris  # or polarisdev"
echo ""
echo "  python -m ttsim.back.tensix_neo.tneoSim \\"
echo "    --inputcfg $WORK_DIR/config/gelu_linear_4_inputcfg.json \\"
echo "    --cfg $POLARIS_ROOT/config/tensix_neo/ttqs_neo4_sep23.json \\"
echo "    --memoryMap $POLARIS_ROOT/config/tensix_neo/ttqs_memory_map_sep23.json \\"
echo "    --ttISAFileName $POLARIS_ROOT/ttsim/config/llk/instruction_sets/ttqs/assembly.yaml \\"
echo "    --odir $WORK_DIR/output \\"
echo "    --exp gelu_linear_4"
echo ""

# Step 4: Verify polaris config files exist
echo ""
echo "Step 4: Verifying polaris configuration files..."
echo "-------------------------------------------"

POLARIS_CONFIGS=(
    "$POLARIS_ROOT/config/tensix_neo/ttqs_neo4_sep23.json"
    "$POLARIS_ROOT/config/tensix_neo/ttqs_memory_map_sep23.json"
    "$POLARIS_ROOT/ttsim/config/llk/instruction_sets/ttqs/assembly.yaml"
)

ALL_FOUND=true
for cfg in "${POLARIS_CONFIGS[@]}"; do
    if [[ -f "$cfg" ]]; then
        echo "✓ Found: $(basename $cfg)"
    else
        echo "✗ Missing: $cfg"
        ALL_FOUND=false
    fi
done

if [[ "$ALL_FOUND" == "true" ]]; then
    echo ""
    echo "✓ All polaris configuration files found!"
else
    echo ""
    echo "⚠ Some configuration files are missing."
    echo "  Check if polaris is properly set up at $POLARIS_ROOT"
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Generate/obtain ELF files for your kernel"
echo "2. Place them in: $WORK_DIR/elfs/"
echo "3. Run the polaris command shown above"
echo ""
