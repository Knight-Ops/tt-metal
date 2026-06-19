#!/bin/bash
# Setup and run quasar simulation using ttsim-private
#
# This script:
# 1. Clones ttsim-private if not present
# 2. Builds the quasar simulator (libttsim.so)
# 3. Sets up the simulator directory with SOC descriptor
# 4. Runs the generic_lut_activation test on quasar simulator
# 5. Extracts the generated ELFs for polaris/neosom

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="/localdev/nkapre/tt-metal"
TTSIM_DIR="/localdev/nkapre/ttsim-private"
SIM_DIR="$HOME/sim/qsr"
ELF_OUTPUT_DIR="$SCRIPT_DIR/quasar_elfs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Quasar Simulation Setup for generic_lut_activation"
echo "=============================================="
echo ""

# Step 1: Clone ttsim-private if not present
echo -e "${YELLOW}Step 1: Checking ttsim-private...${NC}"
if [[ ! -d "$TTSIM_DIR" ]]; then
    echo "Cloning ttsim-private..."
    cd /localdev/nkapre
    git clone git@github.com:tenstorrent/ttsim-private.git
    echo -e "${GREEN}Cloned ttsim-private${NC}"
else
    echo -e "${GREEN}ttsim-private already exists at $TTSIM_DIR${NC}"
    # Update to latest
    echo "Pulling latest changes..."
    cd "$TTSIM_DIR"
    git pull || echo "Warning: Could not pull latest changes"
fi
echo ""

# Step 2: Build quasar simulator
echo -e "${YELLOW}Step 2: Building quasar simulator...${NC}"
cd "$TTSIM_DIR/src"

if [[ -f "_out/release_qsr/libttsim.so" ]]; then
    echo "Quasar simulator already built. Rebuilding to ensure latest..."
fi

# Build the quasar simulator
../make.py _out/release_qsr/libttsim.so

if [[ ! -f "_out/release_qsr/libttsim.so" ]]; then
    echo -e "${RED}ERROR: Failed to build quasar simulator${NC}"
    exit 1
fi
echo -e "${GREEN}Quasar simulator built successfully${NC}"
echo ""

# Step 3: Setup simulator directory
echo -e "${YELLOW}Step 3: Setting up simulator directory...${NC}"
mkdir -p "$SIM_DIR"
cp "$TTSIM_DIR/src/_out/release_qsr/libttsim.so" "$SIM_DIR/"
cp "$TT_METAL_ROOT/tt_metal/soc_descriptors/quasar_1_arch.yaml" "$SIM_DIR/soc_descriptor.yaml"
echo -e "${GREEN}Simulator directory setup at $SIM_DIR${NC}"
echo "  - libttsim.so"
echo "  - soc_descriptor.yaml"
echo ""

# Step 4: Ensure tt-metal is built
echo -e "${YELLOW}Step 4: Checking tt-metal build...${NC}"
cd "$TT_METAL_ROOT"

# Check if generic_lut_activation binary exists
if [[ ! -f "build/programming_examples/generic_lut_activation/generic_lut_activation" ]]; then
    echo "Building tt-metal with programming examples..."
    ./build_metal.sh --build-programming-examples
else
    echo -e "${GREEN}generic_lut_activation binary exists${NC}"
fi
echo ""

# Step 5: Create output directory for ELFs
echo -e "${YELLOW}Step 5: Setting up ELF output directory...${NC}"
mkdir -p "$ELF_OUTPUT_DIR"
echo -e "${GREEN}ELF output directory: $ELF_OUTPUT_DIR${NC}"
echo ""

# Step 6: Run the test on quasar simulator
echo -e "${YELLOW}Step 6: Running generic_lut_activation on quasar simulator...${NC}"
echo ""
echo "Environment:"
echo "  TT_METAL_SIMULATOR=$SIM_DIR/libttsim.so"
echo "  TT_METAL_SLOW_DISPATCH_MODE=1"
echo ""

cd "$TT_METAL_ROOT"

# Set environment and run
export TT_METAL_SIMULATOR="$SIM_DIR/libttsim.so"
export TT_METAL_SLOW_DISPATCH_MODE=1

# Also need to activate python env
if [[ -f "python_env/bin/activate" ]]; then
    source python_env/bin/activate
fi

echo "Running test..."
echo "----------------------------------------------"

# Run the test and capture output
# The JIT will compile kernels for quasar architecture
./build/programming_examples/generic_lut_activation/generic_lut_activation 2>&1 | tee "$SCRIPT_DIR/quasar_run.log"

RUN_STATUS=${PIPESTATUS[0]}

echo "----------------------------------------------"
echo ""

if [[ $RUN_STATUS -eq 0 ]]; then
    echo -e "${GREEN}Test completed successfully!${NC}"
else
    echo -e "${YELLOW}Test exited with status $RUN_STATUS (this may be expected for simulation)${NC}"
fi
echo ""

# Step 7: Find and copy generated ELFs
echo -e "${YELLOW}Step 7: Extracting generated ELFs...${NC}"

# tt-metal JIT puts compiled kernels in generated/ directory
# Look for quasar-specific ELFs
JIT_DIR="$TT_METAL_ROOT/generated"

if [[ -d "$JIT_DIR" ]]; then
    echo "Searching for kernel ELFs in $JIT_DIR..."

    # Find all .elf files generated during the run
    ELF_COUNT=0
    while IFS= read -r -d '' elf_file; do
        # Copy to output directory with descriptive name
        base_name=$(basename "$elf_file")
        rel_path=$(dirname "${elf_file#$JIT_DIR/}")

        # Create subdirectory structure
        mkdir -p "$ELF_OUTPUT_DIR/$rel_path"
        cp "$elf_file" "$ELF_OUTPUT_DIR/$rel_path/"
        echo "  Copied: $rel_path/$base_name"
        ((ELF_COUNT++))
    done < <(find "$JIT_DIR" -name "*.elf" -newer "$SCRIPT_DIR/quasar_run.log" -print0 2>/dev/null || find "$JIT_DIR" -name "*.elf" -print0 2>/dev/null)

    if [[ $ELF_COUNT -eq 0 ]]; then
        # Try finding any recent ELFs
        echo "Looking for any ELFs in generated directory..."
        find "$JIT_DIR" -name "*.elf" -type f 2>/dev/null | head -20 | while read elf_file; do
            base_name=$(basename "$elf_file")
            cp "$elf_file" "$ELF_OUTPUT_DIR/"
            echo "  Copied: $base_name"
        done
    fi

    echo ""
    echo -e "${GREEN}ELFs extracted to: $ELF_OUTPUT_DIR${NC}"
else
    echo -e "${YELLOW}Warning: generated/ directory not found${NC}"
    echo "ELFs may be in a different location"
fi
echo ""

# Step 8: Summary
echo "=============================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "Simulator location: $SIM_DIR/libttsim.so"
echo "ELF output: $ELF_OUTPUT_DIR"
echo "Run log: $SCRIPT_DIR/quasar_run.log"
echo ""
echo "KNOWN LIMITATION: Quasar simulator is early stage."
echo "tt-metal tests may fail with 'coord_to_tile: coord (0,0)' error"
echo "because quasar's functional worker is at (2,2), not (0,0)."
echo ""
echo "To run again manually:"
echo "  cd $TT_METAL_ROOT"
echo "  export TT_METAL_SIMULATOR=$SIM_DIR/libttsim.so"
echo "  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/programming_examples/programming_examples_generic_lut_activation_p1_s4 \\"
echo "      tt_metal/programming_examples/generic_lut_activation/polaris_test/gelu_fp32_4_1_uniform_linear.csv \\"
echo "      --activation gelu --range-min -4 --range-max 4 --tiles 1"
echo ""
echo "ALTERNATIVE: Use the LLK test infrastructure for quasar ELF generation:"
echo "  cd $TT_METAL_ROOT/tt_metal/third_party/tt_llk/tests"
echo "  make archname=quasar testname=sources/quasar/eltwise_unary_datacopy_quasar_test.cpp"
echo "  # ELFs will be in /tmp/tt-llk-build/"
echo ""
