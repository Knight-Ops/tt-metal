#!/bin/bash
# Test gelu p1_s4 (linear, 4 segments) on Polaris/ttsim for Quasar
#
# Naming convention: p{degree}_s{segments}
# p1_s4 = polynomial degree 1 (linear), 4 segments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="/localdev/nkapre/tt-metal"
POLARIS_ROOT="/localdev/nkapre/polaris/polaris"
WORK_DIR="$SCRIPT_DIR"
CACHE_DIR="/tmp/quasar-p1-s4-cache"

# Create directories
mkdir -p "$WORK_DIR/elfs"
mkdir -p "$WORK_DIR/config"
mkdir -p "$WORK_DIR/output"
mkdir -p "$CACHE_DIR"

echo "=============================================="
echo "Polaris/ttsim Quasar Test: gelu p1_s4"
echo "=============================================="
echo "Kernel: p1_s4 (polynomial degree 1, 4 segments)"
echo "Activation: gelu"
echo ""

# Step 1: Try to compile kernels and capture ELFs
echo "Step 1: Attempting to run kernel with Quasar arch..."
echo "-------------------------------------------"

cd "$TT_METAL_ROOT"

# Set environment for quasar compilation
export ARCH_NAME=quasar
export TT_METAL_RUNTIME_ROOT="$TT_METAL_ROOT"
export TT_METAL_CACHE="$CACHE_DIR"
export TT_METAL_SKIP_DELETING_BUILT_CACHE=1
export LD_LIBRARY_PATH="$TT_METAL_ROOT/build_Release/lib:$LD_LIBRARY_PATH"

BINARY="$TT_METAL_ROOT/build_Release/programming_examples/programming_examples_generic_lut_activation_p1_s4"
CSV_FILE="$WORK_DIR/gelu_fp32_4_1_uniform_linear.csv"

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: Binary not found: $BINARY"
    echo "Please build with: ./build_metal.sh --build-programming-examples"
    exit 1
fi

if [[ ! -f "$CSV_FILE" ]]; then
    echo "ERROR: Coefficient CSV not found: $CSV_FILE"
    exit 1
fi

echo "Binary: $BINARY"
echo "Coefficients: $CSV_FILE"
echo ""
echo "Running with ARCH_NAME=quasar..."
echo "(This will fail on device init but may trigger kernel JIT compilation)"
echo ""

# Try to run - this will fail on device init but might generate kernel artifacts
timeout 30s "$BINARY" \
    "$CSV_FILE" \
    --activation gelu \
    --range-min -4.0 \
    --range-max 4.0 \
    --tiles 1 2>&1 || true

echo ""
echo "Step 2: Searching for compiled ELF files..."
echo "-------------------------------------------"

# Search for ELF files in the cache
ELF_LOCATIONS=(
    "$CACHE_DIR"
    "$TT_METAL_ROOT/built"
    "/tmp/tt-metal-cache"
)

for dir in "${ELF_LOCATIONS[@]}"; do
    if [[ -d "$dir" ]]; then
        echo "Checking: $dir"
        find "$dir" -name "*.elf" 2>/dev/null | while read elf; do
            echo "  Found: $elf"
            cp "$elf" "$WORK_DIR/elfs/" 2>/dev/null || true
        done
    fi
done

echo ""
echo "Step 3: Checking ELF files..."
echo "-------------------------------------------"

ELF_COUNT=$(ls -1 "$WORK_DIR/elfs/"*.elf 2>/dev/null | wc -l || echo 0)
if [[ "$ELF_COUNT" -gt 0 ]]; then
    echo "✓ Found $ELF_COUNT ELF file(s) in $WORK_DIR/elfs/"
    ls -la "$WORK_DIR/elfs/"*.elf 2>/dev/null

    # Check ELF architecture
    echo ""
    echo "ELF file details:"
    for elf in "$WORK_DIR/elfs/"*.elf; do
        echo "  $(basename $elf):"
        file "$elf" 2>/dev/null | sed 's/^/    /'
        # Check RISC-V attributes if readelf is available
        readelf -A "$elf" 2>/dev/null | grep -i "riscv\|arch" | head -2 | sed 's/^/    /'
    done
else
    echo "⚠ No ELF files found in cache."
    echo ""
    echo "The kernel JIT compilation didn't happen because:"
    echo "1. ARCH_NAME=quasar has no physical device"
    echo "2. There's no quasar simulator (ttsim) available"
    echo ""
    echo "To get ELFs for polaris testing, you need to:"
    echo "  Option A: Run on wormhole/blackhole hardware and extract ELFs"
    echo "  Option B: Use polaris RTL test data (has pre-compiled ELFs)"
    echo "  Option C: Manually compile using riscv-tt-elf-g++ for quasar"
fi

echo ""
echo "Step 4: Creating polaris input configuration..."
echo "-------------------------------------------"

cat > "$WORK_DIR/config/gelu_p1_s4_inputcfg.json" << EOF
{
  "arch": "ttqs",
  "llkVersionTag": "sep23",
  "debug": 15,
  "numTCores": 1,
  "input": {
    "syn": 0,
    "name": "gelu-p1-s4-lut-activation",
    "tc0": {
      "numThreads": 4,
      "startFunction": "main",
      "th0Elf": "brisc.elf",
      "th0Path": "$WORK_DIR/elfs",
      "th1Elf": "ncrisc.elf",
      "th1Path": "$WORK_DIR/elfs",
      "th2Elf": "trisc0.elf",
      "th2Path": "$WORK_DIR/elfs",
      "th3Elf": "trisc1.elf",
      "th3Path": "$WORK_DIR/elfs"
    }
  },
  "description": {
    "kernel": "p1_s4",
    "activation": "gelu",
    "degree": 1,
    "segments": 4,
    "source": "tt-metal generic_lut_activation"
  }
}
EOF

echo "Created: $WORK_DIR/config/gelu_p1_s4_inputcfg.json"

echo ""
echo "Step 5: Polaris run command"
echo "-------------------------------------------"
echo ""
echo "Once you have ELF files, run:"
echo ""
echo "  cd $POLARIS_ROOT"
echo "  conda activate polaris"
echo ""
echo "  python -m ttsim.back.tensix_neo.tneoSim \\"
echo "    --inputcfg $WORK_DIR/config/gelu_p1_s4_inputcfg.json \\"
echo "    --cfg $POLARIS_ROOT/config/tensix_neo/ttqs_neo4_sep23.json \\"
echo "    --memoryMap $POLARIS_ROOT/config/tensix_neo/ttqs_memory_map_sep23.json \\"
echo "    --ttISAFileName $POLARIS_ROOT/ttsim/config/llk/instruction_sets/ttqs/assembly.yaml \\"
echo "    --odir $WORK_DIR/output \\"
echo "    --exp gelu_p1_s4"
echo ""

echo "=============================================="
echo "Summary"
echo "=============================================="
echo ""
echo "Created files:"
echo "  - $CSV_FILE (LUT coefficients)"
echo "  - $WORK_DIR/config/gelu_p1_s4_inputcfg.json (polaris config)"
echo ""
echo "Missing: ELF files for quasar architecture"
echo ""
echo "The fundamental issue: tt-metal JIT compiles kernels only when"
echo "a device is available. For quasar (simulation-only target),"
echo "there's no device to trigger compilation."
echo ""
echo "Next steps:"
echo "1. Check if polaris RTL test data has compatible ELFs"
echo "2. Or compile kernels manually for quasar architecture"
echo ""
