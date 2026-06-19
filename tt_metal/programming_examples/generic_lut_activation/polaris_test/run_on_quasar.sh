#!/bin/bash
# Run generic_lut_activation on quasar simulator (assumes setup is done)
#
# Usage: ./run_on_quasar.sh [activation] [degree] [segments]
#   activation: gelu, silu, relu, tanh, sigmoid (default: gelu)
#   degree: polynomial degree 1-8 (default: 1)
#   segments: number of segments 4,8,16,32 (default: 4)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="/localdev/nkapre/tt-metal"
SIM_DIR="$HOME/sim/qsr"
ELF_OUTPUT_DIR="$SCRIPT_DIR/quasar_elfs"

# Default parameters
ACTIVATION="${1:-gelu}"
DEGREE="${2:-1}"
SEGMENTS="${3:-4}"

# Validate simulator exists
if [[ ! -f "$SIM_DIR/libttsim.so" ]]; then
    echo "ERROR: Quasar simulator not found at $SIM_DIR/libttsim.so"
    echo "Run ./setup_quasar_ttsim.sh first"
    exit 1
fi

echo "=============================================="
echo "Running on Quasar Simulator"
echo "=============================================="
echo "Activation: $ACTIVATION"
echo "Degree: $DEGREE (p$DEGREE)"
echo "Segments: $SEGMENTS (s$SEGMENTS)"
echo "Config: p${DEGREE}_s${SEGMENTS}"
echo ""

cd "$TT_METAL_ROOT"

# Activate python env if available
if [[ -f "python_env/bin/activate" ]]; then
    source python_env/bin/activate
fi

# Set environment
export TT_METAL_SIMULATOR="$SIM_DIR/libttsim.so"
export TT_METAL_SLOW_DISPATCH_MODE=1

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$ELF_OUTPUT_DIR/${ACTIVATION}_p${DEGREE}_s${SEGMENTS}_$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "Output directory: $RUN_DIR"
echo ""
echo "Running..."
echo "----------------------------------------------"

# Run the test
# Note: The actual activation/degree/segments configuration depends on how
# the generic_lut_activation binary is configured. May need to modify
# the test to accept command-line arguments or use compile-time defines.
./build/programming_examples/generic_lut_activation/generic_lut_activation 2>&1 | tee "$RUN_DIR/run.log"

echo "----------------------------------------------"
echo ""

# Extract ELFs
echo "Extracting ELFs..."
JIT_DIR="$TT_METAL_ROOT/generated"

if [[ -d "$JIT_DIR" ]]; then
    # Find kernel directories for this run
    find "$JIT_DIR" -name "*.elf" -type f 2>/dev/null | while read elf_file; do
        base_name=$(basename "$elf_file")
        parent_dir=$(basename "$(dirname "$elf_file")")

        # Copy with context
        cp "$elf_file" "$RUN_DIR/${parent_dir}_${base_name}" 2>/dev/null || cp "$elf_file" "$RUN_DIR/$base_name"
        echo "  $base_name"
    done
fi

echo ""
echo "Results saved to: $RUN_DIR"
echo ""

# List generated files
echo "Generated files:"
ls -la "$RUN_DIR"
