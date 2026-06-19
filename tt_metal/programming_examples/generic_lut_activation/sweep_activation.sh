#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
PLOT_ONLY=false
ARGS=()
HAS_ARCH=false

for arg in "$@"; do
    if [[ "$arg" == "--plot" ]]; then
        PLOT_ONLY=true
    elif [[ "$arg" == "--arch" ]]; then
        HAS_ARCH=true
        ARGS+=("$arg")
    else
        ARGS+=("$arg")
    fi
done

if [[ "$PLOT_ONLY" == "true" ]]; then
    # Plot mode: just run the plotting script
    echo "=== Plot Mode: Generating error analysis plots ==="

    if [[ "$HAS_ARCH" == "true" ]]; then
        # Arch was specified, use it
        python3 "$SCRIPT_DIR/plots/plot_hardware_error_analysis.py" "${ARGS[@]}" --precision both
    else
        # No arch specified, do both
        echo "No --arch specified, plotting for both wormhole and blackhole"
        python3 "$SCRIPT_DIR/plots/plot_hardware_error_analysis.py" "${ARGS[@]}" --arch wormhole --precision both
        python3 "$SCRIPT_DIR/plots/plot_hardware_error_analysis.py" "${ARGS[@]}" --arch blackhole --precision both
    fi
else
    # Execution mode: run sweeps
    "$SCRIPT_DIR/sweep_best.sh" "${ARGS[@]}" --precision bf16 --use-ulp
    "$SCRIPT_DIR/sweep_best.sh" "${ARGS[@]}" --precision fp32 --use-ulp
    "$SCRIPT_DIR/sweep_native_sfpu.sh" "${ARGS[@]}" --precision bf16 
    "$SCRIPT_DIR/sweep_native_sfpu.sh" "${ARGS[@]}" --precision fp32 
fi
