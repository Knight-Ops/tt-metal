#!/bin/bash
# =============================================================================
# apply_dropin.sh — Generate and apply a LUT-based drop-in SFPU activation
#
# Full pipeline:
#   1. Generate ckernel_sfpu_<activation>.h from coefficient CSVs
#   2. Install shared piecewise_rational evaluator
#   3. Merge upstream/main
#   4. Clean stray files + JIT cache
#   5. Rebuild
#   6. Validate
#
# Usage:
#   ./apply_dropin.sh --activation softplus \
#       --bf16-csv $TT_POLY_FIT_DIR/data/coefficients/softplus_n9d9_s1_uniform_rational_ulp.csv \
#       --fp32-csv $TT_POLY_FIT_DIR/data/coefficients/softplus_n11d11_s2_uniform_rational_ulp.csv \
#       [--skip-test]
#
# Example (BF16 only, same coefficients for both dtypes):
#   ./apply_dropin.sh --activation softplus \
#       --bf16-csv $TT_POLY_FIT_DIR/data/coefficients/softplus_n9d9_s1_uniform_rational_ulp.csv
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)

ACTIVATION=""
BF16_CSV=""
FP32_CSV=""
SKIP_TEST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --activation|-a) ACTIVATION="$2"; shift 2 ;;
        --bf16-csv)      BF16_CSV="$2"; shift 2 ;;
        --fp32-csv)      FP32_CSV="$2"; shift 2 ;;
        --skip-test)     SKIP_TEST=true; shift ;;
        -h|--help)
            echo "Usage: $0 --activation <name> --bf16-csv <path> [--fp32-csv <path>] [--skip-test]"
            exit 0 ;;
        *)
            # Legacy positional: first arg is activation
            if [[ -z "$ACTIVATION" ]]; then
                ACTIVATION="$1"; shift
            elif [[ "$1" == "--skip-test" ]]; then
                SKIP_TEST=true; shift
            else
                echo "Unknown: $1"; exit 1
            fi ;;
    esac
done

[[ -z "$ACTIVATION" ]] && { echo "Error: --activation required"; exit 1; }

BH_SFPU="$REPO_ROOT/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"
WH_SFPU="$REPO_ROOT/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu"
HEADER="ckernel_sfpu_${ACTIVATION}.h"
SHARED="ckernel_sfpu_piecewise_rational.h"

echo "=== Drop-in Replacement: $ACTIVATION ==="
echo ""

# Step 1: Generate drop-in header from coefficient CSVs
if [[ -n "$BF16_CSV" ]]; then
    echo ">>> Step 1: Generating $HEADER from coefficients..."
    GEN_ARGS="--activation $ACTIVATION --bf16-csv $BF16_CSV --output $BH_SFPU/$HEADER"
    [[ -n "$FP32_CSV" ]] && GEN_ARGS="$GEN_ARGS --fp32-csv $FP32_CSV"
    python3 "$SCRIPT_DIR/generate_dropin_header.py" $GEN_ARGS
    echo ""
else
    # No CSV provided — expect header already exists
    if [[ ! -f "$BH_SFPU/$HEADER" ]]; then
        echo "ERROR: No --bf16-csv provided and $BH_SFPU/$HEADER not found"
        echo "Either provide coefficient CSVs or create the header manually."
        exit 1
    fi
    echo ">>> Step 1: Using existing $HEADER (no CSV provided)"
fi

# Ensure shared evaluator exists
if [[ ! -f "$BH_SFPU/$SHARED" ]]; then
    echo "ERROR: Shared evaluator $BH_SFPU/$SHARED not found"
    echo "It should be committed to the repo. Check git status."
    exit 1
fi

# Step 2: Merge upstream/main (best-effort — abort on conflict, don't block)
cd "$REPO_ROOT"
git fetch upstream main --quiet 2>/dev/null || true
if git rev-parse upstream/main >/dev/null 2>&1; then
    MERGE_BASE=$(git merge-base HEAD upstream/main 2>/dev/null || echo "")
    UPSTREAM_HEAD=$(git rev-parse upstream/main 2>/dev/null || echo "")
    if [[ -n "$MERGE_BASE" && "$MERGE_BASE" != "$UPSTREAM_HEAD" ]]; then
        echo "Merging upstream/main..."
        STASHED=false
        if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
            git stash push -m "apply_dropin auto-stash" --quiet && STASHED=true
        fi
        if git merge upstream/main --no-edit --quiet 2>/dev/null; then
            echo "  Merged $(git log --oneline upstream/main ^HEAD~1 | wc -l) commits"
        else
            echo "  Merge conflict — aborting merge (non-blocking)"
            git merge --abort 2>/dev/null
        fi
        [[ "$STASHED" == true ]] && git stash pop --quiet 2>/dev/null || true
    else
        echo "  Already up to date with upstream/main."
    fi
fi

# Step 3: Remove stray copies (critical — stray files shadow real headers via -I<root>)
echo "Cleaning stray copies..."
stray_count=0
for stray in $(find "$REPO_ROOT" -maxdepth 1 -name "ckernel_sfpu_*.h" 2>/dev/null); do
    echo "  REMOVING stray: $stray"
    rm -f "$stray"
    stray_count=$((stray_count + 1))
done
[[ $stray_count -eq 0 ]] && echo "  No stray copies found."

# Step 3b: Standardize header naming (remove ckernel_sfpu_unary_ variants)
# Convention: ckernel_sfpu_{name}.h — update any #include that references the old name.
OLD_HEADER="ckernel_sfpu_unary_${ACTIVATION}.h"
for arch_sfpu in "$BH_SFPU" "$WH_SFPU"; do
    if [[ -f "$arch_sfpu/$OLD_HEADER" ]]; then
        echo "  Removing non-standard $OLD_HEADER from $(basename $(dirname $(dirname $arch_sfpu)))"
        rm -f "$arch_sfpu/$OLD_HEADER"
    fi
    # Fix any include referencing the old name (llk_math, api headers, etc.)
    grep -rl "\"$OLD_HEADER\"" "$arch_sfpu" "$REPO_ROOT/tt_metal/hw/inc" 2>/dev/null | while read -r f; do
        echo "  Updating include in $(basename $f): $OLD_HEADER → $HEADER"
        sed -i "s|\"$OLD_HEADER\"|\"$HEADER\"|g" "$f"
    done
done

# Step 4: Copy blackhole → wormhole_b0
echo "Syncing blackhole → wormhole_b0..."
cp "$BH_SFPU/$HEADER" "$WH_SFPU/$HEADER"
cp "$BH_SFPU/$SHARED" "$WH_SFPU/$SHARED"
echo "  $HEADER: synced"
echo "  $SHARED: synced"

# Step 5: Rebuild
echo ""
echo "Building..."
cd "$REPO_ROOT"
./build_metal.sh 2>&1 | tail -5

# Step 6: Clear JIT kernel cache (headers aren't tracked in dependency hash)
echo ""
echo "Clearing JIT kernel cache..."
rm -rf /home/$USER/.cache/tt-metal-cache
echo "  Done."

# Step 7: Run validation (if not skipped)
if [[ "$SKIP_TEST" == false ]]; then
    echo ""
    echo "=== Running TTNN validation ==="
    source "$REPO_ROOT/python_env/bin/activate"
    PYTHONPATH="$REPO_ROOT" python3 "$SCRIPT_DIR/validate_dropin.py" "$ACTIVATION"

    echo ""
    echo "=== Running unit tests ==="
    PYTHONPATH="$REPO_ROOT" pytest "$REPO_ROOT/tests/ttnn/unit_tests/operations/eltwise/" \
        -k "$ACTIVATION" -vvv --no-header 2>&1 | tail -5
fi

echo ""
echo "=== Drop-in applied successfully ==="
echo ""
echo "Files modified:"
echo "  $BH_SFPU/$HEADER"
echo "  $WH_SFPU/$HEADER"
echo "  $BH_SFPU/$SHARED"
echo "  $WH_SFPU/$SHARED"
echo ""
echo "IMPORTANT: Run 'rm -rf ~/.cache/tt-metal-cache' if you see stale results."
echo "The JIT cache does NOT track ckernel header changes."
