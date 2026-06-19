#!/bin/bash
# =============================================================================
# pr_submission.sh — End-to-end LUT drop-in replacement pipeline
#
# Flow:
#   Step 0: Measure original baseline (BEFORE any changes)
#   Step 1: Validate LUT coefficients (sweep_best.sh → embedded kernel timing)
#   Step 2: Generate + apply drop-in header
#   Step 3: Measure drop-in (ULP + Tracy timing)
#   Step 4: Build results table, generate ULP plots, submit PR
#
# Usage:
#   ./pr_submission.sh --activation softplus
#   ./pr_submission.sh --activation digamma --bf16-csv <path> --fp32-csv <path>
#   ./pr_submission.sh --activation softplus --no-submit
#   ./pr_submission.sh --activation softplus --start-from apply
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)

TT_POLY_FIT_DIR="${TT_POLY_FIT_DIR:-/localdev/nkapre/tt-polynomial-fitter}"
BEST_CSV="$TT_POLY_FIT_DIR/best.csv"
COEFF_DIR="$TT_POLY_FIT_DIR/data/coefficients"
ACCURACY_PYTHON="/usr/bin/python3"
ACCURACY_SCRIPT="$TT_POLY_FIT_DIR/extract_accuracy.py"

ACTIVATION=""
BF16_CSV=""
FP32_CSV=""
NO_SUBMIT=false
START_FROM="baseline"
TICKET_URL=""

# Resolve coefficient CSV path from best.csv
resolve_csv_from_best() {
    local activation="$1"
    local precision="$2"
    local metric="${3:-ulp}"

    [[ ! -f "$BEST_CSV" ]] && { echo ""; return; }

    local row=$(grep "^${activation},${precision}," "$BEST_CSV" | head -1)
    [[ -z "$row" ]] && { echo ""; return; }

    local degree=$(echo "$row" | python3 -c "import sys; r=sys.stdin.read().strip().split(','); print(r[23])")
    local num_segs=$(echo "$row" | python3 -c "import sys; r=sys.stdin.read().strip().split(','); print(r[28])")
    local segmentation=$(echo "$row" | python3 -c "import sys; r=sys.stdin.read().strip().split(','); print(r[29])")
    local fitting=$(echo "$row" | python3 -c "import sys; r=sys.stdin.read().strip().split(','); print(r[30])")

    local csv_name=""
    if [[ "$fitting" == "rational" ]]; then
        local nd=$(echo "$degree" | sed 's|/|d|')
        csv_name="${activation}_n${nd}_s${num_segs}_${segmentation}_rational_${metric}.csv"
    else
        csv_name="${activation}_p${degree}_s${num_segs}_${segmentation}_${fitting}_${metric}.csv"
    fi

    local csv_path="$COEFF_DIR/$csv_name"
    [[ -f "$csv_path" ]] && echo "$csv_path" || echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --activation|-a) ACTIVATION="$2"; shift 2 ;;
        --bf16-csv)      BF16_CSV="$2"; shift 2 ;;
        --fp32-csv)      FP32_CSV="$2"; shift 2 ;;
        --no-submit)     NO_SUBMIT=true; shift ;;
        --start-from)    START_FROM="$2"; shift 2 ;;
        --ticket)        TICKET_URL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --activation <name> [--bf16-csv <path>] [--fp32-csv <path>] [--no-submit] [--start-from <step>] [--ticket <url>]"
            echo ""
            echo "Steps: baseline → validate-lut → apply → dropin → submit"
            echo "  --start-from baseline       (default) Run everything"
            echo "  --start-from validate-lut    Skip baseline, start at LUT validation"
            echo "  --start-from apply           Skip to apply drop-in"
            echo "  --start-from dropin          Skip to measuring drop-in"
            echo "  --start-from submit          Skip to PR submission"
            echo "  --ticket <url>               GitHub issue URL to link in PR"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[[ -z "$ACTIVATION" ]] && { echo "Error: --activation required"; exit 1; }

# Check for coeff_source (e.g. selu/celu reuse elu's LUT)
ACT_JSON_EARLY="$TT_POLY_FIT_DIR/activations/${ACTIVATION}.json"
COEFF_ACTIVATION="$ACTIVATION"
if [[ -f "$ACT_JSON_EARLY" ]]; then
    COEFF_SRC=$(python3 -c "import json; print(json.load(open('$ACT_JSON_EARLY')).get('coeff_source',''))" 2>/dev/null)
    if [[ -n "$COEFF_SRC" ]]; then
        COEFF_ACTIVATION="$COEFF_SRC"
        echo "  coeff_source: using $COEFF_SRC coefficients for $ACTIVATION"
    fi
fi

# Auto-resolve CSVs (from coeff_source activation if specified)
if [[ -z "$BF16_CSV" ]]; then
    BF16_CSV=$(resolve_csv_from_best "$COEFF_ACTIVATION" "bf16")
    [[ -z "$BF16_CSV" ]] && { echo "Error: Could not resolve BF16 CSV from best.csv for $COEFF_ACTIVATION"; exit 1; }
    echo "Auto-resolved BF16 CSV: $BF16_CSV"
fi
if [[ -z "$FP32_CSV" ]]; then
    FP32_CSV=$(resolve_csv_from_best "$COEFF_ACTIVATION" "fp32")
    [[ -n "$FP32_CSV" ]] && echo "Auto-resolved FP32 CSV: $FP32_CSV" || echo "No FP32 CSV (BF16-only mode)"
fi

echo ""
echo "================================================================"
echo "  LUT Drop-in Full Flow: $ACTIVATION"
echo "================================================================"
echo "BF16 CSV: $BF16_CSV"
[[ -n "$FP32_CSV" ]] && echo "FP32 CSV: $FP32_CSV"
echo "Start from: $START_FROM"
echo ""

cd "$REPO_ROOT"

# Detect architecture from environment
ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
ARCH_SHORT="${ARCH_NAME/_b0/}"
echo "  Architecture: $ARCH_NAME ($ARCH_SHORT)"

RESULTS_FILE="/tmp/pr_submission_${ACTIVATION}_results.txt"
WORK_DIR="tt_metal/programming_examples/generic_lut_activation_embedded"
NATIVE_SFPU_DIR="tt_metal/programming_examples/generic_lut_activation/data/hardware_outputs/$ARCH_SHORT/$ACTIVATION"
EMBEDDED_HW_DIR="$WORK_DIR/data/hardware_outputs/$ARCH_SHORT/$ACTIVATION"
source "$SCRIPT_DIR/../profiler_helpers.sh"

SWEEP_NATIVE="$REPO_ROOT/tt_metal/programming_examples/generic_lut_activation/sweep_native_sfpu.sh"

# Clean stale data from previous runs to avoid confusion
echo "  Cleaning stale data for $ACTIVATION..."
rm -rf "$NATIVE_SFPU_DIR"/*_baseline_* "$NATIVE_SFPU_DIR"/*_dropin_* 2>/dev/null
rm -rf "$NATIVE_SFPU_DIR"/*.err.npz 2>/dev/null
rm -rf "$REPO_ROOT/$EMBEDDED_HW_DIR"/*.err.npz 2>/dev/null
rm -rf ~/.cache/tt-metal-cache 2>/dev/null

# =========================================================================
# Step 0: Measure original baseline BEFORE any changes
# =========================================================================
should_run() {
    local step="$1"
    case "$START_FROM" in
        baseline)     return 0 ;;
        validate-lut) [[ "$step" != "baseline" ]] ;;
        apply)        [[ "$step" != "baseline" && "$step" != "validate-lut" ]] ;;
        dropin)       [[ "$step" == "dropin" || "$step" == "plots" || "$step" == "submit" ]] ;;
        submit)       [[ "$step" == "submit" ]] ;;
    esac
}

# Detect native SFPU early (needed for step 0 decision)
# git show needs repo-relative path, not absolute
BH_SFPU_REL="tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"
HEADER="ckernel_sfpu_${ACTIVATION}.h"
# Detect native SFPU: check both upstream/main and origin/main
HAS_NATIVE_SFPU=false
git fetch upstream main --quiet 2>/dev/null || git fetch origin main --quiet 2>/dev/null || true
MAIN_REF=$(git rev-parse --verify upstream/main 2>/dev/null || git rev-parse --verify origin/main 2>/dev/null || echo "")
if [[ -n "$MAIN_REF" ]]; then
    for pattern in "$HEADER" "ckernel_sfpu_unary_${ACTIVATION}.h"; do
        if git show "${MAIN_REF}:$BH_SFPU_REL/$pattern" >/dev/null 2>&1; then
            HAS_NATIVE_SFPU=true
            break
        fi
    done
fi
echo "  Native SFPU: $HAS_NATIVE_SFPU"

# Step 0: Measure baseline on clean upstream/main worktree.
# Uses /localdev/nkapre/tt-metal-clean — pre-built, pristine upstream code.
# Works for both drop-in replacements AND new ops (composite → SFPU).
CLEAN_WORKTREE="/localdev/nkapre/tt-metal-clean"
CLEAN_SWEEP="$CLEAN_WORKTREE/tt_metal/programming_examples/generic_lut_activation/sweep_native_sfpu.sh"

if should_run baseline; then
    echo "================================================================"
    echo "  Step 0/4: Measure baseline (clean worktree: $CLEAN_WORKTREE)"
    echo "================================================================"

    if [[ ! -f "$CLEAN_WORKTREE/build_Release/lib/_ttnn.so" ]]; then
        echo "  ERROR: Clean worktree not built at $CLEAN_WORKTREE"
        echo "  Set up with: git worktree add $CLEAN_WORKTREE upstream/main && cd $CLEAN_WORKTREE && ./build_metal.sh"
        echo "  Skipping baseline."
    else
        rm -rf ~/.cache/tt-metal-cache

        # Run sweep FROM the clean worktree (cd there so JIT uses upstream sources)
        # but with feature branch's python env (has all deps: tracy, torch, etc.)
        # PYTHONPATH puts clean worktree's ttnn first so upstream _ttnn.so loads.
        # Copy sweep scripts to clean worktree (upstream doesn't have them)
        CLEAN_LUT_DIR="$CLEAN_WORKTREE/tt_metal/programming_examples/generic_lut_activation"
        mkdir -p "$CLEAN_LUT_DIR"
        for item in "$REPO_ROOT/tt_metal/programming_examples/generic_lut_activation"/*.sh \
                    "$REPO_ROOT/tt_metal/programming_examples/generic_lut_activation"/*.py \
                    "$REPO_ROOT/tt_metal/programming_examples/generic_lut_activation"/*.dat; do
            [[ -f "$item" ]] && cp "$item" "$CLEAN_LUT_DIR/"
        done

        echo "  Running sweep in clean worktree (upstream code + JIT sources)..."
        (cd "$CLEAN_WORKTREE" && \
            source "$CLEAN_WORKTREE/python_env/bin/activate" && \
            export PYTHONPATH="$CLEAN_WORKTREE/ttnn:$CLEAN_WORKTREE:$CLEAN_WORKTREE/tools:${PYTHONPATH:-}" && \
            export ARCH_NAME="$ARCH_NAME" && \
            bash "$CLEAN_LUT_DIR/sweep_native_sfpu.sh" --activation "$ACTIVATION" --precision both --runs 3 --use-fitting-range)

        # Copy baseline CSVs to feature branch data directory
        CLEAN_SFPU_DIR="$CLEAN_WORKTREE/tt_metal/programming_examples/generic_lut_activation/data/hardware_outputs/$ARCH_SHORT/$ACTIVATION"
        if [[ -d "$CLEAN_SFPU_DIR" ]]; then
            mkdir -p "$REPO_ROOT/$NATIVE_SFPU_DIR"
            for f in "$CLEAN_SFPU_DIR"/native_sfpu_*; do
                [[ -f "$f" ]] && cp "$f" "$REPO_ROOT/$NATIVE_SFPU_DIR/"
            done
            echo "  Copied baseline CSVs to feature branch"
        fi
        rm -rf ~/.cache/tt-metal-cache
    fi
    echo ""
fi

# =========================================================================
# Step 1: Validate LUT coefficients (embedded kernel timing)
# =========================================================================
if should_run validate-lut; then
    echo "================================================================"
    echo "  Step 1/4: Validate LUT coefficients (sweep_best.sh)"
    echo "================================================================"
    echo ""
    "$SCRIPT_DIR/../sweep_best.sh" --activation "$ACTIVATION" --precision bf16 --runs 3
    echo ""
    "$SCRIPT_DIR/../sweep_best.sh" --activation "$ACTIVATION" --precision fp32 --runs 3
    echo ""
fi

# =========================================================================
# Step 2: Generate + apply (auto-detect drop-in vs new SFPU registration)
# =========================================================================
if should_run apply; then
    APPLY_ARGS="--activation $ACTIVATION --bf16-csv $BF16_CSV"
    [[ -n "$FP32_CSV" ]] && APPLY_ARGS="$APPLY_ARGS --fp32-csv $FP32_CSV"

    if [[ "$HAS_NATIVE_SFPU" == true ]]; then
        echo "================================================================"
        echo "  Step 2/4: Drop-in replacement (apply_dropin.sh)"
        echo "================================================================"
        echo ""
        "$SCRIPT_DIR/apply_dropin.sh" $APPLY_ARGS
    else
        echo "================================================================"
        echo "  Step 2/4: New SFPU kernel (register_new_sfpu.sh)"
        echo "================================================================"
        echo ""
        "$SCRIPT_DIR/register_new_sfpu.sh" $APPLY_ARGS
        echo ""
        echo "  Building..."
        ./build_metal.sh 2>&1 | tail -1
        rm -rf ~/.cache/tt-metal-cache
    fi

    # Commit generated headers to feature branch so submit_pr.sh can find them
    echo "  Committing generated headers..."
    cd "$REPO_ROOT"
    git add "$REPO_ROOT"/tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/ckernel_sfpu_${ACTIVATION}.h 2>/dev/null || true
    git add "$REPO_ROOT"/tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/ckernel_sfpu_piecewise_*.h 2>/dev/null || true
    git add "$REPO_ROOT"/tt_metal/hw/ckernels/*/metal/llk_api/llk_sfpu/llk_math_eltwise*${ACTIVATION}*.h 2>/dev/null || true
    git add "$REPO_ROOT"/tt_metal/hw/inc/api/compute/eltwise_unary/${ACTIVATION}.h 2>/dev/null || true
    git commit --no-verify -m "auto: update ${ACTIVATION} drop-in headers" --quiet 2>/dev/null || true
    echo ""
fi

# =========================================================================
# Step 3: Measure drop-in + build results table
# =========================================================================
> "$RESULTS_FILE"

if should_run dropin; then
    echo "================================================================"
    echo "  Step 3/4: Measure drop-in (sweep_native_sfpu.sh)"
    echo "================================================================"

    cd "$REPO_ROOT"
    rm -rf ~/.cache/tt-metal-cache

    # Save baseline CSVs before step 3 overwrites them.
    # Handles both .csv and .csv.gz from sweep_native_sfpu.sh.
    for f in "$NATIVE_SFPU_DIR"/native_sfpu_*_precise_*; do
        [[ -f "$f" && "$f" != *_baseline* && "$f" != *_dropin* ]] || continue
        cp "$f" "${f/_precise_/_precise_baseline_}"
    done

    # sweep_native_sfpu.sh runs TTNN which now uses the drop-in header.
    "$SWEEP_NATIVE" --activation "$ACTIVATION" --precision both --runs 3

    # Tag dropin CSVs, restore baselines for the plot.
    for f in "$NATIVE_SFPU_DIR"/native_sfpu_*_precise_*; do
        [[ -f "$f" && "$f" != *_baseline* && "$f" != *_dropin* ]] || continue
        mv "$f" "${f/_precise_/_precise_dropin_}"
    done
    for f in "$NATIVE_SFPU_DIR"/*_baseline_*; do
        [[ -f "$f" ]] || continue
        restored="${f/_precise_baseline_/_precise_}"
        cp "$f" "$restored"
    done
    echo ""
fi

# =========================================================================
# Step 3b: Generate ULP plots
# =========================================================================
if should_run dropin; then
    echo "================================================================"
    echo "  Step 3b/4: Generate ULP plots"
    echo "================================================================"
    PLOT_SCRIPT="$SCRIPT_DIR/../plots/plot_hardware_error_analysis.py"
    if [[ -f "$PLOT_SCRIPT" ]]; then
        echo "Generating ULP distribution plots..."
        cd "$REPO_ROOT/tt_metal/programming_examples/generic_lut_activation_embedded"
        for prec in bf16 fp32; do
            # Find the dropin CSV (tiles3200 = largest shape from sweep)
            DROPIN_CSV=$(ls "$REPO_ROOT/$NATIVE_SFPU_DIR"/native_sfpu_${ACTIVATION}_${prec}_precise_dropin_tiles3200.csv* 2>/dev/null | grep -v "\.err\.\|\.npz" | head -1)
            [[ -z "$DROPIN_CSV" ]] && continue
            echo "  $prec: baseline (blue) vs dropin (yellow)"
            /usr/bin/python3 plots/plot_hardware_error_analysis.py --arch "$ARCH_SHORT" --metric ulp --full --workers 8 --standalone \
                --activation "$ACTIVATION" --precision "$prec" --csv "$DROPIN_CSV" 2>&1 | tail -10
        done
        cd "$REPO_ROOT"

        PLOT_DIR="$SCRIPT_DIR/../plots/${ARCH_SHORT}/hardware_error_analysis/${ACTIVATION}/ulp_only"
        if [[ -d "$PLOT_DIR" ]]; then
            echo ""
            echo "ULP plots generated:"
            ls "$PLOT_DIR"/*.png 2>/dev/null || true
        fi
    else
        echo "Plot script not found — skipping"
    fi
    echo ""
fi

# =========================================================================
# Step 3c: Run local CI tests (from ci_tests in activation JSON)
# =========================================================================
if should_run dropin; then
    echo "================================================================"
    echo "  Step 3c/4: Local CI tests"
    echo "================================================================"

    # Auto-discover tests matching activation name in eltwise test directory
    TEST_DIR="$REPO_ROOT/tests/ttnn/unit_tests/operations/eltwise"
    rm -rf ~/.cache/tt-metal-cache

    echo "  Running: pytest $TEST_DIR -k $ACTIVATION --timeout=300"
    LOCAL_CI_PASS=true
    if source "$REPO_ROOT/python_env/bin/activate" && \
       PYTHONPATH="$REPO_ROOT" pytest "$TEST_DIR" -k "$ACTIVATION" -x --timeout=300 --no-header -q 2>&1 | tail -5; then
        echo "  ✓ PASSED"
    else
        echo "  ✗ FAILED"
        LOCAL_CI_PASS=false
    fi

    echo ""
    if [[ "$LOCAL_CI_PASS" == true ]]; then
        echo "  All local CI tests PASSED."
    else
        echo "  WARNING: Some local CI tests FAILED."
    fi
    echo ""
fi

# =========================================================================
# Step 4: Submit PR
# =========================================================================
if [[ "$NO_SUBMIT" == true ]]; then
    echo "================================================================"
    echo "  Step 4/4: SKIPPED (--no-submit)"
    echo "================================================================"
    echo ""
    echo "To submit later:"
    echo "  $SCRIPT_DIR/submit_pr.sh --activation $ACTIVATION"
else
    if should_run submit; then
        echo "================================================================"
        echo "  Step 4/4: Submit PR (submit_pr.sh)"
        echo "================================================================"
        echo ""
        SUBMIT_ARGS="--activation $ACTIVATION"
        [[ -n "$TICKET_URL" ]] && SUBMIT_ARGS="$SUBMIT_ARGS --ticket $TICKET_URL"
        "$SCRIPT_DIR/submit_pr.sh" $SUBMIT_ARGS
    fi
fi

# =========================================================================
# Step 5 (optional): Check Copilot review feedback on existing PRs
# =========================================================================
# If a PR already exists for this activation, fetch Copilot feedback
PR_NUM=$(gh pr list --repo tenstorrent/tt-metal --author nkapreTT --state open \
    --json number,title --jq ".[] | select(.title | test(\"$ACTIVATION\"; \"i\")) | .number" 2>/dev/null | head -1)
if [[ -n "$PR_NUM" ]]; then
    echo "================================================================"
    echo "  Copilot Review: PR #$PR_NUM"
    echo "================================================================"
    "$SCRIPT_DIR/pr_copilot.sh" --pr "$PR_NUM" 2>/dev/null || true
fi

echo ""
echo "================================================================"
echo "  Done: $ACTIVATION drop-in flow complete"
echo "================================================================"
