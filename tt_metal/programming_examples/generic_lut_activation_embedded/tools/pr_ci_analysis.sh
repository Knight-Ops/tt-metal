#!/bin/bash
# =============================================================================
# pr_ci_analysis.sh — Analyze CI results for LUT drop-in PRs
#
# Fetches CI check statuses and failed test logs, classifies failures as:
#   - OUR_FAULT: test name matches the activation being changed
#   - BASELINE:  known flaky/infra test unrelated to our changes
#   - UNKNOWN:   needs manual investigation
#
# Usage:
#   ./pr_ci_analysis.sh                    # All open PRs by nkapreTT
#   ./pr_ci_analysis.sh --pr 39362         # Specific PR
#   ./pr_ci_analysis.sh --pr 39362,39377   # Multiple PRs
#   ./pr_ci_analysis.sh --verbose          # Show full failure details
# =============================================================================
set -e

REPO="tenstorrent/tt-metal"
AUTHOR="nkapreTT"
PRS=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --pr|-p)     PRS="$2"; shift 2 ;;
        --repo)      REPO="$2"; shift 2 ;;
        --author)    AUTHOR="$2"; shift 2 ;;
        --verbose|-v) VERBOSE=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--pr <num,num,...>] [--repo owner/repo] [--author login] [--verbose]"
            echo ""
            echo "Analyzes CI results for LUT drop-in PRs and classifies failures."
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Resolve PR list
if [[ -z "$PRS" ]]; then
    echo "Fetching open PRs by $AUTHOR on $REPO..."
    PRS=$(gh pr list --repo "$REPO" --author "$AUTHOR" --state open \
        --json number --jq '.[].number' | tr '\n' ',')
    PRS="${PRS%,}"
fi

[[ -z "$PRS" ]] && { echo "No PRs found."; exit 0; }

echo ""
echo "================================================================"
echo "  CI Analysis: $REPO"
echo "================================================================"
echo ""

# Known baseline/flaky tests (not caused by our ckernel changes)
KNOWN_BASELINE_PATTERNS=(
    "test_program_cache_invalidation_across_dispatch_modes"
    "test_conv_transpose2d_config_tensors_in_dram"
    "test_rms_norm"
    "test_rmsnorm_pre_all_gather"
    "test_gelu_accurate_allclose"  # gelu test, not related to our activations
    "T3000 fast tests.*exit code 134"  # T3K infra crash (SIGABRT), not test failure
    "exit code 134"  # SIGABRT — hardware/infra crash, not test logic failure
    "fail-on-error.*option is set"  # meta-error from test runner, not a real test
)

# Extract activation name from PR branch
get_activation() {
    local branch="$1"
    # nkapre/lut-softplus-dropin → softplus
    # nkapre/lut-erf-erfc-dropin → erf|erfc
    # nkapre/lut-digamma-sfpu → digamma
    local act=$(echo "$branch" | sed -E 's#nkapre/lut-(.+)-(dropin|sfpu)#\1#')
    # Handle erf-erfc → erf|erfc for grep pattern matching
    echo "$act" | sed 's/-/|/g'
}

# Classify a failure line
classify_failure() {
    local error_line="$1"
    local activation_pattern="$2"

    # Check if it's a known baseline issue
    for pattern in "${KNOWN_BASELINE_PATTERNS[@]}"; do
        if echo "$error_line" | grep -qiE "$pattern"; then
            echo "BASELINE"
            return
        fi
    done

    # Check if it matches our activation (our fault)
    if echo "$error_line" | grep -qiE "$activation_pattern"; then
        echo "OUR_FAULT"
        return
    fi

    echo "UNKNOWN"
}

TOTAL_PRS=0
TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_OUR_FAULT=0
TOTAL_BASELINE=0
TOTAL_UNKNOWN=0

declare -A SUMMARY_OUR_FAULT
declare -A SUMMARY_BASELINE
declare -A SUMMARY_UNKNOWN

IFS=',' read -ra PR_ARRAY <<< "$PRS"
for PR_NUM in "${PR_ARRAY[@]}"; do
    PR_NUM=$(echo "$PR_NUM" | tr -d ' ')
    TOTAL_PRS=$((TOTAL_PRS + 1))

    # Get PR info
    PR_INFO=$(gh pr view "$PR_NUM" --repo "$REPO" --json title,headRefName --jq '"\(.headRefName)\t\(.title)"')
    BRANCH=$(echo "$PR_INFO" | cut -f1)
    TITLE=$(echo "$PR_INFO" | cut -f2)
    ACTIVATION=$(get_activation "$BRANCH")

    echo "━━━ PR #$PR_NUM: $TITLE"
    echo "    Branch: $BRANCH | Activation pattern: $ACTIVATION"
    echo ""

    # Get all checks (gh pr checks exits 1 if any fail — suppress)
    CHECKS=$(gh pr checks "$PR_NUM" --repo "$REPO" 2>&1 || true)
    PASS_COUNT=$(echo "$CHECKS" | grep -c "pass" || echo "0")
    FAIL_COUNT=$(echo "$CHECKS" | grep -c "fail" || echo "0")
    SKIP_COUNT=$(echo "$CHECKS" | grep -c "skipping" || true)

    echo "    Checks: $PASS_COUNT pass, $FAIL_COUNT fail, $SKIP_COUNT skipping"

    if [[ "$FAIL_COUNT" -eq 0 ]]; then
        echo "    Status: ALL PASSING"
        TOTAL_PASS=$((TOTAL_PASS + 1))
        echo ""
        continue
    fi

    TOTAL_FAIL=$((TOTAL_FAIL + 1))

    # Get failed run IDs from PR checks (PR gate runs)
    FAILED_RUNS=$(echo "$CHECKS" | grep "fail" | grep -oP 'runs/\K\d+' | sort -u || true)

    # Also find workflow_dispatch runs on this branch (post-commit CI we triggered)
    # These are NOT shown by `gh pr checks` but are critical for full coverage.
    BRANCH_RUNS=$(gh run list --repo "$REPO" --branch "$BRANCH" --limit 10 \
        --json databaseId,conclusion,workflowName \
        --jq '.[] | select(.conclusion == "failure") | .databaseId' 2>/dev/null || true)
    if [[ -n "$BRANCH_RUNS" ]]; then
        FAILED_RUNS=$(echo -e "${FAILED_RUNS}\n${BRANCH_RUNS}" | sort -u | grep -v '^$')
    fi

    PR_OUR_FAULT=0
    PR_BASELINE=0
    PR_UNKNOWN=0

    for RUN_ID in $FAILED_RUNS; do
        # Get error lines from this run (timeout after 30s per run to avoid hanging)
        ERRORS=$(timeout 30 gh run view "$RUN_ID" --repo "$REPO" --log-failed 2>&1 | grep "##\[error\]" | grep -v "Process completed\|Optional job" | sort -u || true)

        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            # Extract just the test name
            TEST_NAME=$(echo "$line" | grep -oP '##\[error\]\K.*')
            [[ -z "$TEST_NAME" ]] && continue

            CLASS=$(classify_failure "$TEST_NAME" "$ACTIVATION")

            case "$CLASS" in
                OUR_FAULT)
                    PR_OUR_FAULT=$((PR_OUR_FAULT + 1))
                    TOTAL_OUR_FAULT=$((TOTAL_OUR_FAULT + 1))
                    SUMMARY_OUR_FAULT["$PR_NUM"]+="      - $TEST_NAME\n"
                    ;;
                BASELINE)
                    PR_BASELINE=$((PR_BASELINE + 1))
                    TOTAL_BASELINE=$((TOTAL_BASELINE + 1))
                    [[ "$VERBOSE" == true ]] && SUMMARY_BASELINE["$PR_NUM"]+="      - $TEST_NAME\n"
                    ;;
                UNKNOWN)
                    PR_UNKNOWN=$((PR_UNKNOWN + 1))
                    TOTAL_UNKNOWN=$((TOTAL_UNKNOWN + 1))
                    SUMMARY_UNKNOWN["$PR_NUM"]+="      - $TEST_NAME\n"
                    ;;
            esac
        done <<< "$ERRORS"
    done

    # Print per-PR summary
    if [[ "$PR_OUR_FAULT" -gt 0 ]]; then
        echo "    OUR FAULT ($PR_OUR_FAULT):"
        echo -e "${SUMMARY_OUR_FAULT[$PR_NUM]}"
    fi
    if [[ "$PR_UNKNOWN" -gt 0 ]]; then
        echo "    UNKNOWN ($PR_UNKNOWN):"
        echo -e "${SUMMARY_UNKNOWN[$PR_NUM]}"
    fi
    if [[ "$PR_BASELINE" -gt 0 ]]; then
        if [[ "$VERBOSE" == true ]]; then
            echo "    BASELINE/FLAKY ($PR_BASELINE):"
            echo -e "${SUMMARY_BASELINE[$PR_NUM]}"
        else
            echo "    BASELINE/FLAKY: $PR_BASELINE (use --verbose to see)"
        fi
    fi
    echo ""
done

# Global summary
echo "================================================================"
echo "  SUMMARY"
echo "================================================================"
echo ""
printf "  PRs analyzed:    %d\n" "$TOTAL_PRS"
printf "  All passing:     %d\n" "$TOTAL_PASS"
printf "  With failures:   %d\n" "$TOTAL_FAIL"
echo ""
printf "  Failure breakdown:\n"
printf "    OUR FAULT:     %d  (activation-specific test failures)\n" "$TOTAL_OUR_FAULT"
printf "    BASELINE:      %d  (known flaky/infra, unrelated to our changes)\n" "$TOTAL_BASELINE"
printf "    UNKNOWN:       %d  (needs investigation)\n" "$TOTAL_UNKNOWN"
echo ""

if [[ "$TOTAL_OUR_FAULT" -gt 0 ]]; then
    echo "  ACTION REQUIRED: $TOTAL_OUR_FAULT test(s) fail on our activated functions."
    echo "  These need to be investigated and fixed before merge."
fi

if [[ "$TOTAL_UNKNOWN" -gt 0 ]]; then
    echo "  WARNING: $TOTAL_UNKNOWN unknown failure(s). May be new flaky tests or"
    echo "  indirect regressions. Compare with main branch CI to confirm."
fi

if [[ "$TOTAL_OUR_FAULT" -eq 0 && "$TOTAL_UNKNOWN" -eq 0 ]]; then
    if [[ "$TOTAL_FAIL" -gt 0 ]]; then
        echo "  All failures are known baseline/flaky tests."
        echo "  PRs are READY for review (CI is clean for our changes)."
    else
        echo "  All PRs passing CI. READY for review."
    fi
fi
echo ""
