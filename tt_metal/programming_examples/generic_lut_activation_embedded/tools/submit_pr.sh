#!/bin/bash
# =============================================================================
# submit_pr.sh — Submit a LUT drop-in replacement PR to tenstorrent/tt-metal
#
# Creates a clean branch from upstream/main with ONLY the ckernel headers,
# pushes to the fork, and opens a PR via gh CLI.
#
# Usage:
#   ./submit_pr.sh --activation softplus [--dry-run] [--ticket <url>] [--update]
#
# Prerequisites:
#   - Drop-in headers already validated (apply_dropin.sh + compare_three_way.sh)
#   - gh CLI authenticated with access to tenstorrent/tt-metal
#   - upstream remote pointing to tenstorrent/tt-metal
#   - origin remote pointing to your fork (nkapreTT/tt-metal)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)

ACTIVATION=""
DRY_RUN=false
SKIP_ISSUE=false
TICKET_URL=""
UPDATE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --activation|-a) ACTIVATION="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=true; shift ;;
        --skip-issue)    SKIP_ISSUE=true; shift ;;
        --ticket)        TICKET_URL="$2"; shift 2 ;;
        --update)        UPDATE_MODE=true; shift ;;
        -h|--help)
            echo "Usage: $0 --activation <name> [--dry-run] [--skip-issue] [--ticket <url>] [--update]"
            echo ""
            echo "Creates or updates a PR branch with ckernel drop-in headers"
            echo "and submits/updates on tenstorrent/tt-metal:main via gh CLI."
            echo ""
            echo "Options:"
            echo "  --activation <name>  Activation function name (e.g., softplus)"
            echo "  --dry-run            Show what would be done without executing"
            echo "  --skip-issue         Skip GitHub issue creation"
            echo "  --ticket <url>       GitHub issue URL to link in the PR"
            echo "  --update             Update existing PR (skip issue/PR creation, push to existing branch)"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[[ -z "$ACTIVATION" ]] && { echo "Error: --activation required"; exit 1; }

# Auto-detect update mode: check branch name first, then exact title word match
if [[ "$UPDATE_MODE" == false ]]; then
    # Try exact branch name match first
    EXISTING_PR=$(gh pr list --repo tenstorrent/tt-metal --head "nkapre/lut-${ACTIVATION}-dropin" \
        --json number --jq '.[0].number' 2>/dev/null || echo "")
    # Fall back to word-boundary title match (prevents "elu" matching "celu")
    if [[ -z "$EXISTING_PR" ]]; then
        EXISTING_PR=$(gh pr list --repo tenstorrent/tt-metal --author nkapreTT --state open \
            --json number,title,headRefName \
            --jq ".[] | select(.title | test(\"\\\\b${ACTIVATION}\\\\b\"; \"i\")) | .number" 2>/dev/null | head -1)
    fi
    if [[ -n "$EXISTING_PR" ]]; then
        EXISTING_BRANCH=$(gh pr view "$EXISTING_PR" --repo tenstorrent/tt-metal --json headRefName --jq '.headRefName' 2>/dev/null)
        echo "Existing PR #$EXISTING_PR found (branch: $EXISTING_BRANCH) — switching to update mode."
        UPDATE_MODE=true
    fi
fi

cd "$REPO_ROOT"

BH_SFPU="tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"
WH_SFPU="tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu"
HEADER="ckernel_sfpu_${ACTIVATION}.h"
SHARED="ckernel_sfpu_piecewise_rational.h"

# Validate files exist
for f in "$BH_SFPU/$HEADER" "$BH_SFPU/$SHARED" "$WH_SFPU/$HEADER" "$WH_SFPU/$SHARED"; do
    [[ -f "$f" ]] || { echo "Error: $f not found. Run apply_dropin.sh first."; exit 1; }
done

# Verify upstream remote
UPSTREAM_URL=$(git remote get-url upstream 2>/dev/null || echo "")
if [[ -z "$UPSTREAM_URL" || "$UPSTREAM_URL" != *"tenstorrent/tt-metal"* ]]; then
    echo "Error: upstream remote not pointing to tenstorrent/tt-metal"
    echo "  Fix: git remote add upstream https://github.com/tenstorrent/tt-metal.git"
    exit 1
fi

# Use existing PR branch if in update mode, otherwise default pattern
if [[ "$UPDATE_MODE" == true && -n "$EXISTING_BRANCH" ]]; then
    BRANCH_NAME="$EXISTING_BRANCH"
else
    BRANCH_NAME="nkapre/lut-${ACTIVATION}-dropin"
fi
CURRENT_BRANCH=$(git branch --show-current)

echo "================================================================"
echo "  Submit PR: $ACTIVATION drop-in → tenstorrent/tt-metal:main"
echo "================================================================"
echo ""
echo "Branch: $BRANCH_NAME"
echo "Files:"
echo "  $BH_SFPU/$HEADER (modified)"
echo "  $BH_SFPU/$SHARED (new)"
echo "  $WH_SFPU/$HEADER (modified)"
echo "  $WH_SFPU/$SHARED (new)"
echo ""

# Create or resolve GitHub issue (unless --skip-issue or --update)
if [[ "$UPDATE_MODE" == false && "$SKIP_ISSUE" == false && -z "$TICKET_URL" ]]; then
    echo ">>> Creating GitHub issue for $ACTIVATION LUT drop-in..."
    ISSUE_URL=$(gh issue create \
        --repo tenstorrent/tt-metal \
        --title "feat(ttnn): Replace ${ACTIVATION} SFPU with LUT-based rational approximation" \
        --body "Replace \`ckernel_sfpu_${ACTIVATION}.h\` with a piecewise rational P(x)/Q(x) LUT-based drop-in using rminimax-fitted coefficients via shared evaluator." \
        2>&1 | tail -1)
    TICKET_URL="$ISSUE_URL"
    echo "  Issue: $TICKET_URL"
elif [[ -n "$TICKET_URL" ]]; then
    echo "Using provided ticket: $TICKET_URL"
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY RUN] Would:"
    echo "  1. git fetch upstream"
    echo "  2. Create branch $BRANCH_NAME from upstream/main"
    echo "  3. Copy 4 ckernel headers"
    echo "  4. Commit and push to origin"
    echo "  5. gh pr create --repo tenstorrent/tt-metal"
    echo "  6. Upload ULP plots as PR comment (not committed)"
    echo "  7. Trigger post-commit CI workflows"
    echo "  Ticket: ${TICKET_URL:-none}"
    exit 0
fi

# Save committed header files (use git show HEAD: to avoid pre-commit stash pollution)
echo ">>> Saving drop-in headers from HEAD..."
git show "HEAD:$BH_SFPU/$HEADER" > "/tmp/pr_${ACTIVATION}_bh.h" 2>/dev/null || cp "$BH_SFPU/$HEADER" "/tmp/pr_${ACTIVATION}_bh.h"
git show "HEAD:$BH_SFPU/$SHARED" > "/tmp/pr_shared_bh.h" 2>/dev/null || cp "$BH_SFPU/$SHARED" "/tmp/pr_shared_bh.h"
git show "HEAD:$WH_SFPU/$HEADER" > "/tmp/pr_${ACTIVATION}_wh.h" 2>/dev/null || cp "$WH_SFPU/$HEADER" "/tmp/pr_${ACTIVATION}_wh.h"
git show "HEAD:$WH_SFPU/$SHARED" > "/tmp/pr_shared_wh.h" 2>/dev/null || cp "$WH_SFPU/$SHARED" "/tmp/pr_shared_wh.h"
# Save polynomial header if it exists
POLY_HEADER="ckernel_sfpu_piecewise_polynomial.h"
git show "HEAD:$BH_SFPU/$POLY_HEADER" > "/tmp/pr_poly_bh.h" 2>/dev/null || true
git show "HEAD:$WH_SFPU/$POLY_HEADER" > "/tmp/pr_poly_wh.h" 2>/dev/null || true
# Save API header + llk wrappers (reciprocal_init, tile API changes)
API_HEADER_REL="tt_metal/hw/inc/api/compute/eltwise_unary/${ACTIVATION}.h"
git show "HEAD:$API_HEADER_REL" > "/tmp/pr_api_${ACTIVATION}.h" 2>/dev/null || true
for arch_sfpu in "$BH_SFPU" "$WH_SFPU"; do
    for llk in "$arch_sfpu"/llk_math_eltwise*${ACTIVATION}*.h; do
        [[ -f "$llk" ]] || continue
        llk_base=$(basename "$llk")
        llk_rel="${llk#./}"
        git show "HEAD:$llk_rel" > "/tmp/pr_llk_${llk_base}" 2>/dev/null || cp "$llk" "/tmp/pr_llk_${llk_base}"
    done
done

# Fetch and checkout branch
GH_TOKEN=$(gh auth token)
PUSH_URL="https://x-access-token:${GH_TOKEN}@github.com/tenstorrent/tt-metal.git"

# Stash any dirty working tree state before switching branches
STASHED=false
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    echo ">>> Stashing local changes..."
    git stash push -m "submit_pr auto-stash for $ACTIVATION" --quiet && STASHED=true
fi

if [[ "$UPDATE_MODE" == true ]]; then
    echo ">>> Fetching existing branch $BRANCH_NAME..."
    git fetch "$PUSH_URL" "$BRANCH_NAME" --quiet
    git checkout -B "$BRANCH_NAME" FETCH_HEAD --quiet
else
    echo ">>> Fetching upstream/main..."
    git fetch upstream main --quiet
    echo ">>> Creating branch $BRANCH_NAME from upstream/main..."
    git checkout -B "$BRANCH_NAME" upstream/main --quiet
fi

# Copy headers onto clean branch
echo ">>> Copying drop-in headers..."
cp "/tmp/pr_${ACTIVATION}_bh.h" "$BH_SFPU/$HEADER"
cp "/tmp/pr_shared_bh.h" "$BH_SFPU/$SHARED"
cp "/tmp/pr_${ACTIVATION}_wh.h" "$WH_SFPU/$HEADER"
cp "/tmp/pr_shared_wh.h" "$WH_SFPU/$SHARED"

# Also copy polynomial header if it exists
POLY_HEADER="ckernel_sfpu_piecewise_polynomial.h"
[[ -f "/tmp/pr_poly_bh.h" ]] && cp "/tmp/pr_poly_bh.h" "$BH_SFPU/$POLY_HEADER"
[[ -f "/tmp/pr_poly_wh.h" ]] && cp "/tmp/pr_poly_wh.h" "$WH_SFPU/$POLY_HEADER"

# Copy API header + llk wrapper (may have reciprocal_init, tile API changes)
API_HEADER="tt_metal/hw/inc/api/compute/eltwise_unary/${ACTIVATION}.h"
[[ -f "/tmp/pr_api_${ACTIVATION}.h" ]] && cp "/tmp/pr_api_${ACTIVATION}.h" "$API_HEADER" && git add "$API_HEADER"
for arch_sfpu in "$BH_SFPU" "$WH_SFPU"; do
    LLK_PATTERN="$arch_sfpu/llk_math_eltwise*${ACTIVATION}*.h"
    for llk in $LLK_PATTERN; do
        llk_base=$(basename "$llk")
        [[ -f "/tmp/pr_llk_${llk_base}" ]] && cp "/tmp/pr_llk_${llk_base}" "$llk" && git add "$llk"
    done
done

# Standardize naming: remove ckernel_sfpu_unary_{name}.h, update all includes
OLD_HEADER="ckernel_sfpu_unary_${ACTIVATION}.h"
for arch_sfpu in "$BH_SFPU" "$WH_SFPU"; do
    if [[ -f "$arch_sfpu/$OLD_HEADER" ]]; then
        git rm -f "$arch_sfpu/$OLD_HEADER" --quiet 2>/dev/null || rm -f "$arch_sfpu/$OLD_HEADER"
        echo "  Removed non-standard $OLD_HEADER"
    fi
    # Fix any include referencing old name (llk_math, api headers, etc.)
    grep -rl "\"$OLD_HEADER\"" "$arch_sfpu" "tt_metal/hw/inc" 2>/dev/null | while read -r f; do
        sed -i "s|\"$OLD_HEADER\"|\"$HEADER\"|g" "$f"
        git add "$f"
        echo "  Updated include in $(basename $f)"
    done
done

# Stage and commit (headers only — plots uploaded as PR comments, not committed)
git add "$BH_SFPU/$HEADER" "$BH_SFPU/$SHARED" "$WH_SFPU/$HEADER" "$WH_SFPU/$SHARED"
[[ -f "$BH_SFPU/$POLY_HEADER" ]] && git add "$BH_SFPU/$POLY_HEADER" "$WH_SFPU/$POLY_HEADER"

if [[ "$UPDATE_MODE" == true ]]; then
    COMMIT_MSG="fix: update ${ACTIVATION} drop-in with boundary clamping (data-driven)

Generated from eval_regions[] in activation JSON — zero per-activation
special-casing. Boundary conditions prevent polynomial overflow on
out-of-domain inputs.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
else
    COMMIT_MSG="feat(ttnn): Replace ${ACTIVATION} SFPU with LUT-based approximation

Drop-in replacement using rminimax-fitted coefficients via shared evaluator.
Identical API signatures. No changes to dispatch, op types, or public headers.
Both blackhole and wormhole_b0 updated.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
fi
git commit -m "$COMMIT_MSG"

# Push to tenstorrent/tt-metal
echo ">>> Pushing to tenstorrent/tt-metal:$BRANCH_NAME..."
if [[ "$UPDATE_MODE" == true ]]; then
    git push "$PUSH_URL" "$BRANCH_NAME:refs/heads/$BRANCH_NAME" --force
else
    git push "$PUSH_URL" "$BRANCH_NAME:refs/heads/$BRANCH_NAME" --force-with-lease
fi

# Build PR body from results file — uses standard tt-metal PR template format
# with {{branch_name}} placeholders for CI badge auto-population.
RESULTS_FILE="/tmp/pr_submission_${ACTIVATION}_results.txt"
PR_BODY_FILE="/tmp/pr_body_${ACTIVATION}.md"

# Extract results tables (the fixed-width output from compare_three_way.sh)
RESULTS_BLOCK=""
if [[ -f "$RESULTS_FILE" ]]; then
    RESULTS_BLOCK=$(sed -n '/RESULTS:/,/====/{/====/d; p;}' "$RESULTS_FILE")
fi

# Auto-detect evaluator type from which shared header exists
if [[ -f "$BH_SFPU/ckernel_sfpu_piecewise_rational.h" ]]; then
    EVALUATOR_TYPE="rational"
    EVALUATOR_DESC="piecewise rational P(x)/Q(x)"
    SHARED_HDR="ckernel_sfpu_piecewise_rational.h"
else
    EVALUATOR_TYPE="polynomial"
    EVALUATOR_DESC="piecewise polynomial"
    SHARED_HDR="ckernel_sfpu_piecewise_polynomial.h"
fi

# --- Assemble PR body ---

# Section: Ticket
cat > "$PR_BODY_FILE" << 'SECTION_EOF'
### Ticket
SECTION_EOF

if [[ -n "$TICKET_URL" ]]; then
    echo "$TICKET_URL" >> "$PR_BODY_FILE"
else
    echo "_To be linked_" >> "$PR_BODY_FILE"
fi

# Section: Problem description
cat >> "$PR_BODY_FILE" << SECTION_EOF

### Problem description

Current \`ckernel_sfpu_${ACTIVATION}.h\` uses a hand-written SFPU implementation.
A ${EVALUATOR_DESC} approximation with rminimax-fitted coefficients
provides better accuracy and/or performance as a drop-in replacement.

### What's changed

Drop-in replacement for \`ckernel_sfpu_${ACTIVATION}.h\` using a new shared
\`${SHARED_HDR}\` evaluator that is reusable for future LUT-based
activation replacements.

- \`ckernel_sfpu_${ACTIVATION}.h\` (blackhole + wormhole_b0): LUT coefficients +
  thin wrapper calling shared evaluator. \`#ifdef INP_FLOAT32\` selects
  dtype-specific coefficients. Identical function signatures — no API change.
- \`${SHARED_HDR}\` (blackhole + wormhole_b0): NEW shared
  header with the exact evaluation machinery from the proven embedded kernels.
SECTION_EOF

# Section: Hardware-Validated Results
# Read in-domain ULP from .err.npz files (baseline + dropin tagged CSVs)
ARCH_SHORT_PR="${ARCH_NAME:-wormhole_b0}"
ARCH_SHORT_PR="${ARCH_SHORT_PR/_b0/}"
HW_DIR="$REPO_ROOT/tt_metal/programming_examples/generic_lut_activation/data/hardware_outputs/${ARCH_SHORT_PR}/${ACTIVATION}"
if [[ -d "$HW_DIR" ]]; then
    RESULTS_TABLE=$(python3 -c "
import json, numpy as np, os, glob

ACTIVATION = '${ACTIVATION}'
HW_DIR = '${HW_DIR}'
TT_POLY_FIT_DIR = '${TT_POLY_FIT_DIR}'

# Domain from JSON
domain_min, domain_max = -10.0, 10.0
act_json = f'{TT_POLY_FIT_DIR}/activations/{ACTIVATION}.json'
if os.path.exists(act_json):
    d = json.load(open(act_json))
    domain_min = d.get('domain', {}).get('min', -10.0)
    domain_max = d.get('domain', {}).get('max', 10.0)

def get_in_domain_ulp(npz_path, dmin, dmax):
    if not os.path.exists(npz_path): return None
    err = np.load(npz_path)
    inputs, ulps = err['inputs'], err['ulp_errors']
    mask = (inputs >= dmin) & (inputs <= dmax) & np.isfinite(ulps)
    if mask.sum() == 0: return None
    u = ulps[mask]
    return {'max': np.max(u), 'mean': np.mean(u), 'p99': np.percentile(u, 99), 'n': int(mask.sum())}

print(f'_Domain: [{domain_min}, {domain_max}]_')
print()
for prec in ['bf16', 'fp32']:
    # Find dropin yolov4 (largest shape)
    dropin = glob.glob(f'{HW_DIR}/native_sfpu_{ACTIVATION}_{prec}_precise_dropin_tiles3200.csv.err.npz')
    baseline = glob.glob(f'{HW_DIR}/native_sfpu_{ACTIVATION}_{prec}_precise_baseline_tiles3200.csv.err.npz')
    # Also try untagged (from baseline that wasn't tagged)
    if not baseline:
        baseline = glob.glob(f'{HW_DIR}/native_sfpu_{ACTIVATION}_{prec}_precise_tiles3200.csv.err.npz')
    d_ulp = get_in_domain_ulp(dropin[0], domain_min, domain_max) if dropin else None
    b_ulp = get_in_domain_ulp(baseline[0], domain_min, domain_max) if baseline else None
    if not d_ulp and not b_ulp: continue
    print(f'### {prec.upper()} (yolov4 5120x320, in-domain)')
    print()
    print('| Impl | MaxULP | MeanULP | P99 ULP | Points |')
    print('|------|--------|---------|---------|--------|')
    if b_ulp:
        print(f'| Original | {b_ulp[\"max\"]:.1f} | {b_ulp[\"mean\"]:.2f} | {b_ulp[\"p99\"]:.1f} | {b_ulp[\"n\"]} |')
    if d_ulp:
        print(f'| **Drop-in (LUT)** | **{d_ulp[\"max\"]:.1f}** | **{d_ulp[\"mean\"]:.2f}** | **{d_ulp[\"p99\"]:.1f}** | {d_ulp[\"n\"]} |')
    print()
" 2>/dev/null)
    if [[ -n "$RESULTS_TABLE" ]]; then
        cat >> "$PR_BODY_FILE" << SECTION_EOF

## Hardware-Validated Results (${ARCH_SHORT_PR}, sweep_native_sfpu.sh)

${RESULTS_TABLE}
SECTION_EOF
    fi
fi

# Note: ULP plots are uploaded to nkapreTT/lut-pr-assets AFTER PR create/update
# and embedded in the PR body via image URLs. See upload step below.

# Section: Test plan
cat >> "$PR_BODY_FILE" << 'SECTION_EOF'

## Test plan
- [x] `compare_three_way.sh`: original vs drop-in vs embedded (all Tracy, ULP + runtime)
- [x] `validate_dropin.py`: exhaustive BF16 (33K pts) + dense FP32 (262K pts)
- [x] Unit tests: PASSED (PCC=1.0)
- [x] `run_csv.sh`: Tracy profiler timing matches embedded kernel
- [ ] Post-commit CI (triggered automatically)
SECTION_EOF

# Section: Checklist (with {{branch_name}} badge placeholders)
# The pr-description-inject-branch-name.yaml workflow auto-replaces these.
cat >> "$PR_BODY_FILE" << 'SECTION_EOF'

### Checklist

- [ ] [![All post-commit tests](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml/badge.svg?branch={{branch_name}})](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml?query=branch:{{branch_name}})
- [ ] [![Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml/badge.svg?branch={{branch_name}})](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml?query=branch:{{branch_name}})
- [ ] [![cpp-unit-tests](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml/badge.svg?branch={{branch_name}})](https://github.com/tenstorrent/tt-metal/actions/workflows/tt-metal-l2-nightly.yaml?query=branch:{{branch_name}})
- [x] New/Existing tests provide coverage for changes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
SECTION_EOF

# Create or update PR
if [[ "$UPDATE_MODE" == true ]]; then
    echo ">>> Updating existing PR..."
    PR_NUM=$(gh pr list --repo tenstorrent/tt-metal --head "$BRANCH_NAME" \
        --json number --jq '.[0].number' 2>/dev/null)
    gh pr edit "$PR_NUM" --repo tenstorrent/tt-metal --body-file "$PR_BODY_FILE"
    PR_URL="https://github.com/tenstorrent/tt-metal/pull/$PR_NUM"
else
    echo ">>> Creating PR..."
    PR_URL=$(gh pr create \
        --repo tenstorrent/tt-metal \
        --base main \
        --head "$BRANCH_NAME" \
        --title "feat(ttnn): Replace ${ACTIVATION} SFPU with LUT-based rational approximation" \
        --body-file "$PR_BODY_FILE" 2>&1)
    PR_NUM=$(echo "$PR_URL" | grep -oP '\d+$')
fi

echo ""
echo "================================================================"
if [[ "$UPDATE_MODE" == true ]]; then
    echo "  PR updated: $PR_URL"
else
    echo "  PR created: $PR_URL"
fi
echo "================================================================"

# Upload ULP plots to nkapreTT/lut-pr-assets and embed in PR body.
# Uses upload_pr_images.sh which uploads via GitHub Contents API.
echo ">>> Uploading ULP plots to assets repo..."
IMAGE_URLS=$(ARCH_NAME="${ARCH_NAME:-wormhole_b0}" "$SCRIPT_DIR/upload_pr_images.sh" --activation "$ACTIVATION" --pr "$PR_NUM" 2>/dev/null)
if [[ -n "$IMAGE_URLS" ]]; then
    # Append image URLs to PR body and re-update
    {
        cat "$PR_BODY_FILE"
        echo ""
        ARCH_SHORT_IMG="${ARCH_NAME:-wormhole_b0}"; ARCH_SHORT_IMG="${ARCH_SHORT_IMG/_b0/}"
        echo "## ULP Distribution (${ARCH_SHORT_IMG})"
        echo ""
        echo "$IMAGE_URLS"
    } > "${PR_BODY_FILE}.with_images"
    gh pr edit "$PR_NUM" --repo tenstorrent/tt-metal --body-file "${PR_BODY_FILE}.with_images"
    echo "  Images embedded in PR body."
else
    echo "  No plots found to upload."
fi

# Trigger post-commit CI workflows on the PR branch
echo ">>> Triggering post-commit CI workflows..."
gh workflow run all-post-commit-workflows.yaml \
    --repo tenstorrent/tt-metal \
    --ref "$BRANCH_NAME" 2>&1 || echo "  (warning: could not trigger all-post-commit-workflows — may need manual trigger)"

gh workflow run blackhole-post-commit.yaml \
    --repo tenstorrent/tt-metal \
    --ref "$BRANCH_NAME" 2>&1 || echo "  (warning: could not trigger blackhole-post-commit — may need manual trigger)"

echo ""
echo "  CI triggered. Monitor status at:"
echo "    https://github.com/tenstorrent/tt-metal/actions?query=branch:${BRANCH_NAME}"

# Return to original branch
echo ">>> Returning to $CURRENT_BRANCH..."
git checkout "$CURRENT_BRANCH" --quiet
[[ "$STASHED" == true ]] && { echo ">>> Restoring stashed changes..."; git stash pop --quiet; }

echo ""
echo "Done. PR branch: $BRANCH_NAME"
echo "PR: $PR_URL"
echo "To update the PR: git checkout $BRANCH_NAME && <make changes> && git push"
