#!/bin/bash
# =============================================================================
# pr_copilot.sh — Fetch and summarize Copilot review feedback from our PRs
#
# Usage:
#   ./pr_copilot.sh                    # All open PRs by nkapreTT
#   ./pr_copilot.sh --pr 39362         # Specific PR
#   ./pr_copilot.sh --pr 39362,39377   # Multiple PRs
# =============================================================================
set -e

REPO="tenstorrent/tt-metal"
AUTHOR="nkapreTT"
PRS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --pr|-p)     PRS="$2"; shift 2 ;;
        --repo)      REPO="$2"; shift 2 ;;
        --author)    AUTHOR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--pr <num,num,...>] [--repo owner/repo] [--author login]"
            echo ""
            echo "Fetches Copilot code review comments from GitHub PRs and"
            echo "summarizes actionable suggestions."
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Resolve PR list
if [[ -z "$PRS" ]]; then
    echo "Fetching open PRs by $AUTHOR on $REPO..."
    PRS=$(gh pr list --repo "$REPO" --author "$AUTHOR" --state open \
        --json number --jq '.[].number' | tr '\n' ',')
    PRS="${PRS%,}"  # trim trailing comma
fi

[[ -z "$PRS" ]] && { echo "No PRs found."; exit 0; }

echo ""
echo "================================================================"
echo "  Copilot Review Feedback: $REPO"
echo "================================================================"
echo ""

TOTAL_COMMENTS=0
TOTAL_PRS=0

IFS=',' read -ra PR_ARRAY <<< "$PRS"
for pr in "${PR_ARRAY[@]}"; do
    pr=$(echo "$pr" | tr -d ' ')
    title=$(gh api "repos/$REPO/pulls/$pr" --jq '.title' 2>/dev/null)

    # Get copilot review IDs
    review_ids=$(gh api "repos/$REPO/pulls/$pr/reviews" \
        --jq '.[] | select(.user.login == "copilot-pull-request-reviewer[bot]") | .id' 2>/dev/null)

    [[ -z "$review_ids" ]] && continue

    # Collect all inline comments
    comments=""
    for rid in $review_ids; do
        batch=$(gh api "repos/$REPO/pulls/$pr/reviews/$rid/comments" \
            --jq '.[] | {path: .path, line: (.line // .original_line), body: .body}' 2>/dev/null)
        [[ -n "$batch" ]] && comments="${comments}${batch}"$'\n'
    done

    comment_count=$(echo "$comments" | grep -c '"path"' 2>/dev/null || echo 0)
    [[ "$comment_count" -eq 0 ]] && continue

    TOTAL_PRS=$((TOTAL_PRS + 1))
    TOTAL_COMMENTS=$((TOTAL_COMMENTS + comment_count))

    echo "================================================================"
    echo "PR #$pr: $title ($comment_count comments)"
    echo "================================================================"
    echo ""

    # Parse and display each comment
    for rid in $review_ids; do
        gh api "repos/$REPO/pulls/$pr/reviews/$rid/comments" \
            --jq '.[] | "  \(.path):\(.line // "?")\n    \(.body | split("\n") | join("\n    "))\n"' 2>/dev/null
    done
    echo ""
done

echo "================================================================"
echo "  Summary: $TOTAL_COMMENTS comments across $TOTAL_PRS PRs"
echo "================================================================"

# Categorize common themes
echo ""
echo "Common themes (grep across all comments):"

# Save all comments to temp for analysis
TMPFILE="/tmp/pr_copilot_all_comments.txt"
> "$TMPFILE"
for pr in "${PR_ARRAY[@]}"; do
    pr=$(echo "$pr" | tr -d ' ')
    review_ids=$(gh api "repos/$REPO/pulls/$pr/reviews" \
        --jq '.[] | select(.user.login == "copilot-pull-request-reviewer[bot]") | .id' 2>/dev/null)
    for rid in $review_ids; do
        gh api "repos/$REPO/pulls/$pr/reviews/$rid/comments" \
            --jq '.[] | "PR#'$pr' \(.path):\(.line // "?"): \(.body)"' 2>/dev/null >> "$TMPFILE"
    done
done

# Count occurrences of common patterns
for pattern in "macro" "static_assert" "underflow" "include" "init" "range" "float_to_fp16b" "comment" "validated"; do
    count=$(grep -ci "$pattern" "$TMPFILE" 2>/dev/null || echo 0)
    [[ "$count" -gt 0 ]] && echo "  $pattern: mentioned in $count comment(s)"
done

echo ""
echo "Full comments saved to: $TMPFILE"
