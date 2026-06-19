# Argument Parsing Refactoring Summary

## Analysis Results

### Common Arguments Across ALL Scripts (100% overlap)

| Argument | sweep_best | sweep_polynomial | sweep_rational | sweep_native_sfpu |
|----------|:----------:|:----------------:|:--------------:|:-----------------:|
| `--activation` | ✅ | ✅ | ✅ | ✅ |
| `--timeout` | ✅ | ✅ | ✅ | ✅ |
| `-h, --help` | ✅ | ✅ | ✅ | ✅ |

### Nearly-Common Arguments (75% overlap)

| Argument | sweep_best | sweep_polynomial | sweep_rational | sweep_native_sfpu |
|----------|:----------:|:----------------:|:--------------:|:-----------------:|
| `--skip-build` | ✅ | ✅ | ❌ | ❌ |
| `--skip-run` | ✅ | ✅ | ✅ | ❌ |
| `--dry-run` | ✅ | ✅ | ✅ | ❌ |

### Script-Specific Arguments

**sweep_best.sh:**
- `--use-mae` - Use MAE instead of max error metric

**sweep_polynomial.sh:**
- `--precision <type>` - bf16/fp32/both
- `--depth <depths>` - Segment counts (4,8,16,32)
- `--degree <degrees>` - Polynomial degrees (1,2,3,4,6,8,12,16)
- `--segmentation <type>` - uniform/chebyshev/curvature
- `--compare` - Show comparison table

**sweep_rational.sh:**
- `--tiles <count>` - Number of tiles (default: 256)
- `--runs <n>` - Runs per config (default: 2)
- `--compare` - Show comparison table

**sweep_native_sfpu.sh:**
- `--precision <type>` - bf16/fp32/both
- `--depth` - No-op flag (for consistency)

## Code Duplication Found

### Environment Setup (duplicated ~30 lines per script)
```bash
# Architecture detection
# Platform prefix setting
# Repository paths
# Build directories
# Polynomial fitter paths
```

### Argument Parsing (~60 lines per script for common args)
```bash
# --activation parsing
# --timeout parsing with validation
# --skip-build parsing
# --skip-run parsing
# --dry-run parsing
# -h/--help parsing
```

### Result File Path Logic (~10 lines per script)
```bash
# Per-activation vs consolidated file logic
# Directory creation
# Path construction
```

### Total Duplication
- **~100 lines** of identical/nearly-identical code per script
- **4 scripts** in generic_lut_activation
- **3 scripts** in generic_lut_activation_embedded
- **Total: ~700 lines** of duplicated code

## Solution Implemented

### Created: `argparse_common.sh`

**Exports:**
- `init_common_env()` - Set up paths and architecture
- `parse_common_args()` - Parse common arguments
- `show_common_options()` - Display common help
- `show_common_env_vars()` - Display env var help
- `get_results_file()` - Get per-activation or consolidated path
- `validate_precision()` - Validate precision values
- `validate_segmentation()` - Validate segmentation types

**Provides Variables:**
- `FILTER_ACTIVATION` - Activation filter
- `TIMEOUT_SECONDS` - Test timeout
- `SKIP_BUILD` - Skip build flag
- `SKIP_RUN` - Skip run flag
- `DRY_RUN` - Dry run flag
- `HELP_REQUESTED` - Help requested flag
- `ARCH_NAME` - Architecture name
- `PLATFORM_PREFIX` - Platform prefix
- `REPO_ROOT` - Repository root
- `BUILD_DIR` - Build directory
- `TT_POLY_FIT_DIR` - Polynomial fitter directory

## Usage Pattern

### Before (sweep_best.sh style)
```bash
#!/bin/bash
set -e

# ~30 lines of environment setup
REPO_ROOT=$(git rev-parse --show-toplevel)
if [[ -z "$ARCH_NAME" ]]; then
    if [[ $(hostname) == *"bh"* ]]; then
        export ARCH_NAME=blackhole
    # ... etc
fi

# ~10 lines of variable defaults
FILTER_ACTIVATION=""
TIMEOUT_SECONDS=120
SKIP_BUILD=false
# ... etc

# ~60 lines of argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --activation) FILTER_ACTIVATION="$2"; shift 2 ;;
        --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        # ... 50 more lines ...
    esac
done

# ~10 lines of result file logic
if [[ -n "$FILTER_ACTIVATION" ]]; then
    RESULTS_FILE="..."
# ... etc

# Actual sweep logic starts here (line ~110)
```

### After (with argparse_common.sh)
```bash
#!/bin/bash
set -e

# Source common library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/argparse_common.sh"

# Initialize (replaces ~30 lines)
init_common_env

# Script-specific defaults
USE_MAX_ERROR=true

# Help function (replaces ~30 lines)
show_help() {
    echo "Usage: $0 [OPTIONS]"
    show_common_options
    echo ""
    echo "Script-Specific Options:"
    echo "  --use-mae    Use MAE config"
    show_common_env_vars
    exit 0
}

# Parse arguments (replaces ~60 lines)
parse_common_args "$@"
set -- "${PARSED_ARGS[@]}"
[[ "$HELP_REQUESTED" == "true" ]] && show_help

# Parse script-specific args (much shorter!)
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-mae) USE_MAX_ERROR=false; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Get result file (replaces ~10 lines)
RESULTS_FILE=$(get_results_file "best" "$WORK_DIR")

# Actual sweep logic starts here (line ~35)
```

## Impact Analysis

### Lines of Code Reduction

| Script | Before | After | Reduction |
|--------|--------|-------|-----------|
| sweep_best.sh | ~716 | ~600 | -116 (-16%) |
| sweep_polynomial.sh | ~820 | ~700 | -120 (-15%) |
| sweep_rational.sh | ~550 | ~440 | -110 (-20%) |
| sweep_native_sfpu.sh | ~340 | ~260 | -80 (-24%) |

**Total:** ~2,426 lines → ~2,000 lines (-426 lines, -18% reduction)

### Maintainability Improvements

1. **Bug Fixes**: Fix once in argparse_common.sh, affects all scripts
2. **Consistency**: All scripts behave identically for common args
3. **Documentation**: Single source of truth for common behavior
4. **Testing**: Can test common logic in isolation
5. **Onboarding**: New developers see clear separation of concerns

### Backward Compatibility

- ✅ Existing scripts work unchanged
- ✅ Common library is opt-in
- ✅ No breaking changes
- ✅ Gradual migration possible

## Files Created

1. **`argparse_common.sh`** (256 lines)
   - Common argument parsing library
   - Environment setup functions
   - Helper functions for validation

2. **`sweep_best_refactored.sh.example`** (100 lines)
   - Example of refactored sweep_best.sh
   - Shows before/after comparison

3. **`ARGPARSE_MIGRATION.md`** (350 lines)
   - Step-by-step migration guide
   - Before/after code examples
   - Testing checklist

4. **`ARGPARSE_SUMMARY.md`** (this file)
   - Analysis results
   - Design decisions
   - Impact summary

## Next Steps

### Option A: Full Migration (Recommended for long-term)
1. Copy `argparse_common.sh` to generic_lut_activation_embedded/
2. Refactor all 7 sweep scripts to use common library
3. Test each refactored script thoroughly
4. Update documentation

### Option B: Gradual Migration (Lower risk)
1. Keep existing scripts as-is
2. Use common library for new scripts only
3. Migrate existing scripts one at a time
4. Keep old scripts as backup until fully tested

### Option C: No Migration (Keep current state)
1. Keep `argparse_common.sh` as reference
2. Use patterns for future script development
3. Don't modify existing working scripts
4. Accept duplication as technical debt

## Recommendation

**Gradual Migration (Option B)** is recommended because:
- ✅ Low risk - existing scripts continue working
- ✅ Immediate benefit for new development
- ✅ Can revert individual scripts if issues found
- ✅ Team can learn new pattern gradually
- ✅ Testing can be done incrementally

Start with sweep_native_sfpu.sh (simplest) → sweep_best.sh → sweep_rational.sh → sweep_polynomial.sh (most complex).

## Testing Plan

For each migrated script:
1. Run side-by-side with old version
2. Compare outputs byte-for-byte
3. Test all argument combinations
4. Verify error handling
5. Check help messages
6. Confirm result file paths correct

## Questions?

- **Q: Does this change sweep behavior?**
  - A: No, only internal organization changes

- **Q: Can we mix old and new scripts?**
  - A: Yes, completely compatible

- **Q: What if argparse_common.sh has a bug?**
  - A: Falls back to old scripts, or fix once in common library

- **Q: Should we migrate embedded scripts too?**
  - A: Yes, same benefits apply (copy argparse_common.sh to embedded directory)
