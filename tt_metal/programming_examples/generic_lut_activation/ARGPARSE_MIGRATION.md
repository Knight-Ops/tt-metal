# Argument Parsing Refactoring Guide

This document explains the common argument parsing library and how to migrate existing sweep scripts to use it.

## Overview

The `argparse_common.sh` library extracts shared argument parsing logic from all sweep scripts:
- Common flags: `--activation`, `--timeout`, `--skip-build`, `--skip-run`, `--dry-run`, `-h/--help`
- Environment setup: Architecture detection, paths, directories
- Result file management: Per-activation vs consolidated CSV paths
- Validation helpers: For precision, segmentation types, etc.

## Common Arguments Supported

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--activation <name>` | string | "" (all) | Filter to single activation |
| `--timeout <seconds>` | int | 120 | Timeout per test |
| `--skip-build` | flag | false | Skip build phase |
| `--skip-run` | flag | false | Skip run phase |
| `--dry-run` | flag | false | Show what would run |
| `-h, --help` | flag | false | Show help message |

## Environment Variables Set Automatically

| Variable | Description | Auto-Detection |
|----------|-------------|----------------|
| `ARCH_NAME` | blackhole or wormhole_b0 | From hostname (bh→blackhole) |
| `PLATFORM_PREFIX` | blackhole_ or wormhole_ | From ARCH_NAME |
| `REPO_ROOT` | Repository root directory | From git |
| `BUILD_DIR` | Build directory path | $REPO_ROOT/build_Release |
| `TT_POLY_FIT_DIR` | Polynomial fitter directory | /localdev/nkapre/... or ~/workspace/... |

## Migration Steps

### 1. Add source line at top of script

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/argparse_common.sh"
```

### 2. Initialize common environment

```bash
init_common_env
cd "$REPO_ROOT"
WORK_DIR="tt_metal/programming_examples/generic_lut_activation"
```

### 3. Replace manual environment setup

**BEFORE:**
```bash
# Auto-detect architecture from environment or hostname
if [[ -z "$ARCH_NAME" ]]; then
    if [[ $(hostname) == *"bh"* ]]; then
        export ARCH_NAME=blackhole
    else
        export ARCH_NAME=wormhole_b0
    fi
fi

# Set platform prefix for result files
if [[ "$ARCH_NAME" == "wormhole_b0" ]]; then
    PLATFORM_PREFIX="wormhole_"
elif [[ "$ARCH_NAME" == "blackhole" ]]; then
    PLATFORM_PREFIX="blackhole_"
fi

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
export TT_METAL_RUNTIME_ROOT=${TT_METAL_RUNTIME_ROOT:-$REPO_ROOT}
BUILD_DIR="$TT_METAL_RUNTIME_ROOT/build_Release"

TT_POLY_FIT_DIR=${TT_POLY_FIT_DIR:-"/localdev/nkapre/tt-polynomial-fitter"}
if [[ ! -d "$TT_POLY_FIT_DIR" ]]; then
    TT_POLY_FIT_DIR="$HOME/workspace/tt-polynomial-fitter"
fi
```

**AFTER:**
```bash
init_common_env  # One line!
```

### 4. Update help function to use common fragments

**BEFORE:**
```bash
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --activation <name>       Single activation (default: all)"
    echo "  --timeout <seconds>       Timeout per test (default: 120)"
    echo "  --skip-build              Skip build phase"
    echo "  --skip-run                Skip run phase"
    echo "  --dry-run                 Show what would run"
    echo "  --use-mae                 Use MAE config (script-specific)"
    echo "  -h, --help                Show help"
    # ... lots more boilerplate ...
}
```

**AFTER:**
```bash
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    show_common_options          # From argparse_common.sh
    echo ""
    echo "Script-Specific Options:"
    echo "  --use-mae                 Use MAE config instead of max error"
    echo ""
    show_common_env_vars         # From argparse_common.sh
    echo ""
    exit 0
}
```

### 5. Update argument parsing loop

**BEFORE:**
```bash
while [[ $# -gt 0 ]]; do
    case $1 in
        --activation)
            FILTER_ACTIVATION="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT_SECONDS="$2"
            if ! [[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
                echo "Error: --timeout must be a positive integer"
                exit 1
            fi
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-run)
            SKIP_RUN=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --use-mae)  # Script-specific
            USE_MAX_ERROR=false
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done
```

**AFTER:**
```bash
# Parse common arguments first
parse_common_args "$@"
set -- "${PARSED_ARGS[@]}"

# Show help if requested
if [[ "$HELP_REQUESTED" == "true" ]]; then
    show_help
fi

# Parse script-specific arguments only
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-mae)  # Only script-specific args remain!
            USE_MAX_ERROR=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done
```

### 6. Replace result file path logic

**BEFORE:**
```bash
if [[ -n "$FILTER_ACTIVATION" ]]; then
    RESULTS_DIR="$WORK_DIR/data/$ARCH_NAME/best"
    mkdir -p "$RESULTS_DIR"
    RESULTS_FILE="$RESULTS_DIR/${FILTER_ACTIVATION}.csv"
else
    RESULTS_FILE="$WORK_DIR/data/$ARCH_NAME/best_results.csv"
fi
```

**AFTER:**
```bash
RESULTS_FILE=$(get_results_file "best" "$WORK_DIR")
```

## Script-Specific Arguments to Keep

### sweep_best.sh
- `--use-mae` - Use MAE instead of max error metric

### sweep_polynomial.sh
- `--precision <type>` - Filter by bf16/fp32/both
- `--depth <depths>` - Comma-separated segment counts
- `--degree <degrees>` - Comma-separated degrees
- `--segmentation <type>` - uniform/chebyshev/curvature
- `--compare` - Show comparison table

### sweep_rational.sh
- `--tiles <count>` - Number of tiles
- `--runs <n>` - Number of runs per config
- `--compare` - Show comparison table

### sweep_native_sfpu.sh
- `--precision <type>` - Filter by bf16/fp32/both
- `--depth` - Accepted but ignored (for consistency)

## Benefits of Refactoring

1. **Reduced Duplication**: ~100 lines of boilerplate per script → 1 line
2. **Consistency**: All scripts parse common args identically
3. **Maintainability**: Fix a bug once, affects all scripts
4. **Clarity**: Script-specific logic clearly separated from common logic
5. **Easier Testing**: Common logic can be tested in isolation

## Validation Helpers

The library provides validation functions you can use:

```bash
# Validate precision value
validate_precision "$PRECISION" || exit 1

# Validate segmentation type
validate_segmentation "$SEGMENTATION" || exit 1
```

## Example: Full Migration

See `sweep_best_refactored.sh.example` for a complete example of how sweep_best.sh would look after migration.

**Before (sweep_best.sh):** ~716 lines
**After (refactored):** ~600 lines (-116 lines of boilerplate)

## Backward Compatibility

All existing sweep scripts continue to work without modification. The common library is opt-in.

## Next Steps

1. Review `argparse_common.sh` implementation
2. Test with `sweep_best_refactored.sh.example`
3. Gradually migrate other scripts as needed
4. Consider migrating generic_lut_activation_embedded scripts as well

## Testing Checklist

After migration, verify:
- [ ] `--help` shows correct help message
- [ ] `--activation gelu` filters correctly
- [ ] `--timeout 300` sets timeout correctly
- [ ] `--skip-build` skips build phase
- [ ] `--skip-run` skips run phase
- [ ] `--dry-run` shows what would run
- [ ] Architecture auto-detection works
- [ ] Result files go to correct paths
- [ ] Script-specific args still work
- [ ] Error messages for invalid args work
