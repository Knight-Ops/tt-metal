# Sweep Scripts Redundancy Analysis

## Overview
Analysis of 4 sweep scripts (3582 total lines) to identify redundant code for refactoring.

## Scripts Analyzed
- sweep_best.sh (841 lines)
- sweep_native_sfpu.sh (731 lines)
- sweep_polynomial.sh (992 lines)
- sweep_rational.sh (1018 lines)

## Redundant Code Patterns Identified

### 1. Initialization Block (All 4 scripts)
**Location**: Lines 1-30 in each script
**Lines**: ~20 lines per script = 80 lines total

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/argparse_common.sh"
source "$SCRIPT_DIR/sweep_helpers.sh"

init_common_env
cd "$REPO_ROOT"
WORK_DIR="tt_metal/programming_examples/generic_lut_activation"
```

**Recommendation**: Create `init_sweep_env()` function in sweep_helpers.sh

---

### 2. Config File Loading (All 4 scripts)
**Location**: Lines 20-37 in each script
**Lines**: ~10 lines per script = 40 lines total

```bash
CONFIG_FILE="$WORK_DIR/sweep_config.dat"
if [[ -f "$CONFIG_FILE" ]]; then
    eval "$(grep -v '^\s*#' "$CONFIG_FILE" | grep -v '^\s*$' | sed 's/,/ /g' | sed 's/\([A-Z_]*\)=/\1=(/' | sed 's/$/)/')"
fi
```

**Recommendation**: Create `load_sweep_config()` function in sweep_helpers.sh

---

### 3. get_test_range() Function (All 4 scripts)
**Location**: Lines 30-58 in each script
**Lines**: ~20 lines per script = 80 lines total

**Identical implementation** in all scripts:
```bash
get_test_range() {
    local activation="$1"
    for entry in "${ACTIVATION_RANGES[@]}"; do
        local name="${entry%%:*}"
        local rest="${entry#*:}"
        local min="${rest%%:*}"
        local max="${rest#*:}"
        if [[ "$name" == "$activation" ]]; then
            echo "$min $max"
            return
        fi
    done
    echo "-10 10"
}
```

**Recommendation**: Move to sweep_helpers.sh (already exists there, scripts should use it)

---

### 4. Device Reset Logic (All 4 scripts)
**Location**: Multiple places in each script
**Lines**: ~10 lines per occurrence × 2 occurrences per script = 80 lines total

```bash
if command -v tt-smi &> /dev/null; then
    tt-smi -r 0 &> /dev/null || true
    sleep 2  # Give device time to reset
elif [[ -f /opt/venv/bin/tt-smi ]]; then
    /opt/venv/bin/tt-smi -r 0 &> /dev/null || true
    sleep 2
fi
```

**Recommendation**: Create `reset_device()` function in sweep_helpers.sh

---

### 5. Build Directory Auto-detection (sweep_polynomial.sh, sweep_rational.sh)
**Location**: Lines 28-35 in sweep_polynomial.sh, lines 134-140 in sweep_rational.sh
**Lines**: ~10 lines per script = 20 lines total

```bash
if [[ -d "$TT_METAL_RUNTIME_ROOT/build_Release/programming_examples" ]]; then
    BUILD_DIR="$TT_METAL_RUNTIME_ROOT/build_Release/programming_examples"
elif [[ -d "$TT_METAL_RUNTIME_ROOT/build/programming_examples" ]]; then
    BUILD_DIR="$TT_METAL_RUNTIME_ROOT/build/programming_examples"
else
    BUILD_DIR="$TT_METAL_RUNTIME_ROOT/build_Release/programming_examples"
fi
```

**Note**: sweep_rational.sh appends `/programming_examples`, sweep_polynomial.sh doesn't
**Recommendation**: Create `get_build_dir()` function with optional subdirectory parameter

---

### 6. Output Directory Creation (All 4 scripts)
**Location**: Lines 36-38 in each script
**Lines**: ~3 lines per script = 12 lines total

```bash
OUTPUT_BASE="$WORK_DIR/data/hardware_outputs"
mkdir -p "$OUTPUT_BASE"
```

**Recommendation**: Include in `init_sweep_env()` function

---

### 7. Python Accuracy Computation (sweep_best.sh, sweep_rational.sh)
**Location**: Lines 625-720 in sweep_best.sh, lines 503-570 in sweep_rational.sh
**Lines**: ~95 lines per script = 190 lines total

**Nearly identical inline Python blocks** that:
- Import ground_truth module from tt-polynomial-fitter
- Read device output CSV (input, output columns)
- Compute ground truth using `compute_ground_truth(activation, inputs)`
- Calculate MAE, RMSE, max_error, mean_rel_error
- Calculate ULP errors (max_ulp, mean_ulp)
- Print comma-separated results

**Minor differences**:
- sweep_best.sh has extra `compute_stats()` function for timing
- sweep_rational.sh uses `refs` instead of `expected` variable name
- Variable extraction slightly different

**Recommendation**: Create external Python script `compute_accuracy_from_csv.py` to replace both

---

### 8. Precision Detection/Filtering (All 4 scripts)
**Location**: Lines 126-131 in sweep_native_sfpu.sh, similar in others
**Lines**: ~5 lines per script = 20 lines total

```bash
if [[ -z "$FILTER_PRECISION" ]]; then
    PRECISIONS=("bf16" "fp32")  # Run both by default
else
    PRECISIONS=("$FILTER_PRECISION")  # Run only specified precision
fi
```

**Recommendation**: Create `get_precisions_to_test()` function in sweep_helpers.sh

---

## Summary

### Total Redundant Lines: ~522 lines (14.6% of codebase)

| Pattern | Lines per Script | Total Scripts | Total Lines | Priority |
|---------|------------------|---------------|-------------|----------|
| Python accuracy computation | 95 | 2 | 190 | HIGH |
| Device reset logic | 10×2 | 4 | 80 | HIGH |
| get_test_range() | 20 | 4 | 80 | HIGH |
| Initialization block | 20 | 4 | 80 | MEDIUM |
| Config file loading | 10 | 4 | 40 | MEDIUM |
| Build directory detection | 10 | 2 | 20 | LOW |
| Precision filtering | 5 | 4 | 20 | LOW |
| Output directory | 3 | 4 | 12 | LOW |

### High Priority Refactoring (350 lines)

1. **Python Accuracy Computation (190 lines)** - Create `compute_accuracy_from_csv.py`
2. **Device Reset Logic (80 lines)** - Create `reset_device()` function
3. **get_test_range() (80 lines)** - Already duplicated, move to sweep_helpers.sh

### Medium Priority Refactoring (120 lines)

4. **Initialization Block (80 lines)** - Create `init_sweep_env()` function
5. **Config Loading (40 lines)** - Create `load_sweep_config()` function

### Low Priority Refactoring (52 lines)

6. **Build Directory Detection (20 lines)** - Create `get_build_dir()` function
7. **Precision Filtering (20 lines)** - Create `get_precisions_to_test()` function
8. **Output Directory (12 lines)** - Include in init function

## Proposed New Functions for sweep_helpers.sh

```bash
# Initialize sweep environment
init_sweep_env() {
    cd "$REPO_ROOT"
    WORK_DIR="tt_metal/programming_examples/generic_lut_activation"
    OUTPUT_BASE="$WORK_DIR/data/hardware_outputs"
    mkdir -p "$OUTPUT_BASE"
}

# Load sweep configuration from sweep_config.dat
load_sweep_config() {
    local config_file="${1:-$WORK_DIR/sweep_config.dat}"
    if [[ -f "$config_file" ]]; then
        eval "$(grep -v '^\s*#' "$config_file" | grep -v '^\s*$' | sed 's/,/ /g' | sed 's/\([A-Z_]*\)=/\1=(/' | sed 's/$/)/')"
    fi
}

# Get test range for an activation
get_test_range() {
    local activation="$1"
    for entry in "${ACTIVATION_RANGES[@]}"; do
        local name="${entry%%:*}"
        local rest="${entry#*:}"
        local min="${rest%%:*}"
        local max="${rest#*:}"
        if [[ "$name" == "$activation" ]]; then
            echo "$min $max"
            return
        fi
    done
    echo "-10 10"
}

# Reset device using tt-smi
reset_device() {
    local device_id="${1:-0}"
    if command -v tt-smi &> /dev/null; then
        tt-smi -r "$device_id" &> /dev/null || true
        sleep 2
    elif [[ -f /opt/venv/bin/tt-smi ]]; then
        /opt/venv/bin/tt-smi -r "$device_id" &> /dev/null || true
        sleep 2
    fi
}

# Get build directory with optional subdirectory
get_build_dir() {
    local subdir="${1:-}"
    local base_dir
    if [[ -d "$TT_METAL_RUNTIME_ROOT/build_Release" ]]; then
        base_dir="$TT_METAL_RUNTIME_ROOT/build_Release"
    elif [[ -d "$TT_METAL_RUNTIME_ROOT/build" ]]; then
        base_dir="$TT_METAL_RUNTIME_ROOT/build"
    else
        base_dir="$TT_METAL_RUNTIME_ROOT/build_Release"
    fi

    if [[ -n "$subdir" ]]; then
        echo "$base_dir/$subdir"
    else
        echo "$base_dir"
    fi
}

# Get list of precisions to test
get_precisions_to_test() {
    if [[ -z "$FILTER_PRECISION" ]]; then
        echo "bf16 fp32"
    else
        echo "$FILTER_PRECISION"
    fi
}
```

## Proposed New Script: compute_accuracy_from_csv.py

```python
#!/usr/bin/env python3
"""
Compute accuracy metrics from device output CSV using ground truth.

Usage:
    compute_accuracy_from_csv.py <activation> <csv_file> <ground_truth_dir>

Output (comma-separated):
    mae,rmse,max_error,mean_rel_error,max_ulp,mean_ulp
"""

import sys
import csv
import numpy as np

def compute_accuracy(activation, csv_file, ground_truth_dir):
    # Add ground_truth module to path
    sys.path.insert(0, ground_truth_dir)
    from ground_truth import compute_ground_truth

    # Read device outputs
    inputs = []
    outputs = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            inputs.append(float(row['input']))
            outputs.append(float(row['output']))

    inputs = np.array(inputs, dtype=np.float32)
    outputs = np.array(outputs, dtype=np.float32)

    # Compute ground truth
    expected = compute_ground_truth(activation, inputs)

    # Compute errors
    errors = np.abs(outputs - expected)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)

    # Compute relative errors
    rel_errors = errors / (np.abs(expected) + 1e-10)
    mean_rel_error = np.mean(rel_errors)

    # Compute ULP errors (filter subnormals)
    min_normal_fp32 = 2.0 ** -126
    ulp_valid_mask = np.abs(expected) >= min_normal_fp32

    if np.any(ulp_valid_mask):
        expected_valid = expected[ulp_valid_mask]
        outputs_valid = outputs[ulp_valid_mask]
        errors_valid = np.abs(expected_valid - outputs_valid)

        # ULP for bf16 (7 mantissa bits)
        abs_y = np.abs(expected_valid)
        exponent = np.floor(np.log2(abs_y))
        golden_ulp = 2.0 ** (exponent - 7)

        ulp_errors = errors_valid / golden_ulp
        max_ulp = np.max(ulp_errors) if len(ulp_errors) > 0 else 0.0
        mean_ulp = np.mean(ulp_errors) if len(ulp_errors) > 0 else 0.0
    else:
        max_ulp = 0.0
        mean_ulp = 0.0

    return mae, rmse, max_error, mean_rel_error, max_ulp, mean_ulp

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: compute_accuracy_from_csv.py <activation> <csv_file> <ground_truth_dir>")
        sys.exit(1)

    activation = sys.argv[1]
    csv_file = sys.argv[2]
    ground_truth_dir = sys.argv[3]

    metrics = compute_accuracy(activation, csv_file, ground_truth_dir)
    print(f"{metrics[0]:.10e},{metrics[1]:.10e},{metrics[2]:.10e},{metrics[3]:.10e},{metrics[4]:.10e},{metrics[5]:.10e}")
```

## Implementation Plan

### Phase 1: High Priority (Immediate Impact)
1. Create `compute_accuracy_from_csv.py` script
2. Add `reset_device()` to sweep_helpers.sh
3. Move `get_test_range()` to sweep_helpers.sh and remove duplicates

**Savings**: ~350 lines (9.8% reduction)

### Phase 2: Medium Priority (Nice to Have)
4. Add `init_sweep_env()` to sweep_helpers.sh
5. Add `load_sweep_config()` to sweep_helpers.sh
6. Update all scripts to use new functions

**Savings**: ~120 lines (3.3% reduction)

### Phase 3: Low Priority (Polish)
7. Add `get_build_dir()` to sweep_helpers.sh
8. Add `get_precisions_to_test()` to sweep_helpers.sh

**Savings**: ~52 lines (1.5% reduction)

**Total Potential Savings**: ~522 lines (14.6% reduction from 3582 to 3060 lines)

## Benefits

1. **Maintainability**: Bug fixes in one place instead of four
2. **Consistency**: Identical behavior across all sweep scripts
3. **Readability**: Shorter scripts focused on script-specific logic
4. **Testability**: Helper functions can be unit tested
5. **Extensibility**: Easy to add new sweep scripts using helpers
