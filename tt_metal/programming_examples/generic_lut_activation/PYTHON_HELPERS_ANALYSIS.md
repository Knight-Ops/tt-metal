# Inline Python Helpers Analysis

## Summary of Inline Python Code

Total inline Python code found: **~750 lines** across 4 sweep scripts

## Detailed Inventory

### 1. Accuracy Computation (sweep_best.sh, sweep_rational.sh)
**Lines**: ~95 lines per script = **190 lines total**
**Location**:
- sweep_best.sh: lines 622-720
- sweep_rational.sh: lines 503-570

**Purpose**: Compute MAE, RMSE, max_error, mean_rel_error, max_ulp, mean_ulp from device CSV output

**Can be factored**: ✅ YES - **HIGH PRIORITY**

**Proposed script**: `compute_accuracy_from_csv.py`
```bash
# Current usage (inline):
stats=$(python3 << EOF
import csv, numpy as np, sys
...95 lines of code...
print(f"{mae},{rmse},{max_err},{mean_rel},{max_ulp},{mean_ulp}")
EOF
)

# Proposed usage (external script):
stats=$(python3 compute_accuracy_from_csv.py "$activation" "$csv_output" "$TT_POLY_FIT_DIR")
```

---

### 2. Comparison Tables (sweep_polynomial.sh, sweep_rational.sh)
**Lines**:
- sweep_polynomial.sh: **164 lines** (lines 677-840)
- sweep_rational.sh: **268 lines** (lines 686-953)
- Total: **432 lines**

**Purpose**: Generate comparison tables between polynomial/rational and SFPU baseline

**Differences**:
- sweep_polynomial.sh: Compares polynomial vs SFPU
- sweep_rational.sh: Compares rational vs polynomial vs SFPU (more complex)

**Can be factored**: ✅ YES - **HIGH PRIORITY**

**Proposed scripts**:
- `compare_polynomial_vs_sfpu.py` (164 lines → external script)
- `compare_rational_vs_all.py` (268 lines → external script)

**Current usage**:
```bash
python3 - "$RESULTS_FILE" "$SFPU_RESULTS_FILE" << 'COMPARE_EOF'
import csv, sys, collections
...164 lines for polynomial...
...268 lines for rational...
COMPARE_EOF
```

**Proposed usage**:
```bash
# Polynomial comparison
python3 compare_polynomial_vs_sfpu.py "$RESULTS_FILE" "$SFPU_RESULTS_FILE"

# Rational comparison
python3 compare_rational_vs_all.py "$RESULTS_FILE" "$POLY_RESULTS_FILE" "$SFPU_RESULTS_FILE"
```

---

### 3. Results Summary Display (sweep_native_sfpu.sh)
**Lines**:
- First occurrence: **35 lines** (lines 257-292) - Skip-run mode summary
- Second occurrence: **52 lines** (lines 666-718) - Post-execution summary
- Total: **87 lines** (but nearly identical, so ~50 unique lines)

**Purpose**: Display formatted results table for native SFPU tests

**Can be factored**: ✅ YES - **MEDIUM PRIORITY**

**Proposed script**: `display_native_sfpu_results.py`
```bash
# Current usage (inline):
export CURRENT_PRECISION RESULTS_FILE
python3 << 'EOFPYTHON'
import csv, numpy as np, os, sys
...35-52 lines of code...
EOFPYTHON

# Proposed usage (external script):
python3 display_native_sfpu_results.py "$RESULTS_FILE" "$CURRENT_PRECISION"
```

---

### 4. Tracy Statistics Parser (sweep_native_sfpu.sh)
**Lines**: **~10 lines** (line 503)
**Purpose**: Parse Tracy profiler output to extract timing statistics

**Can be factored**: ⚠️ MAYBE - **LOW PRIORITY** (small, script-specific)

**Current usage**:
```bash
tracy_stats=$(python3 -c "
import json
data = json.loads('$tracy_output')
# Extract timing stats
print(f'{mean},{min},{stddev}')
")
```

**Note**: Only used in Tracy mode, relatively small, could stay inline

---

### 5. Kernel Exec Min Calculator (sweep_helpers.sh, sweep_native_sfpu.sh)
**Lines**: **~3 lines** per occurrence
**Purpose**: Compute minimum from comma-separated values

**Can be factored**: ❌ NO - **Already a function** in sweep_helpers.sh

**Current implementation in sweep_helpers.sh**:
```bash
compute_kernel_exec_min() {
    local values="$1"
    python3 -c "
import sys
values = [float(x) for x in '$values'.split(',') if x]
print(min(values) if values else 0)
"
}
```

**Note**: This is fine as-is (small, reusable function)

---

## Summary Table

| Script | Purpose | Lines | Priority | Proposed File |
|--------|---------|-------|----------|---------------|
| Accuracy computation | Compute error metrics from CSV | 190 | HIGH | `compute_accuracy_from_csv.py` |
| Polynomial comparison | Compare polynomial vs SFPU | 164 | HIGH | `compare_polynomial_vs_sfpu.py` |
| Rational comparison | Compare rational vs poly vs SFPU | 268 | HIGH | `compare_rational_vs_all.py` |
| SFPU results display | Format and display SFPU results | 87 | MEDIUM | `display_native_sfpu_results.py` |
| Tracy stats parser | Parse Tracy JSON output | 10 | LOW | Keep inline |
| Min calculator | Compute minimum value | 3 | N/A | Already in sweep_helpers.sh ✓ |

**Total Lines to Factor Out**: 190 + 164 + 268 + 87 = **709 lines** (19.8% of 3582 total)

---

## Implementation Priority

### Phase 1A: Accuracy Computation (190 lines)
**Impact**: HIGH - Used in 2 scripts, nearly identical code
**Effort**: LOW - Straightforward extraction

Create: `compute_accuracy_from_csv.py`
- Input: activation name, CSV file, ground_truth directory
- Output: mae,rmse,max_error,mean_rel_error,max_ulp,mean_ulp
- Updates: sweep_best.sh, sweep_rational.sh

---

### Phase 1B: Comparison Tables (432 lines)
**Impact**: HIGH - Large code blocks, complex logic
**Effort**: MEDIUM - Need to preserve formatting and color codes

Create:
- `compare_polynomial_vs_sfpu.py` (164 lines)
- `compare_rational_vs_all.py` (268 lines)

**Note**: These scripts have similar structure but different comparison logic:
- Both read CSV files and compare metrics
- Both generate colored terminal output with ANSI codes
- Rational comparison is more complex (3-way comparison)

**Potential optimization**: Could create a base comparison library and two thin wrappers

---

### Phase 2: SFPU Results Display (87 lines)
**Impact**: MEDIUM - Used twice in same script (skip-run and post-exec)
**Effort**: LOW - Simple extraction

Create: `display_native_sfpu_results.py`
- Input: results file, precision filter
- Output: Formatted table to stdout
- Updates: sweep_native_sfpu.sh (2 locations)

---

## Additional Observations

### Comparison Script Commonalities

Both comparison scripts share significant common code:

1. **CSV Reading** (~30 lines each)
   - Read results CSV
   - Filter by status='pass'
   - Group by activation/precision

2. **ANSI Color Formatting** (~20 lines each)
   - Define color codes (GREEN, RED, YELLOW)
   - Format ratio comparisons with colors
   - Format error values in scientific notation

3. **Table Formatting** (~40 lines each)
   - Unicode box-drawing characters
   - Column alignment
   - Header/footer printing

**Recommendation**: Create `comparison_utils.py` module with shared functions:
- `read_results_csv(file, status_filter='pass')`
- `format_ratio(val1, val2, width=8)` - colored ratio display
- `format_error(val)` - scientific notation formatting
- `print_table(headers, rows, title)` - table display

Then both comparison scripts can import and use these utilities.

---

## Proposed File Structure

```
tt_metal/programming_examples/generic_lut_activation/
├── sweep_best.sh
├── sweep_native_sfpu.sh
├── sweep_polynomial.sh
├── sweep_rational.sh
├── sweep_helpers.sh
└── python_helpers/
    ├── __init__.py
    ├── compute_accuracy_from_csv.py       # 95 lines (accuracy metrics)
    ├── compare_polynomial_vs_sfpu.py      # 164 lines → ~100 lines (using utils)
    ├── compare_rational_vs_all.py         # 268 lines → ~150 lines (using utils)
    ├── display_native_sfpu_results.py     # 52 lines (results table)
    └── comparison_utils.py                # 80 lines (shared comparison code)
```

**Total new Python code**: ~477 lines (well-organized, testable)
**Total inline Python removed**: ~709 lines
**Net reduction**: ~232 lines (6.5%)
**Maintainability**: Significantly improved (Python code can be linted, tested, reused)

---

## Benefits of Factoring Out Python

### 1. **Testability**
- Python helpers can have unit tests
- Easier to test edge cases (empty results, missing files, etc.)
- Can use pytest for comprehensive testing

### 2. **Linting and Type Checking**
- Can run `black`, `flake8`, `mypy` on Python code
- Catch bugs before runtime
- Enforce consistent style

### 3. **IDE Support**
- Better syntax highlighting
- Code completion
- Refactoring tools

### 4. **Reusability**
- comparison_utils.py can be used by future comparison scripts
- Accuracy computation can be called from other contexts (notebooks, analysis scripts)

### 5. **Error Handling**
- Proper exception handling instead of Bash error propagation
- Better error messages
- Can add --help, --version flags to standalone scripts

### 6. **Documentation**
- Can add docstrings, type hints
- Can generate Sphinx documentation
- Easier to understand complex logic

### 7. **Performance**
- Python startup overhead negligible (already calling Python)
- Can optimize Python code more easily than heredoc
- Can add caching if needed

---

## Migration Risk Assessment

### Low Risk ✅
- **Accuracy computation**: Well-defined inputs/outputs, easy to test
- **SFPU results display**: Simple table formatting, no complex logic

### Medium Risk ⚠️
- **Comparison scripts**: Complex formatting with ANSI codes
  - Need to preserve exact table layout
  - Terminal width dependencies
  - Color code handling

### Mitigation
1. Test side-by-side with original inline versions
2. Add --no-color flag for testing exact output
3. Keep original inline versions commented out during transition
4. Run full sweep with both versions and compare outputs

---

## Recommended Implementation Order

1. ✅ **Phase 1A**: `compute_accuracy_from_csv.py` (190 lines, straightforward)
2. ✅ **Phase 1B**: `comparison_utils.py` base library (80 lines, reusable)
3. ✅ **Phase 1C**: `compare_polynomial_vs_sfpu.py` (164→100 lines, test with utils)
4. ✅ **Phase 1D**: `compare_rational_vs_all.py` (268→150 lines, test with utils)
5. ⏸️ **Phase 2**: `display_native_sfpu_results.py` (87 lines, lower priority)

**After Phase 1**: 622 lines factored out, significant maintainability improvement
