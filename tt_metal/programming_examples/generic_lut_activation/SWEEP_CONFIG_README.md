# Sweep Configuration - Single Source of Truth

**Date:** 2026-02-04
**Status:** ✅ Implemented

---

## Overview

All configuration for polynomial and rational approximations is centralized in:

**`sweep_config.dat`** - Single source of truth for:
1. `generate_cmake_all.py` - Generates CMake build targets
2. `sweep_all.sh` - Polynomial sweep testing
3. `sweep_rational.sh` - Rational sweep testing

---

## Configuration File Format

Simple key=value format (bash-sourceable, Python-parseable):

```bash
# Polynomial degrees to build and test
POLY_DEGREES=1,2,3,4,5,6,8,12,16,32

# Segment counts for polynomials
POLY_SEGMENTS=1,2,3,4,6,8,12,16,32

# Rational degrees to build and test
RATIONAL_DEGREES=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16

# Segment counts for rational approximations
RATIONAL_SEGMENTS=1,2
```

---

## How It Works

### 1. generate_cmake_all.py
```python
# Loads sweep_config.dat
config = load_config()
POLY_DEGREES = config.get('POLY_DEGREES', [...])
RATIONAL_DEGREES = config.get('RATIONAL_DEGREES', [...])

# Generates CMakeLists.txt with targets for all configs
```

**Output:** CMakeLists.txt with 155 targets (for degree 1-16)

### 2. sweep_all.sh
```bash
# Source configuration
CONFIG_FILE="$WORK_DIR/sweep_config.dat"
eval "$(grep -v '^\s*#' "$CONFIG_FILE" | ...)"

# Use config arrays
ALL_DEGREES=(${POLY_DEGREES[@]})
ALL_DEPTHS=(${POLY_SEGMENTS[@]})
```

**Output:** Tests all polynomial combinations (degree × segment × segmentation × precision)

### 3. sweep_rational.sh
```bash
# Parse configuration
RATIONAL_DEGREES_STR=$(grep "^RATIONAL_DEGREES=" "$CONFIG_FILE" | cut -d'=' -f2)

# Generate (n,n) and (n+1,n) patterns
for deg in "${RATIONAL_DEGREES_ARRAY[@]}"; do
    RATIONAL_CONFIGS+=("$deg,$deg,$seg")  # n,n pattern
    RATIONAL_CONFIGS+=("$((deg+1)),$deg,$seg")  # n+1,n pattern
done
```

**Output:** Tests all rational configurations (64 configs for degree 1-16)

---

## Adding Higher Degrees (Example: Degree 20)

### Step 1: Edit sweep_config.dat
```bash
# Change this line:
RATIONAL_DEGREES=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16

# To:
RATIONAL_DEGREES=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
```

### Step 2: Regenerate CMakeLists.txt
```bash
python3 generate_cmake_all.py
```

**Output:**
```
Generated CMakeLists.txt
  Polynomial targets: 90 (10 degrees x 9 segments)
  Rational targets: 80 ((n,n) + (n+1,n) patterns for degree 1-20)
  Native SFPU: 1
  Total: 171 targets
```

### Step 3: Rebuild
```bash
./build_metal.sh
```

### Step 4: Test
```bash
# Automatically picks up degree 17-20
./sweep_rational.sh

# Output:
# Loaded configuration from: sweep_config.dat
#   Rational degrees: 1 2 3 4 ... 19 20
#   Rational segments: 1 2
#   Generated 80 rational configurations
```

**That's it!** One file change → everything updates automatically ✅

---

## Benefits

### Before (Hardcoded Lists)
- 3 separate hardcoded lists in 3 different files
- Must manually sync when adding degrees
- Easy to forget to update one file
- Inconsistent configurations

### After (sweep_config.dat)
- ✅ **Single source of truth** - Edit one file
- ✅ **Version controlled** - Git tracks changes
- ✅ **Automatic propagation** - All tools use same config
- ✅ **Zero drift** - Impossible to get out of sync
- ✅ **Clear documentation** - Config file is self-documenting

---

## Configuration Parameters

### Polynomial Configuration
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `POLY_DEGREES` | int list | Polynomial degrees to build | `1,2,3,4,5,6,8,12,16,32` |
| `POLY_SEGMENTS` | int list | Segment counts | `1,2,3,4,6,8,12,16,32` |
| `POLY_SEGMENTATIONS` | string list | Segmentation types | `uniform,chebyshev,curvature` |
| `POLY_PRECISIONS` | string list | Precision types | `bf16,fp32` |

### Rational Configuration
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `RATIONAL_DEGREES` | int list | Rational degrees (both num & den) | `1,2,...,16` |
| `RATIONAL_SEGMENTS` | int list | Segment counts | `1,2` |

**Note:** Rational configs generate both patterns automatically:
- `(n,n)` - Equal degree: `n4_d4`, `n5_d5`, etc.
- `(n+1,n)` - Numerator one higher: `n5_d4`, `n6_d5`, etc.

---

## Error Handling

### Missing config file
```bash
❌ ERROR: Configuration file not found: sweep_config.dat
Please create sweep_config.dat with degree/segment configuration
```

### Invalid format
- Blank lines and `#` comments are ignored
- Lines without `=` are skipped
- Malformed values use defaults (with warning)

---

## Testing

### Verify configuration loads
```bash
# Test generate_cmake_all.py
python3 generate_cmake_all.py --dry-run | tail -20

# Test sweep_rational.sh
./sweep_rational.sh --skip-run | head -20

# Test sweep_all.sh
./sweep_all.sh --skip-run | head -20
```

### Expected output
All three should report consistent degree ranges:
```
Rational degrees: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
Polynomial degrees: 1 2 3 4 5 6 8 12 16 32
```

---

## Workflow

```
┌─────────────────────┐
│  sweep_config.dat   │ ← Single source of truth (git controlled)
└──────────┬──────────┘
           │
           ├────────────────────────────────┐
           │                                │
           ▼                                ▼
┌──────────────────────┐         ┌──────────────────────┐
│ generate_cmake_all.py│         │   Sweep Scripts      │
│  - Reads config      │         │  - sweep_all.sh      │
│  - Generates CMake   │         │  - sweep_rational.sh │
└──────────┬───────────┘         │  - Read config       │
           │                     │  - Test configs      │
           ▼                     └──────────────────────┘
┌──────────────────────┐
│   CMakeLists.txt     │
│  - Build targets     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   ./build_metal.sh   │
│  - Compiles binaries │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Sweep Scripts      │
│  - Test all configs  │
└──────────────────────┘
```

---

## Related Files

- `sweep_config.dat` - Configuration file (single source of truth)
- `generate_cmake_all.py` - CMake generator (reads config)
- `sweep_all.sh` - Polynomial sweep (reads config)
- `sweep_rational.sh` - Rational sweep (reads config)
- `CMakeLists.txt` - Generated file (DO NOT EDIT MANUALLY)

---

## Migration from Old System

### Old System (3 hardcoded lists)
```python
# In generate_cmake_all.py
POLY_DEGREES = [1, 2, 3, 4, 5, 6, 8, 12, 16, 32]
RATIONAL_DEGREES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
```
```bash
# In sweep_all.sh
ALL_DEGREES=(1 2 3 4 6 8 12 16)
```
```bash
# In sweep_rational.sh
RATIONAL_CONFIGS=(
    "1,1,1" "2,1,1" ... "9,9,2"  # 36 hardcoded lines
)
```

### New System (1 config file)
```bash
# In sweep_config.dat
POLY_DEGREES=1,2,3,4,5,6,8,12,16,32
RATIONAL_DEGREES=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
```

**3 files → 1 file** ✅

---

## Summary

**Problem:** Configuration scattered across 3 files, manual sync required

**Solution:** Single `sweep_config.dat` file, all tools read from it

**Result:**
- Zero-maintenance configuration updates
- Guaranteed consistency across tools
- Clear, version-controlled source of truth

**Next:** Just edit `sweep_config.dat` and rebuild! 🚀
