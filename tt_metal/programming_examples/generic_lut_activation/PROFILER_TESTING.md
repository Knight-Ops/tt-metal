# Metal Profiler Integration Testing Report

**Date**: February 12, 2026
**Branch**: `feature/generic-lut-activation`
**Commit**: `2775adb2a8` - Add Metal profiler hooks to all generic LUT activation variants

## Summary

Added Metal device profiler integration to all generic LUT activation variants (native SFPU, piecewise rational, and embedded). This enables detailed kernel-level performance analysis and timing breakdowns.

## Changes Made

### Files Modified

1. **`tt_metal/programming_examples/generic_lut_activation/generic_lut_activation_native.cpp`**
   - Added `#include <tt-metalium/tt_metal_profiler.hpp>` at line 11
   - Added profiler call after line 370 (after `END_TIMER(DEVICE_TO_HOST)`)

2. **`tt_metal/programming_examples/generic_lut_activation/generic_lut_activation_rational.cpp`**
   - Added `#include <tt-metalium/tt_metal_profiler.hpp>` at line 11
   - Added profiler call after line 509 (after `END_TIMER(DEVICE_TO_HOST)`)

3. **`tt_metal/programming_examples/generic_lut_activation_embedded/generic_lut_activation.cpp`**
   - Added `#include <tt-metalium/tt_metal_profiler.hpp>` at line 11
   - Added profiler call after line 318 (after `END_TIMER(DEVICE_TO_HOST)`)

### Code Pattern Added

```cpp
// Read mesh device profiler results if enabled
if (std::getenv("TT_METAL_DEVICE_PROFILER")) {
    ReadMeshDeviceProfilerResults(*mesh_device);
    fmt::print("✓ Device profiler results written to generated/profiler/\n");
}
```

## Build Status

✅ **Compilation Verified**: All changes compiled successfully on original machine
⚠️ **Runtime Testing**: Device timeout error prevented runtime validation (machine may have hardware issues)

### Build Commands

```bash
cd /localdev/nkapre/tt-metal

# Pull latest changes
git pull origin feature/generic-lut-activation

# Full rebuild
./build_metal.sh --build-programming-examples

# Or rebuild specific variants:
cmake --build build --target programming_examples_generic_lut_activation_native_sfpu
cmake --build build --target programming_examples_generic_lut_activation_rational_n4d4_s2
cmake --build build --target programming_examples_generic_lut_activation_embedded_sigmoid_p3_s4
```

## Testing Instructions

### Prerequisites

- Working Tenstorrent hardware (Wormhole/Blackhole)
- tt-metal built with TRACY_ENABLE (default)
- Device properly initialized (no timeout errors)

### Test 1: Native SFPU with Profiler

```bash
export TT_METAL_DEVICE_PROFILER=1

./build/programming_examples/programming_examples_generic_lut_activation_native_sfpu \
  gelu --range-min -10 --range-max 10 --tiles 32 --precision fp32
```

**Expected Output**:
```
TIMING_KERNEL_EXECUTION: XXX.XXX
✓ Execution complete

✓ Device profiler results written to generated/profiler/
```

**Verify Profiler Output**:
```bash
ls -la generated/profiler/
# Should contain:
# - profile_log_device.csv
# - profile_log_host.csv

# Check device profiler data
head -20 generated/profiler/profile_log_device.csv
```

### Test 2: Piecewise Rational with Profiler

```bash
export TT_METAL_DEVICE_PROFILER=1

# Find a rational CSV file
RATIONAL_CSV=$(ls /localdev/nkapre/tt-polynomial-fitter/csvs/rational/*_fp32_*_4_4_uniform_rational.csv | head -1)

# Extract activation name from CSV filename
ACTIVATION=$(basename "$RATIONAL_CSV" | cut -d'_' -f1)

./build/programming_examples/programming_examples_generic_lut_activation_rational_n4d4_s2 \
  "$RATIONAL_CSV" \
  --activation "$ACTIVATION" \
  --range-min -10 --range-max 10 --tiles 32
```

**Expected Output**: Same profiler confirmation message as Test 1

### Test 3: Embedded Variant with Profiler

```bash
export TT_METAL_DEVICE_PROFILER=1

./build/programming_examples/generic_lut_activation_embedded_sigmoid_p3_s4 \
  --activation sigmoid --range-min -10 --range-max 10 --tiles 32 --precision fp32
```

**Expected Output**: Same profiler confirmation message as Test 1

### Test 4: Verify Profiler Data Structure

```bash
# Check CSV headers
head -1 generated/profiler/profile_log_device.csv

# Expected columns should include:
# - Core coordinates
# - RISC processor (BRISC, NCRISC, TRISC0, TRISC1, TRISC2)
# - Kernel name
# - Execution time (cycles/nanoseconds)
# - NOC bandwidth
# - DRAM bandwidth

# View kernel timing breakdown
grep -E "(reader|compute|writer)" generated/profiler/profile_log_device.csv | column -t -s,
```

## Known Issues

### Issue 1: Device Timeout on Original Machine

**Symptom**:
```
✗ Test failed with exception!
Timed out after waiting 5000 ms for arc core (8, 0) to start
```

**Status**: Hardware/device initialization issue on original test machine
**Impact**: Prevented runtime validation of profiler functionality
**Resolution**: Test on different machine with working hardware

**Workaround**: None - requires functional hardware

### Issue 2: Binary Naming Convention

**Binary Locations**:
- Native SFPU: `build/programming_examples/programming_examples_generic_lut_activation_native_sfpu`
- Rational: `build/programming_examples/programming_examples_generic_lut_activation_rational_nXdY_sZ`
- Embedded: `build/programming_examples/programming_examples_generic_lut_activation_embedded_<activation>_pX_sY`

**Note**: Binary names differ from source file names - check `CMakeLists.txt` for exact target names.

## Profiler Output Analysis

### Expected Performance Breakdown

The profiler will show timing for three kernel phases:

1. **Reader Kernel (BRISC/NCRISC)**: DRAM → L1 data movement
2. **Compute Kernel (TRISC0/1/2)**: SFPU polynomial evaluation
3. **Writer Kernel (BRISC/NCRISC)**: L1 → DRAM data movement

### Typical Timing Expectations

**For 32 tiles (32KB input)**:
- **Reader**: ~10-50 μs (depends on DRAM bandwidth)
- **Compute**: Varies by approximation complexity
  - Native SFPU: ~50-100 μs (hardware optimized)
  - Piecewise polynomial: ~100-500 μs (degree/segment dependent)
  - Piecewise rational: ~200-1000 μs (P(x)/Q(x) evaluation)
- **Writer**: ~10-50 μs (similar to reader)

### Comparison Metrics

Use profiler to compare:
- **Native vs Piecewise**: Accuracy vs speed tradeoff
- **Polynomial vs Rational**: P(x) vs P(x)/Q(x) complexity
- **Degree Impact**: Higher degree = better accuracy but slower
- **Segment Impact**: More segments = better accuracy but more overhead

## Next Steps for Testing

### Priority 1: Verify Profiler Works ✅

- [x] Code changes committed and pushed
- [ ] Pull changes on working device
- [ ] Rebuild binaries
- [ ] Run Test 1 (Native SFPU)
- [ ] Verify profiler CSV output exists
- [ ] Verify CSV contains valid timing data

### Priority 2: Comprehensive Profiling

- [ ] Profile all 3 variants (native, rational, embedded) for same activation
- [ ] Compare timing breakdowns
- [ ] Identify performance bottlenecks
- [ ] Document accuracy vs performance tradeoffs

### Priority 3: Performance Analysis

- [ ] Create profiler comparison script
- [ ] Generate timing charts (reader/compute/writer breakdown)
- [ ] Analyze NOC/DRAM bandwidth utilization
- [ ] Identify optimization opportunities

## Validation Checklist

When testing on new device, verify:

- [ ] Device initializes without timeout errors
- [ ] All 3 binaries build successfully
- [ ] Environment variable `TT_METAL_DEVICE_PROFILER=1` is set
- [ ] Profiler confirmation message appears in output
- [ ] `generated/profiler/` directory is created
- [ ] `profile_log_device.csv` exists and contains data
- [ ] CSV contains entries for reader, compute, and writer kernels
- [ ] Timing values are non-zero and reasonable
- [ ] Multiple runs produce consistent results

## Troubleshooting

### Problem: "Device profiler results written" message doesn't appear

**Diagnosis**:
```bash
# Check if profiler was enabled at build time
./build/programming_examples/programming_examples_generic_lut_activation_native_sfpu --help 2>&1 | grep -i profiler

# Check environment variable
echo $TT_METAL_DEVICE_PROFILER

# Verify binary was rebuilt after code changes
ls -l build/programming_examples/programming_examples_generic_lut_activation_native_sfpu
git log --oneline -1 -- tt_metal/programming_examples/generic_lut_activation/generic_lut_activation_native.cpp
```

**Solution**: Ensure `TT_METAL_DEVICE_PROFILER=1` and binary was rebuilt after profiler integration.

### Problem: Generated profiler directory is empty

**Diagnosis**:
```bash
ls -la generated/profiler/

# Check if profiler is enabled in build
grep TRACY_ENABLE build/CMakeCache.txt
```

**Solution**: tt-metal must be built with TRACY_ENABLE (enabled by default in standard builds).

### Problem: CSV file exists but has no data

**Diagnosis**:
```bash
wc -l generated/profiler/profile_log_device.csv
cat generated/profiler/profile_log_device.csv
```

**Solution**:
- Check if kernels actually executed (no early exit/exception)
- Verify device profiler was initialized (`InitDeviceProfiler()` called in Metal)
- Check for Metal profiler errors in stderr output

## References

### Key Files

- **Profiler API**: `tt_metal/api/tt-metalium/tt_metal_profiler.hpp`
- **Profiler Types**: `tt_metal/api/tt-metalium/profiler_types.hpp`
- **Example Usage**: `tt_metal/programming_examples/profiler/test_custom_cycle_count/`
- **Output Format**: `tt_metal/impl/profiler/profiler_paths.hpp`

### Documentation

- Metal Profiler Guide: `docs/source/tools/profiler.rst`
- Tracy Profiler: https://github.com/tenstorrent/tracy
- TT-NN Visualizer: https://github.com/tenstorrent/ttnn-visualizer

### Environment Variables

- `TT_METAL_DEVICE_PROFILER=1` - Enable device profiling
- `TT_METAL_DEVICE_PROFILER_DISPATCH=1` - Include dispatch cores
- `TT_METAL_PROFILER_DIR=<path>` - Set custom output directory
- `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` - Enable NOC event profiling

## Git Information

```bash
# View commit
git show 2775adb2a8

# Pull latest changes
git pull origin feature/generic-lut-activation

# Check if profiler code is present
git diff 11fb8f0678..2775adb2a8 -- \
  tt_metal/programming_examples/generic_lut_activation/generic_lut_activation_native.cpp \
  tt_metal/programming_examples/generic_lut_activation/generic_lut_activation_rational.cpp \
  tt_metal/programming_examples/generic_lut_activation_embedded/generic_lut_activation.cpp
```

## Contact

For issues or questions about this profiler integration, refer to:
- Commit: `2775adb2a8`
- Plan file: `~/.claude/plans/wobbly-nibbling-peacock.md` (if available)
- This report: `PROFILER_TESTING.md`

---

**Status**: ✅ Code integrated, ⏳ Runtime validation pending (waiting for working device)
**Last Updated**: February 12, 2026
