# Multi-Core Integration Plan for Embedded Variant

## Executive Summary

The non-embedded `generic_lut_activation` has been refactored for multi-core execution. This document outlines the plan to integrate the same multi-core support into the **embedded variant** (`generic_lut_activation_embedded`).

**Key Advantage**: Since the embedded variant compiles LUT data directly into the kernel, we **do NOT need per-core LUT buffers**, significantly simplifying the integration compared to the non-embedded version.

## Current State Analysis

### Non-Embedded Version (Multi-Core ✓)
- **Host code**: `generic_lut_activation.cpp` (690 lines)
- **Reader**: `kernels/dataflow/reader.cpp` - 3 runtime args: `(buffer_addr, n_tiles, start_tile_id)`
- **Writer**: `kernels/dataflow/writer.cpp` - 3 runtime args: `(buffer_addr, n_tiles, start_tile_id)`
- **Compute**: `kernels/compute/piecewise_generic.cpp` - 2 runtime args: `(n_tiles, compute_loop_factor)`
- **Work distribution**: Uses `split_work_to_cores()` to divide tiles across `core_group_1` and `core_group_2`
- **Per-core LUT buffers**: Creates individual HEIGHT_SHARDED L1 buffers for each core (lines 299-337)

### Embedded Version (Single-Core Only)
- **Host code**: `generic_lut_activation.cpp` (404 lines) - **needs multi-core refactoring**
- **Reader**: `kernels/dataflow/reader.cpp` - 2 runtime args: `(buffer_addr, n_tiles)` - **needs start_tile_id**
- **Writer**: `kernels/dataflow/writer.cpp` - 2 runtime args: `(buffer_addr, n_tiles)` - **needs start_tile_id**
- **Compute**: Hundreds of generated kernels include `../piecewise_generic.cpp` - 1 runtime arg: `(n_tiles)` - **already supports compute_loop_factor**
- **LUT handling**: **Zero L1 overhead** - LUT compiled into kernel binary at JIT time

## Integration Strategy

### Phase 1: Dataflow Kernels (Reader/Writer)

**Changes Required**:
1. Add `start_tile_id` as 3rd runtime argument to both reader and writer
2. Use `start_tile_id + i` when calling `noc_async_read_tile()` / `noc_async_write_tile()`
3. Add DPRINT debug statements (already present in non-embedded version)

**Files to Modify**:
- `kernels/dataflow/reader.cpp` (24 lines → ~28 lines)
- `kernels/dataflow/writer.cpp` (24 lines → ~28 lines)

**Implementation**:
```cpp
// reader.cpp - Line 8-10
uint32_t in_addr = get_arg_val<uint32_t>(0);
uint32_t n_tiles = get_arg_val<uint32_t>(1);
uint32_t start_tile_id = get_arg_val<uint32_t>(2);  // NEW

// Line 19 - Add debug print
DPRINT << "Reader: start_tile=" << start_tile_id << " n_tiles=" << n_tiles << ENDL();

// Line 24 - Offset tile index
noc_async_read_tile(start_tile_id + i, in_accessor, cb_addr);

// writer.cpp - Similar changes
```

### Phase 2: Host Code Refactoring

**Changes Required** (based on non-embedded lines 274-593):

#### 2.1 Work Distribution (Lines 274-298)
```cpp
// Get grid size and split work across cores
CoreCoord grid_size = mesh_device->compute_with_storage_grid_size();
fmt::print("Grid size: {}x{}\n", grid_size.x, grid_size.y);

auto [num_cores, all_cores, core_group_1, core_group_2,
      tiles_per_core_1, tiles_per_core_2] =
    split_work_to_cores(grid_size, n_tiles);

fmt::print("Work split: {} cores total\n", num_cores);
fmt::print("Group 1: {} cores, {} tiles/core\n",
           core_group_1.num_cores(), tiles_per_core_1);
if (!core_group_2.ranges().empty()) {
    fmt::print("Group 2: {} cores, {} tiles/core\n",
               core_group_2.num_cores(), tiles_per_core_2);
}
```

#### 2.2 Circular Buffer Creation (Lines 341-360)
- **Current**: Create CBs on single core `{0, 0}`
- **New**: Create CBs on `all_cores` (from work split)

```cpp
// Input and output CBs on ALL cores
CreateCircularBuffer(program, all_cores, CircularBufferConfig(...));

// SKIP LUT CB creation entirely (embedded mode advantage!)
```

#### 2.3 Kernel Creation (Lines 415-533)
- **Current**: Create kernels on single core `{0, 0}`
- **New**: Create kernels on `all_cores` / `core_group_1` / `core_group_2`

```cpp
// Reader/Writer on ALL cores
auto reader = CreateKernel(program, "...", all_cores, DataMovementConfig{...});
auto writer = CreateKernel(program, "...", all_cores, DataMovementConfig{...});

// Compute kernel for core_group_1
auto compute_kernel_1 = CreateKernel(program, COMPUTE_KERNEL_PATH, core_group_1, ComputeConfig{...});

// Compute kernel for core_group_2 (if exists)
KernelHandle compute_kernel_2 = 0;
if (!core_group_2.ranges().empty()) {
    compute_kernel_2 = CreateKernel(program, COMPUTE_KERNEL_PATH, core_group_2, ComputeConfig{...});
}
```

#### 2.4 Runtime Arguments (Lines 535-586)
- **Critical**: Set runtime args per-core with correct tile offsets

```cpp
uint32_t num_cores_y = grid_size.y;
uint32_t tiles_written = 0;

for (uint32_t i = 0; i < num_cores; i++) {
    CoreCoord core = {i / num_cores_y, i % num_cores_y};

    // Determine which group this core belongs to
    uint32_t tiles_this_core = 0;
    KernelHandle compute_kernel_id = 0;

    if (core_group_1.contains(core)) {
        tiles_this_core = tiles_per_core_1;
        compute_kernel_id = compute_kernel_1;
    } else if (core_group_2.contains(core)) {
        tiles_this_core = tiles_per_core_2;
        compute_kernel_id = compute_kernel_2;
    }

    // Reader args: (buffer_addr, n_tiles, start_tile_id)
    SetRuntimeArgs(program, reader, core,
        {input_buffer->address(), tiles_this_core, tiles_written});

    // Writer args: (buffer_addr, n_tiles, start_tile_id)
    SetRuntimeArgs(program, writer, core,
        {output_buffer->address(), tiles_this_core, tiles_written});

    // Compute args: (n_tiles) - embedded mode doesn't need compute_loop_factor
    SetRuntimeArgs(program, compute_kernel_id, core, {tiles_this_core});

    tiles_written += tiles_this_core;
}
```

### Phase 3: Testing and Validation

**Test Matrix**:
1. **Tile counts**: 2, 4, 8, 16, 32, 64, 128, 256
2. **Precisions**: bf16, fp32
3. **Activations**: sigmoid, gelu, tanh (representative set)
4. **Validation**: MAE vs ground truth, compare to single-core results

**Verification Steps**:
1. Confirm work distribution matches total tiles
2. Verify no tile overlap or gaps in DRAM access
3. Check accuracy matches single-core implementation
4. Measure performance scaling vs single-core

## Key Differences from Non-Embedded Integration

### What We DON'T Need (Embedded Advantage)
✅ **No per-core LUT buffer creation** (lines 299-337 in non-embedded)
   - Non-embedded creates HEIGHT_SHARDED L1 buffers for each core
   - Non-embedded writes LUT data to each core's L1
   - Non-embedded creates per-core CB pointing to L1 buffer
   - **Embedded**: LUT compiled into kernel binary, zero L1 overhead!

✅ **No LUT circular buffer creation** (CB index c_25)
   - Non-embedded uses `constexpr auto cb_lut = tt::CBIndex::c_25`
   - **Embedded**: Kernels use `#ifdef EMBEDDED_LUT` path, no CB needed

### What We DO Need (Same as Non-Embedded)
- Work distribution via `split_work_to_cores()`
- Create kernels on `all_cores` / `core_group_1` / `core_group_2`
- Set runtime args per-core with tile offsets
- Update reader/writer to accept `start_tile_id`

## Implementation Checklist

### Step 1: Dataflow Kernels ✅
- [ ] Update `reader.cpp` with 3-arg signature
- [ ] Update `writer.cpp` with 3-arg signature
- [ ] Test single-core still works

### Step 2: Host Code - Work Distribution ✅
- [ ] Add `#include <tt-metalium/work_split.hpp>`
- [ ] Add grid size query
- [ ] Add `split_work_to_cores()` call
- [ ] Add work distribution logging

### Step 3: Host Code - Kernel Creation ✅
- [ ] Change reader/writer creation to `all_cores`
- [ ] Create compute kernel for `core_group_1`
- [ ] Create compute kernel for `core_group_2` (conditional)
- [ ] Remove single-core CB creation, use `all_cores`

### Step 4: Host Code - Runtime Arguments ✅
- [ ] Implement per-core loop with tile offset tracking
- [ ] Set reader args: `{addr, tiles_this_core, tiles_written}`
- [ ] Set writer args: `{addr, tiles_this_core, tiles_written}`
- [ ] Set compute args: `{tiles_this_core}` (no compute_loop_factor in embedded)
- [ ] Add tile distribution verification

### Step 5: Testing ✅
- [ ] Test 2-256 tiles with bf16 precision
- [ ] Test 2-256 tiles with fp32 precision
- [ ] Verify accuracy matches single-core
- [ ] Test multiple activations (sigmoid, gelu, tanh)

## Expected Outcomes

### Performance
- **Target**: Linear scaling up to grid size (e.g., 110 cores on Blackhole)
- **Example**: 256 tiles → 110 cores utilized (36×3 + 74×2)

### Accuracy
- **Requirement**: Identical MAE to single-core implementation
- **Verification**: MAE < 1e-7 for all test cases

### Simplicity
- **Embedded advantage**: ~100 fewer lines than non-embedded version
- **No LUT management**: Skip entire per-core buffer creation section
- **Cleaner code**: Fewer moving parts, easier to maintain

## Risk Analysis

### Low Risk
✅ **Reader/writer changes**: Straightforward 3-arg signature
✅ **Work distribution**: Direct copy from non-embedded version
✅ **No LUT buffers**: Major simplification vs non-embedded

### Medium Risk
⚠️ **Runtime args loop**: Must correctly track `tiles_written` offset
⚠️ **Core group logic**: Ensure all cores assigned to correct group

### Mitigation
- Copy runtime args loop verbatim from non-embedded (lines 535-586)
- Add extensive logging to verify tile distribution
- Test with small tile counts first (2, 4, 8) before scaling up

## Timeline Estimate

- **Phase 1** (Dataflow): 30 minutes
- **Phase 2** (Host Code): 2 hours
- **Phase 3** (Testing): 1 hour
- **Total**: ~3-4 hours for complete integration

## References

### Key Files to Study
- `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/generic_lut_activation.cpp` (lines 274-593)
- `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/kernels/dataflow/reader.cpp`
- `/localdev/nkapre/tt-metal/tt_metal/programming_examples/generic_lut_activation/kernels/dataflow/writer.cpp`

### Memory Notes
- See `~/.claude/projects/-localdev-nkapre-tt-metal/memory/MEMORY.md` section "Multi-Core Implementation"
- Key insight: Per-core LUT buffers required for non-embedded, NOT needed for embedded

## Conclusion

Multi-core integration for the embedded variant is **significantly simpler** than the non-embedded version due to zero LUT L1 overhead. The main work is refactoring the host code to distribute work across cores and updating dataflow kernels to support tile offsets. The compute kernels require no changes since they already accept `n_tiles` as a runtime argument.

This integration will unlock 100× performance scaling for large tile counts on multi-core grids.
