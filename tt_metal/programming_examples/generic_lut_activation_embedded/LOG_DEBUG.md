# Log Range Reduction Debug Notes

## Bugs Found & Fixed (Mar 2026)

### Bug 1: `int32_to_float` expects sign-magnitude, not two's complement

**Symptom**: All log outputs for inputs < 1.0 returned `-1,488,522,240` (garbage).

**Root cause**: SFPU's `int32_to_float(vInt, 0)` interprets the integer in **sign-magnitude** format, NOT two's complement. For negative exponents (e.g., `log(0.1)` has exponent `e = -4`), the two's complement representation `0xFFFFFFFC` is misread as sign=1, magnitude=`0x7FFFFFFC` ≈ 2 billion, giving `float(-2^31)`. Then `-2^31 * ln(2) = -1,488,522,240`.

**Fix**: Convert negative integers from two's complement to sign-magnitude before calling `int32_to_float`, matching the pattern used in the native TT log kernel (`ckernel_sfpu_log.h`):
```cpp
v_if(e_int < 0) {
    e_int = setsgn(~e_int + 1, 1);  // twos complement → sign-magnitude
}
v_endif;
vFloat e_float = int32_to_float(e_int, 0);
```

**Lesson**: Always check the SFPU's native kernel implementations (in `hw/ckernels/*/metal/llk_api/llk_sfpu/`) for how they handle integer↔float conversions. The SFPU is NOT a standard CPU — integer formats may differ.

### Bug 2: C++ ternary on SFPU vFloat/vInt

**Symptom**: Kernel failed to compile with `could not convert '__vCond' to 'bool'`.

**Root cause**: Inside a `v_if` block, writing `x_orig < 0.0f ? NaN : -inf` uses a C++ ternary, but `x_orig < 0.0f` returns `__vCond` (a SIMD predicate), not `bool`.

**Fix**: Replace ternary with nested `v_if`/`v_elseif`/`v_else`:
```cpp
// WRONG:
v_if(x_orig <= 0.0f) {
    result = x_orig < 0.0f ? NaN : -inf;  // won't compile
}

// CORRECT:
v_if(x_orig < 0.0f) {
    result = NaN;
}
v_elseif(x_orig == 0.0f) {
    result = -inf;
}
```

**Lesson**: SFPU comparisons always return `__vCond`, never `bool`. Cannot use ternary `?:`, `&&`, `||`, or any bool-expecting construct. Always use `v_if`/`v_elseif`/`v_else`/`v_endif`.

### Bug 3: Base-specific expansion constant

**Symptom**: log2 and log10 produced garbage values — e.g., `log2(4.0)` returned ~1.386 instead of 2.0.

**Root cause**: `log_expand()` hardcoded `constexpr float LN2 = 0.693...` for the reconstruction formula `e * C + poly(m)`. But:
- `log(x) = e * ln(2) + log(m)` → C = ln(2) ≈ 0.6931
- `log2(x) = e * 1 + log2(m)` → C = 1.0
- `log10(x) = e * log10(2) + log10(m)` → C = log10(2) ≈ 0.30103

**Fix**: Kernel uses `LOG_EXPAND_CONSTANT` define (emitted by `generate_adhoc_kernel.py` from CSV metadata). Each activation's CSV carries the correct constant in its `log_ln2_constant` metadata field.

**Lesson**: Range reduction expansion formulas may differ between related functions. Don't hardcode constants — parameterize them.

### Bug 4: Fitter metric computation (update_csv_metrics.py)

**Symptom**: `update_csv_metrics.py` crashed with `int * NoneType` for all log CSVs.

**Root cause**: Two issues:
1. Metadata key capture only matched `range_reduction_*` prefix, missing keys like `log_ln2_constant`
2. Fallback tried `activation_json.get('log_ln2_constant')` at top level, but the JSON doesn't have it there

**Fix**: Read all metadata keys from CSV (not just `range_reduction_*` prefixed ones), and use CSV metadata values directly with sensible defaults.

## Final Results (Blackhole, yolov4 shape, bf16)

| Activation | Config | MaxULP | MeanULP | Time (µs) |
|-----------|--------|--------|---------|-----------|
| log       | p4_s2  | 1      | 0.49    | 24.4      |
| log2      | p3_s4  | 1      | 0.50    | 31.9      |
| log10     | p4_s2  | 1      | 0.50    | 25.4      |

All three achieve sub-ULP accuracy in bf16.

## Files Modified

**tt-metal (kernels)**:
- `piecewise_generic.cpp` — log_expand sign-magnitude fix + LOG_EXPAND_CONSTANT
- `piecewise_generic_specialized.cpp` — v_if ternary fix
- `piecewise_rational.cpp` — log_expand sign-magnitude fix + LOG_EXPAND_CONSTANT
- `piecewise_rational_specialized.cpp` — v_if ternary fix
- `tools/generate_adhoc_kernel.py` — emit LOG_EXPAND_CONSTANT from CSV metadata

**tt-polynomial-fitter**:
- `activations/log.json`, `log2.json`, `log10.json` — added `log_expand_constant`
- `range_reduction/range_reduction.py` — `_build_log_params` reads per-activation constant
- `update_csv_metrics.py` — fixed metadata key capture + range reduction param loading
- 572 CSV files regenerated with correct metrics
