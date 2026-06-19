# Cbrt Range Reduction — Debug Journey

## Summary

Three bugs in the cbrt range reduction kernel, each masking the next.
Total time: ~2 hours of debugging on Blackhole hardware.

## Bug 1: `vInt * int` doesn't exist on SFPU

**Symptom**: Kernel fails to JIT-compile (silently swallowed by `> /dev/null 2>&1` in sweep_best.sh).

**Root cause**: SFPU's `vInt` type has `+`, `-`, `&`, `|`, `^`, `<<`, `>>` but NO multiply operator. The expression `q * 3` in `cbrt_reduce` was a compile error.

**Fix**: Replace `q * 3` with `(q << 1) + q` (shift-left-1 = 2×, plus original = 3×).

## Bug 2: SFPU register spill (8 LREG limit)

**Symptom**: After fixing bug 1, cbrt runs but outputs **exactly 0.0** for all `|x| < 1`. Values with `|x| >= 1` are correct.

**Root cause**: The SFPU has only 8 vector registers (LREGs). The original `cbrt_reduce` function has ~12 temporaries when inlined. Combined with the polynomial evaluation's variables (`x`, `result`, `x_clamped`, plus Horner temporaries), the compiler silently spills registers — overwriting `x_orig`, `cbrt_q`, `cbrt_r`, and `cbrt_sign` with garbage.

**Debug method**: Dumped intermediate values by temporarily replacing the expansion with `result = int32_to_float(cbrt_q, 0)` and `result = x_orig`. Found that:
- `cbrt_q` contained float-like values (not integers) — register reused for Horner intermediates
- `x_orig` was corrupted for 1023/1024 values — register reused by cbrt_reduce's inlined temporaries
- Reading `dst_reg[d]` after poly eval also gave wrong exponent values (bug 3)

**Fix**: Split `cbrt_reduce` into two phases:
1. **`cbrt_reduce_m(x)`** — trivial mantissa extraction (2 SFPU ops: `setsgn` + `setexp`), called before poly eval. Zero register pressure.
2. **Post-poly-eval expansion** — extract biased exponent + sign before poly eval (2 vInts that survive through poly eval = 4 total live vars, well within 8 LREGs), compute q/r after poly eval.

Also skip `NO_CLAMP` for cbrt since range reduction guarantees `m ∈ [1, 2)`.

## Bug 3: SFPU `exexp` returns sign-magnitude integers

**Symptom**: After fixing bug 2, cbrt still outputs 0.0 for `|x| < 1`. Dumping `exexp` output showed `-2147483648` (0x80000000 = INT32_MIN) for ALL negative exponents instead of -1, -2, -3, etc.

**Root cause**: The SFPU's `exexp` instruction with DEBIAS mode returns integers in **sign-magnitude format**, not two's complement. The native log kernel (`ckernel_sfpu_log.h:37-38`) documents this:
```cpp
// Convert negative numbers: signed -> sign-magnitude
v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
```

Similarly, `int32_to_float` on SFPU expects sign-magnitude input. The magic-number floor-division technique produces two's complement integers (from `reinterpret<vInt>(float) - reinterpret<vInt>(float)`), which `int32_to_float` misinterprets for negative values.

**Fix**: Use `exexp_nodebias` (returns unsigned biased exponent, always non-negative) and do ALL arithmetic in float:
```cpp
// Before poly eval:
vInt cbrt_biased_e = exexp_nodebias(setsgn(x_orig, 0));  // unsigned, no sign-magnitude issue

// After poly eval:
vFloat e_float = int32_to_float(cbrt_biased_e, 0) - 127.0f;  // debias in float
vFloat q_approx = e_float * ONE_THIRD;
vFloat q_rounded = q_approx + magic;
vInt q = reinterpret<vInt>(q_rounded) - reinterpret<vInt>(magic);  // q as two's-complement int
vFloat q_back = q_rounded - magic;  // q as float (avoids int32_to_float sign-magnitude trap)
vFloat r_float = e_float - (q_back + q_back + q_back);  // 3*q in float
vInt r = reinterpret<vInt>(r_float + magic) - reinterpret<vInt>(magic);
```

Key insight: `q_back = q_rounded - magic` gives q as a proper float without ever going through `int32_to_float`. This sidesteps the sign-magnitude issue entirely.

## Final Results

| Precision | Config | MaxULP | MeanULP |
|-----------|--------|--------|---------|
| fp32 | p5_s2_uni | 53,886 (fp32 ULP) | 41,288 |
| bf16 | p2_s3_uni | 1 (bf16 ULP) | 0.66 |

Matches the polynomial fitter's predicted accuracy exactly.

## SFPU Lessons Learned

1. **No vInt multiply** — use `(x << 1) + x` for 3×, shifts+adds for other constants.
2. **8 LREG limit is HARD** — the compiler will silently spill without warning. Count your live variables. Inline functions expand all their temporaries into the caller's register space.
3. **`exexp` DEBIAS returns sign-magnitude** — not two's complement. Use `exexp_nodebias` + float debias to avoid the issue.
4. **`int32_to_float` expects sign-magnitude** — the magic-number round-to-nearest technique produces two's complement via `reinterpret<vInt>` subtraction. Don't feed these back through `int32_to_float` for negative values. Use `q_rounded - magic` instead.
5. **`dst_reg[d]` cannot be re-read after poly eval** — register spill corruption may affect the DST read path. Extract all needed values from `x_orig` BEFORE polynomial evaluation.
