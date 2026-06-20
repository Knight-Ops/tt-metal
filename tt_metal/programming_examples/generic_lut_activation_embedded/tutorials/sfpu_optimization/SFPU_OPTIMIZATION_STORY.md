# Optimizing an SFPU Vector Kernel From Scratch

A hands-on, **runnable** walk through optimizing a Tensix SFPU compute kernel — taking a
piecewise activation evaluator from a naive textbook implementation to a tuned one, one
change at a time, **measuring the win at every step on real silicon.**

Every number below is produced by `./run_tutorial.sh` (Blackhole, fp32, 256-tile shape).
Each rung is a self-contained kernel that computes the *same* function; the only thing that
changes between rungs is the one optimization being taught. Correctness is gated at every
step: a rung's output must match the exact reference, so each speedup is "same answer,
fewer microseconds."

> Regenerate all numbers: `./run_tutorial.sh all`. The benchmark coefficients are a fixed
> seed (`gen_bench.py`), so results are deterministic.

---

## 1. The problem

A piecewise activation evaluator does, per input element:
1. **Select a segment** — find which interval `[bᵢ, bᵢ₊₁)` the input `x` falls in.
2. **Evaluate a polynomial** (or rational `P(x)/Q(x)`) for that segment.

On the SFPU this looks simple, but the naive form leaves most of the chip's throughput on
the floor. We'll fix that in six steps for polynomials and five for rationals.

**The benchmark** (deliberately synthetic, so every optimization is exercised):
- *Polynomial:* 16 segments, max degree 8, **odd parity** (even coefficients are zero),
  with **mixed per-segment degree** (some segments need fewer terms).
- *Rational:* 4 segments, n8d8, **odd numerator / even denominator** (denominator ≈ 1, so
  it's well-conditioned).

---

## 2. SFPU primer & the cost model

The SFPU (Special Function Processor Unit) is the Tensix compute engine for transcendental
and elementwise math. Three facts drive every optimization here:

- **It is 32-lane SIMD.** One `vFloat` holds 32 elements; one `vFloat` FMA does 32
  multiply-adds. There is *no scalar mode* — the moment you write a `vFloat` Horner step,
  you're already vectorized. (That's why this tutorial has no "add vectorization" rung — it's
  free.)
- **`v_if` / `v_endif` is predicated, not branching.** *All lanes execute both sides* of a
  `v_if`; the result is masked on write. So a segment cascade `for s: v_if(x >= b[s]) {...}`
  evaluates **every segment's polynomial for every element**, then keeps the right one. Cost
  scales with `Σ(segments × degree)`, not `O(degree)`.
- **The register file is small.** Push too many live `vFloat`s and the compiler's register
  allocator spills — or, at `-O3 -flto`, *crashes* with a reload ICE. Several optimizations
  here are bounded by register pressure, not by math.

Our headline static metric is the **FMA count** — the number of fused multiply-adds the
predicated cascade performs per element. Measured device time tracks it, but not perfectly:
the kernel also pays for segment compares, register moves, and pipeline latency.

---

## 3. Act I — the polynomial ladder

Baseline P0 is the naive textbook kernel: a runtime loop over segments, and inside each, a
runtime Horner loop over the coefficients.

```cpp
for (int d = 0; d < 32; d++) {            // 32 DST rows
    vFloat x = dst_reg[d];
    vFloat result = 0.0f;
    for (uint32_t s = 0; s < NUM_SEGMENTS; s++) {
        v_if (x >= lut[s]) {
            vFloat acc = lut[base + DEG];
            for (int k = DEG - 1; k >= 0; k--) acc = acc * x + lut[base + k];  // runtime Horner
            result = acc;
        }
        v_endif;
    }
    dst_reg[d] = result;
}
```

| rung | optimization | µs | step | cumulative | FMA |
|------|--------------|----:|-----:|-----------:|----:|
| **P0** naive | runtime segment loop + runtime Horner | 85.1 | — | 1.0× | 128 |
| **P1** unrolled | compile-time template/`constexpr` unroll | 22.4 | 3.79× | **3.79×** | 128 |
| **P2** dual-eval | 2 DST rows/iter, shared coeff loads (ILP) | 16.9 | 1.33× | 5.03× | 128 |
| **P3** parity | x²-Horner over odd coeffs (½ the FMAs) | 16.2 | 1.04× | 5.25× | 64 |
| **P4** adaptive | per-segment effective degree | 15.8 | 1.03× | **5.40×** | 56 |

### P1 — compile-time unroll (the dominant win: 3.8×)
Replacing the runtime loops with template recursion (`if constexpr`, `__attribute__((always_inline))`)
lets the compiler emit a flat FMA chain with the coefficient indices folded to constants. The
naive runtime loops paid for loop counters, bounds checks, and — worse — the optimizer
generating `constprop.isra` spill code it couldn't undo. **Same FMA count (128), 3.8× faster.**
Lesson: on the SFPU, *loop overhead and spills dominate* a naive kernel far more than the
arithmetic does.

### P2 — dual-eval (1.33×)
Process two DST rows per iteration with two independent Horner chains that **share each
coefficient load**:
```cpp
vFloat ck = c[K];      // loaded once
a0 = a0 * x0 + ck;     // chain 0
a1 = a1 * x1 + ck;     // chain 1
```
The two chains are independent, so the SFPU can keep its pipeline full (instruction-level
parallelism) instead of stalling on each FMA's latency. Same FMAs, fewer stalls.

### P3 — parity x²-Horner (½ the FMAs)
The benchmark is odd: `P(x) = x·(c₁ + c₃x² + c₅x⁴ + c₇x⁶)`. Evaluate in the `x²` basis with
stride-2 coefficients — **half the Horner length** (FMA 128 → 64). But notice the *time*
barely moves (16.9 → 16.2µs): at this point the kernel is no longer FMA-bound, so halving
FMAs buys little. **A real lesson in not optimizing the wrong thing.**

> **Why P3 is single-eval, not dual+parity.** Stacking parity on top of P2's dual-eval at
> degree 8 overflows the SFPU register file and the compiler **crashes** (`-O3 -flto` reload
> ICE: "maximum number of generated reload insns"). This is the exact register-pressure limit
> the production kernel guards against. Parity's win is the FMA halving, so it's applied on
> single-eval.

### P4 — adaptive per-segment degree (1.03×)
Several segments need fewer terms. Using a per-segment compile-time degree
(`BENCH_SEGMENT_DEGREES[S]`) skips FMAs on the reducible segments (effective FMA 64 → 56).
Small here because only some segments shrink — but free, and larger when the fit varies more.

---

## 4. Act II — the rational ladder

Rationals add a denominator and a **reciprocal** (Newton-Raphson, ~10 SFPU ops). Baseline R0
evaluates numerator and denominator Horners and takes a reciprocal *inside every segment's*
`v_if`.

| rung | optimization | µs | step | cumulative |
|------|--------------|----:|-----:|-----------:|
| **R0** naive | per-seg num/den Horner + reciprocal-in-`v_if` | 45.2 | — | 1.0× |
| **R1** unrolled | compile-time unroll | 13.9 | 3.26× | **3.26×** |
| **R2** interleaved | num + den Horner in lockstep (ILP) | 12.2 | 1.14× | 3.71× |
| **R3** parity | x²-Horner: odd num, even den | 10.9 | 1.12× | 4.15× |
| **R4** deferred reciprocal | ONE reciprocal outside the cascade | 8.8 | 1.24× | **5.16×** |

- **R1 unrolled** — same dominant win as P1 (3.3×): kill loop overhead/spills.
- **R2 interleaved** — run the numerator and denominator Horner chains in lockstep; they're
  independent, so ILP hides latency (like dual-eval, but the two chains are num & den).
- **R3 parity** — odd numerator `P(x)=x·Pₙ(x²)`, even denominator `Q(x²)`; halve both Horners.
- **R4 deferred reciprocal** — the big rational-specific lesson. Because `v_if` is predicated,
  a reciprocal *inside* the cascade computes a full Newton-Raphson reciprocal **on all lanes
  for every segment**. Instead, select `P` and `Q` per segment, then do **one** reciprocal
  after the cascade (1.24× — and it grows with segment count).

```cpp
// R4: defer the expensive reciprocal out of the predicated cascade
vFloat selP = 0.0f, selQ = 1.0f;          // 1.0 default => never divide by zero
seg<0,...>(lut, x, t, selP, selQ);        // cascade selects P, Q
dst_reg[d] = selP * sfpu_reciprocal_iter<3>(selQ);   // ONE reciprocal
```

---

## 5. Results at a glance

| | naive | tuned | speedup |
|---|---:|---:|---:|
| **Polynomial** (16-seg, deg-8) | 85.1µs | 15.8µs | **5.4×** |
| **Rational** (4-seg, n8d8) | 45.2µs | 8.8µs | **5.2×** |

Two ~5× wins from the same playbook: **unroll first** (it dwarfs everything), then **fill the
pipeline** (dual / interleaved), then **cut work** (parity, adaptive, deferred reciprocal) —
while watching register pressure.

---

## 6. When does each optimization apply?

| optimization | applies when | watch out for |
|---|---|---|
| Compile-time unroll | always (degrees/segments known at compile time) | code size; but the win is huge |
| Dual / interleaved (ILP) | independent eval chains exist | uses more registers |
| Parity x²-Horner | function is odd or even | only valid for true parity; **don't stack on dual at high degree** (register ICE) |
| Adaptive degree | per-segment fits vary | needs per-segment degree metadata |
| Deferred reciprocal | rational, multi-segment | keep a safe denominator default (1.0) |

The meta-lesson: **measure before and after every change.** Parity halved the FMAs but barely
moved the clock (not FMA-bound); unroll changed nothing arithmetically but gave 3.8× (it was
overhead-bound). You cannot tell which is which without the device timer.

---

## 7. Reproduce

```bash
cd $TT_METAL_HOME/tt_metal/programming_examples/generic_lut_activation_embedded/tutorials/sfpu_optimization
./run_tutorial.sh all      # builds + profiles + correctness-checks every rung -> results.csv
```

- `gen_bench.py` — fixed-seed benchmark coefficients + exact reference.
- `kernels/compute/p*.cpp`, `r*.cpp` — the rungs (each differs only in the eval body).
- `run_tutorial.sh` — swaps each rung into the adhoc kernel slot, builds, profiles under
  Tracy (3 runs, min), checks output against the exact reference, records `results.csv`.
- `lib/score.py`, `lib/static_analysis.py` — correctness + FMA accounting.

Prerequisites are the example's standard build (see the parent `README.md` → "Setup & Build").
