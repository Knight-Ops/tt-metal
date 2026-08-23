<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
     SPDX-License-Identifier: Apache-2.0 -->
# MTP verify-cost optimisation log

Goal: close the gap between the **measured** traced speculative round and the Phase-A cost model, which
projected 1.7x (42.9 ms/round) but measures 1.06x (69.4 ms/round). Acceptance is not the problem — it
matched prediction exactly (2.53 tokens/round vs 2.522 predicted). The entire miss is verify COST.

All numbers: single P150, tt-metal v0.77.0, 40 layers, gamma=2 (K=3), `tests/bench_mtp.py --rounds 15`.
Baseline to beat: **34.5 tok/s / 29.01 ms** per traced decode step (`bench_decode.py`, A0).

## Ledger

| # | change | verify ms | round ms | tok/s | vs base | note |
|--:|---|--:|--:|--:|--:|---|
| 0 | all-eager rounds | — | 384.5 | 6.6 | 0.19x | host-dispatch bound |
| 1 | + verify trace | 58.9 | 106.3 | 23.8 | 0.69x | draft+commit still eager |
| 2 | + draft & commit traces | 58.9 | 69.4 | 36.5 | **1.06x** | host glue now only ~0.4 ms |
| 3 | hoist `a\|b` tile expansion out of the step loop | 64.11 -> 58.92 | 75.6 -> 69.5 | 36.4 | 1.06x | -5.19 ms; removed ~360 repeat/transpose/reshape ops |
| 4 | pair q\|k and v\|z into 2 zero-pads (was 4) | 58.92 -> 58.79 | 69.5 -> 69.4 | 36.5 | 1.06x | **-0.13 ms = nothing**; ~180 pad/slice ops are ~free |
| 5 | **chained ttl GDN verify kernel** (`decode_step_chain_tt`) | 58.79 -> 57.26 | 69.4 -> 68.0 | 37.3 | 1.08x | -1.53 ms traced (but **-68 ms eager**: 384.5 -> 316.3) |
| 6 | L1-resident verify intermediates (`mc = _MC` when verifying) | — | — | — | — | GDN block 1.0189 -> 1.0100 ms/layer = **-0.9%, noise**. Kept (harmless, correct regime) |
| 7 | conv "stacked window" form: 8 ops instead of 11 | — | — | — | — | **REVERTED: 38% SLOWER** (0.4752 -> 0.6552 ms/layer) |
| 8 | skip `new_conv_state` in verify (it is discarded) | 1.0100 -> 0.9357 /layer | — | — | — | -0.074 ms/layer = **-2.2 ms**; pure dead-code removal |
| 9 | DRAM-shard `w_out` in verify (config accepts any K<=32) | 0.9357 -> 0.8854 /layer | — | — | — | -0.050 ms/layer = **-1.5 ms**; PCC also improved 0.99972 -> 0.99987 |
| **F** | **all of the above, 40L measured** | **53.55** | **64.4** | **40.4** | **1.17x** | parity 39/39 both traced modes; 52 tests pass |

Cost model for reference: verify(3) = 38.36 ms, draft = 4.06 ms -> round 42.92 ms -> 1.70x.
Measured per-trace at #4: **verify 58.79**, draft 4.20, commit 6.02, host glue ~0.4.

## What #3 vs #4 established

Cost is NOT proportional to op count. Removing ~360 `repeat`/`transpose`/`reshape` ops bought 5.19 ms
(~14 us/op); removing ~180 `pad`/`slice` ops bought 0.13 ms (~0.7 us/op). So the expensive ops are a
specific class, and further guesswork is unjustified — hence profiling.

Also refuted along the way: the ~9 per-round `to_tt` host->device writes were NOT the bottleneck (host
glue is 0.4 ms of a 69.4 ms round). The `copy_host_to_device_tensor` + python-tracked-`cur` cleanup is
still correct, just not load-bearing.

## Measurement 1 — verify(K) sweep (`prof_mtp.py --sweep`, 40 layers, traced replay)

    K  verify ms   d/tok
    1     33.533
    2     50.766   17.233
    3     58.822    8.056
    4     66.483    7.661
    fit: verify(K) ~= 22.55 + 10.98*K   (cost model: 24.31 + 4.683*K)

This splits the problem in two, and shows the marginal is NOT linear:

1. **Fixed overhead.** verify(1) = 33.53 ms vs a real decode step at **29.01 ms** -> the verify path
   carries ~4.5 ms of structural overhead even at K=1, before any speculation. Candidates: the conv
   takes `_conv_silu`'s general T>1 branch instead of the T==1 stacked-taps fast path, the tile-row
   relayouts, the extra persistent conv-buffer copy, and `w_out` losing its M=1 DRAM-shard.
2. **Marginal.** A ~17 ms one-time jump at K=1->2, then a steady **~7.8 ms/token** (vs 4.68 modelled,
   1.67x). The K=1->2 jump is consistent with the conv leaving its T==1 fast path (11 extra ops/layer
   x 30 layers) plus the first genuinely multi-step relayout/launch work.

Budget to reach the modelled 38.36 ms: fixed 33.5 -> 29.0 (-4.5) and marginal 7.8 -> 4.7 (-6.2 at K=3).
Even reaching both gives verify ~40 ms -> round ~50.6 ms -> ~1.45x, so **1.7x also needs the 6.02 ms
commit trace to come down** (it is pure overhead the model never accounted for).

## Experiment 5 — the chained ttl GDN kernel (`QWEN36_GDN_VERIFY_CHAIN`, default on)

Folds the K-step recurrence onto the core: one ttl launch for the whole chain instead of K launches
driven from Python, deleting 6 of the 7 ttnn ops per step per layer (~360 ops at K=3, 30 layers).

Correctness: PCCs are **bit-identical** to the per-step-launch path (0.9996929231180836 /
0.9997223152888097 / 0.9997251851627934) and the full-traced token stream is unchanged — expected,
since the on-core math and its order are the same and only the state's SOURCE changes (a CB instead of
DRAM). `QWEN36_GDN_VERIFY_CHAIN=0` reverts.

Three ttl constraints had to be worked around, in order:
1. `for _ in range(K-1)` + a peeled final step (to avoid a conditional) needs **37 DFB indices vs a
   hardware limit of 32** — duplicating the loop body doubles the compiler's temporaries (10 vs 2).
2. Storing the carry back into `Sd` fails with *"dataflow buffer cb_index=8 has multiple producer
   threads"* — `read()` already produces `Sd`, and a DFB may have only one producer THREAD.
3. A dedicated carry buffer needs 33 DFBs — one over. **Solution: reuse `outer`**, which is free the
   moment `st = Sg + outer` is formed and whose producers are all in `compute()`. The carry is stored
   unconditionally and the last step's block is simply left unconsumed (the CB never holds more than
   one block, so it cannot deadlock against `block_count=2`).

**Result: -1.53 ms traced, far short of the ~5.3 ms predicted.** The prediction assumed ~14.8 us/op
(the rate implied by experiment #3); the ops this actually removes are `slice`s at ~4 us. Combined with
#4, the per-op cost picture in a traced graph is roughly:

    repeat / transpose / concat  ~30+ us      (data-movement, expensive)
    slice / pad / reshape        ~1-4 us      (nearly free)

Note the eager saving is huge (-68 ms) because eager pays ~190 us of HOST dispatch per op. That makes
the chained kernel clearly worth keeping regardless, but it is not the traced lever.

**Conclusion: op-count reasoning has now failed twice in a row at predicting traced savings.** Stop
extrapolating and measure per component (next section).

## Measurement 2 — per-component verify cost (`prof_mtp.py --components`, 40L, K=3)

Each sub-block timed in its OWN trace x its per-token layer count, the same method
`bench_decode.bench_components` uses for decode, so the two are directly comparable.

               component    ms/op   xN  total ms
        gdn decode (T=1)   0.3204   30      9.61
        gdn verify (K=3)   1.0189   30     30.57
       attn decode (T=1)   0.2955   10      2.96
       attn verify (K=3)   0.5942   10      5.94
        moe decode (T=1)   0.3272   40     13.09
        moe verify (K=3)   0.4681   40     18.72
      lm_head+argmax (K)   1.3096    1      1.31
      sum of verify totals              56.54   (vs 57.26 measured trace — accounts for it)

    verify-vs-decode EXCESS:  gdn +20.95 ms   moe +5.63 ms   attn +2.99 ms

**GDN is 74% of the excess, and the MoE is fine** (0.468 measured vs 0.489 modelled — the indices-gather
verify path works exactly as designed). Attention is 2x the model but only 3 ms absolute.

### The damning number
**GDN verify (1.0189 ms/layer) is WORSE than three independent decode steps (3 x 0.3204 = 0.961).**
The verify path gets *negative* amortization on the one component that should benefit most. Cause: it
silently drops every optimisation the decode path has —
- **L1 residency**: `forward` sets `mc = _MC` only when `T == 1`; verify runs with T=K so all its
  intermediates live in DRAM. FUTURE_OPTIMIZATIONS measured L1 residency as a real decode win precisely
  because that path is "many tiny dispatch-bound ops" — which describes verify too.
- **conv add-chain**: verify takes `_conv_silu`'s general branch (with a `concat`, a data-movement op in
  the expensive class) instead of the add-chain that replaced it as "the largest gated-delta decode cost".
- **DRAM-sharded `w_out`**: an M=1-only fast path, so verify uses the plain interleaved matmul.

That reframes the work: the target is not "make verify cleverer", it is "stop verify from losing the
decode path's existing wins".

## Is 1.7x actually reachable? — arithmetic says NO, and the model shows why

Round = verify 57.26 + draft 4.20 + commit 6.18 + host glue 0.4 = 68.04 ms, 2.533 tokens/round.

1. **Verify alone caps it.** 57.26 ms / 2.533 tokens = 22.6 ms/token = **1.28x ceiling even if draft and
   commit were free.** So no amount of work on draft/commit can reach 1.7x.
2. **The original 1.70x projection omitted the commit pass entirely.** It was
   round = verify 38.36 + draft 4.06 + glue 0.5 = 42.92 ms. There is no commit term — the Phase-A model
   was built before the rollback existed, and rolling the recurrent + conv state forward by the accepted
   count costs a real 6.18 ms at 40 layers.
3. **Plugging the model's OWN verify number back in gives 1.50x, not 1.70x:**
   38.36 + 4.20 + 6.18 + 0.4 = 49.1 ms -> 19.4 ms/token -> 1.50x.
   Reaching 1.7x would need verify <= ~32 ms — i.e. BELOW a cost model that was itself assembled from
   isolated best-case per-component measurements. Not plausible.

**So the honest ceiling for this design is ~1.5x, and 1.7x was an arithmetic artifact of a model that
forgot a whole pass.** Revised target: verify 57.3 -> ~40 and commit 6.2 -> ~2, which lands ~1.45-1.50x
(50-52 tok/s). Recorded here rather than quietly re-aiming.

## Measurement 3 — inside the GDN verify block (`prof_mtp.py --gdnbreak`, per layer, K=3)

           in_proj (M=K)   0.0845 ms
         conv_silu (T=K)   0.4752 ms   <-- 56% of the block
             xpad concat     0.1234
             4 slice+mul+add 0.2821
             silu            0.0127
             new_state slice 0.0868   <-- ONE slice; verify never uses the result
         4x _to_tilerows   0.0630 ms
            2x ba repeat   0.1104 ms
          _from_tilerows   0.0195 ms
             w_out (M=K)   0.0948 ms
               STAGE SUM   0.8475 ms   (block 1.01 -> chained ttl kernel ~= 0.16, matching probe A2)

Two things this settles:
- **`in_proj` — a real [K,2048]x[2048,12352] matmul with a 25 MB weight read — costs 0.0845 ms, while
  the conv's tiny elementwise ops cost 0.4752.** At these sizes ops are almost pure launch overhead.
- **A single `[conv_k-1, conv_dim]` sub-tile row slice costs 87 us.** `conv_dim = 8192` is 256
  tile-columns, so slicing 3 of 32 rows is a strided gather across all of them. That is why the conv
  dominates, and it is why experiment #7 (which added a `concat` and a `sum` reduction) went backwards.

## What did NOT work, and the rule it establishes

Op count is a bad predictor of traced cost on this hardware. Three predictions failed:

| prediction | expected | actual |
|---|--:|--:|
| #4 pair 4 pads into 2 (~180 ops) | few ms | -0.13 ms |
| #5 chained kernel (~360 ops) | -5.3 ms | -1.53 ms |
| #7 conv window form (11 ops -> 8) | faster | **38% slower** |

Measured per-op classes: `concat` / `repeat` / `sum`-reduction and **sub-tile row slices** are expensive
(~30-90 us); tile-aligned `slice` / `pad` / `reshape` are nearly free (~1-4 us). The wins that landed
(#8, #9) were both "stop doing work" or "reuse an existing fast path", not "use fewer ops".

## Remaining levers, with measured expected value

| lever | worth | why not done |
|---|--:|---|
| conv_silu (0.39 ms/layer after #8) | ~11.7 ms | Near its floor: 11 ops over 256 tile-columns each. Folding it INTO the ttl kernel is the real fix and is blocked by the DFB budget — the chained kernel is already at 32/32. FUTURE_OPTIMIZATIONS Lever 3b documents the same wall for decode. |
| commit trace 6.18 ms | ~4 ms | 30 layers x ~6 tiny ops. The structural fix is one contiguous state buffer across layers instead of 30 separate ones — a broad change to shipping cache layout, so out of scope for "don't break the system". |
| attention verify excess | ~2 ms | Needs the batch-K KV cache (verified PCC 1.000000) at K x the KV memory. |
| 2x `ba` repeat 0.11 ms/layer | ~1.5 ms | One repeat cannot serve both halves: the kernel wants (t*V+h) tile-rows and a single flattened repeat gives (t*2V+j). |

Optimistically all of those together are ~-19 ms -> round ~45 ms -> ~1.63 ms/token... i.e. still short of
1.7x, and they include one item that is architecturally blocked and one that requires a cache-layout
refactor.

## Method notes
- Per-op DEVICE kernel time is identical between eager and traced replay (same kernels), so Tracy
  profiling runs on the EAGER path; host signposts do not fire during `execute_trace` (see
  `tt/signpost.py`). Profile at 4 layers (the device profiler is unreliable at 40) and read the result
  as PROPORTIONS; absolutes come from `bench_mtp.py` at 40 layers.
- Every change is gated on `test_gated_delta_verify.py` / `test_moe_verify.py` /
  `test_attention_verify.py` / `test_mtp_trace.py` staying green, and on the bench's own
  eager-vs-traced token-stream parity (38/38).

---

## The T->0 sampling tests are UNSOUND — measured, and it is the test, not the sampler (2026-08-22)

`test_spec_sampling_at_zero_temperature_equals_greedy_spec` and
`test_spec_sampling_oracle_draft_accepts_everything` have been failing intermittently all through the
2026-08-22 perf work, and their count crept 2 -> 3 -> 5 as unrelated low-bit changes landed. Before
touching either, the question worth answering is whether the SAMPLER is wrong. Measured directly
(`tests/probe_sampling_at_zero_temp.py`, new): one hidden state, both selection tails, same step, no
trajectory — which is the invariant those tests are really about, and the one thing a trajectory
comparison cannot isolate.

| temperature | argmax == sampled | disagreements | top1-top2 gap there | RANK of sampled token |
|---|--:|--:|---|---|
| 1e-4 | 62/64 | 2 | 0.0078 | **2-3** |
| 1e-2 | 62/64 | 2 | 0.0078-0.0312 | **2** |
| 1e-1 | 28/64 | 36 | 0.0000-0.3125 | 2-32 |

**The sampler is fine.** At T->0 it disagrees with `argmax` on ~3% of steps, and when it does it picks
the immediate runner-up (rank 2-3) at a gap of one bf16 ULP at these magnitudes (0.0078). It never
selects a token the target rates as unlikely. At temperature 0.1 it spreads across ranks 2-32, i.e. it
behaves like a sampler. The mechanism is that the candidate values reach `ttnn.sampling` through
`ttnn.topk(..., sorted=False)` + a 32-lane `ttnn.repeat` in bf16, so two logits within ~1 ULP are
indistinguishable to it, and the tie is then broken by a different rule than `argmax`'s first-index.

**The test's stated premise is therefore false.** Its docstring claims *"unlike a PCC gate it is not
tie-sensitive: both paths consume the same verify logits"*. They do consume the same logits — and still
disagree on ~3% of steps, BEFORE any trajectory compounding. Over the 24 tokens it compares,
P(no divergence) = 0.97^24 ~= 48%. **The test is close to a coin flip by construction**, independent of
any change under test.

That is why it has been so hard to read all day: it fails for changes that are provably neutral, it
passes for others by luck, and its failure carries no information about the change. It also actively
misled the triage — two of its three assertions ARE meaningful (the accept path, the bonus token) and
kept passing while the trajectory assertion churned.

### APPLIED — `test_mtp_spec_decode.py` now gates the per-step invariant instead

Three changes; the file went from **3 failed / 13 passed to 17 passed**, with a STRONGER assertion than
it had before:

1. **New `test_sampling_tail_is_near_argmax_at_zero_temperature`** — the invariant, asserted directly:
   one hidden state, both selection tails, same step, no trajectory. `argmax == sampled` on >= 85% of
   steps, and **every disagreement must be rank <= 3**. Measured on the fix: 23/24 agreement, the one
   disagreement at rank 2, reproducing the probe exactly.
   **Negative control (this is what makes it a gate and not a decoration):** the same code path at
   temperature 0.1 gives 28/64 agreement with ranks 2-32 -- it fails both bounds decisively. So a
   sampler with a mis-set temperature, a wrong top-k, or a bad candidate gather is caught; a 1-ULP
   tie-break is not.
2. **The two trajectory assertions are gone**, each replaced by an informational print plus a comment
   recording the measurement and why no prefix bound can substitute (at 3%/step, agreeing for even the
   first 4 tokens is only 88% likely). The three accept-path assertions and the accept-probability
   identity are untouched -- those are the real content and were never what failed.
3. **The T->0 accept-probability tolerance was recalibrated**, and this one was a latent bug of the same
   kind rather than collateral. The assertion demanded `|acc_p_sum - acc_p_n| <= 1% of acc_p_n` while
   its own comment explained that an m-way tie legitimately yields p = 1/m. Measured failure:
   `acc_p_sum` 8.5 over 9 accepts -- eight clean 1.0s and one two-way tie. Now bounds the MEAN at
   >= 0.9, which a broken acceptance rule (mean ~0.1-0.3 here) still fails by an order of magnitude.

Standing rule from this: **an exact-equality assertion over an accumulated trajectory is not a
regression gate on this model.** Decode sits at the bf16-determinism edge, so per-step invariants are
the only stable form. `EXPERIMENTS.md` now has five recorded cases of isolated/accumulated measurements
misleading; this is the first where the MEASUREMENT INSTRUMENT itself was the thing at fault.

## The router rewrite breaks `test_spec_sampling_oracle_draft_accepts_everything` (2026-08-22, OPEN)

`QWEN36_MOE_ROUTER_FAST` (see FUTURE_OPTIMIZATIONS Lever 5 — decode 35.0 -> 37.4 tok/s/user) fails 2 of
the 3 parametrisations of this test. Recorded here rather than fixed, because which side is wrong is a
judgement the MMLU number should make, not me.

**What fails.** Not the accept path itself — the three assertions the test's own docstring calls "the
real subject" all pass at every gamma:

    max(g_sizes) == gamma + 1      full acceptance is reached          PASS
    g_sizes[1] == gamma + 1        the first oracle round fully accepts PASS
    s_st["bonus"] > 0              the bonus path is exercised          PASS
    g_sizes == s_sizes             greedy == T->0-sampled, EXACTLY      FAIL at gamma 1 and 2

    gamma=1  greedy [1,2,2,2,2,2,2,2,2,2,2,2]     sampled [1,2,2,2,2,2,2,2,2,1,1,1,1,1,1]
    gamma=2  greedy [1,3,3,3,3,1,1,...]           sampled [1,3,3,1,1,1,...]
    gamma=3  identical — PASSES

So greedy and T->0 sampling agree for 9 rounds (gamma=1) and 3 rounds (gamma=2), then diverge. Bisected:
`QWEN36_WIDE1D_DECODE=0` still fails; `+ QWEN36_MOE_ROUTER_FAST=0` passes. It is the router rewrite.

**Why.** The 4th assertion demands exact parity between two *different* selection implementations —
`argmax` and the chunked-topk + `ttnn.sampling` tail at T->0 — which only holds while no near-tie occurs
in the window. The router rewrite is not numerically neutral (`softmax(topk(logits))` vs
`renorm(topk(softmax(logits)))` differ in the low bits of the routing weights by <=0.005; expert
SELECTION is bit-identical, measured), and that is enough to move a tie. This is the same regime the
test's own docstring documents: *"a single near-tie can flip an argmax and derail the chain ... a conv
change that IMPROVED PCC 0.9997 -> 0.99998 was enough to move such a tie"*. An accuracy-**improving**
change has already broken this assertion once.

**Note the test runs at `QWEN36_LAYERS=4`.** It exists in that form precisely because the real MTP head
accepts nothing below 40 layers, so the oracle drafter is a reduced-layer scaffold. Whether the
divergence survives at 40 layers is not established.

**Full flag matrix** (the flag was since split three ways; `CONST_SPARSITY` is the `scatter_routing`
deletion, `ROUTER_FAST` the softmax/topk reordering, `WIDE1D_DECODE` the tuned program configs):

| WIDE1D | ROUTER_FAST | CONST_SPARSITY | oracle test |
|--:|--:|--:|---|
| 0 | 0 | 0 | **3 passed** (pristine) |
| 0 | 0 | 1 | **3 passed** |
| 1 | 0 | 1 | 2 failed |
| 0 | 1 | 1 | 2 failed |
| 0 | 1 | 0 | 2 failed |
| 1 | 1 | 1 | 2 failed |

Two things this settles:
- **`CONST_SPARSITY` really is bit-identical.** It is the only one of the three that changes nothing
  here, which is the strongest available confirmation that `sparse_matmul` indexed mode does not read
  the `sparsity` operand. Keep it unconditionally.
- **`ROUTER_FAST` and `WIDE1D_DECODE` each break it on their own.** Not an interaction — two independent
  low-bit perturbations, either of which is enough to move the tie. So this test will fail for *any*
  future change in this class, which is itself the useful finding.

**MMLU-Redux, 200 samples, 40 layers** — the baseline `evaluation/published_scores.json` never had, now
recorded there:

| config | accuracy | avg TTFT |
|---|--:|--:|
| all three flags OFF (pristine) | **79.0%** | 1003 ms |
| shipping defaults (all three on) | **78.5%** | 980 ms |

One question out of 200. At n=200 the standard error is ~2.9 points, so this is **no detectable change**
— not a demonstrated absence of one. It is the same magnitude the BFP8 MoE in0 change was accepted on
("79.0% bf16 vs 78.0% ... noise at that sample count", PREFILL.md §3.8).

## What the 2026-08-22 decode work actually costs in accuracy

Two instruments, because neither alone covers the change. **MMLU-Redux is scored on PREFILL last-token
logits, and most of these optimisations are gated on `T == 1`** — so an MMLU run never executes
`se_gate_up`/`se_down`/`wk`/`w_in_proj`/`const_sparsity` at all. It covers only `ROUTER_FAST` (which
applies at every T) and the `lm_head` config (<=32 rows, so the last-token slice qualifies).

### (a) MMLU-Redux, PAIRED, n=200, 40 layers

| | baseline (all flags off) | defaults (all on) |
|---|--:|--:|
| accuracy | 79.0% | 78.5% |
| avg TTFT | 1003 ms | 980 ms |

Per-question pairing is what makes this readable, and the aggregate hides it:

    identical predicted letter        197/200  (98.5%)
    2x2   both right 156   base-only right 2   new-only right 1   both wrong 41
    discordant pairs 3  ->  McNemar exact two-sided p = 1.000
    the 3 flips: high_school_chemistry, college_computer_science, electrical_engineering

So the −0.5 points is a net −1 question out of 3 that moved, **2 against and 1 for**. At n=200 the
standard error is ~2.9 points; the paired test says p = 1.000. This is a coin toss on near-ties, not a
regression — but note it is also NOT a demonstration of no effect. It bounds the effect below what 200
samples can see.

### (b) Teacher-forced decode logit drift — the instrument for the `T == 1` half
`tests/probe_teacher_forced_drift.py` (new; this is the harness `FUTURE_OPTIMIZATIONS` has been asking
for and that had only ever been run ad hoc). One model build, both arms, arm B replays arm A's token
stream so every step sees identical inputs. 40 layers, 96 steps:

    logit PCC   mean 0.997283   median 0.997894   10th pct 0.996753   min 0.947239
    median 1st half 0.997775 -> 2nd half 0.997906   (delta +1.3e-04, i.e. slightly UP)
    OLS slope -3.52e-05/step, but -3.38e-06/step excluding the single worst step
    VERDICT: STABLE -- fixed rounding offset plus outliers, no accumulation
    argmax agreement 92/96    mean top-5 overlap 4.70/5
    4 flips, at baseline top1-top2 margins of 0.0000 / 0.1250 / 0.2500 / 0.1250
       against an all-step mean margin of 3.2962  -> every flip is on a near-tie

**Context: this is BETTER than what the codebase already ships on.** The conv add-chain
(`QWEN36_CONV_ADDCHAIN`, default on, +9.8% decode) was accepted on "teacher-forced decode logit-PCC vs
baseline (40L, 96 steps) = mean 0.996, stable/no accumulation". This work measures **0.9973, stable**.

**A methodological trap worth keeping.** The first version of this probe judged drift by comparing two
half-MEANS, and reported "DRIFTING". It was wrong: one step (#94, PCC 0.947) supplied ~90% of the
−3.5e-05 slope, and the median trend is *positive*. A single deep outlier late in a ~100-step series is
enough to manufacture a drift signal. The probe now takes its verdict from median half-trend plus a
slope computed with the worst step excluded, and prints both so the reader can see the outlier.

### Verdict on the accuracy question
The cost is **rounding at the bf16-determinism edge, not a quality change**: non-accumulating over 96
steps, better than the shipped conv add-chain's own gate, indistinguishable on paired MMLU (p=1.000),
and every token divergence sits on a near-tie the baseline was going to lose or win by <=0.25 logits.
What it does change is *which* of two near-tied tokens wins — which is why the tie-sensitive
`test_spec_sampling_oracle_draft_accepts_everything` assertion fails, and why that assertion is
measuring determinism rather than quality.

**Where that leaves the decision — deliberately NOT made here.** The evidence says the perf wins cost
nothing measurable and the failing assertion is an exact-parity check on a quantity the codebase
already documents as tie-fragile. But "relax the test that my own change broke" is not a call to make
unilaterally, and the alternative is cheap and real. The options, with prices:

1. **Relax the 4th assertion** to match the intent of the first three (that the accept path is
   exercised), e.g. require agreement over the first N rounds rather than the whole generation. Keeps
   +12.3%. Risk: weakens a real invariant (T->0 sampling *should* equal argmax) — better addressed by
   testing that per-step, on identical state, instead of over an accumulated trajectory.
2. **Revert `ROUTER_FAST` only**, keep `CONST_SPARSITY` + `WIDE1D`. Costs ~1.0 ms/token of the 3.2 —
   but does NOT fix the test, because `WIDE1D` breaks it independently.
3. **Revert both `ROUTER_FAST` and `WIDE1D`,** keep only `CONST_SPARSITY`. Test goes green; keeps
   ~0.9 ms of the 3.2 ms won today (39.2 -> ~37.9 tok/s).
4. Establish whether the divergence is a 4-layer artifact (the test only exists in that form because
   the real MTP head accepts nothing below 40 layers). Not investigated.

**Process note, worth more than the finding.** The router rewrite was reported green on `test_moe*`,
`test_gated_delta*`, `test_trace` and `test_model` — I did not run the MTP suite, and the MTP suite is
where a low-bit change shows up, because it is the only part of the tree that asserts exact agreement
between two selection paths. **Run the whole suite before claiming a change is clean**, not the subset
that looks related: `pytest models/demos/qwen3_6_a3b/tests/ -q` is 7 minutes.

---

# Speculative SAMPLING (2026-08-20)

Greedy speculative decode could only ever serve `temperature == 0`, while `demo/server.py` samples by
default (T 0.6 / top_k 20 / top_p 0.95 / presence 1.5). This adds speculative sampling — acceptance by
rejection sampling instead of exact match — plus the serving integration MTP never had
(`grep -rn mtp demo/` was empty before this: the 1.17x lived only in `bench_mtp.py`).

## The rule, and why the draft graph did not change

The draft is an `argmax`, so q is a point mass at x̂ and `min(1, p(x̂)/q(x̂))` collapses to `p(x̂)`, with
residual = p minus x̂ renormalised. That is exactly p-distributed, so **no draft-side change is needed**
— only the target distribution per verify row (`_verify_candidate_tail`, a per-row lift of
`_select_token_batch`'s validated chunked topk). Truncation and the presence penalty are applied on the
HOST from 256 candidates, which is exact for `top_k <= 32` (proof in `tt/mtp_sampling.py`, checked
against a full-vocab reference in `tests/test_mtp_sampling_rule.py`).

## Measured, 40 layers, gamma=2, fully traced (`bench_mtp.py`)

Baselines re-measured on this build (`bench_decode.py --iters 150`): greedy **28.65** ms/token,
sampled (full thinking-mode config) **29.67** ms/token.

| # | config | tok/round | ms/round | ms/token | tok/s | vs its baseline |
|--:|---|--:|--:|--:|--:|--:|
| 1 | greedy (rule = exact match) | 2.700 | 63.2 | 23.39 | **42.8** | **1.22x** |
| 2 | sampled, exact, presence 1.5 | 2.300 | 65.9 | 28.64 | 34.9 | **1.04x** |
| 3 | sampled, exact, presence 0.0 | 2.500 | 66.5 | 26.61 | 37.6 | ~1.09x (vs ~29.1) |
| 4 | sampled, **relaxed**, presence 1.5 | 2.750 | 66.7 | 24.27 | **41.2** | **1.22x** |

**Rows 2-4 are 20-round windows and are NOT reliable — see the rule sweep below, which supersedes them
(exact 1.00x, relaxed 1.11x).** They are kept because they are how the wrong number got quoted.

Per-trace replay: verify **52.48** ms greedy / **54.20** ms sampled (the candidate tail costs
**+1.72 ms at K=3, i.e. 0.57 ms/row**), draft 4.18, commit 6.18. Eager-vs-traced token-stream parity
holds in every mode (54/54, 46/46, 55/55) — the host RNG is seeded, so sampled streams are comparable.
Row 3 is from the pre-redesign build (two D2H readbacks instead of one, worth ~0.9 ms/round), so it is
slightly pessimistic; rows 1, 2 and 4 are all on the final build. Round-to-round noise between runs of
the same config is ~1% (rows 2 and 4 execute the same graph and differ by 0.8 ms).

## The result that matters: EXACT sampling is a wash, and the host model over-predicted it

The cost side was fine (+1.72 ms of tail on a 63 ms round). The whole miss is **acceptance: 2.70 ->
2.30 tokens/round**, i.e. conditional acceptance 0.85 -> 0.65.

The host-side probe (`probe_mtp_acceptance.py score`, 288 positions against the real `mtp.*` weights)
predicted this would cost almost nothing:

               T/k/p/pen   greedy  exact(amax)  exact(samp)  relaxed  E[p_max]  H(p)  |supp|
        0.60/20/0.95/1.5    0.896        0.858        0.880    0.934     0.936  0.15     1.4
        0.60/20/0.95/0.0    0.896        0.882        0.895    0.958     0.938  0.14     1.4

i.e. exact should have lost only ~0.02-0.04 of acceptance, because at T=0.6 with top_p=0.95 the
truncated target is nearly a point mass (**mean |support| 1.4, H = 0.15, E[p_max] = 0.94**). Device
measured a **0.20** loss — 5-10x the prediction.

Two things are now measured about that gap:
1. **The presence penalty is real but secondary.** Device: 2.50 -> 2.30 tok/round when presence goes
   0 -> 1.5 (conditional 0.75 -> 0.65). The host model saw only -0.024, because it penalises against a
   greedy trajectory's token history rather than the sampled run's.
2. **The rest is off-policy bias, and it is the host model's structural limit.** The probe scores
   positions on the CAPTURED GREEDY trajectory; sampled decode visits states reached by *sampling*,
   which are systematically flatter (lower `p_max`) and on which the one-token-ahead draft head is
   less accurate. No amount of extra host sampling fixes this — it needs on-policy capture.

**So the host acceptance probe is a good instrument for exploring the parameter space and a poor one
for predicting on-policy sampled throughput.** That is the 4th failed prediction recorded in this
file, after #4, #5 and #7 — and unlike those it was not an op-count argument, so the lesson generalises:
predict cost from cost measurements, but never predict ACCEPTANCE off-policy.

## Relaxed acceptance recovers all of it — at a cost that is not measured here

Typical acceptance (`QWEN36_MTP_ACCEPT=relaxed`: accept when `p(x̂) >= min(eps, delta*e^-H(p))`,
eps 0.09 / delta 0.3) gives **2.750 tok/round and 1.23x** — the full greedy-MTP speedup, for sampled
requests, with acceptance 0.88 (*higher* than greedy's 0.85, since a merely-plausible draft is
accepted). At this config `H ~ 0.15` so the threshold is `eps = 0.09`: any draft with >= 9% target
probability is taken.

That is **not** distribution-preserving. **Superseded: a 150-round paired sweep puts relaxed at 1.11x,
not 1.23x** — see "Ranking the acceptance rules" below, which also quantifies the quality cost
directly instead of deferring it to a benchmark that could not have measured it anyway.

## A device-level trap found while integrating: only ONE MTP trace may be resident at capture

The server hung on startup with every thread in `futex_wait`. Root cause, bisected with a 4-layer
repro (`greedy capture -> sampling capture`, vs the same with the traces released in between, vs no
greedy capture at all):

**Capturing the verify trace while any other MTP trace is still resident yields a silently corrupt
trace.** Its candidate rows read back as ~1e9 garbage (values were fine — only the rows written last),
the acceptance rule "sampled" an out-of-vocab id from them, and passing that id to `ttnn.embedding`
hung the device. So the visible symptom (a hung server) was several steps removed from the cause, and
nothing raised.

    greedy capture, then sampling capture, traces kept resident : FAIL (idx range [-1.1e9, 1.1e9])
    same, with release_mtp_traces() in between                  : PASS
    no greedy capture at all (sampling only)                    : PASS
    capturing draft + commit AFTER verify                       : PASS  (order matters)

This is the same failure family as `forward_prefill_traced`'s documented multi-bucket wedge ("the
pinned per-trace intermediate footprint corrupts the larger trace") and the reason
`capture_decode_trace` has always released the prior trace before capturing. Worth stating as a rule:
**capture a large trace only from a clean slate.**

Fixes: `setup_mtp_traces` releases every MTP trace first, there is exactly ONE verify trace with its
tail pinned in `mtp_verify_tail`, and the tail is a PROCESS-level choice (`QWEN36_MTP=1` captures the
sampling tail and greedy requests replay it; `greedy_only` captures the greedy tail and sampled
requests use plain decode). Regression-gated by
`test_mtp_trace.py::test_mtp_candidates_valid_after_a_greedy_capture`, whose assertion is deliberately
just "every candidate index is a real vocab id" — that is exactly what the corruption violated.

Two smaller things fell out of the same debugging:
- The candidate buffer is now ONE float32 `[1,1,2K,W]` tensor (K value rows then K index rows) instead
  of separate float32/int32 tensors. Indices are < 2^24 so float32 holds them exactly, and it halves
  the readback to a single D2H. (The int32 variant was *also* correct in isolation — verified
  model-free for `topk`/`typecast`/`add`/`concat`/`copy`, eager and traced — so the dtype was never the
  bug; this is simply the better shape.)
- Acceptance is **prompt-dependent, and MTP can lose**. Break-even at gamma=2 is round/baseline =
  63.2/28.65 = **2.21 tokens/round**. Measured tokens/round: **2.70** on the bench's chat prompt
  (1.22x), **2.39** on another chat prompt through the server (1.11x), **2.04** on `demo.py`'s short
  raw-text prompt over its first 48 tokens (0.93x — the `<think>` boundary and cold start dominate a
  short run). Quote the steady-state number, never promise it for every prompt.

## Negative result: priming the MTP head's KV cache over the prompt costs 16%

The draft head is one full-attention layer with a PRIVATE KV cache (`mtp_kv`) that nothing ever wrote:
the backbone's prefill fills the backbone's caches. So every draft attended over zeroed rows, and
`TtMtpHead.forward_prefill` -- whose own docstring says it exists "to PRIME the head's own KV cache
over the prompt" -- was never called by any serving path. Priming it looked like free acceptance,
especially since the ~2.5 tok/round the 1.17x rests on was measured WITH context
(`probe_mtp_acceptance._chain_alpha` builds a 96-token teacher-forced prefix for exactly that reason,
and its `--warmup` flag exists to skip the cold-cache positions).

Built it (`TtModel.prime_mtp`, `QWEN36_MTP_PRIME`) and measured it. It is a **16% LOSS**:

| head cache | tok/round | ms/token | vs traced base |
|---|---|---|---|
| cold (default) | **2.608 ± 0.044** | **19.60** | 1.46x |
| primed (992 rows) | 2.252 ± 0.049 | 23.41 | 1.22x |

40L, gamma=2, greedy, 1024-token prompt, 250 rounds/arm, ONE model + ONE captured trace set shared by
both arms (`bench_mtp.py --prime-ab`). Three runs -- 40 and 250 rounds, 40 and 992 primed rows, both
orderings -- all put priming behind. The deficit did **not** grow with prompt length, which is the
opposite of what "the head needs more context" predicts.

The measurement needed an ABBA schedule to be trustworthy, and the control paid for itself in an
unexpected direction. Two straight A-then-B runs gave -0.125 and -0.112 tok/round; ABBA
(cold, primed, primed, cold) gave **-0.356 ± 0.066**, because the LAST block, cold, was the best of the
four (2.792) -- run-order drift had been *masking* most of the effect, not manufacturing it. Caveat on
that sigma: it pools rounds as if independent, and the two cold blocks (2.424 / 2.792) spread as widely
as the effect itself, so the trustworthy signal is the SIGN agreeing across every run, not the
magnitude of any one.

**Why it loses, and why the obvious reasoning had it backwards.** The tempting argument -- the one this
change was built on -- is that zeroed rows steal softmax mass from the head's one genuine signal, the
`(h_i, t_{i+1})` pair arriving through `fc`. The dilution is real; the *consequence* is inverted. Every
K/V row below `pos` is EXACTLY zero, so every score is exactly zero and the softmax spreads
near-uniformly over ~pos zero-valued V rows: the attention output is attenuated toward zero, roughly as
1/pos. **A cold head silently collapses to `fc(h_i, t_{i+1}) + MoE` with its attention branch switched
off -- and that drafts better.** Which makes sense once stated: `h_i` is the post-final-norm backbone
hidden, so it already carries the whole prompt through 40 layers. The head's single attention layer has
little left to add, and evidently adds noise. Priming switches the branch back on.

Two things worth keeping from this:
- the cold head is not a position-invariant baseline either -- its attenuation scales with `pos`, so it
  behaves differently at pos=10 than at pos=1000. Any future acceptance work should account for that.
- the fencepost here is genuinely subtle and is now pinned by tests. `_m_pos_*` is shared with verify
  (which needs true absolute backbone positions), so the draft chain stores the pair for contract row
  `pos-1+j` at head position `pos+j` -- a +1 offset that is INVISIBLE while the cache is cold, because
  there is nothing for the rows to be relative to. `tests/test_mtp_prime.py` asserts primed and drafted
  rows form one contiguous prefix; removing the filler row makes it report exactly the predicted hole
  (`zero row(s) inside the written prefix 0..28: [24]`, T=24).

Kept as a default-off knob rather than deleted: it is the reproduction harness for the finding and the
answer to an obvious "did you try priming the head?". Fourth time on this project a plausible mechanism
has lost to measurement, after conv-fold, norm sharding and the "fast fused chunk_state kernel".

## Serving integration: measure the break-even, but do NOT act on it by default

Because acceptance swings across that 2.0-2.7 range and break-even is 2.21, `QWEN36_MTP=1` on its own
can sometimes make the server SLOWER. So `demo/server.py` measures both rates live:

- warmup calibrates the plain traced decode step on THIS board — measured **28.51-28.64 ms/token**,
  reproducing `bench_decode.py`'s 28.65 to 0.1%, which is a nice independent check of both harnesses;
- every `QWEN36_MTP_CHECK_ROUNDS` (24) rounds `_decode_mtp` logs its own ms/token against that
  reference, as a ratio.

**Acting on it is opt-in, and that is a correction of the original design.** The first version
disengaged automatically whenever a window came in behind, and it misfired immediately:

```
[mtp] disabling speculative decode: 34.8 ms/token vs plain 34.6 (break-even 1.89 tok/round, got 1.88)
```

That is a **0.6% difference** — comfortably inside the noise of a 24-round window — and because
disengaging is sticky for the process, it landed during a benchmark client's own warm-up request and
served every subsequent *timed* request on the plain path. The measurement was right and the decision
was wrong: a rule that flips a process-lifetime switch on a coin toss is worse than no rule, because
its failure is silent and looks like a legitimate result.

So `_mtp_losing_streak` now returns 0 unless `QWEN36_MTP_AUTODISABLE` is set to the loss ratio
required to act (e.g. `1.15` = only when speculation is >=15% slower), sustained for
`QWEN36_MTP_AUTODISABLE_WINDOWS` (2) consecutive windows. Default off: speculation stays engaged and
the operator reads the logged ratio. When it does trip, the handover is unchanged — `sync_mtp_state()`,
release the MTP traces, capture a plain decode trace, finish on the plain path — and still sticky,
since re-engaging costs an ~8 s re-capture per request.

Generalisable: a guard whose threshold sits inside its own measurement error is not a guard. Either
widen the margin past the noise or demote it to an observation.

That disengage path is the only consumer of the `sync_conv_rows` fix below, and the reason it matters:
speculative rounds advance `cache["conv_state"]` while plain decode reads the `conv_rows` add-chain
buffers, so handing over without re-seeding them would silently drop every token the rounds emitted.

### Two measurement traps the guard walked into first
Both produced a **wrong decision**, not just a wrong number — with either in place the guard
disengaged a configuration that measures **1.11x** once they are removed — so they are worth recording:
1. **The eager seed round.** Round 1 of a request is the K=1 seed step (no draft, no trace shaped for
   one row) and costs hundreds of ms. Averaged into a 24-round window it added ~13 ms/round — enough
   to disengage a winning configuration.
2. **Post-capture restore work.** Timing the plain path from step 2 gave **43.1 ms/token** where the
   settled rate is ~29: `capture_decode_trace`'s snapshot/restore (~30 layers of clone+copy) is
   enqueued behind step 1, so the next few replays queue behind it. The tokenizer was the obvious
   suspect and was measured innocent (0.008-0.046 ms per decode, 24-200 ids).

Both are now excluded by a 4-step/round warm-up window (`_RATE_WARM`) plus a long enough window (the
warm generate is 64 tokens, so the plain rate is measured over 48 settled steps). With that in place
the server's own numbers line up with the bench to within noise, which is the check that the harnesses
agree:

    server, plain decode : 28.55 ms in decode_step + 0.9 host = 29.46 ms/token   (bench_decode 28.65)
    server, MTP gamma=2  : 63.3-63.4 ms/round, 2.35-2.42 tok/round -> 37.2-38.2 tok/s  = 1.10-1.13x
                           (bench_mtp 63.2 ms/round at 2.70 tok/round = 1.22x on ITS prompt)

The lesson generalises to any in-server rate measurement on this stack: **skip several steps after a
capture, and measure over enough of them, before believing a timer.** A first attempt also blamed the
per-token tokenizer re-decode for a 17 ms/token gap; it was measured innocent (0.008-0.046 ms for
24-200 ids) and the gap was the capture draining.

## The verify pass was 16% slower than it needed to be — one badly-formed conv

Challenge raised: *"there should be no way MTP verify takes as long as generating the token itself,
otherwise MTP wouldn't be as common as it is."* Correct, and worth taking seriously. For a DENSE
transformer verify(K) really is ~one decode step, because decode is weight-bandwidth-bound and
`per_core_M=1` covers 32 rows, so extra token rows are free. What verify(3) *should* cost here:

    block          xlayers   decode(K=1)   ideal verify(3)   measured   waste
    MoE                 40        13.1         19.6 (K*top_k)    18.7      0   <- already optimal
    attention           10         3.0          3.0               5.9      2.9
    gated-delta         30         9.6         10.5              26.6     16.1  <- the whole problem
    lm_head              1         1.2          1.3               1.3      0
                                  28.6         ~36               52.5     ~19

Part of the gap is architectural and unavoidable: 30 of 40 layers are gated-delta, whose recurrence is
sequential in K and whose depthwise conv needs a sliding window over the K new rows plus 3 carried
ones. That is the K-row penalty dense models do not have, and it is why Qwen3-Next-style hybrids are
genuinely harder to speculate on than the models MTP is famous on.

But most of it was implementation: **the verify conv cost 0.388 ms/layer where the decode path does the
same work in ~0.05**, because verify falls back to `_conv_silu`'s general branch — a concat plus four
`[K, conv_dim]` row-range slices, and a sub-tile row slice over 256 tile-columns is ~108 us.

### The fix: stack the conv into 4 dense ops (`QWEN36_GDN_VERIFY_CONV_STACK`, default on)

`out[t] = sum_j tap[j] * xpad[t+j]` is a contraction over conv_k shifts. Stack every shift into one
tensor of `conv_k*T` rows:

    Z   = S @ xpad     [conv_k*T, D]   ONE matmul instead of conv_k row slices
    Y   = Z * TAPS     [conv_k*T, D]   ONE multiply instead of conv_k multiplies
    out = A @ Y        [T, D]          ONE matmul instead of conv_k-1 adds

with `S[j*T+t, t+j] = 1` and `A[t, j*T+t] = 1`. The reason it wins rather than washes: **in TILE layout
a 12-row tensor occupies the same single 32-row tile as the 3-row tensors it replaces**, so widening is
free while the op count collapses. Restricted to `T <= 8` — at prefill's T the stacked form is
`conv_k*T` rows and the loop is correct instead.

MEASURED, per layer at K=3 (`prof_mtp.py --gdnbreak`), and the trap avoided at each step:

| step | conv ms/layer | note |
|---|--:|---|
| slice loop (before) | 0.475 | 4 row slices @ ~108 us + 4 multiplies + 3 adds + concat |
| stacked | 0.235 | arithmetic 0.282 -> 0.075 (**3.7x**); the concat is now 53% of what remains |
| + concat folded into two half-width selection matmuls | ~0.21 | |

    verify trace   52.48 -> 43.96 ms   (-16%)
    round (greedy) 63.2  -> 54.7  ms
    round (sampled) 66.1 -> 57.5  ms   break-even 2.23 -> 1.94 tokens/round

**PCC IMPROVED**: K=3 verify vs 3 sequential decodes went 0.9997 -> **0.99998**, because the tap sum now
accumulates in fp32 inside the matmul instead of as three chained bf16 adds. The 0/1 selection matmuls
are exact (each output row is one input row x 1.0). This is the same "matmul row-sum" that broke the
reverted decode conv-fold (PCC 0.885 there) — the difference is that this one only reorders the SUM,
in the more accurate direction, rather than changing how the window is gathered.

Two process notes worth keeping:
- An isolated microbench said folding the concat away was a wash (171 vs 168 us). **In-model the
  selection matmuls cost 17 us, not 44**, which flipped it — measure the replacement where it will run.
- The concat's isolated 0.124 ms/layer only returned ~1.0 ms of a predicted 2.4 at 30 layers: it
  pipelines. That is the fourth time an isolated stage timing over-counted in this file.

### Chasing the rest of it: what landed, and what would not

Target was ~1.35x for EXACT sampled decode, i.e. a ~48 ms round. Where each candidate went:

| candidate | worth | outcome |
|---|--:|---|
| full-accept commit as one copy (`QWEN36_GDN_COMMIT_COPY`, on) | **2.6 ms** | **LANDED.** At j == K == conv_k-1 every new conv_state row is a conv_in row in order, so the concat + two row slices collapse to `copy(conv_in, cs)` — no matmul, no transient. Round 54.7 -> 52.1 (greedy), 57.5 -> 55.0 (sampled). conv_state PCC 1.0. |
| general commit as selection matmuls (`QWEN36_GDN_COMMIT_SELECT`, OFF) | 5.1 ms | **NOT SHIPPED** — see below. It works and is bit-identical, but destroys sampled acceptance at 40 layers. |
| attention verify | 2.9 ms | **Not attempted.** It is dispatch-bound on ops that are inherent: `paged_update_cache` writes ONE position per call, so the K cache writes cannot be merged without a batch-K KV cache (K x the KV memory — 500 MB at max_seq 8192, 8 GB at 128K). |
| `2x ba repeat` 0.111 ms/layer | 3.3 ms | **Not attempted.** A repeat IS expressible as a selection matmul, but the measured rates say it is a wash: the repeat is ~55 us and an equivalent selection matmul over the same 98k-element output is ~44-50 us. |
| `4x _to_tilerows` 0.063 ms/layer | 1.9 ms | **Rejected on measurement.** ~16 us per call, cheaper than the ~44 us matmul that would replace it. Already efficient. |

**The commit-select failure, recorded because the mechanism is unexplained.** Two selection matmuls plus
an add for the general j case measured **commit 6.18 -> 1.08 ms, round 49.4, greedy 1.31x** — the target
was in reach. But at 40 layers it collapses SAMPLED acceptance: a1 0.727 -> 0.211, tokens/round
2.18 -> 1.00, reproducible across runs, with identical parameters and only the flag changed. Meanwhile:

- conv_state, recurrent_state and the emitted tokens are **bit-identical** to the old path at 4 layers
  over 8 rounds, EAGER and TRACED (`diag_commit.py`);
- `test_gated_delta_verify` reports conv_state PCC **1.0** at every accept count;
- 40-layer **GREEDY** decode is unaffected (2.267 tok/round, eager-vs-traced parity 136/136);
- removing a transient by writing the sum straight into conv_state with `add(output_tensor=)` did NOT
  fix it, so it is not simply transient count. `matmul` has no `output_tensor=`, so the two matmul
  transients cannot be eliminated.

Correct at 4 layers, broken at 40, in one path only, is the signature of the allocation-pressure
failures already recorded here (the multi-bucket prefill wedge; the trace-capture corruption in
`setup_mtp_decode`). The sampled verify trace carries the extra candidate-tail transients, which is the
only structural difference from the greedy path that works. Chasing it further needs more device time
than the remaining 5.1 ms justifies, so it is left in, default OFF, with its measurement recorded —
5 ms is not worth a silent 3x acceptance regression.

**Where that leaves it (40L, gamma=2):**

    verify trace   52.48 -> 43.96 ms      commit (full-accept) 6.18 -> ~1 ms
    round          63.2  -> 52.1 (greedy),  66.1 -> 55.0 (sampled)
    greedy         1.16x -> 1.25x  (43.5 tok/s)
    exact sampled  1.00x -> 1.11-1.19x depending on the prompt region (round is the solid number;
                   tokens/round varies 2.06-2.20 between trajectories)
    lenient:0.5    1.06x -> 1.18x      relaxed 1.11x -> 1.22x

**1.35x for exact was not reached.** It needs ~48 ms, and the three remaining routes are the blocked
commit-select (5.1 ms), a batch-K KV cache (2.9 ms, K x KV memory) and the cross-layer contiguous state
buffer that would let all 30 commits be one copy. The honest ceiling without those is ~1.2x exact.

## Ranking the acceptance rules — and why the first two attempts could not

`bench_mtp.py --rules exact,relaxed,lenient:0.7,lenient:0.5,lenient:0.3`, 40L, gamma=2,
T=0.6/top_k=20/top_p=0.95/presence=1.5, 150 rounds, one model build and ONE capture for all rules (the
verify trace is rule-independent — it only emits candidates).

**Measurement design matters more than the numbers here.** Two earlier attempts were noise:

| attempt | why it failed |
|---|---|
| 20 rounds/rule | ~45 tokens each. Ranked `lenient:0.3` BELOW `exact` — impossible in expectation, since lenient accepts at least as often as exact at every position. |
| 200 rounds/rule + within-run SE | Still ranked `lenient:0.7` below `exact` (1.5 sigma). The SE only covers round-to-round variance *within* one sequence, but **each rule generates its own sequence**, and that between-sequence variance dominates. |

The fix is a **paired** design (`sweep_rules_paired`): run ONE generation, log the (target
distribution, drafted token) pair at every draft position — the draft chain always runs all gamma steps
and verify always scores all K rows, so every round yields gamma complete pairs regardless of what was
accepted — then score every rule on that same set, using each rule's acceptance PROBABILITY rather than
a coin flip (Rao-Blackwellised). Round cost is taken from the timed run, which is legitimate because it
is rule-independent (measured 65.7-66.8 ms across five rules).

    rule          E[tok/round]     a1     a2  ms/token  tok/s  vs base  mean TV  p<0.1
    exact                2.239  0.770  0.585     29.54   33.8    1.00x   0.0000   0.0%
    lenient:0.7          2.311  0.802  0.613     28.62   34.9    1.04x   0.0304   0.0%
    lenient:0.5          2.359  0.824  0.629     28.03   35.7    1.06x   0.0490   0.0%
    lenient:0.3          2.412  0.844  0.648     27.42   36.5    1.08x   0.0691   0.0%
    relaxed              2.473  0.867  0.673     26.74   37.4    1.11x   0.0939   1.0%

**Re-measured after the stacked conv above cut the round 66.1 -> 57.5 ms** — which changes the
recommendation, because exact now pays on its own:

    rule          E[tok/round]     a1     a2  ms/token  tok/s  vs base  mean TV  p<0.1
    exact                2.204  0.740  0.606     26.09   38.3    1.14x   0.0000   0.0%
    lenient:0.5          2.342  0.797  0.657     24.56   40.7    1.21x   0.0543   0.0%
    relaxed              2.427  0.840  0.680     23.70   42.2    1.25x   0.0903   0.7%

Monotone in both acceptance and divergence, no ordering violations — which is how you know the pairing
worked. `mean TV` is the EXACT per-position total-variation distance from the target
(`tt/mtp_sampling.py::realized_tv`: `TV = |a - p(x̂)|` for a point-mass draft, validated against
simulation in the host tests). It is 0 for `exact` by construction, so it also checks the pipeline.

**Conclusions.**
1. **At the 66 ms round, exact sampled speculation was 1.00x — precisely break-even.** Not "a wash by
   coincidence": break-even against a 29.67 ms baseline was 2.23 tok/round and exact delivered 2.239.
   Its acceptance ceiling IS `E[p(x̂)]` ~ 0.77, so no tuning of the RULE could move it — which is what
   pointed at the round cost instead, and the conv fix took the round to 57.5 ms and exact to **1.14x**.
2. **The rule choice is worth ~10%, the round cost was worth ~14%.** Ranking rules was the wrong lever
   to pull first; the useful thing the sweep did was prove that and quantify the divergence.
3. `relaxed` reaches 1.11x by force-emitting drafts the target rated low — 1% below p=0.1.
   `lenient:0.3` gets 1.08x with **no** sub-0.1 emissions and 26% less divergence, so it is the better
   operating point if sampled speedup is wanted; `lenient` is also a dial (lenience 1.0 == exact).
4. My earlier **1.22x for relaxed was wrong** — a 20-round window measured deeper into a generation.

## Refuted: the draft head not seeing the presence penalty

Plausible story: the draft argmaxes its OWN distribution while the target is presence-penalised, so the
two disagree and acceptance suffers; applying the penalty to the draft would fix it while KEEPING
exactness. It would have been the only exactness-preserving speedup available.

**Measured false.** The same paired sweep at presence 0.0 gives a1 = 0.765 / E[tok] = 2.244 / 1.01x
against 0.770 / 2.239 / 1.00x at presence 1.5 — identical within noise, at every rule. So the penalty
is not what costs acceptance, and the draft-side penalty work is not worth doing. (An earlier
20-round pair had suggested presence cost 0.20 tok/round; that was the same noise the paired design
removes — and the two runs' UNPAIRED columns disagreeing by 0.17 tok/round on identical settings is a
third confirmation that only the paired numbers are readable.)

## What ships

| setting | effect |
|---|---|
| `QWEN36_MTP=greedy_only` | speculate only on `temperature == 0` requests — **1.11-1.22x depending on the prompt, exact**, with a live break-even guard that disengages if it is losing. The recommended production setting. |
| `QWEN36_MTP=1` | also speculate on sampled requests: **1.14x and distribution-EXACT** after the stacked conv (was 1.00x before it). No quality tradeoff. |
| `+ QWEN36_MTP_ACCEPT=relaxed` | **1.25x**, at mean TV 0.090 and ~0.7% of tokens force-emitted below p=0.1. |
| `+ QWEN36_MTP_ACCEPT=lenient` (`QWEN36_MTP_LENIENCE=0.5`) | **1.21x** at mean TV 0.054 and no sub-0.1 emissions — most of relaxed's gain for half its divergence. |

Also landed here: **MoE `gate_up` in0_block_w 16 -> 32** (`QWEN36_SPARSE_IN0BW_GU`). MTP.md's A1b probe
predicted 8-11% on the expert matmuls; the real-model A/B gives MoE 0.327 -> 0.318 ms/op and decode
**29.00 -> 28.65 ms/token (+1.2%)**, with `GU=16` reproducing the historical 29.00/34.5 exactly as a
control. Smaller than the probe implied (again: isolated microbenches over-count), but free and it also
shaves the verify pass.

---

# BFP8 KV cache, and paging it for continuous batching (2026-08-23)

Two separate questions that share one code path, so they were answered together.

## 1. Is a BFP8 KV cache accurate enough to ship?

**MMLU cannot answer this, and that is the first finding.** Single-shot prefill runs SDPA on the LIVE
q/k/v tensors and only *fills* the cache (`tt/attention.py: forward_prefill`), so a prefill-scored
eval — which is what `evaluation/run_mmlu_bench.py` is — never READS the KV cache. Every MMLU number
in this repo is blind to KV precision. The instrument that does read it is teacher-forced **decode**,
so `tests/probe_kv_dtype_accuracy.py` (new) reuses the paired one-build design from
`probe_teacher_forced_drift.py`: prefill 1024 tokens, 48 teacher-forced decode steps, arm B replaying
arm A's token stream so each step sees identical inputs.

40 layers, bf16 KV vs BFP8 KV:

    logit PCC   mean 0.995024   median 0.996008   min 0.977164
    median 1st half 0.995533 -> 2nd half 0.996215   (delta +6.8e-04, i.e. slightly UP)
    VERDICT: STABLE -- no accumulation over the run
    argmax agreement 44/48
    4 flips, at steps [10, 18, 25, 26]
    their top1-top2 margins: mean 0.2344   against an all-step mean of 3.4941  -> all near-ties

Same shape of result as every other rounding-level change measured here, at the same magnitude as what
already ships: the conv add-chain (`QWEN36_CONV_ADDCHAIN`, default on) was accepted at mean 0.996 over
96 steps. **Stable, not drifting** — the second half is marginally *better* than the first, which is the
signature of a fixed rounding offset rather than an error that compounds through the recurrence. The
four token flips all sit on near-ties (0.23 mean margin against 3.49 typical), i.e. they change *which*
of two nearly-tied tokens wins, not whether a confident token is right.

### Why it is NOT the default: it turns four MTP parity tests red
Full suite under `QWEN36_KV_DTYPE=bf8`: **116 passed, 4 failed**, all four in
`test_mtp_spec_decode.py` (`test_spec_decode_token_identical` at gamma 1/2/3 and
`test_spec_decode_full_rejection_is_safe`). Worth working out precisely what that means, because the
number that matters is NOT in the failure list.

That file already documents that MTP output cannot be bit-parity with plain decode (verify runs K-row
projections, chained ttl launches and an indexed-gather MoE), so it gates on two constants:
`MIN_IDENTICAL = 4` — require a real matching prefix, since a gross error diverges immediately — and
`TIE_EPS = 2.0` — a divergence is only acceptable if the baseline's top-2 gap was a near-tie. Comparing
the arms on the SAME test:

| | bf16 KV (passes) | BFP8 KV (fails) |
|---|---|---|
| verify-vs-plain logit PCC (`K=1 row 0`) | 0.997435 | **0.997578** |
| that row's argmax | `==` | `!=` |
| top-2 gap at the divergence | 0.2812 | 0.3438 |
| identical prefix | 8/24 | 1/24 |

So bf8 does not degrade the verify path — its logit PCC is *marginally better* — it flips one near-tie
and thereby moves the first divergence from position 8 into the 4-token prefix window the test reserves
for catching gross errors. The failure trips `MIN_IDENTICAL`; it **passes** `TIE_EPS` (0.3438 vs 2.0),
i.e. the test's own real-error criterion says this is a tie flip. And decisively:
**`test_verify_logits_match_sequential_decode` — the gate that file names as the real bug detector
("a REAL bug looks completely different: PCC 0.58-0.75") — is GREEN under bf8** at PCC 0.997578/0.997407.

**Isolation, and it came back negative.** The MTP head has its OWN single-layer KV cache, which this
work switched to follow `kv_cache_dtype` (0.52 GB at 262144 — worth halving alongside the backbone's).
The obvious hypothesis was that the head's cache was the sensitive one, since it feeds the draft, and
that pinning it at bf16 would buy bf8 on the backbone *and* green tests. Measured, with the head's cache
forced to bf16 and only the backbone at bf8: **identical outcome** — same 4 failures, same first
divergence at position 1, same 0.3438 gap, PCC 0.997578 to six figures. The flip comes from the
backbone cache, so pinning the head's cache buys nothing and the change stays.

This is the same tie-fragility class already recorded for the router rewrite, and it lands the same way:
the evidence says the change is harmless, but "relax the test my own change broke" is not a call to make
in passing. So bf8 stays opt-in, and this section is the record of exactly what enabling it costs —
four red exact-parity assertions, no measured loss of fidelity. Note also that MTP is off by default
(`QWEN36_MTP=0`), so a long-context deployment that is not speculating is unaffected either way.

### The dtype contract, which is not symmetric and cost two failed runs to learn
The two cache-write families disagree about who converts:

| op | used by | input dtype | cache dtype |
|---|---|---|---|
| `ttnn.fill_cache` | single-shot + incremental prefill | **must equal the cache** (`update_cache_device_operation.cpp:72`) | bf16 / bf8 / bf4 |
| `paged_fill_cache` | paged prefill | fp32/bf16 (`:34-35` is a permissive disjunction) | bf8 / bf4 |
| `paged_update_cache` | decode write | **fp32 or bf16 ONLY** (`paged_update_cache_device_operation.cpp:295`) | bf16/bf8/**bf4** (`:45-46`) |

So the prefill path needs an explicit `ttnn.typecast` of k/v to the cache dtype, and the decode path
must NOT have one — casting the decode input to bf8 FATALs, which is the exact opposite of the prefill
requirement. `Attention._to_cache_dtype` does the former; the note in `_kv_update_input` pins the
latter so it does not get "fixed" into a bug later. `paged_update_cache` accepting a **BFP4** cache is
free information for a future probe.

A methodological note worth keeping: the first attempt patched the wrong sites because the patch script
asserted uniqueness on a string that legitimately appears twice, aborted, and — since it wrote the file
only at the end — left the tree unmodified while the same command went on to run the probe. The probe
then failed with the *original* error and looked like a code problem rather than a patch problem. Verify
that an edit landed before spending a device run on it.

## 2. Paging the KV cache under continuous batching

`tt/paged_kv.py` already existed for the single-user path, where the module docstring is careful to say
paging is memory-NEUTRAL: `blocks_per_seq == num_blocks` is forced by
`paged_update_cache_device_operation.cpp:194`, so one sequence cannot be over-subscribed and supporting
256K still costs 256K of blocks. **The multi-slot case is where it pays**, and that is what landed here:
`alloc_batch_caches` now builds ONE pool shared by every slot and every attention layer, so the budget
follows concurrent demand (`QWEN36_KV_POOL_TOKENS`) instead of `batch_size x max_seq`.

Three things fell out of doing it properly rather than minimally:

- **Paged prefill needs no scratch copy.** The flat path stages a prompt in a `[1,...]` cache and then
  row-copies it into the slot with `ttnn.fill_cache(dst, src, slot)`. Paged prefill instead writes
  straight into the slot's pages, because the slot's page table already routes it there — so
  `_scratch_caches` now aliases the real caches for attention layers and only stages GDN state (which
  genuinely needs it: GDN state is `[B,Vh,Dk,Dv]` and prefill writes one row). Allocating a real scratch
  there would have duplicated the entire pool, per layer.
- **The chunked-prefill READ table must be per-slot.** `forward_incremental` was handing chunked-SDPA
  the full `[B, blocks]` decode table; the kernel reads row 0 as *the* sequence's mapping, so every slot
  past 0 would have silently read slot 0's pages. `_kv_read_table(slot)` fixes it, and `slot` is now
  threaded through `prefill_long`/`forward_incremental` so a slot's prompt is not limited to what a
  single-shot prefill can materialize. Without that, lifting the per-slot context cap would have been a
  false promise.
- **Dead slots must not be mapped, and finished ones must be released.** `set_decode_state` takes
  `active_slots`; a dead slot parked at position 0 would otherwise take a physical block that nothing
  ever hands back (it has no request to retire), and every completed request would leak its blocks until
  the pool raised "exhausted" with only a few requests live.

### Gate
`tests/test_paged_cb_kv.py` (new) runs both arrangements in one process against one set of weights,
greedy, with slots carrying DIFFERENT prompt lengths (128 and 64) — the case the two are most likely to
disagree on, since flat seeks by absolute row and paged by page-table entry:

    flat   [[4438, 59427, 26073, 26073, 145104, 148589], [68790, 339, 2808, 77330, 6799, 391]]
    paged  [[4438, 59427, 26073, 26073, 145104, 148589], [68790, 339, 2808, 77330, 6799, 391]]
    PagedKV(block=64, pool=8 blocks = 512 tokens, used=5, per-layer 1 MB)
    OK: 2 slots x 13 tokens identical (prompt lens [128, 64])
    blocks in use 5 -> 0 after releasing every slot

Identical token for token, and the pool accounting closes. Note the pool is deliberately *smaller*
than `B x max_seq` (512 tokens against 2 x 1024), which is the whole point: it holds what the two
sequences actually occupy. Green at 4, 8 and 40 layers, and green again with `QWEN36_KV_DTYPE=bf8`,
which is the combination a long-context deployment would actually run and the one place the paged write
path meets a quantized cache (`paged_fill_cache` with a BFP8 input — untested until this ran).

### A second saving that fell out of it, not planned for
Under the contiguous arrangement, `prefill_into_slot` stages each prompt in a separate
`[1, n_kv, max_seq, hd]` scratch cache and then row-copies it into the slot. That scratch is
**context-sized and pool-independent — 2.5 GB at 128K** — and it is pure overhead. Paged prefill writes
straight into the target slot's pages, so the attention scratch now *aliases* the pool and only GDN
state is staged (~30 MB, context-independent). Allocating a real paged scratch there would have been
much worse than the flat one it replaced: `_alloc_cache` on the paged path returns a whole new pool, per
layer. That trap is worth remembering — with paging, "allocate a scratch cache" no longer means "one
sequence's worth".

### The one thing that broke, and it was a test stub
`test_cache_topology.py` builds a bare `TtModel` via `__new__` carrying only the attributes
`_alloc_cache` reads — and that set grew by `paged_kv`/`kv_pager`. Because the stub bypasses `__init__`
entirely, the growth surfaced as an `AttributeError` inside a *cache-topology* test, nowhere near paged
KV. Worth noting as a pattern: an `__new__`-based stub is a hidden coupling to a method's attribute
list, so it needs a comment saying so (it now has one). Fixed, plus a new `test_paged_cache_topology`
that pins the pool geometry (`[num_blocks, n_kv, block, head_dim]` per layer, one shared pager, and the
`blocks_per_seq <= num_blocks` constraint the kernel enforces). Suite: **120 passed, 0 failed.**

### The bug this shipped with, and how it was found (2026-08-23)
Asked whether 4 concurrent 256K requests would work, the honest answer required testing the 4-slot path
rather than extrapolating from the 2-slot gate. It diverged: **slots 1 and 3 emitted garbage** (token id
`0` repeatedly, when the paged arm ran first). Root cause, and it was in `PagedKV`, not in ttnn:

> `paged_scaled_dot_product_attention_decode` rounds the k-range it READS **up to a whole
> `k_chunk_size`** (128 here). `ensure()` mapped blocks only up to `cur_pos`. So a slot at position 128
> reads positions 0..255 -- four 64-token blocks -- while only three cover its `cur_pos`, and the fourth
> page-table entry was still `UNMAPPED (-1)`. The kernel then attends over whatever that address holds.

The slot pattern is exact arithmetic, which is what confirmed it:

| pos | reads up to | blocks needed | `ensure` mapped | |
|--:|--:|--:|--:|---|
| 64 | 128 | 2 | 2 | ok |
| 128 | 256 | 4 | **3** | garbage |
| 192 | 256 | 4 | 4 | ok |
| 256 | 384 | 6 | **5** | garbage |

Fix: `read_granularity_default()` (`QWEN36_KV_READ_GRAN`, 128) and `ensure` rounds `upto_pos` to it
before computing `need`; the pool is also rounded up to a whole granularity window, since otherwise its
top is a silent cliff. Costs at most one extra block per slot. **This must stay equal to
`Attention._sdpa_pc`'s `k_chunk_size` (attention.py:141)** -- they are one contract, not two knobs.

**Four wrong hypotheses, and what killed each — the debugging record is the useful part:**

1. *"chunked SDPA mis-strides a multi-block BFP8 cache"* (the flat path is ONE block, so it never strides;
   a BFP8 tile is 1088 B, not a power of two). Refuted: paged == flat to 6 decimals.
2. *"the chunked WRITE into a sub-table is wrong at BFP8."* Refuted: identical max and mean error.
3. *"paged sdpa_decode is wrong at B>1."* Refuted at B = 1/2/4/8, both dtypes.
4. *"paged_update_cache quantizes differently into a BFP8 cache."* Refuted, bit-identical.

Then a **direct** comparison of prefilled state showed the two arms bit-identical (KV rows for every slot
AND GDN state, max|delta| exactly 0), which located the fault in decode. What finally cracked it was
**swapping the arm order**: running paged first turned the subtle token divergence into repeated token
`0`, i.e. reading uninitialised memory rather than a numerical difference. Order-dependence plus
"every component is exact" is the signature of an out-of-bounds read, not of arithmetic.

Two method errors worth keeping, because both cost real time:

- **I twice attributed the trigger to the wrong variable** by not holding the others fixed -- first to
  "chunked prefill", then to "4 slots". The failing run also set `QWEN36_KV_DTYPE=bf8` while every
  passing comparison run did not. Only a controlled one-variable sweep (same config, dtype varied)
  settled it; and the real trigger turned out to be neither -- bf8 merely made an out-of-bounds read
  reliably fatal instead of accidentally harmless.
- **The first three isolation probes compared each arm's error against a reference, not the arms against
  each other.** Two arms can share an error magnitude and still differ. Every such probe needs the
  arm-vs-arm diff as its actual assertion.

Why the original gate missed it: `PROMPTS = (128, 64)` at 2 slots passed *with the bug present*. The
default is now `(64, 128, 192, 256)` -- the set that splits on the rounding -- and the docstring says to
run `--long` and `QWEN36_KV_DTYPE=bf8` as well, since bf16 hid it in some pool layouts. Suite after the
fix: **120 passed, 0 failed**, and all six paged configs (2/4 slots x bf16/bf8 x short/chunked) green.

### What the vLLM adapter does with it
`packaging/vllm_bundle/generator_vllm.py` turns paging on by default for `B>1` and drops the per-slot
context cap that the contiguous arrangement forced. The default pool is
`max_num_seqs x QWEN36_CB_TOKENS_PER_SLOT` (8192) — deliberately the same number of tokens the old
contiguous allocation pinned, so **enabling paging changes flexibility, not footprint**: one request may
now use the whole pool, or `B` requests may share it, from the same bytes. `allocate_kv_cache` still
returns `[]` and vLLM's block ids are still used only as stable request keys; the two page tables never
have to agree, because 30 of 40 layers carry GDN state that has no `AttentionSpec` to hand vLLM at all.
