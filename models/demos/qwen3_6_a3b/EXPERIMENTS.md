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
