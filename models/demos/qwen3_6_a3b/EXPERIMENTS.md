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
