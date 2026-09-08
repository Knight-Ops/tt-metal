# Qwen3.6-35B-A3B Prefill — Performance Analysis & Roadmap

Companion to `README.md` (read that first for the model, environments, and how to run). This doc
covers **prefill throughput**: the current baseline, the roofline it's measured against, what's been
optimized, and what's left — including a thorough scope for **traced incremental (multi-turn)
prefill**. Decode is the other engineer's path (see `FUTURE_OPTIMIZATIONS.md`); summarized here only
for the roofline.

> History note, in order, because each revision of this doc was superseded by measurement:
> (1) "prefill is host-dispatch-bound, tracing is the lever" — was true at ~54% dispatch, and tracing
> did buy ~1.5x; (2) "the MoE matmuls are at roofline, no clean MoE win remains" — the conclusion held
> but the roofline was self-referential (see §1.1); (3) once `ttnn.transformer.chunk_gated_delta_rule`
> removed the gated-delta op count, **tracing the prefill is worth only ~1.02x** and prefill is ~75%
> MoE, DRAM-bandwidth-bound. The doc below is state (3).

---

## 1. Current baseline state (single P150, 40 layers, BFP4 experts, tt-metal v0.77.0)

Measured 2026-08-20, eager prefill, warm, real text, best-of-3 (`scratchpad/final_bench.py`; the
standing harness is `tests/bench_prefill.py`).

| config | T=256 | T=512 | T=1024 |
|---|--:|--:|--:|
| v0.77.0 defaults (GDN fused op off, bf16 MoE activations) | 672.3 ms / 381 tok/s | 1277.3 ms / 401 tok/s | 2489.4 ms / 411 tok/s |
| + `ttnn.transformer.chunk_gated_delta_rule` **default on** | 450.3 ms / 568 tok/s | 837.3 ms / 611 tok/s | 1607.4 ms / 637 tok/s |
| + BFP8 MoE matmul in0 & `in0_block_w=8` **default on** | **336.5 ms / 761 tok/s** | **614.3 ms / 833 tok/s** | **1162.4 ms / 881 tok/s** |
| + last-token D2H in ROW_MAJOR & dead `exp` removed (2026-08-22) | **327.9 ms / 781 tok/s** | **607.5 ms / 843 tok/s** | **1154.0 ms / 887 tok/s** |
| **total speedup** | **2.05x** | **2.10x** | **2.16x** |

> The 2026-08-22 row is `scratchpad/bench_prefill_eager.py` (best-of-3 after 2 warm-ups, eager, warm
> cache), NOT paired against the row above it — that row was measured on 2026-08-20 with the same
> defaults. What makes the attribution credible anyway is that the saving is **the same ~8.5 ms at all
> three lengths** (−8.6 / −6.8 / −8.4 ms), which is the signature of a FIXED per-call cost, i.e.
> exactly the one-shot logits readback §4.3 predicted — not a per-token effect. §4.3 estimated ~14 ms
> at T=512 from the 15.9 MB transfer size; the real figure is ~7-9 ms.

> **Read prefill tok/s at a FIXED length.** `tests/benchmark.py` reports `T / TTFT` per prompt and then
> takes an unweighted MEAN over prompts of 24-220 tokens, where that ratio measures fixed per-call cost
> rather than throughput (same run: 37.7 tok/s at 24 tokens, 257 tok/s at 220). The old "~200 tok/s
> prefill" headline was that mean. Steady-state is the table above.

### 0.9 Prefill block cost model, and what it rules out (2026-09-08)

One incremental prefill block, 40 layers, decode trace resident, best of 3
(`hangdbg/blockcost.py`, widths 128/256/384/512 at a fixed offset):

```
time = 127 ms fixed + 0.95 ms per (padded) token
```

The 127 ms floor is a full sweep of the MoE expert weights (~16 GB of the 21.7 GB resident model), so
it is DRAM-bound and irreducible **per block**. Two consequences that overturn the obvious
intuitions:

| | |
|---|---|
| a 512-token tail as **one** block | **613 ms** |
| the same 512 as **4x128** blocks | **995 ms** (62% worse) |

1. **Fewer, wider blocks win.** Chunking finer to reduce padding waste loses to the per-block MoE
   sweep every time.
2. **Finer prefix-reuse checkpoints do NOT pay.** A 128 snapshot grain re-ingests up to 383 fewer
   tokens (~364 ms) but always needs a second block to land the snapshot on the grain (+127 ms) and
   pays padding on both. Measured over a 10-turn growing chat, prefill on the reuse turns:

   ```
   grain 512: 0.33 0.55 0.88 0.33 0.55 0.61 0.88 0.33 0.56   mean 0.558  worst 0.88
   grain 128: 0.56 0.32 0.82 0.54 0.55 0.55 0.56 0.56 0.55   mean 0.557  worst 0.82
   ```

   Identical mean. The finer grain only flattens the sawtooth, and it costs branch-reuse coverage
   (at 4 ring slots, 512 spans 2048 tokens of history against 512). Available as
   `QWEN36_CKPT_ALIGN=128`; not the default.

**Where the win actually is, and it is now implemented:** one block per turn containing only the
genuinely new tokens, with the snapshot taken at a 128 boundary *inside* that block.

`ttnn.transformer.chunk_gated_delta_rule` takes an `initial_state` and returns the final one, so the
recurrence splits **exactly** -- running `[0,j)` then `[j,T)` seeded with the first half's result is
identical to one pass (measured core PCC 1.000000, final-state PCC 1.000000,
`hangdbg/splitprobe.py`). Only the recurrence splits: attention, MoE and the norms still run once
over the whole block, so a resumable checkpoint at an interior 128 boundary costs one extra kernel
launch and one extra conv slice per linear layer instead of a second 127 ms MoE sweep.

Plumbed as `snap_at`/`snap_dst` from `TtModel.forward_incremental` through the decoder into
`GatedDelta.forward`, writing straight into the checkpoint ring slot
(`TtModel._begin_inblock_snapshot`). `GatedDelta.supports_inblock_snapshot` is False when the fused
op is off (`QWEN36_GDN_FUSED_OP=0`), and the planner then falls back to the separate snapshot block.

MEASURED, 40 layers, 10-turn growing chat, prefill per reuse turn:

```
before (512 grain):  0.33 0.55 0.88 0.33 0.56 0.61 0.88 0.33 0.56   mean 0.559  worst 0.88
in-block (128 grain):0.34 0.32 0.62 0.34 0.34 0.34 0.34 0.34 0.34   mean 0.369  worst 0.62
```

**mean -34%, worst -30%, and 7 of 9 turns flat at 0.34 s.** This only pays with the finer padding
ladder ({128,256,384,512}): a ~130-token tail padded to 512 costs ~614 ms, padded to 256 it costs
~370 ms -- which is why the ladder default was widened at the same time.

Also from this pass: **tracing the reuse prefill is not the lever** -- see the note at the top of
this file (~1.02x) and the component profile below (glue is ~1% of prefill; there is no dispatch pool
to reclaim).

### 0.95 Parked conversations — the fix that made reuse actually engage in production (2026-09-08)

Everything in §0.9 measures prefix reuse *in a lab loop*, one conversation at a time. In the shipped
vLLM container it delivered **nothing**: every request logged `reused 0 (0%) ... [full prefill]`.

The cause was not the reuse machinery. Real clients interleave prompt streams — a chat UI runs the
user's conversation *and* short auto-title / auto-tag completions against the same engine. From the
container's own logs:

```
21:12:12  T=6288  start=0  checkpoints=[]                     reused 0    <- conversation, turn 1
21:12:23  T=209   start=0  checkpoints=[5120,5632,6144,6272]  reused 0    <- a side request
21:12:48  T=318   start=0  checkpoints=[]                     reused 0
21:13:57  T=6344  start=0  checkpoints=[]                     reused 0    <- turn 2, only +56 tokens
21:14:18  T=409   start=0  checkpoints=[5120,5632,6144,6272]  reused 0
```

Two single-slot pieces of state destroying each other. The adapter kept ONE `_reuse_hist`, so a side
request overwrote the conversation's history and the next turn's common prefix collapsed to ~0; and
`prefill_reuse` ended with an **unscoped** `drop_state_checkpoints(above=T)`, so a 209-token request
discarded every checkpoint the 6.3K conversation held — which is the `checkpoints=[]` above. Turn 2
extends turn 1 by 56 tokens and should have been ~99% reuse; it paid 7.65 s.

**Why the adapter could not fix this alone.** Reuse restores the gated-delta state at position P and
recomputes only `[P, T)`. That is correct *only if the KV rows below P still belong to this
sequence*. With one flat `[1, n_kv, max_seq, hd]` cache they do not — every prefill writes rows from
0 — so an adapter that merely remembered four histories would restore state against another
conversation's KV and emit plausible garbage instead of a slow answer. Paging is what makes parking
free: each conversation owns its own blocks out of one pool, so nothing overlaps and nothing is
copied. `QWEN36_PAGED_KV` is therefore forced on at B=1 whenever conversation slots are enabled, and
it is **memory-neutral** — the pool defaults to `max_model_len`, exactly the tokens the flat cache
reserved anyway, and the parked conversations share it.

The pieces:

| piece | where |
|---|---|
| N host page-table mapping rows behind ONE device table row | `PagedKV.__init__(nslots=)`, `activate()` |
| per-conversation band of the checkpoint ring (static partition, no cross-eviction) | `TtModel._ckpt_span`, `CKPT_BAND` |
| conversation-scoped search / restore / invalidate | `checkpoint_positions(conv)`, `restore_gdn_state(pos, conv)`, `drop_state_checkpoints(conv=)` |
| routing a prompt to a conversation, LRU aging | `ConversationSlots` (`tt/model.py`) |
| pool headroom check before a prefill | `kv_headroom_tokens()`, `_prefill_single_seq` |

Switching conversations rewrites only the page table's **contents**, at its baked address — the same
in-place `copy_host_to_device_tensor` `PagedKV.ensure` already does on block boundaries, so it is
safe with a decode trace resident (`MEMORY.md` §1). Nothing is reallocated per conversation, which is
what keeps this compatible with everything in `IMPLEMENTATION_GUIDE.md` §1.

**MEASURED, 40 layers, decode trace resident throughout** (`hangdbg/multiconv.py`, 4 slots): the
container's traffic shape — one 6.3K conversation growing +56 tokens a turn, two short side streams
between every pair of turns, 6 decoded tokens per request.

```
chat turn 0: T=6288  reused 0     (  0%)   7.66 s     <- cold, unavoidable
chat turn 1: T=6344  reused 6272  ( 99%)   0.29 s
chat turn 2: T=6400  reused 6272  ( 98%)   0.28 s
chat turn 3: T=6456  reused 6272  ( 97%)   0.35 s
chat turn 4: T=6512  reused 6272  ( 96%)   0.35 s
                                mean 0.32 s, worst 0.35 s
```

**7.66 s → 0.32 s** on the turns that were previously paying full price, no wedge, program cache
saturating at 552 entries and DRAM flat at 10.50 GB free across the run. Correctness is gated
separately: `test_interleaved_conversations_keep_their_prefix` resumes a conversation *after* an
interleaved request and holds PCC 0.99649 against a full prefill of the same prompt — which it can
only do if the interloper's KV went somewhere else.

Cost at 40 layers: the ring goes from 4 slots to `4 x CKPT_BAND` (2), i.e. 8 x 45 MiB = 360 MiB,
against ~10.6 GB free. `QWEN36_CONV_SLOTS=1` restores the single-conversation behaviour exactly.

### 1.0 Tracy per-component profile (2026-09-05) — the MoE is 69% of prefill

`prof_prefill.py` under Tracy, T=512, 4 layers, eager window, **device-kernel time** (not synced, not
inflated by a per-phase barrier). Window total 60.104 ms. Proportions hold at 40 layers (identical op
shapes); absolutes do not.

| component | self % | note |
|---|--:|---|
| `moe.gate_up_mm` | 21.4 | the [E,tc,H]@[E,H,2I] dense expert matmul |
| `moe.swiglu` | 16.1 | |
| `moe.down_mm` | 14.0 | |
| **`moe.repeat`** | **9.0** | pure overhead: the E-fold broadcast of the activations |
| **`moe.reduce`** | **5.9** | pure overhead: the sum over all 256 experts |
| `moe` router/shared/scatter/combine | 2.0 | |
| **MoE total** | **68.6** | |
| `delta.conv` | 7.3 | |
| `delta.qknorm` | 6.2 | |
| `delta.norm_out` | 5.8 | |
| `delta.recurrence` | 4.2 | |
| `delta.in_proj` + split + gate | 1.6 | |
| **gated-delta total** | **25.1** | |
| attention (out/qk_norm/qkv/rope/sdpa/kv_write) | 3.7 | |
| head (lm_head + norm + d2h) | 2.0 | |
| **glue** (norms, residuals, embed, rope table) | **~1.0** | |

Two conclusions that change where the work should go:

1. **There is no "dispatch/glue" pool to reclaim.** Norms, residuals, scatter, combine, split, gate,
   embed and the rope table together are ~1% of prefill. Any earlier framing that put ~40% of prefill
   in elementwise/layout/dispatch is not supported by this capture; it is ~94% real MoE + gated-delta
   work. (Consistent with tracing the prefill being worth only ~1.02x.)
2. **`repeat` + `reduce` alone are 14.9% of ALL prefill time**, and they are work a gathered/sparse
   expert path would not do at all — they exist only to materialise `[E, tc, H]` and then sum it away.
   Together with the 32x FLOP redundancy in the three matmuls, this is where the roofline gap lives.

### 1.1 Where the time goes now

Device-synced per-phase (`QWEN36_PROFILE_PHASES=1`), T=512, 40 layers, warm. Absolute ms are inflated
~2x by the per-phase sync (see `tt/prefill_profiler.py`); read the shares.

| phase | x | before (bf16 MoE) | after (shipped) | share, after |
|---|--:|--:|--:|--:|
| `moe` | 40 | 18.68 ms | **13.78 ms** | ~44% |
| `delta` | 30 | 6.60 ms | 6.57 ms | ~16% |
| `attn` | 10 | 2.97 ms | 2.89 ms | ~2% |
| `head` | 1 | 15.5 ms | 9.6 ms | ~1% |
| `norm` | 80 | 0.16 ms | 0.15 ms | ~1% |

MoE expert stages, per layer per 256-token chunk (`before -> after`, sum **8.27 -> 5.62 ms = 1.47x**):
`gate_up` 2.18 -> 1.70, `swiglu` 2.13 -> 1.40, `down` 1.63 -> 1.14, `repeat` 1.38 -> 0.86,
`reduce` 0.95 -> 0.52.

**Prefill is the MoE, and the MoE is DRAM-bandwidth-bound.** Per layer per 256-token chunk the dense
expert stages move ~2.48 GB — `repeat` 269 MB, gate_up 705, swiglu ~738, down 487, reduce 269 — of
which only ~0.45 GB is weights. Against the measured 8.27 ms of per-op time that is **~330 GB/s, i.e.
~76% of the P150's 432 GB/s DRAM peak**, and the individual stages sit at 45-80% of it.

A previous revision of this doc concluded "the matmuls are essentially at roofline… no clean MoE win
remains" by anchoring the roofline on its own measured 130 TFLOP/s. That number is only **22% of the
card's ~594 TFLOP/s LoFi peak** (110 cores x 5.4 TFLOP/s, `tech_reports/GEMM_FLOPS`); the wall is
bandwidth, not compute. The conclusion was right, the mechanism was not — and the mechanism is what
says which levers can work.

---

## 2. Roofline — where the silicon ceiling is

- **Compute:** ~594 TFLOP/s LoFi (110 Tensix x 5.4 TFLOP/s at 1.35 GHz).
- **DRAM:** **432 GB/s** (8 controllers x 54, tt-npe Blackhole model).
- **Dense MoE FLOPs**, all 256 experts x all tokens: `E*T*(H*2I + I*H)*2` = 16.5 TFLOP at T=256 over
  40 layers — 32x the `top_k=8` requirement, but only ~28 ms of compute at peak. **Irrelevant**: the
  same work needs 99 GB of DRAM traffic, i.e. **>=230 ms**. The MoE is ~1.3x off that floor and cannot
  be tuned closer; the only lever is fewer bytes.
- **Gated-delta**, by contrast, moves ~250 MB/layer at T=256 (a 0.58 ms floor) against several ms
  measured — ~12% of DRAM peak. It is op-count/occupancy-bound, which is exactly why one fused op
  bought 1.5x and why byte-shaving there is pointless.

### 1.2 Where gated-delta PREFILL time goes — and why the glue is NOT the lever either (2026-08-22)

`tt/gated_delta.py` had no per-stage phase timers (the MoE did), so nobody knew. Added them
(`QWEN36_PROFILE_PHASES=1`, gated on `T > 1` so they never sync inside a decode step). T=1024,
40 layers, current tree — synced ms, so read the SHARES:

| stage | ms/layer | x30 | share of `delta` |
|---|--:|--:|--:|
| `delta.qknorm` | 2.65 | 79.4 | **24%** |
| `delta.conv` | 2.64 | 79.3 | **24%** |
| `delta.norm_out` | 2.14 | 64.2 | 19% |
| `delta.recurrence` | 2.10 | 63.1 | **19%** |
| `delta.in_proj` | 0.61 | 18.2 | 6% |
| `delta` (total) | 11.01 | 330.3 | 100% |

**The recurrence is now only 19% of gated-delta prefill.** Every prior optimisation here targeted it --
the ttl `_chunk_state` kernel, the batched chunk-prep, and finally
`ttnn.transformer.chunk_gated_delta_rule` (1.49-1.55x). It worked, and it worked itself out of the top
spot: the GLUE is now 73%.

**But the glue is not worth much, and this is the number that matters.** `delta.qknorm` was 6 ops per
call (`multiply, sum, add, rsqrt, multiply [, multiply]`) x2 for q and k. Replaced with a 2-op identity
(`QWEN36_GDN_L2NORM_RMS`, default on):

    l2norm(x) = x / sqrt(sum(x^2) + eps) = rms_norm(x, epsilon=eps/K) * K**-0.5

with the caller's scale folded into the trailing multiply. PCC-clean at both settings. Paired A/B,
eager, T=1024: **1154.0 -> 1146.3 ms, 887 -> 893 tok/s. +0.7%.**

Eliminating two thirds of the ops in a stage the phase timer calls **24% of `delta`** bought **0.7% of
the call.** So the phase timers substantially over-count `delta`: each timer SYNCS, which serialises
work that otherwise overlaps with the MoE matmuls around it. Back-solving, gated-delta's real
wall-clock share is ~6-7%, not the 13% the synced accounting suggests -- the same "isolated stage
timings over-count because ops pipeline" lesson recorded four times in `EXPERIMENTS.md`, now with a
prefill instance.

**Implication for the remaining prefill plan.** Scaling that result, the other two glue stages
(`conv` 24%, `norm_out` 19%) are worth perhaps ~1.5% between them. So:

    MoE          ~67% of the call   -- 2.5x available, BLOCKED on 12 GB (see 2.0)
    gated-delta  ~6-7%              -- ~2% total available, mostly spent
    attention    ~2%
    head         0.1%               -- was 15.5 ms, now 1.72 ms after the ROW_MAJOR D2H fix

**There is no second lever in prefill.** The MoE reformulation is the only thing with real value left,
and it is a memory problem rather than a compute one. Stop tuning the glue.

### 2.0 The FLAT reformulation: 2.5x, measured, and blocked by 12 GB (2026-08-22)

The dense MoE computes `[E, tc, H] @ [E, H, 2I]`, which forces the E-fold `ttnn.repeat` and an E-fold
output that must then be `sum`ed. Merging the expert axis into the channel axis removes both:

    flat:  hgu = x[tc, H] @ Wgu[H, 2*E*I]      in0 is multicast, never materialised E times
           gate, up = two CONTIGUOUS slices;  h = silu(gate) * up * w
           y   = h[tc, E*I] @ Wdn[E*I, H]      the sum over E IS this matmul's K reduction

Measured at real dims (`tests/probe_flat_moe.py`, traced, BFP4 weights, BFP8 in0), **ms per 256 tokens**
so the chunk sizes are comparable:

| tc | batched (current) | flat | flat + `minimal_matmul(fuse_swiglu)` | batched N-split=4 |
|--:|--:|--:|--:|--:|
| 256 | **4.74** | 6.61 | 8.23 | 9.86 |
| 512 | **L1 FAIL** | 3.50 | 4.35 | 7.77 |
| 1024 | L1 FAIL | 2.88 | **2.50** | 6.90 |
| 2048 | L1 FAIL | 2.60 | **1.91** | — |

PCC vs the batched form is 0.9994-0.9996 at tc=256, which also confirms the one layout claim the design
rests on: **`down_sp` `[E, I, H]` -> `[E*I, H]` is a free view** in TILE layout, no relayout needed.

**The win is 2.48x, and it is NOT about bytes.** My own prediction here -- "2.3x fewer bytes, therefore
~1.75x" -- was wrong twice: at the tc the model actually uses, the flat form is 1.4x **SLOWER**. What it
really does is remove the L1 wall that pins `_DENSE_TMAX` at 256, after which the 453 MB/layer expert
weight read amortises over 4-8x more tokens. `minimal_matmul` also flips from slower to faster at
tc >= 1024, matching tt_transformers' own "only above a length cutoff" rule.

**BLOCKER: it needs a second copy of the gate_up weight, and there is no room.** `[H, 2*E*I]` is
H-major where the sparse decode layout `[1, E, H, 2I]` is E-major, so the flat weight is a genuine
retile, not a view: **302 MB/layer x 40 = 12.1 GB**, against ~9.6 GB free (22.7 of 32.3 GB used,
`MEMORY.md` §5). The down side is free; only gate_up is affected. Decode cannot share the flat layout
because `ttnn.sparse_matmul` needs the `[1, E, K, N]` form to gather 8 of 256 experts.

**Two memory-neutral escapes were measured and both fail:**
1. **Hybrid** (batched gate_up + flat down; both weights are then free views): **15.6 ms at tc=256, 3.3x
   SLOWER**, and it still L1-fails at tc=512. The `[E, tc, I] -> [tc, E*I]` permute of `h` is brutal,
   and the L1 wall is not confined to the down projection as `_dense_experts`' docstring suggests.
2. **N-splitting the batched matmuls** (smaller per-core output block, same weight layout): lifts the L1
   wall -- it runs at tc=512 and 1024 -- but is never faster, because `per_core_N == Nt` means each of
   the `ns` chunks RE-READS the whole `xe` in0 (143 MB x 4 at tc=256).

**So the 2.5x is available but priced in device memory.** Three ways forward, in order of appeal:
- **A second card, or any card with >= 44 GB.** The flat weight fits trivially and needs no other change.
- **Trade decode for prefill:** store ONLY the flat layout and rebuild decode's expert path as 16
  tile-aligned column slices (expert `e`'s gate is `[e*I, (e+1)*I)`, its up is `[E*I + e*I, ...)`) feeding
  one dense `[1, 2048] @ [2048, 8192]`. Rough estimate +93 us/layer = **+3.7 ms/token (~15% decode
  regression)** to buy ~1.5x prefill. Unattractive at a 39 tok/s decode, plausible if prefill is the
  product bottleneck. Measure before believing the estimate.
- **Leave it.** Prefill's non-MoE half (GDN glue, ~20%) has no such blocker and is untouched.

### 2.1 Two structural MoE fixes were built and measured. Both are SLOWER. {#dead-ends}

Recorded so they are not re-attempted; details in `tt/moe.py`'s module docstring.

1. **Per-expert token gathering** — permute tokens into an `[E, C, *]` capacity tile so each expert
   sees only its own tokens. Correct (PCC 0.99998 vs dense) and fast in isolation: **1.83x at T=256,
   3.08x at T=512, 4.60x at T=1024** on the MoE block. It is nevertheless unusable, because of what
   the router actually does. Measured tokens-per-expert over 40 layers at T=512 on real text:

   | | mean | max | experts touched |
   |---|--:|--:|--:|
   | random token ids | 16 | 163-510 (median 491) | 112-239 of 256 |
   | real text | 16 | 138-508 (median 255) | 100-181 of 256 |

   A few **sink experts take nearly every token**, so a uniform capacity must be `C ~ T` and saves
   nothing. In the 40-layer model the gather fired on **1 layer out of 40** and the path came out
   1.05x SLOWER (the routing readback costs 0.43 ms/layer for nothing).
2. **Union-indexed sparse** — `ttnn.sparse_matmul` gather mode over only the experts some token
   selected (the other tail of the same skew: 30-60% of experts are selected by nobody). ~2.7x fewer
   bytes, no permutation needed, and it deletes the `repeat`. Measured end to end: **0.44-0.50x** at
   T=256/512/1024. The 1D sparse program config assigns one N-tile per core, so it runs on `Nt` = 32
   cores (gate_up) or 64 (down) of 110; the grid loss beats the byte saving. This also explains the
   older per-32-token-tile result (0.22-0.37x) — same cause, worse constant.

**So the dense batched matmul over the full 11x10 grid is the right shape on this hardware.** What is
left is narrowing the bytes it moves, which is §3.8.

## 3. Shipped optimizations (what moved the numbers)

1. **MoE dense-matmul program config** (`tt/moe.py` `_dense_expert_pc`) — the default `ttnn.matmul`
   heuristic serialized the 256-expert batch onto few cores (~17 TFLOP/s); an explicit
   `MatmulMultiCoreReuseProgramConfig` over the full 11×10 grid → **gate_up 15.6→2.1 ms, down 5.0→1.5
   ms (PCC 1.0)**. The single biggest eager win.
2. **Last-token lm_head** (`tt/model.py` `_head`) — slice to the final token before the 248k-vocab
   matmul (prefill only needs `logits[0,-1]`); ~T× less compute + D2H.
3. **MoE routing-fold** — fold the routing weight into the down-proj input (`down` is linear), scaling
   `[E,T,I]` instead of `[E,T,H]`; plus `zeros_like→×0` (trace-safe).
4. **Prefill tracing + bucketing** (`tt/model.py` `capture_prefill_trace`/`forward_prefill_traced`/
   `setup_prefill_traces`) — replay the op graph to remove per-op host dispatch (**1.50×**). One trace
   per padded bucket; a gated-delta valid-mask zeros padding (PCC-identical to eager zero-pad); eager
   fallback for T > largest bucket. Required several trace-safety fixes (no in-graph `ttnn.zeros`/
   `tril`/`fill` — pre-allocated pools + mask-multiplies + persistent cache buffers).
5. **Eager incremental (from-cache) prefill** (`forward_incremental`; §4) — ingest only the new tokens
   continuing from cache (gated-delta state-in + attention `chunked_scaled_dot_product_attention` over
   the cache); for multi-turn, O(M) instead of re-prefilling O(P+M). Validated PCC ≥ 0.999.
   **2026-08-23: `slot` is now threaded through `prefill_long`/`forward_incremental`**, so a continuous-
   batching slot's prompt goes through the chunked path instead of being limited to what a single-shot
   prefill can materialize. The subtlety that made this necessary rather than cosmetic: chunked-SDPA's
   READ page table must be **per-slot** (`_kv_read_table`). It was previously handed the full `[B, blocks]`
   decode table, and the kernel treats row 0 as *the* sequence's mapping — so every slot past 0 would
   have silently read slot 0's pages. Flat-path behaviour is unchanged (one cache, trivial table).
6. **MoE large-T chunking** (`_DENSE_TMAX=256`) — the tuned matmul holds each core's full `[T,N]` output
   in L1, overflowing past T≈320; chunk over T in ≤256 blocks (same fast config per chunk). Restores
   arbitrary length; validated T=256/384/512/768.
7. **`ttnn.transformer.chunk_gated_delta_rule` default ON** (`tt/gated_delta.py` `_GDN_FUSED_OP`,
   `QWEN36_GDN_FUSED_OP=0` reverts) — tt-metal v0.77.0's native fused chunked gated-delta op replaces
   the in-tree chunk prep + chunk-state loop. **Measured 40-layer eager prefill 676.6 -> 452.8 ms at
   T=256, 1281.7 -> 840.0 ms at T=512, 2492.5 -> 1610.4 ms at T=1024 (1.49-1.55x)**; core/state PCC
   ~0.999998. It had been implemented but left opt-in; only the vLLM bundle manifest set it, so a plain
   `demo.py` run did not get it. Eager only — the traced path keeps its pre-allocated buffer pool, and
   that costs nothing now that tracing the prefill is worth ~1.02x. Gotcha: it hangs the device
   silently below `head_dim=128`, hence `_FUSED_OP_MIN_HEAD_DIM`.
8. **BFP8 dense-MoE matmul in0 + `in0_block_w=8`** (`tt/moe.py` `_MOE_ACT_DT` / `_DENSE_IN0BW`,
   `QWEN36_MOE_ACT_BF16=1` / `QWEN36_DENSE_IN0BW=1` revert) — the only remaining MoE lever once §2.1
   ruled out both sparse formulations: narrow the two tensors that FEED a matmul (`xe`, the E-fold
   activation broadcast, and `h`, the down projection's input) from BFLOAT16 to BFLOAT8_B. **Measured
   1.43x on the MoE block at tc=256 (8.91 -> 6.22 ms) and 40-layer prefill 450.3 -> 336.5 ms at T=256,
   837.3 -> 614.3 ms at T=512, 1607.4 -> 1162.4 ms at T=1024 (1.34-1.38x).** Because `ttnn.matmul`
   inherits its output dtype from in0, `ye = down(h)` narrows too, so three of the four big
   `[E, tc, *]` intermediates shrink for one knob.

   Numerics: final-logits PCC vs the bf16 path is 0.984 / 0.984 / 0.993 at T=256 / 512 / 1024, with
   **argmax matching at all three** and top-5 overlap 4-5/5. Paired MMLU-Redux over the same 200
   questions: **79.0% bf16 vs 78.0%** with BFP8 forced on every chunk — 2 questions out of 200, i.e.
   noise at that sample count, and a strict upper bound on the cost, because prompts shorter than one
   full chunk are **bit-identical to the previous default** under the chunk gate below and every MMLU
   prompt (53-115 tokens) is in that regime. Both short and long (323-token) chat generations are
   coherent and on topic.

   The two knobs only work as a PAIR, and the L1 limits are sharp — measured CB size for the down
   projection at tc=256 (`per_core_M=8, per_core_N=64`) against a 1.5 MB budget:

   | in0 dtype | `in0_block_w` | static CBs | |
   |---|--:|--:|---|
   | BFLOAT16 | 1 | 1.16 MB | fits (the old default) |
   | BFLOAT16 | 4 | 1.59 MB | TT_THROW |
   | BFP8 | 1 / 2 / 4 | 1.81 / 1.90 / 2.08 MB | TT_THROW |
   | BFP8 | 8 | fits | **the shipped pair** |

   Everything else must stay BFLOAT16: a block-float matmul *output* makes the op allocate a bf16
   intermediate CB on top of the packed output CB, and `ttnn.slice` over a block-float
   `[E, tc, 2*inter]` tensor overflows L1 outright. `MatmulMultiCoreReuseProgramConfig` also
   hard-requires `per_core_N == Nt`, so the output block can only be shrunk by halving the token chunk
   — which doubles the number of chunks and therefore doubles the 453 MB/layer weight read, costing
   more than it saves.

   Two guards, because L1 fit is **not monotonic in the chunk size** — measured, bf16 in0 with block 8
   fits at tc=128 but TT_THROWs at tc=200 and tc=256: (a) the fast pair is used only at
   `tc == _DENSE_TMAX`, the one shape it was validated at, which is every chunk of a prefill except a
   short tail; (b) a tri-state probe (`_FAST_DENSE_OK`) still runs that shape once on the live device
   and falls back to (BFLOAT16, 1) for the process if it does not fit. `QWEN36_MOE_ACT_BF16=1` also
   reverts `in0_block_w`, so the invalid pair cannot be selected by accident. Validated across
   T in {53, 128, 200, 256, 300, 512, 640} x {shipped, reverted, invalid} — no raise, and the fast
   path is retained at every T >= 256 including ragged ones.

   One bug this shook out, worth remembering: `ye` inherits in0's dtype, so it is BFP8 for full chunks
   and BFLOAT16 for a short tail, and `ttnn.concat` **rejects mixed dtypes** — every multi-chunk
   prefill with a tail (e.g. T=640) failed until the per-chunk reduce was pinned to BFLOAT16.
9. **Batched gated-delta chunk-prep** (`tt/gated_delta.py` `_forward_prefill_chunked`, default on;
   `QWEN36_DELTA_BATCH_PREP=0` reverts) — the per-chunk delta-rule prep (cumsum/decay/β-scaling/inverse
   T/w/kcd, ~25 ttnn ops) does NOT depend on the recurrent state S, so all Nc chunks' prep now runs in
   ONE batched set of ops over the stacked `[1, Nc·Vh, C, *]` axis (masks `[1,1,C,C]` broadcast over
   Nc·Vh unchanged) instead of Nc sequential rounds; the ttl `_chunk_state` kernel loop still runs
   per-chunk to carry S. **PCC-identical** (bit-identical to the per-chunk path, both 0.99966–0.99967 vs
   reference at Nc=2/3). Measured same-session 40L A/B: **eager seq-256 1391→1078 ms (−22%, 184→237
   tok/s)**; eager seq-128 ~neutral (fewer chunks); **traced ~neutral** (556 vs 558 ms — dispatch is
   already amortized by replay and the batching doesn't reduce device-kernel work). It is a host-dispatch
   win, so it helps the **eager and (eager) incremental/multi-turn** paths, not the traced replay.

---

## 4. Remaining levers (ranked)

> **MEASURED UPDATE (2026-08-20) — supersedes everything previously ranked here.** With
> `chunk_gated_delta_rule` on, **tracing the prefill is worth ~1.02x** (650 vs 663 ms at T=256; 1243 vs
> 1259 ms at T=512), so every "remove dispatch" lever below is dead: prefill is large-op and
> DRAM-bound, not dispatch-bound. Wiring the fused op into the traced path was also tried and **hangs**
> during capture — and eager+fused beats traced-unfused outright, so the traced prefill path is
> redundant rather than a gap to close. `demo/server.py` already prefers eager (`forward()`), with the
> traced path behind `QWEN36_SERVER_PREFILL_TRACE=1`; leave it there.
>
> Two earlier framings in §4b were also refuted: the bf16 ttl `chunk_state` kernel is **1.48x SLOWER**
> than the fp32/HiFi4 ttnn recurrence (and both are now obsolete — the native fused op supersedes
> them), and "MoE `repeat` is infeasible to remove" is true but was the wrong target; see §2.1.

### Ranked, after §3.7 and §3.8

1. **MoE, remaining ~1.3x to the bandwidth floor.** 99 GB per 256 tokens against a 432 GB/s peak is a
   ~230 ms floor; the stages run at 45-80% of peak, with `repeat` (194 GB/s, 45%) the worst. Nothing
   structural is left (§2.1) — this is pure per-op bandwidth tuning, and `repeat` is where to start.
2. **Gated-delta glue, now ~20% of prefill.** Occupancy-bound, not bandwidth-bound (~12% of DRAM
   peak), so the lever is fewer/larger ops: `delta.conv`, `in_proj`, `norm_out`. Same shape of win the
   fused op just delivered for the recurrence.
3. **~~`head` costs 15.5 ms~~ — DONE (2026-08-22).** `[1, 248320]` in TILE layout is physically
   `[32, 248320]` = 15.9 MB over PCIe for 0.5 MB of logits. `ttnn.to_layout(logits, ROW_MAJOR)` before
   `from_tt` in `_head` (and `_head_all`, which the `evaluation/` path uses, where TILE also pads S up
   to a multiple of 32). **Measured ~8.5 ms per call at every length** — see the §1 note. One line, no
   flag, no accuracy question.
4. **Bigger token chunks are NOT a lever** — measured 2.63 -> 2.50 -> 2.43 ms/token at T=256 -> 512 ->
   1024 (v0.77 defaults). Fixed per-call cost is already amortized by T=256; per-token cost is real
   work. Note `_DENSE_TMAX=256` chunking means a SMALLER chunk is strictly worse (each chunk re-reads
   all 453 MB of a layer's expert weights), so never lower it to buy L1 headroom.

### 4a. Traced incremental (multi-turn) prefill — the next structural feature
Eager incremental prefill (shipped, §3.5) already gives the big multi-turn win (don't re-prefill
history). **Tracing it** would add ~1.5× on the per-turn ingest, but is the **largest remaining effort
and the only cross-cutting one.**

What's already trace-ready: the **read** side — `chunked_scaled_dot_product_attention` with
`chunk_start_idx_tensor` (device offset P) is validated (PCC 0.99995) on the flat cache treated as a
single-block paged cache; gated-delta state-in is a buffer read; RoPE-at-P is a small device-table
gather.

**The hard piece — the traceable dynamic-P KV write.** Eager uses `fill_cache(update_idx=P)`, but
`update_idx` is a **static int** (baking P into the trace ⇒ re-capture per P, defeating tracing). No
flat-cache op writes M tokens at a *device-tensor* offset. The clean solution is a **real paged KV
cache**: `[num_blocks, n_kv, block_s, head_dim]` (block_s = chunk size), a **device page table** updated
per turn, new chunk written via `paged_fill_cache`. **This cache layout is shared with decode** (which
today uses the flat cache + `sdpa_decode`/`paged_update_cache` with `page_table=None`), so it either
touches the decode owner's code or needs a second cache layout — the real escalation.

Effort breakdown:
| piece | effort |
|---|---|
| RoPE-at-P device gather (extend decode's `_rope_for` to M positions) | ~0.5 day |
| **Paged KV cache + page-table management** (the dynamic-P write) | **~3–5 days, cross-cutting with decode** |
| Orchestration: bucket by M, device position P, capture/replay, page-table updates | ~2 days |
| Validation (traced == eager, multi-turn coherence, decode-after) | ~1 day |

**Total ~1–1.5 weeks + decode coordination.** ROI is modest (~1.5× on an already-small per-turn ingest).
**Recommendation: defer unless (a) multi-turn TTFT is a measured bottleneck, or (b) decode adopts a
paged cache anyway** (decode already uses `paged_update_cache` for O(1) long-context KV writes) — in
which case the paged cache is shared infrastructure and this drops to ~2–3 days.

### 4b. Cut the non-matmul overhead (the ~4× gap to roofline)
The 579→~135 ms gap is elementwise/layout/dispatch on non-matmul ops. Per-op profiling (`prof_prefill.py`)
itemizes it. Highest-value, all eager-and-traced wins:
- **Gated-delta chunk-prep.** Two sub-levers:
  - *Dispatch reduction (SHIPPED, §3.7).* Batching the prep across chunks (`QWEN36_DELTA_BATCH_PREP`) cut
    the per-chunk launches ~Nc× → **eager seq-256 −22%**, but **traced neutral** (replay already hides
    dispatch).
  - *Device-work reduction (the remaining TRACED lever, not yet done).* To move the **traced** number,
    the prep's actual compute (cumsum/decay/β-scaling/inverse) must be **folded into the ttl
    `_chunk_state` kernel** so it is one fused launch instead of ~25 ttnn ops — this reduces device-kernel
    time, which is what the traced path is bound by. This is the **largest remaining solo (no
    decode-overlap) traced-prefill lever**, but a sizeable ttl-kernel effort (cf. the tt-lang authoring
    lessons in `tt/ttl_delta.py`). `QWEN36_DELTA_BF16_PREP=1` (bf16 prep) is implemented but ~neutral alone.
- **MoE `repeat` (~69 ms/40L)** — the `[E,T,H]` activation broadcast. **MEASURED INFEASIBLE to remove**
  (2026-06-23): the only repeat-free option is in0-batch-broadcast, which requires switching the expert
  matmul from the grid-tuned `MatmulMultiCoreReuseProgramConfig` to
  `MatmulMultiCoreReuseMultiCast1DProgramConfig` (`mcast_in0=False`, `fuse_batch=False`). A standalone
  microbench (`scratchpad/mm_microbench.py`) at the real dims (E=256, M=256, BFP4) showed the 1D in0-reuse
  matmul is **5–6× SLOWER** than repeat+grid (gate_up 5.16→25.67 ms; down 2.35→14.49 ms; PCC OK) — the 1D
  config serializes the E=256 expert batch while the grid config parallelizes experts across all 110
  cores. The repeat's 256 MB write is far cheaper than that throughput loss, so the repeat **stays**. The
  per-op profile confirms `repeat` (3.16 ms/layer) ≈ the single largest MoE op, but it is irreducible
  here. `reduce` via `ttnn.experimental.fast_reduce_nc` was also benched — **neutral** (0.880 vs 0.889 ms
  vs `ttnn.sum`, ratio 0.99×), no win. **The dense MoE matmul path is already near-roofline; no clean
  MoE non-matmul win remains.**
- **`tilize`/layout churn (~64 ms)** — not isolated as a separable phase in the per-op profile; folded
  into the matmul/repeat device time above. No clean lever identified.

### 4c. Bucketing / attention at large T
- Finer bucket granularity (vs powers of 2) trades startup/trace memory for less padding waste.
- The from-scratch attention `sdpa(is_causal)` fit untuned at T=384; for very long from-scratch buckets,
  bound its L1 like decode's `k_chunk_size=128` if it overflows.

### 4d. Decode (other engineer)
Dispatch/latency-bound at ~33 ms/token, ~10× above its bandwidth floor; already traced. Wins are
op-fusion / fewer launches. Out of scope here.

---

## 5. How to measure

- **Prefill throughput:** `tests/bench_prefill.py` — warm eager vs traced tok/s + TTFT at given seq
  lengths. `QWEN36_LAYERS=40 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/bench_prefill.py --seqs 128,256`.
  Iterate at `QWEN36_LAYERS=4` (same op shapes, ~70 s load).
- **Per-op / device-vs-dispatch split:** `tests/prof_prefill.py` under the Tracy wrapper —
  `QWEN36_LAYERS=4 ./python_env/bin/python -m tracy -r -v --op-support-count 6000 models/demos/qwen3_6_a3b/tests/prof_prefill.py`
  (the `--op-support-count` is REQUIRED — the default drops ops and breaks the host↔device merge).
  Signpost regions: `prefill_eager_*`, `prefill_traced_*`. Load the `generated/profiler/reports/<ts>/`
  folder in ttnn-visualizer (point `--performance-path` at it). Cheap coarse split: env-gated
  device-synced phase timers, `QWEN36_PROFILE_PHASES=1` (`tt/prefill_profiler.py`).
- **Correctness gates:** `tests/test_moe.py`, `test_gated_delta.py`, `test_gated_delta_decode.py` (PCC
  vs reference; run after any change). Equivalence of traced/bucketed/incremental vs eager is checked in
  scratch tests (`test_prefill_trace.py`, `test_bucket.py`, `test_incremental.py`) — fold into `tests/`
  if a standing gate is wanted.
- **Decode:** the decode owner's `bench_decode.py`.

---

## 6. Notes / provenance

- Per-op profiling required `--op-support-count 6000` (default cap drops ops) and a single capture
  (the host `csvexport` segfaults on very large captures). The legacy device-only `OP TO OP LATENCY`
  column is unreliable in this build; use `DEVICE KERNEL DURATION` + op counts, or ttnn-visualizer.
- Roofline FLOP/s anchored on the measured tuned-matmul rate (~130 TFLOP/s BFP4/LoFi); refine with a
  device-spec peak if available.
- Fused-prefill design + ttl authoring constraints: the module docstring in `tt/ttl_delta.py`.
  Trace-safety findings + the incremental-prefill probe: memory note `qwen36-prefill-trace.md`.
