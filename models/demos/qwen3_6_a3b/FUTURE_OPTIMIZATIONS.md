# Qwen3.6-35B-A3B — Future Decode Optimizations

Measured landscape of remaining single-user **decode** perf levers, with the evidence behind each so a
future session doesn't have to re-derive it. All numbers are single P150 (Blackhole), 40 layers, warm
weight cache, traced decode, greedy. See also `DEBUGGING.md` (how to measure) and the memories
`qwen36-decode-bw-analysis`, `qwen36-fused-decode-kernel`, `qwen36-decode-opt-findings`.

## Baseline (2026-07-16) and current

Original baseline **27.7-28.2 tok/s/user (35.5-36.0 ms/token)**, measured by `tests/bench_decode.py`
(`QWEN36_LAYERS=40`). After the earlier DONE levers (input fusion + w_out/wo DRAM-sharding): 30.1 tok/s
(33.2 ms/token). After the 2026-07-16 session below (attn wq/wgate DRAM-shard + MoE `in0_block_w`=16):
31.4 tok/s/user (31.82 ms/token). **After the conv add-chain (Lever 3b, default on, 2026-07-16 s3):
34.6 tok/s/user (28.92 ms/token), +9.8% (+25% over the original 27.7)**, PCC-clean (module + trace tests
pass). **After the per-matmul MoE `in0_block_w` (`gate_up`=32): 34.9 tok/s (28.65 ms). After the router
rewrite (Lever 5, 2026-08-22): 37.4 tok/s/user (26.72 ms).** Per-layer-type breakdown (pre-add-chain
baseline; add-chain takes Gated-DeltaNet 0.451→0.323):

| Component      | ms/layer (orig→now) | ×count | share |
|----------------|---------------------|--------|-------|
| Gated-DeltaNet | 0.510 → 0.451       | ×30    | 43%   |
| MoE            | 0.348 → 0.321       | ×40    | 40%   |
| Attention      | 0.382 → 0.303       | ×10    | 10%   |
| lm_head        | 1.226 (unchanged)   | ×1     | 3.9%  |

The conv (~46% of the GDN step, ~6 ms/token) is now the single biggest untapped pool but is blocked —
see Lever 3b (attempted+reverted).

## The key measured fact: decode is OVERHEAD-bound, not DRAM-bandwidth-bound

Active weight bytes/token = **2.51 GB** (from `.tensorbin` sizes: 30× gdn proj bf8 35.8 MB + 10× attn
bf8 29 MB + 40× MoE [router+shared bf16 7.5 MB + 8/256 experts bf4 14.2 MB] + lm_head bf4 286 MB).
P150 DRAM peak = **432 GB/s** (8 controllers × 54 GB/s, from the tt-npe blackhole model; clock-independent).

- BW floor @432 GB/s = **5.8 ms/token (172 tok/s)**. We measure 35.5 ms → **71 GB/s effective = 16% DRAM
  utilization.** So aggregate bandwidth is NOT the wall.
- Per-component BW util: lm_head **54%** (genuinely BW-bound, already near-optimal — leave it), gated-delta
  **16%**, attention **18%**, MoE **14%**. The layers waste bandwidth on per-op dispatch, the MoE
  multicast scan, and non-matmul "glue" ops (norms, conv, ttl kernel, tilize/untilize, slices) that
  take time while moving almost no data.

Implication: the lever is **reducing op overhead and making each matmul read its weight efficiently**,
NOT "get weights off DRAM." (Moving to L1 is impossible anyway — 17.5 GB model.)

---

## Levers, ranked

### ✅ DONE — Lever 5: the decode router (2026-08-22, `QWEN36_MOE_ROUTER_FAST`, default on)
`moe.router` + `moe.scatter` were **16% of the decode step at ~3% of DRAM peak** — 0.091 + 0.023
ms/layer x 40 = ~4.55 ms/token for a 1 MB matmul and some 256-wide elementwise work. That is the
single least bandwidth-efficient block in the step, and it had never been looked at because the
per-component table in `CURRENT_BENCHMARK.md` §2 folds it into "MoE". Two changes, neither of which
alters what the block computes:

1. **top-k of the raw LOGITS + softmax over the k survivors**, instead of softmax over all E, then
   top-k, then sum+divide. `softmax_E(l)` -> top-k -> renormalize **is** softmax over those k logits,
   and top-k is order-preserving under softmax, so the selected set is identical too. 4 ops -> 2, and
   it is better conditioned in bf16 (one normalization over k, instead of one over E and another over
   the survivors).
2. **the T==1 gather path passes a cached CONSTANT `sparsity`.** `ttnn.sparse_matmul`'s indexed mode
   never reads that operand — verbatim from `matmul_nanobind.cpp:1075`: *"sparsity is still a required
   operand but is NOT read by the kernels in this mode (the indexed loop visits only active groups, so
   there is no per-slot validity scan or multicast)"*. `moe.py:_const_sparsity` (formerly
   `_verify_sparsity`) already exploited this on the verify path; decode was rebuilding it with
   `multiply(probs,0)` + `scatter` + `reshape` + `to_layout` every layer, every token, for a tensor
   the kernels ignore.

**MEASURED, paired A/B in one tree (`bench_decode.py --iters 200`, 40 layers, real weights):**

| arm | ms/token | tok/s/user | MoE ms/op |
|---|--:|--:|--:|
| `QWEN36_MOE_ROUTER_FAST=0` | 28.61 | 35.0 | 0.318 |
| `QWEN36_MOE_ROUTER_FAST=1` (default) | **26.72** | **37.4** | 0.271 |

−1.89 ms/token, **+6.9%**. The MoE delta (−0.047 ms/layer x 40 = −1.88) accounts for the entire
full-model delta, and the control arm reproduces the historical 28.65 ms / 34.9 tok/s to 0.1% — an
independent check that the harness and the baseline agree. Correctness: `test_moe`, `test_moe_sparse`,
`test_moe_verify` (14 tests) green on both arms.

**Estimate vs result, recorded because it is the pattern:** predicted −3 to −3.9 ms, delivered −1.89.
Fifth time in this project an isolated estimate over-predicted a 40-layer result.

### ✅ DONE — Lever 3: fuse the gated-delta input projections
`in_proj_qkv` + `in_proj_z` + `in_proj_b` + `in_proj_a` all consume the same `x` → one
`[hidden, conv_dim+value_dim+2V]` matmul, sliced in `forward`. One dispatch + weight read instead of
three. `QWEN36_GDN_FUSE_IN=1` (default). PCC-clean. See `tt/gated_delta.py` (`w_in_proj`, `_in_off`).

### ✅ DONE — Lever 1: DRAM-shard the gated-delta output projection `w_out`
At T=1 the `[value_dim, hidden]` matmul reads interleaved DRAM at only ~25% BW efficiency. Width-shard
the weight across the **8 physical DRAM banks** + L1-width-shard the activation + sharded output runs it
**2.67× faster including the in/out reshards** (measured; reshards cost only ~4 µs). `QWEN36_GDN_DRAM_SHARD=1`
(default). See `tt/gated_delta.py` (`_build_dram_shard`, `_forward_decode_fused`).

**Core-count sweep (weight always across the 8 DRAM banks; vary compute/worker cores), realistic net incl.
reshards:**

| projection | 8c | 16c | 32c | 64c |
|---|---|---|---|---|
| w_out (K4096,N2048) | **2.67×** | 2.66× | 2.21× | 1.40× |
| w_z (K2048,N4096)   | **1.51×** | 1.50× | 1.26× | 0.82× |
| w_qkv (K2048,N8192) | 1.14×     | 1.17× | 1.11× | 0.75× |

→ **8 (or 16) workers is optimal; it degrades monotonically past that.** At M=1 there is no compute to
parallelize, so more cores only add multicast/dispatch overhead. K-heavy / narrow-N output projections
(`w_out`, attn `wo`) win most (2.67×); wide-N inputs (`w_qkv`, the fused `w_in_proj` N=12352) barely move
(≤1.17×) — so the input projection is deliberately left fused+interleaved, only the output is sharded.
**lm_head cannot use this pattern** (its 248K-wide output OOMs L1-sharded, `bank_manager` fatal) and is
already 54% BW-bound — leave it.

### 📏 Lever 1 AMENDED (2026-08-22) — the sweep this doc never ran: wide 1D mcast on INTERLEAVED weights
The core-count sweep above varied the **DRAM-sharded** core count (8/16/32/64) and correctly found 8
best. It never tried a **tuned 1D `mcast_in0` config on the plain interleaved weight** — which is what
two independent single-P150 implementations ship on this class of skinny (M=1) matmul
(`models/demos/blackhole/qwen36/tt/model_config.py:179-230`; MuseGlimmer's decode `down_proj` at grid
11x10 with `in0_block_w=4`). Now measured, `tests/bench_matmul_layout.py` (traced, 200-500 replays,
real shapes), sweeping `num_cores` x `in0_block_w`:

| case | A: interleaved+auto (today, untuned) | C_net: DRAM-shard **incl. reshards** | D: wide 1D mcast | D best (cores, bw) |
|---|--:|--:|--:|---|
| `gdn.w_in_proj` (fused, 30x) | 83.9 | n/a | **78.1** | (33, 1) |
| `gdn.w_out` (30x) | 82.3 | 30.9 | **30.1** | (22, 16) |
| `attn.wq` (10x) | 47.7 | 31.9 | **31.2** | (22, 8) |
| `attn.wgate` (10x) | 47.9 | 31.9 | **31.1** | (22, 8) |
| `attn.wo` (10x) | 82.5 | 30.9 | **30.4** | (22, 16) |
| `attn.wk` / `wv` (10x each) | 34.4 | 17.8 | **12.7** | (8, 16) |
| `moe.se_gate_up` (40x) | 38.0 | 19.9 | **18.4** | (22, 8) |
| `moe.se_down` (40x) | 19.9 | 17.3 | **12.8** | (22, 4) |
| `moe.gate_w` (40x) | 29.7 | 17.4 | **12.1** | (66, 16) |
| `lm_head` bf4 (1x) | 1125.9 | n/a | **1035.7** | (110, 4) |

Three results, and the middle one is the reason the original sweep reached a different answer:

1. **On the raw matmul alone, the shipped DRAM-sharded config (bare `C`) still wins** by 8-11% — e.g.
   `w_out` 27.3 vs 30.1 us. So Lever 1's conclusion was right *as measured*.
2. **Once you count the reshards the model actually pays, D wins or ties everywhere.** Every
   DRAM-sharded matmul does `to_memory_config` in AND out (~110 of them per decode step); `C_net`
   above times the call the way the model issues it, and D beats it on 9 of 10 cases. D also needs no
   **second copy of the weight** — `build_dram_shard` leaves the interleaved original alive for
   prefill, ~510 MB across the model. **Lesson: a program-config microbench must include the layout
   conversions its config forces on the caller, or it will pick the wrong config.**
3. **The big prize is the matmuls nobody ever tuned.** `attn.wk`/`wv`, `se_gate_up`, `se_down`,
   `gate_w` and `lm_head` run on the ttnn AUTO config today (column A) and are 1.1-2.7x off D.

Projected saving vs what ships today, per token at 40 layers:
`w_in_proj` −0.17 ms, `wk`+`wv` −0.43, `se_gate_up` −0.78, `se_down` −0.28, `gate_w` −0.70,
`lm_head` −0.09, already-sharded projections ~−0.03 → **≈ −2.5 ms/token (26.72 → ~24.2 ms, ~41 tok/s)**,
plus ~510 MB of DRAM reclaimed. Halve it on this project's track record and it is still ~1.5 ms.
Builder: `tt/common.py:wide_1d_decode_pc(m, k, n, num_cores, grid_w=11, in0_block_w=...)`.

#### ✅ SHIPPED (`QWEN36_WIDE1D_DECODE`, default on) — measured 37.4 → 39.2 tok/s/user
Paired A/B, `bench_decode.py --iters 200`, 40 layers, real weights:

| arm | ms/token | tok/s/user | GDN | attn | MoE | lm_head |
|---|--:|--:|--:|--:|--:|--:|
| `QWEN36_WIDE1D_DECODE=0` | 26.74 | 37.4 | 0.319 | 0.295 | 0.270 | 1.225 |
| default | **25.49** | **39.2** | 0.316 | **0.254** | **0.252** | 1.226 |

−1.25 ms/token, **+4.8%**. Attribution: `wk`/`wv` −0.41 ms, `se_gate_up`+`se_down` −0.72,
`w_in_proj` −0.09, `lm_head` **0.00** — the components sum to −1.22 against a measured −1.25.
Cumulative with Lever 5: **34.9 → 39.2 tok/s/user (+12.3%)**.

Three things this cost, all worth keeping:

1. **`gate_w` is DEFAULT-OFF, and it was the largest single win (−0.70 ms).** It is the one site whose
   matmul feeds a `topk`, so a low-bit shift can change *which* expert is 8th. Bisected per site with
   `QWEN36_WIDE1D_SITES` against `test_moe_sparse.py::test_moe_gather_decode` (gate 0.99): every other
   site passes, `gate_w` alone fails at **PCC 0.9857**. Re-enable with `QWEN36_WIDE1D_SITES=all` once
   `evaluation/published_scores.json` has an MMLU baseline — that is this project's rule for anything
   that changes what the model computes. Cheaper first try: sweep `gate_w`'s `in0_block_w` for a value
   that keeps most of the 2.45x with an accumulation order closer to the heuristic's.
2. **"Program-config only, so it cannot change numerics" is FALSE** and the comments that said so are
   corrected. `in0_block_w` is the K-block size, so it sets the matmul's accumulation ORDER; at bf16/bf8
   that moves the low bits. Measured on the MoE block: expert *selection* (`topi`) is bit-identical
   across every subset, routing weights move <=0.005 and the shared expert <=0.006 — rounding, not
   routing — but "rounding" was still enough to fail a 0.99 gate at `gate_w`.
3. **`lm_head` is the 6th microbench in this file to over-predict.** Isolated it went 1125.9 -> 1035.7 us
   (−90 us); in-model it is 1.226 -> 1.226 ms, i.e. exactly zero. Note the first attempt to measure this
   was itself wrong: `bench_decode.py`'s lm_head component called `ttnn.linear(x_h, model.lm_head_w)`
   directly and so was blind to the model's own config. It now calls `model._lmh`, and the answer did
   not change. **If a component row does not move when the full-model number does, check that the
   component harness runs the same code the model runs.**

`in0_block_w` is the knob that had never been touched on this path, and it matters most where K is
large (16 for `w_out`/`wo`/`wk`, 8 for `wq`/`se_gate_up`, 4 for `lm_head`/`se_down`). **It is an L1
bound as well as a legality bound** — it sizes the in0 circular buffer, which competes with any
resident L1 output for the same 1536 KB — so every winner above must be re-A/B'd at 40 layers before
shipping (`blackhole/qwen36/tt/tp_common.py:51-60` documents a model that overflowed exactly this way).

Also fixed in the harness while doing this: it applied BFP8 bytes (1.0625 B/elem) to **every** case, so
the bf16 rows understated their floor 1.88x and the bf4 `lm_head` row overstated it 1.89x — the latter
printed ">100% of DRAM peak", which briefly looked like evidence that the 432 GB/s figure was wrong. It
was the instrument. Any utilization number from an older run of that script is wrong off-bf8.

### ✅/⏭ Lever 1b — extend DRAM-sharding to attention + MoE projections
**DONE:** attention `wo` (`tt/attention.py`, `QWEN36_ATTN_DRAM_SHARD=1`) — attention 0.382→0.332 ms/layer.
**DONE (2026-07-16):** attention `wq`/`wgate` (`QWEN36_ATTN_DRAM_SHARD_IN=1`, default on) — microbench
1.71–1.73×; at 40L attention 0.332→**0.303** ms/layer (−0.29 ms/token). Both wq/wgate share one shard
config (identical `[hidden, n_heads*head_dim]`); the decode branch in `_qkv` reshards `x` once, runs both.
**MEASURED WASH (not shipped, `QWEN36_MOE_SE_DRAM_SHARD` default OFF):** MoE shared-expert `se_down`
(K=512,N=2048) microbenched 1.46× but at 40L netted ~0 — its ~6 µs matmul saving is eaten by the ~8 µs
reshard in/out (K is too small for the fixed reshard cost to amortize). Code kept, opt-in. `wk`/`wv` too small.
Shared helper `common.build_dram_shard(w, K, N)` (returns w_dram, act_mc, out_mc, program_config).

### ❌ Lever 2c (2026-08-22) — `moe_compute` FullLocal: measured, and it is 14x WORSE for decode
Lever 2b concluded that the expert `sparse_matmul`s are core-count-limited and that "the only remaining
lever is a different op that shards experts across the whole grid", i.e. `ttnn.experimental.moe_compute`
in its single-card FullLocal mode. Measured. It does not work here, for a structural reason.

**`moe_compute` streams EVERY expert.** Established by sweeping the expert count
(`QWEN36_MOE_AB_EXPERTS`, new in `tests/bench_moe_ab.py`), FullLocal leg, traced ms/call:

| E | T=32 | T=64 | T=256 |
|--:|--:|--:|--:|
| 32 | 0.470 | 0.478 | 1.170 |
| 64 | 0.919 | 0.919 | 1.507 |
| ratio | **1.96x** | 1.92x | 1.29x |

2x the experts, 2x the time: **linear in E**, so no inactive expert is skipped. Fitting
`t ~= a*E + b*T` gives a = 0.0140 ms/expert, and extrapolating to the real E=256:

| | moe_compute FullLocal @ E=256 | current path | verdict |
|---|--:|--:|---|
| decode (T=1, padded to a 32-tile) | ~3.6 ms/layer | **~0.25 ms/layer** (sparse gather) | **14x WORSE** |
| prefill (T=256) | ~4.2 ms/layer | 5.62 ms/layer (dense) | 1.34x better |

**Decode is arithmetic, not tuning.** The sparse gather path reads 8 of 256 experts = 14.16 MB/layer;
`moe_compute` reads all 256 = 453 MB/layer. That is 32x the bytes to buy at most 110/32 = 3.4x the
cores. No program config can close a 9x gap. **So the decode MoE is DONE**: the gather formulation is
structurally correct, its `in0_block_w` is already optimal (Lever 2b), and its 38%/31% of DRAM peak is
a core-count wall with no legal alternative. Further decode MoE gains need a new op, not a new config.

**Prefill is real but the wrong trade.** 1.34x on the block, because it streams the same 453 MB of
weights the dense path already reads — the saving is only the intermediates. Against that: a new weight
layout, and the host packer needs **~0.034 GB/expert = ~10-12 GB peak at E=256** (measured,
`tests/probe_moe_compute_packmem.py`), which OOM-kills on a 15 GB box. The flattened-dense +
`minimal_matmul(fuse_swiglu)` reformulation (`PREFILL.md`, projected ~1.75x on the same block) needs no
new op, no new memory, and is strictly the better bet.

**Process note.** This bench was OOM-killed three times (rc=137) before producing a number, and rc=137
is silent: the log simply stops mid-build, which reads exactly like a device hang. Two things now guard
it -- `--legs {all,current,moe_compute}` so the two sides' host peaks never add, and a `peak RSS` line
printed around each packer. **If a bench dies with no output and no traceback, check the exit code for
137 before suspecting the device.**

### ❌ Lever 2b (2026-08-22) — re-swept the expert `sparse_matmul` `in0_block_w`: the 2026-07 tuning was already optimal
The 2026-08-22 dense sweep found a SECOND mechanism for this same knob — DRAM *reader*
under-utilisation on Blackhole, where the win comes from fewer/larger contiguous reads per reader
rather than from fewer multicast handshakes (MuseGlimmer: "four K tiles nearly halves this
projection's device time"). Lever 2 was tuned on the handshake theory alone, so it was worth
re-sweeping on the new one. Measured (`tests/probe_sparse_in0bw.py`, traced, real dims, BFP4, top_k=8):

| in0_block_w | gate_up (K=2048, N=1024, 32 cores) | down (K=512, N=2048, 64 cores) |
|--:|--:|--:|
| 1 | 451.7 us (4.8% of peak) | 154.0 us (7.1%) |
| 2 | 244.8 (8.9%) | 86.7 (12.6%) |
| 4 | 141.7 (15.4%) | 54.1 (20.2%) |
| 8 | 90.8 (24.0%) | 38.0 (28.7%) |
| 16 | 66.0 (33.1%) | **35.6 (30.6%) <- default, and Kt=16 is the ceiling** |
| 32 | **56.9 (38.4%) <- default** | — |
| 64 | 57.2 (38.2%) | — |

**No change available: the shipping values are the optima.** The mechanism is confirmed — the curve is
strongly monotone, an 8x span from `in0_block_w` 1 to 32 — but the knob is exhausted: gate_up saturates
at 32 (64 is a wash), and `down` cannot exceed 16 because its `Kt` IS 16.

**The real finding is what the ceiling is.** At its best config gate_up reaches 38% of DRAM peak and
`down` 31%, and `_sparse_pc` puts them on **32 and 64 cores of 110** (the config sets `per_core_N=1`,
so the grid is pinned to `Nt`). 32/110 = 29%, 64/110 = 58%: the achieved fractions track the CORE COUNT,
not the block size. So these two matmuls are **core-count-limited**, which is the same wall that made
the union-indexed prefill variant 0.44-0.50x (`PREFILL.md` §2.1). There is no legal `_sparse_pc` with
more cores, and neither `Nt` (fixed by N) nor `Kt` (fixed by the intermediate size) can grow.

**Consequence: the only remaining lever on the expert matmuls is a DIFFERENT OP** — one that shards the
expert weights across the whole grid. That is exactly what `ttnn.experimental.moe_compute` does
(ring-sharded weights, ring size auto-detected from the DRAM-bank count), which promotes the FullLocal
A/B from "nice to have" to the single gating experiment for the 40% of decode this block costs.

### ✅/⏭ Lever 2 — MoE `sparse_matmul` overhead
The 8-of-256 gather already shipped (74→43 ms/token) and the old "256-slot scan"
premise was measured-false; the real cost is the **per-K-block multicast handshake**, tuned by
`in0_block_w` (`_sparse_pc`, env `QWEN36_SPARSE_IN0BW`).
**DONE (2026-07-16):** raised the default `in0_block_w` 8→**16** (the only higher value dividing both
expert matmuls' Kt: gate_up Kt=64, down Kt=16). Measured 40L: MoE 0.348→**0.321** ms/op (−1.1 ms/token),
PCC-identical (program-config only; `test_moe`/`test_moe_sparse` pass). 16 is the ceiling for this knob.
Any further MoE win needs deep C++ kernel work (reduce the per-K-block handshake itself) — not pursued.

### ✅ Lever 3b (RESOLVED a different way) — decode conv add-chain over separate history rows (`QWEN36_CONV_ADDCHAIN`)
Profiling the reverted fold (below) with a leave-one-out attribution inside the full gated-delta step showed
the decode conv's ~0.177 ms/layer is **87% two dim-0 (row) data-movement ops** — `concat([conv_state,mixed])`
(0.075) + the `xpad[1:K]` state slice (0.079) — and only ~15% the multiply→sum→silu math. Those two ops are
**dispatch-bound** (conv_state L1-vs-DRAM makes no difference) and **cannot be removed bit-identically**
(ttnn `slice`/`concat` have no `output_tensor=`, tiled tensors can't be sub-region-written, and the fp32
`sum`-over-K needs the stacked rows the concat builds). Fix: represent the conv history as **K-1 separate
`[1,conv_dim]` row buffers** (`cache["conv_rows"]`) and compute the decode conv as a **bf16 add-chain**
(4 mul + 3 add over `[1,conv_dim]`) with a **3-cheap-copy window shift** — removing both dim-0 ops. Prefill
path is unchanged (still the concat/stacked `_conv_silu`); `conv_rows` is seeded from `conv_state` once at
`start_decode`. **Measured 40L: Gated-DeltaNet 0.451→0.323 ms/op, decode 31.5→34.6 tok/s/user (+9.8%,
31.79→28.92 ms/token).** NOT bit-identical (bf16 add-chain vs fp32 row-sum) but validated: `test_gated_delta_decode`
0.99979 (== baseline), `test_trace`/`test_model` pass, and a teacher-forced decode logit-PCC vs baseline
(40L, 96 steps) = **mean 0.996, stable/no accumulation** — the codebase's accepted chunked-vs-single regime,
NOT the reverted kernel's 0.885→0.37. (MMLU-Redux can't gate this — it's prefill-scored; the teacher-forced
decode logit-PCC is the right generative gate.) Code: `tt/gated_delta.py` (`_CONV_ADDCHAIN`,
`_conv_addchain_decode`, `sync_conv_rows`), `tt/model.py` (`_alloc_cache` conv_rows, `start_decode` sync,
`_state_tensors` flatten). **Default ON** (`QWEN36_CONV_ADDCHAIN=0` reverts).

### ❌ Lever 3b (original attempt) — fold the depthwise conv into the ttl decode kernel (ATTEMPTED, REVERTED 2026-07-16)
Measured the prize first (traced slope): the T=1 conv (concat+multiply-by-taps+row-sum+silu, ~5 ttnn ops
over `conv_dim`=8192 = 256 tiles) costs **0.207 ms/layer = ~6.2 ms/token (~19% of decode)** — far above the
old "<0.5 ms" guess. Implemented a **separate** on-core conv ttl kernel (`conv_step_tt`; depthwise → one
core per tile-column slab; taps pre-split `taps_prev` zero-padded + `tap_cur`; window row-sum via an
on-core ones-tile matmul; state-shift kept in Python). **Two independent failures killed it:**
1. **Accuracy:** the kernel is numerically correct (conv PCC 0.99999 vs `_conv_silu`, no bias) but
   `test_gated_delta_decode` drops to **0.885** (baseline 0.9998). Localized to the q/k regions: the
   decode kernel's `l2norm(q/k)` amplifies the unavoidable bf16 rounding-order difference (matmul row-sum
   vs `ttnn.sum`); the recurrence tolerates 0.0007 *random* noise (0.998) but not this structured
   rounding. Real 4L greedy tokens diverge after token 1. fp32 DST didn't help. (Decode here is at the
   edge of bf16 determinism — even chunked-vs-single prefill flips argmax.)
2. **Perf:** at 40L, GDN only 0.451→0.437 ms/op — **no net win**. A *separate* kernel launch costs ≈ the
   5 ttnn ops it replaced; the 0.207 ms slope over-counted because those ops pipeline with neighbors in
   the full step while a standalone launch doesn't.
**The only path to the ~6 ms prize is folding the conv INTO `decode_step_tt`** (no separate launch) —
blocked by the **32-DFB budget** on the already-dense recurrence kernel (29/32; needs aggressive buffer
reuse). NOTE: folding in-kernel fixes only the *perf* half; an in-kernel conv still rounds differently
from ttnn's `_conv_silu`, so the §1 accuracy divergence persists — the real gate must be a quality eval
(MMLU-Redux, `evaluation/`), not decode token-identity (this model's decode is at the bf16-determinism
edge). Deferred as higher-effort kernel work.

### ⏭⏭ Lever 1c (BIG, deferred) — ring matmul + DRAM prefetcher (`matmul_1d_ring_config`)
tt_transformers' highest-performance decode matmul family (`model_config.py: matmul_1d_ring_config`,
used by Llama on Blackhole/TG). Two ideas:

1. **Ring / gather-in0** (`gather_in0=True, mcast_in0=False`, `hop_cores`): instead of multicasting the
   activation to every core (the overhead that made 32/64-core DRAM-sharded matmuls *slower* above), each
   core holds an activation slice and passes it systolically around a ring. Trades multicast NoC
   congestion for cheap neighbor hops, so it scales to many cores where the mcast config collapses.
2. **DRAM prefetcher + global circular buffer** (`prefetch=True`, `num_global_cb_receivers`): a dedicated
   prefetcher *sub-device* streams the *next* matmul's weights from DRAM into a shared L1 global CB,
   overlapping weight reads with the current matmul's compute. Across a decode step this hides DRAM
   latency and is what pushes decode toward the true BW floor.

**Why deferred:** it needs a prefetcher sub-device, a global CB, and every weight registered with the
prefetcher, and it only pays off when *most* matmuls in the step route through one ring (amortizing setup).
Here the matmul chain is broken up by the gated-delta ttl kernel, the conv, and the MoE `sparse_matmul`,
so there is no clean ring to prefetch across. Its unique advantage over the 8-bank DRAM-sharded config is
the *cross-matmul prefetch overlap*, which only materializes after a sharded-activation-stream refactor
(below). Per-op at M=1, the 8-bank config is within a few % of what the ring would get.

### ⏭⏭ Enabler — sharded-activation-stream refactor
The DRAM-sharded (and ring) matmuls want their input already L1-width-sharded. Today each sharded matmul
pays ~4 µs to reshard in + ~4 µs out (still a net win, hence piecemeal is fine). The bigger prize is to
keep the whole decode activation stream width-sharded end-to-end (RMSNorm emits sharded, matmuls consume/
produce sharded, residual adds stay sharded — the tt_transformers decode design), which (a) removes all
per-op reshards and the tilize/untilize churn seen in the trace, and (b) is the precondition for the ring
prefetcher. Blocked by the non-matmul ops in the gated-delta path (conv, the `decode_step_tt` ttl kernel,
l2norm, gated RMSNorm) that currently expect interleaved tensors — each would need a sharded-friendly
variant. Large, higher-risk refactor of a working, coherent model; do it only when chasing the last ~2×.

### 📏 Lever 4 MEASURED (2026-08-22) — precision is worth +2.6%, and my re-rating of it was ALSO wrong
Lever 4 below called precision "small now (overhead-bound)". Earlier today I re-rated it UP, arguing
that the gated-delta projections run at 74-87% of DRAM peak so their time is proportional to bytes and
bf8 -> bf4 would nearly halve them (-1.5 to -2.0 ms), and that the bf16 router/shared expert was
"12% of the decode weight budget" (-1.0 to -1.5 ms). **Both estimates were too high, by 4-5x.**
Measured, `bench_decode.py --iters 200`, 40 layers, cumulative arms:

| arm | ms/token | tok/s/user | GDN | attn | MoE |
|---|--:|--:|--:|--:|--:|
| defaults (gdn bf8, attn bf8, shared bf16) | 25.49 | 39.2 | 0.316 | 0.254 | 0.252 |
| `QWEN36_SHARED_DTYPE=bf8` | 25.43 | 39.3 | 0.316 | 0.254 | 0.251 |
| + `QWEN36_GDN_DTYPE=bf4` | 24.97 | 40.0 | **0.301** | 0.254 | 0.251 |
| + `QWEN36_ATTN_DTYPE=bf4` | **24.88** | **40.2** | 0.301 | **0.245** | 0.251 |

Per-change attribution:
- **router + shared expert, bf16 -> bf8: −0.04 ms. Nothing.** 138 MB/token of weight traffic removed for
  no time at all. The reason was in my own table and I didn't apply it: `moe.shared` runs at **21%** of
  DRAM peak and `moe.router` at **3%**. Bytes are not their binding constraint, so removing bytes does
  not buy time. "It is 12% of the weight budget" is not an argument for anything on its own.
- **gated-delta projections, bf8 -> bf4: −0.45 ms** (0.316 -> 0.301 ms/layer x 30). Real, and the
  largest of the three — but only **32%** of the 1.5 ms that 84%-of-peak implies. Halving 35.8 MB/layer
  to 19.0 MB should have saved 0.047 ms/layer at 360 GB/s; it saved 0.015.
- **attention projections, bf8 -> bf4: −0.09 ms.** Same shape, smaller pool.

**The rule this establishes, which is more useful than the 2.6%:** "near-roofline in bytes" licenses at
most the byte saving, and only until the NEXT ceiling binds. Halving the dtype slides you down the byte
axis; here it hit the compute/issue ceiling almost immediately, because a LoFi matmul costs one pass
per tile whatever the tile's dtype, so bf4 has the same tile count as bf8 with half the bytes. A
component at 84% of DRAM peak is therefore not "84% byte-bound" -- it is at a point where bytes and
issue rate are comparable, and only the first ~third of a byte reduction converts to time.

**Corollary for the remaining plan:** every precision idea in this file should be re-rated DOWN, and
the "all-bf4 asymptote ~341 tok/s" figure below is meaningless -- it assumes bytes are the only ceiling.

**NOT SHIPPED. Default stays bf8, because bf4 breaks a correctness gate for its +2.6%:**

    test_long_prefill.py::test_long_prefill_chunk_invariant   bf4 FAIL / bf8 PASS
    (chunked prefill == single-shot prefill; isolated by flipping only QWEN36_GDN_DTYPE/ATTN_DTYPE)

The accuracy evidence was *good* -- paired MMLU-Redux 79.0% -> 80.5% (nominally up; McNemar p=0.508)
and the decode side clean -- which is exactly why the module gates earn their keep: **a 200-question
prefill-scored eval cannot see a chunk-boundary invariant.** Two instruments were not enough; the third
caught it. Opt in with `QWEN36_GDN_DTYPE=bf4 QWEN36_ATTN_DTYPE=bf4` for -0.54 ms/token and ~0.68 GB of
device DRAM if you can accept that gate red.

Kept from the exercise regardless: `QWEN36_ATTN_DTYPE` / `QWEN36_GDN_DTYPE` / `QWEN36_SHARED_DTYPE` now
exist as first-class knobs (they were hardcoded), and `ModelArgs.mlp_weight_dtype` -- assigned and never
read since the model was written -- is superseded by `moe_shared_weight_dtype`, which `decoder.py` and
`mtp.py` actually consume.

### 📏 Lever 6 MEASURED (2026-08-23) — KV cache precision, the lever this doc never had a term for
Every lever above is ranked at position ~0, where the KV cache is empty. That makes this whole document
a budget for the *weight-bound* part of the step, and it omits the one term that grows with the
conversation: SDPA re-reads the entire cache every step, **20 KiB/token** across the 10 attention layers
at bf16. `QWEN36_KV_DTYPE=bf8` halves it (10.9 KiB/token).

Measured in isolation (`tests/probe_paged_kv_perf.py`, per attention layer), and end-to-end
(`bench_decode.py --iters 100 --seq 4096`, 40 layers, `QWEN36_MAX_SEQ=8192`):

| pos | sdpa-decode bf16 | BFP8 | ratio | implied Δ/token (×10 layers) |
|---|--:|--:|--:|--:|
| 1 024 | 36.2 µs | 33.8 µs | 1.07× | −0.02 ms |
| 8 192 | 83.2 µs | 62.4 µs | 1.33× | −0.21 ms |
| 32 768 | 227.7 µs | 137.4 µs | 1.66× | −0.90 ms |
| 131 072 | 796.7 µs | 438.8 µs | 1.82× | −3.58 ms |

| arm (end-to-end, pos ≈ 4096) | ms/token | tok/s/user |
|---|--:|--:|
| `QWEN36_KV_DTYPE=bf16` (default) | 25.69 | 38.9 |
| `QWEN36_KV_DTYPE=bf8` | **25.59** | **39.1** |

**Read the two tables together — this is the point.** End-to-end at 4K the change is −0.10 ms, i.e.
inside noise, and a reader who only ran `bench_decode.py` would file this as another wash. It is not: at
4K the KV term is only ~0.2 ms of a 25.7 ms step, so there is nothing to win *yet*. The isolated table
is what says where the win lives, and the ratio climbing 1.07 → 1.82 with position (96.5% of the 1.88×
byte-ratio asymptote at 128K) says the kernel is doing nothing but moving the cache, so halving the bytes
is the only available lever and it converts almost perfectly. **Crossover is around 8K.**

This is the mirror image of the lesson the rest of this file teaches. Everywhere else, isolated
microbenches *over*-predicted the 40-layer result by 3–5×. Here the end-to-end benchmark *under*-predicts
the deployed result, because the benchmark's context is not the deployment's context. Both failures have
the same cause: measuring in a regime the change does not target.

Accuracy: teacher-forced decode logit PCC mean 0.995 / median 0.996 over 48 steps, **stable** (second
half marginally better than the first), 44/48 argmax agreement with all four flips on near-ties — on par
with the shipped conv add-chain's own 0.996 gate. Note that **MMLU cannot gate this at all**: single-shot
prefill runs SDPA on live q/k/v and only *fills* the cache, so a prefill-scored eval never reads it.
Full detail, plus the non-obvious dtype contract of the cache-write ops, in EXPERIMENTS.md
§"BFP8 KV cache".

Not the default, on the same principle as `QWEN36_SHARED_DTYPE`: it buys nothing at the contexts this
file benchmarks, so the default does not spend the perturbation. **It is the right setting for
long-context serving** — where it is also worth 2.5 GB of DRAM at `max_seq=262144` (5.23 → 2.78 GB).

### ⏭ Lever 4 (cheap, deferred) — precision: lower the byte floor toward BFP4
The one genuinely bandwidth-bound component, lm_head, already runs BFP4 (`QWEN36_LMHEAD_BF4`,
greedy-identical, ~0.7 ms saved) — that was its only lever. The remaining cheap tier: push the
gated-delta projections (~1 GB bf8) and the bf16 shared expert toward BFP4 where PCC holds. Each is a
one-line dtype change + the decode test suite, accuracy-bounded and per-module PCC-gated (opt-in flags
like `QWEN36_EXPERT_DOWN_BF8` already exist for this kind of experiment; gate quality with MMLU-Redux
in `evaluation/`, not decode token-identity). **CORRECTION (2026-08-22).** "Because decode is overhead-bound not BW-bound, this shrinks the
asymptote more than it moves the current number" is wrong for the two projections that matter. Against
the byte table in `ROOFLINE.md` §2, GDN `w_in_proj` runs at **84%** of DRAM peak and `w_out` at **87%**
— they are near-roofline, so their time is *directly proportional to bytes* and bf8→bf4 nearly halves
them (−1.5 to −2.0 ms/token if PCC holds). The aggregate "16% DRAM utilization" figure is an average
over a step whose worst components sit at 3–6%; it says nothing about the components that are already
efficient. The shared expert + router are a separate, larger case: they are **bf16** (294 MB/token,
12% of the decode weight budget) purely because `ModelArgs.mlp_weight_dtype` is set and never read
(`model_config.py:110` vs `decoder.py:57`). Both are accuracy-gated on MMLU-Redux, not on decode
token-identity.

### 📏 MEASURED (2026-08-23) — the B>1 cliff, and why idle slots are not free
`tests/bench_batched_decode.py`, 40 layers, traced, scan GDN:

| B | step ms | per-user tok/s | aggregate tok/s | vs B=1 |
|--:|--:|--:|--:|--:|
| 1 | 25.27 | **39.6** | 39.6 | 1.00x |
| 2 | 137.17 | 7.3 | 14.6 | **0.37x** |
| 4 | 140.64 | 7.1 | 28.4 | **0.72x** |

Two facts worth having in one place, because they decide `--max-num-seqs`:

- **The cliff is at B=1 -> B=2, not at large B.** The step jumps 5.4x, then barely moves from B=2 to
  B=4 (137 -> 141 ms). That is the MoE switching from sparse-gather (top-8, 566 MB/step) to **dense-256**
  (453 MB/layer x 40 = ~18 GB/step) -- a large FIXED cost that amortizes over users. Aggregate therefore
  keeps climbing (28.4 at B=4, ~157 at B=32) and crosses single-user around **B ~ 6**.
- **Idle slots cost full price.** The decode graph runs at POOL width, so a 4-wide pool serving one
  request pays the 4-wide step: ~7 tok/s instead of ~40. Batching is not a free option you can leave
  switched on.

The lever this points at: a **sparse-gather MoE that works at B>1** would remove the cliff entirely and
make small-B batching worthwhile. Today the union of B users' top-8 sets is what makes gather unattractive
(`PREFILL.md` §2.1(2) measured the union approach at 0.44-0.50x), but that was measured for PREFILL token
counts, not for B<=8 decode where the union is at most 8B of 256 experts. Unmeasured, and the highest-value
open question for multi-stream serving. A cheaper partial win: capture decode traces at several widths and
dispatch on the ACTIVE count, so an idle pool costs its active width rather than its configured width.

## Rough headroom summary

| Lever | Effort | Est. decode gain |
|---|---|---|
| 3 input fusion (DONE) | done | delivered |
| 1 gdn w_out shard (DONE) | done | delivered |
| 1b attn wo shard (DONE) | done | delivered |
| — above three together: 27.7 → 30.1 tok/s (+8.7%) | | |
| 1b attn wq/wgate shard (DONE 2026-07-16) | low | delivered −0.29 ms/token |
| 2 MoE in0_block_w=16 (DONE 2026-07-16) | low | delivered −1.1 ms/token |
| — the two above: 30.1 → 31.4 tok/s (+4.3%) | | |
| 1b MoE se_down shard | low | WASH (reshard eats it) — not shipped |
| 3b conv add-chain (`QWEN36_CONV_ADDCHAIN`, default on) | med | **DELIVERED: 31.5→34.6 tok/s (+9.8%), −2.9 ms/token** |
| 3b conv fold (separate kernel) | med | REVERTED: 0 net win + PCC fail (see 3b original) |
| 3b conv fold INTO decode_step_tt | high | superseded by the add-chain (pure ttnn, no DFB fight, accuracy-safe) |
| 2 MoE per-K-block handshake (C++) | high | small; beyond in0_block_w=16 needs kernel work |
| 1c ring+prefetcher (+stream refactor) | very high | pushes toward the 5.8 ms BW floor |
| 5 router rewrite (`QWEN36_MOE_ROUTER_FAST`, default on) | low | **DELIVERED: 35.0→37.4 tok/s (+6.9%), −1.89 ms/token** |
| 4 precision → more BFP4 (gdn proj / shared expert) | low | **re-rated UP — see the correction below** |
| 6 KV precision (`QWEN36_KV_DTYPE=bf8`) | low | **MEASURED: −0.10 ms at 4K (noise) but −0.90 ms at 32K and −3.58 ms at 128K — a long-context lever, and −2.5 GB of DRAM. Crossover ~8K** |

## Deferred research-backed levers (surveyed 2026-07-17; not yet built)

Candidate levers from a modern-research survey, kept here so a future session doesn't re-derive them.
Ranked; each needs the noted open question resolved (usually a microbench) before committing.

- **Norm→matmul operator fusion (helps prefill AND decode).** Fuse each RMSNorm into the matmul it
  feeds, keeping the normalized activation in L1 (no DRAM round-trip): `input_norm→in_proj/qkv`,
  `post_norm→moe gate`, gated-delta `qknorm` and `norm_out→w_out`. We already do matmul–matmul fusion
  (fused `w_in_proj`) and DRAM-sharding, but NOT norm→matmul. Ref: *Operator Fusion for LLM Inference
  on the Tensix Architecture* (arXiv:2606.09879, Wormhole N300, Qwen): measured **−37% attn, −16% MLP,
  −7.9%/decoder-layer, PCC ≥ 98.75%**. Directly attacks the ~7% decode inter-op dispatch gap and the
  prefill norm/glue overhead. Open Q: does ttnn expose a fused norm+matmul (LN-fused-matmul) or does it
  need a ttl kernel? Microbench first.
- **MoE prefill grouped-GEMM over compacted tokens.** Replace the dense broadcast (`moe.repeat`, 15%
  self-time, 268 MB/chunk, 32× FLOP waste) with sort/compact-by-expert → one variable-M batched matmul
  → scatter back. Refs: SonicMoE (arXiv:2512.14080), *Static Batching of Irregular Workloads*
  (arXiv:2501.16103), PyTorch persistent cache-aware grouped-GEMM. Caveat: the dense matmul is already
  near-roofline (§2) and the per-tile sparse path measured 5–7× slower; a true grouped-GEMM was never
  built. Speculative — gated on whether tt-metal has an efficient variable-M grouped-GEMM primitive.
- **Sorting-free sampling (small).** Replace the 8-chunk `topk(32768)` + `ttnn.sampling`
  (`model.py`) with dual-pivot rejection sampling. Refs: FlashInfer sorting-free sampling (2025-03),
  Qrita (arXiv:2602.01518). <1 ms on a ~29 ms step — only if sampling latency becomes a product concern.
- **❌ Width-sharded decode RMSNorm (`QWEN36_NORM_SHARDED`, MEASURED WASH at 40L, 2026-07-17 — default OFF).**
  Microbench (`tests/bench_norm_matmul_fusion.py`): the interleaved single-core `ttnn.rms_norm` on the T=1
  full-hidden vector is ~33 µs, but an 8-core WIDTH-SHARDED `ttnn.rms_norm` is ~10 µs (**3.2×**, PCC 1.0),
  curing decode **row-starvation** (M=1 has no rows to spread across the grid). ~81 full-hidden norms/step
  (40 input + 40 post + 1 final) = ~2.71 ms in isolation, so the isolated microbench projected +3.8% and a
  reshard-free stream +6.5%. **BUT the 40L A/B is a dead wash: pure `execute_trace` 28.73 ms → 28.73 ms
  (identical); full decode_step 28.89 → 29.05.** Root cause = the **conv-fold / se_down lesson**: the norms
  are NOT on the decode critical path — they pipeline/overlap with the matmul chain, so making each one
  cheaper in isolation changes nothing at 40L. (Contrast the DRAM-shard *projections*, which DID help
  because matmuls ARE the critical path.) The isolated microbench over-counts precisely because it can't
  model pipeline overlap. Code kept flag-gated (`tt/rms_norm.py` `_norm_shard_cfg`, `x.volume()==dim`
  guard, PCC 0.99999) as a default-OFF building block for the prefetcher stream, but it is NOT a
  single-user lever on its own. **Implication: glue-op (norm/reshard/tilize) reduction won't move
  single-user decode; only shortening the critical-path matmul chain will (MoE sparse kernel, ring +
  prefetcher, gdn) — or batched decode for aggregate throughput.**
  - **Does NOT help prefill.** Prefill norms already have S rows of real work (not row-starved) AND are
    only ~0.6% of the prefill step (§3b: `input_norm`/`post_norm` ~0.3% each; prefill is `delta.recurrence`
    29% + MoE matmuls ~50%). The sharded config is `block_h=1` (≤32 rows) so it doesn't even apply to S>1.
    (The separate arXiv:2606.09879 norm→matmul *kernel* fusion / SRAM-resident stream is the prefill lever,
    not this norm-sharding trick.)
  - **DEFERRED — per-head qk_norm sharding (small, unproven).** `attn.qk_norm` normalizes over
    head_dim=256 across 16 q-/2 kv-heads → ~25 µs/attn-layer × 10 = ~0.25 ms ≈ **0.9% of decode** (vs the
    full-hidden pool's 9.4%). Currently interleaved (excluded by the `volume()==dim` guard). Uncertain win:
    a 256-wide reduction width-sharded across 8 cores is 1 tile/core + a cross-core handshake that may cost
    more than it saves; the better layout is probably a HEIGHT-shard over heads, a separate design. Low
    priority — settle with a `[1,1,16,256]` microbench case before building. Prior: wash/marginal expected.

Also see the memory note `qwen36-prefill-p1-deadend` and `analyze_chunk_carry_precision.py` for the
measured reason the "fast fused chunk_state kernel" prefill lever was abandoned.
