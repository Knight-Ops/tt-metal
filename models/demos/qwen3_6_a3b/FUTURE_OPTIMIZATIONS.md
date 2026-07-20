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
pass). Per-layer-type breakdown (pre-add-chain baseline; add-chain takes Gated-DeltaNet 0.451→0.323):

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

### ✅/⏭ Lever 1b — extend DRAM-sharding to attention + MoE projections
**DONE:** attention `wo` (`tt/attention.py`, `QWEN36_ATTN_DRAM_SHARD=1`) — attention 0.382→0.332 ms/layer.
**DONE (2026-07-16):** attention `wq`/`wgate` (`QWEN36_ATTN_DRAM_SHARD_IN=1`, default on) — microbench
1.71–1.73×; at 40L attention 0.332→**0.303** ms/layer (−0.29 ms/token). Both wq/wgate share one shard
config (identical `[hidden, n_heads*head_dim]`); the decode branch in `_qkv` reshards `x` once, runs both.
**MEASURED WASH (not shipped, `QWEN36_MOE_SE_DRAM_SHARD` default OFF):** MoE shared-expert `se_down`
(K=512,N=2048) microbenched 1.46× but at 40L netted ~0 — its ~6 µs matmul saving is eaten by the ~8 µs
reshard in/out (K is too small for the fixed reshard cost to amortize). Code kept, opt-in. `wk`/`wv` too small.
Shared helper `common.build_dram_shard(w, K, N)` (returns w_dram, act_mc, out_mc, program_config).

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

### ⏭ Lever 4 (cheap, deferred) — precision: lower the byte floor toward BFP4
The one genuinely bandwidth-bound component, lm_head, already runs BFP4 (`QWEN36_LMHEAD_BF4`,
greedy-identical, ~0.7 ms saved) — that was its only lever. The remaining cheap tier: push the
gated-delta projections (~1 GB bf8) and the bf16 shared expert toward BFP4 where PCC holds. Each is a
one-line dtype change + the decode test suite, accuracy-bounded and per-module PCC-gated (opt-in flags
like `QWEN36_EXPERT_DOWN_BF8` already exist for this kind of experiment; gate quality with MMLU-Redux
in `evaluation/`, not decode token-identity). Because decode is overhead-bound not BW-bound (see the
key measured fact above), this shrinks the asymptote (all-bf4 ceiling ~341 tok/s) more than it moves
the current number — the lowest-effort item left, not a big lever.

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
| 4 precision → more BFP4 (gdn proj / shared expert) | low | small now (overhead-bound); lowers the all-bf4 asymptote |

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
