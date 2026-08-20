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
| **total speedup** | **2.00x** | **2.08x** | **2.14x** |

> **Read prefill tok/s at a FIXED length.** `tests/benchmark.py` reports `T / TTFT` per prompt and then
> takes an unweighted MEAN over prompts of 24-220 tokens, where that ratio measures fixed per-call cost
> rather than throughput (same run: 37.7 tok/s at 24 tokens, 257 tok/s at 220). The old "~200 tok/s
> prefill" headline was that mean. Steady-state is the table above.

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
3. **`head` costs 15.5 ms of a ~614 ms prefill** for one token's logits. `_head` already slices to the
   last position before the 248k-vocab matmul, and decode's identical matmul is 1.23 ms, so the extra
   ~14 ms is the D2H: `[1, 248320]` in TILE layout is physically `[32, 248320]` = 15.9 MB over PCIe.
   `ttnn.to_layout(logits, ROW_MAJOR)` before `from_tt` should move 0.5 MB instead. Untested.
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
