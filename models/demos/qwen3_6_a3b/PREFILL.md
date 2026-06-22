# Qwen3.6-35B-A3B Prefill — Performance Analysis & Roadmap

Companion to `HANDOFF.md` (read that first for the model, environments, and the fused-prefill
integration). This doc covers **prefill throughput**: the current baseline, the roofline it's measured
against, what's been optimized, and what's left — including a thorough scope for **traced incremental
(multi-turn) prefill**. Decode is the other engineer's path; summarized here only for the roofline.

> History note: earlier revisions of this doc hypothesized prefill was purely host-dispatch-bound and
> ranked "trace the prefill" behind everything. Per-op profiling (below) showed it's ~46% device-kernel
> / ~54% dispatch, and tracing turned out to be a real ~1.5× lever. The doc below reflects the measured
> state, superseding that earlier ranking.

---

## 1. Current baseline state (single P150, 40 layers, BFP4 experts)

All numbers warm, seq 256, one model load. Reproduce with `tests/bench_prefill.py` (prefill) and the
decode owner's `bench_decode.py`.

| prefill (40L, seq 256, warm) | time | tok/s | vs original |
|------------------------------|------|-------|-------------|
| original baseline (fused-scan prefill, pre-opt) | 1720 ms | 149 | 1.00× |
| **eager** (current default) | **867 ms** | **295** | **1.98×** |
| **traced** (replay; per-length trace) | **579 ms** | **442** | **2.97×** |

Latest authoritative run, `bench_prefill.py --seqs 128,256` at 40 layers (warm, best-of-3) — confirms
seq-256 traced and adds seq 128 (tracing helps more at shorter prompts: larger dispatch fraction):

| seq | eager | traced | tracing speedup |
|-----|-------|--------|-----------------|
| 128 | 627 ms / 204 tok/s | 343 ms / **374 tok/s** | **1.83×** |
| 256 | 847 ms / 302 tok/s | 579 ms / **442 tok/s** | **1.46×** |

Decode (the other engineer's path, for context): **~9.1 tok/s/user (~110 ms/token)**, captured as a
single replayed trace.

**Shipped since the original baseline** (all additive; eager + decode paths preserved, module PCC
tests pass): MoE matmul program-config tuning, last-token lm_head, MoE routing-fold, **prefill tracing
+ bucketing**, **eager incremental (from-cache) prefill**, MoE large-T chunking. Details in §3.

**Functional envelope:** from-scratch prefill and incremental prefill both work for any length up to
`max_seq_len` (KV-cache limit). Lengths bucket to multiples of 64; incremental advances the context in
64-token-aligned steps.

---

## 2. Roofline — where the silicon ceiling is

Estimates; assumptions stated so they can be refined. Prefill reuses each weight across all T tokens,
so it is **compute-bound** (high arithmetic intensity); decode reads weights per token, so it is
**bandwidth/latency-bound**.

**Prefill compute (40L, seq 256).** The dense MoE computes all 256 experts × 256 tokens:
`E·T·(H·2I + I·H)·2 ≈ 16.5 TFLOP` over 40 layers, plus attention/gated-delta (~1–2 TFLOP) ⇒ **~17.5
TFLOP**. The tuned MoE matmul measures **~130 TFLOP/s** (BFP4/LoFi; gate_up 0.275 TFLOP in 2.1 ms), so
the **achievable dense-compute floor ≈ 135 ms** (and the MoE matmuls themselves already run at it:
~3.6 ms/layer × 40 ≈ 144 ms). Weight-bandwidth floor: 17.5 GB / ~512 GB/s ≈ **34 ms** (not binding).

- **Current traced prefill 579 ms ≈ 4× the ~135 ms dense-compute floor.** The gap is **non-matmul
  overhead** — gated-delta fp32 chunk-prep, MoE `repeat`/`reduce`/tilize, layout churn, and per-op
  dispatch on the long tail of small ops (see §3 levers b).
- A *sparse-ideal* floor (only top-8 of 256 experts) would be ~4× lower again, but is **unreachable
  here**: at seq 256 nearly all experts are touched, and `ttnn.sparse_matmul` over all experts measured
  **7× slower** than the dense batched matmul. Dense is the right call.

**Decode roofline (context).** Per token reads the active weights (lm_head ~254 MB BFP4 + attention/
gated-delta + top-8 experts + shared ≈ 1–2 GB) ⇒ ~2–4 ms/token bandwidth floor. Current ~110 ms/token
is ~30–50× above it → decode is **dispatch/latency-bound**, not bandwidth-bound (it's already traced;
further wins are op-fusion / fewer launches — the decode owner's area).

**Takeaway:** prefill has ~4× of headroom to the achievable floor, concentrated in non-matmul overhead;
the matmuls are essentially at roofline.

---

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

---

## 4. Remaining levers (ranked)

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
paged cache anyway** (the HANDOFF already notes decode wants `paged_update_cache` for O(1) long-context
KV writes) — in which case the paged cache is shared infrastructure and this drops to ~2–3 days.

### 4b. Cut the non-matmul overhead (the ~4× gap to roofline)
The 579→~135 ms gap is elementwise/layout/dispatch on non-matmul ops. Per-op profiling (`prof_prefill.py`)
itemizes it. Highest-value, all eager-and-traced wins:
- **Gated-delta fp32 chunk-prep → fold into the ttl `_chunk_state` kernel** (cumsum/decay/β-scaling),
  cutting dozens of tiny ops/chunk to ~1 launch. Biggest single item (delta prep is the largest
  non-matmul phase). `QWEN36_DELTA_BF16_PREP=1` (bf16 prep) is implemented but measured ~neutral alone —
  the win needs the op-count reduction, not just bf16.
- **MoE `repeat` (~69 ms/40L)** — the `[E,T,H]` activation broadcast; **`tilize`/layout churn (~64 ms)**;
  **`reduce`** via `fast_reduce_nc`.

### 4c. Bucketing / attention at large T
- Finer bucket granularity (vs powers of 2) trades startup/trace memory for less padding waste.
- The from-scratch attention `sdpa(is_causal)` fit untuned at T=384; for very long from-scratch buckets,
  bound its L1 like decode's `k_chunk_size=128` if it overflows.

### 4d. Decode (other engineer)
Dispatch/latency-bound at ~110 ms/token, ~30–50× above its bandwidth floor; already traced. Wins are
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
- Fused-prefill design + ttl authoring constraints: HANDOFF §8–§9. Trace-safety findings + the
  incremental-prefill probe: memory note `qwen36-prefill-trace.md`.
