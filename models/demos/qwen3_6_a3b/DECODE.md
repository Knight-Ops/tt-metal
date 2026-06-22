# Qwen3.6-35B-A3B Decode — Performance Analysis & Roadmap

Companion to `HANDOFF.md` (read that first for the model, environments, and the trace/decode
mechanics) and `PREFILL.md` (the prefill counterpart). This doc is specifically about **single-user
decode throughput (tok/s/user)**: where it stands, the roofline that bounds it, why we are far below
that bound, and the ranked workstreams to close the gap. Prefill is out of scope here.

---

## 1. Where decode stands today

Full 40-layer, single P150, BFP4 experts, traced decode (`demo.py --trace`). Progression:

| path | decode (tok/s/user) | ms/token |
|------|---------------------|----------|
| initial (host scan, gather MoE, concat KV) | 0.52 | 1923 |
| on-device scan + caches + fixed KV + dense decode | 1.45 | 690 |
| + decode trace capture | 2.09 | 478 |
| + ttnn.sparse_matmul MoE decode | 7.28 | 137 |
| + self-contained on-device trace (argmax/rope/pos + O(1) KV) | 7.77 | 129 |
| + fused gate+up MoE sparse_matmul | 9.11 | 110 |
| + multicore argmax (lm_head head stage 9.9 → 1.6 ms) | 9.81 | 102 |
| + MoE `nnz=top_k` (skip the all-256-slot compute scan) | 11.98 | 83 |
| + fused T=1 conv + b\|a + se_gate\|se_up projections | 13.36 | 75 |
| **+ BFP4 lm_head weight** (`QWEN36_LMHEAD_BF4`) | **13.43** | **74** |

Generation is greedy-identical throughout
(`"The capital of France is"` → `" Paris, a city renowned for its rich history, culture,"`),
trace == eager (`tests/test_trace.py`), and every module is PCC-gated.

**Profiling correction (measured, supersedes the original §3/§4 premises).** A per-layer-type decode
microbench (`tests/bench_decode.py`, real dims, traced) revealed two of the original premises were
wrong, and the three cheap wins above (no kernel work) took decode 110 → 75 ms:
- **lm_head was never bandwidth-bound.** The matmul is ~1.5 ms (~340 GB/s, ~66% of peak). The 9.9 ms
  "lm_head" was the **single-core `ttnn.argmax`** over the 248k vocab; `ttnn.argmax(..., use_multicore=True)`
  on a ROW_MAJOR input is ~0.06 ms. DRAM-sharding the lm_head matmul (old §4C) is unnecessary and was
  measured *slower*.
- **MoE `nnz` is the cheap lever, not a hazard here.** Passing `nnz=top_k` makes the kernel's compute
  process only the 8 active experts (`num_batch_compute=nnz`), cutting MoE 64.8 → 46.4 ms. It is SAFE
  for a fixed top-k (softmax→top-k→renormalize gives exactly `top_k` positive weights every token, never
  the gpt-oss zero-flush). Env `QWEN36_MOE_NNZ=0` restores `nnz=None`.
- **Gated-delta's hot spot is the 4-tap conv, not the recurrence.** Measured per layer: conv+silu
  0.40 ms (≈44%) ≫ recurrence 0.19 ms ≫ projections 0.13 ms. For T=1 the conv collapses to one
  `multiply` + one row-sum over the stacked taps (vs K slices+multiplies+adds): gated-delta 1.02 → 0.77 ms.

Current measured per-token breakdown (`bench_decode.py` + Tracy profiler, §3): **MoE ~46 ms (62%)**,
Gated-DeltaNet ~23 ms (31%), attention ~3.8 ms (5%), lm_head ~1.5 ms (2%). The next big lever is
clearly the MoE — `nnz=top_k` removed the compute-scan but the sparse_matmul reader still carries
~0.94 ms/layer of per-slot multicast overhead (≈40× above the ~24 µs the 12.6 MB active weights imply).
The true 8-of-256 weight gather (§4A) is the remaining MoE win and needs C++ kernel work (a ttl gather
was **proven infeasible** — ttl cannot do data-dependent DRAM addressing; `ttnn.gather`/`embedding`
reject bf4; see §4A).

Already landed (the decode step is fully self-contained on device; all PCC-gated, greedy-identical):
- On-device greedy **multicore** `ttnn.argmax` on a ROW_MAJOR input — only the 1 next-token id returns
  to host. (Single-core argmax over the 248k vocab was 8.4 ms; multicore is ~0.06 ms.)
- RoPE indexed from a precomputed device table (`ttnn.embedding`), position advanced on device with
  `ttnn.plus_one` — no per-token host rebuild/copy of cos/sin/pos.
- O(1) KV write via `ttnn.experimental.paged_update_cache` (tensor position, `page_table=None`).
- MoE: gate+up fused (one `sparse_matmul`) + `nnz=top_k` (compute only the 8 active experts).
- Gated-delta: fused T=1 conv (one multiply + row-sum) + fused `in_proj_b|in_proj_a` and shared-expert
  `se_gate|se_up` projections.
- **BFP4 lm_head weight** (`QWEN36_LMHEAD_BF4`, default on; `=0` reverts to BFP8) — lm_head is
  bandwidth-bound, so fewer bytes is the only lever; verified greedy-identical.

So host work per token is now just `execute_trace` + a 1-element readback. The remaining cost is
**all on-device**, and that is what this roadmap attacks.

---

## 2. The roofline: decode is bandwidth-bound, ~220 tok/s is the single-user ceiling

At batch=1 every weight is read from DRAM exactly once per token and consumed by an **M=1** matmul.

**Bytes/token (as-built, mixed precision):**

| Component | active params | dtype | bytes/token |
|---|---|---|---|
| Gated-delta projections (30L) | ~1.0 B | bf8 | ~1008 MB |
| Routed experts (8 of 256, 40L) | ~1.0 B | bf4 | ~504 MB |
| Attention projections (10L) | 0.27 B | bf8 | ~270 MB |
| lm_head | 0.51 B | **bf4** (was bf8) | ~254 MB |
| Shared expert (40L) | 0.12 B | bf16 | ~248 MB |
| Router | 0.02 B | bf16 | ~42 MB |
| **Total** | **~2.9 B** | mixed | **~2.3 GB** |

- **Bandwidth roofline:** 2.3 GB ÷ 512 GB/s ≈ **4.6 ms ≈ ~220 tok/s** (as built, with bf4 lm_head). If
  every weight were bf4 (~1.5 GB) the ceiling rises to **~341 tok/s** — but that costs accuracy (§4E).
- **This ceiling IS a single-user target.** A weight read is a large streaming read; the M=1 compute
  that consumes it is trivial and fully overlappable, so a well-written M=1 matmul approaches peak
  DRAM bandwidth. Bandwidth ceilings do not require batching — they require near-peak BW utilization
  and hidden per-op overhead. A mature stack realistically lands ~100–150 tok/s single-user; ~200 is
  the asymptote to aim at.

**Decode is NOT FLOP-bound.** FLOPs/token ≈ 2 × 2.9B ≈ **6 GFLOP**. Arithmetic intensity =
6e9 / 2.6e9 ≈ **2.2 FLOP/byte**. The P150 ridge point (peak FLOP/s ÷ BW) is **~195+ FLOP/byte** even
at a conservative 100 TFLOP/s bf16 — so M=1 decode sits **~100× on the memory-bound side** of the
roofline. The FLOP ceiling is ~16,000–33,000 tok/s; compute is irrelevant here. FLOPs only bound
throughput when each weight is reused across many tokens — i.e. **prefill or large-batch decode**
(crossover ≈ batch 90–180). More silicon FLOPs would not move single-user decode at all.

**Bottom line:** we are at 74 ms vs a ~4.6 ms bandwidth ceiling — **~16× off** (down from ~22× at the
110 ms start). The gap is not physics. Tracy profiling (§3) shows the traced step is **roughly half
real Tensix compute, half on-device overhead** (per-op setup + inter-op gaps), with matmuls dominating
the compute and the MoE sparse scan dominating the overhead. So closing the gap is two things:
(a) **op-fusion / fewer dispatches** for the overhead half, and (b) **the MoE gather + precision** for
the compute half.

---

## 3. Where the 74 ms goes (measured)

Two complementary measurements (full workflow: `HANDOFF.md` §10):
- **`tests/bench_decode.py`** — real-model decode benchmark (definitive full-model tok/s + per-layer-
  type breakdown). Per-token breakdown: **MoE ~46 ms (62%)**, Gated-DeltaNet ~23 ms (31%), attention
  ~3.8 ms (5%), lm_head ~1.5 ms (2%); reconciles to the full-model number within a few ms.
- **`tests/prof_decode.py`** — Tracy device profiler, per-op kernel/FW/gap timings, with eager and
  traced decode steps in signpost-bounded regions (and a prefill region).

What the Tracy data established (caveat: instrumentation inflates absolutes ~1.8× and overhead more
than kernel time; trust ratios + op *ranking*, not absolute µs):

1. **Decode is NOT host-bound once traced.** The eager step is ~80% host-dispatch gaps; tracing
   collapses those to ~µs (decode is one device trace, host does only `execute_trace` + a 1-elem
   readback). So the remaining cost is genuinely on-device — there is no unrealized win in "apply trace
   capture."
2. **The traced step splits ~half compute / ~half on-device overhead** (per-op setup + inter-op gaps),
   and **63% of ops are tiny (<5 µs kernel)**. Compute is dominated by matmuls: dense `Matmul`
   (lm_head + projections) + the MoE `SparseMatmul`.
3. **The MoE sparse_matmul is scan-overhead-bound, not bandwidth-bound.** The visualizer reports a
   physically-impossible **>100% DRAM** on it (605 GB/s gate_up "118%", 941 GB/s down "183%") — the
   signature that it skips most of its nominal full-256-expert reads, yet still burns ~587 µs/layer,
   ~28× over the ~16 µs its ~8-expert data movement warrants. Pure 256-slot multicast-semaphore scan.
   → the **gather (§4A)** removes it.
4. **lm_head is genuinely bandwidth-bound** (357 GB/s = ~70% peak). Its only lever is fewer bytes
   (BFP4, landed §4E) — op-fusion / sharding does nothing for it.
5. **Gated-delta is diffuse.** The recurrence is already batched over all 32 heads in ttnn (~6 ops, not
   ~96), the SLOW dense projections run at ~23% BW, and `ReshapeView` relayouts cost ~79 µs each. No
   single clean lever (see §4B for why a ttl decode kernel is now low-ROI).

---

## 4. Workstreams (status + ranked remaining levers)

The cheap, low-risk wins are **landed** (§1): multicore argmax, MoE `nnz=top_k`, the conv/projection
fusions, and BFP4 lm_head — together 110 → 74 ms. What remains is harder, and the profiling (§3)
re-ranked it and overturned several original premises. Below: what's done, the remaining levers, and
the current decision.

### 4A. True 8-of-256 expert-weight gather for MoE  *(DONE — see MOE_GATHER_PLAN.md for the measured result)*
- **Full plan + measured result: [`MOE_GATHER_PLAN.md`](MOE_GATHER_PLAN.md).** SHIPPED: **40-layer
  decode 74 → 43 ms/token (13.4 → 23.2 tok/s/user), greedy-identical.**
- **The "scan dominates MoE" premise here was REFUTED by profiling.** Two additive levers delivered the
  win: (1) the **gather** — an opt-in `indices` mode on `ttnn.sparse_matmul` (iterate 8 experts not
  256, compact output) — worth ~14–18%; (2) **`in0_block_w` tuning** (pure Python config, no C++) —
  worth ~22% — because the real bottleneck is per-K-block multicast-handshake overhead, not the scan
  and not weight bandwidth. The Tracy >100% DRAM% on the sparse_matmul is a byte-estimate artifact
  (assumes all 256 experts read), NOT evidence of bandwidth-bound; raising `in0_block_w` (which changes
  iterations, not bytes) cutting time 22% proves it was overhead-bound. Env knobs: `QWEN36_MOE_GATHER`
  (default on), `QWEN36_SPARSE_IN0BW` (default 8).
- *(Historical analysis below; superseded by the measured result above.)*
- **Status:** the prerequisite cheap win (`nnz=top_k`, compute only 8 experts) is landed (64.8 → 46.4 ms).
  The gather itself is **not done**.
- **Why:** MoE is still ~62% (~46 ms). The sparse_matmul reader is **scan-overhead-bound** — Tracy's
  >100%-DRAM signature (§3) proves it skips most reads yet burns ~0.94 ms/layer (~28× over its ~24 µs
  data-movement floor) on the 256-slot multicast scan.
- **Target:** MoE ~46 ms → **~5–15 ms** (the single biggest possible win, → ~25 tok/s).
- **Approach:** make the kernel iterate only the 8 active experts (compacted on-device `topi`) instead
  of scanning 256. **This is C++ kernel work.** A ttl gather was investigated and **proven infeasible**:
  ttl cannot turn a tensor-read index into a DRAM address (`raw_element_read` yields a compute-domain
  scalar with no bridge to the datamovement/NOC addressing domain; only `ttl.node()` gives runtime
  addressing). `ttnn.gather`/`ttnn.embedding` reject bf4. So the realistic paths are:
  1. **Extend the C++ `sparse_matmul` reader** to consume the compacted `topi` and iterate 8 not 256
     (keeps bf4 + trace). Files: `ttnn/cpp/.../sparse/.../sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`
     + the device-op + pybind. *Recommended.*
  2. **Fallback (Python, no C++):** keep a decode-only bf16 copy of experts, `ttnn.gather` the 8 tiles,
     dense batched matmul. Costs ~+6 GB device memory + higher per-token bandwidth (33 vs 12.6 MB) —
     measure before committing; may not beat 46 ms.
- **Risk:** high (core-ttnn C++ + bf4 reader addressing + traceability of the new operand).
- **Validate:** PCC vs the current sparse path (`tests/test_moe_sparse.py`), then ms/layer in
  `bench_decode.py`, then full-demo greedy-identical.

### 4B. Fused single-step Gated-DeltaNet decode kernel  *(REASSESSED — now low-ROI, not recommended)*
- The original plan assumed gated-delta was ~35 tiny dispatches/layer dominated by the recurrence.
  **Profiling disproved this:** the recurrence is **already batched over all 32 heads** in ttnn (~6 ops,
  not ~96) and is only ~0.19 ms/layer; the conv (the real hot spot) is already fused. The remaining
  gated-delta cost is *diffuse* (projections at ~23% BW, ~79 µs `ReshapeView` relayouts, l2norm).
- A per-head ttl decode kernel needs a head-major **tile layout** that, for T=1, requires per-token
  zero-pad/scatter setup costing roughly what the recurrence saves — the chunk-prefill kernel won only
  because it amortized that setup over C=64 tokens, which decode lacks. So it is high-effort/high-risk
  (ttl, device-hang discipline, HANDOFF §9) for an **uncertain few-ms payoff**.
- **Recommendation:** do not build it speculatively. If pursued, run a **time-boxed experiment** first
  to confirm the head-major setup overhead does not eat the gain, before committing to the full kernel.

### 4C. DRAM-sharded weights + 2nd-CQ prefetch  *(SUPERSEDED — premise was wrong)*
- The original premise ("reads run at ~10% of peak") was **wrong**. The lm_head matmul already runs at
  ~70% peak (357 GB/s); DRAM-sharding it was *measured slower* and discarded. The dense projections are
  at ~23% BW (some headroom) but DRAM-sharding adds an `interleaved_to_sharded` + per-shape tuning that
  the lm_head experiment showed is net-neutral-to-negative at these M=1 shapes. **Not a current lever.**
  (A 2nd-CQ weight prefetcher remains theoretically possible but is high-effort and unproven here.)

### 4D. Op-count / control-overhead trimming  *(partially landed; remainder diffuse)*
- **Landed:** shared-expert `se_gate|se_up` fused, gated-delta `in_proj_b|in_proj_a` fused, T=1 conv
  collapsed to multiply+row-sum.
- **Remaining:** the ~79 µs `ReshapeView` relayouts (e.g. the k-transpose for the outer product, the
  post-norm transpose) are largely *intrinsic* to the recurrence formulation — trimming them is fiddly
  relayout work for ~0.3 ms. Low priority unless 4B is taken on (which would subsume them).

### 4E. Precision — lower the byte floor  *(partially landed; the cheap remaining tier)*
- **Landed:** BFP4 lm_head (`QWEN36_LMHEAD_BF4`), greedy-identical, ~0.7 ms saved. lm_head is
  bandwidth-bound so this was its only lever.
- **Remaining (next cheap wins):** push the gated-delta projections (~1 GB bf8) and shared expert
  (bf16) toward BFP4 where PCC holds. Decode-only, accuracy-bounded, per-module PCC-gated. Each is a
  one-line dtype change + the test suite — the lowest-effort items left.

Not needed: long-context KV is already O(1) (`paged_update_cache`); attention is only ~5%.

---

## 5. Current decision & target trajectory

The cheap wins are banked (110 → 74 ms, 13.4 tok/s). The next move is one of three, in priority order:

1. **MoE gather (4A)** — *the* remaining lever. Only thing that materially moves the number
   (→ ~25 tok/s). Cost: high-effort C++ kernel work. **Recommended** if continuing. Implementation
   plan + milestones: [`MOE_GATHER_PLAN.md`](MOE_GATHER_PLAN.md) (start with the microbench probe).
2. **More precision (4E)** — push gated-delta projections / shared expert toward BFP4. Cheap (dtype +
   PCC gate), small per-item but additive; the lowest-effort remaining work. Good to do alongside 4A.
3. **Bank here** at 74 ms / 13.4 tok/s (≈1.5× over the 9.1 start), all low-risk and validated, with the
   profiling harness (`bench_decode.py`, `prof_decode.py`) + ground truth as deliverables.

Not recommended: **4B (gated-delta ttl kernel)** — profiling showed it low-ROI (§4B); only pursue after
a time-boxed feasibility experiment. **4C (DRAM-sharding)** — superseded (§4C).

Indicative trajectory (single-user, full 40-layer; rough, validate each step):

| after | ms/token | tok/s/user | effort |
|---|---|---|---|
| **today (all cheap wins landed)** | **74** | **13.4** | done |
| + 4A (MoE expert gather) | ~40–48 | ~21–25 | high (C++) |
| + 4E (more BFP4) | ~36–44 | ~23–28 | low |
| + 4B (fused gated-delta, if it pays) | ~28–35 | ~30–36 | high (ttl) |
| + near-peak BW / prefetch everywhere | ~15–22 | ~45–65 | high, sustained |
| bandwidth ceiling (as-built precision) | ~4.6 | ~220 | asymptote |
| bandwidth ceiling (all-bf4) | ~3 | ~341 | asymptote |

**~50–65 tok/s is the realistic mature-stack reach** for the remaining hard work; ~150–220 is the
asymptote requiring near-peak BW on every category simultaneously and is not a practical single-user
target on this card. (If aggregate throughput is the real goal, **batching** crosses into FLOP-bound
territory around batch ~90–180 and gets there far more directly than chasing the single-user asymptote.)

---

## 6. How to measure (do this before optimizing)

The tooling is built. The fast inner loop is the microbench; the profiler is for attribution. Full
profiler workflow (Tracy + ttnn-visualizer, prefill + decode): **`HANDOFF.md` §10**.

- **`tests/bench_decode.py`** — benchmark on the **real model** (actual checkpoint weights). Reports
  (1) the **DEFINITIVE** full-model traced decode tok/s (same path as `demo.py --trace`; with
  `QWEN36_LAYERS=40` it is the real 40-layer number), and (2) a per-layer-type breakdown from the
  model's own real components (gated-delta / attention / MoE / lm_head), each timed inside its own
  trace × layer counts (30 / 10 / 40 / 1), splitting MoE into `_shared_and_router` vs
  `forward_sparse_decode`. The breakdown reconciles to the definitive number within a few ms.
  **This is the before/after loop for every change** (use `QWEN36_LAYERS=4`, ~100 s load, for fast
  iteration; `=40` for the authoritative figure). Note: timing is value-independent for these ops, so
  the breakdown matches the previous random-weight microbench — real weights give the trustworthy
  end-to-end number + exact production dtypes/config.
  ```bash
  QWEN36_LAYERS=40 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/bench_decode.py --iters 100
  ```
- **`tests/prof_decode.py`** — Tracy device profiler; per-op kernel / FW / op-to-op-gap, with prefill,
  eager-decode, and traced-decode in signpost-bounded regions. Use to attribute compute-vs-overhead and
  per-op DRAM/FLOPS utilization (run via `python -m tracy -r`; see HANDOFF §10). Caveat: instrumentation
  inflates absolutes ~1.8× — trust ratios + op ranking.
- **TT-NN Visualizer** — `--performance-path` over the profiler report for the interactive per-op view
  (scope to a signpost region, sort by Device Time). HANDOFF §10.5.

Always end with the full 40-layer `demo.py --trace` greedy run to confirm output stays
byte-identical (`"…Paris, a city renowned for its rich history, culture,"`), plus the decode test
suite: `test_trace`, `test_attention_decode`, `test_gated_delta_decode`, `test_moe`, `test_moe_sparse`,
`test_model`.

---

## 7. Notes / provenance

- Decode mechanics, file map, and the trace/cache design: `HANDOFF.md` (§5, §6). Prefill roadmap:
  `PREFILL.md`. User-facing perf summary: `README.md`.
- `ttnn.sparse_matmul`: `nnz=top_k` is **landed and safe** here — it sets `num_batch_compute` so the
  kernel computes only the 8 active experts (64.8 → 46.4 ms). The gpt-oss deadlock (a static `nnz`
  exceeding the live nonzero count) cannot occur for a fixed top-k: softmax→top-k→renormalize yields
  exactly `top_k` positive weights every token. `QWEN36_MOE_NNZ=0` reverts to `nnz=None`. The residual
  cost is the reader's 256-slot multicast **scan** (not weight traffic, not compute) — that is what the
  gather (4A) removes, and it needs C++ (a ttl gather is proven infeasible; §4A).
- The O(1) KV write uses the non-paged `paged_update_cache` path (cache `[1, n_kv, max_seq, hd]`,
  input height-sharded with kv-heads padded to TILE_HEIGHT, `page_table=None`) — mirrors
  `models/tt_transformers/tt/attention.py`.
- Roofline inputs: ~2.9 B active params, ~2.6 GB/token at the current mixed precision, 512 GB/s,
  P150 bf16 in the hundreds of TFLOP/s. Re-derive bytes/token if any dtype in §2 changes.
