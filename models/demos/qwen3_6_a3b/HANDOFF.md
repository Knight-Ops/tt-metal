# Qwen3.6-35B-A3B on Blackhole (tt-nn) — Engineering Handoff

Status as of this handoff: the **full model runs on a single Blackhole P150, generates coherent
text, and decodes at ~9.11 tok/s/user** (≈17.5× the naive baseline; 110 ms/token). The **fused chunked
Gated-DeltaNet prefill kernel is now INTEGRATED and is the DEFAULT** (multi-head, head-dim 128): it
replaces the sequential recurrent scan for the 30 linear layers, giving a **5.4–5.7× warm prefill
speedup at 40 layers** (seq 256: 26.4 → 149 tok/s), while keeping decode identical and generation
coherent. Set `QWEN36_FUSED_PREFILL=0` to fall back to the scan. See §8 for the design + follow-ups.

---

## 1. TL;DR — what works, what's left

**Works (validated on hardware):**
- Full 40-layer Qwen3.6-35B-A3B text model: prefill + cached decode, coherent generation.
- Decode fast path: on-device Gated-DeltaNet recurrent scan + state cache, fixed-shape KV cache +
  `sdpa_decode`, `ttnn.sparse_matmul` MoE (gate+up fused → 2 launches), all captured as one **trace**.
  The step is **self-contained on device** — on-device greedy `argmax` (only the next token id is
  read back, not the 248k-vocab logits), RoPE indexed from a device table, position advanced with
  `ttnn.plus_one`, O(1) `paged_update_cache` KV write — so per token the host does only
  `execute_trace` + a 1-element readback.
- Standalone PyTorch reference cross-validated vs HuggingFace at **PCC = 1.0**; every tt-nn module
  matches it at **PCC > 0.99 on the Blackhole**. ~17 module tests pass.
- Fused chunked Gated-DeltaNet kernel (tt-lang): intra-chunk attention, triangular inverse,
  single-chunk output, 2-chunk state-carry chain, AND **multi-head (32 heads, 8×4 grid) at the real
  head-dim 128** — all PCC ≥ 0.99997.
- **Fused chunked prefill INTEGRATED + DEFAULT** (`QWEN36_FUSED_PREFILL=0` disables): per-chunk prep in ttnn (batched
  over heads) + the fused `_chunk_state` kernel + state carry, wired into `tt/gated_delta.py`. The
  full chunked path matches the torch reference (3-chunk PCC 0.99950); module tests pass; decode is
  unaffected (prefill→decode state handoff validated); 40-layer generation stays coherent.

**Left (lower-priority follow-ups):**
- Optional: also parallelize chunks across the idle cores (32 heads use 32 of ~110), and fold the
  ttnn per-chunk prep into fewer launches. MTP head + vision tower still out of scope.
- Note: the first fused prefill compiles the `_chunk_state` kernel once per (n_heads, head-dim) dims
  (~1 min), then it is disk-cached. Iterate with `QWEN36_LAYERS=4` (loads in ~80 s) — same kernel dims
  as the full model — and run the full 40-layer load only for final checks.

---

## 2. Environments (IMPORTANT)

- **tt-metal runtime venv:** `./python_env` (in the repo root: `/home/carl/projects/tt-metal/python_env`).
  - `transformers==4.53.0`, plus `ttnn` and **`ttl` (tt-lang) 1.1.3**.
  - Run everything model/kernel-side with `./python_env/bin/python`.
  - **DO NOT upgrade packages in this venv** — it is matched to this tt-metal build. In particular do
    NOT upgrade `transformers` here (the HF Qwen3.5-Moe class needs ≥5.2, which would break tt-metal).
- **Reference-only venv:** `/home/carl/qwen_ref_venv` (`transformers==5.12.1`, CPU torch).
  - Used ONLY to cross-validate the standalone reference against canonical HuggingFace.
  - Run `tests/cross_validate_reference.py` with this venv, never with `python_env`.
- **Device:** one Tenstorrent Blackhole at `/dev/tenstorrent/0` (P150, 32 GB).
- **Dev box limits:** 15 GB RAM, no GPU → cannot run the full 35B fp32 reference; correctness is
  per-module (load one module's weights at a time) + generation coherence.

### Weights
- HF repo `Qwen/Qwen3.6-35B-A3B`, downloaded to **`~/models/qwen36`** (72 GB, 26 safetensors shards).
- Override path via env `QWEN36_CKPT`.

---

## 3. How to run

```bash
cd /home/carl/projects/tt-metal

# Full 40-layer demo (traced + sparse decode); coherent text + perf
QWEN36_LAYERS=40 QWEN36_MAX_SEQ=128 ./python_env/bin/python \
    models/demos/qwen3_6_a3b/demo/demo.py --prompt "The capital of France is" --gen 12 --trace

# Reduced layers for fast iteration
QWEN36_LAYERS=4 ./python_env/bin/python models/demos/qwen3_6_a3b/demo/demo.py --seq 16 --gen 4 --trace

# On-device module PCC tests
./python_env/bin/python -m pytest models/demos/qwen3_6_a3b/tests/ -p no:cacheprovider

# Reference vs HuggingFace (SEPARATE venv)
/home/carl/qwen_ref_venv/bin/python models/demos/qwen3_6_a3b/tests/cross_validate_reference.py

# Fused kernel buildout (steps 1–6 validated, incl. multi-head + head-dim 128)
./python_env/bin/python models/demos/qwen3_6_a3b/tt/ttl_delta.py

# bf16-vs-fp32 chunk precision analysis (host only)
./python_env/bin/python models/demos/qwen3_6_a3b/tests/analyze_chunk_precision.py
```

Demo env vars: `QWEN36_LAYERS` (default 40), `QWEN36_MAX_SEQ` (KV/state cache len, default 512 in
ModelArgs; demo caps it; `sdpa_decode` tuned to bound L1 at any length), `QWEN36_CKPT`,
`QWEN36_SPARSE_DECODE=1` (opt into the gather decode path; default is sparse_matmul),
`QWEN36_DENSE_PREFILL=1` / `QWEN36_SPARSE_PREFILL=1` (MoE prefill path; dense is default),
`QWEN36_FUSED_PREFILL=0` (fall back to the recurrent scan; fused chunked Gated-DeltaNet prefill is
the DEFAULT, decode unaffected either way), `QWEN36_DELTA_IPLUSL=0` (use the fp32 doubling-product
inverse; the cheap bf16 T≈I+L inverse is the DEFAULT — both give chunk PCC ≥ 0.9995, I+L is ~5%
faster).

---

## 4. Architecture (Qwen3.6-35B-A3B = Qwen3.5-MoE ≈ Qwen3-Next, text path)

HF arch `Qwen3_5MoeForConditionalGeneration` / text config `qwen3_5_moe_text`. Scope here is
**text-only** (no vision tower, no MTP head). From the real `config.json`:

- hidden 2048, **40 layers**, vocab 248320, rms_eps 1e-6.
- **Hybrid mixers** (`layer_types`, `full_attention_interval=4`): **30 linear-attention** (Gated
  DeltaNet) + **10 full-attention** layers.
- **Full attention**: 16 q / 2 kv heads, head_dim 256; output gate fused into `q_proj`
  (`attn·sigmoid(gate)`); per-head qk-norm; **partial RoPE** (rotary_dim 64), rope_theta 1e7,
  MRoPE-interleaved `[11,11,10]` (collapses to default RoPE for text).
- **Gated DeltaNet** (linear attn): delta-rule; separate `in_proj_qkv/in_proj_z/in_proj_b/in_proj_a`
  (NOT Qwen3-Next's combined `in_proj_qkvz`+`fix_query_key_value_ordering` — KEY difference);
  short causal conv (kernel 4) + silu; 16 key / 32 value heads, head_k/v_dim 128; recurrent state.
- **MoE**: 256 experts, top-8 + renormalize (qwen3_vl_moe router); moe_intermediate 512; plus a
  sigmoid-gated **shared expert** (intermediate 512). `out = routed + sigmoid(gate)·shared`.

Reference math vendored from HF `transformers/models/{qwen3_next,qwen3_5,qwen3_vl_moe}`.

---

## 5. File map (`models/demos/qwen3_6_a3b/`)

```
reference/qwen3_5_moe.py     Standalone torch reference (text path). PCC=1.0 vs HF. The source of truth
                             for correctness. Contains torch_chunk_gated_delta_rule (the kernel ref).
tt/model_config.py           ModelArgs: parses config.json; single-P150 BFP4; precision/flags.
tt/common.py                 to_tt / from_tt / as_weight helpers (single-device replicate mesh).
tt/load_checkpoints.py       Streaming safetensors loader (lazy per-tensor; HF prefix model.language_model.).
tt/rms_norm.py               TtRMSNorm ((1+weight) folded) + TtRMSNormGated.
tt/attention.py              Full attention: fused q/gate split, partial RoPE, qk-norm; KV cache +
                             sdpa_decode (tuned program config, k_chunk_size=128). forward / forward_prefill
                             / forward_decode (O(1) paged_update_cache KV write at a tensor position).
tt/gated_delta.py            Gated DeltaNet: conv (4 taps) + projections + recurrent scan (decode +
                             non-fused prefill) + conv/recurrent state cache (in-place for trace).
                             ALSO the fused chunked prefill: _chunk_prep (ttnn per-chunk prep batched
                             over heads) + _forward_prefill_chunked (calls the ttl _chunk_state kernel,
                             carries state), behind QWEN36_FUSED_PREFILL.
tt/moe.py                    MoE: router; sparse_matmul decode (default; gate+up FUSED → 2 launches) +
                             gather decode (opt-in) + dense/sparse per-tile prefill (single weight
                             set [1,E,H,2I] gate_up + [1,E,I,H] down, sparse layout).
tt/decoder.py, tt/model.py   Decoder layer + full model: embed → 40 hybrid layers → norm → lm_head;
                             prefill (forward) + self-contained device-resident decode (start_decode /
                             decode_step_eager / decode_step_traced; on-device argmax + RoPE-table
                             lookup + plus_one position advance) + trace (capture_decode_trace).
tt/ttl_probe.py              tt-lang sanity: fused silu(a)*b kernel, PCC 1.0. Authoring pattern demo.
tt/ttl_delta.py              *** Fused chunked Gated-DeltaNet kernel (steps 1–6 validated: multi-head
                             8×4 grid, head-dim 128). *** _get_chunk_state_op factory + chunk_state_tt
                             (ttnn-native entry). Read its module docstring for step status + ttl lessons.
demo/demo.py                 Runnable text demo + perf harness (--trace, --gen, --seq, --prompt).
tests/                       Per-module PCC + e2e + cross-validation + precision analysis (see §7).
                             ALSO perf tooling: bench_decode.py (real-model decode benchmark:
                             definitive full-model tok/s + per-layer-type breakdown),
                             prof_decode.py (Tracy per-op profiling, prefill+decode),
                             gen_ttnn_report.py (ttnn-visualizer memory/op-graph db.sqlite). See §10.
README.md                    User-facing overview + perf table + remaining levers.
HANDOFF.md                   This document.
DECODE.md                    Decode performance analysis & roadmap (roofline, measured breakdown, levers).
MOE_GATHER_PLAN.md           Implementation plan for the MoE 8-of-256 gather (decode Tier 1 / DECODE §4A).
PREFILL.md                   Prefill performance analysis & roadmap (roofline, ranked bottlenecks).
```

Reusable upstream pieces: `models/common/rmsnorm.py` (add_unit_offset), `models/demos/gpt_oss/` (MoE
sparse_matmul reference), `models/demos/wormhole/mamba/` (SSM reference),
`ttl/tutorials/{matmul,elementwise}/step_*.py` (tt-lang authoring patterns).

---

## 6. Benchmarks (single P150, full 40 layers, BFP4 experts)

Decode (tok/s/user), the headline serving metric — progression across the work:

| path | decode |
|------|--------|
| initial (host scan, gather MoE, concat KV) | 0.52 |
| on-device scan + caches + fixed KV + dense decode | 1.45 |
| + decode trace capture | 2.09 |
| + ttnn.sparse_matmul MoE decode | 7.28 |
| + self-contained on-device trace (argmax/rope/pos + O(1) KV) | 7.77 |
| + fused gate+up MoE sparse_matmul | **9.11** |

Decode breakdown at 110 ms/token (profiled per layer-type, real dims): **MoE ≈65% (dispatch-bound —
each of the 2 sparse_matmuls scans all 256 expert slots), Gated-DeltaNet ≈24%, lm_head ≈8%,
attention ≈4%.** It is overhead/dispatch-bound, not weight-bandwidth-bound — DRAM-sharding the dense
matmuls would only touch the ~8% lm_head. Highest-value next levers: a true 8-of-256 expert-weight
gather (avoid the 256-slot scan in MoE) and a fused single-step Gated-DeltaNet decode kernel.

Prefill (tok/s, full 40 layers, seq 256, warm) — the **fused chunked prefill is now the default**;
it replaces the sequential scan for the 30 linear layers. Measured in one model load, toggling the
config at runtime (`tests/`-style; see scratch `prefill_bench.py`):

| prefill path (40-layer, seq 256, warm) | tok/s | speedup |
|----------------------------------------|-------|---------|
| recurrent scan (`QWEN36_FUSED_PREFILL=0`) | 26.4 | 1.00× |
| fused chunked, fp32 doubling inverse (`QWEN36_DELTA_IPLUSL=0`) | 142.2 | 5.39× |
| **fused chunked, T≈I+L inverse (DEFAULT)** | **149.0** | **5.65×** |

(The 4-layer microbench shows ~11.7×; at 40 layers the full-attention/MoE/lm_head share dilutes the
linear-layer win to ~5.5×.) Coherent generation confirmed at 40 layers:
`"The capital of France is"` → `" Paris, a city renowned for its rich history, culture,"`; decode
unchanged (~7.76 tok/s/user). The ttl kernel compiles once per (n_heads, head-dim) shape (~1 min)
then is disk-cached, so the first prefill in a fresh process is compile-dominated (warm up to amortize).

Build/load ~800 s (~17.5 GB BFP4 to device).

UPDATE (2026-06-19): the "host-dispatch bound, trace the prefill" hypothesis was **measured and is
wrong** — prefill is **device-COMPUTE-bound** (MoE ~62%, gated-delta ~35%; synced per-phase sum ≈ wall
clock, so dispatch gaps are negligible). Tuning the dense MoE matmul program config (it was
grid-underutilized — the default `ttnn.matmul` heuristic serialized the E=256 batch) plus the
last-token lm_head took **40-layer warm prefill 1720→1367 ms (149→187 tok/s, 1.26×), PCC-preserving**;
coherence + decode (~9 tok/s) unaffected. Tracing is shelved. See **`PREFILL.md` §0** for the measured
split, the fix, and the next lever (gated-delta fp32 chunk-prep).

---

## 7. Tests & correctness strategy

PCC-driven, bottom-up. Reference (`reference/qwen3_5_moe.py`) == HF (PCC 1.0); each tt-nn module ==
reference (PCC > 0.99 on device). Full-model PCC vs HF isn't possible on this box (RAM) → rely on
per-module PCC + a ≤4-layer real-weights slice + coherent generation.

- `tests/test_reference_smoke.py` — reference self-consistency (incl. chunk-vs-recurrent equivalence).
- `tests/cross_validate_reference.py` — reference vs HuggingFace (run in `qwen_ref_venv`). PCC 1.0.
- `tests/test_norms.py`, `test_attention.py`, `test_attention_decode.py`, `test_gated_delta.py`,
  `test_gated_delta_decode.py`, `test_moe.py`, `test_moe_sparse.py` — on-device module PCC.
- `tests/test_trace.py` — traced decode == eager decode (token-identical).
- `tests/test_model.py` — 4-layer real-weights end-to-end (finite logits, timing).
- `tests/analyze_chunk_precision.py` — host bf16-vs-fp32 chunk analysis (decided no fp32 inverse needed).

---

## 8. Fused-kernel prefill integration — DONE (design + follow-ups)

The fused chunked Gated-DeltaNet prefill is integrated and is the **default** (`QWEN36_FUSED_PREFILL=0`
falls back to the scan). How it works and the key findings (which differed from the original plan):

1. **Multi-head batching = the perf lever (done).** The recurrent scan is dispatch-bound (O(T)). The
   chunked kernel only wins if all 32 heads run in ONE launch. `_chunk_state` is now built by a cached
   factory `_get_chunk_state_op(n_heads)` with a 2D grid: Blackhole's compute grid is **11×10**, so a
   single axis caps at 11 — 32 heads use an **8×4 grid** (`_grid_dims`), and each core's linear head
   index is `h = node_n*gm + node_m`. Inputs are head-major stacked on the row axis; only the
   read/write threads add the per-head offset, the compute body is unchanged. (`grid=(32,1)` is
   rejected — that was the first integration bug.)
2. **Head-dim 64 → 128 needed NO tiling (key finding).** An mm probe (`scratchpad/mm_probe.py` pattern)
   showed every `_chunk_state` matmul — incl. `kcd·S` (K=4, 8-tile out) and `kgt·v_new` (16-tile out) —
   is legal with **bf16 DST** (`fp32_dest_acc_en=False`, 16-tile capacity). The old "K=4+8-out illegal"
   note was the **fp32-DST** case only. So head-dim 128 multi-head runs directly (2-chunk PCC 0.99997).
3. **Per-chunk prep runs on-device in ttnn** (`TtGatedDeltaNet._chunk_prep`), batched over the 32-head
   axis as `[1,Vh,C,*]`: `g_cum` via `ttnn.cumsum`; `decay` = `tril(gc_row−gc_col)` masked BEFORE `exp`
   (avoids inf·0=nan in the upper triangle); `v_beta/k_beta/qg/kgt/glast`; `L`; the inverse `T`; and
   `w=T·v_beta`, `kcd=T·(k_beta·exp(g_cum))`. Constant tril/strict-lower/eye masks are built once.
   **Inverse:** cheap `T≈I+L` by default (realistic L is tiny — matches the fp32 doubling product within
   PCC ≥ 0.9995 and is ~5% faster); `QWEN36_DELTA_IPLUSL=0` selects the fp32 doubling product. Only
   `_chunk_state` is a ttl kernel; everything else
   is ttnn (so `_chunk_apply`/`_chunk_inverse` ttl ops are NOT needed in the integration — `_chunk_state`
   with S=0 subsumes the chunk-0 case).
4. **Sequence chaining + integration (done).** `TtGatedDeltaNet._forward_prefill_chunked` pads T to a
   multiple of chunk_size=64 (zero-pad ⇒ g=0 on pad ⇒ state unaffected), loops chunks carrying the
   head-major `[Vh*Dk,Dv]` state, returns `core` `[1,Vh,T,Dv]` into the existing norm+out_proj tail.
   The flag-gated branch in `forward()` replaces the scan for prefill; the final state is reshaped to
   `[1,Vh,Dk,Dv]` and written to `cache["recurrent_state"]` so cached decode continues correctly.
5. **Validation:** `_chunk_state` PCC 0.99997 (8×4 grid, head-dim 128); full chunked path 3-chunk PCC
   0.99950; `test_gated_delta.py` + `test_gated_delta_decode.py` pass with the flag on; 40-layer
   generation coherent (`"…Paris, a city renowned for its rich history, culture,"`); decode unchanged
   (~7.76 tok/s). **Perf (warm, seq 256):** 40-layer prefill 26.4 → **149 tok/s (5.65× with I+L; 5.39×
   with fp32 doubling)**; 4-layer microbench ~11.7×. NOTE: the ttl kernel compiles once per
   (n_heads, head-dim) shape (~1 min) then is disk-cached, so the FIRST prefill in a fresh process is
   compile-dominated — warm up or amortize over the run.

Remaining follow-ups (lower priority):
- Parallelize chunks across the idle cores (32 of ~110 used); fold ttnn prep into fewer launches.
- MTP head + vision tower (currently out of scope).
- Long-context decode: DONE — the KV write is now O(1) via `ttnn.experimental.paged_update_cache`
  (tensor position, `page_table=None`; input height-sharded with kv-heads padded to TILE_HEIGHT),
  replacing the O(max_seq) masked write. Validated to `max_seq=512`.

Decode-perf follow-ups (the profiled bottleneck is MoE ≈65%, dispatch-bound):
- **True 8-of-256 expert-weight gather** (biggest lever). `ttnn.sparse_matmul` with `nnz=None`
  correctly skips the DRAM reads of zero experts BUT still scans all 256 sparsity slots per launch
  (per-slot multicast semaphores) — that scan, not weight bandwidth, is the cost (~1.4 ms/layer for
  the 2 matmuls; the active 12 MB would be ~24 µs at peak BW). A traceable on-device gather of just
  the 8 selected experts' weight tiles → dense [8,…] batched matmul would cut this hugely. The
  existing opt-in gather path (`QWEN36_SPARSE_DECODE=1`) needs a host readback of top-k indices (not
  traceable), so this needs a gathering sparse_matmul kernel or an indexed expert-weight read.
- **Fused single-step Gated-DeltaNet decode** (≈24%): the recurrent scan + 4-tap conv + projections
  are ~35 tiny M=1 dispatches/layer; a fused decode kernel (cf. the chunked prefill kernel) collapses them.
- lm_head DRAM-sharding is the only clean bandwidth win but caps at ~8% — low priority.

---

## 9. tt-lang (ttl) authoring lessons — READ BEFORE EDITING `tt/ttl_delta.py`

- Pattern: `@ttl.operation(grid=(x,y))` defines the op; `@ttl.datamovement` reader/writer threads
  (`ttl.copy(dram_slice, dfb_block).wait()`); `@ttl.compute` does block ops (`@` matmul, `+ - *`,
  `ttl.math.*`, `.store()`); `ttl.make_dataflow_buffer_like(tensor, shape=(tiles), block_count=2)`
  for L1 buffers. See `tt/ttl_probe.py` and `ttl/tutorials/matmul/step_{1,2}.py`.
- **Materialize matmul chains.** Reassigning `p = p @ p` in a Python loop builds an exponential
  symbolic expression that HANGS the compiler — store each result to a DFB block instead.
- **A matmul must be the SOLE op feeding its accumulator.** No `S*g + kgt@v` or `qg@S + A@v` in one
  `.store()`; store each matmul to its own block, then elementwise-combine in a later stage.
- **Matmul operands must be input or materialized blocks** (not arbitrary in-flight expressions).
- **Per-matmul output tile limit**: a single matmul with K=4 tiles + 8-tile output is rejected
  ("explicitly marked illegal" / "invalid block matmul"). Keep output ≤4 tiles or k-loop tile.
- `fp32_dest_acc_en=False` on the op gives bf16 DST (16-tile capacity); `matmul_full_fp32` (fp32
  accumulate) is default-on but matmul INPUTS are bf16 regardless (Tensix). bf16 is fine for the
  realistic small-magnitude gated-delta data (verified in `tests/analyze_chunk_precision.py`).
- Block values are **multi-use** (one block can feed several ops in one scope) — probe-confirmed.
- A list of transaction handles in a DM thread is rejected ("list elements must be constants") — use
  individual variables + `.wait()` per copy.
- **OPERATIONAL: a ttl run killed mid-compile leaves the device HUNG** → `tt-smi -r` (reset) before
  the next run. Use generous timeouts; first-time kernel compiles can take minutes.

---

## 10. Profiling / device tracing (prefill + decode) — ground-truth per-op timing

How to get **measured** per-op device timing for both phases via the tt-metal device profiler (Tracy),
and view it in TT-NN Visualizer. This is how the decode breakdown in `DECODE.md` §1 was produced.

### 10.1 Build requirement
The profiler must be compiled in: `build/CMakeCache.txt` must have `ENABLE_TRACY:BOOL=ON` (this build
has it). If not, rebuild with `./build_metal.sh --enable-profiler`. The **first** profiled run
JIT-recompiles every kernel with device-zone instrumentation (slow, minutes) — it's disk-cached after.

### 10.2 Capture (one run profiles prefill + eager-decode + traced-decode)
The instrumented script is `tests/prof_decode.py`. It runs three **signpost-bounded** regions —
`prefill_start/stop`, `eager_start/stop`, `trace_start/stop` (signposts via `from tracy import
signpost`) — with warmup calls before each measured region so compiles aren't timed. Run it through the
tracy wrapper (NOT plain python — the wrapper sets `TT_METAL_DEVICE_PROFILER=1` and post-processes):

```bash
QWEN36_LAYERS=4 QWEN36_SEQ=32 ./python_env/bin/python -m tracy -r --op-support-count 6000 \
    models/demos/qwen3_6_a3b/tests/prof_decode.py
```
`-r` = generate the ops report; `--op-support-count` must exceed the op count (a 4-layer run is
~1.7k ops; 40 layers needs more). Use `QWEN36_LAYERS=4` for fast iteration — per-op-TYPE timings
generalize; multiply by layer counts (30 gated-delta, 10 attn, 40 MoE) for the per-token estimate.

### 10.3 Output + parsing
- `generated/profiler/.logs/cpp_device_perf_report.csv` — **per-op device summary** (the one to parse).
  Key columns: `DEVICE KERNEL DURATION [ns]` (pure Tensix compute), `DEVICE FW DURATION [ns]`
  (compute + per-op on-device setup), `OP TO OP LATENCY [ns]` (inter-op gap), `METAL TRACE ID`
  (set ⇒ traced op, empty ⇒ eager). OP NAME is blank here — names live in the official report below.
- `generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv` — the official report **with op names +
  signpost rows + DRAM/FLOPS utilization**. Filter rows between the signpost pairs (`OP TYPE ==
  "signpost"`, `OP CODE == prefill_start`…). (If the host↔device join ever asserts, parse the cpp CSV
  directly and ignore the official one.)

### 10.4 Eager vs traced — what each tells you
- **Eager** region: per-op kernel time + **host-dispatch** gaps. Decode eager is ~80% host gaps — that
  is the overhead trace removes; it does not reflect the traced path.
- **Traced** region: same kernels, but gaps are the **on-device** inter-op latency that remains in a
  trace (the only overhead op-fusion can attack). Sum kernel vs sum (FW−kernel) vs gaps to split
  compute / setup / gap.
- **Caveat:** instrumentation inflates absolutes (~1.8× — a 4-layer traced step reads ~16 ms profiled
  vs ~9 ms real), and inflates overhead more than kernel time, so the real path is somewhat *more*
  compute-bound than the profiled split. Trust ratios and op *ranking*, not absolute µs.
- **Reading utilization (visualizer):** a `>100% DRAM` on a sparse_matmul is the signature that it is
  skipping most of its nominal (full-256-expert) reads — i.e. it is scan/overhead-bound, not
  bandwidth-bound. `SLOW`-flagged dense matmuls show their true DRAM%/FLOPS% (e.g. 23%).

### 10.5 TT-NN Visualizer
`ttnn-visualizer` is in `./python_env/bin`. It has two **independent** data sources:
- `--performance-path generated/profiler/reports/<ts>` — reads the profiler **CSV directly** (no DB
  needed); gives the per-op timing / eager-vs-traced / utilization view. **This is all you need for
  overhead analysis** and works from the capture above. Scope to a signpost region and sort by
  *Device Time* (not Total %, which is wall-time-incl-gap and gets dominated by one-time compile stalls).
- `--profiler-path <ttnn report dir>` — expects a **prebuilt `db.sqlite`** (memory/op-graph/buffer
  views). The visualizer does NOT build this from the CSV; TTNN writes it during a logged run.
```bash
./python_env/bin/ttnn-visualizer --performance-path generated/profiler/reports/<ts> --port 8000
```

### 10.6 Generating the memory/op-graph DB (optional, for `--profiler-path`)
`tests/gen_ttnn_report.py` produces `generated/ttnn/reports/<name>/db.sqlite`. It must run **eager**
(the graph/buffers are captured at dispatch, not in a trace). Gotcha: `enable_logging` must be ON **at
load** (it initializes the report path), but the graph + detailed-buffer reports must be toggled on
**only around the decode step** via `ttnn.manage_config` — leaving them on during the model build
snapshots the weight tensors and dies with `parse_py_tensor: ndarray_import failed`. Also note most
decode ops are C++ FastOperations, so the op-graph DB is sparser than the perf CSV.
```bash
TTNN_CONFIG_OVERRIDES='{"enable_logging": true, "report_name": "qwen36_decode", \
    "enable_fast_runtime_mode": false}' QWEN36_LAYERS=4 \
    ./python_env/bin/python models/demos/qwen3_6_a3b/tests/gen_ttnn_report.py
```

### 10.7 Discipline
A profiled run killed mid-compile leaves the device HUNG → `tt-smi -r` before the next run (§9). Use
generous timeouts; the E=256 MoE build + first instrumented compile is minutes. `tests/bench_decode.py`
(real model, no profiler) is the before/after loop for ms/token — use `QWEN36_LAYERS=4` (~100 s load)
for iteration and `=40` for the authoritative figure; reserve the profiler for per-op attribution.

---

## 11. Pointers / provenance

- Plan file: `~/.claude/plans/review-this-codebase-it-dreamy-minsky.md`.
- Memory (background notes): `~/.claude/projects/.../memory/qwen36-*.md`.
- HF reference sources pulled to `/tmp/modeling_qwen3_{next,5}.py`, `/tmp/modular_*` (re-fetch from
  `github.com/huggingface/transformers/main/src/transformers/models/...` if gone).
