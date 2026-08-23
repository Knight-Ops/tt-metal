# Qwen3.6-35B-A3B — vLLM bundle (Milestone V1: single-sequence)

A `tt-kernel --backend vllm` bundle that serves Qwen3.6-35B-A3B through the Tenstorrent
vLLM plugin. **This is V1: single-sequence (B=1)** — it stands up the whole vLLM/OpenAI
serving path and de-risks the plugin integration. It does **not** yet do continuous
batching (that's V2 — see [../VLLM_CONTINUOUS_BATCHING.md](../VLLM_CONTINUOUS_BATCHING.md)).

## Contents
- `vllm_metadata.json` — legacy in-repo descriptor. The v4 `../qwen36_v4_manifest.json` is the
  source of truth now; tt-kernel renders this file from it on pull.
- `generator_vllm.py` — `Qwen36ForCausalLM(Generator)`: model-owned GDN state, bespoke prefill/decode driving this model's own `forward`/`start_decode`/`decode_forward_logits`/`set_decode_tokens` loop.
- `server_example_tt.py` — thin launcher → vLLM OpenAI API server (TT plugin auto-registers).

### What gets published (staged, not checked in)
`tt-kernel push` ships only the `--bundle-dir` subtree, so publish the output of
`../stage_vllm_bundle.py`, not this folder:

```
README.md
server_example_tt.py
models/qwen3_6_a3b/                    <- vendored: OUR model, carrying our fixes
  generator_vllm.py                       (the adapter, addressed as a dotted path)
  tt/…  reference/…
```

`main_class` is `models.qwen3_6_a3b.generator_vllm:Qwen36ForCausalLM`.

Two rules make this work:

- **Vendor our model, host-resolve the platform.** Exactly two modules are host-resolved —
  `models.common.lightweightmodule` and `models.tt_transformers.tt.generator` (the `Generator`
  base class) — because they genuinely ship with tt-metal, so depending on the host for them is as
  safe as depending on it for `ttnn`. Vendoring the `Generator` import alone would pull in 32 extra
  modules / ~890 KB to inherit one base class.

  This is an **allowlist**, and `stage_vllm_bundle.py` fails the stage on anything else. The
  previous rule ("all of `models.common` is upstream") published a bundle that could not import:
  `tt/moe.py` did `from models.common import moe_gather`, and `moe_gather.py` was *our* file that
  happened to live in `models/common/` on the build machine — not in tt-metal, never staged. It now
  lives in the model package, so the import closure vendors it and no host copy can shadow it.
  `packaging/validate_bundle.py` proves self-containment by importing a staged bundle with the
  model directory out of reach; run it before every push.
- **`models/qwen3_6_a3b/`, not `models/demos/qwen3_6_a3b/`.** `models/` has no `__init__.py`
  anywhere, so it is a PEP 420 namespace package that Python *merges* across `sys.path` — which
  is how the host's `models.common` and our `models.qwen3_6_a3b` coexist. The vendored path must
  not collide with one the host already provides: a vendored `models/demos/qwen3_6_a3b` would
  lose, because `tt-kernel serve` *prepends* the tt-metal checkout to `PYTHONPATH` while the
  plugin *appends* the bundle folder. tt-metal has no `models/qwen3_6_a3b`, so nothing shadows
  it. Never write `models/__init__.py` into the bundle — that breaks the merge.

`vllm_metadata.json` is **not** staged: with a v4 `--manifest`, tt-kernel renders it on pull from
the manifest and overwrites anything shipped.

## Push & serve
```bash
# --bundle-dir must be the STAGED directory, NEVER packaging/vllm_bundle itself: the checked-in
# folder holds only the adapter, so pushing it publishes a bundle with no model code. Both steps
# below are load-bearing -- the stager fails on host-only imports and the validator proves
# self-containment by import. The v4 bundle shipped broken because neither check existed.
python models/demos/qwen3_6_a3b/packaging/stage_vllm_bundle.py --out /tmp/qwen36_bundle
python models/demos/qwen3_6_a3b/packaging/validate_bundle.py  --bundle /tmp/qwen36_bundle

tt-kernel push <ns>/qwen3.6-a3b-blackhole --backend vllm \
  --manifest models/demos/qwen3_6_a3b/packaging/qwen36_v4_manifest.json \
  --bundle-dir /tmp/qwen36_bundle

tt-kernel serve <ns>/qwen3.6-a3b-blackhole            # pulls, sets EXTRA_MODELS_DIR, launches vLLM
tt-kernel serve <ns>/qwen3.6-a3b-blackhole --print    # print the launch command instead
```
`tt-kernel serve` runs the bundle's `launch.command` verbatim with `EXTRA_MODELS_DIR=<bundle
parent>` + the manifest's `launch.<machine>.env`. Kernels-less: the plugin JITs at first-run warmup. Needs the TT vLLM
fork/plugin on the serve host (present here per `tt-kernel doctor`); **`tt-api` is NOT needed**
(that's only the dispatch backend). Then hit `POST http://localhost:8000/v1/chat/completions`.

## Deployment sizing — concurrency & context (the two knobs)

Both are set in `vllm_metadata.json`'s `launch` and are tunable per deploy without code changes:

- **Concurrency (slots)** = `--max-num-seqs N` in `launch.command` (shipped bundle sets `1`). Max requests
  decoded concurrently. `N == 1` routes to the fast single-user V1 path (no batching); `N > 1` uses the
  batched continuous-batching path.

  ⚠️ **Leave this at 1 unless you expect ~6+ concurrent requests.** MEASURED 2026-08-23,
  `tests/bench_batched_decode.py`, 40 layers traced:

  | B | step ms | per-user tok/s | aggregate tok/s | vs B=1 |
  |--:|--:|--:|--:|--:|
  | 1 | 25.27 | **39.6** | 39.6 | 1.00x |
  | 2 | 137.17 | 7.3 | 14.6 | **0.37x** |
  | 4 | 140.64 | 7.1 | 28.4 | **0.72x** |

  At B=2 and B=4 the batched path is worse than single-user on **both** axes -- 5.6x the per-token
  latency AND less aggregate throughput. The cause is structural, not tuning: at `B > 1` the MoE
  switches from the sparse-gather path (top-8 experts, 566 MB/step) to **dense-256** (453 MB/layer x 40
  = ~18 GB/step), and the decode graph runs at POOL width regardless of how many slots hold a live
  request. Note B=2 -> B=4 costs almost nothing (137 -> 141 ms): it is one large fixed cost that
  amortizes over users, which is why aggregate keeps climbing (28.4 at B=4, ~157 at B=32) and only
  passes single-user around **B ~ 6**. So idle slots are NOT free -- configuring 4 and using 1 gives you
  7 tok/s, not 40.

  (Caveat on the numbers: this bench drives the scan-based batched GDN via `setup_decode_batch`, not
  `decode_step_traced_batch` with `QWEN36_GDN_BATCH_FUSED=1`, which is 11-28% faster per B. The dense-MoE
  step cost dominates and is identical either way, so the shape of the conclusion holds.)
- **KV budget** — as of 2026-08-23 the batched path (`--max-num-seqs > 1`) uses **paged KV by default**
  (`QWEN36_PAGED_KV`, set automatically). The budget is then an AGGREGATE token count,
  `QWEN36_KV_POOL_TOKENS`, shared by every slot and every attention layer, defaulting to
  `max_num_seqs × QWEN36_CB_TOKENS_PER_SLOT` (8192) — deliberately the same number of tokens the old
  contiguous arrangement pinned. There is **no per-slot context cap** any more: one request may use the
  whole pool, or all of them may share it.
  Set `QWEN36_PAGED_KV=0` to get the old contiguous arrangement back, in which case
  `QWEN36_MAX_SEQ` (code default `8192`; shipped bundle sets `262144`) is again the per-slot reservation.
  ⚠️ Either way this applies only when `--max-num-seqs > 1`; at the shipped `1` the single-user V1 path
  runs unchanged, with a flat cache sized from `--max-model-len` (262144 if that flag is unset).
- **KV precision** = `QWEN36_KV_DTYPE`. **The bundle ships `bf8`** (the model-wide default is `bf16`).
  At this bundle's `QWEN36_MAX_SEQ=262144` it is the single largest decode lever: SDPA re-reads the whole
  cache every step, so BFP8 is worth **−3.58 ms/token at 128K** (1.82× on sdpa-decode, i.e. 96.5% of the
  1.88× byte-ratio ideal) and **halves the cache** (2.66 GiB vs 5.00 GiB for a full 262144-token
  sequence). 128K is the longest position MEASURED (`tests/probe_paged_kv_perf.py`); the term is linear
  in position, so 256K should be roughly twice that -- treat it as extrapolation, not a measurement.
  Set `bf16` to
  revert. It is worth ~nothing below ~8K, which is why it is not the model-wide default — see
  FUTURE_OPTIMIZATIONS.md "Lever 6".
  Accuracy is measured, not assumed: teacher-forced decode logit PCC 0.995 mean / 0.996 median over 48
  steps, **stable** (no accumulation), 44/48 argmax agreement with every flip on a near-tie — on par with
  the conv add-chain this repo already ships on (EXPERIMENTS.md "BFP8 KV cache").
  ⚠️ One thing to know if you enable MTP here: `bf8` turns four `test_mtp_spec_decode.py` **exact-parity**
  assertions red. That is a test property, not a serving risk — greedy speculative decode still emits the
  backbone's argmax, so both the spec and non-spec streams are valid; what fails is the assertion that
  they are token-identical, and the divergence passes that file's own near-tie criterion.
- **Slot margin** = `QWEN36_CB_SLOT_MARGIN` in the launch env (shipped bundle sets `0`). Spare
  slots that absorb the transient where finished requests' replacements are prefilled before the next
  decode's reconcile frees them.

The actual cache width is **`pool = max_num_seqs + margin`** (capped at 32 sampling lanes). ⚠️ **Each
margin slot reserves FULL context too**, so at high context the margin is expensive — `4 + 2 = 6` slots
× 2.5 GB (128K) = 15 GB → OOM. Keep the margin small when context is large.

⚠️ **Collision safety = margin.** A finished request only leaves our slot map at the *next* decode, but
vLLM prefills its replacement in between — so if **K** requests finish on the same step, the pool is
transiently over-subscribed by K. If K > margin, a new request is forced onto a **live** slot and
corrupts it (garbled/derailed output for that request). **Worst case K = max_num_seqs** (all finish at
once — common when many requests share `max_tokens` and start together). So for guaranteed
collision-free batching set **`margin = max_num_seqs`** (`pool = 2 × max_num_seqs`); the shipped
`48K / --max-num-seqs 4 / margin 4` (pool 8) is exactly this. A smaller margin is fine for *staggered*
real traffic (finishes rarely coincide) and saves memory + decode width, at the risk of rare corruption
under bursty/benchmark load.

**Memory model** (measured from the config: 10 of 40 layers are full-attention, GQA with 2 KV heads):

    weights    ≈ 21 GB                          (bf4 experts + attention/router/embed/lm_head)
    KV pool    = QWEN36_KV_POOL_TOKENS × 20 KB   (20 KB per token, bf16 k+v; 10.9 KB at QWEN36_KV_DTYPE=bf8)
    GDN state  = pool_slots × ~30 MB             (context-INDEPENDENT — 3/4 of the layers are gated-delta)
    prefill scratch = GDN only, ~30 MB           (paged: the attention scratch ALIASES the pool)

The card is ~33 GB. After ~21 GB of weights, **~12 GB is left for KV + activations** — call it ~10 GB of
pool once activations are covered, which is **~500K tokens at bf16 or ~900K at bf8**, shared however the
traffic wants them.

What paging changed, stated honestly — it does not create bytes, it removes two worst-case reservations:

1. **The context-sized prefill scratch is gone.** The contiguous path stages each prompt in a separate
   `[1, n_kv, max_seq, hd]` cache and row-copies it into the slot — a fixed **2.5 GB at 128K**, paid
   whatever the pool size. Paged prefill writes straight into the target slot's pages, so that scratch
   aliases the pool and only GDN state is staged.
2. **Per-slot reservation is gone.** Contiguously, *every* slot reserves the full `QWEN36_MAX_SEQ` even
   to serve a 200-token prompt, so capacity had to be sized for worst-case context on every slot. The
   pool is sized for the aggregate the traffic actually occupies, which for realistic mixed-length
   traffic is the predicted **~3–10× more effective capacity** (the ratio of max to average context).

**What ships today, concretely.** `QWEN36_MAX_SEQ=262144` + `QWEN36_KV_DTYPE=bf8` + paged: one
full-length 262144-token sequence is **2.66 GiB** of KV, so weights (~20.3 GiB) + a pool holding one full
sequence ≈ 23 GiB of the ~31.7 GiB usable. Comfortable at `--max-num-seqs 1`. Four *simultaneously*
full-length requests would need 10.6 GiB of pool and land right on the edge — the adapter's pool floor
guarantees any ONE request can reach the advertised context, and beyond that the failure mode is
"block pool exhausted", not corruption. At bf16 the same sequence is 5.00 GiB, which is why bf8 and 256K
were enabled together.

So the old table's ceiling — "128K is single-stream only; keep context ≤ 48K at 4-way" — was a statement
about *reservations*, not about bytes in use. Under paging a 4-way deploy can advertise 128K and serve
it, as long as the concurrent requests do not *simultaneously* need more than the pool; when they do,
the pool raises "block pool exhausted" rather than corrupting anything. Sizing rule:

    QWEN36_KV_POOL_TOKENS ≈ (expected concurrent tokens) × 1.2      # ~20 KB/token bf16, ~11 KB bf8

⚠️ The **collision-safety margin discussion above still applies unchanged** — it is about slot *identity*
(a finished request leaving our slot map one decode step late), not about KV bytes, so paging does not
address it. `QWEN36_CB_SLOT_MARGIN` still wants to be `max_num_seqs` for guaranteed safety; what paging
makes cheaper is precisely that margin, since a spare slot now costs ~30 MB of GDN state instead of a
full context's worth of KV.

⚠️ Slots are released back to the pool when the adapter retires a request (`model.release_slot`), on both
the normal-completion and the over-capacity-eviction paths. Without that the pool leaks one sequence per
completed request; `tests/test_paged_cb_kv.py` asserts the accounting closes.

## Status (verified on the P150, 2026-07-18)

Working end-to-end: launcher (`-m vllm.entrypoints.openai.api_server`), arch registration via
`EXTRA_MODELS_DIR`, multimodal-registry shim (text-only, zero image/video), `prefill_forward`
returning `(logits, rope_deltas)`, and **traced decode** (`capture_decode_logits_trace` /
`decode_step_logits_traced` in `tt/model.py`, driven by the plugin's `enable_trace`). Prefill
~319 tok/s; decode ~16 tok/s after enabling tracing (up from ~2.9 eager).

**Perf tuning in progress — async decode.** `supports_async_decode=True` + `read_from_device=False`
returns on-device logits so the inherited `Generator.read_decode_output` async-`cpu()`s them and
vLLM's async scheduling overlaps per-step CPU work (scheduling/detok/sampling) with the next step's
device forward. Target: recover most of the ~16→~30 tok/s gap while keeping vLLM's **full host
sampler** (logprobs, penalties, guided decoding).

### FALLBACK — on-device sampling (if async overlap underperforms)
To match `server.py`'s ~30 tok/s with certainty, switch to **on-device sampling**: in the adapter
set `supports_sample_on_device=True`, and drive the model's existing validated on-device sampling
trace — `enable_sampling(temperature, top_k, top_p, seed, presence_penalty)` +
`capture_decode_trace()` + `decode_step_traced()` — returning **tokens** instead of logits (the
plugin's device-sampling contract, `perform_device_sampling` → `process_output_decode(is_tokens=True)`).
Tradeoff: bypasses vLLM's host sampler — keeps temp/top_k/top_p/presence/seed (what `server.py`
supports) but loses logprobs, `min_p`, `frequency_penalty`, and guided/structured decoding.

## Remaining assumptions to keep an eye on

- **`allocate_kv_cache` returns `[]`** — vLLM's paged manager is bypassed because the caches are
  MODEL-owned, and that stays true now that the model pages its own KV: our pool, free list and device
  page table live in `tt/paged_kv.py`, and vLLM's block ids are used only as stable request keys. The two
  page tables never have to agree. This is forced by the architecture, not a shortcut — 30 of 40 layers
  carry GDN recurrent+conv state with no `AttentionSpec`, and `model_runner._validate_kv_cache_groups`
  rejects any non-AttentionSpec group, so there is no spec we could hand vLLM for this model.
- **`decode_forward` ignores `start_pos`/`page_table`** — the model tracks positions internally;
  correct only for B=1 continuing from its own prefill.
- **launch env** — `MESH_DEVICE=P150`, `--block-size 64` confirmed working on this host.

## Prerequisites
Same tt-metal build (`ttnn` + `ttl` 1.1.3) as pushed; Blackhole FW ≥ 19.5.0; weights at
`Qwen/Qwen3.6-35B-A3B`; only bf4 experts fit a 32 GB P150. The adapter imports
`models.demos.qwen3_6_a3b.tt.*`, so the tt-metal model tree must be importable on the serve
host (V1 references it; a self-contained vendored bundle can follow like the dispatch wheel).
