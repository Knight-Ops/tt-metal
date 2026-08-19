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
models/autoports/qwen3_6_a3b/          <- vendored: OUR model, carrying our fixes
  generator_vllm.py                       (the adapter, addressed as a dotted path)
  tt/…  reference/…
```

`main_class` is `models.autoports.qwen3_6_a3b.generator_vllm:Qwen36ForCausalLM`.

Two rules make this work:

- **Vendor our model, host-resolve the platform.** `models.common.*` and
  `models.tt_transformers.*` (the `Generator` base class) stay host-resolved — upstream tt-metal
  code we do not modify, so depending on the host for them is as safe as depending on it for
  `ttnn`. Vendoring the `Generator` import alone would pull in 32 extra modules / ~890 KB to
  inherit one base class.
- **`autoports/`, not `demos/`.** `models/` has no `__init__.py` anywhere, so it is a PEP 420
  namespace package that Python *merges* across `sys.path` — which is how the host's
  `models.common` and our `models.autoports.qwen3_6_a3b` coexist. The vendored path must not
  collide with one the host already provides: a vendored `models/demos/qwen3_6_a3b` would lose,
  because `tt-kernel serve` *prepends* the tt-metal checkout to `PYTHONPATH` while the plugin
  *appends* the bundle folder. `autoports/` does not exist in tt-metal, so nothing shadows it.
  Never write `models/__init__.py` or `models/autoports/__init__.py` into the bundle — that
  breaks the merge.

`vllm_metadata.json` is **not** staged: with a v4 `--manifest`, tt-kernel renders it on pull from
the manifest and overwrites anything shipped.

## Push & serve
```bash
tt-kernel push <ns>/qwen3.6-a3b-blackhole --backend vllm \
  --bundle-dir models/demos/qwen3_6_a3b/packaging/vllm_bundle \
  --weights Qwen/Qwen3.6-35B-A3B

tt-kernel serve <ns>/qwen3.6-a3b-blackhole            # pulls, sets EXTRA_MODELS_DIR, launches vLLM
tt-kernel serve <ns>/qwen3.6-a3b-blackhole --print    # print the launch command instead
```
`tt-kernel serve` runs the bundle's `launch.command` verbatim with `EXTRA_MODELS_DIR=<bundle
parent>` + the manifest's `launch.<machine>.env`. Kernels-less: the plugin JITs at first-run warmup. Needs the TT vLLM
fork/plugin on the serve host (present here per `tt-kernel doctor`); **`tt-api` is NOT needed**
(that's only the dispatch backend). Then hit `POST http://localhost:8000/v1/chat/completions`.

## Deployment sizing — concurrency & context (the two knobs)

Both are set in `vllm_metadata.json`'s `launch` and are tunable per deploy without code changes:

- **Concurrency (slots)** = `--max-num-seqs N` in `launch.command` (shipped bundle sets `1`). Max requests decoded
  concurrently. `N == 1` routes to the fast single-user V1 path (no batching); `N > 1` uses the batched
  continuous-batching path.
- **Per-slot context** = `QWEN36_MAX_SEQ` in the launch env (code default `8192`; shipped bundle sets
  `131072`). Bounds the **contiguous** KV each slot reserves, capped by vLLM's `--max-model-len`.
  ⚠️ It is only applied when `--max-num-seqs > 1`, so at the shipped `1` it has **no effect** and KV
  is sized from `--max-model-len` alone (262144 if that flag is unset).
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

    weights    ≈ 21 GB                        (bf4 experts + attention/router/embed/lm_head)
    KV cache   = pool × context × 20 KB       (20 KB per token per slot, bf16, k+v)
    GDN state  = pool × ~30 MB                (context-INDEPENDENT — 3/4 of the layers are gated-delta)
    prefill scratch = context × 20 KB         (ONE [1,…]-wide cache, context-sized, pool-independent)

The card is ~33 GB. After ~21 GB of weights, **~12 GB is left for KV + scratch + activations**. The
prefill scratch is a fixed context-sized overhead (2.5 GB @ 128K) regardless of pool. Per-slot KV cost:
**~2.5 GB @ 128K, ~1.3 GB @ 64K, ~0.6 GB @ 32K.**

Collision-free sizing (`pool = 2 × max_num_seqs`):

| context | GB/slot | scratch | max pool that fits | collision-free `--max-num-seqs` (margin = N) |
|--------|--------|--------|-----|-----|
| 128K | ~2.5 | 2.5 | **1** | 1 — single-stream only (V1 path, no CB) |
| 96K  | ~1.9 | 1.9 | 3 | 1 |
| 64K  | ~1.3 | 1.3 | 6 | 3 |
| 48K  | ~1.0 | 1.0 | 8 | **4** (the shipped default: 48K / seqs 4 / margin 4) |
| 32K  | ~0.6 | 0.6 | 12 | 6 |

Push concurrency higher by accepting a smaller margin (fine for staggered traffic): e.g. `64K / seqs 4 /
margin 2` (pool 6) serves 4-way with only rare bursty-load corruption.

⚠️ **Single-P150 reality: 128K + multi-slot batching does NOT fit** — at 128K the weights + one slot's
KV + scratch already ≈ 26 GB, and pool ≥ 3 OOMs. 128K is single-stream only. For collision-free
continuous batching, keep `QWEN36_MAX_SEQ` ≤ 48K at 4-way.

⚠️ **Contiguous KV**: every slot reserves the FULL `QWEN36_MAX_SEQ` up front (even for short prompts),
so the table is worst-case. Real **paged KV** (V2 follow-on) would allocate only the blocks used →
size for *average* context, ~3–10× more effective capacity. Highest-value remaining optimization.

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

- **`allocate_kv_cache` returns `[]`** — vLLM's paged manager is bypassed (caches model-bound); OK at
  B=1. Real paged KV is V2.
- **`decode_forward` ignores `start_pos`/`page_table`** — the model tracks positions internally;
  correct only for B=1 continuing from its own prefill.
- **launch env** — `MESH_DEVICE=P150`, `--block-size 64` confirmed working on this host.

## Prerequisites
Same tt-metal build (`ttnn` + `ttl` 1.1.3) as pushed; Blackhole FW ≥ 19.5.0; weights at
`Qwen/Qwen3.6-35B-A3B`; only bf4 experts fit a 32 GB P150. The adapter imports
`models.demos.qwen3_6_a3b.tt.*`, so the tt-metal model tree must be importable on the serve
host (V1 references it; a self-contained vendored bundle can follow like the dispatch wheel).
