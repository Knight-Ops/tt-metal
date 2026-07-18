# Qwen3.6-35B-A3B — vLLM bundle (Milestone V1: single-sequence)

A `tt-kernel --backend vllm` bundle that serves Qwen3.6-35B-A3B through the Tenstorrent
vLLM plugin. **This is V1: single-sequence (B=1)** — it stands up the whole vLLM/OpenAI
serving path and de-risks the plugin integration. It does **not** yet do continuous
batching (that's V2 — see [../VLLM_CONTINUOUS_BATCHING.md](../VLLM_CONTINUOUS_BATCHING.md)).

## Contents
- `vllm_metadata.json` — tt-kernel bundle descriptor (`arch`, `main_class`, `hf_weights`, per-machine `launch`).
- `generator_vllm.py` — `Qwen36ForCausalLM(Generator)`: model-owned GDN state, bespoke prefill/decode driving this model's own `forward`/`start_decode`/`decode_forward_logits`/`set_decode_tokens` loop.
- `server_example_tt.py` — thin launcher → vLLM OpenAI API server (TT plugin auto-registers).

## Push & serve
```bash
tt-kernel push <ns>/qwen3.6-a3b-blackhole --backend vllm \
  --bundle-dir models/demos/qwen3_6_a3b/packaging/vllm_bundle \
  --weights Qwen/Qwen3.6-35B-A3B

tt-kernel serve <ns>/qwen3.6-a3b-blackhole            # pulls, sets EXTRA_MODELS_DIR, launches vLLM
tt-kernel serve <ns>/qwen3.6-a3b-blackhole --print    # print the launch command instead
```
`tt-kernel serve` runs the bundle's `launch.command` verbatim with `EXTRA_MODELS_DIR=<bundle
parent>` + `launch.env`. Kernels-less: the plugin JITs at first-run warmup. Needs the TT vLLM
fork/plugin on the serve host (present here per `tt-kernel doctor`); **`tt-api` is NOT needed**
(that's only the dispatch backend). Then hit `POST http://localhost:8000/v1/chat/completions`.

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
- **`launch.env`** — `MESH_DEVICE=P150`, `--block-size 64` confirmed working on this host.

## Prerequisites
Same tt-metal build (`ttnn` + `ttl` 1.1.3) as pushed; Blackhole FW ≥ 19.5.0; weights at
`Qwen/Qwen3.6-35B-A3B`; only bf4 experts fit a 32 GB P150. The adapter imports
`models.demos.qwen3_6_a3b.tt.*`, so the tt-metal model tree must be importable on the serve
host (V1 references it; a self-contained vendored bundle can follow like the dispatch wheel).
