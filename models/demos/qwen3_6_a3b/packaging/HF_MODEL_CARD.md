<!-- Source of truth for the model card published at the ROOT of
     https://huggingface.co/Adartras/qwen3.6-a3b-blackhole (README.md).

     Not staged into the bundle: `tt-kernel push` writes vllm_bundle/ + tt_kernel_manifest.json and
     never touches the card, so edit here and upload separately:
         hf upload Adartras/qwen3.6-a3b-blackhole \
             models/demos/qwen3_6_a3b/packaging/HF_MODEL_CARD.md README.md --repo-type model

     Keep the frontmatter to `tags:` ONLY. `tt-kernel push` calls hub.tag_repo(), which reloads the
     card and replaces card.data with ModelCardData(tags=...) -- it merges tags but DROPS any other
     frontmatter key (license, base_model, pipeline_tag). The body is preserved. -->
---
tags:
- tenstorrent
- tt-metal
- ttnn
- blackhole
- tt-kernel
- vllm
- qwen3
- moe
- speculative-decoding
---

# Qwen3.6-35B-A3B on Tenstorrent Blackhole (p150)

A tt-kernel **vLLM bundle**: the ttnn model implementation and vLLM
adapter for `Qwen/Qwen3.6-35B-A3B`, running on a single Blackhole p150. This repo holds **code, not
weights** — weights are pulled from [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
on first serve. No precompiled kernel cache is shipped; the plugin JITs at warmup.

Qwen3.6-35B-A3B is a hybrid: 40 decoder layers, of which **30 are gated-delta (linear attention)** and
10 are full attention, each with a 256-expert MoE (top-8) plus a shared expert. That hybrid structure
drives most of what is unusual below.

```bash
tt-kernel serve Adartras/qwen3.6-a3b-blackhole
# OpenAI-compatible endpoint on http://localhost:8000
```

> **Updating:** `tt-kernel serve` reuses an already-installed bundle if its directory exists — it does
> **not** compare revisions against the Hub. After a new push, force a refresh:
> ```bash
> tt-kernel rm Adartras/qwen3.6-a3b-blackhole && tt-kernel serve Adartras/qwen3.6-a3b-blackhole
> ```
> Also check `TT_KERNEL_BUNDLES_DIR` — if it points at an old pull, that copy wins.

## Requirements

| | |
|---|---|
| Hardware | 1 × Blackhole **p150** (single device, `1x1` mesh) |
| Platform | `ttnn >= 0.77` — built and measured against `0.77.1.dev0+g9f9cd4fd590` |
| Serving | Tenstorrent vLLM fork + `vllm-tt-plugin` |
| Extra | **`ttl`** — an undeclared platform dependency of the gated-delta kernels; it must be importable and co-versioned with `ttnn` |
| Precision | BFP4 routed experts (default). bf8 experts do **not** fit one p150 |

## Measured performance

All numbers: single p150, 40 layers, BFP4 experts, traced decode, greedy unless stated.

### Through this bundle (vLLM)

`llama-benchy 0.4.0`, `max_num_seqs=1`, concurrency 1:

| test | throughput | TTFT |
|---|--:|--:|
| `pp4096` (prefill) | **782.3 tok/s** | 5254 ms |
| `tg1024` (decode) | **32.5 tok/s** (peak 34.0) | — |

Short prompts are TTFT-dominated rather than throughput-bound — measured 610–856 ms TTFT for 24–220
token chat-templated prompts.

### Model-level decode (`tests/bench_decode.py`)

Device time plus the single-token readback, i.e. the ceiling a serving loop can approach:

| config | ms/token | tok/s |
|---|--:|--:|
| greedy | 28.65 | **34.9** |
| sampling (T=0.6, top_k=20, top_p=0.95, presence=1.5) | 29.67 | 33.7 |

The vLLM figure above (32.5) sits just under this, as expected — the difference is engine and host
overhead, not model time.

### Speculative decode (MTP) — up to **1.25×**, not available through vLLM

| acceptance rule | tokens/round | ms/token | tok/s | vs. its baseline | output distribution |
|---|--:|--:|--:|--:|---|
| greedy (`temperature=0`) | 2.27 | 23.00 | **43.5** | **1.25×** | exact |
| `exact` (sampled) | 2.06 | 26.71 | 37.4 | **1.11×** | **provably unchanged** |
| `lenient` (0.5) | 2.18 | 25.23 | 39.6 | 1.18× | mean TV 0.052 |
| `relaxed` | 2.25 | 24.41 | 41.0 | 1.22× | mean TV 0.088 |

`gamma=2` (K=3), `tests/bench_mtp.py`, 40 layers, one prompt. Round cost 52.1 ms greedy / 55.0 ms
sampled. The sampled rows are a *paired* comparison — every rule scored on identical
target-distribution / draft pairs — because each rule otherwise generates its own text and the
between-sequence variance swamps the difference between rules. On a higher-acceptance prompt `exact`
reaches 1.19×.

**Read the two columns together.** `exact` is rejection sampling: because the draft is an `argmax`, the
draft distribution is a point mass and the accept probability collapses to `p(x̂)`, so the emitted stream
is *exactly* what the model would have sampled — verified by χ² tests over the acceptance rule. The
other rules buy tokens by accepting drafts the target rated unlikely; "mean TV" is the exact
per-token total-variation distance from the true distribution (`|a − p(x̂)|`, validated against
simulation), so the trade is measured rather than assumed. `relaxed` decides *deterministically*, so it
pays the largest divergence its decision allows at every position; `lenient` is exact rejection sampling
against a *tempered* target, so its divergence is bounded and tunable (`lenience=1.0` **is** `exact`).
Neither has been gated by a generative accuracy benchmark.

**Acceptance is prompt-dependent** — 2.04 to 2.70 tokens/round across prompts — and break-even is
≈1.9 tokens/round, so on some prompts speculation is a small *loss*. The standalone server therefore
calibrates the plain decode rate at warmup and automatically disengages if speculation falls behind.

## Speculative decoding is standalone-server only

`tt/mtp.py` and `tt/mtp_sampling.py` **are** in this bundle, but nothing in the vLLM path calls them,
and setting `QWEN36_MTP` under vLLM does nothing. Three reasons, in increasing order of difficulty:

1. `vllm-tt-plugin/platform.py` hard-asserts `not vllm_config.speculative_config`
   ("Speculative decoding is not yet supported for TT backend").
2. The plugin's model runner assumes **one token per sequence per step**; verification needs a
   K+1-token query per sequence (chunked prefill is also disabled on the TT backend).
3. The real blocker: the 30 gated-delta layers carry a **running recurrent state**, not a positional KV
   cache. vLLM rolls back rejected tokens by not writing KV — which cannot express "advance this
   layer's accumulator by exactly *n* accepted tokens". A batched runner also needs a *per-sequence*
   accept count, which the current single-scalar rollback does not provide.

vLLM upstream does have the framework (`v1/spec_decode/`, `method: "mtp"`, a host rejection sampler,
and even a `qwen3_5_mtp.py` head), so this is wiring rather than invention — but the wiring belongs in
the plugin, not in this bundle.

### Running the MTP server

The standalone server is **not part of this bundle** — it lives in the tt-metal checkout that produced
it, at `models/demos/qwen3_6_a3b/demo/server.py` (needs `fastapi`, `uvicorn`, `transformers`). From
that checkout:

```bash
QWEN36_MTP=greedy_only \
QWEN36_LAYERS=40 \
QWEN36_MAX_SEQ=8192 \
python models/demos/qwen3_6_a3b/demo/server.py
```

It exposes the same OpenAI-compatible `/v1/chat/completions` and `/v1/completions`, one request at a
time:

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"Explain speculative decoding in two sentences."}],
  "max_tokens":256, "temperature":0.6, "top_k":20, "top_p":0.95, "presence_penalty":1.5}'
```

| variable | default | effect |
|---|---|---|
| `QWEN36_MTP` | `0` | `greedy_only` speculates on `temperature==0` requests (**recommended**); `1` also on sampled requests |
| `QWEN36_MTP_GAMMA` | `2` | draft depth. Measured optimum — `>=4` is measurably *worse* |
| `QWEN36_MTP_ACCEPT` | `exact` | `exact` (distribution-preserving) / `lenient` (+`QWEN36_MTP_LENIENCE`, default 0.5) / `relaxed` |
| `QWEN36_MTP_CHECK_ROUNDS` | `24` | re-check interval for the break-even guard; `0` disables it |
| `QWEN36_MAX_SEQ` | `8192` | KV / recurrent-state cache length |
| `QWEN36_EXPERT_DTYPE` | `bf4` | routed-expert precision |

`QWEN36_MTP` selects **which requests speculate** — it never changes sampling behaviour. A
`temperature=0.6` request under `greedy_only` is served by the normal sampled path with full
temperature / top-k / top-p / presence support; it simply forgoes the speedup.

Requires a checkpoint that ships the `mtp.*` head (Qwen3.6-35B-A3B does — 19 tensors, ~845M params, a
single extra decoder layer sharing `embed_tokens` and `lm_head`).

## Accuracy

Correctness is PCC-driven per module against a standalone reference that matches HuggingFace at
PCC 1.0. **No end-to-end benchmark score is published for this build** — the MMLU-Redux harness lives
in `evaluation/` and `published_scores.json` is still empty. Treat the accuracy cost of BFP4 experts as
unquantified.

Two numerical regimes are worth knowing:

- **Speculative output is not bit-identical to non-MTP decode.** Verify runs a different kernel path
  (K-row projections, chained recurrence, indexed-gather MoE); measured logit PCC 0.981–0.999 with no
  compounding drift. Greedy MTP therefore tracks greedy baseline until the first near-tie. This is the
  same regime the project already accepts for sparse-vs-dense MoE and chunked-vs-single prefill.
- **`exact` sampled MTP is distribution-preserving** with respect to that verify path, and is
  χ²-gated.

## Bundle contents

`vllm_bundle/models/qwen3_6_a3b/` — the vLLM adapter (`generator_vllm.py`) plus the ttnn model:
`tt/model.py`, `gated_delta.py`, `ttl_delta.py`, `moe.py`, `moe_gather.py`, `attention.py`,
`decoder.py`, `rms_norm.py`, `mtp.py`, `mtp_sampling.py`, `model_config.py`, `load_checkpoints.py`, and
a torch reference. `tt_kernel_manifest.json` at the repo root carries a per-file `sha256`.

Exactly **two** modules resolve from the serve host, both genuine tt-metal code:
`models.common.lightweightmodule` and `models.tt_transformers.tt.generator`. Everything else is
vendored, and the stager fails the build if anything else tries to host-resolve.

Vendored as `models/qwen3_6_a3b/`, **not** `models/demos/qwen3_6_a3b/`: `models/` is a PEP 420
namespace package, so Python merges it across `sys.path` — which lets the host's `models.common` and
this bundle's package coexist. A vendored `models/demos/...` would lose to the host's copy, because
`tt-kernel serve` prepends the tt-metal checkout while the plugin appends the bundle.

## Known limitations

- **Single user.** `max_num_seqs=1`. A batched decode path exists in the model (~4.5× aggregate at
  batch 32 in a separate harness) but continuous batching is not wired into this serving path — and it
  partly conflicts with speculation, which pays most when the device is *underutilised*.
- **Chunked prefill** is unsupported on the TT backend; `max_num_batched_tokens` is bumped to
  `max_model_len`.
- **vLLM speculative decoding** is unsupported (see above).
- **`min_p` / `repetition_penalty`** are accepted but not implemented in the standalone server's
  on-device sampler (no-ops at their defaults).
- **Shutdown can hang.** `close_mesh_device` can take minutes with a large trace region resident.
  Prefer `SIGTERM` and wait; a `SIGKILL` mid-device-call wedges the board and needs `tt-smi -r`.

## License

Model code Apache-2.0. Weights are governed by the
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) license.
