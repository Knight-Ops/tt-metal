# Qwen3.6-35B-A3B on Blackhole (tt-nn)

A naive, pure-Python **tt-nn** bring-up of **Qwen/Qwen3.6-35B-A3B** for a single Tenstorrent
**Blackhole P150**, built bottom-up with PCC correctness gates at every stage.

## What this model is

HF architecture `Qwen3_5MoeForConditionalGeneration` (text config `qwen3_5_moe_text`) — it is
architecturally **Qwen3-Next**: a *hybrid* model, not a plain MoE transformer.

- **40 layers**, hidden 2048, vocab 248320.
- **Hybrid mixers** (`layer_types`, `full_attention_interval=4`): **30 linear-attention** (Gated
  DeltaNet) layers + **10 full-attention** layers.
- **Full attention**: 16 q / 2 kv heads, head_dim 256, the output gate is fused into `q_proj`
  (`attn_out * sigmoid(gate)`), per-head qk-norm, partial RoPE (rotary dim 64), `rope_theta=1e7`.
- **Gated DeltaNet**: delta-rule linear attention — short causal conv (k=4) + recurrent state;
  16 key / 32 value heads, head dim 128.
- **MoE**: 256 experts, top-8 + renormalize, `moe_intermediate_size=512`, plus a sigmoid-gated
  **shared expert**.

Scope here is **text-only** (no vision tower, no MTP head).

## Layout

```
reference/qwen3_5_moe.py   standalone torch reference (text path); PCC=1.0 vs HuggingFace
tt/model_config.py         ModelArgs (parses HF config.json; single-P150 BFP4)
tt/common.py               to_tt / from_tt / as_weight helpers (+ on-disk .tensorbin weight cache)
tt/rms_norm.py             TtRMSNorm (1+weight) + TtRMSNormGated
tt/attention.py            full attention (fused q/gate, partial RoPE, qk-norm, KV cache)
tt/gated_delta.py          Gated DeltaNet (on-device recurrent scan + conv/recurrent state cache)
tt/moe.py                  MoE (dense prefill + gather-top-k sparse decode + shared expert)
tt/load_checkpoints.py     streaming safetensors loader (lazy per-tensor; big weights as thunks)
tt/decoder.py, tt/model.py decoder layer + full hybrid model (prefill + cached decode)
demo/demo.py               runnable text demo + perf harness
demo/generate_weight_cache.py  optional one-time weight-cache populate
tests/                     reference smoke + cross-validation + on-device PCC per module + e2e
```

## Running

> **Blackhole firmware ≥ 19.5.0 required.** On older firmware (observed on bundle **18.8.0**,
> eth-fw 1.4.2) the flash `scaled_dot_product_attention_decode` op **deadlocks** — the model builds
> and prefills, then the first decode step hangs forever with the device idle (two Tensix cores stuck
> in a circular-buffer wait). Upgrading to **19.6.0** (eth-fw ≥ 1.8.1) resolves it. Update with
> `tt-flash ≥ 3.6.0`: `tt-flash flash fw_pack-19.6.0.fwbundle` (from `tenstorrent/tt-firmware`), then
> reset. Standalone repro: `tests/repro_sdpa_decode_hang.py` (hangs on affected FW, passes after the
> upgrade). Check your version with `tt-smi -s`.

Weights are expected at `~/models/qwen36` (override with `QWEN36_CKPT`; 72 GB, 26 safetensors
shards). Use the tt-metal venv.

> **Do not upgrade packages in the tt-metal `python_env`** — it is pinned to this build
> (`transformers==4.53.0`, plus `ttnn` and `ttl` 1.1.3). In particular do NOT upgrade
> `transformers` there. The reference cross-validation needs `transformers>=5.2`, so it runs in a
> SEPARATE venv (CPU torch), never the tt-metal `python_env` (see the cross-validation note below).

```bash
# full 40-layer text demo + perf (--trace = fast on-device decode path)
QWEN36_LAYERS=40 ./python_env/bin/python models/demos/qwen3_6_a3b/demo/demo.py --seq 64 --gen 16 --trace

# reduced layers for quick iteration
QWEN36_LAYERS=4  ./python_env/bin/python models/demos/qwen3_6_a3b/demo/demo.py --prompt "Hello" --gen 8 --trace

# tests (on-device PCC gates)
./python_env/bin/python -m pytest models/demos/qwen3_6_a3b/tests/ -p no:cacheprovider
```

The reference cross-validation (`tests/cross_validate_reference.py`) needs `transformers>=5.2`
and is run with a SEPARATE venv, never the tt-metal python_env.

### Weight cache (fast model load)

The converted (quantized + tilized) weights are cached to disk as `.tensorbin` files so they are
not re-derived from the 72 GB HF checkpoint on every load. **It is on by default** — the first run
populates the cache; later runs reuse it. Measured (40 layers, single P150): cold build **~916 s →
warm load ~120 s**, and warm-vs-cold logits are PCC-identical.

```bash
# Optional: pre-populate the cache once (a normal demo/test run also fills it on its first pass).
QWEN36_LAYERS=40 ./python_env/bin/python models/demos/qwen3_6_a3b/demo/generate_weight_cache.py

# Cache to a specific dir (e.g. fast local NVMe) instead of <ckpt>/tt_weight_cache:
TT_CACHE_PATH=/scratch/qwen36_cache QWEN36_LAYERS=40 \
    ./python_env/bin/python models/demos/qwen3_6_a3b/demo/demo.py --prompt "Hi" --gen 8 --trace
```

- **Location:** `$TT_CACHE_PATH` if set, else `<ckpt>/tt_weight_cache/` (~20 GB for 40 layers).
  Big weights (experts, attention + gated-delta projections, embed, lm_head) are cached; tiny
  tensors (norms, RoPE, conv taps) convert instantly and are not. On a cache hit the corresponding
  HF weight is **not read at all**, so the 72 GB safetensors read is skipped too.
- **Disable / hygiene:** `QWEN36_WEIGHT_CACHE=0` turns it off. Cache files are keyed by name + dtype
  + layout (so flipping e.g. `QWEN36_LMHEAD_BF4` is safe) but **not** by checkpoint contents —
  `rm -rf <cache dir>` if you point at a different checkpoint. Any cache failure falls back to a
  normal from-scratch load with one warning.

### Serving (OpenAI-compatible test server)

`demo/server.py` is a lightweight FastAPI server that exposes the model over the
OpenAI HTTP API for interactive testing. It is single-batch. Sampling is honored
**on device**: `temperature` (with `top_k` / `top_p` / `seed`) drives a chunked
`ttnn.topk` + `ttnn.sampling` decode tail (no host-side logit readback, fully
in-trace, ~0.5 ms/token over greedy); `temperature == 0` selects greedy argmax.
The request defaults are Qwen3's recommended "thinking mode" config (`temperature`
`0.6`, `top_k` `20`, `top_p` `0.95`, `presence_penalty` `1.5`). Qwen3 thinking models
should NOT be run greedy (it degrades into repetition), so requests that omit
`temperature` sample at `0.6` — pass `"temperature": 0` only for deterministic greedy.
The first token (from prefill) is always greedy argmax; decode tokens are sampled.
Since the model is autoregressive it supports real token-by-token SSE streaming
(`"stream": true`), unlike the gemma4 diffusion server.

Runtime deps (`fastapi`, `uvicorn`, `loguru`) are already present in the tt-metal
python_env; nothing extra to install. See [`SAMPLING.md`](SAMPLING.md) for the full
sampling reference (on-device design, supported params, and the perf breakdown).

```bash
# launch (opens the 1x1 mesh directly, like demo.py); trace path on by default
# add QWEN36_MTP=1 for speculative decode (needs the checkpoint's mtp.* head) — exact for
# greedy AND sampled requests, see SAMPLING.md "Speculative decode + sampling"
QWEN36_LAYERS=40 ./python_env/bin/python models/demos/qwen3_6_a3b/demo/server.py

# smoke endpoints
curl localhost:8000/health
curl localhost:8000/v1/models

# non-streaming chat
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":16}'

# streaming chat (SSE deltas, ends with `data: [DONE]`)
curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Tell me a joke"}],"max_tokens":32,"stream":true}'

# raw completion
curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt":"Once upon a time","max_tokens":16}'
```

Generation stops at an EOS id, at `max_tokens`, or before the cache overflows
`QWEN36_MAX_SEQ` (an over-long prompt returns HTTP 400). Set `QWEN36_SERVER_TRACE=0`
to use the eager decode path (no per-request trace capture).

### vLLM / tt-inference-server status

The vLLM adapter lives in `packaging/vllm_bundle/generator_vllm.py`, shipped inside a
`tt-kernel --backend vllm` bundle; `packaging/VLLM_CONTINUOUS_BATCHING.md` holds the full plan
and status. **Milestone V1 (single-sequence, B=1) is drafted but hardware-unverified** — every
`VERIFY-ON-DEVICE` note in that file's docstring is an assumption about the plugin contract that
still needs a running P150 plus the TT plugin to confirm.

Closed since the original barrier list:

- **Linear-attention has no KV-cache spec — closed, by not needing one.** The Tenstorrent
  plugin's KV-cache layer hard-rejects any non-`AttentionSpec`, so it can never carry a
  Mamba-style recurrent spec for the 30 `linear_attention` layers. The adapter therefore
  inherits plain `Generator`, defines **no** `get_kv_cache_spec`, and keeps **both** the
  attention KV and the GDN conv/recurrent state in the model's own device buffers, re-zeroed
  per sequence (`_reset_linear_state`). Note this was a *plugin* constraint, not a framework
  one: upstream vLLM V1 does ship a `MambaSpec` and does serve the Qwen3-Next hybrid on GPU.
- **Sampling — closed.** `decode_forward_logits()` returns `[B, vocab]` for host-side
  sampling; on-device sampling stays available via `enable_sampling()`.
- **Generator-base mismatch — closed.** V1 overrides `prefill_forward` / `decode_forward` to
  drive this model's own loop (`start_decode` → `decode_forward_logits` → `set_decode_tokens`)
  rather than the `tt_transformers.Transformer` contract.

Still open:

1. **Paged KV.** Attention layers use a contiguous `[B, n_kv, max_seq, hd]` cache updated at a
   position tensor (`paged_update_cache`); vLLM's block manager is bypassed
   (`allocate_kv_cache` returns `[]`). This is what forces the per-slot context cap at B>1.
2. **Per-slot (rather than synchronized) continuous batching.** The on-device pieces exist —
   batched GDN state and kernel (`decode_step_batch_tt`, 4.52× at B=32 ≈ 157 tok/s), batched
   attention KV, dense MoE for B>1, and `prefill_into_slot` — but batched prefill and the
   scheduler bridge are unfinished, and the per-slot state writes are unverified on device.

Registration is by HF architecture name: the plugin must resolve
`Qwen3_5MoeForConditionalGeneration` to the bundle's class via `EXTRA_MODELS_DIR`. Add an
`ImplSpec` plus a Blackhole/P150 `ModelSpec` (`InferenceEngine.VLLM`,
`override_tt_config={"optimizations": "performance"}`) in the tt-inference-server workflows.

### Memory: KV cache and Gated-DeltaNet state

Worth knowing before setting `QWEN36_MAX_SEQ`, because the hybrid changes the arithmetic.
Only the **10 `full_attention` layers** hold a KV cache; the **30 `linear_attention` layers**
carry a Gated-DeltaNet recurrent state whose size does not depend on context length.

| | per token | at 32K | at 262144 (native) |
|---|---|---|---|
| this model (10 attn layers) | **20 KiB** | 0.67 GB | **5.4 GB** |
| dense-equivalent (all 40 layers) | 80 KiB | 2.7 GB | 21.5 GB |

`20 KiB = 10 layers x 2 (K,V) x n_kv_heads(2) x head_dim(256) x 2 B` (bfloat16) — a **4x**
reduction, and what makes both long context and B=32 fit a 32 GB P150 alongside ~17.5 GB of
BFP4 weights. Decode reads are bounded by position (`cur_pos_tensor`), not by `max_seq`, so a
generous `max_seq` costs DRAM but not tokens/s.

The state is not free, though, and it does not shrink with short prompts: GDN state is
**~90 MiB per user at B=1** (30 x [1 MiB recurrent + tile-padded conv history]),
sequence-independent. Against 20 KiB/token, the crossover is **~4.6K tokens** — below that
context the "cheap" linear layers cost *more* memory per user than the attention layers do. At
B=32 the conv buffers fill their tiles exactly, so per-user state falls to ~33 MiB and the
crossover to ~1.7K tokens. `tests/test_cache_topology.py` pins all of this.

### Useful env vars

| var | default | effect |
|-----|---------|--------|
| `QWEN36_CKPT` | `~/models/qwen36` | checkpoint dir |
| `QWEN36_LAYERS` | 40 | number of decoder layers to build |
| `QWEN36_MAX_SEQ` | 32768 | KV/state cache length, i.e. the hard prompt+generation limit. At 20 KiB/token (see Memory above) 32K is ~0.67 GB of KV; the checkpoint's native 262144 is ~5.4 GB |
| `QWEN36_KV_DTYPE` | `bf16` | attention KV cache precision (`bf16`/`bf8`/`bf4`). `bf8` HALVES the cache (20 -> 10.9 KiB/token) and makes sdpa-decode markedly cheaper as context grows -- measured per attention layer: 83.2 -> 62.4 us at 8K (1.33x), 227.7 -> 137.4 at 32K (1.66x), 796.7 -> 438.8 at 128K (1.82x), i.e. -0.21 / -0.90 / -3.58 ms per token across the 10 attention layers. Accuracy: teacher-forced decode logit PCC mean 0.995 / median 0.996 over 48 steps, **stable, no accumulation**, 44/48 argmax agreement with every flip on a near-tie -- on par with the shipped conv add-chain's own 0.996 gate (EXPERIMENTS.md). Note MMLU **cannot** gate this: prefill-scored evals never read the KV cache |
| `QWEN36_PAGED_KV` | 0 | paged KV: one block pool + a device page table instead of a flat `[1, n_kv, max_seq, hd]` cache per layer. **Memory-neutral for a single user** (`blocks_per_seq == num_blocks` is forced by the kernel, so one sequence cannot be over-subscribed -- see tt/paged_kv.py); the win is ACROSS sequences, which is why the vLLM adapter defaults it ON for `B>1`. Free at decode (paged sdpa-decode is 0.99-1.00x flat at every position; the write costs +0.3-0.8 us) |
| `QWEN36_KV_READ_GRAN` | 128 | tokens of k-range `sdpa_decode` rounds its READ up to. Paged KV must map every block in that rounded range, not just up to `cur_pos`, or the kernel dereferences an unmapped page-table entry and attends over junk — **must equal `Attention._sdpa_pc`'s `k_chunk_size`**; they are one contract, not two knobs. See EXPERIMENTS.md "The bug this shipped with" |
| `QWEN36_KV_BLOCK` | 64 | tokens per KV block. 64 is what `blackhole/qwen36` and `tt_transformers` ship and what the probe measured the page-table indirection as free at |
| `QWEN36_KV_POOL_TOKENS` | `batch x max_seq` | AGGREGATE pool budget in tokens, shared by all slots and all attention layers. The default preserves the flat footprint; lowering it is what actually reclaims memory, and it also becomes the per-sequence cap |
| `QWEN36_CB_TOKENS_PER_SLOT` | 8192 | vLLM adapter only: the per-slot budget it multiplies by `max_num_seqs` to size the pool when `QWEN36_KV_POOL_TOKENS` is unset |
| `QWEN36_MAX_NEW_DEFAULT` | 8192 | `server.py`: generation cap for a request that sends no `max_tokens`. Deliberately independent of `QWEN36_MAX_SEQ` so widening the context does not lengthen runaway generations |
| `QWEN36_SERVER_HOST` | `0.0.0.0` | server bind host (`server.py`) |
| `QWEN36_SERVER_PORT` | 8000 | server port (`server.py`) |
| `QWEN36_SERVER_TRACE` | 1 | `0` uses the eager decode path (no trace capture) |
| `TT_CACHE_PATH` | `<ckpt>/tt_weight_cache` | weight-cache dir |
| `QWEN36_WEIGHT_CACHE` | 1 | `0` disables the on-disk weight cache |
| `QWEN36_FUSED_PREFILL` | 1 | `0` falls back to the sequential recurrent scan |
| `QWEN36_SPARSE_DECODE` | 0 | `1` uses the gather-top-k decode path (host sync) |
| `QWEN36_LMHEAD_BF4` | 1 | `0` keeps lm_head in BFP8 |
| `QWEN36_EXPERT_DTYPE` | `bf4` | routed-expert precision (`bf4`/`bf8`); used by the accuracy sweep in `evaluation/` |
| `QWEN36_EXPERT_DOWN_BF8` | 0 | `1` pins the sensitive routed-expert `down_proj` to bf8 (gate_up stays bf4) — mixed-precision recipe, ~+5GB, still fits a 32GB P150 |
| `QWEN36_MTP` | 0 | speculative decode with the checkpoint's MTP draft head. `greedy_only` = only `temperature==0` requests (**recommended: 1.22×, exact**); `1` = also sampled requests (exact but ~1.02× — see EXPERIMENTS.md). Needs `mtp.*` weights + decode tracing |
| `QWEN36_MTP_GAMMA` | 2 | draft depth (measured optimum; ≥4 is measured *worse*, see MTP.md) |
| `QWEN36_MTP_ACCEPT` | `exact` | acceptance rule for sampled requests: `exact` (distribution-preserving, 1.14×), `relaxed` (typical acceptance, 1.25×, mean TV 0.090), `lenient` (tunable via `QWEN36_MTP_LENIENCE`, 1.21× at mean TV 0.054), `greedy` (parity testing only — biases sampled output toward the argmax) |
| `QWEN36_TRACE_REGION` | 200 (400 with MTP) | trace region, MiB |
| `QWEN36_MOE_ROUTER_FAST` | 1 | decode router as `topk(logits)` + `softmax(top_k)` (instead of `softmax(E)` + topk + sum + divide) **and** a cached constant `sparsity` for the gather path (`sparse_matmul` indexed mode never reads that operand). Mathematically identical; measured 35.0 → **37.4 tok/s/user** (−1.89 ms/token). `0` reverts, for an A/B |
| `QWEN36_GDN_L2NORM_RMS` | 1 | q/k L2 norm as `rms_norm(x, eps/K) * K**-0.5` (2 ops) instead of a 6-op reduction chain. Measured eager prefill T=1024: 887 → **893 tok/s**. `0` restores the chain |
| `QWEN36_GDN_DTYPE` | `bf8` | gated-delta projection precision (`bf16`/`bf8`/`bf4`). `bf4` is −0.45 ms/token and ~0.5 GB less DRAM, and paired MMLU is clean (79.0→80.5%, p=0.508) — but it **fails `test_long_prefill_chunk_invariant`**, so it is not the default. See FUTURE_OPTIMIZATIONS "Lever 4 MEASURED" |
| `QWEN36_ATTN_DTYPE` | `bf8` | attention projection precision. `bf4` is −0.09 ms/token, same caveat as above |
| `QWEN36_SHARED_DTYPE` | `bf16` | MoE router + shared-expert weight precision. Was hardcoded to the activation dtype because `mlp_weight_dtype` was assigned and never read. `bf8` removes 138 MB/token of weight traffic and buys **−0.04 ms, i.e. nothing** (`moe.shared` runs at 21% of DRAM peak, `moe.router` at 3%) — so the default does not spend the perturbation. Useful only for DRAM footprint |
| `QWEN36_WIDE1D_DECODE` | 1 | tuned wide 1D `mcast_in0` program configs on the decode matmuls that otherwise ran on the ttnn AUTO heuristic (`se_gate_up`, `se_down`, attention `wk`/`wv`, gated-delta `w_in_proj`). Measured 37.4 → **39.2 tok/s/user**. `0` reverts every site. `lm_head` has a tuned config too but is NOT in the default site list — see the row below |
| `QWEN36_WIDE1D_SITES` | `se_gate_up,se_down,wk,w_in_proj` | per-site subset of the above, or `all`/`none`. **`lm_head` is omitted by default**: its tuned config measured **0.00 ms** at 40 layers (the isolated 1125.9 -> 1035.7 us win did not survive; it pipeline-overlaps), so the default does not spend an accumulation-order change for nothing. **`gate_w` is also omitted**: it is worth a further −0.70 ms/token but its matmul feeds a `topk`, so its accumulation-order change can flip which expert is 8th (PCC 0.9857 vs a 0.99 gate). Needs an MMLU baseline first — see FUTURE_OPTIMIZATIONS.md |
| `QWEN36_SPARSE_IN0BW_GU` | 32 | MoE `gate_up` sparse-matmul `in0_block_w` (its Kt=64 is not constrained by `down`'s Kt=16) |
| `QWEN36_GDN_VERIFY_CONV_STACK` | 1 | stacked-matmul verify conv (verify 52.5→44.0 ms, PCC 0.9997→0.99998); `0` reverts to the slice loop |
| `QWEN36_GDN_COMMIT_COPY` | 1 | full-accept conv commit as one copy (round −2.6 ms) |
| `QWEN36_GDN_ALIAS_STATE` | 1 | fused decode kernel writes the recurrent state **in place** (S and Snew are the same buffer), deleting the per-layer `ttnn.copy` and its 1 MB scratch. **Bit-identical** (probe_state_alias.py: max\|Δ\|=0 on state and output at real dims); measured 28.44 → 28.16 ms/token (35.2 → 35.5 tok/s), GDN 0.319 → 0.312 ms/op, −30 MB pinned. `0` reverts |
| `QWEN36_GDN_COMMIT_SELECT` | 0 | general commit as selection matmuls: −5.1 ms but breaks sampled acceptance at 40L — see EXPERIMENTS.md |

## Correctness

PCC-driven, bottom-up. The standalone reference matches HuggingFace at **PCC = 1.0** for
attention, MoE and GatedDeltaNet; every tt-nn module then matches the reference at **PCC > 0.99
on the Blackhole** (norms, attention, MoE prefill+decode, GatedDeltaNet prefill+decode-cache).
A reduced-layer model runs end-to-end on real weights. (Full 40-layer PCC vs HF is not run on the
dev box — 15 GB RAM, no GPU — so end-to-end correctness rests on per-module PCC + generation.)

## Performance status

Full 40-layer decode on a single P150 (text-only, BFP4 experts), measured progression:

| path | decode (tok/s/user) |
|------|---------------------|
| initial (host scan, gather MoE, concat KV) | 0.52 |
| on-device scan + caches + fixed KV + dense decode | 1.45 |
| + decode trace capture | 2.09 |
| + ttnn.sparse_matmul MoE decode | 7.28 |
| + self-contained on-device trace (argmax/rope/pos + O(1) KV) | 7.77 |
| + fused gate+up MoE sparse_matmul | 9.11 |
| **+ decode-opt levers** (8-of-256 MoE gather, fused Gated-DeltaNet decode kernel, conv add-chain, DRAM-sharded projections, MoE `in0_block_w`) | **~30** |
| + per-matmul MoE `in0_block_w` (`gate_up`=32) | 34.9 |
| **+ router rewrite** (`QWEN36_MOE_ROUTER_FAST`: topk-on-logits + constant gather sparsity) | **37.4** |
| **+ wide-1D decode matmul configs** (`QWEN36_WIDE1D_DECODE`) | **39.2** |
| **+ speculative decode (MTP, gamma=2, greedy)** | **43.5** |

Speculative decode is measured at **43.5 tok/s/user greedy** (1.25× over the then-current 34.9,
gamma=2, round 52.1 ms; not yet re-measured on top of the 37.4 baseline). Acceptance is prompt-dependent and break-even is ~1.9
tokens/round, so the server measures both rates live and disengages if speculation is losing. For
*sampled* requests it is **distribution-exact at 1.11–1.19×**;
`QWEN36_MTP_ACCEPT=relaxed` gets 1.22× by giving up exactness and `=lenient` 1.18× at half the
divergence, but exact needs no tradeoff. See EXPERIMENTS.md ("Speculative SAMPLING", "The verify pass
was 16% slower than it needed to be") for the full measured picture.

≈58× faster decode than the naive baseline, and the full model generates coherently:
`"The capital of France is"` → `" Paris, a city renowned for its iconic landmarks such as the"`.
The current decode levers and their measured per-layer-type breakdown live in `FUTURE_OPTIMIZATIONS.md`.

Each step is PCC-gated; trace is verified correct (traced tokens == eager, `tests/test_trace.py`).
Decode path = on-device Gated-DeltaNet recurrent scan + state cache, fixed-shape KV cache +
`sdpa_decode`, `ttnn.sparse_matmul` MoE (gate+up fused → 2 launches; 8 of 256, no host sync). The
whole step is **self-contained on device**: on-device greedy `argmax` (only the next token id comes
back, not the 248k-vocab logits), RoPE indexed from a device table, position advanced with
`ttnn.plus_one`, and an O(1) `paged_update_cache` KV write — all captured as one trace and replayed
per token (host does only `execute_trace` + a 1-element readback). Temperature/top-k/top-p sampling
(`demo.py --temperature`, or the server's `temperature` field) swaps the `argmax` for a chunked
`ttnn.topk` + `ttnn.sampling` tail — still fully on device, in-trace, with the RNG seed advanced by
`ttnn.plus_one` so each replayed step draws fresh randomness (deterministic per seed); ~0.5 ms/token
over greedy. topk is chunked into power-of-2 (≤32768) vocab pieces because a single topk over the
full 248k vocab is single-core (~38 ms) while each power-of-2 chunk runs multicore (~0.2 ms).

Decode is **overhead/dispatch-bound, not weight-bandwidth-bound** (measured ~33 ms/token, ~16% DRAM
utilization); the largest pools are the Gated-DeltaNet step and MoE. The 8-of-256 expert-weight gather
and the fused Gated-DeltaNet decode kernel that this analysis motivated have since shipped (both
default-on). The remaining decode levers and the authoritative per-layer-type breakdown live in
`FUTURE_OPTIMIZATIONS.md`.

**Prefill: fused chunked Gated-DeltaNet kernel (default on; `QWEN36_FUSED_PREFILL=0` disables).** The
sequential recurrent scan for the 30 linear layers is replaced by the chunked delta-rule — ttnn
per-chunk prep batched over all 32 heads + a fused tt-lang `_chunk_state` kernel that runs all heads in
one launch (8×4 grid, head-dim 128) + `[Dk×Dv]` state carry across chunks. The fused kernel gave
~5.65× over the 26.4 tok/s recurrent-scan baseline; with the later MoE matmul tuning + prefill tracing,
current 40-layer prompt processing is **~331 tok/s** (see `PREFILL.md`). Matches the chunked
reference (PCC ≥ 0.9995), decode is unaffected, and 40-layer generation stays coherent. The intra-chunk
`(I−L)⁻¹` uses a numerically-stable recursive block inversion (the cheap `T≈I+L` and the fp32
doubling-product approximations were both removed — they drift to gibberish on this model's O(1) `L`).
Note the default eager prefill runs the per-chunk recurrence in fp32/HiFi4 ttnn (`_chunk_state_ttnn`)
for accuracy; the bf16 tt-lang `chunk_state_tt` kernel is the faster-but-approximate traced/opt-in path.
The kernel compiles once per shape (~70 s, then disk-cached). MoE prefill uses the dense (all-expert) path.

Implemented (Phase B), all PCC-gated:
- On-device Gated-DeltaNet recurrent scan (rank-1 update = batched matmuls) + conv/recurrent state cache.
- Fixed-shape KV cache + `sdpa_decode` for attention (replaces variable-length concat).
- Cached single-token decode loop; dense MoE decode (no host sync, trace-compatible) by default
  (`QWEN36_SPARSE_DECODE=1` for the gather path).

`sdpa_decode` is tuned (`k_chunk_size=128`) so flash-decode L1 stays bounded at any context length
(validated to `max_seq=512`); the KV write is now O(1) (`ttnn.experimental.paged_update_cache` at a
tensor position, page_table=None), replacing the old O(max_seq) masked write.

Remaining levers (prefill-focused):
1. **Fused chunked Gated-DeltaNet kernel — DONE (default on; see above).** The chunked
   delta-rule (`@ttl.operation` `_chunk_state`: intra-chunk `q·kᵀ·decay`, the `(I−L)⁻¹` solve via a
   log-depth blocked inverse, intra/inter contributions, `[Dk×Dv]` state carry) is built in tt-lang
   and integrated into `tt/gated_delta.py` prefill, multi-head over an 8×4 grid at head-dim 128.
   ~11.7× warm prefill. Follow-ups: warm full-40-layer TTFT benchmark; parallelize chunks across the
   idle cores; fold the ttnn per-chunk prep into fewer launches.
2. **`ttnn.sparse_matmul` for MoE *prefill*** — prefill still computes all 256 experts; the sparse
   weights/op are already in place (used for decode), so prefill can reuse them with a multi-token
   sparsity layout.
