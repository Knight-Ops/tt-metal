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

Weights are expected at `~/models/qwen36` (override with `QWEN36_CKPT`). Use the tt-metal venv.

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
OpenAI HTTP API for interactive testing. It is single-batch and **greedy** (decode
bakes the argmax on-device), so `temperature` / `top_p` / `seed` are accepted for
compatibility but ignored. Since the model is autoregressive it supports real
token-by-token SSE streaming (`"stream": true`), unlike the gemma4 diffusion server.

Runtime deps (`fastapi`, `uvicorn`, `loguru`) are already present in the tt-metal
python_env; nothing extra to install.

```bash
# launch (opens the 1x1 mesh directly, like demo.py); trace path on by default
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

`demo/generator_vllm.py` is a **scaffold** of the vLLM `Qwen36ForCausalLM` wrapper
that tt-inference-server would load. It imports cleanly under a vLLM environment and
makes the contract explicit, but its forward / kv-cache methods raise
`NotImplementedError`. The test server above is enough for interactive use; full
vLLM serving is blocked on the barriers below (the same list lives in the scaffold's
docstring, keyed to the method that needs each one closed):

1. **Batched / multi-user.** `TtModel` is batch-1 (input `[1, T]`); vLLM drives
   `max_num_seqs > 1`. A batch dimension must be threaded through prefill + decode.
2. **Paged KV cache.** Attention layers use a contiguous `[1, n_kv, max_seq, hd]`
   cache updated at a scalar position (`paged_update_cache`); vLLM owns a paged cache
   and passes per-layer `page_table` / block tables the model must consume.
3. **Linear-attention has no KV-cache spec (biggest blocker).** `config.layer_types`
   contains `linear_attention` (Gated DeltaNet — recurrent state, not attention KV).
   The hybrid base's `get_kv_cache_spec` only accepts `sliding_attention` /
   `full_attention` and raises on `linear_attention`; these layers need a recurrent
   ("Mamba-style") state spec that the TT vLLM plugin must support. Confirm plugin
   support before committing to the rest.
4. **Sampling.** Decode bakes greedy argmax on-device and returns only the next token
   id; vLLM expects host-side sampling from logits (or a verified on-device-sample
   contract). A logits-returning decode path is needed.
5. **Generator-base mismatch.** Stock wrappers delegate to `prefill_forward_text` /
   `decode_forward` on a `tt_transformers.Transformer`; `TtModel` implements neither,
   so the bridge must be written against this model.

Once those are closed, register `TTQwen3_5MoeForConditionalGeneration` →
`models.demos.qwen3_6_a3b.demo.generator_vllm:Qwen36ForCausalLM` in the tt-vllm-plugin
`ModelRegistry`, and add an `ImplSpec` + Blackhole/P150 `ModelSpec`
(`InferenceEngine.VLLM`, `override_tt_config={"optimizations": "performance"}`) in the
tt-inference-server workflows.

### Useful env vars

| var | default | effect |
|-----|---------|--------|
| `QWEN36_CKPT` | `~/models/qwen36` | checkpoint dir |
| `QWEN36_LAYERS` | 40 | number of decoder layers to build |
| `QWEN36_MAX_SEQ` | 512 | KV/state cache length |
| `QWEN36_SERVER_HOST` | `0.0.0.0` | server bind host (`server.py`) |
| `QWEN36_SERVER_PORT` | 8000 | server port (`server.py`) |
| `QWEN36_SERVER_TRACE` | 1 | `0` uses the eager decode path (no trace capture) |
| `TT_CACHE_PATH` | `<ckpt>/tt_weight_cache` | weight-cache dir |
| `QWEN36_WEIGHT_CACHE` | 1 | `0` disables the on-disk weight cache |
| `QWEN36_FUSED_PREFILL` | 1 | `0` falls back to the sequential recurrent scan |
| `QWEN36_DELTA_IPLUSL` | 1 | `0` uses the fp32 doubling-product inverse |
| `QWEN36_SPARSE_DECODE` | 0 | `1` uses the gather-top-k decode path (host sync) |
| `QWEN36_LMHEAD_BF4` | 1 | `0` keeps lm_head in BFP8 |

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
| **+ fused gate+up MoE sparse_matmul** | **9.11** |

≈17.5× faster decode, and the full model generates coherently:
`"The capital of France is"` → `" Paris, a city renowned for its iconic landmarks such as the"`.

Each step is PCC-gated; trace is verified correct (traced tokens == eager, `tests/test_trace.py`).
Decode path = on-device Gated-DeltaNet recurrent scan + state cache, fixed-shape KV cache +
`sdpa_decode`, `ttnn.sparse_matmul` MoE (gate+up fused → 2 launches; 8 of 256, no host sync). The
whole step is **self-contained on device**: on-device greedy `argmax` (only the next token id comes
back, not the 248k-vocab logits), RoPE indexed from a device table, position advanced with
`ttnn.plus_one`, and an O(1) `paged_update_cache` KV write — all captured as one trace and replayed
per token (host does only `execute_trace` + a 1-element readback).

Profiling the 110 ms/token (real dims, per layer-type): **MoE ≈65% (dispatch-bound — the two
sparse_matmuls each scan all 256 expert slots), Gated-DeltaNet ≈24%, lm_head ≈8%, attention ≈4%.**
The decode is overhead/dispatch-bound, not weight-bandwidth-bound, so the next levers are op-count
reduction (a true 8-of-256 expert-weight gather to avoid the 256-slot scan; a fused Gated-DeltaNet
decode step) rather than DRAM-sharded matmuls.

**Prefill: fused chunked Gated-DeltaNet kernel (default on; `QWEN36_FUSED_PREFILL=0` disables).** The
sequential recurrent scan for the 30 linear layers is replaced by the chunked delta-rule — ttnn
per-chunk prep batched over all 32 heads + a fused tt-lang `_chunk_state` kernel that runs all heads in
one launch (8×4 grid, head-dim 128) + `[Dk×Dv]` state carry across chunks. Warm prefill: **40-layer seq
256, 26.4 → 149 tok/s (~5.65×)**; 4-layer microbench ~11.7× (same kernel dims). Matches the chunked
reference (PCC ≥ 0.9995), decode is unaffected, and 40-layer generation stays coherent. The cheap
`T≈I+L` inverse is the default (`QWEN36_DELTA_IPLUSL=0` selects the fp32 doubling product). The kernel
compiles once per shape (~70 s, then disk-cached). MoE prefill still uses the dense (all-expert) path.

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
