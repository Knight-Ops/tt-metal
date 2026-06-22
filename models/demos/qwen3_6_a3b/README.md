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
tt/common.py               to_tt / from_tt / as_weight helpers
tt/rms_norm.py             TtRMSNorm (1+weight) + TtRMSNormGated
tt/attention.py            full attention (fused q/gate, partial RoPE, qk-norm, KV cache)
tt/gated_delta.py          Gated DeltaNet (on-device recurrent scan + conv/recurrent state cache)
tt/moe.py                  MoE (dense prefill + gather-top-k sparse decode + shared expert)
tt/load_checkpoints.py     streaming safetensors loader (lazy per-tensor)
tt/decoder.py, tt/model.py decoder layer + full hybrid model (prefill + cached decode)
demo/demo.py               runnable text demo + perf harness
tests/                     reference smoke + cross-validation + on-device PCC per module + e2e
```

## Running

Weights are expected at `~/models/qwen36` (override with `QWEN36_CKPT`). Use the tt-metal venv.

```bash
# full 40-layer text demo + perf
QWEN36_LAYERS=40 ./python_env/bin/python models/demos/qwen3_6_a3b/demo/demo.py --seq 64 --gen 16

# reduced layers for quick iteration
QWEN36_LAYERS=4  ./python_env/bin/python models/demos/qwen3_6_a3b/demo/demo.py --prompt "Hello" --gen 8

# tests (on-device PCC gates)
./python_env/bin/python -m pytest models/demos/qwen3_6_a3b/tests/ -p no:cacheprovider
```

The reference cross-validation (`tests/cross_validate_reference.py`) needs `transformers>=5.2`
and is run with a SEPARATE venv, never the tt-metal python_env.

## Correctness

PCC-driven, bottom-up. The standalone reference matches HuggingFace at **PCC = 1.0** for
attention, MoE and GatedDeltaNet; every tt-nn module then matches the reference at **PCC > 0.99
on the Blackhole** (norms, attention, MoE prefill+decode, GatedDeltaNet prefill+decode-cache).
A reduced-layer model runs end-to-end on real weights. (Full 40-layer PCC vs HF is not run on the
dev box — 15 GB RAM, no GPU — so end-to-end correctness rests on per-module PCC + generation.)

## Performance status

Full 40-layer run on a single P150 (text-only, BFP4 experts), measured:

Full 40-layer decode, single P150 (BFP4 experts), measured progression:

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

**Prefill: fused chunked Gated-DeltaNet kernel (`QWEN36_FUSED_PREFILL=1`).** The sequential recurrent
scan for the 30 linear layers is replaced by the chunked delta-rule — ttnn per-chunk prep batched over
all 32 heads + a fused tt-lang `_chunk_state` kernel that runs all heads in one launch (8×4 grid,
head-dim 128) + `[Dk×Dv]` state carry across chunks. Warm prefill (4-layer, seq 256, same kernel dims
as 40 layers): **14.1 → 164.7 tok/s (~11.7×)**; matches the chunked reference (PCC ≥ 0.9995), decode is
unaffected, and 40-layer generation stays coherent. `QWEN36_DELTA_IPLUSL=1` uses the cheap `T≈I+L`
inverse (vs the fp32 doubling product). The kernel compiles once per shape (~70 s, then disk-cached).
MoE prefill still uses the dense (all-expert) path.

Implemented (Phase B), all PCC-gated:
- On-device Gated-DeltaNet recurrent scan (rank-1 update = batched matmuls) + conv/recurrent state cache.
- Fixed-shape KV cache + `sdpa_decode` for attention (replaces variable-length concat).
- Cached single-token decode loop; dense MoE decode (no host sync, trace-compatible) by default
  (`QWEN36_SPARSE_DECODE=1` for the gather path).

`sdpa_decode` is tuned (`k_chunk_size=128`) so flash-decode L1 stays bounded at any context length
(validated to `max_seq=512`); the KV write is now O(1) (`ttnn.experimental.paged_update_cache` at a
tensor position, page_table=None), replacing the old O(max_seq) masked write.

Remaining levers (prefill-focused):
1. **Fused chunked Gated-DeltaNet kernel — DONE (`QWEN36_FUSED_PREFILL=1`, see above).** The chunked
   delta-rule (`@ttl.operation` `_chunk_state`: intra-chunk `q·kᵀ·decay`, the `(I−L)⁻¹` solve via a
   log-depth blocked inverse, intra/inter contributions, `[Dk×Dv]` state carry) is built in tt-lang
   and integrated into `tt/gated_delta.py` prefill, multi-head over an 8×4 grid at head-dim 128.
   ~11.7× warm prefill. Follow-ups: warm full-40-layer TTFT benchmark; parallelize chunks across the
   idle cores; fold the ttnn per-chunk prep into fewer launches.
2. **`ttnn.sparse_matmul` for MoE *prefill*** — prefill still computes all 256 experts; the sparse
   weights/op are already in place (used for decode), so prefill can reuse them with a multi-token
   sparsity layout.
