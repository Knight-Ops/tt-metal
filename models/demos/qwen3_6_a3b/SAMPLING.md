# Sampling (temperature / top-k / top-p / presence-penalty)

Qwen3.6-A3B decode supports **on-device** sampling in addition to greedy argmax. Everything runs
inside the captured decode trace — no host-side logit readback — so sampling keeps the same
"only the next token id comes back to host" design as greedy and adds only a few ms/token.

## TL;DR

- **Greedy is the default and is unchanged** (`temperature == 0` → on-device `ttnn.argmax`, ~0 added cost).
- **Sampling** (`temperature > 0`) swaps the argmax for a chunked `ttnn.topk` + `ttnn.sampling` tail.
- Supported on device: **temperature, top-k, top-p, presence-penalty**, plus a deterministic **seed**.
- Not implemented: **min_p, repetition_penalty** (accepted by the server for API compatibility; their
  recommended values `0.0` / `1.0` are no-ops, and a warning is logged if set otherwise).

## Usage

### Server (`demo/server.py`)
The request defaults are Qwen3's recommended **"thinking mode"** config:

| field | default | applied? |
|---|---|---|
| `temperature` | `0.6` | yes (on device; `0` = greedy) |
| `top_k` | `20` | yes (clamped to (0, 32]) |
| `top_p` | `0.95` | yes |
| `presence_penalty` | `1.5` | yes (on device) |
| `min_p` | `0.0` | no (accepted; no-op at 0.0) |
| `repetition_penalty` | `1.0` | no (accepted; no-op at 1.0) |
| `seed` | `null` → `0` | yes (deterministic per seed) |

Because the default `temperature` is `0.6` (Qwen3's thinking-mode recommendation), requests that omit
it **sample** rather than run greedy. Pass `"temperature": 0` for deterministic greedy.

```bash
curl localhost:8000/v1/chat/completions -d '{
  "messages":[{"role":"user","content":"Explain attention in one paragraph."}],
  "max_tokens":128, "temperature":0.6, "top_k":20, "top_p":0.95, "presence_penalty":1.5, "seed":1234
}'
```

### CLI demo (`demo/demo.py`)
```bash
# greedy (default)
QWEN36_LAYERS=40 python models/demos/qwen3_6_a3b/demo/demo.py --prompt "..." --gen 64 --trace
# sampling (recommended thinking-mode config)
QWEN36_LAYERS=40 python models/demos/qwen3_6_a3b/demo/demo.py --prompt "..." --gen 64 --trace \
  --temperature 0.6 --top-k 20 --top-p 0.95 --presence-penalty 1.5 --seed 1234
```

## How it works (on-device decode tail)

`TtModel.enable_sampling(temperature, top_k, top_p, seed, presence_penalty)` selects the tail recorded
into the decode trace; `TtModel._select_token` runs it. Per step (all in-trace, single device):

1. **lm_head** on the 1-row hidden state → logits `[1, vocab]` (vocab = 248320).
2. **presence penalty** (if > 0): `logits -= presence_penalty * mask`, where `mask` is a vocab-sized
   buffer of tokens generated so far.
3. **chunked top-k**: pad to `8×32768`, reshape `[1,1,8,32768]`, `ttnn.topk(k=32)` per chunk, add
   per-chunk global-index offsets, merge to 256 candidates.
4. **`ttnn.sampling`** over the 256 candidates (replicated to the 32 "users" the op requires), applying
   `temp = 1/T`, `top_k`, `top_p`, with `ttnn.manual_seed`. Token = row 0.
5. **state updates**: write the token into `t_tok`; OR it into the presence `mask`; `plus_one` the
   seed. The seed and presence mask are device-resident state (in `_state_tensors`), so a static trace
   draws fresh randomness and accumulates presence across replays, yet is deterministic for a given seed.

### Why presence updates the mask with a one-hot, not `ttnn.scatter`
The presence mask is `[1,1,1,vocab]`; in TILE layout the singleton row dim is padded to 32, so the
tensor is physically `32 × vocab`. `ttnn.scatter` over that to set a single element costs **~2.1 ms**
(it scans the whole padded tensor). Instead the new token is OR'd in elementwise —
`mask = max(mask, (iota == token))` against a precomputed `iota` index buffer — which is **~0.5 ms**.
The full-vocab `multiply`/`subtract` for applying the penalty are already cheap (~0.1 ms each) even
with the padding, so only the update needed changing.

### Why top-k is chunked
A single `ttnn.topk` over the full 248320 vocab runs **single-core (~38 ms)** — `ttnn.topk` only goes
multicore for **power-of-2 widths ≤ 32768**. Chunking the vocab into 8 power-of-2 pieces keeps every
topk multicore (~0.2 ms each). This is the difference between a ~38 ms and a ~0.5 ms sampling tail.

## Performance

Measured on a single Blackhole, 2-layer model (`bench_decode.py`; the tail cost is layer-count
independent, so the absolute ms add the same way on the full 40-layer model):

| config | ms/token | overhead |
|---|---|---|
| greedy | 3.13 | — |
| temperature + top-k + top-p | 3.54 | **+0.43 ms** |
| + presence_penalty (full recommended config) | 4.14 | **+1.01 ms** |

On the full 40-layer model (~33 ms/token greedy) that is **~+1%** for plain sampling and **~+3%** for
the full thinking-mode config. The presence penalty adds ~0.6 ms (a full-vocab subtract + a one-hot
mask update); plain temperature/top-k/top-p is nearly free. Greedy is unchanged.

> Earlier the presence penalty cost ~2.8 ms because it updated the mask with `ttnn.scatter`
> (~2.1 ms over the tile-padded vocab). Replacing the scatter with an elementwise one-hot
> (`max(mask, iota == token)`) cut it to ~0.6 ms — see "Why presence updates the mask with a one-hot".

Measure it yourself:
```bash
QWEN36_LAYERS=40 python models/demos/qwen3_6_a3b/tests/bench_decode.py \
  --temperature 1.0 --top-k 20 --top-p 0.95 --presence-penalty 1.5
```

## Verification done
- **Greedy unchanged**: byte-identical output and same ms/token as before the change.
- **Correctness**: with `top_k=1, top_p=0, temp=1` the sampler equals `argmax`; sampled tokens always
  lie within the torch top-k set.
- **Reproducible**: same `seed` → identical output across runs (eager and traced); different `seed` →
  different output. Eager and traced sampling agree for a given seed.
- **Presence penalty**: visibly reduces repetition vs `presence_penalty=0`, reproducible per seed.

## Files changed
- **`tt/model.py`** — `enable_sampling()` builds the sampling buffers (k/p/temp/seed, presence mask,
  chunk index/offset tables); `_select_token()` holds the greedy and sampling tails; `_state_tensors()`
  includes the seed and presence mask so trace capture snapshots/restores them.
- **`demo/server.py`** — request schemas default to the recommended thinking-mode config and the
  `temperature/top_k/top_p/presence_penalty/seed` params are threaded through `_generate_ids` into
  `enable_sampling`; `min_p`/`repetition_penalty` are accepted but warned-and-ignored; docstring updated.
- **`demo/demo.py`** — `--temperature/--top-k/--top-p/--presence-penalty/--seed` flags; routes greedy vs
  sampling; opens the mesh with a nonzero `trace_region_size` so `--trace` works.
- **`tests/bench_decode.py`** — `--temperature/--top-k/--top-p/--presence-penalty` benches greedy vs
  sampling side-by-side and reports the per-token overhead.
- **`README.md`** — serving + decode-internals sections updated for sampling.

## Limitations / notes
- `min_p` and `repetition_penalty` are not implemented (no-ops at their recommended defaults).
- The **first** token (from prefill) is always greedy argmax; only decode tokens are sampled.
- The presence penalty applies to generated (decode) tokens; the prompt is not included.
