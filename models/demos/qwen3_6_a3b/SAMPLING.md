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

## Speculative decode + sampling (MTP)

Speculative decode (`MTP.md`, `EXPERIMENTS.md`) originally only supported **greedy** acceptance — a
draft was accepted iff it equalled the verify row's argmax — so it could not serve the server's
default config, which samples. It now runs **speculative sampling**, which accepts drafts while
keeping the emitted stream **exactly** the distribution the model would have sampled from.

### The rule (and why it is cheap here)

The draft head picks each token by `argmax`, so the draft distribution q is a point mass at x̂.
Speculative sampling's `min(1, p(x̂)/q(x̂))` therefore collapses to **`p(x̂)`**, and the rejection
residual `norm((p−q)₊)` collapses to *p with x̂ removed and renormalised*:

    P[emit x̂]    = p(x̂)
    P[emit y≠x̂] = (1 − p(x̂)) · p(y)/(1 − p(x̂)) = p(y)      → exactly p

So the draft graph needs **no change at all** — only the target distribution p per verify row.

### Where p comes from

`TtModel._verify_candidate_tail` runs the same chunked top-k the plain decode tail runs, but per verify
row, and writes each row's 256 (value, global index) candidates into a persistent buffer. The host then
applies presence penalty → temperature → top-k → top-p (`tt/mtp_sampling.py::SpecSampler.probs`) and
runs the acceptance rule. Three consequences:

- **temperature/top_k/top_p are NOT baked into the trace** — a request can change them with no
  re-capture. Only greedy-vs-sampling changes the recorded graph, and that choice is made **once per
  process**, not per request: exactly one MTP verify trace may be resident (see below), so
  `QWEN36_MTP=1` captures the sampling tail and lets greedy requests replay it too (it writes the
  argmax buffer as well; they just also pay the candidate topk, ~1.7 ms/round at gamma=2), while
  `QWEN36_MTP=greedy_only` captures the greedy tail and routes sampled requests to plain decode.
- **Truncating over candidates is exact, not approximate.** Any member of the global top-32 is in its
  own chunk's top-32, so for `top_k ≤ 32` the truncated support is the true one. Applying the presence
  penalty to candidates is exact for the same reason (proof in the `tt/mtp_sampling.py` docstring,
  checked directly against a full-vocab reference in `tests/test_mtp_sampling_rule.py`).
- The presence penalty is in fact **more** faithful than the device mask: the penalised set can include
  tokens accepted earlier *in the same round*, which a per-step device mask cannot express.

### Acceptance rules

`QWEN36_MTP_ACCEPT` selects the rule:

| value | behaviour | exact? |
|---|---|---|
| `exact` (default) | rejection sampling, accept with probability `p(x̂)` | **yes** |
| `relaxed` | Medusa-style typical acceptance: accept when `p(x̂) ≥ min(eps, delta·e^−H(p))` (`QWEN36_MTP_RELAX_EPS`=0.09, `QWEN36_MTP_RELAX_DELTA`=0.3). Deterministic, so it pays the maximal divergence its decision allows | no — measured mean TV 0.094 |
| `lenient` | exact rejection sampling against a *tempered* target: accept with probability `min(1, p(x̂)/QWEN36_MTP_LENIENCE)`. Lenience 1.0 IS `exact`; smaller values buy acceptance at a divergence of exactly `min(1,p/L) − p` per position | tunably — measured mean TV 0.030 (L=0.7) to 0.069 (L=0.3) |
| `greedy` | accept iff `x̂` is the argmax (the pre-sampling rule) | n/a |

### Why sampled acceptance barely costs anything

Measured host-side against the real `mtp.*` head over 384 positions
(`probe_mtp_acceptance.py score`), at the server's default `T=0.6 top_k=20 top_p=0.95`:

| rule | host-model depth-1 acceptance |
|---|--:|
| greedy (exact match) | 0.896 |
| exact, argmax draft (shipped) | 0.858 |
| exact, *sampled* draft | 0.880 |
| relaxed | 0.934 |

That host model says exact should cost almost nothing, because at `T=0.6` with `top_p=0.95` the
truncated target is extremely peaked: mean **|support| = 1.4 tokens, H(p) = 0.15 nats,
E[p_max] = 0.936**. **On device it costs a lot more** — 2.70 → 2.30 tokens/round — because the host
model scores an off-policy greedy trajectory while real sampled decode visits flatter states. See
EXPERIMENTS.md for the measured numbers and the resulting shipping defaults; the short version is
**exact sampled MTP is 1.14× and distribution-preserving**, `lenient` 1.21× and `relaxed` 1.25×. Exact was 1.00× until a verify-conv fix cut the round 66.1 -> 57.5 ms — the round cost, not the acceptance rule, was what made sampled speculation look hopeless.

One design question the host model does settle: **sampling the draft on device is not worth it**
(+0.02 acceptance for an extra per-step topk in the draft trace), so the draft stays `argmax`.

### One resident trace — a hazard worth knowing about
Capturing the verify trace while any other MTP trace is still resident produces a **silently corrupt
trace**: the candidate rows read back as ~1e9 garbage, the acceptance rule then "samples" an
out-of-vocab id, and handing that to `ttnn.embedding` **hangs the device** — so the symptom appears
in the server, far from the cause. `TtModel.setup_mtp_traces` therefore releases every MTP trace
before capturing the set, and `tests/test_mtp_trace.py::test_mtp_candidates_valid_after_a_greedy_capture`
pins it. This is the same failure family as `forward_prefill_traced`'s multi-bucket wedge and the
reason `capture_decode_trace` always releases the prior trace first.

### Determinism
The acceptance/residual randomness is a host `torch.Generator` seeded from the request's `seed`, so
`seed` reproduces a run exactly (gated by `test_spec_sampling_is_seed_reproducible`). It is a different
RNG stream from the on-device `ttnn.sampling` path, so a given seed does not produce the same text with
and without MTP — each is reproducible on its own.
