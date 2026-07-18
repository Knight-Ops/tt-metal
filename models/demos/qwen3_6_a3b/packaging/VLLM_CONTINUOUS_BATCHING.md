# Qwen3.6-35B-A3B — vLLM continuous-batching serving plan

Goal: serve the 35B-A3B MoE hybrid (30 gated-delta + 10 full-attention + MoE) under the
Tenstorrent vLLM plugin with **true continuous batching**, packaged as a `tt-kernel
--backend vllm` bundle in this `packaging/` folder.

## Reality (from investigation, 2026-07-18)

- **Framework is not the blocker.** Upstream vLLM V1 (the fork at `~/projects/tt-vllm`,
  `0.11.2.dev`) supports hybrid gated-delta + attention (Qwen3-Next family) on GPU and
  ships a `MambaSpec`. But the **Tenstorrent plugin does not consume it** — its KV-cache
  layer hard-rejects any non-`AttentionSpec` (`model_runner.py:444`). So GDN state must be
  **model-owned**: inherit plain `Generator`, define **no** `get_kv_cache_spec`, keep the
  conv/recurrent state in fixed device buffers, and re-zero it per new sequence.
- **The off-branch `blackhole/qwen36` stack does NOT apply.** It is a divergent codebase
  for the **dense** Qwen3.5-9B/27B — **no MoE**, so it cannot serve 35B-A3B. Its vLLM
  adapter is single-user; its continuous batching is a model-level *demo*
  (`generate_tp_batched_continuous`) + tested per-slot primitives, **not wired to vLLM**.
  Value: its `prefill_seed_tp` / `reseed_slot_tp` / `decode_tp_batched` / `finalize_seed`
  is a concrete, on-device-validated **reference pattern** for per-slot GDN reseed.
- **Our tree already has the hard on-device pieces** (`models/demos/qwen3_6_a3b`):
  batched GDN state `[B,…]` + kernels (`decode_step_batch_tt`, PCC>0.999 @B=4, benched
  B=32→~157 tok/s), batched attention KV `[B,…]`, dense MoE for B>1, and the
  logits-returning contract (`decode_forward_logits`→`[B,vocab]`, `set_decode_tokens`,
  `_reset_linear_state`). Gaps: **paged KV**, **per-slot (vs synchronized) batching**,
  **batched prefill**, and the **scheduler bridge**.

## Where the code lives (this is NOT confinable to `packaging/`)

| Layer | Location | Work |
|---|---|---|
| vLLM bundle descriptor | `packaging/vllm_bundle/vllm_metadata.json` | new (thin) |
| vLLM adapter (`Generator` subclass) | `packaging/vllm_bundle/generator_vllm.py` (ships in bundle) | new |
| Paged KV, per-slot GDN, batched prefill | `tt/model.py`, `tt/attention.py`, `tt/gated_delta.py` | **the bulk** |

## Milestone V1 — single-user vLLM bundle (shippable foundation)

A real, pushable `--backend vllm` bundle serving **B=1**. No throughput win over the
dispatch runner, but it stands up the whole plugin/OpenAI/tt-inference-server path and
de-risks the integration. The adapter methods map onto primitives **that already exist**:

- `initialize_vllm_model` → build `TtModel` (exists).
- `allocate_kv_cache` → return the attention KV; GDN state stays model-bound. Accept the
  **contiguous** `[1,n_kv,max_seq,hd]` cache at B=1 (gap #D "simplest") — no paged KV yet.
- `prefill_forward` → wrap `TtModel.forward()` → `(logits, rope_deltas=0)`; GDN reset per
  sequence via `_reset_linear_state()` (exists).
- `decode_forward` → drive `decode_forward_logits()` + `set_decode_tokens()` (exist).
  **Decision:** either (a) a bespoke adapter that drives our existing decode loop directly
  (less model change — recommended for V1), or (b) implement the tt_transformers decode
  contract (`prepare_inputs_decode`/`ttnn_decode_forward`/`process_output_decode`) for the
  inherited `WarmupForwardMixin` traced replay (cleaner long-term, more model change).
- `warmup_model_prefill` / `warmup_model_decode` → the plugin's two-phase warmup contract.

Model work: minimal (bridging). Verify on P150 (vLLM plugin present; `tt-api` still needed).

## Decode performance: approaches, measurements & tradeoffs (V1, single-user, on-device)

Measured on one Blackhole P150, 2026-07-18. Prefill is solved (~319 tok/s ≈ server.py's ~350);
this is about the decode-per-token path. The reference is `server.py` at ~30 tok/s.

| Approach | tok/s | vLLM sampler kept? | Status |
|---|---|---|---|
| Eager decode (return logits, host sample) | ~2.9 | ✅ full (logprobs, guided decoding, penalties) | first working V1 |
| **Sync traced** decode (return logits, host sample) | **~16** (peak 21) | ✅ full | **working, feature-complete** |
| Async traced (return device logits, overlap read+sample) | — | ✅ full | **parked — produced NO completions** |
| **On-device sampling** (trace samples, return tokens) | ~30 (server.py parity) | ❌ partial | **chosen** |

Root cause of the sync-traced 16-vs-30 gap: returning full `[1, 248320]` logits means a full-vocab
D2H **plus CPU sampling over 248k classes** on the autoregressive critical path each step. server.py
samples **on-device inside the trace** and reads back only the token id.

### Chosen: on-device sampling — SHORTCOMINGS (document before you rely on it)
Recovers ~30 tok/s by moving sampling into the decode trace (reusing the model's validated
`enable_sampling` + `capture_decode_trace` + `decode_step_traced`). What you give up by bypassing
vLLM's host sampler:
- **Guided / structured decoding** (JSON-schema / regex / grammar-constrained output, incl.
  hard-guaranteed tool-call arg schemas). vLLM masks logits on host per a compiled FSM — impossible
  once sampling is on-device. (Qwen3.6's *trained* native `<tool_call>` XML still works; you only
  lose the hard guarantee.)
- **logprobs / prompt_logprobs** (evals, confidence scoring, spec-decoding).
- **`min_p`, `frequency_penalty`, custom logit processors, `logit_bias`.**
- Kept: `temperature`, `top_k`, `top_p`, `presence_penalty`, `seed` (the set server.py supports).
- **Per-request param changes force a decode-trace re-capture** (the sampling tail is baked into the
  trace), same as server.py's per-request capture. Fine at B=1; a cost to design around at B>1.

### Parked but promising: async decode (KEEP — likely the right long-term answer)
Async **should** be the win: it keeps vLLM's **full** sampler AND overlaps each step's CPU work
(scheduling / detok / host sampling) with the next step's device forward, via vLLM's async scheduling.
The plugin is built for it — `worker.py` defaults `trace_mode="all"` (50 MB trace region,
`enable_trace=True`), there's a `host_sampler`, a forward/sampling split with deferred samplers and
"one forward in flight" (`model_runner.py:196-212`), and `non_dp_async_scheduling` (`:205`).

What we implemented and left in place (dormant, behind `supports_async_decode`):
- adapter `decode_forward` returns on-device logits `[out]` when `read_from_device=False`.
- model `decode_forward_logits(read_from_device=...)` / `decode_step_logits_traced(read_from_device=...)`
  return the device ttnn logits (deferred read); the traced path **clones** the persistent buffer so
  the next replay can't overwrite it before the async read.
- model `process_output_decode(...)` + the inherited `Generator.read_decode_output` (async `.cpu()` +
  event) + `process_decode_output_host` do the deferred host conversion.

Why it's parked: with `supports_async_decode=True` the server ran and did device work but returned
**zero completions** — even after the trace-buffer clone fix. The failure is in the deferred-sampling
/ device-tensor-lifetime interaction, which needs **on-device instrumentation** to pin down, not blind
edits. To revive it, debug on the P150:
1. Log inside `read_decode_output` / `process_output_decode` — are they called? what tensor shape does
   `ttnn.to_torch` produce for the host logits (tile-padding? `[1,vocab]` vs `[32,vocab]`)?
2. Confirm the async `.cpu(blocking=False)` + `record_event` on cq 0 is actually ordered before the
   next `execute_trace` (the clone should guarantee this — verify it holds).
3. Check whether vLLM's async scheduling speculatively runs the next forward before this step's token
   is sampled+written back — if so, the autoregressive `set_decode_tokens` dependency is violated and
   the plugin path needs a different hook (or `bind_decode_trace_inputs`).
4. Verify `warmup_model_decode` being a no-op isn't what async scheduling relies on for setup.

Flip `supports_async_decode` back to `True` in the adapter to resume this line of work.

## Milestone V2 — continuous batching (the prize) — PRECISE GAP MAP

Measured against the actual code 2026-07-18. The surprise: the batched **decode** is done; the
one real blocker is **per-slot prefill**.

### Already WORKS at B>1 (reuse as-is)
- Batched decode compute with **independent per-slot positions**: `_decode_hidden`/`_rope_for`
  (`tt/model.py:540-555, 530-538`), attention `forward_decode` per-slot `current_pos [B]`
  (`tt/attention.py:258-289`), GDN `_forward_decode_batch(_fused)` per-slot state `[B,Vh,Dk,Dv]` +
  `conv_rows [B,conv_dim]` (`tt/gated_delta.py:772-824`).
- Batched **host** sampling: `decode_forward_logits`->`[B,vocab]` + `set_decode_tokens([B])`
  (`tt/model.py:580-600, 568-578`). The plugin already builds `[B,1]` tokens / `[B]` positions /
  `[B]` page tables per decode call.
- Batched cache **storage** (`[B,n_kv,max_seq,hd]`, `[B,Vh,Dk,Dv]`, `[B,conv_dim]`).

### Must BUILD in the model (priority order)
1. **Per-slot prefill (THE blocker).** A `prefill_into_slot(input_ids, slot)` writing only row `s`:
   attention `ttnn.fill_cache(cache, k, s)` (today hardcodes slot `0` at `attention.py:189-190,210-211`);
   GDN sliced writes of `recurrent_state[s]` / `conv_rows[s]` (today whole-buffer `ttnn.copy`,
   `gated_delta.py:701-707, 750-756`). `self.pos` must go per-slot `[B]` (today scalar, `model.py:213`).
2. **Per-slot GDN reset on insert** — generalize `_reset_linear_state` (`model.py:177-185`) to zero
   only slot `s` (non-idempotent scan => every new sequence must re-zero its lane).
3. **Slot eviction/permute** honoring `slot_remap` + `reset_batch` (`model_input.py:117-124`):
   a `reorder_slots(remap)` permuting KV/GDN rows.
4. *(optional, only if keeping on-device sampling at B>1)* per-slot device sampling — `_select_token`
   reads **row 0 only** (`model.py:602-651`); on-device sampling is **B=1-only** today. **First
   increment uses the batched HOST-sampling path instead.**
5. *(larger, deferrable)* true paged KV so vLLM's block manager owns attention KV.

### Must BUILD in the adapter (`generator_vllm.py`)
1. Lift the `max_batch_size == 1` assert.
2. `allocate_kv_cache`: stop returning `[]` — return a real/dummy attention KV of `kv_cache_shape`
   so vLLM's paged manager (needs an `AttentionSpec`, `num_blocks>0`, `model_runner.py:396-411`)
   admits >1 request and hands out per-request `page_table`s. **Returning `[]` at B>1 is a real risk.**
3. `prefill_forward`: the plugin submits **one call with all newly-scheduled requests**
   (`tokens [num_reqs, max_prefill]`, `prompt_lens` list) — loop and route each into a slot.
   WARNING: on **single-device B>1 the plugin does NOT pass `empty_slots`** (DP-only); derive
   request->row from the `page_table`/decode row order or maintain an adapter-side map.
4. `decode_forward` at B>1: use the batched host-sampling path, stop ignoring `start_pos`/`page_table`,
   consume `slot_remap`/`reset_batch`.

### Sampling tension (decide up front)
On-device sampling (the V1 ~30 tok/s win) is **B=1-only** (`_select_token` reads row 0). So the first
B>1 increment runs **host-sampling batched decode** — the win is *aggregate* throughput (benched
B=32->~157 tok/s), not per-user latency. Per-slot on-device sampling is a later add (#4 above).

### Smallest first increment (serve 2 concurrent requests)
Contiguous per-slot cache (no paged KV), host-sampling decode only. Add exactly: (a) model
`prefill_into_slot(ids, slot)` writing KV row `s` + GDN row `s` + per-slot GDN zero; (b) per-slot
`self.pos`/`t_curpos`; (c) adapter that prefills each new request into a free row, tracks a
request->row map, and drives the existing batched `decode_forward_logits`/`set_decode_tokens`. Defer
`slot_remap` compaction (pad/hold freed rows), on-device sampling, and paged KV. This confines new
work to **per-slot prefill write + slot bookkeeping** — the one capability the model lacks.

Effort: substantial + iterative on-device; recommend a git worktree to protect the working B=1 path.

## Packaging & push (both milestones)

```
packaging/vllm_bundle/
  vllm_metadata.json      # arch=TTQwen3_5MoeForConditionalGeneration, main_class=…:Qwen36ForCausalLM,
                          # hf_weights=Qwen/Qwen3.6-35B-A3B, launch={blackhole:{command,env}}
  generator_vllm.py       # the adapter (+ any deps)
```
```bash
tt-kernel push <ns>/qwen3.6-a3b-blackhole --backend vllm \
  --bundle-dir models/demos/qwen3_6_a3b/packaging/vllm_bundle \
  --weights Qwen/Qwen3.6-35B-A3B
```
Kernels-less (the plugin JITs at warmup); co-version `ttnn` + `ttl`. Serving host needs the
TT vLLM fork/plugin (present here) + `tt-api`.

## Hard truth on verification

Every step here requires **hardware-in-the-loop** iteration against the P150 + the vLLM
runtime. This cannot be written-and-verified in one shot; the adapter + model changes must
be driven on-device. That's exactly why the current `demo/generator_vllm.py` shipped as an
accurate map rather than unverified code.
