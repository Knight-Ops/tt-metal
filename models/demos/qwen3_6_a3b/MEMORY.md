# Qwen3.6-35B-A3B — Device Memory & Trace Aliasing

Companion to `README.md`. Two things live here:

1. **The trace/allocator aliasing rule** (§1–§3) — the root cause of the "multi-bucket prefill wedge",
   found and fixed 2026-08-22. This is the part worth reading even if you never touch memory: it
   explains a class of bug that reads as *numerical* (garbage PCC, NaNs, acceptance collapse) but is
   actually an *allocator* problem, and it is the most likely explanation for any future
   "correct-at-4-layers, broken-at-40" or "works alone, breaks alongside" report in this model.
2. **The memory audit** (§4–§6) — where this model's device memory actually goes, what was reclaimed,
   and how to measure it.

Everything here is measured on a single Blackhole P150 (32309 MB DRAM) at 40 layers with BFP4 experts,
tt-metal v0.77.0. Numbers are from `tt/memstat.py` unless noted.

---

## 1. The rule

> **A captured trace bakes the ADDRESSES of every buffer it touches. If any of those buffers is freed
> while the trace is still alive, the allocator will hand that address to something else, and replaying
> the trace will write over whatever now lives there.**

tt-metal states this itself, as a runtime warning
(`tt_metal/impl/allocator/allocator.cpp`, `verify_safe_allocation`):

> *Allocating device buffers is unsafe due to the existence of an active trace. These buffers may be
> corrupted once a trace is executed.*

That warning fires in this model's logs today and is easy to dismiss as cosmetic. It is not. It is the
single most important sentence in this document.

### Why it is easy to hit here

Nothing in ttnn pins a trace's intermediates. An op output lives exactly as long as its Python
reference: rebind the local or return from the frame and the device buffer is freed immediately —
*including* when that happened inside `begin_trace_capture` / `end_trace_capture`. So by the time
`end_trace_capture` returns, every in-graph transient the trace will write on replay has already been
returned to the free list, and the next allocation of a similar size lands right on top of it.

### The two disciplines, and why they are opposites

| context | discipline | why |
|---|---|---|
| **eager** paths (prefill, one-off work) | free dead intermediates EARLY (`ttnn.deallocate`) | peak DRAM is otherwise the sum of everything a frame holds, not its working set |
| **traced** paths | the trace's buffers must OUTLIVE the trace | a freed address gets reused and the replay corrupts the new owner |

Confusing the two is what produced the wedge. `build_trace_pool` had already applied the second
discipline to the gated-delta recurrence; the MoE, attention and norm intermediates were left
unpinned, and that was enough.

---

## 2. The multi-bucket prefill wedge — root cause

### Symptom (as recorded for months)

Capture prefill traces for two bucket lengths in one process and the second one replays as garbage.
Recorded as "512 captured after 256 → PCC ~0.214; 512 alone → PCC 1.0", believed to be a
tt-metal-level multi-trace capture-isolation bug, and worked around by deploying a **single bucket**.

### What measurement actually showed

Run with `QWEN36_GDN_FUSED_OP=0` so eager and traced use the *same* recurrence — otherwise the
comparison is contaminated by the fused-op-vs-in-tree difference, which makes even a single bucket read
0.996 and which is why the old "PCC 1.0 alone" baseline no longer reproduces.

**(a) It reproduces, worse than recorded, and is not bucket-specific:**

| capture order | bucket 256 | bucket 512 |
|---|---|---|
| [256, 512] | 1.000000 | **−0.049535** |
| [512, 256] | **0.009097** | 1.000000 |

512 is *perfect* when captured first. Whichever bucket goes second is destroyed. PCC ≈ 0 means
uncorrelated garbage, not degradation.

**(b) It is not a footprint problem.** Each bucket retains only **18.1–21.4 MB**, against **11130 MB
free**. This is why raising `trace_region_size` never helped — the earlier note inferred that; this
measures it.

**(c) It is neither "second captured" nor "second replayed".** Fix capture at [256, 512] and vary the
replay sequence:

| replay | bucket | PCC | |
|---|---|---|---|
| #0 | 512 | 1.000000 | OK |
| #1 | 256 | 1.000000 | OK |
| #2 | 512 | **−0.049535** | WEDGED |
| #3 | 256 | 1.000000 | OK |

512 was captured second yet replayed perfectly at #0; 256 replayed second at #1 and was fine. One rule
fits all three runs:

> **Replaying an EARLIER-captured trace corrupts the LATER-captured one.** The earlier trace is never
> the victim. The later one is fine until an earlier one runs.

**(d) The mechanism, observed directly rather than inferred.** `pool["S0"]` is bucket 512's *read-only
zero* initial recurrent state — the graph never writes it. Capture [256, 512], then replay 256 once and
read 512's pool back:

| buffer | before the 256 replay | after |
|---|---|---|
| `S0` | 0 | **\|max\| 33.1857, mean 0.198614** |
| `out[0]` | 0 | **NaN** |
| `valid` | 0 | 0 (untouched) |

Trace 256's replay writes into trace 512's pool. Caught in the act.

### The causal chain

1. `capture_prefill_trace(256)` allocates bucket 256's pool, then captures. Its in-graph transients
   (the MoE's ~206 MB peak, attention, norms) are freed when capture ends — nothing pins them.
2. `capture_prefill_trace(512)` then allocates bucket 512's *persistent* pool, which lands in that
   freed hole.
3. Trace 256's recorded commands still name those addresses. Replaying it writes over bucket 512's
   `S0` and `out[]`.
4. Bucket 512 then replays from a corrupted zero-state and emits NaNs.

`valid` survives because it is allocated earlier in the pool build and does not fall in the hole —
which is itself a hint at how address-sensitive this is.

---

## 3. The fix

Allocate **every** bucket's persistent buffers before **any** capture, so they all sit below the region
the captures use for transients:

- `alloc_prefill_buffers(T)` — ids, RoPE tables, output buffer, gated-delta pool. Idempotent.
- `record_prefill_trace(T)` — eager compile pass, then capture.
- `capture_prefill_trace(T)` = both, for single-bucket callers (unchanged behaviour).
- `setup_prefill_traces(buckets)` — **two passes**: allocate all, then record all.

Traces still share the *transient* region afterwards, and that is fine: within a replay every transient
is written before it is read, so stale bytes there are overwritten. The read-only `S0` was the one
buffer for which that was not true, and it is now out of reach.

### Validated

Every repro from §2 passes with the two-pass allocation, and the mechanism test shows the write is
gone rather than merely harmless:

| test | before | after |
|---|---|---|
| capture order [256, 512] | 1.000000 / **−0.049535** | **1.000000 / 1.000000** |
| capture order [512, 256] | **0.009097** / 1.000000 | **1.000000 / 1.000000** |
| replay [512, 256, 512, 256] | OK, OK, **WEDGED**, OK | **1.000000 × 4** |
| bucket 512 `S0` after a 256 replay | \|max\| 33.1857 | **0 (untouched)** |
| bucket 512 `out[0]` after a 256 replay | NaN | **0 (untouched)** |

It is also free: both buckets together retain **39.5 MB**, exactly the 18.1 + 21.4 MB they took when
captured one at a time. Nothing is pinned that was not pinned before — only the *order* of allocation
versus capture changed.

**Consequence: multi-bucket prefill traces work.** The single-bucket deployment constraint recorded in
`tt/gated_delta.py` and `tt/model.py` (`forward_prefill_traced`) can be lifted. Note separately that
tracing the prefill is only worth ~1.01-1.02x now that the fused GDN op is the eager default, so this
matters most for the eval path (`evaluation/run_mmlu_bench.py`, which pre-captures one bucket) rather
than for serving.

### If you need a stronger guarantee

Pinning the prefill graph's intermediates (extending `build_trace_pool` to the MoE and attention) makes
the trace's whole address range untouchable rather than merely non-overlapping-with-state. It costs the
peak footprint per bucket (~206 MB for the MoE alone) and was not needed for this fix.

---

## 4. Checklist for future trace work in this model

Before capturing anything, ask:

1. **Does every buffer the graph reads or writes outlive the trace?** Persistent cache, pooled buffer,
   or module weight — never an op output whose Python reference dies.
2. **Is any per-request buffer reallocated between replays?** If so the trace is silently invalid.
   `start_decode` and `enable_sampling` both did this (see §5); they now write in place via
   `copy_host_to_device_tensor` at stable addresses.
3. **Will another trace be captured after this one?** If yes, allocate both traces' persistent buffers
   before either capture.
4. **Does anything allocate while a trace is live?** That is the allocator warning. Audit it.
5. **If a fresh allocation was zeroing something for you, who zeroes it now?** Reusing a buffer means
   resetting it explicitly. `t_presence` needed this; missing it would leak one request's repetition
   penalty into the next.

### Symptoms that should make you suspect this class of bug

- "Correct at 4 layers, broken at 40" — more layers means more allocation pressure, not different math.
- "Works alone, breaks when captured alongside another trace."
- Output PCC ≈ 0 or NaN rather than degraded — arithmetic errors degrade, clobbered memory does not.
- Raising `trace_region_size` changes nothing (that region holds trace *commands*, not tensors).
- A change that is bit-identical in isolation yet destroys an end-to-end metric.

`_COMMIT_SELECT` in `tt/gated_delta.py` carried all five of these markers and had been disabled for
months on the strength of them. As of 2026-08-22 its failure no longer reproduces: acceptance is
bit-identical with it on and off, greedy (2.500 tok/round, 160/160 token parity) and sampled (2.594,
83/83), with the documented commit saving intact (6.18 → 0.97 ms, round 54.35 → 49.13 ms). The most
likely cause of the recovery is the 150 MB of dead allocation removed from that same verify path (§5),
but that has **not been isolated** — treat it as "does not reproduce on this tree", not "fixed by X".

---

## 5. Memory audit — what was reclaimed

The audit's structural finding: the production path contained **zero** `ttnn.deallocate` calls
(`deepseek_v3` has 465, `llama3_70b_galaxy` 71, and the sibling `blackhole/qwen36` uses it throughout,
including inside traced bodies — so it is legal under capture).

| change | switch | measured |
|---|---|---|
| batched GDN `S`/`Snew` alias (was a 32 MB/layer copy per step at B=32) | `QWEN36_GDN_ALIAS_STATE` | **960 MB** @ B=32; 240 @ B=8 |
| dead verify scratch: `states`/`outs` the chained path never reads | — | **150 MB** (341.3 → 191.3, K∈{1,3}, 30 layers) |
| shared per-head `negA`/`dtb` instead of B-replicated | `QWEN36_GDN_BATCH_SHARED_CONST` | **120 MB** @ B=32 |
| `Snew` ping-pong (2 carry buffers, not `Nc`) | `QWEN36_POOL_SNEW_ALL` | **1020 MB** @ 32K bucket |
| bf16 pool `out` (a read-out, not the fp32 carry) | `QWEN36_POOL_OUT_FP32` | **256 MB** @ 32K bucket |
| MoE dense-prefill deallocations | `QWEN36_MOE_DEALLOC` | peak **449.3 → 205.8 MB** per call |
| fused SiLU into the multiply (**default OFF**, pending MMLU) | `QWEN36_MOE_FUSED_SILU` | 3.1% of the stage @ T=512 |
| decode-trace reuse across requests | `QWEN36_TRACE_REUSE` | **~460–487 ms per request** |

Trace reuse covers **both** decode graphs, because the vLLM adapter drives both and each used to pay
an eager warm-up call plus a full capture on the first step of every request:

| adapter branch | model call | readiness check | what the signature must cover |
|---|---|---|---|
| on-device sampling | `decode_step_*` / `capture_decode_trace` | `decode_trace_ready()` | sampling on/off, **presence-penalty value**, decode batch B, `_samp_nc` |
| host sampling | `decode_forward_logits` / `capture_decode_logits_trace` | `logits_trace_ready()` | decode batch B only — this graph has no sampling tail |

The presence penalty is in the first signature and nothing else from the sampling params is, and the
distinction matters: `top_k` / `top_p` / `temperature` / `seed` are read from device tensors
(`t_k` / `t_p` / `t_temp` / `t_seed`) so a per-request write takes effect under reuse, which is the
whole point. `presence_penalty` is multiplied into the graph as a python scalar, so it is frozen at
capture and a change must force a re-capture. `demo/server.py` uses one server-wide default and would
never have noticed; the vLLM adapter passes the request's own value, and the manifest sets
`sample_on_device_mode: decode_only`, so that path is live in production.

End to end, 40 layers, defaults vs every switch flipped:

| | reverted | new | saved |
|---|---|---|---|
| eager prefill T=512, peak transient | 456.8 MB | **150.8 MB** | 306.0 MB (3.03×) |
| batched decode B=8, caches+scratch | 795.2 MB | **525.1 MB** | 270.1 MB |
| batched decode B=32, caches+scratch | 2994.9 MB | **1914.9 MB** | 1080.0 MB |
| total DRAM at exit | 24096.5 MB | **22716.5 MB** | **1380.0 MB** |

### Two corrections the measurement forced

- **`hgu` is BLOCKFLOAT8, not BFLOAT16.** `ttnn.matmul` inherits its output dtype from in0, and in0
  (`xe`) is BFP8 at the shipping default — so the gate_up output is 68 MB, not 128, and `gate`/`up` are
  34 MB each, not 64. `tt/moe.py` asserted the opposite in two places while its own `_DENSE_IN0BW`
  table three blocks later said "BFP8 in0, block 8 -> fits". Corrected in place.
- **Freeing memory is not a speedup.** Prefill is ~0.2% *slower* with the MoE deallocations (the extra
  host allocator calls), in exchange for 3× less peak transient. The only footprint change that buys
  time is the batched alias, because it deletes real traffic: predicted 4.4 ms/step at B=32 from
  removing 1.92 GB of copy, measured **4.90 ms** (188.97 vs 193.87 ms, +2.53% throughput).

### What the headroom is worth

| | |
|---|---|
| holding B=32 | +1728 tokens of context |
| at max_seq=8192 | max B 61 → **74 (+21%)** |
| at max_seq=32768 | max B 17.3 → 18.2 (+5%) |
| at max_seq=1024 | max B 124 → 194 — unreachable, `SAMP_USERS=32` caps on-device sampling first |

So the headroom is worth most in the middle of the context range. This work bought robustness and
moderate headroom plus 2.5% at B=32; it is not a throughput change.

---

## 6. How to measure

`tt/memstat.py`. The allocator is a **host-side** book-keeper — ttnn enqueues asynchronously but
allocates synchronously — so a query costs no device round-trip and does not perturb what it measures.
That is what makes `probe()` usable inside a hot loop or a capture.

```python
from models.demos.qwen3_6_a3b.tt import memstat

memstat.report(mesh, "after build")          # one-shot DRAM + L1 line
with memstat.track(mesh, "prefill") as t:    # before/after + peak over probes
    model.forward(ids)
print(t)                                     # net retained, and transient peak
```

`track()` only sees points where `probe()` is called, so it is a lower bound on the true peak — place
probes where a claim is about, rather than trusting a coarse before/after. To attribute a peak without
touching the source, wrap the ttnn op that marks it:

```python
real = ttnn.sum
def probed(*a, **k):
    memstat.probe(mesh, "at reduce")
    return real(*a, **k)
ttnn.sum = probed
```

`ttnn.dump_device_memory_state(mesh)` gives the full block table when you need addresses rather than
totals — which is what an aliasing investigation needs.

### Caveats worth knowing before you trust a number

- `ttnn.deallocate` defaults to `force=True`, so it frees the buffer even when another tensor shares
  it. Never pass a `ttnn.reshape` **view** — `gate_up_w`/`down_w` in `_dense_experts` are views of the
  expert *weights*, and freeing those destroys the model.
- Deallocating an operand after its consumer is enqueued is safe for standard ttnn ops (one command
  queue, FIFO). It is **not** safe for the custom `ttl` kernels here — see the `keep` list in
  `tt/gated_delta.py`, which documents a measured corruption from exactly that.
- The first request/iteration of any measurement includes JIT compile and device settling; one
  observed pair differed 3.0 s vs 21.2 s for identical work. Always compare steady-state iterations.
