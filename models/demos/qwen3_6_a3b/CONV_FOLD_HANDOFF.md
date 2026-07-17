# Conv-Fold (Gated-DeltaNet decode) — Engineering Handoff

> **RESOLVED (2026-07-16 session 3) — a DIFFERENT way, NOT the on-core fold this doc describes.**
> Per-op profiling (leave-one-out inside the full gated-delta step) showed the decode conv's ~0.177 ms/layer
> is **87% two dim-0 data-movement ops** (concat 0.075 + state-slice 0.079), only ~15% the math — and those
> two ops **cannot be removed bit-identically** (ttnn `slice`/`concat` have no `output_tensor=`; the fp32
> `sum`-over-K needs the concat's stacked rows). The fix that shipped (behind `QWEN36_CONV_ADDCHAIN`): keep
> the conv history as **K-1 separate `[1,conv_dim]` row buffers** and compute the decode conv as a **bf16
> add-chain** with a cheap 3-copy window shift — removing both dim-0 ops with NO kernel work and NO DFB-budget
> fight. **40L: Gated-DeltaNet 0.451→0.323 ms/op, decode 31.5→34.6 tok/s/user (+9.8%).** Crucially, the
> add-chain uses ttnn's own bf16 ops (unlike the on-core kernel's matmul-ones row-sum) so it rounds ≈
> baseline: `test_gated_delta_decode` 0.99979, `test_trace`/`test_model` pass, teacher-forced decode
> logit-PCC vs baseline (40L) mean **0.996 and stable** — NOT the 0.885→0.37 divergence below. The on-core
> fold (§6) is therefore superseded and no longer worth pursuing. See FUTURE_OPTIMIZATIONS.md Lever 3b.
> The rest of this doc is the record of the earlier reverted on-core attempt.

Status: **on-core attempt: attempted, reverted, not shipped** (2026-07-16); **superseded by the add-chain
above.** This doc + `FUTURE_OPTIMIZATIONS.md` Lever 3b are the record.

---

## 1. Problem statement

In the single-token (T=1) gated-delta **decode** path, the depthwise causal conv + SiLU is computed by
`TtGatedDeltaNet._conv_silu` (`tt/gated_delta.py`) as ~5 separate TT-NN ops over the full
`conv_dim = 8192` (256 tiles): `concat([conv_state, mixed])` → `multiply(xpad, conv_taps_stacked)` →
`sum(dim=0)` → `silu`, plus a `slice` for the new conv-state window.

**Measured cost (this is the motivation):** the conv is **0.207 ms per gated-delta layer** in a trace —
**~46% of the 0.451 ms GDN decode step, ≈ 6.2 ms/token (~19% of the 33 ms decode)**. It is the single
biggest remaining decode pool. (FUTURE_OPTIMIZATIONS previously guessed "<0.5 ms/token" — that was wrong
by >10×; see §3 for how it was measured.)

**Goal:** move the conv on-core so those per-op dispatches disappear, without regressing decode accuracy
or breaking metal-trace capture.

**Outcome:** a separate on-core conv kernel is numerically correct but (a) fails the decode PCC gate and
(b) delivers no net speedup. The real win requires folding the conv *into* the existing fused recurrence
kernel `decode_step_tt`, which is blocked by a hard resource limit (§5). Deferred.

---

## 2. Key facts about the code (orientation)

- Decode conv: `tt/gated_delta.py::_conv_silu` (T==1 branch), called from `forward` (~L556–590). The conv
  output is sliced into q/k/v (`[q(key_dim) | k(key_dim) | v(value_dim)]`, `conv_dim = 2·key_dim +
  value_dim`), which then feed `_forward_decode_fused` → `decode_step_tt`.
- Taps: `conv_taps_stacked` `[conv_k, conv_dim]` (bf16), `conv_k = 4`. Built in `__init__` (~L179).
- conv_state: persistent DRAM buffer `[conv_k-1, conv_dim]`, pre-allocated in `tt/model.py::_alloc_cache`
  (trace-safe: updated in place via `ttnn.copy`, never reallocated).
- The recurrence kernel `decode_step_tt` (`tt/ttl_delta.py::_get_decode_op` / `_decode_step`) already does
  l2norm(q)·scale, l2norm(k), gate/beta, the rank-1 recurrence, and the gated RMSNorm — **all in one ttl
  launch**. It maps **one value-head per Tensix core** (grid `gn·gm == n_v_heads = 32`); each core reads
  its head's tile-aligned **column slabs** of q/k/v.
- **Hard constraint:** ttl's DFB (dataflow-buffer) budget is **32 per op** (`tt/ttl_delta.py` ~L397). The
  recurrence kernel already declares **29**.
- Model dims (real ckpt): `conv_dim=8192`, `key_dim=2048`, `value_dim=4096`, `n_k_heads=16`,
  `n_v_heads=32`, `head_k_dim=head_v_dim=128`, `conv_k=4`, `rep=2`.

---

## 3. What I measured first (Gate 0 — sizing the prize)

- `tests/prof_gdn_phases.py` (eager, per-phase device-synced, **non-fused** decomposition): `conv_silu` =
  26% of a 19.1 ms total. This over-states absolute cost (each phase eats a full sync; total is ~25× the
  0.45 ms traced step) — treat as a rough "conv is a big phase" signal only.
  - Gotcha: this harness references `m.w_qkv`, which doesn't exist under the default fused input proj
    (`_FUSE_IN=1` uses `w_in_proj`). Run it with `QWEN36_GDN_FUSE_IN=0`. (The conv phase is unaffected by
    input-proj fusion.)
- **Authoritative number — traced marginal cost via a slope** (scratchpad `gate0_conv_slope.py`): trace
  the conv done 1×/5×/10× and take the slope to cancel the fixed `execute_trace` replay overhead:
  `conv 1×=0.227, 5×=1.058, 10×=2.094 ms` → **marginal = 0.207 ms/layer** (fixed overhead only ~0.02 ms).
  ×30 gated-delta layers = **6.2 ms/token**.
  - Lesson: do NOT read a tiny isolated trace's absolute time as its in-model cost — the per-`execute_trace`
    fixed overhead dominates a small trace. Use the slope.

---

## 4. What I built (the attempt)

A **separate** on-core conv kernel `conv_step_tt` (in `tt/ttl_delta.py`), gated by `QWEN36_GDN_CONV_FOLD`,
leaving the recurrence kernel untouched. Design rationale: the conv is **depthwise (per-channel)**, so it
partitions perfectly over `conv_dim` tile-columns — one core owns a contiguous slab, no cross-head/cross-
channel reduction, no inter-core comms.

Per core, for its tile-column slab:
```
prod  = conv_state_slab * taps_prev_slab          # [1 tile-row (K-1 valid), cpc]
sumc  = ones[1,1] @ prod                           # row-sum over the K-1 window via ones-tile matmul
out   = silu(sumc + mixed_slab * tap_cur_slab)     # add current-tap term, then silu
```
Design details that mattered:
- **Tap split (Python, once):** `taps_prev = stacked[:K-1]` **explicitly zero-padded to a full tile**, and
  `tap_cur = stacked[K-1]`. Zero-padding `taps_prev`'s rows K-1..31 guarantees `prod`'s padding rows are 0
  regardless of `conv_state`'s tile-padding, so the 32-row `ones @ prod` matmul correctly sums only the
  K-1 window.
- **Row-sum via matmul:** ttl can't reduce a within-tile row axis directly; left-multiplying by an all-ones
  `fill(1.0,(1,1))` tile sums the 32 rows. (`ttl.block.fill` fills the whole tile.)
- **conv_state window-shift stays in Python:** `new = [conv_state[1:], mixed]` is a sub-tile *row* shift
  that ttl can't do cleanly — kept as `ttnn.slice`+`ttnn.concat`+in-place `ttnn.copy`.

**This kernel is CORRECT.** Verified three ways (scratchpad `dbg_conv.py`, `dbg_conv_multi.py`,
`dbg_model_conv.py`):
- vs a torch reference conv: PCC **0.99998**.
- vs `_conv_silu` in-flow on the real model (`conv_dim=8192`): PCC **0.99999**, all q/k/v regions.
- 6-step sequence incl. the state-shift: conv PCC 0.99999, conv_state PCC **1.0** each step.
- No bias: `mean(fold − _conv_silu) ≈ 1e-5`; q/k error std 0.0002, v error std 0.0013.

---

## 5. What did NOT work (two independent, fatal failures)

### 5a. Accuracy — fails the decode PCC gate (0.885 vs 0.9998 baseline)
`tests/test_gated_delta_decode.py` (prefill 8 + decode 6, PCC vs full-recompute reference) drops from
**0.9998 (baseline) to 0.885 with fold on**, and it *accumulates* over decode steps (per-token
0.885→0.66→0.37→…). The conv is 0.99999-accurate, yet decode diverges. Localization + eliminations
(scratchpad `dbg_test_repro.py`, env-gated debug branches):
- **Localized to q/k, not v.** Feeding conv_step_tt's v-region but `_conv_silu`'s q/k → **0.9998** (fine);
  the reverse (conv_step_tt q/k, `_conv_silu` v) → **0.948→0.30** (broken). q/k go through **`l2norm`**
  (direction-sensitive, scale-invariant); v feeds the recurrence directly (robust to small error).
- **It is structured, not magnitude.** Perturbing `_conv_silu`'s output with **random** noise of *larger*
  magnitude (0.0007 std) → recurrence stays **0.998**. conv_step_tt's *smaller* but structured bf16
  rounding-order difference (matmul row-sum vs `ttnn.sum`'s fp32 accumulate) → 0.885. So the recurrence's
  `l2norm(q/k)` tolerates random perturbation but not this particular structured rounding.
- **Ruled out:** WAR hazard (adding `synchronize_device` after the kernel changed nothing — deterministic);
  persistent-buffer reuse (fresh buffer per step — identical failure); bf16 accumulation
  (`fp32_dest_acc_en=True` on the conv kernel — 0.885→0.887, no help); systematic bias (mean err ~1e-5).
- **Real model, not just the random-weight test:** a real 4-layer greedy run diverges from baseline after
  token 1 (genuine hidden-state divergence, not just an argmax flip).

**Why this is the crux:** this model's decode sits at the edge of bf16 determinism (the codebase already
notes argmax flips between equivalent chunked-vs-single prefill, PCC ~0.996, and treats it as "not a bug").
**Any** conv reimplementation whose bf16 rounding differs from ttnn's `_conv_silu` perturbs q/k enough that
`l2norm`+recurrence changes the decode trajectory → it fails a strict PCC / token-identity gate. To pass
that gate you would have to reproduce `_conv_silu`'s exact op sequence (multiply → fp32-accumulate sum),
which defeats the point.

> **Correction to an earlier note:** I initially wrote in FUTURE_OPTIMIZATIONS that folding into
> `decode_step_tt` would avoid the accuracy delta "because l2norm sees the same in-kernel values." **That
> is wrong.** An in-kernel conv still rounds differently from ttnn's `_conv_silu`, so q/k still differ from
> the shipped path and the same `l2norm` sensitivity applies. Folding in-kernel fixes the *perf* problem
> (§5b), **not** the accuracy one.

### 5b. Perf — the separate kernel delivers no net win
40-layer bench with fold on: **GDN 0.451 → 0.437 ms/op** (−0.014 only); full model **33.1 ms (30.2 tok/s)**
= no improvement over the #3-only config (32.81 ms / 30.5 tok/s). A **separate kernel launch costs ≈ the 5
TT-NN ops it replaced**. The 0.207 ms slope over-counted the in-model marginal cost because those TT-NN
ops **pipeline with neighbouring ops** in the full decode step, whereas a standalone launch does not
overlap. (The perf bench was run with `fp32_dest_acc_en=True`, so the number is slightly conservative, but
not enough to change the verdict.)

---

## 6. Recommended path forward (for the ~6 ms prize)

**Fold the conv INTO `decode_step_tt` itself** (no separate launch → solves §5b). The depthwise conv maps
onto the kernel's existing per-value-head column slabs: each core already reads its q/k/v slabs at tile-
aligned offsets `cq = hk·kt_t`, `key_dim//TILE + hk·kt_t`, `2·key_dim//TILE + h·vt`. Replace those 3 input
reads with reads of `mixed` / `conv_state` / `taps_prev` / `tap_cur` slabs, compute the conv (the §4 recipe)
into the existing `qd`/`krd`/`vd` buffers *before* the l2norm, and keep the conv_state shift in Python.

Two obstacles, in order:
1. **DFB budget (hard blocker).** The recurrence kernel is at 29/32; naively adding ~12 conv-input buffers
   (mixed/state/taps × q/k/v slabs) overflows. Options: process q→k→v conv **sequentially**, reusing ~4
   input buffers read 3× (block_count pipelines it); reuse existing pre-l2norm temps (`sq_t`/`cp_t`/`onesm`)
   as conv scratch (they're free before the conv). This is the main engineering effort and needs careful
   ttl work + compile iteration (killing a run mid-ttl-compile hangs the board — use >900 s timeouts,
   `tt-smi -r` to recover).
2. **Accuracy still applies (§5a).** Folding in-kernel does not by itself make q/k bit-match `_conv_silu`.
   Before investing, **decide the real gate**: run an actual **quality eval** (`evaluation/` MMLU-Redux
   harness) on the folded model rather than the random-weight PCC unit test. Decode token-identity is too
   strict here (bf16 determinism edge). If MMLU is unchanged, relax the gate to a quality eval and ship. If
   it regresses, the fold is not viable regardless of perf.

Cheaper alternative if the full fold proves too costly: attack the conv's TT-NN op **count/overlap**
instead of moving it on-core (e.g., verify whether `concat` or an interleaved↔L1 round-trip is the actual
hotspot within the 0.207 ms and remove just that). Lower ceiling, lower risk.

---

## 7. Reproduction / pointers

- Env flag used while attempting: `QWEN36_GDN_CONV_FOLD=1` (removed on revert).
- Accuracy gate: `QWEN36_GDN_CONV_FOLD=1 pytest tests/test_gated_delta_decode.py` → was 0.885.
- Perf: `QWEN36_LAYERS=40 python tests/bench_decode.py --iters 300` (compare GDN ms/op + full-model).
- Conv-share sizing: `tests/prof_gdn_phases.py` (with `QWEN36_GDN_FUSE_IN=0`) + the slope method.
- Scratchpad debug scripts (this session, isolate each layer of the problem):
  `gate0_conv_slope.py` (marginal conv cost), `dbg_conv.py` / `dbg_conv_multi.py` (kernel vs torch,
  single + multi-step), `dbg_model_conv.py` (kernel vs `_conv_silu` on real weights + bias/region check),
  `dbg_test_repro.py` (per-token PCC repro of the unit test, with `MIX`/`NOISE`/`FRESH`/`DBG` probes).
- The correct, reusable pieces if re-attempting: the **tap-split** (zero-padded `taps_prev` + `tap_cur`)
  and the **ones-tile matmul row-sum** — both verified correct.

## 8. One-paragraph TL;DR
The decode conv is ~6 ms/token (~19%), the biggest untapped lever. A separate on-core conv ttl kernel is
numerically correct (0.99999) but (1) a separate launch costs as much as the ops it replaces → no speedup,
and (2) its bf16 rounding differs from ttnn's `_conv_silu` just enough that `l2norm(q/k)` + the recurrence
diverge on the strict PCC gate (0.885). The real win needs the conv folded *into* `decode_step_tt`
(blocked by the 32-DFB budget), and the accuracy question must be settled with a quality eval (MMLU), not
token-identity, because this model's decode is at the edge of bf16 determinism.
