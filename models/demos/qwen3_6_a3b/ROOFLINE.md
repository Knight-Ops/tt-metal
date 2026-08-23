<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
     SPDX-License-Identifier: Apache-2.0 -->
# Qwen3.6-35B-A3B — the roofline, and where the time actually goes

The single reference for "how fast *could* this be, and which component is furthest from its own
limit". Written because the roofline in this tree has now been re-derived three times, each revision
superseding the last (see `PREFILL.md`'s history note), and each time the *mechanism* was what decided
which levers could work. Keep this file current; put the levers themselves in
`FUTURE_OPTIMIZATIONS.md` (decode) and `PREFILL.md` (prefill), and the experiment ledger in
`EXPERIMENTS.md`.

All numbers: single Blackhole P150, 40 layers (30 gated-delta + 10 full-attention), BFP4 experts,
tt-metal v0.77.0.

---

## 1. The machine

| quantity | value | source |
|---|---|---|
| usable Tensix | **110** (11×10) | measured `compute_with_storage_grid_size()`. A *harvested* P150 exposes 11 worker columns, not the 13 the tech report mentions — `models/demos/blackhole/qwen36/tt/tp_common.py:37` |
| L1 per core | 1.5 MB (1 536 KB) | |
| DRAM banks | 8 | `blackhole/qwen36/tt/tp_common.py:17` |
| DRAM peak | **432 GB/s assumed** — *unpinned, see below* | tt-npe P150 model (8 × 54) |
| matmul peak | ~594 TFLOP/s LoFi (110 × 5.4 @ 1.35 GHz); 297 HiFi2; 148 HiFi4 | `tech_reports/GEMM_FLOPS/GEMM_FLOPS.md:67` |
| cycles/tile | LoFi 16, HiFi2 32, HiFi3 48, HiFi4 64 | `GEMM_FLOPS.md:107` |

> **The DRAM peak is not pinned and it matters.** Every "% of peak" below uses 432 GB/s, so they are
> the *optimistic* readings; the p150a datasheet figure is higher (~512 GB/s), which would scale every
> utilization number down by ~16%. `tech_reports/Saturating_DRAM_bandwidth` documents 92% of spec as
> reachable with one reader per bank and per-reader NOC virtual channels. **TODO (Phase 0):** measure
> it with `tests/ttnn/unit_tests/operations/transformers/test_prefetcher_BH_bw_bench.py` and replace
> this row with a measured number.

### Two hardware facts that decide the strategy

1. **`GEMM_FLOPS.md`: an input shorter than 8×16 (e.g. 1×16) gets 1/8 of the FPU throughput.** An M=1
   decode matmul is therefore capped at 12.5% of compute peak no matter what. Decode can only ever be
   bandwidth- or launch-bound, so **the only meaningful target for a decode matmul is % of DRAM peak**
   — never TFLOP/s.
2. **Steady-state traced op-to-op gap is ~600–770 ns** (`models/demos/llama3_70b_galaxy/tests/decoder_perf_targets_6u.json`).
   ~2 600 decode ops therefore contribute only ~1.8 ms of *gap*; the other ~25 ms is kernel time.
   "Reduce dispatch" is the wrong frame. The right one is: **make each kernel move its bytes
   efficiently, and delete kernels that move no bytes.**

---

## 2. Decode budget

Weight bytes per token — the number the whole decode analysis rests on. BFP4_B = 0.5625 B/elem,
BFP8_B = 1.0625, bf16 = 2.

| component | per layer | ×layers | bytes/token |
|---|--:|--:|--:|
| GDN projections (`w_in_proj` 2048×12352 + `w_out` 4096×2048, bf8) | 35.8 MB | 30 | 1 075 MB |
| attention (wq+wgate 2×2048×4096, wk+wv 2×2048×512, wo 4096×2048, bf8) | 29.0 MB | 10 | 290 MB |
| MoE router + shared expert (**bf16**) | 7.34 MB | 40 | 294 MB |
| MoE routed experts, 8 of 256 (bf4) | 14.16 MB | 40 | 566 MB |
| lm_head 2048×248320 (bf4) | — | 1 | 286 MB |
| GDN recurrent state (read+write, aliased in place) | 2.1 MB | 30 | 63 MB |
| KV read @ P=1024 | — | 10 | 21 MB |
| **total** | | | **≈ 2.60 GB/token** |

**Bandwidth floor = 2.60 GB / 432 GB/s = 6.0 ms/token = 166 tok/s.**

### 2.1 Where the 28.6 ms actually goes

Reconstructed from `CURRENT_BENCHMARK.md` §2 (real `execute_trace` per-component replays) and §3a
(signpost self-times), cross-checked against the byte table above. This is the map to optimize against.

| bucket | ms/tok | % step | bytes/tok | achieved | % of 432 GB/s |
|---|--:|--:|--:|--:|--:|
| MoE router (linear + softmax·E + topk + sum + div) | 3.63 | 12.7 | 42 MB | 12 GB/s | **3%** |
| MoE routed experts (2× `sparse_matmul` + swiglu) | 4.36 | 15.2 | 566 MB | 130 GB/s | **30%** |
| MoE shared expert | 2.73 | 9.5 | 252 MB | 92 GB/s | **21%** |
| MoE `scatter_routing` — *dead work, see §2.3* | 0.92 | 3.2 | 0 | — | — |
| 81× `rms_norm` (single-core, interleaved) | 2.43 | 8.5 | ~0 | — | — |
| GDN `w_in_proj` ×30 (interleaved, ttnn AUTO config) | 2.23 | 7.8 | 807 MB | 362 GB/s | **84%** (microbenched 74%) |
| GDN fused recurrence kernel ×30 (**32/110 cores**) | 2.25 | 7.9 | 63 MB | 28 GB/s | **6%** |
| GDN conv add-chain ×30 (10 ops each) | 1.20 | 4.2 | ~2 MB | — | — |
| GDN `w_out` ×30 (DRAM-sharded) | 0.71 | 2.5 | 267 MB | 376 GB/s | **87%** |
| attention qkv ×10 | 1.04 | 3.6 | 200 MB | 192 GB/s | 44% |
| attention `wo` ×10 | 0.34 | 1.2 | 89 MB | 262 GB/s | 61% |
| attention RoPE ×10 (**23 hand-rolled ops each**) | 0.32 | 1.1 | ~0 | — | — |
| attention qk_norm + kv_write + sdpa | 0.65 | 2.3 | ~21 MB | — | — |
| lm_head + argmax | 1.33 | 4.6 | 286 MB | 232 GB/s | 54% |
| op-to-op gap + unattributed | ~4.2 | 14.7 | — | — | — |
| **total** | **28.6** | 100 | 2.60 GB | 91 GB/s | **21%** |

Three readings, in order of importance:

1. **The GDN projections are already at 74–87% of peak** (`w_in_proj` interleaved/auto, `w_out` DRAM-sharded). They are *done*. Their only
   remaining lever is fewer bytes (bf8→bf4), and because they are near-roofline their time is
   *directly proportional* to bytes — which makes precision a real lever there, not the marginal one
   `FUTURE_OPTIMIZATIONS.md` Lever 4 assumed.
2. **The worst offenders are not the big matmuls.** They are the router (3% of peak), the GDN
   recurrence kernel (6%, on 32 of 110 cores) and the shared expert (21%).
3. **~11.8 ms of the step moves essentially no bytes at all** (norms, conv, recurrence kernel, RoPE,
   scatter, glue). That is the floor that survives every bandwidth fix.

### 2.2 Decode ceilings

| scenario | matmul time | step | tok/s |
|---|--:|--:|--:|
| today | 16.9 ms | 28.6 ms | 34.9 |
| every matmul at 80% of peak | 7.3 ms | ~19 ms | ~52 |
| + weight bytes halved (bf4 where PCC holds) | 4.5 ms | ~16 ms | ~60 |
| + non-matmul work halved | 4.5 ms | ~10 ms | ~95 |
| pure bandwidth floor (unreachable — no op takes 0 time) | 6.0 ms | 6.0 ms | 166 |

### 2.3 Measured: routing was 16% of the step and half of it was dead
2026-08-22, `bench_decode.py --iters 200`, 40 layers, real weights, paired A/B in one tree via
`QWEN36_MOE_ROUTER_FAST`:

| arm | ms/token | tok/s/user | MoE ms/op |
|---|--:|--:|--:|
| `QWEN36_MOE_ROUTER_FAST=0` (legacy) | 28.61 | 35.0 | 0.318 |
| `QWEN36_MOE_ROUTER_FAST=1` (default) | **26.72** | **37.4** | 0.271 |
| delta | **−1.89** | **+6.9%** | −0.047 |

Two changes, neither altering what the block computes: top-k of the raw **logits** + softmax over the
k survivors (instead of softmax over E, then top-k, then sum+divide), and passing `sparse_matmul` a
cached constant `sparsity` on the gather path because **indexed mode never reads that operand**
(`ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:1075`). The MoE delta (−0.047 ms/layer × 40 =
−1.88 ms) accounts for the entire full-model delta, and the control reproduces the historical
28.65 ms / 34.9 tok/s baseline to 0.1% — which is also a check that the harness and the baseline agree.

Worth recording: the pre-measurement estimate was −3 to −3.9 ms and the result was −1.89. That is the
fifth time in this project an isolated estimate over-predicted a 40-layer result. Halve your estimates.

### 2.4 The decode MoE is closed (2026-08-22)
40% of the decode step, and both remaining levers are now measured shut:
- `in0_block_w` on the two expert `sparse_matmul`s is **already optimal** (32 / 16); the curve spans 8x
  from 1 to 32 so the mechanism is real, but gate_up saturates at 32 and `down`'s `Kt` IS 16.
- The achieved 38% / 31% of DRAM peak tracks the **core count** (32 and 64 of 110, pinned by
  `per_core_N=1` making the grid equal `Nt`). Neither `Nt` nor `Kt` can grow.
- `moe_compute` FullLocal, the only op that shards experts across the whole grid, **streams every
  expert** (measured linear in E) — 453 MB/layer against the gather path's 14.16 MB. 32x the bytes for
  3.4x the cores: **14x worse for decode**.

So the gather formulation is structurally right and further gains need a new op. See
`FUTURE_OPTIMIZATIONS.md` Levers 2b and 2c.

### 2.4b Measured: the untuned matmuls (`QWEN36_WIDE1D_DECODE`)
Same harness, paired A/B: **26.74 → 25.49 ms/token, 37.4 → 39.2 tok/s/user (−1.25 ms, +4.8%)**, from
putting tuned wide 1D `mcast_in0` configs on the decode matmuls that ran on the ttnn AUTO heuristic.
Attribution: `wk`/`wv` −0.41 ms, `se_gate_up`+`se_down` −0.72, `w_in_proj` −0.09, `lm_head` **0.00**.
Projected −2.5 ms, delivered −1.25 — and −0.70 of the shortfall is `gate_w`, deliberately left off
because its accumulation-order change can flip expert selection (FUTURE_OPTIMIZATIONS.md).

**Cumulative today: 28.65 → 25.49 ms/token, 34.9 → 39.2 tok/s/user (+12.3%).**

### 2.5 The term this budget hides: KV traffic grows with context
Every number in §2.1 is measured near position 0, where the KV cache is empty. It is therefore a
budget for the *weight-bound* part of the step, and it silently omits the one term that grows: SDPA
reads the whole cache every step, `2 · n_kv · head_dim · pos` bytes per attention layer, which at
bf16 is **20 KiB/token** across the 10 attention layers.

Measured, `tests/probe_paged_kv_perf.py`, per attention layer:

| pos | KV MB/step bf16 → bf8 | sdpa-decode bf16 | sdpa-decode BFP8 | ratio | Δ per token (10 layers) |
|---|--:|--:|--:|--:|--:|
| 1 024 | 2.0 → 1.1 | 36.2 µs | 33.8 µs | 1.07× | −0.02 ms |
| 8 192 | 16.0 → 8.5 | 83.2 µs | 62.4 µs | 1.33× | −0.21 ms |
| 32 768 | 64.0 → 34.0 | 227.7 µs | 137.4 µs | 1.66× | −0.90 ms |
| 131 072 | 256.0 → 136.0 | 796.7 µs | 438.8 µs | 1.82× | −3.58 ms |

Two things to read off this:

- **At short context KV is noise; past ~8K it is the largest single line in the budget.** At 128K,
  bf16 SDPA alone is 8.0 ms/token — a third of the whole step — against 0.36 ms at 1K. Any decode
  optimisation ranked at position 0 (i.e. all of §2.1–2.4b) is ranked in the regime where this term
  does not exist.
- **The ratio rising 1.07 → 1.82 with position is the tell that it is pure bandwidth.** BFP8 is
  1.0625 B/elem against bf16's 2, so the asymptote is 1.88×; reaching 1.82 at 128K is **96.5% of that
  ideal**, which says the kernel is doing nothing but moving the cache. That is also why halving the
  bytes is the *only* lever here — there is no compute to tune. At 1K the same op realises just 1.07×,
  because at 1K it is not bandwidth-bound yet.

`QWEN36_KV_DTYPE=bf8` buys that, and §"BFP8 KV cache" in EXPERIMENTS.md has the accuracy: teacher-forced
decode logit PCC 0.995/0.996, stable, on par with the shipped conv add-chain's own gate. It is not the
default because at the short contexts this file benchmarks it buys nothing measurable; it is the right
setting for long-context serving, and the crossover is around 8K.

---

## 3. Prefill budget

MoE expert stages = **5.62 ms/layer per 256-token chunk × 40 = 225 ms of a 336.5 ms T=256 call (67%)**.

Per layer per 256-token chunk at the shipping config (BFP8 matmul in0, BFP4 weights):

| stage | bytes | note |
|---|--:|---|
| `repeat` → `xe [256,256,2048]` | 143 MB | pure broadcast copy; the least bandwidth-efficient stage (~45% of peak) |
| `gate_up` matmul | 516 MB | 302 weights + 143 in0 + 71 out |
| 2× `slice` (gate/up) | 143 MB | |
| `silu` + `multiply` | 178 MB | |
| routing-weight `multiply` | 71 MB | |
| `down` matmul | 329 MB | 151 weights + 36 in0 + 143 out |
| `sum` over E | 144 MB | |
| **total** | **1 524 MB** | of which only **453 MB is weights** |

**1 524 MB in 5.62 ms = 271 GB/s = 63% of peak** — *not* the 76% recorded in `PREFILL.md` §1.1, which
was computed from pre-BFP8 byte counts and should be read as superseded.

Dense MoE FLOPs are `2·E·(H·2I + I·H) = 1.61 GFLOP/token/layer` → **64.4 GFLOP/token** → 16.5 TFLOP per
256 tokens over 40 layers → **127 ms at the measured ~130 TFLOP/s** (22% of LoFi peak).

So the dense MoE is **co-limited**: a ~141 ms DRAM floor *plus* ~127 ms of compute, measured 225 ms.
Neither number alone explains it, which is why single-mechanism reasoning about this block has
repeatedly produced wrong predictions.

### 3.1 The three consequences

1. **70% of the bytes are intermediates, not weights**, and every one scales with `E = 256`. Removing
   them is a *formulation* change, not a tuning change. Arithmetic intensity today is 271 FLOP/byte,
   so DRAM caps the block at ~117 TFLOP/s and we achieve 73.
2. **Flattening the batched per-expert matmul into a plain 2D matmul over a merged `expert×channel`
   axis** takes the bytes to ~670 MB and the intensity to ~616 FLOP/byte, moving the DRAM-implied cap
   to ~266 TFLOP/s — i.e. it converts the block from bandwidth-bound to compute-bound, which is the
   regime where a better matmul primitive (`ttnn.experimental.minimal_matmul`, `fuse_swiglu`) pays.
3. **Dense MoE has a hard ~2 000 tok/s ceiling from FLOPs alone**, at any byte cost. Going past it
   requires genuine top-k sparsity (a variable-M grouped expert FFN), not more tuning.

### 3.2 Calibration against another single-P150 implementation
MuseGlimmer (dense ~13 B active, 52 layers, BFP8 attention + BFP4 MLP) reports 1 549 tok/s prompt
processing. Its dense MLP is 41.5 GFLOP/token, so that is ~64 TFLOP/s achieved — against our
64.4 GFLOP/token at ~800 tok/s ≈ 51 TFLOP/s. **Its prefill is not more efficient than ours; it simply
does less work per token.** That is the cleanest available confirmation that the 32× MoE FLOP
redundancy is the prefill problem, and that both implementations leave 5–9× of matmul efficiency on
the table.

Its decode, by contrast, reports ~9 GB/token at 56 tok/s ≈ **100% of DRAM peak**, using nothing more
exotic than `ttnn.linear` with a wide 11×10 1D-mcast program config and a tuned `in0_block_w` on
interleaved weights. (That byte figure is inferred from its config, not stated by the project, so
treat the number as unverified — but it means our 21% is not a hardware limit.)

---

## 4. How to refresh this file

```bash
# decode: the authoritative ms/token + per-component breakdown
QWEN36_LAYERS=40 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/bench_decode.py --iters 200
# prefill: FIXED lengths only (tests/benchmark.py reports T/TTFT, which is not throughput)
QWEN36_LAYERS=40 QWEN36_MAX_SEQ=2048 ./python_env/bin/python \
    models/demos/qwen3_6_a3b/tests/bench_prefill.py --seqs 256,512,1024
# per-matmul % of DRAM peak, and the layout/grid/in0_block_w sweep behind §2.1
./python_env/bin/python models/demos/qwen3_6_a3b/tests/bench_matmul_layout.py --verbose
# per-op self time (4 layers: PROPORTIONS only, absolutes inflated ~1.8x; --op-support-count REQUIRED)
QWEN36_LAYERS=4 ./python_env/bin/python -m tracy -r -v --op-support-count 6000 \
    models/demos/qwen3_6_a3b/tests/prof_decode.py
python models/demos/qwen3_6_a3b/tests/signpost_report.py --between eager_start eager_stop
```

**Do not run a device benchmark under `timeout`.** Killing a run mid-flight wedges the board
("Read 0xffffffff over PCIe ID 0: the board should be reset"), and recovery needs `tt-smi -r`. Run long
benchmarks detached instead, and check the log.
