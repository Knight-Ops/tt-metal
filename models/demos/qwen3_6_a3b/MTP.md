<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
     SPDX-License-Identifier: Apache-2.0 -->
# MTP / speculative decode — feasibility measurements (Phase A)

Verdict: **GO**. Measured 2026-08-20 on tt-metal v0.77.0, single P150, 40 layers.
Probes: `tests/probe_mtp_verify.py` (verify costs), `tests/probe_mtp_acceptance.py` (acceptance),
`reference/qwen3_5_moe.py::Qwen35MoeMTP` (torch reference for the shipped `mtp.*` head).


## A0 — baseline (DONE, 2026-08-20, v0.77, 40L real weights)

    execute_trace (pure device) :   28.79 ms/token  ->  34.7 tok/s
    decode_step (incl. readback):   29.01 ms/token  ->  34.5 tok/s/user   <- THE BASELINE
    breakdown x layer counts    :   26.80 ms        ->  37.3 tok/s        <- NOT a real rate

**Premise correction:** the "37 tok/s" figure is the *sum of independently-traced per-component
microbenchmarks*, not a measured decode rate. It under-counts the real step by 2.21 ms (7.6%) of
on-device inter-op dispatch gap. Commit 759ac74b235 quoted that derived line. True baseline is
**34.5 tok/s / 29.01 ms**, unchanged from v0.74. This helps MTP slightly (same verify cost against a
slower baseline).

Per-component (this run):

| component | ms/op | x layers | total | share |
|---|--:|--:|--:|--:|
| Gated-DeltaNet | 0.319 | 30 | 9.57 | 33.0% |
| MoE            | 0.326 | 40 | 13.04 | 45.0% |
| Attention      | 0.295 | 10 | 2.95 | 10.2% |
| lm_head        | 1.226 | 1 | 1.23 | 4.2% |
| dispatch gap   | — | — | 2.21 | 7.6% |

MoE split measured separately: router+shared 0.181, sparse experts 0.387 -> 32:68, so of the fused
0.326 ms/layer, **experts ~0.222 and router+shared ~0.104**.

## Amortization budget (what MTP can and cannot recover)

Per decode step, the K-token verify amortizes everything except the routed experts:

| term | ms | amortizes with K? |
|---|--:|---|
| MoE routed experts | 8.88 | **NO — linear in K** (cost is per active expert-slot; slots = K x top_k) |
| Gated-DeltaNet | 9.57 | partially — flat projections/conv + delta per extra on-core step |
| attention | 2.95 | yes (share_cache verify) |
| lm_head | 1.23 | yes (weight-read bound) |
| MoE router+shared | 4.16 | yes (per_core_M=1: 1..32 rows cost the same) |
| dispatch gap | 2.21 | yes (paid once per verify, not per token) |

Asymptotic ceiling (K->inf, alpha=1) = 8.88 + 30*delta ms/token. At delta=0.10 that is 11.9 ms ->
**~2.4x (84 tok/s) absolute ceiling**. Everything below that is lost to alpha.

Why the MoE is structurally linear (first-principles, not just measured): expert weights are
1.57 MB/expert (bf4), 8 experts = 12.6 MB/layer = 0.029 ms at 432 GB/s, but measured 0.222 ms = 13%
DRAM util. The cost is the per-(slot x K-block) multicast handshake (~5.5 us x 5 blocks x 8 slots),
and `per_core_M=1` means adding token rows to a tile is free while adding slots is not. Union dedup
cannot help: at 256 experts / top-8, E[unique] for K=4 is 30.6 of 32 slots (4%).

## A5 host plumbing — VALIDATED (no device needed)

`reference/qwen3_5_moe.py::Qwen35MoeMTP` added; it is an **exact structural match** to the
checkpoint: 19 tensors, all shapes identical, names map 1:1 by stripping the `mtp.` prefix.
Confirms the MTP head is exactly [2 pre-fc norms + fc + ONE full-attention/MoE decoder layer + norm],
sharing embed_tokens and lm_head. `mtp_num_hidden_layers=1`, `mtp_use_dedicated_embeddings=False`
now parsed by `Qwen35MoeConfig`.

Scorer runs end-to-end on this 15 GB box (bf16 weights, row-sliced embeddings, chunked lm_head
argmax; peak ~2.7 GB). Head is 845M params.

## Attention verify (B3) — confirmed trivial

`tt/attention.py::forward_decode` is already fully B-aware with an unpaged
`[B, n_kv, max_seq, head_dim]` cache. MTP verify = pass `share_cache=True` to the two
`paged_update_cache` calls and to `sdpa_decode`, with `current_pos = pos + arange(K)`. Validated the
C++ constraints directly: both ops require cache batch == 1 and reject paged mode, which matches
this model (`page_table=None`). ~6 lines.

## Pending
- A1 verify-MoE gather cost vs K (running)
- A2 chained-GDN delta (the decisive number)
- A3 share_cache attention timing
- A5 capture on device + alpha

## Refinement that changes the verdict: alpha is per-DEPTH, not geometric

`mtp_num_hidden_layers: 1` -> the head is trained to predict exactly ONE token ahead. gamma>1 needs
it run AUTOREGRESSIVELY on its own output (pre-`mtp.norm` hidden fed back), which is off-distribution,
so acceptance decays with depth. The optimistic `1 + a + a^2 + ...` model is wrong. This is why
DeepSeek-V3 ships MTP at depth 1 only. A5 now measures alpha_1..alpha_4 with an explicit draft chain.

alpha needed to clear the 1.25x gate (delta=0.095 assumed):

| gamma | round ms | E[tokens] needed | implies |
|--:|--:|--:|---|
| 1 | 43.2 | 1.861 | **alpha_1 >= 0.86** |
| 2 | 57.0 | 2.456 | alpha ~ 0.79 flat (unlikely off-distribution) |
| 3 | 70.7 | 3.046 | alpha ~ 0.76 flat |

Realistic estimate: gamma=1 at alpha_1 ~ 0.85 -> **~1.24x (43 tok/s)**.

## A1b — a NEW orthogonal lever found while building the probe (untested at time of writing)

`TtMoE._sparse_pc` (moe.py:200-221) uses ONE `in0_block_w` for BOTH expert matmuls, and the comment
at moe.py:204-208 caps it at 16 because it "must divide Kt for both expert matmuls (gate_up Kt=64,
down Kt=16 -> common divisors 1,2,4,8,16)". But the two `sparse_matmul` calls take SEPARATE program
configs, so **gate_up is not actually constrained by down's Kt=16 and could use 32 or 64**.
`in0_block_w` is K-tiles per multicast block, and the per-block multicast-semaphore handshake is the
measured decode-MoE bottleneck (moe.py's own comment). gate_up at in0_block_w=64 does 1 handshake
per expert slot instead of 4.

If it works this speeds up **plain single-user decode**, not just MTP verify: MoE is 45% of the
decode step and the routed experts are 8.88 of 29.01 ms. It also compounds with MTP, whose verify
cost is MoE-dominated. Probed as section A1b over (gu_bw, dn_bw) in {16,32,64}x{16,8}.

## Environment notes / gotchas hit
- `close_mesh_device` reliably hangs for minutes with large bf4 DRAM tensors + a 300 MB trace region
  resident. Mitigation: each probe section persists its numbers to JSON via `_save()` BEFORE
  returning, and tracebacks are printed inside the try block rather than propagating past the hang.
- Always run probes with `python -u`: a SIGKILL of a block-buffered process discards the results.
- `ttnn.from_torch` cannot emit UINT16 from an int32 host tensor -> build indices as uint32 then
  `ttnn.typecast(..., uint16)` + `to_layout(ROW_MAJOR)`, exactly as `moe_gather.topk_to_indices` does.
- Building any device tensor INSIDE a timed lambda fatals with "Writes are not supported during trace
  capture" -- the probe hit precisely the hazard the plan calls out for B7. Pre-build everything.
- `tt-smi` is NOT on PATH; it lives at `/home/carl/.tenstorrent-venv/bin/tt-smi` (`-r` to reset).

---

# MEASURED RESULTS (all Phase-A device work complete except A5)

## A1 — verify-MoE, indices-gather mode: PASS

    decode reference (M=1, na=8): 0.1171 ms/layer   [the shipping path]
      K  num_active  ms/layer   vs K x base
      1           8    0.1095         0.94x
      2          16    0.1903         0.81x
      3          24    0.2704         0.77x
      4          32    0.3552         0.76x
      6          48    0.5124         0.73x
      8          64    0.6820         0.73x

Fit: `ms = 0.0277 + 0.01022 * num_active` (R^2 ~ 1). Two conclusions:
1. **Token rows are free** — a 32-row tile costs the same as 1 row (per_core_M=1). Confirmed.
2. **The expert matmuls are only 0.11 of the 0.326 ms MoE op.** The router + shared expert + topk +
   scatter + combine glue is 0.216 ms — the MAJORITY — and it is flat in K.

So `MoE(K) = 0.2165 + 0.0277 + 0.0818K` per layer, which reproduces the measured 0.326 at K=1.
**This refutes the July NO-GO's central premise.** That analysis treated the MoE as ~45% of the step
AND essentially all-linear; in fact only ~25% of the MoE op scales with K.

## A1b — per-matmul in0_block_w: small win, not a lever

`_sparse_pc` uses ONE in0_block_w for both expert matmuls and moe.py:204-208 caps it at 16 for
`down`'s Kt=16 — but the calls take separate program configs, so gate_up (Kt=64) is unconstrained.

    gu/dn   M=1 ms   M=32,na=32   vs 16/16
    16/16   0.1170       0.3513      1.00x
    32/16   0.1082       0.3152      0.92x   <- best
    64/16   0.1089       0.3205      0.93x
    32/8    0.1104       0.3253      0.94x

8-11% on the expert matmuls = ~0.36 ms of the 29.01 ms step (~1.2%). Real but small; the handshake is
not as dominant at these shapes as moe.py's comment implies. Worth a one-line real-model A/B
(`_sparse_pc` gains a per-call in0_block_w, gate_up passes 32), low priority.

## A2 — verify-GDN chained recurrence: PASS, and it makes the new kernel OPTIONAL

    K   chain ms   batch ms   K launch   PCC
    1     0.0270     0.0614     0.0611   0.99997
    2     0.0427     0.1098     0.1187   0.99997
    3     0.0571     0.1556     0.1767   0.99997
    4     0.0715     0.2011     0.2344   0.99994

- The conditional-free chained kernel COMPILES and is numerically correct (PCC >= 0.9999 on both the
  per-step outputs and the per-step states vs a sequential torch reference).
- delta/token = **0.047 ms** (full kernel) — 6.8x below the 0.319 ms/layer gate. Huge pass.
- **The decisive structural finding: the ttl kernel is only 0.061 of the 0.319 ms GDN layer.** The
  other 81% (w_in_proj, w_out, conv add-chain, relayouts) is FLAT in K. So `GDN(K) = 0.2576 + kernel`.
- Therefore **K sequential launches of the EXISTING `decode_step_tt` already amortize GDN ~2.5x**, and
  the new chained kernel is worth only ~2% of the round. **Plan item B5 is a refinement, not a
  prerequisite** — the opposite of what the plan assumed ("without it every gamma is break-even").
  The plan's premise was wrong because it implicitly attributed the whole GDN layer to the kernel.

## A3 — verify-attention via share_cache: PASS

    K   sdpa ms   vs K=1
    1    0.0224    1.00x
    2    0.0253    1.13x
    4    0.0293    1.31x
    8    0.0336    1.50x

`sdpa_decode(share_cache=True)` accepted a batch-1 KV cache with K query rows and per-row
`cur_pos = pos + arange(K)`. Absolute growth is 0.0016 ms/token — negligible against the 0.295 ms
attention layer. The verify-attention primitive is confirmed working end to end; this was the biggest
unknown-unknown.

## Calibrated cost model (script: model.py)

    MoE  = 0.2165 + 0.0277 + 0.0818K   per layer  (x40)
    GDN  = 0.2576 + 0.0614 + 0.0465(K-1) per layer  (x30)
    attn = 0.295 + 0.0016(K-1)         per layer  (x10)
    head = 1.226, gap = 2.21 (once), draft = 2.03/step (A4 estimate)

**Self-check: verify(1) = 29.00 ms vs the measured 29.01 ms step -> 0.05% error.** The decomposition
is sound.

    K   verify ms   vs K decode steps
    2       33.68              0.58x
    3       38.36              0.44x
    4       43.05              0.37x

Projected speedup (chained kernel; "decayed" = alpha_1=0.85 then 0.55/0.40/0.30):

    gamma  round ms   a=0.8   a=0.85    decayed
        1     36.21   1.44x    1.48x    1.48x (51 tok/s)
        2     42.92   1.65x    1.74x    1.57x (54 tok/s)   <- best
        3     49.63   1.73x    1.86x    1.46x (50 tok/s)

**GATE: PASS on A1/A2/A3 — projected 1.46-1.57x vs the 1.25x threshold.** The one remaining unknown
is alpha (A5), which is now the sole determinant.

## A5 — acceptance rate: PASS (alpha_1 = 0.91)

Captured 3 real chat-templated prompts x 250 greedy tokens on the real 40L model (coherent output),
recording both the pre- and post-final-norm backbone hidden per step, then scored host-side against
the real `mtp.*` weights.

**The fc concat order is resolved unambiguously — `token_first`:**

              hidden/token_first:  alpha_1 = 0.9115  (175/192)  per-prompt 0.766 / 1.000 / 0.969
             hidden/hidden_first:  alpha_1 = 0.0000  (0/192)
          hidden_pre/token_first:  alpha_1 = 0.8854  (170/192)  per-prompt 0.719 / 1.000 / 0.938
         hidden_pre/hidden_first:  alpha_1 = 0.0000  (0/192)

`fc([norm_emb(embed(t_{i+1})), norm_hidden(h_i)])`. `hidden_first` scores EXACTLY zero -- a perfect
discriminator, so there is no residual ambiguity (the two prior in-tree attempts disagreed on this).
Matches the DeepSeek-V3 MTP convention.

**The POST-final-norm hidden wins** (0.9115 vs 0.8854), i.e. `TtModel._decode_hidden()`'s existing
return value is the correct input. No model change needed to source the hidden -- convenient, and the
opposite of what `pre_fc_norm_hidden`'s existence suggested.

**Per-depth chain acceptance** (autoregressive draft, 96-token true prefix, 30 sample positions/prompt):

    prompt 0: cumulative 0.800 0.500 0.333 0.267   conditional 0.800 0.625 0.667 0.800
    prompt 1: cumulative 0.867 0.733 0.400 0.300   conditional 0.867 0.846 0.545 0.750
    prompt 2: cumulative 0.867 0.800 0.600 0.167   conditional 0.867 0.923 0.750 0.278
    MEAN     cumulative 0.844 0.678 0.444 0.244

Acceptance does decay with depth as predicted (the head predicts one token ahead by construction), but
much less steeply than my pre-measurement guess of 0.85/0.55/0.40/0.30.

---

# FINAL VERDICT: GO

Measured cost model x measured acceptance:

    gamma   K  round ms  E[tok]  ms/tok  tok/s  speedup
        1   2     36.21   1.844   19.63   50.9    1.48x
        2   3     42.92   2.522   17.02   58.8    1.70x
        3   4     49.63   2.967   16.73   59.8    1.73x
        4   5     56.34   3.211   17.55   57.0    1.65x

**gamma=2 or 3 -> ~1.7x, 34.5 -> ~59 tok/s.** Gate was >= 1.25x (kill < 1.15x): **PASS**, with ~40%
margin. gamma=2 and gamma=3 are within noise of each other, so **pick gamma=2**: smaller K, less
rollback state, fewer draft steps, and it keeps the MoE verify tile at 24 active slots.

Why this is so much better than the July NO-GO (which measured 0.7-0.9x):
1. Wrong denominator -- verify(K) must be compared to K decode steps, not one.
2. The union-MoE probe used `nnz=None`, paying the 256-slot sparsity scan; indices-gather does 8K.
3. Most of BOTH the MoE op (0.216 of 0.326 ms) and the GDN layer (0.258 of 0.319 ms) is FLAT glue,
   not per-token work. That is where the amortization comes from, and both prior analyses missed it.
4. alpha is genuinely high (0.91 at depth 1) -- the shipped MTP head is good.

## Residual risks (none verdict-changing)
- Draft-step cost is an estimate (2.03 ms). If it were 3.0 ms, gamma=2 drops 1.70x -> 1.62x.
- verify is charged ONE 2.21 ms dispatch gap. At 2x that, gamma=2 drops to ~1.62x.
- alpha measured greedy; the shipping server samples (temp 0.6 / presence 1.5), which lowers
  acceptance. Speculative sampling stays exact, but E[tokens] will be lower than the greedy figure.
- 40L eager capture only; alpha on longer contexts untested.

---

# Phase B — implementation (2026-08-20)

Built and PCC-gated; 39 checkpoint-free tests pass (22 pre-existing + 17 new).

| item | where | validation |
|---|---|---|
| loader + config | `tt/load_checkpoints.py` (`has_mtp`, `mtp_*`), `tt/model_config.py` | all 19 `mtp.*` tensors load through prefix-refactored builders |
| draft head | `tt/mtp.py::TtMtpHead`, `reference/…::Qwen35MoeMTP` | prefill 0.9992, decode 0.9996, pre-norm 0.9996 |
| attention verify | `tt/attention.py::forward_verify` | 0.9996 vs reference, 0.9997 vs K sequential; causality proven |
| MoE verify | `tt/moe.py::forward_verify` | K=1..8: 0.9994 vs reference, 0.9999999 vs K decodes, no row leakage |
| GDN verify + rollback | `tt/gated_delta.py::_forward_verify_seq` / `commit_verify` | 0.9997 vs K sequential; state untouched by verify; rollback rec 0.99998 / conv 1.0 |
| orchestration | `tt/model.py::verify_forward` / `commit_verify` / `mtp_draft` / `spec_decode_step` | verify logits vs sequential decode PCC 0.9978-0.9996, argmax agrees at every K |

## Measured end to end, 40 layers (`tests/bench_mtp.py`)

    gamma  tok/round (mean)   projected traced
        1              1.83             ~1.46x
        2              2.55             **1.72x**
        3              2.90             ~1.70x

A5 predicted 1.844 / 2.522 / 2.967 from a host-side reference; hardware measured 1.83 / 2.55 / 2.90.
**Recommended gamma=2.** The eager tok/s in the harness (3.5-6.0) is dispatch-bound and is not the
shipping number — trace capture (B7) is what realises the ~58 tok/s.

## Not bit-parity with non-MTP decode, by construction

Verify is a different numerical path (K-row projections, chained ttl launches, indexed-gather MoE).
Measured teacher-forced against sequential decode: logit PCC 0.981-0.999, max|delta| 0.34-1.75, and
**no compounding drift** (stationary over 12 rounds, so the rollback is sound). Since that error scale
is comparable to typical top-2 logit gaps, greedy MTP tracks greedy baseline until the first near-tie.
Same regime as the already-shipped sparse-vs-dense MoE and chunked-vs-single prefill.

## Two traps worth knowing

1. **ttl indexes in TILE units, not rows.** `ct = 1` is one 32-row tile and the recurrence's
   `k^T @ delta` contracts over all 32 rows, so each verify step's operand must be a tile whose row 0
   is the token and whose other 31 rows are ZERO (`_to_tilerows`). A row-slice of a `[K, feat]` tensor
   carries neighbouring tokens in its pad: every input compared bit-identical while the output was
   badly wrong (row-0 PCC 0.58). K==1 worked by luck — an M=1 matmul zeroes its own pad.
2. **Acceptance is 0 at reduced layer counts, and that is correct.** The head was trained against the
   40-layer backbone's hiddens. Correctness gates run fine at `QWEN36_LAYERS=4`; throughput must use 40.

## Is a deeper draft (gamma 4-7) worth it? — MEASURED NO

Acceptance measured to depth 7 (host-side chain, 3 prompts x 30 sample positions,
`probe_mtp_acceptance.py score --gmax 7`):

    MEAN cumulative accept P[depth>=j] = 0.844  0.678  0.444  0.244  0.111  0.067  0.011

The economics are a fixed marginal cost against a geometrically decaying marginal benefit. Each extra
gamma costs **4.68 ms verify** (40x0.0818 MoE experts + 30x0.0465 GDN kernel + 10x0.0016 attn) plus one
**2.03 ms draft** = **6.71 ms**, and must buy ~0.39 extra tokens to pay for itself:

     gamma  round ms  E[tok]  ms/tok   tok/s  speedup   d tok  needed  pays?
         1     36.21   1.844   19.63    50.9    1.48x
         2     42.92   2.522   17.02    58.8    1.70x   0.678   0.394    yes
         3     49.64   2.967   16.73    59.8    1.73x   0.444   0.401    yes
         4     56.35   3.211   17.55    57.0    1.65x   0.244   0.383     NO
         5     63.06   3.322   18.98    52.7    1.53x   0.111   0.354     NO
         6     69.77   3.389   20.59    48.6    1.41x   0.067   0.326     NO
         7     76.49   3.400   22.50    44.5    1.29x   0.011   0.298     NO

**Optimum is gamma=2-3** (1.70x / 1.73x — within noise of each other, and the HW run put gamma=2
marginally ahead at 2.55 vs 2.90 tokens/round). gamma>=4 declines monotonically; gamma=5 (1.53x) and
gamma=6 (1.41x) are both WORSE than gamma=2. **Ship gamma=2**: tied on throughput, smaller K, one
fewer draft step, less rollback state.

Why deeper cannot work here, and what would change it:
- `mtp_num_hidden_layers: 1` — the head predicts exactly ONE token ahead, so depth>=2 runs it
  autoregressively on its own output (off-distribution) and acceptance decays ~geometrically.
- Making verify cheaper does not rescue it: even HALVING the MoE expert slope only drops break-even to
  ~0.30 tokens, still above the measured 0.244 at depth 4.
- The real alternative is breadth, not depth (tree/multi-candidate drafting), and that is structurally
  blocked on this model: each tree branch needs its own GDN recurrent-state chain, and the recurrence
  cannot branch. Same structural reason MTP was hard here to begin with.

---

# Phase C — speculative SAMPLING + serving integration (2026-08-20)

Phase B left two gaps: acceptance was exact-match only (so MTP could not serve the server's default
sampled requests), and nothing outside `bench_mtp.py` ever called it. Both are closed. Full measured
detail, including four wrong predictions and a device-level trace trap, is in **EXPERIMENTS.md**; the
summary:

| what | result |
|---|---|
| **speculative sampling** (`tt/mtp_sampling.py`) | exact rejection sampling with a point-mass (argmax) draft, so accept-prob is just `p(x̂)` and the residual is `p` minus `x̂`. Distribution-exactness χ²-gated host-side (`tests/test_mtp_sampling_rule.py`, 18 cases). |
| greedy speculative decode, 40L, gamma=2 | **52.1 ms/round -> 43.5 tok/s = 1.25x** (verify 52.5 -> 44.0 ms, full-accept commit 6.18 -> ~1 ms) |
| sampled, exact, default thinking-mode config | **1.11-1.19x and distribution-EXACT** (was 1.00x; the round cost, not the acceptance rule, was the binding constraint). Round 55.0 ms |
| sampled, `QWEN36_MTP_ACCEPT=relaxed` | **1.22x** (150-round paired sweep), NOT distribution-preserving: mean TV 0.088 |
| sampled, `QWEN36_MTP_ACCEPT=lenient` | a tunable dial between the two: **1.18x** at mean TV 0.052 (lenience 0.5), and it never force-emits a token the target rated <10% |
| sampling the DRAFT on device instead of argmax | **not worth it** (+0.02 acceptance host-measured for an extra per-step topk) |
| serving | `QWEN36_MTP=greedy_only` (recommended) / `1`, wired into `demo/server.py` + `demo/demo.py`, with a live break-even guard that disengages when speculation is losing |

## Why the host acceptance model over-predicted sampled acceptance

The A5-style host probe (extended here with the sampling rules and a presence-penalty dimension) said
exact sampled acceptance would cost only ~0.03: at T=0.6/top_p=0.95 the truncated target is nearly a
point mass (mean |support| **1.4**, H **0.15**, E[p_max] **0.936**). Device measured a **0.20** drop
(0.85 -> 0.65 conditional). Two causes, both measured:

1. the **presence penalty** flattens exactly the tokens the model just used (device: 2.50 -> 2.30
   tok/round going from presence 0 -> 1.5);
2. **off-policy bias** — the probe scores the captured GREEDY trajectory, while sampled decode visits
   flatter states where a one-token-ahead draft head is less accurate. This is structural: no amount of
   extra host sampling fixes it, only an on-policy capture would.

**So the probe is the right tool for exploring the (T, top_k, top_p, presence) space cheaply and the
wrong tool for predicting on-policy throughput.** Predict cost from cost measurements; never predict
acceptance off-policy.
