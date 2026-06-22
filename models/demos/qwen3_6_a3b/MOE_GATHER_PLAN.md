# MoE 8-of-256 Expert Gather — Postmortem (DONE)

Was the implementation plan for the decode MoE expert-gather lever; now shipped. The original
step-by-step plan is preserved in git history (commit that added the `indices` mode). This file is the
result + the lesson + the remaining follow-ups.

## Result

**40-layer decode 74 → 43 ms/token (13.4 → 23.2 tok/s/user, +73%), greedy-identical.** Two
independent, additive levers (measured 2×2 at 4 layers, `execute_trace` ms/token):

| | `in0_block_w=1` | `in0_block_w=8` |
|---|---|---|
| **scan** (default, pre-work) | 8.61 | 6.70 |
| **gather** (this work) | 7.37 | **5.46** |

- **gather** ≈ −14–18% (additive at any block size), **`in0_block_w`** ≈ −22% (additive on either
  path). Together −37% at 4L / 74→43 ms at 40L. Both kept (they attack different costs).
- Env: `QWEN36_MOE_GATHER` (default on), `QWEN36_SPARSE_IN0BW` (default 8; 16 marginally faster).

## The lesson — the original premise was wrong

This doc originally claimed the MoE cost was a 256-slot multicast-semaphore *scan* (~0.94 ms/layer), so
iterating 8 experts would collapse MoE ~46→~10 ms. **Measured, that was false:**
- The scan was only ~0.31 ms/layer. The shipped `nnz=top_k` path already did real multicast/compute
  work for just the 8 active experts; the other 248 slots were a cheap branch+skip. The gather removes
  that branch → the modest ~14–18%.
- The **real** decode-MoE bottleneck is **per-K-block multicast-handshake overhead** (gate_up's 64
  K-tiles × 8 experts at `in0_block_w=1`), NOT weight bandwidth. The Tracy/visualizer ">100% DRAM%" on
  the sparse_matmul is a **byte-estimate artifact** (it assumes all 256 experts are read when 8 are) —
  not evidence of bandwidth saturation. Proof: raising `in0_block_w` (which changes iteration count,
  not bytes moved) cut time 22%, which is impossible for a truly bandwidth-bound op.
- Takeaway: profile before believing a bottleneck label, and check "DRAM%" on sparse ops against the
  actual bytes moved, not the nominal full-expert estimate.

## What shipped (the reusable, generalizable piece)

A new **opt-in `indices` (gather) mode on `ttnn.sparse_matmul`** — benefits *any* MoE on tt-metal:
- Pass an on-device `indices` tensor (uint16, ROW_MAJOR, `[…, num_active]`, expert ids) → the kernels
  iterate only the `num_active` selected experts (`bB = indices[i]`) and emit a **compact**
  `[…, num_active, M, N]` output. Indices stay on device → decode trace preserved.
- **Presence-gated**: absent `indices` ⇒ byte-for-byte the legacy sparsity-scan path. All existing
  callers (gpt_oss, gemma4, unit tests, sweeps, nightly) are unaffected — verified.
- Files: `ttnn/cpp/.../matmul/{matmul.*, matmul_nanobind.cpp}`, the sparse device op +
  `factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`, and the in0/in1 reader kernels
  (compute + in0 receiver unchanged). Reusable helper: `models/common/moe_gather.py`. Op-level probe:
  `tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py`.

## Follow-ups (not done)

- **gpt_oss adoption** (`models/demos/gpt_oss/tt/experts/decode.py`): surface `topi`/`topv` from its
  router; pass `indices` to all 3 matmuls; replace the permute→reshape[E]→mul→sum-over-E combine with
  `moe_gather.gather_combine`. Apply the EP `moe_routing_remap` to the ids in `indices`. Also removes
  its `nnz=None` scan + the static-nnz deadlock risk.
- **gemma4 adoption** (`models/demos/gemma4/tt/experts/decode.py`): cleanest adopter (already
  `nnz=top_k`); add `indices`, switch gate/up/down to compact, swap the sum-over-E combine for
  `gather_combine` (GeGLU stays in `apply_geglu`).
- **`in0_block_w` for other MoEs**: the same per-K-block-overhead win likely applies — check each
  model's sparse-matmul program config (it's a free config change, not C++).
- Optional further trim: drop the always-true validity multicast in indexed mode once correctness is
  long-proven (gate compute's mailbox read under `#ifndef USE_INDICES`).
