# MoE 8-of-256 Expert Gather — Implementation Plan (Decode Tier 1)

The single biggest remaining decode lever for Qwen3.6-35B-A3B. Companion to `DECODE.md` (§4A) which
ranks it; this is the concrete implementation plan. **Status: not started.** Effort: large (multi-day,
core-ttnn C++). Expected win: **MoE ~46 → ~10 ms/token → decode ~74 → ~40 ms → ~25 tok/s/user.**

---

## 1. The problem (confirmed in the kernels)

Decode MoE is ~62% of the step (`bench_decode.py`: 1.14 ms/layer × 40, of which `sparse_decode` is
0.94 ms). It is **scan-overhead-bound, not bandwidth-bound** — the Tracy profiler shows a physically-
impossible ">100% DRAM" on the sparse_matmul, the signature that it skips most of its nominal reads yet
still burns ~28× over its ~16 µs data-movement floor.

The cause is in all three sparse-matmul kernels
(`ttnn/cpp/ttnn/operations/matmul/device/kernels/`):
- `dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp`
- `dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp`
- `compute/bmm_large_block_zm_fused_bias_activation.cpp`

Each loops over **all 256 expert slots** and skips zeros:
```cpp
for (uint32_t bB = 0; bB < batchB_lim; ++bB) {            // batchB_lim = 256
    if (sparsity[bB] == 0) { /* advance tile ids */ continue; }
    ... read expert bB / multicast in0 / compute ...
}
```
`nnz=top_k` (already landed) makes the *compute* expect 8 results, but the kernels still **iterate 256
in lockstep**, paying the per-slot in0 **multicast-semaphore** handshake and tile-id bookkeeping 256×.
That lockstep is the ~0.94 ms/layer.

---

## 2. The change: an on-device compacted index list

The active expert ids `topi[1,8]` already exist on device (`tt/moe.py`, `_shared_and_router`) and are
currently discarded after `scatter`. Pass them as a new optional `indices` operand so the kernels
iterate the active set directly:
```cpp
for (uint32_t i = 0; i < num_active; ++i) {               // num_active = 8
    uint32_t bB = indices[i];                              // jump straight to expert bB
    ... read expert bB / multicast / compute ...           // no scan, no skip-branch
}
```
Iterate 8, not 256 → ~32× fewer multicast handshakes. Output becomes **compact** `[1, num_active, M, N]`
(8 slots, not 256), which also simplifies the combine. The indices stay on device → the decode trace is
preserved (no host sync).

This is a new **indexed/gather mode** on `ttnn.sparse_matmul`, gated by the presence of the `indices`
operand; the existing 256-slot sparsity-scan path stays as the default/fallback.

---

## 3. Files touched & the change in each

**C++ device op**
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation_types.hpp`
  — add an `indices`-mode marker to `SparseMatmulParams` (or infer it from the optional input).
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.{hpp,cpp}`
  — accept `indices` as an **optional input tensor** (uint16/uint32, on device, length ≤ E); validate
  it; compute the **compact output shape** `[1, num_active, M, N]` when present.

**Program factory**
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`
  — plumb the `indices` buffer address as a runtime arg + a `USE_INDICES` compile define; set the
  kernels' batch-loop bound to `num_active`; add the `indices` `TensorAccessor`/CB; keep the
  `get_batch_from_reader` coordination consistent across the three kernels.

**Three kernels** (`ttnn/cpp/ttnn/operations/matmul/device/kernels/`) — under `#ifdef USE_INDICES`
replace the `bB<256 / skip-zero` loop with `i<num_active; bB=indices[i]`:
- `dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` — address in0/mcast by `bB`.
- `dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` — `in1_batch_tile_id = bB*KtNt`;
  write to compact output slot `i`.
- `compute/bmm_large_block_zm_fused_bias_activation.cpp` — same loop substitution so compute stays in
  lockstep with the readers (**the deadlock-sensitive part**).

**Python-facing binding**
- `ttnn/cpp/ttnn/operations/matmul/matmul.{hpp,cpp}` + the sparse pybind — add the `indices` kwarg.

**Model**
- `models/demos/qwen3_6_a3b/tt/moe.py` (`forward_sparse_decode`) — stop discarding `topi`; pass
  `indices=topi` (uint) to **both** sparse_matmuls (gate_up `is_input_a_sparse=False`, **and** the down
  proj `is_input_a_sparse=True` — both need the index treatment); rewrite the combine to
  `sum_i topv[i]·down[i]` over the compact 8, dropping the `scatter`→dense→`sum`-over-256 (which also
  trims the ~7 ms router/shared-expert tier).

**Test**
- `models/demos/qwen3_6_a3b/tests/test_moe_sparse.py` — extend to E=256; PCC the gather path vs the
  dense reference and vs the current sparse path.

---

## 4. Data flow (traceable, no host sync)

`_shared_and_router` → `ttnn.topk` → `topi[1,8]` (on device) → `ttnn.typecast` to uint → pass as
`indices` to both sparse_matmuls. `topv[1,8]` drives the compact combine. Nothing returns to host; the
self-contained decode trace is unchanged.

---

## 5. De-risking milestones (do in order — do not wire the model until the kernel is proven)

1. **Microbench probe FIRST** (a few hours; the go/no-go gate). A standalone bench op with a
   **hard-coded 8-index tensor** — no model, no Python plumbing. Confirm it (a) compiles, (b) PCC-matches
   the 256-scan output, (c) drops ms ~10–30×, (d) the down-proj `is_input_a_sparse` path also works,
   (e) bf4 weight addressing by arbitrary `bB` is correct. This resolves the kernel-coordination and
   bf4-addressing unknowns before any plumbing.
2. Wire `topi` through `moe.py`; PCC vs reference (`test_moe_sparse`, E=256).
3. `bench_decode.py` — measure MoE 46 → ? ms (real weights, definitive).
4. Full 40-layer `demo.py --trace` — greedy-identical + the new tok/s.

---

## 6. Risks / unknowns

- **Kernel coordination is the hard part:** the three kernels must iterate the *same* count or the
  multicast semaphores deadlock. A killed mid-compile hangs the device → `tt-smi -r` (HANDOFF §9).
- **bf4 weight addressing by arbitrary `bB`:** readers compute `in1_batch_tile_id = bB·KtNt` in tiles —
  bf4 page addressing should be unchanged, but verify in the probe (milestone 1e).
- **Both sparse modes:** gate_up (input-B-sparse) and down (input-A-sparse) have different layouts; the
  index path must cover both.
- **Why not ttl:** a ttl gather was investigated and **proven infeasible** — ttl cannot turn a
  tensor-read index into a DRAM address (`raw_element_read` yields a compute-domain scalar with no
  bridge to the datamovement/NOC addressing domain; only `ttl.node()` gives runtime addressing). So
  this lever is necessarily C++.

**Fallback if the C++ stalls:** bf16-promote a decode-only copy of the experts + `ttnn.gather` the 8
tiles + dense batched matmul (Python, no C++). Costs ~+6 GB device memory and higher per-token
bandwidth (33 vs 12.6 MB); measure before committing — it may not beat the current 46 ms.

---

## 7. What the codebase gains

- **A reusable indexed/gather mode on `ttnn.sparse_matmul`** — benefits *any* MoE on tt-metal, not just
  qwen36 (upstreamable).
- **MoE decode ~46 → ~10 ms → ~25 tok/s/user** — the single biggest single-user lever.
- **Router simplification** — `topi` used directly removes the `scatter`→dense→`sum`-over-256 round-trip,
  trimming the ~7 ms router/shared-expert tier.
- **Cleaner combine** (compact 8 vs dense 256), indices stay on device → decode trace unchanged.
