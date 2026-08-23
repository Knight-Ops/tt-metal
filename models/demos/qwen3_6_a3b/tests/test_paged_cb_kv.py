# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Paged KV under CONTINUOUS BATCHING must match the contiguous per-slot arrangement, token for token.

This is the gate for `QWEN36_PAGED_KV=1` on the multi-slot path -- the arrangement the vLLM adapter
turns on by default for B>1 (packaging/vllm_bundle/generator_vllm.py). Both arms run in ONE process
against ONE set of weights, switching `model.paged_kv` between them, because two 40-layer builds do not
fit a 32 GB P150:

  flat  : caches are [B, n_kv, max_seq, head_dim] -- every slot pays max_seq whether it uses it or not,
          and prefill_into_slot stages into a [1,...] scratch then row-copies with ttnn.fill_cache.
  paged : ONE pool of [num_blocks, n_kv, block, head_dim] shared by all slots AND all attention layers,
          with a [B, blocks_per_seq] device page table. prefill writes STRAIGHT into the target slot's
          pages (no scratch, no row copy) because the slot's page table already routes it there.

Greedy decode, so a matching token stream is a real equality claim and not a sampling coincidence.
Slots carry DIFFERENT prompt lengths, which is the case the two arrangements are most likely to
disagree on: the flat path seeks by absolute row, the paged path by page-table entry -- and the flat
path has no notion of an unmapped block at all, so it cannot catch an under-mapped page table.

Run it BOTH ways when touching paging: `--long` (chunked prefill) and with `QWEN36_KV_DTYPE=bf8`. The
under-mapping bug above was invisible at bf16 in some pool layouts (an unmapped read happened to land on
zeros) and only reliably fatal at bf8, so a single-dtype run is not a gate.

Also asserts the pool accounting -- release_slot() must hand every block back, or a long-running
server leaks one sequence's worth of blocks per completed request.

Run: QWEN36_LAYERS=4 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/test_paged_cb_kv.py
"""
import argparse
import os

import torch

import ttnn
from models.demos.qwen3_6_a3b.tests.bench_decode import build_model

STEPS = 12
# The DEFAULT is deliberately the set that discriminates, not the cheapest one. sdpa_decode rounds the
# k-range it reads up to a whole k_chunk_size (128), so a slot at position 128 reads 4 blocks of 64 while
# only 3 cover its cur_pos -- and mapping "up to cur_pos" left the 4th UNMAPPED. These four positions
# split cleanly: 64 and 192 round onto the mapped prefix and were always fine; 128 and 256 do not and
# emitted garbage. An earlier default of (128, 64) passed WITH the bug present, which is why it moved.
PROMPTS = (64, 128, 192, 256)
# --long: prompts that EXCEED the prefill chunk, so each slot is ingested by prefill_long ->
# forward_incremental. That is the path a real long-context request takes, and it is the one place the
# paged arrangement needs a PER-SLOT read table (_kv_read_table): chunked-SDPA treats row 0 of whatever
# table it gets as *the* sequence's mapping, so a shared [B, blocks] table would make every slot past 0
# read slot 0's pages. 1024 = 2 whole chunks; 640 = 1 chunk + a 128-token ragged block.
PROMPTS_LONG = (1024, 640)


def run_arm(model, prompts, steps, paged, pool_tokens=None):
    """Prefill each prompt into its own slot, then greedily decode `steps` batched steps.
    Returns (per-slot token lists, the pager or None)."""
    B = len(prompts)
    model.paged_kv = paged
    model.kv_pager = None  # a fresh pool/page table per arm (the table's address is trace-baked)
    if paged and pool_tokens is not None:
        os.environ["QWEN36_KV_POOL_TOKENS"] = str(pool_tokens)
    else:
        os.environ.pop("QWEN36_KV_POOL_TOKENS", None)
    model.batch_size = B
    model.alloc_batch_caches(B)

    firsts = []
    for s, p in enumerate(prompts):
        lg = model.prefill_into_slot(p, s)
        firsts.append(int(torch.as_tensor(lg).reshape(-1).argmax()))
    pos = list(model.slot_pos)  # per-slot absolute position of the token we are about to feed
    model.setup_batch_decode(firsts)

    out = [[t] for t in firsts]
    toks = list(firsts)
    for _ in range(steps):
        model.set_decode_state(toks, pos, active_slots=list(range(B)))
        nxt = [int(t) for t in model.decode_step_eager_batch()[:B]]
        for s in range(B):
            out[s].append(nxt[s])
        toks, pos = nxt, [p + 1 for p in pos]
    return out, model.kv_pager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true", help="prompts longer than the prefill chunk (chunked path)")
    ap.add_argument("--slots", type=int, default=0, help="N slots with staggered lengths (default: 2)")
    a = ap.parse_args()
    lens = PROMPTS_LONG if a.long else PROMPTS
    if a.slots:
        # Staggered, deliberately NOT uniform: equal lengths would hide an off-by-one in the per-slot
        # block mapping, since every slot would want the same number of blocks.
        base = 512 if a.long else 64
        lens = tuple(base * (i + 1) for i in range(a.slots))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
        n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
        max_seq = int(os.environ.get("QWEN36_MAX_SEQ", "4096" if a.long else "1024"))
        model, args = build_model(mesh, n_layers, ckpt, max_seq, seq=8)
        chunk = model._PREFILL_CHUNK
        assert not a.long or max(lens) > chunk, f"--long needs a prompt > chunk {chunk}"
        print(
            f"[paged-cb] prompt lens {list(lens)}, prefill chunk {chunk} "
            f"-> {'CHUNKED (forward_incremental)' if max(lens) > chunk else 'single-shot'}"
        )
        torch.manual_seed(0)
        prompts = [torch.randint(0, args.vocab_size, (1, T)) for T in lens]

        flat, _ = run_arm(model, prompts, STEPS, paged=False)
        print(f"[paged-cb] flat  {[t[:6] for t in flat]}")
        # Pool deliberately smaller than B*max_seq -- the budget follows concurrent demand.
        pool = max(512, 2 * sum(lens))
        paged, pager = run_arm(model, prompts, STEPS, paged=True, pool_tokens=pool)
        print(f"[paged-cb] paged {[t[:6] for t in paged]}")
        print(f"[paged-cb] {pager}")

        bad = [s for s in range(len(prompts)) if flat[s] != paged[s]]
        for s in bad:
            print(f"[paged-cb] slot {s} DIFFERS\n  flat  {flat[s]}\n  paged {paged[s]}")
        assert not bad, f"paged KV diverges from flat on slots {bad}"
        print(f"[paged-cb] OK: {len(prompts)} slots x {STEPS + 1} tokens identical (prompt lens {list(lens)})")

        # pool accounting: every block a slot took must come back
        used = pager.num_blocks - len(pager._free)
        for s in range(len(prompts)):
            model.release_slot(s)
        after = pager.num_blocks - len(pager._free)
        print(f"[paged-cb] blocks in use {used} -> {after} after releasing every slot")
        assert after == 0, f"release_slot leaked {after} blocks (a server would exhaust the pool)"
        print("[paged-cb] PASS")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
