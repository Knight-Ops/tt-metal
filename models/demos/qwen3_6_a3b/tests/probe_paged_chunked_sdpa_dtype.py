# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does `chunked_scaled_dot_product_attention` read a MULTI-BLOCK **BFP8** paged cache correctly?

WHY THIS EXISTS. `tests/test_paged_cb_kv.py --long` diverges from the contiguous arrangement at
`QWEN36_KV_DTYPE=bf8` and matches exactly at bf16, with everything else held fixed. Three facts narrow
it to one op:
  * paged + bf8 + SINGLE-SHOT prefill  -> matches  (single-shot SDPA runs on live q/k/v; it never reads
    the cache, it only fills it)
  * paged + bf16 + CHUNKED prefill     -> matches
  * paged + bf8  + CHUNKED prefill     -> DIVERGES
The only path unique to the failing case is chunked SDPA *reading* a BFP8 paged cache.

WHY IT IS PLAUSIBLY THE OP AND NOT THE CALLER. On the flat path the same op is handed a
`[1, n_kv, max_seq, hd]` cache, i.e. ONE block covering every position, so it never strides between
blocks. The paged path is `[n_blocks, n_kv, 64, hd]` and strides across many. A BFP8 tile is 1088 B
(1024 data + 64 shared-exponent), not a power of two, so any place that computes a block's byte offset
with a bf16-shaped tile size is correct at bf16 and wrong at BFP8 -- which is exactly the observed
signature. This probe tests that claim with NO model in the loop.

METHOD. Build one set of K/V for positions [0, P+M), write it into (a) a flat cache and (b) a paged cache
holding the same logical content, then run the SAME chunked SDPA call the model makes
(`k_chunk_size=64`, `chunk_start_idx=P`) against both, at bf16 and at BFP8, and score each against a
torch reference. If paged/bf8 is the only arm that loses accuracy, the op is the culprit.

Run:  ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_paged_chunked_sdpa_dtype.py
"""
from __future__ import annotations

import math

import torch

import ttnn

NKV, NH, HD = 2, 16, 256  # this model's attention geometry
BLOCK, P, M = 64, 512, 128  # block size; context already cached; new tokens this chunk
MAXS = 1024


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def reference(q, k, v):
    """Causal attention of the M new rows (positions P..P+M-1) over K/V[0..P+M-1]."""
    g = NH // NKV
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    s = q @ k.transpose(-1, -2) / math.sqrt(HD)  # [1, NH, M, P+M]
    pos = torch.arange(P, P + M).view(-1, 1)
    s = s.masked_fill(torch.arange(P + M).view(1, -1) > pos, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        torch.manual_seed(0)
        k = torch.randn(1, NKV, P + M, HD) * 0.3
        v = torch.randn(1, NKV, P + M, HD) * 0.3
        q = torch.randn(1, NH, M, HD) * 0.3
        ref = reference(q, k, v)
        q_tt = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        pcfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=M,
            k_chunk_size=BLOCK,
        )
        nblk = MAXS // BLOCK

        print(f"  q_heads {NH}  kv {NKV}  head_dim {HD}  block {BLOCK}  P {P}  M {M}")
        print(f"{'arrangement':<22}{'dtype':>7}{'PCC vs torch':>15}{'max|delta|':>13}")
        results = {}
        for dn, dt in (("bf16", ttnn.bfloat16), ("bf8", ttnn.bfloat8_b)):
            for arrangement in ("flat", "paged"):
                if arrangement == "flat":
                    # ONE block covering every position; page table of zeros, as the model does
                    cache = [
                        ttnn.from_torch(torch.zeros(1, NKV, MAXS, HD), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh)
                        for _ in range(2)
                    ]
                    for c, src in zip(cache, (k, v)):
                        t = ttnn.from_torch(src, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh)
                        ttnn.fill_cache(c, t, 0)
                    pt = ttnn.from_torch(
                        torch.zeros(1, 32, dtype=torch.int32),
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh,
                    )
                else:
                    cache = [
                        ttnn.from_torch(
                            torch.zeros(nblk, NKV, BLOCK, HD), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh
                        )
                        for _ in range(2)
                    ]
                    pt_host = torch.arange(nblk, dtype=torch.int32).reshape(1, nblk)  # identity mapping
                    pt = ttnn.from_torch(pt_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
                    for c, src in zip(cache, (k, v)):
                        t = ttnn.from_torch(src, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh)
                        ttnn.experimental.paged_fill_cache(c, t, pt, batch_idx=0)
                out = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_tt,
                    cache[0],
                    cache[1],
                    pt,
                    chunk_start_idx=P,
                    scale=1.0 / math.sqrt(HD),
                    program_config=pcfg,
                )
                got = ttnn.to_torch(out).reshape(ref.shape)
                p, mx = pcc(ref, got), float((ref - got).abs().max())
                results[(dn, arrangement)] = p
                print(f"{arrangement:<22}{dn:>7}{p:>15.6f}{mx:>13.4f}")

        print()
        for dn in ("bf16", "bf8"):
            f, pg = results[(dn, "flat")], results[(dn, "paged")]
            verdict = "MATCH" if abs(f - pg) < 5e-4 else "*** PAGED IS WORSE ***"
            print(f"  {dn:>4}: flat {f:.6f} vs paged {pg:.6f}   delta {pg - f:+.6f}   {verdict}")
        print("\n  If bf16 matches and bf8 does not, chunked SDPA mis-strides a multi-block BFP8 cache")
        print("  and QWEN36_KV_DTYPE=bf8 must not be combined with paged CHUNKED prefill.")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
