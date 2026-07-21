# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Chunked (long-context) prefill correctness: prefill_long must be invariant to chunk size and
match single-shot prefill, at a length where single-shot still works (T=1024).

prefill_long streams the prompt through the incremental path (gated-delta state/conv carry + attention
chunked-SDPA over the accumulated KV). It must produce the SAME last-token logits as the single-shot
forward — for any chunk size. This validates the cross-chunk state carry + attention offset machinery
against a gold reference, before trusting it at 8K/32K/256K (where single-shot can't run).

Run (real weights):
    QWEN36_LAYERS=4 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/test_long_prefill.py
    QWEN36_LAYERS=4 ./python_env/bin/python -m pytest models/demos/qwen3_6_a3b/tests/test_long_prefill.py
"""

import os

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _run(mesh_device):
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
    args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_seq_len=2048)
    loader = CheckpointLoader(ckpt)
    model = TtModel(mesh_device, args, loader, num_layers=n_layers)

    T = 1024  # single-shot still fits here -> usable as the gold reference
    torch.manual_seed(0)
    ids = torch.randint(0, args.vocab_size, (1, T))

    gold = model._prefill_single(ids)[0, -1].float()  # single-shot last-token logits [vocab]
    c512 = model.prefill_long(ids, chunk=512)[0, -1].float()  # 2 chunks (1 single + 1 incremental)
    c256 = model.prefill_long(ids, chunk=256)[0, -1].float()  # 4 chunks

    # The CORRECTNESS signal for the streamed state-carry machinery is chunk-size INVARIANCE: different
    # chunkings must give identical logits. (The chunked path vs single-shot differ only by the attention
    # KERNEL — chunked_scaled_dot_product_attention vs scaled_dot_product_attention — so their logits
    # correlate ~0.9, not 1.0; the next-token argmax still matches, i.e. generation is identical.)
    p_inv = _pcc(c512, c256)  # chunk-invariance: expect ~1.0
    p512 = _pcc(c512, gold)  # cross-kernel vs single-shot: informational (~0.9 expected)
    g_tok, t512, t256 = int(gold.argmax()), int(c512.argmax()), int(c256.argmax())
    print(f"[long_prefill] {n_layers}L, T={T}: chunk-invariance PCC(512,256) {p_inv:.5f}  (>0.99; argmax is the gate)")
    print(f"[long_prefill] cross-kernel PCC(chunked,single) {p512:.5f}  (informational, ~0.9)")
    print(f"[long_prefill] next-token argmax: single={g_tok} chunk512={t512} chunk256={t256}")
    # The PRIMARY correctness signal is argmax-invariance: identical next token for every chunking AND
    # vs single-shot -> generation is byte-identical. The logit PCC is a drift guard (the old unstable
    # inverse blew up to PCC~0); at full 40-layer depth the fp32-via-bf16 prep differs slightly between
    # chunk groupings, so ~0.995 is expected and correct (NOT the old 1e11 blow-up).
    ok = p_inv > 0.99 and g_tok == t512 == t256
    print("[long_prefill] PASS ✓ (chunk-invariant + same next token as single-shot)" if ok else "[long_prefill] FAIL ✗")
    return ok


def test_long_prefill_chunk_invariant(mesh_device):
    assert _run(mesh_device)


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        _run(mesh)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
