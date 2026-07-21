# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Ragged long-prefill regression: prefill_long must accept a prompt length that is NOT a multiple
of the chunk size (the final partial block is right-padded for attention and masked for gated-delta).

Specifically guards the chunked-SDPA alignment rule `chunk_start_idx % q_chunk_size == 0`: the ragged
block must use q_chunk_size=128 (not its padded width), else an offset like P=512 with padded width
384 fatals. T=812 (chunk=512 -> P=512, r=300, padded width 384, 512 % 384 != 0) reproduces it.

Correctness is gated on PCC vs single-shot, NOT exact argmax: chunked prefill is a ~0.99-PCC
approximation of full attention (different kernels), so the greedy top-1 can differ when logits are
close — this is true for aligned chunked prefill too, not specific to the ragged tail.

Uses real weights (QWEN36_LAYERS, default 4).
"""
import os

import torch

from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@torch.no_grad()
def test_ragged_prefill_long(mesh_device):
    n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_seq_len=1024)
    loader = CheckpointLoader(ckpt)
    model = TtModel(mesh_device, args, loader, num_layers=n_layers)

    torch.manual_seed(0)
    for T in (600, 812):  # 812: P=512, r=300, padded width 384 -> exercises the SDPA alignment fix
        prompt = torch.randint(0, args.vocab_size, (1, T))
        ragged = model.prefill_long(prompt, chunk=512)  # must not raise (ragged final block)
        single = model._prefill_single(prompt)
        pcc = _pcc(ragged[0, -1], single[0, -1])
        assert pcc > 0.97, f"T={T}: ragged-chunked vs single-shot PCC {pcc:.4f} too low"
        assert model.pos == T, f"T={T}: self.pos {model.pos} != real prompt length {T}"
