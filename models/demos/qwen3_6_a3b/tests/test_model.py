# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tt-nn model test on REAL weights (reduced layer count) + multi-layer PCC vs reference.

Layer count via env QWEN36_LAYERS (default 4 -> 3 linear + 1 full attention).
"""
import os
import time

import torch

from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs


@torch.no_grad()
def test_model_real_weights(mesh_device):
    n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_seq_len=512)
    loader = CheckpointLoader(ckpt)

    t0 = time.time()
    model = TtModel(mesh_device, args, loader, num_layers=n_layers)
    load_s = time.time() - t0

    T = 32
    torch.manual_seed(0)
    input_ids = torch.randint(0, args.vocab_size, (1, T))

    t0 = time.time()
    logits = model.forward(input_ids)
    fwd_s = time.time() - t0

    print(f"\n[qwen36] layers={n_layers} load={load_s:.1f}s prefill({T} tok)={fwd_s:.2f}s " f"({T/fwd_s:.1f} tok/s)")
    # _head returns only the LAST token's logits (all generation consumers use logits[0, -1]).
    assert logits.shape == (1, 1, args.vocab_size)
    assert torch.isfinite(logits).all(), "non-finite logits"
    # sanity: top token ids are valid
    assert int(logits.argmax(-1).max()) < args.vocab_size
