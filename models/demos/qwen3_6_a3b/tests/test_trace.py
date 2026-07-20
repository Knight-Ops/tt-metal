# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Trace-correctness regression: traced decode must produce identical tokens to eager decode.

Uses real weights (QWEN36_LAYERS, default 4). Builds the model once, runs greedy decode eagerly,
then re-prefills (fresh caches) and runs greedy decode via the captured trace; asserts equal tokens.
"""
import os

import torch

from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs


@torch.no_grad()
def test_trace_matches_eager(mesh_device):
    n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_seq_len=128)
    loader = CheckpointLoader(ckpt)
    model = TtModel(mesh_device, args, loader, num_layers=n_layers)

    torch.manual_seed(0)
    prompt = torch.randint(0, args.vocab_size, (1, 8))
    n_gen = 6

    def greedy_eager():
        logits = model.forward(prompt)
        nxt = int(logits[0, -1].argmax())  # prefill's first token
        model.start_decode(nxt)
        out = [nxt]
        for _ in range(n_gen - 1):
            out.append(model.decode_step_eager())
        return out

    def greedy_traced():
        logits = model.forward(prompt)
        nxt = int(logits[0, -1].argmax())
        model.start_decode(nxt)
        out = [nxt]
        out.append(model.decode_step_eager())  # eager warmup (compiles)
        model.capture_decode_trace()
        for _ in range(n_gen - 2):
            out.append(model.decode_step_traced())
        return out

    eager = greedy_eager()
    traced = greedy_traced()
    assert eager == traced, f"trace diverged:\n eager ={eager}\n traced={traced}"
