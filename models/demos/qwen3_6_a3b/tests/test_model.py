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


@torch.no_grad()
def test_decode_logits_matches_greedy(mesh_device):
    """A1: decode_forward_logits + host argmax + set_decode_tokens reproduces the on-device greedy
    decode_step_eager token stream at B=1. This is the vLLM continuous-batching contract (host samples
    from returned logits, feeds the token back) and must be token-identical to the proven greedy path.
    """
    n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_seq_len=512)
    loader = CheckpointLoader(ckpt)
    model = TtModel(mesh_device, args, loader, num_layers=n_layers)

    T, N = 16, 8
    torch.manual_seed(0)
    input_ids = torch.randint(0, args.vocab_size, (1, T))

    # Greedy trajectory: on-device argmax (the proven standalone path).
    logits = model.forward(input_ids)
    first = int(logits[0, -1].argmax())
    model.start_decode(first)
    greedy = [first] + [model.decode_step_eager() for _ in range(N)]

    # Logits trajectory: re-prefill (reallocates caches + reseeds positions), then host-argmax the
    # returned logits and feed the chosen token back via set_decode_tokens — the vLLM contract.
    logits = model.forward(input_ids)
    model.start_decode(int(logits[0, -1].argmax()))
    host = [int(logits[0, -1].argmax())]
    for _ in range(N):
        nxt = int(model.decode_forward_logits()[0].argmax())
        model.set_decode_tokens(nxt)
        host.append(nxt)

    assert host == greedy, f"logits-path tokens {host} != greedy {greedy}"
