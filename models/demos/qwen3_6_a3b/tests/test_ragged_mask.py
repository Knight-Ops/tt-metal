# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC gate for the ragged-block MASK path against the SLICE path it replaces.

A prompt whose length is not a multiple of the prefill chunk ends in a ragged block: `valid_len` real
tokens right-padded out to a 128-multiple. The old path sliced the block down to `valid_len` before
the gated-delta recurrence; the new one keeps the padded width and zeroes the pad rows' `beta`/`g` so
they are no-ops in the recurrence (`QWEN36_GDN_RAGGED_MASK`, default on).

WHY IT CHANGED, since it is not an accuracy win: slicing makes every op downstream run at
T = valid_len, a different value every turn, so each turn compiles ~90 new device programs and the
program cache grows without bound until the device hangs (measured at 40 layers, ~700-850 entries,
with memory nowhere near full). Masking pins the width to one of four padded values. See
IMPLEMENTATION_GUIDE.md.

So the thing to gate is that the two agree: same last-token logits, and -- what actually matters for
the NEXT block -- the same carried gated-delta state. A masked pad row must contribute nothing to the
recurrent state, and the conv's carried window must still come from the last REAL rows, not the
padding (that is `_conv_silu(state_at=...)`).

Uses real weights (QWEN36_LAYERS, default 4).
"""
import os

import torch

from models.demos.qwen3_6_a3b.tt import gated_delta as gd
from models.demos.qwen3_6_a3b.tt.common import from_tt
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _state(model):
    """The gated-delta state a following block would continue from, per linear layer."""
    out = []
    for c in model.caches:
        if isinstance(c, dict):
            out.append((from_tt(c["recurrent_state"], model.mesh_device), from_tt(c["conv_state"], model.mesh_device)))
    return out


def _run(mesh, prompt, tail, chunk=512):
    """Prefill `prompt`, then ingest `tail` as a ragged incremental block. Returns (logits, state)."""
    n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=2048)
    model = TtModel(mesh, args, CheckpointLoader(ckpt), num_layers=n_layers)
    model._prefill_single(prompt)
    logits = model.forward_incremental(tail, ragged=True, q_chunk=128)
    return logits[0, -1].clone(), _state(model), model.pos


@torch.no_grad()
def test_ragged_mask_matches_slice(mesh_device):
    torch.manual_seed(0)
    args_vocab = 151936  # any valid id range; the tokenizer is not needed here
    prompt = torch.randint(0, args_vocab, (1, 512))
    tail = torch.randint(0, args_vocab, (1, 254))  # -> padded width 256, valid_len 254

    was = gd._RAGGED_MASK
    try:
        gd._RAGGED_MASK = False
        slice_logits, slice_state, slice_pos = _run(mesh_device, prompt, tail)
        gd._RAGGED_MASK = True
        mask_logits, mask_state, mask_pos = _run(mesh_device, prompt, tail)
    finally:
        gd._RAGGED_MASK = was

    assert slice_pos == mask_pos == 766, (slice_pos, mask_pos)

    pcc = _pcc(mask_logits, slice_logits)
    print(f"\n[ragged-mask] last-token logits  PCC {pcc:.6f}")
    assert pcc > 0.99, f"masked ragged block vs sliced: logits PCC {pcc:.4f}"

    # The carried state is what the NEXT block continues from, so it is the stricter gate: a pad row
    # leaking into the recurrence, or a conv window taken past the real tokens, shows up here first.
    worst_rec = worst_conv = 1.0
    for i, ((m_rec, m_conv), (s_rec, s_conv)) in enumerate(zip(mask_state, slice_state)):
        worst_rec = min(worst_rec, _pcc(m_rec, s_rec))
        worst_conv = min(worst_conv, _pcc(m_conv, s_conv))
    print(f"[ragged-mask] recurrent_state    PCC {worst_rec:.6f} (worst of {len(mask_state)} layers)")
    print(f"[ragged-mask] conv_state         PCC {worst_conv:.6f} (worst of {len(mask_state)} layers)")
    assert worst_rec > 0.99, f"recurrent state diverges: PCC {worst_rec:.4f}"
    # conv_state is an exact copy of real input rows on both paths, so it must match to the bit.
    assert worst_conv > 0.9999, f"conv state diverges: PCC {worst_conv:.4f} -- pad rows leaked in?"


@torch.no_grad()
def test_ragged_mask_matches_single_shot(mesh_device):
    """Correctness across remainders, against the real reference: a single-shot prefill of the whole
    prompt. Same gate as test_ragged_prefill (PCC vs single-shot, not exact argmax) -- incremental
    prefill is a ~0.99-PCC approximation of full attention whichever way the padding is handled.

    Not compared against the SLICE path here, deliberately. Slicing a block down to one real row
    makes T == 1, and T == 1 is the DECODE signature throughout this module -- the add-chain conv,
    the decode program configs, the fused single-step kernel. So at valid_len=1 the old path runs a
    prefill block through the decode kernels reading `conv_rows`, which is a different computation
    (and a questionable one mid-prefill). Measured: mask-vs-slice PCC 0.88 at valid_len=1 for that
    reason alone, while both are fine against single-shot. The mask path never narrows the block, so
    it cannot trip that.
    """
    torch.manual_seed(1)
    vocab = 151936
    n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_seq_len=2048)
    model = TtModel(mesh_device, args, CheckpointLoader(ckpt), num_layers=n_layers)

    prompt = torch.randint(0, vocab, (1, 512))
    for valid in (1, 127, 129, 255):
        tail = torch.randint(0, vocab, (1, valid))
        whole = torch.cat([prompt, tail], dim=1)
        model._prefill_single(prompt)
        inc = model.forward_incremental(tail, ragged=True, q_chunk=128)
        assert model.pos == 512 + valid, (valid, model.pos)
        ref = model._prefill_single(whole)  # reference LAST: it resets the state we just built
        pcc = _pcc(inc[0, -1], ref[0, -1])
        print(f"\n[ragged-mask] valid_len={valid:4d}: vs single-shot PCC {pcc:.6f}")
        assert pcc > 0.97, f"valid_len={valid}: masked ragged block vs single-shot PCC {pcc:.4f}"
