# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Gates for priming the MTP head's KV cache over the prompt (``TtModel.prime_mtp``).

These are STRUCTURAL gates on row indices, not accuracy gates, because the thing that can plausibly be
wrong here is the fencepost. The head's KV row is offset +1 from its contract row -- ``_m_pos_*`` is
shared with verify, which needs true absolute backbone positions, so the draft chain stores the pair
for contract row ``pos-1+j`` at head position ``pos+j`` (see prime_mtp's docstring). That offset is
invisible while the cache is cold, since there is nothing for the rows to be relative to; it becomes
load-bearing the moment anything is primed. Get it wrong by one and nothing crashes, no PCC moves, and
the only symptom is silently worse acceptance -- which is exactly what priming was supposed to fix.

So: assert the primed rows are the rows we meant, and assert that primed rows and drafted rows form a
CONTIGUOUS prefix with no zero row wedged between them. A zero row is not inert -- softmax still
spends mass on it -- and one sitting immediately below the first query is the worst place for it.

Needs the real checkpoint (for the ``mtp.*`` head). Correctness here is layer-count-independent, so
the default QWEN36_LAYERS=4 is enough; acceptance is not measured.

    QWEN36_LAYERS=4 ./python_env/bin/python -m pytest -q .../test_mtp_prime.py
"""
import os

import pytest
import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs

CKPT = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
PROMPT_LEN = 24


def _model(mesh_device, max_seq=256):
    args = ModelArgs(mesh_device, ckpt_dir=CKPT, max_seq_len=int(os.environ.get("QWEN36_MAX_SEQ", str(max_seq))))
    loader = CheckpointLoader(CKPT)
    if not loader.has_mtp():
        pytest.skip("checkpoint has no mtp.* head")
    return TtModel(mesh_device, args, loader, num_layers=int(os.environ.get("QWEN36_LAYERS", "4"))), args


def _prompt(args, T=PROMPT_LEN):
    torch.manual_seed(0)
    # Random ids are fine: nothing here depends on the output distribution being peaked.
    return torch.randint(0, args.vocab_size, (1, T))


def _row_activity(model):
    """Per-row max|K| over the head's key cache -> [max_seq]. Zero means "never written"."""
    k = ttnn.to_torch(model.mtp_kv[0]).float()  # [1, n_kv, max_seq, head_dim]
    return k.abs().amax(dim=3).amax(dim=1).reshape(-1)


def _prefill_and_prime(model, prompt):
    model.build_mtp_head()
    model._keep_prefill_hidden = True
    logits = model.forward(prompt)
    first_id = int(logits[0, -1].argmax())
    return model.prime_mtp(prompt, first_id), first_id


def test_prime_writes_exactly_the_rows_it_claims(mesh_device):
    """T+1 rows: a filler at position 0 plus contract rows 0..T-1 at positions 1..T. Nothing beyond."""
    model, args = _model(mesh_device)
    prompt = _prompt(args)
    T = prompt.shape[1]
    n, _ = _prefill_and_prime(model, prompt)
    assert n == T + 1, f"primed {n} rows, expected T+1 = {T + 1}"

    act = _row_activity(model)
    assert (
        act[: T + 1] > 0
    ).all(), f"unwritten row(s) inside the primed range: {(act[: T + 1] == 0).nonzero().tolist()}"
    # The head's own first write lands at pos+1 = T+1 (round 1 is a K=1 backbone seed step that does
    # not run the head), so priming must stop just short of it -- not overlap, not fall short.
    assert act[T + 1] == 0, "priming wrote past its range, into the row the first draft step owns"


def test_primed_and_drafted_rows_are_contiguous(mesh_device):
    """The fencepost gate. After priming and a few speculative rounds the written rows must form one
    unbroken prefix: a gap would mean the +1 offset convention is wrong by one, which costs acceptance
    silently and would never show up as a failure anywhere else."""
    model, args = _model(mesh_device)
    prompt = _prompt(args)
    T = prompt.shape[1]
    n, first_id = _prefill_and_prime(model, prompt)
    assert n == T + 1

    model.setup_mtp_decode()
    model.start_decode(first_id)
    model._mtp_hidden = None
    for _ in range(4):  # round 1 is the eager seed (no head write); later rounds draft
        model.spec_decode_step()

    act = _row_activity(model)
    written = (act > 0).nonzero().reshape(-1).tolist()
    assert written, "nothing was written at all"
    hi = written[-1]
    assert hi > T + 1, f"the draft chain never wrote past the primed range (hi={hi}, primed 0..{T})"
    holes = [i for i in range(hi + 1) if act[i] == 0]
    assert not holes, f"zero row(s) inside the written prefix 0..{hi}: {holes}"


def test_prime_touches_nothing_but_the_head_cache(mesh_device):
    """It must not perturb backbone state: the emitted stream is the backbone's, and priming is only
    allowed to change how much of it a round gets to skip."""
    model, args = _model(mesh_device)
    prompt = _prompt(args)
    model.build_mtp_head()
    model._keep_prefill_hidden = True
    logits = model.forward(prompt)
    pos_before = model.pos
    conv_before = [
        ttnn.to_torch(c["conv_state"]).clone() for c in model.caches if isinstance(c, dict) and "conv_state" in c
    ]

    model.prime_mtp(prompt, int(logits[0, -1].argmax()))

    assert model.pos == pos_before, "prime_mtp advanced the backbone position"
    conv_after = [ttnn.to_torch(c["conv_state"]) for c in model.caches if isinstance(c, dict) and "conv_state" in c]
    for i, (b, a) in enumerate(zip(conv_before, conv_after)):
        assert torch.equal(b, a), f"prime_mtp disturbed layer {i}'s gated-delta conv state"


def test_prime_consumes_the_hidden_so_it_cannot_run_twice(mesh_device):
    """The stashed hidden belongs to ONE prefill. A second prime must no-op rather than re-prime off a
    stale hidden (which is what a request that took the chunked path would otherwise hit)."""
    model, args = _model(mesh_device)
    prompt = _prompt(args)
    n, first_id = _prefill_and_prime(model, prompt)
    assert n == prompt.shape[1] + 1
    assert model.prime_mtp(prompt, first_id) == 0, "primed twice off one prefill"


def test_prime_declines_a_chunked_prefill(mesh_device):
    """prefill_long materialises no full-sequence hidden (only the first chunk's), so priming must
    decline rather than prime off a truncated prompt while claiming the whole one."""
    model, args = _model(mesh_device, max_seq=1024)
    prompt = _prompt(args, T=768)
    model.build_mtp_head()
    model._keep_prefill_hidden = True
    logits = model.prefill_long(prompt, chunk=256)
    assert model.prime_mtp(prompt, int(logits[0, -1].argmax())) == 0
    assert (_row_activity(model) == 0).all(), "the head cache was written despite declining to prime"
