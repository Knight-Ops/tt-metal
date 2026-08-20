# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Trace-safety + parity gate for the captured speculative VERIFY pass.

Verify is ~90% of a round's op count, so capturing it is where the eager->traced win comes from. Two
things must hold:
  1. capture must not fatal — every lazily-built constant (MoE ROWSEL/sparsity, ttl op compile,
     sharded configs) has to exist BEFORE `begin_trace_capture`, because building one during capture
     is a host write ("Writes are not supported during trace capture");
  2. replaying the trace must give the SAME per-row argmax as the eager path, and must keep giving it
     across replays (a stale/aliased buffer shows up as the second replay differing from the first).
"""
import os

import pytest
import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs

CKPT = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))


@torch.no_grad()
@pytest.mark.parametrize("gamma", [1, 2])
def test_mtp_verify_trace_matches_eager(mesh_device, gamma):
    args = ModelArgs(mesh_device, ckpt_dir=CKPT, max_seq_len=int(os.environ.get("QWEN36_MAX_SEQ", "256")))
    loader = CheckpointLoader(CKPT)
    if not loader.has_mtp():
        pytest.skip("checkpoint has no mtp.* head")
    model = TtModel(mesh_device, args, loader, num_layers=int(os.environ.get("QWEN36_LAYERS", "4")))
    torch.manual_seed(0)
    prompt = torch.randint(0, args.vocab_size, (1, 16))

    lg = model.forward(prompt)
    model.start_decode(int(lg[0, -1].argmax()))
    model.build_mtp_head(gamma=gamma)
    K = model.setup_mtp_decode()
    assert K == gamma + 1

    base_pos = model.pos
    toks = [int(ttnn.to_torch(model.t_tok).reshape(-1)[0])] + [7 + i for i in range(gamma)]
    positions = [base_pos + i for i in range(K)]

    # --- eager reference (does not advance any recurrent state) ---
    lgv, _ = model.verify_forward(toks, positions)
    eager = torch.as_tensor(ttnn.to_torch(lgv)).reshape(K, -1).argmax(dim=-1).tolist()

    # --- capture + replay. Capture MUST be primed with the round's real inputs: its warm/capture runs
    # write the attention KV cache, which is outside the snapshot set, so those writes have to be
    # idempotent with the first replay. ---
    model.capture_mtp_verify_trace(toks, positions)
    r1 = model.verify_step_traced(toks, positions)
    r2 = model.verify_step_traced(toks, positions)
    print(f"[trace] gamma={gamma}  eager {eager}")
    print(f"[trace] gamma={gamma}  replay1 {r1}  replay2 {r2}")

    assert r1 == r2, f"replays disagree ({r1} vs {r2}) — a buffer is stale or aliased"
    assert r1 == eager, f"traced verify != eager verify: {r1} vs {eager}"


@torch.no_grad()
def test_mtp_verify_trace_is_repeatable_across_rounds(mesh_device):
    """Replays with DIFFERENT inputs must track the eager path round after round — this is what
    catches an input buffer that was baked into the trace instead of being read each replay."""
    args = ModelArgs(mesh_device, ckpt_dir=CKPT, max_seq_len=int(os.environ.get("QWEN36_MAX_SEQ", "256")))
    loader = CheckpointLoader(CKPT)
    if not loader.has_mtp():
        pytest.skip("checkpoint has no mtp.* head")
    model = TtModel(mesh_device, args, loader, num_layers=int(os.environ.get("QWEN36_LAYERS", "4")))
    torch.manual_seed(1)
    prompt = torch.randint(0, args.vocab_size, (1, 16))
    lg = model.forward(prompt)
    model.start_decode(int(lg[0, -1].argmax()))
    model.build_mtp_head(gamma=2)
    K = model.setup_mtp_decode()
    cur0 = int(ttnn.to_torch(model.t_tok).reshape(-1)[0])
    model.capture_mtp_verify_trace([cur0] + [100 + i for i in range(K - 1)], [model.pos + i for i in range(K)])

    cur = int(ttnn.to_torch(model.t_tok).reshape(-1)[0])
    for r in range(4):
        toks = [cur] + [100 + 13 * r + i for i in range(K - 1)]
        positions = [model.pos + i for i in range(K)]
        traced = model.verify_step_traced(toks, positions)
        lgv, _ = model.verify_forward(toks, positions)
        eager = torch.as_tensor(ttnn.to_torch(lgv)).reshape(K, -1).argmax(dim=-1).tolist()
        print(f"[trace] round {r}: traced {traced}  eager {eager}")
        assert traced == eager, f"round {r}: traced {traced} != eager {eager}"
        model.commit_verify(1)
        cur = traced[0]


@torch.no_grad()
def test_mtp_full_traced_round_matches_eager(mesh_device):
    """All THREE traces together (draft chain, verify, per-accept-count commit) must produce the same
    token stream as fully-eager rounds.

    This is the only gate that can catch a bad COMMIT trace. A wrong accept-count rollback does not
    change any single verify's logits — it corrupts the state the NEXT round starts from — so per-op
    PCC gates and the verify-parity gate are both blind to it; only the emitted stream diverges.
    """
    args = ModelArgs(mesh_device, ckpt_dir=CKPT, max_seq_len=int(os.environ.get("QWEN36_MAX_SEQ", "256")))
    loader = CheckpointLoader(CKPT)
    if not loader.has_mtp():
        pytest.skip("checkpoint has no mtp.* head")
    model = TtModel(mesh_device, args, loader, num_layers=int(os.environ.get("QWEN36_LAYERS", "4")))
    torch.manual_seed(3)
    prompt = torch.randint(0, args.vocab_size, (1, 16))
    gamma, rounds = 2, 6
    model.build_mtp_head(gamma=gamma)

    def run(traced):
        model.mtp_verify_trace = None
        model.mtp_draft_trace = None
        model.mtp_commit_traces = None
        lg = model.forward(prompt)
        model.start_decode(int(lg[0, -1].argmax()))
        model._mtp_hidden = None
        model.setup_mtp_decode()
        if traced:
            # Two eager rounds first: round 1 is the K=1 seed step, round 2 is the first real K-row
            # verify. The commit traces read `_verify_pending`, which only holds the K-SHAPED scratch
            # after a K-row verify has run — so this ordering is required, not incidental.
            out = model.spec_decode_step() + model.spec_decode_step()
            cur = int(ttnn.to_torch(model.t_tok).reshape(-1)[0])
            positions = [model.pos + i for i in range(gamma + 1)]
            model.capture_mtp_verify_trace([cur] * (gamma + 1), positions)
            model._mtp_write_round([cur] * (gamma + 1), positions)
            model.capture_mtp_draft_trace()
            model.capture_mtp_commit_traces()
        else:
            out = model.spec_decode_step() + model.spec_decode_step()
        for _ in range(rounds):
            out.extend(model.spec_decode_step())
        return out

    eager = run(False)
    traced = run(True)
    n = min(len(eager), len(traced))
    div = next((i for i in range(n) if eager[i] != traced[i]), n)
    print(f"[trace] full-traced round vs eager: identical for {div}/{n} tokens")
    print(f"[trace]   eager  {eager[:12]}")
    print(f"[trace]   traced {traced[:12]}")
    assert div == n, (
        f"fully-traced rounds diverged at token {div}: "
        f"eager {eager[max(0,div-2):div+3]} vs traced {traced[max(0,div-2):div+3]}"
    )
