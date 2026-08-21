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
from models.demos.qwen3_6_a3b.tt.mtp_sampling import SpecSampler

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


@torch.no_grad()
@pytest.mark.parametrize("gamma", [1, 2])
def test_mtp_verify_candidates_match_eager_topk(mesh_device, gamma):
    """The SAMPLING verify tail must hand back exactly the target distribution's top candidates.

    Speculative sampling is only distribution-exact if p really is the model's own truncated
    distribution, so the device tail (per-row pad -> chunked topk -> global index offsets -> concat)
    is checked against `torch.topk` of the EAGER full-vocab verify logits. The decisive assertion is
    the VALUE/INDEX PAIRING: `logits[row, returned_index] == returned_value`. That is what catches
    both failure modes this tail can have -- a wrong per-chunk index offset, and the TILE-layout
    cross-row reshape that scrambled candidates in the batched decode path -- and unlike a set
    comparison it is immune to the many exact ties bf16 logits have across a 248k vocab.

    Note what the tail does NOT return: the global top-256. It returns each of the 8 vocab chunks'
    own top-32, which is a different set whenever one chunk holds more than 32 of the global top-256.
    That is exactly the shipping decode tail's design, and it is exact where it matters: any member of
    the global top-32 is necessarily in its OWN chunk's top-32, so for `top_k <= SAMP_K` the truncated
    distribution is the true one. Hence the value assertion below is on the top-SAMP_K prefix, and the
    real check is that the truncated distribution the host rule derives from the device candidates
    equals the one it derives from the full-vocab logits.
    """
    args = ModelArgs(mesh_device, ckpt_dir=CKPT, max_seq_len=int(os.environ.get("QWEN36_MAX_SEQ", "256")))
    loader = CheckpointLoader(CKPT)
    if not loader.has_mtp():
        pytest.skip("checkpoint has no mtp.* head")
    model = TtModel(mesh_device, args, loader, num_layers=int(os.environ.get("QWEN36_LAYERS", "4")))
    torch.manual_seed(0)
    prompt = torch.randint(0, args.vocab_size, (1, 16))

    model.enable_sampling(0.6, 20, 0.95, seed=0)  # selects the candidate tail; also builds _samp_* tables
    lg = model.forward(prompt)
    model.start_decode(int(lg[0, -1].argmax()))
    model.build_mtp_head(gamma=gamma)
    K = model.setup_mtp_decode()
    assert model._mtp_tail() == "sampling"

    toks = [int(ttnn.to_torch(model.t_tok).reshape(-1)[0])] + [7 + i for i in range(gamma)]
    positions = [model.pos + i for i in range(K)]

    lgv, _ = model.verify_forward(toks, positions)  # eager reference (leaves state untouched)
    ref = torch.as_tensor(ttnn.to_torch(lgv)).reshape(K, -1).float()

    model.capture_mtp_verify_trace(toks, positions)
    model.verify_step_traced(toks, positions)
    v, i = model.read_verify_candidates(K)
    W = v.shape[1]
    assert W == model._samp_nc * model.SAMP_K, (W, model._samp_nc, model.SAMP_K)
    rv, ri = ref.topk(W, dim=-1)

    sm = SpecSampler(temperature=0.6, top_k=20, top_p=0.95)
    for r in range(K):
        assert int(i[r].min()) >= 0 and int(i[r].max()) < args.vocab_size, "candidate index out of vocab"
        paired = ref[r, i[r]]
        assert torch.allclose(
            paired, v[r], atol=1e-6
        ), f"row {r}: value/index pairing broken, max delta {float((paired - v[r]).abs().max()):.4f}"
        got = torch.sort(v[r], descending=True).values[: model.SAMP_K]
        assert torch.allclose(got, rv[r][: model.SAMP_K], atol=1e-6), (
            f"row {r}: the global top-{model.SAMP_K} is not fully present in the candidates "
            f"(max delta {float((got - rv[r][: model.SAMP_K]).abs().max()):.4f})"
        )
        p_dev, id_dev = sm.probs(v[r], i[r])
        p_ref, id_ref = sm.probs(rv[r], ri[r])
        print(f"[trace] gamma={gamma} row {r}: |support| {id_dev.numel()}  p_max {float(p_dev[0]):.4f}")
        assert torch.allclose(p_dev, p_ref, atol=1e-6), f"row {r}: truncated distribution differs"
        # Identical ids too, unless the kept prefix holds an exact logit tie (bf16 over a 248k vocab
        # has many), in which case which index wins is arbitrary and does not change p.
        if p_dev.numel() == 1 or bool((p_dev[:-1] > p_dev[1:]).all()):
            assert id_dev.tolist() == id_ref.tolist(), f"row {r}: truncated support differs"


@torch.no_grad()
def test_mtp_candidates_valid_after_a_greedy_capture(mesh_device):
    """REGRESSION: only ONE MTP trace set may be resident when the verify trace is captured.

    Capturing the sampling verify trace while a previous round's traces were still resident produced a
    silently CORRUPT trace: the candidate rows read back as ~1e9 garbage, the acceptance rule then
    "sampled" an out-of-vocab id, and feeding that to ttnn.embedding HUNG the device — a failure that
    looks like a hang in the server, miles from its cause. `setup_mtp_traces` now releases everything
    first; this test pins that down by doing the exact sequence that used to fail (capture the greedy
    tail, run rounds, then switch to sampling and capture again).

    The assertion is deliberately crude — every candidate index must be a real vocab id — because that
    is precisely what the corruption violated, and it is what protects the device from the hang.
    """
    args = ModelArgs(mesh_device, ckpt_dir=CKPT, max_seq_len=int(os.environ.get("QWEN36_MAX_SEQ", "256")))
    loader = CheckpointLoader(CKPT)
    if not loader.has_mtp():
        pytest.skip("checkpoint has no mtp.* head")
    model = TtModel(mesh_device, args, loader, num_layers=int(os.environ.get("QWEN36_LAYERS", "4")))
    torch.manual_seed(0)
    prompt = torch.randint(0, args.vocab_size, (1, 16))

    def run(temperature):
        model.enable_sampling(temperature, 20, 0.95, 0, 0.0)
        model.build_mtp_head(gamma=2)
        lg = model.forward(prompt)
        model.start_decode(int(lg[0, -1].argmax()))
        model.setup_mtp_decode()
        model._mtp_hidden = None
        model.setup_mtp_traces()
        return model.spec_decode_step()

    greedy = run(0.0)
    assert model.mtp_verify_tail == "greedy"
    assert all(0 <= t < args.vocab_size for t in greedy), greedy

    sampled = run(1.0)  # the switch that used to corrupt the newly captured trace
    assert model.mtp_verify_tail == "sampling"
    v, i = model.read_verify_candidates()
    print(
        f"[trace] after greedy->sampling switch: emitted {sampled}  "
        f"cand idx range [{int(i.min())}, {int(i.max())}]  values [{float(v.min()):.2f}, {float(v.max()):.2f}]"
    )
    assert int(i.min()) >= 0 and int(i.max()) < args.vocab_size, (
        f"candidate indices outside the vocab ([{int(i.min())}, {int(i.max())}]) — the verify trace was "
        f"captured while other traces were resident and is corrupt"
    )
    assert all(0 <= t < args.vocab_size for t in sampled), f"emitted an out-of-vocab id: {sampled}"
    # a greedy round must still be servable by the sampling-tail trace (it writes _m_argmax too)
    model.enable_sampling(0.0, 0, 1.0, 0, 0.0)
    assert model.mtp_tail_ready(), "a greedy round must be able to replay the sampling-tail trace"
    again = model.spec_decode_step()
    assert all(0 <= t < args.vocab_size for t in again), again
