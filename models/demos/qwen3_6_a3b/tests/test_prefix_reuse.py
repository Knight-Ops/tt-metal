# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn prefix reuse: resuming a prefill from a gated-delta state checkpoint must match a
from-scratch prefill of the same prompt.

The serving path (demo/server.py, packaging/vllm_bundle/generator_vllm.py) prefills turn N+1 by
CONTINUING from the caches turn N left behind, rewinding the gated-delta recurrent + conv state to
the last checkpoint the new prompt still agrees with. Everything that can go wrong with that is
either an alignment fault (chunked-SDPA / fill_cache offsets) or a stale-state fault (restoring the
wrong context's state), so this checks both against the reference: an unrelated prompt must NOT be
reused, an extension must be, and the resumed logits must track a full prefill.

Correctness is gated on PCC vs single-shot, not exact argmax -- incremental prefill is a ~0.99-PCC
approximation of full attention (different kernels), the same tolerance test_ragged_prefill uses.

Uses real weights (QWEN36_LAYERS, default 4). The pure-policy tests need no device.
"""
import os

import torch

from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import (
    ConversationSlots,
    TtModel,
    common_prefix_len,
    pick_checkpoint,
    plan_prefill_blocks,
)
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# ── policy (device-free) ──────────────────────────────────────────────────────


def test_common_prefix_len():
    assert common_prefix_len([1, 2, 3], [1, 2, 3]) == 3
    assert common_prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix_len([1, 2, 3], [9, 2, 3]) == 0
    assert common_prefix_len([1, 2], [1, 2, 3, 4]) == 2  # history is a prefix of the new prompt
    assert common_prefix_len([], [1, 2]) == 0
    assert common_prefix_len([1, 2], []) == 0


def test_pick_checkpoint():
    ck = [256, 512, 768]
    assert pick_checkpoint(ck, 800) == 768  # deepest usable
    assert pick_checkpoint(ck, 768) == 768  # a checkpoint exactly at the divergence point is usable
    assert pick_checkpoint(ck, 767) == 512  # ...one past it is not
    assert pick_checkpoint(ck, 100) == 0  # nothing early enough
    assert pick_checkpoint([], 9999) == 0
    # min_reuse: don't pay a restore to save a handful of positions
    assert pick_checkpoint(ck, 800, min_reuse=512) == 768
    assert pick_checkpoint(ck, 300, min_reuse=512) == 0
    # a checkpoint may never sit past the divergence point, whatever the inputs
    for lcp in range(0, 1200, 37):
        assert pick_checkpoint(ck, lcp) <= lcp


def test_conversation_slots_route_by_prefix():
    """Interleaved streams must land on their own slot. This is the exact traffic that broke the
    single-slot adapter: one long conversation growing a little each turn, and short side requests
    (a chat UI's auto-title / auto-tag completions) between every pair of turns."""
    long_a = list(range(6288))
    side_b = [900_000 + i for i in range(209)]
    slots = ConversationSlots(4, min_lcp=128)

    a, lcp, ev = slots.match(long_a)
    assert (lcp, ev) == (0, None), "first sight of a conversation reuses nothing"
    slots.commit(a, long_a, 0)

    b, lcp, ev = slots.match(side_b)
    assert b != a, "an unrelated prompt must not take the long conversation's slot"
    assert (lcp, ev) == (0, None)
    slots.commit(b, side_b, 0)

    # ...and now the long conversation's next turn (+56 tokens) still finds its own history intact.
    turn2 = long_a + list(range(6288, 6344))
    a2, lcp, ev = slots.match(turn2)
    assert a2 == a, "the long conversation must route back to its own slot"
    assert lcp == 6288, f"the whole previous prompt is a prefix of this one, got lcp={lcp}"
    assert ev is None
    slots.commit(a2, turn2, 6144)
    assert slots.streak[a2] == 1

    # The side stream grows too, and keeps its own slot and its own depth.
    side2 = side_b + [900_500 + i for i in range(54)]
    b2, lcp, _ = slots.match(side2)
    assert (b2, lcp) == (b, 209)


def test_conversation_slots_min_lcp_and_lru():
    """Below min_lcp a prompt is a NEW conversation (a shared system prompt must not let a stray
    request claim a long conversation's slot), and a full table evicts least-recently-used."""
    sysp = list(range(64))  # a system prompt every request shares
    slots = ConversationSlots(2, min_lcp=128)
    c0 = [*sysp, *range(1000, 2000)]
    i0, _, _ = slots.match(c0)
    slots.commit(i0, c0, 0)
    c1 = [*sysp, *range(5000, 5100)]
    i1, lcp, ev = slots.match(c1)
    assert i1 != i0, "64 shared tokens is below min_lcp: this is a different conversation"
    assert (lcp, ev) == (0, None)
    slots.commit(i1, c1, 0)

    # Table full. c0 was touched first, so it is the victim.
    c2 = list(range(9000, 9500))
    i2, lcp, ev = slots.match(c2)
    assert (i2, ev) == (i0, i0), f"expected to evict the LRU slot {i0}, got {i2} (evicted {ev})"
    assert lcp == 0
    slots.forget(i2)
    slots.commit(i2, c2, 0)
    # ...and c1, having been touched more recently, survived.
    assert slots.match([*c1, 7])[0] == i1


def test_conversation_slots_lru_query():
    """`lru` is what frees KV blocks before a prefill that would exhaust the pool, so it must never
    offer the conversation that is about to run, and must prefer the coldest one."""
    slots = ConversationSlots(3, min_lcp=128)
    for i, base in enumerate((1000, 2000, 3000)):
        ids = list(range(base, base + 200))
        j, _, _ = slots.match(ids)
        slots.commit(j, ids, 0)
    assert slots.lru() == 0, "slot 0 was touched first"
    assert slots.lru(exclude=(0,)) == 1
    assert slots.lru(exclude=(0, 1, 2)) is None, "never evict when nothing is evictable"
    # Touching a slot makes it the newest, so the next victim moves along.
    slots.match(list(range(1000, 1201)))
    assert slots.lru() == 1
    slots.forget(1)
    assert slots.lru() == 2, "a forgotten slot is free, not a victim"


def test_conversation_slots_single_slot_is_the_old_behaviour():
    """n=1 must behave exactly like the single `_reuse_hist` it replaces: every prompt claims the
    one slot and overwrites the history."""
    slots = ConversationSlots(1, min_lcp=128)
    a = list(range(500))
    assert slots.match(a) == (0, 0, None)
    slots.commit(0, a, 0)
    assert slots.match(a + [1]) == (0, 500, None)
    other = list(range(9000, 9200))
    i, lcp, ev = slots.match(other)
    assert (i, lcp, ev) == (0, 0, 0), "with one slot an unrelated prompt must EVICT, not alias"


def _check_plan(T, start, chunk=512, single_shot_max=2048, snap=128):
    """Assert the schedule tiles [start, T) and obeys the rules that are actually enforced, then
    return the positions a checkpoint would be recorded at.

    The rules, and which are real: chunked-SDPA fatals unless ``chunk_start % q_chunk_size == 0``,
    and q_chunk_size defaults to the block width -- that one is a kernel constraint. Offsets on the
    ``snap`` grain and widths off the padding ladder are OUR constraints (snapshot placement and
    program-cache growth respectively). The old "only chunk-wide blocks at chunk-multiple offsets"
    envelope is gone: it existed because P=640/width=256 and P=768/width=256 hung the device, which
    turned out to be the per-width-constant corruption fixed by prewarm_prefill_shapes."""
    plan = plan_prefill_blocks(T, start=start, chunk=chunk, single_shot_max=single_shot_max, snap=snap)
    P, snaps = start, []
    for off, m, kind, q_chunk in plan:
        assert off == P, f"T={T} start={start}: block at {off} does not continue from {P}"
        assert m > 0
        # THE kernel rule. q_chunk=None means "use the width", so the width must divide the offset.
        eff = q_chunk if q_chunk is not None else m
        assert off % eff == 0, (
            f"T={T} start={start}: chunked-SDPA needs chunk_start % q_chunk == 0, got off={off} "
            f"q_chunk={eff} ({kind})"
        )
        if kind == "single":
            assert off == 0, "a single-shot block can only be the first one"
            assert m % snap == 0 or m == T, f"T={T}: single-shot width {m} must be on the snap grain"
        else:
            assert off % snap == 0, f"T={T} start={start}: offset {off} is off the snap grain {snap}"
        if kind == "block":
            assert m % snap == 0 and m <= chunk, f"T={T}: block width {m} (snap {snap}, chunk {chunk})"
        if kind == "ragged":
            assert plan[-1] == (off, m, kind, q_chunk), "a ragged block must be last"
            assert m < snap, f"T={T}: ragged remainder {m} should be below the snap grain {snap}"
            assert q_chunk == 128, f"T={T}: ragged q_chunk {q_chunk} != 128"
        else:
            snaps.append(off + m)
        P = off + m
    assert P == T, f"T={T} start={start}: plan covers [{start},{P}), not [{start},{T})"
    if T < snap:  # degenerate: one bare block, and snapshot_gdn_state rejects its unaligned end
        assert snaps == [T]
        return []
    assert all(x % snap == 0 for x in snaps), f"T={T}: checkpoints {snaps} must be snap multiples"
    return snaps


def test_plan_covers_and_aligns():
    for chunk, snap in ((512, 128), (512, 512), (256, 128), (256, 256), (128, 128)):
        for T in [1, 127, 128, 129, 255, 256, 384, 512, 640, 700, 766, 812, 896, 1026, 2048, 2049, 5000, 32768]:
            snaps = _check_plan(T, 0, chunk=chunk, snap=snap)
            if T >= snap:
                # The deepest checkpoint must be where the next turn's divergence point (T-2, the
                # generation prompt's "<think>\n") floors to, or reuse silently never engages.
                assert max(snaps) == (T // snap) * snap, f"T={T} chunk={chunk} snap={snap}: {max(snaps)}"
                assert pick_checkpoint(snaps, T - 2) == max(snaps) or T % snap < 2
            for st in range(snap, min(T, 4096), snap):
                _check_plan(T, st, chunk=chunk, snap=snap)


def test_plan_splits_only_the_tail():
    """The coarse structure is untouched -- one wide single-shot head, then chunk-wide blocks -- and
    only the TAIL is split so a snapshot lands on the snap grain. Keeping the head chunk-aligned is
    what keeps its padded width on the ladder instead of compiling a new width per prompt length."""
    plan = plan_prefill_blocks(700, start=0, chunk=512, single_shot_max=2048, snap=128)
    assert [(p, m, k) for p, m, k, _ in plan] == [(0, 512, "single"), (512, 128, "block"), (640, 60, "ragged")]
    # the head is still ONE single-shot pass: reuse must not turn a short prefill into a chunked walk
    assert sum(1 for _, _, k, _ in plan if k == "single") == 1

    # A long prompt keeps prefill_long's chunked structure; only where the tail lands moves.
    long_plan = plan_prefill_blocks(5000, start=0, chunk=512, single_shot_max=2048, snap=128)
    kinds = [k for _, _, k, _ in long_plan]
    assert kinds[0] == "single" and kinds[-1] == "ragged"
    # 8 chunk-wide blocks (512..4608) plus ONE narrowing block (4608->4992) so a snapshot lands
    # on the 128 grain. That extra block is the whole cost of the finer grain.
    assert kinds.count("block") == 9, kinds
    assert [(p, m, k) for p, m, k, _ in long_plan][-1:] == [(4992, 8, "ragged")]

    # T below the snap grain cannot be checkpointed at all, and says so by planning one bare block
    assert plan_prefill_blocks(100, start=0, chunk=512, snap=128) == [(0, 100, "single", None)]

    # The two configurations that used to hang are now REACHABLE on purpose -- that is the point of
    # the 128 grain. Resuming at 640 emits width 256 at offset 640, which is exactly the shape the
    # old envelope forbade; it is legal because q_chunk is pinned to 128 (640 % 128 == 0).
    for good in (640, 768):
        plan = plan_prefill_blocks(1026, start=good, chunk=512, snap=128)
        for off, m, kind, q_chunk in plan:
            eff = q_chunk if q_chunk is not None else m
            assert off % eff == 0, (off, m, kind, q_chunk)
    # ...while an offset off the snap grain is still rejected outright
    for bad in (700, 65, 1000):
        try:
            plan_prefill_blocks(1026, start=bad, chunk=512, snap=128)
        except AssertionError:
            continue
        raise AssertionError(f"start={bad} is off the 128 grain and must be rejected")


# ── device: resumed prefill == full prefill ───────────────────────────────────

CH = 256  # test chunk: small enough that a ~700-token prompt records a LADDER of checkpoints


def _build(mesh_device, max_seq=1024):
    n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_seq_len=max_seq)
    model = TtModel(mesh_device, args, CheckpointLoader(ckpt), num_layers=n_layers)
    # Force the chunked walk at test-sized prompts (the shipped 2048 threshold would prefill all of
    # these single-shot, recording exactly one checkpoint). The chunk sets the COARSE structure;
    # CKPT_ALIGN is the separate, finer snapshot grain (128 by default), so a 700-token prompt
    # records checkpoints at both the chunk boundaries and the last 128 multiple.
    model._LONG_PREFILL_THRESHOLD = 128
    model._PREFILL_CHUNK = CH
    return model, args


@torch.no_grad()
def test_prefix_reuse_matches_full_prefill(mesh_device):
    model, args = _build(mesh_device)
    torch.manual_seed(0)
    rnd = lambda n: torch.randint(0, args.vocab_size, (1, n))

    # ── cold turn: no history, so a full prefill that leaves a checkpoint ladder ──
    p1 = rnd(700)
    l1, reused = model.prefill_reuse(p1, None, min_reuse=128)
    assert reused == 0, "no history was supplied, so nothing may be reused"
    assert model.pos == 700
    # 256/512 are block boundaries. 640 is the last 128 multiple below 700 and lands INSIDE the
    # final block: the recurrence exports its carried state there mid-block, so the finer resume
    # grain costs one slice per linear layer rather than a whole extra block. See
    # GatedDelta.supports_inblock_snapshot and TtModel._begin_inblock_snapshot.
    assert model.checkpoint_positions() == [256, 512, 640], model.checkpoint_positions()

    # ── next turn: shares 500 tokens, then diverges (what a dropped <think> block looks like) ──
    p2 = torch.cat([p1[:, :500], rnd(300)], dim=1)  # 800 tokens, divergence at 500
    l2, reused = model.prefill_reuse(p2, p1[0].tolist(), min_reuse=128)
    assert reused == 256, f"expected the deepest checkpoint <= 500, got {reused}"
    assert model.pos == 800
    ref2 = model._prefill_single(p2)  # reference LAST: it resets the state the reuse path just built
    pcc = _pcc(l2[0, -1], ref2[0, -1])
    print(f"\n[prefix-reuse] resumed-from-256 vs full prefill: PCC {pcc:.5f}")
    assert pcc > 0.97, f"resumed prefill vs full prefill PCC {pcc:.4f} too low"

    # ── an unrelated prompt must fall back to a full prefill ──
    model.drop_state_checkpoints()
    _, reused = model.prefill_reuse(rnd(700), p2[0].tolist(), min_reuse=128)
    assert reused == 0, "a prompt with no shared prefix must not reuse anything"
    _, reused = model.prefill_reuse(rnd(700), None, min_reuse=128)
    assert reused == 0


@torch.no_grad()
def test_prefix_reuse_survives_several_turns(mesh_device):
    """Drift check: each resumed turn extends a RESTORED state rather than one recomputed from the
    prompt, so error accumulates across turns in a way single-hop PCC does not measure. Four chained
    turns, then compare against a full prefill of the final prompt."""
    model, args = _build(mesh_device)
    torch.manual_seed(1)
    rnd = lambda n: torch.randint(0, args.vocab_size, (1, n))

    prompt = rnd(400)
    _, reused = model.prefill_reuse(prompt, None, min_reuse=128)
    hist = prompt[0].tolist()
    logits = None
    for turn in range(4):
        # A turn's prompt keeps the whole rendered history and appends: some of the previous prompt's
        # tail is re-rendered (the assistant's visible answer) and new text follows.
        prompt = torch.cat([prompt[:, :-8], rnd(120)], dim=1)
        logits, reused = model.prefill_reuse(prompt, hist, min_reuse=128)
        hist = prompt[0].tolist()
        assert reused > 0, f"turn {turn}: extension should have reused a checkpoint"
        assert model.pos == prompt.shape[1]
    ref = model._prefill_single(prompt)
    pcc = _pcc(logits[0, -1], ref[0, -1])
    print(f"\n[prefix-reuse] 4 chained resumed turns vs full prefill: PCC {pcc:.5f}")
    assert pcc > 0.97, f"after 4 resumed turns, PCC vs full prefill {pcc:.4f} too low"


@torch.no_grad()
def test_checkpoint_bookkeeping(mesh_device):
    """The ring must never hand back a position it does not hold, and must forget positions past the
    end of the current sequence -- a shorter follow-up prompt would otherwise be able to restore the
    previous, longer context's state at a position it also reaches."""
    model, args = _build(mesh_device)
    torch.manual_seed(2)
    long_prompt = torch.randint(0, args.vocab_size, (1, 900))
    model.prefill_reuse(long_prompt, None, min_reuse=128)
    # 256/512/768 are block boundaries; 896 is the in-block checkpoint (last 128 multiple below 900).
    assert model.checkpoint_positions() == [256, 512, 768, 896], model.checkpoint_positions()

    assert model.restore_gdn_state(300) is False, "300 is not a recorded checkpoint"
    assert model.restore_gdn_state(512) is True
    assert model.pos == 512, "a restore must rewind pos too"

    model.drop_state_checkpoints(above=600)
    assert model.checkpoint_positions() == [256, 512]
    model.drop_state_checkpoints()
    assert model.checkpoint_positions() == []
    # ...and a prompt shorter than the recorded ladder gets no stale reuse
    model.prefill_reuse(long_prompt, None, min_reuse=128)
    short = torch.cat([long_prompt[:, :400], torch.randint(0, args.vocab_size, (1, 20))], dim=1)
    _, reused = model.prefill_reuse(short, long_prompt[0].tolist(), min_reuse=128)
    assert reused == 256, f"reuse must floor to a checkpoint at or before the divergence, got {reused}"
    assert max(model.checkpoint_positions()) <= short.shape[1]


# ── device: several conversations parked at once ──────────────────────────────


def _build_multi(monkeypatch, mesh_device, n_conv=2, max_seq=1024, pool=4096):
    """A model with `n_conv` parked conversation slots. Needs paged KV -- see the WHY in
    TtModel.alloc_conversation_slots: with the flat cache every conversation writes KV rows from 0,
    so two of them cannot coexist however much history the caller remembers.

    The env has to stay set for the whole test, not just the constructor: the pager is built lazily
    inside the first `_alloc_cache`, i.e. on the first prefill, and that is where the pool budget is
    read. `pool` must cover all `n_conv` conversations at once -- one pool is the point of paging,
    and the default (max_seq, enough for ONE full-length conversation) exhausts with two."""
    monkeypatch.setenv("QWEN36_PAGED_KV", "1")
    monkeypatch.setenv("QWEN36_KV_POOL_TOKENS", str(pool))
    model, args = _build(mesh_device, max_seq=max_seq)
    model.alloc_conversation_slots(n_conv)
    return model, args


@torch.no_grad()
def test_interleaved_conversations_keep_their_prefix(monkeypatch, mesh_device):
    """THE REGRESSION. Two prompt streams interleaved: one growing conversation and one unrelated
    short request between its turns. Before parked conversations, the short request overwrote the
    single remembered history AND ran an unscoped drop_state_checkpoints(above=its own T), so the
    conversation's next turn found checkpoints=[] and re-prefilled in full every time (measured in
    the shipped container: "reused 0 (0%)" on every request, forever).

    Both halves are checked, because the adapter half alone would be a correctness bug rather than a
    slowdown: reuse must ENGAGE for the conversation, and its logits must still track a full
    prefill, which they only can if the side request wrote its KV somewhere else."""
    model, args = _build_multi(monkeypatch, mesh_device, n_conv=2)
    torch.manual_seed(3)
    rnd = lambda n: torch.randint(0, args.vocab_size, (1, n))

    chat = rnd(700)
    _, reused = model.prefill_reuse(chat, None, min_reuse=128, conv=0)
    assert reused == 0
    chat_ckpts = model.checkpoint_positions(0)
    assert chat_ckpts, "the first turn must leave a checkpoint to resume from"

    # The interloper. A different conversation, shorter than the chat, ending mid-ladder -- which is
    # precisely what made the unscoped `above=T` drop so destructive.
    side = rnd(400)
    _, reused = model.prefill_reuse(side, None, min_reuse=128, conv=1)
    assert reused == 0
    assert model.checkpoint_positions(0) == chat_ckpts, (
        f"the side request destroyed the chat's checkpoints: {chat_ckpts} -> " f"{model.checkpoint_positions(0)}"
    )
    assert model.checkpoint_positions(1), "the side request has its own checkpoints"

    # Turn 2 of the chat: keeps the whole prompt and appends, exactly like a growing transcript.
    turn2 = torch.cat([chat, rnd(100)], dim=1)
    l2, reused = model.prefill_reuse(turn2, chat[0].tolist(), min_reuse=128, conv=0)
    assert reused >= max(p for p in chat_ckpts), f"expected to resume from {chat_ckpts}, reused {reused}"
    assert model.pos == turn2.shape[1]

    # ...and the answer is still right, which is the half that proves the KV went to slot 1.
    ref = model._prefill_single(turn2)
    pcc = _pcc(l2[0, -1], ref[0, -1])
    print(f"\n[multi-conv] resumed-after-interleave vs full prefill: PCC {pcc:.5f} (reused {reused})")
    assert pcc > 0.97, f"resumed-after-interleave PCC {pcc:.4f}: the interloper corrupted the chat"


@torch.no_grad()
def test_conversation_slots_do_not_share_kv_blocks(monkeypatch, mesh_device):
    """Each parked conversation must own its blocks, and aging one out must return them."""
    model, _ = _build_multi(monkeypatch, mesh_device, n_conv=2)
    torch.manual_seed(4)
    a = torch.randint(0, model.args.vocab_size, (1, 700))
    b = torch.randint(0, model.args.vocab_size, (1, 400))
    model.prefill_reuse(a, None, min_reuse=128, conv=0)
    model.prefill_reuse(b, None, min_reuse=128, conv=1)

    pg = model.kv_pager
    rows = [set(int(x) for x in pg._host[i][: pg._mapped[i]].tolist()) for i in range(2)]
    assert rows[0] and rows[1], f"both conversations must have mapped blocks, got {pg._mapped}"
    assert not (rows[0] & rows[1]), f"conversations share physical blocks {rows[0] & rows[1]}"
    assert model.conversation_tokens(0) >= 700 and model.conversation_tokens(1) >= 400

    # Aging out returns the blocks AND forgets the checkpoints; the other conversation is untouched.
    kept = model.checkpoint_positions(0)
    free_before = len(pg._free)
    model.release_conversation(1)
    assert len(pg._free) == free_before + len(rows[1])
    assert model.checkpoint_positions(1) == []
    assert model.checkpoint_positions(0) == kept
    assert model.conversation_tokens(1) == 0


@torch.no_grad()
def test_pool_pressure_ages_a_conversation_out(monkeypatch, mesh_device):
    """A free SLOT is not the same as pool HEADROOM. Parked conversations hold their blocks until
    aged out and the pool has no eviction of its own, so `ensure` would raise "block pool exhausted"
    partway through a prefill -- a failed request, not a slow one. The adapter checks headroom first
    (kv_headroom_tokens) and ages out the coldest conversation; this pins both halves."""
    # A pool deliberately too small for two 700-token conversations at once.
    model, args = _build_multi(monkeypatch, mesh_device, n_conv=2, pool=1024)
    torch.manual_seed(5)
    a = torch.randint(0, args.vocab_size, (1, 700))
    b = torch.randint(0, args.vocab_size, (1, 700))
    slots = ConversationSlots(model.n_conv_slots, 128)

    i, _, _ = slots.match(a[0].tolist())
    model.prefill_reuse(a, None, min_reuse=128, conv=i)
    slots.commit(i, a[0].tolist(), 0)
    assert model.kv_headroom_tokens() < 700, "the pool must be too small for a second 700 tokens"

    j, _, ev = slots.match(b[0].tolist())
    assert (j, ev) != (i, i), "slot 1 is free, so nothing should be evicted on slot grounds alone"
    need = 700 - model.conversation_tokens(j)
    victims = []
    while need > 0 and model.kv_headroom_tokens() < need:
        v = slots.lru(exclude=(j,))
        assert v is not None
        victims.append(v)
        slots.forget(v)
        model.release_conversation(v)
    assert victims == [i], f"the only other conversation should have been aged out, got {victims}"
    assert model.checkpoint_positions(i) == []
    # ...and the prefill now completes instead of raising mid-way.
    _, reused = model.prefill_reuse(b, None, min_reuse=128, conv=j)
    assert reused == 0
    assert model.pos == 700
