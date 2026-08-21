# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exactness gate for the MTP speculative-SAMPLING acceptance rule (`tt/mtp_sampling.py`).

Speculative sampling is only worth anything if it is actually distribution-preserving: the whole
point is to accept draft tokens *without* changing what the model would have sampled. That claim is a
probability statement, not a PCC, so the right gate is a statistical one — draw the round many times
and check the emitted-token histogram against the target distribution.

Deliberately DEVICE-FREE (the module under test imports no ttnn), so this runs on any machine in
under two minutes and gates every change to the rule. The device tests cover the plumbing that FEEDS this
rule (that verify's candidate buffers really are the target's top-k); this file covers the rule.
"""
from __future__ import annotations

import math

import pytest
import torch

from models.demos.qwen3_6_a3b.tt.mtp_sampling import (
    EXACT,
    GREEDY,
    LENIENT,
    RELAXED,
    SpecSampler,
    realized_tv,
    sampler_from_env,
)

NTRIAL = 80_000  # ~105 us/round -> ~8 s per distribution case; 5 sigma resolves a ~0.7% bias
SIGMA = 5.0  # max allowed deviation of any category, in standard errors (deterministic per seed)


def _cands(logits, base=1000):
    """(values, indices) for a candidate row, ids offset off 0 so index/position confusion is visible."""
    v = torch.tensor(logits, dtype=torch.float32)
    return v, torch.arange(base, base + v.numel(), dtype=torch.int64)


def _empirical(sampler, cands, drafts, n=NTRIAL, seed=0):
    """Run `n` independent rounds; return (histogram of the FIRST emitted token, mean accept prob)."""
    g = torch.Generator().manual_seed(seed)
    pool = torch.rand(n * (len(drafts) + 2), generator=g).tolist()
    it = iter(pool)
    hist, acc = {}, 0.0
    for _ in range(n):
        emitted, info = sampler.resolve_round(cands, drafts, lambda: next(it))
        hist[emitted[0]] = hist.get(emitted[0], 0) + 1
        acc += info["accept_probs"][0]
    return hist, acc / n


def _assert_matches(hist, p, ids, n, label):
    """Every category's empirical frequency must sit within SIGMA standard errors of p."""
    worst, chi2 = 0.0, 0.0
    for j, tok in enumerate(ids.tolist()):
        exp = float(p[j])
        obs = hist.get(tok, 0) / n
        se = math.sqrt(max(exp * (1 - exp), 1e-12) / n)
        worst = max(worst, abs(obs - exp) / se)
        chi2 += (obs - exp) ** 2 * n / max(exp, 1e-12)
    leaked = {t: c for t, c in hist.items() if t not in set(ids.tolist())}
    print(f"[specsamp] {label}: worst {worst:.2f} sigma, chi2 {chi2:.2f} (df {len(ids)-1}), leaked {leaked}")
    assert not leaked, f"{label}: emitted tokens outside the target support: {leaked}"
    assert worst < SIGMA, f"{label}: category off by {worst:.2f} sigma (> {SIGMA}) — rule is not exact"


# ── the core claim: exact rejection sampling reproduces the target distribution ───────────────
@pytest.mark.parametrize(
    "logits,draft_pos,label",
    [
        ([5.0, 4.0, 3.0, 2.0, 1.0], 0, "draft = argmax"),
        ([5.0, 4.0, 3.0, 2.0, 1.0], 2, "draft = mid-tail"),
        ([5.0, 4.0, 3.0, 2.0, 1.0], 4, "draft = worst candidate"),
        ([2.0, 2.0, 2.0, 2.0], 1, "near-uniform target"),
        ([12.0, 1.0, 0.5, 0.0], 0, "very peaked, draft = argmax"),
        ([12.0, 1.0, 0.5, 0.0], 3, "very peaked, draft rejected almost always"),
    ],
)
def test_exact_rule_reproduces_target(logits, draft_pos, label):
    s = SpecSampler(temperature=1.0, top_k=0, top_p=1.0, rule=EXACT)
    vals, ids_raw = _cands(logits)
    p, ids = s.probs(vals, ids_raw)
    draft = int(ids_raw[draft_pos])
    hist, mean_acc = _empirical(s, [(vals, ids_raw), (vals, ids_raw)], [draft])
    assert abs(mean_acc - s.mass(p, ids, draft)) < 1e-6, "reported accept prob must equal p(draft)"
    _assert_matches(hist, p, ids, NTRIAL, label)


def test_exact_rule_exact_when_draft_outside_support():
    """A draft outside the truncated support has p(x̂)=0 -> always rejected, and the residual is the
    whole target. This is the case a naive `p(x̂) or fallback` implementation gets wrong."""
    s = SpecSampler(temperature=1.0, top_k=3, top_p=1.0, rule=EXACT)
    vals, ids_raw = _cands([5.0, 4.0, 3.0, 2.0, 1.0])
    p, ids = s.probs(vals, ids_raw)
    assert ids.numel() == 3, "top_k=3 must truncate to 3 candidates"
    draft = int(ids_raw[4])  # outside the kept top-3
    hist, mean_acc = _empirical(s, [(vals, ids_raw)] * 2, [draft])
    assert mean_acc == 0.0
    _assert_matches(hist, p, ids, NTRIAL, "draft outside support")


def test_exact_rule_holds_with_top_k_top_p_and_temperature():
    s = SpecSampler(temperature=0.6, top_k=4, top_p=0.95, rule=EXACT)
    vals, ids_raw = _cands([6.0, 5.4, 5.0, 4.0, 2.0, 1.0, 0.0])
    p, ids = s.probs(vals, ids_raw)
    hist, _ = _empirical(s, [(vals, ids_raw)] * 2, [int(ids[1])])
    _assert_matches(hist, p, ids, NTRIAL, "T=0.6 k=4 p=0.95")


def test_first_token_is_exact_at_every_gamma():
    """The recursion must not leak: whatever gamma is, the FIRST emitted token is row-0 distributed."""
    s = SpecSampler(temperature=0.8, top_k=8, top_p=0.9, rule=EXACT)
    v0, i0 = _cands([5.0, 4.2, 3.1, 2.0, 1.0], base=1000)
    v1, i1 = _cands([1.0, 5.0, 2.0, 0.5, 0.0], base=1000)
    v2, i2 = _cands([0.0, 0.5, 6.0, 1.0, 2.0], base=1000)
    p, ids = s.probs(v0, i0)
    for gamma, cands in ((1, [(v0, i0), (v1, i1)]), (2, [(v0, i0), (v1, i1), (v2, i2)])):
        drafts = [int(i0[1])] + [int(i1[2])] * (gamma - 1)
        hist, _ = _empirical(s, cands, drafts, seed=gamma)
        _assert_matches(hist, p, ids, NTRIAL, f"gamma={gamma} first token")


# ── the commit contract the device path depends on ────────────────────────────────────────────
@pytest.mark.parametrize("gamma", [1, 2, 3])
def test_commit_count_relation(gamma):
    """`len(emitted)` is what gets passed to commit_verify, so it must always be accepted+1 and
    within 1..K — a violation would roll the recurrent state forward by the wrong number of steps."""
    s = SpecSampler(temperature=0.7, top_k=20, top_p=0.95, rule=EXACT)
    g = torch.Generator().manual_seed(7)
    vals = [_cands(torch.randn(16, generator=g).tolist()) for _ in range(gamma + 1)]
    it = iter(torch.rand(4000, generator=g).tolist())
    for _ in range(500):
        drafts = [int(vals[j][1][int(torch.randint(0, 16, (1,), generator=g))]) for j in range(gamma)]
        emitted, info = s.resolve_round(vals, drafts, lambda: next(it))
        assert 1 <= len(emitted) <= gamma + 1
        assert len(emitted) == info["accepted"] + 1
        assert info["bonus"] == (len(emitted) == gamma + 1)
        assert emitted[: info["accepted"]] == drafts[: info["accepted"]], "accepted prefix must be the drafts"


def test_rejection_never_emits_the_rejected_draft():
    """Forced rejection (u=1) must emit from the residual, i.e. never the draft itself."""
    s = SpecSampler(temperature=1.0, top_k=0, top_p=1.0, rule=EXACT)
    vals, ids_raw = _cands([5.0, 4.0, 3.0, 2.0])
    for pos in range(4):
        draft = int(ids_raw[pos])
        for u in (1.0, 0.999999):
            emitted, info = s.resolve_round([(vals, ids_raw)] * 2, [draft], lambda: u)
            assert info["accepted"] == 0 and emitted[0] != draft, (pos, u, emitted)


# ── the other two rules ──────────────────────────────────────────────────────────────────────
def test_greedy_rule_is_exact_match():
    s = SpecSampler(temperature=1.0, rule=GREEDY)
    vals, ids_raw = _cands([5.0, 4.0, 3.0])
    p, ids = s.probs(vals, ids_raw)
    assert s.accept(p, ids, int(ids[0]), 0.99)[0] is True
    assert s.accept(p, ids, int(ids[1]), 0.0)[0] is False


def test_low_temperature_degenerates_to_greedy():
    """As T -> 0 the target becomes a point mass, so EXACT must accept iff the draft is the argmax —
    which is why `temperature=1e-6` is a valid device-side parity check against the greedy path."""
    s = SpecSampler(temperature=1e-4, top_k=20, top_p=0.95, rule=EXACT)
    vals, ids_raw = _cands([5.0, 4.9, 3.0])
    p, ids = s.probs(vals, ids_raw)
    # top-p on a point mass collapses the support to ONE token — the mechanism that makes sampled
    # acceptance nearly as high as greedy at the shipping temperature (measured: mean support 1.3).
    assert ids.numel() == 1 and s.mass(p, ids, int(ids[0])) == pytest.approx(1.0, abs=1e-6)
    for u in (0.0, 0.5, 0.999999):
        assert s.accept(p, ids, int(ids[0]), u)[0], "argmax must always be accepted at T->0"
    for other in (int(ids_raw[1]), int(ids_raw[2])):  # any non-argmax draft is outside the support
        assert not s.accept(p, ids, other, 0.0)[0], "non-argmax must always be rejected at T->0"
        emitted, info = s.resolve_round([(vals, ids_raw)] * 2, [other], lambda: 0.0)
        assert info["accepted"] == 0 and emitted == [int(ids_raw[0])], "residual must fall back to the argmax"


def test_relaxed_is_monotone_and_more_permissive():
    """Typical acceptance must be monotone in p(x̂) and never stricter than exact-in-expectation."""
    s = SpecSampler(temperature=1.0, top_k=0, top_p=1.0, rule=RELAXED, eps=0.09, delta=0.3)
    vals, ids_raw = _cands([3.0, 2.5, 2.0, 1.0, 0.0])
    p, ids = s.probs(vals, ids_raw)
    flags = [s.accept(p, ids, int(t), 0.5)[0] for t in ids]
    assert flags == sorted(flags, reverse=True), f"not monotone in p: {flags} for p={p.tolist()}"
    assert flags[0], "the argmax must always clear the typical-acceptance threshold"


# ── the exactness argument for applying the presence penalty over CANDIDATES only ─────────────
def test_presence_penalty_over_candidates_equals_full_vocab():
    """The module claims that penalising the top-N candidates gives the same kept top-k set as
    penalising the whole vocab first. Check it directly against a full-"vocab" reference."""
    g = torch.Generator().manual_seed(3)
    V, ncand, top_k = 4000, 256, 20
    for trial in range(20):
        full = torch.randn(V, generator=g) * 4.0
        ids_full = torch.arange(V, dtype=torch.int64)
        pen = set(torch.randperm(V, generator=g)[:200].tolist())
        s = SpecSampler(temperature=0.6, top_k=top_k, top_p=1.0, presence_penalty=1.5, rule=EXACT)
        cv, ci = full.topk(ncand)  # what the device tail hands back (penalty applied host-side after)
        p_c, ids_c = s.probs(cv, ci, penalised=pen)
        p_f, ids_f = s.probs(full, ids_full, penalised=pen)
        assert ids_c.tolist() == ids_f.tolist(), f"trial {trial}: kept set differs"
        assert torch.allclose(p_c, p_f, atol=1e-6)


def test_sampler_from_env_validates_rule(expect_error):
    assert sampler_from_env(0.6, 20, 0.95, 1.5, env={}).rule == EXACT
    assert sampler_from_env(0.6, 20, 0.95, 1.5, env={"QWEN36_MTP_ACCEPT": "relaxed"}).rule == RELAXED
    # an unknown rule must fail loudly rather than silently fall back to a different distribution
    with expect_error(ValueError, "QWEN36_MTP_ACCEPT must be one of"):
        sampler_from_env(0.6, 20, 0.95, 1.5, env={"QWEN36_MTP_ACCEPT": "typical"})


# ── the tunable rule, and the divergence metric that makes the rules comparable ───────────────
@pytest.mark.parametrize("logits,draft_pos", [([5.0, 4.0, 3.0, 2.0, 1.0], 0), ([5.0, 4.0, 3.0, 2.0, 1.0], 2)])
def test_lenient_at_unit_lenience_is_exact(logits, draft_pos):
    """`lenient` is exact rejection sampling against a TEMPERED target, so lenience=1 must be exactly
    the exact rule — the property that makes it a dial rather than a separate algorithm."""
    s = SpecSampler(temperature=1.0, top_k=0, top_p=1.0, rule=LENIENT, lenience=1.0)
    vals, ids_raw = _cands(logits)
    p, ids = s.probs(vals, ids_raw)
    hist, _ = _empirical(s, [(vals, ids_raw)] * 2, [int(ids_raw[draft_pos])], seed=draft_pos)
    _assert_matches(hist, p, ids, NTRIAL, f"lenient(1.0) draft@{draft_pos}")


@pytest.mark.parametrize(
    "rule,kw", [(EXACT, {}), (RELAXED, {}), (LENIENT, {"lenience": 0.7}), (LENIENT, {"lenience": 0.3}), (GREEDY, {})]
)
def test_realized_tv_matches_simulation(rule, kw):
    """Validate the `TV = |a - p(x̂)|` algebra EMPIRICALLY, because every rule comparison rests on it.

    Builds the realized one-position distribution by simulation and compares its true TV against the
    target to the closed form. A sign slip or a missing renormalisation in the residual would show up
    here as a mismatch, where the formula alone would look plausible."""
    s = SpecSampler(temperature=1.0, top_k=0, top_p=1.0, rule=rule, **kw)
    vals, ids_raw = _cands([4.0, 2.0, 1.0, 0.0])
    p, ids = s.probs(vals, ids_raw)
    for pos in range(ids.numel()):
        draft = int(ids[pos])
        pd, a = s.accept_prob(p, ids, draft)
        hist, _ = _empirical(s, [(vals, ids_raw)] * 2, [draft], n=40_000, seed=pos)
        emp = torch.tensor([hist.get(int(t), 0) / 40_000 for t in ids])
        tv_emp = float((emp - p).abs().sum()) / 2
        tv_formula = realized_tv(a, pd)
        print(
            f"[specsamp] {rule}{kw or ''} p={pd:.3f} a={a:.3f}: TV formula {tv_formula:.4f} "
            f"vs simulated {tv_emp:.4f}"
        )
        assert abs(tv_emp - tv_formula) < 0.01, (
            f"closed-form TV {tv_formula:.4f} disagrees with the simulated {tv_emp:.4f} "
            f"(rule={rule}{kw}, p={pd:.4f}, a={a:.4f})"
        )


def test_lenience_trades_acceptance_for_divergence_monotonically():
    """Lower lenience must accept more AND diverge more, with no crossovers — that monotonicity is
    what lets an operator pick a point on the curve from a single measured sweep."""
    vals, ids_raw = _cands([4.0, 2.0, 1.0, 0.0])
    rows = []
    for lenience in (1.0, 0.7, 0.5, 0.3, 0.1):
        s = SpecSampler(temperature=1.0, top_k=0, top_p=1.0, rule=LENIENT, lenience=lenience)
        p, ids = s.probs(vals, ids_raw)
        acc = [s.accept_prob(p, ids, int(t))[1] for t in ids]
        tv = [realized_tv(a, float(pp)) for a, pp in zip(acc, p)]
        rows.append((lenience, sum(acc) / len(acc), sum(tv) / len(tv)))
        print(f"[specsamp] lenience {lenience:.1f}: mean accept {rows[-1][1]:.3f}  mean TV {rows[-1][2]:.3f}")
    accs = [r[1] for r in rows]
    tvs = [r[2] for r in rows]
    assert accs == sorted(accs), f"acceptance not monotone in lenience: {accs}"
    assert tvs == sorted(tvs), f"divergence not monotone in lenience: {tvs}"
    assert tvs[0] == pytest.approx(0.0, abs=1e-9), "lenience=1 must be divergence-free"


def test_relaxed_is_the_deterministic_extreme():
    """Relaxed is deterministic, so per position it pays the maximal divergence its decision allows
    (1-p when it accepts, p when it rejects). Recorded as a test because it is the reason a lenient
    rule at comparable acceptance diverges far less."""
    s = SpecSampler(temperature=1.0, top_k=0, top_p=1.0, rule=RELAXED)
    vals, ids_raw = _cands([4.0, 2.0, 1.0, 0.0])
    p, ids = s.probs(vals, ids_raw)
    for pos in range(ids.numel()):
        pd, a = s.accept_prob(p, ids, int(ids[pos]))
        assert a in (0.0, 1.0), f"relaxed must be deterministic, got a={a}"
        assert realized_tv(a, pd) == pytest.approx(1 - pd if a else pd, abs=1e-9)


def test_sampler_from_env_lenience():
    assert sampler_from_env(0.6, 20, 0.95, 0.0, env={"QWEN36_MTP_ACCEPT": "lenient"}).lenience == 0.5
    s = sampler_from_env(0.6, 20, 0.95, 0.0, env={"QWEN36_MTP_ACCEPT": "lenient", "QWEN36_MTP_LENIENCE": "0.8"})
    assert s.rule == LENIENT and s.lenience == 0.8
