# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-side acceptance rules for MTP speculative decode — the sampling counterpart of the greedy
exact-match rule.

Deliberately contains NO ttnn: everything here is pure torch over a handful of candidate values, so
the distribution-exactness argument below is unit-testable on a laptop
(``tests/test_mtp_sampling_rule.py``) instead of only on a board.

WHY A POINT-MASS DRAFT MAKES THIS EASY
--------------------------------------
``TtModel._mtp_draft_graph`` picks each draft token by ``argmax``, so the draft distribution q is a
point mass at x̂. Standard speculative sampling accepts with ``min(1, p(x̂)/q(x̂))``, which collapses to
``p(x̂)``, and its rejection residual ``norm((p-q)+)`` collapses to *p with x̂ removed and
renormalised*. The emitted stream is then EXACTLY p-distributed:

    P[emit x̂]    = p(x̂)
    P[emit y!=x̂] = (1 - p(x̂)) * p(y)/(1 - p(x̂)) = p(y)

so no change to the draft graph is needed — only the target distribution p per verify row.

WHAT p IS HERE
--------------
p is the *truncated* sampling distribution, built from the top-``NCAND`` candidates the verify pass
returns and transformed exactly as the non-speculative decode tail transforms the full vocab
(``TtModel._select_token`` -> ``ttnn.sampling``): presence penalty, temperature, top-k, top-p,
renormalise. Restricting to candidates is EXACT rather than approximate: with ``ncand`` candidates and
``top_k`` kept, if at most ``ncand - top_k`` of them are penalised then at least ``top_k`` unpenalised
candidates survive whose penalised logits equal their raw logits, each >= the raw logit of any token
outside the candidate set >= that token's penalised logit. So the kept top-k set is identical to
penalising the whole vocab first. (256 candidates, top_k <= 32 -> the condition is never close.)

It is in fact *more* faithful than the device tail: the penalty set can include tokens accepted
earlier in the SAME speculative round, which a per-step device mask cannot express.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Acceptance rules. "greedy" is the pre-existing exact-match rule, kept here so they all share one
# code path (and one captured verify trace).
EXACT, RELAXED, LENIENT, GREEDY = "exact", "relaxed", "lenient", "greedy"
RULES = (EXACT, RELAXED, LENIENT, GREEDY)


def realized_tv(accept_prob, p_draft):
    """Total-variation distance between what a rule actually emits at one position and the target.

    For a point-mass draft the realized distribution is
        r = a*delta_x + (1-a)*norm(p minus x)   with a = the rule's acceptance PROBABILITY,
    and expanding sum|r - p| gives, for any a >= p(x),

        TV(r, p) = |a - p(x)|

    which is exact, needs no simulation, and makes the rules directly comparable on one axis:
      * EXACT sets a = p(x) and so has TV = 0 at every position — the algebraic statement of why
        rejection sampling is distribution-preserving;
      * RELAXED is deterministic (a in {0,1}), so it pays TV = 1-p(x) on every ACCEPTED token and
        TV = p(x) on every rejected one — i.e. accepting a draft the target rated 0.1 is a 0.9
        divergence at that position;
      * LENIENT interpolates: a = min(1, p(x)/lenience) gives TV = a - p(x), tunable to 0.

    CAVEAT: this is a LOCAL (per-position) measure. Once a distorted token is emitted the sequence
    conditions on it, so sequence-level divergence compounds and is bounded below, not given, by the
    mean of these."""
    return abs(float(accept_prob) - float(p_draft))


@dataclass
class SpecSampler:
    """The per-request sampling configuration plus the acceptance rule.

    temperature/top_k/top_p/presence_penalty mirror ``TtModel.enable_sampling``. ``rule``:
      * ``exact``   — rejection sampling; the emitted stream is exactly the target distribution.
      * ``relaxed`` — Medusa-style typical acceptance (accept when ``p(x̂) >= min(eps, delta*e^-H(p))``).
                      NOT distribution-preserving, and deterministic, so it pays the LARGEST possible
                      divergence per position (see ``realized_tv``).
      * ``lenient`` — exact rejection sampling against a *tempered* target: accept with probability
                      ``min(1, p(x̂)/lenience)``. ``lenience=1`` IS exact; smaller values buy acceptance
                      at a divergence of exactly ``min(1, p/lenience) - p`` per position. A tunable
                      point on the speed/exactness curve rather than a cliff.
      * ``greedy``  — accept iff the draft is the target's argmax. For parity testing against the
                      greedy path only: on a SAMPLED request it biases hard toward the argmax.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    rule: str = EXACT
    eps: float = 0.09
    delta: float = 0.3
    lenience: float = 0.5

    # ---------------------------------------------------------------- distribution
    def probs(self, vals, idxs, penalised=()):
        """Candidate (vals, idxs) -> the truncated target distribution (p [M], ids [M]), descending.

        vals/idxs are 1-D over the candidates of ONE verify row; ``penalised`` is a container of token
        ids to subtract ``presence_penalty`` from (applied BEFORE truncation — see the module note on
        why that is exact)."""
        lg = torch.as_tensor(vals, dtype=torch.float32).clone()
        ids = torch.as_tensor(idxs, dtype=torch.int64)
        if self.presence_penalty > 0 and len(penalised):
            hit = torch.zeros_like(lg, dtype=torch.bool)
            pen = set(int(t) for t in penalised)
            for j, t in enumerate(ids.tolist()):
                if t in pen:
                    hit[j] = True
            lg[hit] -= float(self.presence_penalty)
        lg = lg / max(float(self.temperature), 1e-6)
        order = torch.argsort(lg, descending=True)
        lg, ids = lg[order], ids[order]
        k = self.top_k
        if k and 0 < k < lg.numel():
            lg, ids = lg[:k], ids[:k]
        p = torch.softmax(lg, dim=-1)
        if 0.0 < self.top_p < 1.0:
            keep = int((p.cumsum(-1) < self.top_p).sum()) + 1  # nucleus; always keep the top-1
            p, ids = p[:keep], ids[:keep]
            p = p / p.sum()
        return p, ids

    # ---------------------------------------------------------------- accept / sample
    @staticmethod
    def mass(p, ids, tok):
        """p(tok), or 0.0 when tok fell outside the truncated support (a legitimate hard reject)."""
        hit = (ids == int(tok)).nonzero()
        return float(p[hit[0, 0]]) if hit.numel() else 0.0

    def accept_prob(self, p, ids, draft):
        """(p(draft), the rule's ACCEPTANCE PROBABILITY for it).

        Splitting this out is what makes the divergence measurable: `realized_tv` needs the acceptance
        probability the rule uses, not just the coin's outcome."""
        pd = self.mass(p, ids, draft)
        if self.rule == GREEDY:
            return pd, float(int(draft) == int(ids[0]))
        if self.rule == RELAXED:
            H = float(-(p * p.clamp_min(1e-12).log()).sum())
            return pd, float(pd >= min(self.eps, self.delta * math.exp(-H)))
        if self.rule == LENIENT:
            return pd, min(1.0, pd / max(self.lenience, 1e-9))
        return pd, pd  # EXACT: accept x̂ with probability exactly p(x̂)

    def accept(self, p, ids, draft, u):
        """Is `draft` accepted? `u` is a uniform(0,1) draw (unused by the deterministic rules).

        Returns (accepted, p_draft, accept_prob)."""
        pd, a = self.accept_prob(p, ids, draft)
        return (u < a if 0.0 < a < 1.0 else a >= 1.0), pd, a

    def sample(self, p, ids, u, exclude=None):
        """Inverse-CDF sample from (p, ids), optionally from the REJECTION RESIDUAL (p with `exclude`
        removed and renormalised). Deterministic in `u`, which is what makes the rule testable.

        Degenerate guard: if removing `exclude` empties the support (only reachable when p(exclude)
        rounds to 1.0 yet u still exceeded it), fall back to the best candidate that is not
        `exclude` — a measure-zero case that must not raise."""
        if exclude is not None:
            keep = ids != int(exclude)
            if not bool(keep.any()):
                return int(ids[0])
            p, ids = p[keep], ids[keep]
            tot = float(p.sum())
            if tot <= 0.0:
                return int(ids[0])
            p = p / tot
        c = torch.cumsum(p, dim=-1)
        j = int(torch.searchsorted(c, torch.tensor(float(u) * float(c[-1]))))
        return int(ids[min(j, ids.numel() - 1)])

    # ---------------------------------------------------------------- one whole round
    def resolve_round(self, cands, drafts, uniforms, generated=()):
        """Decide a whole speculative round. Pure: no device, no RNG state.

        cands:    K = gamma+1 candidate pairs (vals, idxs), one per verify row. Row i is the target
                  distribution for the token FOLLOWING the first i+1 verify inputs, i.e. the row that
                  draft i+1 is trying to predict; row gamma is the bonus row.
        drafts:   the gamma drafted token ids.
        uniforms: a callable returning a fresh uniform(0,1) — one per accept test and one per sample.
        generated: token ids already emitted in this SEQUENCE (for the presence penalty).

        Returns (emitted, info). ``len(emitted)`` is 1..K and is exactly the count to pass to
        ``commit_verify``: verify consumed inputs [cur, d_1..d_gamma], so accepting the first m drafts
        and emitting a residual/bonus token means the backbone consumed m+1 inputs and produced m+1
        new tokens — the same relation the greedy rule has."""
        gamma = len(drafts)
        assert len(cands) == gamma + 1, (len(cands), gamma)
        emitted, probs, tvs, seen = [], [], [], list(generated)
        for i in range(gamma):
            p, ids = self.probs(cands[i][0], cands[i][1], penalised=seen + emitted)
            ok, pd, a = self.accept(p, ids, drafts[i], uniforms())
            probs.append(pd)
            tvs.append(realized_tv(a, pd))  # exact per-position divergence from the target
            if ok:
                emitted.append(int(drafts[i]))
                continue
            emitted.append(self.sample(p, ids, uniforms(), exclude=drafts[i]))
            return emitted, {"accepted": len(emitted) - 1, "bonus": False, "accept_probs": probs, "tvs": tvs}
        p, ids = self.probs(cands[gamma][0], cands[gamma][1], penalised=seen + emitted)
        emitted.append(self.sample(p, ids, uniforms()))  # all drafts accepted -> free bonus token
        return emitted, {"accepted": gamma, "bonus": True, "accept_probs": probs, "tvs": tvs}


def sampler_from_env(temperature, top_k, top_p, presence_penalty, env=None):
    """Build a SpecSampler from the request's sampling params + QWEN36_MTP_ACCEPT/_RELAX_* env."""
    import os

    e = env if env is not None else os.environ
    rule = e.get("QWEN36_MTP_ACCEPT", EXACT).lower()
    if rule not in RULES:
        raise ValueError(f"QWEN36_MTP_ACCEPT must be one of {'/'.join(RULES)}, got {rule!r}")
    return SpecSampler(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        presence_penalty=presence_penalty,
        rule=rule,
        eps=float(e.get("QWEN36_MTP_RELAX_EPS", "0.09")),
        delta=float(e.get("QWEN36_MTP_RELAX_DELTA", "0.3")),
        lenience=float(e.get("QWEN36_MTP_LENIENCE", "0.5")),
    )
