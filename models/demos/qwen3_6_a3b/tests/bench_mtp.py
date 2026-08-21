# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Speculative-decode (MTP) benchmark on the REAL Qwen3.6-35B-A3B model.

Reports the two things that decide whether MTP pays:
  * E[tokens/round] — the ACCEPTANCE, which only means anything at the full layer count (the MTP head
    was trained against the 40-layer backbone's hidden states, so a shallow backbone gives alpha ~ 0),
  * the round cost, decomposed into draft / verify / host glue.

Both are measured EAGER here (tracing is a follow-up), so the absolute tok/s is dispatch-bound and not
the shipping number. The projected traced throughput is computed from the measured E[tokens/round] and
the Phase-A verify cost model (see MTP.md).

    QWEN36_LAYERS=40 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/bench_mtp.py --rounds 24
"""
from __future__ import annotations

import argparse
import os
import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs

CKPT = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
PROMPTS = [
    "Explain why a mixture-of-experts layer is harder to batch efficiently than a dense feed-forward "
    "layer on an accelerator with a fixed tile size.",
    "Write a Python function that merges two sorted lists without using sorted(), with a docstring.",
]

# Phase-A calibrated per-layer costs (MTP.md); used only to PROJECT traced throughput.
# MEASURED traced greedy decode step (bench_decode.py, 40L). Was 29.01 before the per-matmul MoE
# in0_block_w landed (gate_up=32); A/B'd at 29.00 with QWEN36_SPARSE_IN0BW_GU=16, 28.65 with 32.
BASE_STEP_MS = 28.65
N_MOE, N_GDN, N_ATTN = 40, 30, 10
MOE_FLAT, MOE_C0, MOE_SLOPE = 0.2165, 0.0277, 0.0818
GDN_FLAT, GDN_KERN1, GDN_DELTA = 0.2576, 0.0614, 0.0465
ATTN_MS, ATTN_D, HEAD_MS, GAP_MS, DRAFT_MS, GLUE_MS = 0.295, 0.0016, 1.226, 2.21, 2.03, 0.5
# Plain SAMPLED decode step: greedy + the measured 1.02 ms full thinking-mode tail (T=0.6 top_k=20
# top_p=0.95 presence=1.5). This is the baseline a SAMPLED speculative round has to beat. Without the
# presence penalty the tail is only ~+0.43 ms, so pass --presence-penalty 0 and compare against ~29.1.
BASE_SAMPLING_MS = 29.67


def projected_verify_ms(K):
    return (
        N_MOE * (MOE_FLAT + MOE_C0 + MOE_SLOPE * K)
        + N_GDN * (GDN_FLAT + GDN_KERN1 + GDN_DELTA * (K - 1))
        + N_ATTN * (ATTN_MS + ATTN_D * (K - 1))
        + HEAD_MS
        + GAP_MS
    )


def _make_rule(spec, temperature, top_k, top_p, presence):
    """ "lenient:0.4" -> a SpecSampler; a bare name uses that rule's defaults."""
    from models.demos.qwen3_6_a3b.tt.mtp_sampling import RULES, SpecSampler

    name, _, val = spec.partition(":")
    assert name in RULES, f"unknown rule {name!r}; expected one of {RULES}"
    kw = {"lenience": float(val)} if (name == "lenient" and val) else {}
    return SpecSampler(temperature=temperature, top_k=top_k, top_p=top_p, presence_penalty=presence, rule=name, **kw)


def sweep_rules_paired(model, tok, gamma, rounds, specs, base_ref, samp, round_ms):
    """Compare acceptance rules on IDENTICAL inputs, which is the only way to rank them cheaply.

    Running each rule as its own generation is noise-limited: every rule produces a different sequence,
    so the between-sequence variance swamps the difference between rules — two independent 200-round
    sweeps both ranked `lenient:0.7` BELOW `exact`, which is impossible in expectation (lenient accepts
    at least as often as exact at every position).

    So: run ONE generation, record the (target distribution, drafted token) pair at every draft
    position — the draft chain always runs all gamma steps and verify always scores all K rows, so
    every round yields gamma complete pairs regardless of what was accepted — then score every rule on
    that same set. Two further variance reductions:
      * use each rule's acceptance PROBABILITY a_j, not a coin flip, so
        E[tokens/round] = 1 + E[a_1] + E[a_1 a_2] + ... is Rao-Blackwellised;
      * `round_ms` is taken from a measured timed run, since the round cost is rule-INDEPENDENT
        (measured 66.1-66.8 ms across five rules).

    CAVEAT: the trajectory is the reference rule's. A rule that accepted far more would drift to
    somewhat different states, so this is a first-order (paired, off-policy) comparison — but it ranks
    the rules, which the unpaired version could not."""
    from models.demos.qwen3_6_a3b.tt.mtp_sampling import realized_tv

    model.build_mtp_head(gamma=gamma)
    model.release_mtp_traces()
    ids = _prompt(tok, PROMPTS[0])
    _prefill(model, ids, prime=_PRIME)
    model.setup_mtp_decode()
    model._mtp_hidden = None
    model.set_mtp_sampler(_make_rule("exact", *samp))  # reference trajectory: the exact rule
    model.setup_mtp_traces()
    for _ in range(3):
        model.spec_decode_step()

    # collect (candidate rows, drafts) per round
    log = []
    for _ in range(rounds):
        model.spec_decode_step()
        log.append((model._mtp_last_cands, model._mtp_last_drafts))
    print(
        f"[bench_mtp] collected {len(log)} rounds x {gamma} draft positions on the exact-rule " f"trajectory",
        flush=True,
    )

    hdr = (
        f"{'rule':>16}{'E[tok/round]':>14}{'a1':>7}{'a2':>7}{'ms/token':>10}{'tok/s':>8}"
        f"{'vs base':>9}{'mean TV':>9}{'p<0.1':>8}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    rows = []
    for spec in specs:
        sm = _make_rule(spec, *samp)
        acc_by_depth = [[] for _ in range(gamma)]
        prod_terms = [[] for _ in range(gamma)]
        tvs, low, n_pos = [], 0, 0
        for cands, drafts in log:
            run = 1.0
            for j in range(gamma):
                p, pid = sm.probs(cands[j][0], cands[j][1])
                pd, a = sm.accept_prob(p, pid, drafts[j])
                acc_by_depth[j].append(a)
                run *= a
                prod_terms[j].append(run)
                tvs.append(realized_tv(a, pd))
                low += int(a > 0.5 and pd < 0.1)  # would force-emit a token the target rated <10%
                n_pos += 1
        e_tok = 1.0 + sum(sum(t) / len(t) for t in prod_terms)
        a_mean = [sum(t) / len(t) for t in acc_by_depth]
        mspt = round_ms / e_tok
        tv = sum(tvs) / len(tvs)
        rows.append((spec, e_tok, mspt, base_ref / mspt, tv, low / max(n_pos, 1)))
        print(
            f"{spec:>16}{e_tok:>14.3f}{a_mean[0]:>7.3f}"
            f"{(a_mean[1] if gamma > 1 else float('nan')):>7.3f}{mspt:>10.2f}{1000 / mspt:>8.1f}"
            f"{base_ref / mspt:>8.2f}x{tv:>9.4f}{low / max(n_pos, 1):>8.1%}",
            flush=True,
        )
    print(f"\n  round {round_ms:.1f} ms (rule-independent), baseline {base_ref:.2f} ms/token.")
    print("  a1/a2 are mean acceptance PROBABILITIES at draft depth 1/2 on identical inputs.")
    print("  'mean TV' is the exact per-position divergence from the target (0 = unchanged sampling);")
    print("  'p<0.1' is how often the rule would emit a draft the target rated under 10%.")
    return rows


_PRIME = True  # set from --no-prime in main(); mirrors the server default (QWEN36_MTP_PRIME=1)
_PROMPT_TOKENS = 0  # set from --prompt-tokens; 0 = the prompt as written


def sweep_rules(model, tok, gamma, rounds, specs, base_ref, samp):
    """Measure every acceptance rule against ONE captured trace set.

    Reports throughput AND divergence, because a non-exact rule buys tokens/round precisely by
    emitting drafts the target rated unlikely, so the two only mean something together. `mean TV` is
    the exact per-emitted-token total-variation distance from the target distribution -- 0 for `exact`
    by construction, which also makes it an end-to-end check on the whole pipeline. See
    tt/mtp_sampling.py::realized_tv for the algebra (validated against simulation in the host tests)."""
    model.build_mtp_head(gamma=gamma)
    model.release_mtp_traces()
    ids = _prompt(tok, PROMPTS[0])
    _prefill(model, ids, prime=_PRIME)
    model.setup_mtp_decode()
    model._mtp_hidden = None
    warm = model.setup_mtp_traces()  # capture ONCE; every rule below replays it
    print(f"[bench_mtp] captured verify({model.mtp_verify_tail})/draft/commit, " f"{len(warm)} warm tokens", flush=True)

    hdr = (
        f"{'rule':>16}{'tok/round':>15}{'accept':>8}{'ms/round':>10}{'ms/token':>10}{'tok/s':>8}"
        f"{'vs base':>9}{'mean TV':>9}{'p(acc)':>8}{'p<0.1':>7}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    rows = []
    for spec in specs:
        model.set_mtp_sampler(_make_rule(spec, *samp))
        ids = _prompt(tok, PROMPTS[0])
        _prefill(model, ids, prime=_PRIME)  # fresh sequence per rule: same prefix, same head cache
        model.setup_mtp_decode()
        model._mtp_hidden = None
        model.mtp_stats = {"rounds": 0, "tokens": 0, "accepted": 0, "drafted": 0}
        for _ in range(3):
            model.spec_decode_step()  # eager seed round + settle after the re-prefill
        st0, counts = dict(model.mtp_stats), []
        ttnn.synchronize_device(model.mesh_device)
        t0 = time.time()
        for _ in range(rounds):
            counts.append(len(model.spec_decode_step()))
        dt = time.time() - t0
        st = model.mtp_stats
        emitted = sum(counts)
        tpr = emitted / rounds
        # Report the standard error: tokens/round is a mean over Bernoulli acceptances and each rule
        # generates its OWN sequence, so a short run reads as rule differences that are really noise.
        # (A 20-round first pass ranked `lenient:0.3` BELOW `exact`, which is impossible in
        # expectation — lenient accepts at least as often as exact at every position.)
        var = sum((c - tpr) ** 2 for c in counts) / max(rounds - 1, 1)
        se = (var / rounds) ** 0.5
        acc = (st["accepted"] - st0["accepted"]) / max(st["drafted"] - st0["drafted"], 1)
        ms_round = dt / rounds * 1e3
        mspt = ms_round / tpr
        tv, p_acc, p_low = model.mtp_divergence() or (float("nan"),) * 3
        rows.append((spec, tpr, se, ms_round, mspt, base_ref / mspt, tv, p_acc, p_low))
        print(
            f"{spec:>16}{tpr:>9.3f}+-{se:<4.3f}{acc:>8.2f}{ms_round:>10.1f}{mspt:>10.2f}"
            f"{1000 / mspt:>8.1f}{base_ref / mspt:>8.2f}x{tv:>9.4f}{p_acc:>8.3f}{p_low:>7.1%}",
            flush=True,
        )
    print(f"\n  baseline {base_ref:.2f} ms/token. 'mean TV' is the exact per-emitted-token divergence")
    print("  from the target distribution: 0 = the model's own sampling distribution, unchanged.")
    print("  'p(acc)' is the target's mean probability for the drafts the rule ACCEPTED; 'p<0.1' is how")
    print("  often it force-emitted a token the target rated under 10%.")
    return rows


def sweep_prime(model, tok, gamma, rounds, base_ref):
    """PAIRED A/B of MTP-head priming (TtModel.prime_mtp): same model, same captured traces, same
    prompt -- priming is the only difference.

    Paired because the earlier unpaired rule sweep taught the lesson the hard way: tokens/round is a
    mean over Bernoulli acceptances, and a short window of it is wide enough to invert a real ordering.

    ABBA-scheduled (cold, primed, primed, cold) because a straight A-then-B run cannot separate the
    effect from any monotone drift across the run, and the first two attempts at this measurement --
    both A-then-B -- put priming 0.11-0.13 tok/round BEHIND, which is exactly the size an ordering
    artefact could manufacture. Two blocks per arm in mirrored order cancels a linear trend, and the
    per-block numbers are printed so a nonlinear one is visible rather than averaged away.

    Priming cannot change WHAT is emitted (a rejected draft is discarded and the token comes from the
    backbone's own distribution), so tokens/round is the whole of the effect."""
    model.build_mtp_head(gamma=gamma)
    model.release_mtp_traces()
    ids = _prompt(tok, PROMPTS[0])
    _prefill(model, ids, prime=False)
    model.setup_mtp_decode()
    model._mtp_hidden = None
    warm = model.setup_mtp_traces()  # ONE trace set, replayed by every block
    print(f"\n[bench_mtp] prime A/B: captured verify({model.mtp_verify_tail})/draft/commit, {len(warm)} warm tokens")

    SCHEDULE = (False, True, True, False)
    per_block = max(1, rounds // 2)  # each arm gets two blocks, so 2*per_block rounds per arm
    hdr = f"{'block':>7}{'head cache':>12}{'primed':>8}{'tok/round':>15}{'accept':>8}{'ms/round':>10}{'ms/token':>10}"
    print(hdr)
    print("-" * len(hdr))
    acc, msr = {False: [], True: []}, {False: [], True: []}
    for b, prime in enumerate(SCHEDULE):
        ids = _prompt(tok, PROMPTS[0])
        n_primed = _prefill(model, ids, prime=prime)
        model.setup_mtp_decode()
        model._mtp_hidden = None
        model.mtp_stats = {"rounds": 0, "tokens": 0, "accepted": 0, "drafted": 0}
        for _ in range(3):
            model.spec_decode_step()  # eager seed round + settle after the re-prefill
        st0, counts = dict(model.mtp_stats), []
        ttnn.synchronize_device(model.mesh_device)
        t0 = time.time()
        for _ in range(per_block):
            counts.append(len(model.spec_decode_step()))
        dt = time.time() - t0
        st = model.mtp_stats
        tpr = sum(counts) / per_block
        acc[prime].extend(counts)
        ms_round = dt / per_block * 1e3
        msr[prime].append(ms_round)
        a = (st["accepted"] - st0["accepted"]) / max(st["drafted"] - st0["drafted"], 1)
        print(
            f"{b:>7}{'primed' if prime else 'cold':>12}{n_primed:>8}{tpr:>11.3f}    {a:>8.2f}"
            f"{ms_round:>10.1f}{ms_round / tpr:>10.2f}",
            flush=True,
        )

    out = {}
    print()
    for prime in (False, True):
        c = acc[prime]
        m = sum(c) / len(c)
        var = sum((x - m) ** 2 for x in c) / max(len(c) - 1, 1)
        se = (var / len(c)) ** 0.5
        ms_round = sum(msr[prime]) / len(msr[prime])
        out[prime] = (m, se, ms_round / m)
        print(
            f"  {'primed' if prime else 'cold':>6}: {m:.3f}+-{se:.3f} tok/round over {len(c)} rounds"
            f"  ({ms_round:.1f} ms/round -> {ms_round / m:.2f} ms/token, {base_ref / (ms_round / m):.2f}x base)"
        )
    (m0, se0, t0_), (m1, se1, t1) = out[False], out[True]
    d, sed = m1 - m0, (se0**2 + se1**2) ** 0.5
    print(
        f"\n  delta {d:+.3f} tok/round +- {sed:.3f} ({abs(d) / sed if sed else 0:.1f} sigma), {t0_ / t1:.3f}x on ms/token"
    )
    # The +- above pools rounds as if independent, which they are not: acceptance drifts with sequence
    # position, so BLOCK-to-block spread runs wider than round-level noise predicts (measured: the two
    # cold blocks came in at 2.424 and 2.792, a spread as large as the effect). Print it, so nobody
    # reads the pooled sigma as the real confidence -- the trustworthy signal is the SIGN agreeing
    # across runs with different prompt lengths, round counts and orderings.
    blocks = {k: [sum(acc[k][i::2]) / len(acc[k][i::2]) for i in range(2)] for k in (False, True)}
    print(
        f"  block spread: cold {blocks[False][0]:.3f}/{blocks[False][1]:.3f}, primed {blocks[True][0]:.3f}/{blocks[True][1]:.3f}"
    )
    print("  Priming is throughput-only: it cannot change which tokens are emitted, only how many a")
    print("  round gets to skip. Trust the sign across runs before the magnitude of any one.")
    return out


def _prefill(model, ids, prime=True):
    """Prefill + start_decode, priming the MTP head's KV cache when asked (the server's default).

    Priming needs the prefill to keep its per-position post-final-norm hidden, which is opted into
    BEFORE the prefill runs -- so every measured path has to go through here or it silently measures a
    cold head cache. Returns the number of primed rows (0 = not primed)."""
    model._keep_prefill_hidden = bool(prime) and getattr(model, "mtp", None) is not None
    lg = model.forward(ids)
    first = int(lg[0, -1].argmax())
    model.start_decode(first)
    n = model.prime_mtp(ids, first) if model._keep_prefill_hidden else 0
    model._keep_prefill_hidden = False
    model._prefill_hidden = None
    return n


def _prompt(tok, text):
    if _PROMPT_TOKENS:
        # Repeat the question until the chat-templated prompt is about _PROMPT_TOKENS long. Crude, but
        # it lengthens the CONTEXT without changing what is being asked, which is what priming acts on.
        n = max(1, _PROMPT_TOKENS // max(len(tok.encode(text)), 1))
        text = " ".join([text] * n)
    enc = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    if not torch.is_tensor(ids):
        ids = torch.tensor(ids)
    return ids.reshape(1, -1).to(torch.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--gammas", type=str, default="1,2,3")
    ap.add_argument("--baseline-steps", type=int, default=16)
    # Sampling flags. With --temperature>0 BOTH the baseline decode and the speculative rounds run the
    # sampling path, so the reported speedup is against the right baseline: plain sampled decode costs
    # ~1 ms/token more than greedy (SAMPLING.md) and a speculative round pays a per-row candidate topk,
    # so comparing sampled MTP against the GREEDY baseline would flatter it.
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; >0 = speculative SAMPLING")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--no-prime",
        action="store_true",
        help="drop MTP-head priming (TtModel.prime_mtp), i.e. draft from a cold head KV cache",
    )
    ap.add_argument(
        "--prime-ab",
        action="store_true",
        help="paired A/B of priming on vs off, sharing one model + one captured trace set",
    )
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="lengthen the prompt to about this many tokens (repeats the prompt text). Priming scales "
        "with prompt length, so a 40-token prompt is the weakest possible test of it",
    )
    # Acceptance-rule sweep. The captured verify trace is rule-INDEPENDENT (it only emits candidates),
    # so every rule is measured in ONE process against ONE capture: no rebuild, no re-capture, and the
    # numbers are directly comparable. Entries: exact | relaxed | greedy | lenient[:VALUE].
    ap.add_argument(
        "--rules",
        type=str,
        default="",
        help="comma-separated acceptance rules to sweep, e.g. exact,relaxed,lenient:0.5",
    )
    a = ap.parse_args()
    global _PRIME, _PROMPT_TOKENS
    _PRIME = not a.no_prime
    _PROMPT_TOKENS = a.prompt_tokens

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)
    n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        args = ModelArgs(mesh, ckpt_dir=CKPT, max_seq_len=int(os.environ.get("QWEN36_MAX_SEQ", "512")))
        loader = CheckpointLoader(CKPT)
        assert loader.has_mtp(), "checkpoint has no mtp.* head"
        t0 = time.time()
        model = TtModel(mesh, args, loader, num_layers=n_layers)
        print(f"[bench_mtp] built {n_layers}-layer model in {time.time()-t0:.0f}s", flush=True)
        model.build_mtp_head(gamma=1)
        print(f"[bench_mtp] built MTP head", flush=True)
        base_ref = BASE_STEP_MS
        if a.temperature > 0:
            model.enable_sampling(a.temperature, a.top_k, a.top_p, a.seed, a.presence_penalty)
            base_ref = BASE_SAMPLING_MS
            print(
                f"[bench_mtp] SAMPLING: T={a.temperature} top_k={a.top_k} top_p={a.top_p} "
                f"presence={a.presence_penalty} seed={a.seed} "
                f"rule={os.environ.get('QWEN36_MTP_ACCEPT', 'exact')}",
                flush=True,
            )

        # ---- eager baseline decode, for an apples-to-apples eager comparison ----
        ids = _prompt(tok, PROMPTS[0])
        _prefill(model, ids, prime=_PRIME)
        model.decode_step_eager()  # warm/compile
        t0 = time.time()
        for _ in range(a.baseline_steps):
            model.decode_step_eager()
        base_ms = (time.time() - t0) / a.baseline_steps * 1e3
        print(f"\n  eager baseline decode: {base_ms:.1f} ms/token ({1000/base_ms:.1f} tok/s)")
        print(f"  (traced baseline for reference: {BASE_STEP_MS:.2f} ms/token " f"= {1000/BASE_STEP_MS:.1f} tok/s)\n")

        if a.prime_ab:
            for gamma in [int(g) for g in a.gammas.split(",")]:
                print(f"\n===== gamma={gamma} (K={gamma + 1}) =====", flush=True)
                sweep_prime(model, tok, gamma, a.rounds, base_ref)
            return

        if a.rules:
            specs = [r.strip() for r in a.rules.split(",") if r.strip()]
            samp = (a.temperature, a.top_k, a.top_p, a.presence_penalty)
            assert a.temperature > 0, "--rules needs --temperature>0 (the rules only differ when sampling)"
            for gamma in [int(g) for g in a.gammas.split(",")]:
                print(f"\n===== gamma={gamma} (K={gamma + 1}) =====", flush=True)
                measured = sweep_rules(model, tok, gamma, a.rounds, specs, base_ref, samp)
                # round cost is rule-independent, so reuse the measured mean for the paired ranking
                round_ms = sum(r[3] for r in measured) / len(measured)
                sweep_rules_paired(model, tok, gamma, a.rounds, specs, base_ref, samp, round_ms)
            return

        hdr = (
            f"{'gamma':>6}{'K':>3}{'mode':>8}{'tok/round':>11}{'accept':>8}"
            f"{'ms/round':>10}{'ms/token':>10}{'tok/s':>8}{'vs base':>9}{'proj':>8}"
        )
        print(hdr)
        print("-" * len(hdr))
        for gamma in [int(g) for g in a.gammas.split(",")]:
            model.build_mtp_head(gamma=gamma)
            streams = {}
            for mode in ("eager", "verify", "full"):
                model.release_mtp_traces()  # each mode starts from no traces (and never leaks one)
                ids = _prompt(tok, PROMPTS[0])
                _prefill(model, ids, prime=_PRIME)
                model._mtp_hidden = None
                # Same number of warm rounds in EVERY mode: the traced modes need two eager rounds
                # before capture (K=1 seed, then a K-row verify so `_verify_pending` holds the K-shaped
                # scratch), so eager must burn the same two or the timed streams start at different
                # positions and the parity check compares offset sequences.
                model.setup_mtp_decode()
                model.spec_decode_step()
                model.spec_decode_step()
                if mode != "eager":
                    # the pending token is tracked in PYTHON by the spec loop (t_tok is written only by
                    # the plain decode path), so prime capture from _mtp_cur rather than a stale t_tok
                    cur = int(model._mtp_cur)
                    positions = [model.pos + i for i in range(gamma + 1)]
                    model.capture_mtp_verify_trace([cur] * (gamma + 1), positions)
                    if mode == "full":
                        model._mtp_write_round([cur] * (gamma + 1), positions)
                        model.capture_mtp_draft_trace()
                        model.capture_mtp_commit_traces()
                model.spec_decode_step()  # warm
                st0 = dict(model.mtp_stats)
                t0 = time.time()
                emitted, stream = 0, []
                for _ in range(a.rounds):
                    e = model.spec_decode_step()
                    emitted += len(e)
                    stream.extend(e)
                dt = time.time() - t0
                streams[mode] = stream
                st = model.mtp_stats
                tpr = emitted / a.rounds
                acc = (st["accepted"] - st0["accepted"]) / max(st["drafted"] - st0["drafted"], 1)
                ms_round = dt / a.rounds * 1e3
                mspt = ms_round / tpr
                proj = (projected_verify_ms(gamma + 1) + gamma * DRAFT_MS + GLUE_MS) / tpr
                print(
                    f"{gamma:>6}{gamma+1:>3}{mode:>8}{tpr:>11.3f}{acc:>8.2f}{ms_round:>10.1f}"
                    f"{mspt:>10.2f}{1000/mspt:>8.1f}{base_ref/mspt:>8.2f}x"
                    f"{base_ref/proj:>7.2f}x",
                    flush=True,
                )
                _, _, pacc = model.mtp_acceptance()
                if pacc is not None:  # sampled mode: how peaked the target actually was
                    print(
                        f"{'':>17}mean p(draft) {pacc:.4f}   bonus rounds "
                        f"{model.mtp_stats.get('bonus', 0)}   rejections {model.mtp_stats.get('fallbacks', 0)}",
                        flush=True,
                    )
            # parity: every trace is bit-exact vs its eager counterpart, so the token STREAMS must
            # match. A wrong commit trace (the accept-count rollback) shows up here and nowhere else.
            for m in ("verify", "full"):
                a_, b_ = streams["eager"], streams.get(m, [])
                n_ = min(len(a_), len(b_))
                div = next((i for i in range(n_) if a_[i] != b_[i]), n_)
                print(
                    f"         parity eager vs {m:>6}: identical for {div}/{n_} tokens"
                    f"{'' if div == n_ else '  <-- MISMATCH'}",
                    flush=True,
                )
        # ---- decompose the traced round: replay each trace in isolation ----
        if model.mtp_verify_trace is not None:

            def rep(tid, iters=30):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                t0 = time.time()
                for _ in range(iters):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                return (time.time() - t0) / iters * 1e3

            v = rep(model.mtp_verify_trace)
            d = rep(model.mtp_draft_trace) if model.mtp_draft_trace else float("nan")
            c = rep(model.mtp_commit_traces[1]) if model.mtp_commit_traces else float("nan")
            print(
                f"\n  per-trace replay (pure device):  verify {v:.2f} ms   draft(gamma) {d:.2f} ms"
                f"   commit {c:.2f} ms   sum {v+d+c:.2f} ms"
            )
            print(
                f"  cost-model verify({gamma+1}) = {projected_verify_ms(gamma+1):.2f} ms,"
                f" draft = {gamma*DRAFT_MS:.2f} ms -> model round {projected_verify_ms(gamma+1)+gamma*DRAFT_MS+GLUE_MS:.2f} ms"
            )

        print(f"\n  'vs base' is measured against the {base_ref:.2f} ms traced baseline step; 'proj' is the")
        print("  Phase-A cost model ceiling (fully traced round: draft + verify + commit all captured).")
        print("  Only VERIFY is traced here — draft and commit are still eager, so the gap between the")
        print("  measured and projected columns is what capturing those two would recover.")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
