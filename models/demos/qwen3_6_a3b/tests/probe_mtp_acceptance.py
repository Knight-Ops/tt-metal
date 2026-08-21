# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Phase-A5: measure the MTP head's ACCEPTANCE RATE alpha on real prompts.

The cheapest possible kill for MTP, and the gate both prior in-tree attempts skipped: no matter how
fast verify gets, speculative decoding cannot pay unless the draft head is actually right most of the
time. Also resolves the `mtp.fc` CONCAT ORDER empirically (the two prior attempts disagreed; the
wrong order multiplies the operands by the wrong half of a [2048,4096] matrix -> garbage).

Two stages:
  capture  (device) build the real 40L model, prefill a real prompt, then decode N tokens EAGERLY,
           recording the post-final-norm hidden h_i and the greedy token t_{i+1} at every step.
           Saves a .pt bundle. This is the only part that needs the board.
  score    (host)   load the 19 `mtp.*` tensors + the SHARED embed_tokens/lm_head, run the reference
           Qwen35MoeMTP teacher-forced over the captured sequence, and report
               alpha = P[ argmax(lm_head(MTP(h_i, embed(t_{i+1})))) == t_{i+2} ]
           for BOTH concat orders. The head's own attention builds its KV over the captured sequence,
           which is what it would do incrementally during real speculative decoding.

Run:
    QWEN36_LAYERS=40 ./python_env/bin/python .../probe_mtp_acceptance.py capture --gen 300
    ./python_env/bin/python .../probe_mtp_acceptance.py score        # host-only, no board
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

CKPT = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
OUT = Path(os.environ.get("QWEN36_MTP_CAPTURE", "/tmp/qwen36_mtp_capture.pt"))

PROMPTS = [
    "Explain why a mixture-of-experts layer is harder to batch efficiently than a dense feed-forward "
    "layer on an accelerator with a fixed tile size. Be specific about where the inefficiency comes from.",
    "Write a Python function that merges two sorted lists into one sorted list without using sorted(). "
    "Include a short docstring and one example.",
    "A train leaves at 3pm travelling 60 mph. A second train leaves the same station at 4pm travelling "
    "80 mph on the same track. At what time does the second train catch the first? Show your reasoning.",
]


def _decode_hidden_both(model):
    """`TtModel._decode_hidden` (tt/model.py:702), returning (post_final_norm, pre_final_norm).
    Kept as a probe-local copy so the model is not modified during the measurement gate."""
    import ttnn

    B = model.t_tok.shape[0]
    cos, sin = model._rope_for(model.t_ropepos)
    x = ttnn.embedding(model.t_tok, model.embed_weight, layout=ttnn.TILE_LAYOUT)
    x = ttnn.reshape(x, [1, 1, B, model.args.dim])
    for layer, cache in zip(model.layers, model.caches):
        x = layer.forward_decode(x, cos, sin, cache, model.t_curpos)
    pre = ttnn.reshape(x, [B, model.args.dim])
    post = ttnn.reshape(model.final_norm.forward(x), [B, model.args.dim])
    return post, pre


# ============================================================ capture (device)
def capture(gen: int):
    import ttnn
    from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
    from models.demos.qwen3_6_a3b.tt.model import TtModel
    from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs

    n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))
    max_seq = int(os.environ.get("QWEN36_MAX_SEQ", "1024"))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        args = ModelArgs(mesh, ckpt_dir=CKPT, max_seq_len=max_seq)
        model = TtModel(mesh, args, CheckpointLoader(CKPT), num_layers=n_layers)
        print(f"[a5] built {n_layers}-layer model", flush=True)

        runs = []
        for pi, text in enumerate(PROMPTS):
            msgs = [{"role": "user", "content": text}]
            try:
                enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            except Exception:
                enc = tok(text, return_tensors="pt")
            # transformers returns a bare tensor, a list, or a BatchEncoding depending on version
            ids = enc["input_ids"] if hasattr(enc, "keys") else enc
            if not torch.is_tensor(ids):
                ids = torch.tensor(ids)
            ids = ids.reshape(1, -1).to(torch.int64)
            logits = model.forward(ids)
            first = int(logits[0, -1].argmax())
            model.start_decode(first)

            # tokens[i] is the input token consumed at decode step i; hidden[i] is h_i, the state
            # that produced tokens[i+1]. We capture BOTH the pre- and post-final-norm hidden: the MTP
            # head carries its own `pre_fc_norm_hidden`, which implies it consumes the RAW last-layer
            # output (DeepSeek-V3's MTP does exactly that), but `_decode_hidden()` returns the
            # post-final-norm state. Which one is right is an empirical question, so measure both.
            toks, hids, hids_pre = [first], [], []
            for _ in range(gen):
                x, xpre = _decode_hidden_both(model)  # advances mixer state; positions unchanged
                hids.append(ttnn.to_torch(x).reshape(-1).clone().float())
                hids_pre.append(ttnn.to_torch(xpre).reshape(-1).clone().float())
                lg = ttnn.linear(x, model.lm_head_w)
                nxt = int(ttnn.to_torch(lg).reshape(-1).argmax())
                ttnn.plus_one(model.t_curpos)
                ttnn.plus_one(model.t_ropepos)
                model.pos += 1
                model.set_decode_tokens(nxt)
                toks.append(nxt)
            runs.append(
                {
                    "prompt_idx": pi,
                    "prompt_ids": ids[0].tolist(),
                    "tokens": toks,  # len gen+1
                    "hidden": torch.stack(hids),  # [gen, dim]  post-final-norm
                    "hidden_pre": torch.stack(hids_pre),  # [gen, dim]  pre-final-norm (raw layer out)
                    "text": tok.decode(toks, skip_special_tokens=True),
                }
            )
            print(f"[a5] prompt {pi}: {gen} tokens captured | {runs[-1]['text'][:90]!r}", flush=True)

        torch.save({"runs": runs, "n_layers": n_layers, "ckpt": CKPT}, OUT)
        print(f"[a5] wrote {OUT}")
    finally:
        ttnn.close_mesh_device(mesh)


# ============================================================ score (host)
def _load_mtp_state():
    """The 19 `mtp.*` tensors, renamed to Qwen35MoeMTP state_dict keys (strip the prefix). Left in
    their stored bfloat16 -- this box has 15 GB of RAM and fp32 experts alone are 3.2 GB."""
    from safetensors import safe_open

    base = Path(CKPT)
    wmap = json.load(open(base / "model.safetensors.index.json"))["weight_map"]
    want = {k: v for k, v in wmap.items() if k.startswith("mtp.")}
    out = {}
    for shard in sorted(set(want.values())):
        with safe_open(base / shard, framework="pt") as f:
            for k in [k for k, v in want.items() if v == shard]:
                out[k[len("mtp.") :]] = f.get_tensor(k)
    return out


def _embed_rows(token_ids):
    """Only the embedding ROWS we need. The full table is 248320x2048 (1 GB bf16) and we touch a few
    hundred rows, so slice them out of the shard instead of materialising it."""
    from safetensors import safe_open

    base = Path(CKPT)
    wmap = json.load(open(base / "model.safetensors.index.json"))["weight_map"]
    name = "model.language_model.embed_tokens.weight"
    with safe_open(base / wmap[name], framework="pt") as f:
        sl = f.get_slice(name)
        return torch.stack([sl[int(t) : int(t) + 1][0] for t in token_ids])


def _lm_head():
    from safetensors import safe_open

    base = Path(CKPT)
    wmap = json.load(open(base / "model.safetensors.index.json"))["weight_map"]
    with safe_open(base / wmap["lm_head.weight"], framework="pt") as f:
        return f.get_tensor("lm_head.weight")  # bf16 [vocab, hidden], ~1 GB


# Candidate-set width. The device sampling tail keeps nc*SAMP_K = 8*32 = 256 candidates per row
# (tt/model.py: SAMP_CHUNK/SAMP_K), and the truncated distribution is provably a subset of them for
# any top_k <= 32, so scoring host-side over 256 candidates is EXACT, not an approximation.
NCAND = 256

# (temperature, top_k, top_p, presence_penalty) configs to score. The first is demo/server.py's
# default "thinking mode". The presence penalty matters a LOT here and was missing from the first
# version of this probe: it subtracts from exactly the tokens the model already used, i.e. usually the
# high-probability ones, so it FLATTENS the truncated target — which is what sampled speculative
# acceptance depends on. Measured on device: at presence 1.5 the acceptance drop vs greedy is ~0.20,
# at presence 0 it is ~0.02. Configs kept with and without it so the two are directly comparable.
SAMP_CONFIGS = [
    (0.6, 20, 0.95, 1.5),  # demo/server.py default
    (0.6, 20, 0.95, 0.0),
    (0.7, 20, 0.8, 1.5),
    (1.0, 20, 0.95, 0.0),
]
RELAX_EPS, RELAX_DELTA = 0.09, 0.3  # Medusa typical-acceptance defaults


def _topn_vocab(h, lm_w, n=1, chunk=16384):
    """top-`n` over vocab of h @ lm_w.T, chunked over vocab so [S, 248320] is never materialised.
    Returns (values [S, n] fp32, indices [S, n] int64), descending.

    Each weight chunk is cast to fp32 for the GEMM: torch CPU has no fast bf16 matmul kernel, so a
    bf16 @ bf16 here is ~20x slower than casting a 16384x2048 slice (134 MB) and using the fp32 BLAS
    path. Peak stays ~150 MB above the resident bf16 lm_head. Batch S rows in ONE call where possible
    (BLAS is far more efficient per row) — the per-position loop is what makes this probe slow."""
    hf = h.float()
    S = hf.shape[0]
    best_v = torch.full((S, n), float("-inf"), dtype=torch.float32)
    best_i = torch.zeros((S, n), dtype=torch.long)
    for s in range(0, lm_w.shape[0], chunk):
        part = hf @ lm_w[s : s + chunk].float().T  # [S, chunk]
        k = min(n, part.shape[1])
        v, i = part.topk(k, dim=-1)
        cv = torch.cat([best_v, v], dim=-1)
        ci = torch.cat([best_i, i + s], dim=-1)
        best_v, sel = cv.topk(n, dim=-1)
        best_i = ci.gather(-1, sel)
    return best_v, best_i


def _argmax_vocab(h, lm_w, chunk=16384):
    """argmax over vocab of h @ lm_w.T (thin wrapper over _topn_vocab; returns [S] int64)."""
    return _topn_vocab(h, lm_w, n=1, chunk=chunk)[1][:, 0]


def _trunc_probs(vals, idxs, temperature, top_k, top_p, presence_penalty=0.0, penalised=()):
    """Candidate (vals, idxs) -> the truncated sampling distribution, in the SAME order the device
    tail applies it (tt/model.py::_select_token -> ttnn.sampling): temperature scale, top_k, top_p
    (nucleus: the smallest prefix whose cumulative mass reaches top_p), renormalise.

    `penalised` is the set of already-generated token ids; `presence_penalty` is subtracted from their
    logits BEFORE truncation, exactly as the device tail does it.

    Returns (p [M] fp32 summing to 1, ids [M] int64), descending by probability."""
    lg = vals.float().clone()
    if presence_penalty > 0 and len(penalised):
        for j, t in enumerate(idxs.tolist()):
            if t in penalised:
                lg[j] -= float(presence_penalty)
    lg = lg / max(float(temperature), 1e-6)
    order = torch.argsort(lg, descending=True)
    lg, ids = lg[order], idxs[order]
    if top_k and 0 < top_k < lg.numel():
        lg, ids = lg[:top_k], ids[:top_k]
    p = torch.softmax(lg, dim=-1)
    if top_p is not None and 0.0 < top_p < 1.0:
        keep = int((p.cumsum(-1) < top_p).sum()) + 1  # always keep at least the top-1
        p, ids = p[:keep], ids[:keep]
        p = p / p.sum()
    return p, ids


def _pmass(p, ids, tok):
    """p(tok), or 0.0 if tok fell outside the truncated support (a legitimate hard reject)."""
    hit = (ids == int(tok)).nonzero()
    return float(p[hit[0, 0]]) if hit.numel() else 0.0


def _accept_metrics(pv, pi, qv, qi, draft, target_tok, cfg, penalised=()):
    """The four acceptance rules, all evaluated on the SAME (draft, target-distribution) pair.

    pv/pi: the target's top-NCAND (value, index) pair from the BACKBONE hidden.
    qv/qi: the DRAFT head's top-NCAND pair (only needed for the sampled-draft variant).
    draft: the head's argmax token.  target_tok: the captured greedy continuation.

      greedy             what ships today: exact match against the greedy continuation.
      exact_argmax_draft EXACT speculative sampling with a point-mass (argmax) draft. q is a point
                         mass at the draft, so accept-prob = min(1, p(d)/1) = p(d) and the residual
                         is p with d removed, renormalised. No draft-graph change needed.
      exact_sampled_draft EXACT speculative sampling with the draft SAMPLED from q: acceptance is
                         sum_x min(p(x), q(x)) = 1 - TV(p, q). Beats the argmax variant only when q
                         is genuinely close to p; this is the number that decides whether sampling
                         the draft on device is worth the extra topk in the draft trace.
      relaxed            Medusa-style typical acceptance (NOT distribution-preserving): accept when
                         p(d) >= min(eps, delta * exp(-H(p))).
    """
    T, k, pp, pen = cfg
    p, pid = _trunc_probs(pv, pi, T, k, pp, pen, penalised)
    pd = _pmass(p, pid, draft)
    H = float(-(p * p.clamp_min(1e-12).log()).sum())
    q, qid = _trunc_probs(qv, qi, T, k, pp, pen, penalised)
    qm = {int(i): float(v) for i, v in zip(qid, q)}
    overlap = sum(min(float(v), qm.get(int(i), 0.0)) for i, v in zip(pid, p))
    return {
        "greedy": float(int(draft) == int(target_tok)),
        "exact_argmax_draft": pd,
        "exact_sampled_draft": overlap,
        "relaxed": float(pd >= min(RELAX_EPS, RELAX_DELTA * torch.tensor(-H).exp().item())),
        "p_max": float(p[0]),
        "entropy": H,
        "support": float(p.numel()),
    }


def _chain_alpha(head, lm_w, hid, toks, gmax, window, samples, warmup, tgt=None, samp=None):
    """Per-DEPTH acceptance for a gamma-step draft chain.

    `mtp_num_hidden_layers: 1` means the head is trained to predict exactly ONE token ahead, so a
    gamma>1 draft must run it AUTOREGRESSIVELY on its own output (pre-`mtp.norm` hidden fed back as
    the next step's `hidden`, per the DeepSeek MTP playbook). That is off-distribution, so acceptance
    decays with depth and a single-alpha geometric model (1 + a + a^2 + ...) is too generous.

    alpha_j = P[ d_j == t_{i+1+j} ], measured over `samples` positions with a `window`-token true
    prefix so the head's attention has real context.

    `tgt` = (values, indices) of the BACKBONE's top-NCAND at every position (one batched _topn_vocab
    call — see score()). When given, the per-depth SAMPLING acceptance rules are accumulated into
    `samp[cfg][j]` on the very same head forwards, so the sampled numbers cost no extra compute.

    CAVEAT on depth >= 2 under sampling: offline we can only follow the CAPTURED (greedy) trajectory,
    so a draft that speculative sampling would accept but that differs from the greedy continuation
    cannot be scored further. Depth 1 is exact; deeper depths are conditioned on greedy-match at the
    shallower depths — the same convention this function already uses for the greedy numbers, so the
    two columns stay directly comparable.
    """
    N = hid.shape[0]
    lo, hi = max(warmup, window), N - gmax - 2
    if hi <= lo:
        return None
    step = max(1, (hi - lo) // samples)
    idxs = list(range(lo, hi, step))[:samples]
    hits = [0] * gmax
    for i in idxs:
        # teacher-forced true prefix: positions [i-window, i]
        h_seq = [hid[t] for t in range(i - window, i + 1)]
        e_seq = [toks[t + 1] for t in range(i - window, i + 1)]
        emb = _embed_rows(e_seq).to(torch.bfloat16)
        h_stack = torch.stack(h_seq)
        cur_emb = emb
        for j in range(gmax):
            out, pre = head(
                h_stack.unsqueeze(0),
                cur_emb.unsqueeze(0),
                position_ids=torch.arange(i - window, i - window + h_stack.shape[0]).unsqueeze(0),
                return_prenorm=True,
            )
            qv, qi = _topn_vocab(out[0, -1:], lm_w, n=NCAND if samp is not None else 1)
            d = int(qi[0, 0])
            if samp is not None:  # sampled-mode metrics, on the SAME forward (free)
                for cfg in SAMP_CONFIGS:
                    m = _accept_metrics(
                        tgt[0][i + 1 + j],
                        tgt[1][i + 1 + j],
                        qv[0],
                        qi[0],
                        d,
                        toks[i + 2 + j],
                        cfg,
                        penalised=set(toks[1 : i + 2 + j]) if cfg[3] > 0 else (),
                    )
                    acc = samp.setdefault(cfg, [dict() for _ in range(gmax)])[j]
                    acc["n"] = acc.get("n", 0) + 1
                    for key, val in m.items():
                        acc[key] = acc.get(key, 0.0) + val
            if d == toks[i + 2 + j]:
                hits[j] += 1
            else:
                break  # a real chain stops being useful past the first miss; deeper alphas are
                # conditional on all shallower ones accepting, which is what the cost model wants
            # feed back: pre-norm hidden + the drafted token become the next step's inputs
            h_stack = torch.cat([h_stack, pre[0, -1:]], dim=0)
            cur_emb = torch.cat([cur_emb, _embed_rows([d]).to(torch.bfloat16)], dim=0)
    n = len(idxs)
    # hits[j] counts positions where depths 1..j+1 ALL accepted -> conditional alpha per depth
    cond = []
    prev = n
    for j in range(gmax):
        cond.append(hits[j] / prev if prev else 0.0)
        prev = hits[j]
    return {"n": n, "cumulative": [h / n for h in hits], "conditional": cond}


def _depth1_sampling(head, lm_w, hid, toks, warmup, chunk, tgt, acc):
    """DEPTH-1 acceptance under sampling, over the last `chunk` positions -- the DECIDING number.

    Depth 1 needs no draft chain: one teacher-forced head pass gives d_i at every position, and each
    position is scored against its OWN captured target hidden, so there is no trajectory ambiguity
    (unlike depth >= 2, see _chain_alpha). One head forward + one batched _topn_vocab per prompt, so
    this affords ~4x more sample positions than the chain measurement at ~1/30th the cost.

    Accumulates into `acc[cfg]` (shared across prompts)."""
    N = hid.shape[0]
    n = N - 1  # need t_{i+2}
    te = _embed_rows(toks[1 : n + 1]).to(torch.bfloat16)
    h = head(hid[:n].unsqueeze(0), te[:n].unsqueeze(0), position_ids=torch.arange(n).unsqueeze(0))
    lo = max(warmup, n - chunk)
    qv, qi = _topn_vocab(h[0, lo:n], lm_w, n=NCAND)  # the head's own candidates (q)
    for r, i in enumerate(range(lo, n)):
        for cfg in SAMP_CONFIGS:
            # presence set = the tokens generated BEFORE this position in the captured stream
            m = _accept_metrics(
                tgt[0][i + 1],
                tgt[1][i + 1],
                qv[r],
                qi[r],
                int(qi[r, 0]),
                toks[i + 2],
                cfg,
                penalised=set(toks[1 : i + 2]) if cfg[3] > 0 else (),
            )
            a = acc.setdefault(cfg, {})
            a["n"] = a.get("n", 0) + 1
            for key, val in m.items():
                a[key] = a.get(key, 0.0) + val
    return lo, n


def _fmt_acc(a):
    """Sum-accumulator -> per-position means (keeping the count under "n")."""
    n = max(a.get("n", 0), 1)
    out = {k: v / n for k, v in a.items() if k != "n"}
    out["n"] = a.get("n", 0)
    return out


def score(warmup: int, chunk: int, gmax: int = 4, window: int = 96, samples: int = 40):
    from models.demos.qwen3_6_a3b.reference.qwen3_5_moe import Qwen35MoeConfig, Qwen35MoeMTP

    if not OUT.exists():
        sys.exit(f"no capture at {OUT} -- run `capture` on the board first")
    bundle = torch.load(OUT, weights_only=False)
    cfg = Qwen35MoeConfig.from_hf_config(CKPT)
    print(
        f"[a5] mtp_num_hidden_layers={cfg.mtp_num_hidden_layers} " f"dedicated_emb={cfg.mtp_use_dedicated_embeddings}"
    )

    mtp_sd = _load_mtp_state()
    print(f"[a5] loaded {len(mtp_sd)} mtp tensors ({sum(v.numel() for v in mtp_sd.values())/1e6:.0f}M params)")
    lm_w = _lm_head()
    print(f"[a5] lm_head {list(lm_w.shape)} {lm_w.dtype}")

    variants = ["hidden", "hidden_pre"] if "hidden_pre" in bundle["runs"][0] else ["hidden"]
    results, heads = {}, {}
    for order, hkey in [(o, h) for h in variants for o in ("token_first", "hidden_first")]:
        # construct in bf16 so the fp32 params are never allocated (memory, not accuracy: the device
        # itself runs these experts at BFP4, so bf16 host compute is if anything more faithful)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            head = Qwen35MoeMTP(cfg, concat_order=order)
        finally:
            torch.set_default_dtype(prev)
        missing, unexpected = head.load_state_dict(mtp_sd, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        head.eval()

        hits = tot = 0
        per_prompt = []
        with torch.no_grad():
            for run in bundle["runs"]:
                hid = run[hkey].to(torch.bfloat16)  # [N, dim]  h_i
                toks = run["tokens"]  # [N+1]; toks[0] is the token consumed at step 0
                N = hid.shape[0]
                n = N - 1  # need t_{i+2}, so i <= N-2
                te = _embed_rows(toks[1 : n + 1]).to(torch.bfloat16)  # embed(t_{i+1})
                # one teacher-forced pass over the whole sequence (so the head's attention sees real
                # context), but score only the LAST `chunk` positions -- the lm_head argmax is the
                # expensive part and this is just the 4-way discriminator, not the final number.
                h = head(hid[:n].unsqueeze(0), te[:n].unsqueeze(0), position_ids=torch.arange(n).unsqueeze(0))
                lo = max(warmup, n - chunk)
                pred = _argmax_vocab(h[0, lo:n], lm_w)
                tgt = torch.tensor(toks[2 : n + 2])[lo:n]
                ok = pred == tgt
                hits += int(ok.sum())
                tot += ok.numel()
                per_prompt.append(float(ok.float().mean()))
        alpha = hits / max(tot, 1)
        key = f"{hkey}/{order}"
        results[key] = alpha
        print(
            f"  {key:>26}:  alpha_1 = {alpha:.4f}  ({hits}/{tot})   " f"per-prompt {['%.3f' % p for p in per_prompt]}",
            flush=True,
        )
        heads[key] = head

    best = max(results, key=results.get)
    print(f"\n  => WINNER: concat_order={best}, alpha_1={results[best]:.4f}")
    if min(results.values()) > 0.3:
        print("     WARNING: both orders scored high -- weak discriminator, re-check.")

    # per-depth chain acceptance for the winning order -- this is what the cost model needs
    print(
        f"\n  --- autoregressive draft chain (depth 1..{gmax}), window={window}, " f"samples={samples}/prompt ---",
        flush=True,
    )
    chain = []
    hkey_best = best.split("/")[0]
    # The BACKBONE's top-NCAND at every captured position IS the speculative target distribution p.
    # One batched _topn_vocab per prompt (BLAS is far more efficient per row than the per-position
    # loop); cached so the depth-1 and the chain passes share it.
    tgts = {}
    with torch.no_grad():
        for run in bundle["runs"]:
            tgts[run["prompt_idx"]] = _topn_vocab(run[hkey_best].float(), lm_w, n=NCAND)
    samp_d1, samp_chain = {}, {}
    with torch.no_grad():
        for run in bundle["runs"]:
            c = _chain_alpha(
                heads[best],
                lm_w,
                run[hkey_best].to(torch.bfloat16),
                run["tokens"],
                gmax,
                window,
                samples,
                warmup,
                tgt=tgts[run["prompt_idx"]],
                samp=samp_chain,
            )
            if c:
                chain.append(c)
                print(
                    f"    prompt {run['prompt_idx']}: cumulative "
                    f"{['%.3f' % v for v in c['cumulative']]}  conditional "
                    f"{['%.3f' % v for v in c['conditional']]}",
                    flush=True,
                )
    # ---- depth-1 acceptance under SAMPLING (the deciding number), over many more positions ----
    print(f"\n  --- SAMPLED acceptance, depth 1 (teacher-forced, last {chunk} positions/prompt) ---", flush=True)
    with torch.no_grad():
        for run in bundle["runs"]:
            _depth1_sampling(
                heads[best],
                lm_w,
                run[hkey_best].to(torch.bfloat16),
                run["tokens"],
                warmup,
                chunk,
                tgts[run["prompt_idx"]],
                samp_d1,
            )
    hdr = (
        f"    {'T/k/p/pen':>18}{'greedy':>9}{'exact(amax)':>13}{'exact(samp)':>13}"
        f"{'relaxed':>9}{'E[p_max]':>10}{'H(p)':>7}{'|supp|':>8}"
    )
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    d1 = {}
    for cfg in SAMP_CONFIGS:
        a = _fmt_acc(samp_d1.get(cfg, {}))
        d1[str(cfg)] = a
        print(
            f"    {'%.2f/%d/%.2f/%.1f' % cfg:>18}{a.get('greedy', 0):>9.3f}"
            f"{a.get('exact_argmax_draft', 0):>13.3f}{a.get('exact_sampled_draft', 0):>13.3f}"
            f"{a.get('relaxed', 0):>9.3f}{a.get('p_max', 0):>10.3f}"
            f"{a.get('entropy', 0):>7.2f}{a.get('support', 0):>8.1f}",
            flush=True,
        )
    print(
        f"    n = {d1[str(SAMP_CONFIGS[0])].get('n', 0)} positions. 'greedy' must reproduce the alpha_1 above\n"
        f"    (self-check); 'exact(amax)' is E[p(argmax q)], the acceptance the shipping rule would get."
    )

    # ---- the same rules BY DEPTH, from the chain forwards (no extra compute) ----
    print("\n  --- SAMPLED acceptance by DEPTH (conditional; depth >= 2 is on-greedy-trajectory) ---")
    depth = {}
    for cfg in SAMP_CONFIGS:
        rows = [_fmt_acc(a) for a in samp_chain.get(cfg, [])]
        depth[str(cfg)] = rows
        for name in ("greedy", "exact_argmax_draft", "exact_sampled_draft", "relaxed"):
            print(
                f"    {'%.2f/%d/%.2f/%.1f' % cfg:>18} {name:>19}: {['%.3f' % r.get(name, 0.0) for r in rows]}",
                flush=True,
            )

    out = {
        "alpha_1_teacher_forced": results,
        "winner": best,
        "chain": chain,
        "sampled_depth1": d1,
        "sampled_by_depth": depth,
        "configs": [list(c) for c in SAMP_CONFIGS],
    }
    if chain:
        cum = [sum(c["cumulative"][j] * c["n"] for c in chain) / sum(c["n"] for c in chain) for j in range(gmax)]
        out["cumulative_mean"] = cum
        print(f"\n    MEAN cumulative accept P[depth>=j]: {['%.3f' % v for v in cum]}")
        print(f"\n  {'gamma':>6}{'E[tokens/round]':>18}   (1 + sum of cumulative accepts)")
        for g in range(1, gmax + 1):
            print(f"  {g:>6}{1 + sum(cum[:g]):>18.3f}")
        print("\n  Feed E[tokens] into the verify cost model (probe_mtp_verify.py) for the speedup.")
        print("  NOTE: this replaces the optimistic geometric 1+a+a^2+... -- the head predicts ONE")
        print("  token ahead by construction, so depth>=2 runs it off-distribution.")
    with open("/tmp/qwen36_mtp_alpha.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# ============================================================ project (host)
# MEASURED round decomposition, 40 layers, gamma=2, tt-metal v0.77 (EXPERIMENTS.md row F):
#   verify 53.55 + draft 4.20 + commit 6.18 + host glue 0.4 = 64.33 ms  (bench_mtp measured 64.4)
# verify's marginal is the measured 10.98 ms/token slope (EXPERIMENTS.md "Measurement 1"); the
# intercept is back-solved from the post-optimisation verify(3), since the optimisations that landed
# (#5/#8/#9) cut fixed overhead, not the slope.
VERIFY_SLOPE = 10.98
VERIFY_C0 = 53.55 - 3 * VERIFY_SLOPE
DRAFT_PER_STEP = 4.20 / 2
COMMIT_MS = 6.18
GLUE_MS = 0.4
BASE_GREEDY_MS = 29.01  # bench_decode.py A0
BASE_SAMPLING_MS = 30.02  # + the measured 1.01 ms full thinking-mode tail (SAMPLING.md)
# Per verify ROW cost of the candidate tail (pad+reshape+topk+offsets). ESTIMATE from SAMPLING.md's
# measured 0.43 ms single-row sampling tail; replaced by the real bench_mtp number in Phase 3.
TAIL_PER_ROW_MS = 0.43


def _round_ms(gamma, sampled):
    K = gamma + 1
    ms = VERIFY_C0 + VERIFY_SLOPE * K + DRAFT_PER_STEP * gamma + COMMIT_MS + GLUE_MS
    return ms + (K * TAIL_PER_ROW_MS if sampled else 0.0)


def _e_tokens(cond, gamma):
    """E[tokens/round] = 1 + sum_j prod_{l<=j} a_l, from per-depth CONDITIONAL acceptances."""
    e, run = 1.0, 1.0
    for j in range(gamma):
        run *= cond[j] if j < len(cond) else 0.0
        e += run
    return e


def project(gmax: int = 4):
    """Turn the measured acceptances into projected throughput. Host-only; reads the score() output."""
    src = Path("/tmp/qwen36_mtp_alpha.json")
    if not src.exists():
        sys.exit(f"no scores at {src} -- run `score` first")
    a = json.load(open(src))
    print("Projection from the MEASURED round decomposition (EXPERIMENTS.md row F) x measured acceptance.")
    print(f"  baselines: greedy {BASE_GREEDY_MS:.2f} ms/token, sampling {BASE_SAMPLING_MS:.2f} ms/token")
    print(
        f"  round(gamma) = {VERIFY_C0:.2f} + {VERIFY_SLOPE:.2f}K + {DRAFT_PER_STEP:.2f}*gamma"
        f" + {COMMIT_MS:.2f} + {GLUE_MS:.2f}  (+ {TAIL_PER_ROW_MS:.2f}/row when sampling)"
    )

    rows = []
    greedy_cond = None
    if a.get("chain"):
        # conditional acceptance pooled over prompts, weighted by sample count
        tot = sum(c["n"] for c in a["chain"])
        greedy_cum = [sum(c["cumulative"][j] * c["n"] for c in a["chain"]) / tot for j in range(gmax)]
        greedy_cond, prev = [], 1.0
        for j in range(gmax):
            greedy_cond.append(greedy_cum[j] / prev if prev > 0 else 0.0)
            prev = greedy_cum[j]
        rows.append(("greedy MTP (ships today)", False, BASE_GREEDY_MS, greedy_cond))

    for cfg_s, per_depth in (a.get("sampled_by_depth") or {}).items():
        for rule in ("exact_argmax_draft", "exact_sampled_draft", "relaxed"):
            cond = [r.get(rule, 0.0) for r in per_depth]
            rows.append((f"{rule} @ {cfg_s}", True, BASE_SAMPLING_MS, cond))
    print("\n  NOTE: these projections use the HOST acceptance model. Device-measured (bench_mtp, 40L,")
    print("  gamma=2, exact rule) came out LOWER than the host projection because the host model scores")
    print("  the captured GREEDY trajectory while the device runs on-policy sampled text; trust")
    print("  bench_mtp for the shipping number and this table for exploring the parameter space.")

    hdr = f"\n  {'variant':>44}{'gamma':>7}{'E[tok]':>8}{'round ms':>10}{'ms/tok':>8}{'tok/s':>8}{'speedup':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))
    best = {}
    for label, sampled, base, cond in rows:
        for gamma in range(1, gmax + 1):
            e = _e_tokens(cond, gamma)
            if e <= 0:
                continue
            ms = _round_ms(gamma, sampled) / e
            sp = base / ms
            if sp > best.get(label, (0,))[0]:
                best[label] = (sp, gamma, e, ms)
            print(
                f"  {label:>44}{gamma:>7}{e:>8.3f}{_round_ms(gamma, sampled):>10.2f}"
                f"{ms:>8.2f}{1000 / ms:>8.1f}{sp:>8.2f}x"
            )
    print("\n  best gamma per variant:")
    for label, (sp, gamma, e, ms) in sorted(best.items(), key=lambda kv: -kv[1][0]):
        verdict = "GO" if sp >= 1.15 else ("marginal" if sp >= 1.05 else "NO")
        print(f"    {label:>44}: gamma={gamma}  {1000 / ms:5.1f} tok/s  {sp:.2f}x   [{verdict}]")
    print("\n  Gate: >= 1.15x ships on by default; 1.05-1.15x ships flag-gated; < 1.05x greedy-only.")
    print("  CAVEAT: depth >= 2 acceptance is measured on the captured greedy trajectory, and")
    print("  TAIL_PER_ROW_MS is an estimate until bench_mtp measures the real candidate tail.")
    with open("/tmp/qwen36_mtp_projection.json", "w") as f:
        json.dump(
            {k: {"speedup": v[0], "gamma": v[1], "e_tokens": v[2], "ms_per_token": v[3]} for k, v in best.items()},
            f,
            indent=2,
        )
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["capture", "score", "project"])
    ap.add_argument("--gen", type=int, default=300, help="tokens to generate per prompt (capture)")
    ap.add_argument("--warmup", type=int, default=32, help="skip the first N positions (cold MTP KV)")
    ap.add_argument("--chunk", type=int, default=128, help="teacher-forced chunk (score, memory bound)")
    ap.add_argument("--gmax", type=int, default=4, help="max draft depth for the chain measurement")
    ap.add_argument("--window", type=int, default=96, help="true-prefix context for the chain")
    ap.add_argument("--samples", type=int, default=40, help="chain sample positions per prompt")
    a = ap.parse_args()
    if a.stage == "capture":
        capture(a.gen)
    elif a.stage == "project":
        project(a.gmax)
    else:
        score(a.warmup, a.chunk, a.gmax, a.window, a.samples)


if __name__ == "__main__":
    main()
