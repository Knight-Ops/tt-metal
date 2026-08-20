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


def _argmax_vocab(h, lm_w, chunk=16384):
    """argmax over vocab of h @ lm_w.T, chunked over vocab so [S, 248320] is never materialised.

    Each weight chunk is cast to fp32 for the GEMM: torch CPU has no fast bf16 matmul kernel, so a
    bf16 @ bf16 here is ~20x slower than casting a 16384x2048 slice (134 MB) and using the fp32 BLAS
    path. Peak stays ~150 MB above the resident bf16 lm_head."""
    hf = h.float()
    best_v = torch.full((h.shape[0],), float("-inf"))
    best_i = torch.zeros(h.shape[0], dtype=torch.long)
    for s in range(0, lm_w.shape[0], chunk):
        part = hf @ lm_w[s : s + chunk].float().T
        v, i = part.max(dim=-1)
        upd = v > best_v
        best_v = torch.where(upd, v, best_v)
        best_i = torch.where(upd, i + s, best_i)
    return best_i


def _chain_alpha(head, lm_w, hid, toks, gmax, window, samples, warmup):
    """Per-DEPTH acceptance for a gamma-step draft chain.

    `mtp_num_hidden_layers: 1` means the head is trained to predict exactly ONE token ahead, so a
    gamma>1 draft must run it AUTOREGRESSIVELY on its own output (pre-`mtp.norm` hidden fed back as
    the next step's `hidden`, per the DeepSeek MTP playbook). That is off-distribution, so acceptance
    decays with depth and a single-alpha geometric model (1 + a + a^2 + ...) is too generous.

    alpha_j = P[ d_j == t_{i+1+j} ], measured over `samples` positions with a `window`-token true
    prefix so the head's attention has real context.
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
            d = int(_argmax_vocab(out[0, -1:], lm_w)[0])
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
    with torch.no_grad():
        for run in bundle["runs"]:
            c = _chain_alpha(
                heads[best],
                lm_w,
                run[best.split("/")[0]].to(torch.bfloat16),
                run["tokens"],
                gmax,
                window,
                samples,
                warmup,
            )
            if c:
                chain.append(c)
                print(
                    f"    prompt {run['prompt_idx']}: cumulative "
                    f"{['%.3f' % v for v in c['cumulative']]}  conditional "
                    f"{['%.3f' % v for v in c['conditional']]}",
                    flush=True,
                )
    out = {"alpha_1_teacher_forced": results, "winner": best, "chain": chain}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["capture", "score"])
    ap.add_argument("--gen", type=int, default=300, help="tokens to generate per prompt (capture)")
    ap.add_argument("--warmup", type=int, default=32, help="skip the first N positions (cold MTP KV)")
    ap.add_argument("--chunk", type=int, default=128, help="teacher-forced chunk (score, memory bound)")
    ap.add_argument("--gmax", type=int, default=4, help="max draft depth for the chain measurement")
    ap.add_argument("--window", type=int, default=96, help="true-prefix context for the chain")
    ap.add_argument("--samples", type=int, default=40, help="chain sample positions per prompt")
    a = ap.parse_args()
    if a.stage == "capture":
        capture(a.gen)
    else:
        score(a.warmup, a.chunk, a.gmax, a.window, a.samples)


if __name__ == "__main__":
    main()
