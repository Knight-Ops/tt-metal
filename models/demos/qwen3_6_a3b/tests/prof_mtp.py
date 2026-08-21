# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Where does the speculative VERIFY pass spend its time?

Two instruments in one file, because they answer different questions:

  --sweep  (no profiler, 40 layers): capture a verify trace at each K and time the replays. The SLOPE
           is the true marginal cost per speculative token and the intercept is the fixed cost; both
           are absolute and trustworthy at the real layer count. Run this first.

  default  (under `python -m tracy`, few layers): signpost-bracketed EAGER verify so the device
           profiler attributes per-op kernel time to components. Per-op DEVICE time is identical
           between eager and traced replay (same kernels) and host signposts do not fire during
           `execute_trace`, so eager is the right thing to profile — see tt/signpost.py. Read the
           result as PROPORTIONS: the profiler inflates absolutes and 4 layers is not 40.

    QWEN36_LAYERS=40 ./python_env/bin/python .../prof_mtp.py --sweep
    QWEN36_LAYERS=4  ./python_env/bin/python -m tracy -r -v --op-support-count 8000 .../prof_mtp.py
Then: ./python_env/bin/python .../signpost_report.py --between mtp_verify_start mtp_verify_stop
"""
import argparse
import os
import sys

# Signposts are for the Tracy path only; `--sweep`/`--components` do not use them and the markers
# flood the log (hundreds per layer). Opt out with QWEN36_SIGNPOST=0.
if not any(a in sys.argv for a in ("--sweep", "--components", "--gdnbreak")):
    os.environ.setdefault("QWEN36_SIGNPOST", "1")  # read at import time -> must precede model imports

import time

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs

try:
    from tracy import signpost
except Exception:

    def signpost(*a, **k):
        pass


CKPT = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
PROMPT = "Explain why a mixture-of-experts layer is hard to batch on an accelerator."


def _build(mesh, n_layers, max_seq):
    args = ModelArgs(mesh, ckpt_dir=CKPT, max_seq_len=max_seq)
    loader = CheckpointLoader(CKPT)
    assert loader.has_mtp(), "checkpoint has no mtp.* head"
    model = TtModel(mesh, args, loader, num_layers=n_layers)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(CKPT)
        enc = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}], add_generation_prompt=True, return_tensors="pt"
        )
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        ids = (ids if torch.is_tensor(ids) else torch.tensor(ids)).reshape(1, -1).to(torch.int64)
    except Exception:
        torch.manual_seed(0)
        ids = torch.randint(0, args.vocab_size, (1, 24))
    lg = model.forward(ids)
    model.start_decode(int(lg[0, -1].argmax()))
    return model, args


def sweep(mesh, n_layers, max_seq, ks, iters):
    """verify-trace replay time vs K -> (fixed cost, marginal cost per speculative token)."""
    model, args = _build(mesh, n_layers, max_seq)
    print(f"\n  {'K':>3}{'verify ms':>11}{'d/tok':>9}   (traced replay, {n_layers} layers)")
    prev = None
    rows = {}
    for K in ks:
        model.build_mtp_head(gamma=K - 1)
        model.setup_mtp_decode()
        model._mtp_hidden = None
        model.spec_decode_step()  # seed so `_verify_pending` exists and the head is warm
        cur = int(ttnn.to_torch(model.t_tok).reshape(-1)[0])
        pos = [model.pos + i for i in range(K)]
        model.capture_mtp_verify_trace([cur] * K, pos)
        ttnn.execute_trace(mesh, model.mtp_verify_trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        for _ in range(iters):
            ttnn.execute_trace(mesh, model.mtp_verify_trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ms = (time.time() - t0) / iters * 1e3
        rows[K] = ms
        d = "" if prev is None else f"{ms - prev:>9.3f}"
        print(f"  {K:>3}{ms:>11.3f}{d}", flush=True)
        prev = ms
    if len(rows) >= 2:
        k0, k1 = min(rows), max(rows)
        slope = (rows[k1] - rows[k0]) / (k1 - k0)
        print(f"\n  fit: verify(K) ~= {rows[k0] - slope * k0:.2f} + {slope:.3f}*K ms")
        print(f"  cost model says 24.31 + 4.683*K  ->  marginal is {slope / 4.683:.2f}x the model")


def components(mesh, n_layers, max_seq, K, iters):
    """Time each verify SUB-BLOCK in its own trace, x its per-token layer count — the same method
    `bench_decode.bench_components` uses for decode, so the two are directly comparable.

    This is what tells us where verify's excess over the cost model actually sits. Op-count reasoning
    predicted the last two optimisations wrong (see EXPERIMENTS.md), so measure per component."""
    from models.demos.qwen3_6_a3b.tests.bench_decode import time_traced

    model, args = _build(mesh, n_layers, max_seq)
    model.build_mtp_head(gamma=K - 1)
    model.setup_mtp_decode()
    model._mtp_hidden = None
    model.spec_decode_step()
    model.spec_decode_step()  # ensure K-shaped verify scratch exists

    dim = args.dim
    lin = [(l, c) for l, c in zip(model.layers, model.caches) if l.is_linear]
    att = [(l, c) for l, c in zip(model.layers, model.caches) if not l.is_linear]
    n_gdn, n_attn, n_moe = 30, 10, 40  # per-token layer counts at 40 layers

    # EVERY device tensor must be built before capture: a `_pos_tensor`/`to_tt` inside a timed lambda
    # is a host write during trace capture and fatals ("Writes are not supported during trace capture").
    x1 = to_tt_x(mesh, 1, dim)
    xK = to_tt_x(mesh, K, dim)
    posK = model._m_pos_i32
    pos1 = model._pos_tensor([model.pos], dtype=ttnn.int32)
    cos1, sin1 = model._rope_for(model._pos_tensor([model.pos]))
    cosK, sinK = model._rope_for(model._m_pos_u32)

    gl, gc = lin[0]
    al, ac = att[0]
    rows = []
    rows.append(
        ("gdn decode (T=1)", n_gdn, time_traced(mesh, lambda: gl.mixer.forward(x1, cache=gc, decode=True), iters))
    )
    rows.append(
        (f"gdn verify (K={K})", n_gdn, time_traced(mesh, lambda: gl.mixer.forward(xK, cache=gc, verify=True), iters))
    )
    rows.append(
        (
            "attn decode (T=1)",
            n_attn,
            time_traced(mesh, lambda: al.mixer.forward_decode(x1, cos1, sin1, ac, pos1), iters),
        )
    )
    rows.append(
        (
            f"attn verify (K={K})",
            n_attn,
            time_traced(mesh, lambda: al.mixer.forward_verify(xK, cosK, sinK, ac, posK), iters),
        )
    )
    rows.append(("moe decode (T=1)", n_moe, time_traced(mesh, lambda: gl.moe.forward(x1), iters)))
    rows.append((f"moe verify (K={K})", n_moe, time_traced(mesh, lambda: gl.moe.forward_verify(xK), iters)))
    rows.append(
        (
            "lm_head+argmax (K)",
            1,
            time_traced(
                mesh,
                lambda: ttnn.argmax(
                    ttnn.to_layout(ttnn.linear(ttnn.reshape(xK, [K, dim]), model.lm_head_w), ttnn.ROW_MAJOR_LAYOUT),
                    dim=-1,
                ),
                iters,
            ),
        )
    )

    print(f"\n  {'component':>22}{'ms/op':>9}{'xN':>5}{'total ms':>10}")
    for name, n, ms in rows:
        print(f"  {name:>22}{ms:>9.4f}{n:>5}{ms*n:>10.2f}")
    d = {n: ms for n, _, ms in rows}
    gdn_d, gdn_v = d["gdn decode (T=1)"], d[f"gdn verify (K={K})"]
    at_d, at_v = d["attn decode (T=1)"], d[f"attn verify (K={K})"]
    mo_d, mo_v = d["moe decode (T=1)"], d[f"moe verify (K={K})"]
    print(f"\n  verify-vs-decode EXCESS per component (x layer count):")
    print(f"    gdn  {(gdn_v-gdn_d)*n_gdn:>7.2f} ms   ({gdn_v:.4f} vs {gdn_d:.4f} per layer)")
    print(f"    attn {(at_v-at_d)*n_attn:>7.2f} ms   ({at_v:.4f} vs {at_d:.4f})")
    print(f"    moe  {(mo_v-mo_d)*n_moe:>7.2f} ms   ({mo_v:.4f} vs {mo_d:.4f})")
    print(
        f"    sum of component verify totals: "
        f"{gdn_v*n_gdn + at_v*n_attn + mo_v*n_moe + d['lm_head+argmax (K)']:.2f} ms"
    )


def gdnbreak(mesh, n_layers, max_seq, K, iters):
    """Break the GDN verify block into its stages and time each in its own trace.

    Needed because the block-level number (1.01 ms/layer vs a 0.32 ms decode step) is not explained by
    op count, L1 residency, or the recurrence itself, and three rounds of reasoning-from-op-count have
    now mispredicted. Times the SAME sequence `_forward_verify_seq` runs, stage by stage."""
    from models.demos.qwen3_6_a3b.tests.bench_decode import time_traced

    model, args = _build(mesh, n_layers, max_seq)
    model.build_mtp_head(gamma=K - 1)
    model.setup_mtp_decode()
    model._mtp_hidden = None
    model.spec_decode_step()
    model.spec_decode_step()

    m = [l.mixer for l in model.layers if l.is_linear][0]
    cache = [c for l, c in zip(model.layers, model.caches) if l.is_linear][0]
    V, Dk, Dv = m.num_v_heads, m.head_k_dim, m.head_v_dim
    TILE = 32
    x2 = to_tt_x(mesh, K, args.dim)
    x2 = ttnn.reshape(x2, [K, args.dim])

    allp = ttnn.linear(x2, m.w_in_proj)
    o0, o1, o2 = m._in_off
    mixed = ttnn.slice(allp, [0, 0], [K, o0])
    z = ttnn.slice(allp, [0, o0], [K, o1])
    ba = ttnn.slice(allp, [0, o1], [K, o2])
    convd, _ = m._conv_silu(mixed, cache.get("conv_state"))
    q = ttnn.slice(convd, [0, 0], [K, m.key_dim])
    k = ttnn.slice(convd, [0, m.key_dim], [K, 2 * m.key_dim])
    v = ttnn.slice(convd, [0, 2 * m.key_dim], [K, m.conv_dim])
    qT = m._to_tilerows(q, m.key_dim, K)
    kT = m._to_tilerows(k, m.key_dim, K)
    vT = m._to_tilerows(v, m.value_dim, K)
    zT = m._to_tilerows(z, m.value_dim, K)
    bsl = ttnn.slice(ba, [0, 0], [K, V])
    asl = ttnn.slice(ba, [0, V], [K, 2 * V])
    braw = ttnn.reshape(
        ttnn.repeat(ttnn.reshape(bsl, [K * V, 1, 1]), ttnn.Shape([1, TILE, TILE])), [K * V * TILE, TILE]
    )
    araw = ttnn.reshape(
        ttnn.repeat(ttnn.reshape(asl, [K * V, 1, 1]), ttnn.Shape([1, TILE, TILE])), [K * V * TILE, TILE]
    )
    nA, dtbc = m._verify_const(K)
    out_buf, snew_buf = m._verify_chain_scratch(K)
    Sin = ttnn.reshape(cache["recurrent_state"], [V * Dk, Dv])

    xpad_c = ttnn.concat([cache["conv_state"], mixed], dim=0)
    stages = [
        ("in_proj (M=K)", lambda: ttnn.linear(x2, m.w_in_proj)),
        ("conv_silu (T=K)", lambda: m._conv_silu(mixed, cache.get("conv_state"))),
        # inside the conv. The verify conv is now the STACKED form (one matmul selects every tap shift,
        # one multiply applies all taps, one matmul sums them) — see gated_delta._conv_stack_const. The
        # old slice-loop pieces are kept below for comparison because that is where the 0.475 -> 0.235
        # ms/layer came from, but they are NOT what runs any more.
        ("  conv: xpad concat", lambda: ttnn.concat([cache["conv_state"], mixed], dim=0)),
        ("  conv: stacked matmul S", lambda: ttnn.matmul(m._conv_stack_const(K)[0], xpad_c)),
        ("  conv: stacked mul TAPS", lambda: ttnn.multiply(_stack_Z(m, xpad_c, K), m._conv_stack_const(K)[2])),
        ("  conv: stacked matmul A", lambda: ttnn.matmul(m._conv_stack_const(K)[1], _stack_Y(m, xpad_c, K))),
        ("  conv: silu", lambda: ttnn.silu(mixed)),
        ("  [old] conv: 4 slice+mul+add", lambda: _conv_taps_only(m, xpad_c, K)),
        ("  [old] conv: new_state slice", lambda: ttnn.slice(xpad_c, [K, 0], [K + m.conv_k - 1, m.conv_dim])),
        (
            "4x _to_tilerows",
            lambda: [
                m._to_tilerows(q, m.key_dim, K),
                m._to_tilerows(k, m.key_dim, K),
                m._to_tilerows(v, m.value_dim, K),
                m._to_tilerows(z, m.value_dim, K),
            ][-1],
        ),
        (
            "2x ba repeat",
            lambda: ttnn.reshape(
                ttnn.repeat(ttnn.reshape(bsl, [K * V, 1, 1]), ttnn.Shape([1, TILE, TILE])), [K * V * TILE, TILE]
            ),
        ),
        # NB: the chained ttl kernel is deliberately NOT timed here — compiling it in this ad-hoc
        # context segfaults the ttl MLIR pass (materializeToDFB). Infer it as
        # (block cost from --components) - (STAGE SUM below).
        ("_from_tilerows", lambda: m._from_tilerows(out_buf, m.value_dim, K)),
        ("w_out (M=K)", lambda: ttnn.linear(m._from_tilerows(out_buf, m.value_dim, K), m.w_out)),
    ]
    print(f"\n  GDN verify stage breakdown (K={K}), per layer:")
    tot = 0.0
    for name, fn in stages:
        try:
            ms = time_traced(mesh, fn, iters)
        except Exception as e:  # noqa: BLE001
            print(f"    {name:>20}  [FAIL {type(e).__name__}: {str(e)[:50]}]")
            continue
        tot += ms
        print(f"    {name:>20}{ms:>9.4f} ms", flush=True)
    print(f"    {'STAGE SUM':>20}{tot:>9.4f} ms   (isolated stages OVER-count: they do not pipeline)")


def _stack_Z(m, xpad, T):
    """Z = S @ xpad — the stacked conv's shift-selection product (see gated_delta._conv_stack_const)."""
    return ttnn.matmul(m._conv_stack_const(T)[0], xpad)


def _stack_Y(m, xpad, T):
    """Y = Z * TAPS — the stacked conv's tap product, ready for the summing matmul."""
    return ttnn.multiply(_stack_Z(m, xpad, T), m._conv_stack_const(T)[2])


def _conv_taps_only(m, xpad, T):
    """The conv's tap loop with the xpad concat already done — isolates the arithmetic from the concat."""
    acc = None
    for j in range(m.conv_k):
        term = ttnn.multiply(ttnn.slice(xpad, [j, 0], [j + T, m.conv_dim]), m.conv_taps[j])
        acc = term if acc is None else ttnn.add(acc, term)
    return acc


def to_tt_x(mesh, T, dim):
    from models.demos.qwen3_6_a3b.tt.common import to_tt

    return to_tt(torch.randn(1, 1, T, dim) * 0.5, mesh)


def profile(mesh, n_layers, max_seq, gamma):
    """Signpost-bracketed EAGER verify (+ draft, + commit) for the device profiler."""
    model, args = _build(mesh, n_layers, max_seq)
    model.build_mtp_head(gamma=gamma)
    K = model.setup_mtp_decode()
    model._mtp_hidden = None
    model.spec_decode_step()
    model.spec_decode_step()  # a real K-row verify, so _verify_pending is K-shaped
    cur = int(ttnn.to_torch(model.t_tok).reshape(-1)[0])
    toks = [cur] * K
    pos = [model.pos + i for i in range(K)]

    model.verify_forward(toks, pos)  # warm (compile) OUTSIDE the measured region
    ttnn.synchronize_device(mesh)
    signpost("mtp_verify_start")
    model.verify_forward(toks, pos)
    ttnn.synchronize_device(mesh)
    signpost("mtp_verify_stop")

    model.commit_verify(1)
    ttnn.synchronize_device(mesh)
    signpost("mtp_commit_start")
    model.commit_verify(1)
    ttnn.synchronize_device(mesh)
    signpost("mtp_commit_stop")

    signpost("mtp_draft_start")
    model.mtp_draft(model._mtp_hidden, cur, gamma)
    ttnn.synchronize_device(mesh)
    signpost("mtp_draft_stop")
    print("[prof_mtp] regions emitted: mtp_verify / mtp_commit / mtp_draft")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ks", type=str, default="1,2,3,4")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--gamma", type=int, default=2)
    ap.add_argument("--components", action="store_true")
    ap.add_argument("--gdnbreak", action="store_true")
    ap.add_argument("--K", type=int, default=3)
    a = ap.parse_args()
    n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
    max_seq = int(os.environ.get("QWEN36_MAX_SEQ", "512"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=250_000_000)
    try:
        if a.gdnbreak:
            gdnbreak(mesh, n_layers, max_seq, a.K, a.iters)
        elif a.components:
            components(mesh, n_layers, max_seq, a.K, a.iters)
        elif a.sweep:
            sweep(mesh, n_layers, max_seq, [int(k) for k in a.ks.split(",")], a.iters)
        else:
            profile(mesh, n_layers, max_seq, a.gamma)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
