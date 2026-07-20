# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DECODE benchmark on the REAL Qwen3.6-35B-A3B model (actual checkpoint weights).

Two numbers, both on real weights:
  1. DEFINITIVE — the real full-model traced decode tok/s (build TtModel, prefill, capture the decode
     trace, time replays). This is the same path as `demo.py --trace`, minus generation, so it is the
     authoritative single-user decode number. With QWEN36_LAYERS=40 it is the real 40-layer figure.
  2. BREAKDOWN — per-layer-type ms by timing the model's OWN components (a real gated-delta mixer, a
     real attention mixer, a real MoE block, the real lm_head) each inside its own trace, × the
     per-token layer counts (30 gated-delta, 10 attention, 40 MoE, 1 head). Reconciles to (1).

(Timing is value-independent for these ops — dense matmuls don't depend on weight values and the MoE
always activates exactly top_k experts — so the breakdown matches the old random-weight microbench;
the point of real weights here is the DEFINITIVE end-to-end number and that it loads the production
dtypes/config exactly.)

Run (real weights; ~800 s load at 40 layers, ~80 s at 4):
    QWEN36_LAYERS=40 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/bench_decode.py
    QWEN36_LAYERS=4  ./python_env/bin/python models/demos/qwen3_6_a3b/tests/bench_decode.py --iters 300
Env: QWEN36_LAYERS (default 40), QWEN36_CKPT, QWEN36_MAX_SEQ, QWEN36_LMHEAD_BF4, QWEN36_MOE_NNZ.
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

N_GATED_DELTA = 30
N_ATTENTION = 10
N_MOE = 40
N_HEAD = 1


# --------------------------------------------------------------------------- timing
def time_traced(mesh_device, make_call, iters, warmup=3):
    """Capture one trace of make_call() and time `iters` replays. make_call: zero-arg, issues a
    module's decode ops on device-resident (persistent) inputs. Returns ms/op."""
    for _ in range(warmup):  # eager compile (capture must not JIT)
        make_call()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    make_call()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)  # warm replay
    ttnn.synchronize_device(mesh_device)
    t0 = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    dt = time.time() - t0
    ttnn.release_trace(mesh_device, tid)
    return dt / iters * 1e3  # ms/op


# --------------------------------------------------------------------------- build the real model
def build_model(mesh_device, n_layers, ckpt, max_seq, seq):
    args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_seq_len=max_seq)
    loader = CheckpointLoader(ckpt)
    t0 = time.time()
    model = TtModel(mesh_device, args, loader, num_layers=n_layers)
    print(f"[bench] built {n_layers}-layer model in {time.time() - t0:.0f}s")
    torch.manual_seed(0)
    prompt = torch.randint(0, args.vocab_size, (1, seq))
    logits = model.forward(prompt)  # prefill: allocates per-layer caches
    model.start_decode(int(logits[0, -1].argmax()))
    model.decode_step_eager()  # warmup: compile decode kernels
    return model, args


# --------------------------------------------------------------------------- definitive full-model
def bench_full_model(mesh_device, model, iters):
    """Headline: the real traced decode step. execute_trace = pure on-device; decode_step_traced adds
    the 1-element token readback (what demo.py reports as tok/s/user)."""
    model.capture_decode_trace()
    # pure device replay
    ttnn.execute_trace(mesh_device, model.trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    t0 = time.time()
    for _ in range(iters):
        ttnn.execute_trace(mesh_device, model.trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    dev_ms = (time.time() - t0) / iters * 1e3
    # realistic per-token (with readback), matching demo
    for _ in range(3):
        model.decode_step_traced()
    t0 = time.time()
    for _ in range(iters):
        model.decode_step_traced()
    full_ms = (time.time() - t0) / iters * 1e3
    ttnn.release_trace(mesh_device, model.trace_id)  # free the trace region for the component traces
    return dev_ms, full_ms


# --------------------------------------------------------------------------- per-component breakdown
def bench_components(mesh_device, model, args, iters):
    """Time the model's OWN real components, each in its own trace, on persistent dummy-shaped inputs
    (timing is value-independent). Returns dict of ms/op."""
    dim, rdim = args.dim, args.rotary_dim
    lin_idx = next(i for i, l in enumerate(model.layers) if l.is_linear)
    attn_idx = next((i for i, l in enumerate(model.layers) if not l.is_linear), None)
    res = {}

    def rand(*shape):
        return ttnn.from_torch(
            torch.randn(*shape) * 0.5,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # gated-delta mixer (real weights + the layer's real decode cache, in-place scan)
    gdn = model.layers[lin_idx].mixer
    x_g = rand(1, 1, 1, dim)
    cache_g = model.caches[lin_idx]
    res["gated-delta"] = time_traced(mesh_device, lambda: gdn.forward(x_g, cache_g), iters)

    # attention mixer
    if attn_idx is not None:
        attn = model.layers[attn_idx].mixer
        x_a = rand(1, 1, 1, dim)
        cos = rand(1, 1, 1, rdim)
        sin = rand(1, 1, 1, rdim)
        cur = ttnn.from_torch(
            torch.tensor([8], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        kv = model.caches[attn_idx]
        res["attention"] = time_traced(mesh_device, lambda: attn.forward_decode(x_a, cos, sin, kv, cur), iters)

    # MoE block (real experts, real router) — full forward + the router/sparse split
    moe = model.layers[lin_idx].moe
    x_m = rand(1, 1, 1, dim)
    res["moe"] = time_traced(mesh_device, lambda: moe.forward(x_m), iters)
    x2 = ttnn.reshape(x_m, [1, dim])
    res["moe.router"] = time_traced(mesh_device, lambda: moe._shared_and_router(x2), iters)
    _, topv, topi, probs = moe._shared_and_router(x2)
    routing = ttnn.scatter(ttnn.zeros_like(probs), 1, topi, topv)
    sparsity = ttnn.to_layout(ttnn.reshape(routing, [1, 1, 1, args.num_experts]), ttnn.ROW_MAJOR_LAYOUT)
    ttnn.synchronize_device(mesh_device)
    res["moe.sparse"] = time_traced(mesh_device, lambda: moe.forward_sparse_decode(x2, sparsity), iters)

    # lm_head (real weight) + multicore argmax (the production head stage)
    x_h = rand(1, dim)

    def head():
        logits = ttnn.linear(x_h, model.lm_head_w)
        logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.argmax(logits, dim=-1, keepdim=False, use_multicore=True)

    res["lm_head"] = time_traced(mesh_device, head, iters)
    return res


# --------------------------------------------------------------------------- driver
def run(mesh_device, n_layers, ckpt, max_seq, seq, iters, temperature=0.0, top_k=0, top_p=1.0, presence_penalty=0.0):
    model, args = build_model(mesh_device, n_layers, ckpt, max_seq, seq)

    print(f"\n[bench_decode] REAL weights, {n_layers}-layer model, {iters} traced replays\n")
    dev_ms, full_ms = bench_full_model(mesh_device, model, iters)
    print(f"  === DEFINITIVE full-model decode ({n_layers} layers) ===")
    print(f"  execute_trace (pure device) : {dev_ms:7.2f} ms/token  -> {1000/dev_ms:5.1f} tok/s")
    print(f"  decode_step (incl. readback): {full_ms:7.2f} ms/token  -> {1000/full_ms:5.1f} tok/s/user  <- demo metric")

    if temperature > 0:
        # Swap the greedy argmax tail for the on-device topk+sampling tail and re-time. Same model,
        # so the delta vs the greedy number above is purely the sampling head cost.
        model.enable_sampling(temperature, top_k, top_p, seed=0, presence_penalty=presence_penalty)
        model.decode_step_eager()  # compile the sampling kernels before capture
        s_dev_ms, s_full_ms = bench_full_model(mesh_device, model, iters)
        print(
            f"\n  === SAMPLING full-model decode (temp={temperature} top_k={top_k or 32} "
            f"top_p={top_p} presence_penalty={presence_penalty}) ==="
        )
        print(f"  execute_trace (pure device) : {s_dev_ms:7.2f} ms/token  -> {1000/s_dev_ms:5.1f} tok/s")
        print(f"  decode_step (incl. readback): {s_full_ms:7.2f} ms/token  -> {1000/s_full_ms:5.1f} tok/s/user")
        print(
            f"  sampling overhead vs greedy : {s_full_ms - full_ms:+7.2f} ms/token "
            f"({(s_full_ms / full_ms - 1) * 100:+.1f}%)"
        )

    c = bench_components(mesh_device, model, args, iters)
    rows = [
        ("Gated-DeltaNet", c.get("gated-delta", 0), N_GATED_DELTA),
        ("Attention", c.get("attention", 0), N_ATTENTION),
        ("MoE", c["moe"], N_MOE),
        ("lm_head", c["lm_head"], N_HEAD),
    ]
    print(f"\n  === per-layer-type breakdown (real weights) ===")
    for name, ms, n in rows:
        print(f"  {name:16s} {ms:7.3f} ms/op  x{n}")
    print(f"  {'(moe split)':16s} router {c['moe.router']:.3f} + sparse_decode {c['moe.sparse']:.3f}")
    est = sum(ms * n for _, ms, n in rows)
    print(f"\n  breakdown × layer counts (40-layer estimate): {est:6.1f} ms  -> {1000/est:5.1f} tok/s")
    print(f"  measured full-model ({n_layers} layers):       {full_ms:6.1f} ms  -> {1000/full_ms:5.1f} tok/s/user\n")


def test_bench_decode(mesh_device):
    """Pytest entry (uses the shared mesh_device fixture; QWEN36_LAYERS controls depth)."""
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "4"))
    run(mesh_device, n_layers, ckpt, int(os.environ.get("QWEN36_MAX_SEQ", "128")), seq=8, iters=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seq", type=int, default=8, help="prefill prompt length")
    ap.add_argument("--temperature", type=float, default=0.0, help=">0 also benches the sampling tail vs greedy")
    ap.add_argument("--top-k", type=int, default=0, help="top-k for the sampling bench (0/>32 = 32)")
    ap.add_argument("--top-p", type=float, default=1.0, help="top-p for the sampling bench")
    ap.add_argument("--presence-penalty", type=float, default=0.0, help="presence penalty for the sampling bench")
    a = ap.parse_args()
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))
    max_seq = int(os.environ.get("QWEN36_MAX_SEQ", "512"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        run(mesh, n_layers, ckpt, max_seq, a.seq, a.iters, a.temperature, a.top_k, a.top_p, a.presence_penalty)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
