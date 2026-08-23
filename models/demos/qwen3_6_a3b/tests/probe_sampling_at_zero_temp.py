# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does the on-device sampling tail return the ARGMAX at temperature -> 0, on IDENTICAL logits?

This is the invariant `test_spec_sampling_at_zero_temperature_equals_greedy_spec` is really about, and
that test asserts it the hard way -- by comparing two full 24-token speculative TRAJECTORIES. A
trajectory comparison conflates two very different things:

  (1) the sampler disagreeing with argmax on the same logits   -> a BUG in the tail
  (2) the two runs drifting apart after one tie flips          -> compounding, not a defect

Its docstring claims it is immune to (2) -- "unlike a PCC gate it is not tie-sensitive: both paths
consume the same verify logits" -- but that only holds while the round structure agrees; once a tie
flips, the arms group different tokens per K-row verify pass and genuinely compute different logits.
So the test cannot distinguish (1) from (2), and it has been failing intermittently as unrelated
low-bit changes land.

This probe tests (1) directly: one hidden state, both selection tails, same step, no trajectory. Any
disagreement here is a real bug in the tail; agreement means the test failures are (2) and the test is
measuring determinism rather than correctness.

Run:  QWEN36_LAYERS=1 ./python_env/bin/python models/demos/qwen3_6_a3b/tests/probe_sampling_at_zero_temp.py
"""
from __future__ import annotations

import argparse
import os

import torch

import ttnn
from models.demos.qwen3_6_a3b.tt.common import from_tt, to_tt
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--temps", default="1e-4,1e-2,0.1")
    a = ap.parse_args()

    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "1"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        args = ModelArgs(mesh, ckpt_dir=ckpt, max_seq_len=256)
        model = TtModel(mesh, args, CheckpointLoader(ckpt), num_layers=n_layers)
        # A short prefill first: start_decode reads self.pos and needs the per-layer caches allocated.
        logits0 = model.forward(torch.randint(0, args.vocab_size, (1, 8)))
        model.start_decode(int(logits0[0, -1].argmax()))  # allocates t_tok / position / presence
        D = args.dim
        torch.manual_seed(0)

        # Pre-generate the hidden states so every temperature sees the SAME steps.
        xs = [torch.randn(1, D) * 0.6 for _ in range(a.trials)]
        print(
            f"\n{'temp':>8}{'argmax==sampled':>18}{'disagreements':>15}{'exact ties':>12}"
            f"{'gap@disagree':>14}{'rank of sampled':>18}"
        )
        for temp in [float(x) for x in a.temps.split(",")]:
            run_one(model, mesh, xs, temp, D)
    finally:
        ttnn.close_mesh_device(mesh)


def run_one(model, mesh, xs, temp, D):
    agree = ties = 0
    margins_dis, ranks = [], []
    for t in range(len(xs)):
        # A real post-norm hidden is O(1) per element; scale so the logit spread is realistic.
        x = to_tt(xs[t], mesh)

        model.sampling = None  # ---- greedy tail: argmax ----
        model._select_token(x)
        g = int(from_tt(model.t_tok, mesh).flatten()[0])

        # ---- sampling tail at T->0: chunked topk(sorted=False) + ttnn.sampling ----
        model.enable_sampling(temp, 0, 1.0, seed=1234 + t, presence_penalty=0.0)
        model._select_token(x)
        s = int(from_tt(model.t_tok, mesh).flatten()[0])

        # how close was this step to a genuine tie? (top-2 gap of the real logits)
        lg = from_tt(model._lmh(ttnn.reshape(x, [1, D])), mesh).flatten().float()
        order = torch.argsort(lg, descending=True)
        gap = (lg[order[0]] - lg[order[1]]).item()
        if g == s:
            agree += 1
        else:
            margins_dis.append(gap)
            # RANK of the sampled token in the true logit order. This is the diagnostic: rank 2 on a
            # ~1-ULP gap is benign tie-breaking; a large rank means the sampler picked a token the
            # target rates as unlikely, which at T->0 can only be a defect (or an exp overflow --
            # exp(logit/1e-4) is astronomically out of range, so a "greedy" request expressed as a
            # tiny temperature is not the same thing as the greedy path).
            ranks.append(int((order == s).nonzero()[0, 0]) + 1)
            if gap == 0.0:
                ties += 1
        model.sampling = None

    n = len(xs)
    gaps = f"{min(margins_dis):.4f}-{max(margins_dis):.4f}" if margins_dis else "-"
    rk = f"{min(ranks)}-{max(ranks)}" if ranks else "-"
    print(f"{temp:>8.0e}{f'{agree}/{n}':>18}{len(margins_dis):>15}{ties:>12}{gaps:>14}{rk:>18}")


if __name__ == "__main__":
    main()
