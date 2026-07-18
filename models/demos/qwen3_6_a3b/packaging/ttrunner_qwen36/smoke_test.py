# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Device-level contract smoke test for the Qwen3.6 dispatch runner.

Exercises the full runner contract on real hardware WITHOUT needing tt-api or a
tt-kernel push: constructs Qwen36Runner (opens the 1x1 mesh + builds the model +
warms up), then drives generate / generate_stream / benchmark and validates the
contract shapes.

Run (tt-metal python_env, where the P150 device env is set up):
    # test the reference runner (default):
    QWEN36_CKPT=~/models/qwen36 QWEN36_LAYERS=40 \
      ./python_env/bin/python models/demos/qwen3_6_a3b/packaging/ttrunner_qwen36/smoke_test.py

    # test the installed WHEEL instead (after `pip install --no-deps dist/...whl`):
    TTRUNNER_SOURCE=wheel ... ./python_env/bin/python .../smoke_test.py

Note: first run builds the ~20 GB .tensorbin weight cache (~916 s cold; ~120 s warm).
"""

import os
import sys


def _load_runner_class():
    src = os.environ.get("TTRUNNER_SOURCE", "reference").lower()
    if src == "wheel":
        from ttrunner_qwen36.runner import Qwen36Runner  # installed self-contained wheel

        return Qwen36Runner, "wheel (ttrunner_qwen36.runner)"
    from models.demos.qwen3_6_a3b.demo.runner import Qwen36Runner  # reference runner in the tree

    return Qwen36Runner, "reference (models.demos.qwen3_6_a3b.demo.runner)"


def main():
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    max_seq = int(os.environ.get("QWEN36_MAX_SEQ", "512"))
    Qwen36Runner, label = _load_runner_class()
    print(f"[smoke] runner source: {label}")
    print(f"[smoke] ckpt={ckpt} max_seq={max_seq}")

    # dispatch passes device=None because MANAGES_OWN_DEVICE is True
    runner = Qwen36Runner(ckpt, device=None, max_seq=max_seq)

    # --- required attributes ---
    assert runner._tokenizer is not None, "_tokenizer not set"
    assert isinstance(runner._listed, bool), "_listed missing/not bool"
    assert isinstance(runner._community, bool), "_community missing/not bool"
    print(f"[smoke] attrs OK: _listed={runner._listed} _community={runner._community}")

    prompt = "The capital of France is"

    # --- generate ---
    text = runner.generate(prompt, max_new_tokens=16, temperature=0.0, chat=False)
    assert isinstance(text, str) and text, f"generate returned {text!r}"
    print(f"[smoke] generate OK -> {text!r}")

    # --- generate_stream ---
    deltas, final = [], None
    for item in runner.generate_stream(prompt, max_new_tokens=16, temperature=0.6, chat=True):
        if isinstance(item, dict):
            final = item  # must be the LAST item
        else:
            assert isinstance(item, str), f"stream yielded non-str delta {item!r}"
            assert final is None, "a str delta arrived AFTER the usage dict"
            deltas.append(item)
    assert final is not None, "generate_stream did not yield a final usage dict"
    for k in ("finish_reason", "prompt_tokens", "completion_tokens"):
        assert k in final, f"usage dict missing {k!r}: {final}"
    assert final["prompt_tokens"] > 0 and final["completion_tokens"] >= 1, final
    print(f"[smoke] generate_stream OK -> {len(deltas)} deltas, usage={final}")
    print(f"[smoke] streamed text: {''.join(deltas)!r}")

    # --- benchmark ---
    tok_s, btext = runner.benchmark(prompt, n_tokens=32)
    assert isinstance(tok_s, float) and tok_s > 0, f"benchmark tok_s={tok_s}"
    assert isinstance(btext, str) and btext, "benchmark text empty"
    print(f"[smoke] benchmark OK -> {tok_s:.1f} tok/s")

    print("[smoke] PASS — all three contract methods validated on device.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
