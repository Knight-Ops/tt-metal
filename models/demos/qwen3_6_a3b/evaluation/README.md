# Accuracy evaluation — Qwen3.6-35B-A3B (single P150)

The single-card build ships **BFP4** routed experts. Per-module PCC gates prove each op is faithful, but
nothing measures the *end-to-end* accuracy cost of that quantization. Tenstorrent explicitly moved their
27B off bf4/bf8 to bf16 for accuracy — this harness produces the missing datapoint for the MoE build.

## `run_mmlu_bench.py` (self-contained — runs in the tt-metal python_env as-is)

Standard 0-shot MMLU-Redux letter log-prob scoring (compare next-token log-probs of A/B/C/D on the
last-token logits), plus TTFT and decode-TPS binned by prompt length. Depends only on `datasets` /
`huggingface_hub` (already installed) — no `lm-eval` needed.

```bash
# As shipped (BFP4 experts):
QWEN36_LAYERS=40 ./python_env/bin/python -u \
    models/demos/qwen3_6_a3b/evaluation/run_mmlu_bench.py --samples 100 --output /tmp/mmlu_bf4.json

# The BFP4-vs-BFP8 accuracy sweep (the deliverable):
QWEN36_LAYERS=40 QWEN36_EXPERT_DTYPE=bf4 ./python_env/bin/python -u .../run_mmlu_bench.py --samples 200 --output /tmp/mmlu_bf4.json
QWEN36_LAYERS=40 QWEN36_EXPERT_DTYPE=bf8 ./python_env/bin/python -u .../run_mmlu_bench.py --samples 200 --output /tmp/mmlu_bf8.json
#   bf8 experts need ~2x the DRAM; if it OOMs on one card, lower QWEN36_LAYERS.
# lm_head precision is a second knob: QWEN36_LMHEAD_BF4=0 (BFP8 head).
```

Record the resulting `overall_accuracy` per config in `published_scores.json`.

**Speed vs accuracy:** add `--trace` to use the bucketed traced prefill (strips host dispatch, much
faster per sample). CAVEAT: traced prefill runs the bf16 ttl GDN kernel, not the fp32-stable eager
path, so `--trace` measures the *traced path's* accuracy (may be lower than the eager default). Running
both is a useful way to quantify whether the traced serving path is accuracy-safe:

```bash
# eager (accurate, default) vs traced (fast) — compare overall_accuracy
QWEN36_LAYERS=40 ./python_env/bin/python -u .../run_mmlu_bench.py --samples 200 --output /tmp/mmlu_eager.json
QWEN36_LAYERS=40 ./python_env/bin/python -u .../run_mmlu_bench.py --samples 200 --trace --output /tmp/mmlu_traced.json
```

## `lm_eval_wrapper.py` (general lm-evaluation-harness path — needs an install)

Registers `qwen36_tt` with lm-eval so any task works (loglikelihood + generative). Loglikelihood uses
`TtModel.forward_prefill_all_logits`.

```bash
./python_env/bin/pip install lm-eval
QWEN36_LAYERS=40 ./python_env/bin/lm_eval --model qwen36_tt \
    --model_args max_layers=40 --tasks mmlu_redux --limit 100 --batch_size 1
```

## Precision knobs

- `QWEN36_EXPERT_DTYPE=bf4|bf8` — routed-expert + shared-MLP weight precision (default `bf4`).
- `QWEN36_LMHEAD_BF4=0` — BFP8 lm_head instead of BFP4.
- `QWEN36_LAYERS` — layer count (use fewer for quick iteration or if bf8 OOMs).
