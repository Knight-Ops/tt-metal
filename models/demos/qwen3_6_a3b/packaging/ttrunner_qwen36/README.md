# ttrunner-qwen36

Self-contained [tt-kernel](https://github.com/tenstorrent/tt-kernel) **dispatch** runner
for **Qwen3.6-35B-A3B** (Qwen3-Next hybrid gated-delta + full-attention MoE) on a single
Tenstorrent **Blackhole P150**, tt-nn.

It is a renamespaced, self-contained copy of `models/demos/qwen3_6_a3b` (tt-metal) that
implements the runner contract in `tt-kernel-package-manager/docs/authoring_runners.md`:
`generate` / `generate_stream` / `benchmark`, wrapping the validated `Qwen36Engine`
generation loop. The class opens its own 1×1 mesh (`MANAGES_OWN_DEVICE = True`).

## Layout
- `ttrunner_qwen36/runner.py` — `Qwen36Runner`, the contract class.
- `ttrunner_qwen36/engine.py` — `Qwen36Engine`, the prefill + traced-decode + on-device
  sampling loop, extracted from the tt-metal server (FastAPI-free).
- `ttrunner_qwen36/tt/`, `reference/` — the renamespaced model implementation.
- `ttrunner_qwen36/_vendor/` — verbatim tt-metal `models.common` shims
  (`lightweightmodule`, `moe_gather`); see `NOTICE`.

## Build & push
```bash
python -m build --wheel        # -> dist/ttrunner_qwen36-0.1.0-py3-none-any.whl
tt-kernel push <ns>/qwen3.6-a3b-blackhole --private \
  --python-package dist/ttrunner_qwen36-0.1.0-py3-none-any.whl \
  --runner-spec ttrunner_qwen36.runner:Qwen36Runner \
  --entry-point qwen3_6_a3b \
  --weights Qwen/Qwen3.6-35B-A3B
```

## Serving-env prerequisites (co-versioned with the tt-metal build used to push)
- `ttnn` **and** `ttl` (tt-lang 1.1.3) present — the platform, not wheel dependencies.
- `transformers==4.53.0` (the `--no-deps` install cannot enforce this — operator must).
- Blackhole firmware **≥ 19.5.0** (19.6.0 recommended); older FW deadlocks the first
  SDPA-decode.
- Weights dir must contain `config.json` (with `text_config`),
  `model.safetensors.index.json` + shards, tokenizer files, `generation_config.json`.
- First run builds the ~20 GB `.tensorbin` weight cache (~916 s cold → ~120 s warm); size
  the readiness timeout accordingly. Only bf4 experts fit a 32 GB P150.
