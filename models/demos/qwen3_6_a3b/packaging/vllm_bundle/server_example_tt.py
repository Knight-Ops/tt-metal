#!/usr/bin/env python3

"""
Thin launcher for the vLLM OpenAI-compatible API server with the Tenstorrent plugin.

`tt-kernel serve` runs this verbatim (with EXTRA_MODELS_DIR pointing at this bundle's parent
and the bundle's `launch.env` overlaid). The TT vLLM plugin registers itself and its model
architectures on `import vllm` via vLLM's platform-plugin entry point, so this only needs to
hand argv through to vLLM's OpenAI server. Mirrors
`tt-vllm/plugins/vllm-tt-plugin/examples/server_example_tt.py`.

VERIFY-ON-DEVICE: confirm the installed plugin resolves the bundle's `arch`
(Qwen3_5MoeForConditionalGeneration → generator_vllm:Qwen36ForCausalLM) via EXTRA_MODELS_DIR,
and that this script is found on the serve host (tt-kernel runs the command with inherited
cwd; if it isn't the bundle dir, switch the launch command to
`["python3", "-m", "vllm.entrypoints.openai.api_server", ...]`).
"""

import runpy
import sys

from loguru import logger


def main() -> None:
    logger.info(f"[qwen36-vllm] launching vLLM OpenAI server: argv={sys.argv[1:]}")
    # Hand off to vLLM's OpenAI API server; argv (‑‑model/‑‑max-num-seqs/‑‑block-size/‑‑port)
    # is consumed by vLLM's arg parser. The TT plugin is active via its entry point.
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
