# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
lm-evaluation-harness adapter for Qwen3.6-35B-A3B on a single Blackhole P150.

Adapts the bespoke ``TtModel`` to the ``lm_eval.api.model.LM`` interface so the model can be scored on
any lm-eval task (MMLU, GPQA, ...). Loglikelihood scoring uses ``TtModel.forward_prefill_all_logits``
(per-position logits); generative tasks use the model's greedy on-device decode.

For a self-contained accuracy+perf run that needs NO extra install, use ``run_mmlu_bench.py`` instead —
this wrapper requires ``pip install lm-eval`` (not in the tt-metal python_env by default).

Usage (after ``pip install lm-eval`` into ./python_env):
    QWEN36_LAYERS=40 lm_eval --model qwen36_tt \
        --model_args max_layers=40 --tasks mmlu_redux --limit 100 --batch_size 1

    # BFP4-vs-BFP8 sweep: prefix with QWEN36_EXPERT_DTYPE=bf4 / bf8.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm


@register_model("qwen36_tt")
class Qwen36TTEvalWrapper(LM):
    def __init__(
        self,
        max_layers: int | str | None = None,
        ckpt_dir: str | None = None,
        max_length: int = 4096,
        max_seq_len: int | None = None,
        **kwargs,
    ):
        super().__init__()
        import os

        from transformers import AutoTokenizer

        import ttnn
        from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
        from models.demos.qwen3_6_a3b.tt.model import TtModel
        from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs

        ckpt = ckpt_dir or os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
        n_layers = int(max_layers) if max_layers is not None else int(os.environ.get("QWEN36_LAYERS", "40"))
        ms = int(max_seq_len) if max_seq_len is not None else int(os.environ.get("QWEN36_MAX_SEQ", str(max_length)))
        self._max_length = min(max_length, ms)

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
        self.model_args = ModelArgs(self.mesh, ckpt_dir=ckpt, max_seq_len=ms)
        loader = CheckpointLoader(ckpt)
        self.model = TtModel(self.mesh, self.model_args, loader, num_layers=n_layers)
        self.vocab_size = self.model_args.vocab_size

    # ---- lm_eval.LM required properties ----
    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return "tt"

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        ctx_ids = self.tok_encode(context)
        cont_ids = self.tok_encode(continuation)
        total = len(ctx_ids) + len(cont_ids)
        if total > self._max_length:  # keep the continuation, truncate the front of the context
            ctx_ids = ctx_ids[-(self._max_length - len(cont_ids)) :]
        return ctx_ids, cont_ids

    # ---- scoring ----
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results = []
        for instance in tqdm(requests, desc="loglikelihood"):
            context, continuation = instance.arguments
            ctx_ids, cont_ids = self._encode_pair(context, continuation)
            full_ids = torch.tensor([ctx_ids + cont_ids], dtype=torch.long)
            cont_len = len(cont_ids)
            T = full_ids.shape[1]
            # Only the last (cont_len+1) positions' logits are needed to score the continuation, so
            # slice before the 248k-vocab matmul (bounds compute + D2H to the short continuation).
            n_keep = min(T, cont_len + 1)
            logits = self.model.forward_prefill_all_logits(full_ids, last_n=n_keep)  # torch [n_keep, vocab]
            log_probs = F.log_softmax(logits[:, : self.vocab_size].float(), dim=-1)
            shift = cont_len + 1 - n_keep  # 0 unless the context is empty (row for cont token i is i-shift)
            total_ll, is_greedy = 0.0, True
            for i in range(cont_len):
                row = max(i - shift, 0)
                total_ll += log_probs[row, cont_ids[i]].item()
                if int(torch.argmax(log_probs[row]).item()) != cont_ids[i]:
                    is_greedy = False
            results.append((total_ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results = []
        for instance in tqdm(requests, desc="loglikelihood_rolling"):
            token_ids = self.tok_encode(instance.arguments[0])[: self._max_length]
            full_ids = torch.tensor([token_ids], dtype=torch.long)
            logits = self.model.forward_prefill_all_logits(full_ids)  # torch [T, vocab]
            log_probs = F.log_softmax(logits[:, : self.vocab_size].float(), dim=-1)
            total_ll = sum(log_probs[i - 1, token_ids[i]].item() for i in range(1, len(token_ids)))
            results.append((total_ll, True))
        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []
        for instance in tqdm(requests, desc="generate_until"):
            context = instance.arguments[0]
            gen_kwargs = instance.arguments[1] if len(instance.arguments) > 1 else {}
            until = gen_kwargs.get("until") or [self.tokenizer.eos_token]
            max_gen = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))

            ids = torch.tensor([self.tok_encode(context)], dtype=torch.long)
            if ids.shape[1] > self._max_length - max_gen:
                ids = ids[:, -(self._max_length - max_gen) :]

            logits = self.model.forward(ids)  # last-token logits [1,1,vocab]
            next_id = int(logits[0, -1][: self.vocab_size].argmax())
            out_ids = [next_id]
            self.model.start_decode(next_id)
            for _ in range(max_gen - 1):
                next_id = self.model.decode_step_eager()
                if next_id == self.eot_token_id:
                    break
                out_ids.append(next_id)

            text = self.tok_decode(out_ids)
            for stop in until:
                if stop and stop in text:
                    text = text[: text.index(stop)]
                    break
            results.append(text)
        return results

    def close(self):
        import ttnn

        if getattr(self, "mesh", None) is not None:
            ttnn.close_mesh_device(self.mesh)
            self.mesh = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
