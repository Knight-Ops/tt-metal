# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight OpenAI-format server for Qwen3.6-35B-A3B (single Blackhole).

Same request shape as gemma4's diffusion ``server.py``:
    POST /v1/chat/completions   {"messages":[...], "max_tokens":N, "stream":bool}
    POST /v1/completions        {"prompt":"...", "max_tokens":N, "stream":bool}
    GET  /v1/models
    GET  /health
Unlike the diffusion server, Qwen3.6 is autoregressive, so token-by-token SSE
streaming is supported (``"stream": true``). The response is OpenAI-shaped and
additionally carries ``tokens_per_second`` and ``generation_seconds``.

Generation is single-batch. Sampling is honored on device: ``temperature`` (with
``top_k`` / ``top_p`` / ``presence_penalty``) drives an on-device topk + ttnn.sampling
decode tail (no host-side logit readback); ``temperature == 0`` selects greedy argmax.
The request defaults are Qwen3's recommended "thinking mode" config (temperature 1.0,
top_k 20, top_p 0.95, presence_penalty 1.5); ``min_p`` / ``repetition_penalty`` are
accepted but not implemented (their recommended 0.0 / 1.0 are no-ops). Since the default
temperature is 1.0, requests that omit it sample — pass ``"temperature": 0`` for greedy.
The first token (from prefill) is always greedy argmax; decode tokens are sampled.
This is a testing server; for multi-user serving see ``generator_vllm.py``.

Launch (tt-metal python_env), standalone — opens the 1x1 mesh directly like demo.py:
    QWEN36_LAYERS=40 python models/demos/qwen3_6_a3b/demo/server.py
Env knobs: QWEN36_SERVER_HOST (0.0.0.0), QWEN36_SERVER_PORT (8000),
QWEN36_CKPT, QWEN36_LAYERS (40), QWEN36_MAX_SEQ (512),
QWEN36_SERVER_TRACE (1 = capture a decode trace per request for the fast path).
"""

import json
import os
import threading
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

import ttnn
from models.demos.qwen3_6_a3b.tt.load_checkpoints import CheckpointLoader
from models.demos.qwen3_6_a3b.tt.model import TtModel
from models.demos.qwen3_6_a3b.tt.model_config import ModelArgs

MODEL_ID = "qwen3.6-a3b"


# ── OpenAI-format request schemas (subset, matching gemma4's server.py) ───────


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None


# Defaults below are Qwen3's recommended "thinking mode" generation config. Of these, temperature,
# top_k, top_p and presence_penalty are applied on device; min_p and repetition_penalty are accepted
# for API compatibility but not implemented (their recommended values 0.0 / 1.0 are no-ops anyway).
class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: float = 1.0  # 0 = greedy argmax; >0 = on-device temperature sampling
    top_k: int = 20  # 0 (or >32) = 32, the device max
    top_p: float = 0.95  # nucleus probability
    min_p: float = 0.0  # accepted; not implemented (0.0 = no-op)
    presence_penalty: float = 1.5  # subtract from logits of already-generated tokens (on device)
    repetition_penalty: float = 1.0  # accepted; not implemented (1.0 = no-op)
    seed: Optional[int] = None  # deterministic per seed; None -> 0
    stream: bool = False


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    stream: bool = False


def _flatten(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


# ── Engine ────────────────────────────────────────────────────────────────────


class Qwen36Engine:
    """Single-batch Qwen3.6 wrapper. One request at a time (device decode state is
    single-user), so the HTTP layer serializes generation via ``self.lock``."""

    def __init__(self, mesh_device, ckpt: str, n_layers: int, max_seq: int, use_trace: bool):
        from transformers import AutoTokenizer

        self.mesh_device = mesh_device
        self.max_seq = max_seq
        self.use_trace = use_trace

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.eos_ids = self._resolve_eos_ids(ckpt)

        args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_seq_len=max_seq)
        loader = CheckpointLoader(ckpt)
        logger.info(f"Building {n_layers}-layer model ({args.model_name})...")
        t0 = time.time()
        self.model = TtModel(mesh_device, args, loader, num_layers=n_layers)
        logger.info(f"Qwen3.6 model built in {time.time() - t0:.1f}s")
        self.lock = threading.Lock()

    def _resolve_eos_ids(self, ckpt: str) -> set:
        ids = set()
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            ids.add(int(eos))
        # generation_config.json may list extra stop ids (e.g. <|im_end|>)
        try:
            with open(os.path.join(ckpt, "generation_config.json")) as f:
                gc = json.load(f)
            g_eos = gc.get("eos_token_id")
            if isinstance(g_eos, int):
                ids.add(g_eos)
            elif isinstance(g_eos, list):
                ids.update(int(x) for x in g_eos)
        except Exception:
            pass
        return ids

    def ids_from_messages(self, messages: List[ChatMessage]) -> torch.Tensor:
        if self.tokenizer.chat_template:
            out = self.tokenizer.apply_chat_template(
                [{"role": m.role, "content": _flatten(m.content)} for m in messages],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            ids = (out["input_ids"] if not isinstance(out, torch.Tensor) else out).squeeze(0)
        else:
            text = "\n".join(_flatten(m.content) for m in messages)
            ids = self.tokenizer.encode(text, return_tensors="pt").squeeze(0)
        return ids

    def _generate_ids(
        self,
        ids: torch.Tensor,
        max_new: int,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
    ) -> Iterator[int]:
        """Core generator: prefill then yield one token id per decode step.

        ``temperature == 0`` is greedy argmax (on device); ``> 0`` enables the on-device
        temperature/top-k/top-p (+ presence_penalty) sampling tail. ``min_p`` and
        ``repetition_penalty`` are accepted but not implemented (no-ops at their default
        0.0 / 1.0). Mirrors demo.py's loop. Stops on an EOS id, on ``max_new``, or when the
        KV / state cache would overflow ``max_seq``. Holds no lock itself — callers serialize.
        """
        if min_p > 0 or repetition_penalty != 1.0:
            logger.warning(f"min_p={min_p} / repetition_penalty={repetition_penalty} are not implemented; ignoring.")
        ids = ids.reshape(1, -1)
        prompt_len = int(ids.shape[1])
        if prompt_len >= self.max_seq:
            raise ValueError(
                f"prompt length {prompt_len} >= max_seq_len {self.max_seq}; "
                "raise QWEN36_MAX_SEQ or shorten the prompt"
            )
        # leave room: prefill consumes prompt_len positions, each decode step consumes one
        budget = min(max_new, self.max_seq - prompt_len - 1)
        if budget <= 0:
            return

        # Select greedy vs sampling for this request's decode steps (must precede start_decode
        # so the captured trace records the right tail). temperature==0 -> greedy.
        self.model.enable_sampling(temperature, top_k, top_p, seed or 0, presence_penalty)

        logits = self.model.forward(ids)  # prefill -> [1, 1, vocab] host logits
        next_id = int(logits[0, -1].argmax())  # first token is greedy argmax
        self.model.start_decode(next_id)

        traced = False
        for step in range(budget):
            if next_id in self.eos_ids:
                return
            yield next_id
            if self.use_trace and not traced:
                next_id = self.model.decode_step_eager()  # warmup compiles kernels
                self.model.capture_decode_trace()
                traced = True
            elif self.use_trace:
                next_id = self.model.decode_step_traced()
            else:
                next_id = self.model.decode_step_eager()

    def generate(self, ids: torch.Tensor, max_new: int, **sampling):
        """Drain ``_generate_ids`` into a full completion.

        ``sampling`` = temperature/top_k/top_p/seed (forwarded to ``_generate_ids``).
        Returns (text, completion_tokens, gen_seconds, tokens_per_second).
        """
        t0 = time.time()
        out_ids = list(self._generate_ids(ids, max_new, **sampling))
        secs = time.time() - t0
        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        tok_s = len(out_ids) / secs if secs > 0 else 0.0
        return text, len(out_ids), secs, tok_s

    def stream_text(self, ids: torch.Tensor, max_new: int, **sampling) -> Iterator[str]:
        """Yield incremental decoded text deltas (handles multi-byte tokens by
        re-decoding the running id list and emitting the new suffix)."""
        out_ids: List[int] = []
        prev = ""
        for tid in self._generate_ids(ids, max_new, **sampling):
            out_ids.append(tid)
            text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            if len(text) > len(prev):
                yield text[len(prev) :]
                prev = text


# ── HTTP app ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Qwen3.6-A3B server")


def _usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


def _default_max_new(eng: Qwen36Engine, req_max: Optional[int]) -> int:
    return req_max if req_max and req_max > 0 else max(1, eng.max_seq // 4)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _stream_chat(eng: Qwen36Engine, ids: torch.Tensor, max_new: int, model: str, **sampling) -> Iterator[str]:
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
    with eng.lock:
        # first chunk announces the assistant role
        yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        for piece in eng.stream_text(ids, max_new, **sampling):
            yield _sse({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
    yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    yield "data: [DONE]\n\n"


def _stream_completion(eng: Qwen36Engine, ids: torch.Tensor, max_new: int, model: str, **sampling) -> Iterator[str]:
    cid = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    base = {"id": cid, "object": "text_completion", "created": created, "model": model}
    with eng.lock:
        for piece in eng.stream_text(ids, max_new, **sampling):
            yield _sse({**base, "choices": [{"index": 0, "text": piece, "finish_reason": None}]})
    yield _sse({**base, "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]})
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    eng: Qwen36Engine = app.state.engine
    ids = eng.ids_from_messages(req.messages)
    max_new = _default_max_new(eng, req.max_tokens)
    model = req.model or MODEL_ID
    if int(ids.numel()) >= eng.max_seq:
        raise HTTPException(status_code=400, detail=f"prompt too long ({int(ids.numel())} >= {eng.max_seq})")

    sampling = {
        "temperature": req.temperature,
        "top_k": req.top_k,
        "top_p": req.top_p,
        "presence_penalty": req.presence_penalty,
        "min_p": req.min_p,
        "repetition_penalty": req.repetition_penalty,
        "seed": req.seed,
    }
    if req.stream:
        return StreamingResponse(_stream_chat(eng, ids, max_new, model, **sampling), media_type="text/event-stream")

    with eng.lock:
        text, n_tok, secs, tok_s = eng.generate(ids, max_new, **sampling)
    logger.info(f"[chat] {n_tok} tok in {secs:.2f}s = {tok_s:.1f} tok/s")
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": _usage(int(ids.numel()), n_tok),
        "tokens_per_second": round(tok_s, 2),
        "generation_seconds": round(secs, 3),
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    eng: Qwen36Engine = app.state.engine
    ids = eng.tokenizer.encode(req.prompt, return_tensors="pt").squeeze(0)
    max_new = _default_max_new(eng, req.max_tokens)
    model = req.model or MODEL_ID
    if int(ids.numel()) >= eng.max_seq:
        raise HTTPException(status_code=400, detail=f"prompt too long ({int(ids.numel())} >= {eng.max_seq})")

    sampling = {
        "temperature": req.temperature,
        "top_k": req.top_k,
        "top_p": req.top_p,
        "presence_penalty": req.presence_penalty,
        "min_p": req.min_p,
        "repetition_penalty": req.repetition_penalty,
        "seed": req.seed,
    }
    if req.stream:
        return StreamingResponse(
            _stream_completion(eng, ids, max_new, model, **sampling), media_type="text/event-stream"
        )

    with eng.lock:
        text, n_tok, secs, tok_s = eng.generate(ids, max_new, **sampling)
    logger.info(f"[cmpl] {n_tok} tok in {secs:.2f}s = {tok_s:.1f} tok/s")
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": _usage(int(ids.numel()), n_tok),
        "tokens_per_second": round(tok_s, 2),
        "generation_seconds": round(secs, 3),
    }


# ── standalone entrypoint (opens the 1x1 mesh directly, like demo.py) ──────────


def main():
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))
    max_seq = int(os.environ.get("QWEN36_MAX_SEQ", "512"))
    use_trace = os.environ.get("QWEN36_SERVER_TRACE", "1") != "0"
    host = os.environ.get("QWEN36_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("QWEN36_SERVER_PORT", "8000"))

    # Trace capture needs a nonzero trace region; eager-only can use 0.
    trace_region = 200 * 1024 * 1024 if use_trace else 0
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=trace_region)
    try:
        engine = Qwen36Engine(mesh, ckpt, n_layers, max_seq, use_trace)
        app.state.engine = engine

        # Warm up: first forward JIT-compiles every op (and captures the decode trace), so the first
        # real request reports a representative tok/s. Warm BOTH the greedy and the sampling decode
        # tails (their kernels differ) — the default request samples, so without this the first
        # sampling request would pay the sampling-kernel compile (~seconds).
        logger.info("Warming up (greedy + sampling)...")
        t0 = time.time()
        warm_ids = engine.tokenizer.encode("Hello", return_tensors="pt").squeeze(0)
        engine.generate(warm_ids, 4)  # greedy tail
        engine.generate(warm_ids, 4, temperature=1.0, top_k=20, top_p=0.95, presence_penalty=1.5)  # sampling tail
        logger.info(f"Warmup done in {time.time() - t0:.1f}s")

        logger.info(
            f"Serving Qwen3.6-A3B on http://{host}:{port}  "
            f"(POST /v1/chat/completions | /v1/completions; trace={'on' if use_trace else 'off'})"
        )
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
