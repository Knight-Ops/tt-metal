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
The request defaults are Qwen3's recommended "thinking mode" config (temperature 0.6,
top_k 20, top_p 0.95, presence_penalty 1.5); ``min_p`` / ``repetition_penalty`` are
accepted but not implemented (their recommended 0.0 / 1.0 are no-ops). Qwen3 thinking
models should NOT be run greedy (it degrades into repetition), so the default samples at
0.6 — pass ``"temperature": 0`` only if you explicitly want deterministic greedy.
The first token (from prefill) is always greedy argmax; decode tokens are sampled.

Tool/function calling is supported via Qwen3's native XML format. Pass OpenAI-style ``tools``
(and optional ``tool_choice``) on ``/v1/chat/completions``; the model's ``<tool_call>`` blocks
are parsed back into OpenAI ``message.tool_calls`` (with ``finish_reason == "tool_calls"``),
and assistant ``tool_calls`` / ``role:"tool"`` results in the request history are rendered back
into the prompt. ``tool_choice`` honors "auto" (default), "none", and a specific
{"type":"function","function":{"name":...}}; "required" is best-effort (the format is not
grammar-forced). Because thinking mode is on, chat responses split the model's reasoning into a
separate ``reasoning_content`` field and return only the post-</think> text as ``content``.
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

# Opt-in wire logging (QWEN36_LOG_REQUESTS=1): dump each chat request's shape and the model's raw
# output. Use it to capture exactly what an agentic client (e.g. OpenCode) sends/receives —
# whether it streams, what max_tokens/tools/tool_choice it sets, the prompt length, the
# finish_reason, and the parsed tool calls — when debugging tool-calling interop.
_LOG_REQUESTS = os.environ.get("QWEN36_LOG_REQUESTS") == "1"


def _log_chat_request(req: "ChatCompletionRequest", prompt_tokens: int, max_seq: int):
    if not _LOG_REQUESTS:
        return
    tool_names = [(t.get("function", t) or {}).get("name") for t in (req.tools or [])]
    logger.info(
        "[req] stream={} max_tokens={} temp={} pp={} prompt_tokens={}/{} msgs={} tools={} tool_choice={}".format(
            req.stream,
            req.max_tokens,
            req.temperature,
            req.presence_penalty,
            prompt_tokens,
            max_seq,
            len(req.messages),
            tool_names,
            req.tool_choice,
        )
    )


def _log_chat_response(raw_text: str, finish: str, tool_calls):
    if not _LOG_REQUESTS:
        return
    names = [tc["function"]["name"] for tc in (tool_calls or [])]
    logger.info(f"[resp] finish_reason={finish} tool_calls={names}")
    logger.info(f"[resp] raw_output (first 800 chars): {raw_text[:800]!r}")


# ── OpenAI-format request schemas (subset, matching gemma4's server.py) ───────


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None  # tool messages: the function name that produced this result
    tool_call_id: Optional[str] = None  # tool messages: id of the call this result answers
    tool_calls: Optional[List[Dict[str, Any]]] = None  # assistant messages: prior tool calls (history)


# Defaults below are Qwen3's recommended "thinking mode" generation config. Of these, temperature,
# top_k, top_p and presence_penalty are applied on device; min_p and repetition_penalty are accepted
# for API compatibility but not implemented (their recommended values 0.0 / 1.0 are no-ops anyway).
class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: float = 0.6  # Qwen3 thinking-mode rec; 0 = greedy argmax (discouraged); >0 = sampling
    top_k: int = 20  # 0 (or >32) = 32, the device max
    top_p: float = 0.95  # nucleus probability
    min_p: float = 0.0  # accepted; not implemented (0.0 = no-op)
    presence_penalty: float = 1.5  # subtract from logits of already-generated tokens (on device)
    repetition_penalty: float = 1.0  # accepted; not implemented (1.0 = no-op)
    seed: Optional[int] = None  # deterministic per seed; None -> 0
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None  # up to N stop strings; output truncated before the match
    # Tool/function calling (Qwen3 native XML format). ``tools`` is the OpenAI list of
    # {"type":"function","function":{name,description,parameters}}. ``tool_choice`` accepts
    # "auto" (default), "none" (tools hidden from the model), "required" (best-effort: the model
    # is told tools exist but the format is not grammar-forced), or {"type":"function",
    # "function":{"name":...}} to narrow the offered tools to a single function.
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # Qwen3 thinking control. None => AUTO: OFF when ``tools`` are present (non-thinking is the reliable
    # agentic/tool-calling config — thinking can burn the token budget before the tool call and slows
    # every turn), ON otherwise (Qwen3's default). Explicit true/false overrides. Also honored via
    # ``chat_template_kwargs={"enable_thinking": bool}`` for clients that pass it that way.
    enable_thinking: Optional[bool] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = None
    temperature: float = 0.6
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None  # up to N stop strings; output truncated before the match


def _flatten(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


# ── Tool-call (de)serialization (Qwen3 native XML format) ──────────────────────
#
# Qwen3 emits tool calls as XML, NOT Hermes JSON:
#     <tool_call>
#     <function=NAME>
#     <parameter=PNAME>
#     VALUE          (raw text for string params; JSON for everything else)
#     </parameter>
#     ...
#     </function>
#     </tool_call>
# Thinking mode is on (the prompt ends with "<think>\n"), so generation is:
#     <reasoning...></think>\n\n<visible text and/or one-or-more <tool_call> blocks>
# None of <think>/<tool_call>/<function=>/<parameter=> are special tokens, so they survive
# decode(skip_special_tokens=True) and can be parsed straight from the decoded text.

THINK_END = "</think>"
TOOL_OPEN = "<tool_call>"
FUNC_OPEN = "<function="
FUNC_CLOSE = "</function>"
PARAM_OPEN = "<parameter="
PARAM_CLOSE = "</parameter>"
TC_CLOSE = "</tool_call>"
_VALUE_HOLDBACK = len("\n" + PARAM_CLOSE) - 1  # withhold this many chars so a forming close never leaks


def _json_str_frag(s: str) -> str:
    """JSON-escape a fragment of a string for placement between quotes. JSON string escaping is
    per-character, so escaping fragments independently and concatenating equals escaping the
    whole string — which is what lets us stream a string argument value as it is produced."""
    return json.dumps(s, ensure_ascii=False)[1:-1]


def _coerce_arg(value: str, ptype: Optional[str]):
    """Recover a parameter's Python value. The template renders string params raw and all
    other types via tojson, so for non-string params we JSON-parse; on any failure (or an
    unknown/declared-string type) we keep the raw text."""
    if ptype == "string":
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _arg_types(tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Map function name -> {param name -> declared JSON type} from the request's tools, used
    to coerce parsed parameter values back to their schema type."""
    types: Dict[str, Dict[str, Any]] = {}
    for t in tools or []:
        fn = t.get("function", t)
        name = fn.get("name")
        props = (fn.get("parameters") or {}).get("properties") or {}
        if name:
            types[name] = {k: (v or {}).get("type") for k, v in props.items()}
    return types


def _norm_stop(stop) -> List[str]:
    """Normalize the OpenAI ``stop`` field (str | list[str] | None) to a list of non-empty strings."""
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [s for s in stop if isinstance(s, str) and s]


def _earliest_stop(text: str, stops: List[str]) -> Optional[int]:
    """Index of the earliest occurrence of any stop string in ``text``, or None."""
    cut: Optional[int] = None
    for s in stops:
        i = text.find(s)
        if i != -1 and (cut is None or i < cut):
            cut = i
    return cut


def _parse_chat_output(text: str, tools: Optional[List[Dict[str, Any]]] = None, expect_thinking: bool = True):
    """Full parse of a completed generation -> (reasoning, visible_content, tool_calls).

    Drives the SAME ``_ChatStreamParser`` used for streaming over the full text in one shot
    (push + finish), so the streaming and non-streaming endpoints produce identical results.
    Crucially this handles a generation truncated mid ``<tool_call>`` (finish_reason="length",
    e.g. a long file written as a tool argument): ``finish()`` closes the partial arguments JSON
    so the (incomplete but valid) tool call is still surfaced — instead of being dropped by a
    regex that requires a closing ``</tool_call>`` and leaking the half-written call as content."""
    parser = _ChatStreamParser(tools, expect_thinking=expect_thinking)
    reasoning_parts: List[str] = []
    content_parts: List[str] = []
    calls: Dict[int, Dict[str, Any]] = {}
    for kind, payload in [*parser.push(text), *parser.finish()]:
        if kind == "reasoning":
            reasoning_parts.append(payload)
        elif kind == "content":
            content_parts.append(payload)
        else:  # ("tool", <OpenAI tool_calls delta>): first delta carries id/type/name, rest append args
            c = calls.setdefault(
                payload["index"], {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if payload.get("id"):
                c["id"] = payload["id"]
            fn = payload.get("function", {})
            if "name" in fn:
                c["function"]["name"] = fn["name"]
            if "arguments" in fn:
                c["function"]["arguments"] += fn["arguments"]
    tool_calls = [calls[i] for i in sorted(calls)]
    return "".join(reasoning_parts).strip(), "".join(content_parts).strip(), tool_calls


def _normalize_history_tool_call(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAI tool_call from request history into the shape Qwen's chat template
    expects. The template iterates ``arguments | items`` (a mapping), but OpenAI sends
    ``arguments`` as a JSON string, so parse it back to a dict."""
    fn = tc.get("function", tc)
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except Exception:
            args = {}
    elif not isinstance(args, dict):
        args = {}
    return {"function": {"name": fn.get("name"), "arguments": args}}


class _ChatStreamParser:
    """Incrementally turns Qwen3's streamed generation into OpenAI-shaped deltas:

      reasoning text   -> ("reasoning", str)
      visible text     -> ("content", str)
      <tool_call> XML  -> ("tool", openai_tool_call_delta)

    Crucially the tool call is emitted *as it is produced* (function name first, then the
    arguments JSON built fragment-by-fragment) rather than buffered until generation ends — so a
    client sees a large tool argument (e.g. a whole file written via a tool) stream live instead
    of after minutes of silence. String-typed arguments stream incrementally; non-string /
    unknown args are buffered per-parameter and coerced to their declared type, matching the
    non-streaming parser. A tail is always held back so a partial ``</think>``, ``<tool_call>``,
    or ``</parameter>`` marker never leaks into output. ``finish()`` flushes anything held back
    and, if generation was truncated mid-call, closes the arguments JSON so it stays valid."""

    # tool-parsing sub-states (active once the body's first <tool_call> is seen)
    _SEEK_FUNC = "seek_func"  # looking for <function=NAME> (also skips a closed call's </tool_call>)
    _IN_FUNC = "in_func"  # inside a function: next is <parameter=...> or </function>
    _IN_STR = "in_str"  # streaming a string-typed parameter value
    _IN_BUF = "in_buf"  # buffering a non-string/unknown value until </parameter>

    def __init__(self, tools: Optional[List[Dict[str, Any]]] = None, expect_thinking: bool = True):
        self._types = _arg_types(tools)
        self._full = ""
        # reasoning / content. When thinking is disabled the prompt already contains the closed
        # <think></think>, so the model's OUTPUT has no </think> — start with the think block already
        # "closed" so the whole output is parsed as body (content + tool_calls). Otherwise the parser
        # would wait forever for a </think> that never comes and never emit the tool call.
        self._think_closed = not expect_thinking
        self._body_at = 0
        self._reason_emit = 0
        self._content_emit = 0
        self._content_started = False
        # tool calls
        self._in_tools = False
        self._cur = 0  # absolute cursor into _full for tool parsing
        self._tc_state = self._SEEK_FUNC
        self._tc_index = -1
        self._any_tool = False
        self._func_name = ""
        self._first_param = True
        self._args_open = False  # emitted "{" for the current function but not yet "}"
        self._param_name = ""
        self._param_type: Optional[str] = None
        self._param_prefix = ""
        self._val_start = 0  # absolute index where the current value begins
        self._val_emit = 0  # chars of the current string value already emitted

    @property
    def any_tool(self) -> bool:
        return self._any_tool

    def _content_delta(self, abs_end: int, events: list):
        if abs_end <= self._content_emit:
            return
        chunk = self._full[self._content_emit : abs_end]
        self._content_emit = abs_end
        if not self._content_started:
            chunk = chunk.lstrip("\n")  # drop the "\n\n" the template puts after </think>
            if not chunk:
                return
            self._content_started = True
        events.append(("content", chunk))

    def _tool(self, events: list, **fn):
        """Append an OpenAI tool_calls delta for the current call index."""
        delta = {"index": self._tc_index, **fn}
        events.append(("tool", delta))

    def push(self, full: str):
        """Feed the full decoded text so far; return a list of (kind, payload) events."""
        self._full = full
        events: list = []
        if not self._think_closed:
            i = full.find(THINK_END)
            if i == -1:
                # still inside reasoning; hold back a tail that might be a partial "</think>"
                safe = max(0, len(full) - (len(THINK_END) - 1))
                if safe > self._reason_emit:
                    events.append(("reasoning", full[self._reason_emit : safe]))
                    self._reason_emit = safe
                return events
            if i > self._reason_emit:
                events.append(("reasoning", full[self._reason_emit : i]))
            self._think_closed = True
            self._body_at = i + len(THINK_END)
            self._content_emit = self._body_at
        if not self._in_tools:
            # content mode: emit up to the first <tool_call>, holding back a partial-tag tail
            j = full.find(TOOL_OPEN, self._body_at)
            if j == -1:
                self._content_delta(max(self._body_at, len(full) - (len(TOOL_OPEN) - 1)), events)
                return events
            self._content_delta(j, events)
            self._in_tools = True
            self._cur = j + len(TOOL_OPEN)
            self._tc_state = self._SEEK_FUNC
        self._drive_tools(events)
        return events

    def _drive_tools(self, events: list):
        """Advance the tool-call state machine over ``self._full`` as far as the available text
        allows, emitting tool deltas. Each branch either advances the cursor/state or returns to
        wait for more text, so this always terminates."""
        full = self._full
        while True:
            st = self._tc_state
            if st == self._SEEK_FUNC:
                i = full.find(FUNC_OPEN, self._cur)
                gt = full.find(">", i) if i != -1 else -1
                if i == -1 or gt == -1:
                    return  # function header not fully arrived yet
                self._func_name = full[i + len(FUNC_OPEN) : gt].strip()
                self._tc_index += 1
                self._any_tool = True
                self._first_param = True
                self._tool(
                    events,
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    type="function",
                    function={"name": self._func_name, "arguments": ""},
                )
                self._tool(events, function={"arguments": "{"})
                self._args_open = True
                self._cur = gt + 1
                self._tc_state = self._IN_FUNC
            elif st == self._IN_FUNC:
                p = full.find(PARAM_OPEN, self._cur)
                f = full.find(FUNC_CLOSE, self._cur)
                if f != -1 and (p == -1 or f < p):  # function closes
                    self._tool(events, function={"arguments": "}"})
                    self._args_open = False
                    self._cur = f + len(FUNC_CLOSE)
                    self._tc_state = self._SEEK_FUNC  # find() skips the trailing </tool_call>
                elif p != -1:  # a parameter starts
                    gt = full.find(">", p)
                    if gt == -1 or gt + 1 >= len(full):
                        return  # need the full <parameter=...> plus ≥1 char to test the leading \n
                    self._param_name = full[p + len(PARAM_OPEN) : gt].strip()
                    self._param_type = self._types.get(self._func_name, {}).get(self._param_name)
                    vstart = gt + 1 + (1 if full[gt + 1] == "\n" else 0)  # template puts \n after '>'
                    self._val_start = vstart
                    self._val_emit = 0
                    self._cur = vstart
                    self._param_prefix = ("" if self._first_param else ", ") + json.dumps(self._param_name) + ": "
                    self._first_param = False
                    if self._param_type == "string":
                        self._tool(events, function={"arguments": self._param_prefix + '"'})
                        self._tc_state = self._IN_STR
                    else:
                        self._tc_state = self._IN_BUF
                else:
                    return  # neither marker present yet
            elif st == self._IN_STR:
                end = full.find(PARAM_CLOSE, self._val_start)
                if end == -1:  # value still growing: emit the safe prefix, hold back a tail
                    safe = max(self._val_start + self._val_emit, len(full) - _VALUE_HOLDBACK)
                    new = full[self._val_start + self._val_emit : safe]
                    if new:
                        self._tool(events, function={"arguments": _json_str_frag(new)})
                        self._val_emit += len(new)
                    return
                vend = end - 1 if end > self._val_start and full[end - 1] == "\n" else end
                rest = full[self._val_start + self._val_emit : vend]
                self._tool(events, function={"arguments": _json_str_frag(rest) + '"'})
                self._cur = end + len(PARAM_CLOSE)
                self._tc_state = self._IN_FUNC
            elif st == self._IN_BUF:
                end = full.find(PARAM_CLOSE, self._val_start)
                if end == -1:
                    return
                vend = end - 1 if end > self._val_start and full[end - 1] == "\n" else end
                val = json.dumps(_coerce_arg(full[self._val_start : vend], self._param_type), ensure_ascii=False)
                self._tool(events, function={"arguments": self._param_prefix + val})
                self._cur = end + len(PARAM_CLOSE)
                self._tc_state = self._IN_FUNC
            else:
                return

    def finish(self):
        """Flush anything held back. Returns a list of (kind, payload) events (same shape as
        push). For a clean generation this closes nothing extra; for one truncated mid-tool-call
        it closes the arguments JSON so the emitted ``arguments`` string still parses."""
        events: list = []
        if not self._think_closed:
            if len(self._full) > self._reason_emit:  # degenerate: think never closed
                events.append(("reasoning", self._full[self._reason_emit :]))
            return events
        if not self._in_tools:
            self._content_delta(len(self._full), events)
            return events
        if self._tc_state == self._IN_STR:
            self._tool(
                events, function={"arguments": _json_str_frag(self._full[self._val_start + self._val_emit :]) + '"'}
            )
        elif self._tc_state == self._IN_BUF:
            val = json.dumps(_coerce_arg(self._full[self._val_start :], self._param_type), ensure_ascii=False)
            self._tool(events, function={"arguments": self._param_prefix + val})
        if self._args_open:
            self._tool(events, function={"arguments": "}"})
            self._args_open = False
        return events


# ── Engine ────────────────────────────────────────────────────────────────────


class Qwen36Engine:
    """Single-batch Qwen3.6 wrapper. One request at a time (device decode state is
    single-user), so the HTTP layer serializes generation via ``self.lock``."""

    def __init__(self, mesh_device, ckpt: str, n_layers: int, max_seq: int, use_trace: bool):
        from transformers import AutoTokenizer

        self.mesh_device = mesh_device
        self.max_seq = max_seq
        self.use_trace = use_trace
        # Decode tracing (captured once, replayed per request) is the OOM fix and is on with use_trace.
        # Prefill tracing is a separate, EXPERIMENTAL opt-in (QWEN36_SERVER_PREFILL_TRACE=1): the
        # bucketed traced-prefill path is faster and allocation-free, but currently wedges the device
        # at larger buckets (e.g. 256) — under investigation. Default off: the server uses reliable
        # eager prefill (which reuses the persistent caches, so it does NOT leak/OOM either).
        self.use_prefill_trace = use_trace and os.environ.get("QWEN36_SERVER_PREFILL_TRACE", "0") == "1"

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.eos_ids = self._resolve_eos_ids(ckpt)

        args = ModelArgs(mesh_device, ckpt_dir=ckpt, max_seq_len=max_seq)
        loader = CheckpointLoader(ckpt)
        # Speculative decode (MTP). QWEN36_MTP: "0" off (default) | "1" on for every request |
        # "greedy_only" on only for temperature==0. It needs the checkpoint's `mtp.*` head (+845M
        # params and its own KV cache), a nonzero trace region, and decode tracing, so it is opt-in.
        self.mtp_mode = os.environ.get("QWEN36_MTP", "0").lower()
        self.mtp_gamma = int(os.environ.get("QWEN36_MTP_GAMMA", "2"))
        self.plain_ms_per_token = None  # measured at warmup; the break-even reference for MTP
        # Re-check every N rounds whether speculation is beating plain decode, and disengage if not.
        # 0 disables the guard (always speculate).
        self.mtp_check_rounds = int(os.environ.get("QWEN36_MTP_CHECK_ROUNDS", "24"))
        self._RATE_WARM = 4  # steps/rounds skipped before timing anything (see _decode_plain)
        self.use_mtp = self.mtp_mode in ("1", "greedy_only", "true", "on") and use_trace and loader.has_mtp()
        if self.mtp_mode not in ("0", "false", "off") and not self.use_mtp:
            logger.warning(
                f"QWEN36_MTP={self.mtp_mode!r} requested but unavailable "
                f"(has_mtp={loader.has_mtp()}, trace={use_trace}); serving without speculative decode."
            )
        logger.info(f"Building {n_layers}-layer model ({args.model_name})...")
        t0 = time.time()
        self.model = TtModel(mesh_device, args, loader, num_layers=n_layers)
        logger.info(f"Qwen3.6 model built in {time.time() - t0:.1f}s")
        self.lock = threading.Lock()

    def warmup(self):
        """Warm up both decode tails (their kernels differ): the first forward JIT-compiles every op
        and captures the decode trace, so the first real request reports representative tok/s. The
        default request samples, so without warming the sampling tail the first sampling request
        would pay the sampling-kernel compile (~seconds). Each generate() captures (and releases the
        prior) decode trace; persistent caches + that release keep device memory flat across
        requests (the OOM fix)."""
        warm_ids = self.tokenizer.encode("Hello", return_tensors="pt").squeeze(0)
        mtp, self.use_mtp = self.use_mtp, False  # the two warm generates below must stay PLAIN: they
        try:  # compile the plain tails and provide the break-even reference _decode_mtp compares to
            # 24 tokens, not 4: long enough for _decode_plain to record an end-to-end ms/token
            # (it needs >16 steps), which is the reference the MTP break-even guard uses.
            self.generate(warm_ids, 64 if mtp else 4)  # greedy tail
            greedy_ms, greedy_e2e = self._calibrate_plain() if mtp else None, self.plain_ms_per_token
            self.generate(warm_ids, 64 if mtp else 4, temperature=1.0, top_k=20, top_p=0.95, presence_penalty=1.5)
            sampling_ms, sampling_e2e = self._calibrate_plain() if mtp else None, self.plain_ms_per_token
        finally:
            self.use_mtp = mtp
        if self.use_mtp:
            # The reference is the tail MTP will actually be compared against: greedy_only serves
            # temperature==0 requests, mode 1 serves sampled ones too (and the sampling tail is the
            # slower of the two, so using it is the conservative choice there).
            sampled_mode = self.mtp_mode in ("1", "true", "on")
            self.plain_ms_per_token = (sampling_e2e if sampled_mode else greedy_e2e) or (
                sampling_ms if sampled_mode else greedy_ms
            )
            if self.plain_ms_per_token:
                logger.info(
                    f"[mtp] plain decode reference {self.plain_ms_per_token:.2f} ms/token end-to-end "
                    f"(device-only: greedy {greedy_ms:.2f}, sampling {sampling_ms:.2f})"
                )
            # ONE MTP warm generation, whose sampling mode picks the verify tail this process will use
            # for its whole lifetime. Exactly one verify trace may be resident (a second one corrupts
            # it — see TtModel.setup_mtp_decode), so the tail is a process-level choice, not a
            # per-request one:
            #   QWEN36_MTP=1          -> capture the SAMPLING tail; greedy requests replay it too
            #                            (it writes the argmax buffer as well, they just also pay the
            #                            candidate topk, ~1.7 ms/round at gamma=2).
            #   QWEN36_MTP=greedy_only-> capture the GREEDY tail; sampled requests use plain decode.
            n = max(self.mtp_gamma + 2, 6)  # enough rounds to get past the 2 warm/capture rounds
            if self.mtp_mode in ("1", "true", "on"):
                self.generate(warm_ids, n, temperature=1.0, top_k=20, top_p=0.95, presence_penalty=1.5)
            else:
                self.generate(warm_ids, n)

    def _calibrate_plain(self, steps: int = 24):
        """ms/token for the plain traced decode step on this board.

        Speculative decode only pays above a break-even acceptance (round_ms / plain_ms tokens per
        round), and that ratio is board- and build-specific, so it is measured rather than assumed.
        Runs during warmup, where mutating the decode state is harmless. Returns None if there is no
        decode trace to replay (eager mode)."""
        m = self.model
        if getattr(m, "trace_id", None) is None:
            return None
        m.decode_step_traced()  # discard one: settles the queue after the preceding generate
        t0 = time.time()
        for _ in range(steps):
            m.decode_step_traced()
        return (time.time() - t0) / steps * 1e3

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

    @staticmethod
    def _select_tools(tools, tool_choice):
        """Resolve the tools list to actually offer the model given tool_choice. "none" hides
        all tools; a {"type":"function","function":{"name":...}} choice narrows to that one;
        otherwise ("auto"/"required"/None) all tools are offered."""
        if not tools or tool_choice == "none":
            return None
        if isinstance(tool_choice, dict):
            name = (tool_choice.get("function") or {}).get("name")
            if name:
                sel = [t for t in tools if (t.get("function", t)).get("name") == name]
                return sel or tools
        return tools

    @staticmethod
    def _to_template_msg(m: ChatMessage) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": m.role, "content": _flatten(m.content)}
        if m.role == "assistant" and m.tool_calls:
            d["tool_calls"] = [_normalize_history_tool_call(tc) for tc in m.tool_calls]
        return d

    def ids_from_messages(
        self, messages: List[ChatMessage], tools=None, tool_choice=None, enable_thinking=None
    ) -> torch.Tensor:
        if self.tokenizer.chat_template:
            # Only pass enable_thinking when set, so we don't override the template's own default when
            # the caller leaves it unspecified.
            extra = {} if enable_thinking is None else {"enable_thinking": bool(enable_thinking)}
            out = self.tokenizer.apply_chat_template(
                [self._to_template_msg(m) for m in messages],
                tools=self._select_tools(tools, tool_choice),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **extra,
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
        temperature/top-k/top-p (+ presence_penalty) sampling tail. With ``QWEN36_MTP`` enabled the
        decode steps are replaced by speculative rounds (``_decode_mtp``), which emit 1..gamma+1
        tokens each and are distribution-exact for both greedy and sampled requests. ``min_p`` and
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
                f"prompt length {prompt_len} >= max_seq_len {self.max_seq}; raise QWEN36_MAX_SEQ or shorten the prompt"
            )
        # Finish reason for this generation, refined as we go: "length" unless we hit an EOS id
        # ("stop") or, in the text layer, a user stop string ("stop"). Read by callers after the
        # generator is exhausted (generation is serialized under self.lock).
        self._finish_reason = "length"
        # leave room: prefill consumes prompt_len positions, each decode step consumes one
        budget = min(max_new, self.max_seq - prompt_len - 1)
        if budget <= 0:
            return

        # Select greedy vs sampling for this request's decode steps (must precede start_decode
        # so the captured trace records the right tail). temperature==0 -> greedy.
        self.model.enable_sampling(temperature, top_k, top_p, seed or 0, presence_penalty)

        # Prefill: eager by default (reuses persistent caches -> no leak); bucketed traced replay
        # only when the experimental QWEN36_SERVER_PREFILL_TRACE flag is set.
        if self.use_prefill_trace:
            logits = self.model.forward_prefill_traced(ids)  # prefill -> [1, 1, vocab] host logits
        else:
            logits = self.model.forward(ids)
        next_id = int(logits[0, -1].argmax())  # first token is greedy argmax
        self.model.start_decode(next_id)

        if self._mtp_active(temperature):
            yield from self._decode_mtp(next_id, budget)
            return
        yield from self._decode_plain(next_id, budget)

    def _decode_plain(self, next_id: int, budget: int, traced: bool = False) -> Iterator[int]:
        """One-token-per-step decode. Capture the decode trace on the first step, then replay it. The
        capture RELEASES the prior request's trace (see capture_decode_trace) so traces don't
        accumulate; combined with persistent caches, device memory stays flat (no OOM).

        `traced=True` means a usable decode trace is already live — used when `_decode_mtp` disengages
        mid-request and hands the rest of the generation over.

        Also records `plain_ms_per_token` END TO END — i.e. including the caller's per-token work
        (`_generate_text` re-decodes the whole token list each step). That is deliberately the number
        `_decode_mtp` compares against: a speculative round pays the same host cost per token, so
        comparing its round time against a DEVICE-only plain step would penalise it for overhead both
        paths share. Measured here that overhead is small — 28.55 ms in decode_step vs 29.46 end-to-end,
        so ~0.9 ms/token — but it has to be measured over enough SETTLED steps: a 16-step window
        starting right after capture_decode_trace reads 43-46 ms/token, which is the capture's
        snapshot/restore draining, not a real rate."""
        t0, n, dev = None, 0, 0.0
        for step in range(budget):
            if next_id in self.eos_ids:
                self._finish_reason = "stop"
                return
            yield next_id
            t_step = time.time()
            if self.use_trace and not traced:
                next_id = self.model.decode_step_eager()  # warmup compiles kernels
                self.model.capture_decode_trace()
                traced = True
            elif self.use_trace:
                next_id = self.model.decode_step_traced()
            else:
                next_id = self.model.decode_step_eager()
            dev += time.time() - t_step
            n += 1
            # Exclude the first WARM steps, not just the first: step 1 compiles and captures, and
            # capture_decode_trace's state snapshot/restore (~30 layers of clone+copy) is enqueued
            # behind it, so the next few replays queue behind that work. Measured: timing from step 2
            # over 15 steps gave 43 ms/token where the settled rate is ~29 -- enough to make the MTP
            # break-even guard mis-fire.
            if n == self._RATE_WARM:
                t0, dev = time.time(), 0.0
            elif t0 is not None and (n - self._RATE_WARM) >= 16 and (n - self._RATE_WARM) % 16 == 0:
                timed = n - self._RATE_WARM
                self.plain_ms_per_token = (time.time() - t0) / timed * 1e3
                logger.debug(  # the INFO summary is warmup's "plain decode reference" line
                    f"[plain] {timed} steps: {self.plain_ms_per_token:.2f} ms/token end-to-end "
                    f"({dev / timed * 1e3:.2f} in decode_step, "
                    f"{self.plain_ms_per_token - dev / timed * 1e3:.2f} host)"
                )

    def _mtp_active(self, temperature: float) -> bool:
        """Is speculative decode used for THIS request?

        ``QWEN36_MTP=1`` uses it for greedy AND sampled requests: sampled speculative decode is
        distribution-EXACT (rejection sampling over the verify rows' own distributions,
        tt/mtp_sampling.py), so a sampled request's output distribution is unchanged by speculation.
        ``greedy_only`` restricts it to ``temperature == 0``, which is the conservative setting if the
        sampled speedup ever fails to beat plain sampled decode on a given board."""
        if not self.use_mtp:
            return False
        return self.mtp_mode in ("1", "true", "on") or temperature == 0

    def _decode_mtp(self, first_id: int, budget: int) -> Iterator[int]:
        """Speculative-decode drive loop: yields the prefill token then each round's emitted tokens.

        A round emits 1..gamma+1 tokens at once, so EOS and the token budget have to be honoured
        INSIDE the round: the tokens after an EOS have already been computed, but must not be emitted.
        The same goes for the context bound — a round consumes up to gamma+1 positions, so it is only
        started when that many fit.

        Correctness note: draft quality only affects THROUGHPUT. A rejected draft is discarded and the
        emitted token comes from the backbone's own distribution, so a stale MTP KV cache (this flow
        does not prime the head over the prompt — ``TtMtpHead.forward_prefill`` exists for that and is
        the acceptance-improving follow-up) can cost tokens/round but never correctness."""
        m = self.model
        m.build_mtp_head(gamma=self.mtp_gamma)
        m.setup_mtp_decode()  # idempotent for a fixed gamma: keeps the traces captured at warmup
        # Verify reads cache["conv_state"] as the gated-delta conv left-pad while plain decode
        # reads/advances the conv_rows buffers (see model.py). After a prefill the two already agree,
        # so this is a ~10 ms/request no-op here — kept as insurance so any future flow that
        # interleaves plain decode steps with speculative rounds cannot silently drop conv history.
        m.refresh_conv_state()
        m._mtp_hidden = None
        m.mtp_stats = {"rounds": 0, "tokens": 0, "accepted": 0, "drafted": 0}
        pending, n_out, rounds, t_dec, base = [first_id], 0, 0, None, (0, 0)
        while True:
            for tid in pending:
                if tid in self.eos_ids:
                    self._finish_reason = "stop"
                    return
                yield tid
                n_out += 1
                if n_out >= budget:
                    return
            if m.pos + m.mtp_gamma + 2 > self.max_seq:  # a round needs gamma+1 positions of headroom
                return

            if not m.mtp_tail_ready():
                # First MTP request for this tail (warmup normally does this): two warm eager rounds
                # then capture. The warm rounds' tokens are REAL output, so they join the stream.
                t0 = time.time()
                pending = m.setup_mtp_traces()
                logger.info(
                    f"[mtp] captured verify({m.mtp_verify_tail})/draft/commit traces in {time.time() - t0:.1f}s"
                )
            else:
                pending = m.spec_decode_step()
            rounds += 1
            if rounds <= self._RATE_WARM:
                # Exclude the first rounds from the rate. Round 1 is the eager K=1 seed step (no
                # draft, and no captured trace is shaped for one row) and costs hundreds of ms, and
                # when this request also captured, the snapshot/restore work lands in the rounds right
                # after it. Both would otherwise be smeared over the window and make a winning
                # configuration look like a loser.
                t_dec, base = time.time(), (m.mtp_stats["rounds"], m.mtp_stats["tokens"])
                continue
            timed = m.mtp_stats["rounds"] - base[0]
            if self.mtp_check_rounds and timed and timed % self.mtp_check_rounds == 0:
                # Decode-only rate (the [gen] line folds in prefill and any capture). Acceptance is
                # strongly PROMPT-dependent — measured 2.70 tok/round on one chat prompt and 2.17 on
                # another — and a round costs ~2.3x a plain decode step, so below break-even
                # speculation is a net LOSS. Rather than gamble on the prompt, compare the two
                # measured rates and hand the rest of the request to plain decode when we are behind.
                ms_round = (time.time() - t_dec) / timed * 1e3
                tpr = (m.mtp_stats["tokens"] - base[1]) / timed
                ms_tok = ms_round / max(tpr, 1e-9)
                logger.info(
                    f"[mtp] {timed} rounds, {ms_round:.1f} ms/round, {tpr:.2f} tok/round "
                    f"-> {1000 / ms_tok:.1f} tok/s decode-only"
                    + (f" (plain {1000 / self.plain_ms_per_token:.1f})" if self.plain_ms_per_token else "")
                )
                losing = self.plain_ms_per_token and ms_tok > self.plain_ms_per_token
                # Only hand over if a plain decode trace is ALREADY live: capturing one here would
                # capture while the MTP traces are resident, which is the condition that silently
                # corrupts a capture (see TtModel.setup_mtp_decode). Warmup's plain generate leaves
                # one, so this holds in practice; if it somehow does not, keep speculating.
                if losing and self.use_trace:
                    # STICKY: acceptance is a property of the workload more than of one prompt, and
                    # re-engaging would cost an ~8 s MTP re-capture per request. QWEN36_MTP_CHECK_ROUNDS=0
                    # disables the guard for an operator who wants speculation regardless.
                    logger.warning(
                        f"[mtp] disabling speculative decode: {ms_tok:.1f} ms/token vs plain "
                        f"{self.plain_ms_per_token:.1f} (break-even {ms_round / self.plain_ms_per_token:.2f} "
                        f"tok/round, got {tpr:.2f})"
                    )
                    self.use_mtp = False
                    # sync_mtp_state pushes the python-tracked token/position back AND re-seeds the
                    # decode conv add-chain from conv_state, which the speculative rounds have been
                    # advancing instead (see TtModel.sync_mtp_state).
                    m.sync_mtp_state()
                    m.release_mtp_traces()  # capture the plain trace from a clean slate (see above)
                    # _mtp_cur was already yielded, so take one step for the NEXT token, then capture
                    # this request's own decode tail exactly as the plain path's first step does.
                    nxt = self.model.decode_step_eager()
                    self.model.capture_decode_trace()
                    yield from self._decode_plain(nxt, budget - n_out, traced=True)
                    return

    def _generate_text(self, ids: torch.Tensor, max_new: int, stop=None, **sampling) -> Iterator[str]:
        """Drive ``_generate_ids`` and yield the cumulative decoded text after each step.

        Applies user ``stop`` strings: if one appears, the text is truncated just before it,
        the generation ends, and ``self._finish_reason`` becomes "stop". While streaming, a tail
        of up to ``maxlen(stop) - 1`` chars is withheld so a stop string spanning a token
        boundary is never emitted; the final yield flushes it. Also tracks ``self._n_generated``
        (tokens produced). ``self._finish_reason`` is otherwise set by ``_generate_ids``
        ("stop" on EOS, "length" on the token budget)."""
        stops = _norm_stop(stop)
        hold = max((len(s) for s in stops), default=1) - 1  # withheld tail guards boundary-spanning stops
        out_ids: List[int] = []
        self._n_generated = 0
        text = ""
        t_start = time.time()  # progress logging: prove the device is decoding even if the client renders nothing
        for tid in self._generate_ids(ids, max_new, **sampling):
            out_ids.append(tid)
            self._n_generated = len(out_ids)
            if len(out_ids) % 128 == 0:
                dt = time.time() - t_start
                extra = ""
                if self.use_mtp and (self.model.mtp_stats or {}).get("rounds"):
                    tpr, acc, pacc = self.model.mtp_acceptance()
                    extra = f", mtp {tpr:.2f} tok/round (accept {acc:.2f}" + (
                        f", mean p {pacc:.3f})" if pacc is not None else ")"
                    )
                logger.info(f"[gen] {len(out_ids)} tok, {len(out_ids) / dt:.1f} tok/s{extra} (still generating)")
            text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            if stops:
                cut = _earliest_stop(text, stops)
                if cut is not None:
                    self._finish_reason = "stop"
                    yield text[:cut]
                    return
                yield text[: max(0, len(text) - hold)]  # hold back a possible partial stop string
            else:
                yield text
        yield text  # flush any withheld tail (no-op when nothing was held back)

    def generate(self, ids: torch.Tensor, max_new: int, stop=None, **sampling):
        """Drain ``_generate_text`` into a full completion.

        ``sampling`` = temperature/top_k/top_p/seed (forwarded to ``_generate_ids``).
        Returns (text, completion_tokens, gen_seconds, tokens_per_second, finish_reason).
        """
        t0 = time.time()
        text = ""
        for text in self._generate_text(ids, max_new, stop=stop, **sampling):
            pass
        secs = time.time() - t0
        n_tok = self._n_generated
        tok_s = n_tok / secs if secs > 0 else 0.0
        return text, n_tok, secs, tok_s, self._finish_reason

    def stream_text(self, ids: torch.Tensor, max_new: int, stop=None, **sampling) -> Iterator[str]:
        """Yield incremental decoded text deltas (handles multi-byte tokens and stop strings by
        re-decoding the running id list and emitting only the new suffix)."""
        prev = ""
        for full in self._generate_text(ids, max_new, stop=stop, **sampling):
            if len(full) > len(prev):
                yield full[len(prev) :]
                prev = full


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
    # When the client sets max_tokens, honor it. Otherwise offer the FULL remaining context, not a
    # fraction: this is a thinking model and agentic clients (e.g. OpenCode) emit whole files as tool
    # arguments — a small default (the old max_seq//4) truncates the reasoning or the file before the
    # <tool_call> closes, so the call is incomplete/dropped. _generate_ids caps this to the real
    # budget (max_seq - prompt_len - 1) and generation stops early on EOS, so this is an upper bound.
    return req_max if req_max and req_max > 0 else eng.max_seq


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _stream_chat(
    eng: Qwen36Engine,
    ids: torch.Tensor,
    max_new: int,
    model: str,
    tools=None,
    stop=None,
    expect_thinking: bool = True,
    **sampling,
) -> Iterator[str]:
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}

    def chunk(delta, finish=None):
        return _sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    def to_delta(kind, payload):
        if kind == "reasoning":
            return {"reasoning_content": payload}
        if kind == "content":
            return {"content": payload}
        return {"tool_calls": [payload]}  # already an OpenAI tool_calls delta

    parser = _ChatStreamParser(tools, expect_thinking=expect_thinking)
    full = ""
    with eng.lock:
        # first chunk announces the assistant role
        yield chunk({"role": "assistant"})
        for full in eng._generate_text(ids, max_new, stop=stop, **sampling):
            for kind, payload in parser.push(full):
                yield chunk(to_delta(kind, payload))
        for kind, payload in parser.finish():
            yield chunk(to_delta(kind, payload))
        finish = "tool_calls" if parser.any_tool else eng._finish_reason
    _log_chat_response(full, finish, None)  # raw stream text; tool calls were streamed as deltas
    yield chunk({}, finish)
    yield "data: [DONE]\n\n"


def _stream_completion(
    eng: Qwen36Engine, ids: torch.Tensor, max_new: int, model: str, stop=None, **sampling
) -> Iterator[str]:
    cid = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    base = {"id": cid, "object": "text_completion", "created": created, "model": model}
    with eng.lock:
        for piece in eng.stream_text(ids, max_new, stop=stop, **sampling):
            yield _sse({**base, "choices": [{"index": 0, "text": piece, "finish_reason": None}]})
        finish = eng._finish_reason
    yield _sse({**base, "choices": [{"index": 0, "text": "", "finish_reason": finish}]})
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    eng: Qwen36Engine = app.state.engine
    # Resolve thinking: explicit request field > chat_template_kwargs.enable_thinking > AUTO (off when
    # tools are present, else on). Non-thinking is the reliable config for agentic tool calling.
    ctk = req.chat_template_kwargs or {}
    if req.enable_thinking is not None:
        enable_thinking = req.enable_thinking
    elif "enable_thinking" in ctk:
        enable_thinking = bool(ctk["enable_thinking"])
    else:
        enable_thinking = not bool(req.tools)
    ids = eng.ids_from_messages(
        req.messages, tools=req.tools, tool_choice=req.tool_choice, enable_thinking=enable_thinking
    )
    max_new = _default_max_new(eng, req.max_tokens)
    model = req.model or MODEL_ID
    _log_chat_request(req, int(ids.numel()), eng.max_seq)
    if _LOG_REQUESTS:
        logger.info(f"[req] enable_thinking={enable_thinking} (tools={bool(req.tools)})")
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
    stop = _norm_stop(req.stop)
    if req.stream:
        return StreamingResponse(
            _stream_chat(
                eng, ids, max_new, model, tools=req.tools, stop=stop, expect_thinking=enable_thinking, **sampling
            ),
            media_type="text/event-stream",
        )

    with eng.lock:
        text, n_tok, secs, tok_s, finish = eng.generate(ids, max_new, stop=stop, **sampling)
    # Split the raw generation into reasoning / visible content / tool calls (the tool/think
    # markers are plain text, so they survive decode and are parsed here).
    reasoning, content, tool_calls = _parse_chat_output(text, req.tools, expect_thinking=enable_thinking)
    message: Dict[str, Any] = {"role": "assistant", "content": content or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish = "tool_calls"
    logger.info(
        f"[chat] {n_tok} tok in {secs:.2f}s = {tok_s:.1f} tok/s"
        + (f" ({len(tool_calls)} tool call(s))" if tool_calls else "")
    )
    _log_chat_response(text, finish, tool_calls)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
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
    stop = _norm_stop(req.stop)
    if req.stream:
        return StreamingResponse(
            _stream_completion(eng, ids, max_new, model, stop=stop, **sampling), media_type="text/event-stream"
        )

    with eng.lock:
        text, n_tok, secs, tok_s, finish = eng.generate(ids, max_new, stop=stop, **sampling)
    logger.info(f"[cmpl] {n_tok} tok in {secs:.2f}s = {tok_s:.1f} tok/s")
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish}],
        "usage": _usage(int(ids.numel()), n_tok),
        "tokens_per_second": round(tok_s, 2),
        "generation_seconds": round(secs, 3),
    }


# ── standalone entrypoint (opens the 1x1 mesh directly, like demo.py) ──────────


def main():
    ckpt = os.environ.get("QWEN36_CKPT", os.path.expanduser("~/models/qwen36"))
    n_layers = int(os.environ.get("QWEN36_LAYERS", "40"))
    max_seq = int(os.environ.get("QWEN36_MAX_SEQ", "8192"))
    use_trace = os.environ.get("QWEN36_SERVER_TRACE", "1") != "0"
    host = os.environ.get("QWEN36_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("QWEN36_SERVER_PORT", "8000"))

    # Trace capture needs a nonzero trace region; eager-only can use 0. Speculative decode holds more
    # traces concurrently than plain decode — one verify per tail kind (greedy + sampling), the draft
    # chain, and one commit trace per accept count 1..gamma+1, on top of the decode trace — so it gets
    # a larger default. QWEN36_TRACE_REGION (MiB) overrides; raise it if capture starts failing.
    mtp_on = os.environ.get("QWEN36_MTP", "0").lower() not in ("0", "false", "off")
    default_mb = 400 if mtp_on else 200
    trace_region = int(os.environ.get("QWEN36_TRACE_REGION", default_mb)) * 1024 * 1024 if use_trace else 0
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=trace_region)
    try:
        engine = Qwen36Engine(mesh, ckpt, n_layers, max_seq, use_trace)
        app.state.engine = engine

        # Warm up both decode tails (greedy + sampling) so the first real request reports
        # representative tok/s. Each request (re)captures its decode trace and RELEASES the prior one,
        # and the per-layer caches are allocated once and reused — so device memory stays flat and the
        # server no longer OOMs after a few requests. (A single benign allocator.cpp:110 "active
        # trace" notice may still print once per thread; it is cosmetic — see capture_decode_trace.)
        logger.info("Warming up (greedy + sampling)...")
        t0 = time.time()
        engine.warmup()
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
