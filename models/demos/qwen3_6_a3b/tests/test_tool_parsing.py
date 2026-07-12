# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the server's Qwen3 XML tool-call parsing (no device needed).

Covers the OpenCode failure mode: a generation truncated mid <tool_call> (finish_reason="length",
e.g. a big file written as a tool argument) must still surface a valid tool call, and the streaming
and non-streaming paths must agree. Run: ./python_env/bin/python -m pytest models/demos/qwen3_6_a3b/tests/test_tool_parsing.py
"""
import json

from models.demos.qwen3_6_a3b.demo.server import _ChatStreamParser, _parse_chat_output

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_count",
            "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        },
    },
]

# A complete call: reasoning, a sentence of visible content, then a closed <tool_call>.
COMPLETE = (
    "I should write the file.</think>\n\nHere is the game:\n"
    "<tool_call>\n<function=write>\n"
    "<parameter=path>\nindex.html\n</parameter>\n"
    '<parameter=content>\n<html>"x" & {y}\nline2</html>\n</parameter>\n'
    "</function>\n</tool_call>"
)
# Same call truncated mid-content (no closing </parameter></function></tool_call>): finish=length.
TRUNCATED = (
    "I should write the file.</think>\n\nHere is the game:\n"
    "<tool_call>\n<function=write>\n"
    "<parameter=path>\nindex.html\n</parameter>\n"
    "<parameter=content>\n<html>\n  <body>partial code that got cut off"
)


def _args(call):
    return json.loads(call["function"]["arguments"])


def test_complete_tool_call():
    reasoning, content, calls = _parse_chat_output(COMPLETE, TOOLS)
    assert reasoning == "I should write the file."
    assert content == "Here is the game:"
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "write"
    a = _args(calls[0])  # arguments must be valid JSON with both params, special chars escaped
    assert a["path"] == "index.html"
    assert a["content"] == '<html>"x" & {y}\nline2</html>'


def test_truncated_tool_call_still_parses():
    # The OpenCode bug: before the fix this returned 0 calls (regex needed </tool_call>) and the
    # half-written file leaked into content. Now the partial call is surfaced with valid (closed) JSON.
    reasoning, content, calls = _parse_chat_output(TRUNCATED, TOOLS)
    assert len(calls) == 1, "truncated tool call must still be surfaced"
    assert calls[0]["function"]["name"] == "write"
    a = _args(calls[0])  # JSON closed by finish(); content holds whatever streamed so far
    assert a["path"] == "index.html"
    assert "partial code that got cut off" in a["content"]


def test_non_string_arg_coercion():
    txt = "ok</think>\n\n<tool_call>\n<function=set_count>\n<parameter=n>\n42\n</parameter>\n</function>\n</tool_call>"
    _, _, calls = _parse_chat_output(txt, TOOLS)
    assert _args(calls[0]) == {"n": 42}  # integer-typed param coerced, not a string


def test_plain_text_no_tools():
    reasoning, content, calls = _parse_chat_output("thinking...</think>\n\nJust a normal answer.", TOOLS)
    assert calls == []
    assert content == "Just a normal answer."
    assert reasoning == "thinking..."


def _stream_parse(text):
    """Reassemble (reasoning, content, tool_calls) by feeding cumulative prefixes to the stream
    parser, char by char — mimics how the SSE endpoint drives it from growing decoded text."""
    parser = _ChatStreamParser(TOOLS)
    reasoning = content = ""
    calls = {}
    events = []
    for i in range(1, len(text) + 1):
        events = parser.push(text[:i])
        for kind, payload in events:
            if kind == "reasoning":
                reasoning += payload
            elif kind == "content":
                content += payload
            else:
                c = calls.setdefault(payload["index"], {"name": "", "args": ""})
                fn = payload.get("function", {})
                c["name"] = fn.get("name", c["name"]) or c["name"]
                c["args"] += fn.get("arguments", "")
    for kind, payload in parser.finish():
        if kind == "reasoning":
            reasoning += payload
        elif kind == "content":
            content += payload
        else:
            c = calls.setdefault(payload["index"], {"name": "", "args": ""})
            fn = payload.get("function", {})
            c["name"] = fn.get("name", c["name"]) or c["name"]
            c["args"] += fn.get("arguments", "")
    return reasoning.strip(), content.strip(), [calls[i] for i in sorted(calls)]


def test_stream_matches_nonstream_complete():
    s_reason, s_content, s_calls = _stream_parse(COMPLETE)
    n_reason, n_content, n_calls = _parse_chat_output(COMPLETE, TOOLS)
    assert (s_reason, s_content) == (n_reason, n_content)
    assert [c["name"] for c in s_calls] == [c["function"]["name"] for c in n_calls]
    assert json.loads(s_calls[0]["args"]) == _args(n_calls[0])


def test_stream_matches_nonstream_truncated():
    s_reason, s_content, s_calls = _stream_parse(TRUNCATED)
    _, _, n_calls = _parse_chat_output(TRUNCATED, TOOLS)
    assert len(s_calls) == len(n_calls) == 1
    assert json.loads(s_calls[0]["args"]) == _args(n_calls[0])  # both close the JSON identically
