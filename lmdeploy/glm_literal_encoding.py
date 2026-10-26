# Copyright (c) OpenMMLab. All rights reserved.
"""Encode GLM chat prompts so control-token spellings in message content stay literal text.

The GLM chat template inserts structural tags (``<|user|>``, ``<think>``, ``<tool_call>`` ...)
as real control tokens, but the same spellings can also appear inside message content (a user
quoting the format, a tool result containing a tag, ...). Rendering the conversation to a
string and tokenizing it as a whole turns those content spellings into control IDs as well,
so the model cannot tell content from structure. The training pipeline (xtuner #2208,
``glm53_chat.py``) records the content spans while rendering and re-encodes the text runs
between genuine controls with a tokenizer that has no added tokens. This module reproduces
that encoding for inference:

1. :func:`mark_literal_pieces` wraps every content string piece with private-use sentinels.
2. The marked conversation goes through the normal chat template; removing the sentinels
   from the rendered text must reproduce the plain render, which yields the exact character
   spans of content in the rendered prompt (:func:`extract_literal_spans`).
3. :func:`encode_with_literal_spans` tokenizes the rendered prompt, keeps added-token IDs
   that fall outside content spans as controls, and re-encodes everything between them
   with a base tokenizer stripped of added tokens.

If anything does not line up (a piece the template transforms away, a non-fast tokenizer,
no literal collision), ``None`` is returned and the caller falls back to plain encoding.
"""
from __future__ import annotations

import json
import re
from typing import Any

_START = '\ue000{}\ue001'
_END = '\ue002{}\ue003'


def _mark(obj: Any, counter: list[int]) -> Any:
    """Wrap every string leaf of a JSON-like value with sentinels."""
    if isinstance(obj, str):
        index = counter[0]
        counter[0] += 1
        return _START.format(index) + obj + _END.format(index)
    if isinstance(obj, dict):
        # Keys stay unmarked: templates branch on them ('function' in tool, ...).
        return {k: _mark(v, counter) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mark(v, counter) for v in obj]
    return obj


def _mark_content(content: Any, counter: list[int]) -> Any:
    if isinstance(content, str):
        return _mark(content, counter)
    if isinstance(content, list):
        marked = []
        for item in content:
            if isinstance(item, str):
                marked.append(_mark(item, counter))
            elif isinstance(item, dict) and item.get('type') == 'text' and isinstance(item.get('text'), str):
                item = dict(item)
                item['text'] = _mark(item['text'], counter)
                marked.append(item)
            elif isinstance(item, dict) and 'output' in item:
                item = dict(item)
                item['output'] = _mark_content(item['output'], counter)
                marked.append(item)
            else:  # media and other parts keep their original objects
                marked.append(item)
        return marked
    return content


def mark_literal_pieces(messages: list[dict], tools: list | None) -> tuple[list[dict], list | None, int]:
    """Return copies of *messages*/*tools* with content strings sentinel-wrapped.

    Only structures the GLM template inserts as message content are wrapped: message text,
    ``reasoning_content``, tool-call names and argument values, tool results, and string
    leaves of tool definitions. Media payloads are shared, not copied.
    """
    counter = [0]
    marked_messages = []
    for message in messages:
        message = dict(message)
        if 'content' in message:
            message['content'] = _mark_content(message['content'], counter)
        if isinstance(message.get('reasoning_content'), str):
            message['reasoning_content'] = _mark(message['reasoning_content'], counter)
        if message.get('tool_calls'):
            calls = []
            for call in message['tool_calls']:
                call = dict(call)
                function = call.get('function') if isinstance(call.get('function'), dict) else call
                function = dict(function)
                if isinstance(function.get('name'), str):
                    function['name'] = _mark(function['name'], counter)
                if isinstance(function.get('arguments'), (dict, list)):
                    function['arguments'] = _mark(function['arguments'], counter)
                if 'function' in call:
                    call['function'] = function
                else:
                    call = function
                calls.append(call)
            message['tool_calls'] = calls
        marked_messages.append(message)
    marked_tools = _mark(tools, counter) if tools else tools
    return marked_messages, marked_tools, counter[0]


def extract_literal_spans(marked_prompt: str, plain_prompt: str, count: int) -> list[tuple[int, int]] | None:
    """Recover content character spans from a sentinel-marked render.

    Returns ``None`` when removing the sentinels does not reproduce *plain_prompt*
    exactly, in which case the template transformed the markers and spans cannot
    be trusted.
    """
    spans: list[tuple[int, int]] = []
    cleaned: list[str] = []
    cleaned_len = 0
    pos = 0
    open_index: int | None = None
    open_at = 0
    marker = re.compile(r'\ue000(\d+)\ue001|\ue002(\d+)\ue003')
    for match in marker.finditer(marked_prompt):
        chunk = marked_prompt[pos:match.start()]
        cleaned.append(chunk)
        cleaned_len += len(chunk)
        if match.group(1) is not None:
            open_index = int(match.group(1))
            open_at = cleaned_len
        elif open_index is not None and int(match.group(2)) == open_index:
            spans.append((open_at, cleaned_len))
            open_index = None
        pos = match.end()
    cleaned.append(marked_prompt[pos:])
    if ''.join(cleaned) != plain_prompt:
        return None
    return spans


def build_base_backend(tokenizer):
    """A copy of the fast tokenizer backend with all added tokens removed."""
    from tokenizers import Tokenizer

    spec = json.loads(tokenizer.backend_tokenizer.to_str())
    spec['added_tokens'] = []
    return Tokenizer.from_str(json.dumps(spec))


def encode_with_literal_spans(tokenizer,
                              text: str,
                              spans: list[tuple[int, int]],
                              base_backend=None) -> list[int] | None:
    """Tokenize *text* keeping added tokens outside *spans* as single control IDs.

    Added-token IDs intersecting a content span are literals: the maximal text runs
    between genuine controls are encoded with the base tokenizer instead, so a
    literal takes the ordinary BPE tokens it would get in running text. Returns
    ``None`` when no added token intersects a span (nothing to fix).
    """
    if not spans or not getattr(tokenizer, 'is_fast', False):
        return None
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    except Exception:
        return None
    input_ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    added_ids = set(tokenizer.get_added_vocab().values())

    controls: list[tuple[int, int, int]] = []
    has_literal = False
    span_index = 0
    for token_id, (start, end) in zip(input_ids, offsets):
        if token_id not in added_ids or start == end:
            continue
        while span_index < len(spans) and spans[span_index][1] <= start:
            span_index += 1
        if span_index < len(spans) and spans[span_index][0] < end:
            has_literal = True
        else:
            controls.append((token_id, start, end))
    if not has_literal:
        return None

    backend = base_backend if base_backend is not None else build_base_backend(tokenizer)
    result: list[int] = []
    cursor = 0
    for token_id, start, end in [*controls, (None, len(text), len(text))]:
        if cursor < start:
            result.extend(backend.encode(text[cursor:start], add_special_tokens=False).ids)
        if token_id is not None:
            result.append(token_id)
        cursor = end
    return result
