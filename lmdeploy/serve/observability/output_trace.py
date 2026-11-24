# Copyright (c) OpenMMLab. All rights reserved.
"""Payload-free output metadata that travels with the frame actually sent."""

from typing import NamedTuple


def _get(value, key, default=None):
    if value is None:
        return default
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


class OutputMetadata(NamedTuple):
    """Immutable output indices and flags, without request or response
    content."""
    # (choice index, reasoning present, content present, tool output present)
    outputs: tuple = ()
    terminal: bool = False
    error: bool = False
    cancelled: bool = False


def summarize(payload=None, *, terminal=False):
    """Read protocol fields before serialization; retain only indices and
    flags."""
    kind = _get(payload, 'type', '')
    error = _get(payload, 'error')
    cancelled = (_get(_get(payload, 'response'), 'status') == 'cancelled'
                 or _get(error, 'code') == 'request_cancelled')
    failed = bool(error or kind in ('error', 'response.failed')) and not cancelled
    terminal |= kind in ('message_stop', 'response.completed', 'response.incomplete', 'response.failed')
    choices = _get(payload, 'choices')
    outputs = []
    if isinstance(choices, (list, tuple)):
        for choice in choices:
            # Bind once per object: serialized adapters supply dictionaries,
            # while protocol objects are also accepted. No payload is retained.
            get_choice = choice.get if isinstance(choice, dict) else lambda key: getattr(choice, key, None)
            delta = get_choice('delta')
            reasoning = content = tool = False
            if delta is not None:
                get_delta = delta.get if isinstance(delta, dict) else lambda key: getattr(delta, key, None)
                reasoning = bool(get_delta('reasoning_content') or get_delta('reasoning'))
                content = bool(get_delta('content'))
                tool = bool(get_delta('tool_calls') or get_delta('function_call'))
            content |= bool(get_choice('text'))
            index = get_choice('index')
            if reasoning or content or tool:
                # Invalid wire indices still contribute to HTTP observations but
                # must never be attributed to a prompt or retained as arbitrary text.
                index = index if index is None or (type(index) is int and index >= 0) else -1
                outputs.append((index, reasoning, content, tool))
            finish = get_choice('finish_reason')
            cancelled |= finish == 'abort'
            failed |= finish in ('error', 'parse_error')
    else:
        delta = _get(payload, 'delta')
        reasoning = content = tool = False
        if kind in ('response.reasoning_text.delta', 'response.reasoning_summary_text.delta'):
            reasoning = bool(delta)
        elif kind == 'response.output_text.delta':
            content = bool(delta)
        elif kind == 'response.function_call_arguments.delta':
            tool = bool(delta)
        elif kind == 'response.output_item.added':
            tool = _get(_get(payload, 'item'), 'type') == 'function_call'
        else:
            reasoning = bool(_get(delta, 'thinking'))
            content = bool(_get(delta, 'text'))
            tool = bool(_get(delta, 'partial_json'))
        block = _get(payload, 'content_block')
        reasoning |= bool(_get(block, 'thinking'))
        content |= bool(_get(block, 'text'))
        tool |= _get(block, 'type') == 'tool_use'
        meta = _get(payload, 'meta_info')
        if meta is not None:
            content |= bool(_get(payload, 'text'))
            finish = _get(_get(meta, 'finish_reason'), 'type')
            cancelled |= finish == 'abort'
            failed |= finish == 'error'
        if reasoning or content or tool:
            outputs.append((None, reasoning, content, tool))
    return OutputMetadata(tuple(outputs), terminal, failed, cancelled)


class ObservedBytes(bytes):
    """Carry observation metadata alongside unchanged ASGI response bytes."""
    def __new__(cls, value, metadata):
        instance = super().__new__(cls, value)
        instance.trace_metadata = metadata
        return instance


class ObservedString(str):
    """Preserve metadata when Starlette encodes an SSE frame for ASGI send."""

    def __new__(cls, value, metadata):
        instance = super().__new__(cls, value)
        instance.trace_metadata = metadata
        return instance

    def encode(self, encoding='utf-8', errors='strict'):
        return ObservedBytes(super().encode(encoding, errors), self.trace_metadata)
