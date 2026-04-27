# Copyright (c) OpenMMLab. All rights reserved.
"""Streaming helpers for Anthropic-compatible responses."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from .adapter import map_finish_reason


def _format_sse(event: str, data: dict) -> str:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


async def stream_messages_response(result_generator,
                                   *,
                                   request_id: str,
                                   model: str,
                                   response_parser=None) -> AsyncGenerator[str, None]:
    """Convert LMDeploy generation stream to Anthropic SSE events."""

    yield _format_sse(
        'message_start',
        {
            'type': 'message_start',
            'message': {
                'id': request_id,
                'type': 'message',
                'role': 'assistant',
                'content': [],
                'model': model,
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'output_tokens': 0,
                },
            },
        },
    )
    final_res = None
    input_tokens = 0
    next_block_index = 0
    current_block: dict[str, Any] | None = None
    tool_blocks: dict[int, dict[str, Any]] = {}
    streaming_tools = False

    def _close_current_block() -> str | None:
        nonlocal current_block
        if current_block is None:
            return None
        payload = _format_sse(
            'content_block_stop',
            {
                'type': 'content_block_stop',
                'index': current_block['block_index'],
            },
        )
        current_block = None
        return payload

    def _start_text_or_thinking(kind: str) -> list[str]:
        nonlocal current_block, next_block_index
        events: list[str] = []
        if current_block is not None and current_block.get('kind') == kind:
            return events
        closing = _close_current_block()
        if closing:
            events.append(closing)
        block_index = next_block_index
        next_block_index += 1
        content_block = {'type': kind}
        content_block['text' if kind == 'text' else 'thinking'] = ''
        current_block = dict(kind=kind, block_index=block_index)
        events.append(
            _format_sse(
                'content_block_start',
                {
                    'type': 'content_block_start',
                    'index': block_index,
                    'content_block': content_block,
                },
            ))
        return events

    def _start_tool_block(tool_delta) -> list[str]:
        nonlocal current_block, next_block_index
        events: list[str] = []
        tool_index = tool_delta.index
        block = tool_blocks.get(tool_index)
        if block is None:
            closing = _close_current_block()
            if closing:
                events.append(closing)
            block_index = next_block_index
            next_block_index += 1
            block = dict(
                tool_index=tool_index,
                block_index=block_index,
                tool_use_id=tool_delta.id,
                name='',
            )
            tool_blocks[tool_index] = block
            events.append(
                _format_sse(
                    'content_block_start',
                    {
                        'type': 'content_block_start',
                        'index': block_index,
                        'content_block': {
                            'type': 'tool_use',
                            'id': tool_delta.id,
                            'name': '',
                            'input': {},
                        },
                    },
                ))
        current_block = dict(kind='tool_use', block_index=block['block_index'], tool_index=tool_index)
        return events

    def _emit_text_delta(text: str, thinking: bool) -> str:
        block_index = current_block['block_index']
        delta_key = 'thinking' if thinking else 'text'
        delta_type = 'thinking_delta' if thinking else 'text_delta'
        return _format_sse(
            'content_block_delta',
            {
                'type': 'content_block_delta',
                'index': block_index,
                'delta': {
                    'type': delta_type,
                    delta_key: text,
                },
            },
        )

    async for res in result_generator:
        final_res = res
        input_tokens = res.input_token_len
        text = res.response or ''
        delta_token_ids = res.token_ids if getattr(res, 'token_ids', None) is not None else []

        delta_message = None
        tool_emitted = False
        if response_parser is not None:
            delta_message, tool_emitted = response_parser.stream_chunk(text, delta_token_ids)
        elif text:
            for event in _start_text_or_thinking('text'):
                yield event
            if current_block is not None:
                yield _emit_text_delta(text, thinking=False)

        if tool_emitted:
            streaming_tools = True
        if (response_parser is not None and res.finish_reason == 'stop' and streaming_tools):
            res.finish_reason = 'tool_calls'

        if delta_message is None:
            continue

        if delta_message.reasoning_content:
            for event in _start_text_or_thinking('thinking'):
                yield event
            if current_block is not None:
                yield _emit_text_delta(delta_message.reasoning_content, thinking=True)

        if delta_message.content:
            for event in _start_text_or_thinking('text'):
                yield event
            if current_block is not None:
                yield _emit_text_delta(delta_message.content, thinking=False)

        if delta_message.tool_calls:
            for tool_delta in delta_message.tool_calls:
                for event in _start_tool_block(tool_delta):
                    yield event
                function_delta = getattr(tool_delta, 'function', None)
                if function_delta is None:
                    continue
                partial_json = function_delta.arguments or ''
                if partial_json:
                    yield _format_sse(
                        'content_block_delta',
                        {
                            'type': 'content_block_delta',
                            'index': current_block['block_index'],
                            'delta': {
                                'type': 'input_json_delta',
                                'partial_json': partial_json,
                            },
                        },
                    )

    closing = _close_current_block()
    if closing:
        yield closing

    output_tokens = 0 if final_res is None else final_res.generate_token_len
    stop_reason = map_finish_reason(None if final_res is None else final_res.finish_reason)
    yield _format_sse(
        'message_delta',
        {
            'type': 'message_delta',
            'delta': {
                'stop_reason': stop_reason,
                'stop_sequence': None,
            },
            'usage': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
            },
        },
    )
    yield _format_sse('message_stop', {'type': 'message_stop'})
