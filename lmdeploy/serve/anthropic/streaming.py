# Copyright (c) OpenMMLab. All rights reserved.
"""Streaming helpers for Anthropic-compatible responses."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from lmdeploy.serve.openai.protocol import DeltaMessage

from .adapter import map_finish_reason


def _format_sse(event: str, data: dict) -> str:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


async def stream_messages_response(result_generator,
                                   *,
                                   request_id: str,
                                   model: str,
                                   response_parser=None,
                                   return_token_ids: bool = False,
                                   return_routed_experts: bool = False,
                                   logprobs: bool = False) -> AsyncGenerator[str, None]:
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
        same_block = (
            current_block is not None and current_block.get('kind') == 'tool_use'
            and current_block.get('tool_index') == tool_index
            and block is not None
            and current_block.get('block_index') == block['block_index'])
        if not same_block:
            closing = _close_current_block()
            if closing:
                events.append(closing)
        if block is None:
            function_delta = tool_delta.function
            tool_name = '' if function_delta is None else function_delta.name or ''
            block_index = next_block_index
            next_block_index += 1
            block = dict(
                tool_index=tool_index,
                block_index=block_index,
                tool_use_id=tool_delta.id,
                name=tool_name,
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
                            'name': tool_name,
                            'input': {},
                        },
                    },
                ))
        current_block = dict(kind='tool_use', block_index=block['block_index'], tool_index=tool_index)
        return events

    def _emit_text_delta(text: str,
                         thinking: bool,
                         output_ids: list[int] | None = None,
                         output_token_logprobs: list[list] | None = None) -> str:
        block_index = current_block['block_index']
        delta_key = 'thinking' if thinking else 'text'
        delta_type = 'thinking_delta' if thinking else 'text_delta'
        data = {
            'type': 'content_block_delta',
            'index': block_index,
            'delta': {
                'type': delta_type,
                delta_key: text,
            },
        }
        if output_ids is not None:
            data['output_ids'] = output_ids
        if output_token_logprobs is not None:
            data['output_token_logprobs'] = output_token_logprobs
        return _format_sse('content_block_delta', data)

    async for res in result_generator:
        final_res = res
        input_tokens = res.input_token_len
        text = res.response or ''
        delta_token_ids = res.token_ids if res.token_ids is not None else []
        delta_logprobs = None
        if logprobs and res.logprobs and delta_token_ids:
            delta_logprobs = [
                (tok_logprobs[tok], tok)
                for tok, tok_logprobs in zip(delta_token_ids, res.logprobs)
            ]

        stream_deltas = []
        if response_parser is not None:
            stream_deltas = response_parser.stream_chunk(text, delta_token_ids)
        elif text:
            stream_deltas = [(DeltaMessage(role='assistant', content=text), False)]

        should_validate_complete = (
            response_parser is not None
            and res.finish_reason in ('stop', 'length')
            and (return_token_ids or return_routed_experts)
        )
        if should_validate_complete and not response_parser.validate_complete():
            res.finish_reason = 'parse_error'

        for delta_index, (delta_message, tool_emitted) in enumerate(stream_deltas):
            if tool_emitted:
                streaming_tools = True

            if delta_message is None:
                continue

            is_last_delta = delta_index == len(stream_deltas) - 1
            delta_output_ids = delta_token_ids if return_token_ids and is_last_delta else None
            delta_output_logprobs = delta_logprobs if is_last_delta else None
            has_content = bool(delta_message.content)
            has_tools = bool(delta_message.tool_calls)

            if delta_message.reasoning_content:
                for event in _start_text_or_thinking('thinking'):
                    yield event
                if current_block is not None:
                    yield _emit_text_delta(
                        delta_message.reasoning_content,
                        thinking=True,
                        output_ids=None if has_content or has_tools else delta_output_ids,
                        output_token_logprobs=None if has_content or has_tools else delta_output_logprobs,
                    )

            if delta_message.content:
                for event in _start_text_or_thinking('text'):
                    yield event
                if current_block is not None:
                    yield _emit_text_delta(
                        delta_message.content,
                        thinking=False,
                        output_ids=None if has_tools else delta_output_ids,
                        output_token_logprobs=None if has_tools else delta_output_logprobs,
                    )

            if delta_message.tool_calls:
                for tool_index, tool_delta in enumerate(delta_message.tool_calls):
                    for event in _start_tool_block(tool_delta):
                        yield event
                    function_delta = tool_delta.function
                    if function_delta is None:
                        continue
                    partial_json = function_delta.arguments or ''
                    if partial_json:
                        data = {
                            'type': 'content_block_delta',
                            'index': current_block['block_index'],
                            'delta': {
                                'type': 'input_json_delta',
                                'partial_json': partial_json,
                            },
                        }
                        if tool_index == len(delta_message.tool_calls) - 1:
                            if delta_output_ids is not None:
                                data['output_ids'] = delta_output_ids
                            if delta_output_logprobs is not None:
                                data['output_token_logprobs'] = delta_output_logprobs
                        yield _format_sse(
                            'content_block_delta',
                            data,
                        )

        if response_parser is not None and res.finish_reason == 'stop' and streaming_tools:
            res.finish_reason = 'tool_calls'

    closing = _close_current_block()
    if closing:
        yield closing

    output_tokens = 0 if final_res is None else final_res.generate_token_len
    stop_reason = map_finish_reason(None if final_res is None else final_res.finish_reason)
    message_delta_data = {
        'type': 'message_delta',
        'delta': {
            'stop_reason': stop_reason,
            'stop_sequence': None,
        },
        'usage': {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
        },
    }
    if return_routed_experts and final_res is not None:
        message_delta_data['routed_experts'] = final_res.routed_experts
    yield _format_sse('message_delta', message_delta_data)
    yield _format_sse('message_stop', {'type': 'message_stop'})
