# Copyright (c) OpenMMLab. All rights reserved.
"""Responses streaming event helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import shortuuid
from openai.types.responses import (
    ResponseFunctionToolCall as ResponseOutputFunctionCall,
)
from openai.types.responses import (
    ResponseOutputMessage,
)

from lmdeploy.serve.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
)
from lmdeploy.serve.openai.responses.response import _make_response, _response_metadata_kwargs
from lmdeploy.serve.openai.utils import filter_parallel_tool_call_deltas


def _sse(event: str, data: dict[str, Any]) -> str:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


async def _stream_response(result_generator,
                           *,
                           request: ResponsesRequest,
                           model_name: str,
                           created_time: int,
                           response_parser) -> AsyncGenerator[str, None]:
    initial_response = ResponsesResponse(
        id=request.request_id,
        created_at=created_time,
        model=model_name,
        status='in_progress',
        **_response_metadata_kwargs(request),
    ).model_dump(exclude_none=True)
    yield _sse('response.created', {'type': 'response.created', 'sequence_number': 0, 'response': initial_response})
    yield _sse('response.in_progress', {
        'type': 'response.in_progress',
        'sequence_number': 1,
        'response': initial_response,
    })

    message_id = f'msg_{shortuuid.random()}'
    next_output_index = 0
    content_index = 0
    sequence_number = 2
    text_started = False
    text_output_index = None
    text = ''
    tool_states: dict[int, dict[str, Any]] = {}
    streaming_tools = False
    final_res = None

    def _start_text_item() -> list[str]:
        nonlocal next_output_index, sequence_number, text_output_index, text_started
        if text_started:
            return []
        text_started = True
        text_output_index = next_output_index
        events = [
            _sse(
                'response.output_item.added',
                {
                    'type': 'response.output_item.added',
                    'sequence_number': sequence_number,
                    'response_id': request.request_id,
                    'output_index': text_output_index,
                    'item': {
                        'id': message_id,
                        'type': 'message',
                        'role': 'assistant',
                        'status': 'in_progress',
                        'content': [],
                    },
                },
            )
        ]
        sequence_number += 1
        events.append(
            _sse(
                'response.content_part.added',
                {
                    'type': 'response.content_part.added',
                    'sequence_number': sequence_number,
                    'response_id': request.request_id,
                    'output_index': text_output_index,
                    'item_id': message_id,
                    'content_index': content_index,
                    'part': {
                        'type': 'output_text',
                        'text': '',
                        'annotations': [],
                    },
                },
            ))
        sequence_number += 1
        next_output_index += 1
        return events

    def _start_tool_item(tool_delta) -> list[str]:
        nonlocal next_output_index, sequence_number
        tool_index = tool_delta.index
        state = tool_states.get(tool_index)
        function_delta = getattr(tool_delta, 'function', None)
        if state is not None:
            if function_delta is not None and function_delta.name:
                state['name'] = function_delta.name
            return []
        tool_id = tool_delta.id or f'call_{shortuuid.random()}'
        name = '' if function_delta is None else function_delta.name or ''
        state = dict(
            item_id=tool_id,
            call_id=tool_id,
            name=name,
            arguments='',
            output_index=next_output_index,
        )
        tool_states[tool_index] = state
        next_output_index += 1
        event = _sse(
            'response.output_item.added',
            {
                'type': 'response.output_item.added',
                'sequence_number': sequence_number,
                'response_id': request.request_id,
                'output_index': state['output_index'],
                'item': {
                    'id': state['item_id'],
                    'type': 'function_call',
                    'call_id': state['call_id'],
                    'name': state['name'],
                    'arguments': '',
                    'status': 'in_progress',
                },
            },
        )
        sequence_number += 1
        return [event]

    async for res in result_generator:
        final_res = res
        delta = res.response or ''
        delta_token_ids = res.token_ids if getattr(res, 'token_ids', None) is not None else []
        stream_deltas = response_parser.stream_chunk(delta, delta_token_ids)

        for delta_message, tool_emitted in stream_deltas:
            content_delta = getattr(delta_message, 'content', None) or ''
            tool_deltas = getattr(delta_message, 'tool_calls', None)

            if content_delta:
                for event in _start_text_item():
                    yield event
                text += content_delta
                yield _sse(
                    'response.output_text.delta',
                    {
                        'type': 'response.output_text.delta',
                        'sequence_number': sequence_number,
                        'response_id': request.request_id,
                        'item_id': message_id,
                        'output_index': text_output_index,
                        'content_index': content_index,
                        'delta': content_delta,
                    },
                )
                sequence_number += 1
            if tool_deltas:
                for tool_delta in filter_parallel_tool_call_deltas(tool_deltas, request.parallel_tool_calls):
                    if tool_emitted:
                        streaming_tools = True
                    for event in _start_tool_item(tool_delta):
                        yield event
                    function_delta = getattr(tool_delta, 'function', None)
                    if function_delta is None:
                        continue
                    state = tool_states[tool_delta.index]
                    if function_delta.name:
                        state['name'] = function_delta.name
                    arguments_delta = function_delta.arguments or ''
                    if arguments_delta:
                        state['arguments'] += arguments_delta
                        yield _sse(
                            'response.function_call_arguments.delta',
                            {
                                'type': 'response.function_call_arguments.delta',
                                'sequence_number': sequence_number,
                                'response_id': request.request_id,
                                'item_id': state['item_id'],
                                'output_index': state['output_index'],
                                'delta': arguments_delta,
                            },
                        )
                        sequence_number += 1
            elif tool_emitted:
                streaming_tools = True
        if res.finish_reason == 'stop' and streaming_tools:
            res.finish_reason = 'tool_calls'

    input_tokens = 0 if final_res is None else final_res.input_token_len
    output_tokens = 0 if final_res is None else final_res.generate_token_len
    finish_reason = None if final_res is None else final_res.finish_reason
    if not text and not tool_states:
        for event in _start_text_item():
            yield event
    tool_calls = [
        ResponseOutputFunctionCall(
            id=state['item_id'],
            type='function_call',
            call_id=state['call_id'],
            name=state['name'],
            arguments=state['arguments'],
            status='completed',
        ) for _, state in sorted(tool_states.items(), key=lambda item: item[1]['output_index'])
    ]
    final_response = _make_response(
        request=request,
        model_name=model_name,
        created_time=created_time,
        text=text,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
        message_id=message_id,
    )
    output_by_index: dict[int, ResponseOutputMessage | ResponseOutputFunctionCall] = {}
    text_item = next((item for item in final_response.output if isinstance(item, ResponseOutputMessage)), None)
    if text_started and text_output_index is not None and text_item is not None:
        output_by_index[text_output_index] = text_item
    tool_items = {
        item.call_id: item
        for item in final_response.output
        if isinstance(item, ResponseOutputFunctionCall)
    }
    for state in tool_states.values():
        tool_item = tool_items.get(state['call_id'])
        if tool_item is not None:
            output_by_index[state['output_index']] = tool_item
    if output_by_index:
        final_response.output = [item for _, item in sorted(output_by_index.items())]
    if text_started:
        output_text = {
            'type': 'output_text',
            'text': text,
            'annotations': [],
        }
        assert text_output_index is not None
        assert text_item is not None
        yield _sse(
            'response.output_text.done',
            {
                'type': 'response.output_text.done',
                'sequence_number': sequence_number,
                'response_id': request.request_id,
                'item_id': message_id,
                'output_index': text_output_index,
                'content_index': content_index,
                'text': text,
            },
        )
        sequence_number += 1
        yield _sse(
            'response.content_part.done',
            {
                'type': 'response.content_part.done',
                'sequence_number': sequence_number,
                'response_id': request.request_id,
                'item_id': message_id,
                'output_index': text_output_index,
                'content_index': content_index,
                'part': output_text,
            },
        )
        sequence_number += 1
        yield _sse(
            'response.output_item.done',
            {
                'type': 'response.output_item.done',
                'sequence_number': sequence_number,
                'response_id': request.request_id,
                'output_index': text_output_index,
                'item': text_item.model_dump(exclude_none=True),
            },
        )
        sequence_number += 1
    for state in sorted(tool_states.values(), key=lambda item: item['output_index']):
        item = {
            'id': state['item_id'],
            'type': 'function_call',
            'call_id': state['call_id'],
            'name': state['name'],
            'arguments': state['arguments'],
            'status': 'completed',
        }
        yield _sse(
            'response.function_call_arguments.done',
            {
                'type': 'response.function_call_arguments.done',
                'sequence_number': sequence_number,
                'response_id': request.request_id,
                'item_id': state['item_id'],
                'output_index': state['output_index'],
                'name': state['name'],
                'arguments': state['arguments'],
            },
        )
        sequence_number += 1
        yield _sse(
            'response.output_item.done',
            {
                'type': 'response.output_item.done',
                'sequence_number': sequence_number,
                'response_id': request.request_id,
                'output_index': state['output_index'],
                'item': item,
            },
        )
        sequence_number += 1
    terminal_event = {
        'completed': 'response.completed',
        'incomplete': 'response.incomplete',
        'failed': 'response.failed',
        'cancelled': 'response.failed',
    }[final_response.status]
    yield _sse(
        terminal_event,
        {
            'type': terminal_event,
            'sequence_number': sequence_number,
            'response': final_response.model_dump(exclude_none=True),
        },
    )
