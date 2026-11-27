# Copyright (c) OpenMMLab. All rights reserved.
"""Responses streaming event helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any, Literal

import shortuuid
from openai.types.responses import (
    ResponseFunctionToolCall as ResponseOutputFunctionCall,
)
from openai.types.responses import (
    ResponseOutputItem,
    ResponseOutputMessage,
)

from lmdeploy.serve.core.exceptions import RequestError
from lmdeploy.serve.observability.request_trace import trace_frame
from lmdeploy.serve.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
)
from lmdeploy.serve.openai.responses.response import make_reasoning_item, make_response, response_metadata_kwargs
from lmdeploy.serve.openai.utils import filter_parallel_tool_call_deltas


def _sse(event: str, data: dict[str, Any]) -> str:
    return trace_frame(f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n', data)


async def _results_or_error(result_generator):
    try:
        async for result in result_generator:
            yield result
    except RequestError as error:
        yield error


async def stream_response(parsed_stream,
                          *,
                          request: ResponsesRequest,
                          model_name: str,
                          created_time: int) -> AsyncGenerator[str, None]:
    initial_response = ResponsesResponse(
        id=request.request_id,
        created_at=created_time,
        model=model_name,
        status='in_progress',
        **response_metadata_kwargs(request),
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
    reasoning_state: tuple[int, str] | None = None
    reasoning_parts: list[str] = []
    reasoning_items: dict[int, ResponseOutputItem] = {}
    final_chunk = None
    runtime_error = None

    def _reasoning_event(event: str, output_index: int, **payload: Any) -> str:
        nonlocal sequence_number
        frame = _sse(event, dict(type=event, sequence_number=sequence_number, response_id=request.request_id,
                                 output_index=output_index, **payload))
        sequence_number += 1
        return frame

    def _finish_reasoning_item(status: Literal['completed', 'incomplete']) -> list[str]:
        nonlocal reasoning_state
        if reasoning_state is None:
            return []
        index, item_id = reasoning_state
        reasoning_text = ''.join(reasoning_parts)
        item = make_reasoning_item(reasoning_text, item_id=item_id, status=status)
        reasoning_items[index] = item
        reasoning_state = None
        reasoning_parts.clear()
        return [
            _reasoning_event('response.reasoning_text.done', index, item_id=item_id, content_index=0,
                             text=reasoning_text),
            _reasoning_event('response.content_part.done', index, item_id=item_id, content_index=0,
                             part={'type': 'reasoning_text', 'text': reasoning_text}),
            _reasoning_event('response.output_item.done', index, item=item.model_dump(exclude_none=True)),
        ]

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
        function_delta = tool_delta.function
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

    async for chunk in _results_or_error(parsed_stream):
        if isinstance(chunk, RequestError):
            runtime_error = chunk
            break
        final_chunk = chunk
        delta_message = chunk.delta_message
        content_delta = delta_message.content or ''
        tool_deltas = filter_parallel_tool_call_deltas(delta_message.tool_calls or [], request.parallel_tool_calls)
        reasoning_delta = delta_message.reasoning_content
        if reasoning_delta:
            if reasoning_state is None:
                item = make_reasoning_item('', status='in_progress')
                reasoning_state = (next_output_index, item.id)
                next_output_index += 1
                yield _reasoning_event('response.output_item.added', reasoning_state[0],
                                       item=item.model_dump(exclude_none=True))
                yield _reasoning_event('response.content_part.added', reasoning_state[0], item_id=item.id,
                                       content_index=0, part={'type': 'reasoning_text', 'text': ''})
            index, item_id = reasoning_state
            reasoning_parts.append(reasoning_delta)
            yield _reasoning_event('response.reasoning_text.delta', index, item_id=item_id, content_index=0,
                                   delta=reasoning_delta)

        if content_delta or tool_deltas:
            for event in _finish_reasoning_item('completed'):
                yield event

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
            for tool_delta in tool_deltas:
                for event in _start_tool_item(tool_delta):
                    yield event
                function_delta = tool_delta.function
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

    input_tokens = 0 if final_chunk is None else final_chunk.input_token_len
    output_tokens = 0 if final_chunk is None else final_chunk.generate_token_len
    finish_reason = ('error' if runtime_error is not None else
                     None if final_chunk is None else final_chunk.finish_reason)
    reasoning_status = 'incomplete' if finish_reason in ('length', 'error', 'abort') else 'completed'
    for event in _finish_reasoning_item(reasoning_status):
        yield event
    if not (text or tool_states or reasoning_items):
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
    final_response = make_response(
        request=request,
        model_name=model_name,
        created_time=created_time,
        text=text,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        cached_tokens=0 if final_chunk is None else final_chunk.cached_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=0 if final_chunk is None else final_chunk.reasoning_tokens or 0,
        finish_reason=finish_reason,
        message_id=message_id,
        error_code=None if runtime_error is None else runtime_error.code.value,
        error_message=None if runtime_error is None else runtime_error.message,
    )
    output_by_index = reasoning_items.copy()
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
