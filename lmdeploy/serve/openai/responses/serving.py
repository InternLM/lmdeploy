# Copyright (c) OpenMMLab. All rights reserved.
"""Text-first OpenAI Responses API endpoint."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import Any, Literal

import shortuuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, Tool, ToolChoice, ToolChoiceFuncName
from lmdeploy.serve.openai.responses.protocol import (
    ResponseIncompleteDetails,
    ResponseOutputFunctionCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponseUsage,
)
from lmdeploy.serve.utils.server_utils import validate_json_request

logger = logging.getLogger(__name__)


def _error_response(status: HTTPStatus, message: str, *, param: str | None = None) -> JSONResponse:
    payload = {
        'error': {
            'message': message,
            'type': 'invalid_request_error',
            'param': param,
            'code': status.value,
        }
    }
    return JSONResponse(payload, status_code=status.value)


def _get_model_list(server_context) -> list[str]:
    model_names = [server_context.async_engine.model_name]
    cfg = server_context.async_engine.backend_config
    model_names += getattr(cfg, 'adapters', None) or []
    return model_names


def _validate_text_v1_request(request: ResponsesRequest) -> JSONResponse | None:
    if request.background:
        return _error_response(HTTPStatus.BAD_REQUEST, 'background mode is not supported by Responses Text V1.',
                               param='background')
    if request.context_management is not None:
        return _error_response(HTTPStatus.BAD_REQUEST,
                               'context_management is not supported by Responses Text V1.',
                               param='context_management')
    if request.conversation is not None:
        return _error_response(HTTPStatus.BAD_REQUEST, 'conversation is not supported by Responses Text V1.',
                               param='conversation')
    if request.previous_response_id is not None:
        return _error_response(HTTPStatus.BAD_REQUEST,
                               'previous_response_id is not supported by Responses Text V1.',
                               param='previous_response_id')
    if request.prompt is not None:
        return _error_response(HTTPStatus.BAD_REQUEST, 'prompt is not supported by Responses Text V1.',
                               param='prompt')
    if request.input is None:
        return _error_response(HTTPStatus.BAD_REQUEST, 'input is required by Responses Text V1.', param='input')
    return None


def _warn_ignored_request_fields(request: ResponsesRequest) -> None:
    ignored_fields: list[str] = []
    for field_name in (
            'include',
            'max_tool_calls',
            'metadata',
            'logit_bias',
            'prompt_cache_key',
            'prompt_cache_retention',
            'reasoning',
            'safety_identifier',
            'stream_options',
            'top_logprobs',
            'user',
            'presence_penalty',
            'frequency_penalty',
    ):
        if getattr(request, field_name) is not None:
            ignored_fields.append(field_name)
    if request.service_tier not in (None, 'auto'):
        ignored_fields.append('service_tier')
    if request.truncation not in (None, 'disabled'):
        ignored_fields.append('truncation')

    text = _as_dict(request.text)
    if text.get('verbosity') is not None:
        ignored_fields.append('text.verbosity')

    if ignored_fields:
        logger.warning('Ignoring unsupported Responses request fields: %s.', ', '.join(ignored_fields))


def _stringify_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _generation_messages_from_parser(messages: list[dict[str, Any]], parsed_request: ChatCompletionRequest | None):
    if parsed_request is None:
        return messages
    return parsed_request.messages


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, 'model_dump'):
        return value.model_dump(exclude_none=True, by_alias=True)
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    return {}


def _text_from_content(content: Any, field_name: str) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f'Unsupported `{field_name}` content. Expected string or text content parts.')

    text_parts: list[str] = []
    for idx, part in enumerate(content):
        part = _as_dict(part)
        if not part:
            raise ValueError(f'Unsupported `{field_name}` content part at index {idx}.')
        part_type = part.get('type')
        if part_type in ('input_text', 'output_text', 'text'):
            text = part.get('text')
            if text is None:
                raise ValueError(f'Missing `text` in `{field_name}` content part at index {idx}.')
            text_parts.append(text)
            continue
        raise ValueError(f'Unsupported Responses input content part type: {part_type!r}.')
    return ''.join(text_parts)


def _messages_from_input(request: ResponsesRequest) -> list[dict[str, Any]]:
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    if request.instructions:
        system_parts.append(request.instructions)

    if isinstance(request.input, str):
        messages.append(dict(role='user', content=request.input))
        return ([dict(role='system', content='\n\n'.join(system_parts))] if system_parts else []) + messages

    for idx, item in enumerate(request.input):
        item = _as_dict(item)
        if not item:
            raise ValueError(f'Unsupported Responses input item at index {idx}.')

        item_type = item.get('type', 'message')
        if item_type == 'function_call':
            call_id = item.get('call_id') or item.get('id')
            name = item.get('name')
            if not call_id or not name:
                raise ValueError(f'Missing `call_id` or `name` in function_call item at index {idx}.')
            messages.append(
                dict(
                    role='assistant',
                    content=None,
                    tool_calls=[
                        dict(
                            id=call_id,
                            type='function',
                            function=dict(
                                name=name,
                                arguments=_stringify_value(item.get('arguments')),
                            ),
                        )
                    ],
                ))
            continue
        if item_type == 'function_call_output':
            call_id = item.get('call_id')
            if not call_id:
                raise ValueError(f'Missing `call_id` in function_call_output item at index {idx}.')
            messages.append(
                dict(
                    role='tool',
                    tool_call_id=call_id,
                    content=_stringify_value(item.get('output')),
                ))
            continue
        if item_type != 'message':
            raise ValueError(f'Unsupported Responses input item type: {item_type!r}.')

        role = item.get('role')
        if role == 'developer':
            role = 'system'
        if role not in ('system', 'user', 'assistant'):
            raise ValueError(f'Unsupported Responses message role at index {idx}: {role!r}.')
        content = _text_from_content(item.get('content', ''), f'input[{idx}].content')
        if role == 'system':
            system_parts.append(content)
        else:
            messages.append(dict(role=role, content=content))
    return ([dict(role='system', content='\n\n'.join(system_parts))] if system_parts else []) + messages


def _openai_tools_from_responses(request: ResponsesRequest) -> list[Tool] | None:
    """Convert Responses function tools into LMDeploy/OpenAI tool entries."""

    if not request.tools:
        return None
    tools: list[Tool] = []
    for idx, tool in enumerate(request.tools):
        tool = _as_dict(tool)
        if tool.get('type') != 'function':
            logger.warning('Ignoring unsupported Responses tool type at index %s: %r.', idx, tool.get('type'))
            continue
        name = tool.get('name')
        if not name:
            raise ValueError(f'Missing function tool `name` at index {idx}.')
        tools.append(
            Tool(
                type='function',
                function=dict(
                    name=name,
                    description=tool.get('description'),
                    parameters=tool.get('parameters'),
                ),
            ))
    return tools or None


def _tool_choice_from_responses(tool_choice: Any,
                                tools: list[Tool] | None = None) -> ToolChoice | Literal['auto', 'required', 'none']:
    """Map Responses tool_choice to the OpenAI chat tool_choice shape used
    internally."""

    has_tools = bool(tools)
    if tool_choice is None:
        return 'auto' if has_tools else 'none'
    if isinstance(tool_choice, str):
        if tool_choice in ('auto', 'none'):
            return tool_choice if has_tools else 'none'
        if tool_choice == 'required':
            if not has_tools:
                raise ValueError("Tool choice 'required' must be specified with `tools`.")
            return tool_choice
        raise ValueError(f'Unsupported tool_choice: {tool_choice!r}.')
    tool_choice = _as_dict(tool_choice)
    if tool_choice:
        if tool_choice.get('type') == 'function':
            name = tool_choice.get('name')
            if not name:
                raise ValueError('Missing `name` in function tool_choice.')
            tool_names = {tool.function.name for tool in tools or []}
            if name not in tool_names:
                raise ValueError(f"Tool choice 'function' not found in `tools`: {name!r}.")
            return ToolChoice(function=ToolChoiceFuncName(name=name))
        raise ValueError(f'Unsupported tool_choice type: {tool_choice.get("type")!r}.')
    raise ValueError('Unsupported tool_choice. Expected string or function tool choice object.')


def _response_format_from_text(text: Any) -> dict[str, Any] | None:
    if not text:
        return None
    text = _as_dict(text)
    text_format = text.get('format')
    if text_format is None:
        return None
    text_format = _as_dict(text_format)
    if not text_format:
        raise ValueError('`text.format` must be an object.')
    format_type = text_format.get('type', 'text')
    if format_type == 'text':
        return None
    if format_type == 'json_object':
        return dict(type='json_object')
    if format_type == 'json_schema':
        return dict(
            type='json_schema',
            json_schema=dict(
                name=text_format.get('name', 'response'),
                schema=text_format.get('schema'),
                strict=text_format.get('strict', False),
            ),
        )
    raise ValueError(f'Unsupported text.format type: {format_type!r}.')


def _to_generation_config(request: ResponsesRequest) -> GenerationConfig:
    stop_words = [request.stop] if isinstance(request.stop, str) else request.stop
    return GenerationConfig(
        max_new_tokens=request.max_output_tokens,
        do_sample=True,
        top_k=40 if request.top_k is None else request.top_k,
        top_p=1.0 if request.top_p is None else request.top_p,
        temperature=1.0 if request.temperature is None else request.temperature,
        stop_words=stop_words,
        ignore_eos=request.ignore_eos,
        skip_special_tokens=request.skip_special_tokens,
        include_stop_str_in_output=request.include_stop_str_in_output,
        response_format=_response_format_from_text(request.text),
        min_p=request.min_p,
        random_seed=request.seed,
        repetition_penalty=1.0 if request.repetition_penalty is None else request.repetition_penalty,
    )


def _response_metadata_kwargs(request: ResponsesRequest) -> dict[str, Any]:
    return dict(
        instructions=request.instructions,
        metadata=request.metadata,
        max_output_tokens=request.max_output_tokens,
        max_tool_calls=request.max_tool_calls,
        parallel_tool_calls=request.parallel_tool_calls,
        previous_response_id=request.previous_response_id,
        prompt=request.prompt,
        prompt_cache_key=request.prompt_cache_key,
        prompt_cache_retention=request.prompt_cache_retention,
        reasoning=request.reasoning,
        safety_identifier=request.safety_identifier,
        service_tier=request.service_tier,
        background=bool(request.background),
        conversation=request.conversation,
        store=False,
        temperature=request.temperature,
        text=request.text,
        tool_choice=request.tool_choice,
        tools=request.tools,
        top_logprobs=request.top_logprobs,
        top_p=request.top_p,
        truncation=request.truncation,
        user=request.user,
    )


def _filter_parallel_tool_calls(request: ResponsesRequest, tool_calls: list[Any] | None) -> list[Any] | None:
    if request.parallel_tool_calls is not False or not tool_calls:
        return tool_calls
    return tool_calls[:1]


def _make_response(*,
                   request: ResponsesRequest,
                   model_name: str,
                   created_time: int,
                   text: str | None,
                   tool_calls: list[Any] | None = None,
                   input_tokens: int,
                   output_tokens: int,
                   finish_reason: str | None,
                   message_id: str | None = None) -> ResponsesResponse:
    text = text or ''
    tool_calls = _filter_parallel_tool_calls(request, tool_calls)
    status = 'incomplete' if finish_reason == 'length' else 'completed'
    message_status = 'incomplete' if status == 'incomplete' else 'completed'
    incomplete_details = None
    if status == 'incomplete':
        incomplete_details = ResponseIncompleteDetails(reason='max_output_tokens')
    output: list[ResponseOutputMessage | ResponseOutputFunctionCall] = []
    if text or not tool_calls:
        output.append(
            ResponseOutputMessage(
                id=message_id or f'msg_{shortuuid.random()}',
                status=message_status,
                content=[ResponseOutputText(text=text)],
            ))
    if tool_calls:
        for tool_call in tool_calls:
            if isinstance(tool_call, ResponseOutputFunctionCall):
                output.append(tool_call)
                continue
            function = getattr(tool_call, 'function', None)
            if function is None:
                continue
            call_id = getattr(tool_call, 'id', None) or f'call_{shortuuid.random()}'
            output.append(
                ResponseOutputFunctionCall(
                    id=call_id,
                    call_id=call_id,
                    name=function.name,
                    arguments=function.arguments or '',
                ))
    return ResponsesResponse(
        id=request.request_id,
        created_at=created_time,
        model=model_name,
        status=status,
        output=output,
        output_text=text,
        usage=ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        incomplete_details=incomplete_details,
        **_response_metadata_kwargs(request),
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f'event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


async def _stream_response(result_generator,
                           *,
                           request: ResponsesRequest,
                           model_name: str,
                           created_time: int,
                           response_parser=None) -> AsyncGenerator[str, None]:
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
        delta_message = None
        tool_emitted = False
        if response_parser is not None:
            delta_token_ids = res.token_ids if getattr(res, 'token_ids', None) is not None else []
            delta_message, tool_emitted = response_parser.stream_chunk(delta, delta_token_ids)
        elif delta:
            delta_message = dict(content=delta)

        if tool_emitted:
            streaming_tools = True
        if response_parser is not None and res.finish_reason == 'stop' and streaming_tools:
            res.finish_reason = 'tool_calls'

        content_delta = ''
        tool_deltas = None
        if isinstance(delta_message, dict):
            content_delta = delta_message.get('content', '')
        elif delta_message is not None:
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
            for tool_delta in tool_deltas:
                if request.parallel_tool_calls is False and tool_delta.index != 0:
                    continue
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

    input_tokens = 0 if final_res is None else final_res.input_token_len
    output_tokens = 0 if final_res is None else final_res.generate_token_len
    finish_reason = None if final_res is None else final_res.finish_reason
    tool_calls = [
        ResponseOutputFunctionCall(
            id=state['item_id'],
            call_id=state['call_id'],
            name=state['name'],
            arguments=state['arguments'],
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
    yield _sse(
        'response.completed',
        {
            'type': 'response.completed',
            'sequence_number': sequence_number,
            'response': final_response.model_dump(exclude_none=True),
        },
    )


def create_responses_router(server_context) -> APIRouter:
    """Create router for the Text V1 Responses endpoint."""

    router = APIRouter(tags=['openai'])

    @router.post('/v1/responses', dependencies=[Depends(validate_json_request)])
    async def create_response(request: ResponsesRequest, raw_request: Request):
        validation_error = _validate_text_v1_request(request)
        if validation_error is not None:
            return validation_error
        _warn_ignored_request_fields(request)

        model_name = request.model or server_context.async_engine.model_name
        if model_name not in _get_model_list(server_context):
            return _error_response(HTTPStatus.NOT_FOUND, f'The model {model_name!r} does not exist.', param='model')

        try:
            messages = _messages_from_input(request)
        except ValueError as err:
            return _error_response(HTTPStatus.BAD_REQUEST, str(err), param='input')
        try:
            gen_config = _to_generation_config(request)
        except ValueError as err:
            return _error_response(HTTPStatus.BAD_REQUEST, str(err), param='text')
        try:
            tools = _openai_tools_from_responses(request)
        except ValueError as err:
            return _error_response(HTTPStatus.BAD_REQUEST, str(err), param='tools')
        try:
            tool_choice = _tool_choice_from_responses(request.tool_choice, tools)
        except ValueError as err:
            return _error_response(HTTPStatus.BAD_REQUEST, str(err), param='tool_choice')

        parser_cls = getattr(server_context, 'response_parser_cls', None)
        tools_enabled = tools and tool_choice != 'none'
        if tools_enabled and (parser_cls is None or parser_cls.tool_parser_cls is None):
            return _error_response(
                HTTPStatus.BAD_REQUEST,
                'Please launch the api_server with --tool-call-parser if you want to use tool calling.',
                param='tools',
            )
        parser_tools = tools if tools_enabled else None

        response_parser = None
        parsed_request = None
        if parser_cls is not None:
            tokenizer_holder = server_context.async_engine.tokenizer
            tokenizer = getattr(getattr(tokenizer_holder, 'model', None), 'model', tokenizer_holder)
            openai_request = ChatCompletionRequest(
                model=model_name,
                messages=messages,
                max_tokens=request.max_output_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop,
                tools=parser_tools,
                tool_choice=tool_choice,
            )
            response_parser = parser_cls(request=openai_request, tokenizer=tokenizer)
            parsed_request = response_parser.request

        session = server_context.create_session(-1)
        adapter_name = None if model_name == server_context.async_engine.model_name else model_name
        result_generator = server_context.async_engine.generate(
            _generation_messages_from_parser(messages, parsed_request),
            session,
            gen_config=gen_config,
            tools=None if parsed_request is None else parsed_request.tools,
            stream_response=True,
            sequence_start=True,
            sequence_end=True,
            do_preprocess=True,
            adapter_name=adapter_name,
        )
        created_time = int(time.time())

        if request.stream:
            return StreamingResponse(
                _stream_response(
                    result_generator,
                    request=request,
                    model_name=model_name,
                    created_time=created_time,
                    response_parser=response_parser,
                ),
                media_type='text/event-stream',
            )

        text = ''
        final_token_ids: list[int] = []
        final_res = None
        async for res in result_generator:
            if await raw_request.is_disconnected():
                await session.async_abort()
                return _error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected')
            final_res = res
            text += res.response or ''
            if getattr(res, 'token_ids', None):
                final_token_ids.extend(res.token_ids)

        if final_res is None:
            return _error_response(HTTPStatus.INTERNAL_SERVER_ERROR, 'No generation output from engine.')

        tool_calls = None
        if response_parser is not None:
            try:
                text, tool_calls, _reasoning_content = response_parser.parse_complete(text, final_token_ids)
            except Exception as err:
                return _error_response(HTTPStatus.BAD_REQUEST, f'Failed to parse output: {err}')
            if tool_calls and final_res.finish_reason == 'stop':
                final_res.finish_reason = 'tool_calls'

        response = _make_response(
            request=request,
            model_name=model_name,
            created_time=created_time,
            text=text,
            tool_calls=tool_calls,
            input_tokens=final_res.input_token_len,
            output_tokens=final_res.generate_token_len,
            finish_reason=final_res.finish_reason,
        )
        return response.model_dump(exclude_none=True)

    return router
