# Copyright (c) OpenMMLab. All rights reserved.
"""Text-first OpenAI Responses API endpoint."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import Any, Literal

import shortuuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, Tool, ToolChoice, ToolChoiceFuncName
from lmdeploy.serve.utils.server_utils import validate_json_request


class ResponsesRequest(BaseModel):
    """Request body for ``POST /v1/responses``.

    This is intentionally a Text V1 subset. Unsupported agentic fields are accepted by the model so the endpoint can
    return OpenAI-style 400 errors.
    """

    model_config = ConfigDict(extra='allow')

    input: str | list[dict[str, Any]]
    model: str | None = None
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    stream: bool | None = False
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    top_k: int | None = 40
    stop: str | list[str] | None = None
    seed: int | None = None
    min_p: float = 0.0
    ignore_eos: bool | None = False
    skip_special_tokens: bool | None = True
    include_stop_str_in_output: bool | None = False
    text: dict[str, Any] | None = None
    store: bool | None = True
    background: bool | None = False
    previous_response_id: str | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Any = 'auto'
    request_id: str = Field(default_factory=lambda: f'resp_{shortuuid.random()}')


class ResponseUsage(BaseModel):
    """Token usage in Responses API shape."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ResponseOutputText(BaseModel):
    """Text content part in a Responses output message."""

    type: Literal['output_text'] = 'output_text'
    text: str
    annotations: list[Any] = Field(default_factory=list)


class ResponseOutputMessage(BaseModel):
    """Assistant output item."""

    id: str = Field(default_factory=lambda: f'msg_{shortuuid.random()}')
    type: Literal['message'] = 'message'
    role: Literal['assistant'] = 'assistant'
    status: Literal['in_progress', 'completed', 'incomplete'] = 'completed'
    content: list[ResponseOutputText]


class ResponseOutputFunctionCall(BaseModel):
    """Function call output item in Responses API shape."""

    id: str
    type: Literal['function_call'] = 'function_call'
    call_id: str
    name: str
    arguments: str


class ResponsesResponse(BaseModel):
    """Response body for Text V1 ``POST /v1/responses``."""

    id: str
    object: Literal['response'] = 'response'
    created_at: int
    model: str
    status: Literal['in_progress', 'completed', 'incomplete', 'failed'] = 'completed'
    output: list[ResponseOutputMessage | ResponseOutputFunctionCall] = Field(default_factory=list)
    output_text: str = ''
    usage: ResponseUsage | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    store: bool = False


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
    if request.previous_response_id is not None:
        return _error_response(HTTPStatus.BAD_REQUEST,
                               'previous_response_id is not supported by Responses Text V1.',
                               param='previous_response_id')
    return None


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


def _text_from_content(content: Any, field_name: str) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f'Unsupported `{field_name}` content. Expected string or text content parts.')

    text_parts: list[str] = []
    for idx, part in enumerate(content):
        if not isinstance(part, dict):
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
        if not isinstance(item, dict):
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
        if tool.get('type') != 'function':
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
    if isinstance(tool_choice, dict):
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


def _response_format_from_text(text: dict[str, Any] | None) -> dict[str, Any] | None:
    if not text:
        return None
    text_format = text.get('format')
    if text_format is None:
        return None
    if not isinstance(text_format, dict):
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
    )


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
    status = 'incomplete' if finish_reason == 'length' else 'completed'
    message_status = 'incomplete' if status == 'incomplete' else 'completed'
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
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        store=False,
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
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        store=False,
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
