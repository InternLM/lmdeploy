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


class ResponsesResponse(BaseModel):
    """Response body for Text V1 ``POST /v1/responses``."""

    id: str
    object: Literal['response'] = 'response'
    created_at: int
    model: str
    status: Literal['in_progress', 'completed', 'incomplete', 'failed'] = 'completed'
    output: list[ResponseOutputMessage] = Field(default_factory=list)
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
    if request.tools:
        return _error_response(HTTPStatus.BAD_REQUEST, 'tools are not supported by Responses Text V1.', param='tools')
    if request.tool_choice not in ('auto', 'none', None):
        return _error_response(HTTPStatus.BAD_REQUEST, 'tool_choice is not supported by Responses Text V1.',
                               param='tool_choice')
    return None


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


def _messages_from_input(request: ResponsesRequest) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if request.instructions:
        messages.append(dict(role='system', content=request.instructions))

    if isinstance(request.input, str):
        messages.append(dict(role='user', content=request.input))
        return messages

    for idx, item in enumerate(request.input):
        if not isinstance(item, dict):
            raise ValueError(f'Unsupported Responses input item at index {idx}.')

        item_type = item.get('type', 'message')
        if item_type != 'message':
            raise ValueError(f'Unsupported Responses input item type: {item_type!r}.')

        role = item.get('role')
        if role not in ('system', 'user', 'assistant'):
            raise ValueError(f'Unsupported Responses message role at index {idx}: {role!r}.')
        content = _text_from_content(item.get('content', ''), f'input[{idx}].content')
        messages.append(dict(role=role, content=content))
    return messages


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
                   text: str,
                   input_tokens: int,
                   output_tokens: int,
                   finish_reason: str | None,
                   message_id: str | None = None) -> ResponsesResponse:
    status = 'incomplete' if finish_reason == 'length' else 'completed'
    message_status = 'incomplete' if status == 'incomplete' else 'completed'
    return ResponsesResponse(
        id=request.request_id,
        created_at=created_time,
        model=model_name,
        status=status,
        output=[
            ResponseOutputMessage(
                id=message_id or f'msg_{shortuuid.random()}',
                status=message_status,
                content=[ResponseOutputText(text=text)],
            )
        ],
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
                           created_time: int) -> AsyncGenerator[str, None]:
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
    output_index = 0
    content_index = 0
    sequence_number = 2
    started = False
    text = ''
    final_res = None

    async for res in result_generator:
        final_res = res
        delta = res.response or ''
        if not started:
            started = True
            yield _sse(
                'response.output_item.added',
                {
                    'type': 'response.output_item.added',
                    'sequence_number': sequence_number,
                    'output_index': output_index,
                    'item': {
                        'id': message_id,
                        'type': 'message',
                        'role': 'assistant',
                        'status': 'in_progress',
                        'content': [],
                    },
                },
            )
            sequence_number += 1
            yield _sse(
                'response.content_part.added',
                {
                    'type': 'response.content_part.added',
                    'sequence_number': sequence_number,
                    'output_index': output_index,
                    'item_id': message_id,
                    'content_index': content_index,
                    'part': {
                        'type': 'output_text',
                        'text': '',
                        'annotations': [],
                    },
                },
            )
            sequence_number += 1
        if delta:
            text += delta
            yield _sse(
                'response.output_text.delta',
                {
                    'type': 'response.output_text.delta',
                    'sequence_number': sequence_number,
                    'item_id': message_id,
                    'output_index': output_index,
                    'content_index': content_index,
                    'delta': delta,
                },
            )
            sequence_number += 1

    input_tokens = 0 if final_res is None else final_res.input_token_len
    output_tokens = 0 if final_res is None else final_res.generate_token_len
    finish_reason = None if final_res is None else final_res.finish_reason
    final_response = _make_response(
        request=request,
        model_name=model_name,
        created_time=created_time,
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
        message_id=message_id,
    )
    output_text = {
        'type': 'output_text',
        'text': text,
        'annotations': [],
    }
    yield _sse(
        'response.output_text.done',
        {
            'type': 'response.output_text.done',
            'sequence_number': sequence_number,
            'item_id': message_id,
            'output_index': output_index,
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
            'item_id': message_id,
            'output_index': output_index,
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
            'output_index': output_index,
            'item': final_response.output[0].model_dump(exclude_none=True),
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
            gen_config = _to_generation_config(request)
        except ValueError as err:
            return _error_response(HTTPStatus.BAD_REQUEST, str(err), param='input')

        session = server_context.create_session(-1)
        adapter_name = None if model_name == server_context.async_engine.model_name else model_name
        result_generator = server_context.async_engine.generate(
            messages,
            session,
            gen_config=gen_config,
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
                ),
                media_type='text/event-stream',
            )

        text = ''
        final_res = None
        async for res in result_generator:
            if await raw_request.is_disconnected():
                await session.async_abort()
                return _error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected')
            final_res = res
            text += res.response or ''

        if final_res is None:
            return _error_response(HTTPStatus.INTERNAL_SERVER_ERROR, 'No generation output from engine.')

        response = _make_response(
            request=request,
            model_name=model_name,
            created_time=created_time,
            text=text,
            input_tokens=final_res.input_token_len,
            output_tokens=final_res.generate_token_len,
            finish_reason=final_res.finish_reason,
        )
        return response.model_dump(exclude_none=True)

    return router
