# Copyright (c) OpenMMLab. All rights reserved.
"""Responses request validation and conversion helpers."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any, Literal

from fastapi.responses import JSONResponse

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.openai.protocol import Tool, ToolChoice, ToolChoiceFuncName
from lmdeploy.serve.openai.responses.protocol import ResponsesRequest

logger = logging.getLogger('lmdeploy.serve.openai.responses.serving')


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
            if not isinstance(text, str):
                raise ValueError(f'Unsupported `text` in `{field_name}` content part at index {idx}. Expected string.')
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
            arguments = item.get('arguments') or ''
            if not isinstance(arguments, str):
                raise ValueError(f'Unsupported `arguments` in function_call item at index {idx}. Expected string.')
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
                                arguments=arguments,
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
                    content=_text_from_content(item.get('output', ''), f'input[{idx}].output'),
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
