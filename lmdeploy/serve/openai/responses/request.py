# Copyright (c) OpenMMLab. All rights reserved.
"""Responses request validation and conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Literal

from fastapi.responses import JSONResponse

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.core.generation_config import build_generation_config
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, Tool, ToolChoice, ToolChoiceFuncName
from lmdeploy.serve.openai.responses.protocol import ResponsesRequest
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def error_response(status: HTTPStatus | int,
                   message: str,
                   *,
                   param: str | None = None,
                   error_code: ErrorCode | None = None) -> JSONResponse:
    status_value = status.value if isinstance(status, HTTPStatus) else status
    if error_code is None:
        if status_value == HTTPStatus.NOT_FOUND:
            error_code = ErrorCode.MODEL_NOT_FOUND
        elif status_value >= HTTPStatus.INTERNAL_SERVER_ERROR:
            error_code = ErrorCode.INTERNAL_ERROR
        else:
            error_code = ErrorCode.INVALID_REQUEST
    error_type = ('not_found_error' if error_code is ErrorCode.MODEL_NOT_FOUND
                  else 'server_error' if error_code in {
                      ErrorCode.ENGINE_UNAVAILABLE,
                      ErrorCode.INTERNAL_ERROR,
                  } else 'invalid_request_error')
    payload = {
        'error': {
            'message': message,
            'type': error_type,
            'param': param,
            'code': status_value,
        }
    }
    return JSONResponse(payload, status_code=status_value)


def request_error_response(error: RequestError, *, param: str | None = None) -> JSONResponse:
    return error_response(error.status_code,
                          error.message,
                          param=param,
                          error_code=error.code)


@dataclass
class ResponsesRequestContext:
    """Validated Responses request data needed by the serving layer."""

    model_name: str
    chat_request: ChatCompletionRequest


def check_request(
    request: ResponsesRequest,
    server_context,
) -> tuple[ResponsesRequestContext | None, JSONResponse | None]:
    """Validate and adapt a Responses request for chat execution."""

    validation_error = validate_text_v1_request(request)
    if validation_error is not None:
        return None, validation_error
    validation_error = _validate_sampling_request(request)
    if validation_error is not None:
        return None, validation_error

    model_name = request.model or server_context.async_engine.model_name
    if model_name not in _get_model_list(server_context):
        return None, error_response(HTTPStatus.NOT_FOUND, f'The model {model_name!r} does not exist.', param='model')

    try:
        messages = messages_from_input(request)
    except ValueError as err:
        return None, error_response(HTTPStatus.BAD_REQUEST, str(err), param='input')
    try:
        text_response_format = response_format_from_text(request.text)
    except ValueError as err:
        return None, error_response(HTTPStatus.BAD_REQUEST, str(err), param='text')
    try:
        tools = openai_tools_from_responses(request)
    except ValueError as err:
        return None, error_response(HTTPStatus.BAD_REQUEST, str(err), param='tools')
    tool_choice_error = _validate_tool_choice_request(
        request,
        tools,
        server_context.response_parser_cls,
    )
    if tool_choice_error is not None:
        return None, tool_choice_error
    try:
        tool_choice = tool_choice_from_responses(request.tool_choice, tools)
    except ValueError as err:
        return None, error_response(HTTPStatus.BAD_REQUEST, str(err), param='tool_choice')

    tools_enabled = bool(tools and tool_choice != 'none')
    chat_request_kwargs = dict(
        model=model_name,
        messages=messages,
        max_completion_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        tools=tools if tools_enabled else None,
        tool_choice=tool_choice,
        response_format=text_response_format,
        repetition_penalty=request.repetition_penalty,
        min_p=request.min_p,
    )
    for field_name in ('ignore_eos', 'skip_special_tokens', 'include_stop_str_in_output'):
        if field_name in request.model_fields_set:
            chat_request_kwargs[field_name] = getattr(request, field_name)
    chat_request = ChatCompletionRequest(**chat_request_kwargs)
    return ResponsesRequestContext(model_name=model_name, chat_request=chat_request), None


def _get_model_list(server_context) -> list[str]:
    model_names = [server_context.async_engine.model_name]
    cfg = server_context.engine_config
    model_names += getattr(cfg, 'adapters', None) or []
    return model_names


def validate_text_v1_request(request: ResponsesRequest) -> JSONResponse | None:
    if request.background:
        return error_response(HTTPStatus.BAD_REQUEST, 'background mode is not supported by Responses Text V1.',
                              param='background')
    if request.context_management is not None:
        return error_response(HTTPStatus.BAD_REQUEST, 'context_management is not supported by Responses Text V1.',
                              param='context_management')
    if request.conversation is not None:
        return error_response(HTTPStatus.BAD_REQUEST, 'conversation is not supported by Responses Text V1.',
                              param='conversation')
    if request.previous_response_id is not None:
        return error_response(HTTPStatus.BAD_REQUEST, 'previous_response_id is not supported by Responses Text V1.',
                              param='previous_response_id')
    if request.prompt is not None:
        return error_response(HTTPStatus.BAD_REQUEST, 'prompt is not supported by Responses Text V1.', param='prompt')
    if request.input is None:
        return error_response(HTTPStatus.BAD_REQUEST, 'input is required by Responses Text V1.', param='input')
    if isinstance(request.input, list) and len(request.input) == 0:
        return error_response(HTTPStatus.BAD_REQUEST, 'input must not be an empty list.', param='input')
    return None


def _validate_sampling_request(request: ResponsesRequest) -> JSONResponse | None:
    if request.temperature is not None and not (0 <= request.temperature <= 2):
        return error_response(HTTPStatus.BAD_REQUEST, 'temperature must be in [0, 2].', param='temperature')
    if request.top_p is not None and not (0 <= request.top_p <= 1):
        return error_response(HTTPStatus.BAD_REQUEST, 'top_p must be in [0, 1].', param='top_p')
    if request.top_k is not None and request.top_k < 0:
        return error_response(HTTPStatus.BAD_REQUEST, 'top_k cannot be a negative integer.', param='top_k')
    if request.min_p is not None and not (0 <= request.min_p <= 1):
        return error_response(HTTPStatus.BAD_REQUEST, 'min_p must be in [0, 1].', param='min_p')
    return None


def warn_ignored_request_fields(request: ResponsesRequest) -> None:
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


def messages_from_input(request: ResponsesRequest) -> list[dict[str, Any]]:
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


def openai_tools_from_responses(request: ResponsesRequest) -> list[Tool] | None:
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


def tool_choice_from_responses(tool_choice: Any,
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


def _validate_tool_choice_request(request: ResponsesRequest,
                                  tools: list[Tool] | None,
                                  parser_cls) -> JSONResponse | None:
    """Validate tool parser availability for a Responses request."""

    tool_choice = request.tool_choice
    tool_choice_dict = _as_dict(tool_choice)
    if tool_choice_dict:
        tool_choice_type = tool_choice_dict.get('type')
        if tool_choice_type != 'function':
            return error_response(
                HTTPStatus.BAD_REQUEST,
                f'Unsupported tool_choice type: {tool_choice_type!r}.',
                param='tool_choice',
            )
        name = tool_choice_dict.get('name')
        if not name:
            return error_response(
                HTTPStatus.BAD_REQUEST,
                'Missing `name` in function tool_choice.',
                param='tool_choice',
            )
        tool_names = {tool.function.name for tool in tools or []}
        if name not in tool_names:
            return error_response(
                HTTPStatus.BAD_REQUEST,
                f"Tool choice 'function' not found in `tools`: {name!r}.",
                param='tool_choice',
            )
    elif not _is_known_string_tool_choice(tool_choice):
        return error_response(
            HTTPStatus.BAD_REQUEST,
            f'Unsupported tool_choice: {tool_choice!r}.',
            param='tool_choice',
        )

    if tools and tool_choice != 'none':
        if parser_cls is None or parser_cls.tool_parser_cls is None:
            return error_response(
                HTTPStatus.BAD_REQUEST,
                'Please launch the api_server with --tool-call-parser if you want to use tool calling.',
                param='tools',
            )

    return None


def _is_known_string_tool_choice(tool_choice: Any) -> bool:
    return tool_choice is None or tool_choice in ('auto', 'none', 'required')


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


def response_format_from_text(text: Any) -> dict[str, Any] | None:
    """Map Responses ``text.format`` to the internal response_format shape."""
    return _response_format_from_text(text)


def to_generation_config(
    request: ResponsesRequest,
    default_gen_config: dict | None = None,
) -> GenerationConfig:
    stop_words = [request.stop] if isinstance(request.stop, str) else request.stop
    return build_generation_config(
        request,
        default_gen_config or {},
        max_new_tokens=request.max_output_tokens,
        stop_words=stop_words,
        response_format=_response_format_from_text(request.text),
        random_seed=request.seed,
    )
