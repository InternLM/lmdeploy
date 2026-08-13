# Copyright (c) OpenMMLab. All rights reserved.

from http import HTTPStatus
from typing import Any, TypeVar

from fastapi.responses import JSONResponse

from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ErrorResponse,
)

_ToolCallT = TypeVar('_ToolCallT')
_ChatCompletionResponseChoiceT = TypeVar('_ChatCompletionResponseChoiceT', ChatCompletionResponseChoice,
                                         ChatCompletionResponseStreamChoice)


def get_model_list(server_context) -> list[str]:
    """Return the model and adapter names exposed by a server."""
    model_names = [server_context.async_engine.model_name]
    cfg = server_context.engine_config
    model_names += getattr(cfg, 'adapters', None) or []
    return model_names


def create_error_response(
        status: HTTPStatus,
        message: str,
        error_type: str | None = None,
        error_code: ErrorCode | None = None,
        param: str | None = None) -> JSONResponse:
    """Create an OpenAI-compatible error response."""
    status_value = status.value if isinstance(status, HTTPStatus) else status
    error_code = error_code or _error_code_from_status(status_value)
    error_type = error_type or _error_type(error_code)
    payload = ErrorResponse(message=message,
                            type=error_type,
                            code=status_value,
                            param=param)
    return JSONResponse(payload.model_dump(), status_code=status_value)


def create_request_error_response(error: RequestError, *, param: str | None = None) -> JSONResponse:
    """Render a core request error in the OpenAI-compatible envelope."""
    return create_error_response(error.status_code,
                                 error.message,
                                 error_code=error.code,
                                 param=param)


def request_error_payload(error: RequestError, *, param: str | None = None) -> dict:
    """Render an error payload for an already-started OpenAI stream."""
    return ErrorResponse(message=error.message,
                         type=_error_type(error.code),
                         code=error.status_code,
                         param=param).model_dump()


def _error_code_from_status(status_code: int) -> ErrorCode:
    if status_code == HTTPStatus.UNAUTHORIZED:
        return ErrorCode.UNAUTHORIZED
    if status_code == HTTPStatus.NOT_FOUND:
        return ErrorCode.MODEL_NOT_FOUND
    if status_code == HTTPStatus.CONFLICT:
        return ErrorCode.REQUEST_CONFLICT
    if status_code == HTTPStatus.SERVICE_UNAVAILABLE:
        return ErrorCode.ENGINE_UNAVAILABLE
    if status_code == 499:
        return ErrorCode.REQUEST_CANCELLED
    if status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return ErrorCode.INTERNAL_ERROR
    return ErrorCode.INVALID_REQUEST


def _error_type(error_code: ErrorCode) -> str:
    if error_code == ErrorCode.UNAUTHORIZED:
        return 'authentication_error'
    if error_code == ErrorCode.MODEL_NOT_FOUND:
        return 'not_found_error'
    if error_code in (ErrorCode.ENGINE_UNAVAILABLE, ErrorCode.INTERNAL_ERROR):
        return 'server_error'
    return 'invalid_request_error'


def filter_parallel_tool_calls(tool_calls: list[_ToolCallT] | None,
                               parallel_tool_calls: bool | None) -> list[_ToolCallT] | None:
    """Filter to the first tool call only when parallel_tool_calls is false."""

    if parallel_tool_calls is not False or not tool_calls:
        return tool_calls
    return tool_calls[:1]


def filter_parallel_tool_call_deltas(tool_calls: list[Any] | None,
                                     parallel_tool_calls: bool | None) -> list[Any] | None:
    """Filter to index zero tool deltas only when parallel_tool_calls is
    false."""

    if parallel_tool_calls is not False or not tool_calls:
        return tool_calls
    return [tool_call for tool_call in tool_calls if tool_call.index == 0]


def maybe_filter_parallel_tool_calls(
    choice: _ChatCompletionResponseChoiceT,
    request: ChatCompletionRequest,
) -> _ChatCompletionResponseChoiceT:
    """Filter to the first tool call only when parallel_tool_calls is false."""

    if request.parallel_tool_calls is not False:
        return choice

    if isinstance(choice, ChatCompletionResponseChoice) and choice.message.tool_calls:
        choice.message.tool_calls = filter_parallel_tool_calls(
            choice.message.tool_calls, request.parallel_tool_calls)
    elif isinstance(choice, ChatCompletionResponseStreamChoice) and choice.delta.tool_calls:
        choice.delta.tool_calls = filter_parallel_tool_call_deltas(
            choice.delta.tool_calls, request.parallel_tool_calls)

    return choice
