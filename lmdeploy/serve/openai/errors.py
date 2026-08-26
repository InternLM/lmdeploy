# Copyright (c) OpenMMLab. All rights reserved.
"""Error helpers for OpenAI-compatible endpoints."""

from __future__ import annotations

from http import HTTPStatus

from fastapi.responses import JSONResponse

from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.protocol import ErrorResponse

_ERROR_CODE_BY_STATUS = {
    HTTPStatus.UNAUTHORIZED.value: ErrorCode.UNAUTHORIZED,
    HTTPStatus.NOT_FOUND.value: ErrorCode.MODEL_NOT_FOUND,
    HTTPStatus.CONFLICT.value: ErrorCode.REQUEST_CONFLICT,
    HTTPStatus.SERVICE_UNAVAILABLE.value: ErrorCode.ENGINE_UNAVAILABLE,
    499: ErrorCode.REQUEST_CANCELLED,
}

_ERROR_TYPE_BY_CODE = {
    ErrorCode.UNAUTHORIZED: 'authentication_error',
    ErrorCode.MODEL_NOT_FOUND: 'not_found_error',
    ErrorCode.ENGINE_UNAVAILABLE: 'server_error',
    ErrorCode.INTERNAL_ERROR: 'server_error',
}


def create_error_response(
        status: HTTPStatus | int,
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
    error_code = _ERROR_CODE_BY_STATUS.get(status_code)
    if error_code is not None:
        return error_code
    if status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return ErrorCode.INTERNAL_ERROR
    return ErrorCode.INVALID_REQUEST


def _error_type(error_code: ErrorCode) -> str:
    return _ERROR_TYPE_BY_CODE.get(error_code, 'invalid_request_error')
