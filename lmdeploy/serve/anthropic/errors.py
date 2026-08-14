# Copyright (c) OpenMMLab. All rights reserved.
"""Error helpers for Anthropic-compatible endpoints."""

from __future__ import annotations

from http import HTTPStatus

from fastapi.responses import JSONResponse

from lmdeploy.serve.core.exceptions import ErrorCode, RequestError

from .protocol import AnthropicError, AnthropicErrorResponse

_ERROR_TYPE_BY_CODE = {
    ErrorCode.UNAUTHORIZED: 'authentication_error',
    ErrorCode.MODEL_NOT_FOUND: 'not_found_error',
    ErrorCode.ENGINE_UNAVAILABLE: 'overloaded_error',
    ErrorCode.INTERNAL_ERROR: 'api_error',
}


def create_error_response(status: HTTPStatus, message: str, error_type: str = 'invalid_request_error') -> JSONResponse:
    """Create Anthropic-style error response."""

    payload = AnthropicErrorResponse(error=AnthropicError(type=error_type, message=message)).model_dump()
    return JSONResponse(payload, status_code=status.value)


def anthropic_error_from_request(error: RequestError) -> AnthropicError:
    """Map a shared request error to Anthropic's error taxonomy."""
    error_type = _ERROR_TYPE_BY_CODE.get(error.code, 'invalid_request_error')
    return AnthropicError(type=error_type, message=error.message)


def create_request_error_response(error: RequestError) -> JSONResponse:
    payload = AnthropicErrorResponse(error=anthropic_error_from_request(error)).model_dump()
    return JSONResponse(payload, status_code=error.status_code)
