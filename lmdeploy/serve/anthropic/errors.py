# Copyright (c) OpenMMLab. All rights reserved.
"""Error helpers for Anthropic-compatible endpoints."""

from __future__ import annotations

from http import HTTPStatus

from fastapi.responses import JSONResponse

from lmdeploy.serve.core.exceptions import ErrorCode, RequestError

from .protocol import AnthropicError, AnthropicErrorResponse


def create_error_response(status: HTTPStatus, message: str, error_type: str = 'invalid_request_error') -> JSONResponse:
    """Create Anthropic-style error response."""

    payload = AnthropicErrorResponse(error=AnthropicError(type=error_type, message=message)).model_dump()
    return JSONResponse(payload, status_code=status.value)


def anthropic_error_from_request(error: RequestError) -> AnthropicError:
    """Map a shared request error to Anthropic's error taxonomy."""
    if error.code is ErrorCode.UNAUTHORIZED:
        error_type = 'authentication_error'
    elif error.code is ErrorCode.MODEL_NOT_FOUND:
        error_type = 'not_found_error'
    elif error.code is ErrorCode.ENGINE_UNAVAILABLE:
        error_type = 'overloaded_error'
    elif error.code is ErrorCode.INTERNAL_ERROR:
        error_type = 'api_error'
    else:
        error_type = 'invalid_request_error'
    return AnthropicError(type=error_type, message=error.message)


def create_request_error_response(error: RequestError) -> JSONResponse:
    payload = AnthropicErrorResponse(error=anthropic_error_from_request(error)).model_dump()
    return JSONResponse(payload, status_code=error.status_code)
