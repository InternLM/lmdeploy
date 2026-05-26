# Copyright (c) OpenMMLab. All rights reserved.
"""Error helpers for Anthropic-compatible endpoints."""

from __future__ import annotations

from http import HTTPStatus

from fastapi.responses import JSONResponse

from .protocol import AnthropicError, AnthropicErrorResponse


def create_error_response(status: HTTPStatus, message: str, error_type: str = 'invalid_request_error') -> JSONResponse:
    """Create Anthropic-style error response."""

    payload = AnthropicErrorResponse(error=AnthropicError(type=error_type, message=message)).model_dump()
    return JSONResponse(payload, status_code=status.value)
