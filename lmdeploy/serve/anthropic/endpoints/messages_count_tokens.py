# Copyright (c) OpenMMLab. All rights reserved.
"""Endpoint for ``POST /v1/messages/count_tokens``."""

from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from lmdeploy.serve.utils.server_utils import validate_json_request

from ..adapter import count_input_tokens, ensure_tools_not_requested, get_model_list, to_lmdeploy_messages
from ..errors import create_error_response
from ..protocol import CountTokensRequest, CountTokensResponse


def _validate_headers(raw_request: Request):
    anthropic_version = raw_request.headers.get('anthropic-version')
    if not anthropic_version:
        return create_error_response(HTTPStatus.BAD_REQUEST, 'Missing required header: anthropic-version')
    return None


def register(router: APIRouter, server_context) -> None:
    """Register endpoint onto router."""

    @router.post('/v1/messages/count_tokens', dependencies=[Depends(validate_json_request)])
    async def count_tokens(request: CountTokensRequest, raw_request: Request):
        header_error = _validate_headers(raw_request)
        if header_error is not None:
            return header_error

        if request.model not in get_model_list(server_context):
            return create_error_response(
                HTTPStatus.NOT_FOUND,
                f'The model {request.model!r} does not exist.',
                error_type='not_found_error',
            )

        try:
            ensure_tools_not_requested(request)
            messages = to_lmdeploy_messages(request)
            input_tokens = count_input_tokens(server_context.async_engine, messages)
        except NotImplementedError as err:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(err))
        except ValueError as err:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(err))

        return CountTokensResponse(input_tokens=input_tokens).model_dump()
