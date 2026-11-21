# Copyright (c) OpenMMLab. All rights reserved.
"""Endpoint for ``POST /v1/messages/count_tokens``."""

from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from lmdeploy.serve.utils.server_utils import validate_json_request

from ..adapter import (
    chat_template_kwargs_from_thinking,
    count_input_tokens,
    get_model_list,
    render_count_tools,
    to_lmdeploy_messages,
    to_openai_tools,
)
from ..errors import create_error_response
from ..headers import validate_anthropic_headers
from ..protocol import CountTokensRequest, CountTokensResponse


def register(router: APIRouter, server_context, *, merge_inline_system: bool = False) -> None:
    """Register endpoint onto router."""

    @router.post('/v1/messages/count_tokens', dependencies=[Depends(validate_json_request)])
    async def count_tokens(request: CountTokensRequest, raw_request: Request):
        header_error = validate_anthropic_headers(raw_request)
        if header_error is not None:
            return header_error

        if request.model not in get_model_list(server_context):
            return create_error_response(
                HTTPStatus.NOT_FOUND,
                f'The model {request.model!r} does not exist.',
                error_type='not_found_error',
            )

        try:
            messages = to_lmdeploy_messages(request, merge_inline_system=merge_inline_system)
            tools = render_count_tools(getattr(server_context, 'response_parser_cls', None),
                                       to_openai_tools(request.tools))
            input_tokens = count_input_tokens(
                server_context.async_engine,
                messages,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs_from_thinking(request.thinking))
        except (ValueError, AssertionError) as err:
            # chat templates validate message shapes with assert; e.g. deepseek
            # rejects a thinking-mode assistant turn without reasoning_content
            return create_error_response(HTTPStatus.BAD_REQUEST, str(err))

        return CountTokensResponse(input_tokens=input_tokens).model_dump()
