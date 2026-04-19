# Copyright (c) OpenMMLab. All rights reserved.
"""Endpoint for ``POST /v1/messages``."""

from __future__ import annotations

from http import HTTPStatus

import shortuuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from lmdeploy.serve.utils.server_utils import validate_json_request

from ..adapter import (
    ensure_tools_not_requested,
    get_model_list,
    map_finish_reason,
    to_generation_config,
    to_lmdeploy_messages,
)
from ..errors import create_error_response
from ..protocol import MessagesRequest, MessagesResponse, MessageTextBlock, MessageUsage
from ..streaming import stream_messages_response


def _validate_headers(raw_request: Request):
    anthropic_version = raw_request.headers.get('anthropic-version')
    if not anthropic_version:
        return create_error_response(HTTPStatus.BAD_REQUEST, 'Missing required header: anthropic-version')
    return None


def register(router: APIRouter, server_context) -> None:
    """Register endpoint onto router."""

    @router.post('/v1/messages', dependencies=[Depends(validate_json_request)])
    async def create_message(request: MessagesRequest, raw_request: Request):
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
        except NotImplementedError as err:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(err))
        except ValueError as err:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(err))

        session = server_context.get_session(-1)
        adapter_name = None if request.model == server_context.async_engine.model_name else request.model
        result_generator = server_context.async_engine.generate(
            messages,
            session,
            gen_config=to_generation_config(request),
            stream_response=True,
            sequence_start=True,
            sequence_end=True,
            do_preprocess=True,
            adapter_name=adapter_name,
        )

        request_id = f'msg_{shortuuid.random()}'

        if request.stream:
            return StreamingResponse(
                stream_messages_response(result_generator, request_id=request_id, model=request.model),
                media_type='text/event-stream',
            )

        text = ''
        final_res = None
        async for res in result_generator:
            if await raw_request.is_disconnected():
                await session.async_abort()
                return create_error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected')
            final_res = res
            text += res.response or ''

        if final_res is None:
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, 'No generation output from engine.')

        response = MessagesResponse(
            id=request_id,
            model=request.model,
            content=[MessageTextBlock(text=text)],
            stop_reason=map_finish_reason(final_res.finish_reason),
            stop_sequence=None,
            usage=MessageUsage(
                input_tokens=final_res.input_token_len,
                output_tokens=final_res.generate_token_len,
            ),
        )
        return response.model_dump()
