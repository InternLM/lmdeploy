# Copyright (c) OpenMMLab. All rights reserved.
"""Endpoint for ``POST /v1/messages``."""

from __future__ import annotations

from http import HTTPStatus

import shortuuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.utils.server_utils import validate_json_request

from ..adapter import (
    build_message_content_blocks,
    get_model_list,
    map_finish_reason,
    normalize_tool_choice,
    to_generation_config,
    to_lmdeploy_messages,
    to_openai_tools,
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
            messages = to_lmdeploy_messages(request)
        except ValueError as err:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(err))

        parser_cls = getattr(server_context, 'response_parser_cls', None)
        if request.tools and (parser_cls is None or parser_cls.tool_parser_cls is None):
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'Please launch the api_server with --tool-call-parser if you want to use tool.')

        response_parser = None
        parsed_request = None
        if parser_cls is not None:
            tokenizer_holder = server_context.async_engine.tokenizer
            tokenizer = getattr(getattr(tokenizer_holder, 'model', None), 'model', tokenizer_holder)
            openai_request = ChatCompletionRequest(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop=request.stop_sequences,
                tools=to_openai_tools(request.tools),
                tool_choice=normalize_tool_choice(request.tool_choice),
            )
            response_parser = parser_cls(request=openai_request, tokenizer=tokenizer)
            parsed_request = response_parser.request

        session = server_context.get_session(-1)
        adapter_name = None if request.model == server_context.async_engine.model_name else request.model
        result_generator = server_context.async_engine.generate(
            messages,
            session,
            gen_config=to_generation_config(request),
            tools=None if parsed_request is None else parsed_request.tools,
            stream_response=True,
            sequence_start=True,
            sequence_end=True,
            do_preprocess=True,
            adapter_name=adapter_name,
        )

        request_id = f'msg_{shortuuid.random()}'

        if request.stream:
            return StreamingResponse(
                stream_messages_response(
                    result_generator,
                    request_id=request_id,
                    model=request.model,
                    response_parser=response_parser,
                ),
                media_type='text/event-stream',
            )

        text = ''
        final_token_ids: list[int] = []
        final_res = None
        async for res in result_generator:
            if await raw_request.is_disconnected():
                await session.async_abort()
                return create_error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected')
            final_res = res
            text += res.response or ''
            if getattr(res, 'token_ids', None):
                final_token_ids.extend(res.token_ids)

        if final_res is None:
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, 'No generation output from engine.')

        tool_calls = None
        reasoning_content = None
        if response_parser is not None:
            try:
                text, tool_calls, reasoning_content = response_parser.parse_complete(text, final_token_ids)
            except Exception as err:
                return create_error_response(HTTPStatus.BAD_REQUEST, f'Failed to parse output: {err}')
            if tool_calls and final_res.finish_reason == 'stop':
                final_res.finish_reason = 'tool_calls'

        content_blocks = build_message_content_blocks(text, tool_calls, reasoning_content)
        if not content_blocks:
            content_blocks = [MessageTextBlock(text='')]

        response = MessagesResponse(
            id=request_id,
            model=request.model,
            content=content_blocks,
            stop_reason=map_finish_reason(final_res.finish_reason),
            stop_sequence=None,
            usage=MessageUsage(
                input_tokens=final_res.input_token_len,
                output_tokens=final_res.generate_token_len,
            ),
        )
        return response.model_dump()
