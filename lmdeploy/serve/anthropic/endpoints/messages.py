# Copyright (c) OpenMMLab. All rights reserved.
"""Endpoint for ``POST /v1/messages``."""

from __future__ import annotations

from contextlib import aclosing
from http import HTTPStatus

import shortuuid
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.utils.request_cleanup import with_request_cleanup
from lmdeploy.serve.utils.server_utils import validate_json_request

from ..adapter import (
    build_message_content_blocks,
    get_model_list,
    map_finish_reason,
    normalize_tool_choice,
    to_generation_config,
    to_openai_messages,
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


def _is_tool_choice_auto(tool_choice):
    if tool_choice is None:
        return True
    if isinstance(tool_choice, str):
        return tool_choice == 'auto'
    return tool_choice.type == 'auto'


def _validate_extended_outputs(request: MessagesRequest, server_context):
    engine_config = server_context.get_engine_config()
    logprobs_mode = engine_config.logprobs_mode
    if request.return_logprob and logprobs_mode is None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f'return_logprob={request.return_logprob} was requested, but '
            'logprobs_mode is not enabled in the engine configuration.')

    if request.return_routed_experts and not engine_config.enable_return_routed_experts:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            ('routed experts requested but not configured in engine configuration. '
             'May start the api_server with --enable-return-routed-experts flag.'))

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

        extended_outputs_error = _validate_extended_outputs(request, server_context)
        if extended_outputs_error is not None:
            return extended_outputs_error

        # Validate input_ids and image_data constraints.
        # messages has higher priority. input_ids and image_data are only used when
        # messages is empty. image_data requires input_ids.
        messages_empty = (request.messages is None
                          or (isinstance(request.messages, list) and len(request.messages) == 0))
        if not messages_empty:
            if request.input_ids is not None:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    'input_ids cannot be used when messages is non-empty.')
            if request.image_data is not None:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    'image_data cannot be used when messages is non-empty.')
        else:
            if request.input_ids is None:
                if request.image_data is not None:
                    return create_error_response(
                        HTTPStatus.BAD_REQUEST,
                        'image_data requires input_ids to be set when messages is empty.')
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    'messages must not be empty unless input_ids is set.')
            if len(request.input_ids) == 0:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    'input_ids must not be an empty list.')
            if request.system is not None:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    'system cannot be used when input_ids is set because raw input_ids bypass message rendering.')
            if request.tools:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    'tools cannot be used when input_ids is set because raw input_ids bypass message rendering.')
            if request.tool_choice is not None and not _is_tool_choice_auto(request.tool_choice):
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    'tool_choice cannot be used when input_ids is set because raw input_ids bypass message rendering.')

        # Resolve fallback input when messages is empty.
        parser_messages = None
        resolved_input_ids = None
        if messages_empty and request.input_ids is not None:
            resolved_input_ids = request.input_ids
            if request.image_data is not None:
                image_data = request.image_data
                image_input = []
                if not isinstance(image_data, list):
                    image_data = [image_data]
                for img in image_data:
                    if isinstance(img, str):
                        image_input.append(dict(type='image_url', image_url=dict(url=img)))
                    else:
                        image_input.append(dict(type='image_url', image_url=img))
                text_input = dict(type='text', text=request.input_ids)
                parser_messages = [dict(role='user', content=[text_input] + image_input)]
                resolved_input_ids = None
        else:
            try:
                parser_messages = to_openai_messages(request)
            except ValueError as err:
                return create_error_response(HTTPStatus.BAD_REQUEST, str(err))

        parser_cls = server_context.response_parser_cls
        if request.tools and parser_cls.tool_parser_cls is None:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'Please launch the api_server with --tool-call-parser if you want to use tool calling.')

        openai_request = ChatCompletionRequest(
            model=request.model,
            messages=parser_messages or [],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop_sequences,
            tools=to_openai_tools(request.tools),
            tool_choice=normalize_tool_choice(request.tool_choice),
        )
        try:
            response_parser = parser_cls(openai_request)
        except ValueError as err:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(err))
        parsed_request = response_parser.request

        session = server_context.create_session()
        adapter_name = None if request.model == server_context.async_engine.model_name else request.model
        engine_messages = None if resolved_input_ids is not None else parsed_request.messages
        result_generator = server_context.async_engine.generate(
            engine_messages,
            session,
            gen_config=to_generation_config(request),
            tools=parsed_request.tools,
            stream_response=True,
            sequence_start=True,
            sequence_end=True,
            do_preprocess=False if resolved_input_ids is not None else True,
            adapter_name=adapter_name,
            input_ids=resolved_input_ids,
        )

        request_id = f'msg_{shortuuid.random()}'
        session_mgr = server_context.get_session_manager()

        if request.stream:
            return StreamingResponse(
                with_request_cleanup(
                    stream_messages_response(
                        result_generator,
                        request_id=request_id,
                        model=request.model,
                        response_parser=response_parser,
                        return_token_ids=request.return_token_ids or False,
                        return_routed_experts=request.return_routed_experts or False,
                        logprobs=request.return_logprob or False,
                    ),
                    [result_generator],
                    [session],
                    session_mgr,
                ),
                media_type='text/event-stream',
            )

        text = ''
        final_token_ids: list[int] = []
        final_logprobs: list[dict[int, float]] = []
        final_res = None
        async with aclosing(with_request_cleanup(result_generator, [result_generator], [session],
                                                 session_mgr)) as generator:
            async for res in generator:
                if await raw_request.is_disconnected():
                    await session.async_abort()
                    return create_error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected')
                final_res = res
                text += res.response or ''
                if res.token_ids:
                    final_token_ids.extend(res.token_ids)
                if res.logprobs:
                    final_logprobs.extend(res.logprobs)

        if final_res is None:
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, 'No generation output from engine.')

        tool_calls = None
        reasoning_content = None
        try:
            raw_text = text
            text, tool_calls, reasoning_content = response_parser.parse_complete(text, final_token_ids)
        except Exception as err:
            return create_error_response(HTTPStatus.BAD_REQUEST, f'Failed to parse output: {err}')
        should_validate_complete = (
            final_res.finish_reason in ('stop', 'length')
            and (request.return_token_ids or request.return_routed_experts)
        )
        if should_validate_complete and not response_parser.validate_complete(raw_text):
            final_res.finish_reason = 'parse_error'
        if tool_calls and final_res.finish_reason == 'stop':
            final_res.finish_reason = 'tool_calls'

        content_blocks = build_message_content_blocks(text, tool_calls, reasoning_content)
        if not content_blocks:
            content_blocks = [MessageTextBlock(text='')]

        output_token_logprobs = None
        if request.return_logprob and final_logprobs and final_token_ids:
            output_token_logprobs = [
                (tok_logprobs[tok], tok)
                for tok, tok_logprobs in zip(final_token_ids, final_logprobs)
            ]

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
            output_ids=final_token_ids if request.return_token_ids else None,
            output_token_logprobs=output_token_logprobs,
            routed_experts=final_res.routed_experts if request.return_routed_experts else None,
        )
        return response.model_dump()
