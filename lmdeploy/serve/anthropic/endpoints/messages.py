# Copyright (c) OpenMMLab. All rights reserved.
"""Endpoint for ``POST /v1/messages``."""

from __future__ import annotations

from http import HTTPStatus

import shortuuid
from fastapi import APIRouter, Depends, Request

from lmdeploy.serve.core.chat_runner import ChatRunner, ChatRunnerOptions
from lmdeploy.serve.core.exceptions import RequestError
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.utils.server_utils import validate_json_request
from lmdeploy.serve.utils.streaming_response import ManagedStreamingResponse

from ..adapter import (
    build_message_content_blocks,
    map_finish_reason,
    normalize_tool_choice,
    to_openai_messages,
    to_openai_tools,
)
from ..errors import create_error_response, create_request_error_response
from ..protocol import MessagesRequest, MessagesResponse, MessageTextBlock, MessageUsage
from ..streaming import stream_messages_response
from .validation import check_request, messages_empty


def register(router: APIRouter, server_context, *, merge_inline_system: bool = False) -> None:
    """Register endpoint onto router."""

    @router.post('/v1/messages', dependencies=[Depends(validate_json_request)])
    async def create_message(request: MessagesRequest, raw_request: Request):
        validation_error = check_request(request, raw_request, server_context)
        if validation_error is not None:
            return validation_error

        # Resolve fallback input when messages is empty.
        parser_messages = None
        resolved_input_ids = None
        if messages_empty(request) and request.input_ids is not None:
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
                parser_messages = to_openai_messages(request, merge_inline_system=merge_inline_system)
            except ValueError as err:
                return create_error_response(HTTPStatus.BAD_REQUEST, str(err))

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
            return_logprob=bool(request.return_logprob),
            return_token_ids=bool(request.return_token_ids),
            return_routed_experts=bool(request.return_routed_experts),
            include_stop_str_in_output=bool(request.include_stop_str_in_output),
        )
        try:
            chat_runner = await ChatRunner.prepare(
                server_context,
                openai_request,
                ChatRunnerOptions(
                    input_ids=resolved_input_ids,
                    do_preprocess=resolved_input_ids is None,
                ),
            )
        except RequestError as error:
            return create_request_error_response(error)

        request_id = f'msg_{shortuuid.random()}'
        chat_request = chat_runner.request

        if request.stream:
            async def stream_generator():
                try:
                    async for event in stream_messages_response(
                            chat_runner.stream(),
                            request_id=request_id,
                            request=chat_request,
                    ):
                        yield event
                finally:
                    await chat_runner.close()

            return ManagedStreamingResponse(
                stream_generator(),
                cleanup_callbacks=[chat_runner.close],
                media_type='text/event-stream',
            )

        try:
            res = await chat_runner.collect(raw_request)
        except RequestError as error:
            return create_request_error_response(error)

        content_blocks = build_message_content_blocks(
            res.text,
            res.tool_calls,
            res.reasoning_content,
        )
        if not content_blocks:
            content_blocks = [MessageTextBlock(text='')]

        output_token_logprobs = None
        if chat_request.return_logprob and res.logprobs and res.token_ids:
            output_token_logprobs = [
                (tok_logprobs[tok], tok)
                for tok, tok_logprobs in zip(res.token_ids, res.logprobs)
            ]

        response = MessagesResponse(
            id=request_id,
            model=request.model,
            content=content_blocks,
            stop_reason=map_finish_reason(res.finish_reason),
            stop_sequence=None,
            usage=MessageUsage(
                input_tokens=res.input_token_len,
                output_tokens=res.generate_token_len,
            ),
            output_ids=res.token_ids if chat_request.return_token_ids else None,
            output_token_logprobs=output_token_logprobs,
            routed_experts=res.routed_experts if chat_request.return_routed_experts else None,
        )
        return response.model_dump()
