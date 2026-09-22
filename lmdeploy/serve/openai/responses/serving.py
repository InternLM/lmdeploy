# Copyright (c) OpenMMLab. All rights reserved.
"""Text-first OpenAI Responses API endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from lmdeploy.serve.core.chat_runner import ChatRunner, ChatRunnerOptions
from lmdeploy.serve.core.exceptions import RequestError
from lmdeploy.serve.openai.responses.protocol import ResponsesRequest
from lmdeploy.serve.openai.responses.request import (
    check_request,
    request_error_response,
    warn_ignored_request_fields,
)
from lmdeploy.serve.openai.responses.response import make_response
from lmdeploy.serve.openai.responses.streaming import stream_response
from lmdeploy.serve.utils.server_utils import validate_json_request
from lmdeploy.serve.utils.streaming_response import ManagedStreamingResponse


class OpenAIServingResponses:
    """Service object for the Text V1 Responses endpoint."""

    def __init__(self, server_context):
        self.server_context = server_context

    async def create_response(self, request: ResponsesRequest, raw_request: Request):
        request_context, validation_error = check_request(request, self.server_context)
        if validation_error is not None:
            return validation_error
        assert request_context is not None
        warn_ignored_request_fields(request)

        try:
            chat_runner = await ChatRunner.prepare(
                self.server_context,
                request_context.chat_request,
                ChatRunnerOptions(
                    do_preprocess=True,
                    gen_config_kwargs=dict(random_seed=request.seed),
                ),
            )
        except RequestError as error:
            return request_error_response(error)
        created_time = int(time.time())
        model_name = request_context.model_name

        if request.stream:
            async def stream_generator():
                try:
                    async for event in stream_response(
                            chat_runner.stream(),
                            request=request,
                            model_name=model_name,
                            created_time=created_time,
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
            return request_error_response(error)

        response = make_response(
            request=request,
            model_name=model_name,
            created_time=created_time,
            text=res.text,
            tool_calls=res.tool_calls,
            input_tokens=res.input_token_len,
            output_tokens=res.generate_token_len,
            reasoning_tokens=res.reasoning_tokens or 0,
            finish_reason=res.finish_reason,
        )
        return response.model_dump(exclude_none=True)


def create_responses_router(server_context) -> APIRouter:
    """Create router for the Text V1 Responses endpoint."""

    router = APIRouter(tags=['openai'])
    serving = OpenAIServingResponses(server_context)

    @router.post('/v1/responses', dependencies=[Depends(validate_json_request)])
    async def create_response(request: ResponsesRequest, raw_request: Request):
        return await serving.create_response(request, raw_request)

    return router
