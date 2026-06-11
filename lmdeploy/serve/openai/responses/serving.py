# Copyright (c) OpenMMLab. All rights reserved.
"""Text-first OpenAI Responses API endpoint."""

from __future__ import annotations

import time
from contextlib import aclosing
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.openai.responses.protocol import ResponsesRequest
from lmdeploy.serve.openai.responses.request import (
    error_response,
    messages_from_input,
    openai_tools_from_responses,
    to_generation_config,
    tool_choice_from_responses,
    validate_text_v1_request,
    warn_ignored_request_fields,
)
from lmdeploy.serve.openai.responses.response import make_response
from lmdeploy.serve.openai.responses.streaming import stream_response
from lmdeploy.serve.utils.request_cleanup import with_request_cleanup
from lmdeploy.serve.utils.server_utils import validate_json_request


class OpenAIServingResponses:
    """Service object for the Text V1 Responses endpoint."""

    def __init__(self, server_context):
        self.server_context = server_context

    def _get_model_list(self) -> list[str]:
        model_names = [self.server_context.async_engine.model_name]
        cfg = self.server_context.async_engine.backend_config
        model_names += getattr(cfg, 'adapters', None) or []
        return model_names

    def _build_parser(self, request: ResponsesRequest, model_name: str, messages: list[dict], tools, tool_choice):
        parser_cls = self.server_context.response_parser_cls
        tools_enabled = bool(tools and tool_choice != 'none')
        if tools_enabled and parser_cls.tool_parser_cls is None:
            return None, error_response(
                HTTPStatus.BAD_REQUEST,
                'Please launch the api_server with --tool-call-parser if you want to use tool calling.',
                param='tools',
            )

        openai_request = ChatCompletionRequest(
            model=model_name,
            messages=messages,
            max_completion_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            tools=tools if tools_enabled else None,
            tool_choice=tool_choice,
        )
        try:
            response_parser = parser_cls(request=openai_request)
        except ValueError as err:
            return None, error_response(HTTPStatus.BAD_REQUEST, str(err))
        return response_parser, None

    def _generate(self, model_name: str, parsed_request, gen_config):
        session = self.server_context.create_session(-1)
        adapter_name = None if model_name == self.server_context.async_engine.model_name else model_name
        result_generator = self.server_context.async_engine.generate(
            parsed_request.messages,
            session,
            gen_config=gen_config,
            tools=parsed_request.tools,
            stream_response=True,
            sequence_start=True,
            sequence_end=True,
            do_preprocess=True,
            adapter_name=adapter_name,
        )
        return session, result_generator

    async def create_response(self, request: ResponsesRequest, raw_request: Request):
        validation_error = validate_text_v1_request(request)
        if validation_error is not None:
            return validation_error
        warn_ignored_request_fields(request)

        model_name = request.model or self.server_context.async_engine.model_name
        if model_name not in self._get_model_list():
            return error_response(HTTPStatus.NOT_FOUND, f'The model {model_name!r} does not exist.', param='model')

        try:
            messages = messages_from_input(request)
        except ValueError as err:
            return error_response(HTTPStatus.BAD_REQUEST, str(err), param='input')
        try:
            gen_config = to_generation_config(request)
        except ValueError as err:
            return error_response(HTTPStatus.BAD_REQUEST, str(err), param='text')
        try:
            tools = openai_tools_from_responses(request)
        except ValueError as err:
            return error_response(HTTPStatus.BAD_REQUEST, str(err), param='tools')
        try:
            tool_choice = tool_choice_from_responses(request.tool_choice, tools)
        except ValueError as err:
            return error_response(HTTPStatus.BAD_REQUEST, str(err), param='tool_choice')

        response_parser, parser_error = self._build_parser(request, model_name, messages, tools, tool_choice)
        if parser_error is not None:
            return parser_error
        parsed_request = response_parser.request

        session, result_generator = self._generate(model_name, parsed_request, gen_config)
        created_time = int(time.time())
        session_mgr = self.server_context.async_engine.session_mgr

        if request.stream:
            stream_generator = stream_response(
                result_generator,
                request=request,
                model_name=model_name,
                created_time=created_time,
                response_parser=response_parser,
            )
            return StreamingResponse(
                with_request_cleanup(stream_generator, [result_generator], [session], session_mgr),
                media_type='text/event-stream',
            )

        text = ''
        final_token_ids: list[int] = []
        final_res = None
        cleanup_generator = with_request_cleanup(result_generator, [result_generator], [session], session_mgr)
        async with aclosing(cleanup_generator) as generator:
            async for res in generator:
                if await raw_request.is_disconnected():
                    await session.async_abort()
                    return error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected')
                final_res = res
                text += res.response or ''
                if getattr(res, 'token_ids', None):
                    final_token_ids.extend(res.token_ids)

        if final_res is None:
            return error_response(HTTPStatus.INTERNAL_SERVER_ERROR, 'No generation output from engine.')

        tool_calls = None
        try:
            text, tool_calls, _reasoning_content = response_parser.parse_complete(text, final_token_ids)
        except Exception as err:
            return error_response(HTTPStatus.BAD_REQUEST, f'Failed to parse output: {err}')
        if tool_calls and final_res.finish_reason == 'stop':
            final_res.finish_reason = 'tool_calls'

        response = make_response(
            request=request,
            model_name=model_name,
            created_time=created_time,
            text=text,
            tool_calls=tool_calls,
            input_tokens=final_res.input_token_len,
            output_tokens=final_res.generate_token_len,
            finish_reason=final_res.finish_reason,
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
