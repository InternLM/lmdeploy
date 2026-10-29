# Copyright (c) OpenMMLab. All rights reserved.
"""Intern-Decision endpoint: one scoring pass per request."""
from __future__ import annotations

import time
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request

from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.decision_protocol import (
    DecisionRequest,
    OpenAIDecisionRequest,
    OpenAIDecisionResponse,
)
from lmdeploy.serve.openai.errors import create_error_response, create_request_error_response
from lmdeploy.serve.parsers.decision_parser import DecisionParser
from lmdeploy.serve.utils.request_cleanup import cleanup_result_generators
from lmdeploy.serve.utils.server_utils import validate_json_request
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


async def _score_request(server_context, parser: DecisionParser, raw_request=None) -> dict:
    """Run one scoring pass without generating tokens.

    Owns the engine session and result generator; the parser only converts request and output data.
    """
    started_at = time.perf_counter()
    inputs = parser.prepare_input()

    session = server_context.create_session()
    generator = None
    output = None
    preprocess_ms = 0.
    inference_ms = 0.
    try:
        started = time.perf_counter()
        prepared = await server_context.async_engine.preprocess(
            inputs.messages, session, gen_config=inputs.gen_config, **inputs.preprocess_kwargs)
        # Marker positions and logprob_start_len depend on the expanded multimodal input.
        parser.finalize_input(prepared)
        preprocess_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        generator = server_context.async_engine.generate(prepared, stream_response=False)
        async for result in generator:
            if raw_request is not None and await raw_request.is_disconnected():
                await session.async_abort()
                raise RequestError(ErrorCode.INVALID_REQUEST, 'Client disconnected')
            # Scoring yields a single final GenOut carrying the complete logprob snapshot.
            output = result
        inference_ms = (time.perf_counter() - started) * 1000
    finally:
        await cleanup_result_generators(
            [generator] if generator is not None else [],
            [session],
            server_context.session_manager)

    if output is None:
        raise RequestError(ErrorCode.INTERNAL_ERROR, 'No generation output from engine.')
    return parser.parse_output(output,
                               model_name=server_context.async_engine.model_name,
                               timing=dict(preprocess_ms=preprocess_ms,
                                           inference_ms=inference_ms,
                                           server_ms=(time.perf_counter() - started_at) * 1000))


def register(router: APIRouter, server_context) -> None:

    async def handle_request(request: DecisionRequest | OpenAIDecisionRequest, raw_request: Request):
        """Share task checks, scoring and error handling across both protocols."""
        context = server_context.decision_context
        if context is None:
            return create_error_response(
                HTTPStatus.SERVICE_UNAVAILABLE,
                'Start the server with --task intern-decision to use decisions.')
        try:
            if request.model is not None and request.model != server_context.async_engine.model_name:
                raise ValueError(f'Unknown model: {request.model}')
            return await _score_request(server_context, DecisionParser(request, context), raw_request)
        except RequestError as e:
            return create_request_error_response(e)
        except ValueError as e:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
        except Exception:
            logger.exception('[decisions] decision request failed')
            return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR, 'Decision scoring failed.')

    async def decide(request: OpenAIDecisionRequest, raw_request: Request):
        """Score OpenAI Decisions text/image input and return OpenAI Decisions output."""
        return await handle_request(request, raw_request)

    async def jev(request: DecisionRequest, raw_request: Request = None):
        """Preserve the native state/questions/images protocol."""
        return await handle_request(request, raw_request)

    dependencies = [Depends(validate_json_request)]
    router.add_api_route('/v1/decisions', decide, methods=['POST'], dependencies=dependencies,
                         response_model=OpenAIDecisionResponse)
    # Native compatibility endpoint for clients of the original Intern-Decision service.
    router.add_api_route('/v1/jev', jev, methods=['POST'], dependencies=dependencies)
