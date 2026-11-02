# Copyright (c) OpenMMLab. All rights reserved.
"""Intern-Decision endpoint: one scoring pass per request."""
from __future__ import annotations

import asyncio
import json
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError
from starlette.datastructures import Headers

from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.decision_protocol import (
    OpenAIDecisionRequest,
    OpenAIDecisionResponse,
    SystemoneRequest,
    SystemoneResponse,
)
from lmdeploy.serve.openai.errors import create_error_response, create_request_error_response
from lmdeploy.serve.parsers.decision_assets import MAX_TOTAL_BYTES, AssetError, unpack_multipart
from lmdeploy.serve.parsers.decision_parser import DecisionParser
from lmdeploy.serve.utils.request_cleanup import cleanup_result_generators
from lmdeploy.serve.utils.server_utils import validate_json_request
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _validation_error_response(exc: RequestValidationError, *, systemone: bool) -> JSONResponse:
    details = []
    for error in exc.errors():
        loc = list(error['loc'])
        if loc and loc[0] == 'body':
            loc.pop(0)
        # Remove schema branches only at their known positions, never user keys.
        tags = ('choice', 'score', 'noul') if systemone else ('predicate', 'choice', 'score')
        if len(loc) >= 4 and loc[0] == 'questions' and loc[2] in tags:
            del loc[2]
        if not systemone and loc and loc[0] == 'input':
            if len(loc) > 1 and isinstance(loc[1], str) and (
                    loc[1] in ('str', 'constrained-str') or loc[1].startswith('list[')):
                del loc[1]
            if len(loc) > 3 and loc[2] == 'content' and isinstance(loc[3], str):
                if loc[3] == 'str' or loc[3].startswith('list['):
                    del loc[3]
                if len(loc) > 5 and loc[4] in ('input_text', 'input_image'):
                    del loc[4]
        if error['type'] in ('union_tag_invalid', 'union_tag_not_found'):
            loc.append('type')
        message = error['msg']
        if systemone and loc == ['images'] and error['type'] == 'extra_forbidden':
            message = ('images was removed in 0.4.0; use assets and reference images '
                       'in state, instructions or criteria.')
        if systemone and error['type'] == 'union_tag_invalid' and "'box'" in message:
            message = ('box is not supported by this checkpoint; use choice, score or noul. '
                       'No trained bounding-box head or validated grounding decoder is available.')
        details.append(dict(field='.'.join(map(str, loc)) or 'body', loc=['body', *loc],
                            code=error['type'], message=message))
    return JSONResponse(dict(
        detail=[dict(loc=d['loc'], msg=d['message'], type=d['code']) for d in details],
        error=dict(type='validation_error', message='Request body failed validation.', details=details)),
        status_code=422)


class _DecisionRoute(APIRoute):
    """Keep field-level validation responses local to the decision endpoints."""

    systemone = False

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def decision_route(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                return _validation_error_response(exc, systemone=self.systemone)

        return decision_route


class _SystemoneRoute(_DecisionRoute):
    """Normalize bounded multipart input before FastAPI's typed JSON validation.

    Body limits and error details follow intern-decision-inference a8fe089.
    The normal request model remains visible in OpenAPI.
    """

    systemone = True

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def systemone_route(request: Request):
            try:
                declared = request.headers.get('content-length')
                if declared is not None:
                    try:
                        length = int(declared)
                        if length < 0:
                            raise ValueError()
                    except ValueError:
                        raise AssetError('Invalid Content-Length.') from None
                    if length > MAX_TOTAL_BYTES:
                        raise AssetError('Request body exceeds the configured byte limit.', 413)

                async def collect():
                    body = bytearray()
                    async for chunk in request.stream():
                        if len(body) + len(chunk) > MAX_TOTAL_BYTES:
                            raise AssetError('Request body exceeds the configured byte limit.', 413)
                        body.extend(chunk)
                    return bytes(body)

                raw = await asyncio.wait_for(collect(), 30)
                content_type = request.headers.get('content-type', '')

                def normalize():
                    if content_type.lower().startswith('multipart/form-data'):
                        body = unpack_multipart(raw, content_type)
                    else:
                        try:
                            body = json.loads(raw)
                        except (ValueError, UnicodeError, RecursionError):
                            raise AssetError('Malformed JSON request body.') from None
                    return json.dumps(body).encode('utf-8')

                request._body = await asyncio.to_thread(normalize)
                request.scope['headers'] = [(key, value) for key, value in request.scope['headers']
                                            if key not in (b'content-type', b'content-length')]
                request.scope['headers'].extend([(b'content-type', b'application/json'),
                                                (b'content-length', str(len(request._body)).encode())])
                request._headers = Headers(scope=request.scope)
                return await handler(request)
            except asyncio.TimeoutError:
                return JSONResponse({'error': 'Request body read timed out.'}, status_code=408)
            except AssetError as exc:
                return JSONResponse({'error': exc.message}, status_code=exc.status)

        return systemone_route


async def _score_request(server_context, parser: DecisionParser, raw_request=None) -> dict:
    """Run one scoring pass without generating tokens.

    Owns the engine session and result generator; the parser only converts request and output data.
    """
    tokenizer = server_context.async_engine.tokenizer
    try:
        if parser.systemone or parser.request.images:
            processor = getattr(server_context.async_engine, 'prompt_processor', None)
            inputs = await asyncio.to_thread(
                parser.prepare_input, tokenizer=tokenizer,
                allowed_media_domains=getattr(processor, 'allowed_media_domains', None))
        else:
            inputs = parser.prepare_input(tokenizer=tokenizer)
    except ValidationError as exc:
        errors = [{**error, 'loc': ('body', *error['loc'])}
                  for error in exc.errors(include_input=False, include_url=False)]
        raise RequestValidationError(errors) from exc

    session = server_context.create_session()
    generator = None
    output = None
    try:
        prepared = await server_context.async_engine.preprocess(
            inputs.messages, session, gen_config=inputs.gen_config, **inputs.preprocess_kwargs)
        # Marker positions and logprob_start_len depend on the expanded multimodal input.
        parser.finalize_input(prepared)

        generator = server_context.async_engine.generate(prepared, stream_response=False)
        async for result in generator:
            if raw_request is not None and await raw_request.is_disconnected():
                await session.async_abort()
                raise RequestError(ErrorCode.INVALID_REQUEST, 'Client disconnected')
            # Scoring yields a single final GenOut carrying the complete logprob snapshot.
            output = result
    finally:
        await cleanup_result_generators(
            [generator] if generator is not None else [],
            [session],
            server_context.session_manager)

    if output is None:
        raise RequestError(ErrorCode.INTERNAL_ERROR, 'No generation output from engine.')
    return parser.parse_output(output, model_name=server_context.async_engine.model_name)


def register(router: APIRouter, server_context) -> None:

    async def handle_request(request: OpenAIDecisionRequest | SystemoneRequest, raw_request: Request):
        """Share task checks, scoring and error handling across decision protocols."""
        context = server_context.decision_context
        if context is None:
            return create_error_response(
                HTTPStatus.SERVICE_UNAVAILABLE,
                'Start the server with --task intern-decision to use decisions.')
        try:
            model_name = server_context.async_engine.model_name
            models = (model_name, f'InternLM/{model_name}', f'internlm/{model_name.lower()}')
            if isinstance(request, SystemoneRequest):
                models = ('', *models)
            if request.model is not None and request.model not in models:
                raise RequestValidationError([dict(
                    type='value_error', loc=('body', 'model'),
                    msg=f'Value error, Expected one of {models!r}.')])
            return await _score_request(server_context, DecisionParser(request, context), raw_request)
        except AssetError as e:
            return JSONResponse({'error': e.message}, status_code=e.status)
        except RequestValidationError:
            raise
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

    async def systemone(request: SystemoneRequest, raw_request: Request):
        """Score referenced assets using the shared decision engine."""
        return await handle_request(request, raw_request)

    dependencies = [Depends(validate_json_request)]
    router.add_api_route('/v1/decisions', decide, methods=['POST'], dependencies=dependencies,
                         response_model=OpenAIDecisionResponse, route_class_override=_DecisionRoute)
    router.add_api_route('/v1/systemone', systemone, methods=['POST'], response_model=SystemoneResponse,
                         route_class_override=_SystemoneRoute, openapi_extra={
                             'requestBody': {'content': {'multipart/form-data': {
                                 'schema': {'type': 'object', 'required': ['request'], 'properties': {
                                     'request': {'type': 'string', 'description': 'The JSON Systemone request.'}},
                                     'additionalProperties': {'type': 'string', 'format': 'binary'}},
                                 'encoding': {'request': {'contentType': 'application/json'}}}}}})
