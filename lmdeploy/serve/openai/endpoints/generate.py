# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
from contextlib import aclosing
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.core.exceptions import RequestError
from lmdeploy.serve.openai.endpoints.common import build_serving_generation_config, validate_request
from lmdeploy.serve.openai.errors import (
    create_error_response,
    create_request_error_response,
    request_error_payload,
)
from lmdeploy.serve.openai.protocol import (
    GenerateReqInput,
    GenerateReqMetaOutput,
    GenerateReqOutput,
)
from lmdeploy.serve.utils.request_cleanup import with_request_cleanup
from lmdeploy.serve.utils.server_utils import validate_json_request


def _create_token_logprobs(token_ids: list[int] | None = None,
                           logprobs: list[dict[int, float]] | None = None):
    """Create raw (logprob, token_id) pairs."""
    if token_ids is None or logprobs is None:
        return None

    token_logprobs = [(tok_logprobs[tok], tok)
                      for tok, tok_logprobs in zip(token_ids, logprobs, strict=True)]
    return token_logprobs or None


def _create_top_logprobs(token_ids: list[int] | None = None,
                         logprobs: list[dict[int, float]] | None = None,
                         top_logprobs_num: int = 0):
    """Create raw top-logprob pairs."""
    if token_ids is None or logprobs is None or top_logprobs_num <= 0:
        return None

    top_logprobs = []
    for tok, tok_logprobs in zip(token_ids, logprobs, strict=True):
        items = [(prob, top_id) for top_id, prob in tok_logprobs.items()]
        if len(items) > top_logprobs_num:
            # Drop the extra selected/target row; expose model top-k only.
            items = [(prob, top_id) for prob, top_id in items if top_id != tok]
        items.sort(key=lambda x: x[0], reverse=True)
        top_logprobs.append(items[:top_logprobs_num])
    return top_logprobs or None


def check_request(request: GenerateReqInput, server_context) -> str:
    engine_config = server_context.engine_config
    session_manager = server_context.session_manager
    logprobs_mode = getattr(engine_config, 'logprobs_mode', None)
    return_logprob = request.return_logprob
    top_logprobs_num = request.top_logprobs_num or 0
    if top_logprobs_num < 0:
        return f'The top_logprobs_num {request.top_logprobs_num!r} cannot be a negative integer.'
    if top_logprobs_num > 0 and return_logprob is not True:
        return 'top_logprobs_num requires return_logprob=True.'
    if hasattr(engine_config, 'logprobs_mode') and logprobs_mode is None and return_logprob:
        return f'return_logprob({return_logprob}) requested but not enabled logprobs_mode in engine configuration.'

    if request.return_routed_experts:
        if not hasattr(engine_config, 'enable_return_routed_experts'):
            return f'return_routed_experts is not supported in {type(engine_config).__name__}.'
        if not engine_config.enable_return_routed_experts:
            return ('routed experts requested but not configured in engine configuration. '
                    'May start api_server with --enable-return-routed-experts flag.')

    if (request.prompt is not None) ^ (request.input_ids is None):
        return 'You must specify exactly one of prompt or input_ids'

    if request.prompt is not None and request.prompt == '':
        return 'The prompt must not be an empty string'

    if request.input_ids is not None and len(request.input_ids) == 0:
        return 'The input_ids must not be an empty list'

    if request.max_tokens is not None and request.max_tokens < 0:
        return f'The max_tokens {request.max_tokens!r} must be non-negative.'

    # Check for input-logprob specific requirements.
    if request.logprob_start_len >= 0:
        if not isinstance(engine_config, PytorchEngineConfig) or engine_config.role != EngineRole.Hybrid:
            return 'input logprobs are supported only by the PyTorch hybrid engine.'
        if getattr(getattr(server_context, 'async_engine', None), 'speculative_config', None) is not None:
            return 'logprob_start_len is not supported with speculative decoding.'
        if return_logprob is not True:
            return 'logprob_start_len requires return_logprob=True.'
        if logprobs_mode not in ('raw_logits', 'raw_logprobs'):
            return 'logprob_start_len requires raw_logits or raw_logprobs mode.'
        if request.max_tokens != 0:
            return 'logprob_start_len requires max_tokens=0.'
    elif request.max_tokens is not None and request.max_tokens == 0:
        return f'The max_tokens {request.max_tokens!r} must be a positive integer.'

    if session_manager.has(request.session_id):
        return f'The session_id {request.session_id!r} is occupied.'

    # check sampling settings
    if request.top_p is not None and not (0 < request.top_p <= 1):
        return f'The top_p {request.top_p!r} must be in (0, 1].'
    if request.top_k is not None and request.top_k < 0:
        return f'The top_k {request.top_k!r} cannot be a negative integer.'
    if request.temperature is not None and not (0 <= request.temperature <= 2):
        return f'The temperature {request.temperature!r} must be in [0, 2]'

    return ''


def register(router: APIRouter, server_context) -> None:

    @router.post('/generate', dependencies=[Depends(validate_json_request)])
    async def generate(request: GenerateReqInput, raw_request: Request = None):
        error_check_ret = validate_request(request, server_context,
                                           check_request)
        if error_check_ret is not None:
            return error_check_ret

        prompt = request.prompt
        input_ids = request.input_ids
        input_logprobs_requested = request.logprob_start_len >= 0
        image_data = request.image_data
        if image_data is not None:
            # convert to openai format
            image_input = []
            if not isinstance(image_data, list):
                image_data = [image_data]
            for img in image_data:
                if isinstance(img, str):
                    image_input.append(
                        dict(type='image_url', image_url=dict(url=img)))
                else:
                    image_input.append(dict(type='image_url', image_url=img))
            text_input = dict(type='text',
                              text=prompt if prompt else input_ids)
            prompt = [dict(role='user', content=[text_input] + image_input)]
            input_ids = None

        gen_logprobs = (request.top_logprobs_num or 1) if request.return_logprob else None
        gen_config = build_serving_generation_config(
            request,
            server_context,
            logprobs=gen_logprobs,
            logprob_start_len=request.logprob_start_len,
            stop_words=request.stop,
        )

        session = server_context.create_session(request.session_id)
        try:
            prepared_request = await server_context.async_engine.preprocess(
                messages=prompt,
                session_id=session,
                input_ids=input_ids,
                gen_config=gen_config,
                do_preprocess=False,
                media_io_kwargs=request.media_io_kwargs,
                mm_processor_kwargs=request.mm_processor_kwargs)
        except RequestError as e:
            return create_request_error_response(e)
        result_generator = server_context.async_engine.generate(prepared_request, stream_response=True)

        def create_generate_response_json(res,
                                          text,
                                          output_ids,
                                          logprobs,
                                          input_logprobs,
                                          top_logprobs,
                                          input_top_logprobs,
                                          finish_reason,
                                          routed_experts=None):
            # only output router experts in last chunk
            routed_experts = None if finish_reason is None else routed_experts
            meta = GenerateReqMetaOutput(
                finish_reason=dict(
                    type=finish_reason) if finish_reason else None,
                output_token_logprobs=logprobs,
                input_token_logprobs=input_logprobs,
                output_top_logprobs=top_logprobs,
                input_top_logprobs=input_top_logprobs,
                prompt_tokens=res.input_token_len,
                routed_experts=routed_experts,
                completion_tokens=res.generate_token_len)

            response = GenerateReqOutput(text=text,
                                         output_ids=output_ids,
                                         meta_info=meta)
            return response.model_dump_json()

        async def generate_stream_generator():
            try:
                async for res in result_generator:
                    text = res.response or ''
                    output_ids = res.token_ids
                    routed_experts = res.routed_experts
                    if input_logprobs_requested:
                        input_logprob_token_ids = res.logprob_token_ids
                        logprobs = None
                        input_logprobs = _create_token_logprobs(input_logprob_token_ids, res.logprobs)
                        top_logprobs = None
                        input_top_logprobs = (_create_top_logprobs(input_logprob_token_ids, res.logprobs,
                                                                   request.top_logprobs_num)
                                              if request.top_logprobs_num else None)
                    else:
                        logprobs = _create_token_logprobs(res.token_ids, res.logprobs)
                        top_logprobs = (_create_top_logprobs(res.token_ids, res.logprobs, request.top_logprobs_num)
                                        if request.top_logprobs_num else None)
                        input_logprobs = None
                        input_top_logprobs = None
                    response_json = create_generate_response_json(
                        res,
                        text,
                        output_ids,
                        logprobs,
                        input_logprobs,
                        top_logprobs,
                        input_top_logprobs,
                        res.finish_reason,
                        routed_experts=routed_experts)
                    yield f'data: {response_json}\n\n'
            except RequestError as e:
                yield f'data: {json.dumps({"error": request_error_payload(e)})}\n\n'
            yield 'data: [DONE]\n\n'

        if request.stream:
            stream_generator = with_request_cleanup(
                generate_stream_generator(), [result_generator], [session],
                server_context.session_manager)
            return StreamingResponse(stream_generator,
                                     media_type='text/event-stream')

        async def _inner_call():
            text = ''
            output_ids = []
            logprobs = []
            input_logprobs = []
            input_token_logprobs = None
            input_top_logprobs = None
            input_logprob_token_ids = None
            try:
                async with aclosing(
                        with_request_cleanup(
                            result_generator, [result_generator], [session],
                            server_context.session_manager)) as generator:
                    async for res in generator:
                        if await raw_request.is_disconnected():
                            # Abort the request if the client disconnects.
                            await session.async_abort()
                            return create_error_response(HTTPStatus.BAD_REQUEST,
                                                         'Client disconnected')
                        text += res.response or ''
                        output_ids.extend(res.token_ids or [])
                        if input_logprobs_requested:
                            input_logprob_token_ids = res.logprob_token_ids
                            if res.logprobs:
                                input_logprobs = res.logprobs
                        elif res.logprobs:
                            logprobs.extend(res.logprobs)
            except RequestError as e:
                return create_request_error_response(e)

            output_token_logprobs = _create_token_logprobs(output_ids, logprobs) if logprobs else None
            output_top_logprobs = (_create_top_logprobs(output_ids, logprobs, request.top_logprobs_num)
                                   if request.top_logprobs_num and logprobs else None)
            if input_logprobs_requested and input_logprobs:
                input_token_logprobs = _create_token_logprobs(input_logprob_token_ids, input_logprobs)
                input_top_logprobs = (_create_top_logprobs(input_logprob_token_ids, input_logprobs,
                                                           request.top_logprobs_num)
                                      if request.top_logprobs_num else None)

            meta = GenerateReqMetaOutput(
                finish_reason=dict(
                    type=res.finish_reason) if res.finish_reason else None,
                output_token_logprobs=output_token_logprobs or None,
                input_token_logprobs=input_token_logprobs,
                output_top_logprobs=output_top_logprobs,
                input_top_logprobs=input_top_logprobs,
                prompt_tokens=res.input_token_len,
                routed_experts=res.routed_experts,
                completion_tokens=res.generate_token_len)
            return GenerateReqOutput(text=text,
                                     output_ids=output_ids,
                                     meta_info=meta)

        return await _inner_call()
