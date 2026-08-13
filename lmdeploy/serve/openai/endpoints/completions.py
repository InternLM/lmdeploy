# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest
from lmdeploy.serve.openai.endpoints.common import build_serving_generation_config, validate_request
from lmdeploy.serve.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    LogProbs,
    UsageInfo,
)
from lmdeploy.serve.openai.utils import create_error_response
from lmdeploy.serve.utils.request_cleanup import with_request_cleanup
from lmdeploy.serve.utils.server_utils import validate_json_request


def check_request(request: CompletionRequest, server_context) -> str:
    engine_config = server_context.engine_config
    session_manager = server_context.session_manager
    try:
        # Check logprobs settings
        logprobs_mode = engine_config.logprobs_mode
        logprobs = request.logprobs or 0
        if logprobs > 0 and logprobs_mode is None:
            return f'logprobs({logprobs}) requested but not enabled logprobs_mode in engine configuration.'
        if logprobs_mode is not None and logprobs < 0:
            return 'logprobs must be non-negative when logprobs_mode is enabled in engine configuration.'
    except AttributeError:
        pass

    if session_manager.has(request.session_id):
        return f'The session_id {request.session_id!r} is occupied.'

    # check sampling settings
    if request.n <= 0:
        return f'The n {request.n!r} must be a positive int.'
    if request.top_p is not None and not (0 < request.top_p <= 1):
        return f'The top_p {request.top_p!r} must be in (0, 1].'
    if request.top_k is not None and request.top_k < 0:
        return f'The top_k {request.top_k!r} cannot be a negative integer.'
    if request.temperature is not None and not (0 <= request.temperature <= 2):
        return f'The temperature {request.temperature!r} must be in [0, 2]'

    return ''


def register(router: APIRouter, server_context) -> None:

    @router.post('/v1/completions',
                 dependencies=[Depends(validate_json_request)])
    async def completions_v1(request: CompletionRequest,
                             raw_request: Request = None):
        """Completion API similar to OpenAI's API.

        Go to https://platform.openai.com/docs/api-reference/completions/create
        for the API specification.

        The request should be a JSON object with the following fields:

        - **model** (str): model name. Available from /v1/models.
        - **prompt** (str): the input prompt.
        - **suffix** (str): The suffix that comes after a completion of inserted text.
        - **max_completion_tokens** (int | None): output token nums. Default to None.
        - **max_tokens** (int | None): output token nums. Default to 16.
          Deprecated: Use max_completion_tokens instead.
        - **temperature** (float): to modulate the next token probability
        - **top_p** (float): If set to float < 1, only the smallest set of most
          probable tokens with probabilities that add up to top_p or higher
          are kept for generation.
        - **n** (int): How many chat completion choices to generate for each input
          message. **Only support one here**.
        - **stream**: whether to stream the results or not. Default to false.
        - **stream_options**: Options for streaming response. Only set this when you
          set stream: true.
        - **repetition_penalty** (float): The parameter for repetition penalty.
          1.0 means no penalty
        - **frequency_penalty** (float): Additive penalty in ``[-2, 2]`` based
          on a token's frequency in the generated text. ``0`` means no penalty.
        - **user** (str): A unique identifier representing your end-user.
        - **stop** (str | list[str] | None): To stop generating further
          tokens. Only accept stop words that's encoded to one token idex.

        Additional arguments supported by LMDeploy:

        - **ignore_eos** (bool): indicator for ignoring eos
        - **skip_special_tokens** (bool): Whether or not to remove special tokens
          in the decoding. Default to be True.
        - **spaces_between_special_tokens** (bool): Whether or not to add spaces
          around special tokens. The behavior of Fast tokenizers is to have
          this to False. This is setup to True in slow tokenizers.
        - **top_k** (int): The number of the highest probability vocabulary
          tokens to keep for top-k-filtering
        - **min_p** (float): Minimum token probability, which will be scaled by the
          probability of the most likely token. It must be a value between
          0 and 1. Typical values are in the 0.01-0.2 range, comparably
          selective as setting `top_p` in the 0.99-0.8 range (use the
          opposite of normal `top_p` values)
        - **repetition_ngram_size** (int): N-gram length for repetition early stop
          (PyTorch engine). ``0`` disables.
        - **repetition_ngram_threshold** (int): How many times that n-gram must
          repeat to trigger early stop. ``0`` disables.

        Currently we do not support the following features:

        - **logprobs** (not supported yet)
        - **presence_penalty** (replaced with repetition_penalty)
        """
        error_check_ret = validate_request(request, server_context,
                                           check_request)
        if error_check_ret is not None:
            return error_check_ret
        json_request = await raw_request.json()
        migration_request = json_request.pop('migration_request', None)
        with_cache = json_request.pop('with_cache', False)
        preserve_cache = json_request.pop('preserve_cache', False)
        if migration_request:
            migration_request = MigrationRequest.model_validate(
                migration_request)

        model_name = request.model
        adapter_name = None
        if model_name != server_context.async_engine.model_name:
            adapter_name = model_name  # got a adapter name
        request_id = str(request.session_id)
        created_time = int(time.time())
        sessions = []
        if isinstance(request.prompt, str):
            request.prompt = [request.prompt]
            sessions.append(server_context.create_session(request.session_id))
        elif isinstance(request.prompt, list):
            for _ in request.prompt:
                sessions.append(server_context.create_session())
        if isinstance(request.stop, str):
            request.stop = [request.stop]

        gen_config = build_serving_generation_config(
            request,
            server_context,
            stop_words=request.stop,
            random_seed=request.seed,
            migration_request=migration_request,
            with_cache=with_cache,
            preserve_cache=preserve_cache,
        )
        generators = []
        for prompt, session in zip(request.prompt, sessions):
            result_generator = server_context.async_engine.generate(
                prompt,
                session,
                gen_config=gen_config,
                stream_response=True,  # always use stream to enable batching
                do_preprocess=False,
                adapter_name=adapter_name)
            generators.append(result_generator)

        include_usage = bool(request.stream_options
                             and request.stream_options.include_usage)

        def create_stream_response_json(
                index: int,
                text: str,
                finish_reason: str | None = None,
                logprobs: LogProbs | None = None) -> dict:
            choice_data = CompletionResponseStreamChoice(
                index=index,
                text=text,
                finish_reason=finish_reason,
                logprobs=logprobs)
            response = CompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=[choice_data],
                usage=None,
            )
            response_json = response.model_dump(mode='json', exclude_none=True)
            if include_usage:
                response_json['usage'] = None
            return response_json

        def create_stream_usage_response_json(usage: UsageInfo) -> dict:
            response = CompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=[],
                usage=usage,
            )
            return response.model_dump(mode='json', exclude_none=True)

        async def completion_stream_generator() -> AsyncGenerator[str, None]:
            # First chunk with role
            final_usage = UsageInfo() if include_usage else None
            for generator in generators:
                async for res in generator:
                    logprobs = None
                    if request.logprobs and res.logprobs:
                        raise ValueError('logprobs is removed')
                    if res.finish_reason and final_usage is not None:
                        final_res = res
                        total_tokens = sum([
                            final_res.input_token_len,
                            final_res.generate_token_len
                        ])
                        final_usage.prompt_tokens += final_res.input_token_len
                        final_usage.completion_tokens += final_res.generate_token_len
                        final_usage.total_tokens += total_tokens
                    response_json = create_stream_response_json(
                        index=0,
                        text=res.response,
                        finish_reason=res.finish_reason,
                        logprobs=logprobs)
                    if res.cache_block_ids is not None:
                        response_json['cache_block_ids'] = res.cache_block_ids
                        response_json['remote_token_ids'] = res.token_ids
                    yield f'data: {json.dumps(response_json)}\n\n'
            if final_usage is not None:
                yield f'data: {json.dumps(create_stream_usage_response_json(final_usage))}\n\n'
            yield 'data: [DONE]\n\n'

        # Streaming response
        if request.stream:
            stream_generator = with_request_cleanup(
                completion_stream_generator(), generators, sessions,
                server_context.session_manager)
            return StreamingResponse(stream_generator,
                                     media_type='text/event-stream')

        # Non-streaming response
        usage = UsageInfo()
        choices = [None] * len(generators)
        cache_block_ids = []
        remote_token_ids = []

        async def _inner_call(i, generator, session):
            nonlocal cache_block_ids, remote_token_ids
            final_logprobs = []
            final_token_ids = []
            final_res = None
            text = ''
            async with aclosing(
                    with_request_cleanup(generator, [generator], [session],
                                         server_context.session_manager)
            ) as cleanup_generator:
                async for res in cleanup_generator:
                    if await raw_request.is_disconnected():
                        # Abort the request if the client disconnects.
                        await session.async_abort()
                        return create_error_response(HTTPStatus.BAD_REQUEST,
                                                     'Client disconnected')
                    final_res = res
                    text += res.response
                    cache_block_ids.append(res.cache_block_ids)
                    remote_token_ids.append(res.token_ids)
                    if res.token_ids:
                        final_token_ids.extend(res.token_ids)
                    if res.logprobs:
                        final_logprobs.extend(res.logprobs)

            logprobs = None
            assert final_res is not None
            choice_data = CompletionResponseChoice(
                index=i,
                text=text,
                finish_reason=final_res.finish_reason,
                logprobs=logprobs)
            choices[i] = choice_data

            if with_cache:
                cache_block_ids = cache_block_ids[0]
                remote_token_ids = [remote_token_ids[0][-1]]

            total_tokens = sum(
                [final_res.input_token_len, final_res.generate_token_len])
            usage.prompt_tokens += final_res.input_token_len
            usage.completion_tokens += final_res.generate_token_len
            usage.total_tokens += total_tokens

        call_results = await asyncio.gather(*[
            _inner_call(i, generators[i], sessions[i])
            for i in range(len(generators))
        ])
        error_response = next(
            (result for result in call_results if result is not None), None)
        if error_response is not None:
            return error_response

        response = CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        ).model_dump()

        if with_cache:
            response['cache_block_ids'] = cache_block_ids
            response['remote_token_ids'] = remote_token_ids

        return response
