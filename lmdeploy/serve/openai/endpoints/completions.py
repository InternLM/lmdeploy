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
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.endpoints.common import build_serving_generation_config, validate_request
from lmdeploy.serve.openai.errors import (
    create_error_response,
    create_request_error_response,
    request_error_payload,
)
from lmdeploy.serve.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    CompletionTokensDetails,
    LogProbs,
    UsageInfo,
)
from lmdeploy.serve.utils.request_cleanup import with_request_cleanup
from lmdeploy.serve.utils.server_utils import validate_json_request


class _ReasoningTokenCounter:
    """Count generated tokens inside the model reasoning tags.

    Mirrors the token-id accounting used by the chat response parser
    (``ResponseParser._update_reasoning_tokens``) so ``/v1/completions``
    reports the same ``reasoning_tokens`` semantics: the number of tokens
    between the configured reasoning parser open/close tag tokens.
    Counting state is seeded from the prompt token ids: a prompt that
    already ends inside a reasoning block starts in counting mode (the
    prompt tokens themselves are never counted); otherwise counting
    starts at an explicit open tag, so a raw completion that never emits
    reasoning tags reports zero reasoning tokens.
    """

    def __init__(self, open_tag_id: int, close_tag_id: int,
                 prompt_token_ids: list[int] | None = None):
        self._open_tag_id = open_tag_id
        self._close_tag_id = close_tag_id
        self._counting = _ends_inside_reasoning(prompt_token_ids, open_tag_id, close_tag_id)
        self.reasoning_tokens = 0

    def update(self, token_ids: list[int] | None) -> None:
        for token_id in token_ids or []:
            if token_id == self._open_tag_id:
                self._counting = True
            elif token_id == self._close_tag_id:
                self._counting = False
            elif self._counting:
                self.reasoning_tokens += 1


def _resolve_reasoning_tag_ids(server_context) -> tuple[int, int] | None:
    """Resolve reasoning-tag token ids from the configured reasoning parser.

    Returns ``None`` when no reasoning parser is configured (reasoning
    usage is then left unreported, matching the previous behavior).
    """
    parser_cls = getattr(server_context, 'response_parser_cls', None)
    if parser_cls is None:
        return None
    reasoning_cls = getattr(parser_cls, 'reasoning_parser_cls', None)
    tokenizer = getattr(parser_cls, 'tokenizer', None)
    if reasoning_cls is None or tokenizer is None:
        return None
    vocab = tokenizer.get_vocab()
    try:
        return (vocab[reasoning_cls.get_reasoning_open_tag()],
                vocab[reasoning_cls.get_reasoning_close_tag()])
    except KeyError:
        return None


def _ends_inside_reasoning(token_ids: list[int] | None, open_tag_id: int,
                           close_tag_id: int) -> bool:
    """Whether a token sequence ends inside an open reasoning block.

    Decided by the last reasoning tag token present in the sequence.
    """
    for token_id in reversed(token_ids or []):
        if token_id == open_tag_id:
            return True
        if token_id == close_tag_id:
            return False
    return False


def _get_prompt_token_ids(prepared_request) -> list[int] | None:
    """Prompt token ids of a prepared request, when available."""
    inputs = getattr(prepared_request, 'inputs', None)
    if isinstance(inputs, dict):
        return inputs.get('input_ids')
    return getattr(inputs, 'input_ids', None)


def _build_reasoning_counters(reasoning_tag_ids, prepared_requests):
    """Create one reasoning counter per prepared request, or ``None``.

    Each counter is seeded from its prompt's reasoning state so prompts that already end inside a reasoning block are
    counted correctly.
    """
    if reasoning_tag_ids is None:
        return None
    return [
        _ReasoningTokenCounter(*reasoning_tag_ids,
                               prompt_token_ids=_get_prompt_token_ids(prepared_request))
        for prepared_request in prepared_requests
    ]


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
    if request.top_k is not None and request.top_k < -1:
        return f'The top_k {request.top_k!r} must be -1, 0, or a positive integer.'
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
        - **frequency_penalty** (replaced with repetition_penalty)
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
        prepared_requests = await asyncio.gather(*[
            server_context.async_engine.preprocess(prompt,
                                                   session,
                                                   gen_config=gen_config,
                                                   do_preprocess=False,
                                                   adapter_name=adapter_name)
            for prompt, session in zip(request.prompt, sessions)
        ], return_exceptions=True)
        preprocess_error = next((result for result in prepared_requests if isinstance(result, BaseException)), None)
        if preprocess_error is not None:
            for result in prepared_requests:
                if not isinstance(result, BaseException):
                    result.inputs.clear()
            for session in sessions:
                server_context.session_manager.remove(session)
            if isinstance(preprocess_error, asyncio.CancelledError):
                raise preprocess_error
            if not isinstance(preprocess_error, RequestError):
                preprocess_error = RequestError(ErrorCode.INTERNAL_ERROR)
            return create_request_error_response(preprocess_error)
        generators = [
            server_context.async_engine.generate(prepared_request, stream_response=True)
            for prepared_request in prepared_requests
        ]

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
            reasoning_tag_ids = _resolve_reasoning_tag_ids(server_context)
            counters = _build_reasoning_counters(reasoning_tag_ids, prepared_requests)
            try:
                for gen_index, generator in enumerate(generators):
                    async for res in generator:
                        if counters is not None:
                            counters[gen_index].update(res.token_ids)
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
            except RequestError as e:
                yield f'data: {json.dumps({"error": request_error_payload(e)})}\n\n'
                yield 'data: [DONE]\n\n'
                return
            if final_usage is not None:
                if counters is not None:
                    final_usage.completion_tokens_details = CompletionTokensDetails(
                        reasoning_tokens=sum(counter.reasoning_tokens for counter in counters))
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
        reasoning_tag_ids = _resolve_reasoning_tag_ids(server_context)
        counters = _build_reasoning_counters(reasoning_tag_ids, prepared_requests)

        async def _inner_call(i, generator, session, counter=None):
            nonlocal cache_block_ids, remote_token_ids
            final_logprobs = []
            final_token_ids = []
            final_res = None
            text = ''
            try:
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
                        if counter is not None:
                            counter.update(res.token_ids)
                        final_res = res
                        text += res.response
                        cache_block_ids.append(res.cache_block_ids)
                        remote_token_ids.append(res.token_ids)
                        if res.token_ids:
                            final_token_ids.extend(res.token_ids)
                        if res.logprobs:
                            final_logprobs.extend(res.logprobs)
            except RequestError as e:
                return create_request_error_response(e)

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
            _inner_call(i, generators[i], sessions[i],
                        counters[i] if counters is not None else None)
            for i in range(len(generators))
        ])
        error_response = next(
            (result for result in call_results if result is not None), None)
        if error_response is not None:
            return error_response

        if counters is not None:
            usage.completion_tokens_details = CompletionTokensDetails(
                reasoning_tokens=sum(counter.reasoning_tokens for counter in counters))

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
