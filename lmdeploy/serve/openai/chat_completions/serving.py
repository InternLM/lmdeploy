# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from http import HTTPStatus

import shortuuid
from fastapi import APIRouter, Depends, Request

from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest
from lmdeploy.serve.core.exceptions import RequestError
from lmdeploy.serve.openai.endpoints.common import build_serving_generation_config, validate_request
from lmdeploy.serve.openai.errors import (
    create_error_response,
    create_request_error_response,
    request_error_payload,
)
from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChoiceLogprobs,
    DeltaMessage,
    UsageInfo,
)
from lmdeploy.serve.openai.utils import maybe_filter_parallel_tool_calls
from lmdeploy.serve.utils.request_cleanup import with_request_cleanup
from lmdeploy.serve.utils.server_utils import validate_json_request
from lmdeploy.utils import get_logger

from .fanout import fanout_chat_completions
from .logits_processors import logit_bias_logits_processor
from .logprobs import _create_chat_completion_logprobs, _create_output_token_logprobs
from .streaming_response import ManagedStreamingResponse
from .validation import check_request

logger = get_logger('lmdeploy')


def register(router: APIRouter, server_context) -> None:

    @router.post('/v1/chat/completions',
                 dependencies=[Depends(validate_json_request)])
    async def chat_completions_v1(request: ChatCompletionRequest,
                                  raw_request: Request = None):
        """Completion API similar to OpenAI's API.

        Refer to https://platform.openai.com/docs/api-reference/chat/create
        for the API specification.

        The request should be a JSON object with the following fields:

        - **model**: model name. Available from /v1/models.
        - **messages**: string prompt or chat history in OpenAI format. Chat history example:
          ``[{"role": "user", "content": "hi"}]``.
        - **temperature** (float): to modulate the next token probability
        - **top_p** (float): If set to float < 1, only the smallest set of most
          probable tokens with probabilities that add up to top_p or higher
          are kept for generation.
        - **n** (int): How many chat completion choices to generate for each input
          message. Accepts values from 1 to 128.
        - **stream**: whether to stream the results or not. Default to false.
        - **stream_options**: Options for streaming response. Only set this when you
          set stream: true.
        - **max_completion_tokens** (int | None): output token nums. Default to None.
        - **max_tokens** (int | None): output token nums. Default to None.
          Deprecated: Use max_completion_tokens instead.
        - **repetition_penalty** (float): The parameter for repetition penalty.
          1.0 means no penalty
        - **stop** (str | list[str] | None): To stop generating further
          tokens. Only accept stop words that's encoded to one token idex.
        - **response_format** (dict | None): To generate response according to given
          schema. Examples:

          .. code-block:: json

            {
              "type": "json_schema",
              "json_schema":{
                "name": "test",
                "schema":{
                  "properties":{
                    "name":{"type":"string"}
                  },
                  "required":["name"],
                  "type":"object"
                }
              }
            }

          or ``{"type": "regex_schema", "regex_schema": "call me [A-Za-z]{1,10}"}``
        - **logit_bias** (dict): Bias to logits. Only supported in pytorch engine.
        - **tools** (list): A list of tools the model may call. Currently, only
          internlm2 functions are supported as a tool. Use this to specify a
          list of functions for which the model can generate JSON inputs.
        - **tool_choice** (str | object): Controls which (if any) tool is called by
          the model. `none` means the model will not call any tool and instead
          generates a message. Specifying a particular tool via
          ``{"type": "function", "function": {"name": "my_function"}}``
          forces the model to call that tool. `auto` or `required` will put all
          the tools informationto the model.

        Additional arguments supported by LMDeploy:

        - **top_k** (int): The number of the highest probability vocabulary
          tokens to keep for top-k-filtering
        - **ignore_eos** (bool): indicator for ignoring eos
        - **skip_special_tokens** (bool): Whether or not to remove special tokens
          in the decoding. Default to be True.
        - **spaces_between_special_tokens** (bool): Whether or not to add spaces
          around special tokens. The behavior of Fast tokenizers is to have
          this to False. This is setup to True in slow tokenizers.
        - **min_new_tokens** (int): To generate at least numbers of tokens.
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

        - **presence_penalty** (replaced with repetition_penalty)
        - **frequency_penalty** (replaced with repetition_penalty)
        """
        json_request = await raw_request.json()
        error_check_ret = validate_request(
            request,
            server_context,
            check_request,
            json_request=json_request,
        )
        if error_check_ret is not None:
            return error_check_ret
        if request.n is not None and request.n > 1:
            return await fanout_chat_completions(
                chat_completions_v1,
                request,
                raw_request,
                json_request,
            )

        # Resolve input: messages has priority over input_ids/image_data
        messages_empty = (request.messages is None or request.messages == ''
                          or (isinstance(request.messages, list)
                              and len(request.messages) == 0))
        resolved_input_ids = None
        if messages_empty and request.input_ids is not None:
            # /generate-style input: use input_ids (+ optional image_data)
            resolved_input_ids = request.input_ids
            if request.image_data is not None:
                # Convert image_data to OpenAI multimodal content format
                image_data = request.image_data
                image_input = []
                if not isinstance(image_data, list):
                    image_data = [image_data]
                for img in image_data:
                    if isinstance(img, str):
                        image_input.append(
                            dict(type='image_url', image_url=dict(url=img)))
                    else:
                        image_input.append(
                            dict(type='image_url', image_url=img))
                text_input = dict(type='text', text=request.input_ids)
                request.messages = [
                    dict(role='user', content=[text_input] + image_input)
                ]
                resolved_input_ids = None  # image_data conversion takes over
            else:
                # input_ids only — engine requires messages=None
                request.messages = None

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
        request_id = f'chatcmpl-{shortuuid.random()}'
        created_time = int(time.time())

        tokenizer = server_context.async_engine.tokenizer.model.model
        gen_logprobs, logits_processors = None, None
        if request.logprobs:
            gen_logprobs = request.top_logprobs or 1
        elif request.return_logprob:
            gen_logprobs = 1
        if request.logit_bias is not None:
            try:
                logits_processors = [
                    logit_bias_logits_processor(request.logit_bias, tokenizer)
                ]
            except Exception as e:
                return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

        parser_cls = server_context.response_parser_cls
        try:
            response_parser = parser_cls(request)
        except ValueError as e:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
        # request is normalized and may be adjusted by the parser
        # (e.g. GPT-OSS clears response_format and injects the schema into messages)
        request = response_parser.request

        gen_config = build_serving_generation_config(
            request,
            server_context,
            logprobs=gen_logprobs,
            stop_words=request.stop,
            logits_processors=logits_processors,
            random_seed=request.seed,
            migration_request=migration_request,
            with_cache=with_cache,
            preserve_cache=preserve_cache,
        )

        # text completion for string input or input_ids
        do_preprocess = (False if isinstance(request.messages, str)
                         or resolved_input_ids is not None else
                         request.do_preprocess)
        chat_template_kwargs = request.chat_template_kwargs or {}
        if request.enable_thinking is not None:
            logger.warning(
                '`enable_thinking` will be deprecated in the future, '
                'please use `chat_template_kwargs` instead.')
            if chat_template_kwargs.get('enable_thinking') is None:
                chat_template_kwargs['enable_thinking'] = request.enable_thinking
            else:
                logger.warning(
                    '`enable_thinking` in `chat_template_kwargs` will override the value in request.'
                )

        session = server_context.create_session(request.session_id)
        try:
            preprocessed = await server_context.async_engine.preprocess(
                request.messages,
                session,
                gen_config=gen_config,
                tools=request.tools,
                reasoning_effort=request.reasoning_effort,
                do_preprocess=do_preprocess,
                adapter_name=adapter_name,
                chat_template_kwargs=chat_template_kwargs or None,
                input_ids=resolved_input_ids,
                media_io_kwargs=request.media_io_kwargs,
                mm_processor_kwargs=request.mm_processor_kwargs)
        except RequestError as error:
            return create_request_error_response(error)

        result_generator = server_context.async_engine.generate(
                preprocessed,
                stream_response=True)  # always use stream to enable batching
        include_usage = bool(request.stream_options
                             and request.stream_options.include_usage)

        def create_stream_response_json(
                index: int,
                delta_message: DeltaMessage,
                finish_reason: str | None = None,
                logprobs: ChoiceLogprobs | None = None,
                output_token_logprobs: list[tuple[float, int]] | None = None,
                routed_experts=None,
                output_ids=None) -> dict:
            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=delta_message,
                finish_reason=finish_reason,
                logprobs=logprobs,
                output_token_logprobs=output_token_logprobs,
                output_ids=output_ids,
                routed_experts=routed_experts)
            choice_data = maybe_filter_parallel_tool_calls(
                choice_data, request)
            response = ChatCompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=[choice_data],
                usage=None,
            )
            response_dict = response.model_dump(mode='json', exclude_none=True)
            if include_usage:
                response_dict['usage'] = None
            return response_dict

        def create_stream_usage_response_json(usage: UsageInfo) -> str:
            response = ChatCompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=[],
                usage=usage,
            )
            return response.model_dump_json(exclude_none=True)

        async def _completion_stream_generator() -> AsyncGenerator[str, None]:
            streaming_tools = False
            final_usage = None
            async for res in result_generator:
                logprobs = None
                output_token_logprobs = None
                if request.logprobs and res.logprobs:
                    logprobs = _create_chat_completion_logprobs(
                        tokenizer, res.token_ids, res.logprobs)
                if request.return_logprob:
                    output_token_logprobs = _create_output_token_logprobs(
                        res.token_ids, res.logprobs)
                delta_token_ids = res.token_ids if res.token_ids is not None else []
                stream_deltas = response_parser.stream_chunk(
                    res.response,
                    delta_token_ids,
                    final=res.finish_reason is not None,
                )
                if res.finish_reason and include_usage:
                    final_usage = UsageInfo.build(
                        prompt_tokens=res.input_token_len,
                        completion_tokens=res.generate_token_len,
                        cached_tokens=res.cached_tokens,
                        reasoning_tokens=response_parser.reasoning_tokens,
                    )
                if not stream_deltas:
                    # Parser may buffer partial protocol tags and emit no visible delta
                    # while the engine still produced new tokens (e.g. MTP batch). Do not
                    # drop those token ids; emit them once on a placeholder delta.
                    if res.finish_reason is None and not delta_token_ids:
                        continue
                    stream_deltas = [(DeltaMessage(role='assistant',
                                                   content=''), False)]
                should_validate_complete = (res.finish_reason
                                            in ('stop', 'length') and
                                            (request.return_token_ids
                                             or request.return_routed_experts))
                if should_validate_complete and not response_parser.validate_complete():
                    res.finish_reason = 'parse_error'

                for delta_index, (delta_message,
                                  tool_emitted) in enumerate(stream_deltas):
                    if tool_emitted:
                        streaming_tools = True

                    is_last_delta = delta_index == len(stream_deltas) - 1
                    # The chat parser may split one engine yield into multiple protocol deltas,
                    # so attach the engine-level metadata to the last parsed delta.
                    finish_reason = res.finish_reason if is_last_delta else None
                    chunk_logprobs = logprobs if is_last_delta else None
                    chunk_output_token_logprobs = output_token_logprobs if is_last_delta else None

                    if (request.tool_choice != 'none'
                            and response_parser.tool_parser is not None):
                        if finish_reason == 'stop' and streaming_tools is True:
                            finish_reason = 'tool_calls'

                    # Only output routed_experts in the final chunk
                    routed_experts = res.routed_experts if finish_reason is not None else None
                    # Emit token ids once per engine yield on the last parsed delta, when
                    # accumulated delta text and token ids for this step are aligned.
                    stream_output_ids = delta_token_ids if (
                        request.return_token_ids and is_last_delta) else None

                    response_json = create_stream_response_json(
                        index=0,
                        delta_message=delta_message,
                        finish_reason=finish_reason,
                        logprobs=chunk_logprobs,
                        output_token_logprobs=chunk_output_token_logprobs,
                        routed_experts=routed_experts,
                        output_ids=stream_output_ids)
                    if res.cache_block_ids is not None and is_last_delta:
                        response_json['cache_block_ids'] = res.cache_block_ids
                        response_json['remote_token_ids'] = res.token_ids
                    yield f'data: {json.dumps(response_json)}\n\n'
            if final_usage is not None:
                yield f'data: {create_stream_usage_response_json(final_usage)}\n\n'
            yield 'data: [DONE]\n\n'

        async def completion_stream_generator() -> AsyncGenerator[str, None]:
            try:
                async for chunk in _completion_stream_generator():
                    yield chunk
            except RequestError as error:
                yield f'data: {json.dumps({"error": request_error_payload(error)})}\n\n'
                yield 'data: [DONE]\n\n'

        # Streaming response
        if request.stream:
            return ManagedStreamingResponse(
                completion_stream_generator(),
                result_generators=[result_generator],
                sessions=[session],
                session_mgr=server_context.session_manager,
                media_type='text/event-stream')

        # Non-streaming response
        final_logprobs = []
        final_token_ids = []
        final_res = None
        text = ''
        cache_block_ids = []
        remote_token_ids = []
        try:
            async with aclosing(
                    with_request_cleanup(
                        result_generator, [result_generator], [session],
                        server_context.session_manager)) as generator:
                async for res in generator:
                    if await raw_request.is_disconnected():
                        # Abort the request if the client disconnects.
                        await session.async_abort()
                        return create_error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected')
                    final_res = res
                    text += res.response
                    if res.token_ids:
                        final_token_ids.extend(res.token_ids)
                    if res.logprobs:
                        final_logprobs.extend(res.logprobs)
                    cache_block_ids.append(res.cache_block_ids)
                    remote_token_ids.append(res.token_ids)
        except RequestError as error:
            return create_request_error_response(error)

        tool_calls = None
        reasoning_content = None

        try:
            raw_text = text
            text, tool_calls, reasoning_content = response_parser.parse_complete(
                text, final_token_ids)
            should_validate_complete = (
                final_res.finish_reason in ('stop', 'length') and
                (request.return_token_ids or request.return_routed_experts))
            if should_validate_complete and not response_parser.validate_complete(
                    raw_text):
                final_res.finish_reason = 'parse_error'
            if isinstance(tool_calls, list) and len(tool_calls):
                if final_res.finish_reason == 'stop':
                    final_res.finish_reason = 'tool_calls'

        except Exception as e:
            logger.error(f'Failed to parse {text}. Exception: {e}.')
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'Failed to parse fc related info to json format!')

        message = ChatMessage(role='assistant',
                              content=text,
                              tool_calls=tool_calls,
                              reasoning_content=reasoning_content)

        logprobs = None
        if request.logprobs and len(final_logprobs):
            logprobs = _create_chat_completion_logprobs(
                tokenizer, final_token_ids, final_logprobs)
        output_token_logprobs = None
        if request.return_logprob and len(final_logprobs):
            output_token_logprobs = _create_output_token_logprobs(
                final_token_ids, final_logprobs)

        assert final_res is not None
        choices = []
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,
            logprobs=logprobs,
            output_token_logprobs=output_token_logprobs,
            finish_reason=final_res.finish_reason,
            output_ids=final_token_ids if request.return_token_ids else None,
            routed_experts=final_res.routed_experts
            if request.return_routed_experts else None,
        )
        choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
        choices.append(choice_data)

        if with_cache:
            cache_block_ids = cache_block_ids[0]
            remote_token_ids = [remote_token_ids[0][-1]]

        usage = UsageInfo.build(
            prompt_tokens=final_res.input_token_len,
            completion_tokens=final_res.generate_token_len,
            cached_tokens=final_res.cached_tokens,
            reasoning_tokens=response_parser.reasoning_tokens,
        )
        response = ChatCompletionResponse(
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
