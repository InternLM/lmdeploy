# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from http import HTTPStatus

import shortuuid
from fastapi import APIRouter, Depends, Request

from lmdeploy.pytorch.disagg.conn.protocol import MigrationRequest
from lmdeploy.serve.core.chat_runner import (
    ChatRunner,
    ChatRunnerOptions,
)
from lmdeploy.serve.core.exceptions import RequestError
from lmdeploy.serve.openai.endpoints.common import validate_request
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
from lmdeploy.serve.utils.server_utils import validate_json_request
from lmdeploy.serve.utils.streaming_response import ManagedStreamingResponse

from .fanout import fanout_chat_completions
from .logits_processors import logit_bias_logits_processor
from .logprobs import _create_chat_completion_logprobs, _create_output_token_logprobs
from .validation import check_request


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
        - **messages**: chat history in OpenAI format. Chat history example:
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
          generates a message. `auto` lets the model choose whether to call a
          tool, while `required` constrains generation to at least one valid
          call. Specifying a particular tool via
          ``{"type": "function", "function": {"name": "my_function"}}``
          forces the model to call that tool.

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
        messages_empty = request.messages is None or len(request.messages) == 0
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
                request = request.model_copy(
                    update={
                        'messages': [dict(role='user', content=[text_input] + image_input)],
                        'input_ids': None,
                        'image_data': None,
                    })
                resolved_input_ids = None  # image_data conversion takes over

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
        logits_processors = None
        if request.logit_bias is not None:
            try:
                logits_processors = [
                    logit_bias_logits_processor(request.logit_bias, tokenizer)
                ]
            except Exception as e:
                return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

        try:
            chat_runner = await ChatRunner.prepare(
                server_context,
                request,
                ChatRunnerOptions(
                    input_ids=resolved_input_ids,
                    do_preprocess=resolved_input_ids is None,
                    adapter_name=adapter_name,
                    gen_config_kwargs=dict(
                        logits_processors=logits_processors,
                        random_seed=request.seed,
                        migration_request=migration_request,
                        with_cache=with_cache,
                        preserve_cache=preserve_cache,
                    ),
                ),
            )
        except RequestError as error:
            return create_request_error_response(error)
        # request is normalized and may be adjusted by the parser
        # (e.g. GPT-OSS clears response_format and injects the schema into messages)
        request = chat_runner.request
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
            try:
                final_usage = None
                async for chunk in chat_runner.stream():
                    logprobs = None
                    output_token_logprobs = None
                    if request.logprobs and chunk.logprobs:
                        logprobs = _create_chat_completion_logprobs(
                            tokenizer, chunk.token_ids, chunk.logprobs)
                    if request.return_logprob and chunk.logprobs:
                        output_token_logprobs = _create_output_token_logprobs(
                            chunk.token_ids, chunk.logprobs)
                    if chunk.finish_reason and include_usage:
                        final_usage = UsageInfo.build(
                            prompt_tokens=chunk.input_token_len,
                            completion_tokens=chunk.generate_token_len,
                            cached_tokens=chunk.cached_tokens,
                            reasoning_tokens=chunk.reasoning_tokens,
                        )
                    # The chat parser may split one engine yield into multiple protocol deltas,
                    # so attach the engine-level metadata to the last parsed delta.
                    if not chunk.is_last_delta:
                        logprobs = None
                        output_token_logprobs = None
                    # Emit token ids once per engine yield on the last parsed delta, when
                    # accumulated delta text and token ids for this step are aligned.
                    stream_output_ids = chunk.token_ids if (request.return_token_ids and chunk.is_last_delta) else None

                    response_json = create_stream_response_json(
                        index=0,
                        delta_message=chunk.delta_message,
                        finish_reason=chunk.finish_reason,
                        logprobs=logprobs,
                        output_token_logprobs=output_token_logprobs,
                        routed_experts=chunk.routed_experts,
                        output_ids=stream_output_ids)
                    if chunk.cache_block_ids is not None and chunk.is_last_delta:
                        response_json['cache_block_ids'] = chunk.cache_block_ids
                        response_json['remote_token_ids'] = chunk.token_ids
                    yield f'data: {json.dumps(response_json)}\n\n'
                if final_usage is not None:
                    yield f'data: {create_stream_usage_response_json(final_usage)}\n\n'
                yield 'data: [DONE]\n\n'
            finally:
                await chat_runner.close()

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
                cleanup_callbacks=[chat_runner.close],
                media_type='text/event-stream')

        # Non-streaming response
        try:
            res = await chat_runner.collect(raw_request)
        except RequestError as error:
            return create_request_error_response(error)
        text = res.text
        tool_calls = res.tool_calls
        reasoning_content = res.reasoning_content

        message = ChatMessage(role='assistant',
                              content=text,
                              tool_calls=tool_calls,
                              reasoning_content=reasoning_content)

        logprobs = None
        if request.logprobs and len(res.logprobs):
            logprobs = _create_chat_completion_logprobs(
                tokenizer, res.token_ids, res.logprobs)
        output_token_logprobs = None
        if request.return_logprob and len(res.logprobs):
            output_token_logprobs = _create_output_token_logprobs(
                res.token_ids, res.logprobs)

        choices = []
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,
            logprobs=logprobs,
            output_token_logprobs=output_token_logprobs,
            finish_reason=res.finish_reason,
            output_ids=res.token_ids if request.return_token_ids else None,
            routed_experts=res.routed_experts
            if request.return_routed_experts else None,
        )
        choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
        choices.append(choice_data)

        if with_cache:
            cache_block_ids = res.cache_block_ids[0]
            remote_token_ids = [res.remote_token_ids[0][-1]]

        usage = UsageInfo.build(
            prompt_tokens=res.input_token_len,
            completion_tokens=res.generate_token_len,
            cached_tokens=res.cached_tokens,
            reasoning_tokens=res.reasoning_tokens,
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
