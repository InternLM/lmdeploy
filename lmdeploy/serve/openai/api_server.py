# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import copy
import os
import time
from functools import partial
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

from lmdeploy.archs import get_task
from lmdeploy.messages import (GenerationConfig, LogitsProcessor,
                               PytorchEngineConfig, TurbomindEngineConfig)
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.openai.protocol import (  # noqa: E501
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatCompletionTokenLogprob, ChatMessage,
    ChoiceLogprobs, CompletionRequest, CompletionResponse,
    CompletionResponseChoice, CompletionResponseStreamChoice,
    CompletionStreamResponse, DeltaMessage, EmbeddingsRequest, EncodeRequest,
    EncodeResponse, ErrorResponse, FunctionResponse, GenerateRequest,
    GenerateResponse, LogProbs, ModelCard, ModelList, ModelPermission,
    ToolCall, TopLogprob, UsageInfo)
from lmdeploy.tokenizer import DetokenizeState, Tokenizer
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class VariableInterface:
    """A IO interface maintaining variables."""
    async_engine: AsyncEngine = None
    session_id: int = 0
    api_keys: Optional[List[str]] = None
    request_hosts = []


app = FastAPI(docs_url='/')
get_bearer_token = HTTPBearer(auto_error=False)


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    """Check if client provide valid api key.

    Adopted from https://github.com/lm-sys/FastChat/blob/v0.2.35/fastchat/serve/openai_api_server.py#L108-L127
    """  # noqa
    if VariableInterface.api_keys:
        if auth is None or (
                token := auth.credentials) not in VariableInterface.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    'error': {
                        'message': 'Please request with valid api key!',
                        'type': 'invalid_request_error',
                        'param': None,
                        'code': 'invalid_api_key',
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


def get_model_list():
    """Available models.

    If it is a slora serving. The model list would be [model_name,
    adapter_name1, adapter_name2, ...]
    """
    model_names = [VariableInterface.async_engine.model_name]
    cfg = VariableInterface.async_engine.backend_config
    model_names += getattr(cfg, 'adapters', None) or []
    return model_names


@app.get('/v1/models', dependencies=[Depends(check_api_key)])
def available_models():
    """Show available models."""
    model_cards = []
    for model_name in get_model_list():
        model_cards.append(
            ModelCard(id=model_name,
                      root=model_name,
                      permission=[ModelPermission()]))
    return ModelList(data=model_cards)


def create_error_response(status: HTTPStatus,
                          message: str,
                          error_type='invalid_request_error'):
    """Create error response according to http status and message.

    Args:
        status (HTTPStatus): HTTP status codes and reason phrases
        message (str): error message
        error_type (str): error type
    """
    return JSONResponse(ErrorResponse(message=message,
                                      type=error_type,
                                      code=status.value).model_dump(),
                        status_code=status.value)


async def check_request(request) -> Optional[JSONResponse]:
    """Check if a request is valid."""
    if hasattr(request, 'model') and request.model not in get_model_list():
        return create_error_response(
            HTTPStatus.NOT_FOUND,
            f'The model `{request.model}` does not exist.')
    if hasattr(request, 'n') and request.n <= 0:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f'The n `{request.n}` must be a positive int.')
    if hasattr(request,
               'top_p') and not (request.top_p > 0 and request.top_p <= 1):
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f'The top_p `{request.top_p}` must be in (0, 1].')
    if hasattr(request, 'top_k') and request.top_k < 0:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f'The top_k `{request.top_k}` cannot be a negative integer.')
    if hasattr(request, 'temperature') and not (request.temperature <= 2
                                                and request.temperature >= 0):
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f'The temperature `{request.temperature}` must be in [0, 2]')
    return


def _create_completion_logprobs(tokenizer: Tokenizer,
                                token_ids: List[int] = None,
                                logprobs: List[Dict[int, float]] = None,
                                skip_special_tokens: bool = True,
                                offset: int = 0,
                                all_token_ids: List[int] = None,
                                state: DetokenizeState = None):
    """create openai LogProbs for completion.

    Args:
        tokenizer (Tokenizer): tokenizer.
        token_ids (List[int]): output token ids.
        logprobs (List[Dict[int, float]]): the top logprobs for each output
            position.
        skip_special_tokens (bool): Whether or not to remove special tokens
            in the decoding. Default to be True.
        offset (int): text offset.
        all_token_ids (int): the history output token ids.
        state (DetokenizeState): tokenizer decode state.
    """
    if logprobs is None or len(logprobs) == 0:
        return None, None, None, None

    if all_token_ids is None:
        all_token_ids = []
    if state is None:
        state = DetokenizeState()

    out_logprobs = LogProbs()
    out_logprobs.top_logprobs = []
    for token_id, tops in zip(token_ids, logprobs):
        out_logprobs.text_offset.append(offset)
        out_logprobs.token_logprobs.append(tops[token_id])

        res = {}
        out_state = None
        for top_id, prob in tops.items():
            response, _state = tokenizer.detokenize_incrementally(
                all_token_ids + [top_id],
                copy.deepcopy(state),
                skip_special_tokens=skip_special_tokens)
            res[response] = prob
            if top_id == token_id:
                out_state = _state
                offset += len(response)
                out_logprobs.tokens.append(response)

        out_logprobs.top_logprobs.append(res)
        state = out_state
        all_token_ids.append(token_id)

    return out_logprobs, offset, all_token_ids, state


def _create_chat_completion_logprobs(tokenizer: Tokenizer,
                                     token_ids: List[int] = None,
                                     logprobs: List[Dict[int, float]] = None):
    """create openai LogProbs for chat.completion.

    Args:
        tokenizer (Tokenizer): tokenizer.
        token_ids (List[int]): output token ids.
        logprobs (List[Dict[int, float]]): the top logprobs for each output
            position.
    Returns:
        ChoiceLogprobs: logprob result.
    """
    if token_ids is None or logprobs is None:
        return None

    content: List[ChatCompletionTokenLogprob] = []
    for token_id, tops in zip(token_ids, logprobs):
        item = ChatCompletionTokenLogprob(token='',
                                          bytes=[],
                                          logprob=0.0,
                                          top_logprobs=[])
        for top_id, prob in tops.items():
            token = tokenizer.model.model.convert_ids_to_tokens(top_id)
            if isinstance(token, bytes):
                _bytes = list(token)
                token = token.decode('utf-8', errors='backslashreplace')
            else:
                _bytes = list(token.encode())  # token is str
            if top_id == token_id:
                item.token = token
                item.bytes = _bytes
                item.logprob = prob
            else:
                item.top_logprobs.append(
                    TopLogprob(token=token, bytes=_bytes, logprob=prob))
        content.append(item)
    return ChoiceLogprobs(content=content)


@app.get('/health')
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


# modified from https://github.com/vllm-project/vllm/blob/v0.5.4/vllm/entrypoints/openai/logits_processors.py#L51  # noqa
def logit_bias_logits_processor(logit_bias: Union[Dict[int, float],
                                                  Dict[str, float]],
                                tokenizer) -> LogitsProcessor:
    try:
        # Convert token_id to integer
        # Clamp the bias between -100 and 100 per OpenAI API spec
        clamped_logit_bias: Dict[int, float] = {
            int(token_id): min(100.0, max(-100.0, bias))
            for token_id, bias in logit_bias.items()
        }
    except ValueError as exc:
        raise ValueError(
            'Found token_id in logit_bias that is not '
            'an integer or string representing an integer') from exc

    # Check if token_id is within the vocab size
    for token_id, bias in clamped_logit_bias.items():
        if token_id < 0 or token_id >= tokenizer.vocab_size:
            raise ValueError(f'token_id {token_id} in logit_bias contains '
                             'out-of-vocab token id')

    def _logit_bias_processor(
        logit_bias,
        token_ids,
        logits,
    ):
        for token_id, bias in logit_bias.items():
            logits[token_id] = logits[token_id] + bias
        return logits

    return partial(_logit_bias_processor, clamped_logit_bias)


@app.post('/v1/chat/completions', dependencies=[Depends(check_api_key)])
async def chat_completions_v1(request: ChatCompletionRequest,
                              raw_request: Request = None):
    """Completion API similar to OpenAI's API.

    Refer to  `https://platform.openai.com/docs/api-reference/chat/create`
    for the API specification.

    The request should be a JSON object with the following fields:
    - model: model name. Available from /v1/models.
    - messages: string prompt or chat history in OpenAI format. Chat history
        example: `[{"role": "user", "content": "hi"}]`.
    - temperature (float): to modulate the next token probability
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - n (int): How many chat completion choices to generate for each input
        message. **Only support one here**.
    - stream: whether to stream the results or not. Default to false.
    - max_tokens (int | None): output token nums. Default to None.
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - stop (str | List[str] | None): To stop generating further
        tokens. Only accept stop words that's encoded to one token idex.
    - response_format (Dict | None): Only pytorch backend support formatting
        response. Examples: `{"type": "json_schema", "json_schema": {"name":
        "test","schema": {"properties": {"name": {"type": "string"}},
        "required": ["name"], "type": "object"}}}`
        or `{"type": "regex_schema", "regex_schema": "call me [A-Za-z]{1,10}"}`
    - logit_bias (Dict): Bias to logits. Only supported in pytorch engine.
    - tools (List): A list of tools the model may call. Currently, only
        internlm2 functions are supported as a tool. Use this to specify a
        list of functions for which the model can generate JSON inputs.
    - tool_choice (str | object): Controls which (if any) tool is called by
        the model. `none` means the model will not call any tool and instead
        generates a message. Specifying a particular tool via {"type":
        "function", "function": {"name": "my_function"}} forces the model to
        call that tool. `auto` or `required` will put all the tools information
        to the model.

    Additional arguments supported by LMDeploy:
    - top_k (int): The number of the highest probability vocabulary
        tokens to keep for top-k-filtering
    - ignore_eos (bool): indicator for ignoring eos
    - skip_special_tokens (bool): Whether or not to remove special tokens
        in the decoding. Default to be True.

    Currently we do not support the following features:
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    if request.session_id == -1:
        VariableInterface.session_id += 1
        request.session_id = VariableInterface.session_id
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret
    if VariableInterface.async_engine.id2step.get(str(request.session_id),
                                                  0) != 0:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f'The session_id `{request.session_id}` is occupied.')

    model_name = request.model
    adapter_name = None
    if model_name != VariableInterface.async_engine.model_name:
        adapter_name = model_name  # got a adapter name
    request_id = str(request.session_id)
    created_time = int(time.time())

    if isinstance(request.stop, str):
        request.stop = [request.stop]

    gen_logprobs, logits_processors = None, None
    if request.logprobs and request.top_logprobs:
        gen_logprobs = request.top_logprobs
    response_format = None
    if request.response_format and request.response_format.type != 'text':
        if VariableInterface.async_engine.backend != 'pytorch':
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'only pytorch backend can use response_format now')
        response_format = request.response_format.model_dump()

    if request.logit_bias is not None:
        try:
            logits_processors = [
                logit_bias_logits_processor(
                    request.logit_bias,
                    VariableInterface.async_engine.tokenizer.model)
            ]
        except Exception as e:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    random_seed = request.seed if request.seed else None

    gen_config = GenerationConfig(
        max_new_tokens=request.max_tokens,
        do_sample=True,
        logprobs=gen_logprobs,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos,
        stop_words=request.stop,
        skip_special_tokens=request.skip_special_tokens,
        response_format=response_format,
        logits_processors=logits_processors,
        random_seed=random_seed)

    tools = None
    if request.tools and request.tool_choice != 'none':
        gen_config.skip_special_tokens = False
        if request.stream is True:
            logger.warning('Set stream to False for tools')
            request.stream = False
        # internlm2 only uses contents inside function regardless of 'type'
        if not isinstance(request.tool_choice, str):
            tools = [
                item.function.model_dump() for item in request.tools
                if item.function.name == request.tool_choice.function.name
            ]
        else:
            tools = [item.function.model_dump() for item in request.tools]
    result_generator = VariableInterface.async_engine.generate(
        request.messages,
        request.session_id,
        gen_config=gen_config,
        tools=tools,
        stream_response=True,  # always use stream to enable batching
        sequence_start=True,
        sequence_end=True,
        do_preprocess=not isinstance(request.messages,
                                     str),  # text completion for string input
        adapter_name=adapter_name,
    )

    def create_stream_response_json(index: int,
                                    text: str,
                                    finish_reason: Optional[str] = None,
                                    logprobs: Optional[LogProbs] = None,
                                    usage: Optional[UsageInfo] = None) -> str:
        choice_data = ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(role='assistant', content=text),
            finish_reason=finish_reason,
            logprobs=logprobs)
        response = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
            usage=usage,
        )
        response_json = response.model_dump_json()

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        async for res in result_generator:
            logprobs, usage = None, None
            if gen_logprobs and res.logprobs:
                logprobs = _create_chat_completion_logprobs(
                    VariableInterface.async_engine.tokenizer, res.token_ids,
                    res.logprobs)
            if request.stream_options and request.stream_options.include_usage:
                total_tokens = sum([
                    res.history_token_len, res.input_token_len,
                    res.generate_token_len
                ])
                usage = UsageInfo(
                    prompt_tokens=res.input_token_len,
                    completion_tokens=res.generate_token_len,
                    total_tokens=total_tokens,
                )
            response_json = create_stream_response_json(
                index=0,
                text=res.response,
                finish_reason=res.finish_reason,
                logprobs=logprobs,
                usage=usage)
            yield f'data: {response_json}\n\n'
        yield 'data: [DONE]\n\n'

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type='text/event-stream')

    # Non-streaming response
    final_logprobs = []
    final_token_ids = []
    final_res = None
    text = ''
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await VariableInterface.async_engine.stop_session(
                request.session_id)
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         'Client disconnected')
        final_res = res
        text += res.response
        if res.token_ids:
            final_token_ids.extend(res.token_ids)
        if res.logprobs:
            final_logprobs.extend(res.logprobs)

    tool_calls = None
    if request.tool_choice != 'none' and ('<|plugin|>' in text
                                          or '<function=' in text):
        if final_res.finish_reason == 'stop':
            final_res.finish_reason = 'tool_calls'
        try:  # TODO add json_schema guidance to turbomind
            text, action_id, name, parameters = VariableInterface.async_engine.parse_tool_response(  # noqa
                text, request.tools)
            tool_calls = [
                ToolCall(id=str(action_id),
                         function=FunctionResponse(name=name,
                                                   arguments=parameters))
            ]
        except Exception as e:
            logger.error(f'Exception: {e}')
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'Failed to parse fc related info to json format!')

    logprobs = None
    if gen_logprobs and len(final_logprobs):
        logprobs = _create_chat_completion_logprobs(
            VariableInterface.async_engine.tokenizer, final_token_ids,
            final_logprobs)

    assert final_res is not None
    choices = []
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role='assistant',
                            content=text,
                            tool_calls=tool_calls),
        logprobs=logprobs,
        finish_reason=final_res.finish_reason,
    )
    choices.append(choice_data)

    total_tokens = sum([
        final_res.history_token_len, final_res.input_token_len,
        final_res.generate_token_len
    ])
    usage = UsageInfo(
        prompt_tokens=final_res.input_token_len,
        completion_tokens=final_res.generate_token_len,
        total_tokens=total_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    return response


@app.post('/v1/completions', dependencies=[Depends(check_api_key)])
async def completions_v1(request: CompletionRequest,
                         raw_request: Request = None):
    """Completion API similar to OpenAI's API.

    Go to `https://platform.openai.com/docs/api-reference/completions/create`
    for the API specification.

    The request should be a JSON object with the following fields:
    - model (str): model name. Available from /v1/models.
    - prompt (str): the input prompt.
    - suffix (str): The suffix that comes after a completion of inserted text.
    - max_tokens (int): output token nums. Default to 16.
    - temperature (float): to modulate the next token probability
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - n (int): How many chat completion choices to generate for each input
        message. **Only support one here**.
    - stream: whether to stream the results or not. Default to false.
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - user (str): A unique identifier representing your end-user.
    - stop (str | List[str] | None): To stop generating further
        tokens. Only accept stop words that's encoded to one token idex.

    Additional arguments supported by LMDeploy:
    - ignore_eos (bool): indicator for ignoring eos
    - skip_special_tokens (bool): Whether or not to remove special tokens
        in the decoding. Default to be True.
    - top_k (int): The number of the highest probability vocabulary
        tokens to keep for top-k-filtering

    Currently we do not support the following features:
    - logprobs (not supported yet)
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    if request.session_id == -1:
        VariableInterface.session_id += 1
        request.session_id = VariableInterface.session_id
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret
    if VariableInterface.async_engine.id2step.get(str(request.session_id),
                                                  0) != 0:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f'The session_id `{request.session_id}` is occupied.')

    model_name = request.model
    adapter_name = None
    if model_name != VariableInterface.async_engine.model_name:
        adapter_name = model_name  # got a adapter name
    request_id = str(request.session_id)
    created_time = int(time.time())
    if isinstance(request.prompt, str):
        request.prompt = [request.prompt]
    if isinstance(request.stop, str):
        request.stop = [request.stop]
    random_seed = request.seed if request.seed else None

    gen_config = GenerationConfig(
        max_new_tokens=request.max_tokens if request.max_tokens else 512,
        do_sample=True,
        logprobs=request.logprobs,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos,
        stop_words=request.stop,
        skip_special_tokens=request.skip_special_tokens,
        random_seed=random_seed)
    generators = []
    for i in range(len(request.prompt)):
        result_generator = VariableInterface.async_engine.generate(
            request.prompt[i],
            request.session_id + i,
            gen_config=gen_config,
            stream_response=True,  # always use stream to enable batching
            sequence_start=True,
            sequence_end=True,
            do_preprocess=False,
            adapter_name=adapter_name)
        generators.append(result_generator)

    def create_stream_response_json(index: int,
                                    text: str,
                                    finish_reason: Optional[str] = None,
                                    logprobs: Optional[LogProbs] = None,
                                    usage: Optional[UsageInfo] = None) -> str:
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
            usage=usage,
        )
        response_json = response.model_dump_json()

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for generator in generators:
            offset = 0
            all_token_ids = []
            state = DetokenizeState()
            async for res in generator:
                logprobs = None
                usage = None
                if request.logprobs and res.logprobs:
                    logprobs, offset, all_token_ids, state = _create_completion_logprobs(  # noqa E501
                        VariableInterface.async_engine.tokenizer,
                        res.token_ids, res.logprobs,
                        gen_config.skip_special_tokens, offset, all_token_ids,
                        state)
                if request.stream_options and request.stream_options.include_usage:  # noqa E501
                    final_res = res
                    total_tokens = sum([
                        final_res.history_token_len, final_res.input_token_len,
                        final_res.generate_token_len
                    ])
                    usage = UsageInfo(
                        prompt_tokens=final_res.input_token_len,
                        completion_tokens=final_res.generate_token_len,
                        total_tokens=total_tokens,
                    )
                response_json = create_stream_response_json(
                    index=0,
                    text=res.response,
                    finish_reason=res.finish_reason,
                    logprobs=logprobs,
                    usage=usage)
                yield f'data: {response_json}\n\n'
        yield 'data: [DONE]\n\n'

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type='text/event-stream')

    # Non-streaming response
    usage = UsageInfo()
    choices = [None] * len(generators)

    async def _inner_call(i, generator):
        final_logprobs = []
        final_token_ids = []
        final_res = None
        text = ''
        async for res in generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await VariableInterface.async_engine.stop_session(
                    request.session_id)
                return create_error_response(HTTPStatus.BAD_REQUEST,
                                             'Client disconnected')
            final_res = res
            text += res.response
            if res.token_ids:
                final_token_ids.extend(res.token_ids)
            if res.logprobs:
                final_logprobs.extend(res.logprobs)

        logprobs = None
        if request.logprobs and len(final_logprobs):
            logprobs, _, _, _ = _create_completion_logprobs(
                VariableInterface.async_engine.tokenizer, final_token_ids,
                final_logprobs, gen_config.skip_special_tokens)

        assert final_res is not None
        choice_data = CompletionResponseChoice(
            index=i,
            text=text,
            finish_reason=final_res.finish_reason,
            logprobs=logprobs,
        )
        choices[i] = choice_data

        total_tokens = sum([
            final_res.history_token_len, final_res.input_token_len,
            final_res.generate_token_len
        ])
        usage.prompt_tokens += final_res.input_token_len
        usage.completion_tokens += final_res.generate_token_len
        usage.total_tokens += total_tokens

    await asyncio.gather(
        *[_inner_call(i, generators[i]) for i in range(len(generators))])

    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    return response


@app.post('/v1/embeddings', tags=['unsupported'])
async def create_embeddings(request: EmbeddingsRequest,
                            raw_request: Request = None):
    """Creates embeddings for the text."""
    return create_error_response(HTTPStatus.BAD_REQUEST,
                                 'Unsupported by turbomind.')


@app.post('/v1/encode', dependencies=[Depends(check_api_key)])
async def encode(request: EncodeRequest, raw_request: Request = None):
    """Encode prompts.

    The request should be a JSON object with the following fields:
    - input: the prompt to be encoded. In str or List[str] format.
    - do_preprocess: whether do preprocess or not. Default to False.
    - add_bos: True when it is the beginning of a conversation. False when it
        is not. Default to True.
    """

    def encode(prompt: str, do_preprocess: bool, add_bos: bool):
        if do_preprocess:
            prompt = VariableInterface.async_engine.chat_template.get_prompt(
                prompt, sequence_start=add_bos)
        input_ids = VariableInterface.async_engine.tokenizer.encode(
            prompt, add_bos=add_bos)
        return input_ids

    if isinstance(request.input, str):
        encoded = encode(request.input, request.do_preprocess, request.add_bos)
        return EncodeResponse(input_ids=encoded, length=len(encoded))
    else:
        encoded, length = [], []
        for prompt in request.input:
            ids = encode(prompt, request.do_preprocess, request.add_bos)
            encoded.append(ids)
            length.append(len(ids))
        return EncodeResponse(input_ids=encoded, length=length)


@app.post('/v1/chat/interactive', dependencies=[Depends(check_api_key)])
async def chat_interactive_v1(request: GenerateRequest,
                              raw_request: Request = None):
    """Generate completion for the request.

    - On interactive mode, the chat history is kept on the server. Please set
    `interactive_mode = True`.
    - On normal mode, no chat history is kept on the server. Set
    `interactive_mode = False`.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - image_url(str | List[str] | None): the image url or base64 encoded string
        for VL models.
    - session_id: determine which instance will be called. If not specified
        with a value other than -1, using random value directly.
    - interactive_mode (bool): turn on interactive mode or not. On interactive
        mode, session history is kept on the server (and vice versa).
    - stream: whether to stream the results or not.
    - stop (str | List[str] | None): To stop generating further
        tokens. Only accept stop words that's encoded to one token idex.
    - request_output_len (int): output token nums. If not specified, will use
        maximum possible number for a session.
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - top_k (int): The number of the highest probability vocabulary
        tokens to keep for top-k-filtering
    - temperature (float): to modulate the next token probability
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - ignore_eos (bool): indicator for ignoring eos
    - skip_special_tokens (bool): Whether or not to remove special tokens
        in the decoding. Default to be True.
    - adapter_name (str): For slora inference. Choose which lora to do the
        inference.
    """
    if request.cancel:
        if request.session_id != -1:
            await VariableInterface.async_engine.stop_session(
                request.session_id)
            return {
                'text': '',
                'tokens': 0,
                'input_tokens': 0,
                'history_tokens': 0,
                'finish_reason': 'stop'
            }
        else:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'please set a session_id to cancel a request')
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret
    if request.session_id == -1:
        VariableInterface.session_id += 1
        request.session_id = VariableInterface.session_id

    async_engine = VariableInterface.async_engine
    sequence_start = async_engine.id2step.get(str(request.session_id), 0) == 0
    sequence_end = not request.interactive_mode
    if isinstance(request.stop, str):
        request.stop = [request.stop]

    random_seed = request.seed if request.seed else None

    gen_config = GenerationConfig(
        max_new_tokens=request.request_output_len,
        do_sample=True,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos,
        stop_words=request.stop,
        skip_special_tokens=request.skip_special_tokens,
        random_seed=random_seed)
    if request.image_url:
        from lmdeploy.vl import load_image
        if isinstance(request.image_url, List):
            request.prompt = (request.prompt,
                              [load_image(url) for url in request.image_url])
        else:
            request.prompt = (request.prompt, load_image(request.image_url))
        if not hasattr(async_engine, '_convert_prompts'):
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                '`image_url` argument only works for VL model')
        request.prompt = async_engine._convert_prompts(request.prompt)
    generation = async_engine.generate(
        request.prompt,
        request.session_id,
        gen_config=gen_config,
        stream_response=True,  # always use stream to enable batching
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        adapter_name=request.adapter_name)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for out in generation:
            chunk = GenerateResponse(text=out.response,
                                     tokens=out.generate_token_len,
                                     input_tokens=out.input_token_len,
                                     history_tokens=out.history_token_len,
                                     finish_reason=out.finish_reason)
            data = chunk.model_dump_json()
            yield f'{data}\n'

    if request.stream:
        return StreamingResponse(stream_results(),
                                 media_type='text/event-stream')
    else:
        ret = {}
        text = ''
        tokens, input_tokens, history_tokens = 0, 0, 0
        finish_reason = None
        async for out in generation:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await async_engine.stop_session(request.session_id)
                return create_error_response(HTTPStatus.BAD_REQUEST,
                                             'Client disconnected')
            text += out.response
            tokens = out.generate_token_len
            input_tokens = out.input_token_len
            history_tokens = out.history_token_len
            finish_reason = out.finish_reason
        ret = {
            'text': text,
            'tokens': tokens,
            'input_tokens': input_tokens,
            'history_tokens': history_tokens,
            'finish_reason': finish_reason
        }
        return JSONResponse(ret)


def serve(model_path: str,
          model_name: Optional[str] = None,
          backend: Literal['turbomind', 'pytorch'] = 'turbomind',
          backend_config: Optional[Union[PytorchEngineConfig,
                                         TurbomindEngineConfig]] = None,
          chat_template_config: Optional[ChatTemplateConfig] = None,
          server_name: str = '0.0.0.0',
          server_port: int = 23333,
          allow_origins: List[str] = ['*'],
          allow_credentials: bool = True,
          allow_methods: List[str] = ['*'],
          allow_headers: List[str] = ['*'],
          log_level: str = 'ERROR',
          api_keys: Optional[Union[List[str], str]] = None,
          ssl: bool = False,
          **kwargs):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the path of a model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "InternLM/internlm-chat-20b-4bit",
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "internlm/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.
        model_name (str): the name of the served model. It can be accessed
            by the RESTful API `/v1/models`. If it is not specified,
            `model_path` will be adopted
        backend (str): either `turbomind` or `pytorch` backend. Default to
            `turbomind` backend.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig): beckend
            config instance. Default to none.
        chat_template_config (ChatTemplateConfig): chat template configuration.
            Default to None.
        server_name (str): host ip for serving
        server_port (int): server port
        tp (int): tensor parallel
        allow_origins (List[str]): a list of allowed origins for CORS
        allow_credentials (bool): whether to allow credentials for CORS
        allow_methods (List[str]): a list of allowed HTTP methods for CORS
        allow_headers (List[str]): a list of allowed HTTP headers for CORS
        log_level(str): set log level whose value among [CRITICAL, ERROR, WARNING, INFO, DEBUG]
        api_keys (List[str] | str | None): Optional list of API keys. Accepts string type as
            a single api_key. Default to None, which means no api key applied.
        ssl (bool): Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.
    """  # noqa E501
    if os.getenv('TM_LOG_LEVEL') is None:
        os.environ['TM_LOG_LEVEL'] = log_level
    logger.setLevel(log_level)

    if allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
        )
    if api_keys is not None:
        if isinstance(api_keys, str):
            api_keys = api_keys.split(',')
        VariableInterface.api_keys = api_keys
    ssl_keyfile, ssl_certfile, http_or_https = None, None, 'http'
    if ssl:
        ssl_keyfile = os.environ['SSL_KEYFILE']
        ssl_certfile = os.environ['SSL_CERTFILE']
        http_or_https = 'https'

    _, pipeline_class = get_task(model_path)

    VariableInterface.async_engine = pipeline_class(
        model_path=model_path,
        model_name=model_name,
        backend=backend,
        backend_config=backend_config,
        chat_template_config=chat_template_config,
        **kwargs)

    for i in range(3):
        print(
            f'HINT:    Please open \033[93m\033[1m{http_or_https}://'
            f'{server_name}:{server_port}\033[0m in a browser for detailed api'
            ' usage!!!')
    uvicorn.run(app=app,
                host=server_name,
                port=server_port,
                log_level='info',
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile)


if __name__ == '__main__':
    import fire

    fire.Fire(serve)
