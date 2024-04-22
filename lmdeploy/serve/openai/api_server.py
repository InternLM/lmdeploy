# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import copy
import os
import time
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

from lmdeploy.archs import get_task
from lmdeploy.messages import (GenerationConfig, PytorchEngineConfig,
                               TurbomindEngineConfig)
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.openai.protocol import (  # noqa: E501
    ChatCompletionRequest, ChatCompletionRequestQos, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatCompletionTokenLogprob, ChatMessage,
    ChoiceLogprobs, CompletionRequest, CompletionRequestQos,
    CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse, DeltaMessage,
    EmbeddingsRequest, EncodeRequest, EncodeResponse, ErrorResponse,
    GenerateRequest, GenerateRequestQos, GenerateResponse, LogProbs, ModelCard,
    ModelList, ModelPermission, TopLogprob, UsageInfo)
from lmdeploy.serve.qos_engine.qos_engine import QosEngine
from lmdeploy.tokenizer import DetokenizeState, Tokenizer
from lmdeploy.utils import get_logger


class VariableInterface:
    """A IO interface maintaining variables."""
    async_engine: AsyncEngine = None
    session_id: int = 0
    api_keys: Optional[List[str]] = None
    qos_engine: QosEngine = None
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


def create_error_response(status: HTTPStatus, message: str):
    """Create error response according to http status and message.

    Args:
        status (HTTPStatus): HTTP status codes and reason phrases
        message (str): error message
    """
    return JSONResponse(
        ErrorResponse(message=message,
                      type='invalid_request_error',
                      code=status.value).model_dump())


async def check_request(request) -> Optional[JSONResponse]:
    """Check if a request is valid."""
    if request.model in get_model_list():
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND, f'The model `{request.model}` does not exist.')
    return ret


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


@app.post('/v1/chat/completions_qos')
async def chat_completions_v1_qos(request: ChatCompletionRequestQos,
                                  raw_request: Request = None):
    """Completion API similar to OpenAI's API.

    Refer to  `https://platform.openai.com/docs/api-reference/chat/create`
    for the API specification.

    The request should be a JSON object with the following fields:
    - model: model name. Available from /v1/models.
    - messages: string prompt or chat history in OpenAI format.
    - temperature (float): to modulate the next token probability
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - n (int): How many chat completion choices to generate for each input
        message. Only support one here.
    - stream: whether to stream the results or not. Default to false.
    - max_tokens (int): output token nums
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty

    Additional arguments supported by LMDeploy:
    - ignore_eos (bool): indicator for ignoring eos
    - user_id (str): for qos; if not specified, will set to "default"

    Currently we do not support the following features:
    - function_call (Users should implement this by themselves)
    - logit_bias (not supported yet)
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    VariableInterface.session_id += 1
    request.session_id = VariableInterface.session_id
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = str(request.session_id)
    created_time = int(time.time())

    if VariableInterface.qos_engine is None:
        return create_error_response(
            HTTPStatus.NOT_FOUND,
            'cannot parse qos engine config, this api is not work')

    result_generator = await VariableInterface.qos_engine.generate_with_qos(
        request)

    if result_generator is None:
        return create_error_response(HTTPStatus.INTERNAL_SERVER_ERROR,
                                     'Failed to generate completions')

    def create_stream_response_json(
        index: int,
        text: str,
        finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(role='assistant', content=text),
            finish_reason=finish_reason,
        )
        response = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        response_json = response.model_dump_json()

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role='assistant'),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                 choices=[choice_data],
                                                 model=model_name)
            data = chunk.model_dump_json(exclude_unset=True)
            yield f'data: {data}\n\n'

        async for res in result_generator:
            response_json = create_stream_response_json(
                index=0,
                text=res.response,
            )
            yield f'data: {response_json}\n\n'
        yield 'data: [DONE]\n\n'

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type='text/event-stream')

    # Non-streaming response
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
    assert final_res is not None
    choices = []
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role='assistant', content=text),
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
        message. Only support one here.
    - stream: whether to stream the results or not. Default to false.
    - max_tokens (int | None): output token nums. Default to None.
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - stop (str | List[str] | None): To stop generating further
        tokens. Only accept stop words that's encoded to one token idex.

    Additional arguments supported by LMDeploy:
    - top_k (int): The number of the highest probability vocabulary
        tokens to keep for top-k-filtering
    - ignore_eos (bool): indicator for ignoring eos
    - skip_special_tokens (bool): Whether or not to remove special tokens
        in the decoding. Default to be True.

    Currently we do not support the following features:
    - function_call (Users should implement this by themselves)
    - logit_bias (not supported yet)
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    VariableInterface.session_id += 1
    request.session_id = VariableInterface.session_id
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    adapter_name = None
    if model_name != VariableInterface.async_engine.model_name:
        adapter_name = model_name  # got a adapter name
    request_id = str(request.session_id)
    created_time = int(time.time())

    if isinstance(request.stop, str):
        request.stop = [request.stop]

    gen_logprobs = None
    if request.logprobs and request.top_logprobs:
        gen_logprobs = request.top_logprobs

    gen_config = GenerationConfig(
        max_new_tokens=request.max_tokens,
        logprobs=gen_logprobs,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos,
        stop_words=request.stop,
        skip_special_tokens=request.skip_special_tokens)

    result_generator = VariableInterface.async_engine.generate(
        request.messages,
        request.session_id,
        gen_config=gen_config,
        stream_response=True,  # always use stream to enable batching
        sequence_start=True,
        sequence_end=True,
        do_preprocess=not isinstance(request.messages,
                                     str),  # text completion for string input
        adapter_name=adapter_name,
    )

    def create_stream_response_json(
            index: int,
            text: str,
            finish_reason: Optional[str] = None,
            logprobs: Optional[LogProbs] = None) -> str:
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
        )
        response_json = response.model_dump_json()

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role='assistant'),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                 choices=[choice_data],
                                                 model=model_name)
            data = chunk.model_dump_json(exclude_unset=True)
            yield f'data: {data}\n\n'

        async for res in result_generator:
            logprobs = None
            if gen_logprobs and res.logprobs:
                logprobs = _create_chat_completion_logprobs(
                    VariableInterface.async_engine.tokenizer, res.token_ids,
                    res.logprobs)

            response_json = create_stream_response_json(
                index=0,
                text=res.response,
                finish_reason=res.finish_reason,
                logprobs=logprobs)
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

    logprobs = None
    if gen_logprobs and len(final_logprobs):
        logprobs = _create_chat_completion_logprobs(
            VariableInterface.async_engine.tokenizer, final_token_ids,
            final_logprobs)

    assert final_res is not None
    choices = []
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role='assistant', content=text),
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


@app.post('/v1/completions_qos')
async def completions_v1_qos(request: CompletionRequestQos,
                             raw_request: Request = None):
    """Completion API similar to OpenAI's API.

    Go to `https://platform.openai.com/docs/api-reference/completions/create`
    for the API specification.

    The request should be a JSON object with the following fields:
    - model (str): model name. Available from /v1/models.
    - prompt (str): the input prompt.
    - suffix (str): The suffix that comes after a completion of inserted text.
    - max_tokens (int): output token nums
    - temperature (float): to modulate the next token probability
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - n (int): How many chat completion choices to generate for each input
        message. Only support one here.
    - stream: whether to stream the results or not. Default to false.
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - user (str): A unique identifier representing your end-user.

    Additional arguments supported by LMDeploy:
    - top_k (int): The number of the highest probability vocabulary
        tokens to keep for top-k-filtering
    - ignore_eos (bool): indicator for ignoring eos
    - user_id (str): for qos; if not specified, will set to "default"

    Currently we do not support the following features:
    - logprobs (not supported yet)
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    VariableInterface.session_id += 1
    request.session_id = VariableInterface.session_id
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = str(request.session_id)
    created_time = int(time.time())
    if isinstance(request.prompt, str):
        request.prompt = [request.prompt]

    if VariableInterface.qos_engine is None:
        return create_error_response(
            HTTPStatus.NOT_FOUND,
            'cannot parse qos engine config, this api is not work')

    generators = await VariableInterface.qos_engine.generate_with_qos(request)

    def create_stream_response_json(
        index: int,
        text: str,
        finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = CompletionResponseStreamChoice(
            index=index,
            text=text,
            finish_reason=finish_reason,
        )
        response = CompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        response_json = response.model_dump_json()

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for generator in generators:
            for i in range(request.n):
                choice_data = CompletionResponseStreamChoice(
                    index=i,
                    text='',
                    finish_reason=None,
                )
                chunk = CompletionStreamResponse(id=request_id,
                                                 choices=[choice_data],
                                                 model=model_name)
                data = chunk.model_dump_json(exclude_unset=True)
                yield f'data: {data}\n\n'

            async for res in generator:
                response_json = create_stream_response_json(
                    index=0,
                    text=res.response,
                )
                yield f'data: {response_json}\n\n'
        yield 'data: [DONE]\n\n'

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type='text/event-stream')

    # Non-streaming response
    usage = UsageInfo()
    choices = []

    async def _inner_call(i, generator):
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
        assert final_res is not None
        choice_data = CompletionResponseChoice(
            index=0,
            text=text,
            finish_reason=final_res.finish_reason,
        )
        choices.append(choice_data)

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
        message. Only support one here.
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
    VariableInterface.session_id += 1
    request.session_id = VariableInterface.session_id
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret

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
    gen_config = GenerationConfig(
        max_new_tokens=request.max_tokens if request.max_tokens else 512,
        logprobs=request.logprobs,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos,
        stop_words=request.stop,
        skip_special_tokens=request.skip_special_tokens)
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

    def create_stream_response_json(
            index: int,
            text: str,
            finish_reason: Optional[str] = None,
            logprobs: Optional[LogProbs] = None) -> str:
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
        )
        response_json = response.model_dump_json()

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for generator in generators:
            for i in range(request.n):
                choice_data = CompletionResponseStreamChoice(
                    index=i,
                    text='',
                    finish_reason=None,
                )
                chunk = CompletionStreamResponse(id=request_id,
                                                 choices=[choice_data],
                                                 model=model_name)
                data = chunk.model_dump_json(exclude_unset=True)
                yield f'data: {data}\n\n'

            offset = 0
            all_token_ids = []
            state = DetokenizeState()
            async for res in generator:
                logprobs = None
                if request.logprobs and res.logprobs:
                    logprobs, offset, all_token_ids, state = _create_completion_logprobs(  # noqa E501
                        VariableInterface.async_engine.tokenizer,
                        res.token_ids, res.logprobs,
                        gen_config.skip_special_tokens, offset, all_token_ids,
                        state)
                response_json = create_stream_response_json(
                    index=0,
                    text=res.response,
                    finish_reason=res.finish_reason,
                    logprobs=logprobs)
                yield f'data: {response_json}\n\n'
        yield 'data: [DONE]\n\n'

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type='text/event-stream')

    # Non-streaming response
    usage = UsageInfo()
    choices = []

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
            index=0,
            text=text,
            finish_reason=final_res.finish_reason,
            logprobs=logprobs,
        )
        choices.append(choice_data)

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


@app.post('/v1/chat/interactive_qos')
async def chat_interactive_v1_qos(request: GenerateRequestQos,
                                  raw_request: Request = None):
    """Generate completion for the request.

    - On interactive mode, the chat history is kept on the server. Please set
    `interactive_mode = True`.
    - On normal mode, no chat history is kept on the server. Set
    `interactive_mode = False`.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - session_id: determine which instance will be called. If not specified
        with a value other than -1, using random value directly.
    - interactive_mode (bool): turn on interactive mode or not. On interactive
        mode, session history is kept on the server (and vice versa).
    - stream: whether to stream the results or not.
    - stop: whether to stop the session response or not.
    - request_output_len (int): output token nums
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - top_k (int): The number of the highest probability vocabulary
        tokens to keep for top-k-filtering
    - temperature (float): to modulate the next token probability
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - ignore_eos (bool): indicator for ignoring eos
    - user_id (str): for qos; if not specified, will set to "default"
    """
    if request.session_id == -1:
        VariableInterface.session_id += 1
        request.session_id = VariableInterface.session_id

    if VariableInterface.qos_engine is None:
        return create_error_response(
            HTTPStatus.NOT_FOUND,
            'cannot parse qos engine config, this api is not work')

    generation = await VariableInterface.qos_engine.generate_with_qos(request)

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
        tokens = 0
        finish_reason = None
        async for out in generation:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await VariableInterface.qos_engine.stop_session(
                    request.session_id)
                return create_error_response(HTTPStatus.BAD_REQUEST,
                                             'Client disconnected')
            text += out.response
            tokens = out.generate_token_len
            finish_reason = out.finish_reason
        ret = {'text': text, 'tokens': tokens, 'finish_reason': finish_reason}
        return JSONResponse(ret)


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
    if request.session_id == -1:
        VariableInterface.session_id += 1
        request.session_id = VariableInterface.session_id

    async_engine = VariableInterface.async_engine
    sequence_start = async_engine.id2step.get(str(request.session_id), 0) == 0
    sequence_end = not request.interactive_mode
    if isinstance(request.stop, str):
        request.stop = [request.stop]

    gen_config = GenerationConfig(
        max_new_tokens=request.request_output_len,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos,
        stop_words=request.stop,
        skip_special_tokens=request.skip_special_tokens)
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
          tp: int = 1,
          allow_origins: List[str] = ['*'],
          allow_credentials: bool = True,
          allow_methods: List[str] = ['*'],
          allow_headers: List[str] = ['*'],
          log_level: str = 'ERROR',
          api_keys: Optional[Union[List[str], str]] = None,
          ssl: bool = False,
          qos_config_path: str = '',
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
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "InternLM/internlm-chat-7b"
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
        qos_config_path (str): qos policy config path
    """ # noqa E501
    if os.getenv('TM_LOG_LEVEL') is None:
        os.environ['TM_LOG_LEVEL'] = log_level
    logger = get_logger('lmdeploy')
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

    pipeline_type, pipeline_class = get_task(model_path)

    VariableInterface.async_engine = pipeline_class(
        model_path=model_path,
        model_name=model_name,
        backend=backend,
        backend_config=backend_config,
        chat_template_config=chat_template_config,
        tp=tp,
        **kwargs)

    if qos_config_path:
        try:
            with open(qos_config_path, 'r') as file:
                qos_config_str = file.read()
                VariableInterface.qos_engine = QosEngine(
                    qos_tag=qos_config_str,
                    engine=VariableInterface.async_engine,
                    **kwargs)
                VariableInterface.qos_engine.start()
        except FileNotFoundError:
            VariableInterface.qos_engine = None
    else:
        # hide qos functions if not applied
        for i in range(len(app.router.routes)):
            if 'qos' in app.router.routes[i].path:
                app.router.routes[i].include_in_schema = False

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
