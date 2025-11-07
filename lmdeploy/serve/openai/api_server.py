# Copyright (c) OpenMMLab. All rights reserved.
# yapf: disable
import asyncio
import copy
import json
import os
import re
import time
from contextlib import asynccontextmanager
from functools import partial
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Mount

from lmdeploy.archs import get_task
from lmdeploy.messages import GenerationConfig, LogitsProcessor, PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.metrics.metrics_processor import metrics_processor
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.pytorch.disagg.config import DistServeEngineConfig
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeCacheFreeRequest, DistServeConnectionRequest,
                                                   DistServeDropConnectionRequest, DistServeInitRequest,
                                                   MigrationRequest)
from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.openai.harmony_utils import GptOssChatParser
from lmdeploy.serve.openai.protocol import ChatCompletionResponse  # noqa: E501
from lmdeploy.serve.openai.protocol import (AbortRequest, ChatCompletionRequest, ChatCompletionResponseChoice,
                                            ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse,
                                            ChatCompletionTokenLogprob, ChatMessage, ChoiceLogprobs, CompletionRequest,
                                            CompletionResponse, CompletionResponseChoice,
                                            CompletionResponseStreamChoice, CompletionStreamResponse, DeltaMessage,
                                            EmbeddingsRequest, EncodeRequest, EncodeResponse, ErrorResponse,
                                            GenerateReqInput, GenerateReqMetaOutput, GenerateReqOutput, GenerateRequest,
                                            LogProbs, ModelCard, ModelList, ModelPermission, PoolingRequest,
                                            PoolingResponse, TopLogprob, UpdateParamsRequest, UsageInfo)
from lmdeploy.serve.openai.reasoning_parser.reasoning_parser import ReasoningParser, ReasoningParserManager
from lmdeploy.serve.openai.tool_parser.tool_parser import ToolParser, ToolParserManager
from lmdeploy.tokenizer import DetokenizeState, Tokenizer
from lmdeploy.utils import get_logger

# yapf: enable

logger = get_logger('lmdeploy')


class VariableInterface:
    """A IO interface maintaining variables."""
    async_engine: AsyncEngine = None
    session_id: int = 0
    api_keys: Optional[List[str]] = None
    request_hosts = []
    # following are for registering to proxy server
    proxy_url: Optional[str] = None
    api_server_url: Optional[str] = None
    # following are for reasoning parsers
    reasoning_parser: Optional[ReasoningParser] = None
    # following is for tool parsers
    tool_parser: Optional[ToolParser] = None
    allow_terminate_by_client: bool = False
    enable_abort_handling: bool = False


router = APIRouter()
get_bearer_token = HTTPBearer(auto_error=False)


async def check_api_key(auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token), ) -> str:
    """Check if client provide valid api key.

    Adopted from https://github.com/lm-sys/FastChat/blob/v0.2.35/fastchat/serve/openai_api_server.py#L108-L127
    """  # noqa
    if VariableInterface.api_keys:
        if auth is None or (token := auth.credentials) not in VariableInterface.api_keys:
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

    If it is a slora serving. The model list would be [model_name, adapter_name1, adapter_name2, ...]
    """
    model_names = [VariableInterface.async_engine.model_name]
    cfg = VariableInterface.async_engine.backend_config
    model_names += getattr(cfg, 'adapters', None) or []
    return model_names


@router.get('/v1/models', dependencies=[Depends(check_api_key)])
def available_models():
    """Show available models."""
    model_cards = []
    for model_name in get_model_list():
        model_cards.append(ModelCard(id=model_name, root=model_name, permission=[ModelPermission()]))
    return ModelList(data=model_cards)


def create_error_response(status: HTTPStatus, message: str, error_type='invalid_request_error'):
    """Create error response according to http status and message.

    Args:
        status (HTTPStatus): HTTP status codes and reason phrases
        message (str): error message
        error_type (str): error type
    """
    return JSONResponse(ErrorResponse(message=message, type=error_type, code=status.value).model_dump(),
                        status_code=status.value)


async def check_request(request) -> Optional[JSONResponse]:
    """Check if a request is valid."""
    if hasattr(request, 'model') and request.model not in get_model_list():
        return create_error_response(HTTPStatus.NOT_FOUND, f'The model {request.model!r} does not exist.')
    if hasattr(request, 'n') and request.n <= 0:
        return create_error_response(HTTPStatus.BAD_REQUEST, f'The n {request.n!r} must be a positive int.')
    if hasattr(request, 'top_p') and not (request.top_p > 0 and request.top_p <= 1):
        return create_error_response(HTTPStatus.BAD_REQUEST, f'The top_p {request.top_p!r} must be in (0, 1].')
    if hasattr(request, 'top_k') and request.top_k < 0:
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     f'The top_k {request.top_k!r} cannot be a negative integer.')
    if hasattr(request, 'temperature') and not (request.temperature <= 2 and request.temperature >= 0):
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     f'The temperature {request.temperature!r} must be in [0, 2]')
    return


def _create_completion_logprobs(tokenizer: Tokenizer,
                                token_ids: List[int] = None,
                                logprobs: List[Dict[int, float]] = None,
                                skip_special_tokens: bool = True,
                                offset: int = 0,
                                all_token_ids: List[int] = None,
                                state: DetokenizeState = None,
                                spaces_between_special_tokens: bool = True):
    """Create openai LogProbs for completion.

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
        spaces_between_special_tokens (bool): Whether or not to add spaces
            around special tokens. The behavior of Fast tokenizers is to have
            this to False. This is setup to True in slow tokenizers.
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
                skip_special_tokens=skip_special_tokens,
                spaces_between_special_tokens=spaces_between_special_tokens)
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
    """Create openai LogProbs for chat.completion.

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
        item = ChatCompletionTokenLogprob(token='', bytes=[], logprob=0.0, top_logprobs=[])
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
                item.top_logprobs.append(TopLogprob(token=token, bytes=_bytes, logprob=prob))
        content.append(item)
    return ChoiceLogprobs(content=content)


@router.get('/health')
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@router.get('/terminate')
async def terminate():
    """Terminate server."""
    import signal

    if not VariableInterface.allow_terminate_by_client:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            'The server can not be terminated. Please add --allow-terminate-by-client when start the server.')
    os.kill(os.getpid(), signal.SIGTERM)
    return Response(status_code=200)


# modified from https://github.com/vllm-project/vllm/blob/v0.5.4/vllm/entrypoints/openai/logits_processors.py#L51  # noqa
def logit_bias_logits_processor(logit_bias: Union[Dict[int, float], Dict[str, float]], tokenizer) -> LogitsProcessor:
    try:
        # Convert token_id to integer
        # Clamp the bias between -100 and 100 per OpenAI API spec
        clamped_logit_bias: Dict[int, float] = {
            int(token_id): min(100.0, max(-100.0, bias))
            for token_id, bias in logit_bias.items()
        }
    except ValueError as exc:
        raise ValueError('Found token_id in logit_bias that is not '
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


@router.post('/v1/chat/completions', dependencies=[Depends(check_api_key)])
async def chat_completions_v1(request: ChatCompletionRequest, raw_request: Request = None):
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
    - stream_options: Options for streaming response. Only set this when you
        set stream: true.
    - max_completion_tokens (int | None): output token nums. Default to None.
    - max_tokens (int | None): output token nums. Default to None.
        Deprecated: Use max_completion_tokens instead.
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - stop (str | List[str] | None): To stop generating further
        tokens. Only accept stop words that's encoded to one token idex.
    - response_format (Dict | None): To generate response according to given
        schema. Examples: `{"type": "json_schema", "json_schema": {"name":
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
    - spaces_between_special_tokens (bool): Whether or not to add spaces
        around special tokens. The behavior of Fast tokenizers is to have
        this to False. This is setup to True in slow tokenizers.
    - min_new_tokens (int): To generate at least numbers of tokens.
    - min_p (float): Minimum token probability, which will be scaled by the
        probability of the most likely token. It must be a value between
        0 and 1. Typical values are in the 0.01-0.2 range, comparably
        selective as setting `top_p` in the 0.99-0.8 range (use the
        opposite of normal `top_p` values)

    Currently we do not support the following features:
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    json_request = await raw_request.json()
    migration_request = json_request.pop('migration_request', None)
    with_cache = json_request.pop('with_cache', False)
    preserve_cache = json_request.pop('preserve_cache', False)
    if migration_request:
        migration_request = MigrationRequest.model_validate(migration_request)

    if request.session_id == -1:
        VariableInterface.session_id += 1
        request.session_id = VariableInterface.session_id
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret
    if VariableInterface.async_engine.id2step.get(request.session_id, 0) != 0:
        return create_error_response(HTTPStatus.BAD_REQUEST, f'The session_id {request.session_id!r} is occupied.')

    model_name = request.model
    adapter_name = None
    if model_name != VariableInterface.async_engine.model_name:
        adapter_name = model_name  # got a adapter name
    request_id = str(request.session_id)
    created_time = int(time.time())
    gpt_oss_parser = None
    if VariableInterface.async_engine.arch == 'GptOssForCausalLM':
        gpt_oss_parser = GptOssChatParser()

    if isinstance(request.stop, str):
        request.stop = [request.stop]

    gen_logprobs, logits_processors = None, None
    if request.logprobs and request.top_logprobs:
        gen_logprobs = request.top_logprobs
    response_format = None
    if request.response_format and request.response_format.type != 'text':
        response_format = request.response_format.model_dump()

    if request.logit_bias is not None:
        try:
            logits_processors = [
                logit_bias_logits_processor(request.logit_bias, VariableInterface.async_engine.tokenizer.model)
            ]
        except Exception as e:
            return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    random_seed = request.seed if request.seed else None
    max_new_tokens = (request.max_completion_tokens if request.max_completion_tokens else request.max_tokens)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        logprobs=gen_logprobs,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos,
        stop_words=request.stop,
        include_stop_str_in_output=request.include_stop_str_in_output,
        skip_special_tokens=request.skip_special_tokens,
        response_format=response_format,
        logits_processors=logits_processors,
        min_new_tokens=request.min_new_tokens,
        min_p=request.min_p,
        random_seed=random_seed,
        spaces_between_special_tokens=request.spaces_between_special_tokens,
        migration_request=migration_request,
        with_cache=with_cache,
        preserve_cache=preserve_cache,
    )

    tools = None
    if request.tools and request.tool_choice != 'none':
        gen_config.skip_special_tokens = False
        # internlm2 only uses contents inside function regardless of 'type'
        if not isinstance(request.tool_choice, str):
            if gpt_oss_parser:
                tools = [
                    item.model_dump() for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
            else:
                tools = [
                    item.function.model_dump() for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
        else:
            if gpt_oss_parser:
                tools = [item.model_dump() for item in request.tools]
            else:
                tools = [item.function.model_dump() for item in request.tools]
    # text completion for string input
    do_preprocess = False if isinstance(request.messages, str) else request.do_preprocess
    result_generator = VariableInterface.async_engine.generate(
        request.messages,
        request.session_id,
        gen_config=gen_config,
        tools=tools,
        reasoning_effort=request.reasoning_effort,
        stream_response=True,  # always use stream to enable batching
        sequence_start=True,
        sequence_end=True,
        do_preprocess=do_preprocess,
        adapter_name=adapter_name,
        enable_thinking=request.enable_thinking,
    )

    def create_stream_response_json(index: int,
                                    delta_message: DeltaMessage,
                                    finish_reason: Optional[str] = None,
                                    logprobs: Optional[LogProbs] = None,
                                    usage: Optional[UsageInfo] = None) -> str:
        choice_data = ChatCompletionResponseStreamChoice(index=index,
                                                         delta=delta_message,
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
        previous_text = ''
        current_text = ''
        previous_token_ids = []
        current_token_ids = []
        delta_token_ids = []
        has_parser = VariableInterface.tool_parser is not None or VariableInterface.reasoning_parser is not None
        streaming_tools = False
        async for res in result_generator:
            logprobs, usage = None, None
            if gen_logprobs and res.logprobs:
                logprobs = _create_chat_completion_logprobs(VariableInterface.async_engine.tokenizer, res.token_ids,
                                                            res.logprobs)
            # Only stream chunk `usage` in the final chunk according to OpenAI API spec
            if (res.finish_reason and request.stream_options and request.stream_options.include_usage):
                total_tokens = sum([res.history_token_len, res.input_token_len, res.generate_token_len])
                usage = UsageInfo(
                    prompt_tokens=res.input_token_len,
                    completion_tokens=res.generate_token_len,
                    total_tokens=total_tokens,
                )

            delta_token_ids = res.token_ids if res.token_ids is not None else []
            if gpt_oss_parser:
                delta_message = gpt_oss_parser.parse_streaming(res.token_ids)
                if res.finish_reason == 'stop' and len(delta_message.tool_calls) > 0:
                    res.finish_reason = 'tool_calls'
            else:
                delta_message = DeltaMessage(role='assistant', content=res.response)
                if has_parser:
                    current_text = current_text + res.response
                    current_token_ids = current_token_ids + delta_token_ids
                if request.tool_choice != 'none' and VariableInterface.tool_parser is not None:
                    if res.finish_reason == 'stop' and streaming_tools is True:
                        res.finish_reason = 'tool_calls'
                    tool_delta = VariableInterface.tool_parser.extract_tool_calls_streaming(
                        previous_text=previous_text,
                        current_text=current_text,
                        delta_text=delta_message.content,
                        previous_token_ids=previous_token_ids,
                        current_token_ids=current_token_ids,
                        delta_token_ids=delta_token_ids,
                        request=request)
                    if tool_delta is not None:
                        delta_message.tool_calls = tool_delta.tool_calls
                        delta_message.content = tool_delta.content
                        if isinstance(tool_delta.tool_calls, List) and len(tool_delta.tool_calls):
                            streaming_tools = True
                elif (request.tool_choice != 'none' and request.tools is not None
                      and VariableInterface.tool_parser is None):
                    logger.error('Please launch the api_server with --tool-call-parser if you want to use tool.')

                if VariableInterface.reasoning_parser is not None and request.enable_thinking is not False:
                    reasoning_delta = VariableInterface.reasoning_parser.extract_reasoning_content_streaming(
                        previous_text=previous_text,
                        current_text=current_text,
                        delta_text=delta_message.content or '',
                        previous_token_ids=previous_token_ids,
                        current_token_ids=current_token_ids,
                        delta_token_ids=delta_token_ids)
                    if reasoning_delta is not None:
                        delta_message.reasoning_content = reasoning_delta.reasoning_content
                        delta_message.content = reasoning_delta.content
                if has_parser:
                    previous_text = current_text
                    previous_token_ids = current_token_ids
            if request.return_token_ids:
                delta_message.gen_tokens = delta_token_ids
            response_json = create_stream_response_json(index=0,
                                                        delta_message=delta_message,
                                                        finish_reason=res.finish_reason,
                                                        logprobs=logprobs,
                                                        usage=usage)
            if res.cache_block_ids is not None:
                response_json['cache_block_ids'] = res.cache_block_ids
                response_json['remote_token_ids'] = res.token_ids
            yield f'data: {response_json}\n\n'
        yield 'data: [DONE]\n\n'

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(), media_type='text/event-stream')

    # Non-streaming response
    final_logprobs = []
    final_token_ids = []
    final_res = None
    text = ''
    cache_block_ids = []
    remote_token_ids = []
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await VariableInterface.async_engine.stop_session(request.session_id)
            return create_error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected')
        final_res = res
        text += res.response
        if res.token_ids:
            final_token_ids.extend(res.token_ids)
        if res.logprobs:
            final_logprobs.extend(res.logprobs)
        cache_block_ids.append(res.cache_block_ids)
        remote_token_ids.append(res.token_ids)

    if gpt_oss_parser:
        message = gpt_oss_parser.parse_full(final_token_ids)
        if final_res.finish_reason == 'stop' and len(message.tool_calls) > 0:
            final_res.finish_reason = 'tool_calls'
    else:
        tool_calls = None
        reasoning_content = None
        if request.tool_choice != 'none' and VariableInterface.tool_parser is not None:
            try:
                tool_call_info = VariableInterface.tool_parser.extract_tool_calls(text, request=request)
                text, tool_calls = tool_call_info.content, tool_call_info.tool_calls
                if isinstance(tool_calls, List) and len(tool_calls):
                    if final_res.finish_reason == 'stop':
                        final_res.finish_reason = 'tool_calls'

            except Exception as e:
                logger.error(f'Failed to parse {text}. Exception: {e}.')
                return create_error_response(HTTPStatus.BAD_REQUEST, 'Failed to parse fc related info to json format!')
        elif request.tool_choice != 'none' and request.tools is not None and VariableInterface.tool_parser is None:
            logger.error('Please launch the api_server with --tool-call-parser if you want to use tool.')

        if VariableInterface.reasoning_parser is not None and request.enable_thinking is not False:
            reasoning_content, text = VariableInterface.reasoning_parser.extract_reasoning_content(text, request)

        message = ChatMessage(role='assistant',
                              content=text,
                              tool_calls=tool_calls,
                              reasoning_content=reasoning_content)

    logprobs = None
    if gen_logprobs and len(final_logprobs):
        logprobs = _create_chat_completion_logprobs(VariableInterface.async_engine.tokenizer, final_token_ids,
                                                    final_logprobs)

    assert final_res is not None
    choices = []
    if request.return_token_ids:
        message.gen_tokens = final_token_ids
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        logprobs=logprobs,
        finish_reason=final_res.finish_reason,
    )
    choices.append(choice_data)

    if with_cache:
        cache_block_ids = cache_block_ids[0]
        remote_token_ids = [remote_token_ids[0][-1]]

    total_tokens = sum([final_res.history_token_len, final_res.input_token_len, final_res.generate_token_len])
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
    ).model_dump()

    if with_cache:
        response['cache_block_ids'] = cache_block_ids
        response['remote_token_ids'] = remote_token_ids

    return response


@router.post('/v1/completions', dependencies=[Depends(check_api_key)])
async def completions_v1(request: CompletionRequest, raw_request: Request = None):
    """Completion API similar to OpenAI's API.

    Go to `https://platform.openai.com/docs/api-reference/completions/create`
    for the API specification.

    The request should be a JSON object with the following fields:
    - model (str): model name. Available from /v1/models.
    - prompt (str): the input prompt.
    - suffix (str): The suffix that comes after a completion of inserted text.
    - max_completion_tokens (int | None): output token nums. Default to None.
    - max_tokens (int | None): output token nums. Default to 16.
        Deprecated: Use max_completion_tokens instead.
    - temperature (float): to modulate the next token probability
    - top_p (float): If set to float < 1, only the smallest set of most
        probable tokens with probabilities that add up to top_p or higher
        are kept for generation.
    - n (int): How many chat completion choices to generate for each input
        message. **Only support one here**.
    - stream: whether to stream the results or not. Default to false.
    - stream_options: Options for streaming response. Only set this when you
        set stream: true.
    - repetition_penalty (float): The parameter for repetition penalty.
        1.0 means no penalty
    - user (str): A unique identifier representing your end-user.
    - stop (str | List[str] | None): To stop generating further
        tokens. Only accept stop words that's encoded to one token idex.

    Additional arguments supported by LMDeploy:
    - ignore_eos (bool): indicator for ignoring eos
    - skip_special_tokens (bool): Whether or not to remove special tokens
        in the decoding. Default to be True.
    - spaces_between_special_tokens (bool): Whether or not to add spaces
        around special tokens. The behavior of Fast tokenizers is to have
        this to False. This is setup to True in slow tokenizers.
    - top_k (int): The number of the highest probability vocabulary
        tokens to keep for top-k-filtering
    - min_p (float): Minimum token probability, which will be scaled by the
        probability of the most likely token. It must be a value between
        0 and 1. Typical values are in the 0.01-0.2 range, comparably
        selective as setting `top_p` in the 0.99-0.8 range (use the
        opposite of normal `top_p` values)

    Currently we do not support the following features:
    - logprobs (not supported yet)
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    json_request = await raw_request.json()
    migration_request = json_request.pop('migration_request', None)
    with_cache = json_request.pop('with_cache', False)
    preserve_cache = json_request.pop('preserve_cache', False)
    if migration_request:
        migration_request = MigrationRequest.model_validate(migration_request)

    if request.session_id == -1:
        VariableInterface.session_id += 1
        request.session_id = VariableInterface.session_id
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret
    if VariableInterface.async_engine.id2step.get(request.session_id, 0) != 0:
        return create_error_response(HTTPStatus.BAD_REQUEST, f'The session_id {request.session_id!r} is occupied.')

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
    max_new_tokens = (request.max_completion_tokens if request.max_completion_tokens else request.max_tokens)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        logprobs=request.logprobs,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos,
        stop_words=request.stop,
        skip_special_tokens=request.skip_special_tokens,
        min_p=request.min_p,
        random_seed=random_seed,
        spaces_between_special_tokens=request.spaces_between_special_tokens,
        migration_request=migration_request,
        with_cache=with_cache,
        preserve_cache=preserve_cache,
    )
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
                                    gen_tokens: Optional[List[int]] = None,
                                    usage: Optional[UsageInfo] = None) -> str:
        choice_data = CompletionResponseStreamChoice(index=index,
                                                     text=text,
                                                     gen_tokens=gen_tokens,
                                                     finish_reason=finish_reason,
                                                     logprobs=logprobs)
        response = CompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
            usage=usage,
        )
        response_json = response.model_dump()
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
                        VariableInterface.async_engine.tokenizer, res.token_ids, res.logprobs,
                        gen_config.skip_special_tokens, offset, all_token_ids, state,
                        gen_config.spaces_between_special_tokens)
                # Only stream chunk `usage` in the final chunk according to OpenAI API spec
                if (res.finish_reason and request.stream_options and request.stream_options.include_usage):
                    final_res = res
                    total_tokens = sum(
                        [final_res.history_token_len, final_res.input_token_len, final_res.generate_token_len])
                    usage = UsageInfo(
                        prompt_tokens=final_res.input_token_len,
                        completion_tokens=final_res.generate_token_len,
                        total_tokens=total_tokens,
                    )
                gen_tokens = None
                if request.return_token_ids:
                    gen_tokens = res.token_ids or []
                response_json = create_stream_response_json(index=0,
                                                            text=res.response,
                                                            gen_tokens=gen_tokens,
                                                            finish_reason=res.finish_reason,
                                                            logprobs=logprobs,
                                                            usage=usage)
                if res.cache_block_ids is not None:
                    response_json['cache_block_ids'] = res.cache_block_ids
                    response_json['remote_token_ids'] = res.token_ids
                yield f'data: {json.dumps(response_json)}\n\n'
        yield 'data: [DONE]\n\n'

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(), media_type='text/event-stream')

    # Non-streaming response
    usage = UsageInfo()
    choices = [None] * len(generators)
    cache_block_ids = []
    remote_token_ids = []

    async def _inner_call(i, generator):
        nonlocal cache_block_ids, remote_token_ids
        final_logprobs = []
        final_token_ids = []
        final_res = None
        text = ''
        async for res in generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await VariableInterface.async_engine.stop_session(request.session_id)
                return create_error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected')
            final_res = res
            text += res.response
            cache_block_ids.append(res.cache_block_ids)
            remote_token_ids.append(res.token_ids)
            if res.token_ids:
                final_token_ids.extend(res.token_ids)
            if res.logprobs:
                final_logprobs.extend(res.logprobs)

        logprobs = None
        if request.logprobs and len(final_logprobs):
            logprobs, _, _, _ = _create_completion_logprobs(
                VariableInterface.async_engine.tokenizer,
                final_token_ids,
                final_logprobs,
                gen_config.skip_special_tokens,
                spaces_between_special_tokens=gen_config.spaces_between_special_tokens)

        assert final_res is not None
        choice_data = CompletionResponseChoice(index=i,
                                               text=text,
                                               finish_reason=final_res.finish_reason,
                                               logprobs=logprobs,
                                               gen_tokens=final_token_ids if request.return_token_ids else None)
        choices[i] = choice_data

        if with_cache:
            cache_block_ids = cache_block_ids[0]
            remote_token_ids = [remote_token_ids[0][-1]]

        total_tokens = sum([final_res.history_token_len, final_res.input_token_len, final_res.generate_token_len])
        usage.prompt_tokens += final_res.input_token_len
        usage.completion_tokens += final_res.generate_token_len
        usage.total_tokens += total_tokens

    await asyncio.gather(*[_inner_call(i, generators[i]) for i in range(len(generators))])

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


@router.post('/generate', dependencies=[Depends(check_api_key)])
async def generate(request: GenerateReqInput, raw_request: Request = None):

    if request.session_id == -1:
        VariableInterface.session_id += 1
        request.session_id = VariableInterface.session_id
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret
    if VariableInterface.async_engine.id2step.get(request.session_id, 0) != 0:
        return create_error_response(HTTPStatus.BAD_REQUEST, f'The session_id `{request.session_id}` is occupied.')
    if (request.prompt is not None) ^ (request.input_ids is None):
        return create_error_response(HTTPStatus.BAD_REQUEST, 'You must specify exactly one of prompt or input_ids')

    prompt = request.prompt
    input_ids = request.input_ids
    image_data = request.image_data
    if image_data is not None:
        # convert to openai format
        image_input = []
        if not isinstance(image_data, List):
            image_data = [image_data]
        for img in image_data:
            if isinstance(img, str):
                image_input.append(dict(type='image_url', image_url=dict(url=img)))
            else:
                image_input.append(dict(type='image_url', image_url=img))
        text_input = dict(type='text', text=prompt if prompt else input_ids)
        prompt = [dict(role='user', content=[text_input] + image_input)]
        input_ids = None

    gen_config = GenerationConfig(
        max_new_tokens=request.max_tokens,
        do_sample=True,
        logprobs=1 if request.return_logprob else None,
        top_k=request.top_k,
        top_p=request.top_p,
        min_p=request.min_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos,
        stop_words=request.stop,
        stop_token_ids=request.stop_token_ids,
        skip_special_tokens=request.skip_special_tokens,
        spaces_between_special_tokens=request.spaces_between_special_tokens,
        include_stop_str_in_output=request.include_stop_str_in_output,
    )

    result_generator = VariableInterface.async_engine.generate(
        messages=prompt,
        session_id=request.session_id,
        input_ids=input_ids,
        gen_config=gen_config,
        stream_response=True,  # always use stream to enable batching
        sequence_start=True,
        sequence_end=True,
        do_preprocess=False,
    )

    def create_generate_response_json(res, text, output_ids, logprobs, finish_reason):
        meta = GenerateReqMetaOutput(finish_reason=dict(type=finish_reason) if finish_reason else None,
                                     output_token_logprobs=logprobs or None,
                                     prompt_tokens=res.input_token_len,
                                     completion_tokens=res.generate_token_len)
        response = GenerateReqOutput(text=text, output_ids=output_ids, meta_info=meta)
        return response.model_dump_json()

    async def generate_stream_generator():
        async for res in result_generator:
            text = res.response or ''
            output_ids = res.token_ids
            logprobs = []
            if res.logprobs:
                for tok, tok_logprobs in zip(res.token_ids, res.logprobs):
                    logprobs.append((tok_logprobs[tok], tok))
            response_json = create_generate_response_json(res, text, output_ids, logprobs, res.finish_reason)
            yield f'data: {response_json}\n\n'
        yield 'data: [DONE]\n\n'

    if request.stream:
        return StreamingResponse(generate_stream_generator(), media_type='text/event-stream')

    response = None

    async def _inner_call():
        text = ''
        output_ids = []
        logprobs = []
        async for res in result_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await VariableInterface.async_engine.stop_session(request.session_id)
                return create_error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected')
            text += res.response or ''
            output_ids.extend(res.token_ids or [])
            if res.logprobs:
                for tok, tok_logprobs in zip(res.token_ids, res.logprobs):
                    logprobs.append((tok_logprobs[tok], tok))
        nonlocal response
        meta = GenerateReqMetaOutput(finish_reason=dict(type=res.finish_reason) if res.finish_reason else None,
                                     output_token_logprobs=logprobs or None,
                                     prompt_tokens=res.input_token_len,
                                     completion_tokens=res.generate_token_len)
        response = GenerateReqOutput(text=text, output_ids=output_ids, meta_info=meta)

    await _inner_call()
    return response


@router.post('/v1/embeddings', tags=['unsupported'])
async def create_embeddings(request: EmbeddingsRequest, raw_request: Request = None):
    """Creates embeddings for the text."""
    return create_error_response(HTTPStatus.BAD_REQUEST, 'Unsupported by turbomind.')


@router.post('/v1/encode', dependencies=[Depends(check_api_key)])
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
            prompt = VariableInterface.async_engine.chat_template.get_prompt(prompt, sequence_start=add_bos)
        input_ids = VariableInterface.async_engine.tokenizer.encode(prompt, add_bos=add_bos)
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


@router.post('/pooling')
async def pooling(request: PoolingRequest, raw_request: Request = None):
    """Pooling prompts for reward model.

    In vLLM documentation, https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#pooling-api_1,
    the input format of Pooling API is the same as Embeddings API.

    Go to https://platform.openai.com/docs/api-reference/embeddings/create
    for the Embeddings API specification.

    The request should be a JSON object with the following fields:
    - model (str): model name. Available from /v1/models.
    - input (List[int] | List[List[int]] | str | List[str]): input text to be embed
    """

    async_engine = VariableInterface.async_engine

    request_input = request.input
    model_name = request.model or async_engine.model_name

    # Normalize all inputs to be a batch (List[List[int]])
    if isinstance(request_input, str):
        input_ids = [async_engine.tokenizer.encode(request_input)]
    elif isinstance(request_input, List):
        if not request_input:
            return create_error_response(HTTPStatus.BAD_REQUEST, 'Input list cannot be empty.')
        if isinstance(request_input[0], str):  # List[str]
            input_ids = [async_engine.tokenizer.encode(p) for p in request_input]
        elif isinstance(request_input[0], int):  # List[int]
            input_ids = [request_input]
        elif isinstance(request_input[0], List):  # List[List[int]]
            input_ids = request_input
        else:
            return create_error_response(HTTPStatus.BAD_REQUEST, 'Input list contains an invalid type.')
    else:
        return create_error_response(HTTPStatus.BAD_REQUEST, 'Input must be a string or a list.')

    batch_scores = await async_engine._async_get_reward_score(input_ids)
    prompt_tokens = sum(len(ids) for ids in input_ids)
    usage = UsageInfo(prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens)

    data = []
    for i, score in enumerate(batch_scores):
        data.append({
            'index': i,
            'object': 'pooling',
            'data': score,
        })

    response = PoolingResponse(model=model_name, data=data, usage=usage)
    return response.model_dump()


@router.post('/update_weights', dependencies=[Depends(check_api_key)])
def update_params(request: UpdateParamsRequest, raw_request: Request = None):
    """Update weights for the model."""
    VariableInterface.async_engine.engine.update_params(request)
    return JSONResponse(content=None)


@router.post('/sleep', dependencies=[Depends(check_api_key)])
async def sleep(raw_request: Request = None):
    level = raw_request.query_params.get('level', '1')
    VariableInterface.async_engine.sleep(int(level))
    return Response(status_code=200)


@router.post('/wakeup', dependencies=[Depends(check_api_key)])
async def wakeup(raw_request: Request = None):
    tags = raw_request.query_params.getlist('tags')
    tags = tags or None
    VariableInterface.async_engine.wakeup(tags)
    return Response(status_code=200)


@router.get('/is_sleeping', dependencies=[Depends(check_api_key)])
async def is_sleeping(raw_request: Request = None):
    is_sleeping = VariableInterface.async_engine.is_sleeping
    return JSONResponse(content={'is_sleeping': is_sleeping})


""" PD Disaggregation API Begin """


@router.get('/distserve/engine_info')
async def engine_info():
    engine_config = VariableInterface.async_engine.backend_config

    response = DistServeEngineConfig(tp_size=engine_config.tp,
                                     dp_size=engine_config.dp,
                                     pp_size=None,
                                     ep_size=engine_config.ep,
                                     dp_rank=engine_config.dp_rank,
                                     block_size=engine_config.block_size,
                                     num_cpu_blocks=engine_config.num_cpu_blocks,
                                     num_gpu_blocks=engine_config.num_gpu_blocks)

    return response.model_dump_json()


@router.post('/distserve/p2p_initialize')
async def p2p_initialize(init_request: DistServeInitRequest):
    return VariableInterface.async_engine.p2p_initialize(init_request)


@router.post('/distserve/p2p_connect')
async def p2p_connect(conn_request: DistServeConnectionRequest):
    return VariableInterface.async_engine.p2p_connect(conn_request)


@router.post('/distserve/p2p_drop_connect')
async def p2p_drop_connect(drop_conn_request: DistServeDropConnectionRequest):
    return VariableInterface.async_engine.p2p_drop_connect(drop_conn_request)


@router.post('/distserve/free_cache')
async def free_cache(cache_free_request: DistServeCacheFreeRequest) -> JSONResponse:
    session_id = cache_free_request.remote_session_id
    VariableInterface.async_engine.free_cache(session_id)
    return {'status': 'SUCCESS'}


""" PD Disaggregation API End """


@router.post('/abort_request')
async def abort_request(request: AbortRequest, raw_request: Request = None):
    """Abort an ongoing request."""
    if not VariableInterface.enable_abort_handling:
        return Response(
            status_code=501,
            content='This server does not support abort requests. Enable with --enable-abort-handling flag.')

    if request.abort_all:
        await VariableInterface.async_engine.stop_all_session()
    else:
        await VariableInterface.async_engine.stop_session(request.session_id)
    return Response(status_code=200)


@router.post('/v1/chat/interactive', dependencies=[Depends(check_api_key)])
async def chat_interactive_v1(request: GenerateRequest, raw_request: Request = None):
    return create_error_response(
        HTTPStatus.BAD_REQUEST, 'v1/chat/interactive is deprecated, please launch server with --enable-prefix-cache '
        'and use /v1/chat/completions instead.')


def handle_torchrun():
    """To disable mmengine logging logic when using torchrun."""

    def dummy_get_device_id():
        return 0

    if int(os.environ.get('LOCAL_RANK', -1)) > 0:
        from lmdeploy.vl.model.utils import _set_func

        # the replacement can't be recovered
        _set_func('mmengine.logging.logger._get_device_id', dummy_get_device_id)


@router.on_event('startup')
async def startup_event():
    async_engine = VariableInterface.async_engine
    async_engine.start_loop(use_async_api=True)

    if VariableInterface.proxy_url is None:
        return
    try:
        import requests
        engine_config = VariableInterface.async_engine.backend_config
        engine_role = engine_config.role.value if hasattr(engine_config, 'role') else 1
        url = f'{VariableInterface.proxy_url}/nodes/add'
        data = {'url': VariableInterface.api_server_url, 'status': {'models': get_model_list(), 'role': engine_role}}
        headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except Exception as e:
        logger.error(f'Service registration failed: {e}')


@router.on_event('shutdown')
async def shutdown_event():
    async_engine = VariableInterface.async_engine
    if async_engine is not None:
        async_engine.close()


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler for RequestValidationError."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({
            'detail': exc.errors(),
            'body': exc.body
        }),
    )


class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):

    def __init__(self, app: FastAPI, max_concurrent_requests: int):
        super().__init__(app)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def dispatch(self, request: Request, call_next):
        async with self.semaphore:
            response = await call_next(request)
            return response


def set_parsers(reasoning_parser: Optional[str] = None, tool_parser: Optional[str] = None):
    """Set tool parser and reasoning parsers."""
    # set reasoning parser
    if reasoning_parser is not None:
        if reasoning_parser in ReasoningParserManager.module_dict:
            tokenizer = VariableInterface.async_engine.tokenizer
            VariableInterface.reasoning_parser = ReasoningParserManager.get(reasoning_parser)(tokenizer)
        else:
            raise ValueError(
                f'The reasoning parser {reasoning_parser} is not in the parser list: {ReasoningParserManager.module_dict.keys()}'  # noqa
            )
    # set tool parsers
    if tool_parser is not None:
        if tool_parser in ToolParserManager.module_dict:
            tokenizer = VariableInterface.async_engine.tokenizer
            VariableInterface.tool_parser = ToolParserManager.get(tool_parser)(tokenizer)
        else:
            raise ValueError(
                f'The reasoning parser {tool_parser} is not in the parser list: {ToolParserManager.module_dict.keys()}'  # noqa
            )


def mount_metrics(app: FastAPI, backend_config: Union[PytorchEngineConfig, TurbomindEngineConfig]):
    if not getattr(backend_config, 'enable_metrics', False):
        return

    from prometheus_client import REGISTRY, make_asgi_app
    registry = REGISTRY

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount('/metrics', make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile('^/metrics(?P<path>.*)$')
    app.routes.append(metrics_route)


def create_lifespan_handler(backend_config: Union[PytorchEngineConfig, TurbomindEngineConfig],
                            async_engine: AsyncEngine):
    """Factory function to create a lifespan handler."""

    @asynccontextmanager
    async def lifespan_handler(app: FastAPI):
        task = None
        try:
            if getattr(backend_config, 'enable_metrics', False):
                metrics_processor.start_metrics_handler(enable_metrics=True)
                log_interval = 10.

                async def _force_log():
                    while True:
                        await asyncio.sleep(log_interval)

                        # Since scheduled metrics is not changed as frequently as iteration statistics,
                        # we conduct its statistics every `log_interval` seconds
                        schedule_metrics = async_engine.get_schedule_metrics()
                        await metrics_processor.udpate_schedule_stats(schedule_metrics)

                        await async_engine.do_log_stats()

                task = asyncio.create_task(_force_log())

            yield
        finally:
            if task:
                task.cancel()
            await metrics_processor.stop_metrics_handler()

    return lifespan_handler


def serve(model_path: str,
          model_name: Optional[str] = None,
          backend: Literal['turbomind', 'pytorch'] = 'turbomind',
          backend_config: Optional[Union[PytorchEngineConfig, TurbomindEngineConfig]] = None,
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
          proxy_url: Optional[str] = None,
          max_log_len: int = None,
          disable_fastapi_docs: bool = False,
          max_concurrent_requests: Optional[int] = None,
          reasoning_parser: Optional[str] = None,
          tool_call_parser: Optional[str] = None,
          allow_terminate_by_client: bool = False,
          enable_abort_handling: bool = False,
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
        chat_template_config (ChatTemplateConfig): chat template configuration
            Default to None.
        server_name (str): host ip for serving
        server_port (int): server port
        tp (int): tensor parallel
        allow_origins (List[str]): a list of allowed origins for CORS
        allow_credentials (bool): whether to allow credentials for CORS
        allow_methods (List[str]): a list of allowed HTTP methods for CORS
        allow_headers (List[str]): a list of allowed HTTP headers for CORS
        log_level(str): set log level whose value among [CRITICAL, ERROR,
            WARNING, INFO, DEBUG]
        api_keys (List[str] | str | None): Optional list of API keys. Accepts
            string type as a single api_key. Default to None, which means no
            api key applied.
        ssl (bool): Enable SSL. Requires OS Environment variables
            'SSL_KEYFILE' and 'SSL_CERTFILE'.
        proxy_url (str): The proxy url to register the api_server.
        max_log_len (int): Max number of prompt characters or prompt tokens
            being printed in log. Default: Unlimited
        max_concurrent_requests: This refers to the number of concurrent
            requests that the server can handle. The server is designed to
            process the engine’s tasks once the maximum number of concurrent
            requests is reached, regardless of any additional requests sent by
            clients concurrently during that time. Default to None.
        reasoning_parser (str): The reasoning parser name.
        tool_call_parser (str): The tool call parser name.
        allow_terminate_by_client (bool): Allow request from client to terminate server.
    """
    if os.getenv('TM_LOG_LEVEL') is None:
        os.environ['TM_LOG_LEVEL'] = log_level
    logger.setLevel(log_level)

    VariableInterface.allow_terminate_by_client = allow_terminate_by_client
    VariableInterface.enable_abort_handling = enable_abort_handling
    if api_keys is not None:
        if isinstance(api_keys, str):
            api_keys = api_keys.split(',')
        VariableInterface.api_keys = api_keys
    ssl_keyfile, ssl_certfile, http_or_https = None, None, 'http'
    if ssl:
        ssl_keyfile = os.environ['SSL_KEYFILE']
        ssl_certfile = os.environ['SSL_CERTFILE']
        http_or_https = 'https'

    handle_torchrun()
    _, pipeline_class = get_task(model_path)
    if isinstance(backend_config, PytorchEngineConfig):
        backend_config.enable_mp_engine = True
    VariableInterface.async_engine = pipeline_class(model_path=model_path,
                                                    model_name=model_name,
                                                    backend=backend,
                                                    backend_config=backend_config,
                                                    chat_template_config=chat_template_config,
                                                    max_log_len=max_log_len,
                                                    **kwargs)
    # set reasoning parser and tool parser
    set_parsers(reasoning_parser, tool_call_parser)

    # create FastAPI lifespan events
    lifespan = create_lifespan_handler(backend_config, VariableInterface.async_engine)

    if disable_fastapi_docs:
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    else:
        app = FastAPI(docs_url='/', lifespan=lifespan)

    app.include_router(router)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    mount_metrics(app, backend_config)

    if allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
        )

    # set the maximum number of concurrent requests
    if max_concurrent_requests is not None:
        app.add_middleware(ConcurrencyLimitMiddleware, max_concurrent_requests=max_concurrent_requests)

    if proxy_url is not None:
        VariableInterface.proxy_url = proxy_url
        VariableInterface.api_server_url = f'{http_or_https}://{server_name}:{server_port}'  # noqa
    for i in range(3):
        print(f'HINT:    Please open \033[93m\033[1m{http_or_https}://'
              f'{server_name}:{server_port}\033[0m in a browser for detailed api'
              ' usage!!!')
    uvicorn.run(app=app,
                host=server_name,
                port=server_port,
                log_level=os.getenv('UVICORN_LOG_LEVEL', 'info').lower(),
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile)


if __name__ == '__main__':
    import fire

    fire.Fire(serve)
