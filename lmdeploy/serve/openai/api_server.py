# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
import random
import time
from http import HTTPStatus
from typing import AsyncGenerator, List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.openai.protocol import (  # noqa: E501
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, CompletionRequest,
    CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse, DeltaMessage,
    EmbeddingsRequest, EncodeRequest, EncodeResponse, ErrorResponse,
    GenerateRequest, GenerateResponse, ModelCard, ModelList, ModelPermission,
    UsageInfo)


class VariableInterface:
    """A IO interface maintaining variables."""
    async_engine: AsyncEngine = None
    request_hosts = []


app = FastAPI(docs_url='/')


def get_model_list():
    """Available models.

    Only provided one now.
    """
    return [VariableInterface.async_engine.tm_model.model_name]


@app.get('/v1/models')
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


def ip2id(host_ip: str):
    """Convert host ip address to session id."""
    if '.' in host_ip:  # IPv4
        return int(host_ip.replace('.', '')[-8:])
    if ':' in host_ip:  # IPv6
        return int(host_ip.replace(':', '')[-8:], 16)
    print('Warning, could not get session id from ip, set it 0')
    return 0


@app.post('/v1/chat/completions')
async def chat_completions_v1(request: ChatCompletionRequest,
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
    - session_id (int): if not specified, will set random value

    Currently we do not support the following features:
    - function_call (Users should implement this by themselves)
    - logit_bias (not supported yet)
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    if request.session_id == -1:
        request.session_id = random.randint(1, 10086)
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = str(request.session_id)
    created_time = int(time.time())

    result_generator = VariableInterface.async_engine.generate(
        request.messages,
        request.session_id,
        True,  # always use stream to enable batching
        sequence_start=True,
        sequence_end=True,
        request_output_len=request.max_tokens if request.max_tokens else 512,
        stop=request.stop,
        top_p=request.top_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos,
        do_preprocess=not isinstance(request.messages,
                                     str),  # text completion for string input
    )

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
                finish_reason=res.finish_reason,
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
            VariableInterface.async_engine.stop_session(request.session_id)
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


@app.post('/v1/completions')
async def completions_v1(request: CompletionRequest,
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
    - ignore_eos (bool): indicator for ignoring eos
    - session_id (int): if not specified, will set random value

    Currently we do not support the following features:
    - logprobs (not supported yet)
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    if request.session_id == -1:
        request.session_id = random.randint(1, 10086)
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = str(request.session_id)
    created_time = int(time.time())
    if isinstance(request.prompt, str):
        request.prompt = [request.prompt]
    generators = []
    for i in range(len(request.prompt)):
        result_generator = VariableInterface.async_engine.generate(
            request.prompt[i],
            request.session_id + i,
            True,  # always use stream to enable batching
            sequence_start=True,
            sequence_end=True,
            request_output_len=request.max_tokens
            if request.max_tokens else 512,
            stop=False,
            top_p=request.top_p,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            ignore_eos=request.ignore_eos,
            do_preprocess=False)
        generators.append(result_generator)

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
                    finish_reason=res.finish_reason,
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
                VariableInterface.async_engine.stop_session(request.session_id)
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


@app.post('/v1/embeddings', tags=['unsupported'])
async def create_embeddings(request: EmbeddingsRequest,
                            raw_request: Request = None):
    """Creates embeddings for the text."""
    return create_error_response(HTTPStatus.BAD_REQUEST,
                                 'Unsupported by turbomind.')


@app.post('/v1/encode')
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
            prompt = VariableInterface.async_engine.model.get_prompt(
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


@app.post('/generate',
          tags=['deprecated'],
          description='please use /v1/chat/interactive')
@app.post('/v1/chat/interactive')
async def chat_interactive_v1(request: GenerateRequest,
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
    """
    if request.session_id == -1:
        request.session_id = random.randint(10087, 23333)

    async_engine = VariableInterface.async_engine
    sequence_start = async_engine.id2step.get(str(request.session_id), 0) == 0
    sequence_end = not request.interactive_mode

    generation = async_engine.generate(
        request.prompt,
        request.session_id,
        stream_response=True,  # always use stream to enable batching
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        request_output_len=request.request_output_len,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for out in generation:
            chunk = GenerateResponse(text=out.response,
                                     tokens=out.generate_token_len,
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
                async_engine.stop_session(request.session_id)
                return create_error_response(HTTPStatus.BAD_REQUEST,
                                             'Client disconnected')
            text += out.response
            tokens = out.generate_token_len
            finish_reason = out.finish_reason
        ret = {'text': text, 'tokens': tokens, 'finish_reason': finish_reason}
        return JSONResponse(ret)


def serve(model_path: str,
          model_name: Optional[str] = None,
          server_name: str = '0.0.0.0',
          server_port: int = 23333,
          instance_num: int = 64,
          tp: int = 1,
          allow_origins: List[str] = ['*'],
          allow_credentials: bool = True,
          allow_methods: List[str] = ['*'],
          allow_headers: List[str] = ['*'],
          log_level: str = 'ERROR',
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
                    on huggingface.co, such as "InternLM/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "InternLM/internlm-chat-7b"
        server_name (str): host ip for serving
        server_port (int): server port
        instance_num (int): number of instances of turbomind model
        tp (int): tensor parallel
        allow_origins (List[str]): a list of allowed origins for CORS
        allow_credentials (bool): whether to allow credentials for CORS
        allow_methods (List[str]): a list of allowed HTTP methods for CORS
        allow_headers (List[str]): a list of allowed HTTP headers for CORS
        log_level(str): set log level whose value among [CRITICAL, ERROR, WARNING, INFO, DEBUG]
    """ # noqa E501
    os.environ['TM_LOG_LEVEL'] = log_level

    if allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
        )

    VariableInterface.async_engine = AsyncEngine(model_path=model_path,
                                                 model_name=model_name,
                                                 instance_num=instance_num,
                                                 tp=tp,
                                                 **kwargs)
    for i in range(3):
        print(f'HINT:    Please open \033[93m\033[1mhttp://{server_name}:'
              f'{server_port}\033[0m in a browser for detailed api usage!!!')
    uvicorn.run(app=app, host=server_name, port=server_port, log_level='info')


if __name__ == '__main__':
    import fire

    fire.Fire(serve)
