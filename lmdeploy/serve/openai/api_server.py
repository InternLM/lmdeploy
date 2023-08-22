# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import time
from http import HTTPStatus
from typing import AsyncGenerator, Optional

import fire
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from lmdeploy.serve.openai.async_engine import AsyncEngine
from lmdeploy.serve.openai.protocol import (  # noqa: E501
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, EmbeddingsRequest,
    EmbeddingsResponse, ErrorResponse, GenerateRequest, ModelCard, ModelList,
    ModelPermission, UsageInfo)

os.environ['TM_LOG_LEVEL'] = 'ERROR'


class VariableInterface:
    """A IO interface maintaining variables."""
    async_engine: AsyncEngine = None
    request_hosts = []


app = FastAPI()


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
    return JSONResponse(ErrorResponse(message=message,
                                      type='invalid_request_error').dict(),
                        status_code=status.value)


async def check_request(request) -> Optional[JSONResponse]:
    """Check if a request is valid."""
    if request.model in get_model_list():
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND, f'The model `{request.model}` does not exist.')
    return ret


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
    - renew_session (bool): Whether renew the session. Can be used when the
        session length is exceeded.
    - ignore_eos (bool): indicator for ignoring eos

    Currently we do not support the following features:
    - function_call (Users should implement this by themselves)
    - logit_bias (not supported yet)
    - presence_penalty (replaced with repetition_penalty)
    - frequency_penalty (replaced with repetition_penalty)
    """
    instance_id = int(raw_request.client.host.replace('.', ''))

    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = str(instance_id)
    created_time = int(time.time())

    result_generator = VariableInterface.async_engine.generate_openai(
        request.messages,
        instance_id,
        request.stream,
        request.renew_session,
        request_output_len=request.max_tokens if request.max_tokens else 512,
        stop=request.stop,
        top_p=request.top_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos)

    async def abort_request() -> None:
        async for _ in VariableInterface.async_engine.generate_openai(
                request.messages,
                instance_id,
                request.stream,
                request.renew_session,
                stop=True):
            pass

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
        response_json = response.json(ensure_ascii=False)

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
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
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
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(completion_stream_generator(),
                                 media_type='text/event-stream',
                                 background=background_tasks)

    # Non-streaming response
    final_res = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await abort_request()
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         'Client disconnected')
        final_res = res
    assert final_res is not None
    choices = []
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role='assistant', content=final_res.response),
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


@app.post('/v1/embeddings')
async def create_embeddings(request: EmbeddingsRequest,
                            raw_request: Request = None):
    """Creates embeddings for the text."""
    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret

    embedding = await VariableInterface.async_engine.get_embeddings(
        request.input)
    data = [{'object': 'embedding', 'embedding': embedding, 'index': 0}]
    token_num = len(embedding)
    return EmbeddingsResponse(
        data=data,
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=token_num,
            total_tokens=token_num,
            completion_tokens=None,
        ),
    ).dict(exclude_none=True)


@app.post('/generate')
async def generate(request: GenerateRequest, raw_request: Request = None):
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - sequence_start (bool): indicator for starting a sequence.
    - sequence_end (bool): indicator for ending a sequence
    - instance_id: determine which instance will be called. If not specified
        with a value other than -1, using host ip directly.
    - request_output_len (int): output token nums
    - step (int): the offset of the k/v cache
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
    if request.instance_id == -1:
        instance_id = int(raw_request.client.host.replace('.', ''))
        request.instance_id = instance_id

    generation = VariableInterface.async_engine.generate(
        request.prompt,
        request.instance_id,
        stream_response=request.stream,
        sequence_start=request.sequence_start,
        sequence_end=request.sequence_end,
        request_output_len=request.request_output_len,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for out in generation:
            ret = {
                'text': out.response,
                'tokens': out.generate_token_len,
                'finish_reason': out.finish_reason
            }
            yield (json.dumps(ret) + '\0').encode('utf-8')

    if request.stream:
        return StreamingResponse(stream_results())
    else:
        ret = {}
        async for out in generation:
            ret = {
                'text': out.response,
                'tokens': out.generate_token_len,
                'finish_reason': out.finish_reason
            }
        return JSONResponse(ret)


def main(model_path: str,
         server_name: str = 'localhost',
         server_port: int = 23333,
         instance_num: int = 32,
         tp: int = 1):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the path of the deployed model
        server_name (str): host ip for serving
        server_port (int): server port
        instance_num (int): number of instances of turbomind model
        tp (int): tensor parallel
    """
    VariableInterface.async_engine = AsyncEngine(model_path=model_path,
                                                 instance_num=instance_num,
                                                 tp=tp)
    uvicorn.run(app=app, host=server_name, port=server_port, log_level='info')


if __name__ == '__main__':
    fire.Fire(main)
