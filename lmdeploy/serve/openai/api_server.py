# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import time
from http import HTTPStatus
from typing import AsyncGenerator, Optional

import fire
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from lmdeploy.serve.openai.async_engine import AsyncEngine
from lmdeploy.serve.openai.protocol import (  # noqa: E501
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    ModelCard, ModelList, ModelPermission, UsageInfo)

os.environ['TM_LOG_LEVEL'] = 'ERROR'


class WorkerInstance:
    instance: AsyncEngine = None
    request_hosts = []


app = FastAPI()


@app.post('/generate')
async def generate(request: Request):
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    """
    request_dict = await request.json()
    prompt = request_dict.pop('prompt')
    stream_output = request_dict.pop('stream', False)
    renew_session = request_dict.pop('renew_session', False)
    instance_id = int(request.client.host.replace('.', ''))

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for output in WorkerInstance.instance.generate(
                prompt,
                instance_id,
                stream_response=stream_output,
                renew_session=renew_session,
        ):
            ret = {'text': output.response}
            yield (json.dumps(ret) + '\0').encode('utf-8')

    if stream_output:
        return StreamingResponse(stream_results())
    else:
        ret = {}
        async for out in WorkerInstance.instance.generate(
                prompt,
                instance_id,
                renew_session=renew_session,
                stream_response=stream_output):
            ret = {'text': out.response}
        return JSONResponse(ret)


def get_model_list():
    """Available models.

    Only provided one now.
    """
    return [WorkerInstance.instance.tm_model.model_name]


@app.get('/v1/models')
def available_models():
    model_cards = []
    for model_name in get_model_list():
        model_cards.append(
            ModelCard(id=model_name,
                      root=model_name,
                      permission=[ModelPermission()]))
    return ModelList(data=model_cards)


def create_error_response(status: HTTPStatus, message: str):
    return JSONResponse(ErrorResponse(message=message,
                                      type='invalid_request_error').dict(),
                        status_code=status.value)


async def check_request(request) -> Optional[JSONResponse]:
    if request.model in get_model_list():
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND, f'The model `{request.model}` does not exist.')
    return ret


@app.post('/v1/chat/completions')
async def chat_completions_v1(raw_request: Request):
    """Completion API similar to OpenAI's API.

    Refer to  `https://platform.openai.com/docs/api-reference/chat/create`
    for the API specification.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (not supported yet)
    """
    request = await raw_request.json()
    request = ChatCompletionRequest(**request)
    instance_id = int(raw_request.client.host.replace('.', ''))

    error_check_ret = await check_request(request)
    if error_check_ret is not None:
        return error_check_ret

    model_name = request.model
    request_id = str(instance_id)
    created_time = int(time.time())

    result_generator = WorkerInstance.instance.generate(
        request.messages,
        instance_id,
        request.stream,
        request.renew_session,
        stop=request.stop,
        top_p=request.top_p,
        temperature=request.temperature)

    async def abort_request() -> None:
        async for _ in WorkerInstance.instance.generate(request.messages,
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


def main(model_path: str,
         server_name: str = 'localhost',
         server_port: int = 23333):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the path of the deployed model
        server_name (str): host ip for serving
        server_port (int): server port
    """
    WorkerInstance.instance = AsyncEngine(model_path=model_path)
    import uvicorn
    uvicorn.run(app=app, host=server_name, port=server_port, log_level='info')


if __name__ == '__main__':
    fire.Fire(main)
