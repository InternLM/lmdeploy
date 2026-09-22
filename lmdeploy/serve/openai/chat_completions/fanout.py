# Copyright (c) OpenMMLab. All rights reserved.
"""Multi-choice collation for the chat completions endpoint."""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass

import shortuuid
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, UsageInfo
from lmdeploy.serve.utils.streaming_response import ManagedStreamingResponse

ChatEndpoint = Callable[[ChatCompletionRequest, Request],
                        Awaitable[dict | Response]]


@dataclass
class _FanoutResponseError(Exception):
    """Carry an HTTP error response out of concurrent choice invocation."""

    response: Response


class _FanoutRequest:
    """Give each recursive endpoint call an isolated JSON payload."""

    def __init__(self, request: Request, payload: dict):
        self._request = request
        self._payload = payload

    def __getattr__(self, name):
        """Delegate request attributes not overridden by this wrapper."""
        return getattr(self._request, name)

    async def json(self) -> dict:
        """Return an isolated copy of this choice's raw JSON payload."""
        return deepcopy(self._payload)

    async def is_disconnected(self) -> bool:
        """Report the connection state of the original client request."""
        return await self._request.is_disconnected()


def _choice_request(request: ChatCompletionRequest,
                    index: int) -> ChatCompletionRequest:
    """Create an independent single-choice request for one fan-out index."""
    choice_request = request.model_copy(deep=True)
    choice_request.n = 1
    choice_request.session_id = -1
    if request.seed is not None:
        choice_request.seed = (request.seed + index) % (1 << 64)
    return choice_request


async def _cancel_tasks(tasks: list[asyncio.Task]) -> list:
    """Cancel unfinished tasks and collect every terminal result."""
    for task in tasks:
        if not task.done():
            task.cancel()
    return await asyncio.gather(*tasks, return_exceptions=True)


async def _close_streaming_response(response: StreamingResponse) -> None:
    """Close a child response through its explicit or iterator lifecycle."""
    close_response = getattr(response, 'close', None)
    if close_response is not None:
        await close_response()
        return
    close_iterator = getattr(response.body_iterator, 'aclose', None)
    if close_iterator is not None:
        await close_iterator()


async def _close_responses(responses) -> None:
    """Close all streaming responses in an invocation result collection."""
    await asyncio.gather(*(
        _close_streaming_response(response)
        for response in responses
        if isinstance(response, StreamingResponse)
    ), return_exceptions=True)


async def _cleanup_invocations(tasks: list[asyncio.Task]) -> None:
    """Cancel choice invocations and close responses they already created."""
    results = await _cancel_tasks(tasks)
    await _close_responses(results)


def _consume_cleanup_result(task: asyncio.Task) -> None:
    """Retrieve a detached cleanup task's result to suppress task warnings."""
    try:
        task.result()
    except BaseException:  # cleanup is best-effort after caller cancellation
        pass


async def _shield_cleanup(awaitable, name: str) -> None:
    """Let cleanup continue if cancellation interrupts its caller."""
    cleanup_task = asyncio.create_task(awaitable, name=name)
    try:
        await asyncio.shield(cleanup_task)
    except (asyncio.CancelledError, GeneratorExit):
        cleanup_task.add_done_callback(_consume_cleanup_result)
        raise


async def _invoke_choices(
    endpoint: ChatEndpoint,
    request: ChatCompletionRequest,
    raw_request: Request,
    payload: dict,
) -> list[dict | StreamingResponse] | Response:
    """Invoke the single-choice endpoint concurrently for every choice."""

    async def invoke(index: int):
        """Invoke and validate one indexed single-choice response."""
        choice_request = _choice_request(request, index)
        choice_payload = deepcopy(payload)
        choice_payload.update(n=1, session_id=-1, seed=choice_request.seed)
        response = await endpoint(
            choice_request,
            _FanoutRequest(raw_request, choice_payload),
        )
        if isinstance(response, Response) and not isinstance(
                response, StreamingResponse):
            raise _FanoutResponseError(response)
        return response

    tasks = [asyncio.create_task(invoke(index)) for index in range(request.n)]
    try:
        return await asyncio.gather(*tasks)
    except _FanoutResponseError as error:
        await _shield_cleanup(
            _cleanup_invocations(tasks), 'fanout_invocation_cleanup')
        return error.response
    except BaseException:
        await _shield_cleanup(
            _cleanup_invocations(tasks), 'fanout_invocation_cleanup')
        raise


def _cached_tokens(usage: dict) -> int:
    """Read cached prompt tokens from an OpenAI-compatible usage object."""
    details = usage.get('prompt_tokens_details') or {}
    return details.get('cached_tokens', 0)


def _aggregate_usage(usages: list[dict]) -> UsageInfo:
    """Count shared prompt usage once and sum per-choice completion usage."""
    first_usage = usages[0]
    completion_details = [
        usage.get('completion_tokens_details') for usage in usages
    ]
    reasoning_tokens = None
    if all(details is not None for details in completion_details):
        reasoning_tokens = sum(
            details['reasoning_tokens'] for details in completion_details)
    return UsageInfo.build(
        prompt_tokens=first_usage.get('prompt_tokens', 0),
        completion_tokens=sum(
            usage.get('completion_tokens') or 0 for usage in usages),
        cached_tokens=_cached_tokens(first_usage),
        reasoning_tokens=reasoning_tokens,
    )


def _collate_responses(
    responses: list[dict],
    request_id: str,
    created_time: int,
) -> dict:
    """Combine single-choice JSON responses into one multi-choice response."""
    response = deepcopy(responses[0])
    response['id'] = request_id
    response['created'] = created_time
    response['choices'] = []
    usages = []
    for index, choice_response in enumerate(responses):
        choices = choice_response.get('choices') or []
        if len(choices) != 1:
            raise RuntimeError(
                f'Expected one choice from fan-out request, got {len(choices)}'
            )
        choice = deepcopy(choices[0])
        choice['index'] = index
        response['choices'].append(choice)
        usages.append(choice_response.get('usage') or {})
    response['usage'] = _aggregate_usage(usages).model_dump()
    return response


async def _stream_choice(
    index: int,
    response: StreamingResponse,
    queue: asyncio.Queue,
    request_id: str,
    created_time: int,
    model_name: str,
    stopping: asyncio.Event,
) -> None:
    """Parse one child SSE stream and forward normalized events to a queue."""
    buffer = ''
    try:
        async for chunk in response.body_iterator:
            buffer += chunk.decode() if isinstance(chunk, bytes) else chunk
            while '\n\n' in buffer:
                event, buffer = buffer.split('\n\n', 1)
                for line in event.splitlines():
                    if not line.startswith('data: '):
                        continue
                    data = line.removeprefix('data: ')
                    if data == '[DONE]':
                        continue
                    payload = json.loads(data)
                    if payload.get(
                            'usage'
                    ) is not None and not payload.get('choices'):
                        await queue.put(('usage', index, payload['usage']))
                        continue
                    choices = payload.get('choices') or []
                    if len(choices) != 1:
                        raise RuntimeError(
                            'Expected one streaming choice from fan-out request, '
                            f'got {len(choices)}')
                    choices[0]['index'] = index
                    payload.update(
                        id=request_id,
                        created=created_time,
                        model=model_name,
                    )
                    await queue.put(('data', payload))
        await queue.put(('done', index))
    except asyncio.CancelledError:
        if not stopping.is_set():
            await queue.put((
                'error',
                RuntimeError(f'Fan-out choice {index} was cancelled.'),
            ))
        raise
    except Exception as error:  # noqa: BLE001
        await queue.put(('error', error))
        raise
    finally:
        await _close_streaming_response(response)


def _batch_stream_payloads(payloads: list[dict]) -> list[dict]:
    """Combine each choice's Nth ready delta into the Nth output batch."""
    batches: list[dict] = []
    next_batch_by_index: dict[int, int] = {}

    for payload in payloads:
        choice = payload['choices'][0]
        index = choice['index']
        target = next_batch_by_index.get(index, 0)
        if target == len(batches):
            batches.append(payload)
        else:
            batches[target]['choices'].append(choice)
        next_batch_by_index[index] = target + 1

    for payload in batches:
        payload['choices'].sort(key=lambda choice: choice['index'])
    return batches


async def _collate_streams(
    responses: list[StreamingResponse],
    request: ChatCompletionRequest,
    request_id: str,
    created_time: int,
) -> AsyncGenerator[str, None]:
    """Interleave child streams into one multi-choice SSE response."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, len(responses) * 2))
    stopping = asyncio.Event()
    # produce the streaming responses for each fan-out request
    tasks = [
        asyncio.create_task(
            _stream_choice(index, response, queue, request_id, created_time,
                           request.model, stopping))
        for index, response in enumerate(responses)
    ]
    usages: dict[int, dict] = {}
    completed = 0
    include_usage = bool(request.stream_options
                         and request.stream_options.include_usage)
    # consume the streaming responses of each fan-out request, and yield to the client
    try:
        while completed < len(tasks):
            items = [await queue.get()]
            while True:
                try:
                    items.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            payloads = []
            stream_error = None
            for item in items:
                if item[0] == 'data':
                    payloads.append(item[1])
                elif item[0] == 'usage':
                    # item[1]: index, item[2]: usage payload
                    usages[item[1]] = item[2]
                elif item[0] == 'done':
                    completed += 1
                else:
                    stream_error = item[1]
                    break

            for payload in _batch_stream_payloads(payloads):
                yield f'data: {json.dumps(payload)}\n\n'
            if stream_error is not None:
                raise stream_error
        if include_usage and len(usages) == len(tasks):
            ordered_usages = [usages[index] for index in range(len(tasks))]
            usage_response = {
                'id': request_id,
                'object': 'chat.completion.chunk',
                'created': created_time,
                'model': request.model,
                'choices': [],
                'usage': _aggregate_usage(ordered_usages).model_dump(),
            }
            yield f'data: {json.dumps(usage_response)}\n\n'
        yield 'data: [DONE]\n\n'
    finally:
        stopping.set()
        await _shield_cleanup(_cancel_tasks(tasks),
                              'fanout_stream_cleanup')


async def fanout_chat_completions(
    endpoint: ChatEndpoint,
    request: ChatCompletionRequest,
    raw_request: Request,
    payload: dict,
) -> dict | Response:
    """Run the established single-choice endpoint once per requested choice."""
    request_id = f'chatcmpl-{shortuuid.random()}'
    created_time = int(time.time())
    responses = await _invoke_choices(endpoint, request, raw_request, payload)
    if isinstance(responses, Response):
        return responses
    if request.stream:
        if not all(
                isinstance(response, StreamingResponse)
                for response in responses):
            await _close_responses(responses)
            raise RuntimeError(
                'Expected streaming responses from fan-out requests')
        stream = _collate_streams(responses, request, request_id, created_time)
        return ManagedStreamingResponse(
            stream,
            cleanup_callbacks=[
                lambda response=response: _close_streaming_response(response)
                for response in responses
            ],
            media_type='text/event-stream')
    if not all(isinstance(response, dict) for response in responses):
        await _close_responses(responses)
        raise RuntimeError('Expected JSON objects from fan-out requests')
    return _collate_responses(responses, request_id, created_time)
