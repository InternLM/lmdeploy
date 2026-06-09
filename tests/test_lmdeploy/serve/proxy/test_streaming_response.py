# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
from unittest.mock import AsyncMock, MagicMock

from lmdeploy.serve.proxy.streaming_response import ProxyStreamingResponse
from lmdeploy.serve.proxy.upstream.exceptions import APIServerException


async def _fake_upstream():
    yield b'chunk-1'
    await asyncio.sleep(0.2)
    yield b'chunk-2'  # should not be reached after client disconnect


def test_streaming_closes_upstream_on_client_disconnect():
    request = MagicMock()
    request.is_disconnected = AsyncMock(side_effect=[False, True])
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    completed = {'value': False}

    async def run() -> None:
        response = ProxyStreamingResponse(
            _fake_upstream(),
            raw_request=request,
            on_complete=lambda: completed.update(value=True),
            media_type='text/event-stream',
        )
        await response.stream_response(send)

    asyncio.run(run())

    assert completed['value'] is True
    assert sent[0]['type'] == 'http.response.start'
    assert any(msg.get('body') == b'chunk-1' for msg in sent if msg['type'] == 'http.response.body')
    assert not any(msg.get('body') == b'' and msg.get('more_body') is False for msg in sent)


def test_streaming_closes_upstream_before_first_chunk_on_disconnect():
    request = MagicMock()
    request.is_disconnected = AsyncMock(side_effect=[False, True])

    async def slow_upstream():
        await asyncio.sleep(10)
        yield b'late-chunk'

    sent: list[dict] = []
    completed = {'value': False}

    async def send(message: dict) -> None:
        sent.append(message)

    async def run() -> None:
        response = ProxyStreamingResponse(
            slow_upstream(),
            raw_request=request,
            on_complete=lambda: completed.update(value=True),
            media_type='text/event-stream',
        )
        await response.stream_response(send)

    asyncio.run(run())

    assert completed['value'] is True
    assert not sent


def test_streaming_runs_complete_on_upstream_error():
    async def failing_upstream():
        raise APIServerException(status_code=502, body=b'{"error":"bad"}')
        yield b''  # pragma: no cover

    sent: list[dict] = []
    completed = {'value': False}

    async def send(message: dict) -> None:
        sent.append(message)

    async def run() -> None:
        response = ProxyStreamingResponse(
            failing_upstream(),
            on_complete=lambda: completed.update(value=True),
            media_type='text/event-stream',
        )
        await response.stream_response(send)

    asyncio.run(run())

    assert completed['value'] is True
    assert sent[0]['status'] == 502
