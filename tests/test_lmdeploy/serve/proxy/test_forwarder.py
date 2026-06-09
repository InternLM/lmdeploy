# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
from unittest.mock import AsyncMock, MagicMock

from lmdeploy.serve.proxy.upstream.forwarder import UpstreamForwarder


class _SlowResponseContext:
    def __init__(self) -> None:
        self.released = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.released = True

    @property
    def status(self) -> int:
        return 200

    async def read(self) -> bytes:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise
        return b'{}'


def test_forward_raw_buffer_cancels_on_client_disconnect():
    request = MagicMock()
    request.is_disconnected = AsyncMock(side_effect=[False, True])
    request.headers = {}
    request.client = MagicMock(host='127.0.0.1')
    request.url = MagicMock(scheme='http')
    request.body = AsyncMock(return_value=b'{}')

    ctx = _SlowResponseContext()
    session = MagicMock()
    session.post = MagicMock(return_value=ctx)
    forwarder = UpstreamForwarder(session)

    result = asyncio.run(forwarder.forward_raw_buffer(
        request,
        'http://127.0.0.1:19020',
        '/v1/chat/completions',
    ))

    assert result is None
    assert ctx.released is True
