# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import json
from collections.abc import Callable
from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse

from lmdeploy.serve.proxy.upstream.exceptions import APIServerException
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class ProxyStreamingResponse(StreamingResponse):
    """StreamingResponse with upstream error handling and client-disconnect
    cleanup."""

    def __init__(
        self,
        content,
        *,
        raw_request: Request | None = None,
        on_complete: Callable[[], Any] | None = None,
        **kwargs,
    ):
        super().__init__(content, **kwargs)
        self._raw_request = raw_request
        self._on_complete = on_complete

    async def _client_disconnected(self) -> bool:
        return self._raw_request is not None and await self._raw_request.is_disconnected()

    @staticmethod
    async def _close_upstream(iterator) -> None:
        try:
            await iterator.aclose()
        except Exception:
            pass

    def _run_complete(self) -> None:
        if self._on_complete is not None:
            self._on_complete()

    async def stream_response(self, send) -> None:
        iterator = self.body_iterator.__aiter__()
        try:
            try:
                first_chunk = await iterator.__anext__()
            except APIServerException as e:
                headers = self._convert_headers_to_asgi(e.headers) if e.headers else self.raw_headers
                await send({'type': 'http.response.start', 'status': e.status_code, 'headers': headers})
                await send({
                    'type': 'http.response.body',
                    'body': e.body,
                    'more_body': False,
                })
                return

            await send({
                'type': 'http.response.start',
                'status': self.status_code,
                'headers': self.raw_headers,
            })
            await send({
                'type': 'http.response.body',
                'body': first_chunk,
                'more_body': True,
            })

            if await self._client_disconnected():
                logger.info('client disconnected during proxy streaming; closing upstream')
                await self._close_upstream(iterator)
                return

            try:
                async for chunk in iterator:
                    await send({
                        'type': 'http.response.body',
                        'body': chunk,
                        'more_body': True,
                    })
                    if await self._client_disconnected():
                        logger.info('client disconnected during proxy streaming; closing upstream')
                        await self._close_upstream(iterator)
                        return
            except asyncio.CancelledError:
                logger.info('proxy streaming cancelled; closing upstream')
                await self._close_upstream(iterator)
                raise
            except Exception:
                error_data = {'error': True, 'status': 500, 'message': 'Internal streaming error'}
                await send({
                    'type': 'http.response.body',
                    'body': json.dumps(error_data).encode('utf-8'),
                    'more_body': False,
                })
                return

            await send({
                'type': 'http.response.body',
                'body': b'',
                'more_body': False,
            })
        finally:
            self._run_complete()

    def _convert_headers_to_asgi(self, headers: dict) -> list[tuple[bytes, bytes]]:
        """Convert dict headers to ASGI raw header tuples."""
        return [(name.lower().encode('latin-1'), str(value).encode('latin-1')) for name, value in headers.items()]
