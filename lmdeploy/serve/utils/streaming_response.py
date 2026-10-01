# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable

from fastapi.responses import StreamingResponse

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class ManagedStreamingResponse(StreamingResponse):
    """Manage resources allocated before response-body iteration starts.

    Use this response when an endpoint creates resources such as engine
    sessions, result generators, or child streaming responses before returning.
    Cleanup placed only in the body iterator's ``finally`` block is insufficient
    because the client may disconnect before iteration starts, or fan-out may
    discard a child response without consuming it.

    Async cleanup callbacks run exactly once through explicit ``close()`` or
    after the ASGI response lifecycle finishes. Use a regular
    ``StreamingResponse`` when all resources are owned by the body iterator.
    """

    def __init__(
        self,
        content,
        *,
        cleanup_callbacks: Iterable[Callable[[], Awaitable[None]]] = (),
        **kwargs,
    ):
        self._cleanup_callbacks = tuple(cleanup_callbacks)
        self._close_task: asyncio.Task | None = None
        super().__init__(content, **kwargs)

    async def _close(self) -> None:
        """Close the response body and run its cleanup callbacks."""
        body_iterator = self.body_iterator
        close_iterator = getattr(body_iterator, 'aclose', None)
        if close_iterator is not None:
            try:
                await close_iterator()
            except (asyncio.CancelledError, GeneratorExit):
                pass
            except Exception:
                logger.exception('Close response body iterator failed.')

        for callback in self._cleanup_callbacks:
            try:
                await callback()
            except (asyncio.CancelledError, GeneratorExit):
                pass
            except Exception:
                logger.exception('Streaming response cleanup callback failed.')

    async def close(self) -> None:
        """Close the body and its resources exactly once."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close(), name='streaming_response_cleanup')
        try:
            await asyncio.shield(self._close_task)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception:
            logger.exception('Streaming response cleanup failed.')

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.close()
