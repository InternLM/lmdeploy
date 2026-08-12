# Copyright (c) OpenMMLab. All rights reserved.
"""Streaming-response resource ownership for chat completions.

Chat-completion fan-out creates multiple single-choice responses. Each child
allocates an engine session and result generator before Starlette starts its
body iterator. If another child fails, or the client disconnects before the
combined stream starts, cleanup placed only in an iterator ``finally`` block
is never activated and those resources can leak.

``ManagedStreamingResponse`` makes that ownership explicit on the response
itself. Normal iteration still performs cleanup, while ``close()`` also lets
fan-out and the ASGI response lifecycle release resources whose iterators were
never started.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable

from fastapi.responses import StreamingResponse

from lmdeploy.serve.utils.request_cleanup import cleanup_result_generators
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class ManagedStreamingResponse(StreamingResponse):
    """A chat-completion response with explicit, cancellation-safe cleanup.

    Keeping cleanup on the response makes resource ownership independent of whether Starlette entered its body iterator.
    """

    def __init__(
        self,
        content,
        *,
        result_generators: Iterable = (),
        sessions: Iterable = (),
        session_mgr=None,
        cleanup_callbacks: Iterable[Callable[[], Awaitable[None]]] = (),
        **kwargs,
    ):
        self._result_generators = tuple(result_generators)
        self._sessions = tuple(sessions)
        self._session_mgr = session_mgr
        self._cleanup_callbacks = tuple(cleanup_callbacks)
        self._resource_cleanup_task: asyncio.Task | None = None
        self._close_task: asyncio.Task | None = None
        if self._result_generators or self._sessions:
            content = self._with_resource_cleanup(content)
        super().__init__(content, **kwargs)

    async def _with_resource_cleanup(self, content):
        """Clean owned engine resources after normal body iteration."""
        try:
            async for item in content:
                yield item
        finally:
            await self._cleanup_resources()

    async def _cleanup_resources(self) -> None:
        """Close result generators and sessions once, despite cancellation."""
        if not self._result_generators and not self._sessions:
            return
        if self._resource_cleanup_task is None:
            self._resource_cleanup_task = asyncio.create_task(
                cleanup_result_generators(
                    self._result_generators,
                    self._sessions,
                    self._session_mgr,
                ),
                name='streaming_response_resource_cleanup')
        await asyncio.shield(self._resource_cleanup_task)

    async def _close(self) -> None:
        """Close the response body, owned resources, and child callbacks."""
        body_iterator = self.body_iterator
        close_iterator = getattr(body_iterator, 'aclose', None)
        if close_iterator is not None:
            try:
                await close_iterator()
            except (asyncio.CancelledError, GeneratorExit):
                pass
            except Exception:
                logger.exception('Close response body iterator failed.')

        await self._cleanup_resources()
        for callback in self._cleanup_callbacks:
            try:
                await callback()
            except (asyncio.CancelledError, GeneratorExit):
                pass
            except Exception:
                logger.exception('Streaming response cleanup callback failed.')

    async def close(self) -> None:
        """Close the body and its resources exactly once."""
        if (not self._cleanup_callbacks
                and self._resource_cleanup_task is not None
                and self._resource_cleanup_task.done()):
            return
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
