# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import contextlib
from collections.abc import Awaitable
from typing import TypeVar

from fastapi import Request

_POLL_INTERVAL = 0.02
T = TypeVar('T')


async def wait_client_disconnect(raw_request: Request) -> None:
    while True:
        if await raw_request.is_disconnected():
            return
        await asyncio.sleep(_POLL_INTERVAL)


async def race_awaitable_with_disconnect(raw_request: Request | None, upstream: Awaitable[T]) -> T | None:
    """Run *upstream*; return ``None`` if the client disconnects first."""
    if raw_request is None:
        return await upstream

    upstream_task = asyncio.create_task(upstream)
    disconnect_task = asyncio.create_task(wait_client_disconnect(raw_request))
    try:
        done, _ = await asyncio.wait(
            {upstream_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect_task in done:
            upstream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await upstream_task
            return None
        disconnect_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await disconnect_task
        return upstream_task.result()
    except asyncio.CancelledError:
        upstream_task.cancel()
        disconnect_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await upstream_task
            await disconnect_task
        raise
