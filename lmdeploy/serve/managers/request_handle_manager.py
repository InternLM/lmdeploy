# Copyright (c) OpenMMLab. All rights reserved.
import asyncio

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class RequestHandleManager:
    """Manages a pool of request handles for concurrent request processing.

    This class maintains a fixed-size pool of request handles that can be reused
    across multiple inference requests. It implements a lazy-initialized queue-based
    pool pattern to efficiently manage handle lifecycle and enable concurrent
    request handling.

    Each request should acquire a handle from the pool before inference and
    return it after completion. The manager supports:
    - Pool-based handle allocation and deallocation
    - Lazy initialization of the async queue (required for asyncio.Queue)
    - Handle rebuilding after engine wakeup (e.g., turbomind backend)
    - Complete pool cleanup

    Args:
        engine (AsyncEngine): The async inference engine that creates handles.
        size (int): The size of the handle pool, typically set to max_batch_size.

    Note:
        The pool queue is lazily initialized on first access via `get()` method,
        as `asyncio.Queue` must be created within an async context.
    """

    def __init__(self, engine, size: int):
        self.size = size
        self.handles = [engine.create_instance() for _ in range(size)]
        # `asyncio.Queue` must be created in an async context, refer to `get` method
        self.pool: asyncio.Queue = None

    async def get(self):
        """Get a handle from pool."""
        # Lazy initialization: create pool on first use
        if self.pool is None:
            self.pool = asyncio.Queue()
            for inst in self.handles:
                self.pool.put_nowait(inst)

        return await self.pool.get()

    def put(self, handle):
        """Put a handle back to the pool."""
        if handle is not None and self.pool is not None:
            self.pool.put_nowait(handle)

    def rebuild(self, engine):
        """Rebuild all handles.

        Used after wakeup turbomind engine.
        """
        self.handles = [engine.create_instance() for _ in range(self.size)]
        self.pool = None

    def clear(self):
        """Clear all handles."""
        self.handles = []
        self.pool = None
