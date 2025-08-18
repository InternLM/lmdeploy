# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, List, Optional

from lmdeploy.pytorch.disagg.conn.protocol import (DistServeConnectionRequest, DistServeDropConnectionRequest,
                                                   DistServeInitRequest)
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

if TYPE_CHECKING:
    from lmdeploy.pytorch.engine.engine import Engine


class EngineInstancePool:
    """Engine Instance Pool."""

    def __init__(self, engine):
        from lmdeploy.pytorch.engine import Engine
        self.engine: Engine = engine
        self.num_instance = self.engine.engine_config.max_batch_size
        self.pool = None

    def create_instance_pool(self, num_instance: int):
        """Create instance pool."""
        pool = asyncio.Queue(maxsize=num_instance)
        for _ in range(num_instance):
            instance = self.engine.create_instance()
            pool.put_nowait(instance)
        return pool

    @asynccontextmanager
    async def instance(self):
        """Get an instance from the pool."""
        # lazy create pool
        if self.pool is None:
            self.pool = self.create_instance_pool(self.num_instance)
        instance = await self.pool.get()
        try:
            yield instance
        finally:
            self.pool.put_nowait(instance)

    async def async_end(self, session_id: int):
        """End the given session."""
        async with self.instance() as instance:
            return await instance.async_end(session_id)

    async def async_cancel(self, session_id: int):
        """Stop current streaming inference."""
        async with self.instance() as instance:
            return await instance.async_cancel(session_id)

    async def async_stream_infer(self, *args, **kwargs):
        """Send stream inference request."""
        async with self.instance() as instance:
            async for result in instance.async_stream_infer(*args, **kwargs):
                yield result


class EngineWorkerBase:
    """Base class for engine worker."""

    def __init__(self, engine: 'Engine'):
        engine.start_loop()
        self.engine = engine
        self.instance_pool = EngineInstancePool(engine)

    def end_session(self, session_id: int):
        """End session."""
        return self.engine.end_session(session_id)

    def get_engine_config(self):
        """Get engine config."""
        return self.engine.get_engine_config()

    def get_model_config(self):
        """Get model config."""
        return self.engine.get_model_config()

    def p2p_initialize(self, conn_request: DistServeInitRequest):
        """Init rdma link."""
        return self.engine.p2p_initialize(conn_request)

    def p2p_connect(self, conn_request: DistServeConnectionRequest):
        """rdma_connect."""
        return self.engine.p2p_connect(conn_request)

    def p2p_drop_connect(self, drop_conn_request: DistServeDropConnectionRequest):
        """Drop connection.

        1. drop engine connection (zmq connection)
        2. TODO(JimyMa) drop RDMA Connection.
        """
        return self.engine.p2p_drop_connect(drop_conn_request)

    def sleep(self, level: int = 1):
        """sleep."""
        return self.engine.sleep(level)

    def wakeup(self, tags: Optional[List[str]] = None):
        """Wakeup."""
        return self.engine.wakeup(tags)

    def update_params(self, request: Any):
        """Update params."""
        return self.engine.update_params(request)

    def close(self) -> None:
        """Close engine worker."""
        self.engine.close()

    async def instance_async_end(self, session_id: int):
        """End the given session."""
        return await self.instance_pool.async_end(session_id)

    async def instance_async_cancel(self, session_id: int):
        """Stop current streaming inference."""
        return await self.instance_pool.async_cancel(session_id)

    async def instance_async_stream_infer(self, *args, **kwargs):
        """Send stream inference request."""
        async for result in self.instance_pool.async_stream_infer(*args, **kwargs):
            yield result
