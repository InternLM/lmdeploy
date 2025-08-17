import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from lmdeploy.pytorch.engine.engine import Engine
    from lmdeploy.pytorch.engine.engine_instance import EngineInstance


class EngineInstancePool:
    """Engine Instance Pool."""

    def __init__(self, engine):
        self.engine: "Engine" = engine
        self.num_instance = self.engine.engine_config.max_batch_size
        self.pool: asyncio.Queue["EngineInstance"] = None

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