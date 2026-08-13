# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeConnectionRequest,
    DistServeDropConnectionRequest,
    DistServeInitRequest,
)
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

if TYPE_CHECKING:
    from lmdeploy.pytorch.engine.engine import Engine


@dataclass(frozen=True)
class StreamError:
    """Serializable error raised by a remote streaming producer."""

    type_name: str
    message: str

    @classmethod
    def from_exception(cls, error: BaseException):
        """Build a transport-safe error description."""
        error_type = type(error)
        type_name = f'{error_type.__module__}.{error_type.__qualname__}'
        return cls(type_name=type_name, message=str(error))


class MPStreamError(RuntimeError):
    """Error propagated by an MP streaming transport."""

    def __init__(self, error: StreamError):
        self.remote_error = error
        super().__init__(f'{error.type_name}: {error.message}')


@dataclass
class StreamPollResult:
    """Atomic snapshot returned by an MP stream poll."""

    output: Any = None
    has_output: bool = False
    done: bool = False
    error: StreamError | None = None


@dataclass
class StreamMailbox:
    """Single-slot coalescing mailbox shared by MP stream backends."""

    event: asyncio.Event = field(default_factory=asyncio.Event)
    output: Any = None
    pending: bool = False
    done: bool = False
    error: StreamError | None = None

    def publish(self, output: Any):
        """Publish or replace the pending coalesced output."""
        if self.done:
            raise RuntimeError('Cannot publish output after stream completion.')
        self.output = output
        self.pending = True
        self.event.set()

    def finish(self):
        """Mark the producer complete without touching pending output."""
        self.done = True
        self.event.set()

    def fail(self, error: BaseException):
        """Record a producer failure without overwriting pending output."""
        self.error = StreamError.from_exception(error)
        self.done = True
        self.event.set()

    def drain(self):
        """Atomically transfer pending output and terminal state."""
        poll_result = StreamPollResult(
            output=self.output if self.pending else None,
            has_output=self.pending,
            done=self.done,
            error=self.error,
        )
        self.output = None
        self.pending = False
        self.error = None
        self.event.clear()
        return poll_result


def iter_stream_poll_outputs(poll_result: StreamPollResult, method: str):
    """Apply common Ray/ZMQ output and error delivery semantics."""
    if poll_result.has_output:
        yield poll_result.output
    if poll_result.error is not None:
        if method == 'instance_async_stream_infer':
            yield EngineOutput(ResponseType.INTERNAL_ENGINE_ERROR, [])
        else:
            raise MPStreamError(poll_result.error)


class EngineInstancePool:
    """Engine Instance Pool."""

    def __init__(self, engine):
        from lmdeploy.pytorch.engine import Engine
        self.engine: Engine = engine
        # enlarge `num_instance`, otherwise an sequence cannot be stopped in time
        self.num_instance = self.engine.engine_config.max_batch_size * 2
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

    def get_schedule_metrics(self):
        """Get schedule metrics."""
        return self.engine.get_schedule_metrics()

    async def get_health_status(self):
        """Get engine health status."""
        return await self.engine.get_health_status()

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

    async def sleep(self, level: int = 1):
        """sleep."""
        return await self.engine.sleep(level)

    def wakeup(self, tags: list[str] | None = None):
        """Wakeup."""
        return self.engine.wakeup(tags)

    def update_params(self, request: Any):
        """Update params."""
        return self.engine.update_params(request)

    async def init_weights_update_group(self, request: Any):
        """Init disaggregated weights-update process group."""
        return await self.engine.init_weights_update_group(request)

    async def update_weights_from_distributed(self, request: Any):
        """Receive weights through the disaggregated process group."""
        return await self.engine.update_weights_from_distributed(request)

    async def destroy_weights_update_group(self, request: Any):
        """Tear down a previously initialized weights-update process group."""
        return await self.engine.destroy_weights_update_group(request)

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


class EngineOutputGather:
    """Helper class to gather incremental engine output."""

    def __init__(self):
        self._output = dict()

    def get(self, stream_id):
        if stream_id not in self._output:
            self._output[stream_id] = EngineOutput(status=None, token_ids=[], logprobs=None)
        return self._output[stream_id]

    def add(self, stream_id, result):
        if not isinstance(result, EngineOutput):
            return
        output = self.get(stream_id)
        output.token_ids.extend(result.token_ids or [])
        if result.logprobs is not None:
            if output.logprobs is None:
                output.logprobs = []
            output.logprobs.extend(result.logprobs)

    def pop(self, stream_id, result):
        if not isinstance(result, EngineOutput):
            return result
        output = self._output.pop(stream_id, None)
        if output is None:
            return result
        result.token_ids = output.token_ids or []
        result.logprobs = output.logprobs
        return result

    def discard(self, stream_id):
        """Discard gathered output for a stream."""
        self._output.pop(stream_id, None)
