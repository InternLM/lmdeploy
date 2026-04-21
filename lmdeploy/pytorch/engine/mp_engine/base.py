# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeConnectionRequest,
    DistServeDropConnectionRequest,
    DistServeInitRequest,
)
from lmdeploy.utils import get_logger

from ..base import EngineBase, EngineInstanceBase

logger = get_logger('lmdeploy')


@dataclass
class SessionState:
    is_exists: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False


class MPEngine(EngineBase):

    def __init__(self) -> None:
        """Initialize mp engine."""
        self.session_states = defaultdict(SessionState)
        self.engine_config = self._collective_rpc('get_engine_config')
        self.pending_cancel_sessions = set()

    def _collective_rpc(self, func, *args, **kwargs):
        """Collective rpc call."""
        raise NotImplementedError('This method has not been implemented yet.')

    async def _collective_rpc_async(self, func, *args, **kwargs):
        """Collective rpc call."""
        raise NotImplementedError('This method has not been implemented yet.')

    async def _collective_rpc_streaming_async(self, func: str, sess_event: asyncio.Event, *args, **kwargs):
        """Collective rpc call."""
        raise NotImplementedError('This method has not been implemented yet.')

    def close(self) -> None:
        """Close mp engine."""
        raise NotImplementedError('This method has not been implemented yet.')

    def start_loop(self) -> None:
        """Start mp engine loop."""
        raise NotImplementedError('This method has not been implemented yet.')

    def end_session(self, session_id: int):
        """End session."""
        return self._collective_rpc('end_session', session_id)

    async def sleep(self, level: int):
        """sleep."""
        return await self._collective_rpc_async('sleep', level)

    def wakeup(self, tags: list[str] | None = None):
        """Wakeup."""
        return self._collective_rpc('wakeup', tags)

    def update_params(self, request: Any):
        """Update params."""
        return self._collective_rpc('update_params', request)

    async def get_schedule_metrics(self):
        """Get schedule metrics."""
        return await self._collective_rpc_async('get_schedule_metrics')

    def p2p_initialize(self, conn_request: DistServeInitRequest):
        """Init rdma link."""
        return self._collective_rpc('p2p_initialize', conn_request)

    def p2p_connect(self, conn_request: DistServeConnectionRequest):
        """rdma_connect."""
        return self._collective_rpc('p2p_connect', conn_request)

    def p2p_drop_connect(self, drop_conn_request: DistServeDropConnectionRequest):
        """Drop connection.

        1. drop engine connection (zmq connection)
        2. TODO(JimyMa) drop RDMA Connection.
        """
        return self._collective_rpc('p2p_drop_connect', drop_conn_request)

    def create_instance(self, cuda_stream_id=0):
        """Create instance."""
        return MPEngineInstance(self)


class MPEngineInstance(EngineInstanceBase):
    """MP Engine Instance."""

    def __init__(self, engine: MPEngine):
        self.engine = engine
        self.session_states = engine.session_states

    async def async_end(self, session_id: int):
        """End the given session."""
        if session_id not in self.session_states:
            self.engine.pending_cancel_sessions.discard(session_id)
            logger.warning(f'Session {session_id} not found when end session.')
            return ResponseType.SESSION_NOT_EXIST
        await self.session_states[session_id].is_exists.wait()
        ret = await self.engine._collective_rpc_async('instance_async_end', session_id)
        self.session_states.pop(session_id)
        self.engine.pending_cancel_sessions.discard(session_id)
        return ret

    async def async_cancel(self, session_id: int):
        """Stop current streaming inference."""
        if session_id not in self.session_states:
            logger.debug(f'Session {session_id} not found when cancel session.')
            return ResponseType.SESSION_NOT_EXIST
        state = self.session_states[session_id]
        self.engine.pending_cancel_sessions.add(session_id)
        if not state.is_exists.is_set():
            logger.debug(f'Session {session_id} not started yet, recording pending cancel.')
            state.cancelled = True
            return ResponseType.SUCCESS
        return await self.engine._collective_rpc_async('instance_async_cancel', session_id)

    async def async_stream_infer(self, session_id: int, *args, **kwargs):
        """Send stream inference request."""
        state = self.session_states[session_id]
        if state.cancelled or session_id in self.engine.pending_cancel_sessions:
            state.is_exists.set()
            logger.debug(f'Session {session_id} canceld, async_stream_infer')
            yield EngineOutput(ResponseType.CANCEL, [])
            return
        kwargs['session_id'] = session_id
        generator = self.engine._collective_rpc_streaming_async('instance_async_stream_infer',
                                                                state.is_exists,
                                                                *args,
                                                                **kwargs)

        async for result in generator:
            yield result
