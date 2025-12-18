# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional

from lmdeploy.pytorch.disagg.conn.protocol import (DistServeConnectionRequest, DistServeDropConnectionRequest,
                                                   DistServeInitRequest)
from lmdeploy.utils import get_logger

from ..base import EngineBase, EngineInstanceBase

logger = get_logger('lmdeploy')


class MPEngine(EngineBase):

    def __init__(self) -> None:
        """Initialize mp engine."""
        self.engine_config = self._collective_rpc('get_engine_config')

    def _collective_rpc(self, func, *args, **kwargs):
        """Collective rpc call."""
        raise NotImplementedError('This method has not been implemented yet.')

    async def _collective_rpc_async(self, func, *args, **kwargs):
        """Collective rpc call."""
        raise NotImplementedError('This method has not been implemented yet.')

    async def _collective_rpc_streaming_async(self, func, *args, **kwargs):
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

    def sleep(self, level: int):
        """sleep."""
        return self._collective_rpc('sleep', level)

    def wakeup(self, tags: Optional[List[str]] = None):
        """Wakeup."""
        return self._collective_rpc('wakeup', tags)

    def update_params(self, request: Any):
        """Update params."""
        return self._collective_rpc('update_params', request)

    def get_schedule_metrics(self):
        """Get schedule metrics."""
        return self._collective_rpc('get_schedule_metrics')

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

    async def async_end(self, session_id: int):
        """End the given session."""
        return await self.engine._collective_rpc_async('instance_async_end', session_id)

    async def async_cancel(self, session_id: int):
        """Stop current streaming inference."""
        return await self.engine._collective_rpc_async('instance_async_cancel', session_id)

    async def async_stream_infer(self, *args, **kwargs):
        """Send stream inference request."""
        generator = self.engine._collective_rpc_streaming_async('instance_async_stream_infer', *args, **kwargs)
        async for result in generator:
            yield result
