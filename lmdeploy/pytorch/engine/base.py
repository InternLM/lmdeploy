# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeConnectionRequest, DistServeDropConnectionRequest,
                                                   DistServeInitRequest)


class EngineBase:

    def close(self) -> None:
        """Close mp engine."""
        raise NotImplementedError('This method is not implemented.')

    def start_loop(self) -> None:
        """Start mp engine loop."""

    def end_session(self, session_id: int):
        """End session."""
        raise NotImplementedError('This method is not implemented.')

    def p2p_initialize(self, conn_request: DistServeInitRequest):
        """Init rdma link."""
        raise NotImplementedError('This method is not implemented.')

    def p2p_connect(self, conn_request: DistServeConnectionRequest):
        """rdma_connect."""
        raise NotImplementedError('This method is not implemented.')

    def p2p_drop_connect(self, drop_conn_request: DistServeDropConnectionRequest):
        """Drop connection.

        1. drop engine connection (zmq connection)
        2. TODO(JimyMa) drop RDMA Connection.
        """
        raise NotImplementedError('This method is not implemented.')

    def create_instance(self, cuda_stream_id=0):
        """Create instance."""
        raise NotImplementedError('This method is not implemented.')


class EngineInstanceBase:

    async def async_end(self, session_id: int):
        """End the given session."""
        raise NotImplementedError('This method is not implemented.')

    async def async_cancel(self, session_id: int):
        """Stop current streaming inference."""
        raise NotImplementedError('This method is not implemented.')

    async def async_stream_infer(self, *args, **kwargs):
        """Send stream inference request."""
        raise NotImplementedError('This method is not implemented.')
