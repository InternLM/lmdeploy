# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
from typing import TYPE_CHECKING, Dict, List
from urllib.parse import urlparse

import zmq
import zmq.asyncio

from lmdeploy.logger import get_logger
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeCacheFreeRequest, DistServeConnectionRequest,
                                                   DistServeConnectionResponse, DistServeConnectionStatus,
                                                   DistServeDropConnectionRequest, DistServeEngineEndpointInfo,
                                                   DistServeInitRequest, DistServeInitResponse,
                                                   DistServeKVTransferEndpointInfo)
from lmdeploy.pytorch.engine.executor.dist_utils import find_available_port

if TYPE_CHECKING:
    from lmdeploy.pytorch.engine.engine import Engine

logger = get_logger('lmdeploy')


class EngineP2PConnection:

    def __init__(self, engine: 'Engine'):
        self.engine: Engine = engine
        self.p2p_conn_ctx: Dict[str, zmq.asyncio.Context] = {}
        self.p2p_sender: Dict[str, zmq.asyncio.Socket] = {}
        self.p2p_receiver: Dict[str, zmq.asyncio.Socket] = {}

        self.use_unique_kvtransfer_engine = os.environ.get('LMDEPLOY_USE_UNIQUE_KVTRANSFER_ENGINE', False)

    def p2p_initialize(self, init_request: DistServeInitRequest):
        ctx = zmq.asyncio.Context(2)
        sender = ctx.socket(zmq.PUSH)
        sender_port = find_available_port()
        sender_hostname = urlparse(init_request.local_engine_id).hostname
        zmq_address = f'tcp://{sender_hostname}:{sender_port}'
        sender.bind(zmq_address)
        receiver = ctx.socket(zmq.PULL)

        self.p2p_conn_ctx[init_request.remote_engine_id] = ctx
        self.p2p_sender[init_request.remote_engine_id] = sender
        self.p2p_receiver[init_request.remote_engine_id] = receiver

        kvtransfer_endpoint_info: List[DistServeKVTransferEndpointInfo] = self.engine.executor.p2p_initialize(
            init_request)

        return DistServeInitResponse(engine_endpoint_info=DistServeEngineEndpointInfo(zmq_address=zmq_address),
                                     kvtransfer_endpoint_info=kvtransfer_endpoint_info,
                                     status=DistServeConnectionStatus.SUCCESS)

    def p2p_connect(self, conn_request: DistServeConnectionRequest):
        self.p2p_receiver[conn_request.remote_engine_id].connect(conn_request.remote_engine_endpoint_info.zmq_address)
        self.engine.executor.p2p_connect(remote_engine_id=conn_request.remote_engine_id,
                                         conn_request=conn_request.remote_kvtransfer_endpoint_info)
        event_loop = asyncio.get_event_loop()
        event_loop.create_task(self.handle_zmq_recv(conn_request.remote_engine_id))
        return DistServeConnectionResponse(status=DistServeConnectionStatus.SUCCESS)

    def p2p_drop_connect(self, drop_conn_request: DistServeDropConnectionRequest):
        # TODO (JimyMa): drop RDMA Connection
        self.zmq_disconnect(drop_conn_request.remote_engine_id)
        return {'success': True}

    async def zmq_send(self, remote_engine_id: str, remote_session_id: int):
        await self.p2p_sender[remote_engine_id].send_pyobj(
            DistServeCacheFreeRequest(remote_engine_id=remote_engine_id, remote_session_id=remote_session_id))

    async def handle_zmq_recv(self, remote_engine_id: str):
        while True:
            req: DistServeCacheFreeRequest = await self.p2p_receiver[remote_engine_id].recv_pyobj()
            if isinstance(req, DistServeCacheFreeRequest):
                session_id = req.remote_session_id
                if session_id in self.engine.scheduler.sessions:
                    self.engine.scheduler.end_session(session_id=session_id)
                else:
                    logger.error(f'invalid free, {remote_engine_id}, {session_id}')
            else:
                raise ValueError(f'Unsupported zmq request {type(req)}')

    async def zmq_disconnect(self, remote_engine_id: str):
        self.p2p_receiver[remote_engine_id].close()
        self.p2p_sender[remote_engine_id].close()
        self.p2p_conn_ctx[remote_engine_id].term()
