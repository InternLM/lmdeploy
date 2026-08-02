# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import zmq
import zmq.asyncio
from pydantic import ValidationError

from lmdeploy.logger import get_logger
from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeCacheFreeRequest,
    DistServeConnectionRequest,
    DistServeConnectionResponse,
    DistServeConnectionStatus,
    DistServeDropConnectionRequest,
    DistServeEngineEndpointInfo,
    DistServeInitRequest,
    DistServeInitResponse,
    DistServeKVTransferEndpointInfo,
)
from lmdeploy.pytorch.engine.executor.dist_utils import find_available_port

if TYPE_CHECKING:
    from lmdeploy.pytorch.engine.engine import Engine

logger = get_logger('lmdeploy')


class EngineP2PConnection:

    def __init__(self, engine: 'Engine'):
        self.engine: Engine = engine
        self.p2p_conn_ctx: dict[str, zmq.asyncio.Context] = {}
        self.p2p_sender: dict[str, zmq.asyncio.Socket] = {}
        self.p2p_receiver: dict[str, zmq.asyncio.Socket] = {}

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

        kvtransfer_endpoint_info: list[DistServeKVTransferEndpointInfo] = self.engine.executor.p2p_initialize(
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
        req = DistServeCacheFreeRequest(remote_engine_id=remote_engine_id, remote_session_id=remote_session_id)
        # Use JSON rather than pickle on the wire: recv_pyobj()/send_pyobj() call
        # pickle.loads() on peer-supplied bytes, which is remote code execution.
        await self.p2p_sender[remote_engine_id].send_json(req.model_dump())

    async def handle_zmq_recv(self, remote_engine_id: str):
        receiver = self.p2p_receiver[remote_engine_id]
        while True:
            # recv_json() decodes with json.loads (no code execution); model_validate
            # then enforces the DistServeCacheFreeRequest schema before the payload is
            # used, replacing the old recv_pyobj() -> pickle.loads() RCE path. Malformed
            # or off-schema payloads are logged and skipped so a single bad message
            # cannot tear down the receive loop.
            try:
                raw = await receiver.recv_json()
                req = DistServeCacheFreeRequest.model_validate(raw)
            except (ValueError, ValidationError) as e:
                logger.error(f'invalid zmq request from {remote_engine_id}: {e}')
                continue
            session_id = req.remote_session_id
            if session_id in self.engine.scheduler.sessions:
                self.engine.end_session(session_id=session_id)
            else:
                logger.error(f'invalid free, {remote_engine_id}, {session_id}')

    async def zmq_disconnect(self, remote_engine_id: str):
        self.p2p_receiver[remote_engine_id].close()
        self.p2p_sender[remote_engine_id].close()
        self.p2p_conn_ctx[remote_engine_id].term()
