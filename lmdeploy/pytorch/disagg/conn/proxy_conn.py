# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import enum
import json
import os
from typing import Dict, List, Tuple

import aiohttp

from lmdeploy.logger import get_logger
from lmdeploy.pytorch.disagg.config import DistServeEngineConfig
from lmdeploy.pytorch.disagg.messages import PDConnectionMessage
from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeInitRequest,
    DistServeInitResponse,
    DistServeConnectionRequest,
    DistServeConnectionResponse,
)

logger = get_logger("lmdeploy")

AIOHTTP_TIMEOUT = os.getenv("AIOHTTP_TIMEOUT", None)


class PDConnectionStatus(enum.Enum):
    Disconnected = enum.auto()
    Connected = enum.auto()
    Connecting = enum.auto()


class PDConnectionState:
    """PDConnectionState."""

    def __init__(self, status: PDConnectionStatus, event: asyncio.Event):
        self.status = status
        self.event = event

    async def wait(self):
        await self.event.wait()

    def set_status(self, status: PDConnectionStatus):
        self.status = status


class PDConnectionPool:
    """Constructing the link of Prefill and Decode engine for the migration of
    KVCache.

    Note: we use Peer to Peer transportation in KVCache migration.
    Note: Lazy link construction is supported, which perform connection
        at the first LLM request. As a result, we don't need to construct
        PD Communication group when start a engine server.
    Warning: By now, only engines with same parallel configuration can be
        correctly connected.
    """

    def __init__(self):
        # Links of PD Connection.
        self.pool: Dict[Tuple[str, str], PDConnectionState] = {}

        # conn_perform handler queue
        self.waiting_conn: asyncio.Queue[Tuple[PDConnectionMessage, asyncio.Event]] = (
            asyncio.Queue()
        )

        # conn Registry Lock
        self.conn_lock = asyncio.Lock()

        # Connection Retry when failure
        self.max_retry_cnt = 8

        # trigger signal when conn request arrive.
        self.conn_req_event = asyncio.Event()

        # conn initialized signal
        self.initialized = False

    async def perform_conn(self):

        def get_server_api(url: str, api: str):
            return f"{url}/{api}"

        async def get_engine_config(server_endpoint):
            async with self.conn_sem:
                async with self.conn_sess.get(
                    get_server_api(server_endpoint, "distserve/engine_info"),
                    timeout=self.aiotimeout,
                ) as resp:
                    result = await resp.json()
                    return DistServeEngineConfig.model_validate_json(result)

        async def p2p_initialize(server_endpoint, init_request: DistServeInitRequest) -> DistServeInitResponse:
            async with self.conn_sem:
                async with self.conn_sess.post(
                    get_server_api(server_endpoint, "distserve/p2p_initialize"),
                    json=init_request.model_dump(mode="json"),
                    timeout=self.aiotimeout,
                ) as resp:
                    result = await resp.json()

                    return DistServeInitResponse.model_validate(result)

        async def p2p_connect(
            server_endpoint, conn_request: DistServeConnectionRequest
        ) -> DistServeConnectionResponse:
            async with self.conn_sem:
                async with self.conn_sess.post(
                    get_server_api(server_endpoint, "distserve/p2p_connect"),
                    json=conn_request.model_dump(mode="json"),
                    timeout=self.aiotimeout,
                ) as resp:
                    result = await resp.json()
                    return DistServeConnectionResponse.model_validate(result)

        async def conn_worker(conn_req: PDConnectionMessage, conn_event: asyncio.Event):
            # try:
            link = (conn_req.p_url, conn_req.d_url)
            logger.debug(f"{link} connecting...")
            # Step 1. Get Remote Engine Configuration
            prefill_engine_config = await get_engine_config(conn_req.p_url)
            decode_engine_config = await get_engine_config(conn_req.d_url)

            # Note: Only Same Parallel Configurations are supported by now
            assert prefill_engine_config.tp_size == decode_engine_config.tp_size

            # Step 2. Construct Initialize Configuration
            prefill_init_req = DistServeInitRequest(
                protocol=conn_req.protocol,
                local_engine_id=conn_req.p_url,
                local_engine_config=prefill_engine_config,
                remote_engine_id=conn_req.d_url,
                remote_engine_config=decode_engine_config,
                rdma_config=conn_req.rdma_config,
                nvlink_config=conn_req.nvlink_config,
            )
            decode_init_req = DistServeInitRequest(
                protocol=conn_req.protocol,
                local_engine_id=conn_req.d_url,
                local_engine_config=decode_engine_config,
                remote_engine_id=conn_req.p_url,
                remote_engine_config=prefill_engine_config,
                rdma_config=conn_req.rdma_config,
                nvlink_config=conn_req.nvlink_config,
            )

            prefill_init_resp = await p2p_initialize(
                conn_req.p_url, prefill_init_req
            )
            decode_init_resp = await p2p_initialize(
                conn_req.d_url, decode_init_req
            )

            # Step 3. Connection
            prefill_endpoint_conn_reqs = DistServeConnectionRequest(
                protocol=conn_req.protocol,
                remote_engine_id=conn_req.d_url,
                remote_engine_endpoint_info=decode_init_resp.engine_endpoint_info,
                remote_kvtransfer_endpoint_info=decode_init_resp.kvtransfer_endpoint_info
            )
            decode_endpoint_conn_reqs = DistServeConnectionRequest(
                protocol=conn_req.protocol,
                remote_engine_id=conn_req.p_url,
                remote_engine_endpoint_info=prefill_init_resp.engine_endpoint_info,
                remote_kvtransfer_endpoint_info=prefill_init_resp.kvtransfer_endpoint_info
            )
            await p2p_connect(conn_req.p_url, prefill_endpoint_conn_reqs)
            await p2p_connect(conn_req.d_url, decode_endpoint_conn_reqs)
            self.pool[link].set_status(PDConnectionStatus.Connected)
            logger.debug(f"{(conn_req.p_url, conn_req.d_url)} connected")
            # except Exception as e:
            #     self.pool[link].set_status(PDConnectionStatus.Disconnected)
            #     logger.error(f"pd connection error: {e}")
            conn_event.set()

        async def wait_for_conn(
            conn_req: PDConnectionMessage, conn_event: asyncio.Event
        ):
            await self.pool[(conn_req.p_url, conn_req.d_url)].event.wait()
            conn_event.set()

        logger.debug("perform_conn start")
        while True:
            if self.waiting_conn.empty():
                await self.conn_req_event.wait()

            self.conn_req_event.clear()

            while not self.waiting_conn.empty():
                conn_req, conn_event = self.waiting_conn.get_nowait()
                link = (conn_req.p_url, conn_req.d_url)
                if link not in self.pool:
                    self.pool[link] = PDConnectionState(
                        PDConnectionStatus.Disconnected,
                        conn_event,
                    )
                if self.pool[link].status == PDConnectionStatus.Connecting:
                    asyncio.create_task(wait_for_conn(conn_req, conn_event))
                elif self.pool[link].status == PDConnectionStatus.Disconnected:
                    self.pool[link].set_status(PDConnectionStatus.Connecting)
                    asyncio.create_task(conn_worker(conn_req, conn_event))

    async def connect(self, conn_req: PDConnectionMessage):
        if not self.initialized:
            loop = asyncio.get_event_loop()
            loop.create_task(self.perform_conn())
            self.conn_sem = asyncio.Semaphore(1024)
            self.conn_sess = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit_per_host=256),
                timeout=aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT),
            )
            self.aiotimeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
            self.initialized = True
        cnt = 0
        while cnt < self.max_retry_cnt:
            if self.is_connected(conn_req.p_url, conn_req.d_url):
                return
            if cnt > 0:
                logger.warning(f"Connection failure, retry cnt: {cnt}")
            conn_event = asyncio.Event()
            self.waiting_conn.put_nowait((conn_req, conn_event))
            self.conn_req_event.set()
            await conn_event.wait()
            cnt += 1
        async with self.conn_lock:
            self.pool[conn_req.p_url, conn_req.d_url].set_status(
                PDConnectionStatus.Disconnected
            )
        raise TimeoutError("PDConnection Failure")

    def is_connected(self, p_url: str, d_url: str):
        link = self.pool.get((p_url, d_url), None)
        if not link:
            return False
        return link.status == PDConnectionStatus.Connected

    def drop(self, left: str, right: str):
        self.pool.pop((left, right), None)
