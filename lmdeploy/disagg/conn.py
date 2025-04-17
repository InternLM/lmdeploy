import os
import enum
from typing import List, Optional, Tuple

import asyncio
import aiohttp
import requests

from pydantic import BaseModel

from lmdeploy.logger import get_logger

from lmdeploy.disagg.messages import (
    DisaggEngineConfig,
    MigrationInitRequest,
    TCPInitRequest,
    RDMAInitRequest,
    NVLinkInitRequest,
    MigrationTransportProtocol,
    MigrationConnectionRequest
)

logger = get_logger("lmdeploy")


AIOHTTP_TIMEOUT = os.getenv('AIOHTTP_TIMEOUT', None)


class PDConnectionRequest(BaseModel):
    p_url: str
    d_url: str
    protocol: MigrationTransportProtocol = MigrationTransportProtocol.RDMA
    tcp_init_request: Optional[TCPInitRequest] = None
    rdma_init_request: Optional[RDMAInitRequest] = RDMAInitRequest()
    nvlink_init_request: Optional[NVLinkInitRequest] = None


class PDConnectionStatus(enum.Enum):
    Disconnected = enum.auto()
    Connected = enum.auto()
    Connecting = enum.auto()


class PDConnectionPool:
    def __init__(self):
        self.pool = {}
        self.initialized = False
        self.waiting_conn: asyncio.Queue[Tuple[PDConnectionRequest, asyncio.Event]] = asyncio.Queue()
        self.conn_req_event = asyncio.Event()
        self.conn_lock = asyncio.Lock()
        self.max_retry_cnt = 8

    async def perform_conn(self):
        logger.info("perform_conn start")
        def get_server_api(url:str, api: str):
                return f"{url}/{api}"
        
        async def get_engine_config(server_endpoint):
            async with self.conn_sem:
                async with self.conn_sess.get(
                    get_server_api(server_endpoint, "distserve/engine_info"),
                    timeout=self.aiotimeout
                ) as resp:
                    return DisaggEngineConfig.model_validate_json(await resp.json())

        async def p2p_initialize(server_endpoint, init_request: MigrationInitRequest):
            async with self.conn_sem:
                async with self.conn_sess.post(
                    get_server_api(server_endpoint, "distserve/p2p_initialize"),
                    json=init_request.model_dump(mode="json"),
                    timeout=self.aiotimeout,
                ) as resp:
                    return await resp.json()

        async def p2p_connect(server_endpoint, conn_request: List[MigrationConnectionRequest]):
            async with self.conn_sem:
                async with self.conn_sess.post(
                    get_server_api(server_endpoint, "distserve/p2p_connect"),
                    json=[req.model_dump(mode="json") for req in conn_request],
                    timeout=self.aiotimeout,
                ) as resp:
                    return await resp.json()

        async def conn_worker(conn_req: PDConnectionRequest, conn_event: asyncio.Event):
            try:
                logger.info(f"{(conn_req.p_url, conn_req.d_url)} connecting...")
                # Step 1. Get Remote Engine Configuration
                prefill_engine_config = await get_engine_config(conn_req.p_url)
                decode_engine_config = await get_engine_config(conn_req.d_url)

                # Note: Only Same Parallel Configurations are supported by now
                assert prefill_engine_config.tp_size == decode_engine_config.tp_size

                # Step 2. Construct Initialize Configuration
                prefill_init_req = MigrationInitRequest(
                    protocol=conn_req.protocol,
                    remote_engine_id=conn_req.d_url,
                    remote_engine_config=decode_engine_config,
                )
                decode_init_req = MigrationInitRequest(
                    protocol=conn_req.protocol,
                    remote_engine_id=conn_req.p_url,
                    remote_engine_config=prefill_engine_config,
                )

                if conn_req.protocol == MigrationTransportProtocol.RDMA:
                    prefill_init_req.rdma_init_request = conn_req.rdma_init_request
                    decode_init_req.rdma_init_request = conn_req.rdma_init_request
                else:
                    raise NotImplementedError

                prefill_endpoint_info = await p2p_initialize(conn_req.p_url, prefill_init_req)
                decode_endpoint_info = await p2p_initialize(conn_req.d_url, decode_init_req)

                # Step 3. Connection
                if conn_req.protocol == MigrationTransportProtocol.RDMA:
                    prefill_endpoint_conn_reqs = [
                        MigrationConnectionRequest(
                            protocol=conn_req.protocol,
                            remote_engine_id=conn_req.d_url,
                            remote_endpoint_info=info
                        )
                        for info in decode_endpoint_info
                    ]
                    decode_endpoint_conn_reqs = [
                        MigrationConnectionRequest(
                            protocol=conn_req.protocol,
                            remote_engine_id=conn_req.p_url,
                            remote_endpoint_info=info
                        )
                        for info in prefill_endpoint_info
                    ]
                    await p2p_connect(conn_req.p_url, prefill_endpoint_conn_reqs)
                    await p2p_connect(conn_req.d_url, decode_endpoint_conn_reqs)
                logger.info(f"{(conn_req.p_url, conn_req.d_url)} connected")
                self.pool[(conn_req.p_url, conn_req.d_url)] = PDConnectionStatus.Connected
            except:
                self.pool[(conn_req.p_url, conn_req.d_url)] = PDConnectionStatus.Disconnected
            conn_event.set()

        async def wait_for_conn(conn_req, conn_event):
            while True:
                if self.pool[(conn_req.p_url, conn_req.d_url)] == PDConnectionStatus.Connected:
                    conn_event.set()
                    return
                await asyncio.sleep(0.5)

        while True:
            if self.waiting_conn.empty():
                await self.conn_req_event.wait()

            self.conn_req_event.clear()

            while not self.waiting_conn.empty():
                conn_req, conn_event = self.waiting_conn.get_nowait()
                link = (conn_req.p_url, conn_req.d_url)
                if link not in self.pool or self.pool.get(link, None) == PDConnectionStatus.Disconnected:
                    async with self.conn_lock:
                        self.pool[link] = PDConnectionStatus.Connecting
                    asyncio.create_task(conn_worker(conn_req, conn_event))
                elif self.pool.get(link, None) == PDConnectionStatus.Connected:
                    conn_event.set()
                elif self.pool.get(link, None) == PDConnectionStatus.Connecting:
                    asyncio.create_task(wait_for_conn(conn_req, conn_event))

    async def connect(self, conn_req: PDConnectionRequest):
        """pd consolidation."""
        if not self.initialized:
            loop = asyncio.get_event_loop()
            loop.create_task(self.perform_conn())
            self.conn_sem = asyncio.Semaphore(1024)
            self.conn_sess = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit_per_host=256),
                timeout=aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
            )
            self.aiotimeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
            self.initialized = True
        cnt = 0
        while cnt < self.max_retry_cnt:
            conn_event = asyncio.Event()
            self.waiting_conn.put_nowait((conn_req, conn_event))
            self.conn_req_event.set()
            await conn_event.wait()
            if self.is_connected(conn_req.p_url, conn_req.d_url):
                return
            logger.warn("Retry Conn")
            cnt += 1
        else:
            raise TimeoutError("PDConnection Failure")

    def is_connected(self, p_url: str, d_url: str):
        return self.pool.get((p_url, d_url), None) == PDConnectionStatus.Connected

    def drop(self, left: str, right: str):
        self.pool.pop((left, right), None)
