import os
import enum
from typing import List, Tuple

import asyncio
import aiohttp
import requests

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


class PDConnectionStatus(enum.Enum):
    Disconnect = enum.auto()
    Connected = enum.auto()


class PDConnectionPool:
    def __init__(self):
        self.pool = {}
        self.initialized = False

    async def connect(
        self,
        p_url: str,
        d_url: str,
        protocol: MigrationTransportProtocol=MigrationTransportProtocol.RDMA,
        *,
        ib_port: int = 1,
        rdma_link_type: str = None,
    ):
        """pd consolidation."""
        if not self.initialized:
            if not self.initialized:
                self.conn_sem = asyncio.Semaphore(1024)
            self.conn_sess = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit_per_host=32),
                timeout=aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
            )
            self.aiotimeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
            self.initialized = True

        rdma_link_type = rdma_link_type or "Ethernet"
        if rdma_link_type == "IB":
            raise NotImplementedError("Link Type IB is not supported bynow.")

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
                    timeout=self.aiotimeout
                ) as resp:
                    return await resp.json()

        async def p2p_connect(server_endpoint, conn_request: List[MigrationConnectionRequest]):
            async with self.conn_sem:
                async with self.conn_sess.post(
                    get_server_api(server_endpoint, "distserve/p2p_connect"),
                    timeout=5,
                    json=[req.model_dump(mode="json") for req in conn_request]
                ) as resp:
                    return await resp.json()

        logger.info(f"{(p_url, d_url)} connecting...")
        # Step 1. Get Remote Engine Configuration
        prefill_engine_config = await get_engine_config(p_url)
        decode_engine_config = await get_engine_config(d_url)

        # Note: Only tp is supported by now
        assert prefill_engine_config.dp_size is None
        assert prefill_engine_config.pp_size is None

        assert decode_engine_config.dp_size is None
        assert decode_engine_config.pp_size is None

        # Note: Only Same Parallel Configurations are supported by now
        assert prefill_engine_config.tp_size == decode_engine_config.tp_size

        # Step 2. Construct Initialize Configuration
        prefill_init_req = MigrationInitRequest(
            protocol=protocol,
            remote_engine_id=d_url,
            remote_engine_config=decode_engine_config,
        )
        decode_init_req = MigrationInitRequest(
            protocol=protocol,
            remote_engine_id=p_url,
            remote_engine_config=prefill_engine_config,
        )

        if protocol == MigrationTransportProtocol.RDMA:
            rdma_init_req = RDMAInitRequest(
                device_name=None,
                ib_port=ib_port,
                link_type=rdma_link_type
            )
            prefill_init_req.rdma_init_request = rdma_init_req
            decode_init_req.rdma_init_request = rdma_init_req
        else:
            raise NotImplementedError

        prefill_endpoint_info = await p2p_initialize(p_url, prefill_init_req)
        decode_endpoint_info = await p2p_initialize(d_url, decode_init_req)

        # Step 3. Connection
        if protocol == MigrationTransportProtocol.RDMA:
            prefill_endpoint_conn_reqs = [
                MigrationConnectionRequest(
                    protocol=protocol,
                    remote_engine_id=d_url,
                    remote_endpoint_info=info
                )
                for info in decode_endpoint_info
            ]
            decode_endpoint_conn_reqs = [
                MigrationConnectionRequest(
                    protocol=protocol,
                    remote_engine_id=p_url,
                    remote_endpoint_info=info
                )
                for info in prefill_endpoint_info
            ]
            await p2p_connect(p_url, prefill_endpoint_conn_reqs)
            await p2p_connect(d_url, decode_endpoint_conn_reqs)
            logger.info(f"{(p_url, d_url)} connected")
        self.pool[(p_url, d_url)] = PDConnectionStatus.Connected

    def get(self, left: str, right: str):
        return self.pool.get((left, right), None)

    def drop(self, left: str, right: str):
        self.pool.pop((left, right))
