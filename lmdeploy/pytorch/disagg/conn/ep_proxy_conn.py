# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import enum
import os
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

import aiohttp
import requests

from lmdeploy.logger import get_logger
from lmdeploy.pytorch.disagg.config import DistServeEngineConfig, EngineRole
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeCacheFreeRequest, DistServeConnectionRequest,
                                                   DistServeConnectionResponse, DistServeDropConnectionRequest,
                                                   DistServeInitRequest, DistServeInitResponse)
from lmdeploy.pytorch.disagg.messages import EPConnectionMessage

logger = get_logger('lmdeploy')

# Parse timeout env (string -> float) safely
_raw_timeout = os.getenv('AIOHTTP_TIMEOUT', None)
try:
    AIOHTTP_TIMEOUT: Optional[float] = float(_raw_timeout) if _raw_timeout else None
except ValueError:  # fallback silently and log
    logger.warning(f'Invalid AIOHTTP_TIMEOUT value: {_raw_timeout}, fallback to None')
    AIOHTTP_TIMEOUT = None


class EPConnectionStatus(enum.Enum):
    Disconnected = enum.auto()
    Connected = enum.auto()
    Connecting = enum.auto()


class EPConnectionState:
    """EPConnectionState (simple state holder with one event)."""

    def __init__(self, status: EPConnectionStatus, event: asyncio.Event):
        self.status = status
        self.event = event

    async def wait(self):
        await self.event.wait()

    def set_status(self, status: EPConnectionStatus):
        self.status = status


def get_server_api(url: str, api: str):
    return f'{url}/{api}'


class EPConnectionPool:
    """Constructing the link of E & P engine for the migration of KVCache.

    Note: we use Peer to Peer transportation in KVCache migration.
    Note: Lazy link construction is supported, which perform connection
        at the first LLM request. As a result, we don't need to construct
        PD Communication group when start a engine server.
    Note: we perform simple fault tolerance by checkpointing the session_id of a
        request which is under migrating and will trigger `gc` when the decode
        instanceis crushed.
    TODO (JimyMa): By now, only engines with same parallel configuration can be
        correctly connected.
    """

    # Maximum concurrent connections​​
    CONN_SEMAPHORE_SIZE = 2048

    def __init__(self):
        # all prefill and decode instances
        # TODO (JimyMa): Maybe encoding instances
        self.prefill_endpoints: Set[str] = set()
        self.encode_endpoints: Set[str] = set()

        # Links of PD Connection.
        self.pool: Dict[Tuple[str, str], EPConnectionState] = {}

        # put migrating session to `self.migration_session_shelf` for increasing fault tolerance
        # if a session is finished, then pop it from `self.migration_session_shelf`
        # if a decode instance is disconnected, then gc all blocks of these sessions in prefill instance.
        # use tuple (left, right) as key to align with drop() usage
        self.migration_session_shelf: Dict[Tuple[str, str], Set[int]] = defaultdict(set)

        # conn_perform handler queue
        self.waiting_conn: asyncio.Queue[Tuple[EPConnectionMessage, asyncio.Event]] = asyncio.Queue()

        # conn Registry Lock
        self.conn_lock = asyncio.Lock()

        # Connection Retry when failure
        self.max_retry_cnt = 8

        # trigger signal when conn request arrive.
        self.conn_req_event = asyncio.Event()

        # conn initialized signal
        self.initialized = False

    def reg_instance(self, role: EngineRole, endpoint: str):
        if role == EngineRole.Prefill:
            self.prefill_endpoints.add(endpoint)
        elif role == EngineRole.Encoder:
            self.encode_endpoints.add(endpoint)
        else:
            raise ValueError(f'Unsupported role: {role}')

    def dereg_instance(self, endpoint: str):
        # Symmetric cleanup for both roles
        if endpoint in self.encode_endpoints:
            dropped_key = [k for k in self.pool.keys() if k[0] == endpoint]
            for k in dropped_key:
                self.drop(k)
            self.encode_endpoints.remove(endpoint)
        elif endpoint in self.prefill_endpoints:
            dropped_key = [k for k in self.pool.keys() if k[1] == endpoint]
            for k in dropped_key:
                self.drop(k)
            # TODO(JimyMa): handle side-effect by kvcache migration
            self.prefill_endpoints.remove(endpoint)

    async def connect(self, conn_req: EPConnectionMessage):

        async def get_engine_config(server_endpoint):
            async with self.conn_sem:
                async with self.conn_sess.get(
                        get_server_api(server_endpoint, 'distserve/engine_info'),
                        timeout=self.aiotimeout,
                ) as resp:
                    result = await resp.json()
                    # model_validate_json expects a JSON string; result is already dict
                    logger.info(f'engine info from {server_endpoint}: {result}')
                    return DistServeEngineConfig.model_validate_json(result)

        async def p2p_initialize(server_endpoint, init_request: DistServeInitRequest) -> DistServeInitResponse:
            async with self.conn_sem:
                async with self.conn_sess.post(
                        get_server_api(server_endpoint, 'distserve/p2p_initialize'),
                        json=init_request.model_dump(mode='json'),
                        timeout=self.aiotimeout,
                ) as resp:
                    result = await resp.json()
                    logger.info(f'P2P Initialize response from {server_endpoint}: {result}')
                    return DistServeInitResponse.model_validate(result)

        async def p2p_connect(server_endpoint, conn_request: DistServeConnectionRequest) -> DistServeConnectionResponse:
            async with self.conn_sem:
                async with self.conn_sess.post(
                        get_server_api(server_endpoint, 'distserve/p2p_connect'),
                        json=conn_request.model_dump(mode='json'),
                        timeout=self.aiotimeout,
                ) as resp:
                    result = await resp.json()
                    return DistServeConnectionResponse.model_validate(result)

        async def conn_worker(conn_req: EPConnectionMessage, conn_event: asyncio.Event):
            # try:
            link = (conn_req.e_url, conn_req.p_url)
            logger.debug(f'{link} connecting...')
            # Step 1. Get Remote Engine Configuration
            prefill_engine_config = await get_engine_config(conn_req.p_url)
            encode_engine_config = await get_engine_config(conn_req.e_url)
            print(f'prefill_engine_config: {prefill_engine_config}')
            print(f'encode_engine_config: {encode_engine_config}')

            # encode 的 config 大部分字段为 空

            # Step 2. Construct Initialize Configuration
            prefill_init_req = DistServeInitRequest(
                protocol=conn_req.protocol,
                local_engine_id=conn_req.p_url,
                local_engine_config=prefill_engine_config,
                remote_engine_id=conn_req.e_url,
                remote_engine_config=encode_engine_config,
                rdma_config=conn_req.rdma_config,
                nvlink_config=conn_req.nvlink_config,
            )
            encode_init_req = DistServeInitRequest(
                protocol=conn_req.protocol,
                local_engine_id=conn_req.e_url,
                local_engine_config=encode_engine_config,
                remote_engine_id=conn_req.p_url,
                remote_engine_config=prefill_engine_config,
                rdma_config=conn_req.rdma_config,
                nvlink_config=conn_req.nvlink_config,
            )

            print(f'prefill_init_req: {prefill_init_req}')
            print(f'encode_init_req: {encode_init_req}')
            prefill_init_resp = await p2p_initialize(conn_req.p_url, prefill_init_req)
            encode_init_resp = await p2p_initialize(conn_req.e_url, encode_init_req)

            # Step 3. Connection
            encode_endpoint_conn_reqs = DistServeConnectionRequest(
                protocol=conn_req.protocol,
                remote_engine_id=conn_req.p_url,
                remote_engine_endpoint_info=prefill_init_resp.engine_endpoint_info,
                remote_kvtransfer_endpoint_info=prefill_init_resp.kvtransfer_endpoint_info)
            prefill_endpoint_conn_reqs = DistServeConnectionRequest(
                protocol=conn_req.protocol,
                remote_engine_id=conn_req.e_url,
                remote_engine_endpoint_info=encode_init_resp.engine_endpoint_info,
                remote_kvtransfer_endpoint_info=encode_init_resp.kvtransfer_endpoint_info)
            print(f'encode_endpoint_conn_reqs: {encode_endpoint_conn_reqs}')
            print(f'prefill_endpoint_conn_reqs: {prefill_endpoint_conn_reqs}')
            await p2p_connect(conn_req.p_url, prefill_endpoint_conn_reqs)
            await p2p_connect(conn_req.e_url, encode_endpoint_conn_reqs)
            self.pool[link].set_status(EPConnectionStatus.Connected)
            logger.debug(f'{(conn_req.e_url, conn_req.p_url)} connected')
            # except Exception as e:
            #     self.pool[link].set_status(EPConnectionStatus.Disconnected)
            #     logger.error(f'ep connection error: {e}')
            conn_event.set()

        async def wait_for_conn(conn_req: EPConnectionMessage, conn_event: asyncio.Event):
            await self.pool[(conn_req.e_url, conn_req.p_url)].event.wait()
            conn_event.set()

        async def _perform_conn():
            logger.debug('perform_conn start')
            while True:
                if self.waiting_conn.empty():
                    await self.conn_req_event.wait()

                self.conn_req_event.clear()

                while not self.waiting_conn.empty():
                    conn_req, conn_event = self.waiting_conn.get_nowait()
                    link = (conn_req.e_url, conn_req.p_url)
                    if link not in self.pool:
                        self.pool[link] = EPConnectionState(
                            EPConnectionStatus.Disconnected,
                            conn_event,
                        )
                    if self.pool[link].status == EPConnectionStatus.Connecting:
                        asyncio.create_task(wait_for_conn(conn_req, conn_event))
                    elif self.pool[link].status == EPConnectionStatus.Disconnected:
                        self.pool[link].set_status(EPConnectionStatus.Connecting)
                        asyncio.create_task(conn_worker(conn_req, conn_event))

        if not self.initialized:
            loop = asyncio.get_event_loop()
            loop.create_task(_perform_conn())
            self.conn_sem = asyncio.Semaphore(self.CONN_SEMAPHORE_SIZE)
            self.conn_sess = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit_per_host=256),
                timeout=aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT),
            )
            self.aiotimeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
            self.initialized = True

        print(f'EPConnectionPool connect called: {conn_req.e_url} <-> {conn_req.p_url}')
        self.reg_instance(EngineRole.Encoder, conn_req.e_url)
        self.reg_instance(EngineRole.Prefill, conn_req.p_url)

        cnt = 0
        while cnt < self.max_retry_cnt:
            if self.is_connected(conn_req.e_url, conn_req.p_url):
                return
            if cnt > 0:
                logger.warning(f'EP connection failure, retry cnt: {cnt}')
                # simple incremental backoff
                await asyncio.sleep(min(1.0, 0.2 * cnt))
            conn_event = asyncio.Event()
            self.waiting_conn.put_nowait((conn_req, conn_event))
            self.conn_req_event.set()
            await conn_event.wait()
            cnt += 1
        async with self.conn_lock:
            if (conn_req.e_url, conn_req.p_url) in self.pool:
                self.pool[conn_req.e_url, conn_req.p_url].set_status(EPConnectionStatus.Disconnected)
        raise TimeoutError('EPConnection Failure')

    def is_connected(self, e_url: str, p_url: str):
        link = self.pool.get((e_url, p_url), None)
        if not link:
            return False
        return link.status == EPConnectionStatus.Connected

    def drop(self, ep_key: Tuple[str, str]):
        left = ep_key[0]
        right = ep_key[1]

        def cache_free(server_endpoint, cache_free_request: DistServeCacheFreeRequest) -> None:
            try:
                requests.post(get_server_api(server_endpoint, 'distserve/free_cache'),
                              json=cache_free_request.model_dump(mode='json'))
            except Exception as e:
                logger.warning(f'error cache block free {server_endpoint, cache_free_request}. ErrorMsg: {str(e)}')

        def drop_connect(server_endpoint: str, p2p_disconnect_request: DistServeDropConnectionRequest):
            try:
                requests.post(get_server_api(server_endpoint, 'distserve/p2p_drop_connect'),
                              json=p2p_disconnect_request.model_dump(mode='json'))
            except Exception as e:
                logger.warning(f'error drop connect {server_endpoint, p2p_disconnect_request}. ErrorMsg: {str(e)}')

        # trigger gc
        logger.warning('cache block gc triggered.')
        try:
            for session_id in self.migration_session_shelf[(left, right)]:
                cache_free(left, DistServeCacheFreeRequest(remote_engine_id=left, remote_session_id=session_id))
        except Exception as e:
            logger.warning(f'gc error, ErrorMsg: {str(e)}')
        finally:
            self.migration_session_shelf.pop((left, right), None)

        # trigger p2p disconnect
        logger.warning('drop connection triggered.')
        try:
            drop_connect(left, DistServeDropConnectionRequest(engine_id=left, remote_engine_id=right))
            drop_connect(right, DistServeDropConnectionRequest(engine_id=right, remote_engine_id=left))
        except Exception as e:
            logger.warning(f'p2p disconnect error, ErrorMsg: {str(e)}')

        self.pool.pop((left, right), None)

    async def close(self):
        if getattr(self, 'initialized', False):
            try:
                await self.conn_sess.close()
            except Exception as e:
                logger.warning(f'EPConnectionPool close error: {e}')
