# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import enum
import os
from collections import defaultdict

import aiohttp
import requests

from lmdeploy.logger import get_logger
from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.disagg.config import DistServeEngineConfig, EngineRole
from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeCacheFreeRequest,
    DistServeConnectionRequest,
    DistServeConnectionResponse,
    DistServeDropConnectionRequest,
    DistServeInitRequest,
    DistServeInitResponse,
)
from lmdeploy.pytorch.disagg.messages import PDConnectionMessage

logger = get_logger('lmdeploy')

AIOHTTP_TIMEOUT = os.getenv('AIOHTTP_TIMEOUT', None)

# Bound waiters so connect()/warmup cannot hang forever when a handshake is stuck.
# Set LMDEPLOY_PD_CONN_WAIT_TIMEOUT=0 to restore unlimited wait.
CONNECT_WAIT_TIMEOUT = _envs.pd_conn_wait_timeout if _envs.pd_conn_wait_timeout > 0 else None


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


def get_server_api(url: str, api: str):
    return f'{url}/{api}'


class PDConnectionPool:
    """Constructing the link of Prefill and Decode engine for the migration of
    KVCache.

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
    WAITING_CONN_MAXSIZE = 2048
    WARMUP_CONNECT_CONCURRENCY = 128

    def __init__(self):
        # all prefill and decode instances
        # TODO (JimyMa): Maybe encoding instances
        self.prefill_endpoints: set[str] = set()
        self.decode_endpoints: set[str] = set()

        # Links of PD Connection.
        self.pool: dict[tuple[str, str], PDConnectionState] = {}

        # put migrating session to `self.migration_session_shelf` for increasing fault tolerance
        # if a session is finished, then pop it from `self.migration_session_shelf`
        # if a decode instance is disconnected, then gc all blocks of these sessions in prefill instance.
        self.migration_session_shelf: dict[str, set[int]] = defaultdict(set)

        # conn_perform handler queue. maxsize is a safety net; single-flight
        # keeps the queue at one in-flight attempt per P-D link.
        self.waiting_conn: asyncio.Queue[tuple[PDConnectionMessage, asyncio.Event]] = asyncio.Queue(
            maxsize=self.WAITING_CONN_MAXSIZE
        )

        # conn Registry Lock
        self.conn_lock = asyncio.Lock()

        # Connection Retry when failure
        self.max_retry_cnt = 8

        # trigger signal when conn request arrive.
        self.conn_req_event = asyncio.Event()

        # conn initialized signal
        self.initialized = False

        self.connect_wait_timeout = CONNECT_WAIT_TIMEOUT
        self._perform_conn_task: asyncio.Task | None = None
        self._bg_tasks: set[asyncio.Task] = set()
        self._warmup_lock = asyncio.Lock()
        self._warmup_task: asyncio.Task | None = None

    def reg_instance(self, role: EngineRole, endpoint: str):
        if role == EngineRole.Prefill:
            self.prefill_endpoints.add(endpoint)
        elif role == EngineRole.Decode:
            self.decode_endpoints.add(endpoint)
        else:
            raise ValueError(f'Unsupported role: {role}')

    def dereg_instance(self, endpoint: str):
        if endpoint in self.prefill_endpoints:
            self.prefill_endpoints.remove(endpoint)
        elif endpoint in self.decode_endpoints:
            dropped_key = []
            for conn_key in self.pool.keys():
                if conn_key[1] == endpoint:
                    dropped_key.append(conn_key)
            for k in dropped_key:
                self.drop(k)
            # TODO(JimyMa): handle side-effect by kvcache migration
            self.decode_endpoints.remove(endpoint)

    def shelf_prefill_session(self, conn_key: tuple[str, str], session_id: int):
        self.migration_session_shelf[conn_key].add(session_id)

    def unshelf_prefill_session(self, conn_key: tuple[str, str], session_id: int):
        self.migration_session_shelf[conn_key].remove(session_id)

    def _ensure_initialized(self):
        if self.initialized:
            return
        loop = asyncio.get_running_loop()
        self._perform_conn_task = loop.create_task(self._perform_conn(), name='pd-conn-dispatcher')
        self.conn_sem = asyncio.Semaphore(self.CONN_SEMAPHORE_SIZE)
        self.conn_sess = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit_per_host=256),
            timeout=aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT),
        )
        self.aiotimeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
        self.initialized = True

    async def _get_engine_config(self, server_endpoint):
        async with self.conn_sem:
            async with self.conn_sess.get(
                get_server_api(server_endpoint, 'distserve/engine_info'),
                timeout=self.aiotimeout,
            ) as resp:
                result = await resp.json()
                return DistServeEngineConfig.model_validate_json(result)

    async def _p2p_initialize(self, server_endpoint, init_request: DistServeInitRequest) -> DistServeInitResponse:
        async with self.conn_sem:
            async with self.conn_sess.post(
                get_server_api(server_endpoint, 'distserve/p2p_initialize'),
                json=init_request.model_dump(mode='json'),
                timeout=self.aiotimeout,
            ) as resp:
                result = await resp.json()
                return DistServeInitResponse.model_validate(result)

    async def _p2p_connect(
        self, server_endpoint, conn_request: DistServeConnectionRequest
    ) -> DistServeConnectionResponse:
        async with self.conn_sem:
            async with self.conn_sess.post(
                get_server_api(server_endpoint, 'distserve/p2p_connect'),
                json=conn_request.model_dump(mode='json'),
                timeout=self.aiotimeout,
            ) as resp:
                result = await resp.json()
                return DistServeConnectionResponse.model_validate(result)

    async def _handshake(self, conn_req: PDConnectionMessage):
        """Perform the P-D handshake.

        Overridable in tests.
        """
        # Step 1. Get Remote Engine Configuration
        prefill_engine_config = await self._get_engine_config(conn_req.p_url)
        decode_engine_config = await self._get_engine_config(conn_req.d_url)

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

        prefill_init_resp = await self._p2p_initialize(conn_req.p_url, prefill_init_req)
        decode_init_resp = await self._p2p_initialize(conn_req.d_url, decode_init_req)

        # Step 3. Connection
        prefill_endpoint_conn_reqs = DistServeConnectionRequest(
            protocol=conn_req.protocol,
            remote_engine_id=conn_req.d_url,
            remote_engine_endpoint_info=decode_init_resp.engine_endpoint_info,
            remote_kvtransfer_endpoint_info=decode_init_resp.kvtransfer_endpoint_info,
        )
        decode_endpoint_conn_reqs = DistServeConnectionRequest(
            protocol=conn_req.protocol,
            remote_engine_id=conn_req.p_url,
            remote_engine_endpoint_info=prefill_init_resp.engine_endpoint_info,
            remote_kvtransfer_endpoint_info=prefill_init_resp.kvtransfer_endpoint_info,
        )
        await self._p2p_connect(conn_req.p_url, prefill_endpoint_conn_reqs)
        await self._p2p_connect(conn_req.d_url, decode_endpoint_conn_reqs)

    async def _conn_worker(self, conn_req: PDConnectionMessage, conn_event: asyncio.Event):
        link = (conn_req.p_url, conn_req.d_url)
        logger.debug(f'{link} connecting...')
        try:
            await self._handshake(conn_req)
            async with self.conn_lock:
                state = self.pool.get(link)
                if state is not None:
                    state.set_status(PDConnectionStatus.Connected)
            logger.debug(f'{link} connected')
        except Exception as e:
            async with self.conn_lock:
                state = self.pool.get(link)
                if state is not None and state.status != PDConnectionStatus.Connected:
                    state.set_status(PDConnectionStatus.Disconnected)
            logger.error(f'pd connection error: {e}')
        finally:
            conn_event.set()

    async def _perform_conn(self):
        logger.debug('perform_conn start')
        while True:
            if self.waiting_conn.empty():
                await self.conn_req_event.wait()

            self.conn_req_event.clear()

            while not self.waiting_conn.empty():
                conn_req, conn_event = self.waiting_conn.get_nowait()
                link = (conn_req.p_url, conn_req.d_url)
                async with self.conn_lock:
                    state = self.pool.get(link)
                    status = state.status if state is not None else PDConnectionStatus.Disconnected
                    owner_event = state.event if state is not None else None
                if status == PDConnectionStatus.Connected:
                    conn_event.set()
                    continue
                # Single-flight: only the owner event starts a handshake. Duplicate
                # enqueues (if any) wait on the existing Connecting event in connect().
                if status == PDConnectionStatus.Connecting and owner_event is not conn_event:
                    continue
                task = asyncio.create_task(
                    self._conn_worker(conn_req, conn_event), name=f'pd-conn-worker:{link[0]}->{link[1]}'
                )
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)

    async def _join_or_start(self, conn_req: PDConnectionMessage) -> asyncio.Event | None:
        """Return the in-flight event for this P-D link, starting at most one
        handshake.

        Returns None when the link is already Connected.
        """
        link = (conn_req.p_url, conn_req.d_url)
        async with self.conn_lock:
            self._ensure_initialized()
            if self.is_connected(conn_req.p_url, conn_req.d_url):
                return None
            state = self.pool.get(link)
            if state is not None and state.status == PDConnectionStatus.Connecting:
                return state.event
            conn_event = asyncio.Event()
            if state is None:
                self.pool[link] = PDConnectionState(PDConnectionStatus.Connecting, conn_event)
            else:
                state.set_status(PDConnectionStatus.Connecting)
                state.event = conn_event
            try:
                self.waiting_conn.put_nowait((conn_req, conn_event))
            except asyncio.QueueFull:
                state = self.pool[link]
                state.set_status(PDConnectionStatus.Disconnected)
                raise TimeoutError('PDConnection waiting queue is full')
            self.conn_req_event.set()
            return conn_event

    async def _wait_for_event(self, conn_event: asyncio.Event):
        timeout = self.connect_wait_timeout
        if timeout is None:
            await conn_event.wait()
            return
        try:
            await asyncio.wait_for(conn_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f'PDConnection wait timeout after {timeout}s')

    async def connect(self, conn_req: PDConnectionMessage):

        self.reg_instance(EngineRole.Prefill, conn_req.p_url)
        self.reg_instance(EngineRole.Decode, conn_req.d_url)

        cnt = 0
        while cnt < self.max_retry_cnt:
            if self.is_connected(conn_req.p_url, conn_req.d_url):
                return
            if cnt > 0:
                logger.warning(f'Connection failure, retry cnt: {cnt}')
            conn_event = await self._join_or_start(conn_req)
            if conn_event is None:
                return
            await self._wait_for_event(conn_event)
            if self.is_connected(conn_req.p_url, conn_req.d_url):
                return
            cnt += 1
        async with self.conn_lock:
            state = self.pool.get((conn_req.p_url, conn_req.d_url))
            if state is not None and state.status != PDConnectionStatus.Connected:
                state.set_status(PDConnectionStatus.Disconnected)
        raise TimeoutError('PDConnection Failure')

    async def _connect_all(self, messages: list[PDConnectionMessage], concurrency: int):
        sem = asyncio.Semaphore(concurrency)

        async def _one(msg: PDConnectionMessage):
            async with sem:
                await self.connect(msg)

        await asyncio.gather(*[_one(msg) for msg in messages])

    async def warmup_connections(self, messages: list[PDConnectionMessage], concurrency: int | None = None):
        """Connect all P-D pairs, coalescing concurrent warmup callers.

        Concurrent callers join the in-flight warmup task instead of spawning another full P×D gather.
        """
        if not messages:
            return
        if concurrency is None:
            concurrency = self.WARMUP_CONNECT_CONCURRENCY
        async with self._warmup_lock:
            task = self._warmup_task
            if task is None or task.done():
                self._warmup_task = asyncio.create_task(
                    self._connect_all(list(messages), concurrency), name='pd-connection-warmup'
                )
                task = self._warmup_task
        await asyncio.shield(task)

    def is_connected(self, p_url: str, d_url: str):
        link = self.pool.get((p_url, d_url), None)
        if not link:
            return False
        return link.status == PDConnectionStatus.Connected

    def drop(self, pd_key: tuple[str, str]):
        left = pd_key[0]
        right = pd_key[1]

        def cache_free(server_endpoint, cache_free_request: DistServeCacheFreeRequest) -> dict:
            try:
                requests.post(
                    get_server_api(server_endpoint, 'distserve/free_cache'),
                    json=cache_free_request.model_dump(mode='json'),
                )
            except Exception as e:
                logger.warning(f'error cache block free {server_endpoint, cache_free_request}. ErrorMsg: {str(e)}')

        def drop_connect(server_endpoint: str, p2p_disconnect_request: DistServeDropConnectionRequest):
            try:
                requests.post(
                    get_server_api(server_endpoint, 'distserve/p2p_drop_connect'),
                    json=p2p_disconnect_request.model_dump(mode='json'),
                )
            except Exception as e:
                logger.warning(f'error drop connect {server_endpoint, p2p_disconnect_request}. ErrorMsg: {str(e)}')

        # trigger gc
        logger.warning('cache block gc triggered.')
        try:
            for session_id in self.migration_session_shelf[(left, right)]:
                cache_free(left, DistServeCacheFreeRequest(remote_engine_id=left, remote_session_id=session_id))
        except Exception as e:
            logger.warning(f'gc error, ErrorMsg: {str(e)}')

        # trigger p2p disconnect
        logger.warning('drop connection triggered.')
        try:
            drop_connect(left, DistServeDropConnectionRequest(engine_id=left, remote_engine_id=right))
            drop_connect(right, DistServeDropConnectionRequest(engine_id=right, remote_engine_id=left))
        except Exception as e:
            logger.warning(f'p2p disconnect error, ErrorMsg: {str(e)}')

        self.pool.pop((left, right), None)
