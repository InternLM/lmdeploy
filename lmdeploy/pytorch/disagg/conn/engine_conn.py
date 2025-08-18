# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
import time
import functools
from typing import Any, Awaitable, Callable, Dict, List, Tuple, TypeAlias, TYPE_CHECKING 
from urllib.parse import urlparse

import zmq
import zmq.asyncio

import numpy as np

import torch

from lmdeploy.logger import get_logger

from lmdeploy.messages import GenerationConfig
from lmdeploy.pytorch.engine.request import ResponseType

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeCacheFreeRequest, DistServeConnectionRequest,
                                                   DistServeConnectionResponse, DistServeStatus,
                                                   DistServeDropConnectionRequest, DistServeEngineEndpointInfo,
                                                   DistServeInitRequest, DistServeInitResponse,
                                                   DistServeKVTransferEndpointInfo, DistServeRecomputeRequest,
                                                   DistServeRecomputeResponse, DistServeFetchMetaRequest,
                                                   DistServeFetchMetaResponse, DistServeProactiveMigrationRequest,
                                                   DistServeProactiveMigrationResponse)
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch

from lmdeploy.pytorch.engine.executor.dist_utils import find_available_port
from lmdeploy.pytorch.engine.mp_engine.engine_instance_pool import EngineInstancePool
from lmdeploy.pytorch.messages import MessageStatus, HistoryTokenIds

from lmdeploy.pytorch.engine.request import ResponseType
from lmdeploy.pytorch.messages import InferOutput, SchedulerSequence
from lmdeploy.messages import GenerationConfig


if TYPE_CHECKING:
    from lmdeploy.pytorch.engine.engine import Engine

logger = get_logger('lmdeploy')


DistServeEngineConnCallResponse: TypeAlias = (
    DistServeFetchMetaRequest
    | DistServeFetchMetaResponse
    | DistServeProactiveMigrationRequest
    | DistServeProactiveMigrationResponse
    | DistServeCacheFreeRequest
    | DistServeRecomputeRequest
    | DistServeRecomputeResponse
)


class EngineP2PConnection:
    def __init__(self, engine: 'Engine'):
        self.engine: Engine = engine
        self.p2p_conn_ctx: Dict[str, zmq.asyncio.Context] = {}
        self.p2p_sender: Dict[str, zmq.asyncio.Socket] = {}
        self.p2p_receiver: Dict[str, zmq.asyncio.Socket] = {}

        self.use_unique_kvtransfer_engine = os.environ.get('LMDEPLOY_USE_UNIQUE_KVTRANSFER_ENGINE', False)
        self.recomputation_conn_pool = EngineInstancePool(self.engine)

        self.handle_migration_event = asyncio.Event()
        self.handle_meta_migration_event = asyncio.Event()
        self.handle_recomputation_event = asyncio.Event()

        self.release_lock = asyncio.Lock()

        self.resp_que: asyncio.Queue[InferOutput] = None
        self.has_runable_event: asyncio.Event = None

    def _status_jump(self, msg: SchedulerSequence):
        _set_status = lambda msg, status: self.engine.scheduler._set_message_status(msg, status)
        _distserve_state_machine = {
            MessageStatus.META_MIGRATION_WAITING: MessageStatus.META_MIGRATION_RUNNING,
            MessageStatus.META_MIGRATION_RUNNING: MessageStatus.MIGRATION_WAITING,
            MessageStatus.MIGRATION_WAITING: MessageStatus.MIGRATION_RUNNING,
            MessageStatus.MIGRATION_RUNNING: MessageStatus.MIGRATION_DONE,
            MessageStatus.RECOMPUTION_PREEMPTION: MessageStatus.REMOTE_RECOMPUTING,
            MessageStatus.REMOTE_RECOMPUTING: MessageStatus.REMOTE_RECOMPUTED
        }
        if msg.status in _distserve_state_machine:
            # TODO (Jimy): handle TO_BE_MIGRATED
            _set_status(msg, _distserve_state_machine[msg.status])

    async def p2p_initialize(self, init_request: DistServeInitRequest):
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
                                     status=DistServeStatus.SUCCESS)

    def p2p_connect(self, conn_request: DistServeConnectionRequest):
        self.p2p_receiver[conn_request.remote_engine_id].connect(conn_request.remote_engine_endpoint_info.zmq_address)
        self.engine.executor.p2p_connect(remote_engine_id=conn_request.remote_engine_id,
                                         conn_request=conn_request.remote_kvtransfer_endpoint_info)
        event_loop = asyncio.get_event_loop()
        event_loop.create_task(self.handle_zmq_recv(conn_request.remote_engine_id))
        return DistServeConnectionResponse(status=DistServeStatus.SUCCESS)

    def p2p_drop_connect(self, drop_conn_request: DistServeDropConnectionRequest):
        # TODO (JimyMa): drop RDMA Connection
        # self.zmq_disconnect(drop_conn_request.remote_engine_id)
        return {'success': True}

    async def init_engine_conn_loop(self, resp_que, has_runable_event):
        self.resp_que = resp_que
        self.has_runable_event = has_runable_event
        event_loop = asyncio.get_event_loop(resp_que, has_runable_event)
        loop_tasks = []
        loop_migration = event_loop.create_task(
            self.engine_conn._handle_migration(resp_que, has_runable_event=has_runable_event),
            name='MainLoopMigration',
        )
        loop_meta_migration = event_loop.create_task(self.engine_conn.handle_meta_migration(), name="EngineConnHandleMetaMigration")
        loop_recomputation = event_loop.create_task(self.engine_conn.handle_recomputation(), name="HandleRecomputation")
        loop_tasks.extend([loop_migration, loop_meta_migration, loop_recomputation])
        return loop_tasks

    async def handle_meta_migration(self):
        while True:
            await self.handle_meta_migration_event.wait()
            self.handle_meta_migration_event.clear()
            meta_migration_waiting_seqs = self.engine.scheduler.meta_migration_waiting
            for meta_migration_waiting_seq in meta_migration_waiting_seqs:
                await self.zmq_send(
                    meta_migration_waiting_seq.migration_context.prefill_engine_id,
                    DistServeFetchMetaRequest(migration_context=meta_migration_waiting_seq.migration_context)
                )

    async def handle_recomputation(self):
        while True:
            await self.handle_recomputation_event.wait()
            self.handle_recomputation_event.clear()
            preempted_seqs = self.engine.scheduler.recomputation_preemption
            for preempted_seq in preempted_seqs:
                preempted_seq.migration_context.token_ids = preempted_seq.token_ids
                await self.zmq_send(
                    preempted_seq.migration_context.prefill_engine_id,
                    DistServeRecomputeRequest(migration_context=preempted_seq.migration_context)
                )

    @torch.inference_mode()
    async def _handle_migration(self, resp_que: asyncio.Queue, has_runable_event: asyncio.Event):
        """Async loop migration."""
        while True:
            migration_running = self.engine.scheduler._schedule_migration()
            if not migration_running and not self.engine.scheduler.has_migration_waiting():
                await self.handle_migration_event.wait()
            elif migration_running:
                self.handle_migration_event.clear()
                for msg in migration_running:
                    migration_context = msg.migration_context
                    migration_context.decode_block_ids = list(self.engine.scheduler.block_manager.get_block_table(msg=msg))
                    await self.zmq_send(migration_context.prefill_engine_id, DistServeProactiveMigrationRequest(migration_context=migration_context))
            else:
                # release coroutine for decoding
                await asyncio.sleep(.5)

    async def zmq_send(self, remote_engine_id: str, req: DistServeEngineConnCallResponse):
        _get_msg = lambda session_id: list(self.engine.scheduler.sessions[session_id].sequences.values())[0]

        def _send_preprocess(func: Callable[[], Awaitable[Any]]) -> Callable[[], Awaitable[Any]]:
            @functools.wraps(func)
            async def wrapper() -> Any:
                migration_context = req.migration_context
                if self.engine.engine_config.role == EngineRole.Decode:
                    self._status_jump(_get_msg(migration_context.decode_session_id))
                return await func()
            return wrapper

        @_send_preprocess
        async def _send_impl():
            logger.error(f"Sending, {req=}")
            await self.p2p_sender[remote_engine_id].send_pyobj(req)

        await _send_impl()

    async def handle_zmq_recv(self, remote_engine_id: str):
        _get_msg = lambda session_id: list(self.engine.scheduler.sessions[session_id].sequences.values())[0]

        async def _handle_fetch_migration_context_call(req: DistServeFetchMetaRequest):
            logger.error("handle fetch migration context call")
            migration_context = req.migration_context
            msg = _get_msg(migration_context.prefill_session_id)
            migration_context.token_ids = msg.all_ids.tolist()
            migration_context.prefill_block_ids = list(self.engine.scheduler.block_manager.get_block_table(msg=msg))
            await self.zmq_send(
                migration_context.decode_engine_id,
                DistServeFetchMetaResponse(migration_context=migration_context, status=DistServeStatus.SUCCESS)
            )

        async def _handle_fetch_migration_context_resp(req: DistServeFetchMetaResponse):
            migration_context = req.migration_context
            msg = _get_msg(migration_context.decode_session_id)
            msg.history_cache = HistoryTokenIds(np.array(migration_context.token_ids[:-1]))
            msg.migration_context = migration_context
            msg.__post_init__()
            self._status_jump(msg)
            self.handle_migration_event.set()

        async def _handle_remote_preemption_call(req: DistServeRecomputeRequest):
            migration_context = req.migration_context
            async with self.recomputation_conn_pool.instance() as instance:
                gen_config = GenerationConfig(
                    max_new_tokens=1,
                    with_cache=True,
                    preserve_cache=True
                )
                if migration_context.prefill_session_id in self.engine.scheduler.sessions:
                    self.engine.scheduler.end_session(session_id=migration_context.prefill_session_id)
                resp = await instance.async_infer(migration_context.prefill_session_id, req.token_ids, gen_config=gen_config)
            migration_context.prefill_block_ids = resp.cache_block_ids
            migration_context.token_ids = resp.token_ids
            recompute_resp = DistServeRecomputeResponse(
                migration_context=migration_context,
                status=DistServeStatus.SUCCESS
            )
            logger.error(f"{self.p2p_sender[migration_context.decode_engine_id]=}")
            
            await self.zmq_send(migration_context.decode_engine_id, recompute_resp)

        async def _handle_remote_preemption_resp(req: DistServeRecomputeResponse):
            migration_context = req.migration_context
            msg = _get_msg(migration_context.decode_session_id)
            msg.migration_context = migration_context
            logger.error(f"{migration_context=}")
            self._status_jump(msg)
            self.handle_migration_event.set()

        async def _handle_proactive_migration_call(req: DistServeProactiveMigrationRequest):
            migration_context = req.migration_context

            def _handle_cache_free():
                session_id = req.migration_context.prefill_session_id
                if session_id in self.engine.scheduler.sessions:
                    self.engine.scheduler.end_session(session_id=session_id)
                else:
                    logger.error(f'invalid free, {remote_engine_id}, {session_id}')

            migration_execution_requests: List[Tuple[int, List[Tuple[int, int]]]] = []
            logger.error(list(zip(migration_context.prefill_block_ids, migration_context.decode_block_ids)))
            migration_execution_requests.append((
                migration_context.decode_engine_id,
                list(zip(migration_context.decode_block_ids, migration_context.prefill_block_ids)),
            ))
            migration_inputs = MigrationExecutionBatch(protocol=migration_context.protocol,
                                                       requests=migration_execution_requests)
            msg = _get_msg(migration_context.prefill_session_id)
            logger.info(f'migrating session: {msg.session_id} begin')
            migration_context.time_stamp.migration_begine = time.time()
            await self.engine.executor.migrate(migration_inputs)
            migration_context.time_stamp.migration_end = time.time()
            logger.info(f'migrating session: {msg.session_id} done')
            _handle_cache_free()
            migration_resp = DistServeProactiveMigrationResponse(
                migration_context=migration_context,
                status=DistServeStatus.SUCCESS
            )
            await self.zmq_send(migration_context.decode_engine_id, migration_resp)

        async def _handle_proactive_migration_resp(req: DistServeProactiveMigrationResponse):
            # generate output
            migration_context = req.migration_context
            outputs: Dict[int, InferOutput] = dict()
            msg = _get_msg(migration_context.decode_session_id)
            msg.migration_context = migration_context
            msg.resp.type = ResponseType.SUCCESS
            token_ids = [migration_context.token_ids[-1]]
            out = InferOutput(
                session_id=migration_context.decode_session_id,
                resp=msg.resp,
                finish=False,
                token_ids=np.array(token_ids)
            )
            outputs[migration_context.decode_session_id] = out
            self.engine.update_running_migration([msg], np.array([token_ids]), [False], [None])
            self.resp_que.put_nowait(outputs)
            self._status_jump(msg)
            self.has_runable_event.set()

        method_fn: Dict[str, Awaitable[None]] = {}
        def _register_method(primitive: DistServeEngineConnCallResponse, fn: Callable[[DistServeConnectionResponse], None]):
            method_fn[primitive] = fn

        _register_method(DistServeFetchMetaRequest.__name__, _handle_fetch_migration_context_call)
        _register_method(DistServeFetchMetaResponse.__name__, _handle_fetch_migration_context_resp)
        _register_method(DistServeRecomputeRequest.__name__, _handle_remote_preemption_call)
        _register_method(DistServeRecomputeResponse.__name__, _handle_remote_preemption_resp)
        _register_method(DistServeProactiveMigrationRequest.__name__, _handle_proactive_migration_call)
        _register_method(DistServeProactiveMigrationResponse.__name__, _handle_proactive_migration_resp)

        while True:
            logger.error("starting")
            req: DistServeEngineConnCallResponse = await self.p2p_receiver[remote_engine_id].recv_pyobj()
            logger.error(f"recv: {req=}, {req.__class__.__name__}")
            try:
                await method_fn[req.__class__.__name__](req)
            except KeyError:
                logger.error(f'Unsupported zmq request {type(req)}')
                raise KeyError

    async def zmq_disconnect(self, remote_engine_id: str):
        async with self.release_lock:
            self.p2p_receiver[remote_engine_id].close()
            self.p2p_sender[remote_engine_id].close()
            self.p2p_conn_ctx[remote_engine_id].term()
