# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import json
import os
from typing import Dict, List

from dlslime import Assignment as DLSlimeAssignment
from dlslime import NVLinkEndpoint, RDMAEndpoint, available_nic

from lmdeploy.logger import get_logger
from lmdeploy.pytorch.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.pytorch.disagg.backend.base import MigrationBackendImpl
from lmdeploy.pytorch.disagg.config import DistServeEngineConfig, MigrationBackend, MigrationProtocol
from lmdeploy.pytorch.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment
from lmdeploy.pytorch.disagg.request import DistServeConnectionRequest, DistServeInitRequest

logger = get_logger('lmdeploy')

LMDEPLOY_USE_ASYNC_MIGRATION = os.environ.get('LMDEPLOY_USE_ASYNC_MIGRATION', None)


async def read_batch_coroutine(endpoint: RDMAEndpoint, batch: List[DLSlimeAssignment]):
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def _completion_handler(status: int):
        loop.call_soon_threadsafe(future.set_result, status)

    endpoint.read_batch_with_callback(
        batch,
        _completion_handler,
    )
    await future


class DLSlimeMigrationManagement:

    def __init__(self, init_request: DistServeInitRequest):
        self.rank = init_request.rank
        self.local_engine_config: DistServeEngineConfig = init_request.local_engine_config
        self.remote_engine_config: DistServeEngineConfig = init_request.remote_engine_config
        self.endpoint: Dict[MigrationProtocol, RDMAEndpoint] = {
            MigrationProtocol.TCP: None,
            MigrationProtocol.RDMA: None,
            MigrationProtocol.NVLINK: None,
        }
        if init_request.protocol == MigrationProtocol.RDMA:
            nics = available_nic()
            device_name = nics[self.rank % len(nics)]
            logger.info(f'use device {device_name} for kv migration')
            self.endpoint[MigrationProtocol.RDMA] = RDMAEndpoint(device_name=device_name,
                                                                 ib_port=1,
                                                                 link_type=init_request.rdma_config.link_type.name)
        elif init_request.protocol == MigrationProtocol.NVLINK:
            self.endpoint[MigrationProtocol.NVLINK] = NVLinkEndpoint()

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        self.endpoint[register_mr_request.protocol].register_memory_region(register_mr_request.mr_key,
                                                                           register_mr_request.addr,
                                                                           register_mr_request.offset,
                                                                           register_mr_request.length)

    def connect(self, connect_request: DistServeConnectionRequest):
        self.endpoint[connect_request.protocol].connect(json.loads(connect_request.remote_endpoint_info))

    async def p2p_migrate(self, assignment: MigrationAssignment, async_op=False):
        batch = [
            DLSlimeAssignment(
                mr_key=assign.mr_key,
                target_offset=assign.target_offset,
                source_offset=assign.source_offset,
                length=assign.length,
            ) for assign in assignment.batch
        ]

        if not LMDEPLOY_USE_ASYNC_MIGRATION:
            MAX_NUM_READ_BATCH = 4096

            def split(batch: List[DLSlimeAssignment]):
                batch_split = []
                for i in range(0, len(batch), MAX_NUM_READ_BATCH):
                    batch_split.append(batch[i:i + MAX_NUM_READ_BATCH])
                return batch_split

            batch_splited = split(batch)
            for b_split in batch_splited:
                self.endpoint[assignment.protocol].read_batch(b_split)
        else:
            await read_batch_coroutine(self.endpoint[assignment.protocol], batch)


@MIGRATION_BACKENDS.register_module(MigrationBackend.DLSlime.name)
class DLSlimeBackend(MigrationBackendImpl):

    def __init__(self):
        self.links: Dict[int, DLSlimeMigrationManagement] = {}

    def p2p_initialize(self, init_request: DistServeInitRequest):
        self.links[init_request.remote_engine_id] = DLSlimeMigrationManagement(init_request)

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        self.links[register_mr_request.remote_engine_id].register_memory_region(register_mr_request)

    def endpoint_info(self, remote_engine_id: int, protocol: MigrationProtocol):
        return self.links[remote_engine_id].endpoint[protocol].endpoint_info

    def p2p_connect(self, conn_req: DistServeConnectionRequest):
        self.links[conn_req.remote_engine_id].connect(conn_req)

    async def p2p_migrate(self, assignment: MigrationAssignment, async_op: bool = False):
        await self.links[assignment.remote_engine_id].p2p_migrate(assignment, async_op=async_op)

    def store(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError

    def load(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError
