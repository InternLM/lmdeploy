# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import json
import os
from typing import Dict

from dlslime import RDMAEndpoint, available_nic

from lmdeploy.logger import get_logger
from lmdeploy.pytorch.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.pytorch.disagg.backend.base import MigrationBackendImpl
from lmdeploy.pytorch.disagg.config import DistServeEngineConfig, MigrationBackend
from lmdeploy.pytorch.disagg.conn.protocol import (DistServeInitRequest, DistServeKVTransferEndpointInfo,
                                                   MigrationProtocol)
from lmdeploy.pytorch.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment

logger = get_logger('lmdeploy')

LMDEPLOY_USE_ASYNC_MIGRATION = os.environ.get('LMDEPLOY_USE_ASYNC_MIGRATION', None)


class DLSlimeMigrationManagement:

    def __init__(self, init_request: DistServeInitRequest):
        self.rank = init_request.rank
        self.local_engine_config: DistServeEngineConfig = (init_request.local_engine_config)
        self.remote_engine_config: DistServeEngineConfig = (init_request.remote_engine_config)
        self.endpoint: Dict[MigrationProtocol, RDMAEndpoint] = {}
        if init_request.protocol == MigrationProtocol.RDMA:
            nics = available_nic()
            device_name = nics[self.rank % len(nics)]
            logger.info(f'use device {device_name} for kv migration')
            self.endpoint[MigrationProtocol.RDMA] = RDMAEndpoint(
                device_name=device_name,
                ib_port=1,
                link_type=init_request.rdma_config.link_type.name,
            )
        elif init_request.protocol == MigrationProtocol.NVLINK:
            try:
                from dlslime import NVLinkEndpoint
            except ImportError:
                logger.warning('Notice: DLSlime not compiled from source with NVLink. Fallback to RDMAEndpoint.')
                NVLinkEndpoint = RDMAEndpoint
            self.endpoint[MigrationProtocol.NVLINK] = NVLinkEndpoint()

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        self.endpoint[register_mr_request.protocol].register_memory_region(
            register_mr_request.mr_key,
            register_mr_request.addr,
            register_mr_request.offset,
            register_mr_request.length,
        )

    def connect(self, kvtransfer_endpoint_info: DistServeKVTransferEndpointInfo):
        self.endpoint[kvtransfer_endpoint_info.protocol].connect(json.loads(kvtransfer_endpoint_info.endpoint_info))

    async def p2p_migrate(self, assignment: MigrationAssignment):
        batch = [(
            assign.mr_key,
            assign.mr_key,
            assign.target_offset,
            assign.source_offset,
            assign.length,
        ) for assign in assignment.batch]

        future = self.endpoint[assignment.protocol].read(batch)
        if LMDEPLOY_USE_ASYNC_MIGRATION:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, future.wait)
        else:
            return future.wait()


@MIGRATION_BACKENDS.register_module(MigrationBackend.DLSlime.name)
class DLSlimeBackend(MigrationBackendImpl):
    """DLSlime Transfer Engine."""

    def __init__(self):
        self.links: Dict[str, DLSlimeMigrationManagement] = {}

    def p2p_initialize(self, init_request: DistServeInitRequest):
        self.links[init_request.remote_engine_id] = DLSlimeMigrationManagement(init_request)

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        self.links[register_mr_request.remote_engine_id].register_memory_region(register_mr_request)

    def endpoint_info(self, remote_engine_id: str, protocol: MigrationProtocol):
        return self.links[remote_engine_id].endpoint[protocol].endpoint_info()

    def p2p_connect(self, remote_engine_id: str, conn_req: DistServeKVTransferEndpointInfo):
        self.links[remote_engine_id].connect(conn_req)

    async def p2p_migrate(self, assignment: MigrationAssignment, async_op: bool = False):
        await self.links[assignment.remote_engine_id].p2p_migrate(assignment)

    def store(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError

    def load(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError
