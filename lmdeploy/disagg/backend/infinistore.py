# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.disagg.backend.base import MigrationBackendImpl
from lmdeploy.disagg.config import MigrationBackend, MigrationProtocol
from lmdeploy.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment
from lmdeploy.disagg.request import DistServeConnectionRequest, DistServeInitRequest


@MIGRATION_BACKENDS.register_module(MigrationBackend.InfiniStore.name)
class InfiniStoreBackend(MigrationBackendImpl):

    def p2p_initialize(self, init_request: DistServeInitRequest):
        raise NotImplementedError

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        raise NotImplementedError

    def endpoint_info(self, remote_engine_id: int, protocol: MigrationProtocol):
        return NotImplementedError

    def p2p_connect(self, conn_req: DistServeConnectionRequest):
        raise NotImplementedError

    async def p2p_migrate(self, assignment: MigrationAssignment):
        raise NotImplementedError

    async def store(self, assignment: MigrationAssignment):
        raise NotImplementedError

    async def load(self, assignment: MigrationAssignment):
        raise NotImplementedError
