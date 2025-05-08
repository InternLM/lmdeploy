# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.pytorch.disagg.backend.base import MigrationBackendImpl
from lmdeploy.pytorch.disagg.config import MigrationBackend, MigrationProtocol
from lmdeploy.pytorch.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment
from lmdeploy.pytorch.disagg.request import DistServeConnectionRequest, DistServeInitRequest


@MIGRATION_BACKENDS.register_module(MigrationBackend.Mooncake.name)
class MooncakeBackend(MigrationBackendImpl):

    def p2p_initialize(self, init_request: DistServeInitRequest):
        raise NotImplementedError

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        raise NotImplementedError

    def endpoint_info(self, remote_engine_id: int, protocol: MigrationProtocol):
        return NotImplementedError

    def p2p_connect(self, connect_request: DistServeConnectionRequest):
        raise NotImplementedError

    def p2p_migrate(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError

    def store(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError

    def load(self, assignment: MigrationAssignment, async_op: bool = False):
        raise NotImplementedError
