from lmdeploy.disagg.backend.backend import register_migration_backend
from lmdeploy.disagg.backend.base import MigrationBackendImpl
from lmdeploy.disagg.config import MigrationProtocol
from lmdeploy.disagg.messages import DistServeRegisterMRMessage, MigrationAssignment
from lmdeploy.disagg.request import (
    DistServeInitRequest,
    DistServeConnectionRequest
)
from lmdeploy.disagg.config import MigrationBackend


@register_migration_backend(MigrationBackend.Mooncake)
class MooncakeBackend(MigrationBackendImpl):
    def p2p_initialize(self, init_request: DistServeInitRequest):
        raise NotImplementedError

    def register_memory_region(self, register_mr_request:DistServeRegisterMRMessage):
        raise NotImplementedError

    def endpoint_info(self, remote_engine_id: int, protocol: MigrationProtocol):
        return NotImplementedError

    def p2p_connect(self, connect_request: DistServeConnectionRequest):
        raise NotImplementedError

    async def p2p_migrate(self, assignment: MigrationAssignment):
        raise NotImplementedError

    async def store(self, assignment: MigrationAssignment):
        raise NotImplementedError

    async def load(self, assignment: MigrationAssignment):
        raise NotImplementedError
