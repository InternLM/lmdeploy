from lmdeploy.disagg.messages import (
    MigrationBackend,
    MigrationInitRequest,
    MigrationConnectionRequest,
    MigrationAssignment,
    MigrationRegisterMemoryRequest,
    MigrationTransportProtocol
)

from lmdeploy.disagg.backend.backend import register_migration_backend
from lmdeploy.disagg.backend.base import MigrationBackendImpl


@register_migration_backend(MigrationBackend.InfiniStore)
class MooncakeBackend(MigrationBackendImpl):
    def p2p_initialize(self, init_request: MigrationInitRequest):
        raise NotImplementedError

    def register_memory_region(self, register_mr_request:MigrationRegisterMemoryRequest):
        raise NotImplementedError

    def endpoint_info(self, remote_engine_id: int, protocol: MigrationTransportProtocol):
        return NotImplementedError

    def p2p_connect(self, connect_request: MigrationConnectionRequest):
        raise NotImplementedError

    async def p2p_migrate(self, assignment: MigrationAssignment):
        raise NotImplementedError
