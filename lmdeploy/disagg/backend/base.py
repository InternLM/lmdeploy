from abc import abstractmethod

from lmdeploy.disagg.messages import (
    MigrationInitRequest,
    MigrationConnectionRequest,
    MigrationAssignment,
    MigrationRegisterMemoryRequest,
    MigrationTransportProtocol
)


class MigrationBackendImpl:
    @abstractmethod
    def p2p_initialize(self, init_request: MigrationInitRequest):
        raise NotImplementedError

    @abstractmethod
    def register_memory_region(self, register_mr_request:MigrationRegisterMemoryRequest):
        raise NotImplementedError

    @abstractmethod
    def endpoint_info(self, remote_engine_id: int, protocol: MigrationTransportProtocol):
        return NotImplementedError

    @abstractmethod
    def p2p_connect(self, connect_request: MigrationConnectionRequest):
        raise NotImplementedError

    @abstractmethod
    async def p2p_migrate(self, assignment: MigrationAssignment):
        raise NotImplementedError

    @abstractmethod
    async def store(self, assignment: MigrationAssignment):
        raise NotImplementedError

    @abstractmethod
    async def load(self, assignment: MigrationAssignment):
        raise NotImplementedError

