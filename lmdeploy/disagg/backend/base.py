from abc import abstractmethod

from lmdeploy.disagg.request import DistServeConnectionRequest
from lmdeploy.disagg.messages import (
    DistServeRegisterMRMessages,
    MigrationAssignment
)
from lmdeploy.disagg.config import MigrationProtocol
from lmdeploy.disagg.request import DistServeInitRequest


class MigrationBackendImpl:
    @abstractmethod
    def p2p_initialize(self, init_request: DistServeInitRequest):
        raise NotImplementedError

    @abstractmethod
    def register_memory_region(self, register_mr_request:DistServeRegisterMRMessages):
        raise NotImplementedError

    @abstractmethod
    def endpoint_info(self, remote_engine_id: int, protocol: MigrationProtocol):
        return NotImplementedError

    @abstractmethod
    def p2p_connect(self, conn_req: DistServeConnectionRequest):
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

