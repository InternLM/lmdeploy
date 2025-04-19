from typing import Dict

from lmdeploy.disagg.request import DistServeConnectionRequest
from lmdeploy.disagg.messages import (
    DistServeRegisterMRMessage,
    MigrationAssignment
)

from lmdeploy.disagg.backend.base import MigrationBackendImpl
from lmdeploy.disagg.backend.backend import register_migration_backend

from dlslime import RDMAEndpoint, available_nic

from lmdeploy.disagg.config import (
    DistServeEngineConfig,
    MigrationBackend,
    MigrationProtocol
)
from lmdeploy.disagg.request import DistServeInitRequest


class DLSlimeMigrationManagement:
    def __init__(self, init_request: DistServeInitRequest):
        self.rank = init_request.rank
        self.local_engine_config: DistServeEngineConfig = init_request.local_engine_config
        self.remote_engine_config: DistServeEngineConfig = init_request.remote_engine_config
        self.endpoint: Dict[str, RDMAEndpoint] = {
            MigrationProtocol.TCP: None,
            MigrationProtocol.RDMA: None,
            MigrationProtocol.NVLINK: None,
        }
        if init_request.rdma_config:
            nics = self.local_engine_config.available_nics or available_nic()
            device_name = nics[self.rank % len(nics)]
            self.endpoint[MigrationProtocol.RDMA] = RDMAEndpoint(
                device_name=device_name,
                ib_port=1,
                link_type=init_request.rdma_config.link_type.name
            )
        if init_request.nvlink_init_request:
            raise NotImplementedError
        if init_request.tcp_init_request:
            raise NotImplementedError

    def register_memory_region(self, register_mr_request: DistServeRegisterMRMessage):
        self.endpoint[register_mr_request.protocol].register_memory_region(
            register_mr_request.mr_key,
            register_mr_request.addr,
            register_mr_request.length
        )

    def connect_to(self, connect_request: DistServeConnectionRequest):
        self.endpoint[connect_request.protocol].connect_to(connect_request.remote_endpoint_info)

    async def p2p_migrate(self, assignment: MigrationAssignment):
        max_batch = 4096 + 2048
        for i in range(0, len(assignment.target_offset), max_batch):
            await self.endpoint[assignment.protocol].read_batch_async(
                assignment.mr_key,
                assignment.target_offset[i: i+max_batch],
                assignment.source_offset[i: i+max_batch],
                assignment.length
            )


@register_migration_backend(MigrationBackend.DLSlime)
class DLSlimeBackend(MigrationBackendImpl):
    def __init__(self):
        self.links: Dict[int, DLSlimeMigrationManagement] = {}

    def p2p_initialize(self, init_request: DistServeInitRequest):
        self.links[init_request.remote_engine_id] = DLSlimeMigrationManagement(init_request)

    def register_memory_region(self, register_mr_request:DistServeRegisterMRMessage):
        self.links[register_mr_request.remote_engine_id].register_memory_region(register_mr_request)

    def endpoint_info(self, remote_engine_id: int, protocol: MigrationProtocol):
        return self.links[remote_engine_id].endpoint[protocol].local_endpoint_info

    def p2p_connect(self, conn_req: DistServeConnectionRequest):
        self.links[conn_req.remote_engine_id].connect_to(conn_req)

    async def p2p_migrate(self, assignment: MigrationAssignment):
        await self.links[assignment.remote_engine_id].p2p_migrate(assignment)

    async def store(self, assignment: MigrationAssignment):
        raise NotImplementedError

    async def load(self, assignment: MigrationAssignment):
        raise NotImplementedError
