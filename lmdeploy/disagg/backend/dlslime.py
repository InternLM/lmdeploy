from typing import Dict

from lmdeploy.disagg.messages import (
    MigrationBackend,
    MigrationInitRequest,
    MigrationTransportProtocol,
    DisaggEngineConfig,
    MigrationConnectionRequest,
    MigrationAssignment,
    MigrationRegisterMemoryRequest
)

from lmdeploy.disagg.backend.base import MigrationBackendImpl
from lmdeploy.disagg.backend.backend import register_migration_backend

from dlslime import RDMAEndpoint, available_nic


class DLSlimeMigrationManagement:
    def __init__(self, init_request: MigrationInitRequest):
        self.rank = init_request.rank
        self.remote_engine_config: DisaggEngineConfig = init_request.remote_engine_config
        self.endpoint: Dict[str, RDMAEndpoint] = {
            MigrationTransportProtocol.TCP: None,
            MigrationTransportProtocol.RDMA: None,
            MigrationTransportProtocol.NVLINK: None,
        }
        if init_request.rdma_init_request:
            if not init_request.rdma_init_request.device_name:
                nics = available_nic()
                init_request.rdma_init_request.device_name = nics[self.rank % len(nics)]
            self.endpoint[MigrationTransportProtocol.RDMA] = RDMAEndpoint(
                device_name=init_request.rdma_init_request.device_name,
                ib_port=init_request.rdma_init_request.ib_port,
                link_type=init_request.rdma_init_request.link_type
            )
    
    def register_memory_region(self, register_mr_request: MigrationRegisterMemoryRequest):
        self.endpoint[register_mr_request.protocol].register_memory_region(
            register_mr_request.mr_key,
            register_mr_request.addr,
            register_mr_request.length
        )

    def connect_to(self, connect_request: MigrationConnectionRequest):
        self.endpoint[connect_request.protocol].connect_to(connect_request.remote_endpoint_info)

    async def p2p_migrate(self, assignment: MigrationAssignment):
        await self.endpoint[assignment.protocol].read_batch_async(assignment.mr_key, assignment.target_offset, assignment.source_offset, assignment.length)


@register_migration_backend(MigrationBackend.DLSlime)
class DLSlimeBackend(MigrationBackendImpl):
    def __init__(self):
        self.links: Dict[int, DLSlimeMigrationManagement] = {}

    def p2p_initialize(self, init_request: MigrationInitRequest):
        self.links[init_request.remote_engine_id] = DLSlimeMigrationManagement(init_request)

    def register_memory_region(self, register_mr_request:MigrationRegisterMemoryRequest):
        self.links[register_mr_request.remote_engine_id].register_memory_region(register_mr_request)

    def endpoint_info(self, remote_engine_id: int, protocol: MigrationTransportProtocol):
        return self.links[remote_engine_id].endpoint[protocol].local_endpoint_info

    def p2p_connect(self, connect_request: MigrationConnectionRequest):
        self.links[connect_request.remote_engine_id].connect_to(connect_request)

    async def p2p_migrate(self, assignment: MigrationAssignment):
        await self.links[assignment.remote_engine_id].p2p_migrate(assignment)
