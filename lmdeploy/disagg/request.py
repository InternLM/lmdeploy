from typing import List
from pydantic import BaseModel

from typing import Optional

from lmdeploy.disagg.config import (
    DistServeEngineConfig,
    DistServeNVLinkConfig,
    DistServeRDMAConfig,
    DistServeTCPConfig,
    MigrationProtocol
)


class DistServeConnectionRequest(BaseModel):
    protocol: MigrationProtocol
    remote_engine_id: str
    remote_endpoint_info: str


class DistServeInitRequest(BaseModel):
    local_engine_id: str
    local_engine_config: DistServeEngineConfig

    remote_engine_id: str
    remote_engine_config: DistServeEngineConfig

    protocol: MigrationProtocol

    rank: Optional[int] = None

    tcp_init_request: Optional[DistServeTCPConfig] = None
    rdma_init_request: Optional[DistServeRDMAConfig] = None
    nvlink_init_request: Optional[DistServeNVLinkConfig] = None


class MigrationRequest(BaseModel):
    protocol: MigrationProtocol

    remote_engine_id: str
    remote_session_id: int
    remote_token_id: int
    remote_block_ids: List[int]
