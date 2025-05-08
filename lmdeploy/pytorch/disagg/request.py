# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from pydantic import BaseModel

from lmdeploy.pytorch.disagg.config import (DistServeEngineConfig, DistServeNVLinkConfig, DistServeRDMAConfig,
                                            DistServeTCPConfig, MigrationProtocol)


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

    tcp_config: Optional[DistServeTCPConfig] = None
    rdma_config: Optional[DistServeRDMAConfig] = None
    nvlink_config: Optional[DistServeNVLinkConfig] = None


class MigrationRequest(BaseModel):
    protocol: MigrationProtocol

    remote_engine_id: str
    remote_session_id: int
    remote_token_id: int
    remote_block_ids: List[int]
