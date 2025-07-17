# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

from pydantic import BaseModel

from lmdeploy.pytorch.disagg.config import (DistServeNVLinkConfig, DistServeRDMAConfig, DistServeTCPConfig,
                                            MigrationProtocol)


class MigrationExecutionBatch(BaseModel):
    """Input of the Migration."""

    protocol: MigrationProtocol
    requests: List[Tuple[str, List[Tuple[int, int]]]] = []


class AssignmentInstruct(BaseModel):
    """Assignment Batch."""
    mr_key: str
    target_offset: int
    source_offset: int
    length: int


class MigrationAssignment(BaseModel):
    """Migration Assignment."""
    protocol: MigrationProtocol
    remote_engine_id: str
    batch: List[AssignmentInstruct]


class PDConnectionMessage(BaseModel):
    p_url: str
    d_url: str
    protocol: MigrationProtocol = MigrationProtocol.RDMA
    tcp_config: Optional[DistServeTCPConfig] = None
    rdma_config: Optional[DistServeRDMAConfig] = None
    nvlink_config: Optional[DistServeNVLinkConfig] = None


class DistServeRegisterMRMessage(BaseModel):
    protocol: MigrationProtocol

    remote_engine_id: str
    mr_key: str
    addr: int
    offset: int
    length: int
