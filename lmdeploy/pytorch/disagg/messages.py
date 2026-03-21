# Copyright (c) OpenMMLab. All rights reserved.

from pydantic import BaseModel

from lmdeploy.pytorch.disagg.config import DistServeNVLinkConfig, DistServeRDMAConfig, DistServeTCPConfig
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol


class MigrationExecutionBatch(BaseModel):
    """Input of the Migration."""

    protocol: MigrationProtocol
    requests: list[tuple[str, list[tuple[int, int]]]] = []


class AssignmentInstruct(BaseModel):
    """Assignment Batch."""
    mr_key: int
    target_offset: int
    source_offset: int
    length: int


class MigrationAssignment(BaseModel):
    """Migration Assignment."""
    protocol: MigrationProtocol
    remote_engine_id: str
    batch: list[AssignmentInstruct]


class PDConnectionMessage(BaseModel):
    p_url: str
    d_url: str
    protocol: MigrationProtocol = MigrationProtocol.RDMA
    tcp_config: DistServeTCPConfig | None = None
    rdma_config: DistServeRDMAConfig | None = None
    nvlink_config: DistServeNVLinkConfig | None = None


class DistServeRegisterMRMessage(BaseModel):
    protocol: MigrationProtocol

    remote_engine_id: str
    mr_key: int
    addr: int
    offset: int
    length: int
