# Copyright (c) OpenMMLab. All rights reserved.
import enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from lmdeploy.pytorch.disagg.config import (
    DistServeEngineConfig,
    DistServeNVLinkConfig,
    DistServeRDMAConfig,
    DistServeTCPConfig,
)


class MigrationProtocol(enum.Enum):
    """Migration Transport Protocol.

    Attributes:
        RDMA: IB or RoCEv1/v2.
        NVLINK: High device-to-device link.

    Warning: By now, only `GPU Directed RDMA` is supported in DistServe.
        We preserve several protocol and will be implemented in the future.
    """

    TCP = enum.auto()
    RDMA = enum.auto()
    NVLINK = enum.auto()


class DistServeConnectionStatus(enum.Enum):
    # TODO(JimyMa): Add more connection failure handler
    SUCCESS = enum.auto()
    FAIL = enum.auto()


class DistServeInitRequest(BaseModel):
    local_engine_id: str
    local_engine_config: DistServeEngineConfig

    remote_engine_id: str
    remote_engine_config: DistServeEngineConfig

    protocol: MigrationProtocol

    rank: int | None = None

    tcp_config: DistServeTCPConfig | None = None
    rdma_config: DistServeRDMAConfig | None = None
    nvlink_config: DistServeNVLinkConfig | None = None


class DistServeEngineEndpointInfo(BaseModel):
    zmq_address: str


class DistServeCachePoolInfo(BaseModel):
    """Serializable physical layout of one registered cache pool."""

    shape: tuple[int, ...]
    dtype: str
    element_size: int
    entry_axis: int
    group: Literal['target', 'draft', 'state', 'auxiliary', 'legacy_state'] = 'target'


class DistServeKVTransferEndpointInfo(BaseModel):
    protocol: MigrationProtocol
    endpoint_info: str
    cache_pools: tuple[DistServeCachePoolInfo, ...] | None = None
    version: int = 1
    schema: str = ''


class DistServeInitResponse(BaseModel):
    status: DistServeConnectionStatus
    # the control plane initialization feedback
    engine_endpoint_info: DistServeEngineEndpointInfo
    # the KVCache Transfer initialization feedback
    # To ensure generality (where endpoint_info can be initialization information
    # for different media such as RDMA, NVLink, etc.), we use a string (str) to
    # store this information.
    kvtransfer_endpoint_info: list[DistServeKVTransferEndpointInfo]


class DistServeConnectionRequest(BaseModel):
    protocol: MigrationProtocol
    remote_engine_id: str
    remote_engine_endpoint_info: DistServeEngineEndpointInfo
    remote_kvtransfer_endpoint_info: list[DistServeKVTransferEndpointInfo]


class DistServeConnectionResponse(BaseModel):
    status: DistServeConnectionStatus


class PDHandoff(BaseModel):
    """One immutable export, valid only within its source engine incarnation.

    Block IDs address target and draft pools. Compact state IDs are per attention rank; auxiliary and legacy slots
    belong to the request, not the block table. The receiver owns the original prompt and computes its remaining tail.
    """

    version: Literal[2] = 2
    transfer_id: str
    engine_epoch: str
    schema: str
    prompt_digest: str
    prompt_token_count: int = Field(gt=0)
    valid_token_count: int = Field(ge=0)
    block_size: int = Field(gt=0)
    block_ids: list[int]
    state_ids: list[int] = Field(default_factory=list)
    auxiliary_id: int | None = None
    legacy_state_id: int | None = None

    @model_validator(mode='after')
    def validate_boundary(self):
        if self.valid_token_count >= self.prompt_token_count:
            raise ValueError('PD requires a nonempty original prompt tail for decode replay.')
        expected = (self.valid_token_count + self.block_size - 1) // self.block_size
        if self.block_ids and len(self.block_ids) != expected:
            raise ValueError('PD block table must describe exactly the exported prefix.')
        ids = self.block_ids + self.state_ids
        ids.extend(i for i in (self.auxiliary_id, self.legacy_state_id) if i is not None)
        if any(i < 0 for i in ids):
            raise ValueError('PD resource IDs must be nonnegative.')
        return self


class PDControlRequest(BaseModel):
    """Grant and revoke are serialized by the source paging owner."""

    action: Literal['claim', 'revoke', 'release']
    transfer_id: str
    engine_epoch: str
    receiver_epoch: str = ''


class MigrationRequest(BaseModel):
    protocol: MigrationProtocol

    remote_engine_id: str
    remote_session_id: int
    remote_token_id: int = 0
    remote_block_ids: list[int] = Field(default_factory=list)
    handoff: PDHandoff | None = None

    is_dummy_prefill: bool = False


class DistServeCacheFreeRequest(BaseModel):
    remote_engine_id: str
    remote_session_id: int


class DistServeDropConnectionRequest(BaseModel):
    engine_id: str
    remote_engine_id: str
