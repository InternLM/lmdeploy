# Copyright (c) OpenMMLab. All rights reserved.
import enum
from typing import List, Optional

from pydantic import BaseModel

from lmdeploy.pytorch.disagg.config import (DistServeEngineConfig, DistServeNVLinkConfig, DistServeRDMAConfig,
                                            DistServeTCPConfig)


class KVTransferProtocol(enum.Enum):
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


class DistServeStatus(enum.Enum):
    # TODO(JimyMa): Add more connection failure handler
    SUCCESS = enum.auto()
    FAIL = enum.auto()


class DistServeInitRequest(BaseModel):
    local_engine_id: str
    local_engine_config: DistServeEngineConfig

    remote_engine_id: str
    remote_engine_config: DistServeEngineConfig

    kvtransfer_protocol: KVTransferProtocol

    rank: Optional[int] = None

    tcp_config: Optional[DistServeTCPConfig] = None
    rdma_config: Optional[DistServeRDMAConfig] = None
    nvlink_config: Optional[DistServeNVLinkConfig] = None


class DistServeEngineEndpointInfo(BaseModel):
    zmq_address: str


class DistServeKVTransferEndpointInfo(BaseModel):
    protocol: KVTransferProtocol
    endpoint_info: str


class DistServeInitResponse(BaseModel):
    status: DistServeStatus
    # the control plane initialization feedback
    engine_endpoint_info: DistServeEngineEndpointInfo
    # the KVCache Transfer initialization feedback
    # To ensure generality (where endpoint_info can be initialization information
    # for different media such as RDMA, NVLink, etc.), we use a string (str) to
    # store this information.
    kvtransfer_endpoint_info: List[DistServeKVTransferEndpointInfo]


class DistServeConnectionRequest(BaseModel):
    protocol: KVTransferProtocol
    remote_engine_id: str
    remote_engine_endpoint_info: DistServeEngineEndpointInfo
    remote_kvtransfer_endpoint_info: List[DistServeKVTransferEndpointInfo]


class DistServeDropConnectionRequest(BaseModel):
    engine_id: str
    remote_engine_id: str


class DistServeConnectionResponse(BaseModel):
    status: DistServeStatus


class MigrationTimeStamp(BaseModel):
    arrive_time: Optional[float] = None
    migration_begine: Optional[float] = None
    migration_end: Optional[float] = None

    remote_recomputation_begin: Optional[List[float]] = None
    remote_recomputation_end: Optional[List[float]] = None


class MigrationContext(BaseModel):
    protocol: KVTransferProtocol

    decode_engine_id: str
    decode_session_id: Optional[int]
    decode_block_ids: Optional[List[int]]

    prefill_engine_id: str
    prefill_session_id: int
    prefill_block_ids: List[int]

    token_ids: Optional[List[int]] = None

    time_stamp: Optional[MigrationTimeStamp] = None

    is_dummy_prefill: bool = False


class DistServeFetchMetaRequest(BaseModel):
    migration_context: MigrationContext


class DistServeFetchMetaResponse(BaseModel):
    migration_context: MigrationContext
    status: DistServeStatus


class DistServeProactiveMigrationRequest(BaseModel):
    migration_context: MigrationContext


class DistServeProactiveMigrationResponse(BaseModel):
    migration_context: MigrationContext
    status: DistServeStatus


class DistServeCacheFreeRequest(BaseModel):
    migration_context: MigrationContext


class DistServeRecomputeRequest(BaseModel):
    migration_context: MigrationContext


class DistServeRecomputeResponse(BaseModel):
    migration_context: MigrationContext
    status: DistServeStatus
