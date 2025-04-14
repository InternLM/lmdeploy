# Copyright (c) OpenMMLab. All rights reserved.
import enum
from typing import List, Optional, Tuple

from pydantic import BaseModel


class ServingStrategy(enum.Enum):
    NonDisaggregated = enum.auto()
    Disaggregated = enum.auto()


class EngineRole(enum.Enum):
    Hybrid = enum.auto()
    Prefill = enum.auto()
    Decode = enum.auto()


class MigrationBackend(enum.Enum):
    DLSlime = enum.auto()
    Mooncake = enum.auto()
    InfiniStore = enum.auto()


class DisaggEngineConfig(BaseModel):
    # parallel config
    tp_size: int
    ep_size: Optional[int]
    dp_size: Optional[int]
    pp_size: Optional[int]

    # cache config
    block_size: int
    num_cpu_blocks: int
    num_gpu_blocks: int


class MigrationTransportProtocol(enum.Enum):
    # Generate Transport Protocol
    TCP = enum.auto()
    # Engine with IB or RoCE NICs
    RDMA = enum.auto()
    # Engine with high device-to-device link
    NVLINK = enum.auto()


class TCPInitRequest(BaseModel):
    pass


class RDMAInitRequest(BaseModel):
    device_name: Optional[str] = None
    ib_port: int = 1
    link_type: str = "Ethernet"


class NVLinkInitRequest(BaseModel):
    pass


class MigrationInitRequest(BaseModel):
    remote_engine_id: str
    remote_engine_config: DisaggEngineConfig
    protocol: MigrationTransportProtocol

    rank: Optional[int] = None
    tp_rank: Optional[int] = None

    tcp_init_request: Optional[TCPInitRequest] = None
    rdma_init_request: Optional[RDMAInitRequest] = None
    nvlink_init_request: Optional[NVLinkInitRequest] = None


class MigrationRegisterMemoryRequest(BaseModel):
    protocol: MigrationTransportProtocol
    remote_engine_id: str
    mr_key: str
    addr: int
    length: int


class MigrationConnectionRequest(BaseModel):
    protocol: MigrationTransportProtocol
    remote_engine_id: str
    remote_endpoint_info: str


class MigrationRequest(BaseModel):
    remote_engine_id: str
    remote_session_id: int
    remote_token_id: int
    remote_block_ids: List[int]


class MigrationExecutionBatch(BaseModel):
    """Input of the Migration."""

    requests: List[Tuple[str, List[Tuple[int, int]]]] = []


class MigrationAssignment(BaseModel):
    protocol: MigrationTransportProtocol
    remote_engine_id: str
    mr_key: str
    target_offset: List[int]
    source_offset: List[int]
    length: int
