# Copyright (c) OpenMMLab. All rights reserved.
import enum
from typing import List, Optional, Tuple

from pydantic import BaseModel


class EngineRole(enum.Enum):
    Hybrid = enum.auto()
    Prefill = enum.auto()
    Decode = enum.auto()


class RemoteEngineConfig(BaseModel):
    # parallel config
    tp_size: int
    ep_size: Optional[int]
    dp_size: Optional[int]
    pp_size: Optional[int]

    # cache config
    block_size: int
    num_cpu_blocks: int
    num_gpu_blocks: int


class RDMAInitRequest(BaseModel):
    remote_engine_id: int
    remote_engine_config: RemoteEngineConfig
    link_type: str = "Ethernet"


class RDMAConnectRequest(BaseModel):
    remote_engine_id: int
    remote_endpoint_info: List[str]


class MigrationRequest(BaseModel):
    remote_engine_id: int
    remote_session_id: int
    remote_token_id: int
    remote_block_ids: List[int]


class MigrationExecutionInputs(BaseModel):
    """Input of the Migration."""

    requests: List[Tuple[int, List[Tuple[int, int]]]] = []
