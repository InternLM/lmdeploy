# Copyright (c) OpenMMLab. All rights reserved.
import enum
from typing import List, Optional

from pydantic import BaseModel

from lmdeploy.pytorch.disagg.config import (
    DistServeEngineConfig,
    DistServeNVLinkConfig,
    DistServeRDMAConfig,
    DistServeTCPConfig,
)


"""
This File introduces all the interface (*Request* and *Response*) of PD disaggregation protocol.

Figure blow illustrate the flow of connection establish of Prefill and Decode Engine.

    Prefill Engine                                           Proxy                                        Decode Engine
           |                                                   |                                                |
           |----------------------------------------------------------------------------------------------------|
           |                                                                                                    |
           |    Proxy sends Init Request                                                                        |
           |                                                                                                    |
           |--------------------------------------------------------------------------------------------------- |
           | <<<=================================== DistServeInitRequest ===================================>>> |
           |--------------------------------------------------------------------------------------------------- |
           |                                                                                                    |
           |    Engines initialize RDMA/NVLink endpoint for KVCache Migration, and setup control plane via ZMQ  |
           |      socket in scheduler.                                                                          |
           |                                                                                                    |
           |--------------------------------------------------------------------------------------------------- |
           | ===================================>>> DistServeInitResponse <<<================================== |
           |----------------------------------------------------------------------------------------------------|
           |                                                                                                    |
           |    Proxy requests connection info                                                                  |
           |                                                                                                    |
           |----------------------------------------------------------------------------------------------------|
           | <<<================================ DistServeConnectionRequest ================================>>> |
           |----------------------------------------------------------------------------------------------------|
           |                                                                                                    |
           |    Engine modify endpoint status from initialized status to ready to send/recv status              |
           |                                                                                                    |
           |----------------------------------------------------------------------------------------------------|
           | ===============================>>> DistServeConnectionResponse <<<================================ | 
           |                                                   |                                                |
           |                                                   |                                                |
    Prefill Engine                                           Proxy                                        Decode Engine



The figure blow illustrate life span of the served request in DistServe.
Prefill --------------- WAITING -- RUNNING -- TO_BE_MIGRATE ---------------------------------- Prefill Session End
                           /                  \\                                          /              |
                          /                    \\                                        /               |
                         /                      \\                                      /                |
               v1/chat/completions            first_token_id                           /                 |
                (Preserve Cache)                 \\                                   /                  |
                        /                         \\                                 /                   |
                       /                           \\                               v                    |
                      /                             \\                        p2p_migrate                | 
                     /                               \\                   (REMOTE_MEMORY_COPY)           |
Proxy   -------------                                ---                         /                       |
                                                      \\                        /                        |
                                                       \\                      /                     free_cache
                                                        \\                    /                     (ZMQ_Socket)
                                                v1/chat/completion           /                           |
                                                (+MigrationRequest          /                            |
                                                         \\                /                             |
                                                          \\              /                              |
                                                           \\            /                               |
                                                            \\          v                                |            v-^
Decode  ---------------------------------------- MIGRATION_WAITING -- MIGRATION_RUNNING  --------- MIGRATION_DONE  -- | | RUNNING -- Decode Session_end 
                                                                                                                      v-^

"""


class MigrationProtocol(enum.Enum):
    """Migration Transport Protocol.

    Attributes:
        TCP: TCP for General Purpose Transport Protocol.
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

    rank: Optional[int] = None

    tcp_config: Optional[DistServeTCPConfig] = None
    rdma_config: Optional[DistServeRDMAConfig] = None
    nvlink_config: Optional[DistServeNVLinkConfig] = None


class DistServeEngineEndpointInfo(BaseModel):
    zmq_address: str


class DistServeKVTransferEndpointInfo(BaseModel):
    protocol: MigrationProtocol
    endpoint_info: str


class DistServeInitResponse(BaseModel):
    status: DistServeConnectionStatus
    # the control plane initialization feedback
    engine_endpoint_info: DistServeEngineEndpointInfo
    # the KVCache Transfer initialization feedback
    # To ensure generality (where endpoint_info can be initialization information
    # for different media such as RDMA, NVLink, etc.), we use a string (str) to 
    # store this information.
    kvtransfer_endpoint_info: List[DistServeKVTransferEndpointInfo]


class DistServeConnectionRequest(BaseModel):
    protocol: MigrationProtocol
    remote_engine_id: str
    remote_engine_endpoint_info: DistServeEngineEndpointInfo
    remote_kvtransfer_endpoint_info: List[DistServeKVTransferEndpointInfo]


class DistServeConnectionResponse(BaseModel):
    status: DistServeConnectionStatus


class MigrationRequest(BaseModel):
    protocol: MigrationProtocol

    remote_engine_id: str
    remote_session_id: int
    remote_token_id: int
    remote_block_ids: List[int]


class DistServeCacheFreeRequest(BaseModel):
    remote_engine_id: str
    remote_session_id: int

