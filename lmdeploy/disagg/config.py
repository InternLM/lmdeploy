import enum

from typing import List, Optional

from pydantic import BaseModel


class ServingStrategy(enum.Enum):
    """
    Serving Strategy.
    Hybrid: Prefill and Decode workload are co-located in one engine.

    DistServe: Prefill and Decode worload are assigned to different engines. 
        After the execution of prefill phase in Prefill Engine, 
        KVCache is migrated from Prefill to Decode Engine.
    """
    Hybrid = enum.auto()
    DistServe = enum.auto()


class EngineRole(enum.Enum):
    """
    Role of Engine.
    
    Note: In the implementation of LMDeploy-Distserve, all engine is hybrid engine technically,
        the role of engine is up to what kind of request is sent to the engine. However,
        taking implementation into the consideration, the role is still need to be identified
        when starting the engine server for the following reasons:
            1. Make sure the engine can be correctly discovered by the proxy.
            2. The create of ModelInputs is different among hybrid, prefill and decode engines
                in DP Engine (DSV3 DP + EP).
    """
    Hybrid = enum.auto()
    Prefill = enum.auto()
    Decode = enum.auto()


class MigrationBackend(enum.Enum):
    DLSlime = enum.auto()
    Mooncake = enum.auto()
    InfiniStore = enum.auto()


class MigrationProtocol(enum.Enum):
    """
    Migration Transport Protocol.

    Note: bynow, only `GPU Directed RDMA` is supported in DistServe. We preserve several protocal
        and will be implemented in the future.
    """

    # TCP for General Purpose Transport Protocol
    TCP = enum.auto()
    # IB or RoCE NICs
    RDMA = enum.auto()
    # Engine with high device-to-device link
    NVLINK = enum.auto()


class RDMALinkType(enum.Enum):
    """ RDMA Link Type. """
    IB = enum.auto()
    Ethernet = enum.auto()


class DistServeRDMAConfig(BaseModel):
    """
    DistServe RDMA Config.
    
    Warning: Only GDR is supported by now.
    """

    # RDMA with GPU Direct RDMA Access
    with_gdr: bool = True
    link_type: RDMALinkType = RDMALinkType.Ethernet


class DistServeTCPConfig(BaseModel):
    """ TODO: Add TCP Protocol """


class DistServeNVLinkConfig(BaseModel):
    """ TODO: Add NVLink Protocol """


class DistServeEngineConfig(BaseModel):
    """
    DistServe Engine Config.

    In Disaggregated LLM Serving, we need to get engine info of each
    PD Peer for the following reason:
        1. Cache: The stride of cache block for correct offset of KV Transfer.
        2. Parallel: Prefill and decode use different parallel strategy to achieve
            high SLO Attainment or high throughput. In this situation, we need
            to caclculate which prefill-decode worker peers need to connect. For
            example, prefill worker use pp4 and decode worker use tp2pp2, the
            perfill-decode worker conn peer is (0, 0), (0, 1), (1, 0), (1, 1), (2, 2),
            (2, 3), (3, 2), (3, 3). Instead, under the situation of (tp4, tp4),
            perfill-decode worker conn peer is (0, 0), (1, 1), (2, 2), (3, 3).
    """
    # parallel config
    # (dp, pp, tp, ep)
    tp_size: int
    ep_size: int
    dp_size: int
    pp_size: Optional[int]

    # Rank of DP
    dp_rank: int

    # cache config
    block_size: int
    num_cpu_blocks: int
    num_gpu_blocks: int

    # avaliable_nics
    available_nics: Optional[List[str]] = []


class DistServeConfig(BaseModel):
    """ DistServe Config. """
    serving_strategy: ServingStrategy
    distserve_transport_protocol: MigrationProtocol
    rdma_config: Optional[DistServeRDMAConfig] = None
    nvlink_config: Optional[DistServeNVLinkConfig] = None
    tcp_config: Optional[DistServeTCPConfig] = None
