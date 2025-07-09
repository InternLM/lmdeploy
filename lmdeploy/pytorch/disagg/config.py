# Copyright (c) OpenMMLab. All rights reserved.
import enum
from typing import Optional

from pydantic import BaseModel


class ServingStrategy(enum.Enum):
    """Serving Strategy.

    Attributes:
        Hybrid: Prefill and Decode workload are co-located in one engine.
        DistServe: Prefill and Decode workload are assigned to different
            engines. After the execution of prefill phase in Prefill Engine,
            KVCache is migrated from Prefill to Decode Engine.
    """

    Hybrid = enum.auto()
    DistServe = enum.auto()


class EngineRole(enum.Enum):
    """Role of Engine.

    Note: In the implementation of LMDeploy-Distserve, all engine is hybrid
        engine technically, the role of engine is up to what kind of request is
        sent to the engine. However, taking implementation into the consideration,
        the role is still need to be identified when starting the engine server
        for the following reasons:
            1. Make sure the engine can be correctly discovered by the proxy.
            2. The create of ModelInputs is different among hybrid, prefill and
                decode engines in DP Engine (DSV3 DP + EP).
    """

    Hybrid = enum.auto()
    Prefill = enum.auto()
    Decode = enum.auto()


class MigrationBackend(enum.Enum):
    """Migration Backend."""

    DLSlime = enum.auto()
    Mooncake = enum.auto()


class RDMALinkType(enum.Enum):
    """RDMA Link Type."""

    IB = enum.auto()
    RoCE = enum.auto()


class DistServeRDMAConfig(BaseModel):
    """DistServe RDMA Config.

    Args:
        with_gdr: default to True.
        link_type: default to `RDMALinkType.RoCE`.

    Warning: Only GDR is supported by now.
    Warning: Technically, both RoCE and IB are supported.
        However, IB mode is not tested because of unavailable
        testing envoriment.
    """

    # RDMA with GPU Direct RDMA Access
    with_gdr: bool = True
    link_type: RDMALinkType = RDMALinkType.RoCE


class DistServeTCPConfig(BaseModel):
    """TODO: Add TCP Protocol"""


class DistServeNVLinkConfig(BaseModel):
    """TODO: Add NVLink Protocol"""


class DistServeEngineConfig(BaseModel):
    """DistServe Engine Config.

    In Disaggregated LLM Serving, we need to get engine info of each
    PD Peer for the following reason:
        1. Cache: The stride of cache block for correct offset of KV Transfer.
        2. Parallel: Prefill and decode use different parallel strategy to
            achieve high SLO Attainment or high throughput. In this situation,
            we need to caclculate which prefill-decode worker peers need to connect.
            For example, prefill worker use pp4 and decode worker use tp2pp2,
            the perfill-decode worker conn peer is (0, 0), (0, 1), (1, 0), (1, 1),
            (2, 2), (2, 3), (3, 2), (3, 3). Instead, under the situation of
            (tp4, tp4), perfill-decode worker conn peer is (0, 0), (1, 1), (2, 2),
            (3, 3).
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


class MooncakeEngineConfig(DistServeEngineConfig):
    """Mooncake Transfer Engine Config.

    TODO: Support more specific config for Mooncake.
    """
    pass
