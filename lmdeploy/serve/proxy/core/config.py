# Copyright (c) OpenMMLab. All rights reserved.

import enum
from dataclasses import dataclass

from lmdeploy.pytorch.disagg.config import DistServeRDMAConfig, ServingStrategy
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol


class RoutingStrategy(enum.Enum):
    """Strategy to dispatch requests to nodes."""

    RANDOM = enum.auto()
    MIN_EXPECTED_LATENCY = enum.auto()
    MIN_OBSERVED_LATENCY = enum.auto()

    @classmethod
    def from_str(cls, name: str) -> 'RoutingStrategy':
        """Get strategy from string."""
        if name == 'random':
            return cls.RANDOM
        if name == 'min_expected_latency':
            return cls.MIN_EXPECTED_LATENCY
        if name == 'min_observed_latency':
            return cls.MIN_OBSERVED_LATENCY
        raise ValueError(f'Invalid strategy: {name}. Supported: random, '
                         f'min_expected_latency, min_observed_latency.')


@dataclass
class ProxyConfig:
    """Runtime configuration for the proxy server."""

    serving_strategy: ServingStrategy = ServingStrategy.Hybrid
    routing_strategy: RoutingStrategy = RoutingStrategy.MIN_EXPECTED_LATENCY
    migration_protocol: MigrationProtocol = MigrationProtocol.RDMA
    rdma_config: DistServeRDMAConfig | None = None
    dummy_prefill: bool = False
    server_name: str = '0.0.0.0'
    server_port: int = 8000
    api_keys: list[str] | None = None
    ssl: bool = False
    log_level: str = 'INFO'
