# Copyright (c) OpenMMLab. All rights reserved.

from dataclasses import dataclass

from lmdeploy.pytorch.disagg.config import DistServeRDMAConfig, ServingStrategy
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.serve.proxy.utils import RoutingStrategy


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
