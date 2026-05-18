# Copyright (c) OpenMMLab. All rights reserved.

import enum
import os

from pydantic import BaseModel

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

LATENCY_DEQUE_LEN = 15
AIOHTTP_TIMEOUT = os.getenv('AIOHTTP_TIMEOUT', None)
if AIOHTTP_TIMEOUT is not None:
    AIOHTTP_TIMEOUT = int(AIOHTTP_TIMEOUT)
logger.info(f"AIOHTTP_TIMEOUT set to {AIOHTTP_TIMEOUT}. It can be modified before launching the proxy server "
            'through env variable AIOHTTP_TIMEOUT')


class RoutingStrategy(str, enum.Enum):
    """Strategy to dispatch requests to nodes."""
    RANDOM = 'random'
    MIN_EXPECTED_LATENCY = 'min_expected_latency'
    MIN_OBSERVED_LATENCY = 'min_observed_latency'
    MIN_CACHE_USAGE = 'min_cache_usage'

    @classmethod
    def from_str(cls, name):
        """Get strategy from string."""
        try:
            return cls(name)
        except ValueError:
            raise ValueError(f"Invalid strategy: {name}. Supported: random, "
                             f"min_expected_latency, min_observed_latency, min_cache_usage.")


class ServingStrategy(str, enum.Enum):
    """Serving strategy for proxy."""
    HYBRID = 'Hybrid'
    DIST_SERVE = 'DistServe'


class ErrorCodes(enum.Enum):
    """Error codes."""
    MODEL_NOT_FOUND = 10400
    SERVICE_UNAVAILABLE = 10401
    API_TIMEOUT = 10402


err_msg = {
    ErrorCodes.MODEL_NOT_FOUND: 'The request model name does not exist in the model list.',
    ErrorCodes.SERVICE_UNAVAILABLE: 'The service is unavailable now. May retry later.',
    ErrorCodes.API_TIMEOUT: 'Failed to get response after a period of time',
}


class APIServerException(Exception):

    def __init__(self, status_code: int, body: bytes, headers: dict | None = None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        if 'content-type' not in self.headers:
            self.headers['content-type'] = 'application/json'


class ProxyConfig(BaseModel):
    """Configuration for the proxy server."""
    server_name: str = '0.0.0.0'
    server_port: int = 8000
    routing_strategy: RoutingStrategy = RoutingStrategy.MIN_EXPECTED_LATENCY
    serving_strategy: ServingStrategy = ServingStrategy.HYBRID
    disable_cache_status: bool = False
    metrics_poll_interval: float = float(os.getenv('LMDEPLOY_PROXY_POLL_METRICS_INTERVAL', '5.0'))
    migration_protocol: str = 'RDMA'
    link_type: str = 'RoCE'
    disable_gdr: bool = False
    dummy_prefill: bool = False
    api_keys: list[str] | None = None
    ssl: bool = False
    log_level: str = 'INFO'
