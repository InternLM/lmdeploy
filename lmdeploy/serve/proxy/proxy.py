# Copyright (c) OpenMMLab. All rights reserved.

import os
from typing import Literal

import uvicorn

from lmdeploy.serve.proxy.app import create_app
from lmdeploy.serve.proxy.config import ProxyConfig, RoutingStrategy, ServingStrategy
from lmdeploy.serve.proxy.node import NodeRegistry
from lmdeploy.serve.proxy.routing import get_strategy
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

app = None


def proxy(server_name: str = '0.0.0.0',
          server_port: int = 8000,
          serving_strategy: Literal['Hybrid', 'DistServe'] = 'Hybrid',
          routing_strategy: Literal['random', 'min_expected_latency', 'min_observed_latency',
                                    'min_cache_usage'] = 'min_expected_latency',
          api_keys: list[str] | str | None = None,
          ssl: bool = False,
          log_level: str = 'INFO',
          disable_cache_status: bool = False,
          link_type: Literal['RoCE', 'IB'] = 'RoCE',
          migration_protocol: Literal['RDMA'] = 'RDMA',
          dummy_prefill: bool = False,
          **kwargs):
    """Launch the proxy server.

    Args:
        server_name (str): the server name of the proxy. Default to '0.0.0.0'.
        server_port (str): the server port. Default to 8000.
        serving_strategy ('Hybrid' | 'DistServe'): the strategy to serving.
            Hybrid default. DistServe for PD Disaggregation.
        routing_strategy ('random' | 'min_expected_latency' | 'min_observed_latency'
            | 'min_cache_usage'): the strategy to dispatch requests to nodes.
            Default to 'min_expected_latency'.
        api_keys (list[str] | str | None): Optional list of API keys.
        ssl (bool): Enable SSL.
        log_level (str): Set the log level. Default to INFO.
        disable_cache_status (bool): Whether to cache the proxy status.
        link_type: RDMA Link Type.
        migration_protocol: migration protocol for PD disaggregation.
        dummy_prefill: dummy prefill for performance profiler.
    """
    config = ProxyConfig(
        server_name=server_name,
        server_port=server_port,
        routing_strategy=RoutingStrategy.from_str(routing_strategy),
        serving_strategy=ServingStrategy(serving_strategy),
        disable_cache_status=disable_cache_status,
        migration_protocol=migration_protocol,
        link_type=link_type,
        disable_gdr=False,
        dummy_prefill=dummy_prefill,
        api_keys=api_keys if isinstance(api_keys, list) or api_keys is None else [api_keys],
        ssl=ssl,
        log_level=log_level,
    )

    registry = NodeRegistry(cache_status=not config.disable_cache_status)
    strategy = get_strategy(config.routing_strategy, registry, config)
    app_instance = create_app(config, registry, strategy)

    # Preserve app reference for docs/conf.py import
    globals()['app'] = app_instance

    if config.api_keys is not None and (tokens := [key for key in config.api_keys if key]):
        from lmdeploy.serve.utils.server_utils import AuthenticationMiddleware
        app_instance.add_middleware(AuthenticationMiddleware, tokens=tokens)

    ssl_keyfile, ssl_certfile = None, None
    if config.ssl:
        ssl_keyfile = os.environ['SSL_KEYFILE']
        ssl_certfile = os.environ['SSL_CERTFILE']

    logger.setLevel(config.log_level)
    uvicorn_log_level = os.getenv('UVICORN_LOG_LEVEL', 'info').lower()
    uvicorn.run(
        app=app_instance,
        host=config.server_name,
        port=config.server_port,
        log_level=uvicorn_log_level,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )


if __name__ == '__main__':
    import fire
    fire.Fire(proxy)
