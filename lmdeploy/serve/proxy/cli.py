# Copyright (c) OpenMMLab. All rights reserved.

import os
from typing import Literal

import uvicorn

from lmdeploy.pytorch.disagg.config import DistServeRDMAConfig, RDMALinkType, ServingStrategy
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.serve.proxy.app import create_app
from lmdeploy.serve.proxy.core.config import ProxyConfig
from lmdeploy.serve.proxy.utils import RoutingStrategy
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def proxy(server_name: str = '0.0.0.0',
          server_port: int = 8000,
          serving_strategy: Literal['Hybrid', 'DistServe'] = 'Hybrid',
          routing_strategy: Literal['random', 'min_expected_latency', 'min_observed_latency'] = 'min_expected_latency',
          api_keys: list[str] | str | None = None,
          ssl: bool = False,
          log_level: str = 'INFO',
          link_type: Literal['RoCE', 'IB'] = 'RoCE',
          migration_protocol: Literal['RDMA', 'NVLINK'] = 'RDMA',
          dummy_prefill: bool = False,
          disable_gdr: bool = False,
          **kwargs):
    """Launch the proxy server."""
    keys: list[str] | None = None
    if api_keys is not None:
        if isinstance(api_keys, str):
            keys = [api_keys] if api_keys else None
        else:
            keys = list(api_keys)

    config = ProxyConfig(
        serving_strategy=ServingStrategy[serving_strategy],
        routing_strategy=RoutingStrategy.from_str(routing_strategy),
        migration_protocol=MigrationProtocol[migration_protocol],
        rdma_config=DistServeRDMAConfig(
            link_type=RDMALinkType[link_type],
            with_gdr=not disable_gdr,
        ),
        dummy_prefill=dummy_prefill,
        server_name=server_name,
        server_port=server_port,
        api_keys=keys,
        ssl=ssl,
        log_level=log_level,
    )

    app = create_app(config)

    ssl_keyfile, ssl_certfile = None, None
    if ssl:
        ssl_keyfile = os.environ.get('SSL_KEYFILE')
        ssl_certfile = os.environ.get('SSL_CERTFILE')
        if not ssl_keyfile or not ssl_certfile:
            raise ValueError('SSL is enabled but SSL_KEYFILE and SSL_CERTFILE must be set.')

    logger.setLevel(log_level)
    uvicorn_log_level = os.getenv('UVICORN_LOG_LEVEL', 'info').lower()
    uvicorn.run(
        app=app,
        host=server_name,
        port=server_port,
        log_level=uvicorn_log_level,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )


if __name__ == '__main__':
    import fire

    fire.Fire(proxy)
