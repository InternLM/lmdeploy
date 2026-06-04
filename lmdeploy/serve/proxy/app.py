# Copyright (c) OpenMMLab. All rights reserved.

import os
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lmdeploy.serve.proxy.core.config import ProxyConfig
from lmdeploy.serve.proxy.endpoint import admin, distserve, openai
from lmdeploy.serve.proxy.runtime import ProxyRuntime
from lmdeploy.serve.utils.server_utils import AuthenticationMiddleware
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

_DEFAULT_AIOHTTP_LIMIT = 1024
_DEFAULT_AIOHTTP_LIMIT_PER_HOST = 128


def _read_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == '':
        return default
    return int(value)


def _create_upstream_session() -> aiohttp.ClientSession:
    """Shared aiohttp session for forwarding to api_server replicas."""
    timeout_value = os.getenv('AIOHTTP_TIMEOUT')
    if timeout_value is None or timeout_value == '':
        timeout = aiohttp.ClientTimeout(total=None)
    else:
        timeout = aiohttp.ClientTimeout(total=int(timeout_value))

    connector = aiohttp.TCPConnector(
        limit=_read_env_int('AIOHTTP_LIMIT', _DEFAULT_AIOHTTP_LIMIT),
        limit_per_host=_read_env_int('AIOHTTP_LIMIT_PER_HOST', _DEFAULT_AIOHTTP_LIMIT_PER_HOST),
    )
    logger.info(
        f'Proxy upstream aiohttp: timeout={timeout.total}, '
        f'limit={connector.limit}, limit_per_host={connector.limit_per_host}. '
        'Override via env AIOHTTP_TIMEOUT, AIOHTTP_LIMIT, AIOHTTP_LIMIT_PER_HOST.',
    )
    return aiohttp.ClientSession(timeout=timeout, connector=connector)


def create_app(config: ProxyConfig) -> FastAPI:
    """Build FastAPI application for the proxy server."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with _create_upstream_session() as session:
            app.state.runtime = ProxyRuntime(config, session)
            yield

    app = FastAPI(docs_url='/', lifespan=lifespan)
    app.state.proxy_config = config

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    if config.api_keys:
        tokens = [key for key in config.api_keys if key]
        if tokens:
            app.add_middleware(AuthenticationMiddleware, tokens=tokens)

    app.include_router(openai.router)
    app.include_router(admin.router)
    app.include_router(distserve.router)
    return app
