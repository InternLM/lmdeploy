# Copyright (c) OpenMMLab. All rights reserved.

from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lmdeploy.serve.proxy.core.config import ProxyConfig
from lmdeploy.serve.proxy.endpoint import admin, distserve, openai
from lmdeploy.serve.proxy.runtime import ProxyRuntime
from lmdeploy.serve.proxy.utils import AIOHTTP_TIMEOUT
from lmdeploy.serve.utils.server_utils import AuthenticationMiddleware


def create_app(config: ProxyConfig) -> FastAPI:
    """Build FastAPI application for the proxy server."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
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
