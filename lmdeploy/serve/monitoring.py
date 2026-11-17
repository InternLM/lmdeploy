# Copyright (c) OpenMMLab. All rights reserved.
"""Serve monitoring independently of the inference API event loop."""
import asyncio
import re
import socket
from contextlib import asynccontextmanager
from threading import Thread

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.routing import Mount

from lmdeploy.serve.core.health import EngineHealthMonitor
from lmdeploy.serve.utils.server_utils import AuthenticationMiddleware
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def mount_metrics(app: FastAPI, enable_metrics: bool):
    if not enable_metrics:
        return
    from prometheus_client import REGISTRY, make_asgi_app

    metrics_route = Mount('/metrics', make_asgi_app(registry=REGISTRY))
    # Avoid a 307 redirect for /metrics.
    metrics_route.path_regex = re.compile('^/metrics(?P<path>.*)$')
    app.routes.append(metrics_route)


@asynccontextmanager
async def monitoring_server(health_monitor: EngineHealthMonitor,
                            *,
                            host: str,
                            port: int,
                            enable_metrics: bool,
                            api_keys: list[str] | None = None,
                            ssl_keyfile: str | None = None,
                            ssl_certfile: str | None = None):
    """Own an optional monitoring listener for one API process's lifespan.

    The thread reads the shared Prometheus registry and health snapshots only; backend probes and metric updates remain
    on their original event loop.
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get('/health')
    async def health():
        # Never wait for a backend probe on the potentially blocked API loop.
        data = health_monitor.snapshot(include_age=True)
        status_code = 200 if data['status'] in ('healthy', 'sleeping') else 503
        return JSONResponse(data, status_code=status_code)

    mount_metrics(app, enable_metrics)
    if api_keys and (tokens := [key for key in api_keys if key]):
        app.add_middleware(AuthenticationMiddleware, tokens=tokens)

    config = uvicorn.Config(app,
                            host=host,
                            port=port,
                            loop='asyncio',
                            lifespan='off',
                            ws='none',
                            log_config=None,
                            access_log=False,
                            ssl_keyfile=ssl_keyfile,
                            ssl_certfile=ssl_certfile,
                            timeout_graceful_shutdown=1)
    # Validate TLS and bind in the caller so startup errors fail the API startup.
    config.load()
    family = socket.AF_INET6 if ':' in host else socket.AF_INET
    with socket.create_server((host, port), family=family) as sock:
        server = uvicorn.Server(config)
        thread = Thread(target=server.run, kwargs={'sockets': [sock]}, name='lmdeploy-monitoring', daemon=True)
        thread.start()
        try:
            while not server.started:
                if not thread.is_alive():
                    raise RuntimeError('Monitoring server failed to start.')
                await asyncio.sleep(0.01)
            port = sock.getsockname()[1]
            logger.info('Monitoring server listening on %s:%s', host, port)
            yield port
        finally:
            server.should_exit = True
            # Join during lifespan shutdown so cancellation cannot leave a
            # listener behind. Uvicorn bounds in-flight request shutdown above.
            thread.join()
