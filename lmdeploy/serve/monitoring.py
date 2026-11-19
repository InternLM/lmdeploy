# Copyright (c) OpenMMLab. All rights reserved.
"""Serve monitoring independently of the inference API event loop."""
import asyncio
import re
from contextlib import ExitStack, asynccontextmanager
from threading import Event, Thread

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.routing import Mount

from lmdeploy.serve.core.health import EngineHealthMonitor
from lmdeploy.serve.utils.server_utils import AuthenticationMiddleware
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')
_THREAD_JOIN_TIMEOUT = 5


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
                            draining: Event,
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
        if draining.is_set():
            data.update(status='unhealthy', message='API server is shutting down.')
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
                            ssl_keyfile=ssl_keyfile,
                            ssl_certfile=ssl_certfile,
                            timeout_graceful_shutdown=1)
    # Validate TLS and bind in the caller so startup errors fail the API startup.
    config.load()
    loop = asyncio.get_running_loop()
    # Use Uvicorn's address resolution and binding semantics, then hand socket
    # duplicates to the monitoring loop without accepting on the API loop.
    async with await loop.create_server(asyncio.Protocol, host, port, start_serving=False) as listener:
        with ExitStack() as stack:
            sockets = [stack.enter_context(sock.dup()) for sock in listener.sockets]
            server = uvicorn.Server(config)

            def run():
                try:
                    server.run(sockets=sockets)
                finally:
                    if not server.should_exit:
                        logger.error('Monitoring server exited unexpectedly.')

            thread = Thread(target=run, name='lmdeploy-monitoring', daemon=True)
            thread.start()
            try:
                while not server.started:
                    if not thread.is_alive():
                        raise RuntimeError('Monitoring server failed to start.')
                    await asyncio.sleep(0.01)
                port = sockets[0].getsockname()[1]
                logger.info('Monitoring server listening on %s:%s', host, port)
                yield port
            finally:
                server.should_exit = True
                # Bound the wait even if a handler blocks the monitoring loop.
                thread.join(timeout=_THREAD_JOIN_TIMEOUT)
                if thread.is_alive():
                    logger.error('Monitoring thread did not stop within %s seconds; continuing engine cleanup.',
                                 _THREAD_JOIN_TIMEOUT)
