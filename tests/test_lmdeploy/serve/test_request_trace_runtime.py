# Copyright (c) OpenMMLab. All rights reserved.
"""Real import and CORS smoke tests, requiring the installed serving
environment."""
import asyncio
import importlib

import pytest

pytest.importorskip('torch')


def test_trace_real_import_graph():
    for module in (
        'lmdeploy.trace',
        'lmdeploy.serve.core.async_engine',
        'lmdeploy.serve.openai.api_server',
        'lmdeploy.pytorch.kv_connector.mooncake.store.scheduler',
        'lmdeploy.pytorch.paging.kv_load_coordinator',
    ):
        importlib.import_module(module)


def test_outer_trace_header_is_exposed_to_browser(monkeypatch, tmp_path):
    pytest.importorskip('httpx')
    from httpx import ASGITransport, AsyncClient
    from starlette.applications import Starlette
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from lmdeploy.serve.observability import request_trace

    monkeypatch.setattr(request_trace, 'ENABLED', True)
    writer = request_trace.JsonlWriter(tmp_path)
    monkeypatch.setattr(request_trace, 'get_writer', lambda: writer)

    async def endpoint(request):
        return JSONResponse({'ok': True})

    app = Starlette(routes=[Route('/v1/completions', endpoint, methods=['POST'])])
    app.add_middleware(CORSMiddleware, allow_origins=['https://client.example'],
                       expose_headers=[request_trace.HEADER])
    app = request_trace.RequestTraceMiddleware(app)
    try:
        async def request():
            async with AsyncClient(transport=ASGITransport(app),
                                   base_url='http://testserver', trust_env=False) as client:
                return await client.post('/v1/completions',
                                         headers={'Origin': 'https://client.example',
                                                  'x-request-id': 'req_browser_123456'})
        response = asyncio.run(request())
        assert response.status_code == 200
        assert request_trace.HEADER in response.headers
        exposed = response.headers['access-control-expose-headers'].lower()
        assert request_trace.HEADER in exposed
    finally:
        writer.close()


@pytest.mark.parametrize('fail', [False, True])
def test_traced_fastapi_preserves_interface_and_framework_responses(monkeypatch, tmp_path, fail):
    import json
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from lmdeploy.serve.observability import request_trace
    from lmdeploy.serve.openai.api_server import _TracingFastAPI

    monkeypatch.setattr(request_trace, 'ENABLED', True)
    monkeypatch.setattr(request_trace, 'LOG_ENABLED', True)
    writer = request_trace.JsonlWriter(tmp_path)
    monkeypatch.setattr(request_trace, 'get_writer', lambda: writer)
    lifecycle = []

    @asynccontextmanager
    async def lifespan(app):
        lifecycle.append('start')
        yield
        lifecycle.append('stop')

    app = _TracingFastAPI(lifespan=lifespan)
    app.state.marker = 'preserved'

    @app.post('/v1/completions')
    async def endpoint():
        if fail:
            raise RuntimeError('private failure detail')
        return {'ok': True}

    assert isinstance(app, FastAPI)
    assert app.state.marker == 'preserved'
    assert '/v1/completions' in app.openapi()['paths']
    app.middleware_stack = app.build_middleware_stack()
    assert isinstance(app.middleware_stack, request_trace.RequestTraceMiddleware)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post('/v1/completions', headers={'x-request-id': 'req_interface_123456'})
        assert response.status_code == (500 if fail else 200)
        assert response.headers['x-request-id'] == 'req_interface_123456'
        assert lifecycle == ['start', 'stop']
    finally:
        writer.close()
    records = [json.loads(line) for path in tmp_path.rglob('*.jsonl')
               for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]['http_status'] == response.status_code
    assert records[0]['status'] == ('error' if fail else 'success')
    assert 'private failure detail' not in json.dumps(records)
