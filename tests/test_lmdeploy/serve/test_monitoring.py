# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import http.client
import importlib
import json
import socket
import ssl
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import prometheus_client
import pytest

from lmdeploy.serve.core.health import EngineHealthMonitor
from lmdeploy.serve.monitoring import monitoring_server


def get(port, path, headers=None, tls_context=None):
    if tls_context is None:
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
    else:
        connection = http.client.HTTPSConnection('localhost', port, timeout=3, context=tls_context)
    try:
        connection.request('GET', path, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def make_monitor(**kwargs):
    engine = SimpleNamespace(is_sleeping=False,
                              health_probe=AsyncMock(return_value=dict(status='healthy', message='ready')))
    return EngineHealthMonitor(engine, **kwargs)


@pytest.fixture
def registry(monkeypatch):
    registry = prometheus_client.CollectorRegistry()
    monkeypatch.setattr(prometheus_client, 'REGISTRY', registry)
    return registry


def test_monitoring_serves_live_metrics_and_health_while_api_loop_is_blocked(registry):
    async def run():
        monitor = make_monitor()
        await monitor.probe_once()
        counter = prometheus_client.Counter('test_tokens', 'tokens', registry=registry)
        async with monitoring_server(monitor, host='127.0.0.1', port=0, enable_metrics=True) as port:
            # These synchronous HTTP calls deliberately block the owning event
            # loop until a response arrives. A listener on that loop would hang.
            counter.inc(3)
            assert b'test_tokens_total 3.0' in get(port, '/metrics')[1]
            counter.inc(4)
            status, body = get(port, '/metrics/')
            assert status == 200
            assert b'test_tokens_total 7.0' in body
            status, body = get(port, '/health')
            assert status == 200
            assert json.loads(body)['status'] == 'healthy'
            assert json.loads(body)['snapshot_age_seconds'] >= 0
            # Scrapes never submit additional work to the inference loop.
            monitor.async_engine.health_probe.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize('sleeping', [False, True])
def test_expired_health_returns_503_without_refreshing_on_api_loop(monkeypatch, sleeping):
    async def run():
        now = 100.0
        health_module = importlib.import_module('lmdeploy.serve.core.health')
        monkeypatch.setattr(health_module, 'time', SimpleNamespace(monotonic=lambda: now))
        monitor = make_monitor(unhealthy_after=10)
        monitor.async_engine.is_sleeping = sleeping
        await monitor.probe_once()
        calls = monitor.async_engine.health_probe.await_count
        async with monitoring_server(monitor, host='127.0.0.1', port=0, enable_metrics=False) as port:
            assert get(port, '/health')[0] == 200
            now += 11
            status, body = get(port, '/health')
            assert status == 503
            assert json.loads(body)['snapshot_age_seconds'] == 11
            assert 'No successful health probe' in json.loads(body)['message']
            assert monitor.async_engine.health_probe.await_count == calls
            await monitor.probe_once()
            assert get(port, '/health')[0] == 200
    asyncio.run(run())


def test_initializing_and_pending_probe_do_not_block_health():
    async def run():
        monitor = make_monitor()
        probe_started = asyncio.Event()
        finish = asyncio.Event()

        async def probe(**kwargs):
            probe_started.set()
            await finish.wait()
            return dict(status='healthy', message='ready')

        monitor.async_engine.health_probe.side_effect = probe
        pending = asyncio.create_task(monitor.probe_once())
        try:
            await probe_started.wait()
            async with monitoring_server(monitor, host='127.0.0.1', port=0, enable_metrics=False) as port:
                status, body = get(port, '/health')
                assert status == 503
                assert json.loads(body)['snapshot_age_seconds'] is None
                assert json.loads(body)['status'] == 'initializing'
                finish.set()
                await pending
                assert get(port, '/health')[0] == 200
        finally:
            finish.set()
            await pending
    asyncio.run(run())


@pytest.mark.parametrize('enable_metrics', [False, True])
def test_authentication_and_routes_match_monitoring_scope(registry, enable_metrics):
    async def run():
        monitor = make_monitor()
        await monitor.probe_once()
        async with monitoring_server(monitor, host='127.0.0.1', port=0, enable_metrics=enable_metrics,
                                     api_keys=['secret']) as port:
            assert get(port, '/health')[0] == 200
            assert get(port, '/metrics')[0] == 401
            assert get(port, '/metrics', {'Authorization': 'Bearer wrong'})[0] == 401
            authorized = {'Authorization': 'Bearer secret'}
            assert get(port, '/metrics', authorized)[0] == (200 if enable_metrics else 404)
            for path in ['/v1/models', '/docs', '/openapi.json', '/abort_all']:
                assert get(port, path, authorized)[0] == 404
    asyncio.run(run())


@pytest.mark.parametrize('cancelled', [False, True])
def test_shutdown_joins_thread_and_releases_port(cancelled):
    async def run():
        entered = asyncio.Event()
        port = None
        thread = None

        async def serve():
            nonlocal port, thread
            async with monitoring_server(make_monitor(), host='127.0.0.1', port=0, enable_metrics=False) as port:
                thread = next(t for t in threading.enumerate() if t.name == 'lmdeploy-monitoring')
                entered.set()
                if cancelled:
                    await asyncio.Event().wait()

        task = asyncio.create_task(serve())
        await entered.wait()
        if cancelled:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert not thread.is_alive()
        with socket.create_server(('127.0.0.1', port)):
            pass
    asyncio.run(run())


def test_bind_failure_closes_engine_during_api_startup(monkeypatch):
    api = importlib.import_module('lmdeploy.serve.openai.api_server')
    monkeypatch.setattr(api, 'register_to_proxy', Mock())
    engine = SimpleNamespace(start_loop=Mock(), close=Mock(), is_sleeping=False,
                              health_probe=AsyncMock(return_value=dict(status='healthy', message='ready')))
    context = api.ServerContext()

    async def run():
        with socket.create_server(('127.0.0.1', 0)) as occupied:
            lifespan = api.create_lifespan_handler(
                SimpleNamespace(enable_metrics=False), engine, context,
                monitoring_config=dict(host='127.0.0.1', port=occupied.getsockname()[1]))
            with pytest.raises(OSError):
                async with lifespan(None):
                    pytest.fail('API startup must fail when the monitoring port cannot bind')
        engine.close.assert_called_once()
        assert context.health_monitor is None
        assert not any(t.name == 'lmdeploy-monitoring' for t in threading.enumerate())
    asyncio.run(run())


def test_monitoring_uses_configured_tls(tmp_path):
    cert, key = tmp_path / 'cert.pem', tmp_path / 'key.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                    '-subj', '/CN=localhost', '-addext', 'subjectAltName=DNS:localhost',
                    '-keyout', str(key), '-out', str(cert)], check=True, capture_output=True)

    async def run():
        monitor = make_monitor()
        await monitor.probe_once()
        async with monitoring_server(monitor, host='127.0.0.1', port=0, enable_metrics=False,
                                     ssl_keyfile=str(key), ssl_certfile=str(cert)) as port:
            context = ssl.create_default_context(cafile=str(cert))
            assert get(port, '/health', tls_context=context)[0] == 200
    asyncio.run(run())
