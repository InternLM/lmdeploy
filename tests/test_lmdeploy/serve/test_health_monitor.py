import asyncio
import json

from lmdeploy.serve.core.health import EngineHealthMonitor


class _FakeAsyncEngine:

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0
        self.is_sleeping = False

    async def health_probe(self, scheduler_stall_timeout: float) -> dict:
        self.calls += 1
        return self.results.pop(0)


async def _run_refresh_snapshot_updates_cached_unhealthy_status():
    engine = _FakeAsyncEngine([
        dict(status='unhealthy', message='Backend health probe timed out after 10.0s.'),
        dict(status='healthy', message='PyTorch engine is healthy.'),
    ])
    monitor = EngineHealthMonitor(engine)

    await monitor.probe_once()
    assert monitor.snapshot()['status'] == 'unhealthy'

    refreshed = await monitor.refresh_snapshot()

    assert refreshed == dict(status='healthy', message='PyTorch engine is healthy.')
    assert monitor.snapshot() == refreshed
    assert engine.calls == 2


def test_refresh_snapshot_updates_cached_unhealthy_status():
    asyncio.run(_run_refresh_snapshot_updates_cached_unhealthy_status())


async def _run_concurrent_refresh_snapshot_serializes_probes():
    probe_started = asyncio.Event()
    allow_probe_to_finish = asyncio.Event()

    class _BlockingAsyncEngine:

        def __init__(self):
            self.calls = 0
            self.is_sleeping = False

        async def health_probe(self, scheduler_stall_timeout: float) -> dict:
            self.calls += 1
            probe_started.set()
            await allow_probe_to_finish.wait()
            return dict(status='healthy', message='PyTorch engine is healthy.')

    engine = _BlockingAsyncEngine()
    monitor = EngineHealthMonitor(engine)

    first = asyncio.create_task(monitor.refresh_snapshot())
    second = asyncio.create_task(monitor.refresh_snapshot())
    await asyncio.wait_for(probe_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert engine.calls == 1

    allow_probe_to_finish.set()
    await asyncio.gather(first, second)
    assert engine.calls == 2


def test_concurrent_refresh_snapshot_serializes_probes():
    asyncio.run(_run_concurrent_refresh_snapshot_serializes_probes())


async def _run_late_health_probe_result_is_reused():
    allow_probe_to_finish = asyncio.Event()
    probe_finished = asyncio.Event()

    class _SlowAsyncEngine:

        def __init__(self):
            self.calls = 0
            self.is_sleeping = False

        async def health_probe(self, scheduler_stall_timeout: float) -> dict:
            self.calls += 1
            await allow_probe_to_finish.wait()
            probe_finished.set()
            return dict(status='healthy', message='Late probe succeeded.')

    engine = _SlowAsyncEngine()
    monitor = EngineHealthMonitor(engine, probe_timeout=0.01)

    await monitor.probe_once()
    assert monitor.snapshot()['status'] == 'unhealthy'
    assert 'timed out' in monitor.snapshot()['message']
    assert engine.calls == 1

    await monitor.probe_once()
    assert engine.calls == 1
    assert monitor.snapshot()['status'] == 'unhealthy'

    engine.is_sleeping = True
    await monitor.probe_once()
    assert monitor.snapshot() == dict(status='sleeping', message='Engine is sleeping.')
    assert engine.calls == 1

    engine.is_sleeping = False
    allow_probe_to_finish.set()
    await probe_finished.wait()
    await monitor.probe_once()

    assert monitor.snapshot() == dict(status='healthy', message='Late probe succeeded.')
    assert engine.calls == 1


def test_late_health_probe_result_is_reused():
    asyncio.run(_run_late_health_probe_result_is_reused())


async def _run_health_endpoint_refreshes_cached_unhealthy_snapshot():
    from lmdeploy.serve.openai.api_server import ServerContext
    from lmdeploy.serve.openai.endpoints import create_openai_router

    class _FakeMonitor:

        def __init__(self):
            self.refresh_calls = 0

        def snapshot(self):
            return dict(status='unhealthy', message='cached timeout')

        async def refresh_snapshot(self):
            self.refresh_calls += 1
            return dict(status='healthy', message='fresh probe succeeded')

    monitor = _FakeMonitor()
    context = ServerContext()
    context.health_monitor = monitor
    router = create_openai_router(context)
    health = next(route.endpoint for route in router.routes if route.path == '/health')
    response = await health()

    assert response.status_code == 200
    assert json.loads(response.body) == dict(status='healthy', message='fresh probe succeeded')
    assert monitor.refresh_calls == 1


def test_health_endpoint_refreshes_cached_unhealthy_snapshot():
    asyncio.run(_run_health_endpoint_refreshes_cached_unhealthy_snapshot())


async def _run_health_endpoint_does_not_refresh_initializing_snapshot():
    from lmdeploy.serve.openai.api_server import ServerContext
    from lmdeploy.serve.openai.endpoints import create_openai_router

    class _FakeMonitor:

        def __init__(self):
            self.refresh_calls = 0

        def snapshot(self):
            return dict(status='initializing', message='Engine health monitor is starting.')

        async def refresh_snapshot(self):
            self.refresh_calls += 1
            return dict(status='healthy', message='fresh probe succeeded')

    monitor = _FakeMonitor()
    context = ServerContext()
    context.health_monitor = monitor
    router = create_openai_router(context)
    health = next(route.endpoint for route in router.routes if route.path == '/health')
    response = await health()

    assert response.status_code == 503
    assert json.loads(response.body) == dict(status='initializing', message='Engine health monitor is starting.')
    assert monitor.refresh_calls == 0


def test_health_endpoint_does_not_refresh_initializing_snapshot():
    asyncio.run(_run_health_endpoint_does_not_refresh_initializing_snapshot())
