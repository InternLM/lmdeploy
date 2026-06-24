import asyncio
import json

from lmdeploy.serve.core.health import EngineHealthMonitor


class _FakeAsyncEngine:

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    async def health_probe(self, timeout: float, scheduler_stall_timeout: float) -> dict:
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

        async def health_probe(self, timeout: float, scheduler_stall_timeout: float) -> dict:
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


async def _run_health_endpoint_refreshes_cached_unhealthy_snapshot():
    import openai.types.responses.response_create_params as response_create_params

    if not hasattr(response_create_params, 'StreamOptions'):
        response_create_params.StreamOptions = dict

    from lmdeploy.serve.openai.api_server import VariableInterface, health

    class _FakeMonitor:

        def __init__(self):
            self.refresh_calls = 0

        def snapshot(self):
            return dict(status='unhealthy', message='cached timeout')

        async def refresh_snapshot(self):
            self.refresh_calls += 1
            return dict(status='healthy', message='fresh probe succeeded')

    monitor = _FakeMonitor()
    original_monitor = VariableInterface.health_monitor
    VariableInterface.health_monitor = monitor
    try:
        response = await health()
    finally:
        VariableInterface.health_monitor = original_monitor

    assert response.status_code == 200
    assert json.loads(response.body) == dict(status='healthy', message='fresh probe succeeded')
    assert monitor.refresh_calls == 1


def test_health_endpoint_refreshes_cached_unhealthy_snapshot():
    asyncio.run(_run_health_endpoint_refreshes_cached_unhealthy_snapshot())
