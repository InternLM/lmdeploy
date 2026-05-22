import asyncio
import json
from types import SimpleNamespace

from lmdeploy.messages import ScheduleMetrics
from lmdeploy.serve.core.async_engine import AsyncEngine
from lmdeploy.serve.core.health import EngineHealthMonitor
from lmdeploy.serve.openai.api_server import VariableInterface, health


class _FakeBackend:

    def __init__(self, status):
        self.status = status
        self.calls = 0

    async def get_health_status(self):
        self.calls += 1
        return self.status


class _SlowBackend:

    async def get_health_status(self):
        await asyncio.sleep(0.2)
        return dict(
            alive=True,
            message='ok',
            schedule_metrics=ScheduleMetrics(total_blocks=4, free_blocks=2),
        )


class _DoneTask:

    def done(self):
        return True

    def get_name(self):
        return 'done-task'


def _make_async_engine(backend, *, sleeping=False):
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.engine = backend
    engine.backend = 'pytorch'
    engine.model_name = 'fake-model'
    engine.is_sleeping = sleeping
    engine.sleeping_tags = {'weights'} if sleeping else set()
    engine._health_probe_task = None
    engine.session_mgr = SimpleNamespace(request_handle_pool=SimpleNamespace(num_dispatched=0))
    return engine


def test_health_probe_healthy_with_valid_metrics():
    backend = _FakeBackend(
        dict(
            alive=True,
            message='ok',
            schedule_metrics=ScheduleMetrics(total_blocks=8, free_blocks=6),
        ))
    engine = _make_async_engine(backend)

    result = asyncio.run(engine.health_probe(timeout=1.0))

    assert result == {'status': 'healthy', 'message': 'ok'}


def test_health_probe_sleeping_does_not_call_backend():
    backend = _FakeBackend(dict(alive=False, message='should not be called'))
    engine = _make_async_engine(backend, sleeping=True)

    result = asyncio.run(engine.health_probe(timeout=1.0))

    assert result == {'status': 'sleeping', 'message': 'Engine is sleeping.'}
    assert backend.calls == 0


def test_health_probe_rejects_invalid_schedule_metrics():
    backend = _FakeBackend(
        dict(
            alive=True,
            message='ok',
            schedule_metrics=ScheduleMetrics(total_blocks=0, free_blocks=0),
        ))
    engine = _make_async_engine(backend)

    result = asyncio.run(engine.health_probe(timeout=1.0))

    assert result['status'] == 'unhealthy'
    assert set(result) == {'status', 'message'}
    assert 'Invalid total_blocks' in result['message']


def test_health_probe_rejects_idle_metrics_when_request_is_in_backend():
    backend = _FakeBackend(
        dict(
            alive=True,
            message='ok',
            schedule_metrics=ScheduleMetrics(total_blocks=8, free_blocks=6),
        ))
    engine = _make_async_engine(backend)
    engine.session_mgr.request_handle_pool.num_dispatched = 1

    result = asyncio.run(engine.health_probe(timeout=1.0))

    assert result['status'] == 'unhealthy'
    assert 'no active or waiting sequences' in result['message']


def test_health_probe_accepts_busy_metrics_when_request_is_in_backend():
    backend = _FakeBackend(
        dict(
            alive=True,
            message='ok',
            schedule_metrics=ScheduleMetrics(active_seqs=1, total_blocks=8, free_blocks=6),
        ))
    engine = _make_async_engine(backend)
    engine.session_mgr.request_handle_pool.num_dispatched = 1

    result = asyncio.run(engine.health_probe(timeout=1.0))

    assert result == {'status': 'healthy', 'message': 'ok'}


def test_health_probe_timeout_prevents_overlapping_probe():
    engine = _make_async_engine(_SlowBackend())

    async def _run():
        first = await engine.health_probe(timeout=0.01)
        second = await engine.health_probe(timeout=0.01)
        await asyncio.sleep(0.25)
        return first, second

    first, second = asyncio.run(_run())

    assert first['status'] == 'unhealthy'
    assert set(first) == {'status', 'message'}
    assert 'timed out' in first['message']
    assert second['status'] == 'unhealthy'
    assert set(second) == {'status', 'message'}
    assert 'still pending' in second['message']


def test_health_route_returns_monitor_snapshot():
    class _Monitor:

        def snapshot(self):
            return dict(status='healthy', message='ok')

    old_monitor = VariableInterface.health_monitor
    try:
        VariableInterface.health_monitor = _Monitor()
        response = asyncio.run(health())
        body = json.loads(response.body)
        assert response.status_code == 200
        assert body['status'] == 'healthy'
        assert body == {'status': 'healthy', 'message': 'ok'}
    finally:
        VariableInterface.health_monitor = old_monitor


def test_health_route_without_monitor_is_unhealthy():
    old_monitor = VariableInterface.health_monitor
    try:
        VariableInterface.health_monitor = None
        response = asyncio.run(health())
        body = json.loads(response.body)
        assert response.status_code == 503
        assert body['status'] == 'unhealthy'
    finally:
        VariableInterface.health_monitor = old_monitor


def test_health_monitor_probe_once_records_success():
    engine = _make_async_engine(
        _FakeBackend(
            dict(
                alive=True,
                message='ok',
                schedule_metrics=ScheduleMetrics(total_blocks=4, free_blocks=2),
            )))
    monitor = EngineHealthMonitor(engine, poll_interval=100, probe_timeout=1, unhealthy_after=15)

    asyncio.run(monitor.probe_once())

    snapshot = monitor.snapshot()
    assert snapshot['status'] == 'healthy'
    assert snapshot == {'status': 'healthy', 'message': 'ok'}


def test_pytorch_engine_health_fails_when_loop_dead():
    from lmdeploy.pytorch.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.req_manager = SimpleNamespace(is_loop_alive=lambda: False)

    result = asyncio.run(engine.get_health_status())

    assert result['alive'] is False
    assert 'request loop' in result['message']


def test_pytorch_engine_health_fails_when_engine_loop_task_done():
    from lmdeploy.pytorch.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.req_manager = SimpleNamespace(is_loop_alive=lambda: True)
    engine._loop_main = None
    engine._engine_loop = SimpleNamespace(tasks={_DoneTask()})

    result = asyncio.run(engine.get_health_status())

    assert result['alive'] is False
    assert 'engine loop task' in result['message']


def test_pytorch_engine_health_returns_metrics_when_tasks_alive():
    from lmdeploy.pytorch.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.req_manager = SimpleNamespace(is_loop_alive=lambda: True)
    engine._loop_main = None
    engine._engine_loop = SimpleNamespace(tasks=set())
    engine.scheduler = SimpleNamespace(schedule_metrics=ScheduleMetrics(total_blocks=4, free_blocks=1))

    result = asyncio.run(engine.get_health_status())

    assert result['alive'] is True
    assert result['schedule_metrics'].free_blocks == 1
