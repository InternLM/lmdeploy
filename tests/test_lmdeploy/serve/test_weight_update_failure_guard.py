"""A partial distributed update must never become an inference-ready model."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from test_distributed_weight_readiness import make_endpoint

from lmdeploy.pytorch.engine.engine import Engine, LifecycleValidationError
from lmdeploy.serve.openai.endpoints.management import register
from lmdeploy.serve.openai.protocol import (
    UpdateParamsRequest,
    UpdateWeightsFromDistributedRequest,
    UpdateWeightsFromIPCRequest,
)
from lmdeploy.serve.utils.server_utils import EngineSleepingMiddleware


def bucket(finished=True):
    return UpdateWeightsFromDistributedRequest(
        names=['test.weight'], dtypes=['bfloat16'], shapes=[[1]],
        group_name='train', update_id='test-v1', finished=finished)


def route(engine, path, method='POST'):
    router = APIRouter()
    register(router, SimpleNamespace(async_engine=engine))
    return next(r.endpoint for r in router.routes if r.path == path and method in r.methods)


@pytest.mark.parametrize('tags', [None, ['weights', 'kv_cache']])
def test_failed_transfer_cannot_be_woken_or_completed(tags):
    engine, endpoint = make_endpoint(success=False)
    assert asyncio.run(endpoint(bucket())).status_code == 400
    with pytest.raises(RuntimeError, match='Restart the engine'):
        engine.wakeup(tags)
    with pytest.raises(RuntimeError, match='Restart the engine'):
        engine.complete_weights_update()
    with pytest.raises(RuntimeError, match='Restart the engine'):
        asyncio.run(engine.sleep(2))
    assert asyncio.run(endpoint(bucket())).status_code == 409
    engine.engine.update_weights_from_distributed.assert_awaited_once()
    engine.engine.wakeup.assert_not_called()
    assert engine.is_sleeping
    assert not engine.weights_update_busy


@pytest.mark.parametrize('error', [RuntimeError('worker died'), asyncio.CancelledError()])
def test_exception_or_cancellation_latches_failure(error):
    engine, endpoint = make_endpoint()
    engine.engine.update_weights_from_distributed.side_effect = error
    with pytest.raises(type(error)):
        asyncio.run(endpoint(bucket()))
    with pytest.raises(RuntimeError, match='Restart the engine'):
        engine.wakeup()
    assert engine.is_sleeping and not engine.weights_update_busy


def test_finalize_exception_latches_failure():
    engine, endpoint = make_endpoint()
    engine.engine.complete_weights_update.side_effect = RuntimeError('rank 1 finalize failed')
    with pytest.raises(RuntimeError, match='rank 1'):
        asyncio.run(endpoint(bucket()))
    with pytest.raises(RuntimeError, match='Restart the engine'):
        engine.wakeup()


def test_unfinished_bucket_blocks_recovery_until_final_completion():
    engine, endpoint = make_endpoint()
    assert asyncio.run(endpoint(bucket(False))).status_code == 200
    with pytest.raises(RuntimeError, match='incomplete'):
        engine.wakeup(['kv_cache'])
    with pytest.raises(RuntimeError, match='incomplete'):
        engine.wakeup(['weights'])
    with pytest.raises(RuntimeError, match='incomplete'):
        asyncio.run(engine.sleep(2))
    assert engine.is_sleeping
    assert asyncio.run(endpoint(bucket(True))).status_code == 200
    engine.wakeup(['kv_cache'])
    assert not engine.is_sleeping


def test_concurrent_transfer_and_wakeup_are_rejected():
    async def scenario():
        engine, endpoint = make_endpoint()
        entered, release = asyncio.Event(), asyncio.Event()

        async def wait_for_worker(request):
            entered.set()
            await release.wait()
            return True, 'done'

        engine.engine.update_weights_from_distributed.side_effect = wait_for_worker
        first = asyncio.create_task(endpoint(bucket()))
        await entered.wait()
        assert (await endpoint(bucket())).status_code == 409
        with pytest.raises(RuntimeError, match='in progress'):
            engine.wakeup(['kv_cache'])
        release.set()
        assert (await first).status_code == 200
        engine.wakeup(['kv_cache'])
        assert not engine.is_sleeping
    asyncio.run(scenario())


def test_online_update_without_sleep_is_rejected_without_receiving():
    engine, endpoint = make_endpoint(tags=())
    assert asyncio.run(endpoint(bucket())).status_code == 409
    engine.engine.update_weights_from_distributed.assert_not_awaited()
    assert not engine.is_sleeping
    assert engine.weights_update_error is None


@pytest.mark.parametrize('path', ['/generate', '/v1/chat/completions'])
def test_failure_blocks_all_http_inference_routes(path):
    engine, endpoint = make_endpoint(success=False)
    asyncio.run(endpoint(bucket()))
    app = FastAPI()
    app.add_middleware(EngineSleepingMiddleware, is_sleeping=lambda: engine.is_sleeping)
    with TestClient(app) as client:
        assert client.post(path, json={}).status_code == 503
    # No downstream routes exist: 503 demonstrates interception before routing.


def test_health_and_wakeup_report_failure_immediately():
    engine, endpoint = make_endpoint(success=False)
    asyncio.run(endpoint(bucket()))
    assert asyncio.run(route(engine, '/health', 'GET')()).status_code == 503
    request = Request({'type': 'http', 'query_string': b'tags=kv_cache'})
    assert asyncio.run(route(engine, '/wakeup')(request)).status_code == 409


def test_new_engine_is_recovery_boundary():
    failed, endpoint = make_endpoint(success=False)
    asyncio.run(endpoint(bucket()))
    fresh, _ = make_endpoint(tags=())  # Represents fresh checkpoint construction.
    assert fresh.weights_update_error is None and not fresh.is_sleeping
    assert failed.weights_update_error is not None and failed.is_sleeping


@pytest.mark.parametrize('path', ['/update_weights', '/update_weights_from_ipc'])
def test_alternative_update_endpoints_cannot_bypass_guard(path):
    engine, endpoint = make_endpoint(success=False)
    asyncio.run(endpoint(bucket(False)))
    other = route(engine, path)
    request = (UpdateParamsRequest(serialized_named_tensors={}) if path == '/update_weights'
               else UpdateWeightsFromIPCRequest(zmq_handles={}))
    response = other(request)
    if asyncio.iscoroutine(response):
        response = asyncio.run(response)
    assert response.status_code == 409


def test_sleep_must_finish_before_update_or_wakeup():
    async def scenario():
        engine, endpoint = make_endpoint(tags=())
        entered, release = asyncio.Event(), asyncio.Event()
        engine.stop_all_session = AsyncMock()

        async def wait_for_sleep(level):
            entered.set()
            await release.wait()

        engine.engine.sleep = AsyncMock(side_effect=wait_for_sleep)
        task = asyncio.create_task(engine.sleep(2))
        await entered.wait()
        assert (await endpoint(bucket())).status_code == 409
        with pytest.raises(RuntimeError, match='in progress'):
            engine.wakeup()
        release.set()
        await task
        engine.wakeup(['weights'])
        assert (await endpoint(bucket())).status_code == 200
        engine.wakeup(['kv_cache'])
        assert not engine.is_sleeping
    asyncio.run(scenario())


def test_sleep_failure_requires_full_restart():
    engine, _ = make_endpoint(tags=())
    engine.stop_all_session = AsyncMock()
    engine.engine.sleep = AsyncMock(side_effect=RuntimeError('worker disappeared'))
    with pytest.raises(RuntimeError, match='worker disappeared'):
        asyncio.run(engine.sleep(2))
    with pytest.raises(RuntimeError, match='Restart the engine'):
        engine.wakeup()
    assert engine.is_sleeping


def test_worker_wakeup_failure_latches_isolation():
    error = RuntimeError('rank wake failed')
    engine, endpoint = make_endpoint()
    engine.engine.wakeup.side_effect = error
    with pytest.raises(type(error)):
        engine.wakeup(['weights'])
    assert engine.weights_update_error and engine.weights_update_pending
    assert not engine.weights_content_ready and engine.is_sleeping
    assert asyncio.run(endpoint(bucket())).status_code == 409
    with pytest.raises(RuntimeError, match='Restart the engine'):
        engine.wakeup(['kv_cache'])


def _engine_shell():
    """Minimal real-Engine shell for direct engine-level guard checks."""
    engine = Engine.__new__(Engine)
    engine._model_state_lock = threading.Lock()
    engine._weights_update_error = None
    engine._sleeping_tags = set()
    engine._engine_loop = None
    engine.req_manager = SimpleNamespace(
        block_request_types=lambda *_: None, unblock_request_types=lambda *_: None)
    engine._cancel_and_end_all_sessions = lambda: None
    engine.scheduler = SimpleNamespace(reset_cache_after_worker_drain=lambda: None)
    engine.executor = SimpleNamespace(sleep=AsyncMock())
    return engine


def test_engine_level_latch_blocks_direct_write_completion_and_wakeup():
    engine = _engine_shell()
    engine._weights_update_error = 'RuntimeError'
    with pytest.raises(LifecycleValidationError, match='rebuild the engine'):
        engine.wakeup()
    with pytest.raises(LifecycleValidationError, match='rebuild the engine'):
        engine.complete_weights_update()
    with pytest.raises(LifecycleValidationError, match='rebuild the engine'):
        engine._execute_weights_update(lambda: None, model_write=True)


def test_engine_sleep_excludes_concurrent_model_state_writes():
    async def scenario():
        engine = _engine_shell()
        held = engine._get_model_state_lock()
        assert held.acquire(blocking=False)
        with pytest.raises(LifecycleValidationError, match='in progress'):
            await engine.sleep(1)
        held.release()

        started, release = asyncio.Event(), asyncio.Event()

        async def slow_worker_sleep(level):
            started.set()
            await release.wait()

        engine.executor.sleep = slow_worker_sleep
        task = asyncio.create_task(engine.sleep(1))
        await asyncio.wait_for(started.wait(), 5)
        # While worker sleep is still running, a direct writer is refused.
        with pytest.raises(LifecycleValidationError, match='in progress'):
            engine._execute_weights_update(lambda: None, model_write=True)
        release.set()
        await task
    asyncio.run(scenario())
