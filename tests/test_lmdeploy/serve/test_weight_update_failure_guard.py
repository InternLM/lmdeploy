"""Failure isolation: a partial weight transaction never becomes servable."""
import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from test_distributed_weight_readiness import bucket, make_endpoint

from lmdeploy.pytorch.engine.engine import LifecycleValidationError, WeightState
from lmdeploy.serve.openai.endpoints.management import register
from lmdeploy.serve.openai.protocol import UpdateParamsRequest, UpdateWeightsFromIPCRequest
from lmdeploy.serve.utils.server_utils import EngineSleepingMiddleware


def _serve_gate(engine):
    return lambda: engine.is_sleeping or engine.weight_state_visible() != 'ready'


def legacy(finished=False):
    return UpdateParamsRequest(serialized_named_tensors={}, finished=finished)


def _routes(engine):
    router = APIRouter()
    register(router, SimpleNamespace(async_engine=engine))
    return {(r.path, method): r.endpoint for r in router.routes for method in r.methods}


def test_poisoned_engine_refuses_write_completion_wakeup_and_sleep():
    engine, endpoint = make_endpoint(success=False)
    assert asyncio.run(endpoint(bucket())).status_code == 400
    py = engine.engine
    assert py.weight_state is WeightState.POISONED
    with pytest.raises(RuntimeError, match='poisoned'):
        engine.wakeup(['kv_cache'])
    with pytest.raises(LifecycleValidationError, match='rebuild'):
        py.wakeup(['kv_cache'])
    with pytest.raises(LifecycleValidationError, match='rebuild'):
        py.complete_weights_update()
    with pytest.raises(LifecycleValidationError, match='rebuild'):
        py._execute_weights_update(lambda: None, final=True)
    with pytest.raises(RuntimeError, match='poisoned'):
        asyncio.run(engine.sleep(2))
    assert asyncio.run(endpoint(bucket())).status_code == 409
    py.executor.update_weights_from_distributed.assert_called_once()
    assert engine.is_sleeping


def test_worker_exception_poisons_and_reports_rebuild():
    engine, endpoint = make_endpoint()
    engine.engine.executor.update_weights_from_distributed.side_effect = RuntimeError('worker died')
    response = asyncio.run(endpoint(bucket()))
    assert response.status_code == 503
    assert 'rebuilt' in json.loads(response.body)['message']
    assert engine.engine.weight_state is WeightState.POISONED
    with pytest.raises(RuntimeError, match='poisoned'):
        engine.wakeup()


def test_cancelled_waiter_leaves_the_shielded_write_running():
    async def scenario():
        engine, endpoint = make_endpoint()
        entered, release = threading.Event(), threading.Event()
        engine.engine.executor.update_weights_from_distributed.side_effect = \
            lambda req: (entered.set(), release.wait(5), (True, 'done'))[2]
        task = asyncio.create_task(endpoint(bucket()))
        await asyncio.wait_for(asyncio.to_thread(entered.wait), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert engine.engine.weight_state.value == 'updating'  # write still owns the model
        release.set()
        await asyncio.sleep(0.1)
        assert engine.engine.weight_state.value == 'ready'  # resolved to committed
    asyncio.run(scenario())


def test_open_transaction_refuses_wakeup_until_final_bucket():
    engine, endpoint = make_endpoint()
    assert asyncio.run(endpoint(bucket(False))).status_code == 200
    with pytest.raises(RuntimeError, match='updating'):
        engine.wakeup(['kv_cache'])
    assert asyncio.run(endpoint(bucket(True))).status_code == 200
    engine.wakeup(['kv_cache'])
    assert not engine.is_sleeping


def test_concurrent_wakeup_races_the_transfer_and_is_refused():
    async def scenario():
        engine, endpoint = make_endpoint()
        entered, release = threading.Event(), threading.Event()
        engine.engine.executor.update_weights_from_distributed.side_effect = \
            lambda req: (entered.set(), release.wait(5), (True, 'done'))[2]
        first = asyncio.create_task(endpoint(bucket(False)))
        await asyncio.wait_for(asyncio.to_thread(entered.wait), 5)
        with pytest.raises((RuntimeError, LifecycleValidationError)):
            engine.wakeup(['kv_cache'])
        release.set()
        assert (await first).status_code == 200
    asyncio.run(scenario())


def test_online_update_without_sleep_is_rejected_without_receiving():
    engine, endpoint = make_endpoint(tags=())
    assert asyncio.run(endpoint(bucket())).status_code == 409
    engine.engine.executor.update_weights_from_distributed.assert_not_called()
    assert engine.engine.weight_state is WeightState.READY
    assert engine.weight_state_visible() == 'ready'


@pytest.mark.parametrize('path', ['/generate', '/v1/chat/completions'])
def test_failed_update_blocks_all_http_inference_routes(path):
    engine, endpoint = make_endpoint(success=False)
    asyncio.run(endpoint(bucket()))
    assert engine.weight_state_visible() == 'poisoned'
    app = FastAPI()
    app.add_middleware(EngineSleepingMiddleware, is_sleeping=_serve_gate(engine))
    with TestClient(app) as client:
        assert client.post(path, json={}).status_code == 503
    # No downstream routes exist: 503 demonstrates interception before routing.


def test_open_transaction_and_poison_report_through_health():
    engine, endpoint = make_endpoint()
    health = _routes(engine)[('/health', 'GET')]
    assert asyncio.run(endpoint(bucket(False))).status_code == 200
    assert asyncio.run(health()).status_code == 503  # updating
    engine.engine._weight_state = WeightState.POISONED
    body = asyncio.run(health())
    assert body.status_code == 503 and 'Rebuild' in body.body.decode()


def test_alternative_update_endpoints_cannot_bypass_poison():
    engine, endpoint = make_endpoint(success=False)
    asyncio.run(endpoint(bucket()))
    routes = _routes(engine)
    assert asyncio.run(routes[('/update_weights', 'POST')](legacy())).status_code == 409
    assert asyncio.run(
        routes[('/update_weights_from_ipc', 'POST')](UpdateWeightsFromIPCRequest(zmq_handles={}))
    ).status_code == 409
    engine.engine.executor.update_params.assert_not_called()
    engine.engine.executor.update_weights_from_ipc.assert_not_called()


def test_sleep_must_finish_before_update_or_wakeup():
    async def scenario():
        engine, _ = make_endpoint(tags=())
        started, release = asyncio.Event(), asyncio.Event()

        async def slow_worker_sleep(level):
            started.set()
            await release.wait()

        engine.engine.executor.sleep = slow_worker_sleep
        task = asyncio.create_task(engine.sleep(1))  # level 1 keeps valid content
        await asyncio.wait_for(started.wait(), 5)
        with pytest.raises((RuntimeError, LifecycleValidationError)):
            engine.wakeup()
        release.set()
        await task
        engine.wakeup(['weights'])  # valid: level-1 offload preserved content
        assert engine.engine.weight_state is WeightState.READY
    asyncio.run(scenario())


def test_failed_sleep_poisons_the_engine():
    engine, _ = make_endpoint(tags=())
    engine.engine.executor.sleep = AsyncMock(side_effect=RuntimeError('worker disappeared'))
    with pytest.raises(RuntimeError, match='worker disappeared'):
        asyncio.run(engine.sleep(2))
    assert engine.engine.weight_state is WeightState.POISONED
    with pytest.raises(RuntimeError, match='poisoned'):
        engine.wakeup()


def test_partial_wakeup_failure_keeps_the_engine_sleeping():
    engine, _ = make_endpoint()
    engine.engine.executor.wakeup.side_effect = RuntimeError('rank wake failed')
    with pytest.raises(RuntimeError, match='rank wake failed'):
        engine.wakeup(['weights'])
    assert engine.is_sleeping  # serve tags only drop on successful wakeup


def test_new_engine_is_the_recovery_boundary():
    failed, endpoint = make_endpoint(success=False)
    asyncio.run(endpoint(bucket()))
    fresh, _ = make_endpoint(tags=())
    assert fresh.engine.weight_state is WeightState.READY
    assert failed.engine.weight_state is WeightState.POISONED


def test_engine_state_machine_transaction_identity_and_replay():
    py = make_endpoint(tags=())[0].engine
    # An implicit transaction opens; an explicit bucket cannot join it.
    py._execute_weights_update(lambda: None, final=False, txn_id=None)
    assert py.weight_state is WeightState.UPDATING
    with pytest.raises(LifecycleValidationError, match='another weight transaction is open'):
        py._execute_weights_update(lambda: None, final=False, txn_id='v1')
    # The same implicit bucket continues, then commits.
    py._execute_weights_update(lambda: None, final=True, txn_id=None)
    assert py.weight_state is WeightState.READY
    # An explicit id commits once and can never be reused.
    py._execute_weights_update(lambda: None, final=False, txn_id='v9')
    py._execute_weights_update(lambda: None, final=True, txn_id='v9')
    assert py.weight_state is WeightState.READY
    with pytest.raises(LifecycleValidationError, match='already'):
        py._execute_weights_update(lambda: None, final=True, txn_id='v9')
    # A failed tuple result poisons even without an exception.
    py._execute_weights_update(lambda: (False, 'rank 2 failed'), final=True, txn_id='v10')
    assert py.weight_state is WeightState.POISONED


def test_commit_clears_engine_weights_tag_and_reopens_admission():
    engine, endpoint = make_endpoint(tags=('weights', 'kv_cache'))
    py = engine.engine
    assert asyncio.run(endpoint(bucket(False))).status_code == 200
    assert asyncio.run(endpoint(bucket(True))).status_code == 200
    assert py.weight_state is WeightState.READY
    assert 'weights' not in py.sleeping_tags  # commit dropped the engine tag
    engine.wakeup(['kv_cache'])
    assert not py.sleeping_tags  # admission can reopen: no tags left


def test_group_management_rpcs_never_open_a_weight_transaction():
    engine, _ = make_endpoint()
    py = engine.engine
    asyncio.run(py.init_weights_update_group({'group': 'train'}))
    py.executor.init_weights_update_group.assert_called_once()
    assert py.weight_state is WeightState.READY  # not a transaction
    asyncio.run(py.destroy_weights_update_group({'group': 'train'}))
    assert py.weight_state is WeightState.READY
    # And an explicit-id bucket still works right after group init.
    engine2, endpoint2 = make_endpoint()
    assert asyncio.run(endpoint2(bucket(False, update_id='v1'))).status_code == 200


def test_discarded_weights_block_serving_until_reloaded():
    engine, _ = make_endpoint(tags=('weights', 'kv_cache'))
    py = engine.engine
    asyncio.run(py.sleep(2))  # level 2 discards weight content
    py.wakeup(['weights'])  # resource cycle is legal (upstream semantics)...
    assert 'weights' not in py.sleeping_tags
    assert not py.weights_ready  # ...but the content is gone: serving stays refused
    assert engine.serving_blocked()
    with pytest.raises(RuntimeError, match='not ready'):
        engine.wakeup(['kv_cache'])  # and KV wakeup cannot enable inference
    # A committed update reloads content and re-enables serving.
    assert asyncio.run(_distributed_endpoint(engine)(bucket(True))).status_code == 200
    assert py.weights_ready


def _distributed_endpoint(engine):
    router = APIRouter()
    register(router, SimpleNamespace(async_engine=engine))
    return next(r.endpoint for r in router.routes
                if r.path == '/update_weights_from_distributed')


def test_same_transaction_second_bucket_queues_instead_of_409():
    async def scenario():
        engine, endpoint = make_endpoint()
        entered, release = threading.Event(), threading.Event()
        engine.engine.executor.update_weights_from_distributed.side_effect = \
            lambda req: (entered.set(), release.wait(5), (True, 'done'))[2]
        first = asyncio.create_task(endpoint(bucket(False)))
        await asyncio.wait_for(asyncio.to_thread(entered.wait), 5)
        # While bucket 1 is still transferring, the same transaction's next
        # bucket must queue and join, not be refused at the reservation.
        second = asyncio.create_task(endpoint(bucket(False)))
        await asyncio.sleep(0.05)
        assert not second.done(), second.exception() if second.done() else ''
        release.set()
        assert (await first).status_code == 200
        assert (await second).status_code == 200
        assert engine.engine.executor.update_weights_from_distributed.call_count == 2
    asyncio.run(scenario())
