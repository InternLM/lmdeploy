"""Default update API compatibility through the real engine state machine."""
import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_distributed_weight_readiness import bucket, make_endpoint
from test_weight_update_failure_guard import _routes, legacy

from lmdeploy.pytorch.engine.engine import LifecycleValidationError
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.managers import SessionManager
from lmdeploy.serve.openai.protocol import UpdateWeightsFromIPCRequest


def test_default_legacy_update_keeps_null_response_and_finalization():
    engine, _ = make_endpoint(tags=())
    routes = _routes(engine)
    engine.engine.executor.update_params = Mock(return_value=None)
    assert asyncio.run(routes[('/update_weights', 'POST')](legacy(False))).status_code == 200
    body = asyncio.run(routes[('/update_weights', 'POST')](legacy(False)))
    assert body.status_code == 200 and json.loads(body.body) is None
    assert engine.engine.weight_state.value == 'updating'  # transaction open
    assert asyncio.run(routes[('/update_weights', 'POST')](legacy(True))).status_code == 200
    assert engine.engine.weight_state.value == 'ready'
    assert engine.weights_content_ready
    # A new legacy transaction can start again after commit.
    assert asyncio.run(routes[('/update_weights', 'POST')](legacy(True))).status_code == 200


def test_legacy_update_refuses_active_generation_before_worker():
    engine, _ = make_endpoint(tags=(), dispatched=1)
    routes = _routes(engine)
    engine.engine.executor.update_params = Mock()
    assert asyncio.run(routes[('/update_weights', 'POST')](legacy())).status_code == 409
    engine.engine.executor.update_params.assert_not_called()
    assert engine.engine.weight_state.value == 'ready'


def test_default_ipc_readiness_and_update():
    engine, _ = make_endpoint(tags=('kv_cache',))
    routes = _routes(engine)
    status = asyncio.run(routes[('/update_weights_from_ipc', 'GET')]())
    assert status.ready and status.device_uuids == ['GPU-test'] and status.worker_ranks == [0]
    result = asyncio.run(routes[('/update_weights_from_ipc', 'POST')](
        UpdateWeightsFromIPCRequest(zmq_handles={'GPU-test': 'handle'})))
    assert result.status_code == 200 and json.loads(result.body)['success']
    assert engine.engine.weight_state.value == 'ready'  # ipc is single-shot final
    engine.wakeup(['kv_cache'])
    assert not engine.is_sleeping


def test_implicit_and_explicit_transactions_cannot_mix():
    engine, endpoint = make_endpoint()
    first = bucket(False, update_id='v1')
    assert asyncio.run(endpoint(first)).status_code == 200
    assert asyncio.run(endpoint(bucket(False, update_id=None))).status_code == 409
    engine.engine.executor.update_weights_from_distributed.assert_called_once()
    assert asyncio.run(endpoint(bucket(True, update_id='v1'))).status_code == 200
    assert engine.engine.weight_state.value == 'ready'


def test_timeout_leaves_the_write_running_and_state_self_resolves():
    async def scenario(monkeypatch):
        monkeypatch.setenv('LMDEPLOY_WEIGHT_UPDATE_TIMEOUT_SECONDS', '0.05')
        engine, endpoint = make_endpoint()
        entered, release = threading.Event(), threading.Event()
        engine.engine.executor.update_weights_from_distributed.side_effect = \
            lambda req: (entered.set(), release.wait(5), (True, 'done'))[2]
        response = await endpoint(bucket())
        assert response.status_code == 504
        assert 'still running' in json.loads(response.body)['message']
        assert engine.engine.weight_state.value == 'updating'  # in flight
        release.set()
        await asyncio.sleep(0.1)
        assert engine.engine.weight_state.value == 'ready'  # resolved: committed
    asyncio.run(scenario(pytest.MonkeyPatch()))


def test_invalid_timeout_is_refused_before_any_write(monkeypatch):
    engine, endpoint = make_endpoint()
    for value in ['nan', '0', 'not-a-number']:
        monkeypatch.setenv('LMDEPLOY_WEIGHT_UPDATE_TIMEOUT_SECONDS', value)
        assert asyncio.run(endpoint(bucket())).status_code == 409
        engine.engine.executor.update_weights_from_distributed.assert_not_called()
        assert engine.engine.weight_state.value == 'ready'


def test_no_argument_completion_finalizes_open_legacy_transaction():
    engine, _ = make_endpoint(tags=())
    routes = _routes(engine)
    engine.engine.executor.update_params = Mock(return_value=None)
    assert asyncio.run(routes[('/update_weights', 'POST')](legacy(False))).status_code == 200
    engine.complete_weights_update()  # legacy no-argument form still works
    assert engine.engine.weight_state.value == 'ready'
    assert engine.weights_content_ready


def test_no_argument_completion_refused_while_transfer_in_flight():
    async def scenario():
        engine, _ = make_endpoint(tags=())
        routes = _routes(engine)
        entered, release = threading.Event(), threading.Event()
        engine.engine.executor.update_params = Mock(
            side_effect=lambda req: (entered.set(), release.wait(5), None)[2])
        task = asyncio.create_task(routes[('/update_weights', 'POST')](legacy(False)))
        await asyncio.wait_for(asyncio.to_thread(entered.wait), 5)
        with pytest.raises(LifecycleValidationError, match='in progress'):
            engine.complete_weights_update()
        release.set()
        assert (await task).status_code == 200
        engine.complete_weights_update()  # valid after the transfer ended
    asyncio.run(scenario())


def test_ipc_and_distributed_serialize_through_one_state_machine():
    async def scenario():
        engine, endpoint = make_endpoint(tags=('kv_cache',))
        routes = _routes(engine)
        entered, release = threading.Event(), threading.Event()
        engine.engine.executor.update_weights_from_ipc = Mock(
            side_effect=lambda req, reason: (entered.set(), release.wait(5), (True, 'done'))[2])
        ipc = asyncio.create_task(routes[('/update_weights_from_ipc', 'POST')](
            UpdateWeightsFromIPCRequest(zmq_handles={})))
        await asyncio.wait_for(asyncio.to_thread(entered.wait), 5)
        # While the ipc write is in flight the engine is mid-transaction.
        assert engine.engine.weight_state.value == 'updating'
        release.set()
        assert (await ipc).status_code == 200
        assert engine.engine.weight_state.value == 'ready'
        assert (await endpoint(bucket())).status_code == 200  # next txn may start
    asyncio.run(scenario())


def test_late_inference_rechecks_readiness_and_releases_acquired_handle():
    async def scenario():
        engine, _ = make_endpoint(tags=())
        routes = _routes(engine)
        engine.engine.executor.update_params = Mock(return_value=None)
        assert (await routes[('/update_weights', 'POST')](legacy(False))).status_code == 200
        manager = SessionManager()
        handle = SimpleNamespace(async_stream_infer=Mock(), async_end=AsyncMock(),
                                  async_cancel=AsyncMock())
        manager.build_request_handle_pool(SimpleNamespace(create_instance=lambda: handle), 1)
        engine.session_mgr = manager
        session = manager.get(71)
        session._remove_on_request_exit = True
        with pytest.raises(RequestError) as error:
            async with session.request_handle() as acquired:
                async with engine.safe_run(acquired, session):
                    pytest.fail('an open transaction must not start inference')
        assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE
        handle.async_stream_infer.assert_not_called()
        assert manager.request_handle_pool.num_dispatched == 0
        assert not manager.sessions
    asyncio.run(scenario())


def test_timeout_resolution_to_poisoned_when_background_write_fails():
    async def scenario():
        import os
        os.environ['LMDEPLOY_WEIGHT_UPDATE_TIMEOUT_SECONDS'] = '0.05'
        engine, endpoint = make_endpoint()
        entered, release = threading.Event(), threading.Event()

        def slow_failing_write(req):
            entered.set()
            release.wait(5)
            raise RuntimeError('rank 3 died mid-write')

        engine.engine.executor.update_weights_from_distributed.side_effect = slow_failing_write
        response = await endpoint(bucket())
        assert response.status_code == 504  # in flight, outcome not yet known
        release.set()
        await asyncio.sleep(0.1)
        assert engine.engine.weight_state.value == 'poisoned'
        os.environ.pop('LMDEPLOY_WEIGHT_UPDATE_TIMEOUT_SECONDS', None)
    asyncio.run(scenario())


def test_turbomind_backend_keeps_upstream_update_semantics():
    """Backends without the weight-state machine keep their existing flow."""
    engine, endpoint = make_endpoint(tags=())
    engine.backend = 'turbomind'
    engine.engine = SimpleNamespace(  # a turbomind engine has no weight_state
        update_params=Mock(return_value=None))
    routes = _routes(engine)
    response = asyncio.run(routes[('/update_weights', 'POST')](legacy(True)))
    assert response.status_code == 200 and json.loads(response.body) is None
    engine.engine.update_params.assert_called_once()
    assert engine.weight_state_visible() == 'ready'


def test_admission_race_handle_acquired_after_check_is_caught():
    async def scenario():
        engine, _ = make_endpoint(tags=())
        routes = _routes(engine)
        reads = {'n': 0}

        class Pool:
            @property
            def num_dispatched(self):
                reads['n'] += 1
                return 0 if reads['n'] == 1 else 1  # a handle appears right after the check

        engine.session_mgr.request_handle_pool = Pool()
        engine.engine.executor.update_params = Mock(return_value=None)
        response = await routes[('/update_weights', 'POST')](legacy(True))
        assert response.status_code == 409
        engine.engine.executor.update_params.assert_not_called()
        assert engine.engine.weight_state.value == 'ready'  # reservation abandoned
    asyncio.run(scenario())


def test_timeout_committed_write_allows_kv_wakeup_without_serve_mirror():
    async def scenario():
        import os
        os.environ['LMDEPLOY_WEIGHT_UPDATE_TIMEOUT_SECONDS'] = '0.05'
        engine, endpoint = make_endpoint(tags=('weights', 'kv_cache'))
        entered, release = threading.Event(), threading.Event()

        def slow_write(req):
            entered.set()
            release.wait(5)
            return True, 'done'

        engine.engine.executor.update_weights_from_distributed.side_effect = slow_write
        response = await endpoint(bucket())
        assert response.status_code == 504
        release.set()
        await asyncio.sleep(0.1)
        assert engine.engine.weight_state.value == 'ready'
        # Engine truth alone (no serve mirror call ever ran) must allow kv wakeup.
        engine.wakeup(['kv_cache'])
        assert not engine.serving_blocked()
        os.environ.pop('LMDEPLOY_WEIGHT_UPDATE_TIMEOUT_SECONDS', None)
    asyncio.run(scenario())
