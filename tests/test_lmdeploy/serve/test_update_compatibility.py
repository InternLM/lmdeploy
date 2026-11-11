"""Default update API compatibility and opt-in deployment constraints (CPU)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from test_distributed_weight_readiness import make_endpoint
from test_weight_update_failure_guard import bucket, route

from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.managers import SessionManager
from lmdeploy.serve.openai.protocol import UpdateParamsRequest, UpdateWeightsFromIPCRequest


def legacy(finished=True):
    return UpdateParamsRequest(serialized_named_tensors={}, finished=finished)


@pytest.mark.parametrize('backend', ['pytorch', 'turbomind'])
def test_default_legacy_update_keeps_null_response_and_finalization(backend):
    engine, _ = make_endpoint(tags=())
    engine.backend = backend
    engine.engine.update_params = Mock()
    endpoint = route(engine, '/update_weights')
    request = legacy(False)
    response = asyncio.run(endpoint(request))
    assert response.status_code == 200 and json.loads(response.body) is None
    engine.engine.update_params.assert_called_once_with(request)
    assert engine.weights_update_pending and engine.is_sleeping
    assert asyncio.run(endpoint(legacy())).status_code == 200
    assert not engine.is_sleeping and engine.weights_content_ready
    assert not engine.weights_update_pending
    assert engine.engine.complete_weights_update.call_count == (backend == 'pytorch')
    # A new legacy transaction has no permanent synthetic ID to collide with.
    assert asyncio.run(endpoint(legacy())).status_code == 200


def test_legacy_update_refuses_active_generation_before_worker():
    engine, _ = make_endpoint(tags=())
    engine.session_mgr.request_handle_pool.num_dispatched = 1
    engine.engine.update_params = Mock()
    assert asyncio.run(route(engine, '/update_weights')(legacy())).status_code == 409
    assert not engine.is_sleeping and engine.weights_content_ready
    engine.engine.update_params.assert_not_called()


def test_default_ipc_readiness_and_update():
    engine, _ = make_endpoint(tags=('kv_cache',))
    engine.engine.get_checkpoint_engine_status = AsyncMock(return_value=[dict(
        ready=True, message='ready', checkpoint_engine_version='test', device_uuid='GPU-test', rank=0)])
    engine.engine.update_weights_from_ipc = AsyncMock(return_value=(True, 'done'))
    status = asyncio.run(route(engine, '/update_weights_from_ipc', 'GET')())
    assert status.ready and status.device_uuids == ['GPU-test'] and status.worker_ranks == [0]
    request = UpdateWeightsFromIPCRequest(zmq_handles={'GPU-test': 'handle'})
    result = asyncio.run(route(engine, '/update_weights_from_ipc')(request))
    assert result.status_code == 200 and json.loads(result.body)['success']
    engine.engine.update_weights_from_ipc.assert_awaited_once_with(request, None)
    assert engine.weights_content_ready and engine.is_sleeping
    engine.wakeup(['kv_cache'])
    assert not engine.is_sleeping


@pytest.mark.parametrize('explicit_first', [False, True])
def test_implicit_and_explicit_distributed_transactions_cannot_mix(explicit_first):
    engine, endpoint = make_endpoint()
    first = bucket(False).model_copy(update={'update_id': 'v1' if explicit_first else None})
    assert asyncio.run(endpoint(first)).status_code == 200
    other = first.model_copy(update={'update_id': None if explicit_first else 'v1'})
    assert asyncio.run(endpoint(other)).status_code == 409
    assert engine.engine.update_weights_from_distributed.await_count == 1
    assert asyncio.run(endpoint(first.model_copy(update={'finished': True}))).status_code == 200


def test_legacy_distributed_empty_finalize_only_for_an_open_transaction():
    engine, endpoint = make_endpoint()
    first = bucket(False).model_copy(update={'update_id': None})
    final = first.model_copy(update={'names': [], 'dtypes': [], 'shapes': [], 'finished': True})
    assert asyncio.run(endpoint(final)).status_code == 409
    assert asyncio.run(endpoint(first)).status_code == 200
    assert asyncio.run(endpoint(final)).status_code == 200
    assert engine.weights_content_ready
    # Missing ID is a supported legacy schema, not silently rewritten per bucket.
    assert engine.engine.update_weights_from_distributed.call_args.args[0].update_id is None
    assert asyncio.run(endpoint(final)).status_code == 409


@pytest.mark.parametrize('transport', ['legacy', 'ipc'])
@pytest.mark.parametrize('error', [RuntimeError('partial write')])
def test_all_transports_latch_unknown_or_failed_writes(transport, error):
    engine, _ = make_endpoint(tags=('kv_cache',))
    engine.engine.update_params = Mock(side_effect=error)
    engine.engine.update_weights_from_ipc = AsyncMock(side_effect=error)
    path = '/update_weights' if transport == 'legacy' else '/update_weights_from_ipc'
    request = legacy() if transport == 'legacy' else UpdateWeightsFromIPCRequest(zmq_handles={})
    with pytest.raises(type(error)):
        asyncio.run(route(engine, path)(request))
    assert engine.weights_update_error and engine.is_sleeping and not engine.weights_content_ready
    with pytest.raises(RuntimeError, match='Restart'):
        engine.wakeup()


def test_ipc_and_distributed_cannot_overlap():
    async def scenario():
        engine, endpoint = make_endpoint(tags=('kv_cache',))
        entered, release = asyncio.Event(), asyncio.Event()

        async def receive(*args):
            entered.set()
            await release.wait()
            return True, 'done'

        engine.engine.update_weights_from_ipc = AsyncMock(side_effect=receive)
        task = asyncio.create_task(route(engine, '/update_weights_from_ipc')(
            UpdateWeightsFromIPCRequest(zmq_handles={})))
        await asyncio.wait_for(entered.wait(), 1)
        assert (await endpoint(bucket())).status_code == 409
        engine.engine.update_weights_from_distributed.assert_not_awaited()
        release.set()
        assert (await task).status_code == 200
    asyncio.run(scenario())


def test_restricted_server_requires_explicit_id(monkeypatch):
    monkeypatch.setenv('LMDEPLOY_REQUIRE_UPDATE_ID', '1')
    engine, endpoint = make_endpoint()
    assert asyncio.run(endpoint(bucket().model_copy(update={'update_id': None}))).status_code == 409
    engine.engine.update_weights_from_distributed.assert_not_awaited()
    assert asyncio.run(endpoint(bucket())).status_code == 200


@pytest.mark.parametrize('value', ['nan', '0', 'not-a-number'])
def test_invalid_timeout_is_refused_before_any_write(monkeypatch, value):
    monkeypatch.setenv('LMDEPLOY_WEIGHT_UPDATE_TIMEOUT_SECONDS', value)
    engine, endpoint = make_endpoint()
    assert asyncio.run(endpoint(bucket())).status_code == 409
    assert not engine.weights_update_pending and engine.weights_update_error is None
    engine.engine.update_weights_from_distributed.assert_not_awaited()


@pytest.mark.parametrize('name,value', [('LMDEPLOY_WEIGHT_UPDATE_PROTOCOLS', 'unknown'),
                                        ('LMDEPLOY_WEIGHT_UPDATE_PROTOCOLS', ''),
                                        ('LMDEPLOY_REQUIRE_UPDATE_ID', 'maybe')])
def test_bad_deployment_policy_fails_at_registration(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        make_endpoint()


@pytest.mark.parametrize('state', ['sleeping', 'partial', 'failed'])
def test_late_inference_rechecks_readiness_and_releases_acquired_handle(state):
    async def scenario():
        engine, _ = make_endpoint(tags=())
        manager = SessionManager()
        handle = SimpleNamespace(async_stream_infer=Mock(), async_end=AsyncMock(), async_cancel=AsyncMock())
        manager.build_request_handle_pool(SimpleNamespace(create_instance=lambda: handle), 1)
        engine.session_mgr = manager
        engine.engine.update_params = Mock()
        if state == 'sleeping':
            engine.is_sleeping = True
        else:
            # A request already admitted by HTTP can still be waiting before
            # it acquires a real handle. Open a partial update in that window.
            assert (await route(engine, '/update_weights')(legacy(False))).status_code == 200
            if state == 'failed':
                engine.fail_weights_update('unknown worker outcome')
        session = manager.get(71)
        session._remove_on_request_exit = True
        with pytest.raises(RequestError) as error:
            async with session.request_handle() as acquired:
                async with engine.safe_run(acquired, session):
                    pytest.fail('unready engine must not start inference')
        assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE
        handle.async_stream_infer.assert_not_called()
        assert manager.request_handle_pool.num_dispatched == 0
        assert not manager.sessions
    asyncio.run(scenario())
