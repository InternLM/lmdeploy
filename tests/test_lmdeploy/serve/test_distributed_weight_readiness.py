"""Distributed weight transactions through the real engine state machine.

The fixture builds a real ``Engine`` shell (no ``__init__``) whose executor
is mocked, so every assertion below exercises the actual weight-state
machine plus the actual serve projection, not a stubbed lifecycle.
"""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import APIRouter

from lmdeploy.pytorch.engine.engine import (
    Engine, LifecycleValidationError, WeightState, WeightTransactionTracker)
from lmdeploy.serve.core.async_engine import AsyncEngine
from lmdeploy.serve.openai.endpoints.management import register
from lmdeploy.serve.openai.protocol import UpdateWeightsFromDistributedRequest


def make_endpoint(*, success=True, tags=('weights', 'kv_cache'), dispatched=0):
    py = Engine.__new__(Engine)
    py._model_state_lock = threading.Lock()
    py.__dict__['_weight_txn_store'] = WeightTransactionTracker(content_valid=True)
    py._weights_update_lock = None
    py._sleeping_tags = set(tags)
    py._engine_loop = None
    py.req_manager = SimpleNamespace(block_request_types=lambda *_: None,
                                     unblock_request_types=lambda *_: None)
    py._cancel_and_end_all_sessions = lambda: None
    py.scheduler = SimpleNamespace(reset_cache_after_worker_drain=lambda: None)
    py.executor = SimpleNamespace(
        update_params=Mock(return_value=None),
        init_weights_update_group=Mock(return_value=(True, 'group ready')),
        destroy_weights_update_group=Mock(return_value=(True, 'group gone')),
        update_weights_from_ipc=Mock(return_value=(success, 'done')),
        update_weights_from_distributed=Mock(return_value=(success, 'done')),
        get_checkpoint_engine_status=Mock(return_value=[dict(
            ready=True, message='ready', checkpoint_engine_version='test',
            device_uuid='GPU-test', rank=0)]),
        sleep=AsyncMock(),
        wakeup=Mock(),
    )
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.backend = 'pytorch'
    engine.backend_config = SimpleNamespace(device_type='cuda')
    engine.engine = py
    engine.session_mgr = SimpleNamespace(
        request_handle_pool=SimpleNamespace(num_dispatched=dispatched))
    engine.stop_all_session = AsyncMock()
    engine.sleeping_tags = set(tags)
    engine.is_sleeping = bool(tags)
    engine.weights_content_ready = True
    router = APIRouter()
    register(router, SimpleNamespace(async_engine=engine))
    endpoint = next(r.endpoint for r in router.routes if r.path == '/update_weights_from_distributed')
    return engine, endpoint


def bucket(finished=True, update_id='test-v1'):
    return UpdateWeightsFromDistributedRequest(
        names=['test.weight'], dtypes=['bfloat16'], shapes=[[1]],
        group_name='train', update_id=update_id, finished=finished)


@pytest.mark.parametrize('finished', [False, True])
@pytest.mark.parametrize('success', [False, True])
@pytest.mark.parametrize('tags', [('weights', 'kv_cache'), ('kv_cache',)])
def test_only_the_final_successful_bucket_makes_weights_ready(finished, success, tags):
    engine, endpoint = make_endpoint(success=success, tags=tags)
    assert asyncio.run(endpoint(bucket(finished))).status_code == (200 if success else 400)
    py = engine.engine
    py.executor.update_weights_from_distributed.assert_called_once()
    if not success:
        assert py.weight_state is WeightState.POISONED
        assert engine.weight_state_visible() == 'poisoned'
    elif finished:
        assert py.weight_state is WeightState.READY
        assert engine.weights_content_ready
        engine.wakeup(['kv_cache'])
        assert not engine.is_sleeping
    else:
        # Transaction still open: no wakeup, no inference, no sleep.
        assert py.weight_state is WeightState.UPDATING
        with pytest.raises(RuntimeError, match='weight state is updating'):
            engine.wakeup(['kv_cache'])
        with pytest.raises(LifecycleValidationError, match='transaction is open'):
            asyncio.run(py.sleep(1))


def test_empty_finalize_only_completes_an_open_implicit_transaction():
    engine, endpoint = make_endpoint()
    final = bucket(finished=True, update_id=None).model_copy(
        update={'names': [], 'dtypes': [], 'shapes': []})
    assert asyncio.run(endpoint(final)).status_code == 409  # nothing open
    assert asyncio.run(endpoint(bucket(False, update_id=None))).status_code == 200
    assert asyncio.run(endpoint(final)).status_code == 200
    assert engine.engine.weight_state is WeightState.READY
    py = engine.engine
    assert py.executor.update_weights_from_distributed.call_args.args[0].update_id is None
    assert asyncio.run(endpoint(final)).status_code == 409  # transaction already closed


def test_transfer_exception_poisons_instead_of_completing():
    engine, endpoint = make_endpoint()
    engine.engine.executor.update_weights_from_distributed.side_effect = RuntimeError('transfer failed')
    response = asyncio.run(endpoint(bucket()))
    assert response.status_code == 503
    assert engine.engine.weight_state is WeightState.POISONED
    with pytest.raises(RuntimeError, match='poisoned'):
        engine.wakeup()
