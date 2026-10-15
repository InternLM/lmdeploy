"""Distributed weight completion must unblock empty-init inference after KV
wakeup."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import APIRouter

from lmdeploy.serve.core.async_engine import AsyncEngine
from lmdeploy.serve.openai.endpoints.management import register
from lmdeploy.serve.openai.protocol import UpdateWeightsFromDistributedRequest


def make_endpoint(*, success=True, tags=('weights', 'kv_cache')):
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.backend = 'pytorch'
    engine.sleeping_tags = set(tags)
    engine.is_sleeping = bool(tags)
    engine.engine = SimpleNamespace(
        update_weights_from_distributed=AsyncMock(return_value=(success, 'done')),
        complete_weights_update=Mock(),
        wakeup=Mock(),
    )
    router = APIRouter()
    register(router, SimpleNamespace(async_engine=engine))
    endpoint = next(route.endpoint for route in router.routes if route.path == '/update_weights_from_distributed')
    return engine, endpoint


@pytest.mark.parametrize('finished,success', [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize('tags', [('weights', 'kv_cache'), ('kv_cache',)])
def test_only_successful_final_bucket_completes_weights(finished, success, tags):
    engine, endpoint = make_endpoint(success=success, tags=tags)
    request = UpdateWeightsFromDistributedRequest(names=[], dtypes=[], shapes=[], group_name='train', finished=finished)
    response = asyncio.run(endpoint(request))
    assert response.status_code == (200 if success else 400)
    engine.engine.update_weights_from_distributed.assert_awaited_once_with(request)
    if success and finished:
        engine.engine.complete_weights_update.assert_called_once_with()
        assert engine.sleeping_tags == {'kv_cache'}
    else:
        engine.engine.complete_weights_update.assert_not_called()
        assert engine.sleeping_tags == set(tags)
    assert engine.is_sleeping  # Finalization never runs inference/warmup on its own.
    engine.engine.wakeup.assert_not_called()


def test_empty_init_ready_after_finalization_and_kv_wakeup():
    engine, endpoint = make_endpoint()
    request = UpdateWeightsFromDistributedRequest(names=[], dtypes=[], shapes=[], group_name='train', finished=True)
    asyncio.run(endpoint(request))
    engine.wakeup(['kv_cache'])
    assert not engine.is_sleeping
    assert not engine.sleeping_tags
    engine.engine.wakeup.assert_called_once_with(['kv_cache'])


def test_transfer_exception_does_not_complete_weights():
    engine, endpoint = make_endpoint()
    engine.engine.update_weights_from_distributed.side_effect = RuntimeError('transfer failed')
    request = UpdateWeightsFromDistributedRequest(names=[], dtypes=[], shapes=[], group_name='train', finished=True)
    with pytest.raises(RuntimeError, match='transfer failed'):
        asyncio.run(endpoint(request))
    engine.engine.complete_weights_update.assert_not_called()
    assert engine.sleeping_tags == {'weights', 'kv_cache'}
    assert engine.is_sleeping
