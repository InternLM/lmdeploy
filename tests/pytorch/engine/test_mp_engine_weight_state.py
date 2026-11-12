"""The MP-engine proxy mirrors worker weight states through one tracker."""
import asyncio
from typing import Any

import pytest

from lmdeploy.pytorch.engine.engine import LifecycleValidationError, WeightState
from lmdeploy.pytorch.engine.mp_engine.base import MPEngine


class _StubMPEngine(MPEngine):
    """MPEngine whose collectives are scripted; no worker processes."""

    def __init__(self, outcomes=None):
        self.__init_calls__ = True
        # Bypass MPEngine.__init__ (which asks the workers for a config);
        # set exactly what the mirror needs.
        self._weight_txn = __import__('lmdeploy.pytorch.engine.engine',
                                      fromlist=['WeightTransactionTracker']).WeightTransactionTracker()
        import threading
        self._weight_lock = threading.Lock()
        self._mp_sleeping_tags = set()
        self.calls = []
        self.outcomes = outcomes or []

    def _collective_rpc(self, func, *args, **kwargs):
        self.calls.append(func)
        return None

    async def _collective_rpc_async(self, func, *args, **kwargs):
        self.calls.append(func)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return (True, 'done')

    async def _collective_rpc_streaming_async(self, func, init_done, *args, **kwargs):
        raise NotImplementedError


def _bucket(update_id=None, finished=False):
    from lmdeploy.serve.openai.protocol import UpdateWeightsFromDistributedRequest
    return UpdateWeightsFromDistributedRequest(
        names=['w'], dtypes=['bfloat16'], shapes=[[1]], group_name='g',
        update_id=update_id, finished=finished)


def test_proxy_mirror_commits_and_feeds_serve_gates():
    proxy = _StubMPEngine()
    assert proxy.weight_state is WeightState.READY and proxy.weights_ready
    assert asyncio.run(proxy.update_weights_from_distributed(_bucket('v1')))[0]
    assert proxy.weight_state is WeightState.UPDATING and not proxy.weights_ready
    assert asyncio.run(proxy.update_weights_from_distributed(
        _bucket('v1', finished=True)))[0]
    assert proxy.weight_state is WeightState.READY and proxy.weights_ready


def test_proxy_mirror_poisons_on_worker_failure():
    proxy = _StubMPEngine(outcomes=[(False, 'rank 1 died')])
    result = asyncio.run(proxy.update_weights_from_distributed(_bucket('v1')))
    assert result == (False, 'rank 1 died')
    assert proxy.weight_state is WeightState.POISONED
    with pytest.raises(LifecycleValidationError, match='rebuild'):
        asyncio.run(proxy.update_weights_from_distributed(_bucket('v2')))


def test_proxy_refuses_identity_mismatch_without_dispatch():
    proxy = _StubMPEngine()
    assert asyncio.run(proxy.update_weights_from_distributed(_bucket('v1')))[0]
    with pytest.raises(LifecycleValidationError, match='another weight transaction'):
        asyncio.run(proxy.update_weights_from_distributed(_bucket('v2')))
    assert proxy.calls.count('update_weights_from_distributed') == 1  # refused locally
    assert proxy.weight_state is WeightState.UPDATING  # not poisoned by the refusal


def test_proxy_mirror_replay_protection_and_discard():
    proxy = _StubMPEngine()
    asyncio.run(proxy.update_weights_from_distributed(_bucket('v9', finished=True)))
    with pytest.raises(LifecycleValidationError, match='already'):
        asyncio.run(proxy.update_weights_from_distributed(_bucket('v9', finished=True)))
    asyncio.run(proxy.sleep(2))
    assert not proxy.weights_ready  # level-2 discarded content
    assert proxy.sleeping_tags == {'weights', 'kv_cache'}
