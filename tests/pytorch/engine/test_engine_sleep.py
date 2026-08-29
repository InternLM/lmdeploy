# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace

import pytest

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.pytorch.engine.engine import Engine
from lmdeploy.pytorch.engine.engine_instance import EngineInstance
from lmdeploy.pytorch.engine.executor.mp_executor import MPExecutor
from lmdeploy.pytorch.engine.request import RequestManager, RequestType, Response
from lmdeploy.pytorch.exceptions import WeightUpdateError


class _FakeSequence:

    def __init__(self, resp):
        self.resp = resp


class _FakeSession:

    def __init__(self, seq):
        self.sequences = {0: seq}


class _FakeScheduler:

    def __init__(self, session):
        self.sessions = {1: session}
        self.ended_sessions = []
        self.clear_prefix_cache_calls = 0

    def end_session(self, session_id):
        self.ended_sessions.append(session_id)
        self.sessions.pop(session_id)

    def finish_deferred_kv_transfers_after_worker_drain(self):
        pass

    def clear_prefix_cache(self):
        self.clear_prefix_cache_calls += 1


class _FakeEngineLoop:

    def __init__(self, engine):
        self.engine = engine
        self.drained = False
        self.resumed = False

    async def drain_for_sleep(self):
        assert self.engine.req_manager.is_request_blocked(RequestType.ADD_SESSION)
        assert self.engine.req_manager.is_request_blocked(RequestType.ADD_MESSAGE)
        self.engine.events.append('drain')
        self.drained = True

    def resume_from_sleep(self):
        self.engine.events.append('resume')
        self.resumed = True

    def reset_runtime_state(self):
        self.engine.events.append('reset_engine_loop')


class _FakeExecutor:

    def __init__(self, engine):
        self.engine = engine
        self.sleep_calls = []
        self.wakeup_calls = []

    async def sleep(self, level=1):
        assert self.engine.req_manager.is_request_blocked(RequestType.ADD_SESSION)
        assert self.engine.scheduler.sessions == {}
        self.sleep_calls.append(level)
        self.engine.events.append('sleep')

    def wakeup(self, tags=None):
        self.wakeup_calls.append(tags)
        self.engine.events.append(('wakeup', tags))


@pytest.fixture
def event_loop():
    try:
        old_loop = asyncio.get_event_loop()
    except RuntimeError:
        old_loop = None
    new_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(new_loop)
        yield new_loop
    finally:
        pending = asyncio.all_tasks(new_loop)
        for task in pending:
            task.cancel()
        if pending:
            new_loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True))
        new_loop.run_until_complete(new_loop.shutdown_asyncgens())
        new_loop.stop()
        new_loop.close()
        asyncio.set_event_loop(old_loop)


def _build_sleeping_test_engine(event_loop):
    engine = Engine.__new__(Engine)
    engine.req_manager = RequestManager()
    resp = Response(type=ResponseType.INTERNAL_ENGINE_ERROR, sender_id=0, event=asyncio.Event())
    seq = _FakeSequence(resp)
    session = _FakeSession(seq)
    engine.scheduler = _FakeScheduler(session)
    engine._sleeping_tags = set()
    engine.events = []
    engine._engine_loop = _FakeEngineLoop(engine)
    engine.executor = _FakeExecutor(engine)
    return engine, resp


def test_engine_sleep_blocks_inputs_cancels_sessions_then_sleeps(event_loop):
    engine, resp = _build_sleeping_test_engine(event_loop)

    event_loop.run_until_complete(engine.sleep(level=1))

    assert engine.req_manager.is_request_blocked(RequestType.ADD_SESSION)
    assert engine.req_manager.is_request_blocked(RequestType.ADD_MESSAGE)
    assert engine._engine_loop.drained
    assert resp.type == ResponseType.CANCEL
    assert resp.is_done
    assert resp.event.is_set()
    assert engine.scheduler.ended_sessions == [1]
    assert engine.scheduler.clear_prefix_cache_calls == 1
    assert engine.executor.sleep_calls == [1]
    assert engine.events == ['drain', 'sleep', 'reset_engine_loop']


def test_engine_wakeup_reenables_inputs_only_after_all_tags(event_loop):
    engine, _ = _build_sleeping_test_engine(event_loop)
    engine.req_manager.block_request_types({RequestType.ADD_SESSION, RequestType.ADD_MESSAGE})
    engine._sleeping_tags = {'weights', 'kv_cache'}

    engine.wakeup(['weights'])

    assert engine.req_manager.is_request_blocked(RequestType.ADD_SESSION)
    assert engine.req_manager.is_request_blocked(RequestType.ADD_MESSAGE)
    assert not engine._engine_loop.resumed
    assert engine.events == [('wakeup', ['weights'])]

    engine.wakeup(['kv_cache'])

    assert not engine.req_manager.is_request_blocked(RequestType.ADD_SESSION)
    assert not engine.req_manager.is_request_blocked(RequestType.ADD_MESSAGE)
    assert engine._engine_loop.resumed
    assert engine.executor.wakeup_calls == [['weights'], ['kv_cache']]
    assert engine.events == [
        ('wakeup', ['weights']),
        ('wakeup', ['kv_cache']),
        'resume',
    ]


def test_engine_wakeup_all_delegates_warmup_to_executor_wakeup(event_loop):
    engine, _ = _build_sleeping_test_engine(event_loop)
    engine.req_manager.block_request_types({RequestType.ADD_SESSION, RequestType.ADD_MESSAGE})
    engine._sleeping_tags = {'weights', 'kv_cache'}

    engine.wakeup()

    assert not engine.req_manager.is_request_blocked(RequestType.ADD_SESSION)
    assert not engine.req_manager.is_request_blocked(RequestType.ADD_MESSAGE)
    assert engine._engine_loop.resumed
    assert engine.executor.wakeup_calls == [None]
    assert engine.events == [('wakeup', None), 'resume']


def test_mp_executor_wakeup_waits_for_kv_cache():
    executor = MPExecutor.__new__(MPExecutor)
    calls = []

    def _collective_rpc(method, args=None, return_mask=0xff):
        calls.append((method, args, return_mask))

    executor.collective_rpc = _collective_rpc

    executor.wakeup(['weights'])
    executor.wakeup(['kv_cache'])
    executor.wakeup()

    assert calls == [
        ('wakeup', (['weights'], ), 0),
        ('wakeup', (['kv_cache'], ), 0xff),
        ('wakeup', (None, ), 0xff),
    ]


def test_mp_executor_update_params_forwards_to_workers():
    executor = MPExecutor.__new__(MPExecutor)
    calls = []

    def _collective_rpc(method, args=None, return_mask=0xff):
        calls.append((method, args, return_mask))

    executor.collective_rpc = _collective_rpc
    request = object()

    executor.update_params(request)

    assert calls == [('update_params', (request, ), 0xff)]


def test_engine_rejects_weight_update_without_offloaded_kv(event_loop):
    engine = Engine.__new__(Engine)
    engine._sleeping_tags = set()
    engine.misc_config = SimpleNamespace(empty_init=False)
    engine.scheduler = SimpleNamespace(sessions={}, block_trie=SimpleNamespace(leaves=set()),
                                       has_unfinished=lambda: False)
    engine.executor = SimpleNamespace(update_params=lambda _: pytest.fail('update should be rejected'))

    with pytest.raises(WeightUpdateError, match='KV cache to be offloaded'):
        engine.update_params(object())


def test_engine_allows_weight_update_after_cache_offload(event_loop):
    calls = []
    engine = Engine.__new__(Engine)
    engine._sleeping_tags = {'kv_cache'}
    engine.misc_config = SimpleNamespace(empty_init=False)
    engine.scheduler = SimpleNamespace(sessions={}, block_trie=SimpleNamespace(leaves=set()),
                                       has_unfinished=lambda: False)
    engine.executor = SimpleNamespace(update_params=lambda request: calls.append(request))
    request = object()

    engine.update_params(request)

    assert calls == [request]


def test_engine_rejects_weight_update_with_cached_prefix(event_loop):
    engine = Engine.__new__(Engine)
    engine._sleeping_tags = {'kv_cache'}
    engine.misc_config = SimpleNamespace(empty_init=False)
    engine.scheduler = SimpleNamespace(sessions={}, block_trie=SimpleNamespace(leaves={object()}),
                                       has_unfinished=lambda: False)
    engine.executor = SimpleNamespace(update_params=lambda _: pytest.fail('update should be rejected'))

    with pytest.raises(WeightUpdateError, match='empty prefix cache'):
        engine.update_params(object())


def test_engine_distributed_weight_update_rejects_unsafe_state(event_loop):
    engine = Engine.__new__(Engine)
    engine._sleeping_tags = set()
    engine.misc_config = SimpleNamespace(empty_init=False)
    engine.scheduler = SimpleNamespace(sessions={}, block_trie=SimpleNamespace(leaves=set()),
                                       has_unfinished=lambda: False)
    engine._weights_update_lock = None
    engine.executor = SimpleNamespace(update_weights_from_distributed=lambda _: pytest.fail(
        'update should be rejected'))
    request = object()

    success, message = event_loop.run_until_complete(engine.update_weights_from_distributed(request))

    assert not success
    assert 'KV cache to be offloaded' in message


def test_engine_instance_new_request_after_sleep_returns_cancel(event_loop):
    engine = SimpleNamespace(
        req_manager=RequestManager(),
        max_session_len=8,
        engine_config=SimpleNamespace(enable_transfer_obj_ref=False, distributed_executor_backend='uni'),
    )
    engine.req_manager.block_request_types({RequestType.ADD_SESSION, RequestType.ADD_MESSAGE})
    inst = EngineInstance(engine)

    async def __collect():
        outputs: list[EngineOutput] = []
        async for out in inst.async_stream_infer(1, [1, 2, 3]):
            outputs.append(out)
        return outputs

    outputs = event_loop.run_until_complete(__collect())

    assert len(outputs) == 1
    assert outputs[0].status == ResponseType.CANCEL
    assert engine.req_manager._loop_task is None
