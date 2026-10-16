# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.engine.engine import Engine
from lmdeploy.pytorch.engine.engine_instance import EngineInstance
from lmdeploy.pytorch.engine.executor.mp_executor import MPExecutor
from lmdeploy.pytorch.engine.request import RequestManager, RequestType, Response
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.paging import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


class _FakeSequence:

    def __init__(self, resp):
        self.resp = resp


class _FakeSession:

    def __init__(self, session_id, seq):
        self.session_id = session_id
        self.sequences = {0: seq}


class _FakeScheduler:

    def __init__(self, session, events):
        self.sessions = {1: session}
        self.ended_sessions = []
        self.events = events

    def end_session(self, session_id):
        self.ended_sessions.append(session_id)
        self.sessions.pop(session_id)

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def get_sessions(self):
        return list(self.sessions.values())

    def reset_cache_after_worker_drain(self):
        assert self.events[-1] == 'sleep'
        self.events.append('reset_cache')


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
    session = _FakeSession(1, seq)
    engine.events = []
    engine.scheduler = _FakeScheduler(session, engine.events)
    engine._sleeping_tags = set()
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
    assert engine.executor.sleep_calls == [1]
    assert engine.events == ['drain', 'sleep', 'reset_cache', 'reset_engine_loop']


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


@pytest.mark.parametrize('level', [1, 2])
@pytest.mark.parametrize('staged', [False, True])
@pytest.mark.parametrize('state_step', [None, 8, 32, 35])
@pytest.mark.parametrize('prefix_caching', [False, True])
def test_sleep_invalidates_completed_prefix_cache(event_loop, level, staged, state_step, prefix_caching):
    cache = CacheConfig(max_batches=4,
                        block_size=16,
                        num_cpu_blocks=4,
                        num_gpu_blocks=16,
                        enable_prefix_caching=prefix_caching,
                        num_state_caches=13,
                        prefix_cache_state_budget=8,
                        states_shapes=[] if state_step is None else [((1, ), torch.float32)])
    scheduler = Scheduler(SchedulerConfig(max_batches=4, max_session_len=128, max_request_output_len=64),
                          cache, seq_meta=SequenceMeta(16, strategy=ARSequenceStrategy()))
    engine, _ = _build_sleeping_test_engine(event_loop)
    engine.scheduler = scheduler
    step = 32 if state_step is None else state_step
    tokens = [1] * step
    for _ in range(3):
        seq = scheduler.add_session(0).add_sequence(tokens)
        scheduler.block_manager.allocate(seq)
        scheduler.block_trie.allocate(seq)
        if state_step is not None and prefix_caching:
            assert scheduler.state_checkpoints.reserve_save(seq) >= 0
            assert scheduler.state_checkpoints.publish_save(seq)
        scheduler.end_session(0)
        repeat = scheduler.add_session(1).add_sequence(tokens + [2])
        scheduler.block_trie.match(repeat)
        assert repeat.num_history_ids == (step if prefix_caching else 0)
        # Leave a matched request alive, including its restore pin, for sleep
        # to cancel in addition to the already completed producer.
        scheduler.state_checkpoints.pin_restore(repeat)
        event_loop.run_until_complete(engine.sleep(level))
        assert scheduler.block_manager.get_num_free_gpu_blocks() == cache.num_gpu_blocks
        assert scheduler.state_manager.get_num_free() == cache.num_state_caches - 1
        if staged:
            engine.wakeup(['weights'])
            assert engine.req_manager.is_request_blocked(RequestType.ADD_MESSAGE)
            engine.wakeup(['kv_cache'])
        else:
            engine.wakeup()
        assert not engine.req_manager.is_request_blocked(RequestType.ADD_MESSAGE)
        new = scheduler.add_session(2).add_sequence(tokens + [2])
        scheduler.block_trie.match(new)
        assert new.num_history_ids == 0
        assert new.cached_tokens == 0
        assert not new.prefix_cache.restore.is_selected
        scheduler.end_session(2)


def test_complete_weights_update_keeps_kv_cache_sleeping(event_loop):
    engine, _ = _build_sleeping_test_engine(event_loop)
    engine._sleeping_tags = {'weights', 'kv_cache'}

    engine.complete_weights_update()

    assert engine._sleeping_tags == {'kv_cache'}


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
