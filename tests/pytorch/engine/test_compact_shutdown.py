# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from lmdeploy.pytorch.engine.engine import Engine
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from tests.pytorch.engine.test_compact_inputs_maker import _finish_store_maker, _store_maker
from tests.pytorch.kv_connector.test_compact_store_load import _start
from tests.pytorch.kv_connector.test_compact_store_load import setup_load as _setup_load


@pytest.fixture
def setup_load():
    yield from _setup_load.__wrapped__()


def _engine(monkeypatch, scheduler, *, forward_error=False, drain_error=False):
    from lmdeploy.pytorch.engine import engine_loop
    engine = Engine.__new__(Engine)
    engine.scheduler = scheduler
    engine.cache_config = scheduler.cache_config
    engine.req_manager = Mock()
    engine.end_session = scheduler.end_session
    engine._block_new_inputs = Mock()
    engine.executor = Mock()
    engine.executor.drain_for_shutdown = AsyncMock(side_effect=RuntimeError('drain failed') if drain_error else None)
    async def wait():
        if forward_error:
            raise ValueError('original forward failed')
    loop = SimpleNamespace(migration_event=None, start=Mock(), stop=Mock(), wait_tasks=wait)
    monkeypatch.setattr(engine_loop, 'build_engine_loop', lambda _: loop)
    return engine


@pytest.mark.parametrize('operation', ['load', 'save'])
def test_engine_shutdown_drains_before_releasing_compact_io(monkeypatch, setup_load, operation):
    maker = None
    if operation == 'load':
        scheduler, store = setup_load(width=4)
        _start(scheduler)
        scheduler.build_connector_meta([])
    else:
        maker, store = _store_maker(apc=True, width=4)
        scheduler = maker.scheduler
        scheduler.add_session(0).add_sequence([1] * 72)
        payload = maker._make_forward_inputs(True)
        scheduler.state_checkpoints.publish_forward(payload['cache_inputs'].ssm_state_access_id)
    engine = _engine(monkeypatch, scheduler)
    async def drain():
        engine._block_new_inputs.assert_called_once()
        # Controlled worker/I/O completion: no owner may retire it before this returns.
        assert store._compact_loads or store._inflight_saves
    engine.executor.drain_for_shutdown.side_effect = drain
    asyncio.run(engine.async_loop())
    engine.executor.release.assert_called_once()
    assert not store._compact_loads and not store._inflight_saves
    assert not scheduler.state_manager._accesses and scheduler.state_manager._store_refs == 0
    assert not scheduler.sessions
    scheduler.shutdown()  # Scheduler teardown remains idempotent.
    if maker is not None:
        _finish_store_maker(maker)


@pytest.mark.parametrize('forward_error', [False, True])
def test_failed_drain_keeps_io_holds_but_still_releases_executor(monkeypatch, setup_load, forward_error):
    scheduler, store = setup_load(width=4)
    _start(scheduler)
    scheduler.build_connector_meta([])
    engine = _engine(monkeypatch, scheduler, forward_error=forward_error, drain_error=True)
    with pytest.raises((ValueError if forward_error else RuntimeError),
                       match='original forward failed' if forward_error else 'drain failed'):
        asyncio.run(engine.async_loop())
    engine.executor.release.assert_called_once()
    assert store._compact_loads and scheduler.state_manager._accesses
    # Fixture alone provides the mock quiescence proof for cleanup.


def test_scheduler_failure_does_not_mask_original_error_or_skip_release():
    engine = Engine.__new__(Engine)
    engine.migration_event = None
    engine.scheduler = Mock()
    engine.scheduler.shutdown.side_effect = ValueError('scheduler failed')
    engine.executor = Mock()
    engine.executor.release.side_effect = RuntimeError('release failed')
    with pytest.raises(ValueError, match='scheduler failed'):
        engine._loop_finally()
    engine.executor.release.assert_called_once()


@pytest.mark.parametrize('dp', [1, 4])
def test_worker_drain_awaits_producer_cancellation_before_io_close(dp):
    async def run():
        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.dist_config = SimpleNamespace(dp=dp)
        events = []
        async def producer():
            try:
                await asyncio.Event().wait()
            finally:
                events.append('producer stopped')
        task = asyncio.create_task(producer())
        agent.tasks = {task}
        agent._postprocess_task = None
        await asyncio.sleep(0)
        agent.shutdown_kv_connector = lambda: events.append('io closed')
        await agent.drain_for_shutdown()
        assert task.cancelled() and events == ['producer stopped', 'io closed']
    asyncio.run(run())


def test_ray_drain_waits_for_all_rank_rpc_and_propagates_timeout():
    from lmdeploy.pytorch.config import CacheConfig
    from lmdeploy.pytorch.engine.executor.ray_executor import RayExecutor
    executor = RayExecutor.__new__(RayExecutor)
    executor.cache_config = CacheConfig(1, 16, 0, 16)
    executor.collective_rpc_async = AsyncMock(return_value=[None, None])
    asyncio.run(executor.drain_for_shutdown())
    executor.collective_rpc_async.assert_awaited_once_with('drain_for_shutdown')
    executor._get_worker_release_timeout = lambda _: 0.01
    async def blocked(*args):
        await asyncio.Event().wait()
    executor.collective_rpc_async = blocked
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(executor.drain_for_shutdown())


def test_uni_and_worker_wrapper_forward_shutdown_drain():
    from lmdeploy.pytorch.engine.executor.base_worker import WorkerWrapperBase
    from lmdeploy.pytorch.engine.executor.uni_executor import UniExecutor
    for cls in [UniExecutor, WorkerWrapperBase]:
        obj = cls.__new__(cls)
        obj.model_agent = SimpleNamespace(drain_for_shutdown=AsyncMock())
        asyncio.run(obj.drain_for_shutdown())
        obj.model_agent.drain_for_shutdown.assert_awaited_once_with()


def test_mp_drain_awaits_collective_completion():
    from lmdeploy.pytorch.engine.executor.mp_executor import MPExecutor
    executor = MPExecutor.__new__(MPExecutor)
    executor.collective_rpc_async = AsyncMock(return_value=[None, None])
    asyncio.run(executor.drain_for_shutdown())
    executor.collective_rpc_async.assert_awaited_once_with('drain_for_shutdown')


@pytest.mark.parametrize('child_fails', [False, True])
def test_shutdown_waits_for_real_shielded_spec_postprocess(monkeypatch, child_fails):
    from collections import deque
    from contextlib import nullcontext

    import torch

    from lmdeploy.pytorch.engine.model_agent import agent as module

    async def run():
        entered, resume = asyncio.Event(), asyncio.Event()
        events = []
        async def spec_forward(inputs, extra, sampling):
            entered.set()
            await resume.wait()
            events.append('spec finished')
            if child_fails:
                raise RuntimeError('spec failed')
            return extra
        async def model_forward(*args, **kwargs):
            return dict(logits=torch.zeros((1, 1, 8)))
        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.rank, agent.need_output = 1, False
        agent._postprocess_task = None
        agent.kv_connector = SimpleNamespace(shutdown=lambda: events.append('connector closed'))
        agent.cache_engine = agent.memdecode_agent = None
        agent._pending_h2d_transfers = deque()
        agent._pre_in_que, agent._in_que, agent._out_que = [asyncio.Queue() for _ in range(3)]
        agent._prepare_inputs_prefill = lambda inputs, delta: inputs
        agent._async_model_forward = model_forward
        agent._slice_outs = lambda logits, length: logits
        agent._broadcast_next_token = lambda *args, **kwargs: nullcontext()
        extra = SimpleNamespace(next_token_ids=torch.tensor([2]))
        agent.agent_strategy = SimpleNamespace(slice_extra_inputs=lambda *args: extra,
                                              make_dummy_next_token=lambda *args: (torch.tensor([2]), extra),
                                              make_extra_outputs=lambda _: None)
        agent.dist_ctx = SimpleNamespace()
        agent.spec_agent = SimpleNamespace(cache_engine=None, is_enabled=lambda: True,
                                           async_model_forward=spec_forward,
                                           post_broadcast=lambda *args: nullcontext())
        inputs = SimpleNamespace(is_dummy=False, is_decoding=True, ssm_state_meta=object(),
                                 seq_length=torch.tensor([1]), input_ids=torch.tensor([[1]]),
                                 is_chunk=False, is_first_chunk=False, is_last_chunk=False, dp_meta=None)
        cache = SimpleNamespace(ssm_state_access_id=1, kv_restore_plan=None, kv_save_plan=None)
        manager = SimpleNamespace(current_context=lambda: SimpleNamespace(
            dist_config=SimpleNamespace(attn_tp=2, dp=1)))
        monkeypatch.setattr(module, 'get_dist_manager', lambda: manager)
        monkeypatch.setattr(module, 'cache_swapping', lambda *args, **kwargs: None)
        monkeypatch.setattr(torch.cuda, 'synchronize', lambda: events.append('cuda sync'))
        producer = asyncio.create_task(agent._async_step(inputs, extra_inputs=extra, cache_inputs=cache))
        agent.tasks = {producer}
        await entered.wait()
        child = agent._postprocess_task
        drain = asyncio.create_task(agent.drain_for_shutdown())
        try:
            for _ in range(20):
                await asyncio.sleep(0)
            assert producer.cancelled()
            assert not child.done() and not drain.done()
            assert events == []  # No CUDA sync, connector close or successful drain yet.
            resume.set()
            if child_fails:
                with pytest.raises(RuntimeError, match='spec failed'):
                    await drain
                assert events == ['spec finished']
            else:
                await drain
                assert events == ['spec finished', 'cuda sync', 'connector closed']
        finally:
            resume.set()
            await asyncio.gather(producer, child, drain, return_exceptions=True)
    asyncio.run(run())


def test_cancelled_shutdown_does_not_cancel_shielded_producer():
    async def run():
        agent = BaseModelAgent.__new__(BaseModelAgent)
        entered, resume = asyncio.Event(), asyncio.Event()
        async def postprocess():
            entered.set()
            await resume.wait()
        producer = asyncio.create_task(agent._shield_postprocess(postprocess()))
        agent.tasks = {producer}
        await entered.wait()
        child = agent._postprocess_task
        agent.shutdown_kv_connector = Mock()
        drain = asyncio.create_task(agent.drain_for_shutdown())
        for _ in range(20):
            await asyncio.sleep(0)
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain
        assert not child.done()
        agent.shutdown_kv_connector.assert_not_called()
        resume.set()
        await agent.drain_for_shutdown()
        agent.shutdown_kv_connector.assert_called_once()
    asyncio.run(run())
