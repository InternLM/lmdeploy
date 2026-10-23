# Copyright (c) OpenMMLab. All rights reserved.
"""CPU producer quiescence is distinct from task cancellation and GPU drain."""
import asyncio
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs
from lmdeploy.pytorch.engine.engine import Engine
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent


def _agent(monkeypatch, *, output=True):
    from lmdeploy.pytorch.engine.model_agent import agent as module
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.rank = 0
    agent.need_output = output
    agent.compact_runtime = object()
    agent.kv_connector = agent.memdecode_agent = None
    agent.cache_engine = object()
    agent.state_cache_engine = object()
    agent._prepare_inputs_prefill = lambda inputs, delta: inputs
    agent._is_prefill_input_logprobs = lambda inputs: False
    agent.spec_agent = SimpleNamespace(cache_engine=None, is_enabled=lambda: False)
    agent.agent_strategy = SimpleNamespace(slice_extra_inputs=lambda extra, *_: extra)
    agent._slice_outs = lambda logits, lengths: logits
    agent.tasks = set()
    agent._pre_in_que = asyncio.Queue()
    monkeypatch.setattr(module, 'cache_swapping', lambda *a, **kw: None)
    monkeypatch.setattr(module, 'get_dist_manager', lambda: SimpleNamespace(current_context=lambda: SimpleNamespace(
        dist_config=SimpleNamespace(attn_tp=1, dp=1))))

    async def forward(*a, **kw):
        return dict(logits=torch.zeros(1, 1, 2))
    agent._async_model_forward = forward
    inputs = SimpleNamespace(compact_state=object(), is_dummy=False, is_decoding=False,
                             input_ids=torch.ones(1, 1), seq_length=torch.ones(1), is_chunk=False,
                             is_first_chunk=False, is_last_chunk=False, dp_meta=None)
    return agent, inputs


@pytest.mark.parametrize('output', [False, True])
@pytest.mark.parametrize('child_failure', [False, True])
def test_actual_step_preserves_parent_context_until_shielded_child_finishes(monkeypatch, output, child_failure):
    agent, inputs = _agent(monkeypatch, output=output)
    events = []

    async def run():
        began, allow = asyncio.Event(), asyncio.Event()
        context = [False]

        async def postprocess(*args, **kwargs):
            began.set()
            await allow.wait()
            assert context[0], 'parent stream/distributed context exited before child'
            events.append('late-producer')
            if child_failure:
                raise RuntimeError('postprocess failure')
            return (inputs, None, None, None, None, None) if output else (inputs, None, None, None)
        agent._step_postprocess_with_output = postprocess
        agent._step_postprocess_without_output = postprocess

        @contextmanager
        def contexts():
            context[0] = True
            try:
                yield
            finally:
                context[0] = False
                events.append('context-exit')

        async def background():
            with contexts():
                await agent._async_step(inputs, cache_inputs=CacheCheckpointInputs(compact_access_id=1))
        parent = agent._background_task = asyncio.create_task(background())
        agent.tasks.add(parent)
        await began.wait()
        child = agent._compact_postprocess_task
        assert child is not parent and not child.done()
        parent.cancel()
        await asyncio.sleep(0)
        parent.cancel()  # repeated stop must not remove parent-owned contexts
        await asyncio.sleep(0)
        assert not parent.done() and context[0]

        def gpu_drain():
            assert parent.done() and child.done() and not context[0]
            events.append('gpu-drain')
        agent.shutdown_kv_connector = gpu_drain
        drain = asyncio.create_task(agent.drain_compact())
        await asyncio.sleep(0)
        assert not drain.done()
        with pytest.raises(RuntimeError, match='shutting down'):
            agent.set_forward_inputs({})
        with pytest.raises(RuntimeError, match='completed worker drain'):
            agent.release()
        allow.set()
        await asyncio.wait_for(drain, 2)
        assert agent._compact_shutdown_complete and agent._compact_postprocess_task is None
        assert parent.done()
        await agent.drain_compact()  # duplicate drain is a no-op
    asyncio.run(run())
    assert events == ['late-producer', 'context-exit', 'gpu-drain']


def test_cancelled_drain_keeps_background_child_and_physical_caches(monkeypatch):
    agent, inputs = _agent(monkeypatch)

    async def run():
        began, allow = asyncio.Event(), asyncio.Event()
        async def postprocess(*a, **kw):
            began.set()
            await allow.wait()
            return inputs, None, None, None, None, None
        agent._step_postprocess_with_output = postprocess
        parent = agent._background_task = asyncio.create_task(
            agent._async_step(inputs, cache_inputs=CacheCheckpointInputs(compact_access_id=1)))
        agent.tasks.add(parent)
        await began.wait()
        calls = []
        agent.shutdown_kv_connector = lambda: calls.append('gpu-drain')
        drain = asyncio.create_task(agent.drain_compact())
        await asyncio.sleep(0)
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain
        assert not parent.done() and not agent._compact_postprocess_task.done()
        assert agent.cache_engine is not None and agent.state_cache_engine is not None
        assert not getattr(agent, '_compact_shutdown_complete', False) and not calls
        allow.set()
        await agent.drain_compact()
        assert parent.done() and calls == ['gpu-drain']
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['gpu', 'connector'])
def test_failed_physical_drain_never_authorizes_release(monkeypatch, failure):
    agent, _ = _agent(monkeypatch)
    agent._background_task = SimpleNamespace(done=lambda: True)
    def fail():
        raise RuntimeError(f'{failure} failure')
    agent.shutdown_kv_connector = fail
    with pytest.raises(RuntimeError, match=f'{failure} failure'):
        asyncio.run(agent.drain_compact())
    assert not getattr(agent, '_compact_shutdown_complete', False)
    with pytest.raises(RuntimeError, match='completed worker drain'):
        agent.release()
    assert agent.cache_engine is not None and agent.state_cache_engine is not None


@pytest.mark.parametrize('failure', [False, True])
def test_ray_all_worker_drain_acknowledgment_is_required(failure):
    pytest.importorskip('ray')
    from lmdeploy.pytorch.engine.executor.ray_executor import RayExecutor
    executor = RayExecutor.__new__(RayExecutor)
    executor.cache_config = SimpleNamespace(compact_state_capacity=object())
    executor._compact_workers_started = True
    executor._output_steps = deque([(False, 7)])

    async def run():
        arrived, allow = asyncio.Event(), asyncio.Event()
        async def fast():
            arrived.set()
        async def slow():
            await allow.wait()
            if failure:
                raise RuntimeError('unreachable worker')
        executor.workers = [SimpleNamespace(drain_compact=SimpleNamespace(remote=f)) for f in (fast, slow)]
        task = asyncio.create_task(executor.drain_compact())
        await arrived.wait()
        assert not task.done() and len(executor._output_steps) == 1
        with pytest.raises(RuntimeError, match='successful all-worker drain'):
            executor.release()
        allow.set()
        if failure:
            with pytest.raises(RuntimeError, match='unreachable worker'):
                await task
            assert not getattr(executor, '_compact_workers_drained', False)
            assert len(executor._output_steps) == 1
        else:
            await task
            assert executor._compact_workers_drained and not executor._output_steps
    asyncio.run(run())


@pytest.mark.parametrize('failure', [False, True])
def test_engine_finally_only_releases_after_successful_compact_drain(monkeypatch, failure):
    from lmdeploy.pytorch.engine import engine_loop as module
    events = []
    engine = Engine.__new__(Engine)
    engine.cache_config = SimpleNamespace(compact_state_capacity=object())
    engine._block_new_inputs = lambda: events.append('block')
    engine._cancel_and_end_all_sessions = lambda reason: events.append(reason)
    engine.scheduler = SimpleNamespace(finish_kv_transfers_after_worker_drain=lambda: events.append('host-drain'))
    engine._loop_finally = lambda: events.append('release')
    async def drain():
        events.append('worker-drain')
        if failure:
            raise RuntimeError('worker unreachable')
    engine.executor = SimpleNamespace(drain_compact=drain)
    async def wait():
        raise ValueError('original forward failure')
    loop = SimpleNamespace(migration_event=None, start=lambda _: None, wait_tasks=wait,
                           stop=lambda: events.append('stop'))
    monkeypatch.setattr(module, 'build_engine_loop', lambda _: loop)
    with pytest.raises(RuntimeError if failure else ValueError,
                       match='worker unreachable' if failure else 'original forward failure'):
        asyncio.run(engine.async_loop())
    assert events == ['stop', 'block', 'engine shutdown', 'worker-drain'] + (
        [] if failure else ['host-drain', 'release'])
    assert engine._engine_loop is None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two allowed GPUs')
@pytest.mark.parametrize('slow_rank', [0, 1])
def test_real_two_rank_teardown_retains_owner_until_late_gpu_writes_finish(slow_rank):
    pytest.importorskip('ray')
    from lmdeploy.pytorch.config import CompactStateCapacity
    from lmdeploy.pytorch.engine.executor.ray_executor import RayExecutor
    from lmdeploy.pytorch.paging.compact_state import CompactStateManager
    owner = CompactStateManager(CompactStateCapacity(2, 1, 0, 0))
    bank = owner.allocate_bank()
    access = owner.submit_decode((bank,))
    owner.invalidate(access.access_id)
    owner.release_bank(bank)
    agents, pools, streams = [], [], []
    for rank in range(2):
        with torch.cuda.device(rank):
            pool = torch.zeros(owner.capacity.num_units, device='cuda')
            torch.cuda.synchronize()
            pools.append(pool)
            streams.append(torch.cuda.Stream())
        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.tasks = set()
        agent._pre_in_que = agent._in_que = agent._out_que = None
        agent.kv_connector = None
        agent._release_completed_h2d_transfers = lambda: None
        def drain(rank=rank, agent=agent):
            with torch.cuda.device(rank):
                BaseModelAgent.shutdown_kv_connector(agent)
                assert pools[rank][bank.unit_ids[0]].item() == rank + 1
        agent.shutdown_kv_connector = drain
        agents.append(agent)

    async def run():
        gates = [asyncio.Event(), asyncio.Event()]
        started = [asyncio.Event(), asyncio.Event()]
        for rank, agent in enumerate(agents):
            async def producer(rank=rank):
                started[rank].set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await gates[rank].wait()
                    with torch.cuda.device(rank), torch.cuda.stream(streams[rank]):
                        torch.cuda._sleep(50_000_000)
                        pools[rank][bank.unit_ids[0]].fill_(rank + 1)
                    raise
            task = asyncio.create_task(producer())
            agent.tasks.add(task)
        await asyncio.gather(*(event.wait() for event in started))
        executor = RayExecutor.__new__(RayExecutor)
        executor._output_steps = deque([(False, access.access_id)])
        executor.workers = [SimpleNamespace(drain_compact=SimpleNamespace(remote=agent.drain_compact))
                            for agent in agents]
        drain = asyncio.create_task(executor.drain_compact())
        gates[1 - slow_rank].set()
        while not getattr(agents[1 - slow_rank], '_compact_shutdown_complete', False):
            await asyncio.sleep(.001)
        assert not drain.done() and not access.finished and owner.get_num_free() == 0
        assert not getattr(agents[slow_rank], '_compact_shutdown_complete', False)
        gates[slow_rank].set()
        await asyncio.wait_for(drain, 5)
        assert not access.finished  # Worker proof alone never mutates host owner.
        owner.finish_cancelled_forwards_after_worker_drain()
        assert access.finished and access.failed
        assert not owner._accesses and not owner._banks and not owner._units
    asyncio.run(run())


@pytest.mark.parametrize('started', [False, True])
def test_release_allows_unstarted_or_successfully_drained_compact_worker(monkeypatch, started):
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.compact_runtime = object()
    agent._background_task = object() if started else None
    agent._compact_shutdown_complete = started
    agent.kv_connector = None
    agent._checkpoint_engine_zmq_ctx = None
    agent.memdecode_agent = None
    agent.reset_graph_runner = lambda: None
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    agent.release()
    assert agent.cache_engine is None and agent.compact_runtime is None
