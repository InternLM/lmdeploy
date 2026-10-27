# Copyright (c) OpenMMLab. All rights reserved.
"""Compact teardown must not change other model agents' lifecycle."""
import asyncio
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent


def _agent(monkeypatch, *, output=True, compact=True):
    from lmdeploy.pytorch.engine.model_agent import agent as module
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.rank = 0
    agent.need_output = output
    agent.ssm_state_cursor = object() if compact else None
    agent.cache_config = SimpleNamespace(compact_state_capacity=object() if compact else None)
    agent.profiler = agent.guided_decoding_manager = None
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
    inputs = SimpleNamespace(ssm_state_meta=object() if compact else None, is_dummy=False, is_decoding=False,
                             input_ids=torch.ones(1, 1), seq_length=torch.ones(1), is_chunk=False,
                             is_first_chunk=False, is_last_chunk=False, dp_meta=None)
    return agent, inputs


@pytest.mark.parametrize('dp', [1, 4])
@pytest.mark.parametrize('method', ['stop', 'stop_async'])
@pytest.mark.parametrize('compact', [False, True])
def test_stop_keeps_baseline_contract(dp, method, compact):
    async def run():
        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.cache_config = SimpleNamespace(compact_state_capacity=object() if compact else None)
        agent.dist_config = SimpleNamespace(dp=dp)
        agent.profiler = agent.guided_decoding_manager = None
        agent._release_completed_h2d_transfers = lambda: None
        agent.shutdown_kv_connector = lambda: pytest.fail('stop must retain its baseline contract')
        producer = asyncio.create_task(asyncio.Event().wait())
        agent.tasks = {producer}
        await asyncio.sleep(0)
        try:
            if method == 'stop_async':
                await agent.stop_async()
            else:
                agent.stop()
                await asyncio.sleep(0)
            assert '_stop_task' not in vars(agent)
            assert producer.cancelled() == (dp == 1)
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('output', [False, True])
def test_postprocess_retains_baseline_shield(monkeypatch, output, compact):
    agent, inputs = _agent(monkeypatch, output=output, compact=compact)

    async def run():
        began, release = asyncio.Event(), asyncio.Event()
        children = []

        async def postprocess(*args, **kwargs):
            children.append(asyncio.current_task())
            began.set()
            await release.wait()
            return (inputs, None, None, None, None, None) if output else (inputs, None, None, None)
        agent._step_postprocess_with_output = postprocess
        agent._step_postprocess_without_output = postprocess
        cache_inputs = CacheCheckpointInputs(ssm_state_access_id=1) if compact else None
        parent = asyncio.create_task(agent._async_step(inputs, cache_inputs=cache_inputs))
        try:
            await asyncio.wait_for(began.wait(), 2)
            parent.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(parent, 2)
            assert not children[0].done()  # Baseline shield keeps only the child alive.
        finally:
            release.set()
            await asyncio.gather(parent, *children, return_exceptions=True)
    asyncio.run(run())


def test_noncompact_release_needs_no_compact_stop_proof(monkeypatch):
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.cache_config = SimpleNamespace(compact_state_capacity=None)
    agent._background_task = object()
    agent.kv_connector = agent.memdecode_agent = agent._checkpoint_engine_zmq_ctx = None
    agent.reset_graph_runner = lambda: None
    step_inputs = agent.step_inputs = object()
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    agent.release()
    assert agent.step_inputs is step_inputs


@pytest.mark.parametrize('dp', [1, 4])
@pytest.mark.parametrize('compact', [False, True])
def test_ray_stop_keeps_baseline_rpc(dp, compact):
    pytest.importorskip('ray')
    from lmdeploy.pytorch.engine.executor.ray_executor import RayExecutor

    executor = RayExecutor.__new__(RayExecutor)
    executor.dp = dp
    executor.cache_config = SimpleNamespace(compact_state_capacity=object() if compact else None)
    events = []
    executor._prefetch_task = SimpleNamespace(cancel=lambda: events.append('prefetch'))
    executor.collective_rpc = lambda method: events.append(method)
    executor.stop()
    assert events == (['stop_async'] if dp == 1 else []) + ['prefetch']


@pytest.mark.parametrize('compact', [False, True])
def test_engine_finally_keeps_baseline_release(monkeypatch, compact):
    from lmdeploy.pytorch.engine import engine_loop
    from lmdeploy.pytorch.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.cache_config = SimpleNamespace(compact_state_capacity=object() if compact else None)
    events = []
    engine._loop_finally = lambda: events.append('release')
    async def wait():
        raise ValueError('forward failed')
    loop = SimpleNamespace(migration_event=None, start=lambda _: None, wait_tasks=wait,
                           stop=lambda: events.append('stop'))
    monkeypatch.setattr(engine_loop, 'build_engine_loop', lambda _: loop)
    with pytest.raises(ValueError, match='forward failed'):
        asyncio.run(engine.async_loop())
    assert events == ['stop', 'release']


@pytest.mark.parametrize('started', [False, True])
def test_release_clears_compact_resources_without_stop_registration(monkeypatch, started):
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.ssm_state_cursor = object()
    agent.cache_config = SimpleNamespace(compact_state_capacity=object())
    agent._background_task = object() if started else None
    agent.kv_connector = None
    agent._checkpoint_engine_zmq_ctx = None
    agent.memdecode_agent = None
    agent.reset_graph_runner = lambda: None
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    agent.release()
    assert agent.cache_engine is None and agent.ssm_state_cursor is None
