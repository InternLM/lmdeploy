# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch import specialized_execution as api
from lmdeploy.pytorch.backends.cuda.graph_runner.runner import CUDAGraphRunner
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from lmdeploy.pytorch.model_inputs import ModelInputs

PLAN = api.DecodePlan('b8', 8, 128, True)


class Example(api.SpecializedExecutor):
    api_version = 1
    artifact_id = 'example-v1'

    @classmethod
    def describe(cls, ctx):
        return (PLAN,)

    def run(self, plan, **kwargs):
        return self.model(**kwargs)


def inputs(rows=8, length=128, **changes):
    ids = torch.arange(rows)
    values = dict(input_ids=ids[None], seq_length=torch.ones_like(ids), history_lengths=ids, block_offsets=ids[:, None],
                  is_decoding=True, num_ignored_history=torch.zeros_like(ids), max_q_seqlen=1, max_kv_seqlen=length,
                  sum_kv_seqlen=length * rows)
    return ModelInputs(**(values | changes))


@pytest.mark.parametrize('request_inputs, expected', [
    (inputs(), 1), (inputs(length=1), 1), (inputs(length=129), 0), (inputs(rows=7), 0),
    (inputs(is_dummy=True), 0), (inputs(is_chunk=True), 0), (inputs(is_decoding=False), 0),
    (inputs(enable_microbatch=True), 0), (inputs(max_q_seqlen=2), 0), (inputs(vision_inputs=object()), 0),
    (inputs(input_ids=torch.arange(16)[None]), 0),
])
def test_host_only_proposal(request_inputs, expected):
    execution = api.SpecializedExecution((PLAN,), Example(SimpleNamespace(model=None)), 1)
    assert execution.propose(request_inputs, request_inputs.seq_length.numel()) == expected


def test_agreement_and_step_lifetime():
    execution = api.SpecializedExecution((PLAN,), None, 8)
    batch = inputs()
    assert execution.select(batch, 8) is None
    batch.specialized_plan = api.agree_on_plan([1] * 8)
    assert execution.select(batch, 8) == PLAN
    assert execution.select(batch.step(torch.arange(8)), 8) is None
    assert batch.specialized_plan == 1
    assert api.agree_on_plan([1] * 7 + [0]) == api.agree_on_plan([2] + [1] * 7) == 0


def test_versioned_provider_and_config(monkeypatch):
    path = f'{__name__}:Example'
    cfg = PytorchEngineConfig(specialized_executor=path)
    assert ConfigBuilder.build_backend_config(cfg).specialized_executor == path
    assert ConfigBuilder.build_backend_config(PytorchEngineConfig()).specialized_executor is None
    ctx = SimpleNamespace(model=object(), num_spec_tokens=0,
                          dist=SimpleNamespace(dist_config=SimpleNamespace(world_size=1, dp=1)))
    assert api.build_specialized_execution(path, ctx).executor.model is ctx.model
    monkeypatch.setattr(Example, 'api_version', 7)
    with pytest.raises(RuntimeError, match='API version'):
        api.build_specialized_execution(path, ctx)


def test_startup_coverage_must_match_before_prepare(monkeypatch):
    path = f'{__name__}:Example'
    ctx = SimpleNamespace(model=None, num_spec_tokens=0,
                          dist=SimpleNamespace(dist_config=SimpleNamespace(world_size=2), cpu_group=None))
    monkeypatch.setattr(api.dist, 'all_gather_object', lambda dst, val, group: dst.__setitem__(slice(None), [val, ()]))
    monkeypatch.setattr(Example, '__init__', lambda *args: pytest.fail('constructed before consensus'))
    with pytest.raises(RuntimeError, match='inconsistent'):
        api.build_specialized_execution(path, ctx)


def test_remote_preparation_failure_closes_local_executor(monkeypatch):
    path = f'{__name__}:Example'
    ctx = SimpleNamespace(model=None, num_spec_tokens=0)
    monkeypatch.setattr(api, '_gather', lambda ctx, val: [None, 'rank 1: failed'] if val is None else [val, val])
    closed = []
    monkeypatch.setattr(Example, 'close', lambda self: closed.append(self))
    with pytest.raises(RuntimeError, match='rank 1: failed'):
        api.build_specialized_execution(path, ctx)
    assert len(closed) == 1


@pytest.mark.parametrize('device,paradigm,spec,replay', [('ascend', 'ar', False, False), ('maca', 'ar', False, False),
                                                         ('cuda', 'ar_spec', True, False), ('cuda', 'ar', True, False),
                                                         ('cuda', 'ar', False, True)])
def test_unsupported_deployments_skip_provider_import(device, paradigm, spec, replay):
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.backend_config = SimpleNamespace(device_type=device)
    agent.model_config = SimpleNamespace(model_paradigm=paradigm, enable_replayssm=replay)
    agent.spec_agent = SimpleNamespace(is_enabled=lambda: spec)
    agent.memdecode_agent = agent.adapters = None
    agent._specialized_provider = 'deliberately_missing:Factory'
    agent._prepare_specialized_execution(None)
    assert agent.specialized_execution is None


def test_graph_keys_and_capture_callable():
    runner = CUDAGraphRunner.__new__(CUDAGraphRunner)
    runner.specialized_execution = api.SpecializedExecution((PLAN,), Example(SimpleNamespace(model=lambda: 'r11')), 8)
    bucket, query = 8, 1
    runner.get_graph_key = lambda **kwargs: (bucket, True, False, query, 'model-extra')
    runner._get_decode_model_forward = lambda: lambda: 'standard'
    runner._full_graph_runners = {}
    captures = []

    def capture(key, kwargs, forward):
        captures.append(key)
        runner._full_graph_runners[key] = SimpleNamespace(forward=lambda **_: 'replay')
        return forward()

    runner._capture_full_graph = capture
    assert runner._forward_full_graph(PLAN) == 'r11'
    assert runner._forward_full_graph(PLAN) == 'replay'
    assert runner._forward_full_graph(None) == 'standard'
    assert runner._forward_full_graph(replace(PLAN, graph_safe=False)) == 'replay'
    bucket = 16
    assert runner._forward_full_graph(PLAN) == 'standard'
    mtp = replace(PLAN, name='verify', num_spec_tokens=3)
    bucket = 8
    assert runner._forward_full_graph(mtp) == 'replay'  # q1 standard graph
    query = 4
    assert runner._forward_full_graph(mtp) == 'r11'
    assert runner._forward_full_graph(PLAN) == 'standard'
    assert [key[-1] for key in captures] == [PLAN, 'model-extra', 'model-extra', mtp, 'model-extra']


def test_running_failure_is_never_retried(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.graph_runner import runner as cuda
    runner = CUDAGraphRunner.__new__(CUDAGraphRunner)
    batch = inputs()
    ctx = SimpleNamespace(**vars(batch), q_seqlens=batch.seq_length, global_is_decoding=lambda: True)
    runner.ctx_mgr = SimpleNamespace(current_context=lambda: ctx)
    runner._prepare_inputs = lambda **kw: kw
    runner._should_use_full_graph = lambda *args: False
    monkeypatch.setattr(cuda, '_update_deepep_mode', lambda _: None)
    writes = []

    def run(*args, **kwargs):
        writes.append(1)
        raise RuntimeError('failed after writing')

    runner.specialized_execution = api.SpecializedExecution((PLAN,), SimpleNamespace(run=run), 1)
    runner.model = SimpleNamespace(make_output_buffers=lambda x: x)
    runner._forward_eager = lambda **kwargs: pytest.fail('repeated state write')
    with pytest.raises(RuntimeError, match='after writing'):
        runner()
    assert writes == [1]


def test_refresh_retires_graphs_before_repacking(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []
    agent = BaseModelAgent.__new__(BaseModelAgent)
    model = object()
    agent._specialized_provider = 'example:Test'
    agent.all_context = nullcontext
    agent.specialized_execution = SimpleNamespace(executor=SimpleNamespace(close=lambda: events.append('close')))
    agent.patched_model = SimpleNamespace(
        get_model=lambda: model, reset=lambda: events.append('retire graphs'), get_prefill_warmup_token_sizes=list)
    agent.spec_agent = SimpleNamespace(get_model=lambda: None, reset_graph_runner=lambda: None)
    agent.memdecode_agent = None
    agent.state = SimpleNamespace(is_sleeping=False)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: events.append('sync'))
    monkeypatch.setattr(agent_module, 'process_weights_after_loading', lambda m: events.append(('weights', m)))

    def prepare(updated):
        events.append(('prepare', updated))
        agent.specialized_execution = object()

    agent._prepare_specialized_execution = prepare
    agent._finalize_updated_weights()
    assert events == [('weights', model), 'sync', 'sync', 'retire graphs', 'close', ('prepare', model)]
    assert agent.patched_model.specialized_execution is agent.specialized_execution


def test_distributed_setup_follows_successful_local_agreement(monkeypatch):
    events = []
    path = f'{__name__}:Example'
    ctx = SimpleNamespace(model=object(), num_spec_tokens=0,
                          dist=SimpleNamespace(dist_config=SimpleNamespace(dp=8, world_size=8)))
    monkeypatch.setattr(api, '_gather', lambda ctx, x: events.append('agree') or [x] * 8)
    monkeypatch.setattr(Example, 'prepare_distributed', lambda self, context: events.append('prepare EP'))
    assert api.build_specialized_execution(path, ctx).dp_size == 8
    assert events == ['agree', 'agree', 'prepare EP']


def test_collective_failure_retires_prepared_executor(monkeypatch):
    ctx = SimpleNamespace(model=object(), num_spec_tokens=0,
                          dist=SimpleNamespace(dist_config=SimpleNamespace(dp=1, world_size=1)))
    closed = []
    def fail(self, ctx):
        raise RuntimeError('collective setup failed')
    monkeypatch.setattr(Example, 'prepare_distributed', fail)
    monkeypatch.setattr(Example, 'close', lambda self: closed.append(True))
    with pytest.raises(RuntimeError, match='collective setup failed'):
        api.build_specialized_execution(f'{__name__}:Example', ctx)
    assert closed == [True]

def test_mtp3_target_opt_in_and_uniform_rows(monkeypatch):
    verify = replace(PLAN, name='mtp3', num_spec_tokens=3)
    monkeypatch.setattr(Example, 'describe', lambda ctx: (PLAN, verify))
    ctx = SimpleNamespace(model=None, num_spec_tokens=3,
                          dist=SimpleNamespace(dist_config=SimpleNamespace(dp=1, world_size=1)))
    path = f'{__name__}:Example'
    runtime = api.build_specialized_execution(path, ctx)
    assert runtime.plans == (verify,)
    q4 = inputs(input_ids=torch.arange(32)[None], max_q_seqlen=4, seq_length=torch.full((8,), 4))
    assert runtime.propose(q4, 8) == 1
    assert runtime.propose(inputs(), 8) == 0
    # Seven full rows and a short final verification must use stock.
    assert runtime.propose(q4.clone(input_ids=q4.input_ids[:, :-1], seq_length=torch.tensor([4] * 7 + [3])), 8) == 0
    assert runtime.propose(q4.clone(is_decoding=False), 8) == 0
    monkeypatch.setattr(Example, 'describe', lambda ctx: (PLAN,))
    assert api.build_specialized_execution(path, ctx) is None


def test_explicit_ep_mask_binds_frontend_to_selected_device(monkeypatch):
    import importlib
    import os

    launch = importlib.import_module('lmdeploy.serve.openai.launch_server')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    monkeypatch.setenv('LMDEPLOY_EP_CUDA_VISIBLE_DEVICES', '7,6,5,4,3,2,1,0')
    monkeypatch.setattr(os, 'setpgrp', lambda: None)
    bound = []
    monkeypatch.setattr(launch, 'serve', lambda *args, **kwargs: bound.append(os.environ['CUDA_VISIBLE_DEVICES']))
    cfg = PytorchEngineConfig(dp=8, ep=8, tp=1, specialized_executor='example:Runner')
    launch._run_server([2], 'model', backend_config=cfg)
    assert bound == ['5']
