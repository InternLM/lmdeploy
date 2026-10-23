# Copyright (c) OpenMMLab. All rights reserved.
"""Immutable reload transaction, transport drain, and all-rank wakeup
gating."""
import asyncio
import pickle
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pybase64
import pytest
import torch

from lmdeploy.pytorch.engine.engine import Engine
from lmdeploy.pytorch.engine.executor.uni_executor import UniExecutor
from lmdeploy.pytorch.engine.model_agent import agent as module
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent, SleepWakeupState, _CompactWeightReload
from lmdeploy.serve.openai.protocol import (
    CompactStoreWeightIdentity,
    UpdateParamsRequest,
    UpdateWeightsFromDistributedRequest,
    UpdateWeightsFromIPCRequest,
)
from lmdeploy.utils import FlattenedTensorBucket


def _identity(name='new'):
    return CompactStoreWeightIdentity(model_identity=name, draft_identity=name + '-draft')


@pytest.fixture
def worker(monkeypatch):
    events = []

    class Model:
        device = 'cuda'
        fail = False

        def __init__(self, tag):
            self.tag = tag

        def parameters(self):
            yield SimpleNamespace(device=SimpleNamespace(type=self.device))

        def to(self, device, **kwargs):
            self.device = device
            events.append(('to', self.tag, device))

        def load_weights(self, weights):
            events.append(('load', self.tag, [n for n, _ in weights]))
            if self.fail:
                raise RuntimeError('load failure')

    main, draft = Model('main'), Model('draft')
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.cache_config = SimpleNamespace(compact_state_capacity=object(), use_mooncake_store=True,
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
            'compact_model_identity': 'old', 'compact_draft_identity': 'old-draft'}))
    agent.patched_model = SimpleNamespace(get_model=lambda: main)
    agent.spec_agent = SimpleNamespace(get_model=lambda: draft, specdecode_config=object(), cache_engine=None,
        split_updated_weights=lambda weights: ([w for w in weights if w[0] == 'main'],
                                                [w for w in weights if w[0] == 'draft']))
    agent.cache_engine = agent.state_cache_engine = agent.kv_connector = None
    agent._compact_weight_reload = _CompactWeightReload()  # level2 or empty init
    agent._update_params_ipc_tensor = agent._update_params_ipc_event = None
    agent._checkpoint_engine_zmq_ctx = None
    agent._model_update_group = {'trainer': object()}
    agent.all_context = nullcontext
    agent.state = SleepWakeupState(is_sleeping=True)
    agent.dist_config = SimpleNamespace(dp=1)
    agent.memdecode_agent = None
    agent.reset_graph_runner = lambda: events.append('reset_graph')
    agent.shutdown_kv_connector = lambda: events.append('drain_connector')
    agent.reset_runtime_state = lambda: events.append('reset_runtime')
    monkeypatch.setattr(module.ModelWeightLoader, '_rename_weights_iterator', lambda weights, _: iter(weights))
    monkeypatch.setattr(module, 'process_weights_after_loading', lambda model: events.append(('finalize', model.tag)))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: events.append('sync'))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    return agent, main, draft, events


def _ipc(agent, names=('main', 'draft'), *, finished=False, identity='new'):
    bucket = FlattenedTensorBucket(named_tensors=[(name, torch.ones(1)) for name in names])
    if names:
        agent._update_params_ipc_tensor = bucket.get_flattened_tensor()
    request = UpdateParamsRequest(
        serialized_named_tensors=pybase64.b64encode(pickle.dumps({'metadata': bucket.metadata})).decode(),
        load_format='flattened_bucket', finished=finished,
        compact_store_identity=_identity(identity) if identity is not None else None)
    agent.update_params(request)


def test_multibucket_reload_keeps_old_identity_until_empty_finalize(worker):
    agent, _, _, events = worker
    extra = agent.cache_config.kv_transfer_config.kv_connector_extra_config
    _ipc(agent, ('main',))
    _ipc(agent, ('draft',))
    assert extra['compact_model_identity'] == 'old'
    for call in (lambda: agent.wakeup(['kv_cache']), lambda: agent.wakeup(), agent.build_cache_engine, agent.warmup):
        with pytest.raises(RuntimeError, match='full reload'):
            call()
    _ipc(agent, (), finished=True)
    assert agent._compact_weight_reload is None
    assert extra == {'compact_model_identity': 'new', 'compact_draft_identity': 'new-draft'}
    assert events.count(('finalize', 'main')) == events.count(('finalize', 'draft')) == 1
    assert 'reset_graph' in events
    agent.build_cache_engine = lambda: events.append('cache')
    agent.warmup = lambda: events.append('warmup')
    agent.wakeup(['kv_cache'])
    assert events[-2:] == ['cache', 'warmup']
    assert not agent.state.is_sleeping


@pytest.mark.parametrize('bad', ['missing', 'conflict', 'draft_missing', 'blank', 'empty_finish',
                                  'only_main', 'target_failure', 'draft_failure', 'finalizer_failure'])
def test_failed_reload_never_publishes_identity_or_wakes(worker, monkeypatch, bad):
    agent, main, draft, events = worker
    if bad == 'conflict':
        _ipc(agent, ('main',))
    if bad in ('target_failure', 'draft_failure'):
        (main if bad == 'target_failure' else draft).fail = True
    if bad == 'finalizer_failure':
        monkeypatch.setattr(module, 'process_weights_after_loading', lambda _: (_ for _ in ()).throw(
            RuntimeError('finalizer failure')))
    with pytest.raises(RuntimeError):
        if bad in ('draft_missing', 'blank'):
            request = UpdateParamsRequest(serialized_named_tensors='unused', compact_store_identity=
                CompactStoreWeightIdentity(model_identity=' ' if bad == 'blank' else 'new'))
            # Validate through the actual route after deserializing an empty bucket.
            request.serialized_named_tensors = pybase64.b64encode(pickle.dumps({'metadata': []})).decode()
            request.load_format = 'flattened_bucket'
            agent.update_params(request)
        else:
            _ipc(agent, () if bad == 'empty_finish' else ('main',) if bad == 'only_main' else ('main', 'draft'),
                 finished=True, identity=None if bad == 'missing' else 'other' if bad == 'conflict' else 'new')
    assert agent._compact_weight_reload.failed
    assert agent.cache_config.kv_transfer_config.kv_connector_extra_config['compact_model_identity'] == 'old'
    with pytest.raises(RuntimeError, match='full reload'):
        agent.wakeup(['kv_cache'])
    with pytest.raises(RuntimeError, match='previously failed'):
        _ipc(agent, finished=True)


@pytest.mark.parametrize('bad', ['awake', 'cache', 'state_cache', 'connector', 'cpu', 'meta'])
def test_reload_requires_quiescent_cuda_model(worker, bad):
    agent, main, _, events = worker
    if bad == 'awake':
        agent.state.is_sleeping = False
    elif bad in ('cpu', 'meta'):
        main.device = bad
    else:
        setattr(agent, {'cache': 'cache_engine', 'state_cache': 'state_cache_engine', 'connector': 'kv_connector'}[bad],
                object())
    with pytest.raises(RuntimeError, match='requires'):
        _ipc(agent)
    assert not any(isinstance(e, tuple) and e[0] == 'load' for e in events)


def test_level2_discard_and_weight_wakeup_preserve_pending_reload(worker):
    agent, main, draft, events = worker
    agent._compact_weight_reload = None
    agent.state.is_sleeping = False
    asyncio.run(agent.sleep(2))
    assert main.device == draft.device == 'meta'
    assert agent._compact_weight_reload is not None
    assert events.index('drain_connector') < events.index(('to', 'main', 'meta'))
    def rebuild():
        main.device = draft.device = 'cuda'
        events.append('rebuild_empty')
    agent._rebuild_models_after_level2_sleep = rebuild
    agent.wakeup(['weights'])
    assert events[-1] == 'rebuild_empty'
    assert agent._compact_weight_reload.identity is None
    with pytest.raises(RuntimeError, match='full reload'):
        agent.wakeup(['kv_cache'])


def test_level1_sleep_and_reload_starts_a_new_transaction(worker):
    agent, main, draft, events = worker
    agent._compact_weight_reload = None
    agent.state.is_sleeping = False
    asyncio.run(agent.sleep(1))
    assert agent._compact_weight_reload is None
    main.device = draft.device = 'cuda'  # weight-preserving CUDA wakeup
    _ipc(agent, finished=True)
    assert agent._compact_weight_reload is None
    assert agent.cache_config.kv_transfer_config.kv_connector_extra_config['compact_model_identity'] == 'new'


@pytest.mark.parametrize('valid', [False, True])
def test_nccl_receives_before_identity_validation(worker, monkeypatch, valid):
    agent, _, _, events = worker
    real_empty = torch.empty
    monkeypatch.setattr(torch, 'empty', lambda *args, **kwargs: real_empty(*args, **{**kwargs, 'device': 'cpu'}))
    def broadcast(tensor, **kwargs):
        events.append('receive')
        return SimpleNamespace(wait=lambda: events.append('wait_receive'))
    monkeypatch.setattr(module.dist, 'broadcast', broadcast)
    request = UpdateWeightsFromDistributedRequest(names=['main', 'draft'], shapes=[[1], [1]],
        dtypes=['float32', 'float32'], group_name='trainer', finished=True,
        compact_store_identity=_identity() if valid else None)
    success, _ = agent.update_weights_from_distributed(request)
    assert success == valid
    assert events[:4] == ['receive', 'receive', 'wait_receive', 'wait_receive']
    assert (agent._compact_weight_reload is None) == valid


@pytest.mark.parametrize('failure', [None, 'identity', 'transport', 'empty'])
def test_checkpoint_ipc_commit_follows_transport_and_finalize(worker, monkeypatch, failure):
    agent, _, _, events = worker
    def receiver(ctx, handle, device_id, *, run, post_hook):
        events.append('transport')
        if failure != 'empty':
            run([('main', torch.ones(1)), ('draft', torch.ones(1))])
        post_hook()
        assert agent.cache_config.kv_transfer_config.kv_connector_extra_config['compact_model_identity'] == 'old'
        if failure == 'transport':
            raise RuntimeError('post-hook transport failure')
    modules = {'zmq': SimpleNamespace(Context=lambda: object()),
               'checkpoint_engine.worker': SimpleNamespace(update_weights_from_ipc=receiver)}
    monkeypatch.setattr(module.importlib, 'import_module', lambda name: modules[name])
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(uuid='test'))
    success, _ = agent.update_weights_from_ipc(UpdateWeightsFromIPCRequest(zmq_handles={'GPU-test': 'ipc://test'},
        compact_store_identity=None if failure == 'identity' else _identity()))
    assert success == (failure is None)
    assert events[0] == 'transport'
    assert (agent._compact_weight_reload is None) == success


def _engine():
    engine = Engine.__new__(Engine)
    engine.cache_config = SimpleNamespace(compact_state_capacity=object(), use_mooncake_store=True,
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={'compact_model_identity': 'old'}))
    engine._compact_weight_control_lock = threading.Lock()
    engine._compact_weight_reload_state = 'pending'
    engine._weights_update_lock = None
    engine._sleeping_tags = {'kv_cache'}
    engine._engine_loop = None
    engine._unblock_new_inputs = lambda: None
    return engine


@pytest.mark.parametrize('route', ['flat', 'checkpoint', 'nccl'])
@pytest.mark.parametrize('success', [False, True])
def test_engine_all_worker_success_is_required_for_identity_and_wakeup(route, success):
    engine = _engine()
    events = []
    def update(request, *args):
        assert engine._compact_weight_reload_state == 'pending'
        events.append('workers')
        if route == 'flat':
            if not success:
                raise RuntimeError('rank1 failed after rank0 finalized')
        else:
            return success, 'all ranks' if success else 'rank1 failed after rank0 finalized'
    engine.executor = SimpleNamespace(update_params=update, update_weights_from_ipc=update,
        update_weights_from_distributed=update, wakeup=lambda tags: events.append(('wake', tags)))
    request = SimpleNamespace(finished=True, compact_store_identity=_identity())
    if route == 'flat' and not success:
        with pytest.raises(RuntimeError, match='rank1'):
            engine.update_params(request)
    elif route == 'flat':
        engine.update_params(request)
    else:
        result = asyncio.run(engine.update_weights_from_ipc(request) if route == 'checkpoint'
                             else engine.update_weights_from_distributed(request))
        assert result[0] == success
    extra = engine.cache_config.kv_transfer_config.kv_connector_extra_config
    if success:
        assert engine._compact_weight_reload_state is None and extra['compact_model_identity'] == 'new'
        engine.complete_weights_update()
        engine.wakeup(['kv_cache'])
        assert events[-1] == ('wake', ['kv_cache'])
    else:
        assert engine._compact_weight_reload_state == 'failed' and extra['compact_model_identity'] == 'old'
        with pytest.raises(RuntimeError, match='every worker'):
            engine.wakeup(['kv_cache'])
        with pytest.raises(RuntimeError, match='every worker'):
            engine.complete_weights_update()
        # A later spurious all-worker success cannot clear the global failure.
        with pytest.raises(RuntimeError, match='previously failed'):
            engine._apply_weight_update(lambda _: (True, 'ok'), request, True, True)


def test_inflight_update_excludes_sync_wakeup_without_blocking_event_loop():
    engine = _engine()
    started, release = threading.Event(), threading.Event()
    def update(request, *args):
        started.set()
        assert release.wait(10)
        return True, 'ok'
    engine.executor = SimpleNamespace(update_weights_from_ipc=update)
    async def run():
        task = asyncio.create_task(engine.update_weights_from_ipc(SimpleNamespace(compact_store_identity=_identity())))
        assert await asyncio.to_thread(started.wait, 10)
        try:
            with pytest.raises(RuntimeError, match='in progress'):
                engine.wakeup(['weights'])
            with pytest.raises(RuntimeError, match='in progress'):
                await engine.sleep(1)
        finally:
            release.set()
        assert (await task)[0]
    asyncio.run(run())


@pytest.mark.parametrize('method', ['update_params', 'init_weights_update_group',
                                    'update_weights_from_distributed', 'destroy_weights_update_group'])
def test_uni_update_routes_reach_the_model_agent(method):
    executor = UniExecutor.__new__(UniExecutor)
    request = object()
    executor.model_agent = SimpleNamespace(**{method: lambda r: (r is request, 'called')})
    assert getattr(executor, method)(request) == (True, 'called')
