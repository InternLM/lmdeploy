# Copyright (c) OpenMMLab. All rights reserved.
"""Flattened IPC acknowledgments cover both models and exceptional loads."""
import pickle
from contextlib import nullcontext
from types import SimpleNamespace

import pybase64
import pytest
import torch

from lmdeploy.pytorch.engine.model_agent import agent as agent_module
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from lmdeploy.serve.openai.protocol import UpdateParamsRequest
from lmdeploy.utils import FlattenedTensorBucket


def _make_agent(monkeypatch, *, compact=False, failure=None):
    events = []

    def model(tag):
        def load(weights):
            assert list(weights)
            events.append(tag)
            if failure == tag:
                raise RuntimeError('load failed')
        return SimpleNamespace(rename_weight=lambda name: name, load_weights=load, tag=tag)

    main, draft = model('main'), model('draft')
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.cache_config = SimpleNamespace(compact_state_capacity=object() if compact else None,
                                        use_mooncake_store=compact)
    agent.patched_model = SimpleNamespace(get_model=lambda: main)
    agent.spec_agent = SimpleNamespace(get_model=lambda: draft,
                                       split_updated_weights=lambda weights: (weights[:1], weights[1:]))
    agent.all_context = nullcontext
    bucket = FlattenedTensorBucket(named_tensors=[('main', torch.ones(1)), ('draft', torch.ones(1))])
    agent._update_params_ipc_tensor = bucket.get_flattened_tensor()
    agent._update_params_ipc_event = SimpleNamespace(wait=lambda: events.append('wait'),
                                                    record=lambda: events.append('record'))

    def finalize(target):
        events.append('finalize_' + target.tag)
        if failure == 'finalize_' + target.tag:
            raise RuntimeError('finalize failed')

    def synchronize():
        events.append('sync')
        if failure == 'sync':
            raise RuntimeError('sync failed')

    monkeypatch.setattr(agent_module, 'process_weights_after_loading', finalize)
    monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    return agent, bucket, events


def _request(metadata, *, finished=False):
    payload = pybase64.b64encode(pickle.dumps({'metadata': metadata})).decode()
    return UpdateParamsRequest(serialized_named_tensors=payload, load_format='flattened_bucket', finished=finished)


def test_ipc_acknowledges_once_after_both_models(monkeypatch):
    agent, bucket, events = _make_agent(monkeypatch)
    agent.update_params(_request(bucket.metadata))
    assert events == ['wait', 'main', 'draft', 'record']
    assert agent._update_params_ipc_tensor is not None
    assert agent._update_params_ipc_event is not None


def test_compact_rejection_acknowledges_and_releases_without_mutation(monkeypatch):
    agent, bucket, events = _make_agent(monkeypatch, compact=True)
    with pytest.raises(RuntimeError, match='new immutable namespace'):
        agent.update_params(_request(bucket.metadata))
    assert events == ['wait', 'record', 'sync']
    assert agent._update_params_ipc_tensor is None
    assert agent._update_params_ipc_event is None


@pytest.mark.parametrize('failure', ['main', 'draft', 'finalize_main', 'finalize_draft'])
def test_partial_failure_acknowledges_only_after_last_consumer(monkeypatch, failure):
    agent, bucket, events = _make_agent(monkeypatch, failure=failure)
    with pytest.raises(RuntimeError, match='failed'):
        agent.update_params(_request(bucket.metadata, finished=failure.startswith('finalize_')))
    assert events[0] == 'wait'
    assert events.count('record') == 1
    assert events.index('record') > events.index(failure)
    assert events[-1] == 'sync'
    assert agent._update_params_ipc_tensor is None
    assert agent._update_params_ipc_event is None


def test_empty_finalize_is_not_dropped(monkeypatch):
    agent, _, events = _make_agent(monkeypatch)
    agent.update_params(_request([], finished=True))
    assert 'main' not in events and 'draft' not in events
    assert 'finalize_main' in events and 'finalize_draft' in events
    assert events[-1] == 'sync'
    assert agent._update_params_ipc_tensor is None
    assert agent._update_params_ipc_event is None


def test_sync_failure_does_not_release_ipc_storage(monkeypatch):
    agent, bucket, events = _make_agent(monkeypatch, failure='sync')
    with pytest.raises(RuntimeError, match='sync failed'):
        agent.update_params(_request(bucket.metadata, finished=True))
    assert agent._update_params_ipc_tensor is not None
    assert agent._update_params_ipc_event is not None


def test_producer_event_precedes_optional_clone(monkeypatch):
    agent, bucket, events = _make_agent(monkeypatch)

    class IpcTensor:
        def clone(self):
            events.append('clone')
            return bucket.get_flattened_tensor()

    # Stub only handle reconstruction, not the bucket/load lifecycle.
    monkeypatch.setattr(agent_module.ForkingPickler, 'loads', lambda _: {
        'metadata': bucket.metadata, 'flattened_tensor': (lambda *args: IpcTensor(), [None] * 7),
        'event_ipc_handle': b'event', 'require_clone': True,
    })
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    event = agent._update_params_ipc_event
    monkeypatch.setattr(torch.cuda.Event, 'from_ipc_handle', lambda **kwargs: event)
    agent.update_params(_request(bucket.metadata))
    assert events == ['wait', 'clone', 'main', 'draft', 'record']
