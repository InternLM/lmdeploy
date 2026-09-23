# Copyright (c) OpenMMLab. All rights reserved.
"""MTP update routing; real CUDA IPC and model loading are covered by E2E."""
import pickle
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pybase64
import pytest
import torch

from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from lmdeploy.serve.openai.protocol import UpdateParamsRequest
from lmdeploy.utils import FlattenedTensorBucket


@pytest.fixture
def agent():
    agent = BaseModelAgent.__new__(BaseModelAgent)
    config = SimpleNamespace(num_hidden_layers=5, num_nextn_predict_layers=2)
    # Follower ranks have the draft config but no local proposer.
    agent.spec_agent = SimpleNamespace(is_enabled=lambda: True, method='deepseek_mtp', proposer=None,
                                       model_config=SimpleNamespace(hf_config=config, num_layers=2))
    return agent


@pytest.mark.parametrize('method,enabled,indices', [
    ('deepseek_mtp', True, [2, 3]), ('qwen3_5_mtp', True, [6]),
    ('deepseek_mtp', False, []), ('eagle', True, []),
])
def test_split_updated_weights(agent, method, enabled, indices):
    agent.spec_agent.method = method
    agent.spec_agent.is_enabled = lambda: enabled
    names = ['model.embed_tokens.weight', 'model.layers.4.input_layernorm.weight',
             'model.layers.5.eh_proj.weight', 'model.layers.6.shared_head.norm.weight',
             'model.layers.50.eh_proj.weight', 'model.layers.7.eh_proj.weight', 'mtp.fc.weight', 'lm_head.weight']
    weights = [(name, torch.zeros(1)) for name in names]
    main, draft = agent._split_updated_weights(weights)
    assert [n for n, _ in draft] == [names[i] for i in indices]
    assert [n for n, _ in main] == [n for i, n in enumerate(names) if i not in indices]
    assert all(t is dict(weights)[n] for n, t in main + draft)


@pytest.mark.parametrize('entry', ['update_params', 'shared_bucket_loader'])
def test_update_routes_before_renaming(agent, monkeypatch, entry):
    loaded = {'main': [], 'draft': []}

    def model(tag):
        return SimpleNamespace(rename_weight=lambda n: 'renamed.' + n,
                               load_weights=lambda weights: loaded[tag].extend(weights))

    main, draft = model('main'), model('draft')
    agent.patched_model = SimpleNamespace(get_model=lambda: main)
    agent.spec_agent.get_model = lambda: draft
    weights = [('lm_head.weight', torch.tensor([1.])), ('model.layers.5.enorm.weight', torch.tensor([2.]))]
    if entry == 'update_params':
        bucket = FlattenedTensorBucket(named_tensors=weights)
        agent.all_context = nullcontext
        agent._update_params_ipc_tensor = bucket.get_flattened_tensor()
        agent._update_params_ipc_event = Mock()
        monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
        payload = pybase64.b64encode(pickle.dumps({'metadata': bucket.metadata})).decode()
        agent.update_params(UpdateParamsRequest(serialized_named_tensors=payload,
                                                load_format='flattened_bucket', finished=False))
        assert agent._update_params_ipc_event.record.call_count == 2
    else:
        # NCCL and checkpoint-engine IPC use this upstream shared loader.
        agent._load_updated_weight_bucket(weights, 'test')
    for tag, (name, tensor) in zip(('main', 'draft'), weights):
        assert len(loaded[tag]) == 1
        actual_name, actual_tensor = loaded[tag][0]
        assert actual_name == 'renamed.' + name
        torch.testing.assert_close(actual_tensor, tensor)
