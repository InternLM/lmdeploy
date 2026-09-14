# Copyright (c) OpenMMLab. All rights reserved.
"""CPU regressions for MTP routing; transport/CUDA operations are mocked."""
import pickle
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pybase64
import pytest
import torch
from torch import nn


def _make_agent(monkeypatch, method='deepseek_mtp', enabled=True):
    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

    agent = BaseModelAgent.__new__(BaseModelAgent)
    config = SimpleNamespace(num_hidden_layers=5, num_nextn_predict_layers=2,
                             qk_rope_head_dim=2, kv_lora_rank=2, qk_nope_head_dim=2,
                             n_routed_experts=0, tie_word_embeddings=False)
    agent.spec_agent = SimpleNamespace(is_enabled=lambda: enabled, method=method,
                                       model_config=SimpleNamespace(hf_config=config, num_layers=2),
                                       proposer=None, get_model=lambda: None)
    agent.all_context = nullcontext
    agent.dist_ctx = SimpleNamespace(tp_group=SimpleNamespace(rank=1))
    agent._update_params_ipc_tensor = None
    agent._update_params_ipc_event = None
    agent._model_update_group = {'trainer': object()}
    agent.reset_graph_runner = Mock()
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    return agent, config


@pytest.mark.parametrize('method,enabled,expected', [
    ('deepseek_mtp', True, [2, 3]),
    ('qwen3_5_mtp', True, [6]),
    ('deepseek_mtp', False, []),
    ('eagle', True, []),
])
def test_split_update_weights_without_local_proposer(monkeypatch, method, enabled, expected):
    agent, _ = _make_agent(monkeypatch, method, enabled)
    names = ['model.embed_tokens.weight', 'model.layers.4.input_layernorm.weight',
             'model.layers.5.eh_proj.weight', 'model.layers.6.shared_head.norm.weight',
             'model.layers.50.eh_proj.weight', 'model.layers.7.eh_proj.weight', 'mtp.fc.weight',
             'lm_head.weight']
    weights = [(name, torch.zeros(2)) for name in names]
    main, draft = agent._split_updated_weights(weights)
    assert [name for name, _ in draft] == [names[i] for i in expected]
    assert [name for name, _ in main] == [name for i, name in enumerate(names) if i not in expected]
    assert all(tensor is dict(weights)[name] for name, tensor in main + draft)


def _register_parameter(model, name, dtype):
    module = model
    *parts, leaf = name.split('.')
    for part in parts:
        if part not in module._modules:
            module.add_module(part, nn.Module())
        module = module._modules[part]
    module.register_parameter(leaf, nn.Parameter(torch.zeros(2, 2, dtype=dtype), requires_grad=False))


def _make_glm_models(config, dtype, events):
    from lmdeploy.pytorch.models.glm_moe_dsa import GlmMoeDsaForCausalLM
    from lmdeploy.pytorch.models.glm_moe_dsa_mtp import GlmMoeDsaMTPModel

    # Keep the actual GLM loaders and HF -> mtp_block mapping, without building
    # GPU attention/MoE kernels or allocating a full GLM model.
    main = GlmMoeDsaForCausalLM.__new__(GlmMoeDsaForCausalLM)
    draft = GlmMoeDsaMTPModel.__new__(GlmMoeDsaMTPModel)
    main_names = ['model.embed_tokens.weight', 'lm_head.weight', 'model.layers.0.input_layernorm.weight']
    draft_names = [f'model.layers.{idx}.{suffix}' for idx in (5, 6) for suffix in (
        'enorm.weight', 'hnorm.weight', 'eh_proj.weight', 'shared_head.norm.weight',
        'input_layernorm.weight', 'self_attn.o_proj.weight', 'mlp.gate.weight')]
    if dtype == torch.float8_e4m3fn:
        draft_names.append('model.layers.5.eh_proj.weight_scale_inv')
    for model, names, tag in [(main, main_names, 'main'), (draft, draft_names, 'draft')]:
        nn.Module.__init__(model)
        model.config = config
        model.quantization_config = None
        model._load_buffers = {}
        for name in names:
            runtime_name = name
            if model is draft:
                runtime_name = draft._rewrite_spec_layer_name(int(name.split('.')[2]), name)
            param_dtype = torch.float32 if name.endswith('weight_scale_inv') else dtype
            _register_parameter(model, runtime_name, param_dtype)
        model.update_weights = lambda tag=tag: events.append(f'finalize-{tag}')
        load = model.load_weights

        def load_weights(weights, load=load, tag=tag):
            events.append(f'load-{tag}')
            load(weights)

        model.load_weights = load_weights
    # GLM uses the live target embedding and target logits head, not copies.
    draft.model.embed_tokens = main.model.embed_tokens
    return main, draft, main_names, draft_names


@pytest.mark.parametrize('transport', ['ipc', 'nccl'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
def test_glm_mtp_repeated_weight_updates(monkeypatch, transport, dtype):
    from lmdeploy.serve.openai.protocol import UpdateParamsRequest, UpdateWeightsFromDistributedRequest
    from lmdeploy.utils import FlattenedTensorBucket

    agent, config = _make_agent(monkeypatch)
    events = []
    main, draft, main_names, draft_names = _make_glm_models(config, dtype, events)
    agent.patched_model = SimpleNamespace(get_model=lambda: main)
    agent.spec_agent.get_model = lambda: draft
    initial_ptrs = {name: param.data_ptr() for name, param in draft.named_parameters()}

    for version in (1, 2):
        events.clear()
        agent._update_params_ipc_event = SimpleNamespace(wait=lambda: events.append('wait'),
                                                        record=lambda: events.append('record'))
        # One mixed main/draft bucket, then draft-only and (FP8) scale-only
        # buckets. Reuse the IPC buffer for same-dtype buckets as XTuner does.
        groups = [main_names + draft_names[:2], [n for n in draft_names[2:] if not n.endswith('weight_scale_inv')]]
        if dtype == torch.float8_e4m3fn:
            groups.append([draft_names[-1]])
        groups.append([])  # Empty finished=True control request is required.
        for names in groups:
            tensors = [(name, torch.full((2, 2), version,
                                         dtype=torch.float32 if name.endswith('weight_scale_inv') else dtype))
                       for name in names]
            bucket = FlattenedTensorBucket(named_tensors=tensors)
            if transport == 'ipc':
                if names:
                    flat = bucket.get_flattened_tensor()
                    cached = agent._update_params_ipc_tensor
                    if cached is None or cached.dtype != flat.dtype:
                        cached = torch.empty(256, dtype=flat.dtype)
                        agent._update_params_ipc_tensor = cached
                    cached[:flat.numel()].copy_(flat)
                payload = pybase64.b64encode(pickle.dumps({'metadata': bucket.metadata})).decode()
                agent.update_params(UpdateParamsRequest(serialized_named_tensors=['unused-rank-0', payload],
                                                        load_format='flattened_bucket', finished=not names))
            else:
                monkeypatch.setattr(torch.cuda, 'current_device', lambda: 'cpu')
                monkeypatch.setattr(torch.distributed, 'broadcast',
                                    lambda tensor, **kwargs: tensor.copy_(bucket.get_flattened_tensor()))
                ok, message = agent.update_weights_from_distributed(UpdateWeightsFromDistributedRequest(
                    names=names, dtypes=[str(t.dtype).removeprefix('torch.') for _, t in tensors],
                    shapes=[[2, 2]] * len(names), group_name='trainer',
                    load_format='flattened_bucket', finished=not names))
                assert ok, message

        for model in (main, draft):
            for name, param in model.named_parameters():
                assert torch.equal(param.float(), torch.full((2, 2), float(version))), name
        assert {name: param.data_ptr() for name, param in draft.named_parameters()} == initial_ptrs
        assert draft.model.embed_tokens is main.model.embed_tokens
        assert events.count('finalize-main') == events.count('finalize-draft') == 1
        assert events.count('load-main') == 1
        assert events.count('load-draft') == len(groups) - 1
        if transport == 'ipc':
            # Each load is followed by the existing consumer acknowledgement;
            # draft-only buckets must also acknowledge buffer consumption.
            for i, event in enumerate(events):
                if event.startswith('load-'):
                    assert events[i + 1] == 'record'
            assert agent._update_params_ipc_tensor is None
            assert agent._update_params_ipc_event is None
        else:
            assert agent.reset_graph_runner.call_count == version
