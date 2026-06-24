import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import lmdeploy.pytorch.memdecode.agent as agent_module
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, DistConfig, MemDecodeConfig, ModelConfig
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.memdecode.agent import BaseMemDecodeAgent, MemDecodeAgent, build_memdecode_agent


def _model_config(states_shapes=None):
    return ModelConfig(
        hidden_size=8,
        num_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=[2],
        head_dim=8,
        vocab_size=4,
        hf_config=SimpleNamespace(vocab_size=4),
        states_shapes=states_shapes or [],
    )


def _cache_config():
    return CacheConfig(max_batches=2, block_size=4, num_cpu_blocks=1, num_gpu_blocks=1)


def _dist_ctx():
    return DistContext.build(rank=0, dist_config=DistConfig())


def _memdecode_config(states_shapes=None):
    return MemDecodeConfig(
        memory_model_path='memory-model',
        memory_model_config=_model_config(states_shapes=states_shapes),
    )


def test_disabled_agent_is_noop_and_returns_no_model():
    agent = BaseMemDecodeAgent(None, BackendConfig(), _dist_ctx())
    agent.set_cache_config(_cache_config())
    agent.build_model(empty_init=True, build_model_ctx=object())
    agent.build_graph_runner()
    agent.build_cache_engine(cache_stream=None)
    agent.reset_graph_runner()
    agent.release()

    assert agent.is_enabled() is False
    assert agent.get_model() is None
    assert asyncio.run(agent.async_forward(SimpleNamespace())) is None
    with pytest.raises(RuntimeError, match='MemDecode is disabled'):
        agent.get_logits(torch.empty(1, 1))


def test_build_memdecode_agent_returns_disabled_agent_for_missing_config():
    agent = build_memdecode_agent(None, BackendConfig(), _dist_ctx())

    assert isinstance(agent, BaseMemDecodeAgent)
    assert not isinstance(agent, MemDecodeAgent)
    assert agent.is_enabled() is False


def test_build_memdecode_agent_returns_enabled_agent_for_config():
    agent = build_memdecode_agent(_memdecode_config(), BackendConfig(), _dist_ctx())

    assert isinstance(agent, MemDecodeAgent)
    assert agent.is_enabled() is True


def test_release_clears_model_cache_and_state_cache():
    agent = MemDecodeAgent.__new__(MemDecodeAgent)
    agent.model = object()
    agent.cache_engine = object()
    agent.state_cache_engine = object()

    agent.release()

    assert agent.model is None
    assert agent.cache_engine is None
    assert agent.state_cache_engine is None


def test_reset_graph_runner_runs_inside_memory_context_and_calls_model_reset():
    events = []

    class ResettableModel:

        def reset(self):
            events.append('reset')

    @contextmanager
    def memory_context():
        events.append('enter')
        yield
        events.append('exit')

    agent = MemDecodeAgent.__new__(MemDecodeAgent)
    agent.model = ResettableModel()
    agent.memory_context = memory_context

    agent.reset_graph_runner()

    assert events == ['enter', 'reset', 'exit']


def test_get_model_unwraps_graph_runner_when_available():
    raw_model = object()
    graph_runner = SimpleNamespace(get_model=lambda: raw_model)
    agent = MemDecodeAgent.__new__(MemDecodeAgent)
    agent.model = graph_runner

    assert agent.get_model() is raw_model


def test_get_logits_delegates_to_model():
    hidden_states = torch.randn(1, 2)
    logits = torch.randn(1, 4)
    model = SimpleNamespace(get_logits=lambda value: logits if value is hidden_states else None)
    agent = MemDecodeAgent.__new__(MemDecodeAgent)
    agent.model = model

    assert agent.get_logits(hidden_states) is logits


def test_async_forward_runs_memory_forward_inside_memory_context(monkeypatch):
    events = []
    inputs = SimpleNamespace()
    output = {'hidden_states': torch.empty(1, 1)}
    agent = MemDecodeAgent.__new__(MemDecodeAgent)
    agent.model = object()
    agent.model_config = object()
    agent.cache_engine = object()
    agent.state_cache_engine = object()

    @contextmanager
    def memory_context():
        events.append('enter')
        yield
        events.append('exit')

    def fake_memory_model_forward(model, forward_inputs, model_config, cache_engine, state_cache_engine=None):
        assert model is agent.model
        assert forward_inputs is inputs
        assert model_config is agent.model_config
        assert cache_engine is agent.cache_engine
        assert state_cache_engine is agent.state_cache_engine
        events.append('forward')
        return output

    agent.memory_context = memory_context
    monkeypatch.setattr(agent_module, 'memory_model_forward', fake_memory_model_forward)

    assert asyncio.run(agent.async_forward(inputs)) is output
    assert events == ['enter', 'forward', 'exit']


def test_build_model_honors_supplied_context_and_empty_init(monkeypatch):
    built_model = object()
    build_model_ctx = object()
    calls = []
    agent = MemDecodeAgent(_memdecode_config(), BackendConfig(), _dist_ctx(), device='cpu')

    def fake_build_patched_model(model_config, device=None, build_model_ctx=None):
        calls.append(('build', model_config, device, build_model_ctx))
        return built_model

    def fake_load_model_weights(model, model_path, device=None):
        calls.append(('load', model, model_path, device))

    monkeypatch.setattr(agent_module, 'build_patched_model', fake_build_patched_model)
    monkeypatch.setattr(agent_module, 'load_model_weights', fake_load_model_weights)

    agent.build_model(empty_init=True, build_model_ctx=build_model_ctx)

    assert agent.model is built_model
    assert calls == [('build', agent.model_config, 'cpu', build_model_ctx)]

    calls.clear()
    agent.build_model(empty_init=False, build_model_ctx=build_model_ctx)

    assert calls == [
        ('build', agent.model_config, 'cpu', build_model_ctx),
        ('load', built_model, 'memory-model', 'cpu'),
    ]
