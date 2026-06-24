import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

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
    agent.build_model(empty_init=True)
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
