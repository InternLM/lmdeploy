import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import torch

import lmdeploy.pytorch.memdecode.agent as agent_module
from lmdeploy.pytorch.config import BackendConfig, DistConfig, MemDecodeConfig, ModelConfig
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.pytorch.memdecode.agent import MemDecodeAgent, build_memdecode_agent


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


def _dist_ctx():
    return DistContext.build(rank=0, dist_config=DistConfig())


def _memdecode_config(states_shapes=None):
    return MemDecodeConfig(
        memory_model_path='memory-model',
        memory_model_config=_model_config(states_shapes=states_shapes),
    )


def test_build_memdecode_agent_returns_none_for_missing_config():
    agent = build_memdecode_agent(None, BackendConfig(), _dist_ctx())

    assert agent is None


def test_build_memdecode_agent_returns_enabled_agent_for_config():
    agent = build_memdecode_agent(_memdecode_config(), BackendConfig(), _dist_ctx())

    assert isinstance(agent, MemDecodeAgent)


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


def test_fuse_with_base_runs_memory_forward_and_fusion():
    calls = []
    inputs = SimpleNamespace(seq_length=torch.tensor([2]))
    base_hidden = torch.tensor([[[1.0, 2.0]]])
    base_logits = torch.tensor([[[3.0, 4.0]]])
    memory_hidden = torch.tensor([[[5.0, 6.0]]])
    memory_logits = torch.tensor([[[7.0, 8.0]]])
    fused_logits = torch.tensor([[[10.0, 12.0]]])
    base_output = {'hidden_states': base_hidden, 'seq_length': inputs.seq_length}
    memory_output = {'hidden_states': memory_hidden, 'seq_length': inputs.seq_length}

    class _Fusion:

        def __call__(self, **kwargs):
            calls.append(('fusion', kwargs))
            return fused_logits

    async def _memory_forward(forward_inputs):
        calls.append(('memory_forward', forward_inputs))
        return memory_output

    def _postprocess(output, forward_inputs):
        calls.append(('postprocess', output, forward_inputs))
        return output

    agent = MemDecodeAgent.__new__(MemDecodeAgent)
    agent.async_forward = _memory_forward
    agent.model = SimpleNamespace(
        get_logits=lambda hidden_states: memory_logits if hidden_states is memory_hidden else None)
    agent.fusion = _Fusion()

    output = asyncio.run(
        agent.fuse_with_base(
            inputs=inputs,
            base_output=base_output,
            base_logits=base_logits,
            postprocess_output=_postprocess,
        )
    )

    assert output is base_output
    assert output['logits'] is fused_logits
    assert 'all_routed_experts' not in output
    assert calls[0] == ('memory_forward', inputs)
    assert calls[1] == ('postprocess', memory_output, inputs)
    fusion_kwargs = calls[2][1]
    assert fusion_kwargs == {
        'base_logits': base_logits,
        'memory_logits': memory_logits,
        'base_hidden_states': base_hidden,
        'memory_hidden_states': memory_hidden,
    }


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
