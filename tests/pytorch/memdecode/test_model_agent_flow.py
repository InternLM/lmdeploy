import asyncio
from types import SimpleNamespace

import torch

from lmdeploy.pytorch.config import MemDecodeConfig, ModelConfig
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from lmdeploy.pytorch.memdecode import MemDecodeFusion


def _model_config():
    return ModelConfig(
        hidden_size=2,
        num_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=[2],
        head_dim=2,
        vocab_size=3,
        hf_config=SimpleNamespace(vocab_size=3),
    )


def _memdecode_config():
    return MemDecodeConfig(
        memory_model_path='memory-model',
        memory_model_config=_model_config(),
        lambda_value=0.0,
    )


class _FakeBaseModel:

    def __init__(self, calls):
        self.calls = calls
        self.hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        self.weight = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])

    async def async_forward(self, inputs):
        self.calls.append(('base_forward', inputs))
        return {'hidden_states': self.hidden_states.clone(), 'seq_length': inputs.seq_length}

    def get_logits(self, hidden_states):
        self.calls.append(('base_logits_shape', tuple(hidden_states.shape)))
        torch.testing.assert_close(hidden_states, self.hidden_states[:, -1:])
        return hidden_states @ self.weight


class _FakeMemoryAgent:

    def __init__(self, calls):
        self.calls = calls
        self.hidden_states = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
        self.weight = torch.tensor([[2.0, -1.0, 0.5], [1.0, 3.0, -0.5]])

    def is_enabled(self):
        return True

    async def async_forward(self, inputs):
        self.calls.append(('memory_forward', inputs))
        return {'hidden_states': self.hidden_states.clone(), 'seq_length': inputs.seq_length}

    def get_logits(self, hidden_states):
        self.calls.append(('memory_logits_shape', tuple(hidden_states.shape)))
        torch.testing.assert_close(hidden_states, self.hidden_states[:, -1:])
        return hidden_states @ self.weight


def test_async_model_forward_memdecode_fixed_fusion_uses_sliced_logits():
    calls = []
    inputs = SimpleNamespace(seq_length=torch.tensor([2]), is_chunk=False)
    base_model = _FakeBaseModel(calls)
    memory_agent = _FakeMemoryAgent(calls)
    config = _model_config()

    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.patched_model = base_model
    agent.async_forward = base_model.async_forward
    agent.memdecode_agent = memory_agent
    agent.agent_strategy = SimpleNamespace(slice_outputs=lambda hidden, seq_length: hidden[-1:])
    agent.memdecode_fusion = MemDecodeFusion(
        _memdecode_config(),
        base_hidden_size=config.hidden_size,
        memory_hidden_size=config.hidden_size,
        base_vocab_size=config.vocab_size,
    )

    output = asyncio.run(agent._async_model_forward(inputs, return_logits=False))

    assert calls == [
        ('base_forward', inputs),
        ('base_logits_shape', (1, 1, 2)),
        ('memory_forward', inputs),
        ('memory_logits_shape', (1, 1, 2)),
    ]
    assert output['logits'].shape == (1, 1, 3)
    expected_base_logits = base_model.hidden_states[:, -1:] @ base_model.weight
    torch.testing.assert_close(output['logits'], torch.log_softmax(expected_base_logits, dim=-1))
