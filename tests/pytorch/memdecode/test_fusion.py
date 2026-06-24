from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import MemDecodeConfig, ModelConfig
from lmdeploy.pytorch.memdecode import MemDecodeFusion, align_logits_to_base


def _model_config(vocab_size=4, hidden_size=8):
    return ModelConfig(
        hidden_size=hidden_size,
        num_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=[2],
        head_dim=hidden_size,
        vocab_size=vocab_size,
        hf_config=SimpleNamespace(vocab_size=vocab_size),
    )


def _memdecode_config(lambda_value=0.5, adaptive_router=False, router_path=None):
    return MemDecodeConfig(
        memory_model_path='memory',
        memory_model_config=_model_config(),
        lambda_value=lambda_value,
        adaptive_router=adaptive_router,
        router_path=router_path,
    )


def test_align_logits_to_base_truncates_larger_vocab():
    logits = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    aligned = align_logits_to_base(logits, base_vocab_size=3)

    torch.testing.assert_close(aligned, logits[:, :3])


def test_align_logits_to_base_pads_smaller_vocab_with_negative_infinity():
    logits = torch.tensor([[1.0, 2.0]], dtype=torch.float16)

    aligned = align_logits_to_base(logits, base_vocab_size=4)

    assert aligned.dtype is logits.dtype
    assert aligned.device == logits.device
    torch.testing.assert_close(aligned[:, :2], logits)
    assert torch.isneginf(aligned[:, 2:]).all()


def test_fixed_lambda_zero_returns_base_log_softmax_and_no_routing_info():
    fusion = MemDecodeFusion(_model_config(), _memdecode_config(lambda_value=0.0))
    base_logits = torch.tensor([[1.0, 3.0, -1.0, 0.5]])
    memory_logits = torch.tensor([[4.0, -2.0, 8.0, 1.0]])

    fused, routing_info = fusion(base_logits, memory_logits)

    torch.testing.assert_close(fused, torch.log_softmax(base_logits, dim=-1))
    assert routing_info is None


def test_fixed_lambda_one_returns_memory_log_softmax_and_no_routing_info():
    fusion = MemDecodeFusion(_model_config(), _memdecode_config(lambda_value=1.0))
    base_logits = torch.tensor([[1.0, 3.0, -1.0, 0.5]])
    memory_logits = torch.tensor([[4.0, -2.0, 8.0, 1.0]])

    fused, routing_info = fusion(base_logits, memory_logits)

    torch.testing.assert_close(fused, torch.log_softmax(memory_logits, dim=-1))
    assert routing_info is None


def test_fixed_intermediate_lambda_combines_log_probabilities():
    fusion = MemDecodeFusion(_model_config(), _memdecode_config(lambda_value=0.25))
    base_logits = torch.tensor([[1.0, 3.0, -1.0, 0.5]])
    memory_logits = torch.tensor([[4.0, -2.0, 8.0, 1.0]])

    fused, routing_info = fusion(base_logits, memory_logits)

    expected = torch.logaddexp(
        torch.log_softmax(base_logits, dim=-1) + torch.log(torch.tensor(0.75)),
        torch.log_softmax(memory_logits, dim=-1) + torch.log(torch.tensor(0.25)),
    )
    torch.testing.assert_close(fused, expected)
    assert routing_info is None


def test_adaptive_router_requires_hidden_states(tmp_path):
    router_path = tmp_path / 'router.pt'
    torch.save({'config': {'hidden_size': 8}, 'state_dict': {}}, router_path)
    fusion = MemDecodeFusion(_model_config(), _memdecode_config(adaptive_router=True, router_path=str(router_path)))
    base_logits = torch.randn(2, 4)
    memory_logits = torch.randn(2, 4)

    with pytest.raises(ValueError, match='hidden_states are required'):
        fusion(base_logits, memory_logits)
