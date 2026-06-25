import json
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import MemDecodeConfig, ModelConfig
from lmdeploy.pytorch.memdecode import MemDecodeFusion, align_logits_to_base
from lmdeploy.pytorch.memdecode.fusion import RouterNetwork


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


def _memdecode_config(lambda_value=0.5, adaptive_router=False, router_path=None, threshold=-1.0):
    return MemDecodeConfig(
        memory_model_path='memory',
        memory_model_config=_model_config(),
        lambda_value=lambda_value,
        adaptive_router=adaptive_router,
        router_path=router_path,
        lambda_base_only_threshold=threshold,
    )


def _fusion(memdecode_config, base_hidden_size=8, memory_hidden_size=8, base_vocab_size=4):
    return MemDecodeFusion(
        memdecode_config,
        base_hidden_size=base_hidden_size,
        memory_hidden_size=memory_hidden_size,
        base_vocab_size=base_vocab_size,
    )


def _write_router_checkpoint(router_dir, router_config, output_bias):
    router_dir.mkdir()
    (router_dir / 'router_config.json').write_text(json.dumps(router_config))
    router = RouterNetwork(router_config, base_hidden_size=2, memory_hidden_size=3)
    state_dict = router.state_dict()
    for name, value in state_dict.items():
        state_dict[name] = torch.zeros_like(value)
        if name.endswith('bias') and value.shape == torch.Size([2]):
            state_dict[name] = torch.tensor(output_bias, dtype=value.dtype)
    torch.save({'state_dict': state_dict}, router_dir / 'router_10.pt')
    torch.save({'state_dict': state_dict}, router_dir / 'router_2.pt')
    return router_dir


def test_align_logits_to_base_returns_same_tensor_when_vocab_matches():
    logits = torch.randn(2, 4)

    aligned = align_logits_to_base(logits, base_vocab_size=4)

    assert aligned is logits


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


def test_fixed_lambda_zero_returns_base_log_softmax():
    fusion = _fusion(_memdecode_config(lambda_value=0.0))
    base_logits = torch.tensor([[1.0, 3.0, -1.0, 0.5]])
    memory_logits = torch.tensor([[4.0, -2.0, 8.0, 1.0]])

    fused = fusion(base_logits, memory_logits)

    torch.testing.assert_close(fused, torch.log_softmax(base_logits, dim=-1))


def test_fixed_lambda_one_returns_memory_log_softmax():
    fusion = _fusion(_memdecode_config(lambda_value=1.0))
    base_logits = torch.tensor([[1.0, 3.0, -1.0, 0.5]])
    memory_logits = torch.tensor([[4.0, -2.0, 8.0, 1.0]])

    fused = fusion(base_logits, memory_logits)

    torch.testing.assert_close(fused, torch.log_softmax(memory_logits, dim=-1))


def test_fixed_intermediate_lambda_combines_log_probabilities():
    fusion = _fusion(_memdecode_config(lambda_value=0.25))
    base_logits = torch.tensor([[1.0, 3.0, -1.0, 0.5]])
    memory_logits = torch.tensor([[4.0, -2.0, 8.0, 1.0]])

    fused = fusion(base_logits, memory_logits)

    expected = torch.logaddexp(
        torch.log_softmax(base_logits, dim=-1) + torch.log(torch.tensor(0.75)),
        torch.log_softmax(memory_logits, dim=-1) + torch.log(torch.tensor(0.25)),
    )
    torch.testing.assert_close(fused, expected)


def test_fixed_fusion_aligns_mismatched_vocab_sizes_before_fusion():
    fusion = _fusion(_memdecode_config(lambda_value=0.25), base_vocab_size=4)
    base_logits = torch.tensor([[1.0, 3.0, -1.0, 0.5, 100.0]])
    memory_logits = torch.tensor([[4.0, -2.0]])

    fused = fusion(base_logits, memory_logits)

    aligned_base = base_logits[:, :4]
    aligned_memory = torch.tensor([[4.0, -2.0, -torch.inf, -torch.inf]])
    expected = torch.logaddexp(
        torch.log_softmax(aligned_base, dim=-1) + torch.log(torch.tensor(0.75)),
        torch.log_softmax(aligned_memory, dim=-1) + torch.log(torch.tensor(0.25)),
    )
    torch.testing.assert_close(fused, expected)


def test_adaptive_router_requires_base_and_memory_hidden_states(tmp_path):
    router_config = {'num_layers': 1, 'input_mode': 'both', 'use_scalars': False, 'hidden_dim': 4, 'dropout': 0.0}
    router_dir = _write_router_checkpoint(tmp_path / 'router', router_config, output_bias=[0.0, 0.0])
    fusion = _fusion(
        _memdecode_config(adaptive_router=True, router_path=str(router_dir)),
        base_hidden_size=2,
        memory_hidden_size=3,
    )

    with pytest.raises(ValueError, match='base_hidden_states and memory_hidden_states are required'):
        fusion(torch.randn(2, 4), torch.randn(2, 4), base_hidden_states=torch.randn(2, 2))


def test_adaptive_router_loads_state_dict_and_fuses(tmp_path):
    router_config = {'num_layers': 1, 'input_mode': 'both', 'use_scalars': False, 'hidden_dim': 4, 'dropout': 0.0}
    router_dir = _write_router_checkpoint(tmp_path / 'router', router_config, output_bias=[-2.0, 2.0])
    fusion = _fusion(
        _memdecode_config(adaptive_router=True, router_path=str(router_dir)),
        base_hidden_size=2,
        memory_hidden_size=3,
    )
    base_logits = torch.tensor([[1.0, 0.0, -1.0, -2.0]])
    memory_logits = torch.tensor([[-2.0, -1.0, 0.0, 1.0]])

    fused = fusion(
        base_logits,
        memory_logits,
        base_hidden_states=torch.randn(1, 2),
        memory_hidden_states=torch.randn(1, 3),
    )

    log_weights = torch.log_softmax(torch.tensor([[-2.0, 2.0]]), dim=-1)
    expected = torch.logaddexp(
        torch.log_softmax(base_logits, dim=-1) + log_weights[:, 0:1],
        torch.log_softmax(memory_logits, dim=-1) + log_weights[:, 1:2],
    )
    torch.testing.assert_close(fused, expected)


def test_adaptive_router_loads_planned_scalar_projector_checkpoint(tmp_path):
    router_config = {
        'num_layers': 2,
        'input_mode': 'both',
        'use_scalars': True,
        'scalar_proj_dim': 2,
        'hidden_dim': 4,
        'dropout': 0.0,
    }
    router_dir = tmp_path / 'router'
    router_dir.mkdir()
    (router_dir / 'router_config.json').write_text(json.dumps(router_config))
    state_dict = {
        f'scalar_projectors.{idx}.0.weight': torch.zeros(2, 1)
        for idx in range(4)
    }
    state_dict.update({
        f'scalar_projectors.{idx}.0.bias': torch.zeros(2)
        for idx in range(4)
    })
    state_dict.update({
        'mlp.0.weight': torch.zeros(4, 13),
        'mlp.0.bias': torch.zeros(4),
        'mlp.3.weight': torch.zeros(2, 4),
        'mlp.3.bias': torch.tensor([-1.0, 1.0]),
    })
    torch.save(state_dict, router_dir / 'router_1.pt')
    fusion = _fusion(
        _memdecode_config(adaptive_router=True, router_path=str(router_dir)),
        base_hidden_size=2,
        memory_hidden_size=3,
    )

    base_logits = torch.tensor([[1.0, 0.0, -1.0, -2.0]])
    memory_logits = torch.tensor([[-2.0, -1.0, 0.0, 1.0]])

    fused = fusion(
        base_logits,
        memory_logits,
        base_hidden_states=torch.randn(1, 2),
        memory_hidden_states=torch.randn(1, 3),
    )

    log_weights = torch.log_softmax(torch.tensor([[-1.0, 1.0]]), dim=-1)
    expected = torch.logaddexp(
        torch.log_softmax(base_logits, dim=-1) + log_weights[:, 0:1],
        torch.log_softmax(memory_logits, dim=-1) + log_weights[:, 1:2],
    )
    torch.testing.assert_close(fused, expected)


def test_lambda_base_only_threshold_gates_to_base_only(tmp_path):
    router_config = {
        'num_layers': 1,
        'input_mode': 'memory_only',
        'use_scalars': False,
        'hidden_dim': 4,
        'dropout': 0.0,
    }
    router_path = _write_router_checkpoint(tmp_path / 'router', router_config, output_bias=[2.0, -2.0])
    fusion = _fusion(
        _memdecode_config(adaptive_router=True, router_path=str(router_path), threshold=0.1),
        base_hidden_size=2,
        memory_hidden_size=3,
    )
    base_logits = torch.tensor([[1.0, 0.0, -1.0, -2.0]])
    memory_logits = torch.tensor([[-2.0, -1.0, 0.0, 1.0]])

    fused = fusion(
        base_logits,
        memory_logits,
        base_hidden_states=torch.randn(1, 2),
        memory_hidden_states=torch.randn(1, 3),
    )

    torch.testing.assert_close(fused, torch.log_softmax(base_logits, dim=-1))


def test_adaptive_scalar_features_ignore_padded_negative_infinity(tmp_path):
    router_config = {
        'num_layers': 1,
        'input_mode': 'mem_hidden_both_scalars',
        'use_scalars': True,
        'scalar_proj_dim': 0,
        'hidden_dim': 4,
        'dropout': 0.0,
    }
    router_path = _write_router_checkpoint(tmp_path / 'router', router_config, output_bias=[0.5, -0.5])
    fusion = _fusion(
        _memdecode_config(adaptive_router=True, router_path=str(router_path)),
        base_hidden_size=2,
        memory_hidden_size=3,
        base_vocab_size=4,
    )
    base_logits = torch.tensor([[1.0, 0.0, -1.0, -2.0]])
    memory_logits = torch.tensor([[2.0, -2.0]])

    fused = fusion(
        base_logits,
        memory_logits,
        base_hidden_states=torch.randn(1, 2),
        memory_hidden_states=torch.randn(1, 3),
    )

    log_weights = torch.log_softmax(torch.tensor([[0.5, -0.5]]), dim=-1)
    aligned_memory = torch.tensor([[2.0, -2.0, -torch.inf, -torch.inf]])
    expected = torch.logaddexp(
        torch.log_softmax(base_logits, dim=-1) + log_weights[:, 0:1],
        torch.log_softmax(aligned_memory, dim=-1) + log_weights[:, 1:2],
    )
    torch.testing.assert_close(fused, expected)


def test_mem_hidden_both_scalars_uses_implicit_scalar_architecture(tmp_path):
    router_config = {
        'num_layers': 1,
        'input_mode': 'mem_hidden_both_scalars',
        'scalar_proj_dim': 0,
        'hidden_dim': 4,
        'dropout': 0.0,
    }
    router_path = _write_router_checkpoint(tmp_path / 'router', router_config, output_bias=[-0.25, 0.25])
    fusion = _fusion(
        _memdecode_config(adaptive_router=True, router_path=str(router_path)),
        base_hidden_size=2,
        memory_hidden_size=3,
    )

    base_logits = torch.tensor([[1.0, 0.0, -1.0, -2.0]])
    memory_logits = torch.tensor([[-2.0, -1.0, 0.0, 1.0]])

    fused = fusion(
        base_logits,
        memory_logits,
        base_hidden_states=torch.randn(1, 2),
        memory_hidden_states=torch.randn(1, 3),
    )

    log_weights = torch.log_softmax(torch.tensor([[-0.25, 0.25]]), dim=-1)
    expected = torch.logaddexp(
        torch.log_softmax(base_logits, dim=-1) + log_weights[:, 0:1],
        torch.log_softmax(memory_logits, dim=-1) + log_weights[:, 1:2],
    )
    torch.testing.assert_close(fused, expected)
    assert torch.isfinite(fused).all()


def test_memory_only_scalar_router_ignores_base_logits(tmp_path):
    router_config = {
        'num_layers': 1,
        'input_mode': 'memory_only',
        'use_scalars': True,
        'scalar_proj_dim': 0,
        'hidden_dim': 4,
        'dropout': 0.0,
    }
    router_dir = tmp_path / 'router'
    router_dir.mkdir()
    (router_dir / 'router_config.json').write_text(json.dumps(router_config))
    router = RouterNetwork(router_config, base_hidden_size=2, memory_hidden_size=3)
    state_dict = router.state_dict()
    for name, value in state_dict.items():
        state_dict[name] = torch.zeros_like(value)
    state_dict['mlp.0.weight'][0, 3] = 3.0
    state_dict['mlp.0.weight'][1, 4] = -2.0
    torch.save({'state_dict': state_dict}, router_dir / 'router_1.pt')
    fusion = _fusion(
        _memdecode_config(adaptive_router=True, router_path=str(router_dir)),
        base_hidden_size=2,
        memory_hidden_size=3,
    )
    original_router = fusion.router

    class _RecordingRouter(torch.nn.Module):

        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.input_mode = wrapped.input_mode
            self.scalar_features = []

        def forward(self, base_hidden_states, memory_hidden_states, scalar_features):
            self.scalar_features.append(scalar_features.detach().clone())
            return self.wrapped(base_hidden_states, memory_hidden_states, scalar_features)

    recording_router = _RecordingRouter(original_router)
    fusion.router = recording_router
    memory_logits = torch.tensor([[1.5, 0.0, -1.0, -2.0]])
    base_hidden_states = torch.randn(1, 2)
    memory_hidden_states = torch.randn(1, 3)

    fused_a = fusion(
        torch.tensor([[50.0, -50.0, -50.0, -50.0]]),
        memory_logits,
        base_hidden_states=base_hidden_states,
        memory_hidden_states=memory_hidden_states,
    )
    fused_b = fusion(
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        memory_logits,
        base_hidden_states=base_hidden_states,
        memory_hidden_states=memory_hidden_states,
    )

    assert torch.isfinite(fused_a).all()
    assert torch.isfinite(fused_b).all()
    torch.testing.assert_close(recording_router.scalar_features[0], recording_router.scalar_features[1])


def test_router_checkpoint_without_state_dict_fails(tmp_path):
    router_path = tmp_path / 'router.pt'
    torch.save({'config': {'hidden_dim': 4}}, router_path)

    with pytest.raises(ValueError, match='state dict'):
        _fusion(
            _memdecode_config(adaptive_router=True, router_path=str(router_path)),
            base_hidden_size=2,
            memory_hidden_size=3,
        )
