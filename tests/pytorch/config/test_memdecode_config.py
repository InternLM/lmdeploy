from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import DistConfig, MemDecodeConfig, ModelConfig

_OLD_MEMDECODE_FIELDS = (
    'base_model_path',
    'memory_model_path',
    'memory_model_config',
    'lambda_value',
    'adaptive_router',
    'router_path',
    'lambda_base_only_threshold',
)
_FUSION_POLICY_FIELDS = (
    'lambda_value',
    'adaptive_router',
    'router_path',
    'lambda_base_only_threshold',
)


def _hf_config(vocab_size=32000):
    return SimpleNamespace(
        architectures=['Qwen3ForCausalLM'],
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=1024,
        model_type='qwen3',
        num_attention_heads=8,
        num_hidden_layers=2,
        num_key_value_heads=2,
        torch_dtype='float16',
        vocab_size=vocab_size,
    )


def _model_config_from_hf(hf_config, model_path, tp=1, states_shapes=None):
    return ModelConfig(
        hidden_size=hf_config.hidden_size,
        num_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        bos_token_id=hf_config.bos_token_id,
        eos_token_id=[hf_config.eos_token_id],
        head_dim=128,
        vocab_size=hf_config.vocab_size,
        hf_config=hf_config,
        dist_config=DistConfig(tp=tp),
        states_shapes=list(states_shapes or []),
    )


def test_memdecode_config_dataclass_keeps_fusion_policy_off_model_config():
    memory_config = _model_config_from_hf(_hf_config(), 'memory')
    memdecode_config = MemDecodeConfig(
        memory_model_path='memory',
        memory_model_config=memory_config,
        lambda_value=0.25,
        adaptive_router=True,
        router_path='router.pt',
        lambda_base_only_threshold=0.1,
    )

    base_config = _model_config_from_hf(_hf_config(), 'base')
    base_config.memdecode_config = memdecode_config

    assert base_config.memdecode_config.lambda_value == 0.25
    assert not hasattr(base_config, 'lambda_value')
    assert not hasattr(memory_config, 'lambda_value')


def test_memdecode_config_validates_lambda_range():
    memory_config = _model_config_from_hf(_hf_config(), 'memory')

    with pytest.raises(ValueError, match='lambda_value must be in \\[0, 1\\]'):
        MemDecodeConfig(memory_model_path='memory', memory_model_config=memory_config, lambda_value=1.25)


def test_memdecode_config_requires_router_path_for_adaptive_router():
    memory_config = _model_config_from_hf(_hf_config(), 'memory')

    with pytest.raises(ValueError, match='router_path is required when adaptive_router is enabled'):
        MemDecodeConfig(memory_model_path='memory', memory_model_config=memory_config, adaptive_router=True)


def test_from_pretrained_normalizes_flat_memdecode_overrides(monkeypatch):
    returned_hf_configs = {}

    def fake_config_from_pretrained(model_path, trust_remote_code=False):
        assert model_path in {'base', 'memory'}
        returned_hf_configs[model_path] = _hf_config()
        return returned_hf_configs[model_path]

    monkeypatch.setattr('lmdeploy.pytorch.transformers.config_from_pretrained', fake_config_from_pretrained)

    model_config = ModelConfig.from_pretrained(
        'base',
        hf_overrides={
            'memory_model_path': 'memory',
            'lambda_value': 0.25,
            'adaptive_router': True,
            'router_path': 'router.pt',
            'lambda_base_only_threshold': 0.1,
        },
    )

    memdecode_config = model_config.memdecode_config
    assert memdecode_config is not None
    assert memdecode_config.memory_model_path == 'memory'
    assert memdecode_config.lambda_value == 0.25
    assert memdecode_config.adaptive_router is True
    assert memdecode_config.router_path == 'router.pt'
    assert memdecode_config.lambda_base_only_threshold == 0.1
    assert memdecode_config.memory_model_config.hf_config is returned_hf_configs['memory']

    for field_name in _OLD_MEMDECODE_FIELDS:
        assert not hasattr(model_config, field_name)

    for field_name in _FUSION_POLICY_FIELDS:
        assert not hasattr(model_config.hf_config, field_name)
        assert not hasattr(memdecode_config.memory_model_config, field_name)


def test_validate_memdecode_config_rejects_ssm_mismatch():
    base_config = _model_config_from_hf(_hf_config(), 'base', states_shapes=[((1, 2), torch.float16)])
    memory_config = _model_config_from_hf(_hf_config(), 'memory')
    base_config.memdecode_config = MemDecodeConfig(
        memory_model_path='memory',
        memory_model_config=memory_config,
    )

    with pytest.raises(ValueError, match='Base and memory model must both use SSM state caches'):
        base_config.validate_memdecode_config()
