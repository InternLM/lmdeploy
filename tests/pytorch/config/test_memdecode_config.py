from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import DistConfig, MemDecodeConfig, MiscConfig, ModelConfig

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

    assert memdecode_config.lambda_value == 0.25
    assert memdecode_config.memory_model_config is memory_config
    base_config = _model_config_from_hf(_hf_config(), 'base')
    assert not hasattr(base_config, 'memdecode_config')
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


def test_misc_config_can_own_memdecode_config():
    memory_config = _model_config_from_hf(_hf_config(), 'memory')
    memdecode_config = MemDecodeConfig(memory_model_path='memory', memory_model_config=memory_config)

    misc_config = MiscConfig(memdecode_config=memdecode_config)

    assert misc_config.memdecode_config is memdecode_config


def test_model_config_from_pretrained_does_not_build_memdecode_config(monkeypatch):
    returned_hf_configs = {}

    def fake_config_from_pretrained(model_path, trust_remote_code=False):
        assert model_path == 'base'
        returned_hf_configs[model_path] = _hf_config()
        return returned_hf_configs[model_path]

    monkeypatch.setattr('lmdeploy.pytorch.transformers.config_from_pretrained', fake_config_from_pretrained)

    model_config = ModelConfig.from_pretrained('base')

    for field_name in _OLD_MEMDECODE_FIELDS:
        assert not hasattr(model_config, field_name)

    assert not hasattr(model_config, 'memdecode_config')
    assert model_config.hf_config is returned_hf_configs['base']


def test_model_config_has_no_memdecode_validation_method():
    base_config = _model_config_from_hf(_hf_config(), 'base', states_shapes=[((1, 2), torch.float16)])

    assert not hasattr(base_config, 'validate_memdecode_config')
