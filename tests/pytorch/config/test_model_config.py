from types import SimpleNamespace

import pytest

from lmdeploy.pytorch.config import DistConfig, ModelConfig


def _make_model_config(num_attention_heads=32, num_key_value_heads=8, dist_config=None):
    return ModelConfig(
        hidden_size=4096,
        num_layers=1,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        bos_token_id=1,
        eos_token_id=[2],
        head_dim=128,
        dist_config=dist_config or DistConfig(),
    )


def test_get_num_qkv_head_by_tp_from_dist_config():
    model_config = _make_model_config(dist_config=DistConfig(tp=4))

    assert model_config.get_num_qkv_head_by_tp() == (8, 2)


def test_get_num_qkv_head_by_tp_with_none_dist_config():
    model_config = _make_model_config(dist_config=None)

    assert model_config.get_num_qkv_head_by_tp() == (32, 8)


def test_from_hf_config_keeps_dist_config_for_head_split():
    hf_config = SimpleNamespace(
        architectures=['OtherForCausalLM'],
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=4096,
        model_type='other',
        num_attention_heads=32,
        num_hidden_layers=1,
        num_key_value_heads=8,
        vocab_size=32000,
    )

    model_config = ModelConfig.from_hf_config(hf_config, dist_config=DistConfig(tp=4))

    assert model_config.dist_config.tp == 4
    assert model_config.get_num_qkv_head_by_tp() == (8, 2)


def test_get_num_qkv_head_by_tp_with_dist_config_tp():
    model_config = _make_model_config(dist_config=DistConfig(tp=2))

    assert model_config.get_num_qkv_head_by_tp() == (16, 4)


def test_get_num_qkv_head_by_tp_replicated_kv_heads():
    model_config = _make_model_config(num_attention_heads=32, num_key_value_heads=2, dist_config=DistConfig(tp=8))

    assert model_config.get_num_qkv_head_by_tp() == (4, 1)


def test_get_num_qkv_head_by_tp_requires_divisible_heads():
    model_config = _make_model_config(num_attention_heads=30, num_key_value_heads=8, dist_config=DistConfig(tp=4))

    with pytest.raises(AssertionError):
        model_config.get_num_qkv_head_by_tp()
