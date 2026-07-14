from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import ModelConfig
from lmdeploy.pytorch.configurations.deepseek_v32 import (
    DeepseekV32ModelConfigBuilder,
    normalize_glm_moe_dsa_config,
)


def _make_config(**kwargs):
    values = dict(
        model_type='glm_moe_dsa',
        num_hidden_layers=4,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def _patch_v32_base_builder(monkeypatch):
    from lmdeploy.pytorch.configurations.deepseek_v2 import DeepseekV2ModelConfigBuilder

    def fake_build(cls, hf_config, model_path=None, **kwargs):
        return SimpleNamespace()

    monkeypatch.setattr(DeepseekV2ModelConfigBuilder, 'build', classmethod(fake_build))


def _make_fp8_build_config(**quantization_config):
    return _make_config(
        use_flash_mla=True,
        index_head_dim=128,
        index_topk=2048,
        quantization_config=quantization_config,
    )


def test_glm_moe_dsa_enables_online_fp8_moe_only_scope(monkeypatch):
    from lmdeploy.pytorch import envs
    from lmdeploy.pytorch import transformers as pytorch_transformers
    from lmdeploy.pytorch.configurations import deepseek_v2
    from lmdeploy.pytorch.transformers.configuration_glm_moe_dsa import GlmMoeDsaConfig

    monkeypatch.setattr(envs, 'fp8_moe_only', True)
    monkeypatch.setattr(deepseek_v2, 'flash_mla_available', lambda: True)
    cfg = GlmMoeDsaConfig(hidden_size=16,
                          intermediate_size=32,
                          moe_intermediate_size=8,
                          num_hidden_layers=2,
                          num_attention_heads=2,
                          num_key_value_heads=1,
                          q_lora_rank=4,
                          kv_lora_rank=4,
                          qk_nope_head_dim=4,
                          qk_rope_head_dim=4,
                          vocab_size=32,
                          bos_token_id=1,
                          eos_token_id=2,
                          index_head_dim=4,
                          index_n_heads=2,
                          index_topk=2)
    monkeypatch.setattr(pytorch_transformers, 'config_from_pretrained', lambda *args, **kwargs: cfg)

    model_config = ModelConfig.from_pretrained('fake-model', model_format='fp8', dtype='bfloat16')

    assert model_config.quant_config.quant_method == 'fp8'
    assert model_config.quant_config.fp8_quant_scope == 'moe_only'
    assert model_config.quant_config.hf_quant_config['lmdeploy_patched']
    assert model_config.dtype == torch.bfloat16


def test_glm_moe_dsa_fp8_moe_only_scope_requires_env(monkeypatch):
    from lmdeploy.pytorch import envs

    monkeypatch.setattr(envs, 'fp8_moe_only', False)
    _patch_v32_base_builder(monkeypatch)
    cfg = _make_fp8_build_config(quant_method='fp8', lmdeploy_patched=True)

    DeepseekV32ModelConfigBuilder.build(cfg)

    assert 'fp8_quant_scope' not in cfg.quantization_config


def test_glm_moe_dsa_does_not_override_prequantized_fp8_scope(monkeypatch):
    from lmdeploy.pytorch import envs

    monkeypatch.setattr(envs, 'fp8_moe_only', True)
    _patch_v32_base_builder(monkeypatch)
    cfg = _make_fp8_build_config(quant_method='fp8')

    DeepseekV32ModelConfigBuilder.build(cfg)

    assert 'fp8_quant_scope' not in cfg.quantization_config


def test_glm_moe_dsa_normalizes_pattern_indexer_types():
    cfg = _make_config(index_topk_pattern='FSSF')

    normalize_glm_moe_dsa_config(cfg)

    assert cfg.qk_head_dim == 192
    assert cfg.head_dim == 64
    assert cfg.indexer_types == ['full', 'shared', 'shared', 'full']


def test_glm_moe_dsa_normalizes_freq_offset_indexer_types():
    cfg = _make_config(num_hidden_layers=6, index_topk_freq=2, index_skip_topk_offset=2)

    normalize_glm_moe_dsa_config(cfg)

    assert cfg.indexer_types == ['full', 'full', 'shared', 'full', 'shared', 'full']


def test_glm_moe_dsa_recovers_corrupted_rope_head_dim():
    cfg = _make_config(qk_nope_head_dim=192, qk_rope_head_dim=192, qk_head_dim=256)

    normalize_glm_moe_dsa_config(cfg)

    assert cfg.qk_nope_head_dim == 192
    assert cfg.qk_rope_head_dim == 64
    assert cfg.qk_head_dim == 256
    assert cfg.head_dim == 64


def test_glm_moe_dsa_rejects_shared_first_layer():
    cfg = _make_config(index_topk_pattern='SFFF')

    with pytest.raises(ValueError, match='first GLM-MoE-DSA layer'):
        normalize_glm_moe_dsa_config(cfg)


def test_glm_moe_dsa_fallback_config_keeps_rope_head_dim():
    from lmdeploy.pytorch.transformers.configuration_glm_moe_dsa import GlmMoeDsaConfig

    cfg = GlmMoeDsaConfig(qk_nope_head_dim=128,
                          qk_rope_head_dim=64,
                          hidden_size=4096,
                          intermediate_size=8192,
                          num_attention_heads=32,
                          num_hidden_layers=2,
                          vocab_size=32000,
                          bos_token_id=1,
                          eos_token_id=2)

    assert cfg.qk_nope_head_dim == 128
    assert cfg.qk_rope_head_dim == 64
    assert cfg.qk_head_dim == 192
    assert cfg.head_dim == 64
    assert cfg.index_n_heads == 32
    assert cfg.rope_interleave is True
    assert cfg.indexer_rope_interleave is True


def test_glm_moe_dsa_builder_normalizes_mla_kv_heads(monkeypatch):
    from lmdeploy.pytorch.configurations import deepseek_v2

    monkeypatch.setattr(deepseek_v2, 'flash_mla_available', lambda: True)
    cfg = _make_config(num_hidden_layers=2,
                       index_head_dim=128,
                       index_n_heads=32,
                       index_topk=2048,
                       index_skip_topk_offset=3,
                       index_topk_freq=4,
                       indexer_rope_interleave=True,
                       rope_interleave=True,
                       hidden_size=6144,
                       kv_lora_rank=512,
                       num_attention_heads=64,
                       num_key_value_heads=64,
                       bos_token_id=None,
                       eos_token_id=[154820, 154827, 154829],
                       vocab_size=154880,
                       architectures=['GlmMoeDsaForCausalLM'])

    model_config = DeepseekV32ModelConfigBuilder.build(cfg, tp=1)

    assert cfg.num_key_value_heads == 1
    assert model_config.num_key_value_heads == 1
    assert model_config.head_dim == 576
    assert model_config.use_mla_fp8_cache is True
