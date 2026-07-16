import json
from types import SimpleNamespace

from lmdeploy.pytorch.config import ModelConfig
from lmdeploy.pytorch.configurations.glm_moe_dsa import (
    GlmMoeDsaModelConfigBuilder,
    normalize_glm_moe_dsa_config,
)


def test_get_model_arch_registers_deepseek_v32_config(tmp_path):
    from lmdeploy.archs import get_model_arch
    from lmdeploy.pytorch.transformers.configuration_deepseek_v32 import DeepseekV32Config

    config_path = tmp_path / 'config.json'
    config_path.write_text(
        json.dumps({
            'architectures': ['DeepseekV32ForCausalLM'],
            'model_type': 'deepseek_v32',
        }))

    arch, config = get_model_arch(str(tmp_path))

    assert arch == 'DeepseekV32ForCausalLM'
    assert isinstance(config, DeepseekV32Config)


def test_get_model_arch_loads_native_glm_moe_dsa_config(tmp_path):
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    from lmdeploy.archs import get_model_arch

    config_path = tmp_path / 'config.json'
    config_path.write_text(
        json.dumps({
            'architectures': ['GlmMoeDsaForCausalLM'],
            'model_type': 'glm_moe_dsa',
        }))

    arch, config = get_model_arch(str(tmp_path))

    assert arch == 'GlmMoeDsaForCausalLM'
    assert isinstance(config, GlmMoeDsaConfig)


def _make_config(**kwargs):
    values = dict(
        model_type='glm_moe_dsa',
        num_hidden_layers=4,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
    )
    values.update(kwargs)
    values.setdefault('qk_head_dim', values['qk_nope_head_dim'] + values['qk_rope_head_dim'])
    values.setdefault('indexer_types', ['full'] * values['num_hidden_layers'])
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
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    from lmdeploy.pytorch import envs
    from lmdeploy.pytorch import transformers as pytorch_transformers
    from lmdeploy.pytorch.configurations import deepseek_v2

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
                          dtype='float16',
                          index_head_dim=4,
                          index_n_heads=2,
                          index_topk=2)
    monkeypatch.setattr(pytorch_transformers, 'config_from_pretrained', lambda *args, **kwargs: cfg)

    model_config = ModelConfig.from_pretrained('fake-model', model_format='fp8', dtype='bfloat16')

    assert model_config.quant_config.quant_method == 'fp8'
    assert model_config.quant_config.fp8_quant_scope == 'moe_only'
    assert model_config.quant_config.hf_quant_config['lmdeploy_patched']


def test_glm_moe_dsa_does_not_override_prequantized_fp8_scope(monkeypatch):
    from lmdeploy.pytorch import envs

    monkeypatch.setattr(envs, 'fp8_moe_only', True)
    _patch_v32_base_builder(monkeypatch)
    cfg = _make_fp8_build_config(quant_method='fp8')

    GlmMoeDsaModelConfigBuilder.build(cfg)

    assert 'fp8_quant_scope' not in cfg.quantization_config


def test_glm_moe_dsa_runtime_normalizes_native_hf_config():
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    cfg = GlmMoeDsaConfig(qk_nope_head_dim=192,
                          qk_rope_head_dim=64,
                          qk_head_dim=256,
                          head_dim=192,
                          hidden_size=4096,
                          intermediate_size=8192,
                          num_attention_heads=32,
                          num_hidden_layers=2,
                          vocab_size=32000,
                          bos_token_id=1,
                          eos_token_id=2)

    assert cfg.head_dim == 192

    normalize_glm_moe_dsa_config(cfg)

    assert cfg.head_dim == 64
    assert cfg.qk_rope_head_dim == 64


def test_glm_moe_dsa_builder_creates_sparse_mla_config(monkeypatch):
    from lmdeploy.pytorch.configurations import deepseek_v2

    monkeypatch.setattr(deepseek_v2, 'flash_mla_available', lambda: True)
    cfg = _make_config(num_hidden_layers=2,
                       index_head_dim=128,
                       index_topk=2048,
                       hidden_size=6144,
                       kv_lora_rank=512,
                       num_attention_heads=64,
                       num_key_value_heads=64,
                       bos_token_id=None,
                       eos_token_id=[154820, 154827, 154829],
                       vocab_size=154880)

    model_config = GlmMoeDsaModelConfigBuilder.build(cfg, tp=1)

    assert cfg.num_key_value_heads == 1
    assert model_config.num_key_value_heads == 1
    assert model_config.head_dim == 576
    assert model_config.use_mla_fp8_cache is True
