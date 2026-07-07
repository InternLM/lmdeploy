import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, DistConfig, ModelConfig, QuantizationConfig, SpecDecodeConfig


def _qwen35_moe_config_factory(quantization_config=None):
    from transformers import PretrainedConfig

    def _factory():
        text_config = PretrainedConfig(torch_dtype='float16', tie_word_embeddings=False)
        text_config.model_type = 'qwen3_5_moe_text'
        text_config.layer_types = ['full_attention']
        text_config.linear_key_head_dim = 8
        text_config.linear_value_head_dim = 8
        text_config.linear_num_key_heads = 2
        text_config.linear_num_value_heads = 2
        text_config.linear_conv_kernel_dim = 4
        text_config.mtp_num_hidden_layers = 1

        hf_config = PretrainedConfig(torch_dtype='float16', tie_word_embeddings=False)
        hf_config.model_type = 'qwen3_5_moe'
        hf_config.architectures = ['Qwen3_5MoeForConditionalGeneration']
        hf_config.text_config = text_config
        if quantization_config is not None:
            hf_config.quantization_config = dict(quantization_config)
        return hf_config

    return _factory


def _qwen35_dense_config_factory():
    from transformers import PretrainedConfig

    def _factory():
        text_config = PretrainedConfig(torch_dtype='float16', tie_word_embeddings=False)
        text_config.model_type = 'qwen3_5_text'
        text_config.layer_types = ['full_attention']
        text_config.linear_key_head_dim = 8
        text_config.linear_value_head_dim = 8
        text_config.linear_num_key_heads = 2
        text_config.linear_num_value_heads = 2
        text_config.linear_conv_kernel_dim = 4
        text_config.mtp_num_hidden_layers = 1

        hf_config = PretrainedConfig(torch_dtype='float16', tie_word_embeddings=False)
        hf_config.model_type = 'qwen3_5'
        hf_config.architectures = ['Qwen3_5ForConditionalGeneration']
        hf_config.text_config = text_config
        return hf_config

    return _factory


def _plain_config_factory():
    from transformers import PretrainedConfig

    def _factory():
        hf_config = PretrainedConfig(torch_dtype='float16', tie_word_embeddings=False)
        hf_config.model_type = 'fake'
        hf_config.architectures = ['FakeForCausalLM']
        return hf_config

    return _factory


def _patch_config_builder(monkeypatch, config_factory):
    from lmdeploy.pytorch import transformers as pytorch_transformers
    from lmdeploy.pytorch.configurations.default import DefaultModelConfigBuilder

    def fake_config_from_pretrained(pretrained_model_name_or_path, **kwargs):
        return config_factory()

    def fake_build(cls, hf_config, model_path=None, **kwargs):
        return ModelConfig(hidden_size=16,
                           num_layers=1,
                           num_attention_heads=2,
                           num_key_value_heads=2,
                           bos_token_id=1,
                           eos_token_id=[2],
                           head_dim=8,
                           dtype=torch.float16,
                           vocab_size=32,
                           llm_config=hf_config)

    monkeypatch.setattr(pytorch_transformers, 'config_from_pretrained', fake_config_from_pretrained)
    monkeypatch.setattr(DefaultModelConfigBuilder, 'build', classmethod(fake_build))


def test_fp8_moe_only_scope_enabled_for_synthesized_qwen35_moe(monkeypatch):
    from lmdeploy.pytorch import envs

    monkeypatch.setattr(envs, 'fp8_moe_only', True)
    _patch_config_builder(monkeypatch, _qwen35_moe_config_factory())

    model_config = ModelConfig.from_pretrained('fake-model', model_format='fp8')

    assert model_config.quant_config.quant_method == 'fp8'
    assert model_config.quant_config.fp8_quant_scope == 'moe_only'
    assert model_config.quant_config.hf_quant_config['lmdeploy_patched']
    assert model_config.quant_config.hf_quant_config['fp8_quant_scope'] == 'moe_only'


def test_fp8_moe_only_scope_requires_env(monkeypatch):
    from lmdeploy.pytorch import envs

    monkeypatch.setattr(envs, 'fp8_moe_only', False)
    _patch_config_builder(monkeypatch, _qwen35_moe_config_factory())

    model_config = ModelConfig.from_pretrained('fake-model', model_format='fp8')

    assert model_config.quant_config.quant_method == 'fp8'
    assert model_config.quant_config.fp8_quant_scope is None
    assert 'fp8_quant_scope' not in model_config.quant_config.hf_quant_config


def test_fp8_moe_only_scope_only_for_qwen35_moe(monkeypatch):
    from lmdeploy.pytorch import envs

    monkeypatch.setattr(envs, 'fp8_moe_only', True)
    _patch_config_builder(monkeypatch, _plain_config_factory())

    model_config = ModelConfig.from_pretrained('fake-model', model_format='fp8')

    assert model_config.quant_config.quant_method == 'fp8'
    assert model_config.quant_config.fp8_quant_scope is None


def test_fp8_moe_only_scope_does_not_apply_to_dense_qwen35(monkeypatch):
    from lmdeploy.pytorch import envs

    monkeypatch.setattr(envs, 'fp8_moe_only', True)
    _patch_config_builder(monkeypatch, _qwen35_dense_config_factory())

    model_config = ModelConfig.from_pretrained('fake-model', model_format='fp8')

    assert model_config.quant_config.quant_method == 'fp8'
    assert model_config.quant_config.fp8_quant_scope is None
    assert model_config.quant_config.hf_quant_config['lmdeploy_patched']
    assert 'fp8_quant_scope' not in model_config.quant_config.hf_quant_config


def test_fp8_moe_only_scope_does_not_override_prequantized_config(monkeypatch):
    from lmdeploy.pytorch import envs

    monkeypatch.setattr(envs, 'fp8_moe_only', True)
    quantization_config = dict(quant_method='fp8', fmt='e4m3', weight_block_size=[128, 128])
    _patch_config_builder(monkeypatch, _qwen35_moe_config_factory(quantization_config))

    model_config = ModelConfig.from_pretrained('fake-model', model_format='fp8')

    assert model_config.quant_config.quant_method == 'fp8'
    assert model_config.quant_config.fp8_quant_scope is None
    assert 'lmdeploy_patched' not in model_config.quant_config.hf_quant_config
    assert 'fp8_quant_scope' not in model_config.quant_config.hf_quant_config


def test_quantization_config_fp8_moe_only_module_kinds():
    quant_config = QuantizationConfig(quant_method='fp8', fp8_quant_scope='moe_only')

    assert quant_config.get_quant_method('model.layers.0.mlp.experts', module_kind='moe') == 'fp8'
    assert quant_config.get_quant_method('model.layers.0.self_attn.qkv_proj', module_kind='linear') is None
    assert quant_config.get_quant_method('model.layers.0.input_layernorm', module_kind='norm') is None
    assert quant_config.get_quant_method('', module_kind='linear') is None

    all_scope_quant_config = QuantizationConfig(quant_method='fp8')
    assert all_scope_quant_config.get_quant_method('model.layers.0.self_attn.qkv_proj',
                                                   module_kind='linear') == 'fp8'
    assert all_scope_quant_config.get_quant_method('model.layers.0.input_layernorm', module_kind='norm') == 'fp8'
    with pytest.raises(ValueError, match='Unsupported quant module kind'):
        quant_config.get_quant_method(module_kind='gate')
    with pytest.raises(ValueError, match='Unsupported fp8 quant scope'):
        QuantizationConfig(quant_method='fp8', fp8_quant_scope='all').get_quant_method()


def test_fp8_moe_only_scope_survives_mtp_draft_config(monkeypatch):
    from lmdeploy.pytorch import envs

    monkeypatch.setattr(envs, 'fp8_moe_only', True)
    _patch_config_builder(monkeypatch, _qwen35_moe_config_factory())
    target_cache_cfg = CacheConfig(max_batches=1,
                                   block_size=64,
                                   num_cpu_blocks=0,
                                   num_gpu_blocks=0)

    specdecode_config = SpecDecodeConfig.from_config(method='qwen3_5_mtp',
                                                     num_speculative_tokens=1,
                                                     model=None,
                                                     target_model='fake-model',
                                                     target_cache_cfg=target_cache_cfg,
                                                     dist_config=DistConfig(),
                                                     model_format='fp8')
    main_model_config = ModelConfig.from_pretrained('fake-model',
                                                   model_format='fp8',
                                                   spec_method=specdecode_config.method,
                                                   num_spec_tokens=specdecode_config.num_speculative_tokens)

    assert specdecode_config.model_config.quant_config.quant_method == 'fp8'
    assert main_model_config.quant_config.quant_method == 'fp8'
    assert specdecode_config.model_config.quant_config.fp8_quant_scope == 'moe_only'
    assert main_model_config.quant_config.fp8_quant_scope == 'moe_only'
