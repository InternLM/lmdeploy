from types import SimpleNamespace

import pytest
import torch

from lmdeploy.hf_configs import config_from_pretrained
from lmdeploy.hf_configs.configuration_kimi_k2 import KimiK2Config
from lmdeploy.pytorch.config import CacheConfig, DistConfig, ModelConfig, QuantizationConfig
from lmdeploy.pytorch.configurations import AutoModelConfigBuilder
from lmdeploy.pytorch.configurations.deepseek_v4 import update_cache_config as update_deepseek_v4_cache_config


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


def _make_deepseek_v4_hf_config(compress_ratios, num_hidden_layers=3):
    ratio_to_layer_type = {
        0: 'sliding_attention',
        4: 'compressed_sparse_attention',
        128: 'heavily_compressed_attention',
    }
    return SimpleNamespace(
        model_type='deepseek_v4',
        architectures=['DeepseekV4ForCausalLM'],
        hidden_size=4096,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=64,
        num_key_value_heads=1,
        eos_token_id=2,
        sliding_window=128,
        vocab_size=32000,
        layer_types=[ratio_to_layer_type[ratio] for ratio in compress_ratios],
        compress_rates={
            'compressed_sparse_attention': 4,
            'heavily_compressed_attention': 128,
        },
        index_head_dim=128,
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


@pytest.mark.parametrize(
    ('num_kv_heads', 'expected_effective_heads', 'expected_replica_num'),
    [(32, 32, 1), (2, 8, 4), (1, 8, 8)],
)
def test_model_config_records_kv_head_replication(
    num_kv_heads,
    expected_effective_heads,
    expected_replica_num,
):
    hf_config = SimpleNamespace(
        architectures=['OtherForCausalLM'],
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=4096,
        model_type='other',
        num_attention_heads=32,
        num_hidden_layers=1,
        num_key_value_heads=num_kv_heads,
        vocab_size=32000,
    )

    model_config = ModelConfig.from_hf_config(
        hf_config,
        dist_config=DistConfig(tp=8),
    )

    assert model_config.num_key_value_heads == expected_effective_heads
    assert model_config.num_replicate_key_value_heads == expected_replica_num


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


def test_kimi_k2_standalone_eagle_config(tmp_path):
    hf_config = KimiK2Config(
        architectures=['Eagle3DeepseekV2ForCausalLM'],
        vocab_size=163840,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=64,
        max_position_embeddings=262144,
        rope_parameters={
            'rope_type': 'yarn',
            'factor': 64.0,
            'original_max_position_embeddings': 4096,
            'beta_fast': 1.0,
            'beta_slow': 1.0,
            'mscale': 1.0,
            'mscale_all_dim': 1.0,
            'rope_theta': 50000.0,
        },
        dtype='bfloat16',
    )
    hf_config.save_pretrained(tmp_path)

    loaded_config = config_from_pretrained(tmp_path)
    assert isinstance(loaded_config, KimiK2Config)
    assert loaded_config.model_type == 'kimi_k2'
    assert loaded_config.rope_parameters['rope_type'] == 'yarn'

    model_config = ModelConfig.from_pretrained(
        tmp_path,
        is_draft_model=True,
        spec_method='eagle3',
    )
    assert model_config.num_layers == 1
    assert model_config.vocab_size == 163840
    assert model_config.model_paradigm == 'ar_spec'


def test_kimi_compressed_tensors_metadata_uses_existing_quant_config():
    compressed_config = {
        'quant_method': 'compressed-tensors',
        'format': 'pack-quantized',
        'quantization_status': 'compressed',
        'config_groups': {
            'group_0': {
                'targets': ['Linear'],
                'input_activations': None,
                'output_activations': None,
                'weights': {
                    'num_bits': 4,
                    'group_size': 32,
                    'strategy': 'group',
                    'symmetric': True,
                    'dynamic': False,
                    'type': 'int',
                },
            },
        },
        'ignore': [
            r're:.*self_attn.*',
            r're:.*mlp\.(gate|up|gate_up|down)_proj.*',
        ],
    }
    hf_config = SimpleNamespace(
        text_config=SimpleNamespace(quantization_config=compressed_config),
    )

    quant_config = QuantizationConfig.from_config(hf_config)

    assert quant_config.bits == 4
    assert quant_config.group_size == 32
    assert quant_config.get_quant_method(
        'language_model.model.layers.0.mlp.experts',
        module_kind='moe',
    ) == 'compressed-tensors'
    assert quant_config.get_quant_method(
        'language_model.model.layers.0.self_attn.q_a_proj',
        module_kind='linear',
    ) is None
    assert quant_config.get_quant_method(
        'language_model.model.layers.0.mlp.gate_proj',
        module_kind='linear',
    ) is None
    assert quant_config.get_quant_method(
        'language_model.model.layers.0.input_layernorm',
        module_kind='norm',
    ) is None


def test_update_torch_dtype_scopes_ct_fp16_force_to_group_128(monkeypatch):
    """Only group-128 ct checkpoints force fp16; group-32 keeps its dtype.

    The group-32 packed-ct MoE path supports bf16 activations (DeepEP requires them), so forcing fp16 for every
    compressed-tensors config broke existing deployments.
    """
    import lmdeploy.pytorch.config as pytorch_config
    from lmdeploy.pytorch.config import _update_torch_dtype

    # Pin bf16 support so the group-32 fall-through resolves bf16 on hosts
    # without a bf16-capable device (mirrors test_fp8_quant_scope.py).
    monkeypatch.setattr(pytorch_config, 'is_bf16_supported', lambda device_type='auto': True)

    def _ct_config(group_size):
        return SimpleNamespace(
            hf_config=SimpleNamespace(
                quantization_config={
                    'quant_method': 'compressed-tensors',
                    'config_groups': {
                        'group_0': {
                            'weights': {'group_size': group_size},
                        }
                    },
                },
                torch_dtype='bfloat16'),
            dtype=torch.bfloat16)

    # Group-128 normalizes into the fp16 AWQ family: forced.
    config = _update_torch_dtype(_ct_config(128), 'auto')
    assert config.dtype == torch.float16
    assert config.hf_config.torch_dtype == 'float16'

    # Group-32 packed ct keeps the configured dtype.
    config = _update_torch_dtype(_ct_config(32), 'auto')
    assert config.dtype == torch.bfloat16
    assert config.hf_config.torch_dtype == 'bfloat16'

    # AWQ still forces fp16.
    config = _update_torch_dtype(
        SimpleNamespace(
            hf_config=SimpleNamespace(
                quantization_config={'quant_method': 'awq'},
                torch_dtype='bfloat16'),
            dtype=torch.bfloat16),
        'auto')
    assert config.dtype == torch.float16


@pytest.mark.parametrize(('block_size', 'kernel_block_size'), [
    (64, 64),
    (192, 64),
    (256, 128),
    (257, 128),
    (512, 512),
])
def test_deepseek_v4_update_cache_config_forces_block_and_kernel_size(block_size, kernel_block_size):
    cache_config = CacheConfig(max_batches=1,
                               block_size=block_size,
                               kernel_block_size=kernel_block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)

    update_deepseek_v4_cache_config(cache_config)

    assert cache_config.block_size == 256
    assert cache_config.kernel_block_size == 256
    assert cache_config.window_size == -1


def test_deepseek_v4_model_config_uses_native_layer_types():
    hf_config = _make_deepseek_v4_hf_config([0, 4, 128])

    model_config = AutoModelConfigBuilder.build(hf_config)

    state_specs = {spec.name: spec for spec in model_config.state_cache_specs}
    assert state_specs['v4_window_kv_fp8'].layer_ids == [0, 1, 2]
    assert state_specs['v4_compress_state_r4'].layer_ids == [1]
    assert state_specs['v4_compress_state_r4_idx'].layer_ids == [1]
    assert state_specs['v4_compress_state_r128'].layer_ids == [2]


def test_deepseek_v4_model_config_rejects_layer_type_count_mismatch():
    hf_config = _make_deepseek_v4_hf_config([0, 4, 128])
    hf_config.num_hidden_layers = 2

    with pytest.raises(ValueError, match='one layer_type per hidden layer'):
        AutoModelConfigBuilder.build(hf_config)


def test_deepseek_v4_model_config_rejects_unsupported_native_compress_rate():
    hf_config = _make_deepseek_v4_hf_config([4], num_hidden_layers=1)
    hf_config.compress_rates['compressed_sparse_attention'] = 8

    with pytest.raises(ValueError, match='only supports ratios'):
        AutoModelConfigBuilder.build(hf_config)


@pytest.mark.parametrize('num_spec_tokens', [0, 3, 5])
def test_glm5_next_tail_checkpoint_policy_requires_closed_pool_boundary(monkeypatch, num_spec_tokens):
    from lmdeploy.hf_configs.configuration_glm5_next import Glm5NextConfig
    from lmdeploy.pytorch.configurations.glm5_next import Glm5NextModelConfigBuilder
    from lmdeploy.pytorch.consts import (
        GLM5_KDA_CONV_STATE,
        GLM5_KDA_RECURRENT_STATE,
        GLM5_KPOOL_TAIL_K_STATE,
        GLM5_KPOOL_TAIL_SCORE_STATE,
    )

    monkeypatch.setattr('lmdeploy.pytorch.configurations.deepseek_v2.flash_mla_available', lambda: True)
    hf_config = Glm5NextConfig()
    model = Glm5NextModelConfigBuilder.build(hf_config, num_spec_tokens=num_spec_tokens)
    specs = {spec.name: spec for spec in model.state_cache_specs}
    for name in (GLM5_KDA_CONV_STATE, GLM5_KDA_RECURRENT_STATE):
        assert specs[name].prefix_checkpoint
    for name in (GLM5_KPOOL_TAIL_K_STATE, GLM5_KPOOL_TAIL_SCORE_STATE):
        assert not specs[name].prefix_checkpoint
        assert specs[name].shape[1] == hf_config.text_config.index_kpool + 1 + num_spec_tokens
    cache = CacheConfig(max_batches=1, block_size=16, num_cpu_blocks=0, num_gpu_blocks=0,
                        prefix_cache_token_lookahead=int(num_spec_tokens > 0))
    model.update_cache_config_func(cache)
    assert cache.prefix_cache_token_lookahead == int(num_spec_tokens > 0)
    assert cache.prefix_cache_block_aligned
    assert cache.block_size == 64
    assert cache.block_size % hf_config.text_config.index_kpool == 0
    if num_spec_tokens:
        cache.prefix_cache_decode_state_interval = 64
        cache.prefix_cache_decode_save_first = True
        model.update_cache_config_func(cache)
        assert cache.prefix_cache_decode_state_interval == 64
    draft = Glm5NextModelConfigBuilder.build(Glm5NextConfig(), is_draft_model=True,
                                            num_spec_tokens=num_spec_tokens)
    assert not draft.state_cache_specs
    hf_config.text_config.index_kpool = 3
    with pytest.raises(ValueError, match='index_kpool=4'):
        Glm5NextModelConfigBuilder.build(hf_config, num_spec_tokens=num_spec_tokens)


@pytest.mark.parametrize('full_type', ['deepseek_sparse_attention', 'indexed_attention', 'full_attention'])
def test_glm5_next_full_attention_aliases_preserve_geometry(monkeypatch, full_type):
    from transformers import Glm5NextConfig

    from lmdeploy.pytorch.configurations.glm5_next import Glm5NextModelConfigBuilder, is_glm5_kda_layer

    monkeypatch.setattr('lmdeploy.pytorch.configurations.deepseek_v2.flash_mla_available', lambda: True)
    config = Glm5NextConfig(text_config={'num_hidden_layers': 4, 'index_kpool': 4})
    # Assign after HF normalization to exercise every accepted spelling directly.
    config.text_config.layer_types = ['linear_attention'] * 3 + [full_type]
    model = Glm5NextModelConfigBuilder.build(config)
    assert model.num_layers == 1
    assert config.text_config.linear_layer_ids == [0, 1, 2]
    assert config.text_config.full_attention_layer_ids == [3]
    assert [is_glm5_kda_layer(config.text_config, i) for i in range(4)] == [True, True, True, False]
    assert model.state_cache_specs[0].shape == (3, 3 * 64 * 128, 3)
    assert model.state_cache_specs[1].shape == (3, 64, 128, 128)


def test_glm5_next_rejects_unknown_attention_type():
    from transformers import Glm5NextConfig

    from lmdeploy.pytorch.configurations.glm5_next import _resolve_glm5_linear_config

    config = Glm5NextConfig(text_config={'num_hidden_layers': 4})
    config.text_config.layer_types[-1] = 'unknown_attention'
    with pytest.raises(ValueError, match='unknown_attention'):
        _resolve_glm5_linear_config(config.text_config)
