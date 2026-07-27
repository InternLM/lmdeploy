from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, DistConfig, ModelConfig
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
        compress_ratios=compress_ratios,
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


@pytest.mark.parametrize(('block_size', 'kernel_block_size', 'expected_block_size'), [
    (192, 64, 256),
    (256, 128, 256),
    (257, 128, 384),
])
def test_deepseek_v4_update_cache_config_normalizes_block_and_kernel_size(block_size, kernel_block_size,
                                                                          expected_block_size):
    cache_config = CacheConfig(max_batches=1,
                               block_size=block_size,
                               kernel_block_size=kernel_block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)

    update_deepseek_v4_cache_config(cache_config)

    assert cache_config.block_size == expected_block_size
    assert cache_config.kernel_block_size == expected_block_size
    assert cache_config.window_size == -1


def test_deepseek_v4_model_config_normalizes_block_cache_spec_shapes():
    hf_config = _make_deepseek_v4_hf_config([4, 128], num_hidden_layers=2)

    model_config = AutoModelConfigBuilder.build(hf_config)
    model_config.block_size = 192
    model_config.post_build_func(model_config, 192)

    assert model_config.block_size == 256
    block_specs = {spec.name: spec for spec in model_config.block_cache_specs}
    assert block_specs['v4_compressed_kv_r4_fp8'].shape[0] == 64
    assert block_specs['v4_index_kv_r4'].shape == (64, 1, 132)
    assert block_specs['v4_index_kv_r4'].dtype == torch.uint8
    assert block_specs['v4_compressed_kv_r128_fp8'].shape[0] == 2


def test_deepseek_v4_model_config_trims_trailing_zero_compress_ratio():
    hf_config = _make_deepseek_v4_hf_config([0, 4, 128, 0])

    model_config = AutoModelConfigBuilder.build(hf_config)
    model_config.post_build_func(model_config, 256)

    assert hf_config.compress_ratios == [0, 4, 128]
    state_specs = {spec.name: spec for spec in model_config.state_cache_specs}
    assert state_specs['v4_window_kv_fp8'].layer_ids == [0, 1, 2]
    assert state_specs['v4_compress_state_r4'].layer_ids == [1]
    assert state_specs['v4_compress_state_r4_idx'].layer_ids == [1]
    assert state_specs['v4_compress_state_r128'].layer_ids == [2]

    block_specs = {spec.name: spec for spec in model_config.block_cache_specs}
    assert block_specs['v4_compressed_kv_r4_fp8'].layer_ids == [1]
    assert block_specs['v4_index_kv_r4'].layer_ids == [1]
    assert block_specs['v4_compressed_kv_r128_fp8'].layer_ids == [2]


def test_deepseek_v4_model_config_rejects_extra_nonzero_compress_ratio():
    hf_config = _make_deepseek_v4_hf_config([0, 4, 128])
    hf_config.num_hidden_layers = 2

    with pytest.raises(ValueError, match='extra non-zero entries'):
        AutoModelConfigBuilder.build(hf_config)
