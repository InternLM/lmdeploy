from textwrap import dedent
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, DistConfig, ModelConfig
from lmdeploy.pytorch.configurations import AutoModelConfigBuilder
from lmdeploy.pytorch.configurations.deepseek_v4 import update_cache_config as update_deepseek_v4_cache_config
from lmdeploy.pytorch.nn import RopeType, build_rotary_embedding_from_config, build_rotary_params


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


def _make_sparse_conceptlm_hf_config():
    return SimpleNamespace(
        architectures=['ConceptLMV22VQForCausalLM'],
        model_type='conceptlm_v22_vq',
        hidden_size=4096,
        num_hidden_layers=32,
        num_layers=32,
        num_attention_heads=32,
        kv_channels=128,
        bos_token_id=100257,
        eos_token_id=100257,
        pad_token_id=100277,
        vocab_size=100278,
        torch_dtype='bfloat16',
        concept_chunk_size=4,
        concept_shift_feature=True,
        position_embedding_type='yarn',
        max_position_embeddings=65536,
        max_sequence_length=65536,
        rotary_base=500000,
        rotary_percent=1.0,
        yarn_rotary_scaling_factor=8.0,
        yarn_original_max_position_embeddings=8192,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        yarn_mscale=1.0,
        yarn_mscale_all_dim=0.0,
        yarn_correction_range_round_to_int=True,
    )


def _write_conceptlm_training_config(model_dir):
    (model_dir / 'training_config.yaml').write_text(
        dedent("""
        conceptlm_encoder_layers:
          value: 16
        conceptlm_special_layers:
          value: 8
        conceptlm_decoder_layers:
          value: 16
        conceptlm_chunk_merge_method:
          value: meanpooling
        conceptlm_fusion_norm_alpha_init:
          value: 0.1
        conceptlm_v22_vq_codebook_size:
          value: 128
        conceptlm_v22_vq_num_codebooks:
          value: 32
        conceptlm_v21_dd_two_route_add_decoder_use_softmax:
          value: true
        conceptlm_v21_enable_concept_read_encoder:
          value: true
        conceptlm_v21_enable_decoder_read_encoder:
          value: true
        conceptlm_v21_enable_decoder_read_concept:
          value: true
        conceptlm_v21_concept_read_encoder_first_n:
          value: -1
        conceptlm_v21_decoder_read_encoder_first_n:
          value: -1
        """),
        encoding='utf-8')


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


def test_conceptlm_model_config_backfills_new_export_training_config(tmp_path):
    hf_config = _make_sparse_conceptlm_hf_config()
    _write_conceptlm_training_config(tmp_path)

    model_config = AutoModelConfigBuilder.build(hf_config, str(tmp_path))

    assert hf_config.concept_encoder_layers == 16
    assert hf_config.concept_special_layers == 8
    assert hf_config.concept_decoder_layers == 16
    assert hf_config.concept_v22_vq_codebook_size == 128
    assert hf_config.concept_v22_vq_num_codebooks == 32
    assert hf_config.concept_chunk_merge_method == 'meanpooling'
    assert model_config.num_layers == 40
    assert model_config.states_shapes == [((16, 4096), torch.float32), ((9, 4096), torch.bfloat16)]


def test_conceptlm_model_config_reports_sparse_export_without_training_config(tmp_path):
    hf_config = _make_sparse_conceptlm_hf_config()

    with pytest.raises(AttributeError, match='training_config.yaml'):
        AutoModelConfigBuilder.build(hf_config, str(tmp_path))


def test_conceptlm_yarn_rotary_config_uses_original_context_length(tmp_path):
    hf_config = _make_sparse_conceptlm_hf_config()
    _write_conceptlm_training_config(tmp_path)
    model_config = AutoModelConfigBuilder.build(hf_config, str(tmp_path))

    rotary_params = build_rotary_params(hf_config)

    assert model_config.head_dim == 128
    assert hf_config.head_dim == 128
    assert hf_config.rope_theta == 500000
    assert hf_config.partial_rotary_factor == 1.0
    assert hf_config.rope_scaling['rope_type'] == 'yarn'
    assert hf_config.rope_parameters['rope_theta'] == 500000
    assert rotary_params['emb_type'] is RopeType.Yarn
    assert rotary_params['scaling_factor'] == 8.0
    assert rotary_params['max_position_embeddings'] == 8192
    assert rotary_params['yarn_params'].beta_fast == 32.0
    assert rotary_params['yarn_params'].beta_slow == 1.0
    assert rotary_params['yarn_params'].mscale == 1.0
    assert rotary_params['yarn_params'].mscale_all_dim == 0.0
    assert rotary_params['yarn_params'].truncate is True
    assert build_rotary_embedding_from_config(hf_config).base == 500000


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
