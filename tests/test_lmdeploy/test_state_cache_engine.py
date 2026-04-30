from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends.cuda.attention.flashmla_utils import (
    dequantize_model1_fp8_sparse,
    model1_fp8_sparse_token_dim,
    quantize_model1_fp8_sparse,
)
from lmdeploy.pytorch.config import StateCacheSpec
from lmdeploy.pytorch.configurations.deepseek_v4 import DeepseekV4ModelConfigBuilder
from lmdeploy.pytorch.engine.cache_engine import StateCacheEngine
from lmdeploy.pytorch.models.deepseek_v4 import _build_window_positions


def test_state_cache_engine_allocates_layer_scoped_states():
    state_shapes = [((16, 1024), torch.float32), ((256, 512), torch.float32)]
    state_specs = [
        StateCacheSpec('layered_r4', (16, 1024), torch.float32, layer_ids=[1, 3]),
        StateCacheSpec('layered_r128', (256, 512), torch.float32, layer_ids=[0, 2]),
    ]

    _, caches = StateCacheEngine.allocate_caches(num_caches=3,
                                                 state_shapes=state_shapes,
                                                 state_specs=state_specs,
                                                 num_layers=4,
                                                 device='meta')

    assert caches[0].shape == (4, 3, 16, 1024)
    assert caches[1].shape == (4, 3, 256, 512)


def test_state_cache_engine_keeps_legacy_sequence_scoped_states():
    state_shapes = [((16, 1024), torch.float32)]

    _, caches = StateCacheEngine.allocate_caches(num_caches=3, state_shapes=state_shapes, device='meta')

    assert caches[0].shape == (3, 16, 1024)


def test_deepseek_v4_builder_marks_state_cache_layers():
    hf_config = SimpleNamespace(
        model_type='deepseek_v4',
        hidden_size=4096,
        num_hidden_layers=4,
        num_attention_heads=32,
        num_key_value_heads=1,
        bos_token_id=0,
        eos_token_id=[1],
        head_dim=512,
        sliding_window=1024,
        vocab_size=32000,
        compress_ratios=[4, 0, 128, 4],
        index_head_dim=128,
        index_topk=512,
    )

    config = DeepseekV4ModelConfigBuilder.build(hf_config)
    config.hf_config = hf_config
    config.post_build_func(config, 128)
    specs = {spec.name: spec for spec in config.state_cache_specs}
    block_specs = {spec.name: spec for spec in config.block_cache_specs}

    assert specs['v4_window_kv'].layer_ids == [0, 1, 2, 3]
    assert specs['v4_window_kv_fp8'].layer_ids == [0, 1, 2, 3]
    assert specs['v4_compress_state_r4'].layer_ids == [0, 3]
    assert specs['v4_compress_state_r4_idx'].layer_ids == [0, 3]
    assert specs['v4_compress_state_r128'].layer_ids == [2]
    assert 'v4_compress_state_r4_head' not in specs
    assert 'v4_compress_state_r4_idx_head' not in specs
    assert 'v4_raw_kv' not in block_specs
    assert block_specs['v4_compressed_kv_r4'].shape == (32, 512)
    assert block_specs['v4_compressed_kv_r4_fp8'].shape == (32, model1_fp8_sparse_token_dim(64))
    assert block_specs['v4_index_kv_r4'].shape == (32, 128)
    assert block_specs['v4_compressed_kv_r128'].shape == (1, 512)


def test_deepseek_v4_builder_rejects_small_block_size():
    hf_config = SimpleNamespace(
        model_type='deepseek_v4',
        hidden_size=4096,
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=1,
        bos_token_id=0,
        eos_token_id=[1],
        head_dim=512,
        sliding_window=1024,
        vocab_size=32000,
        compress_ratios=[4, 128],
        index_head_dim=128,
        index_topk=512,
    )
    config = DeepseekV4ModelConfigBuilder.build(hf_config)
    config.hf_config = hf_config
    with pytest.raises(ValueError, match='block_size >= 128'):
        config.post_build_func(config, 64)


def test_deepseek_v4_builder_requires_block_size_multiple_of_128():
    hf_config = SimpleNamespace(
        model_type='deepseek_v4',
        hidden_size=4096,
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=1,
        bos_token_id=0,
        eos_token_id=[1],
        head_dim=512,
        sliding_window=1024,
        vocab_size=32000,
        compress_ratios=[4, 128],
        index_head_dim=128,
        index_topk=512,
    )
    config = DeepseekV4ModelConfigBuilder.build(hf_config)
    config.hf_config = hf_config
    with pytest.raises(ValueError, match='multiple of 128'):
        config.post_build_func(config, 192)


def test_build_window_positions_uses_ring_coordinates():
    total_lens = torch.tensor([3, 10])
    positions, window_lens, mask = _build_window_positions(total_lens, 8)

    assert torch.equal(window_lens, torch.tensor([3, 8]))
    assert torch.equal(mask[0], torch.tensor([True, True, True, False, False, False, False, False]))
    assert torch.equal(positions[0, :3], torch.tensor([0, 1, 2]))
    assert torch.equal(positions[1], torch.tensor([2, 3, 4, 5, 6, 7, 0, 1]))


def test_flashmla_model1_fp8_sparse_helper_shape():
    token_dim = model1_fp8_sparse_token_dim(64)
    assert token_dim == 584

    if not torch.cuda.is_available():
        return

    cache = torch.randn(2, 64, 1, 512, dtype=torch.bfloat16, device='cuda')
    packed = quantize_model1_fp8_sparse(cache)
    assert packed.shape == (2, 64, 1, 584)
    assert packed.dtype == torch.float8_e4m3fn
    unpacked = dequantize_model1_fp8_sparse(packed)
    assert unpacked.shape == (2, 64, 1, 512)
    assert unpacked.dtype == torch.bfloat16
