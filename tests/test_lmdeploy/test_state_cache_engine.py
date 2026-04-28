from types import SimpleNamespace

import torch

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
    specs = {spec.name: spec for spec in config.state_cache_specs}
    block_specs = {spec.name: spec for spec in config.block_cache_specs}

    assert specs['v4_window_kv'].layer_ids == [0, 1, 2, 3]
    assert specs['v4_compress_state_r4'].layer_ids == [0, 3]
    assert specs['v4_compress_state_r4_idx'].layer_ids == [0, 3]
    assert specs['v4_compress_state_r128'].layer_ids == [2]
    assert 'v4_raw_kv' not in block_specs


def test_build_window_positions_uses_ring_coordinates():
    total_lens = torch.tensor([3, 10])
    positions, window_lens, mask = _build_window_positions(total_lens, 8)

    assert torch.equal(window_lens, torch.tensor([3, 8]))
    assert torch.equal(mask[0], torch.tensor([True, True, True, False, False, False, False, False]))
    assert torch.equal(positions[0, :3], torch.tensor([0, 1, 2]))
    assert torch.equal(positions[1], torch.tensor([2, 3, 4, 5, 6, 7, 0, 1]))
