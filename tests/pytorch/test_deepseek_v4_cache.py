# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.engine.compressed_cache_engine import CompressedCacheEngine


def test_compressed_cache_engine_window():
    engine = CompressedCacheEngine(window_size=4, compress_ratio=4, head_dim=8)
    slot = 3
    engine.append_window(slot, torch.ones(2, 8))
    engine.append_window(slot, torch.full((3, 8), 2.0))
    window = engine.get_window(slot)
    assert window.shape == (4, 8)
    assert torch.all(window[:1] == 1)
    assert torch.all(window[1:] == 2)


def test_compressed_cache_engine_append_and_reset():
    engine = CompressedCacheEngine(window_size=8, compress_ratio=4, head_dim=16)
    slot = 1
    engine.append_compressed(slot, torch.randn(2, 16))
    engine.append_compressed(slot, torch.randn(3, 16))
    assert engine.get_compressed(slot).shape == (5, 16)
    engine.reset_slot(slot)
    assert engine.get_compressed(slot) is None


def test_compressed_cache_engine_states():
    engine = CompressedCacheEngine(window_size=8, compress_ratio=4, head_dim=12)
    slot = 7
    engine.ensure_states(slot, num_rows=8, device=torch.device('cpu'), dtype=torch.bfloat16)
    kv_state, score_state = engine.get_states(slot)
    assert kv_state.shape == (8, 12)
    assert score_state.shape == (8, 12)


def test_compressed_cache_engine_overlap_state_dim():
    engine = CompressedCacheEngine(window_size=8, compress_ratio=4, head_dim=12, overlap=True, state_dim=24)
    slot = 9
    engine.ensure_states(slot, num_rows=8, device=torch.device('cpu'), dtype=torch.bfloat16)
    kv_state, score_state = engine.get_states(slot)
    assert kv_state.shape == (8, 24)
    assert score_state.shape == (8, 24)


def test_compressed_cache_engine_window_ring_layout():
    engine = CompressedCacheEngine(window_size=4, compress_ratio=4, head_dim=2)
    slot = 5
    kv = torch.arange(12, dtype=torch.float32).view(6, 2)
    engine.set_window(slot, kv)
    window = engine.get_window(slot)
    expected = torch.empty(4, 2)
    expected[2:], expected[:2] = kv[-4:].split([2, 2], dim=0)
    assert torch.equal(window, expected)


def test_compressed_cache_engine_window_update():
    engine = CompressedCacheEngine(window_size=4, compress_ratio=4, head_dim=2)
    slot = 6
    engine.set_window(slot, torch.arange(8, dtype=torch.float32).view(4, 2))
    engine.update_window(slot, 1, torch.tensor([99.0, 100.0]))
    window = engine.get_window(slot)
    assert torch.equal(window[1], torch.tensor([99.0, 100.0]))
