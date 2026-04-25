# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.models.deepseek_v4 import Attention


def test_attention_gather_window_ring_layout():
    """Verify _gather_window returns ring layout compatible with
    get_window_topk_idxs."""
    # Simulate a raw kv cache with 2 layers, 2 blocks, block_size=4, head_dim=8
    raw_kv = torch.arange(2 * 2 * 4 * 8, dtype=torch.float32).view(2, 2, 4, 8)
    block_offsets = torch.tensor([[0, 1]], dtype=torch.int32)

    # total_len=6, window_size=4, window_start=2
    # tokens: t2 t3 t4 t5
    # cutoff = 6 % 4 = 2
    # ring = [t4, t5, t2, t3]
    ring = Attention._gather_window(raw_kv, block_offsets, seq_idx=0, block_size=4,
                                    window_start=2, total_len=6, layer_id=0)
    assert ring.shape == (4, 8)

    # t2 = raw_kv[0, 0, 2], t3 = raw_kv[0, 0, 3], t4 = raw_kv[0, 1, 0], t5 = raw_kv[0, 1, 1]
    t2 = raw_kv[0, 0, 2]
    t3 = raw_kv[0, 0, 3]
    t4 = raw_kv[0, 1, 0]
    t5 = raw_kv[0, 1, 1]
    assert torch.equal(ring[0], t4)
    assert torch.equal(ring[1], t5)
    assert torch.equal(ring[2], t2)
    assert torch.equal(ring[3], t3)


def test_attention_write_to_block_cache():
    """Verify _write_to_block_cache writes entries at correct positions."""
    cache = torch.zeros(1, 2, 4, 8, dtype=torch.float32)
    block_offsets = torch.tensor([[0, 1]], dtype=torch.int32)
    data = torch.tensor([[1.0] * 8, [2.0] * 8, [3.0] * 8])

    Attention._write_to_block_cache(cache, block_offsets, seq_idx=0, block_size=4,
                                    data=data, start_pos=2, layer_id=0)

    # start_pos=2 -> abs_pos 2,3,4 -> block 0 offset 2, block 0 offset 3, block 1 offset 0
    assert torch.equal(cache[0, 0, 2], torch.tensor([1.0] * 8))
    assert torch.equal(cache[0, 0, 3], torch.tensor([2.0] * 8))
    assert torch.equal(cache[0, 1, 0], torch.tensor([3.0] * 8))


def test_attention_gather_compressed():
    """Verify _gather_compressed collects entries from block cache."""
    compressed_kv = torch.arange(1 * 2 * 4 * 8, dtype=torch.float32).view(1, 2, 4, 8)
    block_offsets = torch.tensor([[0, 1]], dtype=torch.int32)

    gathered = Attention._gather_compressed(compressed_kv, block_offsets, seq_idx=0,
                                            block_size=4, num_compressed=5, layer_id=0)
    assert gathered.shape == (5, 8)
    # entry 4 -> block 1 offset 0
    assert torch.equal(gathered[4], compressed_kv[0, 1, 0])


def test_attention_gather_compressed_empty():
    """Verify _gather_compressed returns None when num_compressed is 0."""
    compressed_kv = torch.zeros(1, 2, 4, 8, dtype=torch.float32)
    block_offsets = torch.tensor([[0, 1]], dtype=torch.int32)
    assert Attention._gather_compressed(compressed_kv, block_offsets, seq_idx=0,
                                        block_size=4, num_compressed=0, layer_id=0) is None
