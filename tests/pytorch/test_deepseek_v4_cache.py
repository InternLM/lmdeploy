# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.backends.attention import V4AttentionMetadata
from lmdeploy.pytorch.backends.cuda.attention.v4 import TritonV4AttentionBuilder
from lmdeploy.pytorch.models.deepseek_v4 import Attention


def test_attention_write_cache_entries():
    cache = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    block_offsets = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    batch_idx = torch.tensor([0, 0, 1], dtype=torch.long)
    positions = torch.tensor([2, 4, 7], dtype=torch.long)
    values = torch.tensor([[1.0] * 8, [2.0] * 8, [3.0] * 8], dtype=torch.float32)

    Attention._write_cache_entries(cache, block_offsets, batch_idx, positions, values, block_size=4)

    assert torch.equal(cache[0, 2], torch.tensor([1.0] * 8, dtype=torch.bfloat16))
    assert torch.equal(cache[1, 0], torch.tensor([2.0] * 8, dtype=torch.bfloat16))
    assert torch.equal(cache[1, 3], torch.tensor([3.0] * 8, dtype=torch.bfloat16))


def test_attention_gather_cache_entries_with_padding():
    cache = torch.arange(2 * 4 * 8, dtype=torch.bfloat16).view(2, 4, 8)
    block_offsets = torch.tensor([[0, 1]], dtype=torch.int32)
    positions = torch.tensor([[2, 3, 4, -1]], dtype=torch.long)
    gathered = Attention._gather_cache_entries(cache, block_offsets, positions, block_size=4)

    assert gathered.shape == (1, 4, 8)
    assert torch.equal(gathered[0, 0], cache[0, 2])
    assert torch.equal(gathered[0, 1], cache[0, 3])
    assert torch.equal(gathered[0, 2], cache[1, 0])
    assert torch.equal(gathered[0, 3], torch.zeros(8, dtype=torch.bfloat16))


def test_v4_attention_impl_decode_reads_named_caches():

    class _Kernel:

        @staticmethod
        def sparse_attn(q, kv, attn_sink, topk_idxs, scale):
            assert q.shape == (1, 1, 2, 8)
            assert kv.shape == (1, 6, 8)
            assert topk_idxs.shape == (1, 1, 6)
            return torch.ones_like(q)

    impl = TritonV4AttentionBuilder.build(head_size=8,
                                          scale=1.0,
                                          window_size=4,
                                          compress_ratio=4,
                                          kernel_mod=_Kernel())
    raw_cache = torch.arange(2 * 4 * 8, dtype=torch.bfloat16).view(2, 4, 8)
    compressed_cache = torch.arange(2 * 4 * 8, dtype=torch.bfloat16).view(2, 4, 8) + 100
    meta = V4AttentionMetadata(is_decoding=True,
                               block_offsets=torch.tensor([[0, 1]], dtype=torch.int32),
                               q_seqlens=torch.tensor([1], dtype=torch.int32),
                               kv_seqlens=torch.tensor([6], dtype=torch.int32),
                               state_ids=torch.tensor([0], dtype=torch.int64),
                               topk_indices=torch.arange(6, dtype=torch.int64).view(1, 1, 6),
                               window_positions=torch.tensor([[2, 3, 4, 5]], dtype=torch.long),
                               window_lens=torch.tensor([4], dtype=torch.int32),
                               valid_mask=torch.tensor([True]),
                               compress_ratio=4,
                               compressed_positions=torch.tensor([[0, 1]], dtype=torch.long),
                               compressed_valid_mask=torch.tensor([[True, True]]))
    query = torch.ones((1, 1, 2, 8), dtype=torch.bfloat16)
    attn_sink = torch.zeros(2, dtype=torch.float32)
    out = impl.forward_decode(query,
                              raw_cache,
                              attn_sink,
                              meta,
                              block_size=4,
                              compressed_kv_cache=compressed_cache)
    assert out.shape == query.shape
