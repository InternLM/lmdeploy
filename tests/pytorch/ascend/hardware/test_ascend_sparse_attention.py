# ruff: noqa: E402
# Copyright (c) 2026, DeepLink. All rights reserved.

import pytest
import torch

torch_npu = pytest.importorskip('torch_npu')

if not torch.npu.is_available():
    pytest.skip('Ascend NPU is required', allow_module_level=True)

from lmdeploy.pytorch.kernels.ascend.ops import fill_kv_cache, lightning_indexer, sparse_flash_attention

DTYPE = torch.bfloat16
DEVICE = torch.device('npu')
BLOCK_SIZE = 128
TOPK = 2048


def _randn(shape):
    return torch.randn(shape, dtype=torch.float32).to(device=DEVICE, dtype=DTYPE)


def _metadata():
    cumulative_q = torch.tensor([3, 5], dtype=torch.int32)
    kv_lengths = torch.tensor([3, 2], dtype=torch.int32)
    block_table = torch.tensor([[0], [1]], dtype=torch.int32)
    return cumulative_q, kv_lengths, block_table


def _sparse_attention_reference(query, query_rope, combined_cache, indices, cumulative_q, kv_lengths, block_table):
    query = query.float().cpu()
    query_rope = query_rope.float().cpu()
    combined_cache = combined_cache.float().cpu()
    indices = indices.cpu()
    cumulative_q = cumulative_q.tolist()
    kv_lengths = kv_lengths.tolist()
    block_table = block_table.tolist()
    output = torch.empty_like(query)
    q_start = 0

    for seq_idx, q_end in enumerate(cumulative_q):
        kv_length = kv_lengths[seq_idx]
        num_blocks = (kv_length + BLOCK_SIZE - 1) // BLOCK_SIZE
        cache = torch.cat([combined_cache[block_table[seq_idx][block]] for block in range(num_blocks)])[:kv_length, 0]
        key = cache[:, : query.size(-1)]
        key_rope = cache[:, query.size(-1) :]
        for token_idx in range(q_start, q_end):
            selected = indices[token_idx, 0]
            selected = selected[(selected >= 0) & (selected < kv_length)].long()
            scores = torch.einsum('hd,kd->hk', query[token_idx], key[selected])
            scores += torch.einsum('hd,kd->hk', query_rope[token_idx], key_rope[selected])
            probs = torch.softmax(scores * (query.size(-1) ** -0.5), dim=-1)
            output[token_idx] = torch.einsum('hk,kd->hd', probs, key[selected])
        q_start = q_end
    return output


def test_lightning_indexer_matches_native_torch_npu():
    torch.manual_seed(20260817)
    query = _randn((5, 32, 128))
    key_cache = _randn((2, BLOCK_SIZE, 1, 128))
    weights = _randn((5, 32))
    cumulative_q, kv_lengths, block_table = _metadata()
    cumulative_q_device = cumulative_q.to(DEVICE)
    kv_lengths_device = kv_lengths.to(DEVICE)
    block_table_device = block_table.to(DEVICE)

    expected, _ = torch_npu.npu_lightning_indexer(
        query=query,
        key=key_cache,
        weights=weights,
        actual_seq_lengths_query=cumulative_q_device,
        actual_seq_lengths_key=kv_lengths_device,
        block_table=block_table_device,
        layout_query='TND',
        layout_key='PA_BSND',
        sparse_count=TOPK,
        sparse_mode=3,
    )
    actual = lightning_indexer(
        query,
        key_cache,
        weights,
        actual_seq_lengths_query=cumulative_q_device,
        actual_seq_lengths_key=kv_lengths_device,
        block_table=block_table_device,
        sparse_count=TOPK,
    )

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    assert actual.shape == (5, 1, TOPK)
    assert actual.dtype == torch.int32


def test_sparse_flash_attention_matches_native_torch_npu():
    torch.manual_seed(20260818)
    cumulative_q, kv_lengths, block_table = _metadata()
    cumulative_q_device = cumulative_q.to(DEVICE)
    kv_lengths_device = kv_lengths.to(DEVICE)
    block_table_device = block_table.to(DEVICE)
    index_query = _randn((5, 32, 128))
    index_cache = _randn((2, BLOCK_SIZE, 1, 128))
    index_weights = _randn((5, 32))
    indices, _ = torch_npu.npu_lightning_indexer(
        query=index_query,
        key=index_cache,
        weights=index_weights,
        actual_seq_lengths_query=cumulative_q_device,
        actual_seq_lengths_key=kv_lengths_device,
        block_table=block_table_device,
        layout_query='TND',
        layout_key='PA_BSND',
        sparse_count=TOPK,
        sparse_mode=3,
    )

    query = _randn((5, 64, 512))
    query_rope = _randn((5, 64, 64))
    combined_cache = _randn((2, BLOCK_SIZE, 1, 576))
    key = combined_cache[..., :512]
    key_rope = combined_cache[..., 512:]

    expected, _, _ = torch_npu.npu_sparse_flash_attention(
        query=query,
        key=key,
        value=key,
        sparse_indices=indices,
        scale_value=512**-0.5,
        block_table=block_table_device,
        actual_seq_lengths_query=cumulative_q_device,
        actual_seq_lengths_kv=kv_lengths_device,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=1,
        layout_query='TND',
        layout_kv='PA_BSND',
        sparse_mode=3,
        attention_mode=2,
    )
    actual = sparse_flash_attention(
        query,
        key,
        key,
        indices,
        512**-0.5,
        block_table=block_table_device,
        actual_seq_lengths_query=cumulative_q_device,
        kv_seqlens=kv_lengths_device,
        query_rope=query_rope,
        key_rope=key_rope,
    )

    torch.testing.assert_close(actual.cpu().float(), expected.cpu().float(), rtol=5e-3, atol=5e-3)
    reference = _sparse_attention_reference(
        query,
        query_rope,
        combined_cache,
        indices,
        cumulative_q,
        kv_lengths,
        block_table,
    )
    torch.testing.assert_close(actual.cpu().float(), reference, rtol=3e-2, atol=3e-2)


def test_fill_kv_cache_writes_split_mla_caches():
    torch.manual_seed(20260828)
    key = _randn((3, 1, 576))
    value = _randn((3, 1, 512))
    rope_cache = torch.zeros((2, BLOCK_SIZE, 1, 64), dtype=DTYPE, device=DEVICE)
    nope_cache = torch.zeros((2, BLOCK_SIZE, 1, 512), dtype=DTYPE, device=DEVICE)
    slot_indices = torch.tensor([0, 7, BLOCK_SIZE + 3], dtype=torch.int32, device=DEVICE)

    actual_rope, actual_nope = fill_kv_cache(
        key,
        value,
        rope_cache,
        nope_cache,
        slot_indices,
        k_scales_zeros=(),
        v_scales_zeros=(),
        quant_bits=0,
    )

    flat_rope = actual_rope.flatten(0, 1)
    flat_nope = actual_nope.flatten(0, 1)
    torch.testing.assert_close(flat_rope[slot_indices].cpu(), key[..., -64:].cpu())
    torch.testing.assert_close(flat_nope[slot_indices].cpu(), value.cpu())
