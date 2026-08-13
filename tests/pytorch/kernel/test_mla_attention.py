# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends.cuda.attention.mla import FlashMLAImpl, NSAIndicesUpdater
from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend


def test_nsa_decode_indices_update_repeats_block_table_for_spec_decode():
    updater = NSAIndicesUpdater()
    nsa_indices = torch.tensor([[0, 17, -1], [32, 1, 16], [0, 33, 47], [32, 1, 16]])
    block_offsets = torch.tensor([[100, 101, 102], [200, 201, 202]])

    output = updater._update_decode_multi_impl(nsa_indices, block_offsets, block_size=16)

    expected = torch.tensor([[[1600, 1617, -1], [1632, 1601, 1616]],
                             [[3200, 3233, 3247], [3232, 3201, 3216]]])
    assert torch.equal(output, expected)


def test_nsa_decode_indices_update_keeps_single_token_shape():
    updater = NSAIndicesUpdater()
    nsa_indices = torch.tensor([[0, 17], [32, -1]])
    block_offsets = torch.tensor([[100, 101, 102], [200, 201, 202]])

    output = updater._update_decode_single_impl(nsa_indices, block_offsets, block_size=16)

    expected = torch.tensor([[[1600, 1617]], [[3232, -1]]])
    assert torch.equal(output, expected)


def test_bf16_sparse_decode_uses_strided_cache_view():
    impl = object.__new__(FlashMLAImpl)
    impl.nsa_updater = NSAIndicesUpdater()
    impl.nsa_updater._update_decode_strided_multi_func = impl.nsa_updater._update_decode_strided_multi_impl
    impl._flash_mla_sparse = Mock(return_value=torch.empty(4, 64, 512, dtype=torch.bfloat16))

    query = torch.empty(4, 64, 576, dtype=torch.bfloat16)
    block_size = 16
    block_elements = block_size * 576
    storage = torch.empty(3, block_elements + 128, dtype=torch.bfloat16)
    k_cache = storage[:, :block_elements].view(3, block_size, 1, 576)
    nsa_indices = torch.tensor([[0, 17], [32, -1], [0, 33], [32, 1]])
    metadata = SimpleNamespace(is_decoding=True,
                               q_seqlens=torch.tensor([2, 2]),
                               block_offsets=torch.tensor([[1, 2, 0], [2, 0, 1]]))

    impl._decode_bf16_sparse_flash_mla(query, k_cache, nsa_indices, metadata)

    sparse_query, storage_k, global_indices = impl._flash_mla_sparse.call_args.args
    assert sparse_query is query
    assert storage_k.untyped_storage().data_ptr() == k_cache.untyped_storage().data_ptr()
    assert storage_k.stride() == (64, 576, 1)
    expected = torch.tensor([[[146, 301]], [[0, -1]], [[292, 155]], [[146, 301]]])
    assert torch.equal(global_indices, expected)


def test_bf16_sparse_decode_strided_cache_matches_contiguous_cache():
    pytest.importorskip('flash_mla')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip('FlashMLA BF16 sparse attention requires an SM90 GPU')

    impl = object.__new__(FlashMLAImpl)
    impl.scale = 576**-0.5
    impl.flash_mla_sparse_fwd = None
    impl.nsa_updater = NSAIndicesUpdater()
    impl.nsa_updater._update_decode_strided_multi_func = impl.nsa_updater._update_decode_strided_multi_impl

    batch_size = 2
    query_len = 2
    block_size = 64
    num_blocks = 4
    block_elements = block_size * 576
    storage = torch.empty(num_blocks, block_elements + 128, dtype=torch.bfloat16, device='cuda')
    k_cache = storage[:, :block_elements].view(num_blocks, block_size, 1, 576)
    k_cache.normal_(std=0.1)
    query = torch.randn(batch_size * query_len, 64, 576, dtype=torch.bfloat16, device='cuda')
    nsa_indices = torch.arange(128, dtype=torch.int32, device='cuda').repeat(batch_size * query_len, 1)
    block_offsets = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device='cuda')
    metadata = SimpleNamespace(is_decoding=True,
                               q_seqlens=torch.full((batch_size, ), query_len, dtype=torch.int32, device='cuda'),
                               block_offsets=block_offsets)

    output = impl._decode_bf16_sparse_flash_mla(query, k_cache, nsa_indices, metadata)

    contiguous_k = k_cache.flatten(0, 1)
    contiguous_indices = impl.nsa_updater._update_decode_multi_impl(nsa_indices, block_offsets, block_size)
    contiguous_indices = contiguous_indices.flatten(0, 1)[:, None]
    expected = impl._flash_mla_sparse(query, contiguous_k, contiguous_indices)
    torch.testing.assert_close(output, expected)


def test_fp8_sparse_decode_pads_tp_query_heads_for_aligned_kernel():
    impl = object.__new__(FlashMLAImpl)
    impl.causal = True
    impl.scale = 1.0
    impl.v_head_size = 512
    impl.nsa_updater = Mock()
    impl.nsa_updater.update_decode.return_value = torch.zeros(2, 3, 4, dtype=torch.int32)
    impl.flash_mla_with_kvcache = Mock(
        return_value=(torch.empty(2, 3, 64, 512, dtype=torch.bfloat16), None))

    query = torch.empty(6, 8, 576, dtype=torch.bfloat16)
    k_cache = torch.empty(2, 16, 1, 656, dtype=torch.uint8)
    metadata = SimpleNamespace(
        q_seqlens=torch.tensor([3, 3]),
        kv_seqlens=torch.tensor([16, 16]),
        block_offsets=torch.zeros(2, 1, dtype=torch.int32),
        tile_scheduler_metadata=object(),
        num_splits=None,
    )

    output = impl._decode_paged_flash_mla(query, k_cache, torch.zeros(6, 4, dtype=torch.int32), metadata)

    padded_query = impl.flash_mla_with_kvcache.call_args.args[0]
    assert padded_query.shape == (2, 3, 64, 576)
    assert output.shape == (6, 8, 512)


def test_bf16_sparse_decode_skips_fp8_flashmla_metadata():
    metadata = SimpleNamespace(block_offsets=torch.tensor([[0, 1]], dtype=torch.int64))
    model_config = SimpleNamespace(use_mla_fp8_cache=False, mla_index_topk=2048)

    CudaOpsBackend.update_meta_flashmla(metadata, model_config, decoding_query_len=5)

    assert metadata.block_offsets.dtype == torch.int32
    assert not hasattr(metadata, 'tile_scheduler_metadata')
