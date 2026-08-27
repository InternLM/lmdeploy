# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends.cuda import attention as attention_module
from lmdeploy.pytorch.backends.cuda.attention import TritonAttentionBuilder
from lmdeploy.pytorch.backends.cuda.attention import mla as mla_module
from lmdeploy.pytorch.backends.cuda.attention import sparse_mla as sparse_mla_module
from lmdeploy.pytorch.backends.cuda.attention.sparse_mla import (
    FlashMLAIndexMapper,
    FlashMLASparseImpl,
)
from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend


def _disable_dynamic_compile(monkeypatch):
    monkeypatch.setattr(sparse_mla_module, '_try_dynamic_compile', lambda func, *args, **kwargs: func)


def test_flash_mla_builder_selects_sparse_impl(monkeypatch):
    dense_output = object()
    sparse_output = object()
    dense_impl = Mock(return_value=dense_output)
    sparse_impl = Mock(return_value=sparse_output)
    monkeypatch.setattr(attention_module, '_enable_fa3', lambda *args: False)
    monkeypatch.setattr(attention_module, 'use_fa3', True)
    monkeypatch.setattr(mla_module, 'FlashMLAImpl', dense_impl)
    monkeypatch.setattr(sparse_mla_module, 'FlashMLASparseImpl', sparse_impl)
    kwargs = dict(num_heads=64, head_size=576, num_kv_heads=1, use_flash_mla=True)

    assert TritonAttentionBuilder.build(**kwargs) is dense_output
    assert TritonAttentionBuilder.build(**kwargs, mla_index_topk=2048) is sparse_output
    assert sparse_impl.call_args.kwargs['use_fa3'] is True


def test_flash_mla_decode_index_mapping(monkeypatch):
    _disable_dynamic_compile(monkeypatch)
    mapper = FlashMLAIndexMapper()
    block_offsets = torch.tensor([[100, 101, 102], [200, 201, 202]])

    nsa_indices = torch.tensor([[0, 17, -1], [32, 1, 16], [0, 33, 47], [32, 1, 16]])
    output = mapper.map_paged_decode(nsa_indices, block_offsets, max_q_seqlen=2, block_size=16)
    expected = torch.tensor([[[1600, 1617, -1], [1632, 1601, 1616]],
                             [[3200, 3233, 3247], [3232, 3201, 3216]]])
    assert torch.equal(output, expected)

    nsa_indices = torch.tensor([[0, 17], [32, -1]])
    output = mapper.map_paged_decode(nsa_indices, block_offsets, max_q_seqlen=1, block_size=16)
    assert torch.equal(output, torch.tensor([[[1600, 1617]], [[3232, -1]]]))


def test_flash_mla_decode_index_mapping_caches_query_modes(monkeypatch):
    compile_func = Mock(side_effect=lambda func, *args, **kwargs: func)
    monkeypatch.setattr(sparse_mla_module, '_try_dynamic_compile', compile_func)
    mapper = FlashMLAIndexMapper()
    block_offsets = torch.tensor([[100, 101, 102], [200, 201, 202]])
    single_indices = torch.tensor([[0, 17], [32, -1]])
    multi_indices = torch.tensor([[0, 17], [32, 1], [0, 33], [32, 1]])

    mapper.map_paged_decode(single_indices, block_offsets, max_q_seqlen=1, block_size=16)
    mapper.map_paged_decode(single_indices, block_offsets, max_q_seqlen=1, block_size=16)
    mapper.map_paged_decode(multi_indices, block_offsets, max_q_seqlen=2, block_size=16)
    mapper.map_paged_decode(multi_indices.repeat_interleave(2, dim=0), block_offsets,
                            max_q_seqlen=4, block_size=16)

    assert compile_func.call_count == 2


def test_bf16_sparse_decode_uses_strided_cache_view(monkeypatch):
    _disable_dynamic_compile(monkeypatch)
    impl = object.__new__(FlashMLASparseImpl)
    impl.index_mapper = FlashMLAIndexMapper()
    impl._flash_mla_sparse_forward = Mock(return_value=torch.empty(4, 64, 512, dtype=torch.bfloat16))

    query = torch.empty(4, 64, 576, dtype=torch.bfloat16)
    block_size = 16
    block_elements = block_size * 576
    storage = torch.empty(3, block_elements + 128, dtype=torch.bfloat16)
    k_cache = storage[:, :block_elements].view(3, block_size, 1, 576)
    nsa_indices = torch.tensor([[0, 17], [32, -1], [0, 33], [32, 1]])
    metadata = SimpleNamespace(is_decoding=True,
                               q_seqlens=torch.tensor([2, 2]),
                               block_offsets=torch.tensor([[1, 2, 0], [2, 0, 1]]))

    impl._decoding_sparse_bf16(query, k_cache, nsa_indices, metadata)

    sparse_query, storage_k, global_indices = impl._flash_mla_sparse_forward.call_args.args
    assert sparse_query is query
    assert storage_k.untyped_storage().data_ptr() == k_cache.untyped_storage().data_ptr()
    assert storage_k.stride() == (64, 576, 1)
    expected = torch.tensor([[[146, 301]], [[0, -1]], [[292, 155]], [[146, 301]]])
    assert torch.equal(global_indices, expected)


def test_bf16_sparse_decode_strided_cache_matches_contiguous_cache(monkeypatch):
    pytest.importorskip('flash_mla')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip('FlashMLA BF16 sparse attention requires an SM90 GPU')

    _disable_dynamic_compile(monkeypatch)
    impl = object.__new__(FlashMLASparseImpl)
    impl.scale = 576**-0.5
    impl.flash_mla_sparse_fwd = None
    impl.index_mapper = FlashMLAIndexMapper()

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

    output = impl._decoding_sparse_bf16(query, k_cache, nsa_indices, metadata)

    contiguous_k = k_cache.flatten(0, 1)
    contiguous_indices = impl.index_mapper.map_paged_decode(nsa_indices, block_offsets, query_len, block_size)
    contiguous_indices = contiguous_indices.flatten(0, 1)[:, None]
    expected = impl._flash_mla_sparse_forward(query, contiguous_k, contiguous_indices)
    torch.testing.assert_close(output, expected)


def test_fp8_sparse_decode_pads_tp_query_heads_for_aligned_kernel():
    impl = object.__new__(FlashMLASparseImpl)
    impl.causal = True
    impl.scale = 1.0
    impl.v_head_size = 512
    impl.index_mapper = Mock()
    impl.index_mapper.map_paged_decode.return_value = torch.zeros(2, 3, 4, dtype=torch.int32)
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

    output = impl._decoding_sparse_fp8(query, k_cache, torch.zeros(6, 4, dtype=torch.int32), metadata)

    padded_query = impl.flash_mla_with_kvcache.call_args.args[0]
    assert padded_query.shape == (2, 3, 64, 576)
    assert output.shape == (6, 8, 512)


def test_bf16_sparse_decode_skips_fp8_flashmla_metadata():
    metadata = SimpleNamespace(block_offsets=torch.tensor([[0, 1]], dtype=torch.int64))
    model_config = SimpleNamespace(use_mla_fp8_cache=False, mla_index_topk=2048)

    CudaOpsBackend.update_meta_flashmla(metadata, model_config, decoding_query_len=5)

    assert metadata.block_offsets.dtype == torch.int32
    assert not hasattr(metadata, 'tile_scheduler_metadata')


def test_bf16_mla_flatten_uses_shared_k_latent_as_value():
    impl = object.__new__(mla_module.FlashMLAImpl)
    impl.v_head_size = 512
    flatten_k = torch.empty(3, 1, 576, dtype=torch.bfloat16)
    impl.flatten_kv_cache = Mock(
        return_value=(flatten_k, torch.empty(3, 1, 0, dtype=torch.bfloat16)))
    metadata = SimpleNamespace(
        kv_start_loc=torch.tensor([0]),
        kv_seqlens=torch.tensor([3]),
        block_offsets=torch.tensor([[0]]),
        kv_flatten_size=3,
        quant_policy=0,
    )

    _, flatten_v = impl._flatten_prefill_kv_cache(
        torch.empty(1, 4, 1, 576, dtype=torch.bfloat16),
        torch.empty(1, 4, 1, 0, dtype=torch.bfloat16),
        metadata,
        out_dtype=torch.bfloat16,
        kv_layout='hsd',
    )

    assert flatten_v.shape == (3, 1, 512)
    assert flatten_v.untyped_storage().data_ptr() == flatten_k.untyped_storage().data_ptr()


def test_sparse_mla_prefill_routes_by_kv_length(monkeypatch):
    dense_output = object()
    sparse_output = object()
    dense_prefill = Mock(return_value=dense_output)
    monkeypatch.setattr(mla_module.FlashMLAImpl, '_forward_prefill', dense_prefill)
    impl = object.__new__(FlashMLASparseImpl)
    impl.mla_index_topk = 2048
    impl._flatten_prefill_kv_cache = Mock(return_value=(Mock(), Mock()))
    impl._prefill_sparse = Mock(return_value=sparse_output)
    query, k_cache, v_cache, nsa_indices = (Mock() for _ in range(4))

    dense = impl._forward_prefill(query,
                                  k_cache,
                                  v_cache,
                                  SimpleNamespace(max_kv_seqlen=2048),
                                  nsa_indices=None)
    sparse = impl._forward_prefill(query,
                                   k_cache,
                                   v_cache,
                                   SimpleNamespace(max_kv_seqlen=2049),
                                   nsa_indices=nsa_indices)

    assert dense is dense_output
    assert sparse is sparse_output
    assert dense_prefill.call_args.kwargs['nsa_indices'] is None
