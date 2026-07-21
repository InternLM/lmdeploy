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

    output = updater._update_decode_impl(nsa_indices, block_offsets, max_q_seqlen=2, block_size=16)

    expected = torch.tensor([[[1600, 1617, -1], [1632, 1601, 1616]],
                             [[3200, 3233, 3247], [3232, 3201, 3216]]])
    assert torch.equal(output, expected)


def test_nsa_decode_indices_update_keeps_single_token_shape():
    updater = NSAIndicesUpdater()
    nsa_indices = torch.tensor([[0, 17], [32, -1]])
    block_offsets = torch.tensor([[100, 101, 102], [200, 201, 202]])

    output = updater._update_decode_impl(nsa_indices, block_offsets, max_q_seqlen=1, block_size=16)

    expected = torch.tensor([[[1600, 1617]], [[3232, -1]]])
    assert torch.equal(output, expected)


def test_bf16_sparse_decode_uses_global_cache_view():
    impl = object.__new__(FlashMLAImpl)
    impl.nsa_updater = NSAIndicesUpdater()
    impl.nsa_updater._update_decode_func = impl.nsa_updater._update_decode_impl
    impl._flash_mla_sparse = Mock(return_value=torch.empty(4, 64, 512, dtype=torch.bfloat16))

    query = torch.empty(4, 64, 576, dtype=torch.bfloat16)
    k_cache = torch.empty(3, 16, 1, 576, dtype=torch.bfloat16)
    nsa_indices = torch.tensor([[0, 17], [32, -1], [0, 33], [32, 1]])
    metadata = SimpleNamespace(is_decoding=True,
                               q_seqlens=torch.tensor([2, 2]),
                               block_offsets=torch.tensor([[1, 2, 0], [2, 0, 1]]))

    impl._decode_bf16_sparse_flash_mla(query, k_cache, nsa_indices, metadata)

    sparse_query, flatten_k, global_indices = impl._flash_mla_sparse.call_args.args
    assert sparse_query is query
    assert flatten_k.shape == (48, 1, 576)
    expected = torch.tensor([[[16, 33]], [[0, -1]], [[32, 17]], [[16, 33]]])
    assert torch.equal(global_indices, expected)


def test_bf16_sparse_decode_skips_fp8_flashmla_metadata():
    metadata = SimpleNamespace(block_offsets=torch.tensor([[0, 1]], dtype=torch.int64))
    model_config = SimpleNamespace(use_mla_fp8_cache=False, mla_index_topk=2048)

    CudaOpsBackend.update_meta_flashmla(metadata, model_config, decoding_query_len=5)

    assert metadata.block_offsets.dtype == torch.int32
    assert not hasattr(metadata, 'tile_scheduler_metadata')


def test_nsa_topk_uses_per_query_causal_kv_lengths():
    if not torch.cuda.is_available():
        pytest.skip('requires a CUDA runtime to import the NSA backend')
    from lmdeploy.pytorch.backends.cuda.nsa import _get_causal_k_seqlens

    q_seqlens = torch.tensor([2, 3])
    cu_seqlen_q = torch.tensor([0, 2, 5])
    k_seqlens = torch.tensor([5, 7])

    output = _get_causal_k_seqlens(cu_seqlen_q, q_seqlens, k_seqlens, num_tokens=5)

    assert torch.equal(output, torch.tensor([4, 5, 5, 6, 7]))
