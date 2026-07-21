# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.backends.cuda.attention.mla import NSAIndicesUpdater


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


def test_nsa_topk_uses_per_query_causal_kv_lengths():
    if not torch.cuda.is_available():
        pytest.skip('requires a CUDA runtime to import the NSA backend')
    from lmdeploy.pytorch.backends.cuda.nsa import _get_causal_k_seqlens

    q_seqlens = torch.tensor([2, 3])
    cu_seqlen_q = torch.tensor([0, 2, 5])
    k_seqlens = torch.tensor([5, 7])

    output = _get_causal_k_seqlens(cu_seqlen_q, q_seqlens, k_seqlens, num_tokens=5)

    assert torch.equal(output, torch.tensor([4, 5, 5, 6, 7]))
