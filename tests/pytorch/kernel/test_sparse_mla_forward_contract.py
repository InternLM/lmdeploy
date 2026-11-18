# Copyright (c) OpenMMLab. All rights reserved.
"""CPU contract tests; the FlashMLA kernel itself is deliberately mocked."""
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends.cuda.attention.sparse_mla import FlashMLASparseImpl


@pytest.mark.parametrize('index_width', [512, 517, 640])
@pytest.mark.parametrize('return_lse', [False, True])
@pytest.mark.parametrize('lse_trailing_dim', [False, True])
def test_sparse_forward_preserves_live_tail_and_dcp_contract(index_width, return_lse, lse_trailing_dim):
    impl = object.__new__(FlashMLASparseImpl)
    impl.scale = 0.25
    query = torch.randn(2, 40, 8)
    kv = torch.randn(1024, 1, 8)
    indices = torch.arange(index_width, dtype=torch.int32).repeat(2, 1)
    # DCP compacted rows can include an empty partition. Padding must not
    # change the effective lengths or drop the GLM live tail beyond top-k.
    counts = torch.tensor([0, index_width], dtype=torch.int32) if return_lse else None
    output = torch.randn(2, 64, 8)
    lse = torch.randn(2, 64)
    kernel_lse = lse.unsqueeze(-1) if lse_trailing_dim else lse
    impl.flash_mla_sparse_fwd = Mock(return_value=(output, None, kernel_lse))

    result = impl._flash_mla_sparse_forward(query, kv, indices, return_lse=return_lse, topk_length=counts)

    args, kwargs = impl.flash_mla_sparse_fwd.call_args
    padded_query, actual_kv, padded_indices = args
    assert actual_kv is kv
    assert padded_query.shape == (2, 64, 8)
    torch.testing.assert_close(padded_query[:, :40], query)
    assert torch.count_nonzero(padded_query[:, 40:]) == 0
    assert padded_indices.shape[-1] == (index_width + 127) // 128 * 128
    torch.testing.assert_close(padded_indices[:, :index_width], indices)
    assert torch.all(padded_indices[:, index_width:] == -1)
    assert kwargs['sm_scale'] == impl.scale
    if return_lse:
        assert kwargs['topk_length'] is counts
        torch.testing.assert_close(result[0], output[:, :40])
        torch.testing.assert_close(result[1], lse[:, :40])
    else:
        assert 'topk_length' not in kwargs
        torch.testing.assert_close(result, output[:, :40])
