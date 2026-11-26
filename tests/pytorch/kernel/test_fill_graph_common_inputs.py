import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_padded_query_length_assert(monkeypatch):
    from lmdeploy.pytorch.kernels.cuda.step_metadata.fill_graph_common_inputs import fill_graph_common_inputs

    monkeypatch.setenv('TRITON_DEBUG', '1')
    options = dict(device='cuda', dtype=torch.int32)
    tokens = torch.tensor([[8, 9, 10]], **options)
    positions = torch.tensor([[0, 1, 0]], **options)
    blocks = torch.tensor([[0], [1]], **options)
    starts = torch.tensor([0, 2], **options)
    lengths = torch.tensor([2, 1], **options)
    qkv_lens = torch.empty((3, 4), **options)
    cu_seqlens = torch.empty((2, 5), **options)
    fill_graph_common_inputs(tokens, positions, blocks, starts, lengths, lengths, torch.empty_like(tokens),
                             torch.empty_like(positions), torch.empty((4, 1), **options), qkv_lens, cu_seqlens,
                             pad_query_len=1)

    # Only the second request ends at the real token count. Padding also contributes to the graph offsets.
    assert cu_seqlens.cpu().tolist() == [[0, 2, 3, 4, 5], [0, 2, 3, 4, 5]]
