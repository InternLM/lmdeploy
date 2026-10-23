# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason='requires CUDA device with cc>=9.0')


def _stable_reference(scores, lengths, k, fill):
    expected = torch.full((scores.size(0), k), fill, dtype=torch.int32)
    for row, raw_length in enumerate(lengths):
        length = max(0, min(raw_length, scores.size(1)))
        # Stable descending order breaks score ties by lower logical ID; the
        # kernel emits those selected IDs in ascending order, not score order.
        selected = torch.argsort(scores[row, :length], descending=True, stable=True)[:k]
        expected[row, :selected.numel()] = selected.sort().values.to(torch.int32)
    return expected


def _boundary_scores(k, width):
    scores = torch.full((4, width), -1.0)
    # Make the cutoff bisect equal runs at a warp and a 1024-element boundary.
    # Higher-scoring IDs live later, so selecting the first K IDs cannot pass.
    for row, boundary in enumerate((32, 1024)):
        scores[row, boundary - 5:boundary + 6] = 0
        scores[row, 2048:2048 + k - 5] = 1
    scores[2].fill_(1.0)
    scores[3].zero_()
    scores[3, 1::2] = -0.0
    return scores


@pytest.mark.parametrize('k', [512, 2048])
@pytest.mark.parametrize('padded', [False, True])
def test_deterministic_topk_cutoff_ties_and_partial_tiles(k, padded):
    from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import sparse_index_topk

    # Neither the physical width nor these valid lengths end on a 1024 tile.
    width, fill = 4 * 1024 + 13, -7
    cpu_scores = _boundary_scores(k, width)
    storage = torch.empty(4, width + (128 if padded else 0), device='cuda')
    scores = storage[:, :width]
    scores.copy_(cpu_scores)
    lengths = [width, width - 1, width - 7, width]
    q_seqlens = torch.ones(4, device='cuda', dtype=torch.int32)
    kv_seqlens = torch.tensor(lengths, device='cuda', dtype=torch.int32)
    expected = _stable_reference(cpu_scores, lengths, k, fill)
    # The fifth and final selected tie sits immediately before each boundary.
    assert expected[0, :5].tolist() == list(range(27, 32))
    assert expected[1, :5].tolist() == list(range(1019, 1024))
    for _ in range(3):
        out = sparse_index_topk(scores, q_seqlens, kv_seqlens, k, fill=fill, deterministic=True)
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize('k', [512, 2048])
def test_deterministic_topk_graph_updates_scores_and_valid_lengths(k):
    from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import sparse_index_topk

    rows, width, fill = 12, 4 * 1024 + 13, -7
    generator = torch.Generator().manual_seed(5027)
    cpu_scores = torch.randn(rows, width, generator=generator)
    storage = torch.empty(rows, width + 128, device='cuda')
    scores = storage[:, :width]
    scores.copy_(cpu_scores)
    q_seqlens = torch.ones(rows, device='cuda', dtype=torch.int32)
    lengths = torch.full((rows,), width, device='cuda', dtype=torch.int32)
    sparse_index_topk(scores, q_seqlens, lengths, k, fill=fill, deterministic=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = sparse_index_topk(scores, q_seqlens, lengths, k, fill=fill, deterministic=True)

    length_cases = [-1, 0, 1, k - 1, k, k + 1, 1023, 1025, 3073, width - 1, width, width + 7]
    for iteration, pattern in enumerate(('random', 'ties', 'all_equal', 'signed_zero', 'infinities', 'boundary')):
        cpu_scores.normal_(generator=generator)
        if pattern == 'ties':
            cpu_scores.round_()
        elif pattern == 'all_equal':
            cpu_scores.fill_(1)
        elif pattern == 'signed_zero':
            cpu_scores.zero_()
            cpu_scores[:, 1::2] = -0.0
        elif pattern == 'infinities':
            cpu_scores[:, ::7] = torch.inf
            cpu_scores[:, 1::7] = -torch.inf
        elif pattern == 'boundary':
            cpu_scores.copy_(_boundary_scores(k, width).repeat(3, 1))
        # Rotate lengths so the same captured row grows/shrinks across K and
        # tile boundaries, including transitions into and out of empty rows.
        next_lengths = length_cases[iteration:] + length_cases[:iteration]
        scores.copy_(cpu_scores)
        lengths.copy_(torch.tensor(next_lengths, device='cuda', dtype=torch.int32))
        graph.replay()
        expected = _stable_reference(cpu_scores, next_lengths, k, fill)
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
