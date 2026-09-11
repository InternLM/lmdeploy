# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch


def _requires_sm90_cuda() -> bool:
    return not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9


pytestmark = pytest.mark.skipif(_requires_sm90_cuda(), reason='requires CUDA device with cc>=9.0')


def _assert_topk_ids(scores: torch.Tensor, out: torch.Tensor,
                     seqlens: list[int], k: int, fill: int = -1):
    scores = scores.cpu()
    out = out.cpu()
    score_width = scores.size(1)

    for row, raw_seqlen in enumerate(seqlens):
        seqlen = min(raw_seqlen, score_width)
        row_out = out[row]
        valid = row_out[row_out != fill]

        if seqlen <= k:
            expected = torch.arange(seqlen, dtype=torch.int32)
            torch.testing.assert_close(row_out[:seqlen], expected)
            if seqlen < k:
                assert row_out[seqlen:].eq(fill).all()
            continue

        expected = torch.topk(scores[row, :seqlen], k=k, largest=True, sorted=False).indices
        assert valid.numel() == k
        torch.testing.assert_close(valid.sort().values, expected.to(torch.int32).sort().values)


def test_sparse_index_topk_matches_torch_topk_and_fill():
    from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import (
        is_sparse_index_topk_supported,
        sparse_index_topk,
    )

    assert is_sparse_index_topk_supported(512)

    device = 'cuda'
    k = 512
    fill = -7
    score_width = 1024
    seqlens = [0, 17, 512, 513, 900]
    generator = torch.Generator(device=device).manual_seed(20260709)
    scores = torch.randn(len(seqlens), score_width, device=device,
                         dtype=torch.float32, generator=generator)
    scores += torch.arange(score_width, device=device, dtype=torch.float32) * 1e-6

    q_seqlens = torch.ones(len(seqlens), device=device, dtype=torch.int64)
    kv_seqlens = torch.tensor(seqlens, device=device, dtype=torch.int32)

    out = sparse_index_topk(scores, q_seqlens, kv_seqlens, k=k, fill=fill)
    assert out.shape == (len(seqlens), k)
    assert out.dtype == torch.int32
    _assert_topk_ids(scores, out, seqlens, k, fill=fill)


def test_sparse_index_topk_accepts_padded_score_stride():
    from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import sparse_index_topk

    device = 'cuda'
    k = 512
    score_width = 1024
    padded_width = 1280
    seqlens = [600, 777, 1024, 321]
    generator = torch.Generator(device=device).manual_seed(20260710)
    storage = torch.randn(len(seqlens), padded_width, device=device,
                          dtype=torch.float32, generator=generator)
    scores = storage[:, :score_width]
    scores += torch.arange(score_width, device=device, dtype=torch.float32) * 1e-6
    assert not scores.is_contiguous()

    q_seqlens = torch.ones(len(seqlens), device=device, dtype=torch.int64)
    kv_seqlens = torch.tensor(seqlens, device=device, dtype=torch.int32)

    out = sparse_index_topk(scores, q_seqlens, kv_seqlens, k=k)
    _assert_topk_ids(scores, out, seqlens, k)


def test_sparse_index_topk_expands_batch_kv_seqlens_for_prefill():
    from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import sparse_index_topk

    device = 'cuda'
    k = 512
    score_width = 1536
    q_seqlens_list = [2, 3]
    batch_kv_seqlens = [640, 1100]
    row_seqlens = [640, 640, 1100, 1100, 1100]

    generator = torch.Generator(device=device).manual_seed(260619348)
    scores = torch.randn(sum(q_seqlens_list), score_width, device=device,
                         dtype=torch.float32, generator=generator)
    scores += torch.arange(score_width, device=device, dtype=torch.float32) * 1e-6

    q_seqlens = torch.tensor(q_seqlens_list, device=device, dtype=torch.int64)
    kv_seqlens = torch.tensor(batch_kv_seqlens, device=device, dtype=torch.int32)

    out = sparse_index_topk(scores, q_seqlens, kv_seqlens, k=k)
    _assert_topk_ids(scores, out, row_seqlens, k)


@pytest.mark.parametrize('k', [512, 2048])
def test_sparse_dcp_local_topk_stable_ties_choose_lower_positions(k):
    from lmdeploy.pytorch.kernels.cuda.sparse_index_dcp_topk import sparse_dcp_local_topk

    device = 'cuda'
    score_width = 3 * k
    scores = torch.zeros(2, score_width, device=device, dtype=torch.float32)
    scores[1, :256] = 1
    q_seqlens = torch.ones(2, device=device, dtype=torch.int64)
    kv_seqlens = torch.tensor([score_width, score_width],
                              device=device,
                              dtype=torch.int32)

    out = sparse_dcp_local_topk(scores, q_seqlens, kv_seqlens, k=k)

    expected = torch.arange(k, device=device, dtype=torch.int32).expand(2, -1)
    assert torch.equal(out, expected)


def test_sparse_index_topk_cuda_graph_capture():
    from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import sparse_index_topk

    device = 'cuda'
    k = 512
    score_width = 1024
    seqlens = [600, 777, 1024, 321]
    generator = torch.Generator(device=device).manual_seed(512)
    scores = torch.randn(len(seqlens), score_width, device=device,
                         dtype=torch.float32, generator=generator)
    q_seqlens = torch.ones(len(seqlens), device=device, dtype=torch.int64)
    kv_seqlens = torch.tensor(seqlens, device=device, dtype=torch.int32)

    # Warm the TileLang specialization and PyTorch's graph-aware allocator.
    out = sparse_index_topk(scores, q_seqlens, kv_seqlens, k=k)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = sparse_index_topk(scores, q_seqlens, kv_seqlens, k=k)

    graph.replay()
    torch.cuda.synchronize()
    _assert_topk_ids(scores, out, seqlens, k)


@pytest.mark.parametrize(('dcp_size', 'k', 'local_width'),
                         [(2, 512, 700), (4, 2048, 2300)])
def test_sparse_dcp_global_topk_matches_global_stable_topk(
        dcp_size, k, local_width):
    from lmdeploy.pytorch.kernels.cuda.sparse_index_dcp_topk import (
        pack_dcp_topk_candidates,
        sparse_dcp_global_topk,
    )

    device = 'cuda'
    num_rows = 2
    generator = torch.Generator(device=device).manual_seed(20260902)
    global_scores = torch.randn(num_rows,
                                dcp_size * local_width,
                                dtype=torch.float32,
                                device=device,
                                generator=generator)
    # Exercise stable global-position ties at the selection threshold.
    global_scores[1].zero_()
    packed_by_rank = []
    for rank in range(dcp_size):
        local_scores = global_scores[:, rank::dcp_size].contiguous()
        local_indices = torch.argsort(local_scores,
                                      dim=1,
                                      descending=True,
                                      stable=True)[:, :k].to(torch.int32)
        packed_by_rank.append(
            pack_dcp_topk_candidates(
                local_scores,
                local_indices,
                dcp_world_rank=(dcp_size, rank)))

    gathered = torch.stack(packed_by_rank)
    actual = sparse_dcp_global_topk(gathered, k)
    if k == 512:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = sparse_dcp_global_topk(gathered, k)
        graph.replay()
        torch.cuda.synchronize()
    expected = torch.argsort(global_scores,
                             dim=1,
                             descending=True,
                             stable=True)[:, :k].to(torch.int32)
    candidate_ids = gathered.view(torch.int32)[..., 1]
    candidate_ids = candidate_ids.permute(1, 0, 2).reshape(num_rows, -1)
    expected_in_candidate_order = torch.stack([
        row_ids[torch.isin(row_ids, selected_ids)]
        for row_ids, selected_ids in zip(candidate_ids, expected)
    ])
    assert torch.equal(actual, expected_in_candidate_order)


def test_pack_dcp_topk_candidates_large_score_stride():
    from lmdeploy.pytorch.kernels.cuda.sparse_index_dcp_topk import pack_dcp_topk_candidates

    # Reach a row offset of 2**31 elements without filling the 8 GiB storage.
    if torch.cuda.mem_get_info()[0] < 10 * 2**30:
        pytest.skip('requires 10 GiB free GPU memory for large-stride addressing')
    scores = torch.empty_strided((257, 8), (2**23, 1), dtype=torch.float32, device='cuda')
    scores.copy_(torch.arange(8, dtype=torch.float32, device='cuda')[None])
    indices = torch.full((257, 512), -1, dtype=torch.int32, device='cuda')
    indices[:, :8] = torch.arange(8, dtype=torch.int32, device='cuda')

    packed = pack_dcp_topk_candidates(scores, indices, dcp_world_rank=(2, 1))
    torch.testing.assert_close(packed[..., 0][:, :8], scores)
    assert torch.equal(packed.view(torch.int32)[..., 1][:, :8], indices[:, :8] * 2 + 1)
    assert torch.isneginf(packed[..., 0][:, 8:]).all()
    assert (packed.view(torch.int32)[..., 1][:, 8:] == -1).all()


def test_sparse_dcp_global_topk_preserves_int32_ids_and_padding():
    from lmdeploy.pytorch.kernels.cuda.sparse_index_dcp_topk import sparse_dcp_global_topk

    k = 512
    gathered = torch.empty(2, 1, k, 2, dtype=torch.float32, device='cuda')
    gathered[..., 0].fill_(-torch.inf)
    gathered.view(torch.int32)[..., 1].fill_(-1)
    ids = torch.tensor([2**24 + 1, 2**24 + 3, 7],
                       dtype=torch.int32,
                       device='cuda')
    gathered[0, 0, :2, 0] = 1.0
    gathered[1, 0, 0, 0] = 2.0
    gathered.view(torch.int32)[0, 0, :2, 1].copy_(ids[:2])
    gathered.view(torch.int32)[1, 0, 0, 1].copy_(ids[2])

    actual = sparse_dcp_global_topk(gathered, k)
    assert torch.equal(actual[0, :3], ids)
    assert (actual[0, 3:] == -1).all()
