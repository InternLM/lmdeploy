# Copyright (c) OpenMMLab. All rights reserved.
import sys

import pytest
import torch


def _requires_cuda() -> bool:
    return not torch.cuda.is_available()


pytestmark = pytest.mark.skipif(_requires_cuda(), reason='requires CUDA')


def test_nsa_backend_selects_sparse_topk_for_glm52():
    from lmdeploy.pytorch.backends.cuda.nsa import _get_sparse_index_topk

    assert _get_sparse_index_topk(2048).__name__ == 'sparse_index_topk'
    assert _get_sparse_index_topk(1024) is None


def test_nsa_backend_falls_back_without_tilelang(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import nsa

    nsa._get_sparse_index_topk.cache_clear()
    monkeypatch.setitem(sys.modules,
                        'lmdeploy.pytorch.kernels.cuda.sparse_index_topk', None)
    try:
        assert nsa._get_sparse_index_topk(2048) is None
    finally:
        nsa._get_sparse_index_topk.cache_clear()


def _assert_topk_ids(scores: torch.Tensor,
                     out: torch.Tensor,
                     seqlens: list[int],
                     k: int,
                     fill: int = -1):
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

        expected = torch.topk(scores[row, :seqlen],
                              k=k,
                              largest=True,
                              sorted=False).indices
        assert valid.numel() == k
        torch.testing.assert_close(valid.sort().values,
                                   expected.to(torch.int32).sort().values)


@pytest.mark.parametrize('k', [512, 2048])
def test_sparse_index_topk_matches_torch_topk_and_fill(k: int):
    from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import (
        is_sparse_index_topk_supported,
        sparse_index_topk,
    )

    assert is_sparse_index_topk_supported(k)

    device = 'cuda'
    fill = -7
    score_width = k * 2
    seqlens = [0, 17, k, k + 1, score_width - 124]
    generator = torch.Generator(device=device).manual_seed(20260709 + k)
    scores = torch.randn(len(seqlens),
                         score_width,
                         device=device,
                         dtype=torch.float32,
                         generator=generator)
    scores += torch.arange(score_width, device=device,
                           dtype=torch.float32) * 1e-6

    q_seqlens = torch.ones(len(seqlens), device=device, dtype=torch.int64)
    kv_dtype = torch.int64 if k == 2048 else torch.int32
    kv_seqlens = torch.tensor(seqlens, device=device, dtype=kv_dtype)

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
    storage = torch.randn(len(seqlens),
                          padded_width,
                          device=device,
                          dtype=torch.float32,
                          generator=generator)
    scores = storage[:, :score_width]
    scores += torch.arange(score_width, device=device,
                           dtype=torch.float32) * 1e-6
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
    scores = torch.randn(sum(q_seqlens_list),
                         score_width,
                         device=device,
                         dtype=torch.float32,
                         generator=generator)
    scores += torch.arange(score_width, device=device,
                           dtype=torch.float32) * 1e-6

    q_seqlens = torch.tensor(q_seqlens_list, device=device, dtype=torch.int64)
    kv_seqlens = torch.tensor(batch_kv_seqlens,
                              device=device,
                              dtype=torch.int32)

    out = sparse_index_topk(scores, q_seqlens, kv_seqlens, k=k)
    _assert_topk_ids(scores, out, row_seqlens, k)


@pytest.mark.parametrize('k', [512, 2048])
def test_sparse_index_topk_cuda_graph_capture(k: int):
    from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import sparse_index_topk

    device = 'cuda'
    score_width = k * 2
    seqlens = [k + 1, k + 265, score_width, k // 2]
    generator = torch.Generator(device=device).manual_seed(k)
    scores = torch.randn(len(seqlens),
                         score_width,
                         device=device,
                         dtype=torch.float32,
                         generator=generator)
    q_seqlens = torch.ones(len(seqlens), device=device, dtype=torch.int64)
    kv_seqlens = torch.tensor(seqlens, device=device, dtype=torch.int64)

    # Warm the TileLang specialization and PyTorch's graph-aware allocator.
    out = sparse_index_topk(scores, q_seqlens, kv_seqlens, k=k)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = sparse_index_topk(scores, q_seqlens, kv_seqlens, k=k)

    graph.replay()
    torch.cuda.synchronize()
    _assert_topk_ids(scores, out, seqlens, k)
