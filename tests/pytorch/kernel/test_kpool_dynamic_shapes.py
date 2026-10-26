# Copyright (c) OpenMMLab. All rights reserved.
"""Request-dependent dimensions must preserve bounds, layouts and graph
replay."""
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_indexed_key_cache
from lmdeploy.pytorch.kernels.cuda.kpool import (
    gather_kpool_token_tail,
    kpool_prefill_metadata,
    prepare_kpool_decode_metadata,
    rotate_kpool_query,
    score_paged,
    write_kpool_token_cache,
)
from lmdeploy.pytorch.nn.kpool import kpool_packed_cache_views

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def page_table(batch, pages, strided=False):
    storage = torch.arange(batch * (pages * 2 + 7), device='cuda', dtype=torch.int32)
    storage = storage.reshape(batch, -1)
    return storage[:, :pages * 2:2] if strided else storage[:, :pages].contiguous()


@pytest.mark.parametrize('rows', [0, 1, 16, 17, 255, 256, 257, 1025, 8192])
def test_prefill_bounds_and_rotation(rows):
    q = torch.tensor([rows // 2, 0, rows - rows // 2], device='cuda', dtype=torch.int32)
    history = torch.tensor([64, 128, 192], device='cuda', dtype=torch.int32)
    counts, starts, seq, lengths, begin, end = kpool_prefill_metadata(q, history + q, rows, 4)
    counts_ref = (history + q) // 4
    starts_ref = counts_ref.cumsum(0) - counts_ref
    seq_ref = torch.cat([torch.arange(int(h) + 1, int(h + n) + 1, device='cuda', dtype=torch.int32)
                         for h, n in zip(history.cpu(), q.cpu())])
    begin_ref = torch.repeat_interleave(starts_ref, q.long())
    for actual, expected in zip((counts, starts, seq, lengths, begin, end),
                                (counts_ref, starts_ref, seq_ref, seq_ref // 4, begin_ref,
                                 begin_ref + seq_ref // 4)):
        torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)
    query = torch.randn(rows, 128, device='cuda', dtype=torch.bfloat16)
    expected = query.float()
    d = torch.arange(128, device='cuda')
    for level in range(7):
        partner = expected[:, d ^ (1 << level)]
        expected = torch.where((d & (1 << level)) == 0, expected + partner, partner - expected)
    torch.testing.assert_close(rotate_kpool_query(query), (expected * 128**-.5).to(query.dtype), rtol=0, atol=0)


@pytest.mark.parametrize('pages', [1, 16, 17, 513, 529, 545])
@pytest.mark.parametrize('strided', [False, True])
def test_decode_metadata_graph_replay(pages, strided):
    blocks = page_table(2, pages, strided)
    q = torch.tensor([6, 0], device='cuda', dtype=torch.int32)
    kv = torch.tensor([69, 0], device='cuda', dtype=torch.int32)
    for with_scores in (False, True):

        def call():
            return prepare_kpool_decode_metadata(q, kv, blocks, 12, 4, with_scores, page_step=4)

        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = call()
        for delta in (0, 17):
            kv.copy_(torch.tensor([69 + delta, delta], device='cuda', dtype=kv.dtype))
            graph.replay()
            seq = ((kv - q).long()[:, None] + torch.arange(1, 7, device='cuda')).flatten()
            groups = seq // 4
            table = blocks[:, ::4].repeat_interleave(6, dim=0) if with_scores else blocks.new_empty((12, 0))
            expected = (seq, groups, groups[:, None].int(), groups[:, None].clamp_min(1).int(), table)
            for actual, reference in zip(result, expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize('pages', [1, 16, 17, 513, 529, 545])
@pytest.mark.parametrize('strided', [False, True])
def test_token_cache_bounds(pages, strided):
    blocks = page_table(2, pages, strided)
    blocks[0].fill_(2)
    blocks[1].fill_(0)
    cache = torch.randn(3, 64, 2, 128, device='cuda', dtype=torch.bfloat16)
    q = torch.tensor([6, 3], device='cuda', dtype=torch.int32)
    history = torch.tensor([pages * 64 - 3, 66], device='cuda', dtype=torch.int32)
    kv = history + q
    keys, scores, ids = gather_kpool_token_tail(cache, blocks, q, kv, 4)
    expected_k, expected_s = torch.zeros_like(keys), torch.zeros_like(scores)
    for r, h in enumerate(history.cpu().tolist()):
        for j in range(h % 4):
            pos = h - h % 4 + j
            if pos < pages * 64:
                block = int(blocks[r, pos // 64])
                expected_k[r, j] = cache[block, pos % 64, 0]
                expected_s[r, j] = cache[block, pos % 64, 1]
    torch.testing.assert_close(keys, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(scores, expected_s, rtol=0, atol=0)
    torch.testing.assert_close(ids, torch.arange(2, device='cuda'), rtol=0, atol=0)
    new_k = torch.randn(9, 128, device='cuda', dtype=cache.dtype)
    new_s = torch.randn_like(new_k)
    starts = torch.tensor([0, 6, 9], device='cuda', dtype=torch.int32)
    expected = cache.clone()
    for r, h in enumerate(history.cpu().tolist()):
        for j in range(int(q[r])):
            pos = h + j
            if pos < pages * 64:
                block = int(blocks[r, pos // 64])
                expected[block, pos % 64, 0] = new_k[int(starts[r]) + j]
                expected[block, pos % 64, 1] = new_s[int(starts[r]) + j]
    write_kpool_token_cache(cache, new_k, new_s, blocks, q, kv, starts)
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@pytest.mark.parametrize('fp8', [False, True])
@pytest.mark.parametrize('explicit_requests', [False, True])
@pytest.mark.parametrize('pages', [1, 16, 17, 513, 529, 545])
def test_indexed_cache_strided_and_masked(pages, fp8, explicit_requests):
    blocks = page_table(2, pages, True)
    blocks[0].fill_(2)
    blocks[1].fill_(0)
    if fp8:
        packed = torch.zeros(3, 16, 1, 132, device='cuda', dtype=torch.uint8)
        cache, scale_cache = kpool_packed_cache_views(packed, 128)
    else:
        cache = torch.zeros(3, 16, 2, 128, device='cuda', dtype=torch.bfloat16)[:, :, 1]
        scale_cache = torch.zeros(3, 16, 2, 1, device='cuda')[:, :, 1]
    keys = torch.randn(6, 128, device='cuda').to(cache.dtype)
    scales = torch.rand(6, device='cuda')
    groups = torch.tensor([0, 0, 17, 17, pages * 16 + 2, -1], device='cuda')
    valid = torch.tensor([True, True, True, False, True, False], device='cuda')
    requests = torch.tensor([1, 0, 0, 1, 1, 0], device='cuda') if explicit_requests else None
    expected, expected_scales = cache.clone(), scale_cache.clone()
    for row in range(6):
        if bool(valid[row]):
            group = int(groups[row])
            request = int(requests[row]) if requests is not None else row % 2
            block = int(blocks[request, min(max(group // 16, 0), pages - 1)])
            expected[block, group % 16] = keys[row]
            expected_scales[block, group % 16, 0] = scales[row]
    fill_indexed_key_cache(keys, scales, groups, valid, blocks, cache, scale_cache, request_ids=requests)
    torch.testing.assert_close(cache.float(), expected.float(), rtol=0, atol=0)
    torch.testing.assert_close(scale_cache, expected_scales, rtol=0, atol=0)


@pytest.mark.parametrize('pages', [1, 16, 17, 513, 529, 545])
@pytest.mark.parametrize('strided', [False, True])
def test_score_pages_graph_replay(pages, strided):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip('FP8 dot requires Hopper or newer')
    torch.manual_seed(7)
    rows, heads, dim = 3, 16, 128
    packed = torch.zeros(7, 16, 1, dim + 4, device='cuda', dtype=torch.uint8)
    keys, scales = kpool_packed_cache_views(packed, dim)
    keys.copy_(torch.randn(keys.shape, device='cuda').to(keys.dtype))
    scales.copy_(torch.rand(scales.shape, device='cuda'))
    query = torch.randn(rows, heads, dim, device='cuda').to(keys.dtype)
    weights = torch.randn(rows, heads, device='cuda')
    # score_paged supports padded row strides, with contiguous columns.
    table = page_table(rows, pages + 7)[:, :pages] if strided else page_table(rows, pages)
    table.remainder_(7)
    lengths = torch.tensor([0, min(19, pages * 16), pages * 16 - 1], device='cuda', dtype=torch.int32)
    def call():
        return score_paged(query, weights, packed, lengths, table)

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call()
    for delta in (0, 1):
        lengths.add_(delta).clamp_max_(pages * 16)
        table.add_(delta).remainder_(7)
        graph.replay()
        for r in range(rows):
            length = int(lengths[r])
            positions = torch.arange(length, device='cuda')
            physical = table[r, positions // 16].long()
            selected = keys.float()[physical, positions % 16]
            dots = query[r].float() @ selected.T
            expected = (dots.clamp_min(0) * weights[r, :, None]).sum(0)
            expected *= scales[physical, positions % 16, 0]
            torch.testing.assert_close(result[r, :length], expected, rtol=1e-4, atol=1e-4)
