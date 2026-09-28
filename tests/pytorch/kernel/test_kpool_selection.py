# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
                                reason='DeepGEMM FP8 scoring requires Hopper or newer.')


def reference_selection(query, weight, cache, q_lens, kv_lens, blocks):
    from lmdeploy.pytorch.backends.cuda.kpool import kpool_score_contiguous_cuda, kpool_select_groups_cuda
    from lmdeploy.pytorch.nn.kpool import kpool_expand_selected_groups, kpool_read_packed_cache

    parts = []
    offset = 0
    for request, (q, kv) in enumerate(zip(q_lens.tolist(), kv_lens.tolist())):
        seq = kv - q + torch.arange(1, q + 1, device=query.device)
        lengths = seq // 4
        keys, scales = kpool_read_packed_cache(cache, blocks[request], kv // 4, 4)
        scores = kpool_score_contiguous_cuda(query[offset:offset + q], weight[offset:offset + q], keys, scales, lengths)
        selected = kpool_select_groups_cuda(scores, lengths, group_topk=512, max_group_length=kv // 4)
        parts.append(kpool_expand_selected_groups(selected, lengths, 4, 2048, seq_lens=seq))
        offset += q
    return torch.cat(parts)


def make_case(lengths, histories, tied=False, page_size=64):
    from lmdeploy.pytorch.nn.kpool import kpool_packed_cache_views

    torch.manual_seed(97)
    batch, rows = len(lengths), sum(lengths)
    kv = [q + h for q, h in zip(lengths, histories)]
    columns = max(1, (max(kv) + 63) // 64)
    blocks = (torch.randperm(batch * columns, device='cuda') + 1).reshape(batch, columns)
    cache = torch.empty(batch * columns + 1, page_size, 1, 132, device='cuda', dtype=torch.uint8)
    keys, scales = kpool_packed_cache_views(cache, 128)
    keys.copy_(torch.randn(keys.shape, device='cuda').to(torch.float8_e4m3fn))
    scales.uniform_(0.001, 0.05)
    query = torch.randn(rows, 32, 128, device='cuda').to(torch.float8_e4m3fn)
    weight = torch.zeros(rows, 32, device='cuda') if tied else torch.rand(rows, 32, device='cuda')
    return (query, weight, cache, torch.tensor(lengths, device='cuda'),
            torch.tensor(kv, device='cuda'), blocks, sum(kv))


@pytest.mark.parametrize('lengths,histories', [
    ([0], [0]), ([1, 2], [0, 0]), ([0, 13, 17], [0, 2079, 6287]), ([511, 513], [3, 4096]),
    ([1, 4, 13], [2048, 8189, 32756]),
    ([8192], [0]), ([1] * 256, [2] * 256), ([1] * 256, [3] * 256),
])
@pytest.mark.parametrize('page_size', [16, 64])
@pytest.mark.parametrize('tied', [False, True])
def test_kpool_prefill_selection_matches_request_loop(lengths, histories, tied, page_size):
    pytest.importorskip('deep_gemm')
    from lmdeploy.pytorch.backends.cuda.kpool import kpool_select_prefill_cuda

    args = make_case(lengths, histories, tied, page_size)
    expected = reference_selection(*args[:-1])
    actual = kpool_select_prefill_cuda(*args, 4, 2048)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('page_size', [16, 64])
def test_kpool_prefill_selection_graph_replays_ragged_metadata(page_size):
    pytest.importorskip('deep_gemm')
    from lmdeploy.pytorch.backends.cuda.kpool import kpool_select_prefill_cuda

    args = make_case([13, 17, 0], [2079, 6278, 0], page_size=page_size)
    query, weight, cache, q_lens, kv_lens, blocks, capacity = args
    kpool_select_prefill_cuda(*args, 4, 2048)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = kpool_select_prefill_cuda(*args, 4, 2048)
    for q, kv in [([0, 11, 19], [0, 4097, 4000]), ([17, 0, 13], [3107, 0, 4013])]:
        assert sum(kv) <= capacity
        q_lens.copy_(torch.tensor(q, device='cuda'))
        kv_lens.copy_(torch.tensor(kv, device='cuda'))
        weight.uniform_()
        expected = reference_selection(query, weight, cache, q_lens, kv_lens, blocks)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('page_size', [16, 64])
def test_kpool_compact_mqa_column_contract(monkeypatch, page_size):
    """Exercise the modern adapter even when the installed wheel is legacy.

    This tests column normalization, not the unavailable modern native kernel.
    """
    from types import SimpleNamespace

    from lmdeploy.pytorch.backends.cuda import kpool

    args = make_case([13, 17, 0], [2079, 6278, 0], page_size=page_size)
    expected = reference_selection(*args[:-1])
    real = kpool._get_deep_gemm().fp8_mqa_logits
    native_compact = kpool._mqa_has_local_columns()

    def compact(q, kv, weights, start, end, *, clean_logits, max_seqlen_k):
        if native_compact:
            return real(q, kv, weights, start, end, clean_logits=clean_logits, max_seqlen_k=max_seqlen_k)
        absolute = real(q, kv, weights, start, end, clean_logits=clean_logits)
        ids = start.long()[:, None] + torch.arange(max_seqlen_k, device=q.device)[None]
        return absolute.gather(1, ids.clamp(max=absolute.size(1) - 1)).contiguous()

    monkeypatch.setattr(kpool, '_get_deep_gemm', lambda: SimpleNamespace(fp8_mqa_logits=compact))
    monkeypatch.setattr(kpool, '_mqa_has_local_columns', lambda: True)
    actual = kpool.kpool_select_prefill_cuda(*args, 4, 2048)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
