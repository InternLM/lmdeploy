# Copyright (c) OpenMMLab. All rights reserved.
"""Validate numerical and CUDA Graph correctness across page-table widths."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('steps', [1, 6])
@pytest.mark.parametrize('page_step', [1, 4])
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('with_scores', [False, True])
def test_decode_metadata_dynamic_pages(steps, page_step, dtype, with_scores):
    from lmdeploy.pytorch.kernels.cuda.kpool import prepare_kpool_decode_metadata
    q = torch.full((2,), steps, dtype=dtype, device='cuda')
    kv = torch.tensor([-2, 12345 + steps], dtype=dtype, device='cuda')
    lengths = (kv.long() - q.long())[:, None] + torch.arange(1, steps + 1, device='cuda')
    groups = lengths.flatten() // 4
    for columns in [1, 2, 17, 257, 388, 453, 512]:
        # Both column and row strides differ from the ordinary contiguous bank.
        owner = torch.arange(2 * (columns + 3) * 2, dtype=torch.int32, device='cuda').reshape(2, columns + 3, 2)
        blocks = owner[:, :columns, 1]
        result = prepare_kpool_decode_metadata(q, kv, blocks, 2 * steps, 4,
                                               with_scores=with_scores, page_step=page_step)
        expected_table = (blocks[:, ::page_step].repeat_interleave(steps, 0)
                          if with_scores else blocks[:0].reshape(2 * steps, 0))
        expected = (lengths.flatten(), groups, groups.int()[:, None],
                    groups.clamp_min(1).int()[:, None], expected_table)
        for actual, ref in zip(result, expected):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)


@pytest.mark.parametrize('explicit_requests', [False, True])
@pytest.mark.parametrize('page_step', [1, 4])
@pytest.mark.parametrize('dtype', [torch.uint8, torch.float8_e4m3fn])
def test_index_cache_dynamic_pages(explicit_requests, page_step, dtype):
    from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_indexed_key_cache
    rows, batch, page, width = 6, 2, 16, 32
    keys = torch.arange(rows * width, device='cuda').reshape(rows, width).remainder(64).to(dtype)
    scales = torch.arange(rows, dtype=torch.float32, device='cuda') + 1
    requests = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.int64, device='cuda') if explicit_requests else None
    valid = torch.tensor([True, True, True, False, True, False], device='cuda')
    for columns in [1, 2, 17, 257, 388, 453, 512]:
        table_owner = torch.zeros((batch, columns + 3, 2), dtype=torch.int32, device='cuda')
        blocks = table_owner[:, :columns, 1]
        blocks.copy_(torch.arange(batch * columns, dtype=torch.int32, device='cuda').reshape(batch, columns))
        # Packed layer views; untouched pages/slots must remain byte-identical.
        key_owner = torch.full((batch * columns, 2, page, width), 7, dtype=dtype, device='cuda')
        scale_owner = torch.full((batch * columns, 2, page, 1), -7., device='cuda')
        key_cache, scale_cache = key_owner[:, 1], scale_owner[:, 1]
        groups = torch.tensor([0, 31, columns * page + 2, -999, 3, 999999], device='cuda')
        expected_keys = key_owner.clone().view(torch.uint8)
        expected_scales = scale_owner.clone()
        for row in [0, 1, 2, 4]:
            request = int(requests[row]) if requests is not None else row % batch
            group = int(groups[row])
            col = max(0, min(group // page * page_step, columns - 1))
            dest = int(blocks[request, col])
            expected_keys[dest, 1, group % page] = keys[row].view(torch.uint8)
            expected_scales[dest, 1, group % page, 0] = scales[row]
        fill_indexed_key_cache(keys, scales, groups, valid, blocks, key_cache, scale_cache,
                               page_step=page_step, request_ids=requests)
        torch.testing.assert_close(key_owner.view(torch.uint8), expected_keys, rtol=0, atol=0)
        torch.testing.assert_close(scale_owner, expected_scales, rtol=0, atol=0)


@pytest.mark.parametrize('columns', [1, 388, 513])
def test_dynamic_pages_cuda_graph(columns):
    from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_indexed_key_cache
    from lmdeploy.pytorch.kernels.cuda.kpool import prepare_kpool_decode_metadata
    blocks = torch.zeros((1, columns), dtype=torch.int32, device='cuda')
    q = torch.ones(1, dtype=torch.int64, device='cuda')
    kv = q.clone()
    keys = torch.ones((1, 32), dtype=torch.uint8, device='cuda')
    scales = torch.ones(1, device='cuda')
    valid = torch.ones(1, dtype=torch.bool, device='cuda')
    cache = torch.zeros((1, 16, 32), dtype=torch.uint8, device='cuda')
    scale_cache = torch.zeros((1, 16, 1), device='cuda')

    def launch():
        result = prepare_kpool_decode_metadata(q, kv, blocks, 1, 4, page_step=1)
        fill_indexed_key_cache(keys, scales, result[1], valid, blocks, cache, scale_cache)
        return result

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = launch()
    kv.fill_(9)
    keys.fill_(13)
    graph.replay()
    torch.testing.assert_close(result[0], kv, rtol=0, atol=0)
    assert result[1].item() == 2
    assert torch.equal(cache[0, 2], keys[0])
