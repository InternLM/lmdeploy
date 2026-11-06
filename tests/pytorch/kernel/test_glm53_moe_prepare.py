# Copyright (c) OpenMMLab. All rights reserved.
"""Exact route preparation, including graph-owned dynamic padding counts."""
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.moe.route_mask import prepare_low_latency_routes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')


@pytest.mark.parametrize('rows', [0, 1, 6, 32, 192])
@pytest.mark.parametrize('mode', ['unmasked', 'dummy', 'partial'])
def test_prepare_matches_mask_and_scale(rows, mode):
    ids = torch.randint(0, 288, (rows, 8), device='cuda')
    ids[:, 0] = -1
    weights = torch.randn(rows, 8, device='cuda', dtype=torch.float32)
    expected_ids = ids.clone()
    valid = torch.tensor(rows // 2, device='cuda', dtype=torch.int32) if mode == 'partial' else None
    if mode == 'partial':
        expected_ids[rows // 2:] = -1
    elif mode == 'dummy':
        expected_ids.fill_(-1)
    original_weights = weights.clone()
    actual = prepare_low_latency_routes(ids, weights, valid, 2.5, is_dummy=mode == 'dummy')
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual, weights * 2.5, rtol=0, atol=0)
    torch.testing.assert_close(weights, original_weights, rtol=0, atol=0)


def test_prepare_graph_reads_live_valid_count():
    ids = torch.randint(0, 288, (192, 8), device='cuda')
    original_ids = ids.clone()
    weights = torch.randn(192, 8, device='cuda')
    valid = torch.tensor(192, device='cuda', dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepare_low_latency_routes(ids, weights, valid, 2.5)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        ids.copy_(original_ids)
        scaled = prepare_low_latency_routes(ids, weights, valid, 2.5)
    for count in (0, 1, 6, 191, 192):
        valid.fill_(count)
        weights.normal_()
        graph.replay()
        expected_ids = original_ids.clone()
        expected_ids[count:] = -1
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(scaled, weights * 2.5, rtol=0, atol=0)
