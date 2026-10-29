# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.moe.route_mask import mask_invalid_routes


@pytest.mark.parametrize('rows,top_k', [(1, 1), (4, 8), (16, 8), (128, 8), (5, 6)])
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_mask_changes_after_graph_capture(rows, top_k, dtype):
    """One graph must handle active, dummy, partial, then active steps."""
    original = torch.arange(rows * top_k, device='cuda', dtype=dtype).view(rows, top_k) % 288
    original[0, 0] = -1
    routes = original.clone()
    valid = torch.tensor(rows, device='cuda', dtype=torch.int32)
    mask_invalid_routes(routes, valid)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mask_invalid_routes(routes, valid)
    for count in [rows, 0, max(1, rows // 2), rows, 0]:
        routes.copy_(original)
        valid.fill_(count)
        graph.replay()
        expected = original.clone()
        expected[count:] = -1
        torch.testing.assert_close(routes, expected, rtol=0, atol=0)


def test_empty_routes():
    routes = torch.empty((0, 8), device='cuda', dtype=torch.int64)
    mask_invalid_routes(routes, torch.zeros((), device='cuda', dtype=torch.int32))
    assert routes.shape == (0, 8)
