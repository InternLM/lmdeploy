# Copyright (c) OpenMMLab. All rights reserved.
"""Compare fused GLM routes against the native router and scaling path."""
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.moe.glm53_prepare import glm53_prepare_routes
from lmdeploy.pytorch.kernels.cuda.moe.route_noaux_tc import fused_noaux_tc_routing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')


def check(logits, bias, weights, ids, valid_count):
    ref_w, ref_ids = fused_noaux_tc_routing(logits, bias, num_experts=288, n_group=1,
                                           topk_group=1, top_k=8, routed_scaling_factor=1.0)
    ref_ids = ref_ids.to(torch.int64)
    ref_ids[valid_count:] = -1
    assert ids.dtype == torch.int64
    torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, ref_w * 2.5, rtol=0, atol=0)


@pytest.mark.parametrize('rows', [1, 2, 4, 6, 16, 24, 96, 192])
@pytest.mark.parametrize('kind', ['normal', 'outlier', 'zero'])
def test_fused_prepare_matches_routes_and_scale(rows, kind):
    logits = torch.randn(rows, 288, device='cuda')
    bias = torch.randn(288, device='cuda') * .05
    if kind == 'outlier':
        logits[:, ::8] *= 1000
    elif kind == 'zero':
        logits.zero_()
        bias.zero_()
    valid = torch.tensor(rows // 2, device='cuda', dtype=torch.int32)
    weights, ids = glm53_prepare_routes(logits, bias, valid)
    check(logits, bias, weights, ids, rows // 2)


@pytest.mark.parametrize('dummy', [False, True])
def test_fused_prepare_without_valid_count(dummy):
    logits = torch.randn(6, 288, device='cuda')
    bias = torch.randn(288, device='cuda')
    weights, ids = glm53_prepare_routes(logits, bias, None, is_dummy=dummy)
    check(logits, bias, weights, ids, 0 if dummy else 6)


def test_fused_prepare_empty_rows():
    weights, ids = glm53_prepare_routes(torch.empty(0, 288, device='cuda'),
                                       torch.zeros(288, device='cuda'), None)
    assert weights.shape == ids.shape == (0, 8)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int64


def test_fused_prepare_graph_updates_values_and_padding():
    logits = torch.randn(16, 288, device='cuda')
    bias = torch.randn(288, device='cuda') * .05
    valid = torch.tensor(16, device='cuda', dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        glm53_prepare_routes(logits, bias, valid)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        weights, ids = glm53_prepare_routes(logits, bias, valid)
    for count in (0, 1, 7, 16):
        valid.fill_(count)
        logits.normal_()
        graph.replay()
        check(logits, bias, weights, ids, count)
