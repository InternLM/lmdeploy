# Copyright (c) OpenMMLab. All rights reserved.
"""Compare routing and exact FP8 message bytes, including live graph
padding."""
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.moe.glm53_prepare import glm53_prepare
from lmdeploy.pytorch.kernels.cuda.moe.route_noaux_tc import fused_noaux_tc_routing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
                                reason='SM90 CUDA is required')


def check(logits, bias, x, send, weights, ids, valid_count):
    ref_w, ref_ids = fused_noaux_tc_routing(logits, bias, num_experts=288, n_group=1,
                                            topk_group=1, top_k=8, routed_scaling_factor=1.0)
    ref_ids[valid_count:] = -1
    torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, ref_w * 2.5, rtol=0, atol=0)
    values = x.float().reshape(-1, 32, 128)
    amax = values.abs().amax(-1).clamp_min(1e-4)
    # Tensor/Tensor division mirrors CUDA's 448.f / amax. Scalar/Tensor
    # reverse division uses reciprocal followed by multiply in PyTorch.
    scale = torch.div(torch.full_like(amax, 448.0), amax)
    ref_q = (values * scale[..., None]).to(torch.float8_e4m3fn).reshape(-1, 4096)
    actual_q = send[:len(x), 16:4112].contiguous().view(torch.float8_e4m3fn)
    actual_s = send[:len(x), 4112:].contiguous().view(torch.float32)
    torch.testing.assert_close(actual_q.view(torch.uint8), ref_q.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(actual_s, amax * (1.0 / 448.0), rtol=0, atol=0)
    headers = send[:len(x), :16].contiguous().view(torch.int32)
    torch.testing.assert_close(headers[:, 0], torch.arange(len(x), device=x.device, dtype=torch.int32))
    assert not headers[:, 1:].any()


@pytest.mark.parametrize('rows', [1, 2, 4, 6, 16, 24, 96, 192])
@pytest.mark.parametrize('kind', ['normal', 'outlier', 'zero'])
def test_fused_prepare_matches_route_and_fp8_bytes(rows, kind):
    x = torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16)
    logits = torch.randn(rows, 288, device='cuda')
    bias = torch.randn(288, device='cuda') * .05
    if kind == 'outlier':
        x[:, ::64] *= 1000
    elif kind == 'zero':
        x.zero_()
        logits.zero_()  # Also check the original router's tie-breaking order.
        bias.zero_()
    valid = torch.tensor(rows // 2, device='cuda', dtype=torch.int32)
    send = torch.empty(rows, 4240, device='cuda', dtype=torch.uint8)
    weights, ids = glm53_prepare(logits, bias, x, send, valid)
    check(logits, bias, x, send, weights, ids, rows // 2)


def test_fused_prepare_graph_updates_values_and_padding():
    x = torch.randn(16, 4096, device='cuda', dtype=torch.bfloat16)
    logits = torch.randn(16, 288, device='cuda')
    bias = torch.randn(288, device='cuda') * .05
    send = torch.empty(16, 4240, device='cuda', dtype=torch.uint8)
    valid = torch.tensor(16, device='cuda', dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        glm53_prepare(logits, bias, x, send, valid)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        weights, ids = glm53_prepare(logits, bias, x, send, valid)
    for count in (0, 1, 7, 16):
        valid.fill_(count)
        x.normal_()
        logits.normal_()
        graph.replay()
        check(logits, bias, x, send, weights, ids, count)
