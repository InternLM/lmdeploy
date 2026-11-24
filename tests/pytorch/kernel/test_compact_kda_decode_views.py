# Copyright (c) OpenMMLab. All rights reserved.
"""Compact KDA decode consumes QKV views without changing state or graph
ownership."""
from types import SimpleNamespace

import pytest
import torch


def contiguous_reference(impl, mixed, gate, beta, weight, bias, alog, dtbias, conv, state, meta, heads):
    from lmdeploy.pytorch.backends.cuda.causal_conv1d import CausalConv1dTilelangImpl
    ids = meta.ssm_state_meta
    batch, width = ids.output_state_ids.shape
    mixed = CausalConv1dTilelangImpl.forward_indexed(mixed, weight, bias, conv, meta, 'silu')
    q, k, v = [x.unflatten(-1, (heads, 128)).contiguous() for x in mixed.split(heads * 128, dim=-1)]
    gate = gate.unflatten(-1, (heads, 128)).contiguous()
    beta = beta.contiguous()
    q, k, v, gate, beta = [x[0].unflatten(0, (batch, width)).contiguous() for x in (q, k, v, gate, beta)]
    out, _ = impl._decode_recurrent(q, k, v, g=gate, beta=beta, A_log=alog, dt_bias=dtbias, lower_bound=-5.,
                                    initial_state=state, output_final_state=True, state_indices=ids.input_state_ids,
                                    output_state_indices=ids.output_state_ids)
    return out.flatten(0, 1).unsqueeze(0)


@pytest.mark.parametrize('batch,heads,width', [(64, 64, 6), (64, 64, 1), (8, 64, 6), (64, 8, 6),
                                               (1, 64, 6), (1, 8, 6), (8, 64, 1), (64, 8, 1)])
@pytest.mark.parametrize('weight_dtype', [torch.bfloat16, torch.float32])
def test_compact_decode_qkv_views_full_backing_and_graph(monkeypatch, batch, heads, width, weight_dtype):
    from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl
    torch.manual_seed(202621)
    impl = CudaKdaImpl(indexed_state=True)
    channels, units = 3 * heads * 128, batch * 6 + 2
    mixed = torch.randn(1, batch * width, channels, device='cuda', dtype=torch.bfloat16) / 8
    gate = torch.randn(1, batch * width, heads * 128, device='cuda', dtype=torch.bfloat16)
    beta = torch.randn(1, batch * width, heads, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device='cuda', dtype=weight_dtype) / 8
    bias = torch.randn(channels, device='cuda', dtype=weight_dtype) / 8
    alog, dtbias = torch.zeros(heads, device='cuda'), torch.randn(heads * 128, device='cuda')
    initial_state = torch.randn(units, 3, heads, 128, 128, device='cuda') / 16
    initial_conv = torch.randn(units, 3, channels, 3, device='cuda', dtype=torch.bfloat16) / 16
    initial_state[0].zero_()
    initial_conv[0].zero_()
    actual_state, ref_state = initial_state.clone(), initial_state.clone()
    actual_conv, ref_conv = initial_conv.clone(), initial_conv.clone()
    src = 1 + torch.arange(batch, device='cuda') * 6
    dst = src[:, None] + torch.arange(1, width + 1, device='cuda')
    dst[:, -1] = src
    if batch > 1:
        src[-1] = -1
        dst[-1].fill_(-1)
    meta = SimpleNamespace(is_decoding=True,
                           ssm_state_meta=SimpleNamespace(input_state_ids=src, output_state_ids=dst))
    args = (mixed, gate, beta, weight, bias, alog, dtbias)
    expected = contiguous_reference(impl, *args, ref_conv[:, 1], ref_state[:, 1], meta, heads)
    actual = impl.forward(*args, actual_conv[:, 1], actual_state[:, 1], meta, heads, 128, -5.)
    for x, y in [(actual, expected), (actual_state, ref_state), (actual_conv, ref_conv)]:
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    actual_state.copy_(initial_state)
    actual_conv.copy_(initial_conv)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        captured = impl.forward(*args, actual_conv[:, 1], actual_state[:, 1], meta, heads, 128, -5.)
    for _ in range(2):
        mixed.add_(.03125)
        src.copy_(src.flip(0))
        dst.copy_(dst.flip(0))
        ref_state.copy_(initial_state)
        ref_conv.copy_(initial_conv)
        expected = contiguous_reference(impl, *args, ref_conv[:, 1], ref_state[:, 1], meta, heads)
        actual_state.copy_(initial_state)
        actual_conv.copy_(initial_conv)
        graph.replay()
        for x, y in [(captured, expected), (actual_state, ref_state), (actual_conv, ref_conv)]:
            torch.testing.assert_close(x, y, rtol=0, atol=0)
