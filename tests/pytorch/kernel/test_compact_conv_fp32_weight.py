# Copyright (c) OpenMMLab. All rights reserved.
"""GLM keeps FP32 convolution weights with BF16 activations/state (including
TP4)."""
import itertools
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('lengths', [(1, 3), (72, 137)])
@pytest.mark.parametrize('alias', [False, True])
@pytest.mark.parametrize('with_bias', [False, True])
def test_glm_compact_prefill_fp32_weight(lengths, alias, with_bias):
    from fla.modules.conv.triton.ops import causal_conv1d_fwd

    from lmdeploy.pytorch.backends.cuda.causal_conv1d import compact_causal_conv1d

    torch.manual_seed(2113)
    channels, width = 6144, 4  # GLM5.3 TP4 Q/K/V, not a synthetic smaller channel count.
    x = torch.randn(1, sum(lengths), channels, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(channels, 1, width, device='cuda', dtype=torch.float32) / 4
    bias = torch.randn(channels, device='cuda', dtype=torch.float32) if with_bias else None
    backing = torch.randn(8, 3, channels, width - 1, device='cuda', dtype=x.dtype)
    before = backing.clone()
    pool = backing[:, 1]
    src = torch.tensor([2, 3], device='cuda')
    dst = src.clone() if alias else torch.tensor([4, 5], device='cuda')
    cp = torch.tensor([6, 7], device='cuda')
    offsets = torch.tensor([min(n, 64) for n in lengths], device='cuda')
    cu = torch.tensor([0, *itertools.accumulate(lengths)], device='cuda')
    seq = torch.repeat_interleave(torch.arange(2, device='cuda', dtype=torch.int32),
                                 torch.tensor(lengths, device='cuda'))
    initial = torch.nn.functional.pad(pool[src].contiguous(), (1, 0))
    expected, _ = causal_conv1d_fwd(x, weight[:, 0], bias=bias, residual=None, initial_state=initial,
                                   output_final_state=True, activation='silu', cu_seqlens=cu)
    # An accidental BF16 weight cast is observably different; retaining FP32 is required.
    narrowed, _ = causal_conv1d_fwd(x, weight[:, 0].to(x.dtype), bias=bias, residual=None,
                                   initial_state=initial, output_final_state=True,
                                   activation='silu', cu_seqlens=cu)
    assert not torch.equal(expected, narrowed)
    meta = SimpleNamespace(is_decoding=False,
        compact_state=SimpleNamespace(input_ids=src, output_ids=dst[:, None], allow_input_alias=alias),
        compact_prefill=SimpleNamespace(seq_idx=seq, cu_seqlens=cu,
                                       checkpoint_ids=cp, checkpoint_offsets=offsets))
    actual = compact_causal_conv1d(x, weight, bias, pool, meta, 'silu')
    # Independent FLA legacy math, FP32 accumulation / BF16 output. One output ULP
    # permits exp/FMA differences; history storage must remain bit-exact below.
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.0001)
    for row, length in enumerate(lengths):
        start = sum(lengths[:row])
        for target, end in ((dst[row], length), (cp[row], int(offsets[row]))):
            history = torch.cat((before[src[row], 1], x[0, start:start+end].T), -1)
            torch.testing.assert_close(pool[target], history[:, -3:], rtol=0, atol=0)
    untouched = torch.ones(8, device='cuda', dtype=torch.bool)
    untouched[torch.cat((dst, cp))] = False
    assert torch.equal(pool[untouched], before[:, 1][untouched])
    assert torch.equal(backing[:, 0], before[:, 0]) and torch.equal(backing[:, 2], before[:, 2])


def test_fp32_weight_is_not_rounded_before_multiply():
    from lmdeploy.pytorch.kernels.cuda.causal_conv1d import indexed_causal_conv1d_fwd

    x = torch.full((1, 1, 64), 1.5, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
    weight = torch.zeros(64, 4, device='cuda')
    weight[:, -1] = 1.006
    pool = torch.zeros(3, 64, 3, device='cuda', dtype=x.dtype)
    ids = torch.tensor([1], device='cuda')
    finals = torch.tensor([2], device='cuda')
    actual = indexed_causal_conv1d_fwd(x, weight, pool, ids, finals,
        seq_idx=torch.zeros(1, device='cuda', dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 1], device='cuda'))
    expected = (x.float() * weight[0, -1]).to(x.dtype)
    narrowed = (x.float() * weight[0, -1].to(x.dtype).float()).to(x.dtype)
    assert not torch.equal(expected, narrowed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
