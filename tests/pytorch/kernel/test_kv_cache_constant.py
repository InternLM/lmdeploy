# Copyright (c) OpenMMLab. All rights reserved.
import math

import pytest
import torch


@pytest.mark.parametrize('quant_policy', [4, 8])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('head_dim', [32, 96])
@pytest.mark.parametrize('kv_layout', ['bshd', 'bhsd'])
@pytest.mark.parametrize('constant', [0.0, 1.25, -2.5, None], ids=['zero', 'positive', 'negative', 'random'])
def test_constant_kv_cache(quant_policy, dtype, head_dim, kv_layout, constant):
    """Constant heads must survive cache fill, flatten and GQA decoding."""
    from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache
    from lmdeploy.pytorch.kernels.cuda.flatten_kv_cache import flatten_kv_cache
    from lmdeploy.pytorch.kernels.cuda.pagedattention import flash_attn_with_kvcache

    torch.manual_seed(5155)
    length, block_size, num_heads, num_kv_heads = 19, 16, 4, 2
    k = torch.rand(length, num_kv_heads, head_dim, dtype=dtype, device='cuda') - 0.5
    v = torch.rand_like(k) - 0.5
    # Mix constant and ordinary heads, including a page boundary and decode writes.
    rows = [0, 3, 16, 18]
    if constant is not None:
        k[rows, 0] = constant
        v[rows, 1] = constant

    width = head_dim if quant_policy == 8 else head_dim // 2
    kc = torch.zeros(2, block_size, num_kv_heads, width, dtype=torch.uint8, device='cuda')
    vc = torch.zeros_like(kc)
    ksz = torch.zeros(2, block_size, num_kv_heads, 2, dtype=dtype, device='cuda')
    vsz = torch.zeros_like(ksz)
    if kv_layout == 'bhsd':
        kc, vc, ksz, vsz = [x.transpose(1, 2).contiguous() for x in (kc, vc, ksz, vsz)]
    table = torch.tensor([[1, 0]], dtype=torch.int32, device='cuda')
    starts = torch.tensor([0], dtype=torch.int32, device='cuda')
    lengths = torch.tensor([length], dtype=torch.int32, device='cuda')
    kwargs = dict(k_scales_zeros=ksz, v_scales_zeros=vsz, quant_policy=quant_policy, kv_layout=kv_layout)

    # Prefill across two pages, then append individual decode tokens.
    for start, end in [(0, 17), (17, 18), (18, 19)]:
        q_lens = torch.tensor([end - start], dtype=torch.int32, device='cuda')
        kv_lens = torch.tensor([end], dtype=torch.int32, device='cuda')
        fill_kv_cache(k[start:end], v[start:end], kc, vc, starts, q_lens, kv_lens, end - start, table, **kwargs)

    def logical_cache(cache):
        if kv_layout == 'bhsd':
            cache = cache.transpose(1, 2)
        return cache[table[0].long()].flatten(0, 1)[:length]

    decoded = []
    for states, cache, metadata, head in [(k, kc, ksz, 0), (v, vc, vsz, 1)]:
        codes = logical_cache(cache)
        params = logical_cache(metadata)
        assert torch.isfinite(params).all()
        assert (params[..., 0] > 0).all()
        if quant_policy == 4:
            codes = torch.cat([codes & 15, codes >> 4], dim=-1)
        # Independent decode of the stored representation, evaluated in FP64.
        restored = (codes.double() - params[..., 1, None].double()) * params[..., 0, None].double()
        if constant is not None:
            torch.testing.assert_close(restored[rows, head], states[rows, head].double(), atol=0, rtol=0)
            torch.testing.assert_close(params[rows, head, 0], torch.ones_like(params[rows, head, 0]), atol=0, rtol=0)
            torch.testing.assert_close(params[rows, head, 1], -states[rows, head, 0], atol=0, rtol=0)
            assert (codes[rows, head] == 0).all()
        # Account separately for code rounding and metadata storage precision.
        # This bound also checks the ordinary random heads in every case.
        span = states.double().amax(-1) - states.double().amin(-1)
        bound = span / (2**quant_policy - 1) + 4 * torch.finfo(dtype).eps * states.double().abs().amax(-1)
        assert ((restored - states.double()).abs().amax(-1) <= bound).all()
        decoded.append(restored)

    flat_k, flat_v = flatten_kv_cache(kc, vc, lengths, table, out_size=length,
                                     out_dtype=dtype, flatten_kv_layout='shd', **kwargs)
    tolerance = 2 * torch.finfo(dtype).eps
    for actual, expected in zip((flat_k, flat_v), decoded):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)

    # Zero queries reproduce the issue's exact uniform-attention oracle; random
    # queries also exercise the constant Key vectors in the score calculation.
    for zero_query in (True, False):
        q = torch.randn(1, num_heads, head_dim, dtype=dtype, device='cuda')
        if zero_query:
            q.zero_()
        out = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=lengths, page_table=table,
                                     max_seqlen_q=1, **kwargs)
        assert torch.isfinite(out).all()
        keys, values = [x.repeat_interleave(num_heads // num_kv_heads, dim=1) for x in decoded]
        logits = torch.einsum('bhd,lhd->bhl', q.double(), keys) / math.sqrt(head_dim)
        expected = torch.einsum('bhl,lhd->bhd', logits.softmax(-1), values)
        torch.testing.assert_close(out.double(), expected, atol=tolerance, rtol=tolerance)
