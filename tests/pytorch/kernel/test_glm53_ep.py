# Copyright (c) OpenMMLab. All rights reserved.
"""Numerical contracts shared by TP and DeepEP GLM experts."""
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul, silu_and_mul_moe_ep
from lmdeploy.pytorch.kernels.cuda.moe.ep import ep_gather

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('n', [128, 768, 2048])
@pytest.mark.parametrize('limit,precise', [(None, False), (10.0, True)])
def test_masked_activation_matches_compact_rows(n, limit, precise):
    torch.manual_seed(42)
    counts = torch.tensor([0, 1, 7, 32], dtype=torch.int32, device='cuda')
    x = (torch.randn(4, 32, 2 * n, device='cuda') * 12).bfloat16()
    out = torch.full((4, 32, n), 123.0, dtype=x.dtype, device=x.device)
    silu_and_mul_moe_ep(x, counts, out, swiglu_limit=limit, precise_mul=precise)
    for e, count in enumerate(counts.tolist()):
        if count:
            ref = silu_and_mul(x[e, :count], swiglu_limit=limit, precise_mul=precise)
            torch.testing.assert_close(out[e, :count], ref, rtol=0, atol=0)
        assert torch.all(out[e, count:] == 123.0)


@pytest.mark.parametrize('scale', [1.0, 2.5])
def test_ep_gather_fp32_reduces_before_store_and_scales_once(scale):
    # BF16 stepwise accumulation loses 1 in this cancellation case.
    x = torch.tensor([1000, 1, -1000, 0], dtype=torch.bfloat16, device='cuda')
    x = x[:, None].expand(-1, 1024).contiguous()
    ids = torch.tensor([[0, 1, 2, -1]], dtype=torch.int64, device='cuda')
    weights = torch.ones(1, 4, dtype=torch.float32, device='cuda')
    index = torch.arange(4, dtype=torch.int64, device='cuda').unsqueeze(0)
    out = torch.empty((1, 1024), dtype=x.dtype, device=x.device)
    ep_gather(x, ids, weights, index, out, fp32_acc=True, output_scale=scale)
    torch.testing.assert_close(out, torch.full_like(out, scale), rtol=0, atol=0)
    ep_gather(x, ids, weights, index, out, fp32_acc=False, output_scale=scale)
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


def test_masked_glm_callback_quantization():
    from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul_masked_post_quant_fwd
    from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import _quant_fp8_launcher
    from lmdeploy.pytorch.models.glm5_next import _GLM53_COMPACT_FP8_MOE_ACT
    torch.manual_seed(43)
    x = (torch.randn(4, 32, 4096, device='cuda') * 12).bfloat16()
    counts = torch.tensor([1, 7, 32, 0], dtype=torch.int32, device='cuda')
    out = torch.empty(4, 32, 2048, dtype=torch.float8_e4m3fn, device='cuda')
    scale = torch.empty(4, 32, 16, device='cuda')
    silu_and_mul_masked_post_quant_fwd(x, out, scale, 128, counts, act_func=_GLM53_COMPACT_FP8_MOE_ACT)
    for e, count in enumerate(counts.tolist()):
        if not count:
            continue
        activated = _GLM53_COMPACT_FP8_MOE_ACT(x[e, :count])
        ref = torch.empty_like(out[e, :count])
        ref_scale = torch.empty_like(scale[e, :count])
        _quant_fp8_launcher(activated, 128, ref, ref_scale)
        torch.testing.assert_close(out[e, :count].float(), ref.float(), rtol=0, atol=0)
        torch.testing.assert_close(scale[e, :count], ref_scale, rtol=0, atol=0)
