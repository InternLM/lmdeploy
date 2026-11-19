# Copyright (c) OpenMMLab. All rights reserved.
"""Per-channel (channel-quantized) FP8 compressed-tensors support.

Ornith-style checkpoints carry float-quantized FP8 weights with a
per-output-channel ``.weight_scale`` (BF16, shape [N, 1]) instead of the
blocked ``.weight_scale_inv`` (shape [K//128, N//128]). The scale is expanded
to the K-grouped kernel layout ``[K//128, N]`` by ``FP8Format.finalize``.
"""
import math

import pytest
import torch

from lmdeploy.turbomind.converter import _build_quantized_formats
from lmdeploy.turbomind.weight_format import FP8Format

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')


def test_per_channel_accepts_weight_scale():
    fmt = FP8Format(block_out=1)
    # weight [K, N] after normalize is irrelevant here; accepts() sees the
    # raw checkpoint tensors, weight [N, K], scale [N, 1].
    w = torch.randn(512, 2048).to(torch.float8_e4m3fn)
    s = torch.randn(512, 1).bfloat16()
    assert fmt.accepts({'.weight': w, '.weight_scale': s})
    # blocked scale must NOT be accepted by the per-channel format
    sb = torch.randn(4, 16).bfloat16()  # [K//128, N//128] for K=512, N=2048
    assert not fmt.accepts({'.weight': w, '.weight_scale_inv': sb})


def test_per_channel_rejects_blocked_scale():
    fmt = FP8Format(block_out=1)
    w = torch.randn(512, 2048).to(torch.float8_e4m3fn)
    # blocked layout does not match [N, 1]
    sb = torch.randn(4, 16).bfloat16()
    assert not fmt.accepts({'.weight': w, '.weight_scale_inv': sb})


def test_blocked_format_rejects_per_channel_scale():
    fmt = FP8Format(block_out=128)
    w = torch.randn(512, 2048).to(torch.float8_e4m3fn)
    # per-channel scale [N, 1] does not match the blocked expectation
    s = torch.randn(512, 1).bfloat16()
    assert not fmt.accepts({'.weight': w, '.weight_scale': s})


def test_per_channel_finalize_expands_to_k_groups():
    fmt = FP8Format(block_out=1)
    K, N = 2048, 512
    # finalize runs after normalize(): weight is TM layout [K, N] and the
    # per-channel scale [N, 1] has been transposed to [1, N].
    weight = torch.randn(K, N)
    scales = torch.randn(1, N)
    out = fmt.finalize({'weight': weight, 'scales': scales})
    assert out['scales'].shape == (K // 128, N)
    # every K-group row identical to the original per-channel scale
    assert torch.equal(out['scales'][0], scales.squeeze(0))
    assert torch.equal(out['scales'][-1], scales.squeeze(0))


def test_blocked_finalize_untouched():
    fmt = FP8Format(block_out=128)
    weight = torch.randn(2048, 512)
    scales = torch.randn(16, 4)
    out = fmt.finalize({'weight': weight, 'scales': scales})
    assert out['scales'].shape == (16, 4)


def test_converter_builds_per_channel_fp8():
    fmts = _build_quantized_formats('compressed-tensors', 128, fp8_block_out=1)
    assert len(fmts) == 1 and isinstance(fmts[0], FP8Format)
    assert fmts[0].block_out == 1 and fmts[0].block_in == 128
    # blocked ct float-quantized
    fmts_b = _build_quantized_formats('compressed-tensors', 128, fp8_block_out=128)
    assert isinstance(fmts_b[0], FP8Format) and fmts_b[0].block_out == 128
    # default (int4 pack-quantized) unchanged
    fmts2 = _build_quantized_formats('compressed-tensors', 128)
    assert not isinstance(fmts2[0], FP8Format)


def test_dequant_per_channel_matches_reference():
    fmt = FP8Format(block_out=1)
    K, N = 256, 64
    w = torch.randn(K, N).to(torch.float8_e4m3fn)
    # normalize-transposed per-channel scale [1, N] (was [N, 1] in checkpoint)
    scale = torch.rand(1, N) + 0.1
    out = fmt.finalize({'weight': w, 'scales': scale})
    assert out['scales'].shape == (K // 128, N)
    deq = fmt.dequant(out, torch.float16)
    ref = (w.view(torch.float8_e4m3fn).float() * scale.float().expand(K, N)).to(torch.float16)
    assert torch.allclose(deq['weight'], ref, atol=1e-3)


def test_build_linear_expands_per_channel_scale():
    """The builder path (normalize -> finalize) yields K-grouped scales."""
    from lmdeploy.turbomind.builders.linear import _build_linear

    fmt = FP8Format(block_out=1)
    N, K = 128, 256
    w = torch.randn(N, K).to(torch.float8_e4m3fn)
    s = (torch.rand(N, 1) + 0.5).float()
    linear = _build_linear(fmt, {'.weight': w, '.weight_scale': s})
    # TM layout: weight [K, N], scales [K // 128, N]
    assert linear.tensors['weight'].shape == (K, N)
    assert linear.tensors['scales'].shape == (K // 128, N)
    assert torch.equal(linear.tensors['scales'][0], linear.tensors['scales'][-1])
    deq = fmt.dequant(linear.tensors, torch.float32)
    ref = (w.float().t() * s.float().t()).to(torch.float32)
    assert torch.allclose(deq['weight'], ref, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_per_channel_fp8_native_forward_matches_reference():
    """End-to-end: per-channel FP8 checkpoint -> native GEMM vs dequant ref.

    Exercises the e4m3 groupwise family (block_out=1) on the SM80/SM86 lane;
    FP8 compute is emulated pre-SM89.
    """
    from lmdeploy.turbomind import is_available

    if not is_available():
        pytest.skip('TurboMind is not built')

    from lmdeploy.turbomind.linear import get_linear

    torch.manual_seed(0)
    fmt = FP8Format(block_out=1)
    for K, N, M in ((512, 128, 64), (2048, 256, 8)):
        # dense weight [N, K] quantized per output channel (per-tensor scale 1)
        w_dense = torch.randn(N, K) * (0.1 / math.sqrt(K))
        amax = w_dense.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = (amax / 448.0).float()
        w_fp8 = (w_dense / scale).to(torch.float8_e4m3fn)
        bits = w_fp8.view(torch.uint8)  # checkpoint-layout weight [N, K]

        with torch.cuda.device(0):
            plan = _linear_plan(fmt, torch.bfloat16)
            # post-finalized scales in checkpoint orientation [N, K//128]
            s_pp = fmt.finalize(
                {
                    'weight': fmt.normalize(bits, 'weight'),
                    'scales': fmt.normalize(scale, 'scales'),
                }
            )['scales']
            linear = get_linear()
            with linear._activate():
                weight = linear.prepare_weight(bits.cuda(), plan=plan, scales=s_pp.t().contiguous().cuda())
                x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
                exec_plan, out, _ = linear.tune(x, weight)

        # dequant reference: native path consumed [K//128, N] K-grouped scales
        s_k = s_pp.float()  # [K//128, N]
        w_deq = w_fp8.float().t() * s_k.repeat_interleave(128, dim=0)  # [K, N]
        ref = (x.float() @ w_deq.to(torch.bfloat16).cuda().float()).to(torch.bfloat16)
        err = (out.float() - ref.float()).abs().max().item()
        assert err < 1e-2, f'K={K} N={N} M={M}: max abs err {err}'


def _linear_plan(fmt, dtype):
    from lmdeploy.turbomind.linear import get_linear

    return get_linear().get_weight_plan(weight_format=fmt, dtype=dtype, grouped=False)
