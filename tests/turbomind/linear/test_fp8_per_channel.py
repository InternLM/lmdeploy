# Copyright (c) OpenMMLab. All rights reserved.
"""Per-channel (channel-quantized) FP8 compressed-tensors support.

Ornith-style checkpoints carry float-quantized FP8 weights with a
per-output-channel ``.weight_scale`` (BF16, shape [N, 1]) instead of the
blocked ``.weight_scale_inv`` (shape [K//128, N//128]).
"""
import torch
import pytest

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


def test_per_channel_post_process_expands_to_k_groups():
    fmt = FP8Format(block_out=1)
    K, N = 2048, 512  # down_proj: weight [N=512? no] -> after normalize [K, N]
    # TM layout after normalize: weight [K, N], scale [N, 1]
    weight = torch.randn(K, N)
    scales = torch.randn(N, 1)
    out = fmt.post_process({'weight': weight, 'scales': scales})
    assert out['scales'].shape == (K // 128, N)
    # every K-group row identical to the original per-channel column
    assert torch.equal(out['scales'][0], scales.squeeze(1))
    assert torch.equal(out['scales'][-1], scales.squeeze(1))


def test_blocked_post_process_untouched():
    fmt = FP8Format(block_out=128)
    weight = torch.randn(2048, 512)
    scales = torch.randn(16, 4)
    out = fmt.post_process({'weight': weight, 'scales': scales})
    assert out['scales'].shape == (16, 4)


def test_converter_builds_per_channel_fp8():
    fmts = _build_quantized_formats('compressed-tensors', 128, fp8_block_out=1)
    assert len(fmts) == 1 and isinstance(fmts[0], FP8Format)
    assert fmts[0].block_out == 1 and fmts[0].block_in == 128
    # default (int4 pack-quantized) unchanged
    fmts2 = _build_quantized_formats('compressed-tensors', 128)
    assert not isinstance(fmts2[0], FP8Format)


def test_dequant_per_channel_matches_reference():
    fmt = FP8Format(block_out=1)
    K, N = 256, 64
    w = torch.randn(K, N).to(torch.float8_e4m3fn)
    scale = torch.rand(N, 1) + 0.1
    out = fmt.post_process({'weight': w, 'scales': scale})
    deq = fmt.dequant(out, torch.float16)
    ref = (w.view(torch.float8_e4m3fn).float() * scale.float().expand(K, N)).to(torch.float16)
    assert torch.allclose(deq['weight'], ref, atol=1e-3)
