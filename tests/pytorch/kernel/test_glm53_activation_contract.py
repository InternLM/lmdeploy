# Copyright (c) OpenMMLab. All rights reserved.
"""GLM53 materialized BF16 SwiGLU and routed W2 input-quantization contract."""
import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul, silu_and_mul_masked_post_quant_fwd


def _activation(x):
    gate, up = x.chunk(2, -1)
    return F.silu(gate.clamp(max=10.)) * up.clamp(-10., 10.)


@triton.jit
def _reference_quant_kernel(X, Q, S, GROUP: tl.constexpr):
    group = tl.program_id(0)
    offset = group * GROUP + tl.arange(0, GROUP)
    values = tl.load(X + offset).to(tl.float32)
    scale = tl.clamp(tl.max(tl.abs(values), 0) * (1. / 448.), 1e-12, 3e38)
    # Use the materialized training quantizer's Triton division, including FP8 tie behavior.
    quantized = tl.clamp(values / scale, -448., 448.)
    tl.store(Q + offset, quantized.to(tl.float8e4nv))
    tl.store(S + group, scale)


def _quantize(x):
    x = x.contiguous()
    codes = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((*x.shape[:-1], x.shape[-1] // 128), device=x.device, dtype=torch.float32)
    _reference_quant_kernel[(x.numel() // 128,)](x, codes, scales, 128, num_warps=4)
    return codes, scales


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
class TestGLM53ActivationContract:

    @pytest.mark.parametrize('width', [130, 2048])
    @pytest.mark.parametrize('strided', [False, True])
    def test_materialized_silu(self, width, strided):
        torch.manual_seed(205)
        x = torch.randn(7, 2 * width * (2 if strided else 1), device='cuda', dtype=torch.bfloat16) * 6
        if strided:
            x = x[:, ::2]
        actual = silu_and_mul(x, swiglu_limit=10., precise_mul=True, round_silu=True)
        assert torch.equal(actual.view(torch.uint8), _activation(x).view(torch.uint8))

    @pytest.mark.parametrize('rows', [1, 17, 64])
    @pytest.mark.parametrize('amplitude', [0., 1e-6, 6.])
    def test_w2_codes_scales_and_dummy_rows(self, rows, amplitude):
        torch.manual_seed(206)
        x = torch.randn(3, rows, 4096, device='cuda', dtype=torch.bfloat16) * amplitude
        mask = torch.tensor([0, 1, rows], device='cuda', dtype=torch.int32)
        out = torch.full((3, rows, 2048), -1., device='cuda', dtype=torch.float8_e4m3fn)
        scales = torch.full((3, rows, 16), -1., device='cuda')
        kwargs = dict(swiglu_limit=10., precise_mul=True, training_contract=True)
        silu_and_mul_masked_post_quant_fwd(x, out, scales, 128, mask, **kwargs)
        expected, expected_scales = _quantize(_activation(x))
        valid = torch.arange(rows, device='cuda')[None] < mask[:, None]
        assert torch.equal(out.view(torch.uint8)[valid], expected.view(torch.uint8)[valid])
        assert torch.equal(scales[valid], expected_scales[valid])
        assert bool((out.float()[~valid] == -1).all()) and bool((scales[~valid] == -1).all())
        # W2 contract already implies SiLU rounding; enabling both must not alter codes/scales.
        both = out.clone()
        both_scales = scales.clone()
        silu_and_mul_masked_post_quant_fwd(x, both, both_scales, 128, mask, round_silu=True, **kwargs)
        assert torch.equal(out.view(torch.uint8), both.view(torch.uint8))
        assert torch.equal(scales, both_scales)

    def test_changed_input_and_mask_cuda_graph(self):
        x = torch.zeros(3, 17, 4096, device='cuda', dtype=torch.bfloat16)
        mask = torch.tensor([0, 1, 17], device='cuda', dtype=torch.int32)
        out = torch.empty(3, 17, 2048, device='cuda', dtype=torch.float8_e4m3fn)
        scales = torch.empty(3, 17, 16, device='cuda')

        def run():
            silu_and_mul_masked_post_quant_fwd(x, out, scales, 128, mask, swiglu_limit=10.,
                                             precise_mul=True, training_contract=True)

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        for counts in ([17, 0, 3], [1, 17, 0], [0, 0, 0]):
            x.normal_()
            mask.copy_(torch.tensor(counts, device='cuda', dtype=torch.int32))
            previous, previous_scales = out.clone(), scales.clone()
            graph.replay()
            torch.cuda.synchronize()
            expected, expected_scales = _quantize(_activation(x))
            valid = torch.arange(17, device='cuda')[None] < mask[:, None]
            assert torch.equal(out.view(torch.uint8)[valid], expected.view(torch.uint8)[valid])
            assert torch.equal(scales[valid], expected_scales[valid])
            assert torch.equal(out.view(torch.uint8)[~valid], previous.view(torch.uint8)[~valid])
            assert torch.equal(scales[~valid], previous_scales[~valid])

    @pytest.mark.parametrize('change', [dict(precise_mul=False), dict(swiglu_limit=9.),
                                       dict(quant_group_size=64), dict(scale_fmt='ue8m0')])
    def test_unsupported_contract_is_rejected(self, change):
        x = torch.zeros(1, 1, 4096, device='cuda', dtype=torch.bfloat16)
        out = torch.empty(1, 1, 2048, device='cuda', dtype=torch.float8_e4m3fn)
        scales = torch.empty(1, 1, 16, device='cuda')
        mask = torch.ones(1, device='cuda', dtype=torch.int32)
        kwargs = dict(swiglu_limit=10., precise_mul=True, training_contract=True, quant_group_size=128)
        kwargs.update(change)
        with pytest.raises(ValueError, match='Training W2 contract'):
            silu_and_mul_masked_post_quant_fwd(x, out, scales, masked_m=mask, **kwargs)
