import pytest
import torch

from lmdeploy.utils import is_bf16_supported


def _bf16_mark():
    return pytest.mark.skipif(not is_bf16_supported(), reason='bf16 not supported.')


class TestRMSNorm:

    @pytest.fixture(scope='class')
    def dtype(self, request):
        yield request.param

    @pytest.fixture(scope='class')
    def input_shape(self, request):
        yield request.param

    @pytest.fixture(scope='class')
    def hidden_size(self, input_shape):
        yield input_shape[-1]

    @pytest.fixture(scope='class')
    def input(self, dtype, input_shape):
        yield torch.randn(input_shape, dtype=dtype, device='cuda')

    @pytest.fixture(scope='class')
    def weight(self, dtype, hidden_size):
        yield torch.randn(hidden_size, dtype=dtype, device='cuda')

    @pytest.fixture(scope='class')
    def eps(self):
        yield 1e-6

    @pytest.fixture(scope='class')
    def gt(self, input, weight, eps):
        input_dtype = input.dtype
        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + eps)
        return weight * input.to(input_dtype)

    @pytest.mark.parametrize('input_shape', [(2, 4, 4096), (4, 4096), (4096, )], indirect=True)
    @pytest.mark.parametrize('dtype', [pytest.param(torch.bfloat16, marks=_bf16_mark()), torch.float16], indirect=True)
    def test_rms_norm(self, input, weight, eps, gt):
        from lmdeploy.pytorch.kernels.cuda import rms_norm

        out = rms_norm(input, weight, eps)
        torch.testing.assert_close(out, gt)

    @pytest.fixture(scope='class')
    def residual(self, dtype, input_shape):
        yield torch.randn(input_shape, dtype=dtype, device='cuda')

    @pytest.fixture(scope='class')
    def gt_residual(self, input, residual, weight, eps):

        input = input + residual
        out_res = input
        input_dtype = input.dtype
        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + eps)
        return weight * input.to(input_dtype), out_res

    @pytest.mark.parametrize('input_shape', [(2, 4, 4096), (4, 4096), (4096, )], indirect=True)
    @pytest.mark.parametrize('dtype', [pytest.param(torch.bfloat16, marks=_bf16_mark()), torch.float16], indirect=True)
    def test_rms_norm_residual(self, input, residual, weight, eps, gt_residual):
        from lmdeploy.pytorch.kernels.cuda import rms_norm

        out, out_res = rms_norm(input, weight, eps, residual=residual)
        gt, gt_res = gt_residual
        torch.testing.assert_close(out, gt)
        torch.testing.assert_close(out_res, gt_res)
