import pytest
import torch


def _make_quant_val(shape, out_dtype):
    x = torch.rand(shape, dtype=torch.float32, device='cuda')
    # -1 ~ 1
    x = x * 2 - 1
    # scaling abs max to fmax
    finfo = torch.finfo(out_dtype)
    fmax = finfo.max
    scaling = fmax / x.abs().amax(-1, keepdim=True)
    x *= scaling
    return x.to(out_dtype).to(torch.float32)


def fast_log2_ceil_torch(x: torch.Tensor) -> torch.Tensor:
    bits_x = x.view(torch.int32)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    result = (exp_x - 127).to(torch.int32)
    result = result + torch.where(man_bits != 0, 1, 0)

    return result.to(torch.int32)


def fast_pow2_torch(x: torch.Tensor) -> torch.Tensor:
    bits_x = (x + 127) << 23
    return bits_x.view(torch.float32)


def fast_round_scale_torch(amax: torch.Tensor, fp8_max_inv: torch.Tensor) -> torch.Tensor:
    return fast_pow2_torch(fast_log2_ceil_torch(amax * fp8_max_inv))


def _make_quant_scale_ue8m0(shape, out_dtype):
    scale = torch.randn(shape, dtype=torch.float32, device='cuda')
    finfo = torch.finfo(out_dtype)
    fmax = finfo.max
    scale = fast_round_scale_torch(scale, 1 / fmax)
    return scale


def _make_quant_scale(shape, out_dtype, scale_fmt: str = None):
    if scale_fmt == 'ue8m0':
        return _make_quant_scale_ue8m0(shape, out_dtype)

    # default
    scale = torch.rand(shape, dtype=torch.float32, device='cuda')
    finfo = torch.finfo(out_dtype)
    fmax = finfo.max
    scale /= fmax
    return scale


def _make_A(M, K, group_size, out_dtype, scale_fmt: str = None):
    quant_A = _make_quant_val((M, K // group_size, group_size), out_dtype)

    # create scale and A
    scale = _make_quant_scale((M, K // group_size), out_dtype, scale_fmt)
    A = quant_A * scale[..., None]

    A = A.reshape(M, K)
    quant_A = quant_A.reshape(M, K).to(out_dtype)
    scale = scale.T.contiguous().T
    return A, quant_A, scale


def _aligned_size(a, b):
    return (a + b - 1) // b * b


def _make_B(K, N, group_size, out_dtype, scale_fmt: str = None):
    K_aligned = _aligned_size(K, group_size)
    N_aligned = _aligned_size(N, group_size)

    quant_B = _make_quant_val((K_aligned // group_size, group_size, N_aligned // group_size, group_size), out_dtype)

    scale = _make_quant_scale((K_aligned // group_size, 1, N_aligned // group_size, 1), out_dtype, scale_fmt)

    B = quant_B * scale

    B = B.reshape(K_aligned, N_aligned)[:K, :N]
    quant_B = quant_B.reshape(K_aligned, N_aligned).to(out_dtype)[:K, :N]
    scale = scale.reshape(K_aligned // group_size, N_aligned // group_size)
    quant_B = quant_B.transpose(0, 1).contiguous().transpose(0, 1)
    return B, quant_B, scale


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason='require device with cc>=9.0')
class TestQuantFP8:

    @pytest.fixture
    def M(self, request):
        yield request.param

    @pytest.fixture
    def K(self):
        yield 512

    @pytest.fixture
    def group_size(self):
        yield 128

    @pytest.fixture
    def out_dtype(self):
        yield torch.float8_e4m3fn

    @pytest.fixture
    def scale_fmt(self, request):
        yield request.param

    @pytest.fixture
    def build_A(self, M, K, group_size, out_dtype, scale_fmt):
        return _make_A(M, K, group_size, out_dtype, scale_fmt)

    @pytest.fixture
    def A(self, build_A):
        return build_A[0]

    @pytest.fixture
    def quant_A(self, build_A):
        return build_A[1]

    @pytest.fixture
    def scale(self, build_A):
        return build_A[2]

    @pytest.fixture
    def gt(self, quant_A, scale):
        yield quant_A, scale

    @pytest.mark.parametrize('scale_fmt', [None, 'ue8m0'], indirect=True)
    @pytest.mark.parametrize('M', [65536, 256], indirect=True)
    def test_quant_fp8(self, A, group_size, out_dtype, scale_fmt, gt):
        from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
        quant_A_gt, scale_gt = gt

        quant_A, scale = quant_fp8(A, group_size=group_size, dtype=out_dtype, scale_fmt=scale_fmt)
        torch.testing.assert_close(scale, scale_gt)
        diff = (quant_A.to(torch.float16) - quant_A_gt.to(torch.float16)).abs()
        diff_count = (diff > 1e-5).count_nonzero()
        assert diff_count / diff.numel() < 1e-4


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason='require device with cc>=9.0')
class TestGemmFP8:

    @pytest.fixture
    def M(self):
        yield 256

    @pytest.fixture
    def N(self):
        # test non-aligned
        yield 1024 + 64

    @pytest.fixture
    def K(self):
        yield 512

    @pytest.fixture
    def group_size(self):
        yield 128

    @pytest.fixture
    def quant_dtype(self):
        yield torch.float8_e4m3fn

    @pytest.fixture
    def out_dtype(self):
        yield torch.float16

    @pytest.fixture
    def build_A(self, M, K, group_size, quant_dtype):
        return _make_A(M, K, group_size, quant_dtype)

    @pytest.fixture
    def A(self, build_A, out_dtype):
        return build_A[0].to(out_dtype)

    @pytest.fixture
    def quant_A(self, build_A):
        return build_A[1]

    @pytest.fixture
    def scale_A(self, build_A):
        return build_A[2]

    @pytest.fixture
    def build_B(self, K, N, group_size, quant_dtype):
        return _make_B(K, N, group_size, quant_dtype)

    @pytest.fixture
    def B(self, build_B, out_dtype):
        return build_B[0].to(out_dtype)

    @pytest.fixture
    def quant_B(self, build_B):
        return build_B[1]

    @pytest.fixture
    def scale_B(self, build_B):
        return build_B[2]

    @pytest.fixture
    def gt(self, A, B):
        yield A @ B

    def test_gemm_fp8(self, quant_A, scale_A, quant_B, scale_B, out_dtype, gt):
        from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import blocked_gemm_fp8
        C = blocked_gemm_fp8(quant_A, scale_A, quant_B, scale_B, out_dtype=out_dtype)
        torch.testing.assert_close(C, gt, atol=0.5, rtol=1e-4)
