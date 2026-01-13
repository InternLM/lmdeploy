import pytest
import torch


class TestSiluAndMul:

    @pytest.fixture
    def seqlen(self, request):
        yield request.param

    @pytest.fixture
    def feat_size(self, request):
        yield request.param

    @pytest.fixture
    def x(self, seqlen, feat_size):
        yield torch.rand(seqlen, feat_size, dtype=torch.float16, device='cuda')

    @pytest.fixture
    def gt(self, x):
        gate, up = x.chunk(2, -1)
        gate = torch.nn.functional.silu(gate)
        yield gate * up

    @pytest.mark.parametrize('seqlen', [65536, 256], indirect=True)
    @pytest.mark.parametrize('feat_size', [4096, 768], indirect=True)
    def test_silu_and_mul(self, x, gt):
        from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul

        out = silu_and_mul(x)
        torch.testing.assert_close(out, gt)


class TestSliluAndMulMoEEP:

    @pytest.fixture
    def num_experts(self, request):
        yield request.param

    @pytest.fixture
    def seqlen(self, request):
        yield request.param

    @pytest.fixture
    def feat_size(self, request):
        yield request.param

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def x(self, num_experts, seqlen, feat_size, dtype):
        yield torch.rand(num_experts, seqlen, feat_size, dtype=dtype, device='cuda')

    @pytest.fixture
    def mask_m(self, num_experts, seqlen):
        mask_m = torch.randint(0, seqlen, (num_experts, ), device='cuda')
        yield mask_m

    @pytest.fixture
    def elem_mask(self, mask_m, seqlen):
        elem_mask = torch.arange(seqlen, device='cuda').unsqueeze(0) < mask_m.unsqueeze(1)
        yield elem_mask[..., None]

    @pytest.fixture
    def gt(self, x):
        gate, up = x.chunk(2, -1)
        gate = torch.nn.functional.silu(gate)
        yield gate * up

    @pytest.mark.parametrize('num_experts', [4], indirect=True)
    @pytest.mark.parametrize('seqlen', [1024], indirect=True)
    @pytest.mark.parametrize('feat_size', [4096, 768], indirect=True)
    def test_silu_and_mul(self, x, mask_m, elem_mask, gt):
        from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul_moe_ep

        out = silu_and_mul_moe_ep(x, mask_m)
        out.masked_fill_(~elem_mask, 0.0)
        gt.masked_fill_(~elem_mask, 0.0)
        torch.testing.assert_close(out, gt)
