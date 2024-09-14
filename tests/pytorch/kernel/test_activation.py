import pytest
import torch


class TestSiluAndMul:

    @pytest.fixture
    def seqlen(self):
        yield 256

    @pytest.fixture
    def feat_size(self):
        yield 4096

    @pytest.fixture
    def x(self, seqlen, feat_size):
        yield torch.rand(seqlen, feat_size, dtype=torch.float16, device='cuda')
    
    @pytest.fixture
    def gt(self, x):
        gate, up = x.chunk(2, -1)
        gate = torch.nn.functional.silu(gate)
        yield gate * up

    def test_silu_and_mul(self, x, gt):
        from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul

        out = silu_and_mul(x)
        torch.testing.assert_close(out, gt)