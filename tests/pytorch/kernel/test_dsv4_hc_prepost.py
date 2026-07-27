import pytest
import torch
import torch.nn.functional as F


def _reference_pre_reduce(x, pre, out_dtype):
    return torch.sum(pre.unsqueeze(-1) * x, dim=-2).to(out_dtype)


def _reference_post_expand(x, residual, post, comb):
    y = post.unsqueeze(-1) * x.float().unsqueeze(-2)
    y += torch.sum(comb.unsqueeze(-1) * residual.float().unsqueeze(-3), dim=-2)
    return y.to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
class TestHcPrePost:

    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(0)

    @pytest.mark.parametrize('lead_shape, dim', [
        ((2, 3), 128),
        ((5,), 257),
    ])
    def test_pre_reduce(self, lead_shape, dim):
        from lmdeploy.pytorch.nn import HcPrePost
        hc_mult = 4
        x = torch.randn(*lead_shape, hc_mult, dim, device='cuda', dtype=torch.float32)
        pre = torch.randn(*lead_shape, hc_mult, device='cuda', dtype=torch.float32)

        op = HcPrePost(hc_mult)
        out = op.pre_reduce(x, pre, torch.bfloat16)
        ref = _reference_pre_reduce(x, pre, torch.bfloat16)

        assert out.shape == (*lead_shape, dim)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_pre(self):
        from lmdeploy.pytorch.kernels.cuda.dsv4.hc_split_sinkhorn import hc_split_sinkhorn
        from lmdeploy.pytorch.nn import HcPrePost, rms_scale
        hc_mult = 4
        sinkhorn_iters = 3
        sinkhorn_eps = 1e-6
        norm_eps = 1e-6
        lead_shape = (2, 3)
        dim = 128
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim

        x = torch.randn(*lead_shape, hc_mult, dim, device='cuda', dtype=torch.bfloat16)
        hc_fn = torch.randn(mix_hc, hc_dim, device='cuda', dtype=torch.float32)
        hc_scale = torch.randn(3, device='cuda', dtype=torch.float32)
        hc_base = torch.randn(mix_hc, device='cuda', dtype=torch.float32)

        op = HcPrePost(hc_mult, sinkhorn_iters, sinkhorn_eps)
        out, post, comb = op.pre(x, hc_fn, hc_scale, hc_base, norm_eps)

        x_flat = x.flatten(2).float()
        mixes = rms_scale(F.linear(x_flat, hc_fn), x_flat, eps=norm_eps)
        pre_ref, post_ref, comb_ref = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, sinkhorn_eps)
        out_ref = _reference_pre_reduce(x_flat.view_as(x), pre_ref, x.dtype)

        assert out.shape == (*lead_shape, dim)
        assert post.shape == (*lead_shape, hc_mult)
        assert comb.shape == (*lead_shape, hc_mult, hc_mult)
        torch.testing.assert_close(out.float(), out_ref.float(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(post, post_ref, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(comb, comb_ref, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize('lead_shape, dim', [
        ((2, 3), 128),
        ((5,), 257),
    ])
    def test_post_expand(self, lead_shape, dim):
        from lmdeploy.pytorch.nn import HcPrePost
        hc_mult = 4
        x = torch.randn(*lead_shape, dim, device='cuda', dtype=torch.bfloat16)
        residual = torch.randn(*lead_shape, hc_mult, dim, device='cuda', dtype=torch.bfloat16)
        post = torch.randn(*lead_shape, hc_mult, device='cuda', dtype=torch.float32)
        comb = torch.randn(*lead_shape, hc_mult, hc_mult, device='cuda', dtype=torch.float32)

        op = HcPrePost(hc_mult)
        out = op.post_expand(x, residual, post, comb)
        ref = _reference_post_expand(x, residual, post, comb)

        assert out.shape == (*lead_shape, hc_mult, dim)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
