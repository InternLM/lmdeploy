import pytest
import torch
import torch.nn.functional as F


def _reference_pre_reduce(x, pre, out_dtype):
    return torch.sum(pre.unsqueeze(-1) * x, dim=-2).to(out_dtype)


def _reference_post_expand(x, residual, post, comb):
    # comb is [..., src, out]; the post-expand mixes residuals as
    # matmul(comb.T, residual) == sum_src comb[src, out] * residual[src].
    y = post.unsqueeze(-1) * x.float().unsqueeze(-2)
    y += torch.matmul(comb.float().transpose(-1, -2), residual.float())
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

    @pytest.mark.parametrize('rows', [1, 2, 6, 12, 24, 64, 65])
    @pytest.mark.parametrize('sinkhorn_iters', [3, 20])
    def test_fused_pre_norm(self, rows, sinkhorn_iters):
        from lmdeploy.pytorch.backends.cuda.hc_prepost import TritonHCPrePostImpl
        from lmdeploy.pytorch.kernels.cuda.dsv4.hc_prepost import hc_pre_reduce_norm
        from lmdeploy.pytorch.kernels.cuda.dsv4.hc_split_sinkhorn import hc_split_sinkhorn

        x = torch.randn(1, rows, 4, 4096, device='cuda', dtype=torch.bfloat16)
        mixes = torch.randn(1, rows, 24, device='cuda')
        scale = torch.randn(3, device='cuda')
        base = torch.randn(24, device='cuda')
        weight = torch.randn(4096, device='cuda', dtype=torch.bfloat16)
        impl = TritonHCPrePostImpl(4, sinkhorn_iters, 1e-6)

        def reference():
            pre, post, comb = hc_split_sinkhorn(mixes, scale, base, 4, sinkhorn_iters, 1e-6)
            y = hc_pre_reduce_norm(x, pre, 4, weight, 1e-5, x.dtype)
            return y, post, comb

        def run():
            return impl.pre(x, mixes, scale, base, x.dtype, weight, 1e-5)

        for actual, expected in zip(run(), reference()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        # Captured buffers must consume new values on every replay.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = run()
        for _ in range(3):
            x.normal_()
            mixes.normal_()
            graph.replay()
            for actual, expected in zip(outputs, reference()):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize('rows', [1, 2, 6, 12, 24, 48, 64, 65])
    @pytest.mark.parametrize('eps', [1e-6, 1e-5])
    def test_scale_mixes_graph(self, rows, eps):
        from lmdeploy.pytorch.backends.cuda.hc_prepost import TritonHCPrePostImpl
        from lmdeploy.pytorch.nn.norm import rms_scale

        impl = TritonHCPrePostImpl(4, 20, 1e-6)
        x = torch.randn(1, rows, 16384, device='cuda', dtype=torch.float32)
        mixes = torch.randn(1, rows, 24, device='cuda', dtype=torch.float32)

        def run():
            out = impl.try_scale_mixes(mixes, x, eps)
            assert (out is None) == (rows > 64 or not impl._gluon)
            return rms_scale(mixes, x, eps=eps) if out is None else out

        expected = rms_scale(mixes, x, eps=eps)
        torch.testing.assert_close(run(), expected, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        for index in range(10):
            x.normal_(std=10 ** (index % 7 - 3))
            mixes.normal_(std=10 ** (index % 5 - 2))
            graph.replay()
            torch.testing.assert_close(output, rms_scale(mixes, x, eps=eps), atol=0, rtol=0)

    @pytest.mark.parametrize('rows', [1, 2, 3, 6, 12, 24, 32, 33])
    def test_ordered_projection_weight_reload_graph(self, rows, monkeypatch):
        from lmdeploy.pytorch.models.glm5_next import Glm5NextDecoderLayer
        from lmdeploy.pytorch.nn import HcPrePost

        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip('Projection specialization requires Hopper')
        monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
        # Exercise the actual decoder weight-update lifecycle without allocating
        # an unrelated full attention/expert layer in this operator test.
        layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.hc_prepost = HcPrePost(4, avoid_gemv=True)
        for kind in ('attn', 'ffn'):
            weight = torch.randn(24, 16384, device='cuda') * 0.03
            layer.register_parameter(f'hc_{kind}_fn', torch.nn.Parameter(weight, requires_grad=False))
            layer.register_buffer(f'hc_{kind}_projection', None, persistent=False)
        layer.update_weights()
        pointers = [getattr(layer, f'hc_{k}_projection').data_ptr() for k in ('attn', 'ffn')]
        assert not any('projection' in key for key in layer.state_dict())
        x = torch.randn(1, rows, 16384, device='cuda', dtype=torch.bfloat16).float()

        def reference():
            if rows == 1:
                return F.linear(F.pad(x, (0, 0, 0, 1)), layer.hc_attn_fn)[:, :1].contiguous()
            return F.linear(x, layer.hc_attn_fn)

        def run():
            out = layer.hc_prepost.impl.try_project_mixes(x, layer.hc_attn_projection)
            assert (out is None) == (rows > 32)
            return reference() if out is None else out

        torch.testing.assert_close(run(), reference(), atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        for index in range(5):
            x.copy_(torch.randn_like(x).to(torch.bfloat16).float() * (10 ** (index - 2)))
            for kind in ('attn', 'ffn'):
                getattr(layer, f'hc_{kind}_fn').normal_(std=0.03)
            layer.update_weights()
            assert pointers == [getattr(layer, f'hc_{k}_projection').data_ptr() for k in ('attn', 'ffn')]
            graph.replay()
            torch.testing.assert_close(out, reference(), atol=0, rtol=0)

        monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', True)
        assert layer.hc_prepost.impl.try_project_mixes(x, layer.hc_attn_projection) is None

    def test_scale_mixes_backend_fallback(self):
        from lmdeploy.pytorch.backends.cuda.hc_prepost import TritonHCPrePostImpl

        impl = TritonHCPrePostImpl(4, 20, 1e-6)
        impl._gluon = False
        x = torch.empty(1, 12, 16384, device='cuda')
        mixes = torch.empty(1, 12, 24, device='cuda')
        assert impl.try_scale_mixes(mixes, x, 1e-6) is None


@pytest.mark.parametrize('version', ['development', '3.5.0', '3.8.0'])
def test_hc_gluon_version_fallback(version, monkeypatch):
    import triton

    from lmdeploy.pytorch.backends.cuda.hc_prepost import TritonHCPrePostImpl

    monkeypatch.setattr(triton, '__version__', version)
    impl = TritonHCPrePostImpl(4, 20, 1e-6)
    assert not impl._gluon
