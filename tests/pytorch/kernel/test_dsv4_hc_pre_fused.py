# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
ROWS = [1, 2, 32, 33, 64, 65]


@pytest.fixture(autouse=True)
def seed_and_disable_tf32():
    torch.manual_seed(5027)
    # This backend flag is a dynamic property, not a deletable Python attribute.
    original = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original


def _unfused_pre(x, mixes, scale, base, weight, sinkhorn_iters, norm_eps):
    from lmdeploy.pytorch.kernels.cuda.dsv4.hc_prepost import hc_pre_reduce_norm
    from lmdeploy.pytorch.kernels.cuda.dsv4.hc_split_sinkhorn import hc_split_sinkhorn

    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, 4, sinkhorn_iters, 1e-6)
    out = hc_pre_reduce_norm(x, pre, 4, weight, norm_eps, x.dtype)
    return out, post, comb


def _assert_exact(actual, expected):
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)


@pytest.mark.parametrize('rows', ROWS)
@pytest.mark.parametrize('sinkhorn_iters', [3, 20])
@torch.inference_mode()
def test_pre_norm_matches_unfused_and_replays(rows, sinkhorn_iters, monkeypatch):
    from lmdeploy.pytorch.backends.cuda.hc_prepost import TritonHCPrePostImpl
    from lmdeploy.pytorch.kernels.cuda.dsv4 import hc_pre_fused as kernels

    x = torch.randn(1, rows, 4, 4096, device='cuda', dtype=torch.bfloat16)
    mixes = torch.randn(1, rows, 24, device='cuda')
    scale = torch.randn(3, device='cuda')
    base = torch.randn(24, device='cuda')
    weight = torch.randn(4096, device='cuda', dtype=torch.bfloat16)
    impl = TritonHCPrePostImpl(4, sinkhorn_iters, 1e-6)
    fused = kernels.hc_pre_fused
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return fused(*args, **kwargs)

    monkeypatch.setattr(kernels, 'hc_pre_fused', counted)

    def run():
        return impl.pre(x, mixes, scale, base, x.dtype, weight, 1e-5)

    def reference():
        return _unfused_pre(x, mixes, scale, base, weight, sinkhorn_iters, 1e-5)

    _assert_exact(run(), reference())
    assert bool(calls) == (rows <= 64)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    for _ in range(3):
        x.normal_()
        mixes.normal_()
        scale.normal_()
        base.normal_()
        weight.normal_()
        graph.replay()
        _assert_exact(outputs, reference())


@pytest.mark.parametrize('rows', ROWS)
@pytest.mark.parametrize('eps', [1e-6, 1e-5])
@torch.inference_mode()
def test_scale_mixes_boundary_and_graph_updates(rows, eps):
    from lmdeploy.pytorch.backends.cuda.hc_prepost import TritonHCPrePostImpl
    from lmdeploy.pytorch.nn.norm import rms_scale

    impl = TritonHCPrePostImpl(4, 20, 1e-6)
    x = torch.randn(1, rows, 16384, device='cuda')
    mixes = torch.randn(1, rows, 24, device='cuda')

    def run():
        out = impl.try_scale_mixes(mixes, x, eps)
        assert (out is not None) == (rows <= 64 and impl._gluon)
        return rms_scale(mixes, x, eps=eps) if out is None else out

    torch.testing.assert_close(run(), rms_scale(mixes, x, eps=eps), atol=0, rtol=0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for magnitude in (1e-3, 1.0, 1e3):
        x.normal_(std=magnitude)
        mixes.normal_()
        graph.replay()
        torch.testing.assert_close(out, rms_scale(mixes, x, eps=eps), atol=0, rtol=0)


@pytest.mark.parametrize('rows', ROWS)
@pytest.mark.parametrize('cached_fp32', [False, True])
@torch.inference_mode()
def test_prepared_projection_reload_preserves_graph(rows, cached_fp32):
    from lmdeploy.pytorch.models.glm5_next import Glm5NextDecoderLayer
    from lmdeploy.pytorch.nn import HcPrePost, rms_scale

    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip('ordered projection reference uses Hopper FP32 GEMM')

    # Exercise the real reload hook without allocating attention/expert weights.
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.hc_prepost = HcPrePost(4, avoid_gemv=True)
    for kind in ('attn', 'ffn'):
        fn = torch.randn(24, 16384, device='cuda') * 0.03
        layer.register_parameter(f'hc_{kind}_fn', torch.nn.Parameter(fn, requires_grad=False))
        layer.register_buffer(f'hc_{kind}_projection', None, persistent=False)
    layer.update_weights()
    pointers = [getattr(layer, f'hc_{kind}_projection').data_ptr() for kind in ('attn', 'ffn')]
    assert not any('projection' in key for key in layer.state_dict())

    x = torch.randn(1, rows, 4, 4096, device='cuda', dtype=torch.bfloat16)
    x_fp32 = x.float() if cached_fp32 else None
    scale = torch.randn(3, device='cuda')
    base = torch.randn(24, device='cuda')
    weight = torch.randn(4096, device='cuda', dtype=torch.bfloat16)

    def run():
        return tuple(layer.hc_prepost.pre(
            x, getattr(layer, f'hc_{kind}_fn'), scale, base, 1e-6,
            norm_weight=weight, x_fp32=x_fp32,
            projection_weight=getattr(layer, f'hc_{kind}_projection')) for kind in ('attn', 'ffn'))

    def check(outputs):
        flat = x.flatten(2).float()
        for kind, actual in zip(('attn', 'ffn'), outputs):
            fn = getattr(layer, f'hc_{kind}_fn')
            # Match the existing avoid_gemv path for single-token decode.
            mixes = (F.linear(F.pad(flat, (0, 0, 0, 1)), fn)[:, :1].contiguous()
                     if rows == 1 else F.linear(flat, fn))
            projected = layer.hc_prepost.impl.try_project_mixes(flat, getattr(layer, f'hc_{kind}_projection'))
            assert (projected is not None) == (rows <= 32)
            if projected is not None:
                torch.testing.assert_close(projected, mixes, atol=0, rtol=0)
            mixes = rms_scale(mixes, flat, eps=1e-6)
            _assert_exact(actual, _unfused_pre(x, mixes, scale, base, weight, 20, 1e-6))

    check(run())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    for magnitude in (0.1, 1.0, 10.0):
        x.normal_(std=magnitude)
        if x_fp32 is not None:
            x_fp32.copy_(x)
        for kind in ('attn', 'ffn'):
            getattr(layer, f'hc_{kind}_fn').normal_(std=0.03)
        layer.update_weights()
        assert pointers == [getattr(layer, f'hc_{kind}_projection').data_ptr() for kind in ('attn', 'ffn')]
        assert not any('projection' in key for key in layer.state_dict())
        graph.replay()
        check(outputs)
