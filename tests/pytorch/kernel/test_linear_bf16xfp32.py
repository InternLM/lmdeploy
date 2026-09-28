# Copyright (c) OpenMMLab. All rights reserved.
import importlib

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

# The kernel reconstructs an fp32 weight from two bf16 halves, which costs
# ~2^-17 relative accuracy on the weight. The resulting per-element error of a
# dot product is sum_k x_k * dw_k, which does not scale with the output value
# (cancellation), so the budget has to be RMS-relative rather than rtol-based.
# 8x the 1-sigma error is a >6-sigma margin over these element counts.
_ERR_BUDGET = 8 * 2**-17


def _budget(ref: torch.Tensor) -> float:
    return _ERR_BUDGET * ref.pow(2).mean().sqrt().item()


@pytest.mark.parametrize('n,m,k', [
    (4, 512, 130),      # N below the tl.dot floor, K tail
    (7, 513, 400),      # non power-of-two N, M and K tails
    (16, 512, 512),
    (24, 1024, 512),
    (80, 512, 4096),    # wide N, still a single tile
    (129, 512, 4096),   # N just past one tile
    (192, 512, 4096),   # router-sized N
    (256, 1024, 4096),
    (257, 512, 1024),   # N tail inside the last tile
    (1024, 512, 512),   # many N tiles
])
def test_linear(n, m, k):
    from lmdeploy.pytorch.kernels.cuda import linear_bf16xfp32

    torch.manual_seed(k + m + n)
    x = torch.randn(m, k, device='cuda').bfloat16()
    w = torch.randn(n, k, device='cuda', dtype=torch.float32)
    ref = (x.double() @ w.double().t()).float()

    out = linear_bf16xfp32(x, w)

    assert out.shape == (m, n)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=0.0, atol=_budget(ref))


@pytest.mark.parametrize('n,m,k', [
    (24, 512, 28672),   # DeepSeek-V4 hyper-connection block
    (4, 512, 28672),    # DeepSeek-V4 hyper-connection head
    (256, 1755, 4096),  # DeepSeek-V4 MoE router
    (192, 1755, 4096),  # Hy3 MoE router
])
def test_production_shapes(n, m, k):
    """The two shape families this kernel serves must both take the fast
    path."""
    import triton

    from lmdeploy.pytorch.kernels.cuda import linear_bf16xfp32
    from lmdeploy.pytorch.kernels.cuda.linear_bf16xfp32 import _MIN_M, _get_block_n

    assert m >= _MIN_M, 'this test must exercise the kernel, not the fallback'
    torch.manual_seed(n)
    x = torch.randn(m, k, device='cuda').bfloat16()
    w = torch.randn(n, k, device='cuda', dtype=torch.float32)

    out = linear_bf16xfp32(x, w)
    ref = (x.double() @ w.double().t()).float()

    # A wide N must be tiled rather than rejected.
    assert triton.cdiv(n, _get_block_n(n)) == triton.cdiv(n, min(128, max(16, triton.next_power_of_2(n))))
    torch.testing.assert_close(out, ref, rtol=0.0, atol=_budget(ref))


@pytest.mark.parametrize('n,m,k', [
    (24, 512, 28672),
    (4, 512, 28672),
    (24, 1024, 4096),
])
def test_linear_matches_fp32_path(n, m, k):
    """The kernel must agree with the upcast-then-linear path it replaces."""
    from lmdeploy.pytorch.kernels.cuda import linear_bf16xfp32

    torch.manual_seed(m * k)
    x = torch.randn(m, k, device='cuda').bfloat16()
    w = torch.randn(n, k, device='cuda', dtype=torch.float32)

    ref = F.linear(x.float(), w)
    out = linear_bf16xfp32(x, w)

    torch.testing.assert_close(out, ref, rtol=0.0, atol=_budget(ref))


@pytest.mark.parametrize('lead_shape', [(2, 3, 4), (7, 5)])
def test_linear_keeps_leading_dims(lead_shape):
    from lmdeploy.pytorch.kernels.cuda import linear_bf16xfp32

    m = 1
    for d in lead_shape:
        m *= d
    n, k = 24, 512
    torch.manual_seed(m)
    x = torch.randn(*lead_shape, k, device='cuda').bfloat16()
    w = torch.randn(n, k, device='cuda', dtype=torch.float32)

    out = linear_bf16xfp32(x, w)

    assert out.shape == (*lead_shape, n)
    ref = F.linear(x.reshape(-1, k).float(), w).reshape(*lead_shape, n)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=_budget(ref))


@pytest.mark.parametrize('split_k', [1, 4, 16])
def test_split_k_is_deterministic(split_k, monkeypatch):
    """The partial-buffer reduction must not depend on accumulation order."""
    # importlib, not `import ... as`: the cuda package re-exports the launcher
    # under the same name and would shadow the submodule.
    kernel_mod = importlib.import_module('lmdeploy.pytorch.kernels.cuda.linear_bf16xfp32')

    monkeypatch.setattr(kernel_mod, '_get_split_k', lambda *args: split_k)

    n, m, k = 24, 512, 4096
    torch.manual_seed(split_k)
    x = torch.randn(m, k, device='cuda').bfloat16()
    w = torch.randn(n, k, device='cuda', dtype=torch.float32)

    first = kernel_mod.linear_bf16xfp32(x, w)
    second = kernel_mod.linear_bf16xfp32(x, w)

    assert torch.equal(first, second)

    # Different split factors only change the summation order, so they agree
    # within the kernel error budget.
    monkeypatch.setattr(kernel_mod, '_get_split_k', lambda *args: 1)
    ref = kernel_mod.linear_bf16xfp32(x, w)
    torch.testing.assert_close(first, ref, rtol=0.0, atol=_budget(ref))


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_non_bf16_activation_falls_back(dtype):
    """Non-bf16 activations take the fp32 path, byte for byte."""
    from lmdeploy.pytorch.kernels.cuda import linear_bf16xfp32

    n, m, k = 24, 64, 512
    torch.manual_seed(0)
    x = torch.randn(m, k, device='cuda', dtype=dtype)
    w = torch.randn(n, k, device='cuda', dtype=torch.float32)

    out = linear_bf16xfp32(x, w)

    assert torch.equal(out, F.linear(x.float(), w))


@pytest.mark.parametrize('m', [0, 1, 16, 256])
def test_small_m_falls_back(m):
    """Below `_MIN_M` rows the fp32 path is still the faster one."""
    from lmdeploy.pytorch.kernels.cuda import linear_bf16xfp32

    n, k = 24, 512
    torch.manual_seed(m)
    x = torch.randn(m, k, device='cuda').bfloat16()
    w = torch.randn(n, k, device='cuda', dtype=torch.float32)

    out = linear_bf16xfp32(x, w)

    assert out.shape == (m, n)
    assert out.dtype == torch.float32
    assert torch.equal(out, F.linear(x.float(), w))


def test_wide_n_is_tiled_not_rejected():
    """An N wider than one tile is split along the grid and stays exact."""
    import triton

    from lmdeploy.pytorch.kernels.cuda import linear_bf16xfp32
    from lmdeploy.pytorch.kernels.cuda.linear_bf16xfp32 import _MAX_BLOCK_N, _get_block_n

    n, m, k = _MAX_BLOCK_N * 2 + 1, 512, 256
    torch.manual_seed(0)
    x = torch.randn(m, k, device='cuda').bfloat16()
    w = torch.randn(n, k, device='cuda', dtype=torch.float32)

    assert triton.cdiv(n, _get_block_n(n)) > 1, 'this test must exercise N tiling'
    out = linear_bf16xfp32(x, w)
    ref = (x.double() @ w.double().t()).float()

    assert out.shape == (m, n)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=_budget(ref))


def test_cuda_graph_capture_of_cold_shape_falls_back():
    """Autotune cannot benchmark mid-capture, so an uncompiled shape declines.

    Without the guard the benchmarking sweep invalidates the capture with
    ``cudaErrorStreamCaptureInvalidated``.
    """
    kernel_mod = importlib.import_module('lmdeploy.pytorch.kernels.cuda.linear_bf16xfp32')

    n, m, k = 48, 1024, 2176   # deliberately unused elsewhere, so the cache is cold
    torch.manual_seed(1)
    x = torch.randn(m, k, device='cuda').bfloat16()
    w = torch.randn(n, k, device='cuda', dtype=torch.float32)
    block_n = kernel_mod._get_block_n(n)
    split_k = kernel_mod._get_split_k(m, n, k, block_n, x.device)
    kernel_mod._COMPILED_KEYS.discard((n, k, split_k, block_n, kernel_mod._get_m_hint(m)))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cold = kernel_mod.linear_bf16xfp32(x, w)
    graph.replay()
    torch.cuda.synchronize()

    # The cold capture must have taken the fp32 fallback, bit for bit.
    assert torch.equal(cold, F.linear(x.float(), w))

    # Once compiled eagerly the same shape is safe to capture and use.
    kernel_mod.linear_bf16xfp32(x, w)
    warm_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(warm_graph):
        warm = kernel_mod.linear_bf16xfp32(x, w)
    warm_graph.replay()
    torch.cuda.synchronize()

    ref = (x.double() @ w.double().t()).float()
    torch.testing.assert_close(warm, ref, rtol=0.0, atol=_budget(ref))
