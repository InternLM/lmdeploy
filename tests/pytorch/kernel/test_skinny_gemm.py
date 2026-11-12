# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

from lmdeploy.pytorch.backends.cuda.linear import SkinnyLinearImpl
from lmdeploy.pytorch.backends.cuda.skinny_gemm_plans import SM90_SKINNY_GEMM_PLANS
from lmdeploy.pytorch.kernels.cuda.skinny_gemm import skinny_gemm, warmup_skinny_gemm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
                                reason='SM90 CUDA is required')
CELLS = [(n, k, m, config) for (n, k), plans in SM90_SKINNY_GEMM_PLANS.items() for m, config in plans.items()]


@pytest.mark.parametrize('n,k,m,config', CELLS)
def test_skinny_plan_strided_graph_and_fp64(n, k, m, config):
    torch.manual_seed(65306)
    storage = torch.randn(m, k + 128, device='cuda', dtype=torch.bfloat16)
    x = storage[:, :k]
    weight = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * k**-.5
    warmup_skinny_gemm(n, k, m, config)
    for kind in ('normal', 'outlier', 'sparse'):
        if kind == 'outlier':
            x[:, ::128] *= 16
        if kind == 'sparse':
            x[:, 1::2] = 0
        graph = torch.cuda.CUDAGraph()
        eager = skinny_gemm(x, weight, config)
        with torch.cuda.graph(graph):
            captured = skinny_gemm(x, weight, config)
        x.normal_()
        if kind == 'outlier':
            x[:, ::128] *= 16
        if kind == 'sparse':
            x[:, 1::2] = 0
        graph.replay()
        reference = F.linear(x.double(), weight.double())
        torch.testing.assert_close(captured.double(), reference, rtol=2**-8, atol=7.5e-5)
        assert eager.shape == captured.shape and torch.isfinite(eager).all()


@pytest.mark.parametrize('m,dtype,bias', [(1, torch.bfloat16, False), (3, torch.bfloat16, False),
                                          (12, torch.bfloat16, False), (1, torch.float32, False),
                                          (1, torch.bfloat16, True), (0, torch.bfloat16, False)])
def test_backend_shape_and_fallback(m, dtype, bias):
    n, k = 128, 4096
    impl = SkinnyLinearImpl(n, k, SM90_SKINNY_GEMM_PLANS[n, k])
    x = torch.randn(1, m, k, device='cuda', dtype=dtype)
    weight = torch.randn(n, k, device='cuda', dtype=dtype) * k**-.5
    b = torch.randn(n, device='cuda', dtype=dtype) if bias else None
    actual = impl.forward(x, weight, b)
    expected = F.linear(x, weight, b)
    assert actual.shape == (1, m, n)
    if m in impl.plans and dtype == torch.bfloat16 and not bias:
        torch.testing.assert_close(actual.double(), F.linear(x.double(), weight.double()), rtol=2**-8, atol=7.5e-5)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_batched_projection_keeps_bmm():
    n, k, m = 128, 4096, 2
    impl = SkinnyLinearImpl(n, k, SM90_SKINNY_GEMM_PLANS[n, k])
    x = torch.randn(2, m, k, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(2, n, k, device='cuda', dtype=torch.bfloat16)
    torch.testing.assert_close(impl.forward_batched(x, weight), torch.bmm(x, weight.transpose(1, 2)), rtol=0, atol=0)


def test_cuda_builder_is_opt_in(monkeypatch):
    from lmdeploy.pytorch import envs
    from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
    from lmdeploy.pytorch.backends.default.linear import DefaultLinearImpl
    from lmdeploy.pytorch.backends.linear import LinearBuildSpec
    spec = LinearBuildSpec(4096, 128, False, torch.bfloat16)
    monkeypatch.setattr(envs, 'enable_skinny_gemm', False)
    assert type(CudaOpsBackend.build_op(spec)) is DefaultLinearImpl
    monkeypatch.setattr(envs, 'enable_skinny_gemm', True)
    assert isinstance(CudaOpsBackend.build_op(spec), SkinnyLinearImpl)
    assert type(CudaOpsBackend.build_op(spec, enable_deterministic=True)) is DefaultLinearImpl
    assert type(CudaOpsBackend.build_op(LinearBuildSpec(4096, 128, True, torch.bfloat16))) is DefaultLinearImpl


def test_kpool_projection_backend_preserves_names_and_fp32_gate(monkeypatch):
    from lmdeploy.pytorch import envs
    from lmdeploy.pytorch.nn.kpool import KPOOL_INDEXER_PARAMETER_NAMES, KPoolIndexer
    monkeypatch.setattr(envs, 'enable_skinny_gemm', True)
    indexer = KPoolIndexer(4096, 32, 128, 2048, 1536, 4, dtype=torch.bfloat16, device='cuda')
    assert set(indexer.state_dict()) == set(KPOOL_INDEXER_PARAMETER_NAMES)
    assert isinstance(indexer.wk.impl, SkinnyLinearImpl)
    assert isinstance(indexer._compress_gate_impl, SkinnyLinearImpl)
    assert indexer.weights_proj.weight.dtype == torch.float32
    with torch.no_grad():
        indexer.index_kpool_compress_gate.normal_(std=4096**-.5)
        indexer.weights_proj.weight.normal_(std=4096**-.5)
    x = torch.randn(1, 6, 4096, dtype=torch.bfloat16, device='cuda')
    actual = indexer.project_compress_score(x)
    reference = F.linear(x.double(), indexer.index_kpool_compress_gate.double())
    torch.testing.assert_close(actual.double(), reference, rtol=2**-8, atol=7.5e-5)
    expected_head_gate = F.linear(x.float(), indexer.weights_proj.weight) * 32**-.5
    torch.testing.assert_close(indexer.project_head_gate(x), expected_head_gate, rtol=0, atol=0)
