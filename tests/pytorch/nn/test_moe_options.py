# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch


def test_build_fused_moe_propagates_reduction_options(monkeypatch):
    from lmdeploy.pytorch.nn import moe

    captured = {}

    class FakeFusedMoE:

        def __init__(self, **kwargs):
            captured.update(kwargs)

    import lmdeploy.pytorch.nn.moe.default as default_moe
    monkeypatch.setattr(default_moe, 'FusedMoE', FakeFusedMoE)

    moe.build_fused_moe(16,
                        32,
                        4,
                        2,
                        quant_config=None,
                        fp32_acc=True,
                        output_scale=2.5)

    assert captured['fp32_acc'] is True
    assert captured['output_scale'] == 2.5


@pytest.mark.parametrize('options', [
    {'fp32_acc': True}, {'output_scale': 2.5},
    {'fp32_acc': True, 'output_scale': 2.5},
])
@pytest.mark.parametrize('quant_method', ['smooth_quant', 'fp8', 'compressed-tensors'])
def test_build_fused_moe_rejects_unsupported_reduction_options(monkeypatch, quant_method, options):
    from lmdeploy.pytorch.nn import moe

    class FakeQuantConfig:
        quant_dtype = None
        activation_scheme = 'static'
        weight_block_size = None
        bits = 4
        group_size = 128

        def get_quant_method(self, prefix, module_kind):
            return quant_method

    class FakeContext:
        quant_config = FakeQuantConfig()

    monkeypatch.setattr(moe, 'get_build_model_context', lambda: FakeContext())

    with pytest.raises(NotImplementedError, match='fp32_acc or output_scale'):
        moe.build_fused_moe(16,
                            32,
                            4,
                            2,
                            quant_config={},
                            **options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_moe_reduce_accumulates_in_fp32_before_scaling_and_casting():
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import moe_reduce

    hidden = torch.tensor([[[1.25, -2.5], [3.0, 4.0]]], device='cuda', dtype=torch.bfloat16)
    weights = torch.tensor([[0.2, 0.7]], device='cuda', dtype=torch.float32)
    actual = moe_reduce(hidden, weights, fp32_acc=True, output_scale=2.5)
    expected = ((hidden.float() * weights[..., None]).sum(dim=1) * 2.5).to(hidden.dtype)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture
def dlinfer_moe(monkeypatch):
    # Exercise the real builder without requiring a DLINFER vendor runtime.
    import lmdeploy.pytorch.backends.dlinfer as backend

    kernels = ModuleType('lmdeploy.pytorch.kernels.dlinfer')
    for name in ('DlinferMoECommType', 'DlinferMoeMetadata', 'fused_moe',
                 'fused_moe_w8a8', 'moe_gating_topk_softmax'):
        setattr(kernels, name, Mock())
    spec = importlib.util.spec_from_file_location(
        f'{backend.__name__}._test_moe', f'{backend.__path__[0]}/moe.py')
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, kernels.__name__, kernels)
        spec.loader.exec_module(module)
    return module


def _dlinfer_build_spec(**options):
    from lmdeploy.pytorch.backends.moe import FusedMoEBuildSpec

    return FusedMoEBuildSpec(top_k=2, num_experts=4, renormalize=True,
                            hidden_dim=16, ep_size=1, ep_group=None,
                            layer_idx=0, output_dtype=torch.bfloat16,
                            num_max_dispatch_tokens_per_rank=32, **options)


def test_dlinfer_moe_accepts_default_reduction_options(dlinfer_moe):
    impl = dlinfer_moe._build_fused_moe(_dlinfer_build_spec())
    assert isinstance(impl, dlinfer_moe.DlinferFusedMoEImpl)
    assert (impl.top_k, impl.num_experts, impl.renormalize, impl.ep_size) == (2, 4, True, 1)


@pytest.mark.parametrize('options', [
    {'fp32_acc': True}, {'output_scale': 2.5},
    {'fp32_acc': True, 'output_scale': 2.5},
])
def test_dlinfer_moe_rejects_unsupported_reduction_options(dlinfer_moe, monkeypatch, options):
    constructor = Mock()
    monkeypatch.setattr(dlinfer_moe, 'DlinferFusedMoEImpl', constructor)
    with pytest.raises(NotImplementedError, match='fp32_acc or output_scale'):
        dlinfer_moe._build_fused_moe(_dlinfer_build_spec(**options))
    constructor.assert_not_called()
