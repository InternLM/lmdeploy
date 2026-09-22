# Copyright (c) OpenMMLab. All rights reserved.
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


@pytest.mark.parametrize('quant_method', ['smooth_quant', 'fp8', 'compressed-tensors'])
def test_build_fused_moe_rejects_unsupported_reduction_options(monkeypatch, quant_method):
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
                            fp32_acc=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_moe_reduce_accumulates_in_fp32_before_scaling_and_casting():
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import moe_reduce

    hidden = torch.tensor([[[1.25, -2.5], [3.0, 4.0]]], device='cuda', dtype=torch.bfloat16)
    weights = torch.tensor([[0.2, 0.7]], device='cuda', dtype=torch.float32)
    actual = moe_reduce(hidden, weights, fp32_acc=True, output_scale=2.5)
    expected = ((hidden.float() * weights[..., None]).sum(dim=1) * 2.5).to(hidden.dtype)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
