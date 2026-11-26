# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from lmdeploy.pytorch import distributed
from lmdeploy.pytorch.config import DistConfig, TPMode
from lmdeploy.pytorch.nn.linear.base import LinearBase


@pytest.mark.parametrize('layer_type', ['attn', 'mlp'])
def test_promoted_linear_reduces_fp32_without_lora(monkeypatch, layer_type):
    layer = LinearBase.__new__(LinearBase)
    nn.Module.__init__(layer)
    layer.tp, layer.tp_mode = 2, TPMode.DEFAULT
    layer.all_reduce = True
    layer.tp_reduce_dtype = torch.float32
    layer.layer_type = layer_type
    layer.tp_group = object()
    layer.lora_adapters = nn.ModuleDict()
    local = torch.tensor([[1., 2.]], dtype=torch.bfloat16)
    layer._forward_default = Mock(return_value=local)

    def reduce(tensor, op, group, async_op):
        assert tensor.dtype == torch.float32
        assert group is layer.tp_group
        tensor.add_(0.5)

    collective = Mock(side_effect=reduce)
    monkeypatch.setattr(distributed.dist, 'all_reduce', collective)
    result = layer(local)
    collective.assert_called_once()
    layer._forward_default.assert_called_once_with(local, False, None)
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, local + 0.5)


def test_glm_moe_reduces_fp32_on_moe_group(monkeypatch):
    from lmdeploy.pytorch.models import glm5_next as module

    layer = module.Glm5NextMoE.__new__(module.Glm5NextMoE)
    nn.Module.__init__(layer)
    layer._fp32_tp_reduce = True
    local = torch.tensor([[1., 2.]], dtype=torch.bfloat16)
    monkeypatch.setattr(module.DeepseekV2MoE, 'forward', lambda *args, **kwargs: local)
    group = object()
    context = distributed.DistContext(dist_config=DistConfig(tp=2),
                                      moe_tp_group=distributed.DistGroup(gpu_group=group))

    def reduce(tensor, op, selected_group, async_op):
        assert tensor.dtype == torch.float32
        assert selected_group is group
        tensor.add_(0.5)

    collective = Mock(side_effect=reduce)
    monkeypatch.setattr(distributed.dist, 'all_reduce', collective)
    with distributed.get_dist_manager().context(context):
        result = layer(local)
    collective.assert_called_once()
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, local + 0.5)
