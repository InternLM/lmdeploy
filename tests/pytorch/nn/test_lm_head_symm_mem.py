# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

import torch
from torch import nn

from lmdeploy.pytorch.nn.embedding import ParallelLMHead


def test_lm_head_apply_releases_arena_when_tied_weight_already_moved():
    head = ParallelLMHead.__new__(ParallelLMHead)
    nn.Module.__init__(head)
    head.register_parameter('weight', nn.Parameter(torch.empty(1), requires_grad=False))
    gatherer = Mock()
    head._symm_mem_gatherer = gatherer
    # A tied embedding can move the shared Parameter before LM-head._apply.
    head._symm_mem_device = torch.device('cuda:0')
    head._symm_mem_dtype = head.weight.dtype

    result = head._apply(lambda tensor: tensor)

    assert result is head
    gatherer.release.assert_called_once_with()
    gatherer.prepare.assert_not_called()
    assert head._symm_mem_device == torch.device('cpu')
