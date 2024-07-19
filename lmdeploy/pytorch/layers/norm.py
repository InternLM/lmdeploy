# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..backends import LayerType, get_backend


class RMSNorm(nn.Module):

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(LayerType.RMSNorm)
        self.impl = builder.build(weight, eps)

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        return self.impl.forward(x, residual)
