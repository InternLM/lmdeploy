# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..backends import LayerType, get_backend


class RMSNorm(nn.Module):
    """RMS Norm with add residual."""

    def __init__(self,
                 weight: torch.Tensor,
                 eps: float = 1e-6,
                 is_w8a8: bool = False):
        super().__init__()
        backend = get_backend()
        if is_w8a8:
            builder = backend.get_layer_impl_builder(LayerType.RMSNormW8A8)
        else:
            builder = backend.get_layer_impl_builder(LayerType.RMSNorm)
        self.impl = builder.build(weight, eps)

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        return self.impl.forward(x, residual)
