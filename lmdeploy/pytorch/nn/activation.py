# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor, nn

from ..backends import LayerType, get_backend


class SiluAndMul(nn.Module):
    """Silu and elementwise multiple."""

    def __init__(self, inplace: bool = True):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(LayerType.SiluAndMul)
        self.impl = builder.build(inplace)

    def forward(self, x: Tensor):
        """forward."""
        return self.impl.forward(x)


class GeluAndMul(nn.Module):
    """Gelu and elementwise multiple."""

    def __init__(self, approximate: str = 'none'):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(LayerType.GeluAndMul)
        self.impl = builder.build(approximate)

    def forward(self, x: Tensor):
        """forward."""
        return self.impl.forward(x)
