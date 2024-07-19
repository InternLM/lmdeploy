# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from ..backends import LayerType, get_backend


class SiluAndMul(nn.Module):

    def __init__(self, inplace: bool = True):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(LayerType.SiluAndMul)
        self.impl = builder.build(inplace)

    def forward(self, x):
        return self.impl.forward(x)
