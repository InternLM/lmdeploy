# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from ..backends import LayerType, get_backend


class ApplyRotaryEmb(nn.Module):

    def __init__(self):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(LayerType.ApplyRotaryEmb)
        self.impl = builder.build()

    def forward(self, query, key, cos, sin, inplace: bool = True):
        return self.impl.forward(query, key, cos, sin, inplace)
