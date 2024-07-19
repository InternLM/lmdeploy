# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from ..activation import SiluAndMulBuilder, SiluAndMulImpl


class DefaultSiluAndMulImpl(SiluAndMulImpl, nn.Module):

    def __init__(self, inplace: bool):
        super().__init__()
        self.inplace = inplace
        self.silu = nn.SiLU(inplace)

    def forward(self, x):
        gate, up = x.chunk(2, -1)
        return self.silu(gate) * up


class DefaultSiluAndMulBuilder(SiluAndMulBuilder):

    @staticmethod
    def build(inplace: bool = False):
        return DefaultSiluAndMulImpl(inplace)
