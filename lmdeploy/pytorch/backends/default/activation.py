# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from ..activation import SiluAndMulBuilder, SiluAndMulImpl


class DefaultSiluAndMulImpl(SiluAndMulImpl):
    """silu + multiple residual fused implementation."""

    def __init__(self, inplace: bool):
        self.inplace = inplace
        self.silu = nn.SiLU(inplace)

    def forward(self, x):
        """forward."""
        gate, up = x.chunk(2, -1)
        return self.silu(gate) * up


class DefaultSiluAndMulBuilder(SiluAndMulBuilder):
    """silu and mul implementation builder."""

    @staticmethod
    def build(inplace: bool = False):
        """build."""
        return DefaultSiluAndMulImpl(inplace)
