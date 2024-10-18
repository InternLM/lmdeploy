# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from ..activation import (GeluAndMulBuilder, GeluAndMulImpl, SiluAndMulBuilder,
                          SiluAndMulImpl)


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


class DefaultGeluAndMulImpl(GeluAndMulImpl):
    """gelu + multiple residual fused implementation."""

    def __init__(self, approximate: str = 'none'):
        self.act = nn.GELU(approximate=approximate)

    def forward(self, x):
        """forward."""
        gate, up = x.chunk(2, -1)
        return self.act(gate) * up


class DefaultGeluAndMulBuilder(GeluAndMulBuilder):
    """gelu and mul implementation builder."""

    @staticmethod
    def build(approximate: str = 'none'):
        """build."""
        return DefaultGeluAndMulImpl(approximate)
