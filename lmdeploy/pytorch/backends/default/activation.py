# Copyright (c) OpenMMLab. All rights reserved.

from torch import nn

from ..activation import GeluAndMulImpl, SiluAndMulImpl


class DefaultSiluAndMulImpl(SiluAndMulImpl):
    """Silu + multiple residual fused implementation."""

    def __init__(self, inplace: bool):
        self.inplace = inplace
        self.silu = nn.SiLU(inplace)

    def forward(self, x):
        """forward."""
        gate, up = x.chunk(2, -1)
        return self.silu(gate) * up


class DefaultGeluAndMulImpl(GeluAndMulImpl):
    """Gelu + multiple residual fused implementation."""

    def __init__(self, approximate: str = 'none'):
        self.act = nn.GELU(approximate=approximate)

    def forward(self, x):
        """forward."""
        gate, up = x.chunk(2, -1)
        return self.act(gate) * up
