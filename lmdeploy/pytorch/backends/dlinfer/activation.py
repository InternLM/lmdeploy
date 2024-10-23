# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.kernels.dlinfer.activation import silu_and_mul

from ..activation import SiluAndMulBuilder, SiluAndMulImpl


class AscendSiluAndMulImpl(SiluAndMulImpl):
    """silu + multiple fused implementation."""

    def forward(self, x):
        """forward."""
        return silu_and_mul(x)


class AscendSiluAndMulBuilder(SiluAndMulBuilder):
    """silu and mul implementation builder."""

    @staticmethod
    def build(inplace: bool = False):
        """build."""
        return AscendSiluAndMulImpl()
