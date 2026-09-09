# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.kernels.dlinfer.activation import silu_and_mul

from ..activation import SiluAndMulImpl


class DlinferSiluAndMulImpl(SiluAndMulImpl):
    """Silu + multiple fused implementation."""

    def forward(self, x):
        """forward."""
        return silu_and_mul(x)
