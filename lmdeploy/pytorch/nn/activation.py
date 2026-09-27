# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor, nn

from lmdeploy.pytorch.models.patch import get_build_model_context

from ..backends import get_backend
from ..backends.activation import GeluAndMulBuildSpec, SiluAndMulBuildSpec


class SiluAndMul(nn.Module):
    """Silu and elementwise multiple."""

    def __init__(self, inplace: bool = True):
        super().__init__()
        self.impl = get_backend().build_op(
            SiluAndMulBuildSpec(inplace=inplace),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

    def forward(self, x: Tensor):
        """forward."""
        return self.impl.forward(x)


class GeluAndMul(nn.Module):
    """Gelu and elementwise multiple."""

    def __init__(self, approximate: str = 'none'):
        super().__init__()
        self.impl = get_backend().build_op(
            GeluAndMulBuildSpec(approximate=approximate),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

    def forward(self, x: Tensor):
        """forward."""
        return self.impl.forward(x)
